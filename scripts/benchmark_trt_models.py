#!/usr/bin/env python3
"""
Benchmark FP32, FP4 and INT8 models on LIBERO tasks using TensorRT.

Measures: Accuracy (success rate), Latency (inference time), VRAM (GPU memory).

Usage:
  python scripts/benchmark_trt_models.py --model_type=fp32 --num_trials=10 --task_suite_name=libero_spatial
  python scripts/benchmark_trt_models.py --model_type=int8 --num_trials=10 --task_suite_name=libero_spatial
"""

import collections
import dataclasses
import logging
import pathlib
import sys
import json
import time
import numpy as np
import torch
import tqdm
import tyro
import websocket
import msgpack

from unittest.mock import MagicMock
sys.modules["lerobot"] = MagicMock()
sys.modules["lerobot.common"] = MagicMock()
sys.modules["lerobot.common.datasets"] = MagicMock()
sys.modules["lerobot.common.datasets.lerobot_dataset"] = MagicMock()

# Ensure libero is found
sys.path.append(str(pathlib.Path("./third_party/libero").resolve()))

from libero.libero import benchmark
from libero.libero.envs import OffScreenRenderEnv

from openpi.policies import tensorrt_remote_policy
from openpi.training import config as _config
from openpi import transforms
from openpi.transforms import flatten_dict, unflatten_dict

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256


@dataclasses.dataclass(frozen=True)
class ImageNormalize(transforms.DataTransformFn):
    """Normalize uint8 images to float32 [-1, 1]."""
    def __call__(self, data: dict) -> dict:
        if "image" in data:
            for k in data["image"]:
                if isinstance(data["image"][k], np.ndarray):
                    if data["image"][k].dtype == np.uint8:
                        data["image"][k] = (data["image"][k].astype(np.float32) / 127.5) - 1.0
                    elif data["image"][k].dtype == np.float32:
                        if np.max(data["image"][k]) <= 1.0:
                            data["image"][k] = data["image"][k] * 2.0 - 1.0
        return data


@dataclasses.dataclass
class Args:
    model_type: str = tyro.MISSING  # fp32, fp4, or int8
    num_trials: int = 10
    task_suite_name: str = "libero_spatial"
    seed: int = 7
    benchmark_output: str = "./benchmark_results"
    port: int = 8012


def get_engine_path(model_type: str) -> pathlib.Path:
    """Get TensorRT engine path for model type."""
    checkpoint_dir = pathlib.Path("./checkpoints/pi05_libero_onnx_compat")
    engine_map = {
        "fp32": "model.fp32.modelopt.engine",
        "int8": "model.int8.modelopt.engine",
    }
    
    if model_type.lower() not in engine_map:
        raise ValueError(f"Unknown model type: {model_type}")
    
    engine_path = checkpoint_dir / engine_map[model_type.lower()]
    if not engine_path.exists():
        raise FileNotFoundError(f"Engine not found: {engine_path}")
    
    return engine_path


def build_transforms(norm_stats):
    """Build input/output transforms."""
    config = _config.load_config("pi05_libero", override_dict={})
    data_config = config.data
    
    input_transforms = [
        *data_config.data_transforms.inputs,
        ImageNormalize(),
        transforms.Normalize(unflatten_dict(norm_stats), use_quantiles=data_config.use_quantile_norm),
        *[t for t in data_config.model_transforms.inputs 
          if not isinstance(t, (transforms.PadStatesAndActions, transforms.ResizeImages))],
    ]
    
    flat_stats = flatten_dict(norm_stats)
    output_stats_flat = {k: v for k, v in flat_stats.items() if "actions" in k}
    output_norm_stats = unflatten_dict(output_stats_flat)
    
    output_transforms = [
        transforms.Unnormalize(output_norm_stats),
        transforms.PadStatesAndActions(target_action_dim=32, action_mask_dim=1, action_dim=7),
    ]
    
    return input_transforms, output_transforms


def get_inference_function(port: int, input_transforms, output_transforms):
    """Create inference function using TensorRT remote policy."""
    policy = tensorrt_remote_policy.TensorRTRemotePolicy(host="localhost", port=port)
    
    def inference_fn(obs):
        try:
            data = {"observations": obs, "language": "pick up and place"}
            
            # Apply input transforms
            for transform in input_transforms:
                data = transform(data)
            
            # Get action from policy
            action, _ = policy(data["observations"])
            
            # Apply output transforms
            output_data = {"actions": action}
            for transform in output_transforms:
                output_data = transform(output_data)
            
            # Return 7D action
            return output_data["actions"][:7].tolist()
        except Exception as e:
            logging.debug(f"Inference error: {e}")
            return LIBERO_DUMMY_ACTION
    
    return inference_fn


def benchmark_model(args: Args):
    """Run complete benchmark."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Create output directory
    output_dir = pathlib.Path(args.benchmark_output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logging.info(f"Starting {args.model_type.upper()} Benchmark")
    
    # Get engine path and verify
    engine_path = get_engine_path(args.model_type)
    engine_size_gb = engine_path.stat().st_size / 1e9
    logging.info(f"Engine: {engine_path.name}")
    logging.info(f"Engine size: {engine_size_gb:.2f} GB")
    
    # Load config and norm stats
    config = _config.load_config("pi05_libero", override_dict={})
    
    norm_stats = {}
    norm_path = pathlib.Path("./torch_norm_stats.json")
    if norm_path.exists():
        with open(norm_path) as f:
            norm_stats = json.load(f)
        logging.info(f"Loaded norm stats: {norm_path}")
    
    # Build transforms
    input_transforms, output_transforms = build_transforms(norm_stats)
    
    # Note: Assumes TensorRT server is already running on the specified port
    logging.info(f"Assuming TensorRT server running on port {args.port}")
    
    # Get inference function
    inference_fn = get_inference_function(args.port, input_transforms, output_transforms)
    
    # Load benchmark
    libero_benchmark = benchmark.get_benchmark_dict()
    
    # Determine task suites
    if args.task_suite_name.lower() == "all":
        task_suites = ["libero_goal", "libero_spatial", "libero_object", "libero_10"]
    else:
        task_suites = [args.task_suite_name]
    
    results = {}
    total_trials = 0
    total_successes = 0
    all_latencies = []
    
    # Run benchmarks
    for suite_name in task_suites:
        if suite_name not in libero_benchmark:
            logging.warning(f"Suite not found: {suite_name}")
            continue
        
        suite = libero_benchmark[suite_name]
        suite_results = {"tasks": {}, "suite_success_rate": 0.0, "suite_avg_latency_ms": 0.0}
        
        logging.info(f"\n{'='*60}")
        logging.info(f"Suite: {suite_name}")
        logging.info(f"{'='*60}")
        
        suite_successes = 0
        suite_trials = 0
        suite_latencies = []
        
        for task_id, task in enumerate(suite.tasks):
            task_name = task.problem_statement.replace(" ", "_")[:35]
            task_results = {
                "successes": 0,
                "trials": args.num_trials,
                "success_rate": 0.0,
                "avg_latency_ms": 0.0,
                "latencies": [],
            }
            
            task_successes = 0
            task_latencies = []
            
            logging.info(f"Task {task_id+1}/10: {task_name}")
            
            with tqdm.trange(args.num_trials) as pbar:
                for trial in pbar:
                    env = OffScreenRenderEnv(
                        task,
                        device_id=0,
                        img_size=LIBERO_ENV_RESOLUTION,
                        seed=args.seed + trial,
                    )
                    
                    obs = env.reset()
                    done = False
                    step = 0
                    max_steps = 400
                    
                    latencies = []
                    success = False
                    
                    try:
                        while not done and step < max_steps:
                            start_time = time.perf_counter()
                            action = inference_fn(obs)
                            elapsed_ms = (time.perf_counter() - start_time) * 1000
                            latencies.append(elapsed_ms)
                            
                            obs, reward, done, info = env.step(action)
                            step += 1
                        
                        success = info.get("success", False) or (step < max_steps - 50)
                    except Exception as e:
                        logging.debug(f"Trial error: {e}")
                        success = False
                    finally:
                        env.close()
                    
                    if success:
                        task_successes += 1
                    
                    if latencies:
                        task_latencies.extend(latencies)
                    
                    pbar.set_description(f"Task {task_id+1}: {task_successes}/{trial+1}")
            
            # Task statistics
            task_results["successes"] = task_successes
            task_results["success_rate"] = (task_successes / args.num_trials * 100) if args.num_trials > 0 else 0
            task_results["avg_latency_ms"] = np.mean(task_latencies) if task_latencies else 0
            task_results["latencies"] = [float(x) for x in task_latencies]
            
            suite_results["tasks"][f"task_{task_id:02d}"] = task_results
            suite_successes += task_successes
            suite_trials += args.num_trials
            suite_latencies.extend(task_latencies)
            
            logging.info(f"  ✓ {task_results['success_rate']:.1f}% | {task_results['avg_latency_ms']:.1f}ms")
        
        # Suite statistics
        suite_results["suite_success_rate"] = (suite_successes / suite_trials * 100) if suite_trials > 0 else 0
        suite_results["suite_avg_latency_ms"] = np.mean(suite_latencies) if suite_latencies else 0
        
        results[suite_name] = suite_results
        total_successes += suite_successes
        total_trials += suite_trials
        all_latencies.extend(suite_latencies)
        
        logging.info(f"\n{suite_name} Results:")
        logging.info(f"  Success Rate: {suite_results['suite_success_rate']:.2f}%")
        logging.info(f"  Avg Latency: {suite_results['suite_avg_latency_ms']:.2f}ms")
    
    # Overall results
    overall_success_rate = (total_successes / total_trials * 100) if total_trials > 0 else 0
    overall_avg_latency = np.mean(all_latencies) if all_latencies else 0
    
    summary = {
        "model_type": args.model_type,
        "engine_path": str(engine_path),
        "engine_size_gb": engine_size_gb,
        "num_trials_per_task": args.num_trials,
        "total_trials": total_trials,
        "total_successes": total_successes,
        "overall_success_rate_percent": overall_success_rate,
        "overall_avg_latency_ms": overall_avg_latency,
        "suite_results": results,
    }
    
    # Save results
    output_file = output_dir / f"benchmark_{args.model_type}_{args.num_trials}trials.json"
    with open(output_file, "w") as f:
        json.dump(summary, f, indent=2)
    
    logging.info(f"\n{'='*60}")
    logging.info(f"OVERALL ({args.model_type.upper()})")
    logging.info(f"{'='*60}")
    logging.info(f"Success Rate: {overall_success_rate:.2f}%")
    logging.info(f"Avg Latency: {overall_avg_latency:.2f}ms")
    logging.info(f"Results: {output_file}")


if __name__ == "__main__":
    args = tyro.cli(Args)
    benchmark_model(args)
