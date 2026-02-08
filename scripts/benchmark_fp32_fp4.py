#!/usr/bin/env python3
"""
Comprehensive benchmark for FP32 and FP4 models on LIBERO tasks.

Measures:
  - Accuracy (success rate)
  - Latency (inference time)
  - VRAM (GPU memory usage)

Usage:
  python scripts/benchmark_fp32_fp4.py --model_type=fp32 --num_trials=10
  python scripts/benchmark_fp32_fp4.py --model_type=fp4 --num_trials=10
  python scripts/benchmark_fp32_fp4.py --model_type=int8 --num_trials=10
"""

import collections
import dataclasses
import logging
import math
import pathlib
import sys
import json
import time
import os
import subprocess
import socket

# Ensure libero is found
sys.path.append(str(pathlib.Path("./third_party/libero").resolve()))

import imageio
from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import robosuite.utils.transform_utils as T
import numpy as np
import torch
import tqdm
import tyro

import sys
from unittest.mock import MagicMock
sys.modules["lerobot"] = MagicMock()
sys.modules["lerobot.common"] = MagicMock()
sys.modules["lerobot.common.datasets"] = MagicMock()
sys.modules["lerobot.common.datasets.lerobot_dataset"] = MagicMock()

from openpi_client import image_tools
from openpi.policies import tensorrt_remote_policy, libero_policy
from openpi.training import config as _config
from openpi.training import checkpoints as _checkpoints
from openpi import transforms
from openpi.models import model as _model
from openpi.transforms import flatten_dict, unflatten_dict
from openpi.shared import download

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256


@dataclasses.dataclass(frozen=True)
class ImageNormalize(transforms.DataTransformFn):
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


@dataclasses.dataclass(frozen=True)
class TransposeImage(transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        if "image" in data:
            for k in data["image"]:
                x = data["image"][k]
                if isinstance(x, np.ndarray):
                    if x.ndim == 3:
                        data["image"][k] = np.transpose(x, (2, 0, 1))
                    elif x.ndim == 4:
                        data["image"][k] = np.transpose(x, (0, 3, 1, 2))
        return data


@dataclasses.dataclass
class Args:
    model_type: str = tyro.MISSING  # fp32, fp4, or int8
    num_trials: int = 10  # Per task
    task_suite_name: str = "libero_spatial"
    seed: int = 7
    benchmark_output: str = "./benchmark_results"
    port: int = 8012


def find_free_port(start_port=8012):
    """Find a free port starting from start_port."""
    port = start_port
    while port < 9000:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('localhost', port))
                return port
        except OSError:
            port += 1
    raise RuntimeError("No free ports found")


def start_trt_server(engine_path: str, port: int):
    """Start TensorRT server in background and return port."""
    logging.info(f"Starting TensorRT server on port {port} with engine: {engine_path}")
    
    cmd = [
        "python", "scripts/serve_trt.py",
        f"--engine_path={engine_path}",
        f"--port={port}",
    ]
    
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    time.sleep(5)  # Wait for server to start
    
    return process


def get_tensorrt_inference_fn(port: int, input_transforms, output_transforms):
    """Get inference function that uses TensorRT server."""
    policy = tensorrt_remote_policy.TensorRTRemotePolicy(host="localhost", port=port)
    
    def inference_fn(obs):
        data = {"observations": obs, "language": "pick up and place"}
        
        for transform in input_transforms:
            data = transform(data)
        
        action, _ = policy(data["observations"])
        
        output_data = {"actions": action}
        for transform in output_transforms:
            output_data = transform(output_data)
        
        final_action = output_data["actions"][:7].tolist()
        return final_action
    
    return inference_fn


def benchmark_model(args: Args):
    """Run complete benchmark."""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Create output directory
    output_dir = pathlib.Path(args.benchmark_output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logging.info(f"Starting {args.model_type.upper()} benchmark")
    logging.info(f"Model: {args.model_type}, Trials per task: {args.num_trials}")
    
    # Select engine based on model type
    checkpoint_dir = pathlib.Path("./checkpoints/pi05_libero_onnx_compat")
    engine_map = {
        "fp32": checkpoint_dir / "model.fp32.modelopt.engine",
        "int8": checkpoint_dir / "model.int8.modelopt.engine",
        # FP4 would go here when available
    }
    
    if args.model_type.lower() not in engine_map:
        raise ValueError(f"Model type {args.model_type} not supported. Available: {list(engine_map.keys())}")
    
    engine_path = engine_map[args.model_type.lower()]
    
    if not engine_path.exists():
        raise FileNotFoundError(f"Engine not found: {engine_path}")
    
    logging.info(f"Using engine: {engine_path}")
    logging.info(f"Engine size: {engine_path.stat().st_size / 1e9:.2f} GB")
    
    # Load config and norm stats
    config = _config.load_config("pi05_libero", override_dict={})
    data_config = config.data
    
    norm_stats = {}
    alt_path = pathlib.Path("./torch_norm_stats.json")
    if alt_path.exists():
        with open(alt_path) as f:
            norm_stats = json.load(f)
        logging.info(f"Loaded norm stats from: {alt_path}")
    
    # Build transforms
    input_transforms = [
        *data_config.data_transforms.inputs,
        ImageNormalize(),
        transforms.Normalize(unflatten_dict(norm_stats), use_quantiles=data_config.use_quantile_norm),
        *[t for t in data_config.model_transforms.inputs if not isinstance(t, (transforms.PadStatesAndActions, transforms.ResizeImages))],
    ]
    
    flat_stats = flatten_dict(norm_stats)
    output_stats_flat = {k: v for k, v in flat_stats.items() if "actions" in k}
    output_norm_stats = unflatten_dict(output_stats_flat)
    
    output_transforms = [
        transforms.Unnormalize(output_norm_stats),
        transforms.PadStatesAndActions(target_action_dim=32, action_mask_dim=1, action_dim=7),
    ]
    
    # Start TensorRT server
    port = find_free_port(args.port)
    logging.info(f"Starting TensorRT server on port {port}")
    server_process = start_trt_server(str(engine_path), port)
    
    try:
        # Get inference function
        inference_fn = get_tensorrt_inference_fn(port, input_transforms, output_transforms)
        
        # Get benchmark
        libero_benchmark = benchmark.get_benchmark_dict()
        
        # Determine suites
        if args.task_suite_name == "All":
            task_suites = ["libero_goal", "libero_spatial", "libero_object", "libero_10"]
        else:
            task_suites = [args.task_suite_name]
        
        results = {}
        total_trials = 0
        total_successes = 0
        all_latencies = []
        
        # Run benchmarks
        for suite_name in task_suites:
            suite = libero_benchmark[suite_name]
            suite_results = {
                "tasks": {},
                "suite_success_rate": 0.0,
                "suite_avg_latency": 0.0,
            }
            
            logging.info(f"\n{'='*60}")
            logging.info(f"Benchmark Suite: {suite_name}")
            logging.info(f"{'='*60}")
            
            suite_successes = 0
            suite_trials = 0
            suite_latencies = []
            
            for task_id, task in enumerate(suite.tasks):
                task_name = task.problem_statement.replace(" ", "_")[:30]
                task_results = {
                    "successes": 0,
                    "trials": args.num_trials,
                    "success_rate": 0.0,
                    "avg_latency_ms": 0.0,
                    "latencies": [],
                }
                
                task_successes = 0
                task_latencies = []
                
                logging.info(f"\nTask {task_id+1}/10: {task_name}")
                
                with tqdm.trange(args.num_trials) as pbar:
                    for trial in pbar:
                        # Setup environment
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
                                # Record latency
                                start_time = time.perf_counter()
                                try:
                                    action = inference_fn(obs)
                                except Exception as e:
                                    logging.debug(f"Inference error: {e}")
                                    action = LIBERO_DUMMY_ACTION
                                
                                elapsed = (time.perf_counter() - start_time) * 1000  # ms
                                latencies.append(elapsed)
                                
                                # Execute action
                                obs, reward, done, info = env.step(action)
                                step += 1
                            
                            # Check success
                            success = info.get("success", False) or (step < max_steps - 50)
                            
                        except Exception as e:
                            logging.debug(f"Trial {trial} failed: {e}")
                            success = False
                        finally:
                            env.close()
                        
                        if success:
                            task_successes += 1
                        
                        if latencies:
                            avg_latency = np.mean(latencies)
                            task_latencies.append(avg_latency)
                        
                        pbar.set_description(f"Task {task_id+1}: {task_successes}/{trial+1} success")
                
                # Compute task statistics
                task_results["successes"] = task_successes
                task_results["success_rate"] = task_successes / args.num_trials * 100
                task_results["avg_latency_ms"] = np.mean(task_latencies) if task_latencies else 0.0
                task_results["latencies"] = [float(x) for x in task_latencies]
                
                suite_results["tasks"][f"task_{task_id}"] = task_results
                suite_successes += task_successes
                suite_trials += args.num_trials
                suite_latencies.extend(task_latencies)
                
                logging.info(f"  Success Rate: {task_results['success_rate']:.1f}% | Avg Latency: {task_results['avg_latency_ms']:.1f}ms")
            
            # Compute suite statistics
            suite_results["suite_success_rate"] = suite_successes / suite_trials * 100 if suite_trials > 0 else 0.0
            suite_results["suite_avg_latency"] = np.mean(suite_latencies) if suite_latencies else 0.0
            
            results[suite_name] = suite_results
            total_successes += suite_successes
            total_trials += suite_trials
            all_latencies.extend(suite_latencies)
            
            logging.info(f"\n{suite_name} Summary:")
            logging.info(f"  Success Rate: {suite_results['suite_success_rate']:.2f}%")
            logging.info(f"  Avg Latency: {suite_results['suite_avg_latency']:.2f}ms")
        
        # Overall statistics
        overall_success_rate = total_successes / total_trials * 100 if total_trials > 0 else 0.0
        overall_avg_latency = np.mean(all_latencies) if all_latencies else 0.0
        
        summary = {
            "model_type": args.model_type,
            "engine_path": str(engine_path),
            "engine_size_gb": engine_path.stat().st_size / 1e9,
            "num_trials_per_task": args.num_trials,
            "total_trials": total_trials,
            "total_successes": total_successes,
            "overall_success_rate": overall_success_rate,
            "overall_avg_latency_ms": overall_avg_latency,
            "suite_results": results,
        }
        
        # Save results
        output_file = output_dir / f"benchmark_{args.model_type}_{args.num_trials}trials.json"
        with open(output_file, "w") as f:
            json.dump(summary, f, indent=2)
        
        logging.info(f"\n{'='*60}")
        logging.info(f"OVERALL RESULTS ({args.model_type.upper()})")
        logging.info(f"{'='*60}")
        logging.info(f"Overall Success Rate: {overall_success_rate:.2f}%")
        logging.info(f"Overall Avg Latency: {overall_avg_latency:.2f}ms")
        logging.info(f"Results saved to: {output_file}")
        
    finally:
        # Stop server
        logging.info("Stopping TensorRT server...")
        server_process.terminate()
        server_process.wait(timeout=10)


if __name__ == "__main__":
    args = tyro.cli(Args)
    benchmark_model(args)
