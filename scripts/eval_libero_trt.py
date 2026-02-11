#!/usr/bin/env python3
"""
Evaluate TensorRT engine on LIBERO tasks via WebSocket inference service.
"""

import argparse
import asyncio
import json
import logging
import math
import pathlib
import sys
import time
from collections import deque

import cv2
import msgpack
import msgpack_numpy
import numpy as np
import tqdm
import websockets

# Torch 2.6+ compatibility: allow numpy globals and force weights_only=False
import torch
torch.serialization.add_safe_globals([type(np.ndarray), np.core.multiarray._reconstruct, np.ndarray, np.dtype])
_torch_load = torch.load
def _safe_torch_load(*args, **kwargs):
    if "weights_only" not in kwargs:
        kwargs["weights_only"] = False
    return _torch_load(*args, **kwargs)
torch.load = _safe_torch_load

# Optional dependencies used by openpi tokenizers (stub if missing)
try:
    import jax  # noqa: F401
except Exception:
    import types
    sys.modules.setdefault("jax", types.ModuleType("jax"))
    sys.modules.setdefault("orbax", types.ModuleType("orbax"))
    sys.modules.setdefault("orbax.checkpoint", types.ModuleType("orbax.checkpoint"))

from openpi.models.tokenizer import PaligemmaTokenizer

# Patch msgpack for numpy support
msgpack_numpy.patch()

# Add libero to path
sys.path.append(str(pathlib.Path("./third_party/libero").resolve()))

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256


def resize_with_pad(image, target_height, target_width):
    """Resize image with padding to maintain aspect ratio."""
    h, w = image.shape[:2]
    scale = min(target_height / h, target_width / w)
    new_h, new_w = int(h * scale), int(w * scale)
    image = cv2.resize(image, (new_w, new_h))
    
    top = (target_height - new_h) // 2
    bottom = target_height - new_h - top
    left = (target_width - new_w) // 2
    right = target_width - new_w - left
    
    return cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])


def _quat2axisangle(quat):
    """Convert quaternion to axis-angle representation."""
    if quat[0] > 1.0: quat[0] = 1.0
    elif quat[0] < -1.0: quat[0] = -1.0
    den = np.sqrt(1.0 - quat[0] * quat[0])
    if math.isclose(den, 0.0): return np.zeros(3)
    return (quat[1:] * 2.0 * math.acos(quat[0])) / den


async def call_trt_inference(websocket, image, wrist_image, state, tokenized_prompt, tokenized_prompt_mask):
    """Call TensorRT inference service via WebSocket."""
    # Prepare payload matching the expected format
    # CRITICAL: TensorRT expects NCHW format [batch, channels, height, width]
    # Convert from HWC to CHW format
    image_nchw = np.expand_dims(np.transpose(image, (2, 0, 1)), 0).astype(np.float32)  # [1, 3, 224, 224]
    wrist_nchw = np.expand_dims(np.transpose(wrist_image, (2, 0, 1)), 0).astype(np.float32)

    # Normalize uint8 images to [-1, 1] like preprocessing
    image_nchw = image_nchw / 127.5 - 1.0
    wrist_nchw = wrist_nchw / 127.5 - 1.0
    
    payload = {
        "base_0_rgb": image_nchw,
        "left_wrist_0_rgb": wrist_nchw,
        "right_wrist_0_rgb": np.zeros((1, 3, 224, 224), dtype=np.float32),
        "state": np.expand_dims(state, 0).astype(np.float32),
        "observation.tokenized_prompt": tokenized_prompt,
        "observation.tokenized_prompt_mask": tokenized_prompt_mask,
        "prompt": tokenized_prompt,
        "prompt_mask": tokenized_prompt_mask,
        "noise": np.random.randn(1, 10, 32).astype(np.float32)
    }
    
    start_time = time.time()
    
    # Send request
    message = msgpack.packb(payload)
    await websocket.send(message)
    
    # Receive response with timeout
    try:
        response_data = await asyncio.wait_for(websocket.recv(), timeout=10.0)
    except asyncio.TimeoutError:
        logger.error(f"WebSocket recv timeout after 10s")
        raise RuntimeError("WebSocket inference timeout")
    
    result = msgpack.unpackb(response_data)
    
    latency_ms = (time.time() - start_time) * 1000
    
    # Extract actions
    if "actions" in result:
        actions = result["actions"]
        if actions.ndim == 3:
            actions = actions[0]
        return actions, latency_ms
    else:
        raise RuntimeError(f"No actions in response: {result.keys()}")


async def eval_libero_trt(args):
    """Run LIBERO evaluation using TensorRT inference service."""
    np.random.seed(args.seed)
    tokenizer = PaligemmaTokenizer(max_len=200)

    # Load norm stats for input normalization and output unnormalization
    norm_stats_path = "/home/taco/checkpoints/pi05_libero_pytorch/assets/physical-intelligence/libero/norm_stats.json"
    with open(norm_stats_path, "r") as f:
        norm_stats_data = json.load(f)

    state_mean = np.array(norm_stats_data["norm_stats"]["state"]["mean"], dtype=np.float32)
    state_std = np.array(norm_stats_data["norm_stats"]["state"]["std"], dtype=np.float32)
    action_mean = np.array(norm_stats_data["norm_stats"]["actions"]["mean"], dtype=np.float32)
    action_std = np.array(norm_stats_data["norm_stats"]["actions"]["std"], dtype=np.float32)

    logger.info(
        f"Loaded norm stats: state_mean shape={state_mean.shape}, state_std shape={state_std.shape}, "
        f"action_mean shape={action_mean.shape}, action_std shape={action_std.shape}"
    )
    
    # Initialize task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    num_tasks = task_suite.n_tasks
    
    logger.info(f"Task suite: {args.task_suite_name} ({num_tasks} tasks)")
    logger.info(f"TensorRT server: {args.ws_url}")
    logger.info(f"Trials per task: {args.num_trials_per_task}")
    
    # Task duration map
    steps_map = {
        "libero_spatial": 220,
        "libero_object": 280,
        "libero_goal": 300,
        "libero_10": 520,
        "libero_90": 400
    }
    max_steps = steps_map.get(args.task_suite_name, 400)
    
    # Statistics tracking
    total_episodes = 0
    total_successes = 0
    latencies = []
    debug_logged = False
    
    # Connect to WebSocket server
    try:
        async with websockets.connect(args.ws_url, max_size=None) as websocket:
            # Wait for ready signal
            ready_msg = await websocket.recv()
            ready_data = msgpack.unpackb(ready_msg)
            logger.info(f"Server ready: {ready_data}")
            
            # Run evaluation
            for task_id in tqdm.tqdm(range(num_tasks), desc="Tasks"):
                task = task_suite.get_task(task_id)
                task_description = task.language
                initial_states = task_suite.get_task_init_states(task_id)
                
                # Create environment
                task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
                env_args = {
                    "bddl_file_name": task_bddl_file,
                    "camera_heights": LIBERO_ENV_RESOLUTION,
                    "camera_widths": LIBERO_ENV_RESOLUTION
                }
                env = OffScreenRenderEnv(**env_args)
                env.seed(args.seed)
                
                task_successes = 0
                
                for trial in tqdm.tqdm(range(args.num_trials_per_task), 
                                       desc=f"Task {task_id}: {task_description}", 
                                       leave=False):
                    env.reset()
                    obs = env.set_init_state(initial_states[trial % len(initial_states)])
                    
                    action_queue = deque()
                    done = False
                    t = 0
                    
                    while t < max_steps + args.num_steps_wait:
                        # Wait period
                        if t < args.num_steps_wait:
                            obs, _, _, _ = env.step(LIBERO_DUMMY_ACTION)
                            t += 1
                            continue
                        
                        # Get new actions from model
                        if not action_queue:
                            # Extract observation
                            agentview_image = obs["agentview_image"][::-1, ::-1]
                            wrist_image = obs["robot0_eye_in_hand_image"][::-1, ::-1]
                            robot_state = obs["robot0_joint_pos"]
                            eef_pos = obs["robot0_eef_pos"]
                            eef_quat = obs["robot0_eef_quat"]
                            gripper_qpos = obs["robot0_gripper_qpos"]
                            
                            # Compute axis-angle from quaternion
                            eef_angle = _quat2axisangle(eef_quat)
                            
                            # Concatenate state (eef_pos: 3, eef_angle: 3, gripper_qpos: 2 = 8 total)
                            # Use first two gripper dimensions (Panda gripper has 2 DOF)
                            gripper_state = gripper_qpos[:2] if len(gripper_qpos) >= 2 else gripper_qpos
                            state = np.concatenate([eef_pos, eef_angle, gripper_state]).astype(np.float32)

                            # Normalize state using training stats
                            state = (state - state_mean) / (state_std + 1e-6)
                            
                            # Resize image
                            image = resize_with_pad(agentview_image, args.resize_size, args.resize_size)
                            wrist_image = resize_with_pad(wrist_image, args.resize_size, args.resize_size)
                            
                            # Tokenize prompt (matches model export expectations)
                            tokens, token_mask = tokenizer.tokenize(task_description, state)
                            tokenized_prompt = np.expand_dims(tokens.astype(np.int32), 0)
                            tokenized_prompt_mask = np.expand_dims(token_mask.astype(np.bool_), 0)

                            # Call inference service
                            try:
                                actions, latency = await call_trt_inference(
                                    websocket, image, wrist_image, state, tokenized_prompt, tokenized_prompt_mask
                                )
                                if not debug_logged:
                                    logger.info(
                                        f"DEBUG token_mask sum={int(tokenized_prompt_mask.sum())}, "
                                        f"tokens head={tokenized_prompt[0, :10].tolist()}"
                                    )
                                    logger.info(
                                        f"DEBUG state range=[{state.min():.4f}, {state.max():.4f}]"
                                    )
                                    logger.info(
                                        f"DEBUG actions range (raw)=[{actions.min():.4f}, {actions.max():.4f}]"
                                    )
                                    debug_logged = True
                                latencies.append(latency)
                                logger.info(f"Infer latency (ms): {latency:.2f}")
                                
                                # Extract actions
                                # Model outputs 32D actions, but LIBERO only uses first 7D
                                actions_replan = actions[:args.replan_steps]  # [replan_steps, 32]
                                actions_7d = actions_replan[:, :7]  # [replan_steps, 7]

                                # Unnormalize actions using training stats
                                actions_7d = actions_7d * action_std + action_mean
                                
                                if not debug_logged:
                                    logger.info(
                                        f"DEBUG actions range (7D, unnorm)=[{actions_7d.min():.4f}, {actions_7d.max():.4f}]"
                                    )
                                
                                action_queue.extend(actions_7d)
                                

                            except Exception as e:
                                logger.error(f"Inference failed: {e}")
                                import traceback
                                traceback.print_exc()
                                break
                        
                        # Execute action
                        if action_queue:
                            action = action_queue.popleft()
                            obs, reward, done, info = env.step(action.tolist())
                            
                            if done:
                                task_successes += 1
                                total_successes += 1
                                break
                        
                        t += 1
                    
                    total_episodes += 1
                    
                    if not done:
                        logger.warning(f"Task {task_id} Trial {trial}: FAILED (timeout)")
                
                # Clean up environment
                env.close()
                
                logger.info(f"Task {task_id} ({task_description}): {task_successes}/{args.num_trials_per_task} successes")
    
    except Exception as e:
        logger.error(f"WebSocket connection failed: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    # Final statistics
    success_rate = total_successes / total_episodes if total_episodes > 0 else 0.0
    
    logger.info("=" * 60)
    logger.info(f"Total Success Rate: {success_rate:.2%} ({total_successes}/{total_episodes})")
    
    if latencies:
        latencies_array = np.array(latencies)
        logger.info(f"Latency (ms): Mean={latencies_array.mean():.2f}, "
                   f"Median={np.median(latencies_array):.2f}, "
                   f"P99={np.percentile(latencies_array, 99):.2f}")
    
    logger.info("=" * 60)
    
    # Save results
    results = {
        "task_suite": args.task_suite_name,
        "total_episodes": total_episodes,
        "total_successes": total_successes,
        "success_rate": success_rate,
        "latencies_ms": latencies,
        "latency_mean_ms": float(np.mean(latencies)) if latencies else None,
        "latency_median_ms": float(np.median(latencies)) if latencies else None,
        "latency_p99_ms": float(np.percentile(latencies, 99)) if latencies else None,
    }
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate TensorRT engine on LIBERO tasks")
    parser.add_argument("--task_suite_name", type=str, default="libero_spatial",
                       help="LIBERO task suite to evaluate")
    parser.add_argument("--num_trials_per_task", type=int, default=3,
                       help="Number of trials per task")
    parser.add_argument("--ws_url", type=str, default="ws://localhost:8016",
                       help="TensorRT inference server WebSocket URL")
    parser.add_argument("--resize_size", type=int, default=224,
                       help="Image resize target")
    parser.add_argument("--replan_steps", type=int, default=5,
                       help="Number of action steps to execute before replanning")
    parser.add_argument("--num_steps_wait", type=int, default=10,
                       help="Number of initial steps to wait")
    parser.add_argument("--seed", type=int, default=7,
                       help="Random seed")
    
    args = parser.parse_args()
    
    results = asyncio.run(eval_libero_trt(args))
    
    if results:
        # Print summary
        print("\nFinal Results:")
        print(f"Success Rate: {results['success_rate']:.2%}")
        print(f"Latency: {results['latency_mean_ms']:.2f} ms (mean), "
              f"{results['latency_p99_ms']:.2f} ms (P99)")
    else:
        print("Evaluation failed")


if __name__ == "__main__":
    main()
