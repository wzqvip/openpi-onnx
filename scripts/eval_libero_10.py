#!/usr/bin/env python3
"""
Evaluate model on LIBERO-10 suite via WebSocket inference service.
Based on Physical Intelligence OpenPI example.
"""

import argparse
import collections
import dataclasses
import json
import logging
import math
import pathlib
import sys
import time

import cv2
import numpy as np
import tqdm
import websockets

# Add libero to path
sys.path.append(str(pathlib.Path("./third_party/libero").resolve()))

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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
    if quat[3] > 1.0: quat[3] = 1.0
    elif quat[3] < -1.0: quat[3] = -1.0
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def _get_libero_env(task, resolution, seed):
    """Initialize LIBERO environment."""
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {
        "bddl_file_name": task_bddl_file,
        "camera_heights": resolution,
        "camera_widths": resolution
    }
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)
    return env, task_description


async def infer_websocket(ws, element):
    """Send inference request via WebSocket and get response."""
    # Prepare request
    request = {
        "image": element["observation/image"].tolist(),
        "wrist_image": element["observation/wrist_image"].tolist(),
        "state": element["observation/state"].tolist(),
        "prompt": element["prompt"]
    }
    
    # Send request
    await ws.send(json.dumps(request))
    
    # Receive response
    response = await ws.recv()
    result = json.loads(response)
    
    return {"actions": np.array(result["actions"])}


async def eval_libero_10(
    host="localhost",
    port=8000,
    resize_size=224,
    replan_steps=5,
    num_steps_wait=10,
    num_trials_per_task=20,
    seed=42,
    output_file=None
):
    """Evaluate on LIBERO-10 suite."""
    np.random.seed(seed)
    
    # Initialize LIBERO-10 task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict["libero_10"]()
    num_tasks_in_suite = task_suite.n_tasks
    
    logger.info(f"Task suite: libero_10 ({num_tasks_in_suite} tasks)")
    logger.info(f"Trials per task: {num_trials_per_task}")
    logger.info(f"WebSocket: ws://{host}:{port}")
    
    # libero_10 max steps (conservative estimate)
    max_steps = 300
    
    # Connect to inference server
    uri = f"ws://{host}:{port}"
    async with websockets.connect(uri, max_size=100 * 1024 * 1024) as websocket:
        logger.info(f"Connected to {uri}")
        
        # Statistics
        total_episodes, total_successes = 0, 0
        task_results = []
        all_inference_times = []
        
        # Evaluate each task
        for task_id in tqdm.tqdm(range(num_tasks_in_suite), desc="Tasks"):
            task = task_suite.get_task(task_id)
            initial_states = task_suite.get_task_init_states(task_id)
            env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, seed)
            
            task_episodes, task_successes = 0, 0
            task_inference_times = []
            
            for episode_idx in range(num_trials_per_task):
                logger.info(f"\nTask {task_id}: {task_description} [Episode {episode_idx+1}/{num_trials_per_task}]")
                
                # Reset environment
                env.reset()
                action_plan = collections.deque()
                obs = env.set_init_state(initial_states[episode_idx])
                
                t = 0
                episode_inference_times = []
                
                while t < max_steps + num_steps_wait:
                    try:
                        # Wait for objects to stabilize
                        if t < num_steps_wait:
                            obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                            t += 1
                            continue
                        
                        # Preprocess images (rotate 180 degrees)
                        img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                        wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
                        
                        img = resize_with_pad(img, resize_size, resize_size)
                        wrist_img = resize_with_pad(wrist_img, resize_size, resize_size)
                        
                        # Replan if needed
                        if not action_plan:
                            element = {
                                "observation/image": img,
                                "observation/wrist_image": wrist_img,
                                "observation/state": np.concatenate((
                                    obs["robot0_eef_pos"],
                                    _quat2axisangle(obs["robot0_eef_quat"]),
                                    obs["robot0_gripper_qpos"],
                                )),
                                "prompt": str(task_description),
                            }
                            
                            # Measure inference time
                            start_time = time.perf_counter()
                            result = await infer_websocket(websocket, element)
                            inference_time = (time.perf_counter() - start_time) * 1000  # ms
                            
                            episode_inference_times.append(inference_time)
                            action_chunk = result["actions"]
                            
                            assert len(action_chunk) >= replan_steps, \
                                f"Policy predicts {len(action_chunk)} steps, need >= {replan_steps}"
                            action_plan.extend(action_chunk[:replan_steps])
                        
                        action = action_plan.popleft()
                        obs, reward, done, info = env.step(action.tolist())
                        
                        if done:
                            task_successes += 1
                            total_successes += 1
                            break
                        
                        t += 1
                    
                    except Exception as e:
                        logger.error(f"Exception in episode: {e}")
                        break
                
                task_episodes += 1
                total_episodes += 1
                
                # Episode statistics
                if episode_inference_times:
                    mean_inf = np.mean(episode_inference_times)
                    task_inference_times.extend(episode_inference_times)
                    all_inference_times.extend(episode_inference_times)
                    logger.info(f"Success: {done} | Inference: {mean_inf:.2f}ms (mean, {len(episode_inference_times)} calls)")
                else:
                    logger.info(f"Success: {done}")
            
            # Task statistics
            task_accuracy = task_successes / task_episodes * 100
            task_mean_latency = np.mean(task_inference_times) if task_inference_times else 0
            task_p99_latency = np.percentile(task_inference_times, 99) if task_inference_times else 0
            
            logger.info(f"\n--- Task {task_id} Summary ---")
            logger.info(f"Task: {task_description}")
            logger.info(f"Accuracy: {task_successes}/{task_episodes} ({task_accuracy:.2f}%)")
            logger.info(f"Latency: {task_mean_latency:.2f}ms mean, {task_p99_latency:.2f}ms P99")
            
            task_results.append({
                "task_id": task_id,
                "task_description": task_description,
                "successes": task_successes,
                "episodes": task_episodes,
                "accuracy": task_accuracy,
                "mean_latency_ms": task_mean_latency,
                "p99_latency_ms": task_p99_latency
            })
        
        # Overall statistics
        overall_accuracy = total_successes / total_episodes * 100
        overall_mean_latency = np.mean(all_inference_times) if all_inference_times else 0
        overall_median_latency = np.median(all_inference_times) if all_inference_times else 0
        overall_p99_latency = np.percentile(all_inference_times, 99) if all_inference_times else 0
        
        logger.info("\n" + "="*80)
        logger.info("FINAL RESULTS - LIBERO-10")
        logger.info("="*80)
        logger.info(f"Overall Accuracy: {total_successes}/{total_episodes} ({overall_accuracy:.2f}%)")
        logger.info(f"Latency Statistics:")
        logger.info(f"  Mean:   {overall_mean_latency:.2f} ms")
        logger.info(f"  Median: {overall_median_latency:.2f} ms")
        logger.info(f"  P99:    {overall_p99_latency:.2f} ms")
        logger.info(f"Total inference calls: {len(all_inference_times)}")
        
        # Save results
        if output_file:
            results = {
                "suite": "libero_10",
                "num_tasks": num_tasks_in_suite,
                "num_trials_per_task": num_trials_per_task,
                "total_episodes": total_episodes,
                "total_successes": total_successes,
                "overall_accuracy": overall_accuracy,
                "latency_ms": {
                    "mean": overall_mean_latency,
                    "median": overall_median_latency,
                    "p99": overall_p99_latency
                },
                "task_results": task_results,
                "all_inference_times_ms": all_inference_times
            }
            
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"\nResults saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate on LIBERO-10 suite")
    parser.add_argument("--host", type=str, default="localhost", help="WebSocket host")
    parser.add_argument("--port", type=int, default=8000, help="WebSocket port")
    parser.add_argument("--resize-size", type=int, default=224, help="Image resize size")
    parser.add_argument("--replan-steps", type=int, default=5, help="Steps before replanning")
    parser.add_argument("--num-steps-wait", type=int, default=10, help="Steps to wait for stabilization")
    parser.add_argument("--num-trials", type=int, default=20, help="Trials per task")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", type=str, default=None, help="Output JSON file")
    
    args = parser.parse_args()
    
    import asyncio
    asyncio.run(eval_libero_10(
        host=args.host,
        port=args.port,
        resize_size=args.resize_size,
        replan_steps=args.replan_steps,
        num_steps_wait=args.num_steps_wait,
        num_trials_per_task=args.num_trials,
        seed=args.seed,
        output_file=args.output
    ))


if __name__ == "__main__":
    main()
