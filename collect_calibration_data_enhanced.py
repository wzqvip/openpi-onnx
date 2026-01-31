#!/usr/bin/env python3
"""
Enhanced calibration data collection for INT8 quantization.
Collects 200+ samples across all Libero task suites for better calibration coverage.
"""

import sys
from unittest.mock import MagicMock
sys.modules["lerobot"] = MagicMock()
sys.modules["lerobot.common"] = MagicMock()
sys.modules["lerobot.common.datasets"] = MagicMock()
sys.modules["lerobot.common.datasets.lerobot_dataset"] = MagicMock()

import pathlib
import torch
import numpy as np
from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from openpi_client import image_tools
from tqdm import tqdm

# Configuration
OUTPUT_PATH = "/home/taco/openpi-onnx/calibration_data/calibration_data_enhanced.pt"
TARGET_SAMPLES = 200  # Collect 200+ samples
SAMPLES_PER_TASK = 10  # 10 samples per task
TASK_SUITES = ["libero_spatial", "libero_goal"]  # Cover multiple suites
RESIZE_SIZE = 224
SEED = 42

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256

def _get_libero_env(task, resolution, seed):
    """Create Libero environment"""
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)
    return env, task_description

def _quat2axisangle(quat):
    """Convert quaternion to axis-angle"""
    import math
    if quat[0] > 1.0: quat[0] = 1.0
    elif quat[0] < -1.0: quat[0] = -1.0
    den = np.sqrt(1.0 - quat[0] * quat[0])
    if math.isclose(den, 0.0): return np.zeros(3)
    return (quat[1:] * 2.0 * math.acos(quat[0])) / den

def collect_calibration_samples():
    """Collect calibration samples from Libero environments"""
    print("="*80)
    print("ENHANCED CALIBRATION DATA COLLECTION")
    print("="*80)
    print(f"Target: {TARGET_SAMPLES}+ samples")
    print(f"Task Suites: {TASK_SUITES}")
    print(f"Samples per task: {SAMPLES_PER_TASK}")
    print("="*80)
    
    np.random.seed(SEED)
    calibration_data = []
    
    for suite_name in TASK_SUITES:
        print(f"\n📦 Processing suite: {suite_name}")
        
        benchmark_dict = benchmark.get_benchmark_dict()
        task_suite = benchmark_dict[suite_name]()
        num_tasks = task_suite.n_tasks
        
        print(f"   Tasks in suite: {num_tasks}")
        
        for task_id in tqdm(range(num_tasks), desc=f"{suite_name}"):
            task = task_suite.get_task(task_id)
            initial_states = task_suite.get_task_init_states(task_id)
            env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, SEED)
            
            # Collect SAMPLES_PER_TASK samples from this task
            for sample_idx in range(min(SAMPLES_PER_TASK, len(initial_states))):
                try:
                    # Reset environment with initial state
                    env.reset()
                    env.reset()
                    obs = env.set_init_state(initial_states[sample_idx])
                    
                    # Take a few dummy steps to get varied states
                    for _ in range(np.random.randint(0, 5)):
                        obs, _, _, _ = env.step(LIBERO_DUMMY_ACTION)
                    
                    # Extract observation data
                    base_image = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                    wrist_image = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
                    
                    # Resize images
                    base_image = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(base_image, RESIZE_SIZE, RESIZE_SIZE)
                    )
                    wrist_image = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(wrist_image, RESIZE_SIZE, RESIZE_SIZE)
                    )
                    
                    # Extract state
                    state = np.concatenate((
                        obs["robot0_eef_pos"],
                        _quat2axisangle(obs["robot0_eef_quat"]),
                        obs["robot0_gripper_qpos"],
                    ))
                    
                    # Create sample
                    sample = {
                        "image": base_image,
                        "wrist_image": wrist_image,
                        "state": state,
                        "prompt": str(task_description),
                        "task_suite": suite_name,
                        "task_id": task_id
                    }
                    
                    calibration_data.append(sample)
                    
                except Exception as e:
                    print(f"\n⚠️  Warning: Failed to collect sample from task {task_id}, sample {sample_idx}: {e}")
                    continue
            
            env.close()
    
    print(f"\n✅ Collected {len(calibration_data)} calibration samples")
    
    # Save calibration data
    pathlib.Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    torch.save(calibration_data, OUTPUT_PATH)
    
    # Get file size
    file_size_mb = pathlib.Path(OUTPUT_PATH).stat().st_size / (1024**2)
    
    print("\n" + "="*80)
    print("COLLECTION COMPLETE")
    print("="*80)
    print(f"✅ Saved to: {OUTPUT_PATH}")
    print(f"✅ Total samples: {len(calibration_data)}")
    print(f"✅ File size: {file_size_mb:.1f} MB")
    print(f"\nSample distribution:")
    
    # Print distribution by suite
    from collections import Counter
    suite_counts = Counter(s["task_suite"] for s in calibration_data)
    for suite, count in suite_counts.items():
        print(f"  - {suite}: {count} samples")
    
    return calibration_data

if __name__ == "__main__":
    collect_calibration_samples()
