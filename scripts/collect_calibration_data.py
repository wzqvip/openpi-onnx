
import collections
import dataclasses
import logging
import math
import pathlib
import sys
import torch
import numpy as np
import cv2
import tyro

# Mocks - Must be before openpi imports
from unittest.mock import MagicMock
sys.modules["lerobot"] = MagicMock()
sys.modules["lerobot.common"] = MagicMock()
sys.modules["lerobot.common.datasets"] = MagicMock()
sys.modules["lerobot.common.datasets.lerobot_dataset"] = MagicMock()

# Ensure libero is found
sys.path.append(str(pathlib.Path("./third_party/libero").resolve()))
from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv

from openpi.training import config as _config

# Dummy action for stepping environment
LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256

def resize_with_pad(image, target_height, target_width):
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
    if quat[0] > 1.0: quat[0] = 1.0
    elif quat[0] < -1.0: quat[0] = -1.0
    den = np.sqrt(1.0 - quat[0] * quat[0])
    if math.isclose(den, 0.0): return np.zeros(3)
    return (quat[1:] * 2.0 * math.acos(quat[0])) / den

def _get_libero_env(task, resolution, seed):
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)
    return env, task_description

@dataclasses.dataclass
class Args:
    config: str = "pi05_libero"
    output_path: str = "calibration_data.pt"
    num_samples: int = 64
    resize_size: int = 224
    task_suite_name: str = "libero_spatial"
    seed: int = 7

def main(args: Args):
    logging.basicConfig(level=logging.INFO)
    print(f"Collecting {args.num_samples} samples for calibration...")

    # Load Config (just to get model config structure if needed, mostly for transforms if we were using them)
    # We will manually preprocess to match eval script
    
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    
    # Use first task
    task_id = 0
    task = task_suite.get_task(task_id)
    initial_states = task_suite.get_task_init_states(task_id)
    env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)
    
    env.reset()
    env.set_init_state(initial_states[0])
    
    collected_samples = []
    
    count = 0
    while count < args.num_samples:
        # Step with dummy action (random walk is fine for calibration statistics of images)
        # Or better: just reset if done.
        
        obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
        
        # Preprocessing (matches eval_libero_torch.py)
        img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
        wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
        
        img = resize_with_pad(img, args.resize_size, args.resize_size)
        wrist_img = resize_with_pad(wrist_img, args.resize_size, args.resize_size)
        
        # Create Element
        # Note: We need to match the structure expected by the model wrapper in export script
        # The export script uses OnnxWrapper which expects arguments:
        # base_0_rgb, left_wrist_0_rgb, right_wrist_0_rgb, state, tokenized_prompt, tokenized_prompt_mask, noise
        
        # We save the raw numpy components. 
        # Tokenized prompt will be handled by the export script (it uses internal tokenizer).
        # But we need the prompt text.
        
        sample = {
            "image": img, # [H, W, 3] uint8
            "wrist_image": wrist_img,
            "state": np.concatenate(
                (
                    obs["robot0_eef_pos"],
                    _quat2axisangle(obs["robot0_eef_quat"]),
                    obs["robot0_gripper_qpos"],
                )
            ).astype(np.float32),
            "prompt": str(task_description)
        }
        
        collected_samples.append(sample)
        count += 1
        print(f"Collected {count}/{args.num_samples}", end="\r")
        
        if done:
            env.reset()
            env.set_init_state(initial_states[0])
            
    print(f"\nSaving to {args.output_path}")
    torch.save(collected_samples, args.output_path)
    print("Done.")

if __name__ == "__main__":
    tyro.cli(main)
