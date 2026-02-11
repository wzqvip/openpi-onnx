
import collections
import dataclasses
import logging
import math
import pathlib
import sys

# Ensure libero is found
sys.path.append(str(pathlib.Path("./third_party/libero").resolve()))

import imageio
from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import numpy as np
import tqdm
import tyro
import torch
import time

# PATCH: Fix weights_only=True default in Torch 2.4+
_original_torch_load = torch.load
def safe_torch_load(*args, **kwargs):
    if "weights_only" not in kwargs:
        kwargs["weights_only"] = False
    return _original_torch_load(*args, **kwargs)
torch.load = safe_torch_load

# Mocks - Must be before openpi imports
import sys
from unittest.mock import MagicMock
sys.modules["lerobot"] = MagicMock()
sys.modules["lerobot.common"] = MagicMock()
sys.modules["lerobot.common.datasets"] = MagicMock()
sys.modules["lerobot.common.datasets.lerobot_dataset"] = MagicMock()

from openpi.training import config as _config
from openpi.policies import policy_config
import cv2  # Use cv2 for resize if image_tools not available, or just implement resize
# from openpi_client import image_tools # Try to avoid client dependency if possible, but it might be there.

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

@dataclasses.dataclass
class Args:
    checkpoint: str = "checkpoints/pi05_libero_pytorch"
    config: str = "pi05_libero"
    
    resize_size: int = 224
    replan_steps: int = 5
    task_suite_name: str = "libero_spatial"
    num_steps_wait: int = 10
    num_trials_per_task: int = 1
    video_out_path: str = "data/libero/videos_torch"
    seed: int = 7
    task_id: int | None = None
    force_cpu: bool = False

def _quat2axisangle(quat):
    # Robosuite convention: [w, x, y, z]
    # w is scalar, xyz is vector
    if quat[0] > 1.0: quat[0] = 1.0
    elif quat[0] < -1.0: quat[0] = -1.0
    den = np.sqrt(1.0 - quat[0] * quat[0])
    if math.isclose(den, 0.0): return np.zeros(3)
    return (quat[1:] * 2.0 * math.acos(quat[0])) / den

def eval_libero(args: Args) -> None:
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    logging.info(f"Loading config: {args.config}")
    
    # Hack to disable torch.compile for stability
    torch.compile = lambda x, **k: x
    
    # Mocks
    import sys
    from unittest.mock import MagicMock
    sys.modules["lerobot"] = MagicMock()
    sys.modules["lerobot.common"] = MagicMock()
    sys.modules["lerobot.common.datasets"] = MagicMock()
    sys.modules["lerobot.common.datasets.lerobot_dataset"] = MagicMock()

    train_config = _config.get_config(args.config)
    # Force float32 for baseline stability vs numpy inputs
    train_config = dataclasses.replace(train_config, model=dataclasses.replace(train_config.model, dtype="float32"))
    # [FIX] Override action_dim to 32 to match checkpoint weights
    train_config = dataclasses.replace(train_config, model=dataclasses.replace(train_config.model, action_dim=32))
    
    device = "cpu" if args.force_cpu else ("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Loading policy from {args.checkpoint} on {device}...")
    
    policy = policy_config.create_trained_policy(
        train_config, 
        args.checkpoint, 
        pytorch_device=device
    )
    
    # Inspect Policy for Norm Stats
    print(f"Policy Dir: {dir(policy)}")
    try:
        print(f"Policy Vars: {vars(policy)}")
    except:
        pass
    
    # Recursive Search for norm_stats
    def find_norm_stats(obj, depth=0, visited=None):
        if visited is None: visited = set()
        if depth > 5: return None
        if id(obj) in visited: return None
        visited.add(id(obj))
        
        # Check current object
        if hasattr(obj, "norm_stats") and isinstance(obj.norm_stats, dict):
            return obj.norm_stats
            
        # Search attributes
        if hasattr(obj, "__dict__"):
            for k, v in vars(obj).items():
                if not k.startswith("__"):
                    res = find_norm_stats(v, depth+1, visited)
                    if res: return res
                    
        # Search lists/dicts
        if isinstance(obj, (list, tuple)):
            for item in obj:
                res = find_norm_stats(item, depth+1, visited)
                if res: return res
        
        return None

    print("Searching for norm_stats recursively...")
    norm_stats = find_norm_stats(policy)
    
    if norm_stats:
        print("FOUND norm_stats!")
        import json
        flat_stats = {}
        for k, v in norm_stats.items():
            flat_stats[k] = {"mean": v.mean.tolist(), "std": v.std.tolist(), "q01": v.q01.tolist() if v.q01 is not None else None, "q99": v.q99.tolist() if v.q99 is not None else None}
            
        with open("torch_norm_stats.json", "w") as f:
            json.dump(flat_stats, f, indent=2)
        print("Dumbed stats to torch_norm_stats.json")
    else:
        print("WARNING: Recursive search failed to find norm_stats.")
    
    # Attempt to extract
    # Usually in data_config -> but config failed.
    # What if we just print train_config.data to see what it has?
    print(f"Data Config Type: {type(train_config.data)}")
    print(f"Data Config Dir: {dir(train_config.data)}")

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    logging.info(f"Task suite: {args.task_suite_name}")

    pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)
    
    # Task duration map
    steps_map = {
        "libero_spatial": 220,
        "libero_object": 280,
        "libero_goal": 300,
        "libero_10": 520,
        "libero_90": 400
    }
    max_steps = steps_map.get(args.task_suite_name, 400)

    tasks_to_run = range(num_tasks_in_suite)
    if args.task_id is not None:
        tasks_to_run = [args.task_id]

    total_episodes, total_successes = 0, 0
    latencies = []


    for task_id in tqdm.tqdm(tasks_to_run):
        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)        
        
    # Removed crashing block
    # Will inspect log for structure

        env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)

        task_episodes, task_successes = 0, 0
        
        for episode_idx in tqdm.tqdm(range(args.num_trials_per_task)):
            env.reset()
            env.reset()
            action_plan = collections.deque()
            obs = env.set_init_state(initial_states[episode_idx])
            
            t = 0
            replay_images = []
            
            while t < max_steps + args.num_steps_wait:
                try:
                    if t < args.num_steps_wait:
                        obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                        t += 1
                        continue

                    # Preprocessing (matches main.py)
                    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                    wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
                    
                    
                    img = resize_with_pad(img, args.resize_size, args.resize_size)
                    wrist_img = resize_with_pad(wrist_img, args.resize_size, args.resize_size)
                    
                    replay_images.append(img)

                    if not action_plan:
                        element = {
                            "observation/image": img,
                            "observation/wrist_image": wrist_img,
                            "observation/state": np.concatenate(
                                (
                                    obs["robot0_eef_pos"],
                                    _quat2axisangle(obs["robot0_eef_quat"]),
                                    obs["robot0_gripper_qpos"],
                                )
                            ),
                            "observation/joint_position": obs["robot0_joint_pos"],
                            "prompt": str(task_description),
                        }
                        

                        # Policy Inference
                        if torch.cuda.is_available():
                            torch.cuda.synchronize()
                        start_time = time.time()
                        result = policy.infer(element)
                        if torch.cuda.is_available():
                            torch.cuda.synchronize()
                        end_time = time.time()
                        latency_ms = (end_time - start_time) * 1000
                        latencies.append(latency_ms)
                        logging.info(f"Infer latency (ms): {latency_ms:.2f}")

                        # Convert from Torch to Numpy
                        # result['actions'] is usually a Numpy array (from Policy wrapper)
                        if hasattr(result["actions"], "detach"):
                             raw_actions = result["actions"][0].detach().cpu().numpy()
                        else:
                             # Can be [H, D] or [D]
                             raw_actions = result["actions"][0]
                        
                        # Ensure 2D [H, D]
                        # Ensure 2D [H, D]
                        if raw_actions.ndim == 1:
                            raw_actions = raw_actions[None, :]

                        action_plan.extend(raw_actions[: args.replan_steps])

                    action = action_plan.popleft()
                    # Slice to 7 dimensions if model was padded to 32
                    if len(action) > 7:
                        action = action[:7]
                    obs, reward, done, info = env.step(action.tolist())
                    
                    if done:
                        task_successes += 1
                        total_successes += 1
                        break
                    t += 1
                except Exception as e:
                    logging.error(f"Exception: {e}")
                    import traceback
                    traceback.print_exc()
                    break

            task_episodes += 1
            total_episodes += 1
            
            suffix = "success" if done else "failure"
            try:
                task_segment = task_description.replace(" ", "_")
                imageio.mimwrite(
                    pathlib.Path(args.video_out_path) / f"rollout_{task_segment}_{suffix}.mp4",
                    [np.asarray(x) for x in replay_images],
                    fps=10,
                )
            except Exception as e:
                logging.warning(f"Failed to save video: {e}")
            logging.warning(f"Result: {suffix}")

    print(f"Total Success Rate: {total_successes / total_episodes if total_episodes > 0 else 0}")

    # Print Metrics
    if latencies:
        latencies = np.array(latencies)
        print(f"Latency (ms): Mean={np.mean(latencies):.2f}, Median={np.median(latencies):.2f}, P99={np.percentile(latencies, 99):.2f}")
    
    if torch.cuda.is_available():
        max_mem = torch.cuda.max_memory_allocated() / 1024**3
        print(f"Max GPU Memory: {max_mem:.2f} GB")


def _get_libero_env(task, resolution, seed):
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)
    return env, task_description

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    try:
        eval_libero(tyro.cli(Args))
    except Exception as e:
        print(f"Failed: {e}")
        import traceback
        traceback.print_exc()
