
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
    # Robosuite convention: [x, y, z, w] or [w, x, y, z]?
    # Code in main.py uses quat[3] as w.
    if quat[3] > 1.0: quat[3] = 1.0
    elif quat[3] < -1.0: quat[3] = -1.0
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0): return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den

def eval_libero(args: Args) -> None:
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    logging.info(f"Loading config: {args.config}")
    
    # Hack to disable torch.compile for stability
    torch.compile = lambda x, **k: x
    
    train_config = _config.get_config(args.config)
    
    device = "cpu" if args.force_cpu else ("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Loading policy from {args.checkpoint} on {device}...")
    
    policy = policy_config.create_trained_policy(
        train_config, 
        args.checkpoint, 
        pytorch_device=device
    )

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

    for task_id in tqdm.tqdm(tasks_to_run):
        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)
        env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)

        task_episodes, task_successes = 0, 0
        for episode_idx in tqdm.tqdm(range(args.num_trials_per_task)):
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
                            "prompt": str(task_description),
                        }
                        
                        # Policy Inference
                        result = policy.infer(element)
                        action_chunk = result["actions"]
                        action_plan.extend(action_chunk[: args.replan_steps])

                    action = action_plan.popleft()
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
            task_segment = task_description.replace(" ", "_")
            imageio.mimwrite(
                pathlib.Path(args.video_out_path) / f"rollout_{task_segment}_{suffix}.mp4",
                [np.asarray(x) for x in replay_images],
                fps=10,
            )
            logging.info(f"Result: {suffix}")

    logging.info(f"Total Success Rate: {total_successes / total_episodes if total_episodes > 0 else 0}")

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
