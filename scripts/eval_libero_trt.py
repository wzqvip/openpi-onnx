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
import robosuite.utils.transform_utils as T
import numpy as np
import tqdm
import tyro

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

@dataclasses.dataclass
class Args:
    checkpoint_dir: str = "./checkpoints/pi05_libero_pytorch"
    config_name: str = "pi05_libero"
    host: str = "0.0.0.0"
    port: int = 8000
    resize_size: int = 224
    replan_steps: int = 5
    task_suite_name: str = "libero_spatial"
    num_steps_wait: int = 10
    num_trials_per_task: int = 3 # Small number for quick check, or 50 for full
    video_out_path: str = "data/libero/videos_trt"
    seed: int = 7
    task_id: int | None = None

def eval_libero(args: Args) -> None:
    logging.basicConfig(level=logging.INFO)
    np.random.seed(args.seed)

    # --- Setup Policy (Client Side) ---
    logging.info(f"Loading config: {args.config_name}")
    train_config = _config.get_config(args.config_name)
    model = train_config.model
    data_config = train_config.data.create(train_config.assets_dirs, model)
    asset_id = data_config.asset_id
    
    checkpoint_path = pathlib.Path(args.checkpoint_dir)
    assets_path = checkpoint_path / "assets"
    
    norm_stats = None
    if asset_id and assets_path.exists():
         try:
             norm_stats = _checkpoints.load_norm_stats(assets_path, asset_id)
             logging.info(f"Loaded norm stats for {asset_id}")
         except Exception as e:
             logging.error(f"Failed to load norm stats: {e}")
    if norm_stats is None:
        raise RuntimeError("No norm stats!")

    # Transforms
    model_type = _model.ModelType.PI0 
    libero_inputs = libero_policy.LiberoInputs(model_type=model_type) # Is this a transform? No, it's a wrapper usually.
    # Actually LiberoInputs logic is usually implicitly handled by the user code constructing the 'element' dict
    # or explicitly added.
    # In serve_onnx_policy.py it was just a comment.
    
    input_transforms = [
        *data_config.data_transforms.inputs,
        transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
        *data_config.model_transforms.inputs,
    ]
    
    flat_stats = flatten_dict(norm_stats)
    output_stats_flat = {k: v for k, v in flat_stats.items() if "actions" in k}
    output_norm_stats = unflatten_dict(output_stats_flat)
    
    output_transforms = [
        *data_config.model_transforms.outputs,
        transforms.Unnormalize(output_norm_stats, use_quantiles=data_config.use_quantile_norm),
        *data_config.data_transforms.outputs,
        libero_policy.LiberoOutputs(),
    ]
    
    # Initialize Remote TRT Policy
    client = tensorrt_remote_policy.TensorRTRemotePolicy(
        host=args.host,
        port=args.port,
        transforms=input_transforms,
        output_transforms=output_transforms,
        action_horizon=train_config.model.action_horizon,
        action_dim=train_config.model.action_dim, 
    )
    
    # --- Eval Loop (Copied from main.py) ---
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)

    # Max steps heuristic
    if args.task_suite_name == "libero_spatial": max_steps = 220
    elif args.task_suite_name == "libero_object": max_steps = 280
    elif args.task_suite_name == "libero_goal": max_steps = 300
    elif args.task_suite_name == "libero_10": max_steps = 520
    elif args.task_suite_name == "libero_90": max_steps = 400
    else: max_steps = 400

    total_episodes, total_successes = 0, 0
    tasks_to_run = range(num_tasks_in_suite)
    if args.task_id is not None: tasks_to_run = [args.task_id]

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

                    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                    wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
                    img = image_tools.convert_to_uint8(image_tools.resize_with_pad(img, args.resize_size, args.resize_size))
                    wrist_img = image_tools.convert_to_uint8(image_tools.resize_with_pad(wrist_img, args.resize_size, args.resize_size))
                    
                    replay_images.append(img)

                    if not action_plan:
                        element = {
                            "observation/image": img,
                            "observation/wrist_image": wrist_img,
                            "observation/state": np.concatenate(
                                (
                                    obs["robot0_eef_pos"],
                                    _quat2axisangle(
                                        T.quat_multiply(
                                            T.axisangle2quat(np.array([np.pi, 0, 0])),
                                            obs["robot0_eef_quat"],
                                        )
                                    ),
                                    obs["robot0_gripper_qpos"],
                                )
                            ),
                            "prompt": str(task_description),
                        }
                        
                        action_chunk = client.infer(element)["actions"]
                        action_plan.extend(action_chunk[: args.replan_steps])

                    action = action_plan.popleft()
                    obs, reward, done, info = env.step(action.tolist())
                    if done:
                        task_successes += 1
                        total_successes += 1
                        break
                    t += 1
                except Exception as e:
                    logging.error(f"Error: {e}")
                    break
            
            task_episodes += 1
            total_episodes += 1
            
            suffix = "success" if done else "failure"
            task_segment = task_description.replace(" ", "_")
            imageio.mimwrite(pathlib.Path(args.video_out_path) / f"rollout_{task_segment}_{suffix}.mp4", [np.asarray(x) for x in replay_images], fps=10)
            logging.info(f"Episode {episode_idx+1}/{args.num_trials_per_task}: {suffix}")

        logging.info(f"Task {task_id} Success Rate: {task_successes}/{task_episodes} ({float(task_successes)/float(task_episodes):.2f})")

def _get_libero_env(task, resolution, seed):
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)
    return env, task_description

def _quat2axisangle(quat):
    if quat[3] > 1.0: quat[3] = 1.0
    elif quat[3] < -1.0: quat[3] = -1.0
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0): return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den

if __name__ == "__main__":
    eval_libero(tyro.cli(Args))
