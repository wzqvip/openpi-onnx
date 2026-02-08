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
LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256

@dataclasses.dataclass(frozen=True)
class ImageNormalize(transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        if "image" in data:
            for k in data["image"]:
                if isinstance(data["image"][k], np.ndarray):
                    # Convert to float32 and Normalize to [-1, 1]
                    # Logic: (x/255 - 0.5) / 0.5 = x/127.5 - 1.0
                    if data["image"][k].dtype == np.uint8:
                         data["image"][k] = (data["image"][k].astype(np.float32) / 127.5) - 1.0
                    elif data["image"][k].dtype == np.float32:
                         # Assume already float 0-1? If so, (x-0.5)/0.5.
                         # But usually it comes as uint8. 
                         # If it enters as float 0-1, we do (x-0.5)/0.5 = 2*x - 1.
                         # Just in case.
                         if np.max(data["image"][k]) <= 1.0:
                             data["image"][k] = data["image"][k] * 2.0 - 1.0
        return data

@dataclasses.dataclass(frozen=True)
class TransposeImage(transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        if "image" in data:
            for k in data["image"]:
                x = data["image"][k]
                # [H, W, C] -> [C, H, W]
                if isinstance(x, np.ndarray):
                    if x.ndim == 3:
                        data["image"][k] = np.transpose(x, (2, 0, 1))
                    elif x.ndim == 4: # [B, H, W, C] -> [B, C, H, W]
                        data["image"][k] = np.transpose(x, (0, 3, 1, 2))
        return data

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
        # Fallback to checking ~/.openpi/assets or similar if needed, or just fail
        # Try finding it in the user's home dir if local fails
        alt_assets = pathlib.Path.home() / "openpi" / "assets"
        if alt_assets.exists():
            try:
                norm_stats = _checkpoints.load_norm_stats(alt_assets, asset_id)
                logging.info(f"Loaded norm stats from alt path: {alt_assets}")
            except:
                pass
    
    if norm_stats is None:
        logging.warning(f"No norm stats found at {assets_path}. Using empty stats and injecting defaults.")
        norm_stats = {}
        # raise RuntimeError(f"No norm stats! checked {assets_path}")
    
    # OVERRIDE with local torch_norm_stats.json if present
    if pathlib.Path("torch_norm_stats.json").exists():
        logging.info("Loading local override stats from torch_norm_stats.json")
        import json
        from openpi.shared.normalize import NormStats
        with open("torch_norm_stats.json", "r") as f:
            override_stats = json.load(f)
        
        for k, v in override_stats.items():
            mean = np.array(v["mean"], dtype=np.float32)
            std = np.array(v["std"], dtype=np.float32)
            q01 = np.array(v["q01"], dtype=np.float32) if v.get("q01") is not None else None
            q99 = np.array(v["q99"], dtype=np.float32) if v.get("q99") is not None else None
            norm_stats[k] = NormStats(mean=mean, std=std, q01=q01, q99=q99)
    # Image normalization handled manually by ImageNormalize to [-1, 1]


    # Transforms
    model_type = _model.ModelType.PI0 
    libero_inputs = libero_policy.LiberoInputs(model_type=model_type) # Is this a transform? No, it's a wrapper usually.
    # Actually LiberoInputs logic is usually implicitly handled by the user code constructing the 'element' dict
    # or explicitly added.
    # In serve_onnx_policy.py it was just a comment.
    
    input_transforms = [
        *data_config.data_transforms.inputs,
        ImageNormalize(), # Convert uint8 [0,255] to float32 [-1,1] for TRT engine
        transforms.Normalize(unflatten_dict(norm_stats), use_quantiles=data_config.use_quantile_norm),
        # Filter out PadStatesAndActions because W8A16 engine expects 8-dim state (unpadded)
        # Filter out ResizeImages because we resize manually in the loop (and it breaks on CHW floats)
        *[t for t in data_config.model_transforms.inputs if not isinstance(t, (transforms.PadStatesAndActions, transforms.ResizeImages))],
    ]
    
    flat_stats = flatten_dict(norm_stats)
    output_stats_flat = {k: v for k, v in flat_stats.items() if "actions" in k}
    output_norm_stats = unflatten_dict(output_stats_flat)
    
    # [FIX] Pad action stats to 32 dimensions to match model output
    # The model outputs 32 dims (7 real + 25 embeddings/padding).
    # norm_stats has 7 dims. Unnormalize would fail or broadcast wrongly.
    if "actions" in output_norm_stats:
        act_stats = output_norm_stats["actions"]
        current_dim = act_stats.mean.shape[0]
        if current_dim < 32:
             print(f"DEBUG: Padding action stats from {current_dim} to 32")
             pad_len = 32 - current_dim
             # Pad mean with 0
             act_stats.mean = np.concatenate([act_stats.mean, np.zeros(pad_len, dtype=np.float32)])
             # Pad std with 1 (so unnorm is identity for extra dims)
             act_stats.std = np.concatenate([act_stats.std, np.ones(pad_len, dtype=np.float32)])
             # Pad quantiles if present (safe values)
             if hasattr(act_stats, "q01") and act_stats.q01 is not None:
                 act_stats.q01 = np.concatenate([act_stats.q01, np.full(pad_len, -1.0, dtype=np.float32)])
             if hasattr(act_stats, "q99") and act_stats.q99 is not None:
                 act_stats.q99 = np.concatenate([act_stats.q99, np.full(pad_len, 1.0, dtype=np.float32)])
    
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
        action_dim=32, # Set to 32 to match TRT engine (model expects padded actions) 
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
        
        # DEBUG: Check initial state vector
        s0 = initial_states[0]
        print(f"DEBUG [TRT] InitState[0]: shape={s0.shape}, mean={np.mean(s0)}, first10={s0[:10]}")
        
        for episode_idx in tqdm.tqdm(range(args.num_trials_per_task)):
            env.reset()
            env.reset()
            action_plan = collections.deque()
            obs = env.set_init_state(initial_states[episode_idx])
            
            print(f"DEBUG [TRT] Task ID: {task_id}, Desc: {task_description}")
            
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
                                        obs["robot0_eef_quat"]
                                    ),
                                    obs["robot0_gripper_qpos"],
                                )
                            ),
                            "prompt": str(task_description),
                        }
                        
                        # DEBUG: Print input stats (Unbatched view)
                        print(f"DEBUG [TRT] Image: shape={element['observation/image'].shape}, range=[{np.min(element['observation/image'])}, {np.max(element['observation/image'])}], mean={np.mean(element['observation/image'])}")
                        print(f"DEBUG [TRT] State: shape={element['observation/state'].shape}, first10={element['observation/state'][:10]}")
                        
                        # Note: Shim removed. Relying on correct norm_stats.json
                        
                        action_chunk = client.infer(element)["actions"]
                        if t < 20: # Limit spam
                             print(f"DEBUG Actions: min={np.min(action_chunk)}, max={np.max(action_chunk)}, mean={np.mean(action_chunk)}, has_nan={np.isnan(action_chunk).any()}")

                        action_plan.extend(action_chunk[: args.replan_steps])

                    action = action_plan.popleft()
                    # Slice to 7 dimensions (Libero standard) to drop padding
                    action = action[:7]
                    obs, reward, done, info = env.step(action.tolist())
                    if done:
                        task_successes += 1
                        total_successes += 1
                        break
                    t += 1
                except Exception as e:
                    logging.error(f"Error: {e}", exc_info=True)
                    break
            
            task_episodes += 1
            total_episodes += 1
            
            suffix = "success" if done else "failure"
            task_segment = task_description.replace(" ", "_")
            
            # EXPLICIT PRINT FOR DEBUGGING
            print(f"===== RESULT: Episode {episode_idx+1}/{args.num_trials_per_task}: {suffix.upper()} =====")
            
            imageio.mimwrite(pathlib.Path(args.video_out_path) / f"rollout_{task_segment}_{suffix}.mp4", [np.asarray(x) for x in replay_images], fps=10)
            logging.info(f"Episode {episode_idx+1}/{args.num_trials_per_task}: {suffix}")

        # EXPLICIT PRINT FOR DEBUGGING
        success_rate = float(task_successes)/float(task_episodes) if task_episodes > 0 else 0.0
        print(f"===== TASK {task_id} COMPLETE: {task_successes}/{task_episodes} ({success_rate:.2%}) =====")
        logging.info(f"Task {task_id} Success Rate: {task_successes}/{task_episodes} ({success_rate:.2f})")

def _get_libero_env(task, resolution, seed):
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)
    return env, task_description

def _quat2axisangle(quat):
    if quat[0] > 1.0: quat[0] = 1.0
    elif quat[0] < -1.0: quat[0] = -1.0
    den = np.sqrt(1.0 - quat[0] * quat[0])
    if math.isclose(den, 0.0): return np.zeros(3)
    return (quat[1:] * 2.0 * math.acos(quat[0])) / den

if __name__ == "__main__":
    eval_libero(tyro.cli(Args))
