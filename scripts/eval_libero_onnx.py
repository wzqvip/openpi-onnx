
import collections
import dataclasses
import logging
import math
import pathlib
import sys
import onnxruntime as ort

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
import time

from openpi_client import image_tools
from openpi.policies import policy as _policy
from openpi.policies import libero_policy
from openpi.training import config as _config
from openpi.training import checkpoints as _checkpoints
from openpi import transforms
from openpi.models import model as _model
from openpi.transforms import flatten_dict, unflatten_dict
from openpi.shared import download

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256

class OnnxLocalPolicy(_policy.Policy):
    def __init__(
        self,
        model_path: str,
        transforms=None,
        output_transforms=None,
        action_horizon: int = 10,
        action_dim: int = 7
    ):
        self.transforms = transforms
        self.output_transforms = output_transforms
        self.action_horizon = action_horizon
        self.action_dim = action_dim
        
        logging.info(f"Loading ONNX session: {model_path}")
        self.sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        self.sess_inputs = {i.name: i for i in self.sess.get_inputs()}

    def infer(self, obs):
        # 1. Apply Input Transforms
        def decode_keys(d):
            if isinstance(d, dict):
                return {k.decode("utf-8") if isinstance(k, bytes) else k: decode_keys(v) for k, v in d.items()}
            return d
            
        inputs = decode_keys(obs)
        if self.transforms:
            for transform in self.transforms:
                inputs = transform(inputs)

        # 2. Preprocess for ONNX (Batching + Mapping)
        def add_batch(x):
            return np.expand_dims(np.array(x), axis=0)

        feed_dict = {}
        
        # Prepare Transposed Images (B, C, H, W)
        base_img = None
        left_img = None
        right_img = None
        
        if "image" in inputs:
            if "base_0_rgb" in inputs["image"]: base_img = add_batch(inputs["image"]["base_0_rgb"]).transpose(0, 3, 1, 2)
            if "left_wrist_0_rgb" in inputs["image"]: left_img = add_batch(inputs["image"]["left_wrist_0_rgb"]).transpose(0, 3, 1, 2)
            if "right_wrist_0_rgb" in inputs["image"]: right_img = add_batch(inputs["image"]["right_wrist_0_rgb"]).transpose(0, 3, 1, 2)
        
        # Mapping Logic
        mapping = {
            "base_0_rgb": ["base_0_rgb", "observation.images.base_0_rgb"],
            "left_wrist_0_rgb": ["left_wrist_0_rgb", "observation.images.left_wrist_0_rgb"],
            "right_wrist_0_rgb": ["right_wrist_0_rgb", "observation.images.right_wrist_0_rgb"],
            "state": ["state", "observation.state"],
            "tokenized_prompt": ["tokenized_prompt", "observation.tokenized_prompt"],
            "tokenized_prompt_mask": ["tokenized_prompt_mask", "observation.tokenized_prompt_mask"],
            "noise": ["noise"]
        }

        # Helper to assign if found
        def map_input(key, value):
            if value is None: return
            for name in mapping[key]:
                if name in self.sess_inputs:
                    feed_dict[name] = value
                    return
        
        # Assign Images
        map_input("base_0_rgb", base_img)
        map_input("left_wrist_0_rgb", left_img)
        map_input("right_wrist_0_rgb", right_img)
        
        # Assign State
        if "state" in inputs:
            map_input("state", add_batch(inputs["state"]))
            
        # Assign Prompts
        if "tokenized_prompt" in inputs:
            map_input("tokenized_prompt", add_batch(inputs["tokenized_prompt"]).astype(np.int32))
        if "tokenized_prompt_mask" in inputs:
            map_input("tokenized_prompt_mask", add_batch(inputs["tokenized_prompt_mask"]))
            
        # Generate Noise
        B = 1
        noise = np.random.randn(B, self.action_horizon, self.action_dim).astype(np.float32)
        map_input("noise", noise)

        # 3. Inference
        outputs_onnx = self.sess.run(["actions"], feed_dict)[0]
        
        # 4. Post-process
        actions = outputs_onnx[0] # Unbatch
        
        outputs = {
            "actions": actions,
            "state": inputs.get("state"), 
        }
        
        if self.output_transforms:
            for transform in self.output_transforms:
                outputs = transform(outputs)

        return outputs

@dataclasses.dataclass
class Args:
    checkpoint_dir: str = "./checkpoints/pi05_libero_pytorch"
    onnx_model_path: str = "dist/final_w8a16/model.w8a16.onnx"
    config_name: str = "pi05_libero"
    resize_size: int = 224
    replan_steps: int = 5
    task_suite_name: str = "libero_spatial"
    num_steps_wait: int = 10
    num_trials_per_task: int = 5 
    video_out_path: str = "data/libero/videos_onnx"
    seed: int = 7
    task_id: int | None = None

def eval_libero(args: Args) -> None:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    np.random.seed(args.seed)

    # --- Setup Policy Config ---
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
    
    # Initialize Local ONNX Policy
    client = OnnxLocalPolicy(
        model_path=args.onnx_model_path,
        transforms=input_transforms,
        output_transforms=output_transforms,
        action_horizon=train_config.model.action_horizon,
        action_dim=train_config.model.action_dim, 
    )
    
    # --- Eval Loop ---
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
    inference_times = []
    
    tasks_to_run = range(num_tasks_in_suite)
    if args.task_id is not None: tasks_to_run = [args.task_id]

    print(f"\nSTARTING EVALUATION: {args.task_suite_name}")
    print(f"Model: {args.onnx_model_path}")
    print(f"Process: {len(tasks_to_run)} tasks, {args.num_trials_per_task} trials each\n")

    for task_id in tqdm.tqdm(tasks_to_run):
        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)
        env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)
        
        task_episodes, task_successes = 0, 0
        for episode_idx in range(args.num_trials_per_task):
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
                        
                        start_time = time.time()
                        action_chunk = client.infer(element)["actions"]
                        end_time = time.time()
                        inference_times.append(end_time - start_time)
                        
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
                    import traceback
                    traceback.print_exc()
                    break
            
            task_episodes += 1
            total_episodes += 1
            
            suffix = "success" if done else "failure"
            # task_segment = task_description.replace(" ", "_")
            # imageio.mimwrite(pathlib.Path(args.video_out_path) / f"rollout_{task_segment}_{suffix}.mp4", [np.asarray(x) for x in replay_images], fps=10)
            logging.info(f"Task {task_id} Trial {episode_idx+1}: {suffix}")

        logging.info(f"Task {task_id} SR: {task_successes}/{task_episodes} ({float(task_successes)/float(task_episodes):.2f})")

    avg_latency = np.mean(inference_times) * 1000
    print("\n------------------------------------------------")
    print("EVALUATION COMPLETE")
    print(f"Total Success Rate: {total_successes}/{total_episodes} ({float(total_successes)/float(total_episodes):.2f})")
    print(f"Average Inference Latency (Local ONNX): {avg_latency:.2f} ms")
    print("------------------------------------------------\n")

def _get_libero_env(task, resolution, seed):
    task_description = task.language
    # Hardcode local path to ensure we find the files
    bddl_root = pathlib.Path("./third_party/libero/libero/libero/bddl_files").resolve()
    task_bddl_file = bddl_root / task.problem_folder / task.bddl_file
    
    if not task_bddl_file.exists():
        logging.error(f"BDDL File not found: {task_bddl_file}")
        # Fallback to get_libero_path if local fails?
        # task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
        
    env_args = {"bddl_file_name": str(task_bddl_file), "camera_heights": resolution, "camera_widths": resolution}
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
