import os
import sys
import json
import numpy as np
import pathlib
import tqdm
from unittest.mock import MagicMock

# Mock lerobot to avoid imports if not needed or problematic
sys.modules["lerobot"] = MagicMock()
sys.modules["lerobot.common"] = MagicMock()
sys.modules["lerobot.common.datasets"] = MagicMock()
sys.modules["lerobot.common.datasets.lerobot_dataset"] = MagicMock()

# Add libero to path
sys.path.append(str(pathlib.Path("./third_party/libero").resolve()))

from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import math

def _quat2axisangle(quat):
    if quat[0] > 1.0: quat[0] = 1.0
    elif quat[0] < -1.0: quat[0] = -1.0
    den = np.sqrt(1.0 - quat[0] * quat[0])
    if math.isclose(den, 0.0): return np.zeros(3)
    return (quat[1:] * 2.0 * math.acos(quat[0])) / den

def main():
    task_suite_name = "libero_spatial"
    resolution = 256
    seed = 7
    
    print(f"Loading benchmark: {task_suite_name}")
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[task_suite_name]()
    num_tasks = task_suite.n_tasks
    
    all_states = []
    
    print(f"Collecting initial states from {num_tasks} tasks...")
    # Iterate all tasks
    for task_id in tqdm.tqdm(range(num_tasks)):
        task = task_suite.get_task(task_id)
        # init_states is a list of numpy arrays (sim state) which are obscure
        initial_states = task_suite.get_task_init_states(task_id)
        
        # We need to get the OBSERVATION state (8-dim), not the simulator state
        # So we must spin up the env briefly
        
        task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
        env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
        # Use EGL if available
        if "MUJOCO_GL" not in os.environ:
             os.environ["MUJOCO_GL"] = "egl"
             
        env = OffScreenRenderEnv(**env_args)
        env.seed(seed)
        
        # Check a few init states per task (e.g. 5)
        for i in range(min(5, len(initial_states))):
            obs = env.set_init_state(initial_states[i])
            # Construct 8-dim state
            state_vec = np.concatenate(
                (
                    obs["robot0_eef_pos"],
                    _quat2axisangle(obs["robot0_eef_quat"]),
                    obs["robot0_gripper_qpos"],
                )
            )
            all_states.append(state_vec)
            
        env.close()
        
    all_states = np.array(all_states)
    print(f"Collected {len(all_states)} samples.")
    
    mean = np.mean(all_states, axis=0).tolist()
    std = np.std(all_states, axis=0).tolist()
    min_val = np.min(all_states, axis=0).tolist()
    max_val = np.max(all_states, axis=0).tolist()
    
    print("Mean:", mean)
    print("Std:", std)
    
    # Load template to preserve actions stats
    if os.path.exists("torch_norm_stats.json.bak"):
        with open("torch_norm_stats.json.bak", "r") as f:
            template = json.load(f)
    elif os.path.exists("torch_norm_stats.json"):
        with open("torch_norm_stats.json", "r") as f:
            template = json.load(f)
    else:
        template = {"state": {}, "actions": {}}
        
    template["state"]["mean"] = mean
    # We can use computed std, or keep original std if we trust the scale but not the shift.
    # Usually safer to use computed std for the new domain too.
    template["state"]["std"] = std
    # Clear quantiles or compute them?
    # For now, let's keep q01/q99 from template if we don't compute them, 
    # OR compute them. The eval script prefers q01/q99 if use_quantiles=True.
    
    q01 = np.quantile(all_states, 0.01, axis=0).tolist()
    q99 = np.quantile(all_states, 0.99, axis=0).tolist()
    
    template["state"]["q01"] = q01
    template["state"]["q99"] = q99
    
    with open("torch_norm_stats.json", "w") as f:
        json.dump(template, f, indent=2)
        
    print("Saved calibrated stats to torch_norm_stats.json")

if __name__ == "__main__":
    main()
