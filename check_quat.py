
import sys
import pathlib
sys.path.append(str(pathlib.Path("./third_party/libero").resolve()))
from libero.libero import benchmark
from libero.libero.envs import OffScreenRenderEnv
import numpy as np

def check():
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict["libero_spatial"]()
    task = task_suite.get_task(0)
    from libero.libero import get_libero_path
    bddl_root = get_libero_path("bddl_files")
    task_bddl_file = pathlib.Path(bddl_root) / task.problem_folder / task.bddl_file
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": 256, "camera_widths": 256}
    env = OffScreenRenderEnv(**env_args)
    env.reset()
    obs = env.set_init_state(task_suite.get_task_init_states(0)[0])
    
    quat = obs["robot0_eef_quat"]
    print(f"Quat: {quat}")
    print(f"Type: {type(quat)}")
    print(f"Shape: {quat.shape}")
    
    # Check identity
    # Usually [0, 0, 0, 1] or [1, 0, 0, 0]
    # If [x, y, z, w], w=1 means identity.
    # If [w, x, y, z], w=1 means identity.
    # We can infer from values.
    
    env.close()

if __name__ == "__main__":
    check()
