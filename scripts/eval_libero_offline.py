
import h5py
import numpy as np
import onnxruntime as ort
import os
import glob
from PIL import Image

# Config
DATASET_PATH = "/home/taco/LIBERO/libero/datasets/libero_spatial"
ONNX_MODEL_PATH = "./dist/final_w8a16_new/model.w8a16.onnx"
MAX_EPISODES = 1  # Speed up for quick check
RESIZE_SHAPE = (224, 224)

def load_episodes(dataset_dir):
    files = glob.glob(os.path.join(dataset_dir, "*.hdf5"))
    files.sort()
    return files

def resize_image(img_array):
    # img_array: (128, 128, 3)
    img = Image.fromarray(img_array)
    img = img.resize(RESIZE_SHAPE, Image.BILINEAR)
    return np.array(img, dtype=np.float32) # Standardize to float32 0-255

def get_state(demo, idx):
    # Construct state vector. 
    # Libero usually: joint_states (7) + gripper_states (2) -> use first gripper val?
    # Or matches convert script which yielded 8?
    # We will grab 7 joints + 1 gripper.
    joints = demo["obs/joint_states"][idx] # (7,)
    gripper = demo["obs/gripper_states"][idx] # (2,)
    # Use first gripper dim (width?)
    state = np.concatenate([joints, gripper[:1]]) # (8,)
    
    # Pad to 32
    padded = np.zeros(32, dtype=np.float32)
    padded[:8] = state
    return padded

def main():
    print(f"Loading model: {ONNX_MODEL_PATH}")
    sess = ort.InferenceSession(ONNX_MODEL_PATH, providers=["CPUExecutionProvider"])
    
    files = load_episodes(DATASET_PATH)
    print(f"Found {len(files)} files. Evaluating on first {MAX_EPISODES}...")
    
    total_mse = 0.0
    total_steps = 0
    
    for i, fpath in enumerate(files[:MAX_EPISODES]):
        print(f"Eval {os.path.basename(fpath)}...")
        with h5py.File(fpath, "r") as f:
            # Libero hdf5 has 'data/demo_X'
            demos = list(f["data"].keys())
            # Eval on first demo in file? 
            # Usually one file = 10 or 50 demos.
            # We'll calculate MSE over ALL steps in the first demo of this file.
            demo_key = demos[0]
            demo = f["data"][demo_key]
            
            actions = demo["actions"][:] # (T, 7)
            length = actions.shape[0]
            
            agentview = demo["obs/agentview_rgb"]
            eye_hand = demo["obs/eye_in_hand_rgb"]
            
            # Prepare batch of 1
            for t in range(length):
                # 1. Prepare Inputs
                # Images
                img_base = resize_image(agentview[t])
                img_wrist = resize_image(eye_hand[t])
                
                # Image format: (1, 3, 224, 224)
                # Inputs are float32, usually not normalized? 
                # OpenPi models usually expect uint8 0-255? 
                # Verify_accuracy used float32 randn. Export used float32.
                # If exported model used floats, we pass floats.
                # However, if it contains normalization layers, passing 0-255 is fine.
                # We'll pass float32 (0-255).
                
                inp_base = np.transpose(img_base, (2, 0, 1))[None, ...].astype(np.float32)
                inp_wrist = np.transpose(img_wrist, (2, 0, 1))[None, ...].astype(np.float32)
                inp_right_wrist = np.zeros_like(inp_base)
                
                # State
                state_vec = get_state(demo, t)[None, ...] # (1, 32)
                
                # Prompt (Dummy)
                # "pick up the black bowl..."
                # We can't tokenise dynamically easily without tokenizer.
                # We'll use a dummy prompt (zeros or random) same as verification?
                # export script used random prompt.
                # For "Exact" accuracy we should tokenize properly.
                # But avoiding dependency on tokenizer for "quick" offlinetest.
                # We'll use zeros or random. The prompt guides the task.
                # Since task is consistent per file, maybe constant noise isn't too bad?
                # We'll use Zeros. 
                prompt = np.zeros((1, 200), dtype=np.int32) 
                prompt_mask = np.ones((1, 200), dtype=bool)
                
                # Noise
                # Pi0 is diffusion/flow match?
                # We need to sample noise.
                # To compare expectations, we might want deterministic noise?
                # But Pi0 is conditional generative.
                # We pass noise.
                noise = np.random.randn(1, 10, 32).astype(np.float32)
                
                # Run ONNX
                ort_inputs = {
                    "observation.images.base_0_rgb": inp_base,
                    "observation.images.left_wrist_0_rgb": inp_wrist,
                    "observation.images.right_wrist_0_rgb": inp_right_wrist,
                    "observation.state": state_vec,
                    "observation.tokenized_prompt": prompt,
                    "observation.tokenized_prompt_mask": prompt_mask,
                    "noise": noise
                }
                
                res = sess.run(["actions"], ort_inputs)[0] # (1, 10, 32)
                
                # Compare step 0
                pred_act = res[0, 0, :7] # Slice 7
                gt_act = actions[t]
                
                mse = np.mean((pred_act - gt_act)**2)
                total_mse += mse
                total_steps += 1
                
    print(f"Total Steps: {total_steps}")
    print(f"Average MSE: {total_mse / total_steps}")

if __name__ == "__main__":
    main()
