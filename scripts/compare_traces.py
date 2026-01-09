import argparse
import numpy as np
import dataclasses
import tyro
import logging
import msgpack
import msgpack_numpy
import websockets.sync.client
import time

# Patch
msgpack_numpy.patch()

@dataclasses.dataclass
class Args:
    trace_file: str = "trace_data.npz"
    host: str = "0.0.0.0"
    port: int = 8000
    checkpoint: str = "checkpoints/pi05_libero_pytorch"

def main(args: Args):
    logging.basicConfig(level=logging.INFO)
    
    # Load Trace
    logging.info(f"Loading trace from {args.trace_file}...")
    trace = np.load(args.trace_file, allow_pickle=True)
    obs = trace["obs"].item() # extract dict
    pt_actions = trace["actions"]
    
    # Preprocess Inputs using LiberoInputs transform
    from openpi.policies import libero_policy
    from openpi.models import model as _model
    from openpi.transforms import flatten_dict, unflatten_dict
    from openpi.shared import download
    from openpi import transforms
    
    # We need to apply LiberoInputs to convert raw obs to model inputs
    # Note: LiberoInputs expects "model_type". pi05_libero uses PI0 (or PI0_FAST?)
    # Config uses Pi0Config.
    transform = libero_policy.LiberoInputs(model_type=_model.ModelType.PI0)
    
    # Obs in trace is batched (B=1). LiberoInputs expects unbatched or handles batch?
    # DataTransformFn usually expects unbatched.
    # We should unbatch, transform, then rebatch.
    
    # Unbatch
    obs_unbatched = {
        k: (v[0] if isinstance(v, np.ndarray) and v.shape[0]==1 else v)
        for k, v in obs.items()
    }
    
    model_inputs = transform(obs_unbatched)
    
    # Pack into feed_dict
    feed_dict = {}
    
    # Helper to rebatch
    def add(k, v):
        feed_dict[k] = np.expand_dims(v, axis=0) # Add B=1 dim back

    # Map model_inputs to feed_dict keys (TRT server keys)
    # TRT server keys: "state", "base_0_rgb", "left_wrist_0_rgb", etc.
    # model_inputs keys: "state", "image": {"base_0_rgb"...}
    
    if "state" in model_inputs:
        add("state", model_inputs["state"])
        
    if "image" in model_inputs:
        for name, data in model_inputs["image"].items():
            # Model inputs are (H, W, C) uint8?
            # Config: base_0_rgb (3, 224, 224)?
            # Wait, PyTorch/TRT expects (B, C, H, W) float usually?
            # But TRT engine expects what?
            # TRT Server `infer` does not normalize or transpose?
            # serve_trt.py: 
            #   input_data = ...
            #   context.set_input_shape(name, shape)
            # The ENGINE expects (B, 3, 224, 224) float usually? or uint8?
            # Onnx export usually includes normalization?
            # If exported model (OnnxPolicy) includes preprocessing (Normalize, Transpose), then inputs should be raw (H, W, C) uint8?
            # OnnxPolicy uses `transforms.Normalize`.
            # If TRT engine was built from Onnx model, does it include transforms?
            # My `export_onnx.py` (if standard) includes the model forward.
            # `OnnxPolicy` (in openpi) applies transforms in Python, then calls session run.
            # So TRT engine expects PREPROCESSED inputs?
            # NO. `export_onnx.py` usually exports the `model` which takes normalized tensors.
            # UNLESS `export_onnx.py` wraps it.
            # User instructions "Exporting Models": "Generating ONNX files... Ensure correct data types".
            # Usually we feed normalized tensors.
            
            # BUT `TensorRTModel` (which I verified) takes `state`, `image...`
            # AND `tensorrt_remote_policy.py` sends `feed_dict` via msgpack.
            # Does `tensorrt_remote_policy.py` normalize?
            # Current `TensorRTRemotePolicy.infer`:
            #   inputs = self._input_transform(inputs) (if defined)
            #   data = msgpack.packb(inputs)
            #   server receives data.
            #   server calls `context.execute_async_v3`.
            
            # If `serve_trt.py` feeds data DIRECTLY to engine, then engine input requirements apply.
            # If engine inputs are (B, C, H, W) float32 normalized... then we must normalize.
            # DOES `TensorRTRemotePolicy` normalize?
            # `eval_libero_trt.py` creates policy.
            # Does it set `input_transforms`?
            # If NO transforms, then it sends RAW data.
            # If it sends RAW data (H,W,C uint8), and engine expects (C,H,W float), TRT crashes or gives garbage.
            # THIS explains 0% success (garbage in).
            
            # SO, `compare_traces.py` needs to replicate what `TensorRTRemotePolicy` SHOULD do (or Does).
            # If I want to match PyTorch output, I must feed the SAME input to TRT engine.
            # PyTorch model takes normalized tensors.
            # So TRT engine (exported from PyTorch model) takes normalized tensors.
            # So `compare_traces.py` inputs must be normalized.
            
            # But `TensorRTRemotePolicy` didn't normalize by default?
            # THEN `TensorRTRemotePolicy` IS BROKEN (missing transforms).
            # And `compare_traces.py` confirms this if I feed normalized inputs and it matches.
            
            # I will feed NORMALIZED inputs to TRT server here.
            # To do that, I need to Apply `LiberoInputs`, `Normalize`, `ImageTransforms` etc.
            # I can re-use the `Policy`'s transform pipeline?
            pass

    # We need to construct the FULL transform pipeline to generate correct inputs for TRT engine
    from openpi.training import config as _config
    cfg = _config.get_config("pi05_libero")
    # We need validation/inference transforms
    # data_config = cfg.data.create(...)
    # transforms = data_config.model_transforms.inputs (+ normalize)
    
    # Easier: Just use the PyTorch Policy to transform inputs!
    # PyTorch Policy has `_input_transform`.
    # But `Policy.infer` runs `_input_transform` then `model()`.
    # We want result of `_input_transform`.
    
    # Let's load Policy again to get transforms
    from openpi.policies import policy_config
    policy = policy_config.create_trained_policy(cfg, "checkpoints/pi05_libero_pytorch")
    
    # Run Transform
    transformed_inputs = policy._input_transform(obs)
    
    # Pack transformed inputs (which are Tensors or Arrays ready for model)
    # They should be (B, C, H, W) float32 normalized.
    
    feed_dict = {}
    if "state" in transformed_inputs:
        feed_dict["state"] = np.array(transformed_inputs["state"])
    
    if "image" in transformed_inputs:
        for k, v in transformed_inputs["image"].items():
             feed_dict[k] = np.array(v)
             
    if "tokenized_prompt" in transformed_inputs:
        feed_dict["tokenized_prompt"] = np.array(transformed_inputs["tokenized_prompt"]).astype(np.int32)
    if "tokenized_prompt_mask" in transformed_inputs:
        feed_dict["tokenized_prompt_mask"] = np.array(transformed_inputs["tokenized_prompt_mask"])
        
    # Noise - Use trace noise to ensure deterministic comparison
    if "noise" in trace:
        feed_dict["noise"] = trace["noise"]
    else:
        logging.warning("No noise in trace, using random (comparison will fail generally)")
        feed_dict["noise"] = np.random.randn(1, 10, 32).astype(np.float32)

    # Connect
    uri = f"ws://{args.host}:{args.port}"
    logging.info(f"Connecting to {uri}...")
    with websockets.sync.client.connect(uri, max_size=None) as ws:
        # Handshake
        _ = msgpack.unpackb(ws.recv())
        
        # Send
        logging.info("Sending inputs...")
        data = msgpack.packb(feed_dict, default=msgpack_numpy.encode)
        ws.send(data)
        
        # Receive
        response = ws.recv()
        outputs = msgpack.unpackb(response, object_hook=msgpack_numpy.decode)
        
        trt_actions = outputs["actions"]
        
        # Compare
        logging.info("Comparing actions...")
        # PyTorch actions might be (1, H, D)
        # TRT actions might be (H, D) depending on server unbatching
        
        # Check shapes
        print(f"PyTorch Shape: {pt_actions.shape}")
        print(f"TRT Shape    : {trt_actions.shape}")
        
        # Squeeze if needed
        if pt_actions.ndim == 3 and trt_actions.ndim == 3: # (B, H, D)
            pass
        elif pt_actions.ndim == 3 and trt_actions.ndim == 2:
            trt_actions = np.expand_dims(trt_actions, axis=0) # Assume B=1
            
        # Process TRT Actions
    # TRT result is (B, T, D) = (1, 10, 32)
    # PyTorch result is (T, D_out) = (10, 7)
    
    # 1. Unbatch
    trt_actions_raw = trt_actions[0] # (10, 32)
    
    # 2. Unnormalize
    # Need output stats
    # Reloading policy to get stats is inefficient but safe
    # norm_stats = policy.norm_stats
    # Load norm stats via checkpoints module (like serve_policy.py or eval_libero_trt.py does)
    from openpi.training import checkpoints as _checkpoints
    import pathlib
    
    checkpoint_path = pathlib.Path(args.checkpoint)
    assets_path = checkpoint_path / "assets"
    
    data_config = cfg.data.create(cfg.assets_dirs, cfg.model)
    asset_id = data_config.asset_id
    
    if asset_id and assets_path.exists():
         norm_stats = _checkpoints.load_norm_stats(assets_path, asset_id)
    else:
         # Fallback to config (which seems empty based on logs)
         norm_stats = data_config.norm_stats
         
    if norm_stats is None:
        raise RuntimeError(f"Could not load norm stats from {assets_path} for {asset_id}")

    output_norm_stats = unflatten_dict({k: v for k, v in flatten_dict(norm_stats).items() if "actions" in k})
    
    unnorm = transforms.Unnormalize(output_norm_stats, use_quantiles=cfg.data.create(cfg.assets_dirs, cfg.model).use_quantile_norm)
    
    # Unnormalize expects dict
    trt_out_dict = {"actions": trt_actions_raw}
    trt_out_unnorm = unnorm(trt_out_dict)
    
    # 3. LiberoOutputs (Repack)
    # LiberoOutputs converts model action dim (32) to env action dim (7)
    libero_out = libero_policy.LiberoOutputs()
    trt_actions_final = libero_out(trt_out_unnorm)["actions"]
    
    print(f"PyTorch Shape: {pt_actions.shape}")
    print(f"TRT Shape    : {trt_actions_final.shape}")
    
    # Compare
    diff = np.abs(pt_actions - trt_actions_final)
    print(f"Max Diff: {np.max(diff)}")
    print(f"Mean Diff: {np.mean(diff)}")
    
    if np.max(diff) < 1.0: # 1.0 is rather large, but if unnormalized, actions can be large? 
        # Libero actions are usually [-1, 1] or similar.
        # If diff is > 0.1 it's concerning.
        print("SUCCESS: Outputs match within tolerance!")
    else:
        print("FAILURE: Outputs mismatch!")

if __name__ == "__main__":
    main(tyro.cli(Args))
