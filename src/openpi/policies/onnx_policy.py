import logging
import time
import sys
from typing import Any, Sequence

import numpy as np
import onnxruntime as ort
from overrides import overrides

from openpi import transforms as _transforms
from openpi.policies import policy as _policy
from openpi.shared import download

class OnnxPolicy(_policy.BasePolicy):
    def __init__(
        self,
        onnx_path: str,
        *,
        transforms: Sequence[_transforms.DataTransformFn] = (),
        output_transforms: Sequence[_transforms.DataTransformFn] = (),
        metadata: dict[str, Any] | None = None,
        action_horizon: int = 50, # Default or from config
        action_dim: int = 7,      # Default or from config
    ):
        self._input_transform = _transforms.compose(transforms)
        self._output_transform = _transforms.compose(output_transforms)
        self._metadata = metadata or {}
        
        self.action_horizon = action_horizon
        self.action_dim = action_dim

        # Initialize ONNX Runtime Session
        logging.info(f"Loading ONNX model from: {onnx_path}")
        onnx_path = download.maybe_download(onnx_path)
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self.sess = ort.InferenceSession(str(onnx_path), providers=providers)
        
        self.input_names = [i.name for i in self.sess.get_inputs()]
        self.output_names = [o.name for o in self.sess.get_outputs()]
        logging.info(f"ONNX Model Input Names: {self.input_names}")

    @overrides
    def infer(self, obs: dict) -> dict:
        # Helper to decode bytes keys
        def decode_keys(d):
            if isinstance(d, dict):
                return {k.decode("utf-8") if isinstance(k, bytes) else k: decode_keys(v) for k, v in d.items()}
            return d
            
        obs = decode_keys(obs)
        with open("debug_keys.txt", "w") as f:
            f.write(f"Keys: {list(obs.keys())}\n")
            f.write(f"Sample 'observation/image' type: {type(obs.get('observation/image', None))}\n")
            import pprint
            f.write(pprint.pformat(obs, depth=2)) # Caution with image data size
        
        # Preprocess inputs
        inputs = self._input_transform(obs)
        
        # Add batch dimension (B=1)
        # Note: transforms might produce nested dicts (e.g. image: {base: ...})
        # We need to extract them matching ONNX signature.
        
        # Assuming Policy logic for flattening/batching
        # The transforms produce single-item arrays (no batch dim) usually.
        
        # Helper to batchify
        def add_batch(x):
            return np.expand_dims(np.array(x), axis=0)
            
        # Extract inputs mapped to ONNX names
        # Based on export_onnx.py signature:
        # base_0_rgb, left_wrist_0_rgb, right_wrist_0_rgb, state, tokenized_prompt, tokenized_prompt_mask, noise
        
        feed_dict = {}
        
        # Images need special handling if they are in a dict
        if "image" in inputs:
            # Transpose to (B, C, H, W)
            feed_dict["base_0_rgb"] = add_batch(inputs["image"]["base_0_rgb"]).transpose(0, 3, 1, 2)
            feed_dict["left_wrist_0_rgb"] = add_batch(inputs["image"]["left_wrist_0_rgb"]).transpose(0, 3, 1, 2)
            feed_dict["right_wrist_0_rgb"] = add_batch(inputs["image"]["right_wrist_0_rgb"]).transpose(0, 3, 1, 2)
        
        feed_dict["state"] = add_batch(inputs["state"])
        feed_dict["tokenized_prompt"] = add_batch(inputs["tokenized_prompt"])
        feed_dict["tokenized_prompt_mask"] = add_batch(inputs["tokenized_prompt_mask"])
        
        # Generate Noise
        B = 1
        noise = np.random.randn(B, self.action_horizon, self.action_dim).astype(np.float32)
        # Check current data types of model inputs to match precision
        model_type = feed_dict["state"].dtype
        if model_type == np.float16:
            noise = noise.astype(np.float16)

        feed_dict["noise"] = noise
        
        # DEBUG: Check input types
        for k, v in feed_dict.items():
            if hasattr(v, "dtype"):
                logging.debug(f"Input {k} dtype: {v.dtype}")
                # Safeguard: Cast to float32 if not float (except for prompt/mask which are usually int/bool)
                if k in ["base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb", "state", "noise"]:
                    if not np.issubdtype(v.dtype, np.floating):
                        logging.warning(f"Casting {k} from {v.dtype} to float32 (Normalize skipped?)")
                        feed_dict[k] = v.astype(np.float32)
        
        # Cast inputs to match model expectation if needed (e.g. if transforms return float64)
        for k, v in feed_dict.items():
            if hasattr(v, "dtype") and np.issubdtype(v.dtype, np.floating) and v.dtype != np.float32:
                 logging.debug(f"Casting {k} from {v.dtype} to float32")
                 feed_dict[k] = v.astype(np.float32)

        # Cast tokenized_prompt to int32 (ONNX expects int32)
        if "tokenized_prompt" in feed_dict:
             feed_dict["tokenized_prompt"] = feed_dict["tokenized_prompt"].astype(np.int32)

        start_time = time.monotonic()
        
        # Run inference
        outputs_list = self.sess.run(self.output_names, feed_dict)
        actions = outputs_list[0] # "actions" is likely index 0
        
        model_time = time.monotonic() - start_time
        
        # Unbatch
        outputs = {
            "actions": actions[0],
            "state": inputs["state"], # Pass through state for output transforms if needed
        }
        
        outputs = self._output_transform(outputs)
        outputs["policy_timing"] = {"infer_ms": model_time * 1000}
        
        return outputs

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata
