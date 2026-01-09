import logging
from typing import Any, Dict, List, Optional
import msgpack
import msgpack_numpy
import websockets.sync.client
import numpy as np
from openpi.policies import policy as _policy
from openpi import transforms

# Enable msgpack numpy support
msgpack_numpy.patch()

class TensorRTRemotePolicy(_policy.Policy):
    """
    Policy that runs inference via a remote TensorRT server over WebSockets.
    Performs local pre/post-processing using openpi transforms.
    """
    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 8000,
        transforms: Optional[List[Any]] = None,
        output_transforms: Optional[List[Any]] = None,
        metadata: Optional[Dict] = None,
        action_horizon: Optional[int] = None,
        action_dim: Optional[int] = None,
    ):
        self.host = host
        self.port = port
        self.transforms = transforms
        self.output_transforms = output_transforms
        self._metadata = metadata or {}
        self.action_horizon = action_horizon
        self.action_dim = action_dim
        
        self._uri = f"ws://{self.host}:{self.port}"
        self._ws = self._connect()
        # Expect generic "ready" message
        _ = msgpack.unpackb(self._ws.recv())
        logging.info(f"Connected to TRT Server at {self._uri}")

    @property
    def metadata(self) -> Dict:
        return self._metadata

    def _connect(self):
        return websockets.sync.client.connect(self._uri, max_size=None)

    def infer(self, obs: Dict) -> Dict:
        # 1. Apply Input Transforms (Local)
        # Helper to decode bytes keys (OnnxPolicy logic)
        def decode_keys(d):
            if isinstance(d, dict):
                return {k.decode("utf-8") if isinstance(k, bytes) else k: decode_keys(v) for k, v in d.items()}
            return d
            
        obs = decode_keys(obs)
        
        inputs = obs
        if self.transforms:
            for transform in self.transforms:
                inputs = transform(inputs)

        # 2. Preprocess for TRT (Batching, Transpose, Noise) -- mirroring OnnxPolicy
        def add_batch(x):
            return np.expand_dims(np.array(x), axis=0)
            
        feed_dict = {}
        
        # Images need special handling if they are in a dict
        if "image" in inputs:
            # Transpose to (B, C, H, W)
            if "base_0_rgb" in inputs["image"]:
                feed_dict["base_0_rgb"] = add_batch(inputs["image"]["base_0_rgb"]).transpose(0, 3, 1, 2)
            if "left_wrist_0_rgb" in inputs["image"]:
                feed_dict["left_wrist_0_rgb"] = add_batch(inputs["image"]["left_wrist_0_rgb"]).transpose(0, 3, 1, 2)
            if "right_wrist_0_rgb" in inputs["image"]:
                feed_dict["right_wrist_0_rgb"] = add_batch(inputs["image"]["right_wrist_0_rgb"]).transpose(0, 3, 1, 2)
        
        if "state" in inputs:
            feed_dict["state"] = add_batch(inputs["state"])
        if "tokenized_prompt" in inputs:
            feed_dict["tokenized_prompt"] = add_batch(inputs["tokenized_prompt"]).astype(np.int32)
        if "tokenized_prompt_mask" in inputs:
            feed_dict["tokenized_prompt_mask"] = add_batch(inputs["tokenized_prompt_mask"])
        
        # Generate Noise
        if "noise" not in feed_dict:
            B = 1
            # Check horizon/dim
            horizon = self.action_horizon if self.action_horizon else 50
            dim = self.action_dim if self.action_dim else 10 # Default fallback?
            
            # Let's rely on passed args.
            if self.action_horizon is None: self.action_horizon = 10
            if self.action_dim is None: self.action_dim = 32
            
            noise = np.random.randn(B, self.action_horizon, self.action_dim).astype(np.float32)
            feed_dict["noise"] = noise

        # 3. Serialize and Send
        data = msgpack.packb(feed_dict, default=msgpack_numpy.encode)
        self._ws.send(data)

        # 4. Receive and Deserialize
        response = self._ws.recv()
        outputs_dict = msgpack.unpackb(response, object_hook=msgpack_numpy.decode)
        
        # 5. Unbatch and Post-process
        # "actions" output
        actions = outputs_dict["actions"][0] # Remove batch dim
        
        outputs = {
            "actions": actions,
            "state": inputs.get("state"), 
        }
        
        if self.output_transforms:
            for transform in self.output_transforms:
                outputs = transform(outputs)

        return outputs
