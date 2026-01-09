import argparse
import numpy as np
import dataclasses
import tyro
import logging
from openpi.policies import policy_config
from openpi.training import config as _config

@dataclasses.dataclass
class Args:
    config: str = "pi05_libero"
    checkpoint: str = "checkpoints/pi05_libero_pytorch"
    output: str = "trace_data.npz"

def main(args: Args):
    logging.basicConfig(level=logging.INFO)
    
    # Load Policy
    logging.info(f"Loading policy from {args.checkpoint}...")
    cfg = _config.get_config(args.config)
    policy = policy_config.create_trained_policy(cfg, args.checkpoint)
    
    # Create Dummy Observation
    from openpi.policies import libero_policy
    # make_libero_example returns single example, we need batch?
    # policy.infer expects batched inputs usually, or unbatched? 
    # LiberoInputs expects unbatched?
    # Policy.infer applies DataTransforms which handle batching if configured?
    # Usually infer takes unbatched dict if running in env loop.
    # Check policy.py infer docstring?
    # But let's assume batched for safety or rely on automatic batching if single.
    
    example = libero_policy.make_libero_example()
    # Use unbatched inputs (policy handles batching or expects unbatched single sample)
    obs = example
    # Add prompt (make_libero_example adds it)
    
    # Run Inference
    logging.info("Running inference...")
    # Fix noise for determinism if possible, but the policy interface might handle it.
    # We will pass specific noise to infer if supported, or just capture what happens?
    # Policy.infer signature: infer(self, obs: dict, *, noise: np.ndarray | None = None)
    
    # Create fixed noise
    action_horizon = cfg.model.action_horizon
    action_dim = cfg.model.action_dim
    B = 1
    noise = np.random.randn(B, action_horizon, action_dim).astype(np.float32)
    
    output = policy.infer(obs, noise=noise)
    
    # Save
    logging.info(f"Saving trace to {args.output}")
    np.savez(
        args.output,
        obs=obs,
        noise=noise,
        actions=output["actions"],
        # state_out=output["state"]  # LiberoOutputs drops state
    )
    logging.info("Done.")

if __name__ == "__main__":
    main(tyro.cli(Args))
