import dataclasses
import logging
import socket
import tyro
import pathlib

from openpi.serving import websocket_policy_server
from openpi.policies import libero_policy
from openpi.policies import onnx_policy
from openpi.models import model as _model
from openpi.transforms import flatten_dict
from openpi.training import config as _config
from openpi.policies import policy_config
from openpi import transforms
from openpi.training import checkpoints as _checkpoints


@dataclasses.dataclass
class Args:
    checkpoint_dir: str = "./checkpoints/pi05_libero_pytorch"
    config_name: str = "pi05_libero" # Needed to load correct transforms
    model_name: str = "model.fp32.onnx" # Or model.onnx
    port: int = 8000
    host: str = "0.0.0.0"

def main(args: Args):
    logging.basicConfig(level=logging.INFO)
    
    # 1. Load Config to get Transforms
    logging.info(f"Loading config: {args.config_name}")
    train_config = _config.get_config(args.config_name)
    
    # ... (loading assets logic)
    
    # Instantiate model to get correct data config (factory needs it)
    model = train_config.model
    data_config = train_config.data.create(train_config.assets_dirs, model)

    asset_id = data_config.asset_id
    logging.info(f"Asset ID from config: {asset_id}")
    
    checkpoint_path = pathlib.Path(args.checkpoint_dir)
    assets_path = checkpoint_path / "assets"
    
    # Load norm stats
    norm_stats = None
    if asset_id and assets_path.exists():
         try:
             norm_stats = _checkpoints.load_norm_stats(assets_path, asset_id)
             logging.info(f"Successfully loaded norm stats for {asset_id}")
             # Debug: print norm stats keys
             flat_stats = flatten_dict(norm_stats)
             logging.info(f"Norm stats keys: {list(flat_stats.keys())}")
         except Exception as e:
             logging.error(f"Failed to load norm stats: {e}")

    if norm_stats is None:
        raise RuntimeError("Could not load norm stats! Inference will be garbage. Aborting.")

    # Construct Transforms
    # We MUST include LiberoInputs if we are serving Libero policy!
    # Check if data_config already has it? (Unlikely if it's strictly training config)
    # create_trained_policy in serve_policy.py takes 'repack_transforms'.
    # We need to manually add LiberoInputs.
    
    # Determine model type for LiberoInputs
    # Assuming PI0 (not FAST) based on config name "pi05_libero"
    model_type = _model.ModelType.PI0 
    
    libero_inputs = libero_policy.LiberoInputs(model_type=model_type)
    
    input_transforms = [
        # transforms.InjectDefaultPrompt(default_prompt), # if needed
        *data_config.data_transforms.inputs,
        transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
        *data_config.model_transforms.inputs,
    ]
    
    # Debug: print transform list
    logging.info(f"Input Transforms: {input_transforms}")
    
    # Create output norm stats (filter to actions only)
    # Unnormalize is strict, so we remove input keys from stats
    from openpi.transforms import flatten_dict, unflatten_dict
    flat_stats = flatten_dict(norm_stats)
    output_stats_flat = {k: v for k, v in flat_stats.items() if "actions" in k}
    # Also check if keys are just 'actions' or 'actions/...'
    # If safe, we just use this.
    try:
        output_norm_stats = unflatten_dict(output_stats_flat)
        logging.info(f"Filtered output stats keys: {list(output_stats_flat.keys())}")
    except Exception as e:
        logging.warning(f"Failed to filter output stats: {e}. Using full stats.")
        output_norm_stats = norm_stats

    output_transforms = [
        *data_config.model_transforms.outputs,
        transforms.Unnormalize(output_norm_stats, use_quantiles=data_config.use_quantile_norm),
        *data_config.data_transforms.outputs,
        libero_policy.LiberoOutputs(), # Map actions -> actions[:7]
    ]
    
    # Create ONNX Policy
    onnx_path = checkpoint_path / args.model_name
    policy = onnx_policy.OnnxPolicy(
        str(onnx_path),
        transforms=input_transforms,
        output_transforms=output_transforms,
        metadata=train_config.policy_metadata,
        action_horizon=train_config.model.action_horizon,
        action_dim=train_config.model.action_dim,
    )
    
    # Server
    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host=args.host,
        port=args.port,
        metadata=policy.metadata,
    )
    logging.info(f"Serving ONNX policy on {args.host}:{args.port}")
    server.serve_forever()

if __name__ == "__main__":
    args = tyro.cli(Args)
    main(args)
