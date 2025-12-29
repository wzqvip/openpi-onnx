import dataclasses
import enum
import logging
import socket
import tyro
import os

from openpi.policies import onnx_policy
from openpi.serving import websocket_policy_server

@dataclasses.dataclass
class Args:
    # Path to ONNX model
    onnx_model: str
    # Config name (e.g. pi05_libero)
    config: str = "pi05_libero"
    # Checkpoint dir (for assets/stats)
    checkpoint: str = "gs://openpi-assets/checkpoints/pi05_libero"
    
    # Port to serve on
    port: int = 8000
    
    # Default prompt if needed
    default_prompt: str | None = None
    
    # Device (cuda/cpu)
    device: str = "cuda"

def main(args: Args) -> None:
    logging.basicConfig(level=logging.INFO, force=True)
    
    # Resolve checkpoint path
    if args.checkpoint.startswith("gs://"):
        # Download logic is handled inside create_onnx_policy -> maybe_download
        pass
    else:
        args.checkpoint = os.path.abspath(args.checkpoint)

    policy = onnx_policy.create_onnx_policy(
        config_name=args.config,
        checkpoint_dir=args.checkpoint,
        onnx_path=args.onnx_model,
        default_prompt=args.default_prompt,
        device=args.device
    )
    
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info("Creating ONNX Policy Server (host: %s, ip: %s)", hostname, local_ip)

    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",
        port=args.port,
        metadata=policy.metadata,
    )
    server.serve_forever()

if __name__ == "__main__":
    main(tyro.cli(Args))
