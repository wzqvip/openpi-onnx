import logging
import tyro
import dataclasses
from openpi.training import config as _config
from openpi.models import model as _model

@dataclasses.dataclass
class Args:
    config_name: str = "pi05_libero"

def main(args: Args):
    logging.basicConfig(level=logging.INFO)
    logging.info(f"Loading config: {args.config_name}")
    train_config = _config.get_config(args.config_name)
    model = train_config.model
    # Mock assets dirs
    assets_dirs = train_config.assets_dirs
    
    data_config = train_config.data.create(assets_dirs, model)
    
    print("Data Transforms Inputs:")
    for t in data_config.data_transforms.inputs:
        print(f"  {t}")
        
    print("Model Transforms Inputs:")
    for t in data_config.model_transforms.inputs:
        print(f"  {t}")

if __name__ == "__main__":
    main(tyro.cli(Args))
