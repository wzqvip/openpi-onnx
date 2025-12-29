import torch
import os
import safetensors.torch
from openpi.training import config as _config
from openpi.models_pytorch import pi0_pytorch
import json

def main():
    config_name = "pi05_libero"
    output_path = "./checkpoints/pi05_libero_pytorch"
    os.makedirs(output_path, exist_ok=True)
    
    print(f"Creating dummy checkpoint for {config_name}...")
    
    # HACK: Disable torch.compile to avoid OOM
    torch.compile = lambda x, **k: x
    print("Disabled torch.compile.")

    config = _config.get_config(config_name)
    
    # HACK: Monkey patch gemma.get_config to return tiny configs
    import openpi.models.gemma as _gemma_mod
    original_get_config = _gemma_mod.get_config
    
    def tiny_get_config(variant):
        print(f"DEBUG: tiny_get_config called for {variant}")
        c = original_get_config(variant)
        
        # Reduce dimensions
        if hasattr(c, "depth"): 
            c.depth = 1
            print(f"DEBUG: Set depth=1 for {variant}")
        if hasattr(c, "num_hidden_layers"): 
            c.num_hidden_layers = 1
            print(f"DEBUG: Set num_hidden_layers=1 for {variant}")
            
        # Reduce vocab (if attribute exists or we can inject)
        # Note: GemmaConfig might use hardcoded vocab if not set? 
        # But usually it's in config.
        if hasattr(c, "vocab_size"): 
            c.vocab_size = 1024
            print(f"DEBUG: Set vocab_size=1024 for {variant}")
        else:
            # Try to force set it if it's a data object
            try:
                c.vocab_size = 1024
                print(f"DEBUG: Injected vocab_size=1024 for {variant}")
            except:
                pass

        # Nested configs (if any)
        if hasattr(c, "vision_config"):
             c.vision_config.num_hidden_layers = 1
             print("DEBUG: Reduced vision layers")
        if hasattr(c, "text_config"):
             c.text_config.num_hidden_layers = 1
             c.text_config.vocab_size = 1024 
             print("DEBUG: Reduced text layers & vocab")
             
        return c
        
    _gemma_mod.get_config = tiny_get_config
    print("Monkey-patched openpi.models.gemma.get_config for tiny model.")
    
    # Create Model
    print("Initializing model structure in float16 (Tiny)...")
    # Set default dtype to float16 to save memory during init
    torch.set_default_dtype(torch.float16)
    model = pi0_pytorch.PI0Pytorch(config.model)
    # model.to(torch.float32) # removed
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model initialized with random weights (float16). Param count: {num_params}")
    
    # Save as safetensors
    print(f"Saving to {output_path}/model.safetensors...")
    safetensors.torch.save_model(model, os.path.join(output_path, "model.safetensors"))
    
    # Save config.json
    config_dict = {
        "action_dim": config.model.action_dim,
        "action_horizon": config.model.action_horizon,
        "paligemma_variant": config.model.paligemma_variant,
        "action_expert_variant": config.model.action_expert_variant,
        "precision": "float32",
    }
    with open(os.path.join(output_path, "config.json"), "w") as f:
        json.dump(config_dict, f, indent=2)
        
    print("Dummy checkpoint created.")

if __name__ == "__main__":
    main()
