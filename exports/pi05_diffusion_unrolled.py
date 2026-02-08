"""
Model wrapper that unrolls the diffusion loop for ONNX export compatibility.

This module provides a wrapper around the standard PI0.5 model that converts the
dynamic diffusion while-loop into a fixed sequence of denoising steps. This allows
the model to be exported to ONNX format and subsequently compiled with TensorRT
using the Edge-LLM quantization pipeline.

Key features:
- Unrolls the diffusion sampling loop (while loop -> for loop with unrolled steps)
- Maintains computational equivalence with the original model
- Allows ONNX export without dynamic control flow issues
- Supports variable num_diffusion_steps at model creation time (not inference time)
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict, List, Any
from dataclasses import dataclass

# Import OpenPi modules
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../openpi/src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


@dataclass
class UnrollConfig:
    """Configuration for diffusion unrolling."""
    num_diffusion_steps: int = 10
    """Number of diffusion steps to unroll. This determines the model's fixed architecture."""


class Pi05DiffusionUnrolled(nn.Module):
    """
    Wrapper that unrolls PI0.5 diffusion sampling for ONNX export.
    
    This model wraps a standard PI0.5 checkpoint and modifies its forward pass to:
    1. Use a fixed-length diffusion loop (unrolled from the original while loop)
    2. Expose all intermediate tensors in a way compatible with ONNX export
    3. Maintain numerical equivalence with the original sampling when num_diffusion_steps matches
    
    Usage:
        # Load original model
        base_model = PI0Pytorch.from_pretrained("path/to/pi05_model")
        
        # Wrap for unrolled export (10 steps)
        unrolled = Pi05DiffusionUnrolled(base_model, num_diffusion_steps=10)
        
        # Export to ONNX
        torch.onnx.export(unrolled, (...inputs...), "model.onnx")
    
    Note:
        - The num_diffusion_steps must be set at model creation, not at inference time
        - If you need variable steps, consider using method='loop_unfold' and ONNX Loop operator
        - The model operates in eval mode for inference (no dropout/batch norm randomness)
    """
    
    def __init__(self, base_model: nn.Module, num_diffusion_steps: int = 10):
        """
        Initialize the unrolled diffusion wrapper.
        
        Args:
            base_model: The original PI0.5 model (PI0Pytorch instance)
            num_diffusion_steps: Number of denoising steps to unroll (default: 10)
        """
        super().__init__()
        self.base_model = base_model
        self.num_diffusion_steps = num_diffusion_steps
        self.config = base_model.config
        
        # Move to eval mode (important for reproducibility)
        self.base_model.eval()
        for param in self.base_model.parameters():
            param.requires_grad = False
    
    def forward(
        self,
        state: torch.Tensor,
        images: Dict[str, torch.Tensor],
        image_masks: Dict[str, torch.Tensor],
        noise: Optional[torch.Tensor] = None,
        lang_tokens: Optional[torch.Tensor] = None,
        lang_masks: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass with unrolled diffusion sampling.
        
        Args:
            state: Robot state [batch_size, state_dim]
            images: Dict of images from multiple cameras [batch_size, 3, height, width]
            image_masks: Dict of image validity masks [batch_size]
            noise: Initial noise for diffusion [batch_size, action_horizon, action_dim]
                  If None, sampled from standard normal
            lang_tokens: Optional language tokens [batch_size, seq_len]
            lang_masks: Optional language token masks [batch_size, seq_len]
        
        Returns:
            Sampled actions [batch_size, action_horizon, action_dim]
        """
        batch_size = state.shape[0]
        device = state.device
        
        # === PART 1: Prefix encoding (once per inference) ===
        # This computes the cached key/value pairs for all prefix (image + language) content
        
        if noise is None:
            actions_shape = (batch_size, self.config.action_horizon, self.config.action_dim)
            noise = self.base_model.sample_noise(actions_shape, device)
        
        # Create Observation object from inputs
        from openpi.models.model import Observation
        observation_dict = {
            "image": images,
            "image_mask": image_masks,
            "state": state,
        }
        
        # Add optional language tokens if provided
        if lang_tokens is not None:
            observation_dict["tokenized_prompt"] = lang_tokens
            observation_dict["tokenized_prompt_mask"] = lang_masks
        
        observation = Observation.from_dict(observation_dict)
        
        # Preprocess observation
        preprocessed_output = self.base_model._preprocess_observation(observation, train=False)
        images_list, img_masks_list, lang_tokens_t, lang_masks_t, state_t = preprocessed_output
        
        # Embed prefix (images + language)
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.base_model.embed_prefix(
            images_list, img_masks_list, lang_tokens_t, lang_masks_t
        )
        
        # Make 2D attention masks
        prefix_att_2d_masks = self.base_model.make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        
        # Compute prefix key/value cache
        prefix_att_2d_masks_4d = self.base_model._prepare_attention_masks_4d(prefix_att_2d_masks)
        self.base_model.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"
        
        _, past_key_values = self.base_model.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
        )
        
        # === PART 2: Unrolled diffusion loop ===
        # Instead of while loop, we unroll this into sequential denoising steps
        
        dt = torch.tensor(
            -1.0 / self.num_diffusion_steps,
            dtype=self.base_model.action_in_proj.weight.dtype,
            device=device
        )
        
        x_t = noise
        
        # Unroll the diffusion loop: for each step, compute denoising and update x_t
        for step_idx in range(self.num_diffusion_steps):
            # Current timestep: starts at 1.0 and decreases by dt each step
            time = torch.tensor(
                1.0 + dt.item() * step_idx,
                dtype=self.base_model.action_in_proj.weight.dtype,
                device=device
            )
            expanded_time = time.expand(batch_size)
            
            # Denoise step
            v_t = self.base_model.denoise_step(
                state_t,
                prefix_pad_masks,
                past_key_values,
                x_t,
                expanded_time,
            )
            
            # Euler step: update x_t
            x_t = x_t + dt * v_t
        
        return x_t
    
    @classmethod
    def from_pretrained(cls, model_path: str, num_diffusion_steps: int = 10, **kwargs) -> "Pi05DiffusionUnrolled":
        """
        Load an unrolled diffusion model from a pretrained PI0.5 checkpoint.
        
        Args:
            model_path: Path to the pretrained PI0.5 model directory or checkpoint file
            num_diffusion_steps: Number of steps to unroll
            **kwargs: Additional arguments to pass to the base model loader
        
        Returns:
            Initialized Pi05DiffusionUnrolled instance
        """
        from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
        
        base_model = PI0Pytorch.from_pretrained(model_path, **kwargs)
        return cls(base_model, num_diffusion_steps=num_diffusion_steps)
    
    @torch.no_grad()
    def sample(
        self,
        state: torch.Tensor,
        images: Dict[str, torch.Tensor],
        image_masks: Dict[str, torch.Tensor],
        num_samples: int = 1,
        temperature: float = 1.0,
        **kwargs
    ) -> List[torch.Tensor]:
        """
        Generate multiple action samples (for diversity testing).
        
        Args:
            state: Robot state
            images: Image dict
            image_masks: Image masks
            num_samples: Number of independent samples to generate
            temperature: Temperature for noise scaling (1.0 = standard normal)
            **kwargs: Additional arguments
        
        Returns:
            List of sampled action tensors
        """
        samples = []
        for _ in range(num_samples):
            sample = self.forward(state, images, image_masks, **kwargs)
            samples.append(sample)
        return samples


# Utilities for export

def export_to_onnx(
    model: Pi05DiffusionUnrolled,
    output_path: str,
    sample_input: Dict[str, torch.Tensor],
    opset_version: int = 17,
    do_constant_folding: bool = True,
    verbose: bool = False,
):
    """
    Export the unrolled diffusion model to ONNX format.
    
    Args:
        model: The Pi05DiffusionUnrolled model
        output_path: Path to save the ONNX model
        sample_input: Dictionary of sample inputs for tracing
        opset_version: ONNX opset version (17+ recommended for better op support)
        do_constant_folding: Whether to apply constant folding optimization
        verbose: Whether to print export progress
    
    Returns:
        None (saves to output_path)
    
    Example:
        >>> model = Pi05DiffusionUnrolled.from_pretrained("pi05_model", num_diffusion_steps=10)
        >>> model.eval()
        >>> 
        >>> # Prepare sample input (batch=1 for tracing)
        >>> sample_input = {
        ...     "state": torch.randn(1, state_dim),
        ...     "images": {"base_0_rgb": torch.randn(1, 3, 224, 224), ...},
        ...     "image_masks": {"base_0_rgb": torch.ones(1, dtype=torch.bool), ...},
        ... }
        >>> 
        >>> export_to_onnx(model, "pi05_unrolled.onnx", sample_input)
    """
    import torch.onnx
    
    # Prepare input names and output names
    input_names = []
    output_names = ["actions"]
    
    # Build input specification from sample_input
    inputs_to_export = []
    for key, tensor in sample_input.items():
        if key in ["state"]:
            input_names.append(key)
            inputs_to_export.append(tensor)
        elif key == "images":
            for img_key, img_tensor in tensor.items():
                input_names.append(f"image_{img_key}")
                inputs_to_export.append(img_tensor)
        elif key == "image_masks":
            for mask_key, mask_tensor in tensor.items():
                input_names.append(f"image_mask_{mask_key}")
                inputs_to_export.append(mask_tensor)
    
    # Export
    torch.onnx.export(
        model,
        tuple(inputs_to_export),
        output_path,
        input_names=input_names,
        output_names=output_names,
        opset_version=opset_version,
        do_constant_folding=do_constant_folding,
        verbose=verbose,
        export_params=True,
        use_external_data_format=True,  # For large models
    )
    
    if verbose:
        print(f"✓ Model exported to {output_path}")


def compare_with_original(
    original_model: nn.Module,
    unrolled_model: Pi05DiffusionUnrolled,
    test_input: Dict[str, torch.Tensor],
    num_trials: int = 5,
    rtol: float = 1e-4,
    atol: float = 1e-6,
) -> bool:
    """
    Compare outputs of original and unrolled models to verify correctness.
    
    Args:
        original_model: The original PI0.5 model
        unrolled_model: The unrolled diffusion wrapper
        test_input: Dictionary of test inputs
        num_trials: Number of comparison trials
        rtol: Relative tolerance for allclose check
        atol: Absolute tolerance for allclose check
    
    Returns:
        True if outputs match within tolerance, False otherwise
    
    Example:
        >>> original = PI0Pytorch.from_pretrained("model")
        >>> unrolled = Pi05DiffusionUnrolled(original, num_diffusion_steps=10)
        >>> test_inp = {...}
        >>> matches = compare_with_original(original, unrolled, test_inp)
        >>> print(f"Outputs match: {matches}")
    """
    from openpi.policies.policy import Policy
    
    original_model.eval()
    unrolled_model.eval()
    
    all_match = True
    for trial in range(num_trials):
        with torch.no_grad():
            # Run both models
            output_original = original_model.sample_actions(
                "cuda",  # or appropriate device
                test_input,
                num_steps=unrolled_model.num_diffusion_steps
            )
            output_unrolled = unrolled_model.forward(**test_input)
        
        # Compare
        match = torch.allclose(output_original, output_unrolled, rtol=rtol, atol=atol)
        
        if not match:
            max_diff = (output_original - output_unrolled).abs().max().item()
            print(f"Trial {trial}: MISMATCH - max_diff={max_diff:.2e}")
            all_match = False
        else:
            print(f"Trial {trial}: OK")
    
    return all_match


if __name__ == "__main__":
    # Example usage
    print("Pi05DiffusionUnrolled - ONNX Export Wrapper")
    print("=" * 50)
    print("\nUsage:")
    print("  1. Load model: model = Pi05DiffusionUnrolled.from_pretrained('path')")
    print("  2. Export: export_to_onnx(model, 'output.onnx', sample_input)")
    print("  3. Then use quantize-llm from TensorRT-Edge-LLM")
    print("\nNext steps: Run through Edge-LLM quantization pipeline")
