import sys
from unittest.mock import MagicMock
sys.modules["lerobot"] = MagicMock()
sys.modules["lerobot.common"] = MagicMock()
sys.modules["lerobot.common.datasets"] = MagicMock()
sys.modules["lerobot.common.datasets.lerobot_dataset"] = MagicMock()

# Disable runtime type checking (typeguard/jaxtyping)
def noop_decorator(*args, **kwargs):
    if len(args) >= 1 and callable(args[0]):
        return args[0]
    def wrapper(target):
        return target
    return wrapper

sys.modules["typeguard"] = MagicMock()
sys.modules["typeguard"].typechecked = noop_decorator
sys.modules["jaxtyping"] = MagicMock()
sys.modules["jaxtyping"].jaxtyped = noop_decorator
sys.modules["jaxtyping._decorator"] = MagicMock()

import os
os.environ["JAX_PLATFORM_NAME"] = "cpu"

def mock_custom_jvp(fun, **kwargs):
    def wrapper(*args, **kwargs):
        return fun(*args, **kwargs)
    def defjvp(jvp_fun):
        return jvp_fun
    wrapper.defjvp = defjvp
    return wrapper

try:
    import jax
    jax.jit = noop_decorator
    jax.custom_jvp = mock_custom_jvp
except ImportError:
    pass

import torch
import onnx
import onnx.helper
if not hasattr(onnx.helper, "float32_to_bfloat16"):
    onnx.helper.float32_to_bfloat16 = lambda x: x

import onnx_graphsurgeon as gs
from openpi.training import config as _config
from openpi.models_pytorch import pi0_pytorch
from transformers.models.gemma import modeling_gemma
import numpy as np

# --- Configuration ---
CHECKPOINT_DIR = "/home/taco/checkpoints/pi05_libero_pytorch"
CONFIG_NAME = "pi05_libero"
OUTPUT_PATH = "/home/taco/checkpoints/pi05_libero_pytorch/model.fp32.from_pytorch.onnx"
CALIBRATION_FILE = "/home/taco/openpi-onnx/calibration_data.pt"

# --- 1. Monkey Patching ---
def apply_rotary_pos_emb_patched(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (modeling_gemma.rotate_half(q) * sin)
    k_embed = (k * cos) + (modeling_gemma.rotate_half(k) * sin)
    return q_embed, k_embed

modeling_gemma.apply_rotary_pos_emb = apply_rotary_pos_emb_patched


def get_safe_dtype_patched(target_dtype, device_type):
    return torch.float32

pi0_pytorch.get_safe_dtype = get_safe_dtype_patched
modeling_gemma.GemmaRMSNorm.extra_repr = lambda self: f"eps={self.eps}"


class OnnxWrapperModelOpt(torch.nn.Module):
    def __init__(self, model, num_steps=10):
        super().__init__()
        self.model = model
        self.num_steps = num_steps

    def normalize(self, tensor, mean, std, q01=None, q99=None):
        return tensor

    def unnormalize(self, tensor, mean, std, q01=None, q99=None):
        return tensor

    def forward(self, base_rgb, left_rgb, right_rgb, state, tokenized_prompt, tokenized_prompt_mask, noise):
        bsize = state.shape[0]
        device = state.device

        images = {
            "base_0_rgb": base_rgb,
            "left_wrist_0_rgb": left_rgb,
            "right_wrist_0_rgb": right_rgb,
        }
        image_masks = {
            "base_0_rgb": torch.ones(bsize, dtype=torch.bool, device=device),
            "left_wrist_0_rgb": torch.ones(bsize, dtype=torch.bool, device=device),
            "right_wrist_0_rgb": torch.zeros(bsize, dtype=torch.bool, device=device),
        }

        state_norm = state

        observation = _model.Observation(
            images=images,
            image_masks=image_masks,
            state=state_norm,
            tokenized_prompt=tokenized_prompt,
            tokenized_prompt_mask=tokenized_prompt_mask,
        )

        state_proc = observation.state
        images_proc = observation.images
        img_masks = observation.image_masks
        lang_tokens = observation.tokenized_prompt
        lang_masks = observation.tokenized_prompt_mask

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.model.embed_prefix(
            list(images_proc.values()),
            list(img_masks.values()),
            lang_tokens,
            lang_masks,
        )

        from openpi.models_pytorch.pi0_pytorch import make_att_2d_masks
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

        prefix_att_2d_masks_4d = self.model._prepare_attention_masks_4d(prefix_att_2d_masks)
        self.model.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"

        _, past_key_values = self.model.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
        )

        dt = -1.0 / self.num_steps
        dt_tensor = torch.tensor(dt, dtype=self.model.action_in_proj.weight.dtype, device=device)
        x_t = noise

        for i in range(self.num_steps):
            time = torch.tensor(1.0 + i * dt, dtype=self.model.action_in_proj.weight.dtype, device=device)
            expanded_time = time.expand(bsize)
            v_t = self.model.denoise_step(state_proc, prefix_pad_masks, past_key_values, x_t, expanded_time)
            x_t = x_t + dt_tensor * v_t

        actions = x_t
        actions_out = actions
        return actions_out


def main():
    torch.compile = lambda x, **k: x
    config = _config.get_config(CONFIG_NAME)
    import dataclasses
    config = dataclasses.replace(config, model=dataclasses.replace(config.model, action_dim=32))

    model = pi0_pytorch.PI0Pytorch(config.model)
    ckpt_path = os.path.join(CHECKPOINT_DIR, "model.safetensors")
    from safetensors.torch import load_file
    sd = load_file(ckpt_path)
    model.load_state_dict(sd, strict=False)
    model.eval()
    model.to(dtype=torch.float32)

    wrapper = OnnxWrapperModelOpt(model, num_steps=10)

    # Load calibration data for tracing
    try:
        if os.path.exists(CALIBRATION_FILE):
            calibration_data = torch.load(CALIBRATION_FILE, weights_only=False)
            dummy_inputs = calibration_data[0]
            dummy_inputs = tuple(
                torch.from_numpy(t) if isinstance(t, np.ndarray) else (t.to("cpu") if isinstance(t, torch.Tensor) else t)
                for t in dummy_inputs
            )
        else:
            raise FileNotFoundError(CALIBRATION_FILE)
    except Exception:
        batch_size = 1
        state_dim = 8
        action_dim = config.model.action_dim
        dummy_inputs = (
            torch.randn(batch_size, 3, 224, 224),
            torch.randn(batch_size, 3, 224, 224),
            torch.zeros(batch_size, 3, 224, 224),
            torch.randn(batch_size, state_dim),
            torch.randint(0, 100, (batch_size, config.model.max_token_len), dtype=torch.int32),
            torch.ones(batch_size, config.model.max_token_len, dtype=torch.bool),
            torch.randn(batch_size, 10, action_dim),
        )

    input_names = ["base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb", "state", "prompt", "prompt_mask", "noise"]
    output_names = ["actions"]
    dynamic_axes = {k: {0: "batch_size"} for k in input_names}
    dynamic_axes["actions"] = {0: "batch_size"}

    torch.onnx.export(
        wrapper,
        dummy_inputs,
        OUTPUT_PATH,
        export_params=True,
        opset_version=19,
        do_constant_folding=True,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        dynamo=False,
    )

    # GraphSurgeon cleanup
    graph = gs.import_onnx(onnx.load(OUTPUT_PATH))
    graph.cleanup().toposort()

    intermediate_path = OUTPUT_PATH.replace(".onnx", ".gs_clean.onnx")
    onnx.save(
        gs.export_onnx(graph),
        intermediate_path,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location="model.fp32.from_pytorch.gs_clean.data",
        size_threshold=1024,
        convert_attribute=False,
    )

    # CumSum patch
    model_proto = onnx.load(intermediate_path)
    from onnx import helper, TensorProto

    new_nodes = []
    for node in model_proto.graph.node:
        if node.op_type == "CumSum":
            input_name = node.input[0]
            original_output_name = node.output[0]

            cast_in_name = input_name + "_cast_int32"
            cumsum_out_name = original_output_name + "_int32_intermediate"

            cast_in_node = helper.make_node(
                "Cast",
                inputs=[input_name],
                outputs=[cast_in_name],
                to=TensorProto.INT32,
                name=node.name + "_cast_in_patch",
            )

            node.input[0] = cast_in_name
            node.output[0] = cumsum_out_name

            cast_out_node = helper.make_node(
                "Cast",
                inputs=[cumsum_out_name],
                outputs=[original_output_name],
                to=TensorProto.INT64,
                name=node.name + "_cast_out_patch",
            )

            new_nodes.append(cast_in_node)
            new_nodes.append(node)
            new_nodes.append(cast_out_node)
        else:
            new_nodes.append(node)

    model_proto.graph.ClearField("node")
    model_proto.graph.node.extend(new_nodes)

    cleaned_path = OUTPUT_PATH.replace(".onnx", ".cleaned.onnx")
    onnx.save(
        model_proto,
        cleaned_path,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location="model.fp32.from_pytorch.cleaned.data",
        size_threshold=1024,
        convert_attribute=False,
    )


if __name__ == "__main__":
    from openpi.models import model as _model
    main()
