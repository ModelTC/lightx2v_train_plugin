"""Convert native HunyuanVideo 1.5 checkpoints to the Diffusers tensor layout."""

import json
import re
from pathlib import Path

import torch
from accelerate import init_empty_weights
from diffusers import AutoencoderKLHunyuanVideo15, HunyuanVideo15Transformer3DModel
from safetensors.torch import load_file


def _component_path(root, component, transformer_version=None):
    root = Path(root)
    if (root / "config.json").is_file():
        config = json.loads((root / "config.json").read_text(encoding="utf-8"))
        if config.get("_class_name") in {
            "HunyuanVideo_1_5_DiffusionTransformer",
            "HunyuanVideo15Transformer3DModel",
            "AutoencoderKLConv3D",
            "AutoencoderKLHunyuanVideo15",
        }:
            return root
    path = root / component
    if component == "transformer" and not (path / "config.json").is_file():
        path /= transformer_version
    if not (path / "config.json").is_file():
        raise FileNotFoundError(f"HunyuanVideo 1.5 {component} config not found: {path / 'config.json'}")
    return path


def _read_weights(path):
    index_path = path / "diffusion_pytorch_model.safetensors.index.json"
    if index_path.is_file():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        files = sorted(set(index["weight_map"].values()))
    else:
        files = ["diffusion_pytorch_model.safetensors"]
    state = {}
    for filename in files:
        shard = load_file(str(path / filename), device="cpu")
        duplicate = state.keys() & shard.keys()
        if duplicate:
            raise ValueError(f"Duplicate checkpoint tensor keys: {sorted(duplicate)}")
        state.update(shard)
    return state


def _assign_weights(model, state, torch_dtype):
    expected = model.state_dict()
    missing = expected.keys() - state.keys()
    unexpected = state.keys() - expected.keys()
    if missing or unexpected:
        raise ValueError(f"HunyuanVideo 1.5 checkpoint keys differ: missing={sorted(missing)}, unexpected={sorted(unexpected)}")
    for key, tensor in state.items():
        if tensor.shape != expected[key].shape:
            raise ValueError(f"HunyuanVideo 1.5 tensor {key} has shape {tuple(tensor.shape)}, expected {tuple(expected[key].shape)}")
        if tensor.is_floating_point():
            state[key] = tensor.to(dtype=torch_dtype)
    model.load_state_dict(state, strict=True, assign=True)
    return model


def convert_transformer_state_dict(state):
    """Convert native tensor names and layouts without altering the source dict."""
    result = {}
    top_level = {
        "img_in.proj": "x_embedder.proj",
        "vision_in.proj.0": "image_embedder.norm_in",
        "vision_in.proj.1": "image_embedder.linear_1",
        "vision_in.proj.3": "image_embedder.linear_2",
        "vision_in.proj.4": "image_embedder.norm_out",
        "txt_in.input_embedder": "context_embedder.proj_in",
        "txt_in.t_embedder.mlp.0": "context_embedder.time_text_embed.timestep_embedder.linear_1",
        "txt_in.t_embedder.mlp.2": "context_embedder.time_text_embed.timestep_embedder.linear_2",
        "txt_in.c_embedder.linear_1": "context_embedder.time_text_embed.text_embedder.linear_1",
        "txt_in.c_embedder.linear_2": "context_embedder.time_text_embed.text_embedder.linear_2",
        "byt5_in.layernorm": "context_embedder_2.norm",
        "byt5_in.fc1": "context_embedder_2.linear_1",
        "byt5_in.fc2": "context_embedder_2.linear_2",
        "byt5_in.fc3": "context_embedder_2.linear_3",
        "time_in.mlp.0": "time_embed.timestep_embedder.linear_1",
        "time_in.mlp.2": "time_embed.timestep_embedder.linear_2",
        "time_r_in.mlp.0": "time_embed.timestep_embedder_r.linear_1",
        "time_r_in.mlp.2": "time_embed.timestep_embedder_r.linear_2",
        "cond_type_embedding": "cond_type_embed",
        "final_layer.linear": "proj_out",
        "final_layer.adaLN_modulation.1": "norm_out.linear",
    }
    block_map = {
        "img_mod.linear": "norm1.linear",
        "txt_mod.linear": "norm1_context.linear",
        "img_attn_q": "attn.to_q",
        "img_attn_k": "attn.to_k",
        "img_attn_v": "attn.to_v",
        "img_attn_q_norm": "attn.norm_q",
        "img_attn_k_norm": "attn.norm_k",
        "img_attn_proj": "attn.to_out.0",
        "txt_attn_q": "attn.add_q_proj",
        "txt_attn_k": "attn.add_k_proj",
        "txt_attn_v": "attn.add_v_proj",
        "txt_attn_q_norm": "attn.norm_added_q",
        "txt_attn_k_norm": "attn.norm_added_k",
        "txt_attn_proj": "attn.to_add_out",
        "img_mlp.fc1": "ff.net.0.proj",
        "img_mlp.fc2": "ff.net.2",
        "txt_mlp.fc1": "ff_context.net.0.proj",
        "txt_mlp.fc2": "ff_context.net.2",
    }
    refiner_map = {
        "norm1": "norm1",
        "norm2": "norm2",
        "self_attn_proj": "attn.to_out.0",
        "mlp.fc1": "ff.net.0.proj",
        "mlp.fc2": "ff.net.2",
        "adaLN_modulation.1": "norm_out.linear",
    }
    for key, tensor in state.items():
        stem, suffix = key.rsplit(".", 1)
        if stem in top_level:
            target = f"{top_level[stem]}.{suffix}"
            if stem == "final_layer.adaLN_modulation.1":
                # Tencent emits shift, scale; AdaLayerNormContinuous uses scale, shift.
                shift, scale = tensor.chunk(2, dim=0)
                tensor = torch.cat((scale, shift), dim=0)
        elif match := re.fullmatch(r"double_blocks\.(\d+)\.(.+)", stem):
            index, layer = match.groups()
            if layer not in block_map:
                raise ValueError(f"Unsupported native HunyuanVideo 1.5 tensor: {key}")
            target = f"transformer_blocks.{index}.{block_map[layer]}.{suffix}"
        elif match := re.fullmatch(r"txt_in\.individual_token_refiner\.blocks\.(\d+)\.(.+)", stem):
            index, layer = match.groups()
            prefix = f"context_embedder.token_refiner.refiner_blocks.{index}"
            if layer == "self_attn_qkv":
                if tensor.shape[0] % 3:
                    raise ValueError(f"Invalid fused QKV tensor: {key}")
                for projection, part in zip(("to_q", "to_k", "to_v"), tensor.chunk(3, dim=0)):
                    result[f"{prefix}.attn.{projection}.{suffix}"] = part
                continue
            if layer not in refiner_map:
                raise ValueError(f"Unsupported native HunyuanVideo 1.5 tensor: {key}")
            target = f"{prefix}.{refiner_map[layer]}.{suffix}"
        else:
            raise ValueError(f"Unsupported native HunyuanVideo 1.5 tensor: {key}")
        if target in result:
            raise ValueError(f"Duplicate converted HunyuanVideo 1.5 tensor: {target}")
        result[target] = tensor
    return result


def native_transformer_config(config):
    """Translate Tencent's released dense base transformer config."""
    patch_size = config.get("patch_size", (1, 1, 1))
    if config.get("mm_single_blocks_depth", 0) != 0 or len(patch_size) != 3 or patch_size[1] != patch_size[2]:
        raise ValueError("Only HunyuanVideo 1.5 dense double-stream transformers with square spatial patches are supported.")
    if config.get("ideal_task") not in {"t2v", "i2v"} or not config.get("concat_condition", True):
        raise ValueError("HunyuanVideo 1.5 training requires a base t2v/i2v transformer, not a super-resolution model.")
    if not config.get("qk_norm", True) or config.get("qk_norm_type", "rms") != "rms":
        raise ValueError("HunyuanVideo 1.5 training requires RMS query/key normalization.")
    if config.get("is_reshape_temporal_channels", False) or config.get("guidance_embed", False):
        raise ValueError("Temporal channel reshaping and guidance-distilled transformers are unsupported.")
    num_heads = config["heads_num"]
    hidden_size = config["hidden_size"]
    if hidden_size % num_heads:
        raise ValueError("hidden_size must be divisible by heads_num.")
    latent_channels = config.get("in_channels", 32)
    return {
        "in_channels": 2 * latent_channels + 1,
        "out_channels": config.get("out_channels", latent_channels),
        "num_attention_heads": num_heads,
        "attention_head_dim": hidden_size // num_heads,
        "num_layers": config["mm_double_blocks_depth"],
        "num_refiner_layers": 2,
        "mlp_ratio": config.get("mlp_width_ratio", 4.0),
        "patch_size": patch_size[1],
        "patch_size_t": patch_size[0],
        "qk_norm": "rms_norm",
        "text_embed_dim": config.get("text_states_dim", 3584),
        "text_embed_2_dim": 1472,
        "image_embed_dim": config.get("vision_states_dim", 1152),
        "rope_theta": config.get("rope_theta", 256),
        "rope_axes_dim": config.get("rope_dim_list", (16, 56, 56)),
        "target_size": {"480p": 640, "720p": 960}.get(config.get("ideal_resolution"), 640),
        "task_type": config["ideal_task"],
        "use_meanflow": config.get("use_meanflow", False),
    }


def load_transformer(model_path, *, torch_dtype=torch.bfloat16, transformer_version="480p_t2v"):
    """Return a CPU transformer from either a Tencent root or Diffusers layout."""
    path = _component_path(model_path, "transformer", transformer_version)
    config = json.loads((path / "config.json").read_text(encoding="utf-8"))
    class_name = config.get("_class_name")
    if class_name == "HunyuanVideo15Transformer3DModel":
        return HunyuanVideo15Transformer3DModel.from_pretrained(path, torch_dtype=torch_dtype, local_files_only=True)
    if class_name != "HunyuanVideo_1_5_DiffusionTransformer":
        raise ValueError(f"Unsupported HunyuanVideo 1.5 transformer class: {class_name}")
    with init_empty_weights():
        model = HunyuanVideo15Transformer3DModel(**native_transformer_config(config))
    state = convert_transformer_state_dict(_read_weights(path))
    return _assign_weights(model, state, torch_dtype)


def convert_vae_state_dict(state):
    """Map native Conv3D VAE tensors to Diffusers, preserving their layout."""
    result = {}
    for key, value in state.items():
        target = re.sub(r"\.down\.(\d+)\.block\.(\d+)\.", r".down_blocks.\1.resnets.\2.", key)
        target = re.sub(r"\.down\.(\d+)\.downsample\.", r".down_blocks.\1.downsamplers.0.", target)
        target = re.sub(r"\.up\.(\d+)\.block\.(\d+)\.", r".up_blocks.\1.resnets.\2.", target)
        target = re.sub(r"\.up\.(\d+)\.upsample\.", r".up_blocks.\1.upsamplers.0.", target)
        target = target.replace(".mid.block_1.", ".mid_block.resnets.0.")
        target = target.replace(".mid.block_2.", ".mid_block.resnets.1.")
        target = target.replace(".mid.attn_1.", ".mid_block.attentions.0.")
        target = re.sub(r"(\.attentions\.0)\.([qkv])\.", r"\1.to_\2.", target)
        if target in result:
            raise ValueError(f"Duplicate converted VAE tensor: {target}")
        result[target] = value
    return result


def load_vae(model_path, *, torch_dtype=torch.float32):
    """Return a CPU VAE from a Tencent root, component directory or Diffusers root."""
    path = _component_path(model_path, "vae")
    config = json.loads((path / "config.json").read_text(encoding="utf-8"))
    class_name = config.get("_class_name")
    if class_name == "AutoencoderKLHunyuanVideo15":
        return AutoencoderKLHunyuanVideo15.from_pretrained(path, torch_dtype=torch_dtype, local_files_only=True)
    if class_name != "AutoencoderKLConv3D":
        raise ValueError(f"Unsupported HunyuanVideo 1.5 VAE class: {class_name}")
    config_keys = ("in_channels", "out_channels", "latent_channels", "block_out_channels", "layers_per_block", "downsample_match_channel", "upsample_match_channel", "scaling_factor")
    kwargs = {key: config[key] for key in config_keys if key in config}
    kwargs["spatial_compression_ratio"] = config.get("ffactor_spatial", 16)
    kwargs["temporal_compression_ratio"] = config.get("ffactor_temporal", 4)
    with init_empty_weights():
        model = AutoencoderKLHunyuanVideo15(**kwargs)
    return _assign_weights(model, convert_vae_state_dict(_read_weights(path)), torch_dtype)
