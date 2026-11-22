"""Frozen Qwen2.5-VL and glyph ByT5 encoders in native or Diffusers layouts."""

import json
from pathlib import Path

import torch
from accelerate import init_empty_weights
from diffusers import HunyuanVideo15Pipeline
from safetensors import safe_open
from transformers import AutoConfig, AutoTokenizer, Qwen2_5_VLTextModel, T5Config, T5EncoderModel


def _load_native_qwen(path, dtype):
    """Load only language weights, without constructing the visual tower."""
    config = AutoConfig.from_pretrained(path, local_files_only=True)
    config = getattr(config, "text_config", config)
    config._attn_implementation = "sdpa"
    with init_empty_weights():
        encoder = Qwen2_5_VLTextModel(config)
    expected_keys = set(encoder.state_dict())
    index_path = path / "model.safetensors.index.json"
    if index_path.is_file():
        weight_map = json.loads(index_path.read_text(encoding="utf-8"))["weight_map"]
        shards = sorted(set(weight_map.values()))
    else:
        shards = ["model.safetensors"]
    state_dict = {}
    for shard in shards:
        with safe_open(path / shard, framework="pt", device="cpu") as weights:
            for key in weights.keys():
                name = key.removeprefix("model.").removeprefix("language_model.")
                if name in expected_keys:
                    state_dict[name] = weights.get_tensor(key).to(dtype=dtype)
    encoder.load_state_dict(state_dict, strict=True, assign=True)
    return encoder


def _load_native_byt5(root, dtype):
    path = root / "text_encoder" / "byt5-small"
    glyph_path = root / "text_encoder" / "Glyph-SDXL-v2"
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    font_ids = json.loads((glyph_path / "assets" / "multilingual_10-lang_idx.json").read_text(encoding="utf-8"))
    color_ids = json.loads((glyph_path / "assets" / "color_idx.json").read_text(encoding="utf-8"))
    # Vocabulary order must match the checkpoint's embedding rows.
    tokens = [f"<color-{i}>" for i in range(len(color_ids))]
    tokens += [f"<{code[:2]}-font-{font_ids[code]}>" for code in font_ids]
    tokenizer.add_tokens(tokens, special_tokens=True)
    config = T5Config.from_pretrained(path, local_files_only=True)
    config.vocab_size = len(tokenizer)
    with init_empty_weights():
        encoder = T5EncoderModel(config)
    state = torch.load(glyph_path / "checkpoints" / "byt5_model.pt", map_location="cpu", weights_only=True)
    if "state_dict" in state:
        prefix = "module.text_tower.encoder."
        state = {name.removeprefix(prefix): value for name, value in state["state_dict"].items() if name.startswith(prefix)}
    converted = {f"encoder.{name}": value.to(dtype=dtype) for name, value in state.items()}
    converted["shared.weight"] = converted["encoder.embed_tokens.weight"]
    encoder.load_state_dict(converted, strict=True, assign=True)
    return tokenizer, encoder


class HunyuanVideo15ConditionEncoder:
    """Encode prompts using the official pipeline's template and glyph rules."""

    def __init__(self, model_path, *, device, torch_dtype=torch.bfloat16, max_sequence_length=1000):
        self.device = torch.device(device)
        self.dtype = torch_dtype
        self.max_sequence_length = int(max_sequence_length)
        if self.max_sequence_length <= 0:
            raise ValueError("HunyuanVideo 1.5 max_sequence_length must be positive.")
        root = Path(model_path)
        native_text_path = root / "text_encoder" / "llm"
        is_native = native_text_path.is_dir()
        tokenizer_path = native_text_path if is_native else root / "tokenizer"
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, padding_side="right", local_files_only=True)
        if is_native:
            self.text_encoder = _load_native_qwen(native_text_path, torch_dtype)
            self.tokenizer_2, self.text_encoder_2 = _load_native_byt5(root, torch_dtype)
        else:
            self.text_encoder = Qwen2_5_VLTextModel.from_pretrained(
                root / "text_encoder",
                torch_dtype=torch_dtype,
                local_files_only=True,
            )
            self.tokenizer_2 = AutoTokenizer.from_pretrained(root / "tokenizer_2", local_files_only=True)
            self.text_encoder_2 = T5EncoderModel.from_pretrained(
                root / "text_encoder_2",
                torch_dtype=torch_dtype,
                local_files_only=True,
            )
        for encoder in (self.text_encoder, self.text_encoder_2):
            encoder.requires_grad_(False)
            encoder.eval()
            encoder.to(device=self.device, dtype=torch_dtype)

    @torch.no_grad()
    def encode_prompt(self, prompts):
        prompts = [prompts] if isinstance(prompts, str) else list(prompts)
        if not prompts or any(not isinstance(prompt, str) for prompt in prompts):
            raise ValueError("HunyuanVideo 1.5 requires a non-empty prompt list of strings.")
        # Keep the pipeline's video template, hidden-state layer and glyph rules.
        embeddings, mask = HunyuanVideo15Pipeline._get_mllm_prompt_embeds(
            self.text_encoder,
            self.tokenizer,
            prompts,
            self.device,
            tokenizer_max_length=self.max_sequence_length,
        )
        glyph_embeddings, glyph_mask = HunyuanVideo15Pipeline._get_byt5_prompt_embeds(
            self.tokenizer_2,
            self.text_encoder_2,
            prompts,
            self.device,
            tokenizer_max_length=256,
        )
        return {
            "encoder_hidden_states": embeddings.to(device=self.device, dtype=self.dtype),
            "encoder_attention_mask": mask.to(device=self.device, dtype=torch.int64),
            "encoder_hidden_states_2": glyph_embeddings.to(device=self.device, dtype=self.dtype),
            "encoder_attention_mask_2": glyph_mask.to(device=self.device, dtype=torch.int64),
        }

    encode = encode_prompt
    __call__ = encode_prompt
