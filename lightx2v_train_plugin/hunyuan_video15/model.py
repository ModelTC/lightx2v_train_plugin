"""HunyuanVideo 1.5 T2V training with native or Diffusers checkpoints."""

import json
from contextlib import contextmanager, nullcontext
from copy import deepcopy
from pathlib import Path

import torch
from diffusers import HunyuanVideo15Pipeline
from diffusers.video_processor import VideoProcessor
from lightx2v_train.model_capabilities import FlowMatchingSFTCapability
from lightx2v_train.model_zoo.base import BaseModel
from lightx2v_train.runtime.attention import prepare_diffusers_attention
from lightx2v_train.runtime.backend import get_backend
from lightx2v_train.runtime.distributed import is_main_process
from lightx2v_train.runtime.fsdp import fsdp2_enabled
from lightx2v_train.utils.registry import MODEL_REGISTER
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from .attention import HunyuanVideo15CompactAttnProcessor2_0
from .data_process import validate_video_shape
from .sft import HunyuanVideo15SFTCapability


@MODEL_REGISTER("hunyuan_video15_t2v")
class HunyuanVideo15T2VModel(BaseModel):
    pipeline_cls = HunyuanVideo15Pipeline
    default_unconditional_prompt = ""
    vae_scale_factor_spatial = 16
    vae_scale_factor_temporal = 4
    latent_channels = 32

    def __init__(self, config):
        config = deepcopy(config)
        model = config["model"]
        self.transformer_version = model.setdefault("transformer_version", "480p_t2v")
        if self.transformer_version not in {"480p_t2v", "720p_t2v"}:
            raise ValueError("HunyuanVideo 1.5 SFT supports transformer_version=480p_t2v or 720p_t2v.")
        if config.get("distributed", {}).get("sequence_parallel", {}).get("enabled", False):
            raise ValueError("HunyuanVideo 1.5 training currently supports DDP/FSDP2; disable sequence_parallel.")
        model.setdefault("running_dtype", "bf16")
        model.setdefault("latent_dtype", "bf16")
        model.setdefault("vae_dtype", "fp32")
        super().__init__(config)
        self.condition_encoder = None
        self.image_processor = VideoProcessor(vae_scale_factor=16)

    def register_capabilities(self):
        super().register_capabilities()
        self.capabilities.register(FlowMatchingSFTCapability, HunyuanVideo15SFTCapability(self))

    def load_components(self, *, load_transformer, load_vae, load_condition_encoder):
        from .loading import load_transformer as load_dit
        from .loading import load_vae as load_autoencoder

        root = self.config["model"]["pretrained_model_name_or_path"]
        self._load_vae_config(root)
        if load_condition_encoder:
            from .condition import HunyuanVideo15ConditionEncoder

            self.condition_encoder = HunyuanVideo15ConditionEncoder(
                root,
                device=self.device,
                torch_dtype=self.text_encoder_dtype,
                max_sequence_length=self.config["model"].get("max_sequence_length", 1000),
            )
        if load_vae:
            self.vae = load_autoencoder(root, torch_dtype=self.vae_dtype).to(self.device)
            self.vae.requires_grad_(False).eval()
            if self.config["model"].get("vae_tiling", True):
                self.vae.enable_tiling()
        if load_transformer:
            self.transformer = load_dit(
                self.config["model"].get("transformer_path", root),
                torch_dtype=self.transformer_param_dtype,
                transformer_version=self.transformer_version,
            )
            if self.transformer.config.task_type != "t2v" or self.transformer.config.use_meanflow:
                raise ValueError("HunyuanVideo 1.5 SFT requires a non-distilled T2V transformer.")
            if self.transformer.config.in_channels != 65 or self.transformer.config.out_channels != 32:
                raise ValueError("HunyuanVideo 1.5 requires 65 transformer input channels and 32 output channels.")
            for block in self.transformer.transformer_blocks:
                previous = block.attn.get_processor()
                processor = HunyuanVideo15CompactAttnProcessor2_0()
                processor._attention_backend = previous._attention_backend
                processor._parallel_config = previous._parallel_config
                block.attn.set_processor(processor)
            device = get_backend().host_device if fsdp2_enabled(self.config) else self.device
            self.transformer.to(device)
            prepare_diffusers_attention(self.transformer, self.config)

    def _load_vae_config(self, root):
        path = Path(root) / "vae" / "config.json"
        self.vae_config = json.loads(path.read_text(encoding="utf-8"))
        if int(self.vae_config.get("latent_channels", 32)) != 32:
            raise ValueError("Expected a HunyuanVideo 1.5 VAE with 32 latent channels.")
        scaling = float(self.vae_config.get("scaling_factor", 1.03682))
        if not 0 < scaling < float("inf"):
            raise ValueError("HunyuanVideo 1.5 VAE scaling_factor must be positive and finite.")

    def reuse_frozen_components_from(self, source):
        super().reuse_frozen_components_from(source)
        self.condition_encoder = source.condition_encoder

    def denoiser_module(self):
        return self.transformer

    def fsdp2_shard_plan(self, fsdp_config):
        reshard = fsdp_config["reshard_after_forward"]
        return [
            {"modules": self.transformer.transformer_blocks, "reshard_after_forward": reshard["block_reshard"]},
            {"module": self.transformer, "reshard_after_forward": reshard["root_reshard"]},
        ]

    def add_lora(self, rank, alpha, target_modules):
        super().add_lora(rank, alpha, target_modules or ["to_q", "to_k", "to_v", "to_out.0", "add_q_proj", "add_k_proj", "add_v_proj", "to_add_out"])

    def save_lora_weights(self, save_dir, adapter_name=None, weights_subdir=None, **kwargs):
        super().save_lora_weights(save_dir, adapter_name, weights_subdir, **kwargs)
        if not is_main_process():
            return
        output_dir = Path(save_dir) / weights_subdir if weights_subdir else Path(save_dir)
        path = output_dir / "pytorch_lora_weights.safetensors"
        with safe_open(path, framework="pt", device="cpu") as weights:
            metadata = dict(weights.metadata() or {})
        adapter_config = self.denoiser_module().peft_config[adapter_name or "default"].to_dict()
        metadata["lora_adapter_metadata"] = json.dumps(
            {key: sorted(value) if isinstance(value, set) else value for key, value in adapter_config.items()},
            sort_keys=True,
        )
        tensors = load_file(path, device="cpu")
        temporary = path.with_suffix(".safetensors.tmp")
        save_file(tensors, temporary, metadata=metadata)
        temporary.replace(path)

    def _freeze_unused_output_branches(self):
        # Discarded final text outputs never get gradients and break FSDP optimizer restore.
        last_block = self.transformer.transformer_blocks[-1]
        last_block.attn.to_add_out.requires_grad_(False)
        last_block.ff_context.requires_grad_(False)

    def set_lora_trainable(self):
        super().set_lora_trainable()
        self._freeze_unused_output_branches()

    def set_full_trainable(self):
        super().set_full_trainable()
        self._freeze_unused_output_branches()

    def save_consolidated_weights(self, output_path):
        super().save_consolidated_weights(output_path)
        if is_main_process():
            self.transformer.save_config(Path(output_path).parent)

    @contextmanager
    def transformer_forward_context(self):
        offload = self.config.get("training", {}).get("gradient_checkpointing_offload", False)
        if offload and self.is_fsdp2_wrapped():
            raise ValueError("HunyuanVideo 1.5 FSDP2 requires gradient_checkpointing_offload=false; gradient_checkpointing remains supported.")
        offload_context = nullcontext()
        if offload and torch.is_grad_enabled():
            offload_context = torch.autograd.graph.save_on_cpu(pin_memory=self.device.type == "cuda")
        with super().transformer_forward_context(), offload_context:
            yield

    def encode_to_latent(self, sample):
        return self._encode_video(sample, mode="sample")

    def encode_to_cache_latent(self, sample):
        return self._encode_video(sample, mode="mode")

    def _encode_video(self, sample, *, mode):
        video = sample["inputs"]["video"]
        if video.ndim == 4:
            video = video.unsqueeze(0)
        if video.ndim != 5 or video.shape[1] != 3:
            raise ValueError("HunyuanVideo 1.5 video must have shape [B,3,F,H,W].")
        validate_video_shape(*video.shape[-3:])
        with self._active_vae() as vae:
            posterior = vae.encode(video.to(device=self.device, dtype=self.vae_dtype)).latent_dist
            latent = getattr(posterior, mode)()
        return (latent * float(self.vae_config["scaling_factor"])).to(self.latent_dtype)

    def encode_condition(self, sample):
        return self.encode_prompt_condition(sample["conditioning"]["prompt"])

    def encode_prompt_condition(self, prompt):
        if self.condition_encoder is None:
            raise RuntimeError("Text encoders are not loaded; provide a training cache or enable load_text_encoder.")
        return self.condition_encoder.encode_prompt(prompt)

    def prepare_denoiser_input(self, noisy_latent, condition=None):
        del condition
        if noisy_latent.ndim != 5 or noisy_latent.shape[1] != 32 or min(noisy_latent.shape) < 1:
            raise ValueError("HunyuanVideo 1.5 latents must have shape [B,32,F,H,W].")
        noisy_latent = noisy_latent.to(device=self.device, dtype=self.running_dtype)
        mask = noisy_latent.new_zeros(noisy_latent.shape[0], 1, *noisy_latent.shape[2:])
        return torch.cat([noisy_latent, torch.zeros_like(noisy_latent), mask], dim=1)

    def denoise(self, denoiser_input, timestep_or_sigma, condition):
        batch_size = denoiser_input.shape[0]
        sigma = timestep_or_sigma.to(device=self.device, dtype=torch.float32).reshape(-1)
        if sigma.numel() == 1:
            sigma = sigma.expand(batch_size)
        if sigma.numel() != batch_size:
            raise ValueError("HunyuanVideo 1.5 sigma batch does not match the latent batch.")
        encoder_inputs = {}
        for key in ("encoder_hidden_states", "encoder_hidden_states_2", "encoder_attention_mask", "encoder_attention_mask_2"):
            if key not in condition:
                raise KeyError(f"HunyuanVideo 1.5 condition is missing {key}.")
            tensor = condition[key]
            expected_ndim = 2 if "mask" in key else 3
            if tensor.ndim != expected_ndim or tensor.shape[0] != batch_size:
                raise ValueError(f"HunyuanVideo 1.5 condition {key} has incompatible shape {tuple(tensor.shape)}.")
            encoder_inputs[key] = tensor.to(device=self.device, dtype=torch.bool if "mask" in key else self.running_dtype)
        # T2V still requires zero image tokens; the transformer masks them.
        image_embeds = denoiser_input.new_zeros(batch_size, 729, self.transformer.config.image_embed_dim)
        with self.transformer_forward_context():
            return self.transformer(
                hidden_states=denoiser_input,
                timestep=sigma * 1000.0,
                image_embeds=image_embeds,
                return_dict=False,
                **encoder_inputs,
            )[0]

    def postprocess_denoiser_output(self, prediction, denoiser_input):
        expected = (denoiser_input.shape[0], 32, *denoiser_input.shape[2:])
        if tuple(prediction.shape) != expected:
            raise ValueError(f"HunyuanVideo 1.5 prediction shape {tuple(prediction.shape)} does not match {expected}.")
        return prediction

    def prepare_infer_latents(self, height, width, generator=None):
        frames = self.config.get("inference", {}).get("num_frames", 49)
        validate_video_shape(frames, height, width)
        return torch.randn(
            (1, 32, (frames - 1) // 4 + 1, height // 16, width // 16),
            device=self.device,
            dtype=self.latent_dtype,
            generator=generator,
        )

    def decode_latent(self, latent):
        with self._active_vae() as vae:
            video = vae.decode(latent.to(self.vae_dtype) / float(self.vae_config["scaling_factor"]), return_dict=False)[0]
        return self.image_processor.postprocess_video(video, output_type="pil")
