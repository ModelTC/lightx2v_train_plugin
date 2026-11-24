"""Z-Image adapter with native image-list inputs and flow-matching loss."""

import math

import torch
from diffusers import AutoencoderKL, ZImagePipeline, ZImageTransformer2DModel
from diffusers.image_processor import VaeImageProcessor
from lightx2v_train.model_capabilities import FlowMatchingSFTCapability
from lightx2v_train.model_zoo.base import BaseModel
from lightx2v_train.model_zoo.capability_adapters.common import (
    GenericFlowMatchingCapability,
)
from lightx2v_train.runtime.attention import prepare_diffusers_attention
from lightx2v_train.runtime.backend import get_backend
from lightx2v_train.runtime.distributed import get_sequence_parallel_world_size
from lightx2v_train.runtime.fsdp import fsdp2_enabled
from lightx2v_train.utils.registry import MODEL_REGISTER
from lightx2v_train.utils.utils import is_cache_build, is_train_cache_dataset


@MODEL_REGISTER("z_image")
class ZImageModel(BaseModel):
    pipeline_cls = ZImagePipeline
    default_unconditional_prompt = ""

    def register_capabilities(self):
        super().register_capabilities()
        self.capabilities.register(
            FlowMatchingSFTCapability,
            GenericFlowMatchingCapability(self),
        )

    def load_components(
        self,
        *,
        load_transformer,
        load_vae,
        load_condition_encoder,
    ):
        if get_sequence_parallel_world_size() > 1:
            raise ValueError("Z-Image does not support sequence parallel training; set distributed.sequence_parallel.enabled=false.")
        model_path = self.config["model"]["pretrained_model_name_or_path"]
        if load_condition_encoder:
            self._load_condition_encoder(model_path)
        if load_vae:
            self._load_vae(model_path)
            self.vae_config = self.vae.config
        else:
            self.vae_config = AutoencoderKL.load_config(model_path, subfolder="vae")
        self.vae_scale_factor = self._scale_factor_from_config(self.vae_config)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor * 2)
        if load_transformer:
            self.transformer = self.load_transformer(model_path)

    def _load_condition_encoder(self, model_path):
        self.text_pipeline = self.pipeline_cls.from_pretrained(
            model_path,
            transformer=None,
            vae=None,
            torch_dtype=self.text_encoder_dtype,
        ).to(self.device)
        self.text_pipeline.text_encoder.requires_grad_(False)
        self.text_pipeline.text_encoder.eval()

    def _load_vae(self, model_path):
        use_cpu = not is_cache_build(self.config) and is_train_cache_dataset(self.config) and self.config.get("inference", {}).get("vae_cpu_offload", False)
        device = torch.device("cpu") if use_cpu else self.device
        self.vae = AutoencoderKL.from_pretrained(
            model_path,
            subfolder="vae",
            torch_dtype=self.vae_dtype,
        ).to(device)
        self.vae.requires_grad_(False)
        self.vae.eval()

    @staticmethod
    def _scale_factor_from_config(config):
        block_out_channels = config.get("block_out_channels")
        if not block_out_channels:
            raise ValueError("Z-Image VAE config has no block_out_channels.")
        return 2 ** (len(block_out_channels) - 1)

    def reuse_frozen_components_from(self, source):
        super().reuse_frozen_components_from(source)
        self.vae_scale_factor = source.vae_scale_factor

    def load_transformer(self, model_path):
        device = get_backend().host_device if fsdp2_enabled(self.config) else self.device
        transformer = ZImageTransformer2DModel.from_pretrained(
            model_path,
            subfolder="transformer",
            torch_dtype=self.transformer_param_dtype,
        ).to(device)
        return prepare_diffusers_attention(transformer, self.config)

    def denoiser_module(self):
        return self.transformer

    def fsdp2_shard_plan(self, fsdp_config):
        reshard = fsdp_config["reshard_after_forward"]
        blocks = (
            self.transformer.noise_refiner,
            self.transformer.context_refiner,
            self.transformer.layers,
        )
        plan = [{"modules": modules, "reshard_after_forward": reshard["block_reshard"]} for modules in blocks]
        plan.append({"module": self.transformer, "reshard_after_forward": reshard["root_reshard"]})
        return plan

    def add_lora(self, rank, alpha, target_modules):
        super().add_lora(
            rank,
            alpha,
            target_modules or ["to_q", "to_k", "to_v", "to_out.0", "w1", "w2", "w3"],
        )

    @property
    def latent_channels(self):
        channels = self.vae_config.get("latent_channels")
        if channels is None:
            channels = self.transformer.config.in_channels
        return int(channels)

    def _latent_scale_shift(self, latents):
        scaling = float(self.vae_config.get("scaling_factor", 1.0))
        shift = self.vae_config.get("shift_factor", 0.0)
        shift = 0.0 if shift is None else float(shift)
        if not math.isfinite(scaling) or scaling <= 0:
            raise ValueError("Z-Image VAE scaling_factor must be positive and finite.")
        if not math.isfinite(shift):
            raise ValueError("Z-Image VAE shift_factor must be finite.")
        return latents.new_tensor(scaling), latents.new_tensor(shift)

    def _normalize_latents(self, latents):
        scaling, shift = self._latent_scale_shift(latents)
        return (latents - shift) * scaling

    def _denormalize_latents(self, latents):
        scaling, shift = self._latent_scale_shift(latents)
        return latents / scaling + shift

    def encode_to_latent(self, sample):
        return self._encode_target_latent(sample, mode="sample")

    def encode_to_cache_latent(self, sample):
        return self._encode_target_latent(sample, mode="mode")

    def _encode_target_latent(self, sample, *, mode):
        image = sample["inputs"]["target_pixel_values"]
        if image.ndim == 3:
            image = image.unsqueeze(0)
        if image.ndim != 4:
            raise ValueError(f"Expected target_pixel_values with shape [B, C, H, W], got {tuple(image.shape)}")
        height, width = image.shape[-2:]
        multiple = self.vae_scale_factor * 2
        if height <= 0 or width <= 0 or height % multiple or width % multiple:
            raise ValueError(f"Z-Image target dimensions must be positive multiples of {multiple}, got {height}x{width}.")
        image = image.to(device=self.device, dtype=self.vae_dtype)
        with self._active_vae() as vae:
            latent = getattr(vae.encode(image).latent_dist, mode)()
        return self._normalize_latents(latent.to(dtype=self.latent_dtype))

    def encode_condition(self, sample):
        return self.encode_prompt_condition(sample["conditioning"]["prompt"])

    def encode_prompt_condition(self, prompt):
        prompts = [prompt] if isinstance(prompt, str) else list(prompt)
        if not prompts:
            raise ValueError("Z-Image training requires at least one prompt.")
        prompt_embeds, _ = self.text_pipeline.encode_prompt(
            prompt=prompts,
            device=self.device,
            do_classifier_free_guidance=False,
            max_sequence_length=self.config["model"].get("max_sequence_length", 512),
        )
        return {"prompt_embed": prompt_embeds}

    def prepare_denoiser_input(self, noisy_latent, condition=None):
        del condition
        if noisy_latent.ndim != 4:
            raise ValueError(f"Z-Image expects image latents with shape [B, C, H, W], got {tuple(noisy_latent.shape)}")
        # The native transformer takes one [C, 1, H, W] tensor per image.
        return [latent.to(device=self.device, dtype=self.running_dtype).unsqueeze(1) for latent in noisy_latent.unbind(dim=0)]

    def denoise(self, denoiser_input, timestep_or_sigma, condition):
        prompt_embeds = condition.get("prompt_embed")
        if prompt_embeds is None:
            raise KeyError("Z-Image condition is missing prompt_embed.")
        if not isinstance(prompt_embeds, list):
            prompt_embeds = [prompt_embeds]
        batch_size = len(denoiser_input)
        if len(prompt_embeds) != batch_size:
            if len(prompt_embeds) == 1:
                prompt_embeds = prompt_embeds * batch_size
            else:
                raise ValueError(f"Z-Image prompt batch size {len(prompt_embeds)} does not match latent batch size {batch_size}.")
        sigma = timestep_or_sigma.float().reshape(-1)
        if sigma.numel() == 1:
            sigma = sigma.expand(batch_size)
        if sigma.numel() != batch_size:
            raise ValueError(f"Z-Image timestep batch size {sigma.numel()} does not match {batch_size}.")
        # Z-Image uses the reverse timestep and opposite velocity sign.
        transformer_t = (1.0 - sigma).to(device=self.device, dtype=torch.float32)
        prompt_embeds = [item.to(device=self.device, dtype=self.running_dtype) for item in prompt_embeds]
        with self.transformer_forward_context():
            outputs = self.transformer(
                denoiser_input,
                transformer_t,
                prompt_embeds,
                return_dict=False,
                patch_size=self.transformer.config.all_patch_size[0],
                f_patch_size=self.transformer.config.all_f_patch_size[0],
            )[0]
        prediction = torch.stack([item[:, 0] for item in outputs], dim=0)
        return -prediction

    def postprocess_denoiser_output(self, prediction, denoiser_input):
        del denoiser_input
        return prediction

    def apply_cfg(self, positive, negative, guidance_scale):
        # Native Z-Image CFG starts at the conditional prediction (scale=0).
        positive, negative = positive.float(), negative.float()
        return positive + guidance_scale * (positive - negative)

    def prepare_infer_latents(self, height, width, generator=None):
        multiple = self.vae_scale_factor * 2
        if height <= 0 or width <= 0 or height % multiple or width % multiple:
            raise ValueError(f"Z-Image dimensions must be positive multiples of {multiple}, got {height}x{width}.")
        shape = (
            1,
            self.latent_channels,
            height // self.vae_scale_factor,
            width // self.vae_scale_factor,
        )
        return torch.randn(shape, generator=generator, device=self.device, dtype=self.latent_dtype)

    def decode_latent(self, latent):
        latent = self._denormalize_latents(latent.to(dtype=self.vae_dtype))
        with self._active_vae() as vae:
            image = vae.decode(latent, return_dict=False)[0]
        return self.image_processor.postprocess(image, output_type="pil")
