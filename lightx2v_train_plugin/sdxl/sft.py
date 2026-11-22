"""SDXL posterior caching and discrete DDPM epsilon training."""

import math
from collections.abc import Mapping
from pathlib import Path

import torch
import torch.nn.functional as F
from diffusers import DDPMScheduler
from lightx2v_train.model_capabilities import LossResult
from lightx2v_train.model_zoo.capability_adapters.common import (
    GenericFlowMatchingCapability,
    _cached_condition,
    _require_singleton_tensor,
    _uses_prompt_dropout,
)

SDXL_CACHE_VERSION = 1


def model_root(model):
    return str(Path(model.config["model"]["pretrained_model_name_or_path"]).expanduser().resolve())


def validate_training_scheduler(scheduler):
    if not isinstance(scheduler, DDPMScheduler):
        raise TypeError("SDXL SFT requires the model's discrete DDPMScheduler, not a flow-matching scheduler.")
    if scheduler.config.prediction_type != "epsilon":
        raise ValueError(f"SDXL SFT requires epsilon prediction, got {scheduler.config.prediction_type!r}.")
    if scheduler.config.num_train_timesteps <= 0:
        raise ValueError("SDXL requires a positive num_train_timesteps.")
    return scheduler


def validate_condition(condition, name):
    if not isinstance(condition, Mapping):
        raise TypeError(f"SDXL {name} condition must be a mapping.")
    for key, shape in (("prompt_embed", (1, 77, 2048)), ("pooled_prompt_embed", (1, 1280)), ("time_ids", (1, 6))):
        value = condition.get(key)
        if not torch.is_tensor(value) or tuple(value.shape) != shape:
            raise ValueError(f"SDXL {name}.{key} must have shape {shape}.")
        if not torch.isfinite(value).all():
            raise ValueError(f"SDXL {name}.{key} contains non-finite values.")


class SDXLSFTCapability(GenericFlowMatchingCapability):
    """Cache frozen encoders; train the UNet with DDPM epsilon MSE."""

    def encode_training_cache(self, batch):
        model = self.model
        inputs = {}
        if batch.get("inputs", {}).get("target_pixel_values") is not None:
            posterior = model.encode_to_cache_posterior(batch)
            inputs = {key: posterior[key].float() for key in ("latent_mean", "latent_logvar")}

        positive = model.encode_condition(batch)
        validate_condition(positive, "positive")
        conditioning = {"prompt": batch["conditioning"]["prompt"], "positive": positive}

        # Empty CFG negatives use zeros; caption dropout uses an encoded caption.
        negative_prompt = str(model.config.get("inference", {}).get("negative_prompt", ""))
        negative_sample = {**batch, "conditioning": {**batch["conditioning"], "prompt": negative_prompt}}
        negative = model.encode_inference_condition(negative_sample, is_negative=True)
        validate_condition(negative, "negative")
        conditioning["negative"] = negative
        conditioning["negative_prompt"] = negative_prompt
        if _uses_prompt_dropout(model):
            unconditional_sample = {**batch, "conditioning": {**batch["conditioning"], "prompt": model.unconditional_prompt}}
            unconditional = model.encode_condition(unconditional_sample)
            validate_condition(unconditional, "unconditional")
            conditioning["unconditional"] = unconditional
            conditioning["unconditional_prompt"] = model.unconditional_prompt

        meta = dict(batch.get("meta", {}))
        meta.update(
            sdxl_cache_version=SDXL_CACHE_VERSION,
            sdxl_model_root=model_root(model),
            sdxl_vae_scaling_factor=float(model.vae_scaling_factor),
        )
        return {"inputs": inputs, "conditioning": conditioning, "meta": meta}

    def _cached_posterior(self, batch):
        model = self.model
        meta = batch.get("meta", {})
        cache_path = meta.get("training_cache_path", "<unknown>")
        if meta.get("sdxl_cache_version") != SDXL_CACHE_VERSION:
            raise ValueError(f"SDXL training requires posterior cache version {SDXL_CACHE_VERSION}; rebuild {cache_path}.")
        if meta.get("sdxl_model_root") != model_root(model):
            raise ValueError(f"SDXL cache was encoded with a different base model; rebuild {cache_path}.")
        scale = meta.get("sdxl_vae_scaling_factor")
        if not isinstance(scale, (int, float)) or not math.isclose(scale, float(model.vae_scaling_factor), rel_tol=1e-7):
            raise ValueError(f"SDXL cache VAE scaling factor differs from the current model; rebuild {cache_path}.")

        values = []
        for key in ("latent_mean", "latent_logvar"):
            value = _require_singleton_tensor(batch.get("inputs", {}).get(key), f"SDXL inputs.{key}")
            if value.ndim != 4 or value.shape[1] != 4 or min(value.shape[-2:]) <= 0 or not value.is_floating_point():
                raise ValueError(f"SDXL inputs.{key} must be floating point [1, 4, height, width].")
            value = value.to(device=model.device, dtype=torch.float32)
            if not torch.isfinite(value).all():
                raise ValueError(f"SDXL inputs.{key} contains non-finite values in {cache_path}.")
            values.append(value)
        mean, logvar = values
        if mean.shape != logvar.shape:
            raise ValueError(f"SDXL cached posterior mean and logvar have different shapes in {cache_path}.")
        return values

    def compute_loss(self, batch, context):
        model = self.model
        scheduler = validate_training_scheduler(context.noise_scheduler)
        broadcast = context.broadcast
        with torch.no_grad():
            mean, logvar = self._cached_posterior(batch)
            # Resample the posterior on each visit to a cached image.
            posterior_noise = torch.randn_like(mean)
            std = (0.5 * logvar.clamp(-30.0, 20.0)).exp()
            latent = broadcast((mean + std * posterior_noise) * float(model.vae_scaling_factor))
            noise = broadcast(torch.randn_like(latent))
            timestep = broadcast(torch.randint(scheduler.config.num_train_timesteps, (1,), device=model.device, dtype=torch.long))
            noisy_latent = scheduler.add_noise(latent, noise, timestep)
            condition = _cached_condition(batch, model)
            validate_condition(condition, "training")
            condition = broadcast(condition)

        prediction = model.predict_denoiser_output(noisy_latent, timestep, condition)
        if prediction.shape != noise.shape:
            raise ValueError(f"SDXL UNet output shape {tuple(prediction.shape)} differs from epsilon target {tuple(noise.shape)}.")
        mse = F.mse_loss(prediction.float(), noise)
        return LossResult(loss=mse, metrics={"mse": mse.detach(), "timestep": timestep.float().mean()})
