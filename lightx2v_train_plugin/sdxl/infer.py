"""SDXL epsilon inference with its native Euler noise schedule."""

import hashlib
import json
import math
from pathlib import Path

import torch
from PIL.PngImagePlugin import PngInfo
from diffusers import EulerDiscreteScheduler
from lightx2v_train.infer.base import BaseInferencer, cached_condition
from lightx2v_train.runtime.distributed import barrier, get_rank, get_world_size
from lightx2v_train.utils.registry import INFERENCER_REGISTER
from loguru import logger

from .sft import validate_condition


@INFERENCER_REGISTER("sdxl_infer")
class SDXLInferencer(BaseInferencer):
    def __init__(self, config):
        # Skip BaseInferencer's flow-matching scheduler.
        self.config = config
        self.infer_config = config.get("inference", {})
        self.output_infer_dir = self.infer_config.get("output_dir")
        if not self.output_infer_dir:
            raise ValueError("SDXL inference requires inference.output_dir.")
        Path(self.output_infer_dir).mkdir(parents=True, exist_ok=True)
        self.model = None
        self.dataloader_val = None
        self.enable_cfg = bool(self.infer_config.get("enable_cfg", True))
        self.guidance_scale = float(self.infer_config.get("cfg_guidance_scale", 5.0))
        self.scheduler = None

    def set_model(self, model):
        self.model = model
        root = self.config["model"]["pretrained_model_name_or_path"]
        self.scheduler = EulerDiscreteScheduler.from_pretrained(root, subfolder="scheduler", local_files_only=True)
        if self.scheduler.config.prediction_type != "epsilon":
            raise ValueError("SDXL inference requires an epsilon-prediction scheduler.")

    def _conditions(self, sample):
        meta = sample.get("meta", {})
        if meta.get("sdxl_cache_version") != 1:
            raise ValueError("SDXL inference requires an SDXL v1 conditioning cache; rebuild cache_val.")
        expected_root = str(Path(self.config["model"]["pretrained_model_name_or_path"]).resolve())
        if meta.get("sdxl_model_root") != expected_root:
            raise ValueError("SDXL conditioning cache belongs to a different base model; rebuild cache_val.")
        if not math.isclose(float(meta.get("sdxl_vae_scaling_factor", 0)), self.model.vae_scaling_factor, rel_tol=1e-7):
            raise ValueError("SDXL conditioning cache VAE scale differs from the model; rebuild cache_val.")
        positive = cached_condition(sample, self.model, "positive")
        negative = cached_condition(sample, self.model, "negative") if self.enable_cfg else None
        if positive is None or (self.enable_cfg and negative is None):
            raise ValueError("SDXL inference requires cached positive and CFG negative conditions.")
        validate_condition(positive, "positive")
        if negative is not None:
            validate_condition(negative, "negative")
        expected_negative = str(self.infer_config.get("negative_prompt", ""))
        if self.enable_cfg and sample["conditioning"].get("negative_prompt") != expected_negative:
            raise ValueError("inference.negative_prompt differs from the cached negative prompt; rebuild cache_val.")
        height, width = (int(value) for value in positive["time_ids"][0, -2:].tolist())
        if height <= 0 or width <= 0 or height % 8 or width % 8:
            raise ValueError(f"Invalid cached SDXL target size: {height}x{width}.")
        return positive, negative, height, width

    @torch.no_grad()
    def infer(self):
        dataset = self.dataloader_val.dataset
        if not getattr(dataset, "uses_cache_dataset", False):
            raise ValueError("sdxl_infer requires cache_dataset; precompute text encoders with cache_sdxl.sh --data_split val.")
        rank, world_size = get_rank(), get_world_size()
        steps = int(self.infer_config.get("num_inference_steps", 30))
        if steps < 1:
            raise ValueError("num_inference_steps must be positive.")
        seed_base = int(self.infer_config.get("seed", 42))
        lora = self.infer_config.get("lora_config") or {}
        if lora.get("path") and getattr(self.model, "_infer_lora_adapter_name", None) is None:
            self.model.load_lora_for_infer(lora["path"])
        self.model.set_denoiser_eval()
        records = []
        # Dummy slots keep FSDP collectives in step on ranks with fewer prompts.
        for slot in range((len(dataset) + world_size - 1) // world_size):
            index = slot * world_size + rank
            has_sample = index < len(dataset)
            sample = dataset[index if has_sample else 0]
            positive, negative, height, width = self._conditions(sample)
            seed = seed_base + index if has_sample else seed_base
            generator = torch.Generator(device=self.model.device).manual_seed(seed)
            self.scheduler.set_timesteps(steps, device=self.model.device)
            noise = self.model.prepare_infer_latents(height, width, generator).float()
            noise_hash = hashlib.sha256(noise.cpu().numpy().tobytes()).hexdigest()
            latent = noise * self.scheduler.init_noise_sigma
            for timestep in self.scheduler.timesteps:
                scaled = self.scheduler.scale_model_input(latent, timestep)
                prediction = self.model.predict_denoiser_output(scaled, timestep, positive)
                if self.enable_cfg:
                    uncond = self.model.predict_denoiser_output(scaled, timestep, negative)
                    prediction = self.model.apply_cfg(prediction, uncond, self.guidance_scale)
                latent = self.scheduler.step(prediction.float(), timestep, latent, return_dict=False)[0].float()
            if not torch.isfinite(latent).all():
                raise RuntimeError(f"Non-finite SDXL inference latent at prompt {index}.")
            if not has_sample:
                continue
            image = self.model.decode_latent(latent)
            if isinstance(image, (list, tuple)):
                image = image[0]
            path = Path(self.output_infer_dir) / f"{index:04d}_seed{seed}.png"
            prompt = sample["conditioning"]["prompt"]
            record = {
                "index": index,
                "prompt": prompt,
                "seed": seed,
                "height": height,
                "width": width,
                "steps": steps,
                "enable_cfg": self.enable_cfg,
                "guidance_scale": self.guidance_scale,
                "initial_noise_sha256": noise_hash,
                "unet_path": self.config["model"].get("unet_path"),
                "lora_path": lora.get("path"),
                "scheduler": type(self.scheduler).__name__,
                "path": str(path),
            }
            info = PngInfo()
            info.add_text("sdxl", json.dumps(record, ensure_ascii=False))
            image.save(path, pnginfo=info)
            records.append(record)
            logger.info("[sdxl_infer] saved {} seed={} size={}x{}", path, seed, height, width)
        manifest = Path(self.output_infer_dir) / f"manifest_rank{rank}.jsonl"
        manifest.write_text("".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records), encoding="utf-8")
        barrier()
        return [record["path"] for record in records]
