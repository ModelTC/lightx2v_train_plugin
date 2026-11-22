"""SDXL Base UNet, frozen VAE and dual CLIP encoders."""

import json
import math
from pathlib import Path

import torch
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionXLPipeline, UNet2DConditionModel
from diffusers.image_processor import VaeImageProcessor
from diffusers.utils import convert_state_dict_to_diffusers, convert_unet_state_dict_to_peft
from diffusers.utils.peft_utils import get_adapter_name
from lightx2v_train.model_capabilities import FlowMatchingSFTCapability
from lightx2v_train.model_zoo.base import BaseModel
from lightx2v_train.runtime.backend import get_backend
from lightx2v_train.runtime.distributed import get_sequence_parallel_world_size, is_main_process
from lightx2v_train.runtime.fsdp import fsdp2_enabled, is_fsdp2_module
from lightx2v_train.utils.registry import MODEL_REGISTER
from loguru import logger
from peft.utils import get_peft_model_state_dict, set_peft_model_state_dict
from safetensors import safe_open
from safetensors.torch import load_file
from torch.distributed.checkpoint.state_dict import StateDictOptions, get_state_dict
from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer


@MODEL_REGISTER("sdxl")
class SDXLModel(BaseModel):
    pipeline_cls = StableDiffusionXLPipeline
    default_unconditional_prompt = ""
    default_lora_target_modules = ("to_q", "to_k", "to_v", "to_out.0")

    def __init__(self, config):
        model_config = dict(config["model"])
        model_config.setdefault("running_dtype", "bf16")
        model_config.setdefault("vae_dtype", "fp32")
        config = {**config, "model": model_config}
        if config.get("training", {}).get("method") == "sdxl_sft":
            if config.get("data", {}).get("train", {}).get("name") != "cache_dataset":
                raise ValueError("SDXL SFT requires data.train.name=cache_dataset; encode the frozen VAE and dual CLIP outputs first.")
            if config.get("inference", {}).get("method", "none") != "none":
                raise ValueError("SDXL SFT requires inference.method=none; run image inference separately after saving the UNet.")
        super().__init__(config)
        if self.vae_dtype != torch.float32:
            raise ValueError("SDXL Base requires model.vae_dtype=fp32 for stable posterior caching and decoding.")
        self.unet = None
        self.text_encoder = None
        self.text_encoder_2 = None
        self.tokenizer = None
        self.tokenizer_2 = None
        self.vae_scale_factor = 8
        self.vae_scaling_factor = 0.13025
        self.force_zeros_for_empty_prompt = True

    def register_capabilities(self):
        from .sft import SDXLSFTCapability

        super().register_capabilities()
        self.capabilities.register(FlowMatchingSFTCapability, SDXLSFTCapability(self))

    def load_components(self, *, load_transformer, load_vae, load_condition_encoder):
        if get_sequence_parallel_world_size() > 1:
            raise ValueError("SDXL does not support sequence parallelism; set distributed.sequence_parallel.enabled=false.")
        model_path = self.config["model"]["pretrained_model_name_or_path"]
        self._load_vae_config(model_path)
        model_index = Path(model_path) / "model_index.json"
        if model_index.is_file():
            self.force_zeros_for_empty_prompt = json.loads(model_index.read_text(encoding="utf-8")).get("force_zeros_for_empty_prompt", True)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)
        if load_condition_encoder:
            self._load_condition_encoder(model_path)
        if load_vae:
            self.vae = AutoencoderKL.from_pretrained(model_path, subfolder="vae", torch_dtype=torch.float32).to(self.device)
            self.vae.requires_grad_(False)
            self.vae.eval()
        if load_transformer:
            self.unet = self.load_transformer(model_path)
        logger.info(
            "[sdxl] loaded unet={}, vae={}, text_encoder={}, text_encoder_2={}",
            self.unet is not None,
            self.vae is not None,
            self.text_encoder is not None,
            self.text_encoder_2 is not None,
        )

    def _load_vae_config(self, model_path):
        self.vae_config = AutoencoderKL.load_config(model_path, subfolder="vae")
        self.vae_scale_factor = 2 ** (len(self.vae_config["block_out_channels"]) - 1)
        self.vae_scaling_factor = float(self.vae_config["scaling_factor"])
        if self.vae_scale_factor != 8 or int(self.vae_config["latent_channels"]) != 4:
            raise ValueError("SDXL Base requires a four-channel VAE with spatial downsampling factor eight.")
        if not math.isfinite(self.vae_scaling_factor) or self.vae_scaling_factor <= 0:
            raise ValueError("SDXL VAE scaling_factor must be positive and finite.")
        if self.vae_config.get("shift_factor") not in (None, 0, 0.0):
            raise ValueError("SDXL Base does not use a latent shift factor.")

    def _load_condition_encoder(self, model_path):
        self.tokenizer = CLIPTokenizer.from_pretrained(model_path, subfolder="tokenizer")
        self.tokenizer_2 = CLIPTokenizer.from_pretrained(model_path, subfolder="tokenizer_2")
        self.text_encoder = CLIPTextModel.from_pretrained(model_path, subfolder="text_encoder", torch_dtype=self.text_encoder_dtype).to(self.device)
        self.text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(model_path, subfolder="text_encoder_2", torch_dtype=self.text_encoder_dtype).to(self.device)
        for encoder in (self.text_encoder, self.text_encoder_2):
            encoder.requires_grad_(False)
            encoder.eval()

    def reuse_frozen_components_from(self, source):
        super().reuse_frozen_components_from(source)
        for name in ("text_encoder", "text_encoder_2", "tokenizer", "tokenizer_2", "vae_scale_factor", "vae_scaling_factor", "force_zeros_for_empty_prompt"):
            setattr(self, name, getattr(source, name))

    def load_transformer(self, model_path):
        unet_path = Path(self.config["model"].get("unet_path") or Path(model_path) / "unet")
        if not (unet_path / "config.json").is_file() and (unet_path / "unet/config.json").is_file():
            unet_path = unet_path / "unet"
        unet_config = UNet2DConditionModel.load_config(unet_path)
        expected = {
            "in_channels": 4,
            "out_channels": 4,
            "cross_attention_dim": 2048,
            "addition_embed_type": "text_time",
            "addition_time_embed_dim": 256,
            "projection_class_embeddings_input_dim": 2816,
        }
        for name, value in expected.items():
            if unet_config.get(name) != value:
                raise ValueError(f"Only SDXL Base UNets are supported: expected {name}={value!r}, got {unet_config.get(name)!r}.")
        device = get_backend().host_device if fsdp2_enabled(self.config) else self.device
        unet = UNet2DConditionModel.from_pretrained(unet_path, torch_dtype=self.transformer_param_dtype).to(device)
        if self.config.get("training", {}).get("gradient_checkpointing", True):
            unet.enable_gradient_checkpointing()
        return unet

    def denoiser_module(self):
        return self.unet

    def fsdp2_shard_plan(self, fsdp_config):
        reshard = fsdp_config["reshard_after_forward"]
        blocks = [*self.unet.down_blocks, self.unet.mid_block, *self.unet.up_blocks]
        return [
            {"modules": [block for block in blocks if block is not None], "reshard_after_forward": reshard["block_reshard"]},
            {"module": self.unet, "reshard_after_forward": reshard["root_reshard"]},
        ]

    def add_lora(self, rank, alpha, target_modules):
        super().add_lora(rank, alpha, target_modules or list(self.default_lora_target_modules))
        self._upcast_lora_parameters()

    def _upcast_lora_parameters(self):
        for name, parameter in self.unet.named_parameters():
            if "lora_" in name:
                parameter.data = parameter.data.float()

    def set_lora_trainable(self):
        super().set_lora_trainable()
        self._upcast_lora_parameters()

    def load_training_scheduler(self):
        scheduler_config = DDPMScheduler.load_config(self.config["model"]["pretrained_model_name_or_path"], subfolder="scheduler")
        overrides = self.config.get("scheduler", {})
        if overrides.get("prediction_type", "epsilon") != "epsilon" or scheduler_config.get("prediction_type", "epsilon") != "epsilon":
            raise ValueError("SDXL Base SFT only supports epsilon prediction.")
        steps = int(overrides.get("num_train_timesteps", scheduler_config.get("num_train_timesteps", 1000)))
        if steps != 1000:
            raise ValueError("SDXL Base SFT requires scheduler.num_train_timesteps=1000.")
        return DDPMScheduler(
            num_train_timesteps=steps,
            beta_start=scheduler_config.get("beta_start", 0.00085),
            beta_end=scheduler_config.get("beta_end", 0.012),
            beta_schedule=scheduler_config.get("beta_schedule", "scaled_linear"),
            trained_betas=scheduler_config.get("trained_betas"),
            prediction_type="epsilon",
            clip_sample=False,
        )

    @torch.no_grad()
    def encode_to_cache_posterior(self, sample):
        if self.vae is None:
            raise RuntimeError("Load the frozen SDXL VAE before building the latent posterior cache.")
        image = sample["inputs"]["target_pixel_values"]
        if image.ndim == 3:
            image = image.unsqueeze(0)
        if image.ndim != 4 or image.shape[1] != 3:
            raise ValueError(f"SDXL target_pixel_values must have shape [B, 3, H, W], got {tuple(image.shape)}.")
        height, width = image.shape[-2:]
        if height <= 0 or width <= 0 or height % 8 or width % 8:
            raise ValueError(f"SDXL target dimensions must be positive multiples of eight, got {height}x{width}.")
        with self._active_vae() as vae:
            posterior = vae.encode(image.to(device=self.device, dtype=torch.float32)).latent_dist
        # Store the unscaled distribution; every optimizer step samples it anew.
        return {"latent_mean": posterior.mean.float(), "latent_logvar": posterior.logvar.float()}

    def encode_to_latent(self, sample):
        posterior = self.encode_to_cache_posterior(sample)
        mean, logvar = posterior["latent_mean"], posterior["latent_logvar"]
        return (mean + torch.exp(0.5 * logvar) * torch.randn_like(mean)) * self.vae_scaling_factor

    def encode_to_cache_latent(self, sample):
        raise RuntimeError("SDXL caches a VAE posterior, not a fixed latent; use encode_to_cache_posterior through the SDXL SFT capability.")

    @staticmethod
    def _prompts(prompt):
        prompts = [prompt] if isinstance(prompt, str) or prompt is None else list(prompt)
        if not prompts or any(value is not None and not isinstance(value, str) for value in prompts):
            raise ValueError("SDXL prompts must be a nonempty string sequence.")
        return [value or "" for value in prompts]

    def _time_ids(self, sample, batch_size):
        meta = sample.get("meta", {}) if sample is not None else {}
        time_ids = meta.get("time_ids")
        if time_ids is None:
            inference = self.config.get("inference", {})
            height, width = int(inference.get("default_height", 1024)), int(inference.get("default_width", 1024))
            keys = ("original_height", "original_width", "crop_top", "crop_left", "target_height", "target_width")
            defaults = (height, width, 0, 0, height, width)
            columns = [torch.as_tensor(meta.get(key, default)).reshape(-1) for key, default in zip(keys, defaults, strict=True)]
            time_ids = torch.stack(torch.broadcast_tensors(*columns), dim=-1)
        time_ids = torch.as_tensor(time_ids, device=self.device, dtype=torch.float32)
        if time_ids.ndim == 1:
            time_ids = time_ids.unsqueeze(0)
        if time_ids.ndim != 2 or time_ids.shape[1] != 6 or time_ids.shape[0] not in (1, batch_size):
            raise ValueError(f"SDXL time_ids must have shape [{batch_size}, 6], got {tuple(time_ids.shape)}.")
        time_ids = time_ids.expand(batch_size, -1)
        if not torch.isfinite(time_ids).all() or (time_ids[:, (0, 1, 4, 5)] <= 0).any() or (time_ids[:, (2, 3)] < 0).any():
            raise ValueError("SDXL time_ids require positive original/target sizes and nonnegative crop offsets.")
        return time_ids

    @torch.no_grad()
    def _encode_condition(self, prompts, sample=None, *, zero_prompt=False):
        if zero_prompt:
            prompt_embed = torch.zeros(len(prompts), 77, 2048, device=self.device, dtype=self.text_encoder_dtype)
            pooled_embed = torch.zeros(len(prompts), 1280, device=self.device, dtype=self.text_encoder_dtype)
        else:
            if self.text_encoder is None or self.text_encoder_2 is None:
                raise RuntimeError("Load both frozen SDXL text encoders before encoding prompts.")
            hidden_states = []
            for tokenizer, encoder in ((self.tokenizer, self.text_encoder), (self.tokenizer_2, self.text_encoder_2)):
                token_ids = tokenizer(prompts, padding="max_length", max_length=77, truncation=True, return_tensors="pt").input_ids.to(self.device)
                output = encoder(token_ids, output_hidden_states=True)
                hidden_states.append(output.hidden_states[-2])
            prompt_embed = torch.cat(hidden_states, dim=-1).to(dtype=self.text_encoder_dtype)
            pooled_embed = output.text_embeds.to(dtype=self.text_encoder_dtype)
        return {
            "prompt_embed": prompt_embed,
            "pooled_prompt_embed": pooled_embed,
            "time_ids": self._time_ids(sample, len(prompts)),
        }

    def encode_prompt_condition(self, prompt):
        return self._encode_condition(self._prompts(prompt), zero_prompt=prompt is None and self.force_zeros_for_empty_prompt)

    def encode_condition(self, sample):
        # An empty positive prompt still passes through both CLIP encoders.
        return self._encode_condition(self._prompts(sample["conditioning"].get("prompt", "")), sample)

    def encode_inference_condition(self, sample, *, is_negative=False):
        prompts = self._prompts(sample["conditioning"].get("prompt", ""))
        zero_prompt = is_negative and self.force_zeros_for_empty_prompt and not any(prompts)
        return self._encode_condition(prompts, sample, zero_prompt=zero_prompt)

    def prepare_denoiser_input(self, noisy_latent, condition=None):
        del condition
        if noisy_latent.ndim != 4 or noisy_latent.shape[1] != 4:
            raise ValueError(f"SDXL expects latents with shape [B, 4, H, W], got {tuple(noisy_latent.shape)}.")
        return noisy_latent

    def denoise(self, denoiser_input, timesteps, condition):
        batch_size = denoiser_input.shape[0]
        prompt = condition["prompt_embed"].to(device=self.device, dtype=self.running_dtype)
        pooled = condition["pooled_prompt_embed"].to(device=self.device, dtype=self.running_dtype)
        time_ids = condition["time_ids"].to(device=self.device, dtype=torch.float32)
        if prompt.shape != (batch_size, 77, 2048) or pooled.shape != (batch_size, 1280) or time_ids.shape != (batch_size, 6):
            raise ValueError(f"Invalid SDXL conditioning shapes: prompt={tuple(prompt.shape)}, pooled={tuple(pooled.shape)}, time_ids={tuple(time_ids.shape)}.")
        timesteps = torch.as_tensor(timesteps, device=self.device).reshape(-1)
        if timesteps.numel() == 1:
            timesteps = timesteps.expand(batch_size)
        if timesteps.numel() != batch_size:
            raise ValueError("SDXL timestep batch size does not match latent batch size.")
        with self.transformer_forward_context():
            return self.unet(
                sample=denoiser_input.to(device=self.device, dtype=self.running_dtype),
                timestep=timesteps,
                encoder_hidden_states=prompt,
                added_cond_kwargs={"text_embeds": pooled, "time_ids": time_ids},
                return_dict=False,
            )[0]

    def postprocess_denoiser_output(self, prediction, denoiser_input):
        del denoiser_input
        return prediction

    def denoiser_prediction_type(self):
        return "epsilon"

    def prepare_infer_latents(self, height, width, generator=None):
        if height <= 0 or width <= 0 or height % 8 or width % 8:
            raise ValueError("SDXL inference dimensions must be positive multiples of eight.")
        return torch.randn((1, 4, height // 8, width // 8), generator=generator, device=self.device, dtype=torch.float32)

    @torch.no_grad()
    def decode_latent(self, latent):
        if self.vae is None:
            raise RuntimeError("Load the frozen SDXL VAE before decoding latents.")
        with self._active_vae() as vae:
            image = vae.decode(latent.to(device=self.device, dtype=torch.float32) / self.vae_scaling_factor, return_dict=False)[0]
        if not torch.isfinite(image).all():
            raise RuntimeError("SDXL VAE decoded a non-finite image.")
        return self.image_processor.postprocess(image, output_type="pil")

    def save_lora_weights(self, save_dir, adapter_name=None, weights_subdir=None, *, auxiliary_parameter_names=(), auxiliary_weights_name=None):
        del auxiliary_weights_name
        if auxiliary_parameter_names:
            raise ValueError("SDXL UNet LoRA does not have auxiliary trainable parameters.")
        peft_state, _ = self._get_lora_and_auxiliary_state_dict_for_save(adapter_name=adapter_name)
        if not is_main_process():
            return
        output_dir = Path(save_dir) / (weights_subdir or "")
        self.pipeline_cls.save_lora_weights(
            output_dir,
            unet_lora_layers=convert_state_dict_to_diffusers(peft_state),
            unet_lora_adapter_metadata=self.unet.peft_config[adapter_name or "default"].to_dict(),
            safe_serialization=True,
        )

    @staticmethod
    def _read_lora_checkpoint(lora_path):
        weight_path = Path(lora_path)
        if weight_path.is_dir():
            weight_path = weight_path / "pytorch_lora_weights.safetensors"
        raw = load_file(str(weight_path), device="cpu")
        if not raw:
            raise RuntimeError("Cannot load an empty SDXL LoRA checkpoint.")
        if any(key.startswith(("text_encoder.", "text_encoder_2.", "transformer.")) for key in raw):
            raise ValueError("SDXL training supports UNet LoRA weights only.")
        state = {key.removeprefix("unet."): value for key, value in raw.items()}
        if len(state) != len(raw):
            raise ValueError("Duplicate normalized SDXL LoRA keys.")
        if any("lora_A" not in key and "lora_B" not in key for key in state):
            state = convert_unet_state_dict_to_peft(state)
        for key, value in state.items():
            if not key.endswith((".lora_A.weight", ".lora_B.weight")):
                raise ValueError(f"Unsupported SDXL LoRA tensor: {key}")
            pair = key.replace(".lora_A.weight", ".lora_B.weight") if key.endswith(".lora_A.weight") else key.replace(".lora_B.weight", ".lora_A.weight")
            if pair not in state or not torch.isfinite(value).all():
                raise ValueError(f"SDXL LoRA tensor is non-finite or lacks its pair: {key}")
        with safe_open(str(weight_path), framework="pt", device="cpu") as checkpoint:
            raw_metadata = (checkpoint.metadata() or {}).get("lora_adapter_metadata")
        metadata = json.loads(raw_metadata) if raw_metadata else None
        if metadata is not None and any(key.startswith("unet.") for key in metadata):
            metadata = {key.removeprefix("unet."): value for key, value in metadata.items() if key.startswith("unet.")}
        return str(weight_path), state, metadata

    def _verify_lora_state(self, expected, adapter_name):
        loaded = get_peft_model_state_dict(self.unet, adapter_name=adapter_name)
        if loaded.keys() != expected.keys():
            raise RuntimeError(f"Incomplete SDXL LoRA load: missing={sorted(expected.keys() - loaded.keys())}, unexpected={sorted(loaded.keys() - expected.keys())}.")
        for key, value in loaded.items():
            actual = value.detach().cpu()
            if not torch.equal(actual, expected[key].to(dtype=actual.dtype)):
                raise RuntimeError(f"SDXL LoRA tensor was not loaded correctly: {key}")

    def load_lora_for_infer(self, lora_path, adapter_name=None):
        adapter_name = adapter_name or get_adapter_name(self.unet)
        weight_path, state, metadata = self._read_lora_checkpoint(lora_path)
        self.unet.load_lora_adapter(state, adapter_name=adapter_name, prefix=None, metadata=metadata)
        self._verify_lora_state(state, adapter_name)
        self._infer_lora_adapter_name = adapter_name
        logger.info("Loaded SDXL UNet inference LoRA from {} adapter={} tensors={}", weight_path, adapter_name, len(state))

    def load_lora_weights_for_resume(self, lora_path, adapter_name=None, weights_subdir=None):
        adapter_name = adapter_name or "default"
        weights_dir = Path(lora_path) / (weights_subdir or "")
        _, state, metadata = self._read_lora_checkpoint(weights_dir)
        if not metadata:
            raise ValueError("SDXL LoRA resume requires saved adapter metadata to preserve rank/alpha scaling.")
        current = self.unet.peft_config[adapter_name].to_dict()
        for key in ("r", "lora_alpha", "rank_pattern", "alpha_pattern", "use_rslora", "use_dora", "lora_bias"):
            if metadata.get(key) != current.get(key):
                raise ValueError(f"SDXL LoRA resume mismatch for {key}: checkpoint={metadata.get(key)!r}, current={current.get(key)!r}.")
        if set(metadata.get("target_modules") or ()) != set(current.get("target_modules") or ()):
            raise ValueError("SDXL LoRA resume target_modules do not match the saved adapter.")
        incompatible = set_peft_model_state_dict(self.unet, state, adapter_name=adapter_name)
        if incompatible and incompatible.unexpected_keys:
            raise RuntimeError(f"Unexpected SDXL LoRA resume keys: {incompatible.unexpected_keys}")
        self._verify_lora_state(state, adapter_name)
        self._upcast_lora_parameters()

    def save_full_model(self, save_dir):
        unet_dir = Path(save_dir) / "unet"
        if not is_fsdp2_module(self.unet):
            if is_main_process():
                self.unet.save_pretrained(unet_dir, safe_serialization=True)
            return
        options = StateDictOptions(full_state_dict=True, cpu_offload=True, ignore_frozen_params=False, strict=False)
        logger.info("[checkpoint] gathering SDXL full UNet state dict")
        state_dict, _ = get_state_dict(self.unet, (), options=options)
        if is_main_process():
            self._save_full_state_dict(unet_dir, self.unet, state_dict)
            # Export the Diffusers class, not FSDP's runtime wrapper.
            config_path = unet_dir / "config.json"
            unet_config = json.loads(config_path.read_text(encoding="utf-8"))
            unet_config["_class_name"] = "UNet2DConditionModel"
            temporary = config_path.with_suffix(".json.tmp")
            temporary.write_text(json.dumps(unet_config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            temporary.replace(config_path)
