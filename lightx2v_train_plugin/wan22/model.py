import json
from contextlib import contextmanager, nullcontext
from copy import deepcopy
from pathlib import Path

import torch
from lightx2v_train.model_capabilities import FlowMatchingSFTCapability
from lightx2v_train.model_zoo.base import BaseModel
from lightx2v_train.model_zoo.wan.wan_t2v import WanT2VModel
from lightx2v_train.utils.registry import MODEL_REGISTER
from peft import LoraConfig

from .sft import Wan22SFTCapability


@MODEL_REGISTER("wan22_t2v_a14b")
class Wan22T2VModel(WanT2VModel):
    def __init__(self, config):
        config = deepcopy(config)
        model = config["model"]
        self.expert = model.get("expert")
        if self.expert not in {"high", "low"}:
            raise ValueError("model.expert must be 'high' or 'low'.")
        if model.get("causal", False):
            raise ValueError("Wan2.2 T2V SFT uses the non-causal transformer.")
        root = Path(model["pretrained_model_name_or_path"])
        model.setdefault("components_root", str(root))
        model["transformer_path"] = str(root / f"{self.expert}_noise_model")
        model.setdefault("running_dtype", "bf16")
        model.setdefault("transformer_param_dtype", "bf16")
        model.setdefault("latent_dtype", "bf16")
        model.setdefault("vae_dtype", "bf16")
        super().__init__(config)

    def register_capabilities(self):
        BaseModel.register_capabilities(self)
        self.capabilities.register(FlowMatchingSFTCapability, Wan22SFTCapability(self))

    def _load_transformer(self, model_path):
        config = json.loads((Path(model_path) / "config.json").read_text(encoding="utf-8"))
        if config.get("model_type") != "t2v" or config.get("in_dim", 16) != 16:
            raise ValueError(f"Expected a native Wan2.2 T2V checkpoint: {model_path}")
        return super()._load_transformer(model_path)

    def add_lora(self, rank, alpha, target_modules):
        # Match DiffSynth's PEFT defaults, rather than the host's Gaussian init.
        self._inject_lora(LoraConfig(r=rank, lora_alpha=alpha, target_modules=target_modules))
        for parameter in self.transformer.parameters():
            if parameter.requires_grad:
                parameter.data = parameter.data.to(self.running_dtype)

    @contextmanager
    def transformer_forward_context(self):
        offload = self.config.get("training", {}).get("gradient_checkpointing_offload", False)
        saved_tensors = torch.autograd.graph.save_on_cpu(pin_memory=self.device.type == "cuda") if offload and torch.is_grad_enabled() else nullcontext()
        with super().transformer_forward_context(), saved_tensors:
            yield

    def prepare_infer_latents(self, height, width, generator=None):
        if self.config.get("inference", {}).get("rand_device", "cpu") != "cpu":
            return super().prepare_infer_latents(height, width, generator)
        frames = self.config.get("inference", {}).get("num_frames", 49)
        shape = (1, self._latent_channels(), (frames - 1) // self.vae_scale_factor_temporal + 1, height // self.vae_scale_factor_spatial, width // self.vae_scale_factor_spatial)
        cpu_generator = None if generator is None else torch.Generator("cpu").manual_seed(generator.initial_seed())
        noise = torch.randn(shape, generator=cpu_generator, dtype=torch.float32)
        return noise.to(device=self.device, dtype=self.latent_dtype)
