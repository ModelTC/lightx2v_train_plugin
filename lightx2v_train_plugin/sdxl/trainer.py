"""UNet-only SDXL training and checkpoint compatibility checks."""

import hashlib
from collections.abc import Mapping
from pathlib import Path

import yaml
from lightx2v_train.runtime.distributed import barrier, is_main_process
from lightx2v_train.trainers.flow_matching import FlowMatchingTrainer
from lightx2v_train.utils.registry import TRAINER_REGISTER

from .sft import SDXL_CACHE_VERSION, model_root, validate_training_scheduler


def _plain_value(value):
    """Convert config objects to values accepted by YAML."""
    if isinstance(value, Mapping):
        return {str(key): _plain_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain_value(item) for item in value]
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return value


@TRAINER_REGISTER("sdxl_sft")
class SDXLSFTTrainer(FlowMatchingTrainer):
    trainer_name = "sdxl_sft"

    def __init__(self, config):
        if config["model"].get("name") != "sdxl":
            raise ValueError("sdxl_sft requires model.name=sdxl.")
        if config.get("data", {}).get("train", {}).get("name") != "cache_dataset":
            raise ValueError("SDXL training requires data.train.name=cache_dataset; run the frozen VAE/text cache pass first.")
        if config.get("inference", {}).get("method", "none") != "none":
            raise ValueError("SDXL UNet-only training requires inference.method=none; run inference separately after saving.")
        super().__init__(config)

    def set_model(self, model):
        super().set_model(model)
        if model.denoiser_module() is not model.unet:
            raise ValueError("SDXL training must use the UNet as its only denoiser.")
        for name in ("vae", "text_encoder", "text_encoder_2", "text_pipeline"):
            if getattr(model, name, None) is not None:
                raise ValueError(f"SDXL cached training must not load {name}; use a separate cache process.")
        self.noise_scheduler = validate_training_scheduler(model.load_training_scheduler())

    def _training_identity(self):
        base_path = model_root(self.model)
        unet_path = Path(self.model_config.get("unet_path") or Path(base_path) / "unet").expanduser().resolve()
        data = self.config["data"]["train"]
        paths = data["data_path"]
        if isinstance(paths, (str, Path)):
            paths = [paths]
        manifests = []
        for path in paths:
            source = Path(path).expanduser().resolve()
            if not source.is_file():
                raise RuntimeError(f"SDXL cache manifest does not exist: {source}")
            manifests.append({"path": str(source), "sha256": hashlib.sha256(source.read_bytes()).hexdigest()})
        lora = None
        if self.train_type == "lora":
            targets = self.lora_target_modules
            if not targets:
                targets = ["to_q", "to_k", "to_v", "to_out.0"]
            lora = {"rank": int(self.lora_rank), "alpha": int(self.lora_alpha), "target_modules": targets}
        # Exclude Diffusers' bookkeeping fields.
        scheduler = {key: value for key, value in self.noise_scheduler.config.items() if not key.startswith("_")}
        return _plain_value(
            {
                "version": 1,
                "model_root": base_path,
                "unet_path": str(unet_path),
                "vae_scaling_factor": float(self.model.vae_scaling_factor),
                "train_type": self.train_type,
                "lora": lora,
                "scheduler": scheduler,
                "cache_version": SDXL_CACHE_VERSION,
                "cache_manifests": manifests,
                "prompt_dropout_rate": float(data.get("prompt_dropout_rate", 0.0)),
                "generation_shapes": data.get("generation_shapes"),
                "running_dtype": self.model_config["running_dtype"],
            }
        )

    def save_checkpoint(self, iteration, save_total_limit):
        super().save_checkpoint(iteration, save_total_limit)
        checkpoint_dir = Path(self.output_train_dir) / f"checkpoint-{iteration:09d}"
        if self.train_type == "full":
            # FSDP state collection needs every rank.
            self.model.save_full_model(str(checkpoint_dir))
        barrier()
        if is_main_process():
            saved = {**self.config, "sdxl_training_identity": self._training_identity()}
            path = checkpoint_dir / "config.yaml"
            temporary = path.with_suffix(".yaml.tmp")
            temporary.write_text(yaml.safe_dump(_plain_value(saved), sort_keys=False), encoding="utf-8")
            temporary.replace(path)
        barrier()

    def _validate_checkpoint_metadata(self, state, state_path, resume_ckpt_path):
        super()._validate_checkpoint_metadata(state, state_path, resume_ckpt_path)
        path = Path(resume_ckpt_path) / "config.yaml"
        if not path.is_file():
            raise RuntimeError(f"SDXL resume requires the saved resolved configuration: {path}")
        saved = yaml.safe_load(path.read_text(encoding="utf-8"))
        previous = saved.get("sdxl_training_identity") if isinstance(saved, Mapping) else None
        if not isinstance(previous, Mapping):
            raise RuntimeError(f"SDXL resume requires checkpoint provenance metadata: {path}")
        if self.train_type == "full":
            unet = Path(resume_ckpt_path) / "unet"
            weights = list(unet.glob("*.safetensors")) + list(unet.glob("*.bin"))
            if not (unet / "config.json").is_file() or not weights:
                raise RuntimeError(f"SDXL full checkpoint is incomplete: its Diffusers UNet export is missing from {unet}.")
        current = self._training_identity()
        for key in sorted(set(previous) | set(current)):
            if previous.get(key) != current.get(key):
                raise RuntimeError(f"Cannot resume SDXL checkpoint with changed {key}: saved={previous.get(key)!r}, current={current.get(key)!r}; {path}")
