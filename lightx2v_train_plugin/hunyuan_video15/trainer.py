"""Flow-matching trainer with HunyuanVideo base-checkpoint resume checks."""

from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path

import yaml
from lightx2v_train.runtime.distributed import barrier, is_main_process
from lightx2v_train.trainers.flow_matching import FlowMatchingTrainer
from lightx2v_train.utils.registry import TRAINER_REGISTER


@TRAINER_REGISTER("hunyuan_video15_sft")
class HunyuanVideo15SFTTrainer(FlowMatchingTrainer):
    trainer_name = "hunyuan_video15_sft"

    @staticmethod
    def _model_identity(model_config):
        root = model_config.get("pretrained_model_name_or_path")
        if not isinstance(root, str) or not root:
            raise RuntimeError("HunyuanVideo 1.5 resume requires model.pretrained_model_name_or_path.")
        return {
            "transformer_version": model_config.get("transformer_version", "480p_t2v"),
            "pretrained_model_name_or_path": root,
            "transformer_path": model_config.get("transformer_path", root),
        }

    def save_checkpoint(self, iteration, save_total_limit):
        super().save_checkpoint(iteration, save_total_limit)
        if is_main_process():
            path = Path(self.output_train_dir) / f"checkpoint-{iteration:09d}" / "config.yaml"
            saved = deepcopy(self.config)
            # LoRA cannot verify moved/replaced base weights; retain exact base paths.
            saved["model"].update(self._model_identity(self.model_config))
            temporary = path.with_suffix(".yaml.tmp")
            temporary.write_text(yaml.safe_dump(saved, sort_keys=False), encoding="utf-8")
            temporary.replace(path)
        barrier()

    def _validate_checkpoint_metadata(self, state, state_path, resume_ckpt_path):
        super()._validate_checkpoint_metadata(state, state_path, resume_ckpt_path)
        path = Path(resume_ckpt_path) / "config.yaml"
        if not path.is_file():
            raise RuntimeError(f"HunyuanVideo 1.5 resume requires the saved config: {path}")
        saved = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(saved, Mapping) or not isinstance(saved.get("model"), Mapping):
            raise RuntimeError(f"HunyuanVideo 1.5 resume requires a saved model config: {path}")
        previous = self._model_identity(saved["model"])
        current = self._model_identity(self.model_config)
        for key, value in current.items():
            if previous[key] != value:
                raise RuntimeError(f"Cannot resume HunyuanVideo 1.5 checkpoint with model.{key}={previous[key]!r} as {value!r}; resume requires the same model variant and exact base paths: {path}")
