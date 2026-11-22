import random
from pathlib import Path

import numpy as np
import torch
import yaml
from lightx2v_train.runtime.distributed import barrier, is_main_process
from lightx2v_train.trainers.flow_matching import FlowMatchingTrainer
from lightx2v_train.utils.registry import TRAINER_REGISTER


@TRAINER_REGISTER("wan22_sft")
class Wan22SFTTrainer(FlowMatchingTrainer):
    trainer_name = "wan22_sft"

    def __init__(self, config):
        seed = int(config.get("runtime", {}).get("seed", 42))
        random.seed(seed)
        np.random.seed(seed)
        super().__init__(config)

    def _build_lr_scheduler(self, optimizer, num_training_steps=None, num_warmup_steps=None):
        warmup_steps = self.lr_warmup_iters if num_warmup_steps is None else num_warmup_steps
        warmup_steps = max(1, warmup_steps)
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: min(1.0, (step + 1) / warmup_steps))

    def save_checkpoint(self, iteration, save_total_limit):
        super().save_checkpoint(iteration, save_total_limit)
        if is_main_process():
            path = Path(self.output_train_dir) / f"checkpoint-{iteration:09d}" / "config.yaml"
            temporary = path.with_suffix(".yaml.tmp")
            temporary.write_text(yaml.safe_dump(self.config, sort_keys=False), encoding="utf-8")
            temporary.replace(path)
        barrier()

    def _validate_checkpoint_metadata(self, state, state_path, resume_ckpt_path):
        super()._validate_checkpoint_metadata(state, state_path, resume_ckpt_path)
        path = Path(resume_ckpt_path) / "config.yaml"
        if not path.is_file():
            raise RuntimeError(f"Wan2.2 resume requires the saved config: {path}")
        saved = yaml.safe_load(path.read_text(encoding="utf-8"))
        previous = saved.get("model", {}).get("expert")
        current = self.model_config["expert"]
        if previous != current:
            raise RuntimeError(f"Cannot resume {previous!r} expert checkpoint as {current!r}: {path}")
