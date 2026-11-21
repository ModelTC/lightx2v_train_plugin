import math

import torch
from lightx2v_train.infer.video import WanT2VDualInferencer
from lightx2v_train.utils.registry import INFERENCER_REGISTER


@INFERENCER_REGISTER("wan22_t2v_infer")
class Wan22T2VInferencer(WanT2VDualInferencer):
    def __init__(self, config):
        super().__init__(config)
        self.negative_prompt = self.infer_config.get("negative_prompt", self.negative_prompt)

    def _inference_sigmas(self, num_inference_steps):
        shift = float(self.infer_config.get("sigma_shift", 5.0))
        if not math.isfinite(shift) or shift <= 0:
            raise ValueError("inference.sigma_shift must be positive and finite.")
        sigmas = torch.linspace(1, 0, num_inference_steps + 1)[:-1]
        return (shift * sigmas / (1 + (shift - 1) * sigmas)).tolist()
