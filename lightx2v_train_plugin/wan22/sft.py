"""Wan2.2 SFT using the DiffSynth experiment's timestep and loss recipe."""

import math

import torch
import torch.nn.functional as F
from lightx2v_train.model_capabilities import LossResult
from lightx2v_train.model_zoo.capability_adapters.common import (
    _cached_condition,
    _cached_latent,
    _require_singleton_tensor,
)
from lightx2v_train.model_zoo.wan.capability_adapters.wan_flow_matching_capability import (
    WanFlowMatchingCapability,
)


class Wan22SFTCapability(WanFlowMatchingCapability):
    def __init__(self, model):
        super().__init__(model)
        scheduler = model.config.get("scheduler", {})
        if scheduler.get("num_train_timesteps", 1000) != 1000:
            raise ValueError("Wan2.2 SFT requires scheduler.num_train_timesteps=1000.")
        shift = float(scheduler.get("shift", 5.0))
        if not math.isfinite(shift) or shift <= 0:
            raise ValueError("scheduler.shift must be positive and finite.")

        expert = model.config["model"]["expert"]
        if expert not in {"high", "low"}:
            raise ValueError("model.expert must be 'high' or 'low'.")
        self.timestep_start, self.timestep_end = (0, 417) if expert == "high" else (417, 1000)

        # Keep the reference's CPU float32 table, including its endpoints.
        sigmas = torch.linspace(1.0, 0.0, 1001)[:-1]
        self.sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)
        self.timesteps = self.sigmas * 1000
        weights = torch.exp(-2 * ((self.timesteps - 500) / 1000) ** 2)
        weights = weights - weights.min()
        self.training_weights = weights * (1000 / weights.sum())

    def compute_loss(self, batch, context):
        model = self.model
        broadcast = context.broadcast
        with torch.no_grad():
            latent = _cached_latent(batch, model)
            if latent is None:
                latent = model.encode_to_latent(batch)
            latent = _require_singleton_tensor(
                broadcast(latent.to(device=model.device, dtype=torch.bfloat16)),
                "Wan2.2 SFT latent",
            )

            # DiffSynth samples an index before drawing fresh bf16 noise.
            index = torch.randint(self.timestep_start, self.timestep_end, (1,))
            timestep = broadcast(self.timesteps[index].to(device=model.device, dtype=torch.bfloat16))
            noise = broadcast(torch.randn_like(latent))

            # The reference quantizes t to bf16, then looks up sigma and weight
            # again. Keep that behavior, including the small boundary overlap.
            table_index = torch.argmin((self.timesteps - timestep.cpu()).abs())
            sigma = self.sigmas[table_index]
            weight = self.training_weights[table_index]
            noisy_latent = (1 - sigma) * latent + sigma * noise
            target = noise - latent
            condition = _cached_condition(batch, model)
            if condition is None:
                condition = model.encode_condition(batch)
            condition = broadcast(condition)

        prediction = model.predict_denoiser_output(
            noisy_latent,
            timestep.float() / 1000,
            condition,
        )
        mse = F.mse_loss(prediction.float(), target.float())
        loss = mse * weight.to(device=mse.device)
        return LossResult(loss=loss, metrics={"mse": mse.detach(), "sigma": sigma, "loss_weight": weight})
