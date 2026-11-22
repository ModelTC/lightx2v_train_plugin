"""Video cache support for the host flow-matching training objective."""

from lightx2v_train.model_zoo.capability_adapters.common import (
    GenericFlowMatchingCapability,
    _training_cache_data,
    _uses_prompt_dropout,
)


class HunyuanVideo15SFTCapability(GenericFlowMatchingCapability):
    def encode_training_cache(self, batch):
        prompts = {"positive": batch["conditioning"]["prompt"]}
        if _uses_prompt_dropout(self.model):
            prompts["unconditional"] = self.model.unconditional_prompt
        return _training_cache_data(
            self.model,
            batch,
            inputs={"latents": self.model.encode_to_cache_latent(batch)},
            prompts=prompts,
            contextual_roles=prompts,
        )
