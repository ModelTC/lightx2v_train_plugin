# Copyright 2025 The Hunyuan Team and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""HunyuanVideo 1.5 joint attention with a linear-size padding mask.

Adapted from ``HunyuanVideo15AttnProcessor2_0`` in Diffusers 0.38.0:
https://github.com/huggingface/diffusers/blob/v0.38.0/src/diffusers/models/transformers/transformer_hunyuan_video15.py

Mask keys during attention and zero invalid queries before output projection
to avoid the upstream [B, 1, N, N] validity mask.
"""

import torch
import torch.nn.functional as F
from diffusers.models.attention_dispatch import dispatch_attention_fn
from diffusers.models.attention_processor import Attention
from diffusers.models.embeddings import apply_rotary_emb
from diffusers.models.transformers.transformer_hunyuan_video15 import HunyuanVideo15AttnProcessor2_0


class HunyuanVideo15CompactAttnProcessor2_0(HunyuanVideo15AttnProcessor2_0):
    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        image_rotary_emb: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        query = query.unflatten(2, (attn.heads, -1))
        key = key.unflatten(2, (attn.heads, -1))
        value = value.unflatten(2, (attn.heads, -1))

        query = attn.norm_q(query)
        key = attn.norm_k(key)

        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb, sequence_dim=1)
            key = apply_rotary_emb(key, image_rotary_emb, sequence_dim=1)

        if encoder_hidden_states is not None:
            encoder_query = attn.add_q_proj(encoder_hidden_states)
            encoder_key = attn.add_k_proj(encoder_hidden_states)
            encoder_value = attn.add_v_proj(encoder_hidden_states)

            encoder_query = encoder_query.unflatten(2, (attn.heads, -1))
            encoder_key = encoder_key.unflatten(2, (attn.heads, -1))
            encoder_value = encoder_value.unflatten(2, (attn.heads, -1))

            if attn.norm_added_q is not None:
                encoder_query = attn.norm_added_q(encoder_query)
            if attn.norm_added_k is not None:
                encoder_key = attn.norm_added_k(encoder_key)

            query = torch.cat([query, encoder_query], dim=1)
            key = torch.cat([key, encoder_key], dim=1)
            value = torch.cat([value, encoder_value], dim=1)

        batch_size, seq_len = query.shape[:2]
        # Only encoder tokens can be padding; prepended latent tokens are valid.
        valid_tokens = F.pad(attention_mask, (seq_len - attention_mask.shape[1], 0), value=True).bool()
        key_mask = valid_tokens.view(batch_size, 1, 1, seq_len)

        hidden_states = dispatch_attention_fn(
            query,
            key,
            value,
            attn_mask=key_mask,
            dropout_p=0.0,
            is_causal=False,
            backend=self._attention_backend,
            parallel_config=self._parallel_config,
        )
        # Zero invalid queries before projection so padded tokens retain its bias.
        hidden_states = hidden_states.masked_fill(~valid_tokens[:, :, None, None], 0)
        hidden_states = hidden_states.flatten(2, 3).to(query.dtype)

        if encoder_hidden_states is not None:
            encoder_tokens = encoder_hidden_states.shape[1]
            hidden_states, encoder_hidden_states = hidden_states[:, :-encoder_tokens], hidden_states[:, -encoder_tokens:]

            if getattr(attn, "to_out", None) is not None:
                hidden_states = attn.to_out[0](hidden_states)
                hidden_states = attn.to_out[1](hidden_states)

            if getattr(attn, "to_add_out", None) is not None:
                encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

        return hidden_states, encoder_hidden_states
