# coding=utf-8
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
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
"""PyTorch LLaMA model."""

from typing import Optional, Tuple
import torch
import math
from torch import nn
from transformers.utils import logging
from transformers.models.llama.modeling_llama import LlamaConfig
from transformers.activations import ACT2FN

logger = logging.get_logger(__name__)


class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)

        # convert into half-precision if necessary
        if self.weight.dtype in [torch.float16, torch.bfloat16]:
            hidden_states = hidden_states.to(self.weight.dtype)

        return self.weight * hidden_states



def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class LlamaMLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
    ):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.act_fn = ACT2FN[hidden_act]

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class LlamaAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = getattr(config, "rope_theta", 10000.0)

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        self.q_proj = nn.Linear(
            self.hidden_size, self.num_heads * self.head_dim, bias=False
        )
        self.k_proj = nn.Linear(
            self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim, self.hidden_size, bias=False
        )
        self.attn = None

        self.layer_idx = layer_idx

        #### InfiniGen Hyperparams ####
        self.cache_ratio = None
        self.partial_weight_ratio = None
        self.previous_attn = None

        self.current_hidden_states = None
        self.partial_weight_q = None
        self.skewing_matrx = None
        self.alpha = 5
        self.capacity = 1.0
        self.budget = 0.2
        self.eviction_policy = "counter"
        self.density = None
        ###############################

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return (
            tensor.view(bsz, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
            .contiguous()
        )

    def _get_previous_hidden_states(self):
        if self.previous_attn is None:
            return None
        return self.previous_attn.current_hidden_states

    def kv_cache_mask(self, attn):
        # Hyperparameters
        # budget: maximum kv cache percentage to prefetch per layer
        # capacity: maximum kv cache percentage to store in cpu
        assert self.budget < self.capacity

        b, h, tgt_len, src_len = attn.shape
        m_inf = -10000.
        if tgt_len == 1:
            fetch_max = int(src_len * self.budget)
            _, fetch_idxs = torch.topk(attn, dim=-1, k=fetch_max)
            fetch_mask = torch.full_like(attn, m_inf).scatter_(-1, fetch_idxs, 0.)
        else:
            fetch_mask = torch.zeros_like(attn)
        return fetch_mask, None


    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()
        self.current_hidden_states = hidden_states

        ### Ours  ###########################################
        query_states = (
            self.q_proj(hidden_states)
            .view(bsz, q_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )

        key_states = (
            self.k_proj(hidden_states)
            .view(bsz, q_len, self.num_key_value_heads, self.head_dim)
            .transpose(1, 2)
        )
        value_states = (
            self.v_proj(hidden_states)
            .view(bsz, q_len, self.num_key_value_heads, self.head_dim)
            .transpose(1, 2)
        )

        layer_idx = self.layer_idx
        is_dynamic_cache = past_key_value is not None and hasattr(
            past_key_value, "update"
        )

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            if is_dynamic_cache:
                kv_seq_len += past_key_value.get_seq_length(layer_idx)
            else:
                kv_seq_len += past_key_value[0].shape[-2]
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin,
        )
        # [bsz, nh, t, hd]

        if past_key_value is not None:
            if is_dynamic_cache:
                key_states, value_states = past_key_value.update(
                    key_states, value_states, layer_idx
                )
            else:
                key_states = torch.cat([past_key_value[0], key_states], dim=2)
                value_states = torch.cat([past_key_value[1], value_states], dim=2)

        past_key_value = (
            past_key_value
            if (use_cache and is_dynamic_cache)
            else ((key_states, value_states) if use_cache else None)
        )

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        infinigen_active = (self._get_previous_hidden_states() is not None and 
            self.partial_weight_q is not None and hidden_states.size(1) == 1)
        ### Speculate attention after prefill###
        if infinigen_active:
            query = (
                (
                    torch.matmul(
                        self._get_previous_hidden_states(),
                        self.q_proj.weight.data.transpose(-1, -2),
                    )
                )
                .view(bsz, q_len, self.num_heads, self.head_dim)
                .transpose(1, 2)
            )
            query, _ = apply_rotary_pos_emb(query, key_states, cos, sin, position_ids)
            query = query @ self.skewing_matrix.unsqueeze(0)
            mask = (
                self.partial_weight_q[0]
                .view(-1, 128)
                .unsqueeze(0)
                .unsqueeze(2)
                .expand(1, -1, query_states.shape[2], -1)
            )
            query = query * mask.to(query.dtype)

            attn = torch.matmul(
                query, (key_states @ self.skewing_matrix).transpose(2, 3)
            ) / math.sqrt(self.head_dim)

            attn_mask, density = self.kv_cache_mask(attn)
            self.density = density
        ###########################

        ### Build combined additive mask ###
        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )

        if infinigen_active and attention_mask is not None:
            combined_mask = attn_mask + attention_mask
        elif infinigen_active:
            combined_mask = attn_mask
        else:
            combined_mask = attention_mask  # may be None

        if output_attentions:
            # Materialize full attention matrix only when caller needs the weights
            attn_weights = torch.matmul(
                query_states, key_states.transpose(2, 3)
            ) / math.sqrt(self.head_dim)
            if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention weights should be of size {(bsz * self.num_heads, q_len, kv_seq_len)}, but is"
                    f" {attn_weights.size()}"
                )
            if combined_mask is not None:
                attn_weights = attn_weights + combined_mask
                attn_weights = torch.max(
                    attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min)
                )
            self.attn = attn_weights
            # upcast attention to fp32
            attn_weights = nn.functional.softmax(
                attn_weights, dim=-1, dtype=torch.float32
            ).to(query_states.dtype)
            attn_output = torch.matmul(attn_weights, value_states)
        else:
            # Memory-efficient path: never materializes the full [bsz, heads, q, kv] matrix
            self.attn = None
            if combined_mask is None:
                attn_kwargs = dict(is_causal=(query_states.size(2) > 1))  # prefill
                logger.warning("Please use attn_implementation=eager when loading the model. Now trying to infer attention mask")
            else:
                attn_kwargs = dict(attn_mask=combined_mask)
            attn_output = nn.functional.scaled_dot_product_attention(
                query_states,
                key_states,
                value_states,
                dropout_p=0.0,
                **attn_kwargs
            )
            attn_weights = None

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        return attn_output, attn_weights


def replace_attention_layers(model, config):
    for layer_idx, layer in enumerate(model.model.layers):
        old_attn = layer.self_attn
        new_attn = LlamaAttention(config, layer_idx).to(
            dtype=old_attn.q_proj.weight.dtype, device=old_attn.q_proj.weight.device
        )
        for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
            getattr(new_attn, proj).weight.data.copy_(
                getattr(old_attn, proj).weight.data
            )
            if getattr(old_attn, proj).bias is not None:
                getattr(new_attn, proj).bias.data.copy_(
                    getattr(old_attn, proj).bias.data
                )
        layer.self_attn = new_attn


def patch_llama_model(
    model,
    skewing_matrix_path: Optional[str] = None,
    partial_weight_path: Optional[str] = None,
    # InfiniGen parameters
    partial_weight_ratio=0.2,
    capacity=1.0,
    alpha=99,
    budget=0.2,
    **other_kwargs,
):
    replace_attention_layers(model, model.config)
    if skewing_matrix_path is not None:
        A = torch.load(skewing_matrix_path).to(next(model.parameters()).dtype)
    if partial_weight_path is not None:
        for layer in range(len(model.model.layers)):
            model.model.layers[
                layer
            ].self_attn.partial_weight_ratio = partial_weight_ratio
            model.model.layers[layer].self_attn.partial_weight_q = torch.load(
                partial_weight_path + "/partial_weight_q_" + str(layer) + ".pt"
            )
            model.model.layers[layer].self_attn.alpha = alpha
            model.model.layers[layer].self_attn.capacity = capacity
            model.model.layers[layer].self_attn.budget = budget
            if skewing_matrix_path is not None:
                model.model.layers[layer].self_attn.skewing_matrix = A[layer]

            if layer > 1:  # bug consistent with the original implementation
                model.model.layers[layer].self_attn.previous_attn = model.model.layers[layer - 1].self_attn
            else:
                model.model.layers[layer].self_attn.previous_attn = None

    return model
