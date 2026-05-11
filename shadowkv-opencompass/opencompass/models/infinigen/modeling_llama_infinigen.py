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
""" PyTorch LLaMA model."""
import math
from typing import Optional, Tuple

import torch
import torch.utils.checkpoint
from torch import nn

from transformers.utils import logging
from transformers.activations import ACT2FN
from transformers.models.llama.configuration_llama import LlamaConfig


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


class LlamaRotaryEmbedding(torch.nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float().to(device) / dim))
        self.register_buffer("inv_freq", inv_freq)

        # Build here to make `torch.jit.trace` work.
        self.max_seq_len_cached = max_position_embeddings
        t = torch.arange(self.max_seq_len_cached, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        # This `if` block is unlikely to be run after we build sin/cos in `__init__`. Keep the logic here just in case.
        if seq_len > self.max_seq_len_cached:
            self.max_seq_len_cached = seq_len
            t = torch.arange(self.max_seq_len_cached, device=x.device, dtype=self.inv_freq.dtype)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            # Different from paper, but it uses a different permutation in order to obtain the same calculation
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
            self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
            self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)
        return (
            self.cos_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
            self.sin_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
        )


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    gather_indices = position_ids[:, None, :, None]  # [bs, 1, seq_len, 1]
    gather_indices = gather_indices.expand(-1, cos.shape[1], -1, cos.shape[3])
    cos = torch.gather(cos.expand(gather_indices.shape[0], -1, -1, -1), 2, gather_indices)
    sin = torch.gather(sin.expand(gather_indices.shape[0], -1, -1, -1), 2, gather_indices)
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
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
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
        self.rope_theta = getattr(config, 'rope_theta', 10000.0)

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        self.rotary_emb = LlamaRotaryEmbedding(self.head_dim, max_position_embeddings=self.max_position_embeddings,
                                               base=self.rope_theta)
        self.attn = None

        self.layer_idx = layer_idx
        
        #### InfiniGen Hyperparams ####
        self.cache_ratio = None
        self.partial_weight_ratio = None
        self.previous_hidden_states = None
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
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def kv_cache_mask(self, attn):
        # Hyperparameters
        # budget: maximum kv cache percentage to prefetch per layer
        # capacity: maximum kv cache percentage to store in cpu
        assert self.budget < self.capacity

        b, h, tgt_len, src_len = attn.shape
        attn = attn.view(b*h, tgt_len, src_len)
        heads = b * h

        attn_mask = torch.full(attn.shape, -10000, dtype=attn.dtype, device=attn.device)
        attn_mask = torch.triu(attn_mask, diagonal = 1)
        fetch_mask = torch.zeros_like(attn)
        m_inf = torch.tensor([[-10000]], dtype=attn.dtype, device=attn.device)
        attn = attn + attn_mask
        del attn_mask
        
        max = torch.max(attn, dim = -1, keepdim = True)[0][0]
        threshold = max - self.alpha
        fetch_num  = (attn >= threshold).sum(dim = -1) #heads, tgt_len
        del threshold

        fetch_num = torch.mean(fetch_num.to(attn.dtype), dim = 0).to(torch.int32) # need to fetch same amount for each head
        fetch_max = int(src_len * self.budget)
        fetch_num = torch.where(fetch_num >= fetch_max, fetch_max, fetch_num) # tgt_len

        store_max = int(src_len * self.capacity)

        fetch_mask[:, :fetch_max] = torch.tril(torch.ones((fetch_max, src_len), dtype = attn.dtype, device = attn.device)).unsqueeze(0)

        for i in range(fetch_max, store_max):
            _, ind = torch.topk(attn[:,i, :i+1], k = fetch_num[i], dim = -1)
            fetch_mask[:, i, :i+1] = fetch_mask[:, i, :i + 1].scatter(-1, ind, 1)

        for i in range(store_max, tgt_len):
            _, ind = torch.topk(attn[:,i, :i+1], k = fetch_num[i], dim = -1)
            fetch_mask[:, i, :i + 1] = fetch_mask[:, i, :i + 1].scatter(-1, ind, 1)

            if i == (tgt_len - 1):
                continue

            # Before adding KV cache, evict one
            if self.eviction_policy == "fifo":
                evict_idx = i - store_max
                attn[:, (i + 1):, evict_idx] = -10000

            elif self.eviction_policy == "lru":
                idx = torch.arange(i + 1, device = attn.device).unsqueeze(0).unsqueeze(-1)
                idx = idx * fetch_mask[:, :i + 1, :int(i / 2)] # avoid evicting recently added ones
                # Most recently fetched idx per each KV cache
                _, idx = torch.max(idx, dim = 1, keepdim = True) # heads, 1, i/2
                _, ind = torch.min(idx, dim = -1, keepdim = True) # heads, 1, 1
                ind = ind.repeat(1, tgt_len - (i + 1), 1)
                attn[:, (i + 1):] = attn[:, (i + 1):].scatter(-1, ind, -10000)

            elif self.eviction_policy == "counter":
                counter = torch.sum(fetch_mask[:, :i + 1, :int(i / 2)], dim = 1, keepdim = True) #heads, 1, i/2
                _, ind = torch.min(counter, dim = -1, keepdim = True) #heads, 1, 1
                ind = ind.repeat(1,tgt_len-(i+1),1)
                attn[:, (i + 1):] = attn[:, (i + 1):].scatter(-1, ind, -10000)

            else:
                raise NotImplementedError

        density = fetch_mask.float().sum().item() / heads / (tgt_len * (tgt_len + 1) / 2)
        fetch_mask = torch.where(fetch_mask == 1, 0, m_inf)
        fetch_mask = fetch_mask.view(b, h, tgt_len, src_len)
        return fetch_mask, density
    
    def forward(
        self,
        hidden_states: torch.Tensor,
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
        query_states = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)

        key_states = self.k_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        layer_idx = kwargs.get('layer_idx', 0)
        is_dynamic_cache = past_key_value is not None and hasattr(past_key_value, 'update')

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            if is_dynamic_cache:
                kv_seq_len += past_key_value.get_seq_length(layer_idx)
            else:
                kv_seq_len += past_key_value[0].shape[-2]
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
        # [bsz, nh, t, hd]

        if past_key_value is not None:
            if is_dynamic_cache:
                key_states, value_states = past_key_value.update(key_states, value_states, layer_idx)
            else:
                key_states = torch.cat([past_key_value[0], key_states], dim=2)
                value_states = torch.cat([past_key_value[1], value_states], dim=2)

        past_key_value = past_key_value if (use_cache and is_dynamic_cache) else \
                         ((key_states, value_states) if use_cache else None)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        ### Speculate attention ###
        if (self.previous_hidden_states is not None) and (self.partial_weight_q is not None):
            query = (torch.matmul(self.previous_hidden_states, self.q_proj.weight.data.transpose(-1,-2))).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
            query, _ = apply_rotary_pos_emb(query, key_states, cos, sin, position_ids)
            query = query @ self.skewing_matrix.unsqueeze(0)
            mask = self.partial_weight_q[0].view(-1,128).unsqueeze(0).unsqueeze(2).expand(1,-1,query_states.shape[2], -1)
            query = query * mask.to(query.dtype)

            attn = torch.matmul(query, (key_states @ self.skewing_matrix).transpose(2, 3))/math.sqrt(self.head_dim)

            attn_mask, density = self.kv_cache_mask(attn)
            self.density = density
        ###########################

        ### Build combined additive mask ###
        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )

        infinigen_active = (self.previous_hidden_states is not None) and (self.partial_weight_q is not None)
        if infinigen_active and attention_mask is not None:
            combined_mask = attn_mask + attention_mask
        elif infinigen_active:
            combined_mask = attn_mask
        else:
            combined_mask = attention_mask  # may be None

        if output_attentions:
            # Materialize full attention matrix only when caller needs the weights
            attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
            if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention weights should be of size {(bsz * self.num_heads, q_len, kv_seq_len)}, but is"
                    f" {attn_weights.size()}"
                )
            if combined_mask is not None:
                attn_weights = attn_weights + combined_mask
                attn_weights = torch.max(attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min))
            self.attn = attn_weights
            # upcast attention to fp32
            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
            attn_output = torch.matmul(attn_weights, value_states)
        else:
            # Memory-efficient path: never materializes the full [bsz, heads, q, kv] matrix
            self.attn = None
            attn_output = nn.functional.scaled_dot_product_attention(
                query_states, key_states, value_states,
                attn_mask=combined_mask,
                dropout_p=0.0,
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
        new_attn = LlamaAttention(config, layer_idx).to(dtype=old_attn.q_proj.weight.dtype,
                                             device=old_attn.q_proj.weight.device)
        for proj in ('q_proj', 'k_proj', 'v_proj', 'o_proj'):
            getattr(new_attn, proj).weight.data.copy_(getattr(old_attn, proj).weight.data)
            if getattr(old_attn, proj).bias is not None:
                getattr(new_attn, proj).bias.data.copy_(getattr(old_attn, proj).bias.data)
        layer.self_attn = new_attn
