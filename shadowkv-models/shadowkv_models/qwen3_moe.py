################################################################################
#
# Copyright 2024 ByteDance Ltd. and/or its affiliates. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
################################################################################


import torch
import torch.nn.functional as F
import gc
import time
import typing as tp

import transformers
from transformers import Qwen3MoeForCausalLM, Qwen3MoeConfig, AutoTokenizer
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeDecoderLayer
from transformers.activations import ACT2FN

transformers.logging.set_verbosity_error()

from .tensor_op import layer_norm, apply_rotary_pos_emb, apply_rotary_pos_emb_single, sample_token
from .prompt_template import Templates, Chat_Templates
from .base import LLM


class Qwen3MoeLayer:
    def __init__(self, layer_idx) -> None:
        self.wq: torch.Tensor = None
        self.wk: torch.Tensor = None
        self.wv: torch.Tensor = None
        self.wo: torch.Tensor = None

        self.q_norm_weight: torch.Tensor = None
        self.q_norm_variance_epsilon: float = 0.0
        self.k_norm_weight: torch.Tensor = None
        self.k_norm_variance_epsilon: float = 0.0

        self.bq: torch.Tensor = None
        self.bk: torch.Tensor = None
        self.bv: torch.Tensor = None

        self.is_moe: bool = False
        self.act_fn = None

        self.gate_proj: torch.Tensor = None
        self.up_proj: torch.Tensor = None
        self.down_proj: torch.Tensor = None

        self.moe_gate: torch.Tensor = None
        self.expert_gate_proj: torch.Tensor = None
        self.expert_up_proj: torch.Tensor = None
        self.expert_down_proj: torch.Tensor = None
        self.expert_gate_proj_t: torch.Tensor = None
        self.expert_up_proj_t: torch.Tensor = None
        self.expert_down_proj_t: torch.Tensor = None
        self.num_experts: int = 0
        self.top_k: int = 0
        self.norm_topk_prob: bool = False

        self.input_layernorm_weight: torch.Tensor = None
        self.input_layernorm_variance_epsilon: float = 0.0

        self.post_attention_layernorm_weight: torch.Tensor = None
        self.post_attention_layernorm_variance_epsilon: float = 0.0

        self.layer_idx = layer_idx

    def init_parameters(self, hf_layer: Qwen3MoeDecoderLayer):
        self.wq = hf_layer.self_attn.q_proj.weight.detach()
        self.wk = hf_layer.self_attn.k_proj.weight.detach()
        self.wv = hf_layer.self_attn.v_proj.weight.detach()
        self.wo = hf_layer.self_attn.o_proj.weight.detach()

        self.q_norm_weight = hf_layer.self_attn.q_norm.weight
        self.q_norm_variance_epsilon = hf_layer.self_attn.q_norm.variance_epsilon
        self.k_norm_weight = hf_layer.self_attn.k_norm.weight
        self.k_norm_variance_epsilon = hf_layer.self_attn.k_norm.variance_epsilon

        self.attention_bias = hf_layer.self_attn.config.attention_bias
        if self.attention_bias:
            self.bq = hf_layer.self_attn.q_proj.bias.detach()
            self.bk = hf_layer.self_attn.k_proj.bias.detach()
            self.bv = hf_layer.self_attn.v_proj.bias.detach()
        else:
            self.bq = None
            self.bk = None
            self.bv = None

        self.act_fn = ACT2FN[hf_layer.self_attn.config.hidden_act]
        self.is_moe = hasattr(hf_layer.mlp, "experts")

        if self.is_moe:
            self.moe_gate = hf_layer.mlp.gate.weight.detach()
            if hasattr(hf_layer.mlp.experts, "gate_up_proj"):
                self.num_experts = hf_layer.mlp.experts.num_experts
                self.top_k = hf_layer.mlp.gate.top_k
                self.norm_topk_prob = hf_layer.mlp.gate.norm_topk_prob
                expert_gate_proj, expert_up_proj = torch.chunk(
                    hf_layer.mlp.experts.gate_up_proj.detach(), chunks=2, dim=1
                )
                expert_down_proj = hf_layer.mlp.experts.down_proj.detach()

                self.expert_gate_proj_t = expert_gate_proj.transpose(1, 2).contiguous()
                self.expert_up_proj_t = expert_up_proj.transpose(1, 2).contiguous()
                self.expert_down_proj_t = expert_down_proj.transpose(1, 2).contiguous()
                self.expert_gate_proj = None
                self.expert_up_proj = None
                self.expert_down_proj = None
            else:
                self.num_experts = hf_layer.mlp.num_experts
                self.top_k = hf_layer.mlp.top_k
                self.norm_topk_prob = hf_layer.mlp.norm_topk_prob
                expert_gate_proj = [expert.gate_proj.weight.detach() for expert in hf_layer.mlp.experts]
                expert_up_proj = [expert.up_proj.weight.detach() for expert in hf_layer.mlp.experts]
                expert_down_proj = [expert.down_proj.weight.detach() for expert in hf_layer.mlp.experts]

                self.expert_gate_proj = expert_gate_proj
                self.expert_up_proj = expert_up_proj
                self.expert_down_proj = expert_down_proj
                self.expert_gate_proj_t = None
                self.expert_up_proj_t = None
                self.expert_down_proj_t = None
        else:
            self.gate_proj = hf_layer.mlp.gate_proj.weight.detach()
            self.up_proj = hf_layer.mlp.up_proj.weight.detach()
            self.down_proj = hf_layer.mlp.down_proj.weight.detach()

        self.input_layernorm_weight = hf_layer.input_layernorm.weight
        self.input_layernorm_variance_epsilon = hf_layer.input_layernorm.variance_epsilon

        self.post_attention_layernorm_weight = hf_layer.post_attention_layernorm.weight
        self.post_attention_layernorm_variance_epsilon = hf_layer.post_attention_layernorm.variance_epsilon
    
    def init_gpu(self, device:str = 'cuda:0'):
        self.input_layernorm_weight = self.input_layernorm_weight.to(device, non_blocking=True)
        self.post_attention_layernorm_weight = self.post_attention_layernorm_weight.to(device, non_blocking=True)
        self.wq = self.wq.to(device, non_blocking=True)
        self.wk = self.wk.to(device, non_blocking=True)
        self.wv = self.wv.to(device, non_blocking=True)
        self.wo = self.wo.to(device, non_blocking=True)

        self.q_norm_weight = self.q_norm_weight.to(device, non_blocking=True)
        self.k_norm_weight = self.k_norm_weight.to(device, non_blocking=True)

        if self.attention_bias:
            self.bq = self.bq.to(device, non_blocking=True)
            self.bk = self.bk.to(device, non_blocking=True)
            self.bv = self.bv.to(device, non_blocking=True)

        if self.is_moe:
            self.moe_gate = self.moe_gate.to(device, non_blocking=True)
            if self.expert_gate_proj is not None:
                self.expert_gate_proj = [expert_gate_proj.to(device, non_blocking=True) for expert_gate_proj in self.expert_gate_proj]
                self.expert_up_proj = [expert_up_proj.to(device, non_blocking=True) for expert_up_proj in self.expert_up_proj]
                self.expert_down_proj = [expert_down_proj.to(device, non_blocking=True) for expert_down_proj in self.expert_down_proj]
            if self.expert_gate_proj_t is not None:
                self.expert_gate_proj_t = self.expert_gate_proj_t.to(device, non_blocking=True)
                self.expert_up_proj_t = self.expert_up_proj_t.to(device, non_blocking=True)
                self.expert_down_proj_t = self.expert_down_proj_t.to(device, non_blocking=True)
        else:
            self.gate_proj = self.gate_proj.to(device, non_blocking=True)
            self.up_proj = self.up_proj.to(device, non_blocking=True)
            self.down_proj = self.down_proj.to(device, non_blocking=True)


class Qwen3Moe(LLM):
    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-30B-A3B-Instruct-2507",
        batch_size: int = 1,
        max_length: int = 64 * 1024,
        device: str = "cuda:0",
        dtype=torch.bfloat16,
        attn_mode: str = "full",
        sparse_budget: tp.Union[int, float] = 2048,
        local_budget: tp.Union[int, float] = 4 * 8,
        outlier_budget: tp.Union[int, float] = 48 * 8,
        rank=160,
        chunk_size=8,
        minference=False,
        keys_on_device: bool = True,
        use_nvfp4_quantization: bool = False,
        nvfp4_channel_size: int = 1024,
        use_higgs_quantization: bool = False,
        higgs_hadamard_groupsize: int = 128,
        higgs_edenn_d: int = 16,
        higgs_edenn_n: int = 256,
        higgs_channel_size: int = 1024,
        higgs_chunk_size: int = 64,
        use_residual_quantization: bool = False,
        meta_chunk_size: int = 1,
        residual_higgs_edenn_d: int = 16,
        residual_higgs_edenn_n: int = 256,
        keys_fp8: bool = False,
        keys_nvfp4: bool = False,
        keys_higgs: bool = False,
        values_fp8: bool = False,
        values_nvfp4: bool = False,
        values_higgs: bool = False,
        keys_higgs_hadamard_groupsize: int = 128,
        keys_higgs_edenn_d: int = 16,
        keys_higgs_edenn_n: int = 256,
        keys_higgs_channel_size: int = 1024,
        keys_higgs_chunk_size: int = 64,
        values_higgs_hadamard_groupsize: int = 128,
        values_higgs_edenn_d: int = 16,
        values_higgs_edenn_n: int = 256,
        values_higgs_channel_size: int = 1024,
        values_higgs_chunk_size: int = 64,
    ) -> None:

        assert batch_size == 1, "Batch size must be 1"
        self.batch_size = batch_size
        self.device = device
        self.dtype = dtype
        self.config = Qwen3MoeConfig.from_pretrained(model_name)
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, legacy=False)
        self.max_length = max_length
        self.hidden_size = self.config.hidden_size
        self.num_heads = self.config.num_attention_heads
        self.head_dim = self.config.head_dim
        self.num_key_value_heads = self.config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = self.config.max_position_embeddings
        rope_scaling = getattr(self.config, "rope_scaling", None)
        if rope_scaling is not None and "rope_theta" in rope_scaling:
            self.rope_theta = rope_scaling["rope_theta"]
        else:
            self.rope_theta = self.config.rope_theta

        self.init_parameters()
        self.attn_mode = attn_mode
        self.minference = minference

        self.ctx_template = Templates["qwen"]
        self.chat_template = Chat_Templates["qwen"]

        self._init_shadowkv_params(
            chunk_size=chunk_size, rank=rank,
            sparse_budget=sparse_budget, local_budget=local_budget, outlier_budget=outlier_budget, keys_on_device=keys_on_device,
            use_nvfp4_quantization=use_nvfp4_quantization, nvfp4_channel_size=nvfp4_channel_size,
            use_higgs_quantization=use_higgs_quantization, higgs_hadamard_groupsize=higgs_hadamard_groupsize,
            higgs_edenn_d=higgs_edenn_d, higgs_edenn_n=higgs_edenn_n,
            higgs_channel_size=higgs_channel_size, higgs_chunk_size=higgs_chunk_size,
            use_residual_quantization=use_residual_quantization, meta_chunk_size=meta_chunk_size,
            residual_higgs_edenn_d=residual_higgs_edenn_d, residual_higgs_edenn_n=residual_higgs_edenn_n,
            keys_fp8=keys_fp8, keys_nvfp4=keys_nvfp4, keys_higgs=keys_higgs,
            values_fp8=values_fp8, values_nvfp4=values_nvfp4, values_higgs=values_higgs,
            keys_higgs_hadamard_groupsize=keys_higgs_hadamard_groupsize,
            keys_higgs_edenn_d=keys_higgs_edenn_d, keys_higgs_edenn_n=keys_higgs_edenn_n,
            keys_higgs_channel_size=keys_higgs_channel_size, keys_higgs_chunk_size=keys_higgs_chunk_size,
            values_higgs_hadamard_groupsize=values_higgs_hadamard_groupsize, 
            values_higgs_edenn_d=values_higgs_edenn_d, values_higgs_edenn_n=values_higgs_edenn_n, 
            values_higgs_channel_size=values_higgs_channel_size, values_higgs_chunk_size=values_higgs_chunk_size
        )

    def _set_cos_sin_cache(self, inv_freq: torch.Tensor):
        t = torch.arange(self.max_length, device=self.device, dtype=torch.int64).type_as(inv_freq)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().to(self.dtype), emb.sin().to(self.dtype)

    def init_parameters(self):
        hf_model = Qwen3MoeForCausalLM.from_pretrained(self.model_name, torch_dtype=self.dtype)
        self.embed_tokens = hf_model.model.embed_tokens.weight.detach().to(self.device)
        self.lm_head = hf_model.lm_head.weight.detach().to(self.device)
        self.norm_weight = hf_model.model.norm.weight.detach().to(self.device)
        self.norm_variance_epsilon = hf_model.model.norm.variance_epsilon
        self.cos_cache, self.sin_cache = self._set_cos_sin_cache(hf_model.model.rotary_emb.inv_freq.to(self.device))
        self.cos_sin_cache = torch.cat((self.cos_cache[:, :64], self.sin_cache[:, :64]), dim=-1)
        self.layers: list[Qwen3MoeLayer] = []

        for idx, hf_layer in enumerate(hf_model.model.layers):
            layer = Qwen3MoeLayer(idx)
            layer.init_parameters(hf_layer=hf_layer)
            layer.init_gpu(self.device)
            self.layers.append(layer)
            hf_model.model.layers[idx] = None
            gc.collect()

        self.num_layers = len(self.layers)

    def pre_attention_compute(
        self,
        hidden_states: torch.Tensor,
        buffer: Qwen3MoeLayer,
        num_heads: int,
        num_key_value_heads: int,
        head_dim: int,
    ):
        hidden_states = layer_norm(hidden_states, buffer.input_layernorm_variance_epsilon, buffer.input_layernorm_weight)
        bsz, q_len, _ = hidden_states.size()
        query_states = F.linear(hidden_states, buffer.wq, bias=buffer.bq)
        key_states = F.linear(hidden_states, buffer.wk, bias=buffer.bk)
        value_states = F.linear(hidden_states, buffer.wv, bias=buffer.bv)
        query_states = layer_norm(
            query_states.view(bsz, q_len, num_heads, head_dim),
            buffer.q_norm_variance_epsilon,
            buffer.q_norm_weight,
        ).transpose(1, 2)
        key_states = layer_norm(
            key_states.view(bsz, q_len, num_key_value_heads, head_dim),
            buffer.k_norm_variance_epsilon,
            buffer.k_norm_weight,
        ).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, num_key_value_heads, head_dim).transpose(1, 2)
        return query_states, key_states, value_states

    def _dense_mlp(self, hidden_states: torch.Tensor, buffer: Qwen3MoeLayer) -> torch.Tensor:
        up = F.linear(hidden_states, buffer.up_proj)
        gate = buffer.act_fn(F.linear(hidden_states, buffer.gate_proj))
        hidden_states = gate * up
        return F.linear(hidden_states, buffer.down_proj)

    def _grouped_mm(self, x: torch.Tensor, w: torch.Tensor, offs: torch.Tensor) -> torch.Tensor:
        if hasattr(F, "grouped_mm"):
            return F.grouped_mm(x.to(w.dtype), w, offs=offs)
        if hasattr(torch, "_grouped_mm"):
            return torch._grouped_mm(x.to(w.dtype), w, offs=offs)
        raise RuntimeError("No grouped_mm backend is available")

    def _moe_mlp(self, hidden_states: torch.Tensor, buffer: Qwen3MoeLayer) -> torch.Tensor:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)

        router_logits = F.linear(hidden_states, buffer.moe_gate)
        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, buffer.top_k, dim=-1)
        if buffer.norm_topk_prob:
            routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        routing_weights = routing_weights.to(hidden_states.dtype)

        if buffer.expert_gate_proj_t is not None:
            num_tokens = hidden_states.shape[0]
            token_idx = torch.arange(num_tokens, device=hidden_states.device).unsqueeze(1)
            token_idx = token_idx.expand(-1, buffer.top_k).reshape(-1)
            sample_weights = routing_weights.reshape(-1)
            expert_ids = selected_experts.reshape(-1)

            perm = torch.argsort(expert_ids)
            inv_perm = torch.empty_like(perm)
            inv_perm[perm] = torch.arange(perm.size(0), device=hidden_states.device)

            expert_ids_g = expert_ids[perm]
            sample_weights_g = sample_weights[perm]
            selected_hidden_states_g = hidden_states[token_idx[perm]].contiguous()

            tokens_per_expert = torch.bincount(expert_ids_g, minlength=buffer.num_experts)
            offsets = torch.cumsum(tokens_per_expert, dim=0, dtype=torch.int32)

            gate = self._grouped_mm(selected_hidden_states_g, buffer.expert_gate_proj_t, offs=offsets)
            up = self._grouped_mm(selected_hidden_states_g, buffer.expert_up_proj_t, offs=offsets)
            proj_out = self._grouped_mm(buffer.act_fn(gate) * up, buffer.expert_down_proj_t, offs=offsets)

            weighted_out = proj_out * sample_weights_g.unsqueeze(-1)
            weighted_out = weighted_out[inv_perm]
            final_hidden_states = weighted_out.view(num_tokens, buffer.top_k, hidden_dim).sum(dim=1)
            return final_hidden_states.reshape(batch_size, sequence_length, hidden_dim).to(hidden_states.dtype)

        final_hidden_states = torch.zeros_like(hidden_states)

        for expert_idx in range(buffer.num_experts):
            token_idx, slot_idx = torch.where(selected_experts == expert_idx)
            if token_idx.numel() == 0:
                continue

            current_state = hidden_states.index_select(0, token_idx)
            up = F.linear(current_state, buffer.expert_up_proj[expert_idx])
            gate = buffer.act_fn(F.linear(current_state, buffer.expert_gate_proj[expert_idx]))
            current_hidden_states = F.linear(gate * up, buffer.expert_down_proj[expert_idx])
            current_hidden_states = current_hidden_states * routing_weights[token_idx, slot_idx, None]
            final_hidden_states.index_add_(0, token_idx, current_hidden_states.to(hidden_states.dtype))

        return final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)

    def post_attention_compute(
        self,
        attn_output: torch.Tensor,
        residual: torch.Tensor,
        buffer: Qwen3MoeLayer,
    ):
        hidden_states = F.linear(attn_output, buffer.wo)
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = layer_norm(
            hidden_states,
            buffer.post_attention_layernorm_variance_epsilon,
            buffer.post_attention_layernorm_weight,
        )

        if buffer.is_moe:
            hidden_states = self._moe_mlp(hidden_states, buffer)
        else:
            hidden_states = self._dense_mlp(hidden_states, buffer)

        hidden_states = residual + hidden_states
        return hidden_states

    @torch.inference_mode()
    def apply_rotary_pos_emb_single(self, x: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        return apply_rotary_pos_emb_single(x, self.cos_cache.to(x.device), self.sin_cache.to(x.device), position_ids)

    @torch.inference_mode()
    def apply_rotary_pos_emb(self, q: torch.Tensor, k: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        return apply_rotary_pos_emb(q, k, cos=self.cos_cache, sin=self.sin_cache, position_ids=position_ids)
