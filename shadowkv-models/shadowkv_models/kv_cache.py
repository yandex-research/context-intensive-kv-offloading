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
import math
import gc
from torch import nn
from .tensor_op import batch_gather_gemm_rotary_pos_emb_cuda
from . import kernels
from .quantization.higgs import HiggsQuantizer
from .quantization.nvfp4 import NVFP4Quantizer
from .quantization.fp8 import FP8Quantizer

class KV_Cache:
    """Full Attention"""
    def __init__(self, 
        config :object,
        batch_size :int = 1,
        max_length :int = 32*1024, 
        device :str = 'cuda:0',
        dtype = torch.bfloat16) -> None:

        self.config = config
        self.max_length = max_length
        self.device = device
        self.dtype = dtype
        self.k_cache = torch.zeros(
            config.num_hidden_layers,
            batch_size,
            config.num_key_value_heads,
            max_length,
            config.head_dim,
            device='cpu',
            dtype=self.dtype
        )

        self.v_cache = torch.zeros(
            config.num_hidden_layers,
            batch_size,
            config.num_key_value_heads,
            max_length,
            config.head_dim,
            device='cpu',
            dtype=self.dtype
        )
        self.num_layers = config.num_hidden_layers
        self.kv_offset = 0

        # batch prefill record
        self.prefilled_batch = 0
        self.batch_size = batch_size

    def update_kv_cache(self, 
            new_k_cache :torch.Tensor,
            new_v_cache :torch.Tensor,
            layer_idx :int
            ):

        bsz, _, incoming, _ = new_v_cache.shape # [bsz, num_kv_heads, incoming, head_dim]

        if bsz == self.batch_size:
            self.prefilled_batch = 0

        self.k_cache[layer_idx][self.prefilled_batch:self.prefilled_batch + bsz, :, self.kv_offset:self.kv_offset + incoming].copy_(new_k_cache)
        self.v_cache[layer_idx][self.prefilled_batch:self.prefilled_batch + bsz, :, self.kv_offset:self.kv_offset + incoming].copy_(new_v_cache)

        key = self.k_cache[layer_idx][self.prefilled_batch:self.prefilled_batch + bsz, :, :self.kv_offset + incoming]
        value = self.v_cache[layer_idx][self.prefilled_batch:self.prefilled_batch + bsz, :, :self.kv_offset + incoming]

        if incoming > 1: # prefill
            key = key.to(self.device)
            value = value.to(self.device)

        if layer_idx == self.num_layers - 1:
            self.prefilled_batch += bsz
            if self.prefilled_batch == self.batch_size:
                self.kv_offset += incoming
        
        return key.to(self.device), value.to(self.device)
    
    def print_stats(self):
        print(f"KVCache | max_length {self.max_length} | dtype {self.dtype} | cached {self.kv_offset}")

    def H2D(self):
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        self.k_cache = self.k_cache.to(self.device)
        self.v_cache = self.v_cache.to(self.device)

    def clear(self):
        self.kv_offset = 0
        self.prefilled_batch = 0

    def get_kv_len(self):
        return self.kv_offset

class ShadowKVCache:
    """ShadowKV, only for accuracy measurement and understanding, not for efficiency, please refer to ShadowKV_CPU for the efficient implementation"""
    def __init__(self, 
        config :object,
        batch_size :int = 1,
        max_length :int = 32*1024, 
        device :str = 'cuda:0',
        dtype = torch.bfloat16,
        sparse_budget: int = 2048,
        chunk_size=8,
        rank=160,
        local_budget=4*8,
        outlier_budget=48*8,
        keys_on_device: bool = True,
        # nvfp4 quantization
        use_nvfp4_quantization: bool = False,
        nvfp4_channel_size: int = 1024,
        # HIGGS quantization
        use_higgs_quantization: bool = False,
        higgs_hadamard_groupsize: int = 128, 
        higgs_edenn_d: int = 16, 
        higgs_edenn_n: int = 256,
        higgs_channel_size: int = 1024, 
        higgs_chunk_size: int = 64,
        # Residual quantization
        use_residual_quantization: bool = False,
        meta_chunk_size: int = 1,
        residual_higgs_edenn_d: int = 16, 
        residual_higgs_edenn_n: int = 256,
        # keys low bit format
        keys_fp8: bool = False,
        keys_nvfp4: bool = False,
        keys_higgs: bool = False,
        # values low bit format
        values_fp8: bool = False,
        values_nvfp4: bool = False,
        values_higgs: bool = False,
        # HIGGS quantization for keys
        keys_higgs_hadamard_groupsize: int = 128,
        keys_higgs_edenn_d: int = 16,
        keys_higgs_edenn_n: int = 256,
        keys_higgs_channel_size: int = 1024,
        keys_higgs_chunk_size: int = 64,
        #  HIGGS quantization for values
        values_higgs_hadamard_groupsize: int = 128,
        values_higgs_edenn_d: int = 16,
        values_higgs_edenn_n: int = 256,
        values_higgs_channel_size: int = 1024,
        values_higgs_chunk_size: int = 64,
    ) -> None:
        
        self.config = config
        self.batch_size = batch_size
        self.max_length = max_length
        self.device = device
        self.dtype = dtype
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.head_dim = config.head_dim
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads

        self.sparse_budget = int(sparse_budget)
        self.chunk_size = chunk_size
        self.rank = rank

        assert local_budget % chunk_size == 0
        assert outlier_budget % chunk_size == 0
        self.local_chunk = local_budget // chunk_size
        self.outlier_chunk = outlier_budget // chunk_size
        self.use_nvfp4_quantization = use_nvfp4_quantization
        self.nvfp4_channel_size = nvfp4_channel_size
        self.use_higgs_quantization = use_higgs_quantization
        self.higgs_hadamard_groupsize = higgs_hadamard_groupsize
        self.higgs_edenn_d = higgs_edenn_d
        self.higgs_edenn_n = higgs_edenn_n
        self.higgs_channel_size = higgs_channel_size
        self.higgs_chunk_size = higgs_chunk_size
        self.use_residual_quantization = use_residual_quantization
        self.meta_chunk_size = meta_chunk_size
        self.residual_higgs_edenn_d = residual_higgs_edenn_d
        self.residual_higgs_edenn_n = residual_higgs_edenn_n

        assert self.batch_size == 1, "ShadowKV class only supports batch_size=1, please use ShadowKV_CPU class for batch_size > 1"

        self.selected_chunk_idx = torch.zeros(
            config.num_hidden_layers,
            batch_size,
            config.num_key_value_heads,
            self.sparse_budget // self.chunk_size,
            device=self.device,
            dtype=torch.long
        )

        self.v_cache_cpu = torch.zeros(
            config.num_hidden_layers,
            batch_size,
            config.num_key_value_heads,
            self.max_length,
            self.config.head_dim,
            device=self.device,
            dtype=self.dtype
        )

        self.k_cache_buffer = torch.zeros(
            config.num_hidden_layers,
            batch_size,
            config.num_key_value_heads,
            self.sparse_budget + 4096,
            self.config.head_dim,
            device=self.device,
            dtype=self.dtype
        )

        self.v_cache_buffer = torch.zeros(
            config.num_hidden_layers,
            batch_size,
            config.num_key_value_heads,
            self.sparse_budget + 4096,
            self.config.head_dim,
            device=self.device,
            dtype=self.dtype
        )


        self.num_layers = config.num_hidden_layers
        self.kv_offset = 0
        self.prefill = 0
        self.gen_offset = 0

        self.k_landmark = None
        self.k_landmark_idx = None
        self.U = None
        self.SV = None

        assert not (use_nvfp4_quantization and use_higgs_quantization), (
            "Only one of use_nvfp4_quantization or use_higgs_quantization should be specified"
        )
        assert not (use_residual_quantization and not use_higgs_quantization), (
            "use_residual_quantization requires use_higgs_quantization"
        )

        if use_nvfp4_quantization:
            self.quantizer = NVFP4Quantizer(
                channel_size=nvfp4_channel_size
            )
        if use_higgs_quantization:
            self.quantizer = HiggsQuantizer(
               hadamard_groupsize=higgs_hadamard_groupsize,
               edenn_d=higgs_edenn_d,
               edenn_n=higgs_edenn_n,
               channel_size=higgs_channel_size,
               chunk_size=higgs_chunk_size
            )

        self.use_residual_quantization = use_residual_quantization
        if use_residual_quantization:
            self.residual_quantizer = HiggsQuantizer(
                hadamard_groupsize=higgs_hadamard_groupsize,
                edenn_d=residual_higgs_edenn_d,
                edenn_n=residual_higgs_edenn_n,
                channel_size=higgs_channel_size,
                chunk_size=higgs_chunk_size,
            )
        else:
            self.residual_quantizer = None

        if keys_fp8 or keys_nvfp4 or keys_higgs:
            assert rank == -1
        assert sum([keys_fp8, keys_nvfp4, keys_higgs]) <= 1, (
            "Only one of [keys_fp8, keys_nvfp4, keys_higgs] should be specified"
        )
        self.keys_low_bit_format = False
        if keys_fp8:
            self.keys_quantizer = FP8Quantizer(self.dtype)
            self.keys_low_bit_format = True
        if keys_nvfp4:
            self.keys_quantizer = NVFP4Quantizer(
                channel_size=nvfp4_channel_size
            )
            self.keys_low_bit_format = True
        if keys_higgs:
            self.keys_quantizer = HiggsQuantizer(
                hadamard_groupsize=keys_higgs_hadamard_groupsize,
                edenn_d=keys_higgs_edenn_d,
                edenn_n=keys_higgs_edenn_n,
                channel_size=keys_higgs_channel_size,
                chunk_size=keys_higgs_chunk_size
            )
            self.keys_low_bit_format = True

        assert sum([values_fp8, values_nvfp4, values_higgs]) <= 1, (
            "Only one of values_fp8 or values_nvfp4 or values_higgs should be specified"
        )
        self.values_low_bit_format = False
        if values_fp8:
            self.value_quantizer = FP8Quantizer(self.dtype)
            self.values_low_bit_format = True
        if values_nvfp4:
            self.value_quantizer = NVFP4Quantizer(
                channel_size=nvfp4_channel_size
            )
            self.values_low_bit_format = True
        if values_higgs:
            self.value_quantizer = HiggsQuantizer(
                hadamard_groupsize=values_higgs_hadamard_groupsize,
                edenn_d=values_higgs_edenn_d,
                edenn_n=values_higgs_edenn_n,
                channel_size=values_higgs_channel_size,
                chunk_size=values_higgs_chunk_size
            )
            self.values_low_bit_format = True
        self.copy_stream = torch.cuda.Stream()
        self.keys_on_device = keys_on_device

    def print_stats(self):
        print(f"ShadowKV | sparse budget {self.sparse_budget} | chunk size {self.chunk_size} |rank {self.rank} | cached {self.kv_offset} | local_chunk {self.local_chunk} | outlier_chunk {self.outlier_chunk}")

    def get_svd(self, new_k_cache, layer_idx):
        new_k_cache = new_k_cache.to(self.device)
        # [bsz, 8, prefill, 128] OR [bsz, prefill, 1024]
        if new_k_cache.ndim == 4: # new_k_cache.shape[1] <= 32
            # [bsz, 8, prefill, 128] --> [bsz, prefill, 1024]
            k_cache = new_k_cache.transpose(1, 2).reshape(self.batch_size, -1, self.num_key_value_heads * self.head_dim)
        else:
            # [bsz, prefill, 1024]
            k_cache = new_k_cache
        
        if layer_idx == 0: 
            if self.rank == -1 or self.keys_low_bit_format:
                if self.keys_on_device:
                    keys_device = self.device
                else:
                    keys_device = "cpu"
                self.keys = torch.zeros(self.num_layers, self.batch_size, k_cache.shape[1], self.num_key_value_heads * self.head_dim, device=keys_device, dtype=self.dtype)
            else:
                # init U, SV
                self.U = torch.zeros(self.num_layers, self.batch_size, k_cache.shape[1], self.rank, device=self.device, dtype=self.dtype)
                self.SV = torch.zeros(self.num_layers, self.batch_size, self.num_key_value_heads, self.rank, self.head_dim, device=self.device, dtype=self.dtype)
        if self.rank != -1:
            u, s, v = torch.svd(k_cache.float())
            v = v.transpose(1,2)
            # [bsz, 128k, 1024] --> [bsz, 128k, 160] [bsz, 160, 1024] (bsz, 8, 160, 128)
            self.U[layer_idx].copy_(u[:, :, :self.rank].to(self.dtype)) # [bsz, 128k, 160]
            self.SV[layer_idx].copy_(torch.matmul(torch.diag_embed(s[:, :self.rank]), v[:, :self.rank]).to(self.dtype).view(self.batch_size, -1, self.num_key_value_heads, self.head_dim).transpose(1, 2)) # [bsz, 8, 160, 128]
        else:
            if self.keys_low_bit_format:
                bsz, n_tokens, channel_size = k_cache.shape
                k_cache = self.keys_quantizer.quantize_dequantize(k_cache.view(bsz * n_tokens, channel_size))
            self.keys[layer_idx].copy_(k_cache)
    
    def register_k_landmark(self, k_landmark, k_landmark_idx, layer_idx):
        k_landmark = k_landmark.to(self.device)
        num_landmarks = k_landmark.shape[-2]
        if layer_idx == 0:
            # init k_landmark, k_landmark_idx
            self.k_landmark = torch.zeros(self.num_layers, self.batch_size, self.num_key_value_heads, num_landmarks, self.head_dim, device=self.device, dtype=self.dtype)
            self.k_landmark_idx = torch.zeros(self.num_layers, self.batch_size, self.num_key_value_heads, num_landmarks, device=self.device, dtype=torch.long)
        
        # Maybe quantize k_landmark
        if self.use_higgs_quantization or self.use_nvfp4_quantization:
            if self.use_residual_quantization:
                # Pad to nearest multiple of meta_chunk_size if needed
                remainder = num_landmarks % self.meta_chunk_size
                if remainder != 0:
                    pad = self.meta_chunk_size - remainder
                    k_landmark = torch.cat(
                        [k_landmark, k_landmark[:, :, -1:, :].expand(-1, -1, pad, -1)],
                        dim=2
                    )
                num_padded = k_landmark.shape[2]
                num_meta_landmarks = num_padded // self.meta_chunk_size
                # Average landmarks
                meta_landmarks = k_landmark.view(
                    self.batch_size,
                    self.num_key_value_heads,
                    num_meta_landmarks,
                    self.meta_chunk_size,
                    self.head_dim
                ).mean(dim=-2)
                # Reshape meta landmarks
                meta_landmarks = meta_landmarks.transpose(1, 2).reshape(-1, self.num_key_value_heads * self.head_dim)
                # Quantize landmarks
                meta_landmarks = self.quantizer.quantize_dequantize(meta_landmarks)
                meta_landmarks = meta_landmarks.reshape(self.batch_size, num_meta_landmarks, self.num_key_value_heads, self.head_dim).transpose(1, 2)
                residual_landmark = k_landmark - meta_landmarks.repeat_interleave(self.meta_chunk_size, dim=2)
                residual_landmark = residual_landmark.transpose(1, 2).reshape(-1, self.num_key_value_heads * self.head_dim)
                residual_landmark = self.residual_quantizer.quantize_dequantize(residual_landmark)
                k_landmark = (
                    residual_landmark.reshape(
                        self.batch_size,
                        num_padded,
                        self.num_key_value_heads,
                        self.head_dim
                    ).transpose(1, 2)
                    + meta_landmarks.repeat_interleave(self.meta_chunk_size, dim=2)
                )[:, :, :num_landmarks, :]
            else:
                k_landmark = k_landmark.transpose(1, 2).reshape(-1, self.num_key_value_heads * self.head_dim)
                k_landmark = self.quantizer.quantize_dequantize(k_landmark)
                k_landmark = k_landmark.reshape(self.batch_size, num_landmarks, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        
        self.k_landmark[layer_idx].copy_(k_landmark.contiguous())
        self.k_landmark_idx[layer_idx].copy_(k_landmark_idx.contiguous())

    def prefill_kv_cache(self,
            new_v_cache :torch.Tensor,
            layer_idx :int,
            key_states_roped: torch.Tensor,
            query: torch.Tensor=None
            ):
        
        new_v_cache = new_v_cache.to(self.device)
        key_states_roped = key_states_roped.to(self.device)
        if query is not None:
            query = query.to(self.device)

        bsz, num_kv_heads, incoming = new_v_cache.shape[:3] # [bsz, num_kv_heads, incoming, head_dim]
        self.prefill = incoming
        v_cache = new_v_cache.clone()
        if self.values_low_bit_format:
            v_cache = v_cache.transpose(1, 2).view(bsz * incoming, -1)
            v_cache = self.value_quantizer.quantize_dequantize(v_cache).view(bsz, incoming, num_kv_heads, -1).transpose(1, 2)
        self.v_cache_cpu[layer_idx][:, :, :incoming] = v_cache

        # [x0, x1, ...., self.chunks*chunk_size, local_chunk, rest]
        self.chunks = incoming // self.chunk_size - self.local_chunk 
        self.select_sets = self.sparse_budget // self.chunk_size
        
        assert self.select_sets * self.chunk_size == self.sparse_budget, f"({self.select_sets}) * {self.chunk_size} != {self.sparse_budget}"
        
        # store Post-RoPE k cache <prefill_local> to the cache
        self.prefill_local = incoming - self.chunks * self.chunk_size # local chunks + align to chunk_size
        self.k_cache_buffer[layer_idx][:, :, :self.prefill_local].copy_(key_states_roped[:, :, -self.prefill_local:])
        self.v_cache_buffer[layer_idx][:, :, :self.prefill_local].copy_(new_v_cache[:, :, -self.prefill_local:])

        key_states_roped_ctx = key_states_roped[:,:,:self.chunks*self.chunk_size].view(self.batch_size, self.num_key_value_heads, self.chunks, self.chunk_size, self.head_dim)
        landmark_candidates = key_states_roped_ctx.mean(dim=-2) # [bsz, kv_heads, chunks, head_dim]
        
        # compute the cos similarity between it and the original key cache
        cos_sim = torch.nn.functional.cosine_similarity(landmark_candidates.unsqueeze(3).expand(-1, -1, -1, self.chunk_size, -1), key_states_roped_ctx, dim=-1) # [bsz, kv_heads, chunks, chunk_size]
        
        # get the outlier_chunk idx for each head # [bsz, kv_heads, outlier_chunk]
        if self.chunk_size > 1:
            outlier_chunk_idx = cos_sim.min(dim=-1).values.topk(self.outlier_chunk, largest=False).indices
        else:
            outlier_chunk_idx = torch.arange(self.outlier_chunk, device=cos_sim.device, dtype=torch.long).unsqueeze(0).unsqueeze(0).expand(bsz, num_kv_heads, -1)
        # [bsz, kv_heads, chunks, chunk_size, head_dim] --gather[bsz, kv_heads, outlier_chunk]-->[bsz, kv_heads, outlier_chunk, chunk_size, head_dim]
        outlier_chunk_k_cache = key_states_roped_ctx.gather(dim=2, index=outlier_chunk_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, self.chunk_size, self.head_dim)).view(self.batch_size, self.num_key_value_heads, self.outlier_chunk*self.chunk_size, self.head_dim)
        
        outlier_chunk_v_cache = new_v_cache[:,:,:self.chunks*self.chunk_size].view(self.batch_size, self.num_key_value_heads, self.chunks, self.chunk_size, self.head_dim).gather(dim=2, index=outlier_chunk_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, self.chunk_size, self.head_dim)).view(self.batch_size, self.num_key_value_heads, self.outlier_chunk*self.chunk_size, self.head_dim)

        self.sparse_start = self.prefill_local + self.outlier_chunk*self.chunk_size
        self.sparse_end = self.prefill_local + self.outlier_chunk*self.chunk_size + self.sparse_budget
        
        # store outlier_chunk to the cache
        self.k_cache_buffer[layer_idx][:, :, self.prefill_local:self.sparse_start].copy_(outlier_chunk_k_cache)
        self.v_cache_buffer[layer_idx][:, :, self.prefill_local:self.sparse_start].copy_(outlier_chunk_v_cache)

        # filter landmark_candidates using outlier_chunk and register the rest to k_landmark
        # [bsz, kv_heads, chunks, head_dim] --> [bsz, kv_heads, chunks - outlier_chunk, head_dim]
        # get rest_idx: [bsz, kv_heads, chunks] --filter--> [bsz, kv_heads, chunks - outlier_chunk]
        all_idx = torch.arange(self.chunks, device=key_states_roped.device).unsqueeze(0).unsqueeze(0).expand(self.batch_size, self.num_key_value_heads, -1) # [bsz, kv_heads, chunks]
        mask = torch.ones_like(all_idx, dtype=torch.bool)
        mask.scatter_(dim=-1, index=outlier_chunk_idx, value=False)
        rest_idx = all_idx.masked_select(mask).view(self.batch_size, self.num_key_value_heads, -1)

        # register rest_idxed landmarks to k_landmark
        self.register_k_landmark(landmark_candidates.gather(dim=2, index=rest_idx.unsqueeze(-1).expand(-1, -1, -1, self.head_dim)).view(self.batch_size, self.num_key_value_heads, -1, self.head_dim), rest_idx, layer_idx)

        if layer_idx == self.num_layers - 1:
            # assert self.sparse_budget < incoming
            self.kv_offset += incoming

    def get_retrieval_position_ids(self, layer_idx, query_states):
        query_states = query_states.to(self.device)
        # self.k_landmark[layer_idx][:, :, :self.chunks] is [bsz, 8, chunks, head_dim]
        # chunk_attn: [bsz, 32, window_size, chunks]
        self.incoming_q_len = query_states.shape[-2] # 1
        # print(query_states.view(-1, self.num_key_value_heads, self.num_key_value_groups, self.incoming_q_len, self.head_dim).shape, self.k_landmark[layer_idx].transpose(2, 3).shape)
        # [bsz, 8, 4, q_len, 128] * [bsz, 8, 128, chunks] --> [bsz, 8, 4, q_len, chunks]
        chunk_attn = torch.einsum('bhgqd,bhdc->bhgqc', query_states.view(-1, self.num_key_value_heads, self.num_key_value_groups, self.incoming_q_len, self.head_dim), self.k_landmark[layer_idx].transpose(2, 3)).squeeze(2) / math.sqrt(self.head_dim)
        chunk_attn = nn.functional.softmax(chunk_attn, dim=-1, dtype=torch.float32).to(self.dtype) # [bsz, 8, 4, q_len, chunks]
        chunk_attn = chunk_attn.sum(dim = -2) # [bsz, 8, 4, chunks]
        if self.num_key_value_groups > 1:
            chunk_attn, _ = torch.max(chunk_attn, dim=-2) # [bsz, 8, chunks]
        merged_results = torch.topk(chunk_attn, k=self.select_sets, dim=-1).indices # [bsz, 8, select_sets(256)]

        # use merged_results to gather the position_ids: [bsz, 8, select_sets] --> [bsz, 8, select_sets]
        selected_chunks = self.k_landmark_idx[layer_idx].gather(dim=-1, index=merged_results) # [bsz, 8, select_sets]

        # this is chunk idx, which can be used to offload value cache and decide if the cache hits
        self.selected_chunk_idx[layer_idx].copy_(selected_chunks, non_blocking=True)

        position_ids = (selected_chunks.unsqueeze(-1) * self.chunk_size + torch.arange(self.chunk_size, device=chunk_attn.device).unsqueeze(0).unsqueeze(0).unsqueeze(0)).view(self.batch_size, self.num_key_value_heads, -1) # [bsz, 8, select_sets * chunk_size]

        return position_ids
        
    def get_value_cache(self, layer_idx, position_ids):
        position_ids = position_ids.to(self.device)
        # gather value cache
        value_ = self.v_cache_cpu[layer_idx].gather(dim=-2, index=position_ids.unsqueeze(-1).expand(-1, -1, -1, self.head_dim))
        self.v_cache_buffer[layer_idx][:, :, self.sparse_start:self.sparse_end].copy_(value_, non_blocking=True)
        gen_offset = self.gen_offset if layer_idx == self.num_layers - 1 else self.gen_offset + self.incoming_q_len

        return self.v_cache_buffer[layer_idx][:, :, :self.sparse_end + gen_offset]

    def get_key_cache(self, layer_idx, position_ids, rope_func, cos_sin_cache):
        position_ids = position_ids.to(self.device)
        # gather key cache and rope them
        if self.rank == -1:
            result = self.keys[layer_idx]
            bs, n_tokens, channel_size = result.shape
            result = result.view(bs, n_tokens, self.num_key_value_heads, self.head_dim).transpose(1, 2)
            index_expanded = position_ids.unsqueeze(-1).expand(-1, -1, -1, self.head_dim) # [bsz, 8, sparse_budget, head_dim]
            result = torch.gather(result.to(self.device), 2, index_expanded)
            # print(result.shape, result.device)
        else:
            u = self.U[layer_idx] # [bsz, 128k, rank]
            sv = self.SV[layer_idx] # [bsz, 8, rank, 128]

            # indexing, [bsz, 8, sparse_budget, rank]
            index_expanded = position_ids.unsqueeze(-1).expand(-1, -1, -1, u.size(-1)) # [bsz, 8, sparse_budget, rank]
            u_expand = u.unsqueeze(1).expand(-1, self.num_key_value_heads, -1, -1) # [bsz, 8, 128k, rank]
            U_head = torch.gather(u_expand, 2, index_expanded)

            # [bsz, 8, sparse_budget, rank] -matmul- [8, rank, 128] --> [bsz, 8, sparse_budget, 128]
            result = torch.einsum('bhrk,bhkd->bhrd', U_head, sv)

        # rope the key cache
        result = rope_func(result, position_ids.to("cpu"))  # for double gpu inference

        # send to buffer
        self.k_cache_buffer[layer_idx][:, :, self.sparse_start:self.sparse_end].copy_(result, non_blocking=True)
        gen_offset = self.gen_offset if layer_idx == self.num_layers - 1 else self.gen_offset + self.incoming_q_len

        return self.k_cache_buffer[layer_idx][:, :, :self.sparse_end + gen_offset]

    def update_kv_cache(self, 
            new_k_cache :torch.Tensor,
            new_v_cache :torch.Tensor,
            layer_idx :int,
            ):

        new_k_cache = new_k_cache.to(self.device)
        new_v_cache = new_v_cache.to(self.device)

        incoming = new_k_cache.shape[-2]
        self.v_cache_buffer[layer_idx][:, :, self.sparse_end+self.gen_offset:self.sparse_end+self.gen_offset+incoming].copy_(new_v_cache, non_blocking=True)
        self.k_cache_buffer[layer_idx][:, :, self.sparse_end+self.gen_offset:self.sparse_end+self.gen_offset+incoming].copy_(new_k_cache, non_blocking=True)

        if layer_idx == self.num_layers - 1:
            self.kv_offset += incoming
            self.gen_offset += incoming

    def clear(self):
        self.k_cache_buffer.zero_()
        self.v_cache_buffer.zero_()
        self.selected_chunk_idx.zero_()
        self.k_landmark = None
        self.k_landmark_idx = None
        self.U = None
        self.SV = None

        self.kv_offset = 0
        self.prefill = 0
        self.gen_offset = 0
        self.prefill_local = 0
    
    def H2D(self):
        pass

    def get_kv_len(self):
        return self.kv_offset


class ShadowKVCache_CPU:
    """ShadowKV, can be used for Llama-3-8B, Llama-3.1-8B, GLM-4-9B, Yi-200K.

    Budget sizes scale dynamically with the input sequence length via ratios, and
    the generation buffer grows automatically as new tokens are produced.
    """
    def __init__(self,
        config: object,
        batch_size: int = 1,
        device: str = 'cuda:0',
        dtype=torch.bfloat16,
        sparse_budget_ratio: float = 0.0625,
        local_budget_ratio: float = 0.001,
        outlier_budget_ratio: float = 0.012,
        chunk_size: int = 8,
        rank: int = 160,
        gen_buffer_init_size: int = 256,
        max_length: int = None,  # retained for API compatibility, unused
        ) -> None:

        self.config = config
        self.batch_size = batch_size
        self.device = device
        self.dtype = dtype
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.head_dim = config.head_dim
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_layers = config.num_hidden_layers

        self.sparse_budget_ratio = sparse_budget_ratio
        self.local_budget_ratio = local_budget_ratio
        self.outlier_budget_ratio = outlier_budget_ratio
        self.chunk_size = chunk_size
        self.rank = rank
        self._gen_buffer_init_size = gen_buffer_init_size

        # Scalar state (reset on clear)
        self.kv_offset = 0
        self.prefill = 0
        self.gen_offset = 0
        self.prefilled_batch = 0

        # Computed from input length during first prefill (reset on clear)
        self.sparse_budget = None
        self.local_chunk = None
        self.outlier_chunk = None
        self.select_sets = None

        # Lazily allocated tensors (reset on clear)
        self.v_cache_cpu = None
        self.k_cache_buffer = None
        self.v_cache_buffer = None
        self.temp = None
        self.offsets = None
        self.cnts = None
        self.signals = None
        self.position_ids = None
        self.output = None

        self.k_landmark = None
        self.k_landmark_idx = None
        self.U = None
        self.SV = None

        # multi-stream
        self.copy_stream = torch.cuda.Stream()

    def _compute_budgets(self, incoming: int) -> None:
        """Compute integer token budgets from ratios and input length, aligned to chunk_size."""
        cs = self.chunk_size
        local_tokens = max(cs, math.floor(self.local_budget_ratio * incoming / cs) * cs)
        outlier_tokens = max(cs, math.floor(self.outlier_budget_ratio * incoming / cs) * cs)
        sparse_tokens = max(cs, math.floor(self.sparse_budget_ratio * incoming / cs) * cs)

        self.local_chunk = local_tokens // cs
        self.outlier_chunk = outlier_tokens // cs
        self.sparse_budget = sparse_tokens
        self.select_sets = sparse_tokens // cs

        assert self.select_sets * cs == self.sparse_budget, \
            f"({self.select_sets}) * {cs} != {self.sparse_budget}"

    def _init_buffers(self, incoming: int) -> None:
        """Allocate all cache buffers sized for the given input sequence length.
        Called once at the start of each new sequence (layer_idx=0, prefilled_batch=0).
        """
        self._compute_budgets(incoming)

        cs = self.chunk_size
        # Compute prefill_local the same way prefill_kv_cache does
        chunks = incoming // cs - self.local_chunk
        chunks = chunks - chunks % 8
        prefill_local = incoming - chunks * cs

        buf_size = (prefill_local
                    + self.outlier_chunk * cs
                    + self.sparse_budget
                    + self._gen_buffer_init_size)

        max_ctx_chunks = incoming // cs

        self.v_cache_cpu = torch.zeros(
            self.num_layers,
            self.batch_size,
            self.num_key_value_heads,
            max_ctx_chunks,
            self.head_dim * cs,
            device='cpu',
            dtype=self.dtype,
            pin_memory=True,
        )

        self.k_cache_buffer = torch.zeros(
            self.num_layers,
            self.batch_size,
            self.num_key_value_heads,
            buf_size,
            self.head_dim,
            device=self.device,
            dtype=self.dtype,
        )

        self.v_cache_buffer = torch.zeros(
            self.num_layers,
            self.batch_size,
            self.num_key_value_heads,
            buf_size,
            self.head_dim,
            device=self.device,
            dtype=self.dtype,
        )

        block_num = self.batch_size * self.num_key_value_heads
        self.offsets = torch.zeros(
            block_num * self.select_sets, device=self.device, dtype=torch.int32
        ).contiguous()
        self.cnts = torch.zeros(block_num, device=self.device, dtype=torch.int32).contiguous()
        self.signals = torch.zeros(block_num, device=self.device, dtype=torch.int32).contiguous()
        self.position_ids = torch.zeros(
            self.num_layers, self.batch_size, self.num_key_value_heads, self.select_sets,
            device=self.device, dtype=torch.int64,
        ).fill_(-1).contiguous()

        self.temp = torch.zeros(
            self.batch_size,
            self.num_key_value_heads,
            self.select_sets,
            cs * self.head_dim,
            device='cpu',
            dtype=self.dtype,
        ).contiguous()

        self.output = torch.zeros(
            self.batch_size,
            self.num_key_value_heads,
            self.sparse_budget,
            self.head_dim,
            device='cpu',
            dtype=self.dtype,
        ).contiguous()

        self.kernel_stride = buf_size * self.head_dim

    def _extend_generation_buffers(self, additional: int) -> None:
        """Grow k_cache_buffer and v_cache_buffer to fit more generated tokens."""
        old_size = self.k_cache_buffer.shape[-2]
        new_size = old_size + max(additional, self._gen_buffer_init_size)

        new_k = torch.zeros(
            self.num_layers, self.batch_size, self.num_key_value_heads,
            new_size, self.head_dim, device=self.device, dtype=self.dtype,
        )
        new_v = torch.zeros(
            self.num_layers, self.batch_size, self.num_key_value_heads,
            new_size, self.head_dim, device=self.device, dtype=self.dtype,
        )
        new_k[:, :, :, :old_size, :].copy_(self.k_cache_buffer)
        new_v[:, :, :, :old_size, :].copy_(self.v_cache_buffer)

        self.k_cache_buffer = new_k
        self.v_cache_buffer = new_v
        self.kernel_stride = new_size * self.head_dim

    def print_stats(self):
        if self.sparse_budget is None:
            print(f"ShadowKV_CPU | sparse ratio {self.sparse_budget_ratio} | chunk size {self.chunk_size} | rank {self.rank} | (not yet prefilled)")
        else:
            print(f"ShadowKV_CPU | sparse budget {self.sparse_budget} | chunk size {self.chunk_size} | rank {self.rank} | cached {self.kv_offset} | local_chunk {self.local_chunk} | outlier_chunk {self.outlier_chunk}")

    ##### Encoding #####
    def get_svd(self, new_k_cache, layer_idx):
        # [bsz, 8, prefill, 128] OR [bsz, prefill, 1024]
        if new_k_cache.shape[1] <= 32:
            # [bsz, 8, prefill, 128] --> [bsz, prefill, 1024]
            k_cache = new_k_cache.transpose(1, 2).reshape(self.batch_size, -1, self.num_key_value_heads*self.head_dim)
        else:
            # [bsz, prefill, 1024]
            k_cache = new_k_cache
        
        if layer_idx == 0 and self.prefilled_batch == 0:
            # init U, SV
            self.U = torch.zeros(self.num_layers, self.batch_size, k_cache.shape[1], self.rank, device='cpu', dtype=self.dtype)
            self.SV = torch.zeros(self.num_layers, self.batch_size, self.num_key_value_heads, self.head_dim, self.rank, device='cpu', dtype=self.dtype)
        
        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        u, s, v = torch.svd(k_cache.float())
        v = v.transpose(1,2)
        
        bsz = k_cache.shape[0]
        # [bsz, 128k, 1024] --> [bsz, 128k, 160] [bsz, 160, 1024] (bsz, 8, 160, 128)
        self.U[layer_idx][self.prefilled_batch:self.prefilled_batch + bsz].copy_(u[:, :, :self.rank].to(self.dtype)) # [bsz, 128k, 160]
        
        temp_sv = torch.matmul(torch.diag_embed(s[:, :self.rank]), v[:, :self.rank]).to(self.dtype).view(bsz, -1, self.num_key_value_heads, self.head_dim).transpose(1, 2) # [bsz, 8, 160, 128]

        # used for kernel
        temp_sv = temp_sv.transpose(-1, -2) # [bsz, 8, 128, 160]
        
        self.SV[layer_idx][self.prefilled_batch:self.prefilled_batch + bsz].copy_(temp_sv) # [bsz, 8, 128, 160]

        del u, s, v

    def register_k_landmark(self, k_landmark, k_landmark_idx, layer_idx):
        num_landmarks = k_landmark.shape[-2]
        bsz = k_landmark.shape[0]
        if layer_idx == 0 and self.prefilled_batch == 0:
            # init k_landmark, k_landmark_idx
            self.k_landmark = torch.zeros(self.num_layers, self.batch_size, self.num_key_value_heads, num_landmarks, self.head_dim, device='cpu', dtype=self.dtype)
            self.k_landmark_idx = torch.zeros(self.num_layers, self.batch_size, self.num_key_value_heads, num_landmarks, device='cpu', dtype=torch.long)

            # for fused gemm kernel
            self.gemm_o = torch.zeros(self.batch_size, self.num_key_value_heads, self.num_key_value_groups, num_landmarks, device='cpu', dtype=torch.bfloat16).contiguous()
            self.softmax_o = torch.zeros(self.batch_size, self.num_key_value_heads, self.num_key_value_groups, num_landmarks, device='cpu', dtype=torch.bfloat16).contiguous()
            self.norm = torch.zeros(self.batch_size*self.num_key_value_heads, self.num_key_value_groups, (num_landmarks + 256 - 1) // 256, device='cpu', dtype=torch.float).contiguous()
            self.sum = torch.zeros(self.batch_size*self.num_key_value_heads, self.num_key_value_groups, (num_landmarks + 256 - 1) // 256, device='cpu', dtype=torch.float).contiguous()
        
        self.k_landmark[layer_idx][self.prefilled_batch:self.prefilled_batch + bsz].copy_(k_landmark)
        self.k_landmark_idx[layer_idx][self.prefilled_batch:self.prefilled_batch + bsz].copy_(k_landmark_idx)

    def prefill_kv_cache(self,
            new_v_cache :torch.Tensor,
            layer_idx :int,
            key_states_roped: torch.Tensor,
            last_query_states=None
            ):

        bsz, _, incoming, _ = new_v_cache.shape # [bsz, num_kv_heads, incoming, head_dim]
        self.prefill = incoming

        # Lazy initialisation: allocate all buffers once per new sequence
        if layer_idx == 0 and self.prefilled_batch == 0:
            self._init_buffers(incoming)

        max_ctx_chunks = incoming // self.chunk_size
        self.max_ctx_chunks_len = max_ctx_chunks * self.chunk_size
        self.v_cache_cpu[layer_idx][self.prefilled_batch:self.prefilled_batch + bsz, :, :max_ctx_chunks].copy_(new_v_cache[:, :, :self.max_ctx_chunks_len].reshape(bsz, self.num_key_value_heads, max_ctx_chunks, self.chunk_size*self.head_dim), non_blocking=True) # [bsz, num_kv_heads, max_ctx_chunks, chunk_size*head_dim]

        # [x0, x1, ...., self.chunks*chunk_size, local_chunk, rest]
        self.chunks = incoming // self.chunk_size - self.local_chunk 
        # ensure self.chunks is even
        self.chunks = self.chunks - self.chunks % 8
        
        # store Post-RoPE k cache <prefill_local> to the cache
        self.prefill_local = incoming - self.chunks * self.chunk_size # local chunks + align to chunk_size
        self.k_cache_buffer[layer_idx][self.prefilled_batch:self.prefilled_batch + bsz, :, :self.prefill_local].copy_(key_states_roped[:, :, -self.prefill_local:])
        self.v_cache_buffer[layer_idx][self.prefilled_batch:self.prefilled_batch + bsz, :, :self.prefill_local].copy_(new_v_cache[:, :, -self.prefill_local:])

        key_states_roped_ctx = key_states_roped[:,:,:self.chunks*self.chunk_size].view(bsz, self.num_key_value_heads, self.chunks, self.chunk_size, self.head_dim)
        landmark_candidates = key_states_roped_ctx.mean(dim=-2) # [bsz, kv_heads, chunks, head_dim]
        
        # compute the cos similarity between it and the original key cache
        cos_sim = torch.nn.functional.cosine_similarity(landmark_candidates.unsqueeze(3).expand(-1, -1, -1, self.chunk_size, -1), key_states_roped_ctx, dim=-1) # [bsz, kv_heads, chunks, chunk_size]
        
        # get the outlier_chunk idx for each head # [bsz, kv_heads, outlier_chunk]
        outlier_chunk_idx = cos_sim.min(dim=-1).values.topk(self.outlier_chunk, largest=False).indices
    
        # [bsz, kv_heads, chunks, chunk_size, head_dim] --gather[bsz, kv_heads, outlier_chunk]-->[bsz, kv_heads, outlier_chunk, chunk_size, head_dim]
        outlier_chunk_k_cache = key_states_roped_ctx.gather(dim=2, index=outlier_chunk_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, self.chunk_size, self.head_dim)).view(bsz, self.num_key_value_heads, self.outlier_chunk*self.chunk_size, self.head_dim)
        
        outlier_chunk_v_cache = new_v_cache[:,:,:self.chunks*self.chunk_size].view(bsz, self.num_key_value_heads, self.chunks, self.chunk_size, self.head_dim).gather(dim=2, index=outlier_chunk_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, self.chunk_size, self.head_dim)).view(bsz, self.num_key_value_heads, self.outlier_chunk*self.chunk_size, self.head_dim)

        self.sparse_start = self.prefill_local + self.outlier_chunk*self.chunk_size
        self.sparse_end = self.prefill_local + self.outlier_chunk*self.chunk_size + self.sparse_budget

        self.kernel_offset = self.sparse_start * self.head_dim
        
        # store outlier_chunk to the cache
        self.k_cache_buffer[layer_idx][self.prefilled_batch:self.prefilled_batch + bsz, :, self.prefill_local:self.sparse_start].copy_(outlier_chunk_k_cache)
        self.v_cache_buffer[layer_idx][self.prefilled_batch:self.prefilled_batch + bsz, :, self.prefill_local:self.sparse_start].copy_(outlier_chunk_v_cache)

        # filter landmark_candidates using outlier_chunk and register the rest to k_landmark
        # [bsz, kv_heads, chunks, head_dim] --> [bsz, kv_heads, chunks - outlier_chunk, head_dim]
        # get rest_idx: [bsz, kv_heads, chunks] --filter--> [bsz, kv_heads, chunks - outlier_chunk]
        all_idx = torch.arange(self.chunks, device=key_states_roped.device).unsqueeze(0).unsqueeze(0).expand(bsz, self.num_key_value_heads, -1) # [bsz, kv_heads, chunks]
        mask = torch.ones_like(all_idx, dtype=torch.bool)
        mask.scatter_(dim=-1, index=outlier_chunk_idx, value=False)
        rest_idx = all_idx.masked_select(mask).view(bsz, self.num_key_value_heads, -1)

        # register rest_idxed landmarks to k_landmark
        self.register_k_landmark(landmark_candidates.gather(dim=2, index=rest_idx.unsqueeze(-1).expand(-1, -1, -1, self.head_dim)).view(bsz, self.num_key_value_heads, -1, self.head_dim), rest_idx, layer_idx)

        # fill cache for the first time
        chunk_attn = torch.einsum('bhgd,bhcd->bhgc', last_query_states.view(-1, self.num_key_value_heads, self.num_key_value_groups, self.head_dim), self.k_landmark[layer_idx][self.prefilled_batch:self.prefilled_batch + bsz].to(last_query_states.device)) / math.sqrt(128) # [bsz, 8, 4, chunks]
        chunk_attn = nn.functional.softmax(chunk_attn, dim=-1, dtype=torch.float32).to(self.dtype)
        chunk_attn, _ = torch.max(chunk_attn, dim=-2) # [bsz, 8, chunks]
        merged_results = torch.topk(chunk_attn, k=self.select_sets, dim=-1).indices # [bsz, 8, select_sets(256)]
        selected_chunks = self.k_landmark_idx[layer_idx][self.prefilled_batch:self.prefilled_batch + bsz].to(last_query_states.device).gather(dim=-1, index=merged_results) # [bsz, 8, select_sets]
        self.position_ids[layer_idx][self.prefilled_batch:self.prefilled_batch + bsz].copy_(selected_chunks)
        assert self.position_ids[layer_idx][self.prefilled_batch:self.prefilled_batch + bsz].max() < self.chunks, f"position_ids exceed the max_length {self.position_ids[layer_idx].max()}"
        assert self.position_ids[layer_idx][self.prefilled_batch:self.prefilled_batch + bsz].min() >= 0, f"position_ids exceed the min_length {self.position_ids[layer_idx].min()}"
        position_ids = (selected_chunks.unsqueeze(-1) * self.chunk_size + torch.arange(self.chunk_size, device=chunk_attn.device).unsqueeze(0).unsqueeze(0).unsqueeze(0)).view(bsz, self.num_key_value_heads, -1)
        value_ = new_v_cache.gather(dim=-2, index=position_ids.unsqueeze(-1).expand(-1, -1, -1, self.head_dim))
        self.v_cache_buffer[layer_idx][self.prefilled_batch:self.prefilled_batch + bsz, :, self.sparse_start:self.sparse_end].copy_(value_, non_blocking=True)
        key_ = key_states_roped.gather(dim=-2, index=position_ids.unsqueeze(-1).expand(-1, -1, -1, self.head_dim))
        self.k_cache_buffer[layer_idx][self.prefilled_batch:self.prefilled_batch + bsz, :, self.sparse_start:self.sparse_end].copy_(key_, non_blocking=True)

        if layer_idx == self.num_layers - 1:
            assert self.sparse_budget < incoming
            # self.kv_offset += incoming
            self.prefilled_batch += bsz

            if self.prefilled_batch == self.batch_size:
                self.kv_offset += incoming

                assert torch.any(self.position_ids == -1) == False, f"The cache for offloading is not built correctly, {self.position_ids}"

    ##### Decoding #####
    def get_retrieval_position_ids(self, layer_idx, query_states):
        # self.k_landmark[layer_idx][:, :, :self.chunks] is [bsz, 8, chunks, head_dim]
        # chunk_attn: [bsz, 32, window_size, chunks]
        self.incoming_q_len = query_states.shape[-2] # 1

        # gemm_softmax
        kernels.batch_gemm_softmax(
            query_states.contiguous(),
            self.k_landmark[layer_idx].contiguous(),
            self.gemm_o,
            self.norm,
            self.sum,
            self.softmax_o,
            self.batch_size * self.num_key_value_heads,
            self.num_key_value_groups * self.incoming_q_len,
            self.k_landmark[layer_idx].shape[-2],
            self.head_dim,
            1 / math.sqrt(128),
            0
        )
        if self.num_key_value_groups > 1:
            chunk_attn, _ = torch.max(self.softmax_o.view(self.batch_size, self.num_key_value_heads, self.num_key_value_groups, -1), dim=-2) # [bsz, 8, chunks]

        # [bsz, 8, seq] --> [bsz, 8, select_sets(256)]
        merged_results = torch.topk(chunk_attn.view(self.batch_size, self.num_key_value_heads, -1), k=self.select_sets, dim=-1).indices
        # use merged_results to gather the position_ids: [bsz, 8, select_sets] --> [bsz, 8, select_sets]
        selected_chunks = self.k_landmark_idx[layer_idx].gather(dim=-1, index=merged_results) # [bsz, 8, select_sets]
        kernels.reorder_keys_and_compute_offsets(self.position_ids[layer_idx], selected_chunks, self.offsets, self.cnts, self.batch_size, self.num_key_value_heads, self.select_sets)

        return self.position_ids[layer_idx]

    def get_value_cache(self, layer_idx, position_ids):

        kernels.gather_copy_with_offsets(self.v_cache_cpu[layer_idx], self.v_cache_buffer[layer_idx], self.temp, self.offsets, self.cnts, self.signals, self.batch_size, self.num_key_value_heads, int(self.max_ctx_chunks_len*self.head_dim), int(self.sparse_budget*self.head_dim), self.kernel_offset, self.kernel_stride, self.select_sets)

        gen_offset = self.gen_offset if layer_idx == self.num_layers - 1 else self.gen_offset + self.incoming_q_len

        return self.v_cache_buffer[layer_idx][:, :, :self.sparse_end + gen_offset]

    def get_key_cache(self, layer_idx, position_ids, rope_func, cos_sin_cache):

        # gather key cache and rope them
        u = self.U[layer_idx] # [bsz, 128k, rank]
        sv = self.SV[layer_idx] # [bsz, 8, 128, rank]

        kernels.gather_copy_d2d_with_offsets(self.k_cache_buffer[layer_idx], self.offsets, self.cnts, self.batch_size, self.num_key_value_heads, int(self.sparse_budget*self.head_dim), self.kernel_offset, self.kernel_stride, self.select_sets)
        batch_gather_gemm_rotary_pos_emb_cuda(u, sv, cos_sin_cache, position_ids, self.output, self.chunk_size, self.k_cache_buffer[layer_idx], self.sparse_start, self.sparse_end, self.cnts)

        gen_offset = self.gen_offset if layer_idx == self.num_layers - 1 else self.gen_offset + self.incoming_q_len

        return self.k_cache_buffer[layer_idx][:, :, :self.sparse_end + gen_offset]

    def H2D(self):
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        self.SV = self.SV.to(self.device)
        self.U = self.U.to(self.device)
        self.k_landmark = self.k_landmark.to(self.device)
        self.k_landmark_idx = self.k_landmark_idx.to(self.device)

        self.gemm_o = self.gemm_o.to(self.device)
        self.softmax_o = self.softmax_o.to(self.device)
        self.norm = self.norm.to(self.device)
        self.sum = self.sum.to(self.device)

        self.temp = self.temp.to(self.device)
        self.output = self.output.to(self.device)

        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    def update_kv_cache(self,
            new_k_cache :torch.Tensor,
            new_v_cache :torch.Tensor,
            layer_idx :int,
            ):

        incoming = new_k_cache.shape[-2]
        needed = self.sparse_end + self.gen_offset + incoming
        if needed > self.k_cache_buffer.shape[-2]:
            self._extend_generation_buffers(needed - self.k_cache_buffer.shape[-2])

        self.v_cache_buffer[layer_idx][:, :, self.sparse_end+self.gen_offset:self.sparse_end+self.gen_offset+incoming].copy_(new_v_cache, non_blocking=True)
        self.k_cache_buffer[layer_idx][:, :, self.sparse_end+self.gen_offset:self.sparse_end+self.gen_offset+incoming].copy_(new_k_cache, non_blocking=True)

        if layer_idx == self.num_layers - 1:
            self.kv_offset += incoming
            self.gen_offset += incoming

    def clear(self):
        self.v_cache_cpu = None
        self.k_cache_buffer = None
        self.v_cache_buffer = None
        self.temp = None
        self.offsets = None
        self.cnts = None
        self.signals = None
        self.position_ids = None
        self.output = None

        self.k_landmark = None
        self.k_landmark_idx = None
        self.U = None
        self.SV = None

        self.sparse_budget = None
        self.local_chunk = None
        self.outlier_chunk = None
        self.select_sets = None

        self.kv_offset = 0
        self.prefill = 0
        self.gen_offset = 0
        self.prefill_local = 0
        self.prefilled_batch = 0

    def get_kv_len(self):
        return self.kv_offset
