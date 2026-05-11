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

# Base LLM class
import typing as tp

import torch
import torch.nn.functional as F
import time
import gc
from tqdm import tqdm

from flash_attn import flash_attn_with_kvcache

from .tensor_op import sample_token, layer_norm, minference_prefill_kernel
from .kv_cache import KV_Cache, ShadowKVCache, ShadowKVCache_CPU

class LLM:

    def _init_shadowkv_params(self,
                              chunk_size: int,
                              rank: int,
                              sparse_budget,
                              local_budget,
                              outlier_budget,
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
                              values_higgs_chunk_size: int = 64
                              ):
        self.chunk_size = chunk_size
        self.rank = rank
        self.sparse_budget = sparse_budget
        self.local_budget = local_budget
        self.outlier_budget = outlier_budget
        self.keys_on_device = keys_on_device
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
        self.keys_fp8 = keys_fp8
        self.keys_nvfp4 = keys_nvfp4
        self.keys_higgs = keys_higgs
        self.values_fp8 = values_fp8
        self.values_nvfp4 = values_nvfp4
        self.values_higgs = values_higgs
        self.keys_higgs_hadamard_groupsize = keys_higgs_hadamard_groupsize
        self.keys_higgs_edenn_d = keys_higgs_edenn_d
        self.keys_higgs_edenn_n = keys_higgs_edenn_n
        self.keys_higgs_channel_size = keys_higgs_channel_size
        self.keys_higgs_chunk_size = keys_higgs_chunk_size
        self.values_higgs_hadamard_groupsize = values_higgs_hadamard_groupsize
        self.values_higgs_edenn_d = values_higgs_edenn_d
        self.values_higgs_edenn_n = values_higgs_edenn_n
        self.values_higgs_channel_size = values_higgs_channel_size
        self.values_higgs_chunk_size = values_higgs_chunk_size

    def __str__(self) -> str:
        gpu_mem = f"{round(torch.cuda.memory_allocated(self.device) / 1024**3, 2)} GB / {round(torch.cuda.get_device_properties(self.device).total_memory / 1024**3, 2)} GB"
        return f"LLM: {self.model_name}, attn_mode: {self.attn_mode}, max_length: {self.max_length}, batch_size: {self.batch_size}, device: {self.device}, dtype: {self.dtype}, GPU mem: {gpu_mem}"

    def init_kv_cache(
        self,
        sparse_budget: tp.Union[int, float],
        local_budget: tp.Union[int, float],
        outlier_budget: tp.Union[int, float],
        rank: int,
        chunk_size: int,
        config,
        prompt_length: int = 0,
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
    ):
        if type(sparse_budget) == float:
            sparse_budget = int(sparse_budget * prompt_length) // chunk_size * chunk_size
        if type(local_budget) == float:
             local_budget = int(local_budget * prompt_length) // chunk_size * chunk_size
        if type(outlier_budget) == float:
             outlier_budget = int(outlier_budget * prompt_length) // chunk_size * chunk_size
        if hasattr(self, "kv_cache"):
            del self.kv_cache
        if self.attn_mode == 'full':
            self.kv_cache = KV_Cache(
                config,
                max_length=self.max_length,
                device=self.device,
                dtype=self.dtype,
                batch_size=self.batch_size,
            )
        elif self.attn_mode.lower() == 'shadowkv':
            self.kv_cache = ShadowKVCache(
                config,
                max_length=self.max_length,
                device=self.device,
                dtype=self.dtype,
                batch_size=self.batch_size,
                sparse_budget=sparse_budget,
                local_budget=local_budget,
                outlier_budget=outlier_budget,
                keys_on_device=keys_on_device,
                rank=rank,
                chunk_size=chunk_size,
                use_nvfp4_quantization=use_nvfp4_quantization,
                nvfp4_channel_size=nvfp4_channel_size,
                use_higgs_quantization=use_higgs_quantization,
                higgs_hadamard_groupsize=higgs_hadamard_groupsize,
                higgs_edenn_d=higgs_edenn_d,
                higgs_edenn_n=higgs_edenn_n,
                higgs_channel_size=higgs_channel_size,
                higgs_chunk_size=higgs_chunk_size,
                use_residual_quantization=use_residual_quantization,
                meta_chunk_size=meta_chunk_size,
                residual_higgs_edenn_d=residual_higgs_edenn_d,
                residual_higgs_edenn_n=residual_higgs_edenn_n,
                keys_fp8=keys_fp8,
                keys_nvfp4=keys_nvfp4,
                keys_higgs=keys_higgs,
                values_fp8=values_fp8,
                values_nvfp4=values_nvfp4,
                values_higgs=values_higgs,
                keys_higgs_hadamard_groupsize=keys_higgs_hadamard_groupsize,
                keys_higgs_edenn_d=keys_higgs_edenn_d,
                keys_higgs_edenn_n=keys_higgs_edenn_n,
                keys_higgs_channel_size=keys_higgs_channel_size,
                keys_higgs_chunk_size=keys_higgs_chunk_size,
                values_higgs_hadamard_groupsize=values_higgs_hadamard_groupsize,
                values_higgs_edenn_d=values_higgs_edenn_d,
                values_higgs_edenn_n=values_higgs_edenn_n,
                values_higgs_channel_size=values_higgs_channel_size,
                values_higgs_chunk_size=values_higgs_chunk_size,
            )
        elif self.attn_mode.lower() == 'shadowkv_cpu':
            raise NotImplementedError("bugged implementation. Check https://github.com/ByteDance-Seed/ShadowKV/issues/13#issuecomment-3274036980 for more details")
            self.kv_cache = ShadowKVCache_CPU(
                config,
                device=self.device,
                dtype=self.dtype,
                batch_size=self.batch_size,
                sparse_budget_ratio=sparse_budget,
                local_budget=local_budget,
                outlier_budget=outlier_budget,
                rank=rank,
                chunk_size=chunk_size,
            )
        else:
            raise ValueError(f"Invalid attention mode {self.attn_mode}")

    def print_kv_stats(self):
        self.kv_cache.print_stats()
    
    def get_ctx(self, input_ids: torch.LongTensor):
        input_len = input_ids.size(1)
        past_len = self.kv_cache.get_kv_len()
        position_ids = torch.arange(past_len, past_len + input_len, device=self.device, dtype=torch.long).unsqueeze(0).repeat(input_ids.size(0), 1)
        return position_ids

    @torch.inference_mode()
    def inference(self,
            input_ids: torch.LongTensor,
            position_ids: torch.LongTensor):

        hidden_states = F.embedding(input_ids, self.embed_tokens)

        for idx in range(self.num_layers):
            hidden_states = self.layer_compute(self.layers[idx], idx, hidden_states, position_ids)
        
        hidden_states = layer_norm(hidden_states, w=self.norm_weight, eps=self.norm_variance_epsilon)
        
        if hidden_states.shape[1] > 16: # prefill
            hidden_states = hidden_states[:, -1:, :]
        logits = F.linear(hidden_states, self.lm_head).float()
        
        return logits

    @torch.inference_mode()
    def prefill(self, input_ids: torch.LongTensor):
        prompt_length = input_ids.shape[-1]
        self.init_kv_cache(
            self.sparse_budget,
            self.local_budget,
            self.outlier_budget,
            self.rank,
            self.chunk_size,
            self.config,
            prompt_length,
            keys_on_device=self.keys_on_device,
            use_nvfp4_quantization=self.use_nvfp4_quantization,
            nvfp4_channel_size=self.nvfp4_channel_size,
            use_higgs_quantization=self.use_higgs_quantization,
            higgs_hadamard_groupsize=self.higgs_hadamard_groupsize,
            higgs_edenn_d=self.higgs_edenn_d,
            higgs_edenn_n=self.higgs_edenn_n,
            higgs_channel_size=self.higgs_channel_size,
            higgs_chunk_size=self.higgs_chunk_size,
            use_residual_quantization=self.use_residual_quantization,
            meta_chunk_size=self.meta_chunk_size,
            residual_higgs_edenn_d=self.residual_higgs_edenn_d,
            residual_higgs_edenn_n=self.residual_higgs_edenn_n,
            keys_fp8=self.keys_fp8,
            keys_nvfp4=self.keys_nvfp4,
            keys_higgs=self.keys_higgs,
            values_fp8=self.values_fp8,
            values_nvfp4=self.values_nvfp4,
            values_higgs=self.values_higgs,
            keys_higgs_hadamard_groupsize=self.keys_higgs_hadamard_groupsize,
            keys_higgs_edenn_d=self.keys_higgs_edenn_d,
            keys_higgs_edenn_n=self.keys_higgs_edenn_n,
            keys_higgs_channel_size=self.keys_higgs_channel_size,
            keys_higgs_chunk_size=self.keys_higgs_chunk_size,
            values_higgs_hadamard_groupsize=self.values_higgs_hadamard_groupsize,
            values_higgs_edenn_d=self.values_higgs_edenn_d,
            values_higgs_edenn_n=self.values_higgs_edenn_n,
            values_higgs_channel_size=self.values_higgs_channel_size,
            values_higgs_chunk_size=self.values_higgs_chunk_size,
        )
        logits = self.inference(input_ids=input_ids, position_ids=self.get_ctx(input_ids))

        assert self.kv_cache.get_kv_len() == prompt_length, f"KV length mismatch, got {self.kv_cache.get_kv_len()}, expected {input_ids.shape[-1]}"
        return logits


    @torch.inference_mode()
    def prefill_cont(self, input_ids: torch.LongTensor):
        logits = self.inference(input_ids=input_ids, position_ids=self.get_ctx(input_ids))
        return logits
    
    def encode(self, text: str, template=None, truncation=False):
        if template == 'chat':
            text = self.chat_template.format(msg=text)
            input_ids = self.tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids.to(self.device)
            if self.tokenizer.bos_token_id is not None:
                assert self.tokenizer.bos_token_id not in input_ids, f"bos_token_id found in input_ids"
            return input_ids
        if template == 'ctx':
            text = self.ctx_template.format(ctx=text)
        if template == 'prefix':
            text = self.prefix_template.format(ctx=text)
        input_ids = self.tokenizer(text, return_tensors="pt", truncation=truncation).input_ids.to(self.device)
        return input_ids

    @torch.inference_mode()
    def layer_compute(self, 
            buffer,
            layer_idx :int, 
            hidden_states: torch.FloatTensor, 
            position_ids: torch.LongTensor):

        residual = hidden_states
        bsz, q_len, _ = hidden_states.size()
        query_states, key_states, value_states = self.pre_attention_compute(
            hidden_states,
            buffer,
            self.num_heads,
            self.num_key_value_heads,
            self.head_dim
        )
        
        if isinstance(self.kv_cache, KV_Cache):
            query_states, key_states = self.apply_rotary_pos_emb(query_states, key_states, position_ids)
            key_states, value_states = self.kv_cache.update_kv_cache(key_states, value_states, layer_idx)
            
            if self.minference == True and q_len > 1:
                hidden_states = minference_prefill_kernel(query_states=query_states, key_states=key_states, value_states=value_states, minference_parttern=self.minference_parttern[layer_idx])
            else:
                hidden_states = flash_attn_with_kvcache(q=query_states.transpose(1, 2), k_cache=key_states.transpose(1, 2), v_cache=value_states.transpose(1, 2), causal=True)

        elif isinstance(self.kv_cache, ShadowKVCache) or isinstance(self.kv_cache, ShadowKVCache_CPU):

            if q_len > 1: # prefill
                # svd unrope key and save
                self.kv_cache.get_svd(key_states, layer_idx=layer_idx)
                query_states, key_states = self.apply_rotary_pos_emb(query_states, key_states, position_ids)
                self.kv_cache.prefill_kv_cache(value_states, layer_idx, key_states, query_states[:, :, -1:])
                
                if self.minference == True:
                    hidden_states = minference_prefill_kernel(query_states=query_states, key_states=key_states, value_states=value_states, minference_parttern=self.minference_parttern[layer_idx])
                else:
                    hidden_states = flash_attn_with_kvcache(q=query_states.transpose(1, 2), k_cache=key_states.transpose(1, 2), v_cache=value_states.transpose(1, 2), causal=True)

            else: # decode
                # rope query and key
                query_states, key_states = self.apply_rotary_pos_emb(query_states, key_states, position_ids)

                # update kv cache to buffer
                self.kv_cache.update_kv_cache(key_states, value_states, layer_idx)

                # get retrieval idx
                position_ids = self.kv_cache.get_retrieval_position_ids(layer_idx=layer_idx, query_states=query_states).to(self.device)

                # multi-stream
                curr_stream = torch.cuda.current_stream()
                get_value_stream = self.kv_cache.copy_stream

                with torch.cuda.stream(get_value_stream):
                    get_value_stream.wait_stream(curr_stream)
                    value_states = self.kv_cache.get_value_cache(layer_idx, position_ids).to(self.device)

                # gather key cache from GPU and RoPE it (should be hide by CPU offloading time)
                key_states = self.kv_cache.get_key_cache(layer_idx=layer_idx, position_ids=position_ids, rope_func=self.apply_rotary_pos_emb_single, cos_sin_cache=self.cos_sin_cache).to(self.device)

                curr_stream.wait_stream(get_value_stream)

                # flash attention
                hidden_states = flash_attn_with_kvcache(q=query_states.transpose(1, 2), k_cache=key_states.transpose(1, 2), v_cache=value_states.transpose(1, 2), causal=True)

        else:
            raise ValueError(f"Invalid attention mode {self.attn_mode}")

        hidden_states = hidden_states.reshape(bsz, q_len, -1)  # not self.hidden_size possibly
        # if hidden_states.size()[-1] != 4096:
        #     breakpoint()
        
        if bsz*q_len > 64*1024: # [bsz, seq, 128]
            output = torch.empty(bsz, q_len, self.hidden_size,
                                 device=hidden_states.device,
                                 dtype=hidden_states.dtype)
            prop_iter = bsz * q_len // (8*1024)
            prefill_chunk_size = bsz * q_len // prop_iter
            prefill_iter = (q_len + prefill_chunk_size - 1) // prefill_chunk_size
            for i in range(prefill_iter):
                start = i*prefill_chunk_size
                end = (i+1)*prefill_chunk_size
                tmp = self.post_attention_compute(hidden_states[:, start:end], residual[:, start:end], buffer)
                # if tmp.size(-1) != 4096:
                #     breakpoint()
                output[:, start:end] = tmp
            hidden_states = output

        else:
            hidden_states = self.post_attention_compute(hidden_states, residual, buffer)
        
        return hidden_states

    def decode(self, input_ids: torch.Tensor, skip_special_tokens: bool = False):
        return self.tokenizer.batch_decode(input_ids, skip_special_tokens=skip_special_tokens)

    @torch.inference_mode()
    def generate(self, input_ids: torch.Tensor, gen_len: int = 256, temperature: float = 0.0, top_p: float = 0.9, top_k :int = 50, verbose: bool = False, benchmark: bool = False, cont: bool = False):
        """accuracy eval usage, not for throughput eval"""
        assert type(input_ids) == torch.Tensor, f"input_ids must be a torch.Tensor, got {type(input_ids)}"

        # prefill
        if cont == False:
            if input_ids.size(1) > self.max_length:
                raise ValueError(f"Input length must be less than {self.max_length}, but got {input_ids.size(1)}")
            logits = self.prefill(input_ids)
        else:
            raise NotImplementedError("not tested with dynamic batch size")
            if input_ids.size(1) + self.kv_cache.get_kv_len() >= self.max_length:
                raise ValueError(f"Input length must be less than {self.max_length}, but got {input_ids.size(1)}")
            logits = self.prefill_cont(input_ids)
        next_token = sample_token(logits[:, -1, :], temperature=temperature, top_p=top_p, top_k=top_k)
        
        n = 0
        pos = 0
        generated_ids = []
        generated_ids.extend(next_token[0].tolist())
        
        self.kv_cache.H2D()

        if benchmark == True:
            start = time.time()
        
        while n < gen_len:
            logits = self.inference(input_ids=next_token, position_ids=self.get_ctx(next_token))
            next_token = sample_token(logits[:, -1, :], temperature=temperature, top_p=top_p, top_k=top_k)
            
            n += 1
            generated_ids.extend(next_token[0].tolist())
            if verbose == True:
                generated_text = (
                    self.tokenizer.decode(
                        generated_ids,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=True,
                        spaces_between_special_tokens=False,
                    ).strip().split(" ")
                )
                now = len(generated_text) - 1
                if now > pos:
                    print(" ".join(generated_text[pos:now]), end=" ", flush=True)
                    pos = now

            if next_token[0] == self.tokenizer.eos_token_id:
                break
            if self.tokenizer.decode(next_token[0]) == "<|eot_id|>": # llama-3
                break
            if self.tokenizer.decode(next_token[0]) == "<|im_end|>": # yi
                break
            if next_token[0] in [151329, 151336, 151338]: # glm
                break
            if self.tokenizer.decode(next_token[0]) == "<|endoftext|>": # glm
                break
            if self.tokenizer.decode(next_token[0]) == "<|end|>": # phi
                break

        if verbose == True and n!=0:
            print(" ".join(generated_text[pos:]), end=" ", flush=True)
        if benchmark == True:
            end = time.time()
            print(f"\nPrefill {input_ids.size(1)} tokens | Generate {n} tokens in {round(end - start, 2)}s, {round(n / (end - start), 2)} tokens/s | cached {self.kv_cache.get_kv_len()}\n")

        # feed new token to the model
        self.inference(input_ids=next_token, position_ids=self.get_ctx(next_token))

        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        return [self.tokenizer.decode(generated_ids, skip_special_tokens=True)]
    
    @torch.inference_mode()
    def batch_prefill(self, input_ids: torch.Tensor, benchmark: bool = False):
        self.kv_cache.clear()
        batch_size = input_ids.size(0)
        
        assert batch_size == self.batch_size, f"batch_size mismatch, got {batch_size}, expected {self.batch_size}"
        
        if input_ids.size(1) > self.max_length:
                raise ValueError(f"Input length must be less than {self.max_length}, but got {input_ids.size(1)}")
        
        logits = torch.zeros(batch_size, 1, self.vocab_size, device=self.device, dtype=torch.float32)

        if input_ids.shape[-1] > 120*1024 and input_ids.shape[-1] < 200*1024:
            T = 8
        else:
            T = 4
        # for bsz in range(0, batch_size, T):
        for bsz in tqdm(range(0, batch_size, T), desc=f"Prefilling (batch size={batch_size})"):
            req_input_ids = input_ids[bsz:bsz+T]
            logits[bsz:bsz+T].copy_(self.inference(input_ids=req_input_ids, position_ids=self.get_ctx(req_input_ids)))
        assert self.kv_cache.get_kv_len() == input_ids.shape[-1], f"KV length mismatch, got {self.kv_cache.get_kv_len()}, expected {input_ids.shape[-1]}"

        return logits


    @torch.inference_mode()
    def warmup(self):

        a = torch.randn(self.batch_size, 1024, 1024).to(self.dtype).to(self.device)
        b = torch.randn(self.batch_size, 1024, 1024).to(self.dtype).to(self.device)
        for _ in range(100):
            torch.bmm(a, b)
        del a, b

        print("Warmup done")

    @torch.inference_mode()
    def batch_generate(self, input_ids: torch.Tensor, gen_len: int = 256, temperature: float = 0.0, top_p: float = -1, top_k :int = 50, verbose: bool = False, benchmark: bool = False, cont: bool = False):
        """throughput eval usage"""
        raise NotImplementedError("not tested with dynamic sparse budget")
        assert type(input_ids) == torch.Tensor, f"input_ids must be a torch.Tensor, got {type(input_ids)}"

        # prefill
        if cont == False:
            if input_ids.size(1) > self.max_length:
                raise ValueError(f"Input length must be less than {self.max_length}, but got {input_ids.size(1)}")
            logits = self.batch_prefill(input_ids)
        else:
            logits = self.prefill_cont(input_ids)
        
        next_token = sample_token(logits[:, -1, :], temperature=temperature, top_p=top_p, top_k=top_k)
        
        n = 0
        generated_ids = []
        generated_ids.append(next_token[:, -1].tolist())
        
        self.kv_cache.H2D()
        self.warmup()

        if benchmark == True:
            start = time.time()
        
        while n < gen_len:
            logits = self.inference(input_ids=next_token, position_ids=self.get_ctx(next_token))
            next_token = sample_token(logits[:, -1, :], temperature=temperature, top_p=top_p, top_k=top_k)
            
            n += 1
            generated_ids.append(next_token[:, -1].tolist())

        if benchmark == True:
            end = time.time()
            print(f"\nPrefill {input_ids.size(1)} tokens | Generate {n} tokens in {round(end - start, 2)}s | Throughput: {round(self.batch_size * n / (end - start), 2)} tokens/s, Latency: {round((end - start)*1000 / n, 2)} ms/step | cached {self.kv_cache.get_kv_len()}\n")

        # feed new token to the model
        self.inference(input_ids=next_token, position_ids=self.get_ctx(next_token))

        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        generated_ids = torch.LongTensor(generated_ids).t().tolist()

        if benchmark == True:
            return self.decode(generated_ids, skip_special_tokens=True), self.batch_size * n / (end - start)

        return self.decode(generated_ids, skip_special_tokens=True)
