
import re

from shadowkv_models import Qwen3, Qwen3Moe
from .base import BaseModel
from .huggingface_above_v4_33 import _convert_chat_messages
from opencompass.utils.logging import get_logger

config = {
    "60k": {
        "sparse_budget": 1024,
        "min_prompt_len": 1024*60,
        "baseline_bsz": 8,
        "shadowkv_bsz": 48,
    },
    "122k": {
        "sparse_budget": 2048,
        "min_prompt_len": 1024*122,
        "baseline_bsz": 4,
        "shadowkv_bsz": 24,
    },
    "244k": {
        "sparse_budget": 4096,
        "min_prompt_len": 1024*244,
        "baseline_bsz": 2,
        "shadowkv_bsz": 12,
    }
}


class Qwen3ShadowKV(BaseModel):
    def __init__(
        self,
        path,
        sparse_budget,
        local_budget=32,
        outlier_budget=384,
        max_length=256 * 1024,
        chunk_size=8,
        rank=160,
        batch_size=1,
        keys_on_device: bool = True,
        # Quantization parameters
        use_nvfp4_quantization: bool = False,
        nvfp4_channel_size: int = 1024,
        use_higgs_quantization: bool = False,
        higgs_hadamard_groupsize: int = 128,
        higgs_edenn_d: int = 16,
        higgs_edenn_n: int = 256,
        higgs_channel_size: int = 1024,
        higgs_chunk_size: int = 64,
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
        # Residual quantization
        use_residual_quantization: bool = False,
        meta_chunk_size: int = 1,
        residual_higgs_edenn_d: int = 16, 
        residual_higgs_edenn_n: int = 256,
    ):
        super().__init__(path, max_seq_len=max_length)
        model_class = Qwen3Moe if re.search(r'A\d+B', path) else Qwen3
        self.model = model_class(model_name=path,
                                 batch_size=batch_size,
                                 sparse_budget=sparse_budget,
                                 local_budget=local_budget,
                                 outlier_budget=outlier_budget,
                                 keys_on_device=keys_on_device,
                                 attn_mode="shadowkv",
                                 max_length=max_length,
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
                                 use_residual_quantization=use_residual_quantization,
                                 meta_chunk_size=meta_chunk_size,
                                 residual_higgs_edenn_d=residual_higgs_edenn_d,
                                 residual_higgs_edenn_n=residual_higgs_edenn_n,
                                 )
        self.model.tokenizer.pad_token = self.model.tokenizer.eos_token
        self.logger = get_logger()

    def generate(self, inputs: list[str], max_out_len: int) -> list[str]:
        messages = _convert_chat_messages(inputs)
        batch_size = len(messages)
        assert batch_size == 1

        tokenize_kwargs = dict(
            return_tensors='pt',
            padding=True,
            truncation=True,
            add_special_tokens=False,
            max_length=self.max_seq_len
        )
        messages = [self.model.tokenizer.apply_chat_template(m,
                                                             add_generation_prompt=True,
                                                             tokenize=False,
                                                             enable_thinking=False) for m in messages]
        input_ids = self.model.tokenizer(messages, **tokenize_kwargs).input_ids.to(self.model.device)
        self.logger.info(input_ids.shape)

        return self.model.generate(input_ids, gen_len=max_out_len)


    def get_token_len(self, prompt: str) -> int:
        m = _convert_chat_messages([prompt])[0]
        t = self.model.tokenizer.apply_chat_template(m, add_generation_prompt=True, return_dict=True)
        return len(t['input_ids'])
