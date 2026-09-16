import math
import torch
from dataclasses import dataclass

from minisgl.models.config import ModelConfig
from minisgl.distributed import get_tp_info
from minisgl.utils import div_even, init_logger, print_object_cuda_tensors_capacity

from minisgl.kernel.shadowkv import (
    shadowkv_gather_scales_kernel,
    shadowkv_score_landmarks_kernel_hd128,
    shadowkv_topk_kernel,
)

from minisgl.kernel import store_cache

from minisgl.shadowkv_kernels import (
    fill_prefill_metadata,
    fill_decode_metadata,
    gather_kv_cache,
    map_to_gpu,
)

from minisgl.quantization.higgs import (
    QuantizedTensor,
    higgs_quantize_heads,
    higgs_score,
    higgs_dequantize_heads,
    HiggsQuantizationMeta,
)

logger = init_logger(__name__)

DTYPE_MAP = {
    "bf16": torch.bfloat16,
    "fp8": torch.float8_e4m3fn,
}


@dataclass(frozen=False)
class ShadowKVConfig:
    enabled: bool = False
    chunk_size: int = 8
    prefix_budget: float = 0.006125
    sparse_budget: float = 0.125
    suffix_budget: float = 0.06125
    total_budget: float = 0.0
    min_seqlen_to_prune: int = 512
    quantize_landmarks: bool = False
    quantize_kvcache: bool = False
    enable_offloading: bool = False
    # kv_cache_dtype: str | torch.dtype = "bf16"
    # landmarks_dtype: str | torch.dtype = "bf16"
    # use_naive_topk: bool = False

    def __post_init__(self):
        self.total_budget = self.prefix_budget + self.sparse_budget + self.suffix_budget
        assert self.total_budget <= 1.0, "Total Budget cannot be greater than 100%"

        if self.enabled:
            logger.info(f"INITTED shadow kv with {self}")

    @classmethod
    def from_dict(cls, config: dict):
        return cls(**config)


class ShadowKVPool:
    def __init__(
        self,
        config: ShadowKVConfig,
        model_config: ModelConfig,
        max_batch_size: int,
        max_seq_len: int,
        device,
        dtype,
    ):
        if not config.enabled:
            raise RuntimeError("INITTING ShadowKV pool with shadowkv disabled is prohibited!")

        self.config: ShadowKVConfig = config
        self.model_config: ModelConfig = model_config
        self.device = device
        self.dtype = dtype

        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len

        assert (self.model_config.num_qo_heads % self.model_config.num_kv_heads) == 0
        self.gqa_factor = self.model_config.num_qo_heads // self.model_config.num_kv_heads

        tp_info = get_tp_info()
        self.local_kv_heads = div_even(
            model_config.num_kv_heads, tp_info.size, allow_replicate=True
        )
        self.local_qo_heads = self.local_kv_heads * self.gqa_factor

        self.max_num_landmarks = max_seq_len // config.chunk_size
        self.head_dim = model_config.head_dim

        if not self.config.quantize_landmarks:
            self.landmarks_buffer = torch.empty(
                (
                    model_config.num_layers,
                    max_batch_size,
                    self.local_kv_heads,
                    self.max_num_landmarks,
                    model_config.head_dim,
                ),
                device=device,
                dtype=self.dtype,
            ).contiguous()

            logger.info(
                f"ShadowkvPool: Allocated {(self.landmarks_buffer.numel() * self.landmarks_buffer.element_size()) / 2**30:.2f} GiB for Landmarks ({self.dtype})"
            )
        else:
            self.landmarks_quantization_meta = HiggsQuantizationMeta.from_config(
                "2bit", head_dim=self.head_dim, device=self.device, dtype=self.dtype
            )
            self.landmarks_buffer = QuantizedTensor(
                idx=torch.zeros(
                    (
                        self.model_config.num_layers,
                        max_batch_size,
                        self.max_num_landmarks,
                        self.local_kv_heads,
                        self.landmarks_quantization_meta.quantized_head_dim,
                    ),
                    dtype=self.landmarks_quantization_meta.idx_dtype,
                    device=self.device,
                ).contiguous(),
                scales=torch.zeros(
                    (
                        self.model_config.num_layers,
                        max_batch_size,
                        self.max_num_landmarks,
                        self.local_kv_heads,
                    ),
                    dtype=self.landmarks_quantization_meta.scales_dtype,
                    device=self.device,
                ).contiguous(),
            )

            __allocated_space = (
                self.landmarks_buffer.idx.numel() * self.landmarks_buffer.idx.element_size()
                + self.landmarks_buffer.scales.numel() * self.landmarks_buffer.scales.element_size()
            ) / 2**30

            logger.info(f"ShadowkvPool: Allocated {__allocated_space:.2f} GiB for 2-bit Landmarks")

        self.prefix_lens = torch.empty((max_batch_size,), dtype=torch.int32, device=self.device)
        self.infix_lens = torch.empty((max_batch_size,), dtype=torch.int32, device=self.device)
        self.pruned_infix_lens = torch.empty(
            (max_batch_size,), dtype=torch.int32, device=self.device
        )
        self.pruned_seq_lens = torch.empty((max_batch_size,), dtype=torch.int32, device=self.device)
        self.cu_pruned_seq_lens = torch.empty(
            (max_batch_size + 1,), dtype=torch.int32, device=self.device
        )
        self.prefix_end_indices = [0] * max_batch_size
        self.suffix_start_indices = [0] * max_batch_size
        self.num_chunks_to_select = torch.empty(
            (max_batch_size,), dtype=torch.int32, device=self.device
        )
        self.max_num_chunks_to_select = 0

        self.total_num_chunks = torch.empty(
            (max_batch_size,), dtype=torch.int32, device=self.device
        ).contiguous()
        self.batch_indices = torch.empty(
            (max_batch_size,), dtype=torch.int32, device=self.device
        ).contiguous()

        self._cpu_batch_indices = [0] * max_batch_size
        self.seqlens = [0] * max_batch_size

        self.store_cache_indices = None

        # KV-allocation
        self._kv_cache_dtype = self.dtype
        self.kv_cache_headdim = self.head_dim
        _kv_cache_device = self.device if not self.config.enable_offloading else "cpu"

        if self.config.quantize_kvcache:
            self.kvcache_quantization_meta = HiggsQuantizationMeta.from_config(
                "4bit", head_dim=self.head_dim, dtype=self.dtype, device=self.device
            )
            self._kv_cache_dtype = self.kvcache_quantization_meta.idx_dtype
            self.kv_cache_headdim = self.kvcache_quantization_meta.quantized_head_dim

            self.quantized_kvcache_scales = torch.empty(
                (
                    2,
                    model_config.num_layers,
                    max_batch_size,
                    max_seq_len,
                    self.local_kv_heads,
                ),
                device=self.device,
                dtype=self.kvcache_quantization_meta.scales_dtype,
            ).contiguous()

            self.dequantized_kvcache_buffer = torch.empty(
                (
                    2,
                    max_batch_size,
                    max_seq_len,
                    self.local_kv_heads,
                    self.head_dim,
                ),
                dtype=self.dtype,
                device=self.device,
            ).contiguous()

            self.gathered_quantized_kv_cache_scales = torch.empty(
                (
                    2,
                    max_batch_size,
                    max_seq_len,
                    self.local_kv_heads,
                ),
                device=self.device,
                dtype=self.kvcache_quantization_meta.scales_dtype,
            ).contiguous()

            # per-slot token counts consumed by higgs_quantize_heads / higgs_dequantize_heads
            # (both index `lengths` through block_indices, i.e. per kv-cache slot)
            self._slot_seq_lens = torch.zeros(
                (max_batch_size,), dtype=torch.int32, device=self.device
            )
            self._slot_pruned_seq_lens = torch.zeros(
                (max_batch_size,), dtype=torch.int32, device=self.device
            )
            # on decode the flat (max_batch_size * max_seq_len) cache is treated as
            # max_batch_size * max_seq_len slots of one token each (zero-storage expand)
            self._flat_ones = torch.ones((1,), dtype=torch.int32, device=self.device).expand(
                max_batch_size * max_seq_len
            )

        self.full_kv_buffer = torch.empty(
            (
                2,
                model_config.num_layers,
                max_batch_size,
                max_seq_len,
                self.local_kv_heads,
                self.kv_cache_headdim,
            ),
            device=_kv_cache_device,
            dtype=self._kv_cache_dtype,
        ).contiguous()

        if self.config.enable_offloading:
            logger.debug(
                f"Mapping kv cache on gpu ({_kv_cache_device, self._kv_cache_dtype, self.full_kv_buffer.shape})"
            )
            self.full_kv_buffer = map_to_gpu(self.full_kv_buffer)

        logger.info(
            f"ShadowkvPool: Allocated {(self.full_kv_buffer.numel() * self.full_kv_buffer.element_size()) / 2**30:.2f} GiB for KV cache on {_kv_cache_device}"
            f"({self._kv_cache_dtype if not self.config.quantize_kvcache else 8 * self.full_kv_buffer.element_size() // self.kvcache_quantization_meta.edenn_d})"
        )

        # SCRATCH BUFFERS

        self.selected_chunks = torch.empty(
            (max_batch_size, self.local_kv_heads, self.max_num_landmarks),
            dtype=torch.int64,
            device=self.device,
        ).contiguous()

        self._topk_values_buffer = torch.empty(
            (self.local_kv_heads, self.max_num_landmarks),
            dtype=self.dtype,
            device=self.device,
        ).contiguous()

        self._mean_query_states = torch.empty(
            (max_batch_size, self.local_kv_heads, 1, self.model_config.head_dim),
            dtype=self.dtype,
            device=self.device,
        ).contiguous()

        self.gathered_kv_buffer = torch.zeros(
            (2, max_batch_size, max_seq_len, self.local_kv_heads, self.kv_cache_headdim),
            dtype=self._kv_cache_dtype,
            device=self.device,
        ).contiguous()

        self.scores = torch.zeros(
            (max_batch_size, self.local_kv_heads, self.max_num_landmarks),
            dtype=dtype,
            device=device,
        ).contiguous()

        self.imag_page_table = torch.stack(
            [
                torch.arange(i * max_seq_len, (i + 1) * max_seq_len, device=device)
                for i in range(max_batch_size)
            ],
            dim=0,
        ).to(dtype=torch.int32)

        logger.debug(print_object_cuda_tensors_capacity(self))

    def store_kv(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        layer_idx: int,
        is_prefill: bool = False,
        batch_indices: torch.Tensor = None,
    ):
        if self.config.quantize_kvcache:
            assert batch_indices is not None
            num_tokens = k.shape[0]

            if is_prefill:
                # Whole new sequence is quantized into cache slot batch_indices[0]:
                # codes -> full_kv_buffer[:, layer_idx], scales -> quantized_kvcache_scales.
                # `_slot_seq_lens` (per slot) is filled in prepare_shadowkv_metadata.
                x_k = k.reshape(1, num_tokens, self.local_kv_heads, self.head_dim).contiguous()
                x_v = v.reshape(1, num_tokens, self.local_kv_heads, self.head_dim).contiguous()
                lengths = self._slot_seq_lens
                block_indices = batch_indices
                out_k = QuantizedTensor(
                    idx=self.full_kv_buffer[0, layer_idx],
                    scales=self.quantized_kvcache_scales[0, layer_idx],
                )
                out_v = QuantizedTensor(
                    idx=self.full_kv_buffer[1, layer_idx],
                    scales=self.quantized_kvcache_scales[1, layer_idx],
                )
            else:
                # Decode: treat the cache as (max_batch_size * max_seq_len) one-token slots
                # and quantize each new token directly into its flat row
                # (batch_idx * max_seq_len + seq_len - 1)
                x_k = k.reshape(num_tokens, 1, self.local_kv_heads, self.head_dim).contiguous()
                x_v = v.reshape(num_tokens, 1, self.local_kv_heads, self.head_dim).contiguous()
                lengths = self._flat_ones
                block_indices = self.store_cache_indices
                flat_tokens = self.max_batch_size * self.max_seq_len
                out_k = QuantizedTensor(
                    idx=self.full_kv_buffer[0, layer_idx].view(
                        flat_tokens, 1, self.local_kv_heads, self.kv_cache_headdim
                    ),
                    scales=self.quantized_kvcache_scales[0, layer_idx].view(
                        flat_tokens, 1, self.local_kv_heads
                    ),
                )
                out_v = QuantizedTensor(
                    idx=self.full_kv_buffer[1, layer_idx].view(
                        flat_tokens, 1, self.local_kv_heads, self.kv_cache_headdim
                    ),
                    scales=self.quantized_kvcache_scales[1, layer_idx].view(
                        flat_tokens, 1, self.local_kv_heads
                    ),
                )

            higgs_quantize_heads(
                x_k,
                grid=self.kvcache_quantization_meta.grid,
                out=out_k,
                heads_first=False,
                block_indices=block_indices,
                lengths=lengths,
            )

            higgs_quantize_heads(
                x_v,
                grid=self.kvcache_quantization_meta.grid,
                out=out_v,
                heads_first=False,
                block_indices=block_indices,
                lengths=lengths,
            )

            return

        store_cache(
            k_cache=self.full_kv_buffer[0, layer_idx].view(
                self.max_batch_size * self.max_seq_len,
                self.local_kv_heads,
                self.kv_cache_headdim,
            ),
            v_cache=self.full_kv_buffer[1, layer_idx].view(
                self.max_batch_size * self.max_seq_len,
                self.local_kv_heads,
                self.kv_cache_headdim,
            ),
            indices=self.store_cache_indices,
            k=k,
            v=v,
        )

    def compute_and_store_landmarks(
        self, key_states: torch.Tensor, layer_idx: int, batch_indices: list[int]
    ):
        assert len(batch_indices) == 1
        # assert key_states.is_contiguous()

        batch_index = batch_indices[0]
        num_chunks = int(self.total_num_chunks[batch_index])

        if num_chunks == 0:
            return

        # self.scores[batch_index].fill_(float("-inf"))

        new_landmarks = key_states[
            self.prefix_end_indices[batch_index] : self.suffix_start_indices[batch_index]
        ]

        if self.config.chunk_size > 1:
            new_landmarks = new_landmarks.view(
                num_chunks,
                self.config.chunk_size,
                self.local_kv_heads,
                self.model_config.head_dim,
            )

            new_landmarks = torch.mean(new_landmarks, dim=1)

        new_landmarks = new_landmarks.contiguous()

        if not self.config.quantize_landmarks:
            new_landmarks = new_landmarks.transpose(0, 1)

            self.landmarks_buffer[layer_idx, batch_index, :, :num_chunks].copy_(
                new_landmarks.to(dtype=self.dtype)
            )
            # self.landmarks_buffer[layer_idx, batch_index].index_copy_(
            #     dim=1, index=torch.arange(num_chunks, device=self.device), source=new_landmarks
            # )
        else:
            higgs_quantize_heads(
                new_landmarks.view(
                    1, num_chunks, self.local_kv_heads, self.model_config.head_dim
                ).contiguous(),
                lengths=self.total_num_chunks,
                grid=self.landmarks_quantization_meta.grid,
                out=self.landmarks_buffer,
                block_indices=torch.Tensor([batch_index]).to(
                    device=self.device, dtype=torch.int32, non_blocking=True
                ),
                layer_idx=layer_idx,
                heads_first=False,
            )

    def prepare_shadowkv_metadata(self, seqlens: list[int], batch_indices: list[int]):
        fill_prefill_metadata(
            self.prefix_lens,
            self.infix_lens,
            self.pruned_infix_lens,
            torch.Tensor(seqlens).to(device=self.device, dtype=torch.int32, non_blocking=True),
            torch.Tensor(batch_indices).to(
                device=self.device, dtype=torch.int32, non_blocking=True
            ),
            self.config.prefix_budget,
            self.config.sparse_budget,
            self.config.suffix_budget,
            self.config.min_seqlen_to_prune,
            self.config.chunk_size,
        )
        self.prefix_end_indices = self.prefix_lens.cpu().tolist()
        self.suffix_start_indices = (self.prefix_lens.cpu() + self.infix_lens.cpu()).tolist()
        torch.div(
            self.infix_lens,
            self.config.chunk_size,
            rounding_mode="floor",
            out=self.total_num_chunks,
        )

        torch.div(
            self.pruned_infix_lens,
            self.config.chunk_size,
            rounding_mode="floor",
            out=self.num_chunks_to_select,
        )

        self.store_cache_indices = (
            torch.arange(0, seqlens[0]) + (batch_indices[0] * self.max_seq_len)
        ).to(device=self.device, dtype=torch.int32, non_blocking=True)

        if self.config.quantize_kvcache:
            self._slot_seq_lens[batch_indices[0]] = seqlens[0]

    def prepare_decode_metadata(self, seqlens: list[int], batch_indices: list[int]):
        BS = len(batch_indices)

        self._cpu_batch_indices[:BS] = batch_indices
        self.batch_indices[:BS].copy_(
            torch.tensor(batch_indices, dtype=torch.int32), non_blocking=True
        )

        fill_decode_metadata(
            self.prefix_lens,
            self.infix_lens,
            self.pruned_infix_lens,
            torch.Tensor(seqlens).to(device=self.device, dtype=torch.int32, non_blocking=True),
            self.batch_indices[:BS],
            self.pruned_seq_lens,
            self.cu_pruned_seq_lens,
        )

        self.store_cache_indices = torch.tensor(
            [
                batch_idx * self.max_seq_len + seqlens[i] - 1
                for i, batch_idx in enumerate(batch_indices)
            ]
        ).to(device=self.device, dtype=torch.int32, non_blocking=True)

        if self.config.quantize_kvcache:
            # per-slot pruned lengths, consumed by higgs_dequantize_heads in select_kv
            self._slot_pruned_seq_lens.index_copy_(
                0, self.batch_indices[:BS].long(), self.pruned_seq_lens[:BS]
            )

        return (
            self.cu_pruned_seq_lens[: BS + 1],
            max(seqlens),  # TODO: pruned estimate here
            self.pruned_seq_lens[:BS],
            self.imag_page_table[:BS, : max(seqlens)],
        )

    def select_kv(
        self,
        query_states: torch.Tensor,
        layer_idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        BS, num_qo_heads, HD = query_states.shape

        # SCORING
        if not self.config.quantize_landmarks:
            torch.mean(
                query_states.view(BS, self.local_kv_heads, self.gqa_factor, 1, HD),
                dim=2,
                out=self._mean_query_states[:BS],
            )

            landmarks = self.landmarks_buffer[layer_idx]

            shadowkv_score_landmarks_kernel_hd128(
                self._mean_query_states[:BS],
                self.scores,
                landmarks,
                self.local_kv_heads,
                self.total_num_chunks,
                self.max_num_landmarks,
                self.batch_indices[:BS],
            )
        else:
            higgs_score(
                self.landmarks_buffer,
                lengths=self.total_num_chunks,
                grid=self.landmarks_quantization_meta.grid,
                query=query_states,
                hadamard_scale=self.landmarks_quantization_meta.hadamard_scale,
                block_indices=self.batch_indices[:BS],
                out=self.scores,
                layer_idx=layer_idx,
            )

        # TOPK
        shadowkv_topk_kernel(
            scores=self.scores,
            total_num_chunks=self.total_num_chunks,
            num_chunks_to_select=self.num_chunks_to_select,
            batch_indices=self.batch_indices[:BS],
            out=self.selected_chunks,
        )

        gather_kv_cache(
            self.prefix_lens,
            self.infix_lens,
            self.pruned_infix_lens,
            self.batch_indices[:BS],
            self.pruned_seq_lens[:BS],
            self.cu_pruned_seq_lens[: BS + 1],
            self.selected_chunks,
            self.full_kv_buffer[0, layer_idx],
            self.full_kv_buffer[1, layer_idx],
            self.gathered_kv_buffer[0],
            self.gathered_kv_buffer[1],
            self.config.chunk_size,
        )

        if self.config.quantize_kvcache:
            shadowkv_gather_scales_kernel(
                self.prefix_lens,
                self.infix_lens,
                self.pruned_infix_lens,
                self.batch_indices[:BS],
                self.pruned_seq_lens[:BS],
                self.selected_chunks,
                self.quantized_kvcache_scales[0, layer_idx],
                self.quantized_kvcache_scales[1, layer_idx],
                self.gathered_quantized_kv_cache_scales[0],
                self.gathered_quantized_kv_cache_scales[1],
                self.config.chunk_size,
            )

            for i in range(2):  # for K and V respectively
                higgs_dequantize_heads(
                    quantized=QuantizedTensor(
                        idx=self.gathered_kv_buffer[i],
                        scales=self.gathered_quantized_kv_cache_scales[i],
                    ),
                    lengths=self._slot_pruned_seq_lens,
                    grid=self.kvcache_quantization_meta.grid,
                    hadamard_scale=self.kvcache_quantization_meta.hadamard_scale,
                    block_indices=self.batch_indices[:BS],
                    out=self.dequantized_kvcache_buffer[i],
                    heads_first=False,
                )

            return self.dequantized_kvcache_buffer[0], self.dequantized_kvcache_buffer[1]

        return self.gathered_kv_buffer[0], self.gathered_kv_buffer[1]
