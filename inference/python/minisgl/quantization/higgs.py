import os
import torch
import typing as tp

from dataclasses import dataclass

from collections import namedtuple

# from minisgl.shadowkv_kernels import higgs_quantize, higgs_dequantize, higgs_dequantize_full
import minisgl.shadowkv_kernels as higgs_quantization_cuda


def is_power_of_two(x: int) -> bool:
    return (x & (x - 1)) == 0


def quantize(x: torch.Tensor):
    assert is_power_of_two(x.shape[-1])


def get_2bit_grid(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    file = f"{os.path.dirname(os.path.abspath(__file__))}/grids/EDEN4-256.pt"
    grid = torch.load(file, map_location="cpu").to(device=device, dtype=dtype)

    return grid

def get_4bit_grid(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    file = f"{os.path.dirname(os.path.abspath(__file__))}/grids/EDEN2-256.pt"
    grid = torch.load(file, map_location="cpu").to(device=device, dtype=dtype)

    return grid

@dataclass(frozen=False)
class HiggsQuantizationMeta:
    cfg: str = None
    edenn_d: int = None
    grid: torch.Tensor = None
    hadamard_scale: torch.Tensor = None
    idx_dtype: torch.dtype = None
    scales_dtype: torch.dtype = torch.float32
    quantized_head_dim: int = None

    @classmethod
    def from_config(cls, cfg: str, head_dim, device, dtype):
        if cfg == '2bit':
            return cls(
                cfg=cfg,
                edenn_d=4,
                grid=get_2bit_grid(device, dtype),
                hadamard_scale=(1 / head_dim),
                idx_dtype=torch.uint8,
                quantized_head_dim=(head_dim // 4),
            )
        elif cfg == '4bit':
            return cls(
                cfg=cfg,
                edenn_d=2,
                grid=get_4bit_grid(device, dtype),
                hadamard_scale=(1 / head_dim),
                idx_dtype=torch.uint8,
                quantized_head_dim=(head_dim // 2),
            )


QuantizedTensor = namedtuple("QuantizedTensor", ["idx", "scales"])


def higgs_quantize(x: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
    return higgs_quantization_cuda.higgs_quantize(x, grid)


def higgs_dequantize(quantized: QuantizedTensor, grid: torch.Tensor) -> torch.Tensor:
    return higgs_quantization_cuda.higgs_dequantize(quantized.idx, quantized.scales, grid)


def higgs_dequantize_full(
    quantized: QuantizedTensor,
    grid: torch.Tensor,
    add_prediction: torch.Tensor,
    out: torch.Tensor,
    hadamard_scale,
) -> torch.Tensor:
    higgs_quantization_cuda.higgs_dequantize_full(
        quantized.idx, quantized.scales, grid, add_prediction, out, hadamard_scale
    )


def _cum_lengths(lengths: torch.Tensor, device) -> torch.Tensor:
    lengths = lengths.to(torch.int64)
    cum_lengths = torch.zeros(lengths.shape[0] + 1, dtype=torch.int32, device=device)
    cum_lengths[1:] = torch.cumsum(lengths, 0).to(torch.int32)
    return cum_lengths


def higgs_quantize_heads(
    x: torch.Tensor,
    lengths: torch.Tensor,
    grid: torch.Tensor,
    out: tp.Optional[QuantizedTensor] = None,
    heads_first: bool = False,
    block_indices: tp.Optional[torch.Tensor] = None,
    layer_idx: tp.Optional[int] = None,
) -> QuantizedTensor:
    # Prefill kernel. x: [batch_size, padded_T, n_kv_heads, head_dim] bf16 keys
    # ([batch_size, n_kv_heads, padded_T, head_dim] if heads_first) -- the new requests.
    # `out` is the whole preallocated kv cache with max_batch_size as the first dim and
    # max_seq_len as the token dim (may exceed padded_T); sample b is written into cache slot
    # block_indices[b] (identity by default; values must be distinct -- this is a scatter).
    # `lengths`: [max_batch_size] int, tokens per cache slot, indexed through block_indices;
    # padding tokens are skipped, their idx/scales are left untouched. A warp owns a whole
    # token in this kernel, so lengths need no rounding here (unlike higgs_score).
    # `grid`: [256, 4] for 2-bit or [256, 2] for 4-bit quantization.
    # Returns codes [max_batch_size, ..., head_dim // d] uint8 and per-head norms float32 --
    # with heads_first=False that is exactly the landmarks format higgs_score consumes, with
    # heads_first=True the format higgs_dequantize_heads consumes.
    d = grid.shape[1]
    batch_size = x.shape[0]

    if out is None:
        assert (
            block_indices is None
        ), "cannot infer max_batch_size: pass the preallocated out together with block_indices"
        out = QuantizedTensor(
            idx=torch.empty(*x.shape[:3], x.shape[3] // d, dtype=torch.uint8, device=x.device),
            scales=torch.empty(x.shape[:3], dtype=torch.float32, device=x.device),
        )

        assert layer_idx is None

    out_idx = out.idx
    out_scales = out.scales

    if layer_idx is not None:
        out_idx = out_idx[layer_idx]
        out_scales = out_scales[layer_idx]

    assert (
        lengths.shape[0] == out_idx.shape[0]
    ), f"lengths must be per cache slot: {lengths.shape[0]} != max_batch_size {out_idx.shape[0]}"
    if block_indices is None:  # identity mapping: a slice replaces the gather
        block_indices = torch.arange(batch_size, dtype=torch.int32, device=x.device)
        active_lengths = lengths[:batch_size]
    else:
        block_indices = block_indices.to(torch.int32)
        active_lengths = lengths[block_indices]
    cum_lengths = _cum_lengths(active_lengths, x.device)

    higgs_quantization_cuda.higgs_quantize_heads(
        x, cum_lengths, block_indices, grid, out_idx, out_scales, heads_first
    )

    return out


def higgs_dequantize_heads(
    quantized: QuantizedTensor,
    lengths: torch.Tensor,
    grid: torch.Tensor,
    hadamard_scale: float,
    block_indices: tp.Optional[torch.Tensor] = None,
    out: tp.Optional[torch.Tensor] = None,
    heads_first: bool = True,
) -> torch.Tensor:
    # Decode kernel, inverse of higgs_quantize_heads(..., heads_first=heads_first) with a
    # [256, 2] (4-bit) grid. `quantized` is the whole preallocated kv cache:
    # codes [max_batch_size, n_kv_heads, max_T, head_dim // 2] + scales
    # [max_batch_size, n_kv_heads, max_T] (with heads_first=False the t/h dims are swapped:
    # codes [max_batch_size, max_T, n_kv_heads, head_dim // 2] + scales
    # [max_batch_size, max_T, n_kv_heads]). Only batch_size <= max_batch_size requests are
    # active; out spans the whole cache too (same layout as codes, head_dim last, bf16) and
    # the dequantized entries land in the same slots:
    # out[block_indices[b]] = dequant(cache[block_indices[b]]) (identity mapping over all
    # slots by default; block_indices values must be distinct). Untouched slots keep their
    # old values. `lengths`: [max_batch_size] int, tokens per cache slot, indexed through
    # block_indices; padding tokens (t >= lengths[block_indices[b]]) are skipped as well.
    # hadamard_scale should be 1 / head_dim.
    d = grid.shape[1]
    max_batch_size = quantized.idx.shape[0]
    assert (
        lengths.shape[0] == max_batch_size
    ), f"lengths must be per cache slot: {lengths.shape[0]} != max_batch_size {max_batch_size}"

    if block_indices is None:  # identity mapping over the whole cache: a slice replaces the gather
        block_indices = torch.arange(max_batch_size, dtype=torch.int32, device=quantized.idx.device)
        active_lengths = lengths
    else:
        block_indices = block_indices.to(torch.int32)
        active_lengths = lengths[block_indices.long()]

    cum_lengths = _cum_lengths(active_lengths, quantized.idx.device)

    if out is None:
        out = torch.empty(
            max_batch_size,
            *quantized.idx.shape[1:3],
            quantized.idx.shape[3] * d,
            dtype=torch.bfloat16,
            device=quantized.idx.device,
        )

    higgs_quantization_cuda.higgs_dequantize_heads(
        quantized.idx,
        quantized.scales,
        cum_lengths,
        block_indices,
        grid,
        out,
        hadamard_scale,
        heads_first,
    )

    return out


def higgs_score(
    landmarks: QuantizedTensor,
    lengths: torch.Tensor,
    grid: torch.Tensor,
    query: torch.Tensor,
    hadamard_scale: float,
    block_indices: tp.Optional[torch.Tensor] = None,
    out: tp.Optional[torch.Tensor] = None,
    layer_idx: tp.Optional[int] = None,
) -> torch.Tensor:
    # Decode kernel. landmarks hold the whole preallocated kv cache
    # ([max_batch_size, max_T, n_kv_heads, ...]); only batch_size <= max_batch_size requests
    # are active. `block_indices`: [batch_size] int, query b scores the tokens of kv-cache
    # entry block_indices[b] (defaults to the identity mapping), and the scores land in the
    # same slot of out ([max_batch_size, n_kv_heads, max_T] bf16):
    # out[block_indices[b], h, t] = score of query b -- so block_indices values must be
    # distinct; untouched slots keep their old values. `lengths`: [max_batch_size] int,
    # tokens per cache slot, indexed through block_indices. The kernel needs each query's
    # flattened token range to be a multiple of tokens-per-warp (so a warp never straddles a
    # query boundary), so we round up here and pass the cumsum.
    if query.ndim == 4:
        assert query.shape[1] == 1
        query = query.squeeze(1)

    landmarks_idx = landmarks.idx
    landmarks_scales = landmarks.scales

    if layer_idx is not None:
        landmarks_idx = landmarks_idx[layer_idx]
        landmarks_scales = landmarks_scales[layer_idx]

    batch_size = query.shape[0]
    n_kv_heads = landmarks_idx.shape[2]
    assert (
        lengths.shape[0] == landmarks_idx.shape[0]
    ), f"lengths must be per cache slot: {lengths.shape[0]} != max_batch_size {landmarks_idx.shape[0]}"

    if block_indices is None:  # identity mapping: a slice replaces the gather
        block_indices = torch.arange(batch_size, dtype=torch.int32, device=landmarks_idx.device)
        lengths = lengths[:batch_size]
    else:
        block_indices = block_indices.to(torch.int32)
        lengths = lengths[block_indices]

    # tokens-per-warp = (32 lanes * 16 bytes) / (n_kv_heads * head_dim/d bytes);  head_dim=128, d=4 -> 16 // n_kv
    tokens_per_warp = 16 // n_kv_heads
    lengths = lengths.to(torch.int64)
    rounded = ((lengths + tokens_per_warp - 1) // tokens_per_warp) * tokens_per_warp
    cum_lengths = torch.zeros(batch_size + 1, dtype=torch.int32, device=landmarks_idx.device)
    cum_lengths[1:] = torch.cumsum(rounded, 0).to(torch.int32)

    if out is None:
        out = torch.empty(
            landmarks_idx.shape[0],
            n_kv_heads,
            landmarks_idx.shape[1],
            dtype=query.dtype,
            device=query.device,
        )  # [max_batch, n_kv_head, T] scores

    higgs_quantization_cuda.higgs_score(
        landmarks_idx,
        landmarks_scales,
        cum_lengths,
        block_indices,
        grid,
        query,
        out,
        hadamard_scale,
    )

    return out
