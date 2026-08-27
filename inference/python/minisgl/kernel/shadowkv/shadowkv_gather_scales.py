import torch

import triton
import triton.language as tl


# grid: (BS, 2 /* K and V */, NUM_TILE_PIDS)
@triton.jit
def __shadowkv_gather_scales_kernel_triton(
    prefix_lens,  # (max_batch_size,) int32, indexed by kv cache slot
    infix_lens,  # (max_batch_size,) int32, indexed by kv cache slot
    pruned_infix_lens,  # (max_batch_size,) int32, indexed by kv cache slot
    batch_indices,  # (BS,) int32, active batch pos -> kv cache slot
    pruned_seq_lens,  # (BS,) int32, indexed by active batch pos
    selected_chunks,  # (max_batch_size, local_kv_heads, max_num_chunks) int64
    selected_chunks_stride_b,
    selected_chunks_stride_h,
    selected_chunks_stride_c,
    k_scales_src,  # (max_batch_size, max_seq_len, local_kv_heads) float32
    v_scales_src,  # (max_batch_size, max_seq_len, local_kv_heads) float32
    k_scales_dst,  # (max_batch_size, max_seq_len, local_kv_heads) float32
    v_scales_dst,  # (max_batch_size, max_seq_len, local_kv_heads) float32
    scales_stride_b,
    scales_stride_t,
    chunk_size,
    NUM_TILE_PIDS: tl.constexpr,
    BLOCK_T: tl.constexpr,
    H: tl.constexpr,  # local_kv_heads (power of 2 padded via mask)
    H_POW2: tl.constexpr,
):
    pid_b, pid_kv, pid_tile = tl.program_id(0), tl.program_id(1), tl.program_id(2)

    kv_cache_idx = tl.load(batch_indices + pid_b)
    num_tokens = tl.load(pruned_seq_lens + pid_b)

    prefix_len = tl.load(prefix_lens + kv_cache_idx)
    infix_len = tl.load(infix_lens + kv_cache_idx)
    pruned_infix_len = tl.load(pruned_infix_lens + kv_cache_idx)

    if pid_kv == 0:
        src_base = k_scales_src
        dst_base = k_scales_dst
    else:
        src_base = v_scales_src
        dst_base = v_scales_dst

    src_base = src_base + kv_cache_idx.to(tl.int64) * scales_stride_b
    dst_base = dst_base + kv_cache_idx.to(tl.int64) * scales_stride_b
    chunks_base = selected_chunks + kv_cache_idx.to(tl.int64) * selected_chunks_stride_b

    h_off = tl.arange(0, H_POW2)
    h_mask = h_off < H

    for tile_start in range(
        pid_tile * BLOCK_T, num_tokens, NUM_TILE_PIDS * BLOCK_T
    ):
        t_off = tile_start + tl.arange(0, BLOCK_T)
        t_mask = t_off < num_tokens

        # (BLOCK_T, H_POW2) masks / indices
        mask_2d = t_mask[:, None] & h_mask[None, :]

        is_prefix = t_off < prefix_len
        is_sparse = (t_off >= prefix_len) & (t_off < prefix_len + pruned_infix_len)

        # sparse part: per (token, head) source index via selected chunks
        sparse_pos = t_off - prefix_len
        top_landmark_idx = sparse_pos // chunk_size
        token_in_chunk = sparse_pos - top_landmark_idx * chunk_size

        chunk_load_mask = (is_sparse & t_mask)[:, None] & h_mask[None, :]
        landmark_idx = tl.load(
            chunks_base
            + h_off[None, :] * selected_chunks_stride_h
            + top_landmark_idx[:, None] * selected_chunks_stride_c,
            mask=chunk_load_mask,
            other=0,
        )
        sparse_src_t = prefix_len + landmark_idx * chunk_size + token_in_chunk[:, None]

        # suffix part: shifted by the pruned-out tokens
        suffix_src_t = (t_off + (infix_len - pruned_infix_len))[:, None].to(tl.int64)

        prefix_src_t = t_off[:, None].to(tl.int64)

        src_t = tl.where(
            is_prefix[:, None],
            prefix_src_t,
            tl.where(is_sparse[:, None], sparse_src_t, suffix_src_t),
        )

        values = tl.load(
            src_base + src_t * scales_stride_t + h_off[None, :],
            mask=mask_2d,
            other=0.0,
        )
        tl.store(
            dst_base + t_off[:, None].to(tl.int64) * scales_stride_t + h_off[None, :],
            values,
            mask=mask_2d,
        )


def shadowkv_gather_scales_kernel(
    prefix_lens: torch.Tensor,
    infix_lens: torch.Tensor,
    pruned_infix_lens: torch.Tensor,
    batch_indices: torch.Tensor,
    pruned_seq_lens: torch.Tensor,
    selected_chunks: torch.Tensor,
    k_scales_src: torch.Tensor,
    v_scales_src: torch.Tensor,
    k_scales_dst: torch.Tensor,
    v_scales_dst: torch.Tensor,
    chunk_size: int,
    num_tile_pids: int = 32,
    block_t: int = 64,
):
    """
    Gathers per-(token, head) quantization scales the same way gather_kv_cache gathers
    the quantized codes: dst[b, t, h] = src[b, src_t(t, h), h] where src_t is
      t                                              for t < prefix_len
      prefix + selected_chunks[b, h, (t - prefix) // chunk] * chunk + (t - prefix) % chunk
                                                     for prefix <= t < prefix + pruned_infix
      t + (infix_len - pruned_infix_len)             otherwise (suffix)

    All source/destination tensors are (max_batch_size, max_seq_len, local_kv_heads)
    float32, indexed by kv-cache slot (batch_indices[b]). The grid is fixed
    (graph-capture safe): each program strides over token tiles.
    """
    BS = batch_indices.shape[0]
    H = k_scales_src.shape[2]
    H_POW2 = triton.next_power_of_2(H)

    assert k_scales_src.stride(2) == 1 and k_scales_dst.stride(2) == 1

    grid = (BS, 2, num_tile_pids)
    __shadowkv_gather_scales_kernel_triton[grid](
        prefix_lens,
        infix_lens,
        pruned_infix_lens,
        batch_indices,
        pruned_seq_lens,
        selected_chunks,
        selected_chunks.stride(0),
        selected_chunks.stride(1),
        selected_chunks.stride(2),
        k_scales_src,
        v_scales_src,
        k_scales_dst,
        v_scales_dst,
        k_scales_src.stride(0),
        k_scales_src.stride(1),
        chunk_size,
        NUM_TILE_PIDS=num_tile_pids,
        BLOCK_T=block_t,
        H=H,
        H_POW2=H_POW2,
    )
