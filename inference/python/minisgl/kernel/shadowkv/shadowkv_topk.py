import torch
import triton
import triton.language as tl

INT16_MIN = -32768
SORT_MAX_L = 1024  # auto-dispatch threshold, tuned on H200
RADIX_BLOCK_L = 8192


@triton.jit
def _topk_indices_sort_kernel(
    scores_ptr,  # bf16  (MAX_BS, H, L)
    k_ptr,  # int32 (MAX_BS,)
    total_chunks_ptr,  # int32 (MAX_BS,)
    batch_indices_ptr,  # int32 (CUR_BS,)
    out_ptr,  # int64 (MAX_BS, H, L)
    stride_sb,
    stride_sh,
    stride_sl,
    stride_ob,
    stride_oh,
    stride_ol,
    L,
    BLOCK_L: tl.constexpr,
):
    pid_b = tl.program_id(0)
    head = tl.program_id(1)

    batch_idx = tl.load(batch_indices_ptr + pid_b)
    n = tl.load(total_chunks_ptr + batch_idx)  # valid row length of this request
    n = tl.minimum(n, L)
    k = tl.load(k_ptr + batch_idx)
    k = tl.minimum(k, n)
    if k <= 0:
        return

    offs = tl.arange(0, BLOCK_L)
    mask = offs < n

    row_ptr = scores_ptr + batch_idx.to(tl.int64) * stride_sb + head.to(tl.int64) * stride_sh
    s = tl.load(row_ptr + offs * stride_sl, mask=mask, other=0.0)

    # ---- bf16 -> order-preserving 16-bit key -------------------------------
    bits = s.to(tl.int16, bitcast=True).to(tl.int32) & 0xFFFF
    # negative floats: flip all 16 bits; non-negative: set the sign bit
    key = tl.where((bits & 0x8000) != 0, bits ^ 0xFFFF, bits | 0x8000)
    # center into signed int16 range: [0, 65535] -> [-32768, 32767]
    key = key - 0x8000
    # padding lanes must lose against every real score (even -inf -> -32641)
    key = tl.where(mask, key, -32768)

    # ---- pack (key | index) and sort descending ----------------------------
    packed = (key << 16) | offs
    packed = tl.sort(packed, descending=True)

    idx = (packed & 0xFFFF).to(tl.int64)

    out_row_ptr = out_ptr + batch_idx.to(tl.int64) * stride_ob + head.to(tl.int64) * stride_oh
    tl.store(out_row_ptr + offs * stride_ol, idx, mask=offs < k)


@triton.jit
def _topk_indices_radix_kernel(
    scores_ptr,  # bf16  (MAX_BS, H, L)
    k_ptr,  # int32 (MAX_BS,)
    total_chunks_ptr,  # int32 (MAX_BS,)
    batch_indices_ptr,  # int32 (CUR_BS,)
    out_ptr,  # int64 (MAX_BS, H, L)
    stride_sb,
    stride_sh,
    stride_sl,
    stride_ob,
    stride_oh,
    stride_ol,
    L,
    BLOCK_L: tl.constexpr,
):
    pid_b = tl.program_id(0)
    head = tl.program_id(1)

    batch_idx = tl.load(batch_indices_ptr + pid_b)
    n = tl.load(total_chunks_ptr + batch_idx)  # valid row length of this request
    n = tl.minimum(n, L)
    k = tl.load(k_ptr + batch_idx)
    k = tl.minimum(k, n)
    if k <= 0:
        return

    row_ptr = scores_ptr + batch_idx.to(tl.int64) * stride_sb + head.to(tl.int64) * stride_sh
    # only blocks containing valid data are visited (short rows are cheap)
    num_blocks = tl.cdiv(n, BLOCK_L)
    bins = tl.arange(0, 256)

    # ---- pass 1: histogram of the 8 high key bits ---------------------------
    # key(x) = monotone uint16 image of bf16 x: flip all bits if negative,
    # else set the sign bit; padding contributes to bucket 0 and cancels out.
    hist_hi = tl.zeros((256,), dtype=tl.int32)
    for blk in range(num_blocks):
        offs = blk * BLOCK_L + tl.arange(0, BLOCK_L)
        mask = offs < n
        s = tl.load(row_ptr + offs * stride_sl, mask=mask, other=0.0)
        bits = s.to(tl.int16, bitcast=True).to(tl.int32) & 0xFFFF
        key = tl.where((bits & 0x8000) != 0, bits ^ 0xFFFF, bits | 0x8000)
        key = tl.where(mask, key, 0)
        hist_hi += tl.histogram(key >> 8, 256)

    # inclusive suffix counts: inc_hi[b] = #{key_hi >= b}  (non-increasing)
    inc_hi = tl.flip(tl.cumsum(tl.flip(hist_hi, 0), 0), 0)
    # B = max b with inc_hi[b] >= k  (b = 0 always qualifies: inc_hi[0] >= L)
    B = tl.sum(tl.where(inc_hi >= k, 1, 0), 0) - 1
    inc_B = tl.sum(tl.where(bins == B, inc_hi, 0), 0)
    hist_B = tl.sum(tl.where(bins == B, hist_hi, 0), 0)
    # pads live only in bucket 0, so this difference is always pad-free
    count_above_hi = inc_B - hist_B
    k2 = k - count_above_hi  # rank of the threshold inside bucket B (>= 1)

    # ---- pass 2: histogram of the 8 low key bits inside bucket B ------------
    hist_lo = tl.zeros((256,), dtype=tl.int32)
    for blk in range(num_blocks):
        offs = blk * BLOCK_L + tl.arange(0, BLOCK_L)
        mask = offs < n
        s = tl.load(row_ptr + offs * stride_sl, mask=mask, other=0.0)
        bits = s.to(tl.int16, bitcast=True).to(tl.int32) & 0xFFFF
        key = tl.where((bits & 0x8000) != 0, bits ^ 0xFFFF, bits | 0x8000)
        sel = mask & ((key >> 8) == B)
        lo = tl.where(sel, key & 0xFF, 0)
        hist_lo += tl.histogram(lo, 256)
        # lanes with sel == False polluted bin 0 above; remove them
        n_bad = BLOCK_L - tl.sum(sel.to(tl.int32), 0)
        hist_lo = tl.where(bins == 0, hist_lo - n_bad, hist_lo)

    inc_lo = tl.flip(tl.cumsum(tl.flip(hist_lo, 0), 0), 0)
    B_lo = tl.sum(tl.where(inc_lo >= k2, 1, 0), 0) - 1
    inc_Blo = tl.sum(tl.where(bins == B_lo, inc_lo, 0), 0)
    hist_Blo = tl.sum(tl.where(bins == B_lo, hist_lo, 0), 0)

    T = (B << 8) | B_lo  # threshold key (k-th largest)
    A = count_above_hi + (inc_Blo - hist_Blo)  # #{key > T}, guaranteed < k
    remaining = k - A  # how many key == T to take

    out_row_ptr = out_ptr + batch_idx.to(tl.int64) * stride_ob + head.to(tl.int64) * stride_oh

    # ---- pass 3: compaction --------------------------------------------------
    # keys > T go to out[0:A] (stream order); the first `remaining` keys == T
    # go to out[A:k].  Positions are disjoint and dense -> exactly k stores.
    n_gt = tl.zeros((1,), dtype=tl.int32)
    n_eq = tl.zeros((1,), dtype=tl.int32)
    for blk in range(num_blocks):
        offs = blk * BLOCK_L + tl.arange(0, BLOCK_L)
        mask = offs < n
        s = tl.load(row_ptr + offs * stride_sl, mask=mask, other=0.0)
        bits = s.to(tl.int16, bitcast=True).to(tl.int32) & 0xFFFF
        key = tl.where((bits & 0x8000) != 0, bits ^ 0xFFFF, bits | 0x8000)

        is_gt = mask & (key > T)
        is_eq = mask & (key == T)

        excl_gt = tl.cumsum(is_gt.to(tl.int32), 0) - is_gt.to(tl.int32)
        excl_eq = tl.cumsum(is_eq.to(tl.int32), 0) - is_eq.to(tl.int32)

        pos_gt = n_gt + excl_gt
        pos_eq = A + n_eq + excl_eq
        take_eq = is_eq & (n_eq + excl_eq < remaining)

        pos = tl.where(is_gt, pos_gt, pos_eq).to(tl.int64)
        do_store = is_gt | take_eq
        tl.store(out_row_ptr + pos * stride_ol, offs.to(tl.int64), mask=do_store)

        n_gt += tl.sum(is_gt.to(tl.int32), 0)
        n_eq += tl.sum(is_eq.to(tl.int32), 0)


def shadowkv_topk_kernel(
    scores: torch.Tensor,  # (max_bs, H, L) bf16
    num_chunks_to_select: torch.Tensor,  # (max_bs,) int32 -- K per request
    total_num_chunks: torch.Tensor,  # (max_bs,) int32 -- valid length per request
    batch_indices: torch.Tensor,  # (cur_bs,) int32
    out: torch.Tensor,  # (max_bs, H, L) int64
    algo: str = "auto",
) -> None:
    max_bs, H, L = scores.shape
    cur_bs = batch_indices.shape[0]
    assert scores.dtype == torch.bfloat16
    assert num_chunks_to_select.dtype == torch.int32
    assert num_chunks_to_select.shape == (max_bs,)
    assert total_num_chunks.dtype == torch.int32
    assert total_num_chunks.shape == (max_bs,)
    assert batch_indices.dtype == torch.int32
    assert out.dtype == torch.int64 and out.shape == (max_bs, H, L)
    if cur_bs == 0:
        return

    if algo == "auto":
        algo = "sort" if L <= SORT_MAX_L else "radix"

    if algo == "sort":
        assert L <= 65536, "sort kernel: landmark index must fit into 16 bits"
        BLOCK_L = max(2, triton.next_power_of_2(L))
        if BLOCK_L <= 1024:
            num_warps = 4
        elif BLOCK_L <= 4096:
            num_warps = 8
        else:
            num_warps = 16
        _topk_indices_sort_kernel[(cur_bs, H)](
            scores,
            num_chunks_to_select,
            total_num_chunks,
            batch_indices,
            out,
            scores.stride(0),
            scores.stride(1),
            scores.stride(2),
            out.stride(0),
            out.stride(1),
            out.stride(2),
            L,
            BLOCK_L=BLOCK_L,
            num_warps=num_warps,
        )
    elif algo == "radix":
        assert L <= 2**24, "radix kernel tested for L <= 2**24"
        BLOCK_L = min(RADIX_BLOCK_L, max(256, triton.next_power_of_2(L)))
        _topk_indices_radix_kernel[(cur_bs, H)](
            scores,
            num_chunks_to_select,
            total_num_chunks,
            batch_indices,
            out,
            scores.stride(0),
            scores.stride(1),
            scores.stride(2),
            out.stride(0),
            out.stride(1),
            out.stride(2),
            L,
            BLOCK_L=BLOCK_L,
            num_warps=8,
        )
    else:
        raise ValueError(f"unknown algo: {algo}")
