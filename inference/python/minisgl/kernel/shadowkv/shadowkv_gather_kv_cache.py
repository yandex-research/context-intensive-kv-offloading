import torch

import triton
import triton.language as tl


# grid: (BS, local_kv_heads, 2)
@triton.jit
def __shadowkv_gather_kv_cache_kernel_triton(
    page_table,  # (BS, max_seq_len)
    page_table_stride,
    k_cache,  # (num_pages, page_size = 1, local_kv_heads, head_dim)
    k_cache_stride,
    v_cache,  # (num_pages, page_size = 1, local_kv_heads, head_dim)
    v_cache_stride,
    k_buffer_res,  # (max_batch_size * max_num_tokens, 1, local_kv_heads, head_dim)
    k_buffer_res_stride,
    v_buffer_res,  # (max_batch_size * max_num_tokens, 1, local_kv_heads, head_dim)
    v_buffer_res_stride,
    indices,  # int64 tensor (BS, local_kv_heads, max_num_chunks_to_select)
    indices_stride,
    num_tokens_to_gather,  # (BS,)
    num_tokens_to_gather_stride,
    batch_indices,
    batch_indices_stride,
    max_num_tokens,
    HD: tl.constexpr,
):
    pid_x, pid_y, pid_z = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    batch_index = tl.load(batch_indices + pid_x * batch_indices_stride[0])
    num_tokens = tl.load(num_tokens_to_gather + batch_index * num_tokens_to_gather_stride[0])

    sourse_table = page_table + batch_index * page_table_stride[0]
    if pid_z == 0:
        sourse_buffer = k_cache
        sourse_buffer_stride = k_cache_stride
        dst_buffer = (
            k_buffer_res
            + pid_x * max_num_tokens * k_buffer_res_stride[0]
            + pid_y * k_buffer_res_stride[2]
        )
        dst_buffer_stride = k_buffer_res_stride
    else:
        sourse_buffer = v_cache
        sourse_buffer_stride = v_cache_stride
        dst_buffer = (
            v_buffer_res
            + pid_x * max_num_tokens * v_buffer_res_stride[0]
            + pid_y * v_buffer_res_stride[2]
        )
        dst_buffer_stride = v_buffer_res_stride

    indices_ptr = indices + batch_index * indices_stride[0] + pid_y * indices_stride[1]
    offset = tl.arange(0, HD)

    for i in range(num_tokens):
        token_index = tl.load(indices_ptr + i * indices_stride[2])
        token_ptr = tl.load(sourse_table + token_index * page_table_stride[1])
        token = tl.load(
            sourse_buffer
            + token_ptr * sourse_buffer_stride[0]
            + pid_y * sourse_buffer_stride[2]
            + offset
        )
        tl.store(dst_buffer + i * dst_buffer_stride[0] + offset, token)


def shadowkv_gather_kv_cache_kernel_hd128(
    page_table: torch.Tensor,
    k_cache: torch.Tensor,  # contiguous over head dim,
    v_cache: torch.Tensor,
    k_buffer_res: torch.Tensor,
    v_buffer_res: torch.Tensor,
    local_kv_heads: int,
    indices: torch.LongTensor,
    num_tokens_to_gather: torch.Tensor,
    batch_indices: torch.Tensor,
    max_num_tokens,
):
    """
    Tokens will be gathered already sorted for current batch. Meaning that result page_table indices will be [0, 1, ..BS]
    """
    BS, HD = len(batch_indices), 128

    grid = lambda meta: (BS, local_kv_heads, 2)
    __shadowkv_gather_kv_cache_kernel_triton[grid](
        page_table,
        page_table.stride(),
        k_cache,
        k_cache.stride(),
        v_cache,
        v_cache.stride(),
        k_buffer_res,
        k_buffer_res.stride(),
        v_buffer_res,
        v_buffer_res.stride(),
        indices,
        indices.stride(),
        num_tokens_to_gather,
        num_tokens_to_gather.stride(),
        batch_indices,
        batch_indices.stride(),
        max_num_tokens,
        HD=HD,
    )
