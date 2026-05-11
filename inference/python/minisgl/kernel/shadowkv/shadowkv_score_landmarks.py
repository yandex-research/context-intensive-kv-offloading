import math
import torch

import triton
import triton.language as tl


@triton.jit
def __shadowkv_score_landmarks_kernel_triton(
    query_states,  # (BS, num_heads, 1, head_dim)
    query_states_stride,
    landmarks,  # (BS, num_heads, max_num_chunks, head_dim) we should transpose in mind
    landmarks_stride,
    scores,  # (BS, num_heads, max_num_chunks)
    scores_stride,
    num_chunks,
    num_chunks_stride,
    batch_indices,
    batch_indices_stride,
    HD: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    All buffers must be contiguous.
    """
    pid_x, pid_y, pid_z = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    sl_ptr = tl.arange(0, BLOCK_SIZE) + BLOCK_SIZE * pid_z

    batch_index = tl.load(batch_indices + pid_x * batch_indices_stride[0])
    mask = sl_ptr < tl.load(num_chunks + batch_index * num_chunks_stride[0])

    query_ptr = query_states + pid_x * query_states_stride[0] + pid_y * query_states_stride[1]
    landmark_ptr = (
        landmarks
        + batch_index * landmarks_stride[0]
        + pid_y * landmarks_stride[1]
        + sl_ptr[None, :] * landmarks_stride[2]
    )
    score_ptr = scores + batch_index * scores_stride[0] + pid_y * scores_stride[1] + sl_ptr[None, :]

    offsets = tl.arange(0, HD)
    x = tl.load(query_ptr + offsets[None, :])
    y = tl.load(landmark_ptr + offsets[:, None], mask=mask[None, :])

    tl.store(score_ptr, tl.dot(x, y), mask=mask[None, :])


def shadowkv_score_landmarks_kernel_hd128(
    query_states: torch.Tensor,
    scores_output: torch.Tensor,
    landmarks: torch.Tensor,
    local_kv_heads: int,
    num_chunks: torch.Tensor,
    max_num_chunks: int,
    batch_indices: torch.Tensor,
    # device: torch.device,
    # dtype: torch.dtype,
) -> None:
    BS, HD = len(batch_indices), 128

    query_states = query_states.to(landmarks.dtype).contiguous()
    assert landmarks.stride(-1) == 1

    assert query_states.shape == (BS, local_kv_heads, 1, HD), (
        query_states.shape,
        (BS, local_kv_heads, 1, HD),
    )

    grid = lambda meta: (BS, local_kv_heads, math.ceil(max_num_chunks / meta['BLOCK_SIZE']))

    __shadowkv_score_landmarks_kernel_triton[grid](
        query_states,
        query_states.stride(),
        landmarks,
        landmarks.stride(),
        scores_output,
        scores_output.stride(),
        num_chunks,
        num_chunks.stride(),
        batch_indices,
        batch_indices.stride(),
        HD=HD,
        BLOCK_SIZE=64
    )

    # scores *= (HD ** -.5)

    # return scores
