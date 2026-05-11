#include "metadata_kernels.cuh"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <tuple>

#include <c10/cuda/CUDAException.h>

namespace yakv {

namespace {

__device__ __forceinline__ std::tuple<size_t, size_t, size_t>
find_prefix_infix_and_pruned_infix_lengths(size_t full_seq_len,
                                           size_t chunk_len,
                                           float sink_chunks_ratio,
                                           float sparse_chunks_ratio,
                                           float local_chunks_ratio,
                                           size_t min_seq_len_to_prune) {
  if (full_seq_len <= min_seq_len_to_prune) {
    return {full_seq_len, 0, 0};
  }

  size_t prefix_len =
      std::min(static_cast<size_t>(std::ceil(sink_chunks_ratio * full_seq_len)),
               full_seq_len);
  size_t remaining = full_seq_len - prefix_len;
  if (remaining == 0) {
    return {prefix_len, 0, 0};
  }
  size_t suffix_len = std::min(
      static_cast<size_t>(std::ceil(local_chunks_ratio * full_seq_len)),
      remaining);
  remaining -= suffix_len;
  size_t infix_len = remaining;
  const size_t infix_extra = infix_len % chunk_len;
  if (infix_extra != 0) {
    suffix_len += infix_extra;
    infix_len -= infix_extra;
  }

  size_t pruned_infix_len = std::min(
      (static_cast<size_t>(sparse_chunks_ratio * full_seq_len) / chunk_len) *
          chunk_len, infix_len);
  const size_t pruned_seq_len = prefix_len + pruned_infix_len + suffix_len;
  if (pruned_seq_len < full_seq_len && pruned_seq_len < min_seq_len_to_prune) {
    const size_t min_pruned_infix_len =
        min_seq_len_to_prune - prefix_len - suffix_len;
    const size_t extra = min_pruned_infix_len % chunk_len;
    pruned_infix_len = min_pruned_infix_len -
        extra; // Make sure min_pruned_infix_len is aligned to chunk len
  }

  assert(prefix_len + pruned_infix_len + suffix_len <= full_seq_len);
  assert(prefix_len + infix_len + suffix_len == full_seq_len);
  return {prefix_len, infix_len, pruned_infix_len};
}

__global__ void shadowkv_fill_prefill_state_kernel(
    const FillPrefillMetadataParams params) {
  for (size_t batch_idx = threadIdx.x; batch_idx < params.batch_size;
       batch_idx += blockDim.x) {
    const size_t kv_cache_idx =
        params.block_indices ? params.block_indices[batch_idx] : batch_idx;
    const auto [prefix_len, infix_len, pruned_infix_len] =
        find_prefix_infix_and_pruned_infix_lengths(
            params.full_seq_lens[batch_idx], params.chunk_len,
            params.sink_chunks_ratio, params.sparse_chunks_ratio,
            params.local_chunks_ratio, params.min_seq_len_to_prune);
    params.prefix_lens[kv_cache_idx] = prefix_len;
    params.infix_lens[kv_cache_idx] = infix_len;
    params.pruned_infix_lens[kv_cache_idx] = pruned_infix_len;
  }
}

} // namespace

void fill_prefill_metadata_launcher(const FillPrefillMetadataParams& params,
                                    cudaStream_t stream) {
  shadowkv_fill_prefill_state_kernel<<<1, 128, 0, stream>>>(params);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

namespace {

__global__ void shadowkv_fill_decode_state_kernel(
    const FillDecodeMetadataParams params) {
  params.cu_pruned_seq_lens[0] = 0;
  for (size_t batch_idx = 0; batch_idx < params.batch_size; ++batch_idx) {
    const size_t kv_idx = params.block_indices[batch_idx];
    params.pruned_seq_lens[batch_idx] = params.full_seq_lens[batch_idx] -
        params.infix_lens[kv_idx] + params.pruned_infix_lens[kv_idx];
    params.cu_pruned_seq_lens[batch_idx + 1] =
        params.pruned_seq_lens[batch_idx] +
        params.cu_pruned_seq_lens[batch_idx];
  }
}

} // namespace

void fill_decode_metadata_launcher(const FillDecodeMetadataParams& params,
                                   cudaStream_t stream) {
  shadowkv_fill_decode_state_kernel<<<1, 1, 0, stream>>>(params);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

} // namespace yakv