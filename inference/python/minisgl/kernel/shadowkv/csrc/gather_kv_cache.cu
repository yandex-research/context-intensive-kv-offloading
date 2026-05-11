#include "gather_kv_cache.cuh"

#include <cassert>
#include <variant>

#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>

namespace yakv {

namespace {

__device__ __forceinline__ size_t cu_seqlen_upper_bound(const int* cu_seqlens,
                                                        size_t batch_size,
                                                        int value) {
  size_t first = 0;
  size_t last = batch_size;
  while (first < last) {
    const size_t mid = first + (last - first) / 2;
    if (value < cu_seqlens[mid]) {
      last = mid;
    } else {
      first = mid + 1;
    }
  }

  assert(first >= 1);

  return first;
}

constexpr size_t kBytesPerLoad = 16;

template <size_t HeadSizeBytes>
__global__ void gather_kv_cache_kernel(GatherKVCacheImplParams params) {
  for (size_t cur_block_idx = blockIdx.x;
       cur_block_idx < params.cu_pruned_seq_lens[params.batch_size];
       cur_block_idx += gridDim.x) {
    const size_t batch_idx =
        cu_seqlen_upper_bound(params.cu_pruned_seq_lens, params.batch_size,
                              static_cast<int>(cur_block_idx)) -
        1;
    const size_t dst_token_idx =
        cur_block_idx - params.cu_pruned_seq_lens[batch_idx];

    const size_t kv_idx = threadIdx.y;
    const int pruned_seq_len = params.pruned_seq_lens[batch_idx];
    if (dst_token_idx >= pruned_seq_len) {
      return;
    }

    const bool is_key = threadIdx.z == 0; // we either copy K or V
    const size_t kv_cache_idx =
        params.block_indices ? params.block_indices[batch_idx] : batch_idx;
    const int prefix_len = params.prefix_lens[kv_cache_idx];
    const int infix_len = params.infix_lens[kv_cache_idx];
    const int pruned_infix_len = params.pruned_infix_lens[kv_cache_idx];

    const size_t src_token_idx = [&]() -> size_t {
      if (dst_token_idx < prefix_len) {
        return dst_token_idx;
      } else if (dst_token_idx < prefix_len + pruned_infix_len) {
        const size_t top_landmark_idx =
            (dst_token_idx - prefix_len) / params.chunk_len;
        const size_t token_in_landmark_idx =
            (dst_token_idx - prefix_len) % params.chunk_len;
        const int landmark_idx =
            params.top_landmarks_indices
                [kv_cache_idx * params.top_landmarks_indices_strides[0] +
                 kv_idx * params.top_landmarks_indices_strides[1] +
                 top_landmark_idx];
        return prefix_len + landmark_idx * params.chunk_len +
            token_in_landmark_idx;
      } else {
        return dst_token_idx + (infix_len - pruned_infix_len);
      }
    }();

    const size_t head_offset = threadIdx.x * kBytesPerLoad;

    const uint8_t* src_tensor_ptr =
        is_key ? params.src_k_cache : params.src_v_cache;
    const auto& src_tensor_strides =
        is_key ? params.src_k_cache_strides : params.src_v_cache_strides;
    uint8_t* dst_tensor_ptr = is_key ? params.dst_k_cache : params.dst_v_cache;
    const auto& dst_tensor_strides =
        is_key ? params.dst_k_cache_strides : params.dst_v_cache_strides;

    const uint8_t* src_ptr = src_tensor_ptr +
        kv_cache_idx * src_tensor_strides[0] +
        src_token_idx * src_tensor_strides[1] + kv_idx * src_tensor_strides[2] +
        head_offset;
    uint8_t* dst_ptr = dst_tensor_ptr + kv_cache_idx * dst_tensor_strides[0] +
        dst_token_idx * dst_tensor_strides[1] + kv_idx * dst_tensor_strides[2] +
        head_offset;
    int4 thread_data = *reinterpret_cast<const int4*>(src_ptr);
    *reinterpret_cast<int4*>(dst_ptr) = thread_data;
  }
}

template <size_t... Values>
std::variant<std::integral_constant<size_t, Values>...> make_size_t_variant(
    size_t value) {
  std::variant<std::integral_constant<size_t, Values>...> result;
  bool found = false;
  ([&] {
    if (value == Values) {
      result.template emplace<std::integral_constant<size_t, Values>>();
      found = true;
    }
  }(), ...);
  if (!found) {
    TORCH_CHECK(false, "Invalid mode value provided: ", value);
  }
  return result;
}

} // namespace

void gather_kv_cache_launcher(const GatherKVCacheImplParams& params,
                              cudaStream_t stream) {
  TORCH_CHECK(params.head_size_bytes % kBytesPerLoad == 0,
              "Only head size aligned by 16 bytes is supported");
  std::visit([&](auto head_size_bytes_variant) -> void {
    constexpr size_t kHeadSizeBytes = head_size_bytes_variant.value;
    auto* kernel_instance = gather_kv_cache_kernel<kHeadSizeBytes>;

    const dim3 block_dim{kHeadSizeBytes / kBytesPerLoad,
                         static_cast<unsigned int>(params.num_kv_heads),
                         /* K and V */ 2};
    int num_blocks = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &num_blocks, kernel_instance, block_dim.x * block_dim.y * block_dim.z,
        0);
    const dim3 grid_dim{
        static_cast<unsigned>(num_blocks) *
            static_cast<unsigned>(params.num_sms),
        1,
        1};

    kernel_instance<<<grid_dim, block_dim, 0, stream>>>(params);
  }, make_size_t_variant<64, 128, 256>(params.head_size_bytes));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

} // namespace yakv