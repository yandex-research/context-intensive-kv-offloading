#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

#include <cuda_bf16.h>

namespace yakv {

struct GatherKVCacheImplParams {
  const uint8_t* src_k_cache;
  std::array<size_t, 4> src_k_cache_strides;
  const uint8_t* src_v_cache;
  std::array<size_t, 4> src_v_cache_strides;
  const int64_t* top_landmarks_indices = nullptr;
  std::array<size_t, 3> top_landmarks_indices_strides;
  size_t batch_size = 0;
  size_t num_kv_heads = 0;
  size_t chunk_len = 0;
  size_t head_size_bytes = 0;
  const int* prefix_lens = nullptr;
  const int* infix_lens = nullptr;
  const int* pruned_infix_lens = nullptr;
  const int* pruned_seq_lens = nullptr;
  const int* cu_pruned_seq_lens = nullptr;
  const int* block_indices = nullptr;
  uint8_t* dst_k_cache = nullptr;
  std::array<size_t, 4> dst_k_cache_strides;
  uint8_t* dst_v_cache = nullptr;
  std::array<size_t, 4> dst_v_cache_strides;
  int* src_page_table = nullptr;
  std::array<size_t, 2> src_page_table_strides;
  int* dst_page_table = nullptr;
  std::array<size_t, 2> dst_page_table_strides;
  size_t num_sms = 0;
};

void gather_kv_cache_launcher(const GatherKVCacheImplParams& params,
                              cudaStream_t stream);

} // namespace yakv