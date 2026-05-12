#pragma once

#include <cstddef>

#include <cuda_runtime_api.h>

namespace yakv {

struct FillPrefillMetadataParams {
  int* prefix_lens = nullptr;
  int* infix_lens = nullptr;
  int* pruned_infix_lens = nullptr;
  float sink_chunks_ratio = 0.0f;
  float sparse_chunks_ratio = 0.0f;
  float local_chunks_ratio = 0.0f;
  size_t min_seq_len_to_prune = 0;
  size_t chunk_len = 0;
  const int* block_indices = nullptr;
  const int* full_seq_lens = nullptr;
  size_t batch_size = 0;
};

void fill_prefill_metadata_launcher(const FillPrefillMetadataParams& params,
                                    cudaStream_t stream);

struct FillDecodeMetadataParams {
  const int* prefix_lens = nullptr;
  const int* infix_lens = nullptr;
  const int* pruned_infix_lens = nullptr;
  const int* block_indices = nullptr;
  const int* full_seq_lens = nullptr;
  int* pruned_seq_lens = nullptr;
  int* cu_pruned_seq_lens = nullptr;
  size_t batch_size = 0;
};

void fill_decode_metadata_launcher(const FillDecodeMetadataParams& params,
                                   cudaStream_t stream);

} // namespace yakv