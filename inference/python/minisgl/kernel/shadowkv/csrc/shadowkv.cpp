#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>

#include "gather_kv_cache.cuh"
#include "metadata_kernels.cuh"

namespace yakv {

void fill_prefill_metadata(torch::Tensor prefix_lens, torch::Tensor infix_lens,
                           torch::Tensor pruned_infix_lens,
                           torch::Tensor seq_lens, torch::Tensor batch_indices,
                           float sink_chunks_ratio, float sparse_chunks_ratio,
                           float local_chunks_ratio,
                           size_t min_seq_len_to_prune, size_t chunk_len) {
  fill_prefill_metadata_launcher(
      FillPrefillMetadataParams{
          .prefix_lens = prefix_lens.data_ptr<int>(),
          .infix_lens = infix_lens.data_ptr<int>(),
          .pruned_infix_lens = pruned_infix_lens.data_ptr<int>(),
          .sink_chunks_ratio = sink_chunks_ratio,
          .sparse_chunks_ratio = sparse_chunks_ratio,
          .local_chunks_ratio = local_chunks_ratio,
          .min_seq_len_to_prune = min_seq_len_to_prune,
          .chunk_len = chunk_len,
          .block_indices = batch_indices.data_ptr<int>(),
          .full_seq_lens = seq_lens.data_ptr<int>(),
          .batch_size = batch_indices.size(0),
      }, c10::cuda::getCurrentCUDAStream());
}

void fill_decode_metadata(torch::Tensor prefix_lens, torch::Tensor infix_lens,
                          torch::Tensor pruned_infix_lens,
                          torch::Tensor seq_lens, torch::Tensor batch_indices,
                          torch::Tensor pruned_seq_lens,
                          torch::Tensor cu_pruned_seq_lens) {
  fill_decode_metadata_launcher(
      FillDecodeMetadataParams{
          .prefix_lens = prefix_lens.data_ptr<int>(),
          .infix_lens = infix_lens.data_ptr<int>(),
          .pruned_infix_lens = pruned_infix_lens.data_ptr<int>(),
          .block_indices = batch_indices.data_ptr<int>(),
          .full_seq_lens = seq_lens.data_ptr<int>(),
          .pruned_seq_lens = pruned_seq_lens.data_ptr<int>(),
          .cu_pruned_seq_lens = cu_pruned_seq_lens.data_ptr<int>(),
          .batch_size = batch_indices.size(0),
      }, c10::cuda::getCurrentCUDAStream());
}

void gather_kv_cache(torch::Tensor prefix_lens, torch::Tensor infix_lens,
                     torch::Tensor pruned_infix_lens,
                     torch::Tensor batch_indices, torch::Tensor pruned_seq_lens,
                     torch::Tensor cu_pruned_seq_lens,
                     torch::Tensor selected_chunks, torch::Tensor src_k_cache,
                     torch::Tensor src_v_cache, torch::Tensor out_k_cache,
                     torch::Tensor out_v_cache, size_t chunk_len) {
  const size_t el_size = src_k_cache.element_size();
  gather_kv_cache_launcher(
      GatherKVCacheImplParams{
          .src_k_cache = reinterpret_cast<uint8_t*>(src_k_cache.data_ptr()),
          .src_k_cache_strides =
              {
                  src_k_cache.stride(0) * el_size,
                  src_k_cache.stride(1) * el_size,
                  src_k_cache.stride(2) * el_size,
                  src_k_cache.stride(3) * el_size,
              },
          .src_v_cache = reinterpret_cast<uint8_t*>(src_v_cache.data_ptr()),
          .src_v_cache_strides =
              {
                  src_v_cache.stride(0) * el_size,
                  src_v_cache.stride(1) * el_size,
                  src_v_cache.stride(2) * el_size,
                  src_v_cache.stride(3) * el_size,
              },
          .top_landmarks_indices = selected_chunks.data_ptr<int64_t>(),
          .top_landmarks_indices_strides = {selected_chunks.stride(0),
                                            selected_chunks.stride(1),
                                            selected_chunks.stride(2)},
          .batch_size = pruned_seq_lens.size(0),
          .num_kv_heads = src_k_cache.size(2),
          .chunk_len = chunk_len,
          .head_size_bytes = src_k_cache.size(3) * el_size,
          .prefix_lens = prefix_lens.data_ptr<int>(),
          .infix_lens = infix_lens.data_ptr<int>(),
          .pruned_infix_lens = pruned_infix_lens.data_ptr<int>(),
          .pruned_seq_lens = pruned_seq_lens.data_ptr<int>(),
          .cu_pruned_seq_lens = cu_pruned_seq_lens.data_ptr<int>(),
          .block_indices = batch_indices.data_ptr<int>(),
          .dst_k_cache = reinterpret_cast<uint8_t*>(out_k_cache.data_ptr()),
          .dst_k_cache_strides =
              {
                  out_k_cache.stride(0) * el_size,
                  out_k_cache.stride(1) * el_size,
                  out_k_cache.stride(2) * el_size,
                  out_k_cache.stride(3) * el_size,
              },
          .dst_v_cache = reinterpret_cast<uint8_t*>(out_v_cache.data_ptr()),
          .dst_v_cache_strides =
              {
                  out_v_cache.stride(0) * el_size,
                  out_v_cache.stride(1) * el_size,
                  out_v_cache.stride(2) * el_size,
                  out_v_cache.stride(3) * el_size,
              },
          .num_sms =
              at::cuda::getCurrentDeviceProperties()->multiProcessorCount,
      }, c10::cuda::getCurrentCUDAStream());
}

torch::Tensor map_to_gpu(torch::Tensor tensor) {
  TORCH_CHECK(tensor.device().is_cpu(), "Tensor must be on CPU");

  void* base_ptr = tensor.storage().data_ptr().get();
  size_t size_bytes = tensor.storage().nbytes();

  cudaError_t err =
      cudaHostRegister(base_ptr, size_bytes, cudaHostRegisterMapped);

  bool needs_unregister = false;
  if (err == cudaSuccess) {
    needs_unregister = true;
  } else if (err == cudaErrorHostMemoryAlreadyRegistered) {
    cudaGetLastError();
  } else {
    TORCH_CHECK(false, "cudaHostRegister: ", cudaGetErrorString(err));
  }

  void* data_ptr = tensor.data_ptr();

  torch::Tensor* keep_alive = new torch::Tensor(tensor);

  auto deleter = [keep_alive, base_ptr, needs_unregister](void* /*ptr*/) {
    if (needs_unregister) {
      cudaHostUnregister(base_ptr);
    }
    delete keep_alive;
  };

  auto options = torch::TensorOptions()
                     .dtype(tensor.dtype())
                     .device(torch::kCUDA)
                     .layout(tensor.layout());

  torch::Tensor gpu_tensor = torch::from_blob(
      data_ptr, tensor.sizes(), tensor.strides(), deleter, options);

  return gpu_tensor;
}

} // namespace yakv


void init_gather_lib(py::module &m) {
  m.def("gather_kv_cache", &yakv::gather_kv_cache, "gather_kv_cache");
  m.def("fill_prefill_metadata", &yakv::fill_prefill_metadata,
        "fill_prefill_metadata");
  m.def("fill_decode_metadata", &yakv::fill_decode_metadata,
        "fill_decode_metadata");
  m.def("map_to_gpu", &yakv::map_to_gpu, "map_to_gpu");
}
