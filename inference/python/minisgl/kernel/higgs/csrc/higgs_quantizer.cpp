#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include "higgs_quantizer.h"
#include "c10/util/Exception.h"

void higgs_quantize_d4_n256_cuda(QuantizeParams &params, cudaStream_t stream);
void higgs_dequantize_d4_n256_cuda(DequantizeParams &params, cudaStream_t stream, int channel_size);
void higgs_dequantize_d4_n256_full_cuda(DequantizeFullParams &params, cudaStream_t stream);
void LandmarksScore(at::Tensor &landmarks, at::Tensor &scales, at::Tensor &cum_lengths, at::Tensor &block_indices, at::Tensor &lattice, at::Tensor &query, at::Tensor &out, float hadamard_scale);
void HeadsQuantize(at::Tensor &x, at::Tensor &cum_lengths, at::Tensor &block_indices, at::Tensor &lattice, at::Tensor &idx, at::Tensor &scales, bool heads_first);
void HeadsDequantize(at::Tensor &idx, at::Tensor &scales, at::Tensor &cum_lengths, at::Tensor &block_indices, at::Tensor &lattice, at::Tensor &out, float hadamard_scale, bool heads_first);

void set_quantize_params(
    QuantizeParams &params,
    at::Tensor &x,
    at::Tensor &lattice,
    at::Tensor &quantized, 
    int batch_size,
    int channel_size
) {
    memset(&params, 0, sizeof(params));

    params.batch = batch_size;
    params.channel_size = channel_size;

    params.x_ptr = x.data_ptr();
    params.lattice_ptr = lattice.data_ptr();
    params.idx_ptr = quantized.data_ptr();
}

void set_dequantize_params(
    DequantizeParams &params, 
    at::Tensor &index, 
    at::Tensor &scales, 
    at::Tensor &lattice,
    at::Tensor &dequantized, 
    int batch_size
) {
    memset(&params, 0, sizeof(params));

    params.batch = batch_size;

    // uintptr_t ptr = reinterpret_cast<uintptr_t>(lattice.data_ptr());
    // printf("ptr adrress %llu\n", ptr);
    // TORCH_CHECK(ptr % alignof(float4) == 0, "lattice_ptr is not 16-byte aligned");
    
    params.idx_ptr         = index.data_ptr();
    params.scales_ptr      = scales.data_ptr();
    params.lattice_ptr     = lattice.data_ptr();
    params.dequantized_ptr = dequantized.data_ptr();

    // uintptr_t ptr = reinterpret_cast<uintptr_t>(dequantized.data_ptr());
    // printf("ptr adrress %llu\n", ptr);
    // TORCH_CHECK(ptr % alignof(float4) == 0, "dequantized_ptr is not 16-byte aligned");

    params.idx_batch_stride   = index.stride(0);
    params.lattice_n_stride   = lattice.stride(0);
    params.dequantized_stride = dequantized.stride(0);
}

void set_dequantize_full_params(
    DequantizeFullParams &params,
    int flatten_batch, 
    int n_tokens,
    int channel_size,
    int n,
    int d,
    float hadamard_scale,
    at::Tensor &index,
    at::Tensor &scales,
    at::Tensor &lattice,
    at::Tensor &add_prediction,
    at::Tensor &out
) {
    memset(&params, 0, sizeof(params));
    
    params.n_tokens = n_tokens;
    params.flatten_batch = flatten_batch;
    params.channel_size = channel_size;
    params.n = n;
    params.d = d;

    params.hadamard_scale = hadamard_scale;
    
    int out_element_size = out.element_size();
    params.out_token_stride = out.stride(1) * out_element_size;
    params.out_batch_stride = out.stride(0) * out_element_size;

    params.idx_ptr = index.data_ptr();
    params.scales_ptr = scales.data_ptr();
    params.lattice_ptr = lattice.data_ptr();
    params.add_ptr = add_prediction.data_ptr();
    params.out_ptr = out.data_ptr();
}


at::Tensor higgs_dequantize_d4_n256(at::Tensor &index, at::Tensor &scales, at::Tensor &lattice) {
    at::ScalarType index_type  = index.scalar_type();
    at::ScalarType scales_type = scales.scalar_type();
    at::ScalarType lattice_type = lattice.scalar_type();

    int device_index = index.get_device();
    at::Device device = index.device();
    
    TORCH_CHECK(index.is_cuda());
    TORCH_CHECK(scales.is_cuda());
    TORCH_CHECK(lattice.is_cuda());
    
    TORCH_CHECK(device_index == scales.get_device());
    TORCH_CHECK(device_index == lattice.get_device());
    TORCH_CHECK(device == scales.device());
    TORCH_CHECK(device == lattice.device());
    
    TORCH_CHECK(index_type == at::ScalarType::Byte);
    TORCH_CHECK(scales_type == at::ScalarType::BFloat16);
    TORCH_CHECK(lattice_type == at::ScalarType::BFloat16);
    
    TORCH_CHECK(index.size(0) == scales.size(0));
    TORCH_CHECK(index.dim() == 2);
    TORCH_CHECK(scales.dim() == 1);
    TORCH_CHECK(lattice.dim() == 2);
    TORCH_CHECK(lattice.size(0) == 256);
    TORCH_CHECK(lattice.size(1) == 4);

    TORCH_CHECK(index.stride(1) == 1);
    TORCH_CHECK(lattice.stride(1) == 1);
    TORCH_CHECK(lattice.stride(0) == 4); // 16 loaded bytes will end up in 8 elements - 2 values from lattice, so the elements should be contiguous
    
    int batch_size = index.size(0);
    int n_bytes = index.size(1);
    TORCH_CHECK(n_bytes % 16 == 0);
    TORCH_CHECK(n_bytes / 16 <= 1024);
    constexpr int kD = 4;
    int dequantized_channel_size = n_bytes * kD;
    float inf = std::numeric_limits<float>::infinity();
    at::Tensor dequantized = torch::full(
        {batch_size, dequantized_channel_size},
        std::numeric_limits<float>::infinity(),
        torch::TensorOptions().dtype(at::kBFloat16).device(device)
    );
    TORCH_CHECK(dequantized.stride(0) == dequantized_channel_size); 

    DequantizeParams params;
    set_dequantize_params(params, index, scales, lattice, dequantized, batch_size);
    
    at::cuda::CUDAGuard device_guard{(char)index.get_device()};
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    
    higgs_dequantize_d4_n256_cuda(params, stream, dequantized_channel_size);
    
    return dequantized;
}

at::Tensor higgs_quantize_d4_n256(at::Tensor &x, at::Tensor &lattice) {
    at::ScalarType x_type = x.scalar_type();
    TORCH_CHECK(x_type == at::ScalarType::BFloat16);
    TORCH_CHECK(lattice.size(0) == 256);
    TORCH_CHECK(lattice.size(1) == 4);
    
    int channel_size = x.size(1);
    int quantized_size = channel_size / 4;
    TORCH_CHECK(channel_size % 4 == 0);

    int device_index = x.get_device();
    at::Device device = x.device();
    
    TORCH_CHECK(device_index == lattice.get_device());
    TORCH_CHECK(device == lattice.device());

    at::Tensor quantized = torch::zeros(
        {x.size(0), quantized_size},
        torch::TensorOptions().dtype(at::kByte).device(device)
    );
    TORCH_CHECK(quantized.stride(0) == quantized_size); 
    
    QuantizeParams params;
    set_quantize_params(params, x, lattice, quantized, x.size(0), x.size(1));

    at::cuda::CUDAGuard device_guard{(char)x.get_device()};
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    higgs_quantize_d4_n256_cuda(params, stream);

    return quantized;
}


void higgs_dequantize_full(at::Tensor &index, at::Tensor &scales, at::Tensor &lattice, at::Tensor &add_prediction, at::Tensor &out, float hadamard_scale) {
    at::ScalarType index_type = index.scalar_type();
    at::ScalarType scales_type = scales.scalar_type();
    at::ScalarType lattice_type = lattice.scalar_type();
    at::ScalarType add_prediction_type = add_prediction.scalar_type();
    at::ScalarType out_type = out.scalar_type();

    TORCH_CHECK(index_type == at::ScalarType::Byte);
    TORCH_CHECK(scales_type == at::ScalarType::BFloat16);
    TORCH_CHECK(lattice_type == at::ScalarType::BFloat16);
    TORCH_CHECK(add_prediction_type == at::ScalarType::BFloat16);
    TORCH_CHECK(out_type == at::ScalarType::BFloat16);
    
    TORCH_CHECK(index.is_cuda());
    TORCH_CHECK(scales.is_cuda());
    TORCH_CHECK(lattice.is_cuda());
    TORCH_CHECK(add_prediction.is_cuda());
    TORCH_CHECK(out.is_cuda());

    TORCH_CHECK(index.is_contiguous());
    TORCH_CHECK(add_prediction.is_contiguous());

    int n = lattice.size(0);
    int d = lattice.size(1);

    TORCH_CHECK(out.dim() == 3);
    TORCH_CHECK(index.dim() == 2);
    TORCH_CHECK(add_prediction.dim() == 2);

    int flattened_batch_size = index.size(0);
    int quantized_channel_size = index.size(1);

    TORCH_CHECK(flattened_batch_size == add_prediction.size(0));

    int batch_size = out.size(0);
    int n_tokens = out.size(1);
    int channel_size = out.size(2);

    TORCH_CHECK(flattened_batch_size == batch_size * n_tokens);
    TORCH_CHECK(quantized_channel_size == channel_size / d);

    int device_index = index.get_device();
    at::Device device = index.device();
    
    TORCH_CHECK(device_index == lattice.get_device());
    TORCH_CHECK(device_index == scales.get_device());
    TORCH_CHECK(device_index == add_prediction.get_device());
    TORCH_CHECK(device_index == out.get_device());
    
    TORCH_CHECK(device == lattice.device());
    TORCH_CHECK(device == scales.device());
    TORCH_CHECK(device == add_prediction.device());
    TORCH_CHECK(device == out.device());
    
    DequantizeFullParams params;
    set_dequantize_full_params(
        params, 
        flattened_batch_size,
        n_tokens, 
        quantized_channel_size * d,
        n,
        d,
        hadamard_scale,
        index,
        scales,
        lattice,
        add_prediction,
        out
    );

    at::cuda::CUDAGuard device_guard{(char)device_index};
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    higgs_dequantize_d4_n256_full_cuda(params, stream);
}

void higgs_score_d4_n256(
	at::Tensor &landmarks, // [kv_cache_size, padded_T, n_kv_heads, head_dim // d]
	at::Tensor &scales, // [kv_cache_size, padded_T, n_kv_heads]
	at::Tensor &cum_lengths, // [batch + 1] prefix sums of the (rounded) per-query lengths
	at::Tensor &block_indices, // [batch]: kv-cache entry scored by each query (also the out slot, values distinct)
	at::Tensor &lattice, // [256, 4]
	at::Tensor &query, // [batch, n_q_heads, head_dim]
	at::Tensor &out, // [kv_cache_size, n_kv_heads, padded_T] scores, written at out[block_indices[b]]
	float hadamard_scale
    ) {
    at::ScalarType index_type = landmarks.scalar_type();
    at::ScalarType scales_type = scales.scalar_type();
    at::ScalarType cum_lengths_type = cum_lengths.scalar_type();
    at::ScalarType block_indices_type = block_indices.scalar_type();
    at::ScalarType lattice_type = lattice.scalar_type();
    at::ScalarType query_type = query.scalar_type();
    at::ScalarType out_type = out.scalar_type();

    TORCH_CHECK(index_type == at::ScalarType::Byte);
    TORCH_CHECK(scales_type == at::ScalarType::Float);
    TORCH_CHECK(cum_lengths_type == at::ScalarType::Int);
    TORCH_CHECK(block_indices_type == at::ScalarType::Int);
    TORCH_CHECK(lattice_type == at::ScalarType::BFloat16);
    TORCH_CHECK(query_type == at::ScalarType::BFloat16);
    TORCH_CHECK(out_type == at::ScalarType::BFloat16);
    
    TORCH_CHECK(landmarks.is_cuda());
    TORCH_CHECK(scales.is_cuda());
    TORCH_CHECK(cum_lengths.is_cuda());
    TORCH_CHECK(block_indices.is_cuda());
    TORCH_CHECK(lattice.is_cuda());
    TORCH_CHECK(query.is_cuda());
    TORCH_CHECK(out.is_cuda());

    int device_index = landmarks.get_device();
    at::Device device = landmarks.device();
    TORCH_CHECK(device_index == scales.get_device());
    TORCH_CHECK(device_index == cum_lengths.get_device());
    TORCH_CHECK(device_index == block_indices.get_device());
    TORCH_CHECK(device_index == lattice.get_device());
    TORCH_CHECK(device_index == query.get_device());
    TORCH_CHECK(device_index == out.get_device());
    TORCH_CHECK(device == scales.device());
    TORCH_CHECK(device == cum_lengths.device());
    TORCH_CHECK(device == block_indices.device());
    TORCH_CHECK(device == lattice.device());
    TORCH_CHECK(device == query.device());
    TORCH_CHECK(device == out.device());

    TORCH_CHECK(landmarks.ndimension() == 4);
    TORCH_CHECK(scales.ndimension() == 3);
    TORCH_CHECK(cum_lengths.ndimension() == 1);
    TORCH_CHECK(block_indices.ndimension() == 1);
    TORCH_CHECK(lattice.ndimension() == 2);
    TORCH_CHECK(query.ndimension() == 3, "query must be 3D, got ", query.ndimension(), "D");
    TORCH_CHECK(out.ndimension() == 3, "out must be 3D, got ", out.ndimension(), "D");

    constexpr int kD = 4;

    // index: [kv_cache_size, t, n_kv_head, head_dim / d] -- the whole kv cache; the query batch
    // b can be smaller, block_indices maps each query to its cache entry
    int kv_cache_size = landmarks.size(0);
    int b = query.size(0);
    int t = landmarks.size(1);
    int n_kv_head = landmarks.size(2);
    int head_dim = landmarks.size(3) * kD; // index stores codes, so size(3) == head_dim / d
    TORCH_CHECK(landmarks.is_contiguous()); // batch/token strides are recomputed in the kernel

    // scales: [kv_cache_size, t, n_kv_head] float, contiguous (strides recomputed in the kernel)
    TORCH_CHECK(scales.size(0) == kv_cache_size, "scales.size(0) (", scales.size(0), ") != kv_cache_size (", kv_cache_size, ")");
    TORCH_CHECK(scales.size(1) == t, "scales.size(1) (", scales.size(1), ") != t (", t, ")");
    TORCH_CHECK(scales.size(2) == n_kv_head, "scales.size(2) (", scales.size(2), ") != n_kv_head (", n_kv_head, ")");
    TORCH_CHECK(scales.is_contiguous());

    // cum_lengths: [b + 1] int32 prefix sums (kernel reads cum_lengths[b])
    TORCH_CHECK(cum_lengths.size(0) == b + 1, "cum_lengths.size(0) (", cum_lengths.size(0), ") != b + 1 (", b + 1, ")");
    TORCH_CHECK(cum_lengths.is_contiguous());

    // block_indices: [b] int32; values must be distinct (out is written at out[block_indices[b]])
    // and < kv_cache_size (neither is validated, lives on device)
    TORCH_CHECK(block_indices.size(0) == b, "block_indices.size(0) (", block_indices.size(0), ") != b (", b, ")");
    TORCH_CHECK(block_indices.is_contiguous());

    // lattice: [256, 4]
    TORCH_CHECK(lattice.size(0) == 256, "lattice.size(0) must be 256, got ", lattice.size(0));
    TORCH_CHECK(lattice.size(1) == kD, "lattice.size(1) must be ", kD, ", got ", lattice.size(1));

    // query: [b, n_q_head, head_dim]; batch stride is passed, inner two dims must be packed
    TORCH_CHECK(query.size(2) == head_dim,
                "query last dim (", query.size(2), ") != head_dim (", head_dim, ")");
    TORCH_CHECK(query.size(1) % n_kv_head == 0,
                "query.size(1) (", query.size(1),
                ") must be divisible by n_kv_head (", n_kv_head, ")");
    TORCH_CHECK(query.stride(2) == 1, "query head_dim must be contiguous");
    TORCH_CHECK(query.stride(1) == head_dim, "query heads must be packed (stride(1) == head_dim)");

    // out (scores): [kv_cache_size, n_kv_head, t]; batch stride is passed, inner two dims must be packed
    TORCH_CHECK(out.size(0) == kv_cache_size, "out.size(0) (", out.size(0), ") != kv_cache_size (", kv_cache_size, ")");
    TORCH_CHECK(out.size(1) == n_kv_head, "out.size(1) (", out.size(1), ") != n_kv_head (", n_kv_head, ")");
    TORCH_CHECK(out.size(2) == t, "out.size(2) (", out.size(2), ") != t (", t, ")");
    TORCH_CHECK(out.stride(2) == 1, "out last dim must be contiguous");
    TORCH_CHECK(out.stride(1) == t, "out head stride must be t");

    LandmarksScore(landmarks, scales, cum_lengths, block_indices, lattice, query, out, hadamard_scale);
}

void higgs_quantize_heads(
	at::Tensor &x, // [batch, padded_T, n_kv_heads, head_dim], or [batch, n_kv_heads, padded_T, head_dim] if heads_first
	at::Tensor &cum_lengths, // [batch + 1] int32 prefix sums of the true lengths
	at::Tensor &block_indices, // [batch]: cache slot each input sample is written to (must be distinct)
	at::Tensor &lattice, // [256, 4] (2 bit) or [256, 2] (4 bit)
	at::Tensor &idx, // codes (out), the whole kv cache: [max_batch, max_T, n_kv_heads, head_dim // d] (layout like x)
	at::Tensor &scales, // float norms (out), [max_batch, max_T, n_kv_heads] (layout like x)
	bool heads_first
    ) {
    at::ScalarType x_type = x.scalar_type();
    at::ScalarType block_indices_type = block_indices.scalar_type();
    at::ScalarType cum_lengths_type = cum_lengths.scalar_type();
    at::ScalarType lattice_type = lattice.scalar_type();
    at::ScalarType idx_type = idx.scalar_type();
    at::ScalarType scales_type = scales.scalar_type();

    TORCH_CHECK(x_type == at::ScalarType::BFloat16);
    TORCH_CHECK(cum_lengths_type == at::ScalarType::Int);
    TORCH_CHECK(block_indices_type == at::ScalarType::Int);
    TORCH_CHECK(lattice_type == at::ScalarType::BFloat16);
    TORCH_CHECK(idx_type == at::ScalarType::Byte);
    TORCH_CHECK(scales_type == at::ScalarType::Float);

    TORCH_CHECK(x.is_cuda());
    TORCH_CHECK(block_indices.is_cuda());
    TORCH_CHECK(cum_lengths.is_cuda());
    TORCH_CHECK(lattice.is_cuda());
    TORCH_CHECK(idx.is_cuda());
    TORCH_CHECK(scales.is_cuda());

    int device_index = x.get_device();
    at::Device device = x.device();
    TORCH_CHECK(device_index == cum_lengths.get_device());
    TORCH_CHECK(device_index == block_indices.get_device());
    TORCH_CHECK(device_index == lattice.get_device());
    TORCH_CHECK(device_index == idx.get_device());
    TORCH_CHECK(device_index == scales.get_device());
    TORCH_CHECK(device == cum_lengths.device());
    TORCH_CHECK(device == block_indices.device());
    TORCH_CHECK(device == lattice.device());
    TORCH_CHECK(device == idx.device());
    TORCH_CHECK(device == scales.device());

    TORCH_CHECK(x.ndimension() == 4, "x must be 4D, got ", x.ndimension(), "D");
    TORCH_CHECK(cum_lengths.ndimension() == 1);
    TORCH_CHECK(block_indices.ndimension() == 1);
    TORCH_CHECK(lattice.ndimension() == 2);
    TORCH_CHECK(idx.ndimension() == 4, "idx must be 4D, got ", idx.ndimension(), "D");
    TORCH_CHECK(scales.ndimension() == 3, "scales must be 3D, got ", scales.ndimension(), "D");

    constexpr int kHeadDim = 128;

    // x: [b, t, n_kv_head, head_dim] (or [b, n_kv_head, t, head_dim] if heads_first),
    // strides are recomputed in the kernel
    int b = x.size(0);
    int head_dim = x.size(3);
    TORCH_CHECK(head_dim == kHeadDim, "only head_dim = ", kHeadDim, " is supported, got ", head_dim);
    TORCH_CHECK(x.is_contiguous());

    // cum_lengths: [b + 1] int32 prefix sums (kernel reads cum_lengths[b])
    TORCH_CHECK(cum_lengths.size(0) == b + 1, "cum_lengths.size(0) (", cum_lengths.size(0), ") != b + 1 (", b + 1, ")");
    TORCH_CHECK(cum_lengths.is_contiguous());

    // block_indices: [b] int32; values must be distinct and < idx.size(0) (not validated, lives on device)
    TORCH_CHECK(block_indices.size(0) == b, "block_indices.size(0) (", block_indices.size(0), ") != b (", b, ")");
    TORCH_CHECK(block_indices.is_contiguous());

    // lattice: [256, 4] or [256, 2]
    int d = lattice.size(1);
    TORCH_CHECK(lattice.size(0) == 256, "lattice.size(0) must be 256, got ", lattice.size(0));
    TORCH_CHECK(d == 4 || d == 2, "lattice.size(1) must be 4 or 2, got ", d);
    TORCH_CHECK(lattice.is_contiguous());

    // idx/scales are the preallocated kv cache: same layout as x, but the batch dim is
    // max_batch_size (>= b is not checkable against block_indices values here) and the token
    // dim is max_seq_len (may exceed x's padding; lengths must fit both)
    TORCH_CHECK(idx.size(0) == scales.size(0), "idx.size(0) (", idx.size(0), ") != scales.size(0) (", scales.size(0), ")");
    int n_kv_head = heads_first ? x.size(1) : x.size(2);
    if (heads_first) {
        TORCH_CHECK(idx.size(1) == n_kv_head, "idx.size(1) (", idx.size(1), ") != n_kv_head (", n_kv_head, ")");
        TORCH_CHECK(scales.size(1) == n_kv_head, "scales.size(1) (", scales.size(1), ") != n_kv_head (", n_kv_head, ")");
        TORCH_CHECK(idx.size(2) == scales.size(2), "idx.size(2) (", idx.size(2), ") != scales.size(2) (", scales.size(2), ")");
    } else {
        TORCH_CHECK(idx.size(2) == n_kv_head, "idx.size(2) (", idx.size(2), ") != n_kv_head (", n_kv_head, ")");
        TORCH_CHECK(scales.size(2) == n_kv_head, "scales.size(2) (", scales.size(2), ") != n_kv_head (", n_kv_head, ")");
        TORCH_CHECK(idx.size(1) == scales.size(1), "idx.size(1) (", idx.size(1), ") != scales.size(1) (", scales.size(1), ")");
    }
    TORCH_CHECK(idx.size(3) == head_dim / d, "idx.size(3) (", idx.size(3), ") != head_dim / d (", head_dim / d, ")");
    TORCH_CHECK(idx.is_contiguous());
    TORCH_CHECK(scales.is_contiguous());

    HeadsQuantize(x, cum_lengths, block_indices, lattice, idx, scales, heads_first);
}

void higgs_dequantize_heads(
	at::Tensor &idx, // [max_batch, n_kv_heads, max_T, head_dim // 2] codes (the whole kv cache), [max_batch, max_T, n_kv_heads, head_dim // 2] if not heads_first
	at::Tensor &scales, // [max_batch, n_kv_heads, max_T] float norms (t/h swapped if not heads_first)
	at::Tensor &cum_lengths, // [batch + 1] int32 prefix sums of the active requests' lengths
	at::Tensor &block_indices, // [batch]: kv-cache slots to dequantize (also the out slots, values distinct)
	at::Tensor &lattice, // [256, 2] (4 bit)
	at::Tensor &out, // [max_batch, n_kv_heads, max_T, head_dim], written at out[block_indices[b]] (t/h swapped if not heads_first)
	float hadamard_scale,
	bool heads_first
    ) {
    at::ScalarType idx_type = idx.scalar_type();
    at::ScalarType scales_type = scales.scalar_type();
    at::ScalarType block_indices_type = block_indices.scalar_type();
    at::ScalarType cum_lengths_type = cum_lengths.scalar_type();
    at::ScalarType lattice_type = lattice.scalar_type();
    at::ScalarType out_type = out.scalar_type();

    TORCH_CHECK(idx_type == at::ScalarType::Byte);
    TORCH_CHECK(scales_type == at::ScalarType::Float);
    TORCH_CHECK(cum_lengths_type == at::ScalarType::Int);
    TORCH_CHECK(block_indices_type == at::ScalarType::Int);
    TORCH_CHECK(lattice_type == at::ScalarType::BFloat16);
    TORCH_CHECK(out_type == at::ScalarType::BFloat16);

    TORCH_CHECK(idx.is_cuda());
    TORCH_CHECK(scales.is_cuda());
    TORCH_CHECK(cum_lengths.is_cuda());
    TORCH_CHECK(block_indices.is_cuda());
    TORCH_CHECK(lattice.is_cuda());
    TORCH_CHECK(out.is_cuda());

    int device_index = idx.get_device();
    at::Device device = idx.device();
    TORCH_CHECK(device_index == scales.get_device());
    TORCH_CHECK(device_index == cum_lengths.get_device());
    TORCH_CHECK(device_index == block_indices.get_device());
    TORCH_CHECK(device_index == lattice.get_device());
    TORCH_CHECK(device_index == out.get_device());
    TORCH_CHECK(device == scales.device());
    TORCH_CHECK(device == cum_lengths.device());
    TORCH_CHECK(device == block_indices.device());
    TORCH_CHECK(device == lattice.device());
    TORCH_CHECK(device == out.device());

    TORCH_CHECK(idx.ndimension() == 4, "idx must be 4D, got ", idx.ndimension(), "D");
    TORCH_CHECK(scales.ndimension() == 3, "scales must be 3D, got ", scales.ndimension(), "D");
    TORCH_CHECK(cum_lengths.ndimension() == 1);
    TORCH_CHECK(block_indices.ndimension() == 1);
    TORCH_CHECK(lattice.ndimension() == 2);
    TORCH_CHECK(out.ndimension() == 4, "out must be 4D, got ", out.ndimension(), "D");

    constexpr int kD = 2;
    constexpr int kHeadDim = 128;

    // idx/scales/out all span the whole kv cache; only b = block_indices.size(0) requests are
    // active and only their slots are touched
    int b = block_indices.size(0);
    TORCH_CHECK(idx.size(3) == kHeadDim / kD, "idx.size(3) (", idx.size(3), ") != head_dim / d (", kHeadDim / kD, ")");
    TORCH_CHECK(out.size(3) == kHeadDim, "out.size(3) (", out.size(3), ") != head_dim (", kHeadDim, ")");
    TORCH_CHECK(scales.size(0) == idx.size(0), "scales.size(0) (", scales.size(0), ") != idx.size(0) (", idx.size(0), ")");
    TORCH_CHECK(out.size(0) == idx.size(0), "out.size(0) (", out.size(0), ") != idx.size(0) (", idx.size(0), ")");
    for (int i = 1; i < 3; ++i) {
        TORCH_CHECK(scales.size(i) == idx.size(i), "scales.size(", i, ") (", scales.size(i), ") != idx.size(", i, ") (", idx.size(i), ")");
        TORCH_CHECK(out.size(i) == idx.size(i), "out.size(", i, ") (", out.size(i), ") != idx.size(", i, ") (", idx.size(i), ")");
    }
    TORCH_CHECK(idx.is_contiguous());
    TORCH_CHECK(scales.is_contiguous());
    TORCH_CHECK(out.is_contiguous());

    // cum_lengths: [b + 1] int32 prefix sums (kernel reads cum_lengths[b])
    TORCH_CHECK(cum_lengths.size(0) == b + 1, "cum_lengths.size(0) (", cum_lengths.size(0), ") != b + 1 (", b + 1, ")");
    TORCH_CHECK(cum_lengths.is_contiguous());

    // lattice: [256, 2]
    TORCH_CHECK(lattice.size(0) == 256, "lattice.size(0) must be 256, got ", lattice.size(0));
    TORCH_CHECK(lattice.size(1) == kD, "lattice.size(1) must be ", kD, ", got ", lattice.size(1));
    TORCH_CHECK(lattice.is_contiguous());

    HeadsDequantize(idx, scales, cum_lengths, block_indices, lattice, out, hadamard_scale, heads_first);
}

void init_higgs_lib(py::module &m) {
    m.def("higgs_quantize", &higgs_quantize_d4_n256, "quantization");
    m.def("higgs_dequantize", &higgs_dequantize_d4_n256, "dequantization");
    m.def("higgs_dequantize_full", &higgs_dequantize_full, "full_dequantization");
    m.def("higgs_score", &higgs_score_d4_n256, "landmark scoring");
    m.def("higgs_quantize_heads", &higgs_quantize_heads, "per-head quantization (hadamard size = head_dim)");
    m.def("higgs_dequantize_heads", &higgs_dequantize_heads, "per-head dequantization (hadamard size = head_dim), inverse of higgs_quantize_heads");
}
