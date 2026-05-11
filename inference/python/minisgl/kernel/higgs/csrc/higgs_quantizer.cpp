#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include "higgs_quantizer.h"

void higgs_quantize_d4_n256_cuda(QuantizeParams &params, cudaStream_t stream);
void higgs_dequantize_d4_n256_cuda(DequantizeParams &params, cudaStream_t stream, int channel_size);
void higgs_dequantize_d4_n256_full_cuda(DequantizeFullParams &params, cudaStream_t stream);

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
    // at::Tensor &add_prediction,
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
    // params.add_ptr = add_prediction.data_ptr();
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


// void higgs_dequantize_full(at::Tensor &index, at::Tensor &scales, at::Tensor &lattice, at::Tensor &add_prediction, at::Tensor &out, float hadamard_scale) {
void higgs_dequantize_full(at::Tensor &index, at::Tensor &scales, at::Tensor &lattice, at::Tensor &out, float hadamard_scale) {
    at::ScalarType index_type = index.scalar_type();
    at::ScalarType scales_type = scales.scalar_type();
    at::ScalarType lattice_type = lattice.scalar_type();
    // at::ScalarType add_prediction_type = add_prediction.scalar_type();
    at::ScalarType out_type = out.scalar_type();

    TORCH_CHECK(index_type == at::ScalarType::Byte);
    TORCH_CHECK(scales_type == at::ScalarType::BFloat16);
    TORCH_CHECK(lattice_type == at::ScalarType::BFloat16);
    // TORCH_CHECK(add_prediction_type == at::ScalarType::BFloat16);
    TORCH_CHECK(out_type == at::ScalarType::BFloat16);
    
    TORCH_CHECK(index.is_cuda());
    TORCH_CHECK(scales.is_cuda());
    TORCH_CHECK(lattice.is_cuda());
    // TORCH_CHECK(add_prediction.is_cuda());
    TORCH_CHECK(out.is_cuda());

    TORCH_CHECK(index.is_contiguous());
    // TORCH_CHECK(add_prediction.is_contiguous());

    int n = lattice.size(0);
    int d = lattice.size(1);

    TORCH_CHECK(out.dim() == 3);
    TORCH_CHECK(index.dim() == 2);
    // TORCH_CHECK(add_prediction.dim() == 2);

    int flattened_batch_size = index.size(0);
    int quantized_channel_size = index.size(1);

    // TORCH_CHECK(flattened_batch_size == add_prediction.size(0));

    int batch_size = out.size(0);
    int n_tokens = out.size(1);
    int channel_size = out.size(2);

    TORCH_CHECK(flattened_batch_size == batch_size * n_tokens);
    TORCH_CHECK(quantized_channel_size == channel_size / d);

    int device_index = index.get_device();
    at::Device device = index.device();
    
    TORCH_CHECK(device_index == lattice.get_device());
    TORCH_CHECK(device_index == scales.get_device());
    // TORCH_CHECK(device_index == add_prediction.get_device());
    TORCH_CHECK(device_index == out.get_device());
    
    TORCH_CHECK(device == lattice.device());
    TORCH_CHECK(device == scales.device());
    // TORCH_CHECK(device == add_prediction.device());
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
        // add_prediction,
        out
    );

    at::cuda::CUDAGuard device_guard{(char)device_index};
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    higgs_dequantize_d4_n256_full_cuda(params, stream);
}

void init_higgs_lib(py::module &m) {
    m.def("higgs_quantize", &higgs_quantize_d4_n256, "quantization");
    m.def("higgs_dequantize", &higgs_dequantize_d4_n256, "dequantization");
    m.def("higgs_dequantize_full", &higgs_dequantize_full, "full_dequantization");
}
