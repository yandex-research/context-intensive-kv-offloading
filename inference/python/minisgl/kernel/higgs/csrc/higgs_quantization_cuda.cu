#include <iostream>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

#include "higgs_quantizer.h"

#define DISPATCH_ROW_SIZE(CHANNEL_SIZE, ...) \
    if (CHANNEL_SIZE == 128) { \
        using kQuantizeTraits = quantize_traits<128>; \
        __VA_ARGS__(); \
    } else if (CHANNEL_SIZE == 256) { \
        using kQuantizeTraits = quantize_traits<256>; \
        __VA_ARGS__(); \
    } else if (CHANNEL_SIZE == 512) { \
        using kQuantizeTraits = quantize_traits<512>; \
        __VA_ARGS__(); \
    } else if (CHANNEL_SIZE == 1024) { \
        using kQuantizeTraits = quantize_traits<1024>; \
        __VA_ARGS__(); \
    } else if (CHANNEL_SIZE == 2048) { \
        using kQuantizeTraits = quantize_traits<2048>; \
        __VA_ARGS__(); \
    } else if (CHANNEL_SIZE == 4096) { \
        using kQuantizeTraits = quantize_traits<4096>; \
        __VA_ARGS__(); \
    } else if (CHANNEL_SIZE == 8192) { \
        using kQuantizeTraits = quantize_traits<8192>; \
        __VA_ARGS__(); \
    }

template<int kChannelSize_>
struct quantize_traits {
    static constexpr int kThreadsPerThreadBlock = 256;
    static constexpr int kBytesPerElem = 2;
    static constexpr int kChannelSize = kChannelSize_;
    static constexpr int kSharedMemValues = kChannelSize * kBytesPerElem;
    static constexpr int kSharedMemReduce = kThreadsPerThreadBlock / kWarpSize * sizeof(float) * 2; // 2 - both values and indexes
};

template <typename KLatticeTraits, typename kQuantizeTraits>
__global__ __launch_bounds__(256, 8)
void higgs_quantize_d4_n256_kernel(QuantizeParams params) {
    extern __shared__ __align__(16) char smem_[];
    auto smem_lattice = reinterpret_cast<bfloat4*>(smem_);
    auto smem_values = reinterpret_cast<float4*>(smem_ + KLatticeTraits::kLatticeBytes);
    auto smem_reduce = reinterpret_cast<float2*>(smem_ + KLatticeTraits::kLatticeBytes + kQuantizeTraits::kSharedMemValues);
    auto smem_save = reinterpret_cast<uint8_t*>(smem_ + KLatticeTraits::kLatticeBytes + kQuantizeTraits::kSharedMemValues + kQuantizeTraits::kSharedMemReduce);
    auto smem_out = reinterpret_cast<int4*>(smem_save);

    auto lattice_read = reinterpret_cast<float4*>(params.lattice_ptr);
    auto x_read = reinterpret_cast<float4*>(params.x_ptr);
    auto idx_write = reinterpret_cast<int4*>(params.idx_ptr);

    #pragma unroll
    for (int i = threadIdx.x; i < KLatticeTraits::kLoadsPerLattice; i += kQuantizeTraits::kThreadsPerThreadBlock) {
        assert(16 * i < KLatticeTraits::kLatticeBytes);
        auto lattice_values = lattice_read[i];
        constexpr int loaded_lattice_elements = kBytesPerLoad / KLatticeTraits::kSmemWordSize; 
        #pragma unroll
        for (int j = 0; j < loaded_lattice_elements; ++j) {
            auto lattice_value = reinterpret_cast<bfloat4*>(&lattice_values)[j];
            int smem_location = i * loaded_lattice_elements + j; 
            smem_lattice[smem_location] = lattice_value;
        }
    }
    
    int min_rows_per_thread_block = params.batch / gridDim.x;
    int extra_row = blockIdx.x < (params.batch - min_rows_per_thread_block * gridDim.x);
    
    // only channel_size <= 8192 will work because 1024 threads * 16 bytes per load = 16k loaded bytes / 2 bytes per element = 8k elements
    constexpr int threads_to_load = kQuantizeTraits::kChannelSize * kQuantizeTraits::kBytesPerElem / sizeof(float4);
    constexpr int kItersOfLoading = (threads_to_load + kQuantizeTraits::kThreadsPerThreadBlock - 1) / kQuantizeTraits::kThreadsPerThreadBlock;
    // TODO try inner for with 2/4 iters to unroll it
    for (int i = 0; i < min_rows_per_thread_block + extra_row; ++i) {
        int row_index;
        int row_shift;
        if (extra_row > 0 && i == min_rows_per_thread_block) {
            row_index = gridDim.x * min_rows_per_thread_block + blockIdx.x; // TODO
        } else {
            row_index = blockIdx.x * min_rows_per_thread_block + i;
        }
        row_shift = row_index * kQuantizeTraits::kChannelSize * kQuantizeTraits::kBytesPerElem / sizeof(float4);
        for (int j = 0; j < kItersOfLoading; ++j) {
            int shift_within_row = j * kQuantizeTraits::kThreadsPerThreadBlock + threadIdx.x;
            if (shift_within_row < threads_to_load) {
                assert(row_index < params.batch);
                assert(16 * (row_shift + shift_within_row) < params.batch * kQuantizeTraits::kChannelSize * 2);
                assert(16 * shift_within_row < kQuantizeTraits::kSharedMemValues);
                smem_values[shift_within_row] = x_read[row_shift + shift_within_row]; 
            }
        }
        auto smem_values_read = reinterpret_cast<bfloat4*>(smem_values);
        // kNThreadGroups
        int thread_group = threadIdx.x / KLatticeTraits::kN;
        int thread_shift = threadIdx.x % KLatticeTraits::kN;
        __syncthreads();
        
        #pragma unroll
        for (int j = 0; j < kQuantizeTraits::kChannelSize / KLatticeTraits::kD; ++j) {
            bfloat4 lattice_value = smem_lattice[thread_shift];
            bfloat4 x_value = smem_values_read[j * KLatticeTraits::kD * kQuantizeTraits::kBytesPerElem / kSmemWordSize + thread_group];
            float2 dot_norm = dot_and_norm(x_value, lattice_value);
            float res = -2 * dot_norm.x + dot_norm.y;
            uint8_t index = thread_shift;
            #pragma unroll
            float other_res;
            uint8_t other_index;
            for (int k = 1; k < kWarpSize; k *= 2) {
                other_res = __shfl_down_sync(0xffffffff, res, k);
                other_index = __shfl_down_sync(0xffffffff, index, k);
                if (other_res < res) {
                    res = other_res;
                    index = other_index;
                }
            }

            __syncwarp();
            if (threadIdx.x % kWarpSize == 0) {
                smem_reduce[threadIdx.x / kWarpSize] = {res, float(index)};
            }
            __syncthreads();
            
            constexpr int kValuesPerThreadGroup = KLatticeTraits::kN / kWarpSize; // 256 / 32 = 8
            // 
            if (threadIdx.x < kValuesPerThreadGroup) {
                auto res_index = smem_reduce[threadIdx.x];
                res = res_index.x;
                index = uint8_t(res_index.y);

                #pragma unroll
                for (int k = 1; k < kValuesPerThreadGroup; k *= 2) {
                    other_res = __shfl_down_sync(0x000000ff, res, k);
                    other_index = __shfl_down_sync(0x000000ff, index, k);
                    if (other_res < res) {
                        res = other_res;
                        index = other_index;
                    }
                }
            }
            __syncwarp();

            if (threadIdx.x == 0) {
                smem_save[j] = index;
            }
            __syncthreads();
        }
    
        constexpr int kSharedMemOutput = kQuantizeTraits::kChannelSize / KLatticeTraits::kD;
        #pragma unroll
        for (int j = threadIdx.x; j < kSharedMemOutput / kBytesPerLoad; j += kQuantizeTraits::kThreadsPerThreadBlock) {
            idx_write[row_index * kSharedMemOutput / kBytesPerLoad + j] = smem_out[j];
        }
    }
}

template <typename kQuantizeTraits>
void higgs_quantize_d4_n256_cuda_launch(QuantizeParams &params, cudaStream_t stream) {
    constexpr int sms = 108;
    using kLatticeTraits = lattice_traits;

    constexpr int kSharedMemOutput = kQuantizeTraits::kChannelSize / kLatticeTraits::kD;
    constexpr int kSharedMemTotal = kLatticeTraits::kLatticeBytes + kQuantizeTraits::kSharedMemValues + kQuantizeTraits::kSharedMemReduce + kSharedMemOutput;

    auto kernel = &higgs_quantize_d4_n256_kernel<kLatticeTraits, kQuantizeTraits>;
    
    cudaFuncSetAttribute(
        kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSharedMemTotal
    );
    // cudaFuncSetCacheConfig(kernel, cudaFuncCachePreferShared);
    kernel<<<8 * sms, kQuantizeTraits::kThreadsPerThreadBlock, kSharedMemTotal, stream>>>(params);

    C10_CUDA_KERNEL_LAUNCH_CHECK();
    // cudaDeviceSynchronize();
}

void higgs_quantize_d4_n256_cuda(QuantizeParams &params, cudaStream_t stream) {
    cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);
    
    DISPATCH_ROW_SIZE(params.channel_size, [&] {
        higgs_quantize_d4_n256_cuda_launch<kQuantizeTraits>(params, stream);
    });
}