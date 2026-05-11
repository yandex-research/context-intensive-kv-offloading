#include <iostream>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include "higgs_quantizer.h"

const int SIZEOF_INT4 = 16;

template<int kChannelSize_, int kN_, int kD_, int kLatticeSharedMemCopy_>
struct higgs_dequantize_d4_n256_full_kernel_traits {
    static constexpr int kNThreads = 1024;

    static constexpr int kChannelSize = kChannelSize_;
    static constexpr int kN = kN_;
    static constexpr int kD = kD_;

    static constexpr int kThreadsPerChannelToLoad = kChannelSize / kD / SIZEOF_INT4;
    static constexpr int kMaxRowsPerThreadBlock = kNThreads / kThreadsPerChannelToLoad;

    // Lattice params
    static constexpr int kLatticeSharedMemCopy = kLatticeSharedMemCopy_;
    static constexpr int kLatticeBytesPerElem = 2;
    static constexpr int kLatticeElemsPerLoad = SIZEOF_INT4 / (kD * kLatticeBytesPerElem);
    static constexpr int kNLoadsForLattice = kD * kN * kLatticeBytesPerElem / SIZEOF_INT4;
    static constexpr int kSharedMemSizeLattice = kD * kN * kLatticeSharedMemCopy * kLatticeBytesPerElem;
    
    static constexpr int kSharedMemSizeSaveIndex = kNThreads * SIZEOF_INT4;
    static constexpr int kScalesBytesPerElem = 2;
    static constexpr int kSharedMemSizeScales = kMaxRowsPerThreadBlock * kScalesBytesPerElem;
    
    // Hadamard params
    static constexpr int kNThreadsPerChannelHadamard = kChannelSize / 8;
    static constexpr int kLogNThreadsPerChannelInWarp = cilog2(constexpr_min(32, kNThreadsPerChannelHadamard));
    static constexpr int kNWarpsPerChannel = kNThreadsPerChannelHadamard / 32;

    static constexpr int kSharedMemSizeHadamardTranspose = kNWarpsPerChannel > 1 ? kNThreads * sizeof(float2) : 0;

    static constexpr int kSharedMemSize = kSharedMemSizeLattice + kSharedMemSizeSaveIndex + kSharedMemSizeScales + kSharedMemSizeHadamardTranspose;
};


__forceinline__ __device__ void hadamard_thread(float dequantized[8]) {
    #pragma unroll
    for (int i = 0; i < 3; ++i) {
        const int stride = 1 << i;
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int lo = j & (stride - 1);
            const int idx = (j - lo) * 2 + lo;
            #pragma unroll
            const float a = dequantized[idx];
            const float b = dequantized[idx + stride];
            dequantized[idx] = a + b;
            dequantized[idx + stride] = a - b;
        }
    }
}

template<int kLogNThreadsPerChannelInWarp> // 16 for channel_size = 128 in Qwen2.5-0.5B, 32 for everything else
__forceinline__ __device__ void hadamard_warp(float dequantized[8]) {
    constexpr int N = 1 << kLogNThreadsPerChannelInWarp;
    int lane_id = threadIdx.x % N;
    #pragma unroll
    for (int step = 0; step < kLogNThreadsPerChannelInWarp; ++step) {
        const int lane_mask = 1 << step;
        const float sign = (lane_id & lane_mask) ? -1.f : 1.f;
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            float x_val_other = __shfl_xor_sync(__activemask(), dequantized[i], lane_mask);        
            dequantized[i] = sign * dequantized[i] + x_val_other;
        }
    }
}

template<int kNThreadsPerChannel, int kNWarpsPerChannel, bool Pre>
inline __device__ void exchange(float dequantized[8], float2 *smem) {
    int channel_idx = threadIdx.x / kNThreadsPerChannel;
    int warp_idx = threadIdx.x / 32 % kNWarpsPerChannel;
    int lane_idx = threadIdx.x % 32;
    int reverse_warp_idx = threadIdx.x % kNWarpsPerChannel;
    int shift_within_warp = threadIdx.x % kNThreadsPerChannel / kNWarpsPerChannel;
    // для d=2 видимо word smem'а будет не по 8 байт, а по 4 поэтому и грузить нужно будет по другому. Хотя из-за этого будет больше синхронизаций
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        // кажется банк-конфликт исправляется  ^ threadIdx.x / 16 в случае двух варпов и  ^ threadIdx.x / 8 в случае трёх
        // но надо изменить и откуда читать будем. Изначально сделать без этого 
        smem[Pre ? channel_idx * kNThreadsPerChannel + lane_idx * kNWarpsPerChannel + warp_idx : channel_idx * kNThreadsPerChannel + reverse_warp_idx * 32 + shift_within_warp] = ((float2*)dequantized)[i];
    
        __syncthreads();

        ((float2*)dequantized)[i] = smem[threadIdx.x];

        __syncthreads();
    }
}


template<typename traits>
__global__ __launch_bounds__(1024, 2)
void higgs_dequantize_d4_n256_full_kernel(DequantizeFullParams params, int n_iters_full, int rows_per_threadBlock, int extra_row) {
    extern __shared__ __align__(16) char smem_[];
    auto smem      = reinterpret_cast<bfloat4*>(smem_);
    auto smem_save_index = reinterpret_cast<int4*>(smem_ + traits::kSharedMemSizeLattice); // smem + Ktraits::kSharedMemSizeLattice / sizeof(bfloat4);
    auto smem_scale = reinterpret_cast<__nv_bfloat16*>(smem_ + traits::kSharedMemSizeLattice + traits::kSharedMemSizeSaveIndex);
    auto smem_hadamard_exchange = reinterpret_cast<float2*>(smem_ + traits::kSharedMemSizeLattice + traits::kSharedMemSizeSaveIndex + traits::kSharedMemSizeScales);

    auto lattice_load = reinterpret_cast<float4*>(params.lattice_ptr);
    auto scales_load = reinterpret_cast<__nv_bfloat16*>(params.scales_ptr);
    auto idx_load = reinterpret_cast<int4*>(params.idx_ptr);
    const int lane_id = threadIdx.x % traits::kLatticeSharedMemCopy;

    #pragma unroll
    for (int i = threadIdx.x; i < traits::kNLoadsForLattice; i += blockDim.x) {
        float4 latticeValues = lattice_load[i];
        bfloat4* latticeElems = reinterpret_cast<bfloat4*>(&latticeValues); 
        #pragma unroll
        for (int j = 0; j < traits::kLatticeElemsPerLoad; j++) {
            bfloat4 latticeElement = latticeElems[j];
            #pragma unroll
            for (int k = 0; k < traits::kLatticeSharedMemCopy; k++) {
                int smem_location = i * traits::kLatticeElemsPerLoad * traits::kLatticeSharedMemCopy + j * traits::kLatticeSharedMemCopy + k ^ lane_id;
                assert(smem_location < traits::kSharedMemSizeLattice / 8);
                smem[smem_location] = latticeElement;
            }
        }
    }

    auto out_store = reinterpret_cast<float4*>(params.out_ptr);
    int n_iters = n_iters_full + (rows_per_threadBlock > 0) + (blockIdx.x < extra_row);
    for (int i = 0; i < n_iters; ++i) {
        int row_threadBlock_level;
        int operating_n_rows;
        if (i < n_iters_full) {
            operating_n_rows = traits::kMaxRowsPerThreadBlock;
            row_threadBlock_level = (i * gridDim.x + blockIdx.x) * operating_n_rows;
        } else {
            if (i == n_iters_full) {
                // we can be here only if rows_per_threadBlock > 0 or extra_row == 1 (or both)
                operating_n_rows = rows_per_threadBlock > 0 ? rows_per_threadBlock : 1;
                row_threadBlock_level = n_iters_full * gridDim.x * traits::kMaxRowsPerThreadBlock + blockIdx.x * operating_n_rows;
            } else {
                // case when rows_per_threadBlock > 0 and extra_row = 1
                operating_n_rows = 1;
                row_threadBlock_level = n_iters_full * gridDim.x * traits::kMaxRowsPerThreadBlock + gridDim.x * rows_per_threadBlock + blockIdx.x * operating_n_rows;
            }
        }
        int row_index = row_threadBlock_level + threadIdx.x / traits::kThreadsPerChannelToLoad;
        int row_scale = row_threadBlock_level + threadIdx.x;
        
        if (threadIdx.x < operating_n_rows * traits::kThreadsPerChannelToLoad && row_index < params.flatten_batch) {
            smem_save_index[threadIdx.x] = idx_load[row_threadBlock_level * traits::kThreadsPerChannelToLoad + threadIdx.x]; // row_shift_bytes
        }

        if (threadIdx.x < operating_n_rows && row_scale < params.flatten_batch) {
            smem_scale[threadIdx.x] = scales_load[row_scale];
        }

        __syncthreads();
        auto smem_save_index_read = reinterpret_cast<int2*>(smem_save_index);
    
        float dequantized[8]; // sizeof(int4) / sizeof(type) = 16 / 2 = 8
        float2* dequantized_pointer = (float2*)dequantized;
        __align__(16) __nv_bfloat162 dequantized_store_local[4];
        constexpr int kBytesPerThread = 8 / traits::kD; // sizeof word in shared memory = 8
        // kThreadsGroupSize = 8 / kBytesPerThread = 8 / (8 /  traits.kD) = traits.kD
        constexpr int kLookUpNThreadsPerRow = traits::kChannelSize / 8; // 8 = sizeof(float4) / sizeof(bf16)
        int nLookUpsNeeded = kLookUpNThreadsPerRow * operating_n_rows;
        int nLookUpIters = ceildiv(nLookUpsNeeded, traits::kNThreads);
        for (int j = 0; j < nLookUpIters; ++j) {
            int smem_save_index_shift = j * traits::kNThreads + threadIdx.x;
            int output_row_index = smem_save_index_shift / kLookUpNThreadsPerRow;
            if (smem_save_index_shift < nLookUpsNeeded) {
                int2 codes = smem_save_index_read[smem_save_index_shift / traits::kD]; // сhannel_size / 8 / d * 8 = quantized_channel)size
                int thread_shift = kBytesPerThread * (threadIdx.x % traits::kD);
                const uint8_t* codes_adrress = reinterpret_cast<const uint8_t*>(&codes);
                float scale = float(smem_scale[output_row_index]);
                #pragma unroll
                for (int k = 0; k < kBytesPerThread; ++k) {
                    if constexpr (traits::kD == 4) {
                        bfloat4 dequantized_bf16 = smem[traits::kLatticeSharedMemCopy * codes_adrress[thread_shift + k] + lane_id];
                        dequantized_pointer[traits::kD / 2 * k] = __bfloat1622float2(dequantized_bf16.lo);
                        dequantized_pointer[traits::kD / 2 * k + 1] = __bfloat1622float2(dequantized_bf16.hi);
                        
                        dequantized[traits::kD * k] *= scale;
                        dequantized[traits::kD * k + 1] *= scale;
                        dequantized[traits::kD * k + 2] *= scale;
                        dequantized[traits::kD * k + 3] *= scale;
                    } else {
                        // TODO smem should be changed from bfloat4 to __nv_bfloat162
                        // __nv_bfloat162 dequantized_bf16 = smem[traits::kLatticeSharedMemCopy * codes_adrress[thread_shift + k] + lane_id];
                        // dequantized_pointer[traits::kD / 2 * k] = __bfloat1622float2(dequantized_bf16);

                        // dequantized[traits::kD * k] *= scale;
                        // dequantized[traits::kD * k + 1] *= scale;
                    }
                }

                hadamard_thread(dequantized);
                hadamard_warp<traits::kLogNThreadsPerChannelInWarp>(dequantized);
                if constexpr (traits::kNWarpsPerChannel > 1) {
                    exchange<traits::kNThreadsPerChannelHadamard, traits::kNWarpsPerChannel, true>(dequantized, smem_hadamard_exchange);
                    hadamard_warp<cilog2(traits::kNWarpsPerChannel)>(dequantized);
                    exchange<traits::kNThreadsPerChannelHadamard, traits::kNWarpsPerChannel, false>(dequantized, smem_hadamard_exchange);
                }

                #pragma unroll
                for (int k = 0; k < 8; ++k) {
                    dequantized[k] *= params.hadamard_scale;
                }
 
                assert(output_row_index < operating_n_rows);
                assert(row_threadBlock_level + output_row_index < params.flatten_batch);
                int flatten_batch_row_index = row_threadBlock_level + output_row_index;
                int64_t add_shift = flatten_batch_row_index * traits::kChannelSize * 2 / 16 + threadIdx.x % kLookUpNThreadsPerRow; // sizeof(bfloat16) = 2
                int64_t output_shift = (flatten_batch_row_index / params.n_tokens * params.out_batch_stride + flatten_batch_row_index % params.n_tokens * params.out_token_stride) / 16 + threadIdx.x % kLookUpNThreadsPerRow;
                // __align__(16) __nv_bfloat16 add_values[8];
                // auto add_load = reinterpret_cast<float4*>(params.add_ptr);
                // *reinterpret_cast<float4*>(add_values) = add_load[add_shift];
                
                // #pragma unroll
                // for (int k = 0; k < 8; ++k) {
                //     dequantized[k] += float(add_values[k]);
                // }

                #pragma unroll
                for (int k = 0; k < 4; ++k) {
                    dequantized_store_local[k] = __float22bfloat162_rn(dequantized_pointer[k]);
                    // dequantized_store_local[k].x = __float2bfloat16_rn(dequantized_pointer[k].x);
                    // dequantized_store_local[k].y = __float2bfloat16_rn(dequantized_pointer[k].y);
                    
                }
                out_store[output_shift] = *reinterpret_cast<float4*>(dequantized_store_local);
            }
        }
        __syncthreads();
    }
}


template<int KNChannels, int kN, int kD, int kLatticeSharedMemCopy>
void higgs_dequantize_d4_n256_full_launch(DequantizeFullParams &params, cudaStream_t stream) {
    using Ktraits = higgs_dequantize_d4_n256_full_kernel_traits<KNChannels, kN, kD, kLatticeSharedMemCopy>;

    int n_sms = 2 * 108;
    
    int n_iters_full = params.flatten_batch / (n_sms * Ktraits::kMaxRowsPerThreadBlock);
    int rows_per_threadBlock = (params.flatten_batch - n_iters_full * n_sms * Ktraits::kMaxRowsPerThreadBlock) / n_sms;
    int extra_row = params.flatten_batch - n_iters_full * n_sms * Ktraits::kMaxRowsPerThreadBlock - rows_per_threadBlock * n_sms;
    /*
    int rows_per_threadBlock = params.flatten_batch / n_sms;
    int n_iters;
    if (rows_per_threadBlock == 0) {
        rows_per_threadBlock = 1;
        n_iters = 0;
    } else {
        if (rows_per_threadBlock <= Ktraits::kMaxRowsPerThreadBlock) {
            n_iters = 1;
        } else {
            rows_per_threadBlock = Ktraits::kMaxRowsPerThreadBlock;
            n_iters = params.flatten_batch / (rows_per_threadBlock * n_sms);
        }
    }
    */
    // printf("rows_per_threadBlock, n_iters, flatten_batch - %d, %d, %d %d \n", params.flatten_batch, n_iters_full, rows_per_threadBlock, extra_row);
    auto kernel = &higgs_dequantize_d4_n256_full_kernel<Ktraits>;
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, Ktraits::kSharedMemSize);
    kernel<<<n_sms, Ktraits::kNThreads, Ktraits::kSharedMemSize, stream>>>(params, n_iters_full, rows_per_threadBlock, extra_row);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void higgs_dequantize_d4_n256_full_cuda(DequantizeFullParams &params, cudaStream_t stream) {
    if (params.channel_size == 128) {
        if (params.n == 16) {
            if (params.d == 4) higgs_dequantize_d4_n256_full_launch<128, 16, 4, 32>(params, stream);
            // else higgs_dequantize_d4_n256_full_launch<128, 16, 2, 32>(params, stream);
        } else if (params.n == 64) {
            if (params.d == 4) higgs_dequantize_d4_n256_full_launch<128, 64, 4, 32>(params, stream);
            // else higgs_dequantize_d4_n256_full_launch<128, 64, 2, 32>(params, stream);
        } else if (params.n == 256) {
            if (params.d == 4) higgs_dequantize_d4_n256_full_launch<128, 256, 4, 16>(params, stream);
            // else higgs_dequantize_d4_n256_full_launch<128, 256, 2, 32>(params, stream);
        }
    } else if (params.channel_size == 256) {
        if (params.n == 16) {
            if (params.d == 4) higgs_dequantize_d4_n256_full_launch<256, 16, 4, 32>(params, stream);
            // else higgs_dequantize_d4_n256_full_launch<256, 16, 2, 32>(params, stream);
        } else if (params.n == 64) {
            if (params.d == 4) higgs_dequantize_d4_n256_full_launch<256, 64, 4, 32>(params, stream);
            // else higgs_dequantize_d4_n256_full_launch<256, 64, 2, 32>(params, stream);
        } else if (params.n == 256) {
            if (params.d == 4) higgs_dequantize_d4_n256_full_launch<256, 256, 4, 16>(params, stream);
            // else higgs_dequantize_d4_n256_full_launch<256, 256, 2, 32>(params, stream);
        }
    } else if (params.channel_size == 512) {
        if (params.n == 16) {
            if (params.d == 4) higgs_dequantize_d4_n256_full_launch<512, 16, 4, 32>(params, stream);
            // else higgs_dequantize_d4_n256_full_launch<512, 16, 2, 32>(params, stream);
        } else if (params.n == 64) {
            if (params.d == 4) higgs_dequantize_d4_n256_full_launch<512, 64, 4, 32>(params, stream);
            // else higgs_dequantize_d4_n256_full_launch<512, 64, 2, 32>(params, stream);
        } else if (params.n == 256) {
            if (params.d == 4) higgs_dequantize_d4_n256_full_launch<512, 256, 4, 16>(params, stream);
            // else higgs_dequantize_d4_n256_full_launch<512, 256, 2, 32>(params, stream);
        }
    } else if (params.channel_size == 1024) {
        if (params.n == 16) {
            if (params.d == 4) higgs_dequantize_d4_n256_full_launch<1024, 16, 4, 32>(params, stream);
            // else higgs_dequantize_d4_n256_full_launch<1024, 16, 2, 32>(params, stream);
        } else if (params.n == 64) {
            if (params.d == 4) higgs_dequantize_d4_n256_full_launch<1024, 64, 4, 32>(params, stream);
            // else higgs_dequantize_d4_n256_full_launch<1024, 64, 2, 32>(params, stream);
        } else if (params.n == 256) {
            if (params.d == 4) higgs_dequantize_d4_n256_full_launch<1024, 256, 4, 16>(params, stream);
            // else higgs_dequantize_d4_n256_full_launch<1024, 256, 2, 32>(params, stream);
        }
    }
}
