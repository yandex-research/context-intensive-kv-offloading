#include <iostream>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include <vector>
#include "higgs_quantizer.h"


struct __align__(64) float16 {
    float4 x, y, z, w;
};

class dequantize_kernel_traits {
    public:
        int KBankWordSize = 8;
        int kThreadsPerWarp = 32;
    
        int kDequantizedRowSize;
        int kDequantizedByterPerElem = 2;
        int kDequantizedBytesPerRow;
        int kNLattice = 256, kDLattice = 4, kLatticeBytesPerElem = 2, KSharedMemCopy = 32;
        int kQuantizedRowSize;
        int kRowBytes;
    
        int kBytesPerLoad = 16;
        int kThreadsPerRow;
    
        int kScaleBytesPerElem = 2;
    
        int kNThreads;
        int nActiveThreads;
        int nRowsPerThreadBlock;
        int kLatticeElemsPerLoad;
        int kNLoadsForLattice;
        int kSharedMemSizeLattice;
        int kSharedMemSizeSave;
        int kSharedMemSizeScales;
        
        int kSharedMemSize;
    
        dequantize_kernel_traits(int kNThreads_, int kDequantizedRowSize_, int nActiveThreads_)
            : kNThreads(kNThreads_),
              nActiveThreads(nActiveThreads_),
              kDequantizedRowSize(kDequantizedRowSize_),
              kDequantizedBytesPerRow(kDequantizedRowSize_ * kDequantizedByterPerElem),
              kQuantizedRowSize(kDequantizedRowSize_ / kDLattice),
              kRowBytes(kQuantizedRowSize * 1),
              kThreadsPerRow(kRowBytes / kBytesPerLoad),
              nRowsPerThreadBlock(nActiveThreads_ / kThreadsPerRow),
              kLatticeElemsPerLoad(kBytesPerLoad / (kDLattice * kLatticeBytesPerElem)),
              kNLoadsForLattice(kNLattice / kLatticeElemsPerLoad),
              kSharedMemSizeLattice(kNLattice * kDLattice * kLatticeBytesPerElem * KSharedMemCopy),
              kSharedMemSizeSave(nActiveThreads_ * kBytesPerLoad),
              kSharedMemSizeScales(nRowsPerThreadBlock * kScaleBytesPerElem),
              kSharedMemSize(kSharedMemSizeLattice + kSharedMemSizeSave + kSharedMemSizeScales) {}
    };
    

__device__ __forceinline__ bfloat4 multiply_bfloat4_scalar(bfloat4 a, __nv_bfloat16 scalar) {
    __nv_bfloat162 scalar_vec = __nv_bfloat162(scalar, scalar);
    bfloat4 result;
    result.lo = __hmul2(a.lo, scalar_vec);
    result.hi = __hmul2(a.hi, scalar_vec);
    return result;
}

__global__ __launch_bounds__(1024, 2)
void higgs_dequantize_d4_n256_kernel(DequantizeParams params, dequantize_kernel_traits traits) {

    extern __shared__ __align__(16) char smem_[];
    auto smem      = reinterpret_cast<bfloat4*>(smem_);
    auto smem_save = reinterpret_cast<bfloat4*>(smem_ + traits.kSharedMemSizeLattice); // smem + Ktraits::kSharedMemSizeLattice / sizeof(bfloat4);
    auto smem_scale = reinterpret_cast<__nv_bfloat16*>(smem_ + traits.kSharedMemSizeLattice + traits.kSharedMemSizeSave);

    auto lattice_load = reinterpret_cast<float4*>(params.lattice_ptr); //// 
    auto scales_load = reinterpret_cast<__nv_bfloat16*>(params.scales_ptr);
    auto idx_load = reinterpret_cast<int4*>(params.idx_ptr);
    assert(reinterpret_cast<uintptr_t>(params.dequantized_ptr) % alignof(float4) == 0);
    auto dequantized_store = reinterpret_cast<float4*>(params.dequantized_ptr);
    const int lane_id = threadIdx.x % traits.KSharedMemCopy;

    #pragma unroll
    for (int i = threadIdx.x; i < traits.kNLoadsForLattice; i += blockDim.x) {
        float4 latticeValues = lattice_load[i];
        bfloat4* latticeElems = reinterpret_cast<bfloat4*>(&latticeValues); 
        #pragma unroll
        for (int j = 0; j < traits.kLatticeElemsPerLoad; j++) {
            bfloat4 latticeElement = latticeElems[j];
            #pragma unroll
            for (int k = 0; k < traits.KSharedMemCopy; k++) {
                int smem_location = i * traits.kLatticeElemsPerLoad * traits.KSharedMemCopy + j * traits.KSharedMemCopy + k ^ lane_id;
                assert(smem_location < traits.kSharedMemSizeLattice / 8);
                smem[smem_location] = latticeElement;
            }
        }
    }

    if (threadIdx.x < traits.nActiveThreads) {
        int nIters = ceildiv(params.batch, (gridDim.x * traits.nRowsPerThreadBlock));
        int last_part_bytes_loaded = traits.nActiveThreads * traits.kBytesPerLoad; // case when all threads loaded usefull stuff
        const int padded_batches = ceildiv(params.batch, traits.nRowsPerThreadBlock) * traits.nRowsPerThreadBlock;
        auto smem_save_idxes = reinterpret_cast<int4*>(smem_save); // change correct adrress latter
        #pragma unroll
        for (int i = 0; i < nIters; ++i) {
            int rowIndexThreadBlockGranularity = i * gridDim.x * traits.nRowsPerThreadBlock + blockIdx.x * traits.nRowsPerThreadBlock;
            int rowIndex = rowIndexThreadBlockGranularity + threadIdx.x / traits.kThreadsPerRow;
            // assert(rowIndex < params.batch);
            if (rowIndex < params.batch) {
                // Ktraits::kThreadsPerRow неудачное название тут, но это то что нужно - кол-во байт в строке делённое на кол-во байт в int4 чтении (а у нас теперь idx_load - int4)
                assert(16 * (rowIndex * traits.kThreadsPerRow + threadIdx.x % traits.kThreadsPerRow) < traits.kRowBytes * params.batch);
                smem_save_idxes[threadIdx.x] = idx_load[rowIndex * traits.kThreadsPerRow + threadIdx.x % traits.kThreadsPerRow];
            }
            #pragma unroll
            for (int i = threadIdx.x; i < traits.nRowsPerThreadBlock; i += blockDim.x) {
                if (rowIndexThreadBlockGranularity + i < params.batch) {
                    smem_scale[i] = scales_load[rowIndexThreadBlockGranularity + i];
                }
            }
            __syncthreads();

            // first threadBlock that dont have all active threads - last active threadBlock 
            // case when some threads dont need to load anything, so we dont need to process it
            // if (params.batch % Ktraits::kRowsPerThreadBlock != 0 && padded_batches - rowIndex <= Ktraits::kRowsPerThreadBlock) {
                // last_part_bytes_loaded = params.batch % Ktraits::kRowsPerThreadBlock * Ktraits::kThreadsPerRow * Ktraits::kBytesPerLoad;
            // }

            if (i == nIters - 1) {
                if (rowIndex >= padded_batches) {
                    last_part_bytes_loaded = 0;
                } else { 
                    if (params.batch % traits.nRowsPerThreadBlock != 0 && rowIndex >= padded_batches - traits.nRowsPerThreadBlock) {
                        last_part_bytes_loaded = params.batch % traits.nRowsPerThreadBlock * traits.kThreadsPerRow * traits.kBytesPerLoad;
                    }
                }
            }
            if (i < nIters - 1) {
                assert(last_part_bytes_loaded == traits.kNThreads * traits.kBytesPerLoad);
            }

            bfloat4 dequantized[2]; //traits.KBankWordSize / 4
            int kSharedMemoryReads = traits.kBytesPerLoad / traits.KBankWordSize;
            auto smem_read = reinterpret_cast<int2*>(smem_save_idxes); // тут кастится к int2 потому что sizeof(int2) = KBankWordSize. И кажется KBankWordSize = 4 всегда хуже поэтому мб и не нужен параметр KBankWordSize
        
            int thread_shift = threadIdx.x % 4;
            int thread_group = threadIdx.x / 4;
            #pragma unroll
            for (int j = 0; j < kSharedMemoryReads; ++j) {
                for (int k = 0; k < 4; ++k) {
                    if (8 * (j * traits.nActiveThreads + k * traits.nActiveThreads / 4 + thread_group) < last_part_bytes_loaded) {
                        int2 codes = smem_read[j * traits.nActiveThreads + k * traits.nActiveThreads / 4 + thread_group];
                        const uint8_t* codes_adrress = reinterpret_cast<const uint8_t*>(&codes);
                        // in case when channels_size = 64 / 128 and d=4 we have 1 or 2 activeThreads
                        // in case when channels_size = 32 / 64 and d=2 we also have only 1 or 2 activeThreads
                        // so this if can be constexpr in case of template impl, and thread_shift, thread_group can be const variables to
                        // doesn't work because nActiveThreads can be equal to 6 in there are 3 rows and 2 threads per row
                        // if (traits.nActiveThreads < 4) {
                            // thread_shift = threadIdx.x % 4 + traits.nActiveThreads * (k % traits.nActiveThreads);
                        // }
                        #pragma unroll
                        for (int l = 0; l < traits.KBankWordSize / 4; ++l) {
                            dequantized[l] = smem[traits.KSharedMemCopy * codes_adrress[2 * thread_shift + l] + lane_id];
                        }
                        int shift_inside_thread_block = j * 4 * traits.nActiveThreads + k * traits.nActiveThreads + threadIdx.x;
                        int output_row_index = shift_inside_thread_block * traits.kBytesPerLoad / traits.kDequantizedBytesPerRow;
                        int shift = ((i * gridDim.x + blockIdx.x) * traits.nRowsPerThreadBlock * traits.kDequantizedBytesPerRow) / sizeof(float4) + shift_inside_thread_block;  // threadIdx.x = 4 * thread_group + thread_shift
                        assert(i * gridDim.x * traits.nRowsPerThreadBlock + blockIdx.x * traits.nRowsPerThreadBlock < params.batch);
                        assert(16 * shift < traits.kRowBytes * params.batch * traits.kDLattice * traits.kLatticeBytesPerElem);
                        assert(output_row_index < traits.nRowsPerThreadBlock);
                        dequantized[0] = multiply_bfloat4_scalar(dequantized[0], smem_scale[output_row_index]);
                        dequantized[1] = multiply_bfloat4_scalar(dequantized[1], smem_scale[output_row_index]);
                        dequantized_store[shift] = *reinterpret_cast<float4*>(dequantized);
                        // 6 
                    }
                }
            }
            __syncthreads();
            // #pragma unroll
            // for (int j = 0; j < Ktraits::KBankWordSize; ++j) { // тут захардкожено 8 тоже потому что KBankWordSize = sizeof(int2) = 8
                // dequantized[j] = smem[Ktraits::KSharedMemCopy * codes_adrress[j] + lane_id];
            // }
        }  
    }
}


void higgs_dequantize_d4_n256_cuda(DequantizeParams &params, cudaStream_t stream, int channel_size) {
    /*
    int device = -1;
    cudaDeviceProp prop;
    cudaGetDevice(&device);
    cudaGetDeviceProperties(&prop, device);
    int sms = prop.multiProcessorCount;
    */
    constexpr int sms = 108;
    // должно всегда запускаться 2 * sm тредблоков потому что больше не нужно чтобы избежать лишних загрузок
    // хотя бы sm надо чтобы sm'ки не простаивали и 2 * потому что есть __syncthreads()
    int nThreadsPerRow = channel_size / 4 / 16;
    constexpr int kMaxThreadsPerThreadBlock = 1024, kWarpSize = 32;
    int nRowsPerThreadBlock = min(ceildiv(params.batch, sms * 2), kMaxThreadsPerThreadBlock / nThreadsPerRow);
    int nActiveThreads = nRowsPerThreadBlock * nThreadsPerRow;
    int nThreadsPerThreadBlock = ceildiv(nActiveThreads, kWarpSize) * kWarpSize;
    int nThreadBlocks = sms * 2;

    cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);
    // std::cout << "batch" << params.batch << " sms = " << sms << "n active threads" << nActiveThreads << " batch / 2sm " << ceildiv(params.batch, sms * 2) << " " << nRowsPerThreadBlock << " ceil div =" << ceildiv(nActiveThreads, kWarpSize) << " warp size " << kWarpSize << " * = " << ceildiv(nActiveThreads, kWarpSize) * kWarpSize << " round up to 32 = "  << nThreadsPerThreadBlock << " " << channel_size << '\n';
    
    auto traint = dequantize_kernel_traits(nThreadsPerThreadBlock, channel_size, nActiveThreads);
    
    auto kernel = &higgs_dequantize_d4_n256_kernel;
    cudaFuncSetAttribute(
        kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, traint.kSharedMemSize
    );
    // cudaFuncSetCacheConfig(kernel, cudaFuncCachePreferShared);
    kernel<<<nThreadBlocks, traint.kNThreads, traint.kSharedMemSize, stream>>>(params, traint);

    // C10_CUDA_KERNEL_LAUNCH_CHECK();
}