#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include "higgs_quantizer.h"

// Quantizes every head vector of x independently (hadamard size == head_dim):
//   scale[b, t, h] = ||x[b, t, h, :]||                                    (pre-hadamard norm, float)
//   codes[b, t, h, g] = argmin_k || hadamard(x[b, t, h, :]) / scale [kD*g : kD*(g+1)] - lattice[k] ||^2
//
// Templated on the grid ([4, 256] -> 2 bit, [2, 256] -> 4 bit) and on the layout of x/idx/scales:
// kTokensMajor is [B, T, n_kv_heads, ...] (the landmarks format higgs_score consumes),
// kHeadsMajor is [B, n_kv_heads, T, ...] (the kv-cache format the dequantize kernel consumes).
// The layout only changes the flat row index of a head vector -- head_dim stays contiguous.
//
// One warp owns one token (all its kv heads, one head at a time); lane l holds elements
// [4l, 4l + 4) of the head -- the same layout LandmarksScoreKernel dequantizes with.
// Tokens are enumerated through cumulativeLengthes exactly like in the score kernel, so padding
// tokens (beyond each sample's true length) are never touched: idx/scales keep their old values
// there. Since a warp owns a whole token, lengths need no rounding on the python side.
//
// This is the prefill kernel: x holds batchSize new requests, while idx/scales are the
// preallocated kv cache ([max_batch_size, max_seq_len, ...]-shaped). Sample b is written into
// cache slot blockIndices[b], so xT (padding of the input) and outT (max_seq_len of the cache)
// can differ. blockIndices values must be distinct -- this is a scatter write.

enum class HeadsLayout { kTokensMajor, kHeadsMajor };

template <int kD, int kN, HeadsLayout kLayout, size_t kThreadsPerBlock>
__global__ void __launch_bounds__(kThreadsPerBlock) HeadsQuantizeKernel(
    const void *__restrict__ xPtr,
    void *__restrict__ idxPtr,
    void *__restrict__ scalesPtr,
    const int *__restrict__ cumulativeLengthes,
    const int *__restrict__ blockIndices, // [batchSize]: cache slot each input sample is written to
    const void *__restrict__ latticePtr,
    size_t batchSize,
    size_t xT,
    size_t outT,
    size_t nKvHeads
) {
    static_assert(kD == 2 || kD == 4, "only [2, 256] and [4, 256] grids are supported");
    static_assert(kN == 256, "codes are single bytes");
    constexpr int kHeadDim = 128;
    constexpr int kNCodesPerHead = kHeadDim / kD;
    constexpr int kElemsPerLane = kHeadDim / kWarpSize;   // 4
    constexpr int kGroupsPerLane = kElemsPerLane / kD;    // 1 for d=4, 2 for d=2
    using LatticeVec = std::conditional_t<kD == 4, bfloat4, __nv_bfloat162>;

    // Unlike the score kernel, every lane of a warp reads the same lattice entry at the same
    // iteration of the argmin loop below, so the reads are smem broadcasts -- a single copy
    // without swizzling is enough. Entries are stored pre-converted to float (bf16->float
    // conversions in the hot loop saturate the integer ALU pipe otherwise), together with
    // precomputed ||g||^2 / 2 so that the whole distance is one FMA chain.
    using LatticeFloatVec = std::conditional_t<kD == 4, float4, float2>;
    __shared__ LatticeFloatVec lattice[kN];
    __shared__ float latticeNormHalf[kN];

    auto latticeLoad = reinterpret_cast<const float4*>(latticePtr);
    constexpr int kNLoadsForLattice = kD * kN * sizeof(nv_bfloat16) / sizeof(float4);
    constexpr int kNLoadedValues = sizeof(float4) / sizeof(LatticeVec); // lattice points per float4
    #pragma unroll
    for (int i = threadIdx.x; i < kNLoadsForLattice; i += blockDim.x) {
        float4 latticeValues = latticeLoad[i];
        LatticeVec* latticeElems = reinterpret_cast<LatticeVec*>(&latticeValues);
        #pragma unroll
        for (int j = 0; j < kNLoadedValues; ++j) {
            LatticeVec latticeElement = latticeElems[j];
            if constexpr (kD == 4) {
                float2 lo = __bfloat1622float2(latticeElement.lo);
                float2 hi = __bfloat1622float2(latticeElement.hi);
                lattice[i * kNLoadedValues + j] = {lo.x, lo.y, hi.x, hi.y};
                latticeNormHalf[i * kNLoadedValues + j] = 0.5f * (lo.x * lo.x + lo.y * lo.y + hi.x * hi.x + hi.y * hi.y);
            } else {
                float2 v = __bfloat1622float2(latticeElement);
                lattice[i * kNLoadedValues + j] = v;
                latticeNormHalf[i * kNLoadedValues + j] = 0.5f * (v.x * v.x + v.y * v.y);
            }
        }
    }
    __syncthreads();

    const int laneId = threadIdx.x % kWarpSize;
    const int globalWarpIdx = (blockIdx.x * kThreadsPerBlock + threadIdx.x) / kWarpSize;
    const int nWarps = gridDim.x * (kThreadsPerBlock / kWarpSize);
    const int totalTokens = cumulativeLengthes[batchSize];

    auto xLoad = reinterpret_cast<const bfloat4*>(xPtr);
    auto idxStore = reinterpret_cast<uint8_t*>(idxPtr);
    auto scalesStore = reinterpret_cast<float*>(scalesPtr);

    int batchHint = 0;
    for (int currentToken = globalWarpIdx; currentToken < totalTokens; currentToken += nWarps) {
        const int batchIdx = FindBatchIdxFromHint(cumulativeLengthes, batchSize, currentToken, batchHint);
        batchHint = batchIdx;
        // batchIdx indexes the input samples; the outputs are written into cache slot
        // blockIndices[batchIdx] (with its own token stride outT)
        const int kvCacheIdx = blockIndices[batchIdx];
        const int tokenWithinSample = currentToken - cumulativeLengthes[batchIdx];
        for (int head = 0; head < nKvHeads; ++head) {
            size_t xRow, outRow; // flat indices of the (batch, token, head) vector, head_dim is contiguous
            if constexpr (kLayout == HeadsLayout::kTokensMajor) {
                xRow = (size_t(batchIdx) * xT + tokenWithinSample) * nKvHeads + head;
                outRow = (size_t(kvCacheIdx) * outT + tokenWithinSample) * nKvHeads + head;
            } else {
                xRow = (size_t(batchIdx) * nKvHeads + head) * xT + tokenWithinSample;
                outRow = (size_t(kvCacheIdx) * nKvHeads + head) * outT + tokenWithinSample;
            }

            bfloat4 xValues = xLoad[xRow * (kHeadDim / kElemsPerLane) + laneId];
            float x[kElemsPerLane];
            float2 lo = __bfloat1622float2(xValues.lo);
            float2 hi = __bfloat1622float2(xValues.hi);
            x[0] = lo.x; x[1] = lo.y; x[2] = hi.x; x[3] = hi.y;

            float normSq = x[0] * x[0] + x[1] * x[1] + x[2] * x[2] + x[3] * x[3];
            #pragma unroll
            for (int mask = kWarpSize / 2; mask > 0; mask >>= 1) {
                normSq += __shfl_xor_sync(0xffffffffu, normSq, mask);
            }
            const float scale = sqrtf(normSq);
            const float invScale = 1.0f / scale;

            hadamard_thread(x);
            hadamard_warp<cilog2(kWarpSize)>(x);
            #pragma unroll
            for (int i = 0; i < kElemsPerLane; ++i) {
                x[i] *= invScale;
            }

            // All groups of the lane share one pass over the lattice. dist = ||g||^2 / 2 - <x, g>
            // (a monotone rescale of the true distance) is a single FMA chain, and the candidate
            // index k is packed into the low mantissa byte so the argmin is a plain fminf --
            // no integer compare-selects in the hot loop. The packing perturbs dist by at most
            // 256 ulps (~3e-5 relative), which only affects genuine near-ties.
            float best[kGroupsPerLane];
            #pragma unroll
            for (int g = 0; g < kGroupsPerLane; ++g) {
                best[g] = INFINITY;
            }
            #pragma unroll 32
            for (int k = 0; k < kN; ++k) {
                LatticeFloatVec latticeValue = lattice[k];
                float normHalf = latticeNormHalf[k];
                #pragma unroll
                for (int g = 0; g < kGroupsPerLane; ++g) {
                    float dist;
                    if constexpr (kD == 4) {
                        dist = fmaf(-x[3], latticeValue.w,
                               fmaf(-x[2], latticeValue.z,
                               fmaf(-x[1], latticeValue.y,
                               fmaf(-x[0], latticeValue.x, normHalf))));
                    } else {
                        dist = fmaf(-x[kD * g + 1], latticeValue.y,
                               fmaf(-x[kD * g], latticeValue.x, normHalf));
                    }
                    float candidate = __uint_as_float((__float_as_uint(dist) & 0xFFFFFF00u) | unsigned(k));
                    best[g] = fminf(best[g], candidate);
                }
            }

            if constexpr (kGroupsPerLane == 1) {
                idxStore[outRow * kNCodesPerHead + laneId] = uint8_t(__float_as_uint(best[0]) & 0xFFu);
            } else {
                uint8_t codes[kGroupsPerLane];
                #pragma unroll
                for (int g = 0; g < kGroupsPerLane; ++g) {
                    codes[g] = uint8_t(__float_as_uint(best[g]) & 0xFFu);
                }
                reinterpret_cast<uint16_t*>(idxStore + outRow * kNCodesPerHead)[laneId] = *reinterpret_cast<uint16_t*>(codes);
            }
            if (laneId == 0) {
                scalesStore[outRow] = scale;
            }
        }
    }
}

template <int kD, HeadsLayout kLayout, size_t kThreadsPerBlock>
void HeadsQuantizeImpl(
    at::Tensor &x,
    at::Tensor &cum_lengths,
    at::Tensor &block_indices,
    at::Tensor &lattice,
    at::Tensor &idx,
    at::Tensor &scales
) {
    at::cuda::CUDAGuard device_guard{(char)x.get_device()};

    auto* kernelInstance = HeadsQuantizeKernel<kD, 256, kLayout, kThreadsPerBlock>;
    int numBlocks = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocks, kernelInstance, kThreadsPerBlock, 0);

    // raw cudaGetDeviceProperties takes ~1ms per call and would stall every launch, torch caches it
    int smCount = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;

    const dim3 gridDim{unsigned(numBlocks * smCount), 1, 1};
    const dim3 blockDim{unsigned(kThreadsPerBlock), 1, 1};

    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    size_t xT = kLayout == HeadsLayout::kTokensMajor ? x.size(1) : x.size(2);
    size_t outT = kLayout == HeadsLayout::kTokensMajor ? idx.size(1) : idx.size(2);
    size_t nKvHeads = kLayout == HeadsLayout::kTokensMajor ? x.size(2) : x.size(1);

    kernelInstance<<<gridDim, blockDim, 0, stream>>>(
        x.data_ptr(),
        idx.data_ptr(),
        scales.data_ptr(),
        reinterpret_cast<int*>(cum_lengths.data_ptr()),
        reinterpret_cast<const int*>(block_indices.data_ptr()),
        lattice.data_ptr(),
        static_cast<size_t>(x.size(0)),   // batchSize (input samples)
        xT,                               // padded seq len of the input
        outT,                             // max_seq_len of the kv cache
        nKvHeads
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void HeadsQuantize(
    at::Tensor &x,
    at::Tensor &cum_lengths,
    at::Tensor &block_indices,
    at::Tensor &lattice,
    at::Tensor &idx,
    at::Tensor &scales,
    bool heads_first
) {
    int d = lattice.size(1); // validated in higgs_quantize_heads
    if (d == 4) {
        if (heads_first) HeadsQuantizeImpl<4, HeadsLayout::kHeadsMajor, 512>(x, cum_lengths, block_indices, lattice, idx, scales);
        else HeadsQuantizeImpl<4, HeadsLayout::kTokensMajor, 512>(x, cum_lengths, block_indices, lattice, idx, scales);
    } else {
        if (heads_first) HeadsQuantizeImpl<2, HeadsLayout::kHeadsMajor, 512>(x, cum_lengths, block_indices, lattice, idx, scales);
        else HeadsQuantizeImpl<2, HeadsLayout::kTokensMajor, 512>(x, cum_lengths, block_indices, lattice, idx, scales);
    }
}
