#include "ATen/core/TensorBody.h"
#include "ATen/core/interned_strings.h"
#include "c10/util/ArrayRef.h"
#include <cstddef>
#include <cstdint>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include "higgs_quantizer.h"
#include "c10/util/Exception.h"
#include <variant>

constexpr int next_pow2(int v) {
    int p = 1;
    while (p < v) p <<= 1;
    return p;
}

// Applies the same 128-point Hadamard transform the score kernel needs (H_32 x H_4 with lane l
// holding dims [4l, 4l+4)) to the queries, once, and folds hadamard_scale in. H is symmetric,
// so (H k) . q == k . (H q): transforming the tiny query tensor here lets the main kernel skip
// the 20-shuffle hadamard_warp for every (token, head). Output is fp32 so the main loop dots
// bf16 lattice values against it without converting q back from bf16.
template <size_t kThreadsPerBlock>
__global__ void __launch_bounds__(kThreadsPerBlock) HadamardQueryKernel(
    const void *__restrict__ queryPtr,
    float *__restrict__ outPtr,
    int nHeadsTotal,          // batch * n_q_heads
    int nQHeads,
    size_t queryBatchStride,  // in bf16 elements
    float hadamardScale
) {
    constexpr int kHeadDim = 128;
    constexpr int D = 4;
    int warpId = (blockIdx.x * kThreadsPerBlock + threadIdx.x) / kWarpSize;
    int laneId = threadIdx.x % kWarpSize;
    if (warpId >= nHeadsTotal) return;  // warpId is uniform across the warp
    int b = warpId / nQHeads;
    int h = warpId % nQHeads;

    auto src = reinterpret_cast<const nv_bfloat16*>(queryPtr) + b * queryBatchStride + h * kHeadDim + laneId * D;
    bfloat4 qv = *reinterpret_cast<const bfloat4*>(src);
    float2 lo = __bfloat1622float2(qv.lo);
    float2 hi = __bfloat1622float2(qv.hi);
    float x[D] = {lo.x, lo.y, hi.x, hi.y};

    hadamard_thread(x);
    hadamard_warp<cilog2(kWarpSize)>(x);

    float4 out = {x[0] * hadamardScale, x[1] * hadamardScale, x[2] * hadamardScale, x[3] * hadamardScale};
    *reinterpret_cast<float4*>(outPtr + (b * nQHeads + h) * kHeadDim + laneId * D) = out;
}

// Reduces kP per-lane partials to full 32-lane sums in ONE packed butterfly: after each xor
// stage every lane holds a valid pairwise sum, so two accumulators fold into one register
// (selected by the stage's lane bit) and the register count halves while the reduction keeps
// going. Lane l ends up with the complete sum of accumulator (l & (kPInitial - 1)), i.e. all
// kP sums finish in ~2*kP shuffles instead of kP separate 5-shuffle butterflies.
template <int kMask, int kP>
__device__ __forceinline__ float PackedWarpReduce(float (&acc)[kP], int laneId) {
    static_assert(1 <= kMask && kMask <= kWarpSize / 2, "mask must stay within the warp");
    #pragma unroll
    for (int g = 0; g < kP; ++g) {
	acc[g] += __shfl_xor_sync(0xffffffffu, acc[g], kMask);
    }
    constexpr int kNext = kP > 1 ? kP / 2 : 1;
    float packed[kNext];
    if constexpr (kP > 1) {
	#pragma unroll
	for (int g = 0; g < kNext; ++g) {
	    packed[g] = (laneId & kMask) ? acc[2 * g + 1] : acc[2 * g];
	}
    } else {
	packed[0] = acc[0];
    }
    if constexpr (kMask == kWarpSize / 2) {
	static_assert(kP <= 2, "initial kP must not exceed the warp size");
	return packed[0];
    } else {
	return PackedWarpReduce<kMask * 2, kNext>(packed, laneId);
    }
}

template <size_t kNumQHeads, size_t kNumKVHeads, size_t kThreadsPerBlock, size_t kLatticeSharedMemCopy>
__global__ void __launch_bounds__(kThreadsPerBlock) LandmarksScoreKernel(
    void *__restrict__ landmarksPtr,
    void *__restrict__ scalesPtr,
    const float *__restrict__ queryPtr,  // hadamard-transformed, hadamard_scale folded in: [batch, n_q, 128] fp32 contiguous
    void *__restrict__ scoresPtr,
    size_t batchSize,                    // number of queries; landmarks/scales/scores hold max_batch entries
    size_t T,
    size_t scoresBatchStride,
    int *__restrict__ cumulativeLengthes,
    const int *__restrict__ blockIndices, // [batchSize]: kv-cache entry scored by each query; also
                                          // the scores slot written -> values must be distinct
    void *__restrict__ lattice_ptr
) {
    constexpr int N = 256;
    constexpr int D = 4;
    constexpr int kLatticeSize = D * N * sizeof(nv_bfloat16) * kLatticeSharedMemCopy / sizeof(bfloat4);
    __shared__ bfloat4 lattice[kLatticeSize];

    auto lattice_load = reinterpret_cast<float4*>(lattice_ptr);
    int lane_id = threadIdx.x % kWarpSize;
    int latticeLane = threadIdx.x % kLatticeSharedMemCopy;                          // which lattice copy this thread uses
    constexpr int kNLoadsForLattice = D * N * sizeof(nv_bfloat16) / sizeof(float4); // float4s to cover one copy of the grid
    #pragma unroll
    for (int i = threadIdx.x; i < kNLoadsForLattice; i += blockDim.x) {
	float4 latticeValues = lattice_load[i];
        bfloat4* latticeElems = reinterpret_cast<bfloat4*>(&latticeValues);
	constexpr int kNLoadedValues = sizeof(float4) / sizeof(bfloat4);   // bfloat4 (d=4 point) per float4 = 2
	#pragma unroll
	for (int j = 0; j < kNLoadedValues; ++j) {
            bfloat4 latticeElement = latticeElems[j];
            #pragma unroll
            for (int k = 0; k < kLatticeSharedMemCopy; k++) {
                int smem_location = i * kNLoadedValues * kLatticeSharedMemCopy + j * kLatticeSharedMemCopy + k ^ latticeLane;
                lattice[smem_location] = latticeElement;
	    }
	}
    }
    __syncthreads();

    __shared__ int4 quantizedBytes[kThreadsPerBlock];  // one int4 (16 code bytes) per thread
    constexpr int kHeadDim = 128;
    constexpr int kNBytesPerHead = kHeadDim / D;
    constexpr int kNBytesPerToken = kNumKVHeads * kNBytesPerHead; // 256 in case of k_kv_head = 8 and 128 in case of 4
    constexpr int kNThreadsPerToken = kNBytesPerToken / sizeof(float4);
    constexpr int kNBytesReadPerWarp = kWarpSize * sizeof(float4); // just 512
    constexpr int kNTokensPerWarp = kNBytesReadPerWarp / kNBytesPerToken;
    static_assert(kNTokensPerWarp == 2 || kNTokensPerWarp == 4 || kNTokensPerWarp == 8, "Only n_kv_heads = 2, n_kv_heads = 4 and n_kv_heads = 8 are supported");
    constexpr int kNTokesPerBlock = kThreadsPerBlock / kWarpSize * kNTokensPerWarp;

    __shared__ __align__(16) float scales[kNTokesPerBlock * kNumKVHeads];
    using ScaleVec = std::conditional_t<kNumKVHeads == 2, float2, float4>;
    constexpr int kScaleVecFloats  = sizeof(ScaleVec) / sizeof(float);   // 2 or 4
    constexpr int kScaleVecsPerTok = kNumKVHeads / kScaleVecFloats;      // 1, 1, 2

    constexpr int kGQAGroupSize = kNumQHeads / kNumKVHeads;
    constexpr int kNWarpsPerBlock = kThreadsPerBlock / kWarpSize;
    constexpr int kScoresPerWarp = kNTokensPerWarp * kNumKVHeads;   // always 16
    __shared__ nv_bfloat16 scoresSmem[kNWarpsPerBlock * kScoresPerWarp];

    // GQA dots are reduced in chunks of up to 4 query heads per packed butterfly: enough to cut
    // the shuffle count ~2x while keeping the per-chunk query slice in 4 float4 registers.
    constexpr int kChunk = constexpr_min(kGQAGroupSize, 4);
    constexpr int kP = next_pow2(kChunk);
    constexpr int kNChunks = (kGQAGroupSize + kChunk - 1) / kChunk;

    // all threads within a warp process tokens in the same sample (they have the same batch_idx)
    int warpIdx = threadIdx.x / kWarpSize;
    int firstTokenByWarp = blockIdx.x * kNTokesPerBlock + warpIdx * kNTokensPerWarp;
    int batchHint = 0;
    constexpr int landmarksTokenStride = kNumKVHeads * kNBytesPerHead / sizeof(int4); // codes are head_dim/d bytes per head
    int landmarksBatchStride = T * landmarksTokenStride;
    constexpr int scalesTokenStride = kNumKVHeads;
    int scalesBatchStride = T * scalesTokenStride;
    auto landmarksLoad = reinterpret_cast<int4*>(landmarksPtr);
    auto scalesLoad = reinterpret_cast<float*>(scalesPtr);
    auto queryLoad = reinterpret_cast<const float4*>(queryPtr);
    auto scoresStore = reinterpret_cast<nv_bfloat16*>(scoresPtr);
    const int totalTokens = cumulativeLengthes[batchSize];
    for (int currentToken = firstTokenByWarp; currentToken < totalTokens; currentToken += gridDim.x * kNTokesPerBlock) {
	const int batchIdx = FindBatchIdxFromHint(cumulativeLengthes, batchSize, currentToken, batchHint);
	batchHint = batchIdx;
	// batchIdx indexes queries/scores/lengths; the landmarks and scales of the kv-cache entry
	// this query attends to live at blockIndices[batchIdx]
	const int kvCacheIdx = blockIndices[batchIdx];
	int tokenNumberWithinSample = currentToken - cumulativeLengthes[batchIdx];
	int tokenNumberWithinWarp = lane_id / kNThreadsPerToken;
	int shiftWithinToken = lane_id % kNThreadsPerToken;
	// инвариант -- варп грузит те токены которые обрабатывает. И все они принадлежат одному примеру батча.
	if (tokenNumberWithinSample + tokenNumberWithinWarp < T) {
	    int shift = kvCacheIdx * landmarksBatchStride + (tokenNumberWithinSample + tokenNumberWithinWarp) * landmarksTokenStride + shiftWithinToken;
	    quantizedBytes[threadIdx.x] = landmarksLoad[shift];
	}
	if (lane_id < kNTokensPerWarp) {
	    int scaleToLoad = tokenNumberWithinSample + lane_id;
	    if (scaleToLoad < T) {
		const float* src = scalesLoad + kvCacheIdx * scalesBatchStride + scaleToLoad * scalesTokenStride;
		float* dst = &scales[(warpIdx * kNTokensPerWarp + lane_id) * kNumKVHeads];
		#pragma unroll
		for (int c = 0; c < kScaleVecsPerTok; ++c) {
		    reinterpret_cast<ScaleVec*>(dst)[c] = reinterpret_cast<const ScaleVec*>(src)[c];
		}
	    }
	}
	// every warp reads/writes only its own slice of quantizedBytes/scales/scoresSmem
	// (thread t owns int4 #t and warp w spans exactly [32w, 32w+32)), so a warp-level
	// barrier is enough -- no need to couple the block's warps with __syncthreads.
	__syncwarp();

	int warpTokenBase = warpIdx * kNTokensPerWarp;
	const uint8_t* warpBytes = reinterpret_cast<const uint8_t*>(quantizedBytes);
	// head-outer so each chunk's query registers are reused across the warp's tokens
	#pragma unroll
	for (int currentHead = 0; currentHead < kNumKVHeads; ++currentHead) {
	    float score[kNTokensPerWarp];
	    #pragma unroll
	    for (int c = 0; c < kNChunks; ++c) {
		// this lane's 4 dims of up to kChunk hadamard-transformed query heads
		float4 q[kP];
		#pragma unroll
		for (int j = 0; j < kP; ++j) {
		    int g = c * kChunk + j;
		    if (g < kGQAGroupSize) {
			q[j] = queryLoad[(batchIdx * kNumQHeads + currentHead * kGQAGroupSize + g) * (kHeadDim / D) + lane_id];
		    }
		}
		#pragma unroll
		for (int t = 0; t < kNTokensPerWarp; ++t) {
		    int code = warpBytes[(warpTokenBase + t) * kNBytesPerToken + currentHead * kNBytesPerHead + lane_id];
		    bfloat4 latticeValue = lattice[code * kLatticeSharedMemCopy + latticeLane];
		    float2 lo = __bfloat1622float2(latticeValue.lo);
		    float2 hi = __bfloat1622float2(latticeValue.hi);
		    float acc[kP];
		    #pragma unroll
		    for (int j = 0; j < kP; ++j) {
			if (c * kChunk + j < kGQAGroupSize) {
			    acc[j] = lo.x * q[j].x + lo.y * q[j].y + hi.x * q[j].z + hi.y * q[j].w;
			} else {
			    acc[j] = acc[0];  // pad slot duplicates a real dot; the max below is unaffected
			}
		    }
		    // lane l now holds the full dot for gqa head (c * kChunk + (l & (kP - 1)))
		    float v = PackedWarpReduce<1>(acc, lane_id);
		    // the token-head scale is uniform over the warp: one multiply per chunk instead
		    // of scaling every dequantized value, applied before the max so any sign works
		    v *= scales[(warpTokenBase + t) * kNumKVHeads + currentHead];
		    #pragma unroll
		    for (int m = 1; m < kP; m <<= 1) {
			v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, m));
		    }
		    score[t] = (c == 0) ? v : fmaxf(score[t], v);
		}
	    }
	    if (lane_id == 0) {
		#pragma unroll
		for (int t = 0; t < kNTokensPerWarp; ++t) {
		    scoresSmem[warpIdx * kScoresPerWarp + t * kNumKVHeads + currentHead] = __float2bfloat16(score[t]);
		}
	    }
	}

	// every warp now has its 16 maxes in smem; 16 lanes each do one 2-byte write. Scores are
	// [max_batch, n_kv, T] and land in the cache slot's row (kvCacheIdx, not batchIdx): each
	// group of kNTokensPerWarp lanes writes the consecutive scores of one head row, so the
	// stores coalesce into kNumKVHeads short runs strided by T.
	__syncwarp();
	if (lane_id < kScoresPerWarp) {
	    int headIdx = lane_id / kNTokensPerWarp;
	    int tokenInWarp = lane_id % kNTokensPerWarp;
	    int tokenWithinSample = tokenNumberWithinSample + tokenInWarp;
	    if (tokenWithinSample < T) {  // skip padding tokens
	        scoresStore[kvCacheIdx * scoresBatchStride + headIdx * T + tokenWithinSample]
	            = scoresSmem[warpIdx * kScoresPerWarp + tokenInWarp * kNumKVHeads + headIdx];
	    }
	}
    }
}

template <size_t kThreadsPerBlock, size_t kLatticeSharedMemCopy>
void LandmarksScoreImpl(
    at::Tensor &landmarks,
    at::Tensor &scales,
    at::Tensor &cum_lengths,
    at::Tensor &block_indices,
    at::Tensor &lattice,
    at::Tensor &query,
    at::Tensor &out,
    float hadamard_scale
) {
    int n_q_heads = query.size(1);
    int n_kv_heads = landmarks.size(2);

    at::cuda::CUDAGuard device_guard{(char)landmarks.get_device()};
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    // hadamard-transform the queries once (fp32, hadamard_scale folded in) so the main kernel
    // never runs the warp hadamard in its per-token loop
    auto query_transformed = at::empty({query.size(0), query.size(1), query.size(2)},
                                       query.options().dtype(at::kFloat));
    {
	constexpr int kQhThreads = 128;
	int nHeadsTotal = query.size(0) * query.size(1);
	int nBlocks = ceildiv(nHeadsTotal, kQhThreads / kWarpSize);
	HadamardQueryKernel<kQhThreads><<<nBlocks, kQhThreads, 0, stream>>>(
	    query.data_ptr(),
	    reinterpret_cast<float*>(query_transformed.data_ptr()),
	    nHeadsTotal,
	    static_cast<int>(query.size(1)),
	    static_cast<size_t>(query.stride(0)),
	    hadamard_scale
	);
    }

    try {
	std::visit([&](auto numQHeads, auto numKVHeads) {
	    constexpr int NumQHeads = std::decay_t<decltype(numQHeads)>::value;
	    constexpr int NumKVHeads = std::decay_t<decltype(numKVHeads)>::value;

	    auto* kernelInstance = LandmarksScoreKernel<NumQHeads, NumKVHeads, kThreadsPerBlock, kLatticeSharedMemCopy>;
	    int numBlocks = 0;
	    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocks, kernelInstance, kThreadsPerBlock, 0);

	    cudaDeviceProp* props = at::cuda::getCurrentDeviceProperties();  // cached by pytorch

	    const dim3 gridDim{static_cast<unsigned>(numBlocks * props->multiProcessorCount), 1, 1};
	    const dim3 blockDim{kThreadsPerBlock, 1, 1};

	    kernelInstance<<<gridDim, blockDim, 0, stream>>>(
		landmarks.data_ptr(),
		scales.data_ptr(),
		reinterpret_cast<const float*>(query_transformed.data_ptr()),
		out.data_ptr(),
		static_cast<size_t>(query.size(0)),       // batchSize (number of queries)
		static_cast<size_t>(landmarks.size(1)),   // T (padded seq len)
		static_cast<size_t>(out.stride(0)),       // scoresBatchStride
		reinterpret_cast<int*>(cum_lengths.data_ptr()),  // cumulativeLengthes (int32)
		reinterpret_cast<const int*>(block_indices.data_ptr()), // blockIndices (int32)
		lattice.data_ptr()
	    );
	},
	make_int_variant<8, 16, 24, 32, 64>(n_q_heads, "n_q_heads"),  // 8/16: tp4/tp2 shards with replicated kv heads
	make_int_variant<2, 4, 8>(n_kv_heads, "n_kv_heads"));
    } catch (const std::bad_variant_access& _) {
	TORCH_CHECK(false, "Unsupported Landmarks configuration: n_kv_head = ", n_kv_heads, " , n_q_heads = ", n_q_heads)
    }
}


void LandmarksScore(
    at::Tensor &landmarks,
    at::Tensor &scales,
    at::Tensor &cum_lengths,
    at::Tensor &block_indices,
    at::Tensor &lattice,
    at::Tensor &query,
    at::Tensor &out,
    float hadamard_scale
) {
    LandmarksScoreImpl<512, 16>(landmarks, scales, cum_lengths, block_indices, lattice, query, out, hadamard_scale);
}
