#pragma once

#include <type_traits>
#include <variant>

#include <c10/util/Exception.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

const int kBytesPerLoad = 16;
const int kSmemWordSize = 8;
constexpr int kWarpSize = 32;
    
constexpr int cilog2(int val) { return val > 0 ? 1 + cilog2(val >> 1) : -1; }

template <typename T>
constexpr T constexpr_min(T a, T b) {
    return a < b ? a : b;
}

inline int __host__ __device__ ceildiv(int a, int b) {
    return (a + b - 1) / b;
}
  
struct __align__(8) bfloat4 {
    __nv_bfloat162 lo;
    __nv_bfloat162 hi;
};

inline float2 __device__ dot_and_norm(bfloat4 a, bfloat4 b) {
    float2 a_float = __bfloat1622float2(a.lo);
    float2 b_float = __bfloat1622float2(b.lo);
    float dot = a_float.x * b_float.x + a_float.y * b_float.y;
    float norm = b_float.x * b_float.x + b_float.y * b_float.y;
    a_float = __bfloat1622float2(a.hi);
    b_float = __bfloat1622float2(b.hi);
    dot += a_float.x * b_float.x + a_float.y * b_float.y;
    norm += b_float.x * b_float.x + b_float.y * b_float.y;
    return {dot, norm};
}

// In-place N-point Walsh-Hadamard transform (N a power of two: 4 for score, 8 for full dequant).
// N stages = log2(N), N/2 butterflies per stage. N is deduced from the array argument.
template <int N>
__forceinline__ __device__ void hadamard_thread(float (&x)[N]) {
    static_assert(N >= 2 && (N & (N - 1)) == 0, "hadamard_thread: N must be a power of two");
    constexpr int kLogN = cilog2(N);
    #pragma unroll
    for (int i = 0; i < kLogN; ++i) {
        const int stride = 1 << i;
        #pragma unroll
        for (int j = 0; j < N / 2; ++j) {
            const int lo  = j & (stride - 1);
            const int idx = (j - lo) * 2 + lo;
            const float a = x[idx];
            const float b = x[idx + stride];
            x[idx]          = a + b;
            x[idx + stride] = a - b;
        }
    }
}

// Cross-lane (warp) stage of the Hadamard transform.
// kLogNThreadsPerChannelInWarp = log2(#lanes participating); kNElems (per-thread count) is
// deduced from the array. Guarded by __CUDACC__ because the warp-shuffle intrinsics are not
// declared for the host compiler (this header is also included from higgs_quantizer.cpp).
#ifdef __CUDACC__
template <int kLogNThreadsPerChannelInWarp, int kNElems>
__forceinline__ __device__ void hadamard_warp(float (&x)[kNElems]) {
    constexpr int kLanes = 1 << kLogNThreadsPerChannelInWarp;
    int lane_id = threadIdx.x % kLanes;
    #pragma unroll
    for (int step = 0; step < kLogNThreadsPerChannelInWarp; ++step) {
        const int lane_mask = 1 << step;
        const float sign = (lane_id & lane_mask) ? -1.f : 1.f;
        #pragma unroll
        for (int i = 0; i < kNElems; ++i) {
            float x_val_other = __shfl_xor_sync(__activemask(), x[i], lane_mask);
            x[i] = sign * x[i] + x_val_other;
        }
    }
}
#endif

template <int... Values>
auto make_int_variant(int x, const char* name = "value") {
    using Variant = std::variant<std::integral_constant<int, Values>...>;

    Variant v;

    bool matched = ((x == Values ? (v = std::integral_constant<int, Values>{}, true) : false) || ...);

    TORCH_CHECK(matched, "unsupported ", name, ": ", x, " (supported: ", c10::Join(", ", std::vector<int>{Values...}), ")");
    return v;
}

__device__ __forceinline__ int FindBatchIdxFromHint(const int* __restrict__ cumulativeLengthes, int batchSize, int value, int hint) {
    int idx = hint;
    while (idx + 1 < batchSize && value >= cumulativeLengthes[idx + 1]) {
	++idx;
    }
    return idx;
}

struct lattice_traits {
    static constexpr int kD = 4;
    static constexpr int kN = 256;
    static constexpr int kBytesPerLatticeElem = 2;
    static constexpr int kLatticeBytes = kN * kD * kBytesPerLatticeElem;
    static constexpr int kLoadsPerLattice = kLatticeBytes / kBytesPerLoad;
    static constexpr int kSmemWordSize = 8;
};

struct QuantizeParams {
    int batch;
    int channel_size;

    void *__restrict__ debug;
    void *__restrict__ x_ptr;
    void *__restrict__ lattice_ptr;
    void *__restrict__ idx_ptr;
};

struct DequantizeParams {
    int batch;
    
    int64_t idx_batch_stride;
    int64_t lattice_n_stride;
    int64_t dequantized_stride;

    void *__restrict__ idx_ptr;
    void *__restrict__ scales_ptr;
    void *__restrict__ lattice_ptr;
    void *__restrict__ dequantized_ptr;
};

struct DequantizeFullParams {
    int flatten_batch;
    int n_tokens;
    int channel_size;
    int n, d;
    float hadamard_scale;

    int64_t out_token_stride;
    int64_t out_batch_stride;

    void *__restrict__ idx_ptr;
    void *__restrict__ scales_ptr;
    void *__restrict__ lattice_ptr;
    void *__restrict__ add_ptr;
    void *__restrict__ out_ptr;
};
