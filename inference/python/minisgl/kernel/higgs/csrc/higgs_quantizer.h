#pragma once 
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
    // void *__restrict__ add_ptr;
    void *__restrict__ out_ptr;
};