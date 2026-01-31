#include "cuda_fp16.h"
#include "cuda_bf16.h"
#include<stdint.h>

template <typename T>
__device__ void fill(T *dst, const T value, const size_t numel) {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numel) return;
    dst[idx] = value;
}

#define FILL_OP(TYPENAME, RUST_NAME) \
  extern "C" __global__ void fill_##RUST_NAME( \
      TYPENAME *dst, \
      const TYPENAME value, \
      const size_t numel) { \
    fill<TYPENAME>(dst, value, numel); \
  } \

#if __CUDA_ARCH__ >= 800
FILL_OP(__nv_bfloat16, bf16)
#endif

#if __CUDA_ARCH__ >= 530
FILL_OP(__half, f16)
#endif

FILL_OP(float, f32)
FILL_OP(double, f64)
FILL_OP(int64_t, i64)
FILL_OP(uint8_t, u8)
