#include "cuda_fp16.h"
#include "cuda_bf16.h"
#include<stdint.h>

template<typename T, typename I>
__device__ void index_select(
    const int32_t numel,
    const int32_t dim,
    const I *ids,
    const T *src,
    T *dst
) {
    int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    int32_t j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= numel || j >= dim) {
      return;
    }
    dst[i * dim + j] = src[ids[i] * dim + j];
}

#define IS_OP(TYPENAME, INDEX_TYPENAME, FN_NAME) \
extern "C" __global__ void FN_NAME(  \
    const int32_t numel,  \
    const int32_t dim, \
    const INDEX_TYPENAME *ids, \
    const TYPENAME *src, \
    TYPENAME *dst \
) { index_select(numel, dim, ids, src, dst); } \

#if __CUDA_ARCH__ >= 800
IS_OP(__nv_bfloat16, uint32_t, is_u32_bf16);
#endif
#if __CUDA_ARCH__ >= 530
IS_OP(__half, uint32_t, is_u32_f16);
#endif

IS_OP(float, uint32_t, is_u32_f32);

// Causality mask kernel
// Sets dst[idx_b * t1 * t2 + idx1 * t2 + idx2] = -inf where idx2 > offset + idx1
template<typename T>
__device__ void apply_causality_mask(
    T *dst,
    const uint32_t bh,
    const uint32_t t1,
    const uint32_t t2,
    const uint32_t offset
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t total = bh * t1 * t2;
    if (idx >= total) {
        return;
    }
    // Decompose linear index into (idx_b, idx1, idx2)
    uint32_t idx2 = idx % t2;
    uint32_t tmp = idx / t2;
    uint32_t idx1 = tmp % t1;
    // Query at position offset + idx1 can attend to keys at positions 0..=offset+idx1
    // Mask positions where idx2 > offset + idx1
    if (idx2 > offset + idx1) {
        dst[idx] = -INFINITY;
    }
}

#define CAUSALITY_MASK_OP(TYPENAME, FN_NAME) \
extern "C" __global__ void FN_NAME( \
    TYPENAME *dst, \
    const uint32_t bh, \
    const uint32_t t1, \
    const uint32_t t2, \
    const uint32_t offset \
) { apply_causality_mask(dst, bh, t1, t2, offset); }

#if __CUDA_ARCH__ >= 800
CAUSALITY_MASK_OP(__nv_bfloat16, causality_mask_bf16)
#endif
#if __CUDA_ARCH__ >= 530
CAUSALITY_MASK_OP(__half, causality_mask_f16)
#endif
CAUSALITY_MASK_OP(float, causality_mask_f32)
CAUSALITY_MASK_OP(double, causality_mask_f64)
