// Shared definitions for the GEMM lab kernels (gfx950, MXFP4 A4W4, bf16 output).
#pragma once

#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>

#include <cstdint>

using i32x4 = int __attribute__((ext_vector_type(4)));
using i32x8 = int __attribute__((ext_vector_type(8)));
using f32x4 = float __attribute__((ext_vector_type(4)));

// C[M, N] = A[M, K] * B[N, K]^T, everything plain row-major (no pre-shuffle):
//   A  [M, K/2]  two e2m1 values per byte     B  [N, K/2]  same layout
//   As [M, K/32] one E8M0 scale per 32 K      Bs [N, K/32] same layout
//   C  [M, N]    bf16
struct Problem {
    int M, N, K;
    const uint8_t* A;
    const uint8_t* B;
    const uint8_t* As;
    const uint8_t* Bs;
    __hip_bfloat16* C;
};

// One v_mfma_scale_f32_16x16x128_f8f6f4 with both operands in fp4 (format 4).
// Lane L passes row L%16 of its operand, K block L/16 (32 fp4 = 16 bytes), and the
// E8M0 scale of that (row, block). Lane L gets C[4*(L/16) + i][L%16] in acc[i],
// rows taken from `a`, columns from `b`.
__device__ __forceinline__ f32x4 mfma_fp4(i32x4 a, i32x4 b, f32x4 c, int sa, int sb)
{
    const i32x8 a8 = {a[0], a[1], a[2], a[3], 0, 0, 0, 0};
    const i32x8 b8 = {b[0], b[1], b[2], b[3], 0, 0, 0, 0};
    return __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(a8, b8, c, 4, 4, 0, sa, 0, sb);
}
