#pragma once

#include <cstdint>

// Pure (torch-free) launcher for the MXFP4 A4W4 GEMM on gfx950.
//
// Computes  out[M, N] = dequant(A) @ dequant(B)^T   where A/B are 4-bit (MXFP4
// e2m1) packed two-per-byte and each contiguous group of 32 K-values shares one
// E8M0 block scale. This matches the input contract of aiter.gemm_a4w4_asm:
//
//   A       : [M, K/2]  fp4x2 (uint8), plain row-major
//   B       : [N, K/2]  fp4x2 (uint8), PRESHUFFLED via shuffle_weight(layout=(16,16))
//   AScale  : E8M0 (uint8), SHUFFLED  (quant shuffle=True layout)
//   BScale  : E8M0 (uint8), SHUFFLED
//   out     : [Mpad, N]  bf16  (Mpad = round_up(M, 32); caller slices [:M])
//
// scaleN_pad = round_up(ceil(K/32), 8) is the padded K-block count used by the
// scale shuffle (identical for A and B since they share K).
//
// MBS factors (optional; both or neither): one uint8 per row of A / of B,
// plain contiguous [M] / [N]. Byte b means f(b) = 1 / (1 + b/256), applied
// after the K accumulation as out[m][n] *= f(AFactor[m]) * f(BFactor[n]).
//
// `stream` is a hipStream_t passed as void* to keep this header HIP-free.
// Returns false (nothing launched) if the problem needs the single-wave
// fallback kernel, which does not support the MBS factors.
bool gemm_a4w4_mxfp4_launch(const uint8_t* A,
                            const uint8_t* B,
                            const uint8_t* AScale,
                            const uint8_t* BScale,
                            const uint8_t* AFactor,   // [M] or nullptr
                            const uint8_t* BFactor,   // [N] or nullptr
                            void*          out,       // __hip_bfloat16[Mpad, N]
                            int            M,
                            int            N,
                            int            K,
                            int            Mpad,
                            int            scaleN_pad,
                            void*          stream);
