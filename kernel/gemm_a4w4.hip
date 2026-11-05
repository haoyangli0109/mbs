// MXFP4 A4W4 GEMM for gfx950 (MI350).
//
// Computes out[M,N] = dequant(A) @ dequant(B)^T with MXFP4 (e2m1) 4-bit
// operands packed two-per-byte and per-32-K-block E8M0 scales, consuming the
// production shuffled-input layout (plain row-major A, 16x16-preshuffled B,
// shuffled E8M0 scales -- see aiter/ops/shuffle.py, aiter/utility/fp4_utils.py).
//
// Compute core: the gfx950 scaled MFMA
//   v_mfma_scale_f32_16x16x128_f8f6f4 (fmt=4 -> fp4 e2m1),
// decoded empirically with kernel/mfma_probe.hip:
//   * input fragment: lane L holds row (L%16), K-block g=(L/16); the block's
//     32 fp4 (16 bytes = i32x4) are the operand (fp4 ignores the nominal high
//     128 bits of the 256-bit f8f6f4 slot).
//   * scale is PER-LANE: lane L passes the E8M0 byte of its own (row, block)
//     in the low byte (opsel=0); block g = L/16 so the 4 lane-groups cover the
//     four 32-K blocks of the 128-K MFMA -> correct per-block MXFP4 scaling.
//   * output: acc[i] on lane L == C[(L/16)*4 + i][L%16].
//
// Layout decode (flat offsets):
//   B preshuffle, into the [N, K/2] contiguous buffer:
//     flatB(n,kp) = (n/16)*(KT*512) + (kp/32)*512 + ((kp%32)/16)*256
//                   + (n%16)*16 + (kp%16)         with KT = K/64
//   Scale shuffle, into the scale buffer (row r, K-block b):
//     flatS(r,b) = ((r%32)/16) + ((b%8)/4)*2 + ((r%32)%16)*4
//                  + ((b%8)%4)*64 + (b/8)*256 + (r/32)*32*scaleN_pad
//
// Performance design (what actually moves gfx950 MXFP4 GEMM throughput):
//   * The matrix engine is the limiter, NOT occupancy. Profiling shows the
//     production asm kernel sustains ~56% MFMA-unit utilization at <1 wave/SIMD;
//     occupancy does nothing. What keeps the pipe fed is MFMA ISSUE DENSITY --
//     many independent MFMAs in flight per K-step (deep ILP).
//   * So each wave computes a LARGE tile: up to 128x128 (MSUB=NSUB=8) = 64
//     independent 16x16 MFMAs per 128-K step. That needs 64 f32x4 accumulators
//     = 256 registers, which only fit if they live in the AGPR file -- hence the
//     inline-asm MFMA (mfma_fp4_acc, "+a" accumulator). The builtin keeps
//     accumulators in VGPRs (capped at 256) and cannot reach this tile size.
//   * The K loop is software-pipelined with ping-pong register buffers: this
//     overlaps the next chunk's loads with the current MFMAs AND is required for
//     correctness -- the MFMA reads its source VGPRs over many cycles, opaque to
//     the compiler's hazard model, so a single buffer lets the next load clobber
//     operands still in flight.
//   * The launcher picks a tile per problem (HUGE 128x128 down to SMALL 16x32)
//     so small M*N shapes, which can't fill the GPU with 128x128 tiles, instead
//     get more workgroups.

#include "gemm_a4w4.h"

#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>

#include <type_traits>
#include <utility>
#include <algorithm>
#include <cstdlib>  // getenv (MBS_FORCE_TILE_ENV debug builds)

#ifndef MBS_TIMING
#define MBS_TIMING 0  // per-phase s_memtime instrumentation, see mbs_timing_read
#endif

#if MBS_TIMING
// Debug instrumentation (MBS_TIMING=1): wg-kernel cycles (s_memtime) per
// phase, summed over all waves -- [0] section-0 wait, [1] section-0 body,
// [2] section-1 wait + barrier, [3] section-1 body, [4] epilogue, [5]/[6]
// section-0 wait of a tile's first/second step, [7] their section-1 waits;
// [8] wave count. Read (and optionally reset) from the host.
__device__ unsigned long long mbs_timing[16];
extern "C" void mbs_timing_read(unsigned long long* host, int reset)
{
    (void)hipMemcpyFromSymbol(host, HIP_SYMBOL(mbs_timing), sizeof(mbs_timing));
    if (reset)
    {
        unsigned long long z[16] = {};
        (void)hipMemcpyToSymbol(HIP_SYMBOL(mbs_timing), z, sizeof(z));
    }
}
#endif

namespace {

typedef int   i32x4 __attribute__((ext_vector_type(4)));
typedef int   i32x8 __attribute__((ext_vector_type(8)));
typedef float f32x4 __attribute__((ext_vector_type(4)));

// Flat offset of weight byte (n, kp) inside the preshuffled B buffer.
__device__ __forceinline__ int64_t flatB(int n, int kp, int KT)
{
    int n_tile = n >> 4;
    int n_in   = n & 15;
    int k_tile = kp >> 5;
    int rem    = kp & 31;
    int kpack  = rem >> 4;
    int k_in   = kp & 15;
    return (int64_t)n_tile * ((int64_t)KT * 512) + (int64_t)k_tile * 512 +
           (int64_t)kpack * 256 + (int64_t)n_in * 16 + k_in;
}

// Flat offset of scale byte for row r, K-block b inside the shuffled buffer.
__device__ __forceinline__ int64_t flatS(int r, int b, int scaleN_pad)
{
    int r32 = r & 31;
    int b8  = b & 7;
    return (int64_t)(r32 >> 4) + (int64_t)(b8 >> 2) * 2 + (int64_t)(r32 & 15) * 4 +
           (int64_t)(b8 & 3) * 64 + (int64_t)(b >> 3) * 256 +
           (int64_t)(r >> 5) * 32 * (int64_t)scaleN_pad;
}

// Load one lane's 32-fp4 (16-byte, 16-byte-aligned) K-block fragment. For the
// fp4 (fmt=4) MFMA the operand is 128 bits = 4 i32 (the instruction ignores the
// high 128 bits of the nominal 256-bit f8f6f4 slot), so we carry just an i32x4.
__device__ __forceinline__ i32x4 load_frag(const uint8_t* p, bool valid)
{
    return valid ? *reinterpret_cast<const i32x4*>(p) : i32x4{0, 0, 0, 0};
}

// Scaled MXFP4 MFMA with the accumulator pinned to the AGPR file ("+a"). Keeping
// the MSUB*NSUB f32x4 accumulators in AGPRs (not VGPRs) frees the capped
// 256-VGPR pool for software-pipeline buffers -- which is what lets the K loop
// prefetch. The asm is volatile, so MFMAs keep their order; the compiler may
// still place the (non-asm) loads between them.
//   D, A, B, C, scaleA, scaleB  (D==C == the accumulator, in place)
__device__ __forceinline__ void mfma_fp4_acc(f32x4& acc, i32x4 a, i32x4 b,
                                             int sa, int sb)
{
    asm volatile(
        "v_mfma_scale_f32_16x16x128_f8f6f4 %0, %1, %2, %0, %3, %4 "
        "op_sel_hi:[0,0,0] cbsz:4 blgp:4"
        : "+a"(acc)
        : "v"(a), "v"(b), "v"(sa), "v"(sb));
}

// 2D register blocking: each wave computes an (16*MSUB) x (16*NSUB) output tile.
// Per K-step it loads MSUB A fragments (+scales) and NSUB B fragments (+scales)
// and issues MSUB*NSUB MFMAs, so A traffic is cut by NSUB and B traffic by MSUB.
//
// MSUB/NSUB are a compile-time tile shape. Bigger tiles give more A/B reuse
// (higher peak on large shapes) but cost more VGPRs -> lower occupancy; smaller
// tiles raise occupancy and the workgroup count (better for small M*N). The
// launcher instantiates a few shapes and picks one per problem (see dispatch).

// One wave (64 lanes) computes a (16*MSUB) x (16*NSUB) output tile.
//
// launch_bounds min-waves/EU: the HUGE/TALL tiles (MSUB*NSUB >= 32, up to 256
// AGPR accumulators + 2x fragment buffers) get no occupancy constraint; BIG
// (16 MFMAs) asks for 2 waves/EU (256 registers each), which it fits without
// spilling. MID/SMALL are small enough that 1 (no constraint) lets them reach
// occupancy 5-7.
template <int MSUB, int NSUB>
__global__ void __launch_bounds__(64, MSUB* NSUB >= 32 ? 1 : (MSUB* NSUB >= 16 ? 2 : 1))
gemm_a4w4_mfma_kernel(const uint8_t* __restrict__ A,
                                      const uint8_t* __restrict__ B,
                                      const uint8_t* __restrict__ AScale,
                                      const uint8_t* __restrict__ BScale,
                                      __hip_bfloat16* __restrict__ out,
                                      int M, int N, int K, int Mpad,
                                      int scaleN_pad)
{
    constexpr int MTILE = 16 * MSUB;
    constexpr int NTILE = 16 * NSUB;
    const int lane  = threadIdx.x;   // 0..63
    const int row16 = lane & 15;     // fragment row/col index
    const int g     = lane >> 4;     // K-block group (0..3)
    const int m0    = blockIdx.y * MTILE;
    const int n0    = blockIdx.x * NTILE;

    const int Kp      = K >> 1;       // packed bytes per row
    const int KT      = K / 64;       // k_tiles for B preshuffle
    const int Kchunks = K / 128;      // 128-fp4 K steps (one MFMA each)

    f32x4 acc[MSUB][NSUB];
#pragma unroll
    for (int mi = 0; mi < MSUB; ++mi)
#pragma unroll
        for (int ni = 0; ni < NSUB; ++ni)
            acc[mi][ni] = f32x4{0.f, 0.f, 0.f, 0.f};

    // Load chunk kc's A/B fragments + E8M0 scales into the given buffers.
    auto load = [&](int kc, i32x4 (&af)[MSUB], int (&as)[MSUB],
                    i32x4 (&bf)[NSUB], int (&bs)[NSUB]) {
        const int kblk     = 4 * kc + g;
        const int ktile    = 2 * kc + (g >> 1);
        const int kpack    = g & 1;
        const int kpstart  = ktile * 32 + kpack * 16;
        const int64_t aoff = (int64_t)(kc * 64 + g * 16);
#pragma unroll
        for (int mi = 0; mi < MSUB; ++mi)
        {
            int am = m0 + mi * 16 + row16;
            af[mi] = load_frag(A + (int64_t)am * Kp + aoff, am < M);
            as[mi] = (am < M) ? AScale[flatS(am, kblk, scaleN_pad)] : 127;
        }
#pragma unroll
        for (int ni = 0; ni < NSUB; ++ni)
        {
            int  bn = n0 + ni * 16 + row16;
            bool vb = bn < N;
            bf[ni] = load_frag(B + flatB(bn, kpstart, KT), vb);
            bs[ni] = vb ? BScale[flatS(bn, kblk, scaleN_pad)] : 127;
        }
    };

    auto mma = [&](i32x4 (&af)[MSUB], int (&as)[MSUB], i32x4 (&bf)[NSUB],
                   int (&bs)[NSUB]) {
#pragma unroll
        for (int mi = 0; mi < MSUB; ++mi)
#pragma unroll
            for (int ni = 0; ni < NSUB; ++ni)
                mfma_fp4_acc(acc[mi][ni], af[mi], bf[ni], as[mi], bs[ni]);
    };

    // Software-pipelined K loop with statically named ping-pong buffers (A/B).
    // Besides overlapping loads with compute, double-buffering is REQUIRED for
    // correctness here: the MFMA reads its source VGPRs over many cycles, but
    // the inline asm is opaque to the compiler's hazard model, so a single
    // buffer would let the next chunk's loads clobber operands still being read.
    // Separate buffers keep each chunk's operands alive until its MFMAs finish.
    i32x4 afA[MSUB], bfA[NSUB], afB[MSUB], bfB[NSUB];
    int   asA[MSUB], bsA[NSUB], asB[MSUB], bsB[NSUB];
    load(0, afA, asA, bfA, bsA);

    int kc = 0;
    for (; kc + 1 < Kchunks; kc += 2)
    {
        load(kc + 1, afB, asB, bfB, bsB);   // prefetch odd
        mma(afA, asA, bfA, bsA);            // compute even
        if (kc + 2 < Kchunks)
            load(kc + 2, afA, asA, bfA, bsA);  // prefetch next even
        mma(afB, asB, bfB, bsB);            // compute odd
    }
    if (kc < Kchunks)                        // odd Kchunks tail
        mma(afA, asA, bfA, bsA);

    // acc[mi][ni][i] on lane L == C[(L/16)*4 + i][L%16] of subtile (mi, ni).
#pragma unroll
    for (int ni = 0; ni < NSUB; ++ni)
    {
        const int on = n0 + ni * 16 + row16;
        if (on >= N)
            continue;
#pragma unroll
        for (int mi = 0; mi < MSUB; ++mi)
#pragma unroll
            for (int i = 0; i < 4; ++i)
            {
                int om = m0 + mi * 16 + g * 4 + i;
                if (om < M)
                    out[(int64_t)om * N + on] =
                        __float2bfloat16(acc[mi][ni][i]);
            }
    }
}

// ---------------------------------------------------------------------------
// Workgroup-cooperative kernel (modeled on the production asm f4gemm 160x256).
//
// One workgroup = WAVES waves laid out 1 x WAVES along N; each wave computes a
// (16*MSUB) x (16*NSUB) tile, so the workgroup tile is (16*MSUB) x
// (16*NSUB*WAVES). Compared to the single-wave kernel above:
//   * A is shared by all waves of the workgroup through LDS. It is streamed
//     with direct-to-LDS buffer loads (no VGPR staging) into a 4-deep ring,
//     one K=256 stage (MTILE rows x 128 B) per buffer, 2 stages ahead of
//     compute. The 16 B chunks of each row are XOR-swizzled by (row & 7) so
//     the ds_read_b128 fragment reads (16 lanes on 16 rows, same K chunk) do
//     not bank-conflict.
//   * B (16x16 preshuffled) still goes straight from global into VGPRs -- each
//     wave owns distinct N columns, so there is nothing to share -- 2 stages
//     ahead (1 for the tiles that cannot afford a third B register set).
//   * E8M0 scales are one dword per lane per (16-row pair, K=256 stage): the
//     shuffled scale layout packs [tile parity][K-128 half] into the four
//     bytes, which the MFMA selects for free via op_sel (byte = 2*khalf + par).
//     B scales go into VGPRs; A scales, shared by all waves, are DMA'd once per
//     workgroup into the stage's LDS buffer behind the A data.
//   * Each K=256 stage runs as two sections (M halves) as in the asm kernel,
//     with A fragments read a whole section ahead and loads threaded one at a
//     time through the MFMA stream (see `step`). All global loads and the
//     MFMAs are inline asm, loads address the asm way (fixed per-op VGPR
//     offsets, per-stage descriptor base, see issue_one); the loop's vmcnt
//     waits are explicit and the accumulators are pinned to AGPRs.
//   * Persistent (one workgroup per CU at 1 wave/SIMD) with an XCD-aware tile
//     order; the pipeline runs across tile boundaries.
//   * The MFMA computes C^T (operands swapped) so each lane holds 4 contiguous
//     outputs of a row; the epilogue regroups them across lanes
//     (permlane16_swap + DPP) into 16-byte stores of whole 128 B row segments.
// Requirements (checked by the launcher): K % 1024 == 0 (K=256 stage count a
// multiple of the LDS ring), N % 16 == 0, MSUB/NSUB even, NSUB in {4, 6, 8}.
// ---------------------------------------------------------------------------

// Scaled fp4 MFMA with compile-time E8M0 byte selection (byte = op_sel_hi*2 +
// op_sel per operand), as inline asm with the accumulator pinned to AGPRs
// ("+a"). With the builtin, the register allocator is free to move
// accumulators between AGPRs and VGPRs; for the big tiles (up to 256
// accumulator registers) it does so inside the K loop, costing hundreds of
// accvgpr copies per stage. Hazard note: the asm is opaque to the hazard
// recognizer, but every accumulator is touched again only after many other
// MFMAs, and its fragment/scale sources are rewritten only by loads issued
// long after their last MFMA.
#define MBS_MFMA_FP4(OS0, OS1, OH0, OH1)                                                   \
    asm volatile("v_mfma_scale_f32_16x16x128_f8f6f4 %0, %1, %2, %0, %3, %4 op_sel:[" #OS0 \
                 "," #OS1 ",0] op_sel_hi:[" #OH0 "," #OH1 ",0] cbsz:4 blgp:4"              \
                 : "+a"(acc)                                                              \
                 : "v"(a), "v"(b), "v"(sa), "v"(sb))
template <int OA, int OB>
__device__ __forceinline__ void mfma_fp4_sel(f32x4& acc, i32x4 a, i32x4 b, int sa, int sb)
{
    static_assert(OA >= 0 && OA < 4 && OB >= 0 && OB < 4, "scale byte select");
    if constexpr (OA == 0 && OB == 0) MBS_MFMA_FP4(0, 0, 0, 0);
    else if constexpr (OA == 1 && OB == 0) MBS_MFMA_FP4(1, 0, 0, 0);
    else if constexpr (OA == 2 && OB == 0) MBS_MFMA_FP4(0, 0, 1, 0);
    else if constexpr (OA == 3 && OB == 0) MBS_MFMA_FP4(1, 0, 1, 0);
    else if constexpr (OA == 0 && OB == 1) MBS_MFMA_FP4(0, 1, 0, 0);
    else if constexpr (OA == 1 && OB == 1) MBS_MFMA_FP4(1, 1, 0, 0);
    else if constexpr (OA == 2 && OB == 1) MBS_MFMA_FP4(0, 1, 1, 0);
    else if constexpr (OA == 3 && OB == 1) MBS_MFMA_FP4(1, 1, 1, 0);
    else if constexpr (OA == 0 && OB == 2) MBS_MFMA_FP4(0, 0, 0, 1);
    else if constexpr (OA == 1 && OB == 2) MBS_MFMA_FP4(1, 0, 0, 1);
    else if constexpr (OA == 2 && OB == 2) MBS_MFMA_FP4(0, 0, 1, 1);
    else if constexpr (OA == 3 && OB == 2) MBS_MFMA_FP4(1, 0, 1, 1);
    else if constexpr (OA == 0 && OB == 3) MBS_MFMA_FP4(0, 1, 0, 1);
    else if constexpr (OA == 1 && OB == 3) MBS_MFMA_FP4(1, 1, 0, 1);
    else if constexpr (OA == 2 && OB == 3) MBS_MFMA_FP4(0, 1, 1, 1);
    else MBS_MFMA_FP4(1, 1, 1, 1);
}
#undef MBS_MFMA_FP4

// Two floats -> packed bf16x2 (RNE), one v_cvt_pk_bf16_f32 (the generic
// conversion lowers to a cvt + shift + or per pair).
__device__ __forceinline__ uint32_t pack_bf16x2(float lo, float hi)
{
    uint32_t r;
    asm("v_cvt_pk_bf16_f32 %0, %1, %2" : "=v"(r) : "v"(lo), "v"(hi));
    return r;
}

template <int... Is, typename F>
__device__ __forceinline__ void static_for_impl(std::integer_sequence<int, Is...>, F&& f)
{
    (f(std::integral_constant<int, Is>{}), ...);
}
template <int N, typename F>
__device__ __forceinline__ void static_for(F&& f)
{
    static_for_impl(std::make_integer_sequence<int, N>{}, f);
}

template <int MSUB, int NSUB, int WAVES>
__global__ void __launch_bounds__(64 * WAVES, 1)
gemm_a4w4_wg_kernel(const uint8_t* __restrict__ A,
                    const uint8_t* __restrict__ B,
                    const uint8_t* __restrict__ AScale,
                    const uint8_t* __restrict__ BScale,
                    const uint8_t* __restrict__ AFactor,
                    const uint8_t* __restrict__ BFactor,
                    __hip_bfloat16* __restrict__ out,
                    int M, int N, int K, int scaleN_pad)
{
    static_assert(MSUB % 2 == 0 && NSUB % 2 == 0, "scale dwords cover tile pairs");
    constexpr int MTILE   = 16 * MSUB;
    constexpr int WNTILE  = 16 * NSUB;            // per-wave N extent
    constexpr int NTILE   = WNTILE * WAVES;
    constexpr int STAGE_A = MTILE * 128;          // LDS bytes per K=256 A stage
    constexpr int A_LDS_PER_WAVE = MTILE / 8 / WAVES;  // 1 KB (8 rows) each
    // The stage's A scales follow the A data: one 256 B block (a dword per
    // lane) per 32-row pair of fragments, DMA'd once per workgroup, split over
    // the waves (the last block is repeated to even out the per-wave op count).
    constexpr int NSB     = MSUB / 2;                    // scale blocks per stage
    constexpr int S_LDS_PER_WAVE = (NSB + WAVES - 1) / WAVES;
    constexpr int STAGE_B = STAGE_A + NSB * 256;
    static_assert((MTILE / 8) % WAVES == 0, "A stage must split evenly across waves");

    // Ring of RING LDS stage buffers. A (via LDS) is prefetched PF stages ahead,
    // B and the scales (straight into registers) PF_B stages ahead. Each B
    // stage in flight costs a register set (2*NSUB*4 VGPRs): the 256-row and
    // the wide-N tiles, which run 2*MSUB*NSUB >= 128 MFMAs per stage, keep B
    // one stage ahead; the rest measured better at two.
    constexpr int RING = 4;
    constexpr int PF   = 2;
    constexpr int PF_B = (NSUB <= 4 && MSUB < 16) ? 2 : 1;
    // Step s refills buffer (s+PF) % RING in its section 1. Even with RING =
    // PF+1 that would be stage s-1's buffer, whose last fragment reads issue in
    // step s-1's section 0 and are retired at its section-1 barrier
    // (lgkmcnt(0)); RING = 4 (so nstages % RING == 0 for K % 1024 == 0) leaves
    // a stage of margin, which is why section 0 needs no barrier.
    static_assert(RING >= PF + 1, "refill target must be clear of readers");
    static_assert(PF_B <= PF, "B is never further ahead than A");
    __shared__ __attribute__((aligned(16))) uint8_t smem0[STAGE_B];
    __shared__ __attribute__((aligned(16))) uint8_t smem1[STAGE_B];
    __shared__ __attribute__((aligned(16))) uint8_t smem2[STAGE_B];
    __shared__ __attribute__((aligned(16))) uint8_t smem3[STAGE_B];
    // MBS factors of the current tile, as f32 (see byte_to_factor): the M
    // factors are shared by the workgroup, the N factors are per wave.
    static_assert(MTILE <= 64 * WAVES, "one M factor per thread");
    __shared__ __attribute__((aligned(16))) float fa_lds[MTILE];
    __shared__ __attribute__((aligned(16))) float fb_lds[WAVES][WNTILE];
    // Distinct arrays, statically selected per (unrolled) ring slot.
    auto lds_buf = [&](auto u) __attribute__((always_inline)) -> uint8_t* {
        constexpr int i = decltype(u)::value % RING;
        if constexpr (i == 0) return smem0;
        else if constexpr (i == 1) return smem1;
        else if constexpr (i == 2) return smem2;
        else return smem3;
    };

    const int lane  = threadIdx.x & 63;
    // readfirstlane: the compiler cannot tell threadIdx.x >> 6 is wave-uniform;
    // made explicit, every wave-derived tile base lives in SGPRs, not VGPRs.
    const int wave  = __builtin_amdgcn_readfirstlane(threadIdx.x >> 6);
    const int row16 = lane & 15;
    const int g     = lane >> 4;

    const int Kp      = K >> 1;
    const int KT      = K / 64;
    const int nstages = K / 256;

    // ---- persistent tile schedule ---------------------------------------------
    // One resident workgroup per CU walks tiles id = blockIdx.x + i*gridDim.x.
    // The software pipeline runs straight across tile boundaries: the next
    // tile's first stages are prefetched under the current tile's last MFMAs,
    // so only the very first tile pays the load latency before compute starts.
    //
    // Rasterization: workgroups are dispatched round-robin over the XCDs (each
    // with a private L2), so a plain row-major order gives every XCD a
    // scattered set of tiles with little B reuse. Remap so each XCD walks a
    // contiguous range of tile ids, ordered in GROUP_M-tall column groups: the
    // ~32 workgroups resident on one XCD then cover a GROUP_M x 8 block, sharing
    // each B tile GROUP_M times and each A tile ~8 times in L2. (gridDim.x is a
    // multiple of NXCD, so id % NXCD is the XCD of the workgroup for every i.)
    constexpr int NXCD = 8, GROUP_M = 4;
    const int num_m = (M + MTILE - 1) / MTILE;
    const int num_n = (N + NTILE - 1) / NTILE;
    const int total = num_m * num_n;

    // Load offsets = per-lane part (tile-independent, below) + per-tile part
    // (wave-uniform, kept in SGPRs and folded into the descriptor base, see
    // issue_one). Buffer loads bounds-check voffset + soffset + inst offset
    // against the descriptor's range, so everything past M / N (rows, subtiles,
    // scale blocks) reads 0 in hardware.
    struct TileOff
    {
        int m0, n0;  // n0 includes this wave's column offset
        int a, b, as, bs;
    };
    // A -> LDS: wave instruction j fills rows [8*grp, 8*grp+8) of the stage;
    // lane writes LDS slot (row = lane/8, phys chunk = lane%8) and so fetches
    // logical chunk phys ^ (row & 7). j advances 8 rows (a_jstride).
    const int a_lane = (lane >> 3) * Kp + (((lane & 7) ^ (lane >> 3)) * 16);
    // B fragment (n, K-128 chunk kc): 1 KB contiguous per 16-row subtile.
    const int b_lane = g * 256 + row16 * 16;
    // Scale dword of a 32-row block-stage: the shuffled layout is lane-major.
    const int s_lane = lane * 4;
    auto tile_off = [&](int id) __attribute__((always_inline)) {
        id = min(id, total - 1);  // past-the-end prefetches: any valid tile
        if (id < total / NXCD * NXCD)  // ragged tail keeps its natural id
            id = (id % NXCD) * (total / NXCD) + id / NXCD;
        const int group   = id / (GROUP_M * num_n);
        const int first_m = group * GROUP_M;
        const int gsize   = min(num_m - first_m, GROUP_M);
        TileOff t;
        t.m0 = (first_m + (id % (GROUP_M * num_n)) % gsize) * MTILE;
        t.n0 = (id % (GROUP_M * num_n)) / gsize * NTILE + wave * WNTILE;
        t.a  = (t.m0 + wave * A_LDS_PER_WAVE * 8) * Kp;
        t.b  = (t.n0 >> 4) * KT * 512;
        t.as = (t.m0 >> 5) * 32 * scaleN_pad;  // block holding subtiles (2p, 2p+1)
        t.bs = (t.n0 >> 5) * 32 * scaleN_pad;
        return t;
    };

    constexpr int RSRC_W3 = 0x00020000;
    const int Mpad32 = (M + 31) & ~31;
    const int Npad32 = (N + 31) & ~31;
    // Plain SGPR-quad resources for the inline-asm loads below.
    auto rsrc_of = [&](const void* p, int bytes) __attribute__((always_inline)) {
        const uint64_t a = reinterpret_cast<uint64_t>(p);
        return i32x4{(int)a, (int)(a >> 32) & 0xffff, bytes, RSRC_W3};
    };
    // readfirstlane once, here: the descriptors are then plain SGPR values for
    // the "s" operands below. (Applied at each use instead, the compiler parks
    // the descriptor words in VGPRs to feed v_readfirstlane, keeping ~10 VGPRs
    // live across the whole tile loop -- the 256x256 build spilled them.)
    auto sq = [](const i32x4& r) __attribute__((always_inline)) {
        return i32x4{__builtin_amdgcn_readfirstlane(r[0]), __builtin_amdgcn_readfirstlane(r[1]),
                     __builtin_amdgcn_readfirstlane(r[2]), __builtin_amdgcn_readfirstlane(r[3])};
    };
    const i32x4 a_rsrc  = sq(rsrc_of(A, M * Kp));
    const i32x4 b_rsrc  = sq(rsrc_of(B, N * Kp));
    const i32x4 as_rsrc = sq(rsrc_of(AScale, Mpad32 * scaleN_pad));
    const i32x4 bs_rsrc = sq(rsrc_of(BScale, Npad32 * scaleN_pad));
    const i32x4 af_rsrc = sq(rsrc_of(AFactor, M));
    const i32x4 bf_rsrc = sq(rsrc_of(BFactor, N));
    const auto o_rsrc  = __builtin_amdgcn_make_buffer_rsrc(
        (void*)out, 0, M * N * (int)sizeof(__hip_bfloat16), RSRC_W3);

    f32x4 acc[MSUB][NSUB];
#pragma unroll
    for (int mi = 0; mi < MSUB; ++mi)
#pragma unroll
        for (int ni = 0; ni < NSUB; ++ni)
            acc[mi][ni] = f32x4{0.f, 0.f, 0.f, 0.f};

    // Global traffic of one pipeline step, as LOADS_PER_STAGE individually
    // addressable ops (k), in issue order: B fragments, B scales (into the
    // given registers), then A rows and A scales into LDS buffer `lds`.
    //
    // All of it is inline asm, and every vmcnt wait in the K loop is explicit
    // (see W_START / W_MID). The LDS DMA cannot be left to the compiler (it
    // drains vmcnt(0) before every LDS read it cannot prove disjoint), and once
    // it is invisible, compiler-placed waits for B miscount either way --
    // forcing the next stage's A DMA to land early. The loaded registers are
    // only read by the (asm) MFMAs, after the explicit wait.
    constexpr int NB              = 2 * NSUB;
    constexpr int LOADS_PER_STAGE = NB + NSUB / 2 + A_LDS_PER_WAVE + S_LDS_PER_WAVE;
    // op ranges: [0, KBS) B, [KBS, KA) B scales, [KA, KAS) A DMA, [KAS, L) A-scale DMA
    constexpr int KBS = NB, KA = KBS + NSUB / 2, KAS = KA + A_LDS_PER_WAVE;
    const uint32_t lds_wave_off   = wave * A_LDS_PER_WAVE * 1024;
    const int a_jstride           = 8 * Kp;  // A DMA op j covers rows 8j..8j+7
    // A ops load stage `st` of tile `to`; B/scale ops load stage `stb` of tile
    // `tob`.
    // Load addressing as in the asm kernel: each load op has its own fixed
    // per-lane VGPR offset (computed once; opaque, so the compiler keeps it
    // rather than re-adding it in front of every load), soffset is 0, and the
    // per-tile/per-stage part moves the descriptor base -- a few SALU per
    // descriptor per step instead of a SALU chain per load. The descriptor's
    // range shrinks by the same amount, so the hardware range check (which
    // covers voffset + inst offset) still clips rows past M / N. Measured: an
    // SGPR soffset costs the SQ a second cycle to send each VMEM instruction
    // (SQ_INST_CYCLES_VMEM_RD), and a per-load SALU (or VALU) offset chain in
    // front of each load stalls the in-order stream; together ~6% per stage.
    auto opaque = [](int v) __attribute__((always_inline)) {
        asm volatile("" : "+v"(v));
        return v;
    };
    // On the 256-row tile the scale loads (NSUB/2 + S_LDS_PER_WAVE per stage)
    // share one lane offset and take their block offset as SGPR soffset
    // instead: one more issue cycle each, but 4 VGPRs saved -- that tile
    // otherwise spills across its epilogue.
    constexpr bool SCALE_SOFF = MSUB >= 16;
    int a_voff[A_LDS_PER_WAVE], b_voff[NSUB];
    int bs_voff[SCALE_SOFF ? 1 : NSUB / 2], as_voff[SCALE_SOFF ? 1 : S_LDS_PER_WAVE];
    static_for<A_LDS_PER_WAVE>([&](auto j) { a_voff[j] = opaque(a_lane + j * a_jstride); });
    static_for<NSUB>([&](auto ni) { b_voff[ni] = opaque(b_lane + ni * KT * 512); });
    if constexpr (SCALE_SOFF)
    {
        bs_voff[0] = as_voff[0] = opaque(s_lane);
    }
    else
    {
        static_for<NSUB / 2>([&](auto p) { bs_voff[p] = opaque(s_lane + p * 32 * scaleN_pad); });
        static_for<S_LDS_PER_WAVE>([&](auto j) {
            as_voff[j] = opaque(s_lane + min(j * WAVES + wave, NSB - 1) * 32 * scaleN_pad);
        });
    }
    auto rsrc_at = [](const i32x4& r, int off) __attribute__((always_inline)) {
        const uint64_t a = (((uint64_t)(uint32_t)r[1] << 32) | (uint32_t)r[0]) + (uint32_t)off;
        return i32x4{(int)a, (int)(a >> 32), max(r[2] - off, 0), r[3]};
    };
    auto issue_one = [&](auto kc, const TileOff& to, int st, uint8_t* lds, const TileOff& tob,
                         int stb, i32x4 (&bf)[NSUB][2],
                         int (&bs)[NSUB / 2]) __attribute__((always_inline)) {
        constexpr int k = decltype(kc)::value;
        // clang: a capture used only in a dependent branch is missed
        (void)a_jstride, (void)a_rsrc, (void)b_rsrc, (void)as_rsrc, (void)bs_rsrc;
        (void)a_lane, (void)b_lane, (void)s_lane, (void)wave;
        (void)a_voff, (void)b_voff, (void)bs_voff, (void)as_voff, (void)rsrc_at;
        if constexpr (k < KBS)
        {
            constexpr int ni = k / 2, h = k % 2;
            asm volatile("buffer_load_dwordx4 %0, %1, %2, 0 offen offset:%3"
                         : "=v"(bf[ni][h])
                         : "v"(b_voff[ni]), "s"(rsrc_at(b_rsrc, tob.b + stb * 2048)), "i"(h * 1024));
        }
        else if constexpr (k < KA)
        {
            constexpr int p = k - KBS;
            if constexpr (SCALE_SOFF)
                asm volatile("buffer_load_dword %0, %1, %2, %3 offen"
                             : "=v"(bs[p])
                             : "v"(bs_voff[0]), "s"(rsrc_at(bs_rsrc, tob.bs + stb * 256)),
                               "s"(__builtin_amdgcn_readfirstlane(p * 32 * scaleN_pad)));
            else
                asm volatile("buffer_load_dword %0, %1, %2, 0 offen"
                             : "=v"(bs[p])
                             : "v"(bs_voff[p]), "s"(rsrc_at(bs_rsrc, tob.bs + stb * 256)));
        }
        else
        {
            const uint32_t lds_base = (uint32_t)(uintptr_t)(__attribute__((address_space(3))) void*)lds;
            if constexpr (k < KAS)
            {
                constexpr int j = k - KA;
                asm volatile("s_mov_b32 m0, %0\n\t"
                             "buffer_load_dwordx4 %1, %2, 0 offen lds"
                             :
                             : "s"(lds_base + lds_wave_off + j * 1024), "v"(a_voff[j]),
                               "s"(rsrc_at(a_rsrc, to.a + st * 128))
                             : "memory", "m0");
            }
            else
            {
                constexpr int j = k - KAS;
                const int p = min(j * WAVES + wave, NSB - 1);
                if constexpr (SCALE_SOFF)
                    asm volatile("s_mov_b32 m0, %0\n\t"
                                 "buffer_load_dword %1, %2, %3 offen lds"
                                 :
                                 : "s"(lds_base + STAGE_A + p * 256), "v"(as_voff[0]),
                                   "s"(rsrc_at(as_rsrc, to.as + st * 256)),
                                   "s"(__builtin_amdgcn_readfirstlane(p * 32 * scaleN_pad))
                                 : "memory", "m0");
                else
                    asm volatile("s_mov_b32 m0, %0\n\t"
                                 "buffer_load_dword %1, %2, 0 offen lds"
                                 :
                                 : "s"(lds_base + STAGE_A + p * 256), "v"(as_voff[j]),
                                   "s"(rsrc_at(as_rsrc, to.as + st * 256))
                                 : "memory", "m0");
            }
        }
    };

    // MBS factors (one byte per row of A and per row of B), applied in the
    // epilogue as C[m][n] *= f(AFactor[m]) * f(BFactor[n]). A tile's factor
    // bytes are loaded right after the previous tile's epilogue (in the
    // prologue for the first tile) with FAC_LOADS ops: thread t gets M factor
    // t, lane l N factors FB*l..FB*l+FB-1 of its wave. Like the other loads
    // these are explicit-vmcnt asm; the bytes have landed by step 1's
    // section-1 wait (its target, stage 2's A DMA, is issued later), where
    // they are converted and written to fa_lds / fb_lds. Every wave has
    // passed the step-1 barrier only after the previous epilogue (the last
    // reader), and the step-2 barrier publishes the writes to the epilogue.
    // Rows past M / N read 0 (range check) and are never stored.
    constexpr int FB        = (WNTILE + 63) / 64;  // N factor bytes per lane
    constexpr int FAC_LOADS = 2;
    static_assert(FB <= 2, "N factors: ubyte / ushort per lane");
    int fac_a, fac_b;  // raw factor bytes, in flight from issue to step 1
    auto issue_factors = [&](const TileOff& t) __attribute__((always_inline)) {
        // Lane id recomputed per issue (as in the epilogue): a long-lived
        // voffset spills on the 256x256 tile, and its reload drains vmcnt.
        int ln;
        asm volatile("v_mbcnt_lo_u32_b32 %0, -1, 0\n\t"
                     "v_mbcnt_hi_u32_b32 %0, -1, %0"
                     : "=v"(ln));
        asm volatile("buffer_load_ubyte %0, %1, %2, 0 offen"
                     : "=v"(fac_a)
                     : "v"(ln), "s"(rsrc_at(af_rsrc, t.m0 + wave * 64)));
        if constexpr (FB == 1)
            asm volatile("buffer_load_ubyte %0, %1, %2, 0 offen"
                         : "=v"(fac_b)
                         : "v"(ln), "s"(rsrc_at(bf_rsrc, t.n0)));
        else
            asm volatile("buffer_load_ushort %0, %1, %2, 0 offen"
                         : "=v"(fac_b)
                         : "v"(ln * 2), "s"(rsrc_at(bf_rsrc, t.n0)));
    };
    // MBS factor byte b -> 1 / (1 + b/256): b fills the top 8 mantissa bits
    // of 1.0f (v_lshl_or_b32). The IEEE f32 division expands to ~10 VALU
    // (div_scale / div_fmas / div_fixup); v_rcp_f32 alone is 1 ulp off for
    // 21 of the 256 bytes, and one Newton step makes it bit-exact for all
    // of them (checked exhaustively on gfx950 against the double reference).
    auto byte_to_factor = [](uint32_t b) __attribute__((always_inline)) {
        const float x = __uint_as_float(0x3F800000u | (b << 15));
        const float r = __builtin_amdgcn_rcpf(x);
        return __builtin_fmaf(r, __builtin_fmaf(-x, r, 1.0f), r);
    };
    auto write_factors = [&]() __attribute__((always_inline)) {
        // Volatile asm, ordered after the explicit vmcnt wait: the compiler
        // sees the asm-loaded bytes as ready at issue and would otherwise
        // hoist the conversion above the wait; and the lane id, recomputed
        // here, keeps the LDS addresses from living (and spilling) across
        // the K loop.
        int a = fac_a, b = fac_b, ln;
        asm volatile("v_mbcnt_lo_u32_b32 %0, -1, 0\n\t"
                     "v_mbcnt_hi_u32_b32 %0, -1, %0"
                     : "=v"(ln), "+v"(a), "+v"(b));
        const int t = wave * 64 + ln;
        if (t < MTILE)
            fa_lds[t] = byte_to_factor(a);
#pragma unroll
        for (int i = 0; i < FB; ++i)
            if (ln * FB + i < WNTILE)
                fb_lds[wave][ln * FB + i] = byte_to_factor((b >> (8 * i)) & 0xff);
    };

    // One pipeline step (stage s), modeled on the asm kernel's loop: two
    // sections of MFMAs, section 0 over M fragments [0, MS0), section 1 over
    // [MS0, MSUB), each over both K-128 halves.
    //   * Each section starts with lgkmcnt(0): its A fragments, ds_read during
    //     the previous section, have landed. Section 1 also waits for stage
    //     s+1's A DMA and barriers, publishing it to every wave. That is the only
    //     barrier per stage: with RING > PF + 1 no buffer is refilled while a
    //     wave can still read it (a section-0 barrier as in the asm kernel only
    //     measured ~2% slower).
    //   * During a section the wave ds_reads ALL A fragments of the next section
    //     (section 1 of this stage, or section 0 of stage s+1) into the other
    //     fragment buffer: a whole section of latency slack, and no LDS waits
    //     between the MFMAs.
    //   * Section 0 threads the register loads (B fragments and scales) of stage
    //     s+PF_B through its first LOAD_WIN/8 MFMAs, section 1 the A and A-scale
    //     DMA of stage s+PF, one load at a time (issuing ~20 back to back stalls
    //     the in-order stream on the VMEM queue).
    //   * MFMA order inside a section is the asm's 2x2 zig-zag: per K half, per B
    //     fragment pair, per A fragment pair: (n,m) (n,m+1) (n+1,m) (n+1,m+1).
    // sched_barriers pin this interleave, which the compiler would re-cluster.
    constexpr int MS0      = (MSUB / 2 + 1) / 2 * 2;  // even, so both sections pair up
    constexpr int MS1      = MSUB - MS0;
    constexpr int LOAD_WIN = 7;  // measured: ~2% faster per stage than spreading over all 8/8
    static_assert(PF >= 2, "stage s+1's A DMA is issued before step s");
    static_assert(2 * MS0 * NSUB >= KA && 2 * MS1 * NSUB >= LOADS_PER_STAGE - KA,
                  "need at least one MFMA per load");
    // vmcnt bookkeeping (vmcnt retires in issue order; each step issues its
    // register loads in section 0, its A DMA last, in section 1):
    //  * W_START, before step s's first MFMA: its B/scale registers (issued in
    //    step s-PF_B) have landed once at most the newer loads are left: that
    //    step's A DMA plus all of every later step.
    //  * W_MID, at the section-1 barrier: the A DMA for stage s+1 (the tail of
    //    step s+1-PF) has landed once only the later loads are left: whole steps
    //    s+2-PF..s-1 and this step's register loads.
    //  * A tile's epilogue issues EPI_STORES buffer stores (vmcnt counts them
    //    too) and then the next tile's FAC_LOADS factor loads between steps,
    //    so the first step of the next tile allows those.
    constexpr int EPI_STORES = MSUB * (NSUB / 2) + FAC_LOADS;
    constexpr int W_START    = (LOADS_PER_STAGE - KA) + LOADS_PER_STAGE * (PF_B - 1);
    constexpr int W_MID      = KA + LOADS_PER_STAGE * (PF - 2);
    static_assert(W_MID + EPI_STORES <= 63 && W_START + EPI_STORES <= 63, "vmcnt is 6 bits");
    // s_waitcnt simm16 (gfx9): vmcnt[3:0] | expcnt[6:4] | lgkmcnt[11:8] | vmcnt[15:14]
    auto wait_vm_lgkm0 = [](auto nc) __attribute__((always_inline)) {
        constexpr int n = decltype(nc)::value;
        __builtin_amdgcn_s_waitcnt((n & 15) | (7 << 4) | (0 << 8) | ((n >> 4) << 14));
    };
    constexpr int WAIT_ALL = (7 << 4) | (15 << 8);

    auto frag_at = [&](const uint8_t* lds, int h, int mi) __attribute__((always_inline)) {
        const int row = mi * 16 + row16;
        const int chk = (h * 4 + g) ^ (row & 7);
        return *reinterpret_cast<const i32x4*>(lds + row * 128 + chk * 16);
    };
    auto scale_at = [&](const uint8_t* lds, int p) __attribute__((always_inline)) {
        return *reinterpret_cast<const int*>(lds + STAGE_A + p * 256 + lane * 4);
    };
    // A fragment and A-scale buffers: section 0 computes from afr0/asc0 and
    // fills afr1/asc1, section 1 the reverse. Fragment index = h * (section's
    // M count) + local mi; scale index = local mi / 2.
    i32x4 afr0[2 * MS0], afr1[2 * MS0];
    int asc0[MS0 / 2], asc1[MS0 / 2];

#if MBS_TIMING
    // Phase timer: tick(b) adds the time since the previous tick to bucket b.
    // Buckets: 0 sec0 wait, 1 sec0 body, 2 sec1 wait, 3 sec1 body, 4 epilogue,
    // 5/6 sec0 wait of the 1st/2nd step after an epilogue, 7 their sec1 waits.
    unsigned long long t_last;
    asm volatile("s_memtime %0\n\ts_waitcnt lgkmcnt(0)" : "=s"(t_last));
    uint32_t tacc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    auto tick = [&](int bkt) __attribute__((always_inline)) {
        unsigned long long now;
        asm volatile("s_memtime %0\n\ts_waitcnt lgkmcnt(0)" : "=s"(now));
        tacc[bkt] += (uint32_t)(now - t_last);
        t_last = now;
    };
#define MBS_TICK(b) tick(b);
#else
#define MBS_TICK(b)
#endif
    // Section SIG of a step: TS MFMAs over its MC M fragments, while reading
    // the next section's MN fragments (and their scales) from rd_lds and
    // issuing loads [KLO, KLO + NL).
    auto section = [&](auto sc, i32x4 (&cf)[2 * MS0], int (&cs)[MS0 / 2], i32x4 (&nf)[2 * MS0],
                       int (&ns)[MS0 / 2], const uint8_t* rd_lds, i32x4 (&bf)[NSUB][2],
                       int (&bs)[NSUB / 2], const TileOff& nto, int nst, uint8_t* nlds,
                       const TileOff& ntob, int nstb, i32x4 (&nbf)[NSUB][2],
                       int (&nbs)[NSUB / 2]) __attribute__((always_inline)) {
        constexpr int SIG = decltype(sc)::value;
        constexpr int MC = SIG == 0 ? MS0 : MS1, MB = SIG == 0 ? 0 : MS0;    // computed
        constexpr int MN = SIG == 0 ? MS1 : MS0, MBN = SIG == 0 ? MS0 : 0;  // read ahead
        constexpr int TS  = 2 * MC * NSUB;
        constexpr int NF  = 2 * MN, NR = NF + MN / 2;  // fragment + scale reads
        constexpr int KLO = SIG == 0 ? 0 : KA, NL = (SIG == 0 ? KA : LOADS_PER_STAGE) - KLO;
        constexpr int TLW = TS * LOAD_WIN / 8 >= NL ? TS * LOAD_WIN / 8 : NL;
        static_assert(NR <= TS, "one fragment read per MFMA at most");
        static_for<TS>([&](auto tc) __attribute__((always_inline)) {
            constexpr int t = decltype(tc)::value;
            // fragment/scale read r goes before MFMA floor(r*TS/NR)
            constexpr int r = (t * NR + TS - 1) / TS;
            if constexpr (r < NR && r * TS / NR == t)
            {
                if constexpr (r < NF)
                    nf[r] = frag_at(rd_lds, r / MN, MBN + r % MN);
                else
                    ns[r - NF] = scale_at(rd_lds, MBN / 2 + r - NF);
                __builtin_amdgcn_sched_barrier(0);
            }
            // load KLO+k goes before MFMA floor(k*TLW/NL)
            constexpr int k = (t * NL + TLW - 1) / TLW;
            if constexpr (k < NL && k * TLW / NL == t)
            {
                issue_one(std::integral_constant<int, KLO + k>{}, nto, nst, nlds, ntob, nstb, nbf,
                          nbs);
                __builtin_amdgcn_sched_barrier(0);
            }
            // 2x2 zig-zag: t -> (h, B pair np, A pair mp, (q>>1, q&1))
            constexpr int q = t % 4, mp = (t / 4) % (MC / 2), np = (t / 2 / MC) % (NSUB / 2);
            constexpr int h = t / (MC * NSUB), ml = 2 * mp + (q & 1), ni = 2 * np + (q >> 1);
            constexpr int mi = MB + ml;
            // Operands swapped (B first): computes the C^T subtile, so lane L
            // ends up holding row L%16, columns (L/16)*4..+3 -- 4 contiguous
            // outputs of one row (see the epilogue).
            mfma_fp4_sel<2 * h + (ni & 1), 2 * h + (mi & 1)>(acc[mi][ni], bf[ni][h],
                                                             cf[h * MC + ml], bs[ni / 2], cs[ml / 2]);
            __builtin_amdgcn_sched_barrier(0);
        });
    };

    auto step = [&](const uint8_t* lds, const uint8_t* next_lds, i32x4 (&bf)[NSUB][2],
                    int (&bs)[NSUB / 2], const TileOff& nto, int nst, uint8_t* nlds,
                    const TileOff& ntob, int nstb, i32x4 (&nbf)[NSUB][2], int (&nbs)[NSUB / 2],
                    bool after_epi, int sidx) __attribute__((always_inline)) {
        MBS_TICK(3)
        // This stage's B/scale registers and section-0 fragments have landed.
        if (after_epi)
            wait_vm_lgkm0(std::integral_constant<int, W_START + EPI_STORES>{});
        else
            wait_vm_lgkm0(std::integral_constant<int, W_START>{});
        MBS_TICK(sidx == 0 ? 5 : sidx == 1 ? 6 : 0)
        __builtin_amdgcn_sched_barrier(0);
        section(std::integral_constant<int, 0>{}, afr0, asc0, afr1, asc1, lds, bf, bs, nto, nst,
                nlds, ntob, nstb, nbf, nbs);
        MBS_TICK(1)
        // Stage s+1's A DMA has landed in every wave (and its section-0
        // fragments may be read); this wave's section-1 fragments have landed.
        if (after_epi)
            wait_vm_lgkm0(std::integral_constant<int, W_MID + EPI_STORES>{});
        else
            wait_vm_lgkm0(std::integral_constant<int, W_MID>{});
        __syncthreads();
        MBS_TICK(sidx < 2 ? 7 : 2)
        if (sidx == 1)
            write_factors();
        __builtin_amdgcn_sched_barrier(0);
        section(std::integral_constant<int, 1>{}, afr1, asc1, afr0, asc0, next_lds, bf, bs, nto,
                nst, nlds, ntob, nstb, nbf, nbs);
    };

    // Epilogue: acc[mi][ni][i] on lane L == C[mi*16 + L%16][ni*16 + (L/16)*4 + i].
    // Per (mi, N-subtile pair ni = 2np, 2np+1), pack to bf16 and exchange halves
    // between lane groups with v_permlane16_swap (odd groups of the first operand
    // <-> even groups of the second): lane group g then holds the 16 bytes of
    // columns np*32 + (g&1)*16 + (g>>1)*8 .. +7 of its row -- 64 B per row over
    // the wave. Two such pairs (np = 2q, 2q+1) are then recombined with a DPP
    // row_ror:8, so that each 16-byte store covers 8 whole 128 B rows instead of
    // 16 half rows: under a full-GPU write burst, half-line stores take ~2.5x
    // longer to be acknowledged, and the next tile's loads queue behind them
    // (vmcnt retires in order). No LDS round trip. Rows >= M fall outside the
    // buffer (the range check covers voffset + soffset) and are dropped; chunks
    // past N (N % WNTILE != 0) get an out-of-range offset.
    auto epilogue = [&](const TileOff& to) __attribute__((always_inline)) {
        // Lane id from volatile asm: the compiler cannot hoist the per-lane
        // epilogue addresses out of the tile loop (or keep threadIdx.x live
        // across it) -- hoisted values stay live across the K loop and spill,
        // and every spill reload drains the whole prefetch pipeline (vmcnt(0)).
        int lane;
        asm volatile("v_mbcnt_lo_u32_b32 %0, -1, 0\n\t"
                     "v_mbcnt_hi_u32_b32 %0, -1, %0"
                     : "=v"(lane));
        const int row16 = lane & 15, g = lane >> 4;
        using u32x4 = __attribute__((ext_vector_type(4))) unsigned int;
        // 64 B per row of N-subtile pair np: {s0[0], s1[0], s0[1], s1[1]}.
        auto pair = [&](int mi, int np) __attribute__((always_inline)) {
            // AGPR -> VGPR copy pinned here by the volatile asm: left to the
            // register allocator, all 256 accumulator reads get hoisted to the
            // top of the epilogue, and spilled.
            f32x4 x, y;
            asm volatile("" : "=v"(x) : "0"(acc[mi][2 * np]));
            asm volatile("" : "=v"(y) : "0"(acc[mi][2 * np + 1]));
            // MBS: C *= f(AFactor[row]) * f(BFactor[col]), before the bf16 cast.
            const float  fa  = fa_lds[mi * 16 + row16];
            const f32x4 fb0 = *reinterpret_cast<const f32x4*>(&fb_lds[wave][2 * np * 16 + g * 4]);
            const f32x4 fb1 = *reinterpret_cast<const f32x4*>(&fb_lds[wave][(2 * np + 1) * 16 + g * 4]);
#pragma unroll
            for (int i = 0; i < 4; ++i)
            {
                x[i] *= fa * fb0[i];
                y[i] *= fa * fb1[i];
            }
            // Reset for the next tile right after the last read, made opaque
            // ("+a"): a plain constant reset is folded into the tile loop's
            // accumulator phi, and the register allocator then shuffles (and
            // spills) the accumulators around it.
            acc[mi][2 * np] = acc[mi][2 * np + 1] = f32x4{0.f, 0.f, 0.f, 0.f};
            asm volatile("" : "+a"(acc[mi][2 * np]), "+a"(acc[mi][2 * np + 1]));
            const auto s0 = __builtin_amdgcn_permlane16_swap(
                pack_bf16x2(x[0], x[1]), pack_bf16x2(y[0], y[1]), false, false);
            const auto s1 = __builtin_amdgcn_permlane16_swap(
                pack_bf16x2(x[2], x[3]), pack_bf16x2(y[2], y[3]), false, false);
            return u32x4{s0[0], s1[0], s0[1], s1[1]};
        };
        // Whole offset in voffset (soffset 0): per-row-block soffsets would be
        // hoisted into MSUB long-lived SGPRs.
        constexpr int NQ = NSUB / 4;  // 128 B pairs of pairs; an odd pair stays 64 B
        const bool lo8   = row16 < 8;
        const int  ccol  = to.n0 + (row16 >> 3) * 32 + (g & 1) * 16 + (g >> 1) * 8;  // full-line
        const int  hcol  = to.n0 + (g & 1) * 16 + (g >> 1) * 8;                       // half-line
        int vofs_f = ((to.m0 + (row16 & 7)) * N + ccol) * 2;
        int vofs_h = ((to.m0 + row16) * N + hcol) * 2;
#pragma unroll
        for (int mi = 0; mi < MSUB; ++mi)
        {
            // Fence each row block: unfenced, the scheduler hoists the
            // AGPR->VGPR accumulator reads of many blocks to the top.
            __builtin_amdgcn_sched_barrier(0);
#pragma unroll
            for (int q = 0; q < NQ; ++q)
            {
                const u32x4 v0 = pair(mi, 2 * q), v1 = pair(mi, 2 * q + 1);
                u32x4 va, vb;  // rows 0-7 / 8-15 of the block, 128 B each
#pragma unroll
                for (int i = 0; i < 4; ++i)
                {
                    const unsigned r0 = __builtin_amdgcn_mov_dpp(v0[i], 0x128, 0xF, 0xF, false);
                    const unsigned r1 = __builtin_amdgcn_mov_dpp(v1[i], 0x128, 0xF, 0xF, false);
                    va[i] = lo8 ? v0[i] : r1;  // row_ror:8 -- lane L reads L^8 in its row
                    vb[i] = lo8 ? r0 : v1[i];
                }
                // Unconditional: the K loop's vmcnt bookkeeping counts exactly
                // EPI_STORES stores per epilogue.
                const bool in = ccol + q * 64 < N;
                __builtin_amdgcn_raw_buffer_store_b128(va, o_rsrc, in ? vofs_f + q * 128 : 0x7fffffff,
                                                       0, 0);
                __builtin_amdgcn_raw_buffer_store_b128(
                    vb, o_rsrc, in ? vofs_f + 8 * N * 2 + q * 128 : 0x7fffffff, 0, 0);
            }
            if constexpr ((NSUB / 2) % 2)
            {
                constexpr int np = NSUB / 2 - 1;
                __builtin_amdgcn_raw_buffer_store_b128(
                    pair(mi, np), o_rsrc, hcol + np * 32 < N ? vofs_h + np * 64 : 0x7fffffff, 0, 0);
            }
            vofs_f += 16 * N * 2;
            vofs_h += 16 * N * 2;
        }
    };

    // Software pipeline: stage s's A loads are issued while stages s-PF..s-1
    // compute (B: PF_B), the stage index running on across tiles
    // (stage >= nstages means stage - nstages of the next tile). Unrolled by
    // RING so every LDS buffer and B/scale register set is statically named;
    // nstages % RING == 0, so every tile starts on ring slot 0. The barrier
    // inside step s both publishes every wave's LDS DMA for stage s+1 and
    // guarantees stage s-1 is fully read before step s+1 refills its buffer.
    // Prefetches past the last tile still issue (for a clamped valid tile)
    // rather than branch: a conditional issue would keep every ring register
    // set live across the loop and spill. They land in a buffer / register set
    // nobody reads.
    i32x4 bf[RING][NSUB][2];
    int bs[RING][NSUB / 2];

    int id      = blockIdx.x;
    TileOff cur = tile_off(id);
    TileOff nxt = tile_off(id + gridDim.x);
    issue_factors(cur);  // older than every stage load: landed by step 1
    // Prologue, laid out as the steps "-PF..-1" would issue it so the vmcnt
    // bookkeeping is identical: step -PF+u issues A stage u, B stage u+PF_B-PF.
    static_for<PF>([&](auto u) __attribute__((always_inline)) {
        constexpr int sb = u + PF_B - PF;
        static_for<LOADS_PER_STAGE>([&](auto k) __attribute__((always_inline)) {
            if constexpr (decltype(k)::value >= KA || sb >= 0)  // skip B ops of stages < 0
                issue_one(k, cur, u, lds_buf(u), cur, sb, bf[sb < 0 ? 0 : sb],
                          bs[sb < 0 ? 0 : sb]);
        });
    });
    // Stage 0 must have landed before its section-0 fragments are read; later
    // stages do this in the previous step's section 1.
    wait_vm_lgkm0(std::integral_constant<int, LOADS_PER_STAGE * (PF - 1)>{});
    __syncthreads();
    static_for<2 * MS0>([&](auto r) __attribute__((always_inline)) {
        afr0[r] = frag_at(smem0, r / MS0, r % MS0);
    });
    static_for<MS0 / 2>([&](auto p) __attribute__((always_inline)) { asc0[p] = scale_at(smem0, p); });
    bool after_epi = false;  // first step of a tile that follows an epilogue
    for (; id < total; id += gridDim.x)
    {
        for (int st = 0; st < nstages; st += RING)
        {
            static_for<RING>([&](auto u) __attribute__((always_inline)) {
                constexpr int nu  = (u + PF) % RING;
                constexpr int nub = (u + PF_B) % RING;
                const int nst     = st + u + PF;
                const int nstb    = st + u + PF_B;
                const bool wrap   = nst >= nstages;   // uniform
                const bool wrapb  = nstb >= nstages;  // uniform
                const TileOff nto{0, 0, wrap ? nxt.a : cur.a, 0, wrap ? nxt.as : cur.as, 0};
                const TileOff ntob{0, 0, 0, wrapb ? nxt.b : cur.b, 0, wrapb ? nxt.bs : cur.bs};
                step(lds_buf(u), lds_buf(std::integral_constant<int, u + 1>{}), bf[u], bs[u], nto, wrap ? nst - nstages : nst,
                     lds_buf(std::integral_constant<int, nu>{}), ntob,
                     wrapb ? nstb - nstages : nstb, bf[nub], bs[nub],
                     u == 0 && st == 0 && after_epi, st + u);
            });
        }
        MBS_TICK(3)
        epilogue(cur);
        MBS_TICK(4)
        after_epi = true;
        cur = nxt;
        nxt = tile_off(id + 2 * gridDim.x);
        issue_factors(cur);
    }
    __builtin_amdgcn_s_waitcnt(WAIT_ALL);  // retire the overrun LDS DMA
#if MBS_TIMING
    if (lane == 0)
    {
        for (int i = 0; i < 8; ++i)
            atomicAdd(&mbs_timing[i], (unsigned long long)tacc[i]);
        atomicAdd(&mbs_timing[8], 1ull);
    }
#endif
}

}  // namespace

// Persistent launch of the workgroup kernel: one workgroup per CU (1 wave/SIMD).
template <int MS, int NS>
void launch_wg(const uint8_t* A, const uint8_t* B, const uint8_t* AScale,
               const uint8_t* BScale, const uint8_t* AFactor, const uint8_t* BFactor,
               void* out, int M, int N, int K, int scaleN_pad, hipStream_t stream)
{
    constexpr int WV = 4;
    const int tiles  = ((N + 16 * NS * WV - 1) / (16 * NS * WV)) * ((M + 16 * MS - 1) / (16 * MS));
    hipLaunchKernelGGL((gemm_a4w4_wg_kernel<MS, NS, WV>), dim3(std::min(tiles, 256)),
                       dim3(64 * WV), 0, stream, A, B, AScale, BScale, AFactor, BFactor,
                       reinterpret_cast<__hip_bfloat16*>(out), M, N, K, scaleN_pad);
}

struct WgTile
{
    int mt, nt;
    void (*launch)(const uint8_t*, const uint8_t*, const uint8_t*, const uint8_t*,
                   const uint8_t*, const uint8_t*, void*, int, int, int, int, hipStream_t);
};
// Workgroup tiles (MTILE x NTILE = 16*MSUB x 64*NSUB). Wider wave tiles (NSUB 6,
// 8) cut loads and LDS reads per MFMA and give the heuristic more ways to
// quantize N onto the CUs; MSUB*NSUB <= 64 keeps the accumulators in AGPRs.
constexpr WgTile kWgTiles[] = {
    {256, 256, launch_wg<16, 4>}, {192, 256, launch_wg<12, 4>}, {160, 256, launch_wg<10, 4>}, {128, 256, launch_wg<8, 4>},
    {96, 256, launch_wg<6, 4>},   {64, 256, launch_wg<4, 4>},
    {128, 384, launch_wg<8, 6>},  {96, 384, launch_wg<6, 6>},   {64, 384, launch_wg<4, 6>},
    {96, 512, launch_wg<6, 8>},   {64, 512, launch_wg<4, 8>},
};

bool gemm_a4w4_mxfp4_launch(const uint8_t* A,
                            const uint8_t* B,
                            const uint8_t* AScale,
                            const uint8_t* BScale,
                            const uint8_t* AFactor,
                            const uint8_t* BFactor,
                            void*          out,
                            int            M,
                            int            N,
                            int            K,
                            int            Mpad,
                            int            scaleN_pad,
                            void*          stream)
{
    // Grid: one wave (64 threads) per (16*MSUB) x (16*NSUB) output tile.
    auto launch = [&](auto msub, auto nsub) {
        constexpr int MS = decltype(msub)::value;
        constexpr int NS = decltype(nsub)::value;
        constexpr int MTILE = 16 * MS;
        constexpr int NTILE = 16 * NS;
        dim3 block(64);
        dim3 grid((N + NTILE - 1) / NTILE, (Mpad + MTILE - 1) / MTILE);
        hipLaunchKernelGGL((gemm_a4w4_mfma_kernel<MS, NS>), grid, block, 0,
                           reinterpret_cast<hipStream_t>(stream), A, B, AScale,
                           BScale, reinterpret_cast<__hip_bfloat16*>(out), M, N,
                           K, Mpad, scaleN_pad);
    };

    // Workgroup-cooperative path (LDS-shared A, 4 waves/WG, persistent).
    // Tile choice: minimize ceil(tiles / NCU) * (MTILE + 64) * NTILE -- rounds
    // of one tile per CU, each costing the tile area plus a fixed per-tile
    // overhead (worth ~64 rows: prologue/epilogue, and the lower MFMA:load
    // ratio of short tiles).
    if (K % 1024 == 0)
    {
        constexpr int NCU_WG = 256;
        int best = -1;
        long best_cost = 0, best_tiles = 0;
        for (int i = 0; i < (int)(sizeof(kWgTiles) / sizeof(kWgTiles[0])); ++i)
        {
            const int mt = kWgTiles[i].mt, nt = kWgTiles[i].nt;
            const long tiles = (long)((M + mt - 1) / mt) * ((N + nt - 1) / nt);
            const long cost  = (tiles + NCU_WG - 1) / NCU_WG * (long)(mt + 64) * nt;
            if (best < 0 || cost < best_cost)
            {
                best       = i;
                best_cost  = cost;
                best_tiles = tiles;
            }
        }
#ifdef MBS_FORCE_TILE_ENV
        // Debug builds: MBS_TILE=<index into kWgTiles> forces a workgroup tile.
        if (const char* e = getenv("MBS_TILE"))
        {
            kWgTiles[atoi(e)].launch(A, B, AScale, BScale, AFactor, BFactor, out, M, N, K,
                                     scaleN_pad, reinterpret_cast<hipStream_t>(stream));
            return true;
        }
#endif
        // Too few tiles to occupy a quarter of the GPU: the single-wave kernel's
        // small tiles expose more parallelism.
        if (best_tiles >= NCU_WG / 4)
        {
            kWgTiles[best].launch(A, B, AScale, BScale, AFactor, BFactor, out, M, N, K,
                                  scaleN_pad, reinterpret_cast<hipStream_t>(stream));
            return true;
        }
    }

    // The single-wave kernel does not apply the MBS factors yet.
    if (AFactor || BFactor)
        return false;

    // Tile-shape heuristic: prefer the largest tile (best reuse) whose workgroup
    // count still fills the GPU (~>= 2 workgroups per CU); otherwise shrink the
    // tile to expose more parallelism. Thresholds measured on gfx950 (256 CUs).
    constexpr int NCU        = 256;
    constexpr int FILL       = 2 * NCU;  // target minimum workgroup count
    auto grid_wgs = [&](int mt, int nt) {
        return (long)((Mpad + mt - 1) / mt) * ((N + nt - 1) / nt);
    };
    using m8 = std::integral_constant<int, 8>;
    using m4 = std::integral_constant<int, 4>;
    using m2 = std::integral_constant<int, 2>;
    using m1 = std::integral_constant<int, 1>;

    // Larger per-wave tiles issue more independent MFMAs per K-step, which is
    // what keeps the matrix pipe busy (the real limiter -- occupancy is not).
    // HUGE 128x128 = 64 MFMAs/chunk (256 AGPR accumulators). Step down when a
    // tile that big can't fill the GPU, trading ILP for workgroup parallelism.
    if (grid_wgs(128, 128) >= FILL)
        launch(m8{}, m8{});        // HUGE 128x128: deepest MFMA ILP
    else if (grid_wgs(64, 128) >= FILL)
        launch(m4{}, m8{});        // TALL 64x128: 32-MFMA ILP, fills small-M/
                                   // large-N where 128x128 can't (e.g. M=1024)
    else if (grid_wgs(64, 64) >= FILL)
        launch(m4{}, m4{});        // BIG  64x64
    else if (grid_wgs(32, 32) >= FILL)
        launch(m2{}, m2{});        // MID  32x32
    else
        launch(m1{}, m2{});        // SMALL 16x32: tiny M, maximize occupancy
    return true;
}
