#pragma once

#include "common.h"

namespace v0 {
// A8192 5120 *B 5120 8192=C 8192 8192
// 64*64*128*128=8192 8192
constexpr int BM = 128, BN = 128, BK = 128;
constexpr int ROWB = BK / 2; //128/2=64B


__global__ __launch_bounds__(256) void gemm_v0_naive(Problem p)
{
    // A,B 128*128，128*128
    __shared__ uint8_t sA[BM * ROWB]; // LDS 128*64=8192B=8KB
    __shared__ uint8_t sB[BN * ROWB];
    long long start = clock64();

    // 一个cu 四个wave 4*64=256 lane, 每个wave 64个lane，
    // lane本wave 0-63线程，wave本cu 0-3
    const int tid = threadIdx.x, lane = tid % 64, wave = tid / 64;
    const int wm = wave / 2, wn = wave % 2;// wm 0-1 wn 0-1
    const int r16 = lane % 16, g = lane / 16; // 16*4=64， r16第几行0-15，g第几列 0-3
    const int m0 = blockIdx.y * BM, n0 = blockIdx.x * BN; // 64*64 BM=BN=128
    const int KB = p.K / 2, KS = p.K / 32; // 8192/2=4096 B， 8192/32=256 B
    long long end = clock64();
    printf("load time: %llu\n", end - start);
    f32x4 acc[4][4] = {}; // 4*4个fp32*4, 4*4个16字节
    for (int k0 = 0; k0 < p.K; k0 += BK) //一次加载A 128*128
    {
        // i(2047)是第几个dword，tid(255)是谁在搬, 哪个线程
        // 这个搬运就直接直接搬行不连续的HB,数据到LDS中了
        for (int i = tid; i < BM * ROWB / 4; i += 256)  // BM=128*64B / 4B=512次load_dword(4字节每次，每lane)， 256个lane跳一次，一共要跳8次才能加载完
        {
            const int r = i / (ROWB / 4), c = i % (ROWB / 4);// 本次加载是128*128中的第几行第几列，这里的列是按照4字节一组的小块
            reinterpret_cast<uint32_t*>(sA)[i] =
                *reinterpret_cast<const uint32_t*>(p.A + (size_t)(m0 + r) * KB + k0 / 2 + c * 4);
            reinterpret_cast<uint32_t*>(sB)[i] =
                *reinterpret_cast<const uint32_t*>(p.B + (size_t)(n0 + r) * KB + k0 / 2 + c * 4);
        }
        __syncthreads();

#pragma unroll
        // 128*128，分给四个wave，那么wave0要0-63*0-63，wave1要0-63，64-127， 
        // mi*16是第几个16行，r16 是16行中的第几个行
        for (int mi = 0; mi < 4; ++mi) 
        {
            const int ra = wm * 64 + mi * 16 + r16; // 128行里面的第几行
            const i32x4 a = *reinterpret_cast<const i32x4*>(sA + ra * ROWB + g * 16); // 128行中第几行*每行64B=本行起始地址，g*16
            const int sa = p.As[(size_t)(m0 + ra) * KS + k0 / 32 + g];
#pragma unroll
            for (int ni = 0; ni < 4; ++ni) {
                const int rb = wn * 64 + ni * 16 + r16;
                const i32x4 b = *reinterpret_cast<const i32x4*>(sB + rb * ROWB + g * 16);
                const int sb = p.Bs[(size_t)(n0 + rb) * KS + k0 / 32 + g];
                acc[mi][ni] = mfma_fp4(a, b, acc[mi][ni], sa, sb);
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int mi = 0; mi < 4; ++mi)
#pragma unroll
        for (int ni = 0; ni < 4; ++ni)
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const int m = m0 + wm * 64 + mi * 16 + 4 * g + i;
                const int n = n0 + wn * 64 + ni * 16 + r16;
                p.C[(size_t)m * p.N + n] = __float2bfloat16(acc[mi][ni][i]);
            }
}

inline void launch(const Problem& p, hipStream_t s)
{
    gemm_v0_naive<<<dim3(p.N / BN, p.M / BM), 256, 0, s>>>(p);
    // gemm_v0_naive<<<dim3(p.N / BN, p.M / BM), 256, 0, s>>>(p);
    // gemm_v0_naive<<<dim3(1, 1), 64, 0, s>>>(p);

}

}  // namespace v0
