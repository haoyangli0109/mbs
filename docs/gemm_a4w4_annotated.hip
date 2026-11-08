// =============================================================================
// gemm_a4w4.hip 中文逐行注释版（学习用）
//
// 本文件是 kernel/gemm_a4w4.hip（提交 80614af）的注释版：代码与原文件逐字相同，
// 只是把英文注释改写成了更详细的中文注释。原版英文注释以 kernel/gemm_a4w4.hip 为准。
// 它不参与构建（Makefile 只编译 kernel/gemm_a4w4.hip），但可以用同样的选项单独编译，
// 生成的指令和原文件完全一样（在 mbs 目录下执行）：
//   hipcc -O3 -std=c++20 --offload-arch=gfx950 -Ikernel --cuda-device-only -S
//         docs/gemm_a4w4_annotated.hip -o /tmp/annotated.s
//
// -----------------------------------------------------------------------------
// 建议的阅读顺序
// -----------------------------------------------------------------------------
//   1. 先读下面的“背景知识”和 docs/kernel_intro.md，弄清 MXFP4 / E8M0 和三种数据布局。
//   2. 读 flatB、flatS、mfma_fp4_acc，配合 kernel/mfma_probe.hip（make probe 之后运行
//      ./build/mfma_probe）理解一条 MFMA 里 lane 和数据的对应关系。
//   3. 读单 wave kernel gemm_a4w4_mfma_kernel（约 100 行）。它是“正确但朴素”的参考
//      实现，片段、子块、累加器、乒乓缓冲、输出布局这些概念都在这里。
//   4. 再读 wg kernel gemm_a4w4_wg_kernel，按这个顺序：常量 → tile 调度 tile_off →
//      描述符和寻址 rsrc_of / rsrc_at / issue_one → LDS 读取 frag_at / scale_at →
//      section / step（指令编排和等待）→ epilogue → 最后的主循环。
//      读的时候对照 gemm_tiling.png、gemm_dataflow.png 和 mbs_note.md。
//   5. 最后读 launcher（gemm_a4w4_mxfp4_launch）里的 tile 选择。
//   6. 动手：把 kernel 编译成汇编看 K 循环；用 bench_one.py 测时间；改 LOAD_WIN、PF_B
//      之类的参数再测；用 -DMBS_TIMING=1 看各阶段的 cycle；和 kernel/gemm_a4w4_tmp.hip
//      （上一版）做 diff，看优化是怎么一步步来的。
//
// 文中的“【例】”都以主力 tile 256×256（gemm_a4w4_wg_kernel<16, 4, 4>）为例；
// 问题规模默认取 A = [8192, 5120]、B = [8192, 5120]（即 out = A @ B^T → [8192, 8192]，
// 也就是常说的 8192×5120 × 5120×8192）。由此推出来的常量：
//   Kp = 2560，KT = 80，nstages = 20，scaleN_pad = 160，
//   num_m = num_n = 32，total = 1024 个 tile，每组 16 行 B 占 KT*512 = 8·K = 40960 B。
//
// -----------------------------------------------------------------------------
// 背景知识
// -----------------------------------------------------------------------------
// 计算：out[M, N] = dequant(A) @ dequant(B)^T，输出 bf16。
//   * A 是激活 [M, K]，B 是权重 [N, K]，都是 MXFP4。
//   * MXFP4 的每个元素是 4 bit 的 e2m1 浮点（1 位符号、2 位指数、1 位尾数，能表示
//     0、±0.5、±1、±1.5、±2、±3、±4、±6）。两个元素打包在一个字节里，所以一行 K 个
//     元素占 K/2 字节。
//   * 沿 K 每 32 个元素共享一个 E8M0 scale：8 bit 纯指数，值为 2^(e-127)，127 表示 1.0。
//     反量化 = fp4 值 × 2^(e-127)。
//
// 三种输入布局（和 aiter 的 gemm_a4w4_asm 一致）：
//   * A：普通行主序，[M, K/2] 字节。
//   * B：aiter 的 shuffle_weight(layout=(16, 16)) 预重排，见 flatB。
//   * scale：aiter 量化时 shuffle=True 的布局，见 flatS。
//
// 硬件（gfx950 / MI355X）：
//   * 一个 wave = 64 个 lane（线程），共用一条指令流，按顺序发射（in-order issue）。
//   * 一个 CU 有 4 个 SIMD 和 160 KB LDS（工作组内共享的片上内存）。整卡 256 个 CU，
//     分布在 8 个 XCD 上，每个 XCD 有自己的 L2。
//   * 寄存器：VGPR 是每个 lane 各一份的向量寄存器；SGPR 是整个 wave 共用的标量寄存器；
//     AGPR 是 MFMA 用的累加寄存器。一个 SIMD 只跑 1 个 wave 时，每个 lane 最多有
//     256 个 VGPR + 256 个 AGPR。
//   * MFMA：矩阵核心指令。本文件只用 v_mfma_scale_f32_16x16x128_f8f6f4：一条算 16×16
//     的输出、K = 128，自带 E8M0 scale，fp32 累加。FP4 时一条大约占用矩阵核心 16 个
//     cycle，这段时间里同一个 wave 还可以发射别的指令。
//   * buffer load / store：用一个 128 bit 的“描述符”（基址 + 字节范围）寻址的访存指令。
//     地址 = 基址 + VGPR 偏移（voffset）+ SGPR 偏移（soffset）+ 立即数。超出范围的读
//     返回 0，写被丢弃——本文件就是这样处理 M、N 边界的。
//   * 等待计数：vmcnt 是还没完成的全局访存条数（load、LDS DMA、store 都算，按发射顺序
//     完成）；lgkmcnt 是还没完成的 LDS / 标量访存条数。s_waitcnt vmcnt(n) 的意思是
//     “等到最多只剩最新发出的 n 条没完成”。
//
// 本文件里的两个 kernel：
//   * gemm_a4w4_mfma_kernel：一个工作组只有 1 个 wave，A、B 都直接从全局读进寄存器。
//     作为回退路径（K 不是 1024 的倍数，或 tile 太少时）。
//   * gemm_a4w4_wg_kernel：一个工作组 4 个 wave，A 经 LDS 共享，指令顺序在编译期排定，
//     persistent。这是主力路径。
// =============================================================================

#include "gemm_a4w4.h"         // 对外的 launcher 声明（不依赖 torch）

#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>      // __hip_bfloat16 和 float -> bf16 的转换

#include <type_traits>         // std::integral_constant（编译期整数）
#include <utility>             // std::integer_sequence（用来实现 static_for）
#include <algorithm>           // std::min
#include <cstdlib>             // getenv / atoi：只在 MBS_FORCE_TILE_ENV 调试构建里用

// MBS_TIMING = 1 时打开分阶段计时（调试用），默认关闭。
#ifndef MBS_TIMING
#define MBS_TIMING 0
#endif

#if MBS_TIMING
// 计时调试：wg kernel 的每个 wave 用 s_memtime 统计各阶段花的 cycle，最后累加到这个
// 全局数组里。下标含义：[0] 段 0 的等待，[1] 段 0 的计算，[2] 段 1 的等待 + barrier，
// [3] 段 1 的计算，[4] epilogue，[5]/[6] 每个 tile 第 1/2 个 step 的段 0 等待，
// [7] 这两个 step 的段 1 等待，[8] wave 总数。
__device__ unsigned long long mbs_timing[16];
// 主机端读取计时结果（reset 非 0 时顺便清零）。可以用 ctypes 从 libmbs_gemm_kernel.so 调用。
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

// 向量类型。
//   i32x4：4 个 int = 128 bit，正好是一个 lane 的一份 fp4 操作数（32 个 fp4 = 16 字节）。
//   i32x8：本文件里没有用到。
//   f32x4：4 个 float，是一个 lane 上一个 16×16 子块的累加结果。
typedef int   i32x4 __attribute__((ext_vector_type(4)));
typedef int   i32x8 __attribute__((ext_vector_type(8)));
typedef float f32x4 __attribute__((ext_vector_type(4)));

// -----------------------------------------------------------------------------
// flatB：B 经 16×16 预重排后，权重字节 (n, kp) 在 B 缓冲区里的偏移。
//   n  是 B 的行（也就是输出的列，0..N-1）；kp 是行内第几个字节（0..K/2-1）。
//   KT = K/64，是每行有多少个 32 字节（64 个 fp4）的 K tile。【例】K = 5120 → KT = 80。
// 排布：B 先按 16 行分组，每组的数据连续存放（每组 KT*512 = 8·K 字节；【例】40960 B）；
// 组内再按 K tile 排，每个 K tile 512 字节 = 16 行的前 16 字节（256 B）+ 16 行的后 16 字节
// （256 B）。【例】一行组沿 K 共 80 个 K tile = 40 KB。
// 效果：一条 MFMA 要的 B 操作数（16 行 × 128 个 K = 1 KB）在内存里是连续的，64 个 lane
// 各读 16 字节，就是一次完全合并的读。
// （wg kernel 不调用这个函数，而是直接用由它推出来的 b_lane 公式。）
// -----------------------------------------------------------------------------
__device__ __forceinline__ int64_t flatB(int n, int kp, int KT)
{
    int n_tile = n >> 4;     // 第几个 16 行组
    int n_in   = n & 15;     // 组内第几行
    int k_tile = kp >> 5;    // 第几个 32 字节的 K tile
    int rem    = kp & 31;    // K tile 内的字节号 0..31
    int kpack  = rem >> 4;   // 在 K tile 的前 16 字节（0）还是后 16 字节（1）
    int k_in   = kp & 15;    // 16 字节内的字节号
    // 组偏移 + K tile 偏移 + 半区偏移 + 行偏移（每行 16 字节）+ 字节号
    return (int64_t)n_tile * ((int64_t)KT * 512) + (int64_t)k_tile * 512 +
           (int64_t)kpack * 256 + (int64_t)n_in * 16 + k_in;
}

// -----------------------------------------------------------------------------
// flatS：shuffle 过的 E8M0 scale 缓冲区里，第 r 行、第 b 个 K 块（每块 32 个 K）的 scale
// 字节的偏移。scaleN_pad 是每行的 K 块数向上取整到 8 的倍数，也就是每行 scale 的字节数。
// 【例】K = 5120 → K/32 = 160，已是 8 的倍数，scaleN_pad = 160。
// 排布：
//   * 行按 32 行分组，每组占 32*scaleN_pad 字节；【例】5120 B
//   * 组内 K 块按 8 块一段，每段 256 字节（64 个 dword）；【例】160/8 = 20 段
//   * 段内的 dword 号 = (b%4)*16 + (r%16)，dword 里的字节号 = ((b%8)/4)*2 + (r%32)/16。
// 关键性质：段内第 L 个 dword 正好是“lane L”要用的 scale——它的行 r%16 = L%16、K 块
// b%4 = L/16，和 MFMA 里 lane L 负责的 (行, K 块) 一致；而这个 dword 的 4 个字节分别对应
// {相邻两个 16 行子块} × {K 的前后两个 128}。所以 wg kernel 可以每个 lane 读一个 dword，
// 再让 MFMA 用 op_sel 挑字节，完全不需要解包。
// -----------------------------------------------------------------------------
__device__ __forceinline__ int64_t flatS(int r, int b, int scaleN_pad)
{
    int r32 = r & 31;   // 32 行组内的行号
    int b8  = b & 7;    // 8 块一段内的块号
    // 字节号（r32>>4、b8>>2）+ dword 号 ×4（r32&15、b8&3）+ 段偏移（b>>3）+ 行组偏移（r>>5）
    return (int64_t)(r32 >> 4) + (int64_t)(b8 >> 2) * 2 + (int64_t)(r32 & 15) * 4 +
           (int64_t)(b8 & 3) * 64 + (int64_t)(b >> 3) * 256 +
           (int64_t)(r >> 5) * 32 * (int64_t)scaleN_pad;
}

// 读一个 lane 的 fp4 片段：32 个 fp4 = 16 字节（地址 16 字节对齐）。FP4 的 MFMA 操作数
// 只用 128 bit（指令名义上是 256 bit 的 f8f6f4 槽位，fp4 时高 128 bit 被忽略），所以
// i32x4 就够了。valid 为假（越界的行）时返回 0。
// 注意：这种逐 lane 的“valid ? 读 : 0”会被编译成带 exec 掩码的分支，导致单 wave kernel
// 的预取后面被插入 vmcnt(0)，预取基本失效（见 summary.md）。
__device__ __forceinline__ i32x4 load_frag(const uint8_t* p, bool valid)
{
    return valid ? *reinterpret_cast<const i32x4*>(p) : i32x4{0, 0, 0, 0};
}

// -----------------------------------------------------------------------------
// 带 scale 的 FP4 MFMA（单 wave kernel 用）：acc += A 片段 × B 片段（各自乘上 scale）。
// 指令格式：v_mfma_scale_f32_16x16x128_f8f6f4 D, A, B, C, scaleA, scaleB
//   * D 和 C 都是 %0（acc），即原地累加；
//   * cbsz:4 / blgp:4：A、B 两个操作数的格式都是 4，即 FP4（e2m1）；
//   * op_sel_hi:[0,0,0]（op_sel 省略即为 0）：scale 取寄存器里的第 0 个字节。
// mfma_probe 实测得到的 lane 对应关系：
//   * 操作数：lane L 提供第 L%16 行、第 L/16 个 32-K 块的 32 个 fp4；
//   * scale：每个 lane 提供自己那个 (行, K 块) 的 scale 字节；
//   * 输出：acc[i] 在 lane L 上是 C[(L/16)*4 + i][L%16]。
// 为什么用 inline asm 而不是 builtin：约束 "+a" 把累加器钉在 AGPR 里，VGPR 就能全部
// 留给流水线的缓冲；用 builtin 时累加器可能被放进 VGPR（上限 256 个），大 tile 放不下。
// asm volatile 保证 MFMA 之间的顺序不被打乱（编译器仍然可以把 load 插到它们中间）。
// -----------------------------------------------------------------------------
__device__ __forceinline__ void mfma_fp4_acc(f32x4& acc, i32x4 a, i32x4 b,
                                             int sa, int sb)
{
    asm volatile(
        "v_mfma_scale_f32_16x16x128_f8f6f4 %0, %1, %2, %0, %3, %4 "
        "op_sel_hi:[0,0,0] cbsz:4 blgp:4"
        : "+a"(acc)                               // %0：累加器，读写，放在 AGPR
        : "v"(a), "v"(b), "v"(sa), "v"(sb));      // %1 A 片段，%2 B 片段，%3 / %4 两个 scale
}

// =============================================================================
// 第一部分：单 wave kernel（回退路径，也是最好的入门读物）
// =============================================================================
// 一个工作组只有 1 个 wave（64 个 lane），计算一块 (16*MSUB) × (16*NSUB) 的输出。
// 二维寄存器分块：每个 K 步（128 个 K）读 MSUB 个 A 片段和 NSUB 个 B 片段，发出
// MSUB*NSUB 条 MFMA。每个 A 片段被 NSUB 条 MFMA 复用，每个 B 片段被 MSUB 条复用，
// 所以 A 的读取量降为 1/NSUB，B 的降为 1/MSUB。
// MSUB / NSUB 是编译期常量：tile 大，复用多、峰值高，但寄存器多、工作组少；tile 小，
// 工作组多，适合小 shape。launcher 实例化了几种，按问题挑一种。
//
// __launch_bounds__ 的第二个参数是“每个 SIMD 至少要能同时驻留几个 wave”，它限制每个
// wave 能用的寄存器数：1 表示不限制（最多 512 个），2 表示每个 wave 不超过 256 个。
//   * MSUB*NSUB ≥ 32（128×128、64×128）：最多 256 个 AGPR 累加器，再加两套片段缓冲，
//     不加限制；
//   * MSUB*NSUB = 16（64×64）：要求每个 SIMD 2 个 wave，256 个寄存器够用、不会 spill；
//   * 更小的 tile 不加限制也能跑到每个 SIMD 5~7 个 wave。
//
// 参数：
//   A       [M, K/2] fp4x2，行主序          B      [N, K/2]，16×16 预重排
//   AScale  A 的 E8M0 scale（shuffle 布局）  BScale B 的 E8M0 scale（shuffle 布局）
//   out     [Mpad, N] bf16 输出（只写前 M 行）
//   Mpad    M 向上取整到 32；kernel 里没有用到，只在 launcher 里用来算 grid
//   scaleN_pad  每行 scale 的字节数（K/32 向上取整到 8）
// =============================================================================
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
    constexpr int MTILE = 16 * MSUB;        // 每个 wave 负责的输出行数 16*16=256
    constexpr int NTILE = 16 * NSUB;        // 每个 wave 负责的输出列数 16*4=64
    const int lane  = threadIdx.x;          // lane 号 0..63
    const int row16 = lane & 15;            // 片段里的行号0-15（A 片段是 M 方向，B 片段是 N 方向）
    const int g     = lane >> 4;            // 第几个 32-K 块（0..3）；一条 MFMA 的 K=128 正好 4 块
    const int m0    = blockIdx.y * MTILE;   // 本 tile 的起始行（grid.y 沿 M）
    const int n0    = blockIdx.x * NTILE;   // 本 tile 的起始列（grid.x 沿 N）

    const int Kp      = K >> 1;     // 每行的字节数（一个字节放 2 个 fp4）2560
    // 写成 K / 64 只是沿用了 aiter shuffle_weight 里 BK = 32 字节（等于 64 个 fp4）的叫法
    const int KT      = K / 64;     // B 预重排里每行的Ktile数(K tile=64个k，不同于最小块32个k)【例】80
    // K 步数，每步 128 个 K，正好一条 MFMA 的 K。【例】K = 5120 → 40 步
    // 注意：K 不是 128 的倍数时，余下的 K 会被直接丢掉（已知缺陷，见 summary.md）。
    const int Kchunks = K / 128;

    // 累加器：MSUB×NSUB 个 16×16 子块，每个子块在每个 lane 上是256/64 = 4 个 float。
    // 【例】8×8 时一共 64 个 f32x4 = 256 个寄存器，只能放在 AGPR（靠 mfma_fp4_acc 的 "+a"）。
    f32x4 acc[MSUB][NSUB]; // 16， 4
#pragma unroll
    for (int mi = 0; mi < MSUB; ++mi)
#pragma unroll
        for (int ni = 0; ni < NSUB; ++ni)
            acc[mi][ni] = f32x4{0.f, 0.f, 0.f, 0.f};   // 清零

    // load：把第 kc 个 K 步的 A / B 片段和 E8M0 scale 读进给定的缓冲。
    // 每个 lane 只负责“自己那一行 × 自己那个 32-K 块”：A 片段 MSUB 个，B 片段 NSUB 个。
    // af MSUB 个 i32x4,第 mi 个 16 行子块的 A 片段,数组内的相邻的32个k在地址上隔了16行，mi=0-15
    // as MSUB 个 int, af[mi] 对应的 E8M0 scale
    // bf NSUB 个 i32x4,第 ni 个 16 列子块的 B 片段, 16字节
    // bs NSUB 个 int bf[ni] 对应的 E8M0 scale
    auto load = [&](int kc, i32x4 (&af)[MSUB], int (&as)[MSUB],
                    i32x4 (&bf)[NSUB], int (&bs)[NSUB]) {
        // kc 是 K 步数，g 是 32k 块号，128个k为一步，4个32k块为一步，
        const int kblk     = 4 * kc + g;                // 整行里第几个 32-K 块：一步 4 块，本 lane 是这一步的第 g 块（查 scale 用）
        const int ktile    = 2 * kc + (g >> 1);         // 整行里第几个Ktile（64个K=32字节）：一步 2 个，本 lane 在这一步的第 g/2 个
        const int kpack    = g & 1;                     // K tile 里的前 16 字节（0）还是后 16 字节（1）
        const int kpstart  = ktile * 32 + kpack * 16;   // 行内起始字节，等于 kc*64 + g*16【例】kc = 3、g = 2 → ktile 7、kpack 0、kpstart 224
        const int64_t aoff = (int64_t)(kc * 64 + g * 16);   // A 行内的字节偏移（同一个值）
#pragma unroll
        for (int mi = 0; mi < MSUB; ++mi)
        {
            int am = m0 + mi * 16 + row16;   // 本 lane 要读的 A 行
            // 读 16 字节的片段；越界的行读成 0
            af[mi] = load_frag(A + (int64_t)am * Kp + aoff, am < M);
            // scale 逐字节读；越界时给 127（= 1.0，反正数据是 0）
            as[mi] = (am < M) ? AScale[flatS(am, kblk, scaleN_pad)] : 127;
        }
#pragma unroll
        for (int ni = 0; ni < NSUB; ++ni)
        {
            int  bn = n0 + ni * 16 + row16;   // 本 lane 要读的 B 行（即输出列）
            bool vb = bn < N;
            bf[ni] = load_frag(B + flatB(bn, kpstart, KT), vb);
            bs[ni] = vb ? BScale[flatS(bn, kblk, scaleN_pad)] : 127;
        }
    };

    // mma：用一组缓冲发出 MSUB×NSUB 条 MFMA：acc[mi][ni] += A 片段[mi] × B 片段[ni]。
    // 相邻的 MFMA 写不同的累加器，互不依赖，可以背靠背发射（这就是指令级并行 ILP）。
    auto mma = [&](i32x4 (&af)[MSUB], int (&as)[MSUB], i32x4 (&bf)[NSUB],
                   int (&bs)[NSUB]) {
#pragma unroll
        for (int mi = 0; mi < MSUB; ++mi)
#pragma unroll
            for (int ni = 0; ni < NSUB; ++ni)
                mfma_fp4_acc(acc[mi][ni], af[mi], bf[ni], as[mi], bs[ni]);
    };

    // 软件流水的 K 循环：两套名字固定的缓冲（A 组、B 组）轮流使用（乒乓）。
    // 双缓冲有两个作用：
    //   1. 让下一步的 load 和这一步的 MFMA 重叠；
    //   2. 正确性也需要它：MFMA 要花很多 cycle 读源寄存器，而 inline asm 对编译器的 hazard
    //      检查是不可见的。只有一套缓冲的话，下一步的 load 可能覆盖还在被 MFMA 读的寄存器；
    //      两套缓冲保证每一步的操作数一直活到它的 MFMA 结束。
    // 实测编译结果：因为 load_frag 里的分支，编译器在每批预取之后插了 vmcnt(0)，重叠基本
    // 没有发生。这是回退路径只有 asm 一半速度的主要原因。
    i32x4 afA[MSUB], bfA[NSUB], afB[MSUB], bfB[NSUB];   // 片段缓冲：A 组 / B 组
    int   asA[MSUB], bsA[NSUB], asB[MSUB], bsB[NSUB];   // scale 缓冲：A 组 / B 组
    load(0, afA, asA, bfA, bsA);   // 序幕：先读第 0 步

    int kc = 0;
    for (; kc + 1 < Kchunks; kc += 2)   // 每次迭代处理两步：偶数步 + 奇数步
    {
        load(kc + 1, afB, asB, bfB, bsB);       // 预取奇数步到 B 组
        mma(afA, asA, bfA, bsA);                // 计算偶数步（A 组）
        if (kc + 2 < Kchunks)
            load(kc + 2, afA, asA, bfA, bsA);   // 预取下一个偶数步到 A 组
        mma(afB, asB, bfB, bsB);                // 计算奇数步（B 组）
    }
    if (kc < Kchunks)   // Kchunks 是奇数时，最后一步还留在 A 组里
        mma(afA, asA, bfA, bsA);

    // 写回：子块 (mi, ni) 的 acc[i] 在 lane L 上是 C[(L/16)*4 + i][L%16]，
    // 即本 lane 负责第 row16 列、第 g*4 .. g*4+3 行。逐个元素转成 bf16，每次写 2 字节
    // （没有向量化，这也是这个 kernel 慢的原因之一）。
#pragma unroll
    for (int ni = 0; ni < NSUB; ++ni)
    {
        const int on = n0 + ni * 16 + row16;   // 输出列
        if (on >= N)
            continue;                          // 越界的列不写
#pragma unroll
        for (int mi = 0; mi < MSUB; ++mi)
#pragma unroll
            for (int i = 0; i < 4; ++i)
            {
                int om = m0 + mi * 16 + g * 4 + i;   // 输出行
                if (om < M)                          // 越界的行不写（out 有 Mpad 行，只写前 M 行）
                    out[(int64_t)om * N + on] =
                        __float2bfloat16(acc[mi][ni][i]);
            }
    }
}

// =============================================================================
// 第二部分：工作组协作 kernel（主力路径，参照生产 asm 的 f4gemm 写成）
// =============================================================================
// 一个工作组 = WAVES（=4）个 wave，沿 N 并排：每个 wave 算 (16*MSUB) × (16*NSUB)，整个
// 工作组算 (16*MSUB) × (16*NSUB*WAVES)。【例】256×256 时每个 wave 算 256×64。
// 和单 wave kernel 相比：
//   * A 由 4 个 wave 共享：用“直接写进 LDS 的 buffer load”（不经过 VGPR）流进一个 4 槽
//     的环形缓冲，每槽放一个 stage（K = 256，即 MTILE 行 × 128 字节），提前 2 个 stage
//     发出。每行的 8 个 16 字节块按 (row & 7) 做 XOR swizzle，让读片段的 ds_read_b128
//     （16 个 lane 读 16 行的同一个 K 块）不发生 bank conflict。
//   * B（16×16 预重排）仍然从全局直接读进 VGPR：每个 wave 的 N 列互不相同，没有可共享的。
//     提前 2 个 stage（放不下第三套 B 寄存器的 tile 只提前 1 个）。
//   * E8M0 scale：每个 lane 每个 stage 每 32 行读一个 dword。shuffle 布局把
//     [子块奇偶][K 的前后 128] 打包在这 4 个字节里，MFMA 用 op_sel 直接挑字节
//     （字节号 = 2*K 半段 + 子块奇偶），零开销。B 的 scale 进 VGPR；A 的 scale 被所有
//     wave 共用，每个工作组只 DMA 一次，放在同一个 LDS 槽里 A 数据的后面。
//   * 每个 stage 像 asm kernel 一样分成两段（M 的前后两半）：A 片段提前一整段读好，全局
//     load 一条一条穿插在 MFMA 之间（见 step）。所有全局 load 和 MFMA 都是 inline asm，
//     寻址方式也照搬 asm（每条 load 有固定的 VGPR 偏移，tile / stage 的偏移加到描述符
//     基址上，见 issue_one）；循环里的 vmcnt 等待全部手写；累加器钉在 AGPR。
//   * persistent：每个 CU 常驻一个工作组（每个 SIMD 1 个 wave），按 XCD 感知的顺序领
//     tile；软件流水跨 tile 不中断。
//   * MFMA 交换两个操作数，算的是 C^T，于是每个 lane 拿到同一行的 4 个连续输出；
//     epilogue 用 permlane16_swap + DPP 在 lane 之间重组，用 16 字节的 store 写出完整
//     的 128 字节行段。
// 使用条件（由 launcher 检查）：K % 1024 == 0（stage 数是 4 槽环的整数倍）、N % 16 == 0、
// MSUB / NSUB 为偶数、NSUB ∈ {4, 6, 8}。
// =============================================================================

// -----------------------------------------------------------------------------
// mfma_fp4_sel<OA, OB>：带编译期 scale 字节选择的 FP4 MFMA。
//   OA / OB：操作数 A / B 的 scale 取寄存器里的第几个字节（0..3）。
//   字节号编码在指令里：字节号 = op_sel_hi*2 + op_sel。
// 宏 MBS_MFMA_FP4 用来拼出 16 种组合的 asm 文本（字节号必须是指令里的立即数）。
// 同样用 inline asm + "+a" 把累加器钉在 AGPR：用 builtin 的话，寄存器分配器可以把累加器
// 在 AGPR 和 VGPR 之间搬来搬去，大 tile（最多 256 个累加寄存器）在 K 循环里每个 stage
// 会多出几百条 accvgpr 拷贝。
// hazard 说明：inline asm 对编译器的 hazard 检查不可见；但每个累加器要隔很多条 MFMA 才会
// 被再次写，片段 / scale 寄存器也要在最后一次被 MFMA 读之后很久才会被新的 load 覆盖，
// 所以不需要额外的 s_nop。
// 注意：宏定义的续行（行尾的反斜杠）里不能放注释，所以说明都写在这里。
// -----------------------------------------------------------------------------
#define MBS_MFMA_FP4(OS0, OS1, OH0, OH1)                                                   \
    asm volatile("v_mfma_scale_f32_16x16x128_f8f6f4 %0, %1, %2, %0, %3, %4 op_sel:[" #OS0 \
                 "," #OS1 ",0] op_sel_hi:[" #OH0 "," #OH1 ",0] cbsz:4 blgp:4"              \
                 : "+a"(acc)                                                              \
                 : "v"(a), "v"(b), "v"(sa), "v"(sb))
template <int OA, int OB>
__device__ __forceinline__ void mfma_fp4_sel(f32x4& acc, i32x4 a, i32x4 b, int sa, int sb)
{
    static_assert(OA >= 0 && OA < 4 && OB >= 0 && OB < 4, "scale byte select");
    // 下面 16 行按 (OA, OB) 展开。宏参数依次是 A 的 op_sel、B 的 op_sel、A 的 op_sel_hi、
    // B 的 op_sel_hi，即字节号的低位和高位。例如 OA = 3、OB = 0 对应 MBS_MFMA_FP4(1, 0, 1, 0)。
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

// 两个 float 打包成两个 bf16（就近舍入到偶数），一条 v_cvt_pk_bf16_f32 完成（通用的转换
// 写法会变成每对一条 cvt 加移位和或运算）。lo 放在低 16 位，hi 放在高 16 位。
__device__ __forceinline__ uint32_t pack_bf16x2(float lo, float hi)
{
    uint32_t r;
    asm("v_cvt_pk_bf16_f32 %0, %1, %2" : "=v"(r) : "v"(lo), "v"(hi));
    return r;
}

// static_for<N>(f)：编译期循环，依次调用 f(integral_constant<int, 0>) …
// f(integral_constant<int, N-1>)。和 #pragma unroll 不同，循环变量在 f 里是编译期常量，
// 可以用在模板参数、if constexpr 和数组下标里。wg kernel 靠它让每条指令的位置、每个缓冲的
// 名字都在编译期确定。
template <int... Is, typename F>
__device__ __forceinline__ void static_for_impl(std::integer_sequence<int, Is...>, F&& f)
{
    (f(std::integral_constant<int, Is>{}), ...);   // 折叠表达式：按 0..N-1 的顺序展开
}
template <int N, typename F>
__device__ __forceinline__ void static_for(F&& f)
{
    static_for_impl(std::make_integer_sequence<int, N>{}, f);
}

// -----------------------------------------------------------------------------
// gemm_a4w4_wg_kernel<MSUB, NSUB, WAVES>
//   __launch_bounds__(256, 1)：每个工作组 256 个线程（4 个 wave）；每个 SIMD 只要求驻留
//   1 个 wave，所以每个 wave 可以用满 512 个寄存器（256 VGPR + 256 AGPR）。一个 CU 有
//   4 个 SIMD，正好放下一个工作组。
// 参数和单 wave kernel 一样（没有 Mpad）。out 的范围按 M 行算，第 M 行以后不会被写。
// 【例】A/B = [8192, 5120]，scaleN_pad = 160；MSUB=16、NSUB=4、WAVES=4 → 256×256 tile。
// -----------------------------------------------------------------------------
template <int MSUB, int NSUB, int WAVES>
__global__ void __launch_bounds__(64 * WAVES, 1)
gemm_a4w4_wg_kernel(const uint8_t* __restrict__ A,
                    const uint8_t* __restrict__ B,
                    const uint8_t* __restrict__ AScale,
                    const uint8_t* __restrict__ BScale,
                    __hip_bfloat16* __restrict__ out,
                    int M, int N, int K, int scaleN_pad)
{
    // 一个 scale dword 覆盖相邻两个 16 行子块，所以 M、N 方向的子块数都必须是偶数。
    static_assert(MSUB % 2 == 0 && NSUB % 2 == 0, "scale dwords cover tile pairs");
    constexpr int MTILE   = 16 * MSUB;          // 工作组 tile 的行数【例】256
    constexpr int WNTILE  = 16 * NSUB;          // 每个 wave 的列数【例】64
    constexpr int NTILE   = WNTILE * WAVES;     // 工作组 tile 的列数【例】256
    // 每个 stage 的 A 在 LDS 里占的字节数：K = 256 个 fp4 = 每行 128 字节【例】32 KB
    constexpr int STAGE_A = MTILE * 128;
    // A 的 DMA：每条指令搬 1 KB = 8 行 × 128 字节；一个 stage 共 MTILE/8 条，平分给 4 个
    // wave。【例】每个 wave 8 条
    constexpr int A_LDS_PER_WAVE = MTILE / 8 / WAVES;
    // A 的 scale 跟在同一个 LDS 槽里 A 数据的后面：
    // scale 256*8, 如果32行作为一个整体，32*8=256字节，这样64个线程一人加载4字节就可以了，
    // 每个工作组只 DMA 一次，分给各个 wave,每个wave两次。块数不能被 wave 数整除时，
    // 多出来的指令重复搬最后一块，让每个 wave 的指令条数相同。
    // NSB就是以256*8 以32*8拆分，有多少scale块，就是8块
    constexpr int NSB     = MSUB / 2;                    // 每个 stage 的 scale 块数【例】8
    constexpr int S_LDS_PER_WAVE = (NSB + WAVES - 1) / WAVES;   // 每个 wave 搬几块【例】2
    constexpr int STAGE_B = STAGE_A + NSB * 256;         // 一个 LDS 槽的总字节数【例】34 KB
    static_assert((MTILE / 8) % WAVES == 0, "A stage must split evenly across waves");

    // LDS 环形缓冲：RING 个槽，每槽放一个 stage。“槽”就是 LDS 里一块固定大小的缓冲区。
    //   PF   = A（经 LDS）提前几个 stage 发出；
    //   PF_B = B 和 B scale（直接进寄存器）提前几个 stage 发出。
    // B 每多提前一个 stage，就多占一套寄存器（2*NSUB*4 个 VGPR）。256 行的 tile 和宽 N 的
    // tile（每个 stage 2*MSUB*NSUB ≥ 128 条 MFMA）只提前 1 个 stage，其余的实测提前 2 个
    // 更好。【例】PF_B = 1
    constexpr int RING = 4;
    constexpr int PF   = 2;
    constexpr int PF_B = (NSUB <= 4 && MSUB < 16) ? 2 : 1; // 1
    // 为什么不会覆盖还在读的槽：step s 在它的段 1 里重填 (s+PF) % RING 号槽。即使 RING 只有
    // PF+1，被填的也是 stage s-1 的槽：它的最后一批片段读取在 step s-1 的段 0 发出，并在
    // step s-1 段 1 开头的 lgkmcnt(0) + barrier 处全部完成。RING = 4 再多留一个 stage 的
    // 余量，这就是段 0 不需要 barrier 的原因。另外 K % 1024 == 0 时 stage 数是 4 的倍数。
    static_assert(RING >= PF + 1, "refill target must be clear of readers");
    static_assert(PF_B <= PF, "B is never further ahead than A");
    // 4 个独立的 LDS 数组（而不是一个二维数组），展开之后每个槽的地址都是编译期常量。
    // 【例】4 × 34 KB = 136 KB（gfx950 每个 CU 有 160 KB）
    // 这一行是在 LDS（每个 CU 上的片上共享内存）里开一块 STAGE_B 字节的数组，起始地址按 16 字节对齐。
    // 它是 A 的 LDS 环形缓冲的第 0 个槽，放一个 stage 的 A 数据和 A scale。256×256 tile 时一个槽是 34 KB。
    // __attribute__((aligned(16)))：保证数组的起始地址是 16 的倍数。uint8_t 数组默认只要求 1 字节对齐。
    // 但 kernel 从这里读 A，用的是一次读 16 字节的 ds_read_b128（frag_at 把地址转成了 i32x4*）。
    // 槽内偏移 row*128 + chk*16 都是 16 的倍数，基址也对齐了，每次读才是对齐的
    // uint8_t smem0[STAGE_B]：按字节开的数组，里面是打包好的 fp4 和 E8M0 原始字节。
    // 读的时候再按 16 B（A 片段）或 4 B（scale dword）来解释
    __shared__ __attribute__((aligned(16))) uint8_t smem0[STAGE_B]; // 32kb+2KB
    __shared__ __attribute__((aligned(16))) uint8_t smem1[STAGE_B];
    __shared__ __attribute__((aligned(16))) uint8_t smem2[STAGE_B];
    __shared__ __attribute__((aligned(16))) uint8_t smem3[STAGE_B];
    // lds_buf(u)：按编译期的 u % RING 选出对应的槽。
    auto lds_buf = [&](auto u) __attribute__((always_inline)) -> uint8_t* {
        constexpr int i = decltype(u)::value % RING;
        if constexpr (i == 0) return smem0;
        else if constexpr (i == 1) return smem1;
        else if constexpr (i == 2) return smem2;
        else return smem3;
    };

    // 工作组 256 个线程（4 个 wave）。threadIdx.x = 0..255：
    //   wave 0 → tid 0..63，wave 1 → 64..127，wave 2 → 128..191，wave 3 → 192..255。
    const int lane  = threadIdx.x & 63;   // wave 内的 lane 号 0..63【例】tid 101 → lane = 37
    // wave 号 0..3。【例】tid 101 → wave = 1。threadIdx.x >> 6 在一个 wave 内都相同，但
    // 编译器看不出来；用 readfirstlane 显式告诉它，之后所有由 wave 推出来的 tile 基址就都
    // 放在 SGPR 里，不占 VGPR。
    // 以16*128来约定的
    const int wave  = __builtin_amdgcn_readfirstlane(threadIdx.x >> 6); // 0-4
    // 一条 MFMA 16×16×128 的 B 操作数是 16 行 × 128 个 K
    // row16和g指示了本lane必须要拿的是原A,B中指定的行号，格号，你可以shuffle,swizzle，但最终要按照符合这个地址的元素给入
    const int row16 = lane & 15;   // 片段里的行号【例】lane 37 → 5
    const int g     = lane >> 4;   // 32-K 块号 0..3【例】lane 37 → 2

    // 【例】M = N = 8192，K = 5120（A/B 都是 [8192, 5120]，out = A @ B^T）
    const int Kp      = K >> 1;    // 每行的字节数（2 个 fp4 / 字节）【例】2560
    // ktile 是B的概念，
    const int KT      = K / 64;    // B 预重排里每行的 K tile 数（一个 K tile = 64 个 K = 32 字节）【例】80
    const int nstages = K / 256;   // stage 数，每个 stage K = 256【例】20
    // scaleN_pad 是 launcher 传入的：K/32 向上取整到 8。【例】160

    // ---- persistent tile 调度 -------------------------------------------------
    // 每个 CU 常驻一个工作组，依次处理 tile id = blockIdx.x + i*gridDim.x。软件流水跨 tile
    // 不中断：下一个 tile 的前几个 stage 在当前 tile 最后几批 MFMA 期间就已经预取，所以只有
    // 第一个 tile 要付一次完整的 load 延迟。
    //
    // tile 顺序（光栅化）：工作组是轮流派发到 8 个 XCD 上的（工作组 b 在第 b%8 个 XCD），
    // 每个 XCD 有自己的 L2。如果按行主序直接编号，每个 XCD 拿到的是一堆分散的 tile，B 几乎
    // 没有复用。这里重新映射：
    //   1. 让每个 XCD 处理一段连续的 tile 编号；
    //   2. XCD 内的编号按 GROUP_M（= 4）个 M tile 高的“列组”排列：同一时刻驻留在一个 XCD
    //      上的约 32 个工作组覆盖 4（M）× 8（N）个 tile，每个 B tile 在 L2 里被用 4 次，
    //      每个 A tile 约 8 次。
    // （gridDim.x 是 NXCD 的倍数，所以对每个 i，id % NXCD 都是这个工作组所在的 XCD。）
    // 【例】1024 个 tile → 每个 XCD 连续处理 128 个；M 方向 32 个 tile → 8 个列组。
    constexpr int NXCD = 8, GROUP_M = 4;
    const int num_m = (M + MTILE - 1) / MTILE;   // M 方向的 tile 数【例】8192/256 = 32
    const int num_n = (N + NTILE - 1) / NTILE;   // N 方向的 tile 数【例】32
    const int total = num_m * num_n;             // tile 总数【例】32×32 = 1024

    // 加载地址 = 每个 lane 自己的部分（和 tile 无关，见下面的 *_lane）+ 每个 tile 的部分
    // （整个 wave 相同，放在 SGPR 里，最后加到描述符基址上，见 issue_one）。buffer load
    // 会用描述符的范围检查 voffset + soffset + 立即数偏移，所以 M / N 之外的行、子块、
    // scale 块都会被硬件读成 0。
    struct TileOff
    {
        // m0、n0 描述的是输出 C 里的位置
        // 一个 tile 指的是输出 C 的一块，大小 256 × 256。算这一块需要 A 的 256 行和 B 的 256 行，每行都是全部 5120 个 K
        // m0： 本 tile 在 C 里的起始全局行号，也是要读的 A 的起始行。4 个 wave 共用同一个值。256，512，768，1024
        // n0： 本 tile 里本 wave 的起始全局列号，也是要读的 B 的起始行。64的倍数0，64，128，192，256
        int m0, n0;         // tile 的起始行、起始列；n0 已经加上了本 wave 的列偏移
        int a, b, as, bs;   // A、B、A scale、B scale 在各自缓冲区里的字节偏移
    };
    // A → LDS 的 DMA：一条指令（第 j 条）填 stage 里的 8 行。LDS 的写入位置由 lane 号固定：
    // lane 写到 (行 = lane/8, 物理块 = lane%8)，每块 16 字节。为了实现 XOR swizzle，改的是
    // “从全局读哪一块”：读逻辑块 = 物理块 ^ (行 & 7)。结果是 LDS 第 row 行的物理块 p 里放的
    // 是逻辑块 p ^ (row & 7)，读的时候按同样的异或换算回来（见 frag_at）。第 j 条指令再往下
    // 偏移 8 行（a_jstride）。
    // 【例】lane 37：行 = 37/8 = 4，物理块 = 37%8 = 5，逻辑块 = 5^4 = 1
    // → a_lane = 4·Kp + 1·16 = 4·2560 + 16 = 10256（本 wave 第 0 条 DMA 的 VGPR 偏移）
    // 这里是HBM地址
    const int a_lane = (lane >> 3) * Kp + (((lane & 7) ^ (lane >> 3)) * 16); 
    // B 片段：一个 16 行子块的一个 K-128 段是连续的 1 KB，lane L 读其中第 L 个 16 字节
    // 这是一个16*128内的偏移，比wave还小，
    // (row16, g）代表本lane要拿的B的是把本次用16行128列得行号0-16，格号0-4, 做shuffle，小块被先按格在行排到新矩阵的行方向了，
    // 这也就是说原本的1行2格，被排到从左往右，1*16+2这个小格位了，也就是2*16*16+1*16=2*256+1*16
    const int b_lane = g * 256 + row16 * 16;
    // scale：shuffle 布局里，32 行 × 一个 stage 是 256 字节，lane L 读第 L 个 dword。
    // 【例】lane 37 → s_lane = 148
    const int s_lane = lane * 4; // TODO
    // tile_off(id)：输入一个 tile 的编号，输出“当前这个 wave 算这个 tile 时，各种数据从哪里开始”。
    // 逻辑上，我们希望为一个xcd指定4*8的tile作为一轮来计算，但是硬件上只认id，block总是按照工作组轮流派发，
    // 那么我们可以针对每个id，不按照行主序来给id排序实际地址，而是按照调整后的L2cache的方式来赋予每个id相应的地址
    // 原id映射行主序的某块tile，某行某列，在函数内，被映射到另外一个某行某列的tile
    auto tile_off = [&](int id) __attribute__((always_inline)) {
        // 预取会越过最后一个 tile：把编号夹到一个合法的 tile 上，读进来的数据没人用。
        id = min(id, total - 1); // id 是全局编号，取值 0 到 1023
        // 第 1 步：把编号 id = q*8 + x（x 是 XCD 号）映射成 x*(total/8) + q，于是第 x 个 XCD
        // 处理连续的一段编号。凑不满 8 个的尾巴保持原编号。
        // 原本是0-1023，现在通过公式，映射到0-127，128-255，256-383，384-511，512-639，640-767，768-895，896-1023
        
        if (id < total / NXCD * NXCD)
        // （8个xcd一次循环，这次落在第几个xcd上了）*每个xcd处理128个tile + （该xcd拿到的第几个tile号）
        // XCD 0 原本拿到的是隔 8 取 1 的号（0、8、16……），重排后变成连续的 0 到 127。XCD 1 变成 128 到 255，依此类推。
        // 按 XCD 重新编号。id = 8q + x 变成 r = 128·x + q，让每个 XCD 拿到连续的一段号。
            id = (id % NXCD) * (total / NXCD) + id / NXCD;
        // 第 2 步：按 GROUP_M 个 M tile 一组的“列组”展开，组内先走 M、再走 N。

        const int group   = id / (GROUP_M * num_n);        // id/(4*32)第这个 tile 落在 C 的第几个 4 行条带里
        const int first_m = group * GROUP_M;               // 这个条带的第一行tile号，0，4，8，12 - 32
        const int gsize   = min(num_m - first_m, GROUP_M); // 最后一组可能不满 4 个， 这里看作4先
        TileOff t;
        // 现在只考虑映射后的，id值所对应的实际地址，
        // 比如id为135，映射为57，那就看57在C中的地址，总之总有128个id，会被映射为0-127，
        // 这里黄括号，(该条的第一个tile号 + (条内第几个tile0-128) % 一个组有几个tile)
        // 分组时是4格一列，所以取余就知道这个id的是由上到下第几个格，然后乘以tile大小256*256，就是该tile在C中的起始行(最小单位）
        // 以xcd0为例，总有被映射后的0-128作为id输入过来，那么就很自然的，按列分组，4个一组
        t.m0 = (first_m + (id % (GROUP_M * num_n)) % gsize) * MTILE; // 行号0-8191
        // 带内第几个tile，除法之后就是第几组，组数乘以tile大小 256，然后，加上本wave的偏移
        t.n0 = (id % (GROUP_M * num_n)) / gsize * NTILE + wave * WNTILE; // 列号0-8191
        // A：本 wave 负责 DMA 的那 1/4 行（A_LDS_PER_WAVE*8 = 64 行）的起始字节
        // 【例】tile (m0,n0)=(0,0)、wave 1 → t.a = (0 + 1·64)·2560 = 163840
        // 八是一条DMA搬运几行，64*16B=1KB，1KB/（256*0.5）= 8，也就是八行
        // t.a 是每个 wave 自己那块 64 行 × 5120 的左上角，不是整条 256 行的左上角。
        t.a  = (t.m0 + wave * A_LDS_PER_WAVE * 8) * Kp; // 这样就拿到了本wave取的64行的在全局中的起始地址
        // B：本 wave 第一个 16 列子块的起始字节（每个 16 行组占 KT*512 = 16 × K/2 = 8·K 字节）
        // 512 是一个 K tile 在一组 16 行里占的字节数：16 行 × 32 B = 512 B
        // 【例】wave 1 → n0 = 64，t.b = (64/16)·80·512 = 4·40960 = 163840
        // t.n0是B的N方向的序号，shuffle是16行N一组来shuffle的
        // 按我的话来说，就是16*5120这一个shuffle组，有多少字节，行数除以16是有多少组，16*5120*0.5=40960字节
        t.b  = (t.n0 >> 4) * KT * 512;// t.b 是本 wave 要读的那 64 行 B 在重排后的 B 里的起始字节
        // scale：本 tile 第一行 A 所在的那个 32 行 scale 块的起始字节
        // 先除以32看是第几块*每块32行，乘以scaleN_pad，就是该scale块在全局中的起始地址，scale是一字节一个的
        // 【例】m0 = 0 → t.as = 0；n0 = 64 → t.bs = (64/32)·32·160 = 2·5120 = 10240
        t.as = (t.m0 >> 5) * 32 * scaleN_pad;
        t.bs = (t.n0 >> 5) * 32 * scaleN_pad;
        return t;
    };

    // 描述符的第 4 个字：gfx9 系列 buffer 描述符的常用取值（32 位数据格式，CK / aiter 用的
    // 也是这个值）。
    constexpr int RSRC_W3 = 0x00020000;
    // scale 缓冲区按 32 行补齐【例】M = N = 8192 时 Mpad32 = Npad32 = 8192
    const int Mpad32 = (M + 31) & ~31;
    const int Npad32 = (N + 31) & ~31;
    // p：这块显存的起始地址，比如 A、B、AScale、BScalebytes：这块显存有多少字节
    // 输入：p = 0x7f2a9e200000（48 位虚拟地址），bytes = 8192·2560 = 20971520
    // 字 0 = (int)a                = 0x9e200000   地址的低 32 位
    // 字 1 = (int)(a>>32) & 0xffff = 0x00007f2a   地址的高 16 位；这个字的高 16 位是 stride，被清成 0
    // 字 2 = bytes                 = 0x01400000 = 20971520   范围（越界检查用）
    // 字 3 = RSRC_W3               = 0x00020000   格式 / 标志
    // WHY 每个 lane 只需一个 32 位偏移 省寄存器，LDS 需要，
    auto rsrc_of = [&](const void* p, int bytes) __attribute__((always_inline)) {
        const uint64_t a = reinterpret_cast<uint64_t>(p);
        return i32x4{(int)a, (int)(a >> 32) & 0xffff, bytes, RSRC_W3};
    };
    // sq：对描述符的 4 个字各做一次 readfirstlane，明确变成 SGPR 值。只在这里做一次：如果
    // 每次使用时才做，编译器会把描述符放在 VGPR 里喂给 v_readfirstlane，约 10 个 VGPR 在
    // 整个 tile 循环里一直活着——256×256 就曾因此 spill。
    auto sq = [](const i32x4& r) __attribute__((always_inline)) {
        return i32x4{__builtin_amdgcn_readfirstlane(r[0]), __builtin_amdgcn_readfirstlane(r[1]),
                     __builtin_amdgcn_readfirstlane(r[2]), __builtin_amdgcn_readfirstlane(r[3])};
    };
    // 【例】A/B 各 8192·2560 = 20 971 520 B；A/B scale 各 8192·160 = 1 310 720 B
    const i32x4 a_rsrc  = sq(rsrc_of(A, M * Kp));                     // A：M 行
    const i32x4 b_rsrc  = sq(rsrc_of(B, N * Kp));                     // B：N 行
    const i32x4 as_rsrc = sq(rsrc_of(AScale, Mpad32 * scaleN_pad));   // A scale：按 32 行补齐
    const i32x4 bs_rsrc = sq(rsrc_of(BScale, Npad32 * scaleN_pad));   // B scale：按 32 行补齐
    // 输出的描述符（给 epilogue 里的 buffer store builtin 用）：范围是 M*N*2 字节，所以第 M
    // 行以后的写会被硬件丢弃。
    // MFMA计算的到float32,然后输出的时候epilogue用指令把两个fp32打包位两个bf16
    const auto o_rsrc  = __builtin_amdgcn_make_buffer_rsrc(
        (void*)out, 0, M * N * (int)sizeof(__hip_bfloat16), RSRC_W3);

    // 累加器：每个 wave 有 MSUB×NSUB 个 16×16 子块。【例】16×4 = 64 个 f32x4 = 256 个 AGPR
    f32x4 acc[MSUB][NSUB];
#pragma unroll
    for (int mi = 0; mi < MSUB; ++mi)
#pragma unroll
        for (int ni = 0; ni < NSUB; ++ni)
            acc[mi][ni] = f32x4{0.f, 0.f, 0.f, 0.f};

    // ---- 每个 step 的全局访存 ---------------------------------------------------
    // 一个 step（处理一个 stage）的全局访存被拆成 LOADS_PER_STAGE 条可以单独发射的指令，
    // 编号为 k，按发射顺序依次是：B 片段、B scale（读进给定的寄存器），然后是 A 行和
    // A scale（DMA 进 LDS 槽 lds）。
    //
    // 全部用 inline asm 写，K 循环里的每个 vmcnt 等待都是手写的（W_START / W_MID）。原因：
    //   * LDS DMA 不能交给编译器：编译器会在每个无法证明地址不重叠的 LDS 读之前插入
    //     vmcnt(0)，把预取全部排空；
    //   * 一旦 DMA 对编译器不可见，编译器替 B 插入的等待就会算错（多等或少等），逼得下一个
    //     stage 的 A DMA 提前落地。
    // 读进来的寄存器只会被（同样是 asm 的）MFMA 读，而且是在手写的等待之后。
     // 对一次加载的一条256(个k)*16(行一组)=2kb, 一个wave一条load 1kb， 所以要2次，一个wave有NSUB条， 2*8就是一个wave要发几条B的load【例】8
    constexpr int NB              = 2 * NSUB;  
    // 一个wave在一个 step 的总条数 = B 片段 + B scale + A DMA + A scale DMA【例】8 + 2 + 8 + 2 = 20
    // NSUB/2：B_scale，对一个小条32*(256/32）来说(B scale 32行为一组)，需要256B，每个lane用load_dword 4B,
    // 一个wave load 256B。NSUB表示有4个16*(256/32),而上面的分组是32，那么NSUB/2就是这个wave需要发的数量
    // A_LDS_PER_WAVE=8条，每次wave加载1B，4个wave,合起来32KB=256*256/2
    constexpr int LOADS_PER_STAGE = NB + NSUB / 2 + A_LDS_PER_WAVE + S_LDS_PER_WAVE;
    // 编号区间：[0, KBS) B 片段，[KBS, KA) B scale，[KA, KAS) A DMA，[KAS, L) A scale DMA
    constexpr int KBS = NB, KA = KBS + NSUB / 2, KAS = KA + A_LDS_PER_WAVE;
    // 本 wave 在 LDS 槽里负责的那 1/4 的起点（每条 DMA 1 KB）
    const uint32_t lds_wave_off   = wave * A_LDS_PER_WAVE * 1024; // 每个wave，发8条，8条是8*128=1KB
    // 从HBM取是行主序，8*256=1KB=一次，然后4个wave，每个8次，一共32kb
    const int a_jstride           = 8 * Kp;   // 第 j 条 A DMA 覆盖第 8j .. 8j+7 行
    // A 的指令加载 tile to 的第 st 个 stage；B / scale 的指令加载 tile tob 的第 stb 个 stage
    // （两者的提前量不同，所以分开传）。
    //
    // 寻址方式照搬 asm kernel：
    //   * 每条 load 有自己固定的、按 lane 算好的 VGPR 偏移，只算一次，用 opaque 包起来，
    //     编译器就不会在每条 load 前重新做加法；
    //   * soffset 固定为 0；
    //   * 随 tile / stage 变化的部分加到描述符的基址上（rsrc_at）：每个 step 每个描述符只要
    //     几条标量指令，而不是每条 load 前一串计算。描述符的范围同步缩小，所以硬件的范围检查
    //     （覆盖 voffset + 立即数偏移）依然会裁掉 M / N 之外的行。
    // 实测：非 0 的 SGPR soffset 会让 SQ 发一条 VMEM 指令要 2 个 cycle
    // （SQ_INST_CYCLES_VMEM_RD）；每条 load 前面的一串标量（或向量）地址计算会卡住按顺序
    // 发射的指令流。两者加起来每个 stage 约 6%。
    auto opaque = [](int v) __attribute__((always_inline)) {
        asm volatile("" : "+v"(v));   // 空 asm：让编译器把 v 当成一个“看不透”的 VGPR 值
        return v;
    };
    // 256 行的 tile 上，scale 的 load（每个 step NSUB/2 + S_LDS_PER_WAVE 条）共用一个 lane
    // 偏移，块偏移改走 SGPR soffset：每条多一个发射 cycle，但省下 4 个 VGPR——不这样的话，
    // 这个 tile 会在 epilogue 附近 spill。
    constexpr bool SCALE_SOFF = MSUB >= 16;
    // 对A每个wave发8条，NSUB=4
    int a_voff[A_LDS_PER_WAVE], b_voff[NSUB];   // 每条 A DMA / 每个 B 子块一个 VGPR 偏移
    int bs_voff[SCALE_SOFF ? 1 : NSUB / 2], as_voff[SCALE_SOFF ? 1 : S_LDS_PER_WAVE];
    // 以下三个偏移，都是本 lane 的偏移，每个 lane 的值都不一样，所以放在 VGPR 里。是数组也确实因为一个 stage 要发多条 load。每条 load 都从同一个基址出发，去不同的位置，各自需要一个固定的偏移。
    // 第 j 条 A DMA：lane 偏移 + j × 8 行， j = 0-7
    // 偏移真的会占用VGPR，这里就i占了8个VGPR了
    // 本lane, 共有j=8次取数，8行*256个=1kb，每次都是行内同一个位置块1*32，
    static_for<A_LDS_PER_WAVE>([&](auto j) { a_voff[j] = opaque(a_lane + j * a_jstride); });
    // 第 ni 个 B 子块：lane 偏移 + ni 个 16 行组。这里还是b_lane的偏移
    // NSUB = 4：每个 wave 在 N 方向有 4 个 16 行子块，一个 stage 共 64(N) × 256(K)。
    // b_voff[ni] = b_lane（本 lane 在一条 MFMA 的 1 KB 里的位置，含 32-K 块 g）
    //            + ni 个 16 行组（每组跨整个 K，KT·512 = 16·K/2 字节）。
    // K 的另外两层不在这里：K 半段 h 走立即数 h·1024，stage 走描述符基址 stb·2048。
    // 然后ni表示第几个16行组，那么一组是16*5120/2字节=（5120/64）* 512；KT这个值真的很抽象
    static_for<NSUB>([&](auto ni) { b_voff[ni] = opaque(b_lane + ni * KT * 512); });
    if constexpr (SCALE_SOFF)
    {
        // 256 行的 tile：A、B 的 scale 共用一个 lane 偏移，块偏移放到 soffset
        bs_voff[0] = as_voff[0] = opaque(s_lane);
    }
    else
    {
        // 第 p 个 B scale 块（覆盖子块 2p 和 2p+1）
        static_for<NSUB / 2>([&](auto p) { bs_voff[p] = opaque(s_lane + p * 32 * scaleN_pad); });
        // 第 j 条 A scale DMA 搬第 min(j*WAVES + wave, NSB-1) 块（各 wave 轮流分）
        static_for<S_LDS_PER_WAVE>([&](auto j) {
            as_voff[j] = opaque(s_lane + min(j * WAVES + wave, NSB - 1) * 32 * scaleN_pad);
        });
    }
    // rsrc_at(r, off)：把描述符的基址往后移 off 字节，范围同步减少 off（不小于 0）。这样随
    // tile / stage 变化的偏移只改 SGPR 里的描述符，硬件的范围检查依然有效。
    auto rsrc_at = [](const i32x4& r, int off) __attribute__((always_inline)) {
        const uint64_t a = (((uint64_t)(uint32_t)r[1] << 32) | (uint32_t)r[0]) + (uint32_t)off;
        return i32x4{(int)a, (int)(a >> 32), max(r[2] - off, 0), r[3]};
    };
    // 一个 step 处理的是一个 stage，也就是 256 个 K，处理一个 stage（两段）
    // 段 0 发下一个 stage 的 B：8 条 load 加 2 条 B scale。
    // 段 1 发再下一个 stage 的 A：8 条 DMA 加 2 条 A scale。
    // 到了一个 tile 的最后两个 step，这些 load 读的已经是下一个 tile 的数据
    // issue_one<k>(...)：发出一个 step 里的第 k 条全局访存。k 是编译期常量，所以每次调用都
    // 展开成一条确定的指令。
    //   to / st / lds        ：A 的目标 tile、stage 号、LDS 槽
    //   tob / stb / bf / bs  ：B 的目标 tile、stage 号、目的寄存器
    // k 的含义：一个 step 里第几条全局访存
    // 编号区间：[0, KBS) B 片段，[KBS, KA) B scale，[KA, KAS) A DMA，[KAS, L) A scale DMA
    // 256×256 的 tile 每个 step 一共有 20 条全局访存（693 行的 LOADS_PER_STAGE = 8 + 2 + 8 + 2）
    // 这里只有从HBM加载的指令，A的寄存器加载，在后面交错在MFMA
    // kc：第几条访存（k = 0..19）；
    // to/st/lds：A 的目标 tile、stage、LDS 槽（k = 10–19 用）；
    // tob/stb：B 的目标 tile、stage；bf/bs：B 片段、B scale 写进的寄存器（k = 0–9 用）

    // 共用的偏移（wave级别的)放在 SGPR 和立即数里，每个 lane 不同的那部分放在 VGPR 里，
    // 比如 a_voff[j]、b_voff[ni]。每个 lane 从自己的地址开始读 16 B，64 个 lane 合起来是 1 KB
    auto issue_one = [&](auto kc, const TileOff& to, int st, uint8_t* lds, const TileOff& tob,
                         int stb, i32x4 (&bf)[NSUB][2],
                         int (&bs)[NSUB / 2]) __attribute__((always_inline)) {
        constexpr int k = decltype(kc)::value;
        // 绕开 clang 的一个问题：只在某个 if constexpr 分支里用到的捕获变量会被漏掉，所以先
        // 把它们全部“用”一遍。
        (void)a_jstride, (void)a_rsrc, (void)b_rsrc, (void)as_rsrc, (void)bs_rsrc;
        (void)a_lane, (void)b_lane, (void)s_lane, (void)wave;
        (void)a_voff, (void)b_voff, (void)bs_voff, (void)as_voff, (void)rsrc_at;
        if constexpr (k < KBS)
        {
            // B 片段：第 ni 个子块的第 h 个 K 半段（每个 K-128 段 1 KB）。地址 = 描述符基址
            // （tile 偏移 + stage 偏移，每个 stage 2 KB）+ VGPR 偏移 b_voff[ni] + 立即数 h*1024。
            // 这里k取值为8，那么ni=0-4，h=0-1
            constexpr int ni = k / 2, h = k % 2; // 把k
            // soffset固定为0，offen启用voffset，
            // 本lane的偏移voffset, 
            // 描述符：基址已经挪到“目标 tile 的这一组、第 stb 个 stage”；
            // offset%3立即数, 编译期常数，前半段是 0，后半段是 1024。
            asm volatile("buffer_load_dwordx4 %0, %1, %2, 0 offen offset:%3"
                         : "=v"(bf[ni][h]) // 目的寄存器
                         // tob.b本 wave 在目标 tile 里第一个 16 行组的起点，wave 的偏移已经包含在里面
                         // stb·2048：这一组里第 stb 个 stage 的位置。一个 stage 在一组里占 16 行 × 128 B = 2 KB。
                         : "v"(b_voff[ni]), "s"(rsrc_at(b_rsrc, tob.b + stb * 2048)), "i"(h * 1024));
        }
        else if constexpr (k < KA)
        {
            // B scale：第 p 个 32 行块里本 lane 的 dword（每个 stage 256 字节）
            constexpr int p = k - KBS;
            if constexpr (SCALE_SOFF)
                // 256 行的 tile：块偏移走 soffset（%3）
                // 本wave负责的64，是32*8为一组，发两次
                // 一行5120，scale有160个，现在要发下一个32*160的scale指令，那么就是要隔32*160字节，意思是换行了
                asm volatile("buffer_load_dword %0, %1, %2, %3 offen"
                             : "=v"(bs[p])
                             : "v"(bs_voff[0]), "s"(rsrc_at(bs_rsrc, tob.bs + stb * 256)),
                               "s"(__builtin_amdgcn_readfirstlane(p * 32 * scaleN_pad)));
            else
                // 其他 tile：块偏移已经算进 bs_voff[p]，soffset 为 0
                asm volatile("buffer_load_dword %0, %1, %2, 0 offen"
                             : "=v"(bs[p])
                             : "v"(bs_voff[p]), "s"(rsrc_at(bs_rsrc, tob.bs + stb * 256)));
        }
        else
        {
            // 下面两类是 LDS DMA：buffer_load ... lds 把读到的数据直接写进 LDS，写入地址 =
            // M0 + lane × 每个 lane 的字节数，不经过 VGPR。所以先用 s_mov_b32 把目的地址写进
            // M0。（编译器会对 clobber 列表里的 m0 报警告，见 summary.md。）
            // address_space(3) 这个转换确实是在告诉编译器“这个指针指向 LDS”，但 lds_base 这个变量本身，
            // 是为了拿到一个具体的数：这个槽在 LDS 里的字节地址，比如 0、0x8800。拿到之后写进 M0，DMA 指令才知道把数据写到 LDS 的哪里
            const uint32_t lds_base = (uint32_t)(uintptr_t)(__attribute__((address_space(3))) void*)lds;
            if constexpr (k < KAS)
            {
                // A DMA 的第 j 条：搬本 wave 负责的第 j 个 8 行组（1 KB）。j=0-7
                // 源地址 = 描述符基址（tile 偏移 + stage 偏移，每个 stage 每行 128 字节）+ a_voff[j]；
                // 目的地址 = 本 wave 在槽里的区域 + j KB。
                constexpr int j = k - KA;
                asm volatile("s_mov_b32 m0, %0\n\t"
                             "buffer_load_dwordx4 %1, %2, 0 offen lds"
                             :
                             : "s"(lds_base + lds_wave_off + j * 1024), "v"(a_voff[j]),
                             // 本wave那64行起点，第st个stage,每个128字节
                               "s"(rsrc_at(a_rsrc, to.a + st * 128))
                             : "memory", "m0");
            }
            else
            {
                // A scale DMA 的第 j 条：搬第 p 块（256 字节），放在槽里 A 数据的后面
                constexpr int j = k - KAS;
                const int p = min(j * WAVES + wave, NSB - 1);
                if constexpr (SCALE_SOFF)
                    asm volatile("s_mov_b32 m0, %0\n\t"
                                 "buffer_load_dword %1, %2, %3 offen lds"
                                 :
                                 : "s"(lds_base + STAGE_A + p * 256), "v"(as_voff[0]),
                                 // 本wave那64行起点，第st个stage,每个256字节
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

    // ---- 一个 step 的指令编排 ----------------------------------------------------
    // 一个 step 处理一个 stage，循环结构照搬 asm kernel：两段 MFMA，段 0 算 M 片段
    // [0, MS0)，段 1 算 [MS0, MSUB)，每段都覆盖两个 K-128 半段。
    //   * 每段开头都有 lgkmcnt(0)：这一段要用的 A 片段（上一段期间 ds_read 的）已经到了。
    //     段 1 开头还要等 stage s+1 的 A DMA，并做 barrier，让所有 wave 都能看到它。这是
    //     每个 stage 唯一的 barrier：RING > PF + 1 时，不会有哪个槽在还有 wave 读它的时候
    //     被重填（像 asm 那样在段 0 也放一个 barrier，实测只慢约 2%）。
    //   * 一段计算期间，wave 把下一段的全部 A 片段（本 stage 的段 1，或 stage s+1 的段 0）
    //     读进另一套片段缓冲：有整整一段的延迟余量，MFMA 之间不用等 LDS。
    //   * 段 0 把 stage s+PF_B 的寄存器 load（B 片段和 B scale）穿插在前 LOAD_WIN/8 的 MFMA
    //     之间；段 1 穿插 stage s+PF 的 A DMA 和 A scale DMA。一次只插一条（约 20 条连着发
    //     会把 VMEM 队列塞满，卡住按顺序发射的指令流）。
    //   * 段内 MFMA 的顺序是 asm 的 2×2 之字形：每个 K 半段内，按 B 片段对、A 片段对展开，
    //     对内顺序是 (n,m) (n,m+1) (n+1,m) (n+1,m+1)。
    // 每个元素之间都放了 sched_barrier，把这个交错固定下来——否则编译器会把它们重新归堆。
    //
    // MS0：段 0 的 M 片段数，取偶数，这样两段都能成对使用 scale（一个 scale dword 覆盖两个
    // 子块）。【例】MSUB = 16 时 MS0 = MS1 = 8；MSUB = 10 时 MS0 = 6、MS1 = 4。
    constexpr int MS0      = (MSUB / 2 + 1) / 2 * 2; // 8
    constexpr int MS1      = MSUB - MS0; // 8 
    // load 只撒在每段前 7/8 的 MFMA 里：实测比撒满整段每个 stage 快约 2%。
    constexpr int LOAD_WIN = 7;
    static_assert(PF >= 2, "stage s+1's A DMA is issued before step s");
    // 每条 load 至少要有一条 MFMA 可以穿插
    static_assert(2 * MS0 * NSUB >= KA && 2 * MS1 * NSUB >= LOADS_PER_STAGE - KA,
                  "need at least one MFMA per load");
    // vmcnt 记账（vmcnt 按发射顺序完成；每个 step 在段 0 发寄存器 load，在段 1 最后发 A DMA）：
    //   * W_START，在 step s 的第一条 MFMA 之前：本 stage 的 B / scale 寄存器（在 step s-PF_B
    //     里发出）必须已经到了。在它们之后发出的只有：那个 step 的 A DMA，以及之后每个 step
    //     的全部指令。只要“最多剩这么多条没完成”，它们就一定到了。
    //   * W_MID，在段 1 的 barrier 处：stage s+1 的 A DMA（step s+1-PF 的尾部）必须已经到了。
    //     在它之后发出的只有：step s+2-PF .. s-1 的全部，以及本 step 段 0 的寄存器 load。
    //   * 每个 tile 的 epilogue 在两个 step 之间发出 EPI_STORES 条 buffer store（vmcnt 也算
    //     store），所以下一个 tile 的第一个 step 要多放宽这么多条。
    // 【例】W_START = (20 - 10) + 20*0 = 10，W_MID = 10 + 20*0 = 10，EPI_STORES = 32
    // vmcnt 记的是本 wave 发出去、还没完成的显存访问条数，load 和 store 都算
    // 所以“等某一条 load 到齐”，就等于“等到最多只剩它后面发的那几条”。
    constexpr int EPI_STORES = MSUB * (NSUB / 2); // 16 * (4/2) = 32
    // 等本 stage 的 B，B(s) 是在 step s−1 段 0 那一列发的（蓝格），到 step s 段 0 才用，① 就是用之前的那条虚线。一般情况下 B(s) 在 step s−PF_B 的段 0 发出，在它之后发出的有两部分：
    // 同一个 step 段 1 的 A 类：LOADS_PER_STAGE − KA = 10 条，这里的shape A是8，ka 10，等10条指令到
    // 中间隔着的 PF_B − 1 个完整 step，每个 20 条。
    constexpr int W_START    = (LOADS_PER_STAGE - KA) + LOADS_PER_STAGE * (PF_B - 1); // (20 - 10) + 20*0 = 10
    // 等下个 stage 的 A。 A(s+1) 是在 step s−1 段 1 发的，而且是那个 step 的最后 10 条。在它之后，只发过本 step 段 0 的 10 条 B(s+1)，也就是 KA；如果 PF 更大，中间还要加上 PF − 2 个整 step。PF = 2，所以结果是 10 + 0 = 10。
    constexpr int W_MID      = KA + LOADS_PER_STAGE * (PF - 2); //1 0
    static_assert(W_MID + EPI_STORES <= 63 && W_START + EPI_STORES <= 63, "vmcnt is 6 bits");
    // s_waitcnt 立即数的编码（gfx9）：vmcnt 的低 4 位在 [3:0]、高 2 位在 [15:14]，expcnt 在
    // [6:4]，lgkmcnt 在 [11:8]。
    // wait_vm_lgkm0<n>：等到 vmcnt ≤ n 并且 lgkmcnt = 0（expcnt 填最大值，表示不等）。
    auto wait_vm_lgkm0 = [](auto nc) __attribute__((always_inline)) {
        constexpr int n = decltype(nc)::value;
        __builtin_amdgcn_s_waitcnt((n & 15) | (7 << 4) | (0 << 8) | ((n >> 4) << 14));
    };
    // WAIT_ALL：只等 vmcnt(0)（expcnt、lgkmcnt 都填最大值，表示不等）
    // 表示等到 vmcnt = 0，也就是所有全局访存都完成
    constexpr int WAIT_ALL = (7 << 4) | (15 << 8);

    // frag_at(lds, h, mi)：从 LDS 槽里读本 lane 的 A 片段——第 mi 个 16 行子块、第 h 个 K
    // 半段。行 = mi*16 + row16，逻辑块 = h*4 + g（每行 8 个 16 字节块），再按写入时的 XOR
    // swizzle 换算成物理块。对应一条 ds_read_b128。
    auto frag_at = [&](const uint8_t* lds, int h, int mi) __attribute__((always_inline)) {
        const int row = mi * 16 + row16;
        const int chk = (h * 4 + g) ^ (row & 7);
        return *reinterpret_cast<const i32x4*>(lds + row * 128 + chk * 16);
    };
    // scale_at(lds, p)：读第 p 个 A scale 块里本 lane 的 dword（一条 ds_read_b32）。
    auto scale_at = [&](const uint8_t* lds, int p) __attribute__((always_inline)) {
        return *reinterpret_cast<const int*>(lds + STAGE_A + p * 256 + lane * 4);
    };
    // A 片段和 A scale 的双缓冲：段 0 用 afr0 / asc0 计算，同时填 afr1 / asc1；段 1 反过来。
    // 片段下标 = h × (这一段的 M 片段数) + 段内的 mi；scale 下标 = 段内的 mi / 2。
    // 大小按较大的段 MS0 分配。【例】每套 16 个片段 = 64 个 VGPR，两套 128 个
    // 这里两份缓冲区，一份用来算，另一份同时装下一批数据，这一轮算完，两份互换角色
    i32x4 afr0[2 * MS0], afr1[2 * MS0]; // 两份A片段
    int asc0[MS0 / 2], asc1[MS0 / 2]; // 两套A scale

#if MBS_TIMING
    // 分阶段计时：tick(b) 把距上一次 tick 的 cycle 数累加到第 b 个桶。
    // 桶：0 段 0 等待，1 段 0 计算，2 段 1 等待，3 段 1 计算，4 epilogue，
    //     5 / 6 每个 tile 第 1 / 2 个 step 的段 0 等待，7 这两个 step 的段 1 等待。
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
    // ---- section：一个 step 里的一段 -------------------------------------------
    // 第 SIG 段（0 或 1）：发出 TS 条 MFMA，计算本段的 MC 个 M 片段；同时从 rd_lds 读出下一段
    // 的 MN 个 M 片段（和它们的 scale）放进 nf / ns，并发出编号 [KLO, KLO + NL) 的全局访存。
    //   cf / cs   ：本段计算用的 A 片段 / A scale
    //   nf / ns   ：要填好的下一段 A 片段 / A scale
    //   rd_lds    ：下一段的片段所在的 LDS 槽（段 0 是本 stage 的槽，段 1 是下一个 stage 的槽）
    //   bf / bs   ：本 stage 的 B 片段 / B scale 寄存器
    //   nto, nst, nlds        ：本 step 要发的 A DMA 的目标（tile、stage、LDS 槽）
    //   ntob, nstb, nbf, nbs  ：本 step 要发的 B load 的目标（tile、stage、寄存器）
    // 一个section分M为两段，段0算A前128行256列，MS0 = 8 的意思就是段 0 算前 8 个 16 行，也就是前 128 行
    auto section = [&](auto sc, i32x4 (&cf)[2 * MS0], int (&cs)[MS0 / 2], i32x4 (&nf)[2 * MS0],
                       int (&ns)[MS0 / 2], const uint8_t* rd_lds, i32x4 (&bf)[NSUB][2],
                       int (&bs)[NSUB / 2], const TileOff& nto, int nst, uint8_t* nlds,
                       const TileOff& ntob, int nstb, i32x4 (&nbf)[NSUB][2],
                       int (&nbs)[NSUB / 2]) __attribute__((always_inline)) {
        // SIG是段号，0或1
        constexpr int SIG = decltype(sc)::value;
        // MC / MB：本段计算的 M 片段数和起始下标，SIG=0， MC=8， MB=0；SIG=1, MC=8, MB=8，
        constexpr int MC = SIG == 0 ? MS0 : MS1, MB = SIG == 0 ? 0 : MS0;
        // MN / MBN：要为下一段预读的 M 片段数和起始下标
        // SIG=0，MN=8，MBN=8；SIG=1, MN=8, MBN=0，
        constexpr int MN = SIG == 0 ? MS1 : MS0, MBN = SIG == 0 ? MS0 : 0;
        // 本段负责的MFMA条数，2个K半段，MC=8，NSUB=4，所以TS=64
        constexpr int TS  = 2 * MC * NSUB;   // 本段的 MFMA 条数（2 个 K 半段）【例】64
        // NF：预读的片段数（每个 M 片段 2 个 K 半段）；
        // NR：加上 scale 的总读取次数（一个 ds_read_b32 管 2 个子块）
        // 从LDS到寄存器，128*256*0.5=16KB，一个ds_read_b128，一个wave只能处理1kb,因此是16次
        // scale同理128*8=1024B，ds_read_b32，一个wave只能处理64*4B,因此是4次
        constexpr int NF  = 2 * MN, NR = NF + MN / 2; //【例】16、20
        // 本段负责的全局访存编号区间：段 0 是 [0, KA)（B），段 1 是 [KA, L)（A DMA）【例】各 10 条
        // SIG=0，KLO=0，NL=10；SIG=1, KLO=10, NL=10，
        constexpr int KLO = SIG == 0 ? 0 : KA, NL = (SIG == 0 ? KA : LOADS_PER_STAGE) - KLO;
        // load 撒在前 TLW 条 MFMA 里：取段长的 7/8，但至少保证每条 load 有一条 MFMA【例】56
        constexpr int TLW = TS * LOAD_WIN / 8 >= NL ? TS * LOAD_WIN / 8 : NL; // 56
        static_assert(NR <= TS, "one fragment read per MFMA at most");
        // 按 MFMA 编号 t = 0 .. TS-1 展开整段。每个 t 之前可能插一个 LDS 读、一条全局访存，
        // 然后是第 t 条 MFMA；它们之间都隔着 sched_barrier。
        static_for<TS>([&](auto tc) __attribute__((always_inline)) {
            constexpr int t = decltype(tc)::value; // 0-63
            // 第 r 个 LDS 读放在第 floor(r*TS/NR) 条 MFMA 之前（均匀撒开）。这里反过来求哪个 r
            // 落在当前的 t 上（没有就跳过）。【例】r = 0..19 落在 t = 0, 3, 6, 9, 12, 16, 19, …
            constexpr int r = (t * NR + TS - 1) / TS; // 0-19
            if constexpr (r < NR && r * TS / NR == t)
            {
                if constexpr (r < NF) // r=0-15
                    // 片段：K 半段 = r / MN，M 片段 = MBN + r % MN， MN = 8
                    nf[r] = frag_at(rd_lds, r / MN, MBN + r % MN);
                else // r=16-19
                    // scale：第 MBN/2 + (r - NF) 个 scale 块
                    ns[r - NF] = scale_at(rd_lds, MBN / 2 + r - NF);
            // 它前面的指令不能挪到后面，后面的指令也不能挪到前面。参数 0 表示所有种类的指令都不准越过。
            // 不要扰乱我们精心设计的指令流！
                __builtin_amdgcn_sched_barrier(0);
            }
            // 第 KLO+k 条全局访存放在第 floor(k*TLW/NL) 条 MFMA 之前。
            // 【例】k = 0..9 落在 t = 0, 5, 11, 16, 22, 28, 33, 39, 44, 50
            constexpr int k = (t * NL + TLW - 1) / TLW;
            if constexpr (k < NL && k * TLW / NL == t)
            {
                issue_one(std::integral_constant<int, KLO + k>{}, nto, nst, nlds, ntob, nstb, nbf,
                          nbs);
                __builtin_amdgcn_sched_barrier(0);
            }
            // 2×2 之字形：t → (K 半段 h, B 片段对 np, A 片段对 mp, 对内位置 q)。
            //   q = t%4 在 2×2 里依次走 (n,m) (n,m+1) (n+1,m) (n+1,m+1)；
            //   每 4 条换一个 A 片段对，走完本段的 MC/2 对再换下一个 B 片段对；
            //   走完 NSUB/2 个 B 片段对再换 K 半段。
            constexpr int q = t % 4, mp = (t / 4) % (MC / 2), np = (t / 2 / MC) % (NSUB / 2);
            constexpr int h = t / (MC * NSUB), ml = 2 * mp + (q & 1), ni = 2 * np + (q >> 1);
            constexpr int mi = MB + ml;   // 在整个 tile 里的 M 片段号
            // 操作数交换（B 片段放在第一个位置）：算的是子块的 C^T，于是 lane L 拿到第 L%16 行、
            // 第 (L/16)*4 .. +3 列——同一行的 4 个连续输出（见 epilogue）。
            // scale 字节号 = 2*h（哪个 K 半段）+ 子块的奇偶（ni&1 / mi&1）。
            mfma_fp4_sel<2 * h + (ni & 1), 2 * h + (mi & 1)>(acc[mi][ni], bf[ni][h],
                                                             cf[h * MC + ml], bs[ni / 2], cs[ml / 2]);
            __builtin_amdgcn_sched_barrier(0);
        });
    };

    // ---- step：处理一个 stage（两段）------------------------------------------
    //   lds / next_lds ：本 stage / 下一个 stage 的 LDS 槽
    //   bf / bs        ：本 stage 的 B 寄存器
    //   after_epi      ：是不是紧跟在 epilogue 后面的第一个 step（等待要多放宽 store 的条数）
    //   sidx           ：tile 内的 stage 号，只给计时用
    auto step = [&](const uint8_t* lds, const uint8_t* next_lds, i32x4 (&bf)[NSUB][2],
                    int (&bs)[NSUB / 2], const TileOff& nto, int nst, uint8_t* nlds,
                    const TileOff& ntob, int nstb, i32x4 (&nbf)[NSUB][2], int (&nbs)[NSUB / 2],
                    bool after_epi, int sidx) __attribute__((always_inline)) {
        (void)sidx;
        MBS_TICK(3)
        // 等待 1：本 stage 的 B / scale 寄存器到了（vmcnt），段 0 的 A 片段也到了（lgkmcnt(0)）。
        if (after_epi)
            wait_vm_lgkm0(std::integral_constant<int, W_START + EPI_STORES>{});
        else
            wait_vm_lgkm0(std::integral_constant<int, W_START>{});
        MBS_TICK(sidx == 0 ? 5 : sidx == 1 ? 6 : 0)
        __builtin_amdgcn_sched_barrier(0);
        // 段 0：用 afr0 / asc0 计算前 MS0 个 M 片段；从本 stage 的槽读段 1 的片段，放进
        // afr1 / asc1；中间穿插 B 的 load。
        section(std::integral_constant<int, 0>{}, afr0, asc0, afr1, asc1, lds, bf, bs, nto, nst,
                nlds, ntob, nstb, nbf, nbs);
        MBS_TICK(1)
        // 等待 2：stage s+1 的 A DMA 在本 wave 这边已经落地（vmcnt），本 wave 段 1 的片段也到了
        // （lgkmcnt(0)）。再做 barrier，保证所有 wave 的那份 DMA 都落地了——之后才能读别的 wave
        // 搬进来的数据（stage s+1 的段 0 片段）。
        if (after_epi)
            wait_vm_lgkm0(std::integral_constant<int, W_MID + EPI_STORES>{});
        else
            wait_vm_lgkm0(std::integral_constant<int, W_MID>{});
        __syncthreads();
        MBS_TICK(sidx < 2 ? 7 : 2)
        __builtin_amdgcn_sched_barrier(0);
        // 段 1：用 afr1 / asc1 计算后 MS1 个 M 片段；从下一个 stage 的槽读它段 0 的片段，放进
        // afr0 / asc0；中间穿插 A 的 DMA。
        section(std::integral_constant<int, 1>{}, afr1, asc1, afr0, asc0, next_lds, bf, bs, nto,
                nst, nlds, ntob, nstb, nbf, nbs);
    };

    // ---- epilogue：把一个 tile 的结果写回 ---------------------------------------
    // 写回之前：acc[mi][ni][i] 在 lane L 上是 C[mi*16 + L%16][ni*16 + (L/16)*4 + i]，即每个
    // lane 拿着一行里 4 个连续的输出。
    // 做法（不经过 LDS）：
    //   1. pair(mi, np)：取相邻两个 N 子块（ni = 2np 和 2np+1，共 32 列），转成 bf16，再用
    //      v_permlane16_swap 在 lane 组之间交换一半。之后 lane 组 g 拿着本行第
    //      np*32 + (g&1)*16 + (g>>1)*8 .. +7 列的 16 字节——整个 wave 合起来是每行 64 字节。
    //   2. 再把两个这样的 pair（np = 2q 和 2q+1）用 DPP row_ror:8 加选择重新组合：每条 16 字节
    //      的 store 指令覆盖 8 行完整的 128 字节，而不是 16 行、每行半条。
    // 为什么要写整行：实测全 GPU 同时写的时候，半条 cache line 的 store 要约 2.5 倍的时间才被
    // 确认；而 vmcnt 按顺序完成，下一个 tile 的 load 会排在这些 store 后面。
    // 第 M 行以后的写超出描述符的范围，会被丢弃；N 方向越界的块（N 不是 WNTILE 的倍数时）
    // 给一个超出范围的偏移。
    auto epilogue = [&](const TileOff& to) __attribute__((always_inline)) {
        // 用 volatile asm 现算 lane 号（v_mbcnt 数的是比自己编号低的 lane 有几个）：这样编译器
        // 无法把 epilogue 的地址计算提到 tile 循环外面，也不用让 threadIdx.x 一直活着。被提出去
        // 的值会跨整个 K 循环占着寄存器并 spill，而每次 spill 回读都会排空整个预取流水
        // （vmcnt(0)）。
        int lane;
        asm volatile("v_mbcnt_lo_u32_b32 %0, -1, 0\n\t"
                     "v_mbcnt_hi_u32_b32 %0, -1, %0"
                     : "=v"(lane));
        const int row16 = lane & 15, g = lane >> 4;
        // u32x4 表示 4 个 32 位无符号整数打包在一起，一共 16 B
        using u32x4 = __attribute__((ext_vector_type(4))) unsigned int;
        // pair(mi, np)：返回本 lane 在第 mi 个行块、第 np 个 N 子块对上的 16 字节
        // 整个tiel算完 256*8192 8192*256=256*256， 每个wave256*64, 就是16*4个16*16的子块；
        // 64个lane,分别保存16*4个16*16的子块的相同位置的1行4列个数，1*4*4=16B，是4个寄存器，64个子块就是256个寄存器用满了就
        // 然后是acc[16][4], mi16,n=4
        // （4 个 dword = 8 个 bf16），顺序是 {s0[0], s1[0], s0[1], s1[1]}。
        auto pair = [&](int mi, int np) __attribute__((always_inline)) {
            // AGPR → VGPR 的拷贝用 volatile asm 钉在这里：交给寄存器分配器的话，它会把全部
            // 256 个累加器的读取都提到 epilogue 的最前面，然后 spill。
            f32x4 x, y;
            // 是读数：这两行就是把两个累加器的值拷到 x、y，阻止编译器自把256个读取都提到最前一起做
            // 从wave看，acc[16][4]每个元素代表一个子块C 16*16,但是实际存储是对每个lane而言的，也就是acc每个元素是1行4列
            // 但是从lane，看，acc[16][4]只是一个f32*4,也就是本wave负责的某个子块的1行4列这种小块的连续4个，L%16,4*(L/16)
            // 
            asm volatile("" : "=v"(x) : "0"(acc[mi][2 * np]));       // 子块 2np
            asm volatile("" : "=v"(y) : "0"(acc[mi][2 * np + 1]));   // 子块 2np+1
            // 读完立刻清零，为下一个 tile 做准备；清零后再用 "+a" 的空 asm 让它对编译器不透明。
            // 直接写常量 0 的话，会被合并进 tile 循环里累加器的 phi 节点，寄存器分配器就会在它
            // 附近搬动（并 spill）累加器。
            acc[mi][2 * np] = acc[mi][2 * np + 1] = f32x4{0.f, 0.f, 0.f, 0.f};
            asm volatile("" : "+a"(acc[mi][2 * np]), "+a"(acc[mi][2 * np + 1]));
            // permlane是一个跨lane指令，交换
            // permlane16_swap(a, b)：把 a 的奇数 lane 组（16 个 lane 为一组）和 b 的偶数组互换；
            // 返回值 [0] 是交换后的 a，[1] 是交换后的 b。s0 处理每个 lane 的第 0、1 列，s1 处理
            // 第 2、3 列。
            // lane5时，x[0-3]是行5，0-3列；y[0-3]是行5，16-19列，lane 21时，x[0-3]是行5，4-7列；y[0-3]是行5，20-23列
            // 那交换是lane之间的交换，我们希望lane21的x0-3和lane5的y0-3互换，而让lane5 x0-3不和lane21的y0-3互换，这是参数顺序，硬件性质一起决定的
            const auto s0 = __builtin_amdgcn_permlane16_swap(
                pack_bf16x2(x[0], x[1]), pack_bf16x2(y[0], y[1]), false, false);
            const auto s1 = __builtin_amdgcn_permlane16_swap(
                pack_bf16x2(x[2], x[3]), pack_bf16x2(y[2], y[3]), false, false);
            // 结果：lane 组 g 拿到本行（相对这个子块对的）第 (g&1)*16 + (g>>1)*8 .. +7 列
            return u32x4{s0[0], s1[0], s0[1], s1[1]};
        };
        // 整个偏移都放在 voffset 里（soffset 为 0）：如果每个行块用一个 soffset，编译器会把
        // MSUB 个偏移提前算好，长期占着 SGPR。
        // NQ：每个行块有几组“两个子块对”（= 64 列 = 128 字节）。NSUB/2 是奇数时，最后一个子块
        // 对单独按每行 64 字节写。
        constexpr int NQ = NSUB / 4; // 1
        const bool lo8   = row16 < 8;   // 本 lane 是不是行块的前 8 行
        // 整行写法下本 lane 负责的列：后 8 行的 lane 负责第二个子块对（+32 列）
        //  本 lane 最终要写的那 8 列从第几列开始
        const int  ccol  = to.n0 + (row16 >> 3) * 32 + (g & 1) * 16 + (g >> 1) * 8;
        // 半行写法（单独的子块对）下本 lane 负责的列，
        const int  hcol  = to.n0 + (g & 1) * 16 + (g >> 1) * 8;// NSUB=6有用
        // 整行写法：va 写行块的第 0..7 行，本 lane 写其中第 (row16 & 7) 行
        int vofs_f = ((to.m0 + (row16 & 7)) * N + ccol) * 2;
        // 半行写法：本 lane 写第 row16 行
        int vofs_h = ((to.m0 + row16) * N + hcol) * 2; // NSUB=6有用
#pragma unroll
        for (int mi = 0; mi < MSUB; ++mi)   // 每次处理一个 16 行的行块
        {
            // 每个行块之前放一个调度屏障：不然调度器会把很多行块的 AGPR→VGPR 读取都提到前面。
            __builtin_amdgcn_sched_barrier(0);
#pragma unroll
            for (int q = 0; q < NQ; ++q) // NQ=1
            {
                // 两个子块对：v0 是第 64q .. 64q+31 列，v1 是第 64q+32 .. 64q+63 列
                // 对lane5而言就是v0为0-7列，v1为32-39列；
                // 回忆一下256*64，64个lane每个lane写32次16B，也就是1*8列，16*32为一组
                // 但是这样不利于合并访存，因此通过变化，16*32改为8*64，这样可以每行合并访存128B，
                // 那么需要做一下数据交换，也就是下面的__builtin_amdgcn_mov_dpp
                const u32x4 v0 = pair(mi, 2 * q), v1 = pair(mi, 2 * q + 1); // 出来就是b16了
                u32x4 va, vb;   // 上8行的每lane的8列数据， 下8行的每lane的8列数据，
#pragma unroll
                for (int i = 0; i < 4; ++i) // 每次dword只能取4B，而8列是16B，所以四次
                {
                    // DPP row_ror:8（0x128）：在 16 个 lane 组成的一行里循环右移 8 位，即 lane L
                    // 拿到 lane L^8 的值（第 r 行和第 r±8 行互相交换）。
                    const unsigned r0 = __builtin_amdgcn_mov_dpp(v0[i], 0x128, 0xF, 0xF, false);
                    const unsigned r1 = __builtin_amdgcn_mov_dpp(v1[i], 0x128, 0xF, 0xF, false);
                    // va：前 8 行的 lane 放自己的 v0（第 r 行的前 32 列），后 8 行的 lane 放第
                    //     r-8 行的 v1（后 32 列），合起来是第 0..7 行、每行 64 列 = 128 字节
                    // 变换之后就变成行块的第 0..7 行 / 第 8..15 行，每行 128 字节
                    va[i] = lo8 ? v0[i] : r1;
                    // vb：同理，拼出第 8..15 行
                    vb[i] = lo8 ? r0 : v1[i];
                }
                // 无条件发 store：K 循环的 vmcnt 记账假定每个 epilogue 恰好 EPI_STORES 条。
                // 越界（这 8 列超出 N）时给 0x7fffffff 这个偏移，被硬件丢弃。
                const bool in = ccol + q * 64 < N;
                __builtin_amdgcn_raw_buffer_store_b128(va, o_rsrc, in ? vofs_f + q * 128 : 0x7fffffff,
                                                       0, 0);
                __builtin_amdgcn_raw_buffer_store_b128(
                    vb, o_rsrc, in ? vofs_f + 8 * N * 2 + q * 128 : 0x7fffffff, 0, 0);
            }
            if constexpr ((NSUB / 2) % 2)
            {
                // NSUB = 6 时有 3 个子块对：最后一个单独写，每行 64 字节（半条 cache line）
                constexpr int np = NSUB / 2 - 1;
                __builtin_amdgcn_raw_buffer_store_b128(
                    pair(mi, np), o_rsrc, hcol + np * 32 < N ? vofs_h + np * 64 : 0x7fffffff, 0, 0);
            }
            vofs_f += 16 * N * 2;   // 移到下一个 16 行的行块
            vofs_h += 16 * N * 2;
        }
    };

    // ---- 主循环：软件流水 -------------------------------------------------------
    // stage s 的 A load 在 stage s-PF .. s-1 计算期间发出（B 是 s-PF_B），stage 编号跨 tile
    // 连续（stage ≥ nstages 表示下一个 tile 的第 stage - nstages 个）。循环按 RING 展开，让每个
    // LDS 槽和每套 B 寄存器都有固定的名字；nstages % RING == 0，所以每个 tile 都从 0 号槽开始。
    // step s 里的 barrier 同时做两件事：让各 wave 为 stage s+1 做的 DMA 对所有 wave 可见；保证
    // stage s-1 已经读完，step s+1 才会重填它。
    // 越过最后一个 tile 的预取照样发出（目标是夹到合法范围内的 tile），不加分支：条件发射会让
    // 每一套环形缓冲寄存器在整个循环里都活着，导致 spill。这些数据落进没人读的缓冲。
    // B 寄存器按槽声明了 RING 套，但同一时刻只有 PF_B+1 套是活的，寄存器分配器会把它们复用
    // 到 2（或 3）套物理寄存器上。【例】2 套，共 64 个 VGPR
    i32x4 bf[RING][NSUB][2]; // RING是4，4LDS流水
    int bs[RING][NSUB / 2];

    int id      = blockIdx.x;                 // 本工作组的第一个 tile
    TileOff cur = tile_off(id);               // 当前 tile
    TileOff nxt = tile_off(id + gridDim.x);   // 下一个 tile（流水会提前用到它）
    // 序幕：按“第 -PF .. -1 个 step”的样子发出，这样 vmcnt 记账和正常的 step 完全一致：
    // 第 -PF+u 个 step 发 A 的 stage u 和 B 的 stage u+PF_B-PF（是负数就跳过 B）。
    // 【例】PF_B = 1：u = 0 只发 A 的 stage 0；u = 1 发 B 的 stage 0 和 A 的 stage 1。
    static_for<PF>([&](auto u) __attribute__((always_inline)) {
        constexpr int sb = u + PF_B - PF;
        static_for<LOADS_PER_STAGE>([&](auto k) __attribute__((always_inline)) {
            if constexpr (decltype(k)::value >= KA || sb >= 0)  // 跳过 stage < 0 的 B 指令
                issue_one(k, cur, u, lds_buf(u), cur, sb, bf[sb < 0 ? 0 : sb],
                          bs[sb < 0 ? 0 : sb]);
        });
    });
    // stage 0 必须先落地，才能读它段 0 的片段；之后的 stage 都在上一个 step 的段 1 里读。
    // 等到最多只剩最后一个序幕 step 的 LOADS_PER_STAGE 条没完成，再做 barrier。
    wait_vm_lgkm0(std::integral_constant<int, LOADS_PER_STAGE * (PF - 1)>{});
    __syncthreads();
    // 读 stage 0 段 0 的 A 片段和 scale（相当于“第 -1 个 step 的段 1”要做的事）
    static_for<2 * MS0>([&](auto r) __attribute__((always_inline)) {
        afr0[r] = frag_at(smem0, r / MS0, r % MS0);
    });
    static_for<MS0 / 2>([&](auto p) __attribute__((always_inline)) { asc0[p] = scale_at(smem0, p); });
    bool after_epi = false;   // 当前 step 是否紧跟在 epilogue 后面
    for (; id < total; id += gridDim.x)   // persistent：一个接一个地处理 tile
    {
        for (int st = 0; st < nstages; st += RING)   // 每次展开 RING（= 4）个 step
        {
            static_for<RING>([&](auto u) __attribute__((always_inline)) {
                constexpr int nu  = (u + PF) % RING;     // 本 step 的 A DMA 写进哪个槽
                constexpr int nub = (u + PF_B) % RING;   // 本 step 的 B load 写进哪套寄存器
                const int nst     = st + u + PF;         // 本 step 发的 A 是第几个 stage
                const int nstb    = st + u + PF_B;       // 本 step 发的 B 是第几个 stage
                // 超出本 tile 的 stage 属于下一个 tile（这两个值在整个 wave 里都相同）
                const bool wrap   = nst >= nstages;
                const bool wrapb  = nstb >= nstages;
                // A DMA 的目标：本 tile 或下一个 tile 的 A / A scale 偏移
                const TileOff nto{0, 0, wrap ? nxt.a : cur.a, 0, wrap ? nxt.as : cur.as, 0};
                // B load 的目标：本 tile 或下一个 tile 的 B / B scale 偏移
                const TileOff ntob{0, 0, 0, wrapb ? nxt.b : cur.b, 0, wrapb ? nxt.bs : cur.bs};
                // 本 step：读槽 u（本 stage）和槽 u+1（下一个 stage，段 1 预读用），用第 u 套 B
                // 寄存器计算；把 A 发到槽 nu，把 B 发到第 nub 套寄存器。
                step(lds_buf(u), lds_buf(std::integral_constant<int, u + 1>{}), bf[u], bs[u], nto, wrap ? nst - nstages : nst,
                     lds_buf(std::integral_constant<int, nu>{}), ntob,
                     wrapb ? nstb - nstages : nstb, bf[nub], bs[nub],
                     u == 0 && st == 0 && after_epi, st + u);
            });
        }
        MBS_TICK(3)
        epilogue(cur);   // 写回当前 tile（此时下一个 tile 的前几个 stage 已经在路上了）
        MBS_TICK(4)
        after_epi = true;
        cur = nxt;                            // 换到下一个 tile
        nxt = tile_off(id + 2 * gridDim.x);   // 再下一个
    }
    // 退出前等所有全局访存完成：最后几个 step 还为“下一个 tile”（夹过的假 tile）发了 LDS
    // DMA，不能让它们在工作组结束、LDS 被回收之后才写进来。
    __builtin_amdgcn_s_waitcnt(WAIT_ALL);
#if MBS_TIMING
    // 每个 wave 的 lane 0 把自己的计时累加到全局数组
    if (lane == 0)
    {
        for (int i = 0; i < 8; ++i)
            atomicAdd(&mbs_timing[i], (unsigned long long)tacc[i]);
        atomicAdd(&mbs_timing[8], 1ull);
    }
#endif
}

}  // namespace

// =============================================================================
// 第三部分：主机端 launcher
// =============================================================================

// launch_wg<MS, NS>：启动 wg kernel。persistent：grid = min(tile 数, 256)，每个 CU 一个
// 工作组（每个 SIMD 1 个 wave）；block = 4 个 wave = 256 个线程。
template <int MS, int NS>
void launch_wg(const uint8_t* A, const uint8_t* B, const uint8_t* AScale,
               const uint8_t* BScale, void* out, int M, int N, int K, int scaleN_pad,
               hipStream_t stream)
{
    constexpr int WV = 4;   // 每个工作组的 wave 数
    // tile 数 = N 方向的 tile 数 × M 方向的 tile 数（工作组 tile 为 16*MS × 16*NS*4）
    const int tiles  = ((N + 16 * NS * WV - 1) / (16 * NS * WV)) * ((M + 16 * MS - 1) / (16 * MS));
    hipLaunchKernelGGL((gemm_a4w4_wg_kernel<MS, NS, WV>), dim3(std::min(tiles, 256)),
                       dim3(64 * WV), 0, stream, A, B, AScale, BScale,
                       reinterpret_cast<__hip_bfloat16*>(out), M, N, K, scaleN_pad);
}

// 一种工作组 tile：行数 mt、列数 nt，以及对应的启动函数。
struct WgTile
{
    int mt, nt;
    void (*launch)(const uint8_t*, const uint8_t*, const uint8_t*, const uint8_t*, void*,
                   int, int, int, int, hipStream_t);
};
// 全部 11 种工作组 tile（MTILE × NTILE = 16*MSUB × 64*NSUB）。更宽的 wave tile（NSUB = 6、8）
// 每条 MFMA 摊到的 load 和 LDS 读更少，也让启发式多了几种把 N 切到 CU 上的方式。
// MSUB*NSUB ≤ 64 保证累加器放得进 256 个 AGPR。
// 数组下标就是调试用的 MBS_TILE 编号：0 = 256×256，1 = 192×256，……，10 = 64×512。
constexpr WgTile kWgTiles[] = {
    {256, 256, launch_wg<16, 4>}, {192, 256, launch_wg<12, 4>}, {160, 256, launch_wg<10, 4>}, {128, 256, launch_wg<8, 4>},
    {96, 256, launch_wg<6, 4>},   {64, 256, launch_wg<4, 4>},
    {128, 384, launch_wg<8, 6>},  {96, 384, launch_wg<6, 6>},   {64, 384, launch_wg<4, 6>},
    {96, 512, launch_wg<6, 8>},   {64, 512, launch_wg<4, 8>},
};

// 对外入口（声明在 gemm_a4w4.h）：按问题规模选 kernel 和 tile，然后启动。
//   Mpad = M 向上取整到 32；scaleN_pad = K/32 向上取整到 8；stream 实际是 hipStream_t。
void gemm_a4w4_mxfp4_launch(const uint8_t* A,
                            const uint8_t* B,
                            const uint8_t* AScale,
                            const uint8_t* BScale,
                            void*          out,
                            int            M,
                            int            N,
                            int            K,
                            int            Mpad,
                            int            scaleN_pad,
                            void*          stream)
{
    // launch(msub, nsub)：启动单 wave kernel。grid 里的每个工作组是一个 wave（64 个线程），
    // 负责一块 (16*MSUB) × (16*NSUB) 的输出；grid.x 沿 N，grid.y 沿 M（按 Mpad 算）。
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

    // 工作组协作路径（A 经 LDS 共享、每个工作组 4 个 wave、persistent）。
    // tile 选择：让 ceil(tiles / NCU) * (MTILE + 64) * NTILE 最小，也就是“轮数 × 每轮代价”：
    // 每轮每个 CU 处理一个 tile；每轮的代价 = tile 面积 + 约 64 行的固定开销（序幕、epilogue，
    // 以及矮 tile 更低的 MFMA / load 比）。
    // 【例】M = 1024、N = 17408：256×256 有 272 个 tile，要 2 轮且第二轮只有 16 个 CU 在干活，
    // 代价 163840；160×256 有 476 个 tile，两轮都比较满，代价 114688，所以选它。
    if (K % 1024 == 0)
    {
        constexpr int NCU_WG = 256;   // CU 数（写死为 MI355X 的 256）
        int best = -1;
        long best_cost = 0, best_tiles = 0;
        for (int i = 0; i < (int)(sizeof(kWgTiles) / sizeof(kWgTiles[0])); ++i)
        {
            const int mt = kWgTiles[i].mt, nt = kWgTiles[i].nt;
            // 这种 tile 下的 tile 数
            const long tiles = (long)((M + mt - 1) / mt) * ((N + nt - 1) / nt);
            // 轮数 × 每轮代价
            const long cost  = (tiles + NCU_WG - 1) / NCU_WG * (long)(mt + 64) * nt;
            if (best < 0 || cost < best_cost)   // 代价相同时保留下标靠前的那个
            {
                best       = i;
                best_cost  = cost;
                best_tiles = tiles;
            }
        }
#ifdef MBS_FORCE_TILE_ENV
        // 调试构建：环境变量 MBS_TILE=<kWgTiles 的下标> 强制使用某个工作组 tile。
        if (const char* e = getenv("MBS_TILE"))
        {
            kWgTiles[atoi(e)].launch(A, B, AScale, BScale, out, M, N, K, scaleN_pad,
                                     reinterpret_cast<hipStream_t>(stream));
            return;
        }
#endif
        // tile 少到连 1/4 的 GPU（64 个 CU）都占不满时，改用单 wave kernel：它的小 tile 能
        // 提供更多的并行度。
        if (best_tiles >= NCU_WG / 4)
        {
            kWgTiles[best].launch(A, B, AScale, BScale, out, M, N, K, scaleN_pad,
                                  reinterpret_cast<hipStream_t>(stream));
            return;
        }
    }

    // 单 wave kernel 的 tile 启发式：优先用最大的 tile（复用最好），只要它的工作组数还能铺满
    // GPU（约每个 CU 2 个工作组）；否则逐级缩小 tile，换取并行度。阈值是在 gfx950（256 个 CU）
    // 上测出来的。
    constexpr int NCU        = 256;
    constexpr int FILL       = 2 * NCU;   // 目标最少工作组数：512
    // grid_wgs(mt, nt)：用 mt × nt 的 tile 时有多少个工作组
    auto grid_wgs = [&](int mt, int nt) {
        return (long)((Mpad + mt - 1) / mt) * ((N + nt - 1) / nt);
    };
    using m8 = std::integral_constant<int, 8>;
    using m4 = std::integral_constant<int, 4>;
    using m2 = std::integral_constant<int, 2>;
    using m1 = std::integral_constant<int, 1>;

    // 每个 wave 的 tile 越大，每个 K 步里互不依赖的 MFMA 越多，矩阵核心就越忙——真正的瓶颈
    // 是这个，而不是 occupancy。HUGE 128×128 每步 64 条 MFMA（256 个 AGPR 累加器）。大 tile
    // 铺不满 GPU 时就逐级缩小，用 ILP 换工作组的并行度。
    if (grid_wgs(128, 128) >= FILL)
        launch(m8{}, m8{});        // HUGE 128×128：MFMA 的 ILP 最深
    else if (grid_wgs(64, 128) >= FILL)
        launch(m4{}, m8{});        // TALL 64×128：每步 32 条 MFMA，适合小 M、大 N（如 M = 1024）
    else if (grid_wgs(64, 64) >= FILL)
        launch(m4{}, m4{});        // BIG 64×64
    else if (grid_wgs(32, 32) >= FILL)
        launch(m2{}, m2{});        // MID 32×32
    else
        launch(m1{}, m2{});        // SMALL 16×32：M 极小，尽量提高 occupancy
}
