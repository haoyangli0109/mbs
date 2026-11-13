// =============================================================================
// v0：GEMM 实验室的第一版 kernel。能跑对，每个选择都是最顺手的写法。
//
// 读法：先读这段总览，再按代码顺序往下读，每行代码上面的注释解释这一行。
// 配套文档是 docs/naive_gemm_lab.md：第 1 节讲结构，第 2 节是 profile 的结果。
//
// 文中的【例】统一取：
//   * 问题规模 M = N = 8192，K = 5120（gemm_lab 的默认值）；
//   * blockIdx = (x = 3, y = 5) 的那个 workgroup；
//   * tid = 165 的线程，也就是 wave 2 里的 lane 37。
//
// -----------------------------------------------------------------------------
// 算什么
// -----------------------------------------------------------------------------
//   C[M, N] = A[M, K] × B[N, K]^T，A、B 都是 MXFP4，输出 bf16。
//   每个 fp4 先乘上它所在的那 32 个 K 共享的 scale，再做乘加：
//     C[m][n] = Σ_k  A[m][k] × 2^(As[m][k/32] - 127)  ×  B[n][k] × 2^(Bs[n][k/32] - 127)
//   数据布局（common.h 里的 Problem）全部是普通行主序，不做任何 shuffle：
//     A  [M, K/2]   字节，每字节放 2 个 fp4（e2m1）    B  [N, K/2]   同样
//     As [M, K/32]  字节，每 32 个 K 一个 E8M0 scale   Bs [N, K/32]  同样
//     C  [M, N]     bf16
//   【例】A、B 每行 2560 字节、160 个 scale；C 是 8192 × 8192 个 bf16，共 128 MiB。
//
// -----------------------------------------------------------------------------
// 怎么分工
// -----------------------------------------------------------------------------
//   * 一个 workgroup 算 C 的一块 128 × 128（BM × BN）。【例】grid 是 64 × 64 = 4096 块。
//   * 一个 workgroup 有 256 个线程，也就是 4 个 wave，排成 2 × 2，每个 wave 算 64 × 64：
//
//                        N 0..63          N 64..127
//                   +----------------+----------------+
//      M 0..63      |     wave 0     |     wave 1     |
//                   |   wm=0, wn=0   |   wm=0, wn=1   |
//                   +----------------+----------------+
//      M 64..127    |     wave 2     |     wave 3     |
//                   |   wm=1, wn=0   |   wm=1, wn=1   |
//                   +----------------+----------------+
//
//   * 每个 wave 的 64 × 64 再切成 4 × 4 个 16 × 16 的块，每块每步 K 由一条 MFMA 来算：
//
//                 ni=0        ni=1        ni=2        ni=3
//      mi=0     acc[0][0]   acc[0][1]   acc[0][2]   acc[0][3]
//      mi=1     acc[1][0]   acc[1][1]   acc[1][2]   acc[1][3]
//      mi=2     acc[2][0]   acc[2][1]   acc[2][2]   acc[2][3]
//      mi=3     acc[3][0]   acc[3][1]   acc[3][2]   acc[3][3]
//
//     mi 管 M 方向（每块 16 行），ni 管 N 方向（每块 16 列）。
//   * K 方向每步走 BK = 128 个元素，正好是一条 MFMA 的 K。【例】K = 5120 一共 40 步；
//     每步每个 wave 发 16 条 MFMA，40 步共 640 条。
//
// -----------------------------------------------------------------------------
// 一步 K 做什么（k0 -> k0 + 128）
// -----------------------------------------------------------------------------
//   1. 拷贝：256 个线程一起把 A tile（128 行 × 64 字节 = 8 KB）和 B tile（8 KB）
//      从显存搬进 LDS，每个线程每次搬 4 字节，A、B 各搬 8 次。
//   2. __syncthreads()：等所有线程写完 LDS。
//   3. 计算：每个 wave 从 LDS 读自己要的 A、B 片段，从显存直接读 scale，发 16 条 MFMA。
//   4. __syncthreads()：等所有 wave 读完 LDS，下一步才能覆盖它。
//   40 步做完，把累加器转成 bf16 写回 C。
//
// -----------------------------------------------------------------------------
// 故意写得朴素的地方，以及 profile 的结论（docs/naive_gemm_lab.md 2.6 节）
// -----------------------------------------------------------------------------
//   * LDS 照搬显存的布局，每行 64 字节，不 padding、不 swizzle：读片段有 bank conflict，
//     但只占 wave 时间的 0.3%。
//   * scale 不进 LDS，要用时每个 lane 才从显存读 1 个字节：MFMA 前要等一次显存往返，
//     约占 23%；scale 的 load 发不出去，又占约 5%。
//   * 一步 K 就是“拷贝 -> barrier -> 计算 -> barrier”，没有双缓冲，也没有预取。
//   * 结果一次写 1 个 bf16（2 字节）：只占 0.3%。
//   * 最大的问题不在上面这几条里：拷贝循环被编译成了“每搬一次就等一次显存”，
//     一步 K 串行等 8 次，约占 45%。详见下面拷贝循环处的注释。
// =============================================================================
#pragma once  // 这个头文件只会被包含一次

#include "common.h"  // Problem（kernel 参数）、i32x4 / f32x4 向量类型、mfma_fp4（一条 MFMA 的包装）

// 每个版本放在自己的命名空间里：gemm_lab.hip 会同时 include v0、v1……，BM、launch 这些名字
// 就不会冲突。kernel 的符号名也因此带上了命名空间，编译后是 _ZN2v013gemm_v0_naiveE7Problem，
// make resources 和 rocprofv3 的输出里看到的就是它。
namespace v0 {

// tile 的大小：一个 workgroup 算 C 的 BM × BN = 128 × 128，K 方向每步走 BK = 128。
// BK 数的是 fp4 元素，不是字节：128 个 fp4 = 64 字节。BK = 128 正好等于一条
// v_mfma_scale_f32_16x16x128_f8f6f4 的 K，所以每个 16 × 16 的块每步只要一条 MFMA。
constexpr int BM = 128, BN = 128, BK = 128;  // BK 以 fp4 元素计
// tile 每行（某个 M 行或 N 行在这一步的 128 个 K）占的字节数：128 / 2 = 64。
constexpr int ROWB = BK / 2;                 // tile 每行的字节数（64）

// __global__：在 GPU 上执行、由 host 启动的函数，也就是 kernel。
// __launch_bounds__(256)：告诉编译器每个 workgroup 最多 256 个线程，编译器据此决定每个线程
//   最多能用多少寄存器。不写的话，编译器按最多 1024 个线程来估，寄存器预算会紧很多。
// Problem p 按值传：整个结构体放在 kernel 的参数区里，kernel 开头用标量指令（s_load）读进 SGPR。
__global__ __launch_bounds__(256) void gemm_v0_naive(Problem p)
{
    // LDS 是同一个 workgroup 的 256 个线程共享的片上内存，这里放这一步 K 的 A tile 和 B tile：
    //   sA：128 行 × 64 字节 = 8 KB，第 r 行是 A 的第 m0 + r 行在这一步的 128 个 K；
    //   sB：同样 8 KB，第 r 行是 B 的第 n0 + r 行，也就是 C 的第 n0 + r 列要用的数据。
    // 一共 16 KB（make resources 报的 LDS Size 16384）。只有一份，没有双缓冲。
    // 编译器把两个数组紧挨着放：sA 从第 0 字节开始，sB 从第 8192 字节开始。
    __shared__ uint8_t sA[BM * ROWB]; // 128 * 64 = 8192字节
    __shared__ uint8_t sB[BN * ROWB]; // 128 * 64 = 8192字节

    // tid：线程在 workgroup 里的编号，0–255。
    // lane：线程在自己 wave 里的编号，0–63；wave：第几个 wave，0–3。
    // 【例】tid = 165 -> lane = 37，wave = 2。
    const int tid = threadIdx.x, lane = tid % 64, wave = tid / 64;
    // 4 个 wave 排成 2 × 2（见开头的图）：wm 管上下（0 上、1 下），wn 管左右（0 左、1 右）。
    // 这个 wave 负责 tile 里第 wm×64 .. wm×64+63 行、第 wn×64 .. wn×64+63 列。
    // 【例】wave 2 -> wm = 1，wn = 0：tile 的第 64–127 行、第 0–63 列，即左下角。
    const int wm = wave / 2, wn = wave % 2;
    // r16 和 g 是这个 lane 在一条 MFMA 里的分工（common.h 的 mfma_fp4）。
    //
    //   输入：lane 交出第 r16 行的第 g 个 K 块，一块 32 个 fp4 = 16 字节，128 个 K 分 4 块。
    //   A、B 都这样交；B 的“行”就是 C 的列。
    //
    //                  g=0       g=1       g=2       g=3
    //                  K 0-31    K 32-63   K 64-95   K 96-127
    //      r16=0       lane 0    lane 16   lane 32   lane 48
    //      r16=1       lane 1    lane 17   lane 33   lane 49
    //      ...
    //      r16=15      lane 15   lane 31   lane 47   lane 63
    //
    //   输出：lane 拿回 16 × 16 结果块里第 4g .. 4g+3 行、第 r16 列，共 4 个 fp32。
    //   下表每行括号里的数字，是这一行 16 个 lane 拿的那 4 行结果：
    //
    //                  r16=0     r16=1     ...   r16=15
    //      g=0 (0-3)   lane 0    lane 1    ...   lane 15
    //      g=1 (4-7)   lane 16   lane 17   ...   lane 31
    //      g=2 (8-11)  lane 32   lane 33   ...   lane 47
    //      g=3 (12-15) lane 48   lane 49   ...   lane 63
    //
    // 【例】lane 37 -> r16 = 5，g = 2：交出第 5 行的 K 64–95，拿回第 8–11 行、第 5 列。
    const int r16 = lane % 16, g = lane / 16;
    // 这个 workgroup 负责的 C 块的左上角：第 m0 行、第 n0 列。
    // blockIdx.x 管 N 方向，blockIdx.y 管 M 方向（见最后 launch 里的 grid）。
    // 【例】blockIdx = (3, 5) -> m0 = 640，n0 = 384：算 C 的第 640–767 行、第 384–511 列。
    // block是64*64,每个block64算128*128的tile
    const int m0 = blockIdx.y * BM, n0 = blockIdx.x * BN;
    // KB：A、B 每行的字节数（K / 2）；KS：As、Bs 每行的 scale 个数（K / 32）。
    // 【例】KB = 2560，KS = 160。
    const int KB = p.K / 2, KS = p.K / 32;

    // 累加器。f32x4 是 4 个 float 组成的向量（common.h）。
    // 4 × 4 个块，每块 16 × 16 = 256 个 fp32，分给 64 个 lane，每个 lane 4 个；
    // 所以每个 lane 一共 16 × 4 = 64 个 fp32。编译器把它们放进了 64 个 AGPR
    // （make resources：AGPRs 64、VGPRs 73，每个 SIMD 只能驻留 3 个 wave）。
    // = {} 把它们全部清零。
    f32x4 acc[4][4] = {};
    // K 循环：k0 = 0, 128, 256, ..., 4992，共 K / BK 步。【例】40 步。
    // k0 数的是 fp4 元素：这一步的数据从每行第 k0 / 2 字节开始，scale 从第 k0 / 32 个开始。
    // 【例】k0 = 256：每行的第 128–191 字节，第 8–11 个 scale。
    for (int k0 = 0; k0 < p.K; k0 += BK) // 每步128个k,一共5120个k
    {
        // ---- 1. 拷贝：A tile、B tile 从显存搬进 LDS ----
        // 一个 tile 是 128 行 × 64 字节 = 2048 个 dword（4 字节），i 是 dword 的编号。
        // 256 个线程，每个搬 2048 / 256 = 8 个：i = tid, tid + 256, ..., tid + 1792。
        //
        // 这个循环是 v0 最大的问题（文档 2.2、4.1 节）。循环次数写成了依赖 tid 的形式，
        // 编译器没能证明每个线程都正好跑 8 次，于是没有展开，而是编成了 exec 循环：
        // 每个 lane 自己判断结束，全部结束才退出（s_andn2_b64 exec + s_cbranch_execnz）。
        // 每一圈都是“发 A、B 两条 load -> s_waitcnt vmcnt(0) 等它们回来 -> 写 LDS”，
        // 下一圈的 load 要到下一圈才发，一步 K 就串行等了 8 次显存往返，约占 wave 时间的 45%。
        // 怎么改见文档第 3 节的练习。
        // BM=128，ROW=64字节, i是dword的编号，每个线程发一个dword，然后，256个lane一组，发8次dword
        for (int i = tid; i < BM * ROWB / 4; i += 256) {
            // 第 i 个 dword 在 tile 里的位置。每行 ROWB / 4 = 16 个 dword：
            // r 是行（0–127），c 是行内第几个 dword（0–15，即第 4c .. 4c+3 字节）。
            // 第 j 圈 i = tid + 256j，所以 r = tid / 16 + 16j，c = tid % 16：
            // 每圈 256 个线程正好铺满 16 行，每行 16 个线程，一人 4 字节。
            //
            //                  c=0    c=1    ...    c=15
            //    r = 16j+0     t0     t1     ...    t15
            //    r = 16j+1     t16    t17    ...    t31
            //      ...
            //    r = 16j+15    t240   t241   ...    t255
            //
            // 【例】tid = 165 -> c = 5（第 20–23 字节），8 圈依次搬第 10、26、42、……、122 行。
            // ROWB是一行有多少dword
            const int r = i / (ROWB / 4), c = i % (ROWB / 4);
            // 搬 A：读 A 的第 m0 + r 行、这一步的第 c 个 dword（第 k0/2 + 4c 字节起的 4 个字节，
            // 即 8 个 fp4），写进 sA 的第 i 个 dword。sA 的第 i 个 dword 正好是第 r 行第 c 个，
            // 所以 LDS 里的排法和显存一样，只是行距从 KB 字节变成了 64 字节。
            //   reinterpret_cast<uint32_t*>(sA)[i]：把字节数组 sA 当成 uint32_t 数组，取第 i 个元素；
            //   *reinterpret_cast<const uint32_t*>(地址)：从这个字节地址一次读 4 个字节。
            // 这一句跨两行，汇编和 ATT 里的源码行号按物理行记：
            //   第一行（左边，写 sA）编成 s_waitcnt vmcnt(0) + ds_write2st64_b32，拷贝循环等显存的时间
            //   全记在这一行上（文档 2.5 节“等得最久的源码行”）；
            //   第二行（右边，读 A）编成地址运算 + global_load_dword，load 发不出去时的 stall 记在这一行上。
            // (size_t)：乘法按 64 位做，M、K 再大也不会溢出。用指针直接寻址，地址本来就是 64 位的，
            //   所以每条 load 前都有 2–3 条 64 位的地址运算（v_mad_i64_i32、v_lshl_add_u64），
            //   这是文档里的问题 3；换成 buffer load 的 32 位偏移就能省掉。
            // 合并访存：同一条 load 里，每 16 个 lane 读一行里连续的 64 字节，一个 wave 读 4 行。
            //   这 4 行在显存里相隔 KB 字节，每段只用到 128 字节 cache line 的一半，
            //   所以 vL1D 合并率只有 51.6%（文档 2.4 节）。
            // 【例】tid = 165、k0 = 0、第 0 圈：读 A 第 650 行的第 20–23 字节，
            //       写到 sA 的第 660–663 字节（第 10 行的第 20–23 字节）。
            reinterpret_cast<uint32_t*>(sA)[i] =
                *reinterpret_cast<const uint32_t*>(p.A + (size_t)(m0 + r) * KB + k0 / 2 + c * 4);
            // 搬 B：位置关系完全一样，读 B 的第 n0 + r 行，写进 sB 的第 i 个 dword。
            // sA[i] 和 sB[i] 在 LDS 里正好相差 8192 字节，编译器把两次写合成了一条 ds_write2st64_b32
            // （offset1:32，即 32 × 256 = 8192 字节），所以每圈只有一条写 LDS 的指令。
            reinterpret_cast<uint32_t*>(sB)[i] =
                *reinterpret_cast<const uint32_t*>(p.B + (size_t)(n0 + r) * KB + k0 / 2 + c * 4);
        }
        // ---- 2. 第一个 barrier ----
        // 等 256 个线程全部写完 LDS 才能开始读：一个 lane 要用的数据大多是别的线程搬的。
        // 【例】tid = 165 搬的是第 10、26、……行，可它在 MFMA 里要交出的 A 行是第 69、85、101、117 行；
        //       第 69 行是 tid 80–95（wave 1）在 j = 4 那一圈搬的。
        // 汇编里是 s_waitcnt lgkmcnt(0)（自己的 LDS 写都已完成）+ s_barrier（等其他 wave 也到这里）。
        __syncthreads();

        // ---- 3. 计算：每个 wave 发 4 × 4 = 16 条 MFMA ----
        // #pragma unroll：把紧跟着的循环完全展开。这两层循环的次数是常数 4，编译器能展开。
        // 展开以后，b、sb 只和 ni 有关，编译器只读 4 次而不是 16 次：一步 K 每个 wave 一共
        // 8 条 ds_read_b128（A、B 各 4 条）和 8 条 global_load_ubyte（As、Bs 各 4 条）。
#pragma unroll
        for (int mi = 0; mi < 4; ++mi) 
        // 128*128， 4*4个16*16
        {
            // 这个 lane 在第 mi 块里要交出的 A 行（tile 内的行号，0–127）：
            //   wm × 64（wave 在上半还是下半）+ mi × 16（第几块）+ r16（块内第几行）。
            // 【例】lane 37（wave 2，r16 = 5）、mi = 0 -> ra = 64 + 0 + 5 = 69，即 A 的第 640 + 69 = 709 行。
            const int ra = wm * 64 + mi * 16 + r16;
            // 从 LDS 读 A 片段：第 ra 行的第 g 个 K 块，16 字节（i32x4 是 4 个 int，16 字节 = 32 个 fp4），
            // 编成一条 ds_read_b128。
            // LDS 行距 64 字节、没有 swizzle：同一条指令里，同一个 g 的 16 个 lane 读 16 行的同一个位置，
            // 地址相隔 64 字节的整数倍，会撞到相同的 bank（bank conflict）。实测每条多约 4 个周期，
            // 只占 wave 时间的 0.3%，所以先不修（文档 2.4 节）。
            // 【例】lane 37、mi = 0：sA 第 69 × 64 + 2 × 16 = 4448 字节起的 16 字节，
            //       也就是 A 第 709 行的第 k0+64 .. k0+95 个元素。
            const i32x4 a = *reinterpret_cast<const i32x4*>(sA + ra * ROWB + g * 16);
            // A 的 scale：第 m0 + ra 行、第 k0/32 + g 个 32-K 块，1 字节的 E8M0（值为 2^(e-127)）。
            // 直接从显存读（global_load_ubyte），而且是在 barrier 之后、MFMA 之前才读，MFMA 只能等
            // 这一次显存往返：这是 v0 的第二大问题（文档问题 2）。
            // 一条指令里 64 个 lane 读 16 行 × 4 个相邻字节，16 行在显存里相隔 KS 字节，非常零散。
            // 读出来存成 int，MFMA 只用它的最低字节（common.h 里 opsel 填的是 0）。
            // 【例】lane 37、mi = 0、k0 = 0：As[709 × 160 + 0 + 2]，A 第 709 行 K 64–95 的 scale。
            const int sa = p.As[(size_t)(m0 + ra) * KS + k0 / 32 + g];
#pragma unroll
            for (int ni = 0; ni < 4; ++ni) {
                // 这个 lane 在第 ni 块里要交出的 B 行（tile 内 0–127），也就是 C 的列：
                //   wn × 64（wave 在左半还是右半）+ ni × 16 + r16。
                // 【例】lane 37（wave 2，wn = 0）、ni = 0 -> rb = 5，即 B 的第 384 + 5 = 389 行（C 的第 389 列）。
                const int rb = wn * 64 + ni * 16 + r16;
                // 从 LDS 读 B 片段：第 rb 行的第 g 个 K 块，16 字节，同样是一条 ds_read_b128。
                const i32x4 b = *reinterpret_cast<const i32x4*>(sB + rb * ROWB + g * 16);
                // B 的 scale：第 n0 + rb 行、第 k0/32 + g 块，同样从显存读 1 个字节。
                const int sb = p.Bs[(size_t)(n0 + rb) * KS + k0 / 32 + g];
                // 一条 v_mfma_scale_f32_16x16x128_f8f6f4，在第 (mi, ni) 块上累加这一步的 128 个 K：
                //   C[m][n] += Σ_k A[m][k] × scaleA × B[n][k] × scaleB（m、n 各取这一块的 16 行、16 列）。
                // 64 个 lane 各交出一行的 32 个 K 和它的 scale，硬件在 lane 之间交换数据，
                // 每个 lane 拿回 4 个结果（第 4g .. 4g+3 行、第 r16 列），累加进 acc[mi][ni]。
                // FP4 时一条 MFMA 大约占用矩阵核心 16 个周期。
                // ATT 里“MFMA 前等 scale 和 LDS 读”的时间记在 common.h:34（mfma_fp4 里的那条内建函数），
                // 它的内联链下一层就是这一行。
                acc[mi][ni] = mfma_fp4(a, b, acc[mi][ni], sa, sb);
            }
        }
        // ---- 4. 第二个 barrier ----
        // 等所有 wave 都读完 LDS，下一步的拷贝才能覆盖 sA、sB，否则快的 wave 会把别人还没读的数据冲掉。
        // LDS 只有一份，下一步的 load 只能等到这之后才发，读显存和算 MFMA 完全没有重叠
        // （没有预取，文档问题 4）。
        __syncthreads();
    }

    // ---- 收尾：累加器转成 bf16，写回 C ----
    // 三层循环全部展开，每个 lane 写 4 × 4 × 4 = 64 个 bf16，每个一条 2 字节的 store。
#pragma unroll
    for (int mi = 0; mi < 4; ++mi)
#pragma unroll
        for (int ni = 0; ni < 4; ++ni)
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                // acc[mi][ni][i] 在 C 里的行：m0 + wm × 64 + mi × 16，再加块内的行 4g + i
                // （MFMA 的输出排法：lane 拿第 4g .. 4g+3 行）。
                // 注意它和交 A 时的行 r16 不是一回事：输入按 r16 分行，输出按 4g + i 分行。
                // 【例】lane 37（g = 2）、mi = 0：m = 640 + 64 + 0 + 8 + i，即第 712–715 行。
                const int m = m0 + wm * 64 + mi * 16 + 4 * g + i;
                // 列：n0 + wn × 64 + ni × 16，再加块内的列 r16。
                // 【例】lane 37（r16 = 5）、ni = 0：n = 384 + 0 + 0 + 5 = 389。
                const int n = n0 + wn * 64 + ni * 16 + r16;
                // fp32 就近舍入成 bf16，写 2 个字节。同一条 store 里，每 16 个 lane 写一行里连续的 16 列
                // （32 字节），一个 wave 写 4 行。L2 会把这些小块的写合并起来，写回显存的总量正好等于
                // C 的大小，只是指令多：每个 wave 64 条（文档问题 8，只占 0.3%）。
                p.C[(size_t)m * p.N + n] = __float2bfloat16(acc[mi][ni][i]);
            }
}

// host 端的启动函数，gemm_lab.hip 的 kVersions 表里登记的就是它。
// inline：函数定义写在头文件里，加 inline 才不会在多个源文件都包含它时重复定义。
inline void launch(const Problem& p, hipStream_t s)
{
    // <<<grid, 每个 workgroup 的线程数, 动态 LDS 字节数, stream>>>：
    //   grid = (N / 128, M / 128)：blockIdx.x 管 N 方向，blockIdx.y 管 M 方向。【例】64 × 64 = 4096 个 workgroup。
    //   256：每个 workgroup 的线程数，和 __launch_bounds__(256) 对应。
    //   0：不用动态 LDS（sA、sB 是静态声明的）。
    // kernel 里没有任何边界检查，M、N、K 都必须是 128 的倍数，由 gemm_lab.hip 在启动前检查。
    // 这不是常驻型 kernel：每个 CU 同时最多放 3 个 workgroup，256 个 CU 一次放 768 个，
    // 做完一个，硬件再派下一个。也没有做 MBS 那种按 XCD / L2 的 tile 编号重排。
    gemm_v0_naive<<<dim3(p.N / BN, p.M / BM), 256, 0, s>>>(p);
}

}  // namespace v0
