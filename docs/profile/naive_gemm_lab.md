# GEMM 实验室：从一个 naive kernel 开始，用 profile 一步步找问题

这个实验室和 MBS 算同一道题：MXFP4 A4W4 的 GEMM，M = N = 8192，K = 5120，输出 bf16。区别是从一个“最普通的写法”出发，每一轮按 [`profiling_guide.md`](./profiling_guide.md) 的步骤 profile 一遍，找出最大的问题，只改这一处，再测。这份记录每做完一轮就往后追加一节。

代码在 `mbs/lab/` 下：

| 文件 | 内容 |
|---|---|
| `common.h` | 问题的数据布局（全部普通行主序，A、B、scale 都不做 shuffle）、MFMA 包装函数 |
| `v0_naive.h` | 第一版 kernel |
| `gemm_lab.hip` | 测试程序：生成随机数据、CPU 抽样对拍、稳态计时；新版本在 `kVersions` 里登记 |
| `Makefile` | `make`、`make asm`（生成带行号的 `.s`）、`make src`（汇编和源码行交错排列）、`make resources`（寄存器和占用率报告） |
| `pmc_lab.txt` | 四轮硬件计数器 |
| `att_summary.py` | 通用的 ATT 汇总：每个 SIMD 的 MFMA 忙闲、wave 时间的分配、等待最久的源码行 |
| `att_dump.py` | 把某个 wave 某一步 K 的原始 ATT 记录和汇编、源码行并排打出来 |
| `draw_lab_figs.py` | 画每一版的 wave 时间分配图和 SIMD 时间线 |

## 0. 怎么用

```bash
cd mbs/lab
make                                    # 生成 build/gemm_lab
export HIP_VISIBLE_DEVICES=4            # 这台机器上对应 amd-smi 的 7 号卡
build/gemm_lab v0                       # 对拍 1024 个点，热身 500 ms，计时 200 次
build/gemm_lab v0 --iters 1000 --warmup-ms 1000 --check 0
```

**对拍怎么算。** 随机抽 1024 个输出点，在 CPU 上用 double 重算，误差除以这个点各 32 元素块乘积的平方和开方（块和范数）。合格线是 3e-2；bf16 的舍入误差在 1e-2 以内，写错的 kernel 误差在 1 左右。

**计时怎么算。** 先连续跑 `--warmup-ms` 毫秒，让时钟稳定下来，再用 hipEvent 计 `--iters` 次求平均（原因见 `profiling_guide.md` 3.1 节）。

profile 时，被测命令用 `--iters 2 --warmup-ms 0 --check 0`，算上对拍那一次一共 3 个 dispatch。rocprofv3 会在当前目录留下 `.rocprofv3/` 临时目录，所以要换到仓库外面去跑：

```bash
mkdir -p /tmp/lab_prof && cd /tmp/lab_prof
LAB=/home/haoyanli/a121_quark/mbs/lab
EXE=$LAB/build/gemm_lab
# 这次实验用的解码器和 rocprof-compute 都在 /tmp 下（安装方法见 profiling_guide.md 2.2、2.3），/tmp 可能被清掉
DEC=/tmp/mbs_check/decoder/rocprof-trace-decoder-manylinux-2.28-0.1.6-Linux/opt/rocm/lib
RPC="/tmp/rpc/venv/bin/python /tmp/rpc/root/opt/rocm-7.2.3/bin/rocprof-compute"

rocprofv3 --kernel-trace --stats --output-format csv -d v0_kt -o kt -- $EXE v0 --iters 2 --warmup-ms 0 --check 0
rocprofv3 -i $LAB/pmc_lab.txt --kernel-include-regex gemm_v0 --output-format csv -d v0_pmc -o pmc \
    -- $EXE v0 --iters 2 --warmup-ms 0 --check 0
python3 $LAB/../scripts/pmc_summary.py v0_pmc 8192 8192 5120
$RPC profile --name v0 --no-roof -k gemm_v0 -p /tmp/lab_prof/rpc_v0 -- $EXE v0 --iters 2 --warmup-ms 0 --check 0
$RPC analyze -p /tmp/lab_prof/rpc_v0 -b 2 7 11 12 16 17
rocprofv3 --att --att-library-path $DEC --kernel-include-regex gemm_v0 --kernel-iteration-range 2 \
    -d v0_att -- $EXE v0 --iters 2 --warmup-ms 0 --check 0
python3 $LAB/att_summary.py v0_att/ui_output_agent_*_dispatch_* $LAB/v0_naive.h $LAB/common.h
python3 $LAB/draw_lab_figs.py v0 v0_att/ui_output_agent_*_dispatch_*
```

这个测试程序不经过 Python，所以 profile 很快：完整的 rocprof-compute 只要 33 秒，ATT 不到 2 秒。

**加一个新版本：**

1. 把 `v0_naive.h` 复制成 `v1_xxx.h`，改掉命名空间和 kernel 名（比如 `gemm_v1_xxx`）；
2. 在 `gemm_lab.hip` 里 `#include` 它，并在 `kVersions` 里加一行；
3. 把上面命令里的 `v0` 换成 `v1`，重跑一遍。

**每一轮的规矩：**

1. 测时间，并换算成周期；
2. 编译期体检；
3. 采计数器；
4. 跑 ATT；
5. 根据证据提出一个假设，预测改完能快多少；
6. 只改一处；
7. 对拍通过后回到第 1 步。

## 1. v0：一个能跑对的 naive 版本

下面是简化过的结构，完整代码见 `lab/v0_naive.h`：

```cpp
constexpr int BM = 128, BN = 128, BK = 128;  // BK 以 fp4 元素计
constexpr int ROWB = BK / 2;                 // tile 每行的字节数（64）

__global__ __launch_bounds__(256) void gemm_v0_naive(Problem p)
{
    __shared__ uint8_t sA[BM * ROWB];
    __shared__ uint8_t sB[BN * ROWB];
    ...
    for (int k0 = 0; k0 < p.K; k0 += BK) {
        for (int i = tid; i < BM * ROWB / 4; i += 256) {          // 拷贝 A、B tile，每次 4 字节
            ...sA[i] = A[...];  ...sB[i] = B[...];
        }
        __syncthreads();
        for (int mi = 0; mi < 4; ++mi) {                          // 每个 wave 4 x 4 条 MFMA
            a  = LDS 里 A 的片段（行距 64 字节，不做 swizzle）;
            sa = p.As[...];                                       // scale 直接从显存读，每 lane 1 字节
            for (int ni = 0; ni < 4; ++ni) {
                b  = LDS 里 B 的片段;  sb = p.Bs[...];
                acc[mi][ni] = mfma_fp4(a, b, acc[mi][ni], sa, sb);
            }
        }
        __syncthreads();
    }
    // 每个 lane 写 64 个 bf16，一次 2 字节
}
```

每个 workgroup 算 128×128 的输出，256 个线程分成 4 个 wave，排成 2×2，每个 wave 算 64×64，也就是 4×4 个 16×16 的 MFMA 块。K 方向每步走 128，正好是一条 MFMA 的 K，所以一步里每个 wave 发 16 条 MFMA，一共走 40 步。grid 是 64×64 = 4,096 个 workgroup，不是常驻型 kernel。

这些都是第一版最自然的写法：LDS 照搬显存的布局，scale 要用的时候才去读，一步 K 就是“拷贝 → barrier → 计算 → barrier”，结果算完直接写出去。

> 练习：先别往下看。你觉得这几个写法里，哪个最拖后腿？把你的排序记下来，看完 profile 再对答案。

## 2. 第 0 轮：v0 慢在哪

### 2.1 测时间，并换算成周期

```text
v0   M=8192 N=8192 K=5120     1018.4 us    674.8 TFLOPS  (1000 launches after 1000 ms warm-up)
v0   M=8192 N=8192 K=5120     1019.5 us    674.1 TFLOPS
v0   M=8192 N=8192 K=5120     1019.1 us    674.3 TFLOPS
```

v0 每次 **1,019 µs，674 TFLOPS**，MBS 是 179 µs，看起来慢 5.7 倍。但连续跑时 amd-smi 显示，v0 只用了约 740 W，时钟一直在 2.19 GHz；MBS 顶着 1000 W 的功耗上限，只有 1.49 GHz。换成周期数：

- v0：1,019 µs × 2.19 GHz ≈ 223 万周期；
- MBS：179 µs × 1.49 GHz ≈ 26.7 万周期。

所以 v0 实际上慢了 **8.4 倍**。只比时间会低估差距：v0 做的有效工作少，耗电少，时钟反而跑得高。

### 2.2 编译期体检

```bash
make resources
```

```text
Function Name: _ZN2v013gemm_v0_naiveE7Problem
    TotalSGPRs: 28
    VGPRs: 73
    AGPRs: 64
    ScratchSize [bytes/lane]: 0
    Occupancy [waves/SIMD]: 3
    VGPRs Spill: 0
    LDS Size [bytes/block]: 16384
```

编译器把 64 个累加用的 float 放进了 AGPR，加上 73 个 VGPR，每个 SIMD 只能驻留 3 个 wave。也就是说，每个 CU 同时跑 3 个 workgroup；LDS 只用了 16 KB，不是限制因素。没有 spill。

再看汇编（`make asm`，主循环在 `.LBB0_5`）。下面是一步 K 里最关键的几段：

```text
.LBB0_6:                                ; 拷贝循环，跑 8 次
	v_mad_i64_i32 v[46:47], s[10:11], v43, s18, v[24:25]
	...
	global_load_dword v48, v[46:47], off          ; A 的 4 字节
	...
	global_load_dword v0, v[46:47], off           ; B 的 4 字节
	...
	s_waitcnt vmcnt(0)                            ; 等这两条 load 回来
	ds_write2st64_b32 v42, v48, v0 offset1:32     ; 写进 LDS
	...
	s_cbranch_execnz .LBB0_6
	...
	s_waitcnt lgkmcnt(0)
	s_barrier                                     ; 第一个 __syncthreads
	global_load_ubyte v0, v[26:27], off           ; scale，每个 lane 1 字节，共 8 条
	...
	ds_read_b128 v[24:27], v34                    ; A、B 片段，共 8 条
	...
	s_waitcnt vmcnt(4) lgkmcnt(3)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[24:27], v[42:45], a[0:3], v0, v62 ...
	...                                           ; 共 16 条 MFMA，中间夹着等待
	s_waitcnt lgkmcnt(0)
	s_barrier                                     ; 第二个 __syncthreads
```

> 练习：只看汇编，你能找出几个问题？

能看出来的：

1. **拷贝循环没有展开。** 循环次数取决于 `tid`，编译器没法确定所有线程跑一样多次，只好按“每个线程各自循环”来编（`s_cbranch_execnz`）。结果每一次循环都是“发 2 条 load → `s_waitcnt vmcnt(0)` → 写 LDS”，一步 K 要**串行等 8 次**全局内存。这一条不在我刻意设计的毛病里，是编译器的行为造成的，只有看汇编或者 profile 才会发现。
2. **scale 在 barrier 之后才去读。** 8 条 `global_load_ubyte`，MFMA 只能等它们回来，于是每步 K 又多一次往返。
3. **算地址用的是 64 位运算**（`v_mad_i64_i32`、`v_lshl_add_u64`），每条 load 前面都有好几条 VALU。
4. **每步 K 有两个 barrier。**

### 2.3 算上限

- **MFMA**：和 MBS 一样，每个 SIMD 要算 10,240 条 MFMA，每条 16 周期，最少 163,840 周期。v0 用了 223 万周期，所以 MFMA 管线最多只忙了 7.3%。
- **访存**：每个 workgroup 每步 K 要从 L2 读 16 KB 的 A、B（scale 另算），做 2 × 128 × 128 × 128 = 4.19 MFLOP，算术强度约 256 FLOP/B，正好在 L2 的拐点（257）附近。这说明，即使其他问题全修好，128×128 的 tile 最终也会被 L2 带宽卡住。这是以后的事，现在离那里还很远。

### 2.4 硬件计数器

`pmc_summary.py` 的输出（整块 GPU，一次 dispatch）：

| 计数器 | v0 | 怎么读 |
|---|---:|---|
| `MfmaUtil` | 7.3% | MBS 是 55% |
| 每个 wave 的寿命 | 374,492 周期 | `SQ_WAVE_CYCLES` × 4 ÷ 16,384 个 wave |
| `SQ_WAIT_ANY` × 4 ÷ wave 数 | 286,174 周期 | 每个 wave 有 **76%** 的时间在等依赖 |
| `SQ_WAIT_INST_ANY` × 4 ÷ wave 数 | 50,768 周期 | 13.6% 是指令备好了、执行单元接不住 |
| `SQ_INSTS_VMEM_RD` ÷ wave 数 | 960 | = 40 步 × 24 条，和汇编一致 |
| `SQ_INSTS_VALU` ÷ wave 数 | 4,847 | 是 MFMA 条数（640）的 7.6 倍，大多在算地址 |
| `SQ_INSTS_LDS` ÷ wave 数 | 640 | 40 步 × 16 条（8 条写、8 条读） |
| `SQ_LDS_BANK_CONFLICT` | 20,971,520 周期 | ÷ 524 万条 `ds_read_b128` = 每条多 **4 个周期** |
| L2 命中率 | 96.0% | 数据都在 L2 里 |
| 写请求 × 64 B | 128 MiB | 正好等于 C：一次写 2 字节在 L2 里合并了，没有多写 |

rocprof-compute 给出的同一件事：

| 编号 | 指标 | v0 | MBS |
|---|---|---:|---:|
| 2.1.12 | MFMA Utilization | 7.25% | 56.5% |
| 7.2.4–7.2.6 | Dependency Wait / Issue Wait / Active | 77% / 13% / 10% | 14% / 29% / 57% |
| 2.1.16 | IPC | 0.25 | 0.41 |
| 12.2.9 | LDS Bank Conflicts/Access | 0.5 | 0 |
| 16.1.3 | vL1D Coalescing | 51.6% | |
| 2.1.22 | L2 Hit Rate | 96% | 77% |
| 2.1.23 | L2 Cache BW | 29% of peak | 29% of peak |
| 17.2.9 | L2 Read Latency | 623 周期 | |

> 练习：bank conflict 确实存在（每条 `ds_read_b128` 多 4 个周期）。它值得先修吗？

不值得。每个 wave 一步 K 读 8 条，40 步一共 320 条，多出来 1,280 周期，只占它寿命的 **0.3%**。而“等依赖”占了 76%。计数器告诉我们：wave 绝大部分时间在等访存结果，但带宽并没有用满（L2 只用了 29%，HBM 几乎没用），所以问题出在**延迟没被藏住**，而不是带宽不够。

### 2.5 ATT：等的是哪一行

```bash
python3 att_summary.py v0_att/ui_output_agent_*_dispatch_* v0_naive.h common.h
```

```text
1. 每个 SIMD 的 MFMA 管线
   SIMD 0: 16 个 wave（最多同时 3 个），10,240 条 MFMA，活跃 2,167,960 周期，MFMA 忙 7.6%
   ...
2. wave 的时间花在哪（64 个 wave 合计，平均每个 wave 378,261 周期）
   等 vmcnt（全局内存）                  169,400   44.8%
   等 vmcnt + lgkmcnt               88,437   23.4%
   发全局读                            38,008   10.0%
   向量指令                            28,789    7.6%
   空档（没有指令可发）                      15,948    4.2%
   等 barrier                       11,256    3.0%
   MFMA                             9,248    2.4%
   ...
3. 等得最久的 s_waitcnt / s_barrier（按源码行）
   等 vmcnt（全局内存）          v0_naive.h:193      169,400 周期/wave  reinterpret_cast<uint32_t*>(sA)[i] =
   等 vmcnt + lgkmcnt      common.h:34        88,437 周期/wave  return __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(...
   等 barrier              amd_device_functions.h:692       11,256 周期/wave
4. stall 最多的其他指令（按源码行）
   v0_naive.h:194       11,926 周期/wave  global_load_dword ...   *reinterpret_cast<const uint32_t*>(p.A + ...
   v0_naive.h:232       11,879 周期/wave  global_load_ubyte ...   const int sa = p.As[...];
   v0_naive.h:242        8,361 周期/wave  global_load_ubyte ...   const int sb = p.Bs[...];
   v0_naive.h:199        8,336 周期/wave  global_load_dword ...   *reinterpret_cast<const uint32_t*>(p.B + ...
```

![图 1：v0 的 wave 时间分配](./images/lab_v0_wave_time.svg)

图 1：v0 每个 wave 的时间分配。紫色是在等全局内存，深灰是发全局读，蓝色的 MFMA 只占 2.4%。

![图 2：v0 在 SIMD 0 上的时间线](./images/lab_v0_timeline.svg)

图 2：SIMD 0 上约两步 K 的时间线。最上面是 MFMA 管线，浅红是空着；下面三行是同时驻留的 3 个 wave。每一步 K 都先有一串等全局内存的“梳齿”，也就是拷贝循环的 8 次往返；接着是一大段 MFMA 前的等待。3 个 wave 轮流在等，MFMA 管线大部分时间空着。

把一个 wave 的 37.8 万周期除以 40 步，每步 K 约 9,460 周期：

| 一步 K 里 | 周期 | 来源 |
|---|---:|---|
| 拷贝循环里等 load（8 次 × 约 530） | 4,235 | 193 行，和 L2 读延迟 623 周期吻合 |
| MFMA 前等 scale 和 LDS 读 | 2,211 | scale 是 1 字节 load，每条要访问 16 行 |
| 发全局读（含被堵） | 950 | scale load 和 dword load 发不出去 |
| 向量指令（多半是 64 位地址运算） | 720 | |
| 空档 | 399 | |
| barrier | 281 | |
| 16 条 MFMA 自己 | 231 | 真正干活的部分 |

### 2.6 问题清单，按 profile 排序

| # | 问题 | 证据 | 占 wave 时间 | 怎么修 |
|---|---|---|---:|---|
| 1 | 拷贝循环串行，每次 load 都等 | 汇编里循环体内有 `vmcnt(0)`；ATT 193 行 | 45% | 让 16 条 load 一起发出去，只等一次 |
| 2 | scale 在 barrier 后才从显存读，MFMA 前等它 | ATT `common.h:34`；232、242 行发射被堵 | 23% + 约 5% | 和 tile 一起提前读，存进 LDS 或寄存器；或者换成 shuffle 过的 scale 布局，一条 dword 读完 |
| 3 | 地址用 64 位算，VALU 很多 | 每个 wave 4,847 条 VALU；ATT 7.6% | 7.6% | 用 32 位偏移（buffer load 的 voffset） |
| 4 | 每步 K 两个 barrier，没有预取 | ATT barrier 3%；剩下的延迟都暴露着 | 3% 以上 | 双缓冲：算第 k 步时，发第 k+1 步的 load |
| 5 | load 太窄（dword、ubyte），合并率 51% | `SQ_INSTS_VMEM_RD` 960；vL1D coalescing 51.6% | 算在第 2、3 条里 | dwordx4；直接写进 LDS（`buffer_load ... lds`） |
| 6 | LDS bank conflict（4 路） | 每条 `ds_read_b128` 多 4 个周期 | 0.3% | XOR swizzle，等 LDS 读变成瓶颈时再修 |
| 7 | tile 小，每步每个 wave 只有 16 条 MFMA | 算术强度约 256 FLOP/B，在 L2 拐点上 | 以后会卡 | 更大的 tile、更大的 BK |
| 8 | 结果一次写 2 字节 | 每个 wave 64 条 store | 0.3% | 打包成 16 字节再写 |

对一下你在第 1 节的排序：我刻意埋的 bank conflict 只占 0.3%；最大的问题（拷贝循环串行）根本不在我想埋的清单里，是编译器把循环编成了“每次都等”。这正是要先 profile 再动手的原因：凭直觉改，很可能改的是 0.3% 那一项。

## 3. 下一轮：先修第 1 条

**假设。** 只要让一步 K 的 16 条 load 同时在路上、只等一次，这一步的等待就能从 8 次往返缩短到 1 次左右。拷贝循环里那 4,235 周期会降到 600–900 周期，一步 K 从约 9,460 周期降到约 6,000 周期，时间从 1,019 µs 降到约 650 µs。

**改完要验证这些：**

- 时间和 TFLOPS；
- ATT 里 193 行的等待是否变成一次；
- 汇编里拷贝部分是否变成“16 条 load → 一次 `s_waitcnt` → 8 条写 LDS”。

**预测下一轮的瓶颈。** 如果这个假设成立，第 2 条（MFMA 前等 scale）会成为最大的问题。

> 练习：在 `v0_naive.h` 里只改拷贝循环，让编译器能把它展开。提示：循环次数要在编译时就确定，比如写成 `for (int j = 0; j < 8; ++j) { int i = tid + j * 256; ... }`，再加 `#pragma unroll`。

## 4. 细读：v0 的汇编、计数器和 ATT 各自怎么看

三种数据回答的问题不一样：

| 数据 | 回答 | 看不到 |
|---|---|---|
| 汇编 | 编译器实际生成了什么指令、哪里会等 | 每处等多久 |
| 计数器 | 整个 kernel 的时间大体花在哪类事情上，各个部件忙到几成 | 是哪一行、哪一条指令 |
| ATT | 某个 wave 的每一条指令什么时候发、等了多久 | 只追了一个 CU，要用计数器确认它有代表性 |

一般按“汇编 → 计数器 → ATT”的顺序读，三者互相核对。

### 4.1 汇编

**找到循环。** `make asm` 生成的 `.s` 里，从 `_ZN2v013gemm_v0_naiveE7Problem:` 到 `.Lfunc_end0` 是这个 kernel。编译器会在每个循环头上写注释，循环的结尾是往回跳的那条分支：

```text
.LBB0_5:                                ; =>This Loop Header: Depth=1      ← K 循环
.LBB0_6:                                ;   Parent Loop BB0_5 Depth=1      ← 套在里面的拷贝循环
	s_cbranch_execnz .LBB0_6                                            ← 拷贝循环的结尾
	s_cbranch_scc0 .LBB0_5                                              ← K 循环的结尾
```

**指令名怎么拆。**

| 部分 | 例子 | 含义 |
|---|---|---|
| 前缀 | `s_` | 标量指令，整个 wave 只算一份 |
| | `v_` | 向量指令，64 个 lane 各算一份 |
| | `global_`、`buffer_` | 读写显存 |
| | `ds_` | 读写 LDS |
| | `v_mfma` | 矩阵指令 |
| 后缀 | `_ubyte`、`_dword`、`_dwordx4` | 每个 lane 读 1、4、16 字节 |
| | `_b32`、`_b128` | LDS 读写的宽度：4、16 字节 |
| | `_e32`、`_e64` | 指令编码的长度，不影响含义 |
| 寄存器 | `s18`、`v43`、`a[0:3]` | SGPR、VGPR、AGPR；`v[46:47]` 是两个相邻寄存器拼成 64 位 |

**逐行读拷贝循环。**

```text
.LBB0_6:
	v_mad_i64_i32 v[46:47], s[10:11], v43, s18, v[24:25]  ; 64 位：行号 r (v43) × 每行字节数 K/2 (s18) + A 的基址 (v[24:25])
	v_and_b32_e32 v0, 60, v42                              ; 列偏移 (i % 16) × 4，从 LDS 地址 v42 = 4i 里截出来
	v_lshl_add_u64 v[46:47], v[46:47], 0, v[0:1]          ; 64 位加法，得到这个 lane 要读的地址
	global_load_dword v48, v[46:47], off                   ; 读 A 的 4 字节到 v48
	v_mad_i64_i32 v[46:47], s[10:11], v44, s18, v[26:27]  ; B 的地址，同上
	v_lshl_add_u64 v[46:47], v[46:47], 0, v[0:1]
	global_load_dword v0, v[46:47], off                    ; 读 B 的 4 字节到 v0
	v_add_u32_e32 v45, 0x100, v45                          ; i += 256
	v_cmp_lt_u32_e32 vcc, s8, v45                          ; 哪些 lane 的 i 已经超过 2047
	v_add_u32_e32 v44, 16, v44                             ; 行号 += 16
	v_add_u32_e32 v43, 16, v43
	s_or_b64 s[2:3], vcc, s[2:3]                           ; 记下已经做完的 lane
	s_waitcnt vmcnt(0)                                     ; ← 等刚发的两条 load 都回来
	ds_write2st64_b32 v42, v48, v0 offset1:32              ; v48 写到 sA[i]，v0 写到往后 8192 字节处，也就是 sB[i]
	v_add_u32_e32 v42, 0x400, v42                          ; LDS 地址 += 1024
	s_andn2_b64 exec, exec, s[2:3]                         ; 关掉做完的 lane
	s_cbranch_execnz .LBB0_6                               ; 还有 lane 没做完，就再来一圈
```

从这段能读出三件事：

1. **这是一个 exec 循环。** `s_andn2_b64 exec …` 加 `s_cbranch_execnz` 是“每个 lane 的循环次数可能不同”时的编法：每个 lane 自己判断结束，全部结束才退出。v0 的循环写成 `for (int i = tid; i < 2048; i += 256)`，次数依赖 `tid`，编译器证明不了所有 lane 都正好跑 8 次，只能这样编，也就没法展开。
2. **循环里有 `s_waitcnt vmcnt(0)`。** 写 LDS 要用 load 回来的数据，所以必须等；下一圈的 load 要到下一圈才发，于是每一圈都要完整地等一次往返，8 圈就是 8 次。
3. **地址是 64 位算的。** 每条 load 前面有 2–3 条 VALU 在做 64 位运算。

**等待计数器。** 每个 wave 有两个计数器，记录“发出去还没完成的访存”：

- `vmcnt`：显存读写（`global_`、`buffer_`），读按发出的顺序完成；
- `lgkmcnt`：LDS 操作、标量读（`s_load`）等，LDS 操作也按顺序完成。

`s_waitcnt vmcnt(N)` 的意思是“等到最多只剩 N 条没完成”。因为按顺序完成，等到只剩 N 条，就等于更早发的都回来了。计算段里的例子：

```text
	global_load_ubyte v0  ...       ; 第 1 条：A scale（mi 0）
	global_load_ubyte v62 ...       ; 第 2 条：B scale（ni 0）
	global_load_ubyte v63 ...       ; 第 3–6 条：其余的 scale
	...
	ds_read_b128 v[24:27] ...       ; 第 1 条：A 片段（mi 0）
	ds_read_b128 v[42:45] ...       ; 第 2 条：B 片段（ni 0）
	ds_read_b128 ...                ; 第 3–5 条
	s_waitcnt vmcnt(4) lgkmcnt(3)   ; 6 条 load 只剩 4 条没回 → v0、v62 到了；5 条 LDS 读只剩 3 条 → 前 2 条到了
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[24:27], v[42:45], a[0:3], v0, v62 op_sel_hi:[0,0,0] cbsz:4 blgp:4
```

第一条 MFMA 正好要用 `v0`、`v62`、`v[24:27]`、`v[42:45]`，编译器就只等这 4 样到齐。

**MFMA 这一行的操作数：**

- `a[0:3]`：结果，也就是累加器；
- `v[24:27]`、`v[42:45]`：A 片段和 B 片段；
- 第二个 `a[0:3]`：累加的输入，表示原地累加；
- `v0`、`v62`：A 和 B 的 scale；
- `cbsz:4 blgp:4`：A、B 都是 fp4 格式。

**一步 K 的指令账**（从 ATT 里数的，一步 205 条）：

| 指令 | 条数 | 说明 |
|---|---:|---|
| MFMA | 16 | 真正干活的 |
| `global_load_dword` | 16 | A、B 的 tile，每条 4 字节 |
| `global_load_ubyte` | 8 | scale，每条 1 字节 |
| `ds_write2st64_b32` | 8 | 写 LDS，一条写 A、B 各 4 字节 |
| `ds_read_b128` | 8 | 读 A、B 片段 |
| `s_waitcnt` | 16 | 其中 8 条在拷贝循环里 |
| `s_barrier` | 2 | |
| VALU（不含 MFMA） | 96 | 大多是 64 位地址运算和循环控制 |
| 标量指令和分支 | 35 | |

MFMA 只占一步指令的 8%。

**读汇编时要找的东西：**

- 循环里有没有 `s_waitcnt vmcnt(0)`，尤其是紧跟在 load 后面的；
- load 是不是成批发出去的；
- 有没有意料之外的循环（`s_cbranch_execnz`）；
- 有没有 `scratch_` 指令；
- 地址是不是在用 64 位算；
- MFMA 是不是挤在一起。

### 4.2 计数器

先记住三条规矩：

1. **计数器是整块 GPU、一次 dispatch 的总和。** 除以 `SQ_WAVES`（这里是 16,384），就是“每个 wave”的值。
2. **注意单位。** `SQ_WAVE_CYCLES`、`SQ_WAIT_ANY`、`SQ_WAIT_INST_ANY`、`SQ_BUSY_CYCLES` 以 4 个周期为单位，要乘 4；`GRBM_GUI_ACTIVE` 是 8 个 XCD 的总和，要除以 8。
3. **先核对，再解读。** 能从代码算出来的数，先拿来对。对不上，说明你对代码的理解有误，后面的结论也就站不住。

v0 的核对：

| 计数器 ÷ 16,384 | 实测 | 从汇编算 |
|---|---:|---|
| `SQ_INSTS_VMEM_RD` | 960 | 40 步 × 24 条（16 条 dword + 8 条 ubyte） |
| `SQ_INSTS_VMEM_WR` | 64 | 结尾的 4 × 4 × 4 个 bf16 |
| `SQ_INSTS_FLAT` | 1,024 | 960 + 64：`global_` 指令属于 FLAT 类 |
| `SQ_INSTS_LDS` | 640 | 40 × 16（8 写 + 8 读） |
| `SQ_INSTS_MFMA` | 640 | 40 × 16 |
| `SQ_INSTS_VALU` | 4,847 | 40 × 112（96 条 VALU 加 16 条 MFMA），再加开头结尾 |

全对上了。再看时间：

| 每个 wave | 周期 | 占比 | 含义 |
|---|---:|---:|---|
| `SQ_WAVE_CYCLES` × 4 | 374,492 | 100% | 寿命 |
| `SQ_WAIT_ANY` × 4 | 286,174 | 76% | 在等依赖：`s_waitcnt` 等数据，`s_barrier` 等其他 wave |
| `SQ_WAIT_INST_ANY` × 4 | 50,768 | 14% | 指令准备好了，但要用的单元接不住，比如 load 队列满了 |
| 其余 | 约 37,500 | 10% | 在发指令 |

再看比例：MFMA 利用率 7.3%；L2 命中率 96%，数据都在 L2 里；LDS 的冲突周期除以 `ds_read_b128` 的条数，每条多 4 个周期；L1 合并率 51.6%。

**判断瓶颈的套路：**

| 现象 | 结论 | 方向 |
|---|---|---|
| MFMA 利用率高 | 算力受限 | 减少非 MFMA 指令，让 MFMA 更密 |
| 某一级带宽接近峰值 | 带宽受限 | 减少流量：更大的 tile、更好的复用 |
| 两样都不高，等依赖的时间很多 | 延迟受限（v0 就是这样） | 让更多访存同时在路上：成批发出、预取、提高占用率 |

计数器能告诉你“76% 的时间在等访存”，但说不出等的是哪条 load、在哪一行。这要靠 ATT。

### 4.3 ATT

**每条记录是什么。** wave 文件里每条记录是 `[cycle, type, stall, duration, 指令下标]`：

- wave 在 `cycle` 开始尝试发这条指令，`stall` 周期之后才发出去；
- `duration` 周期之后开始尝试下一条，所以下一条的 `cycle` 通常等于这一条的 `cycle + duration`；
- 对 `s_waitcnt` 和 `s_barrier` 来说，`duration` 就是等的时间。

用 `att_dump.py` 把 SIMD 0 中间那个 wave 的第 10 步展开：

```bash
python3 att_dump.py v0_att/ui_output_agent_*_dispatch_* --step 10          # 全部 205 条
python3 att_dump.py v0_att/ui_output_agent_*_dispatch_* --step 10 --waits  # 只看 load、等待和 MFMA
```

拷贝循环的第 1 圈：

```text
    周期  stall    dur   指令
    +152      0      4   v_mad_i64_i32 v[46:47], s[10:11], v43, s18, v[24:25]   v0_naive.h:194
    +156      0      4   v_and_b32_e32 v0, 60, v42                              v0_naive.h:194
    +160      0      4   v_lshl_add_u64 v[46:47], v[46:47], 0, v[0:1]           v0_naive.h:194
    +172     28     32   global_load_dword v48, v[46:47], off                   v0_naive.h:194
    +204     12     16   v_mad_i64_i32 v[46:47], s[10:11], v44, s18, v[26:27]   v0_naive.h:199
    +220     12     16   v_lshl_add_u64 v[46:47], v[46:47], 0, v[0:1]           v0_naive.h:199
    +244      4      8   global_load_dword v0, v[46:47], off                    v0_naive.h:199
    ...
    +280    164    164   s_waitcnt vmcnt(0)                                     v0_naive.h:193
    +444      0     12   ds_write2st64_b32 v42, v48, v0 offset1:32              v0_naive.h:193
    ...
    +468      0      4   s_cbranch_execnz                                       v0_naive.h:162
```

读法：+172 发 A 的 load，它被堵了 28 周期才发出去；+244 发 B 的 load；+280 开始等，等了 164 周期；+444 数据到了，才写 LDS。

这一步的 8 次等待分别是 164、160、164、520、540、576、1,704、448 周期：

- 前 3 次很短，大概率是 L1 命中：同一个 CU 上相邻的 workgroup 刚读过同样的 A 行，L1 命中率有 77%；
- 中间几次约 550，是 L2 命中（rocprof-compute 量到的 L2 读延迟是 623）；
- 1,704 那次可能是在排队，或者数据离得更远。

8 次加起来，拷贝循环占了这一步 10,464 周期里的约 5,800。

计算段是 `--waits` 输出的后半：

```text
   +6104      0      4   global_load_ubyte v0, ...                  ← 读 scale
   +6176    168    172   global_load_ubyte v64, ...                 ← 发不出去，被堵了 168 周期
   +6588    352    352   s_waitcnt vmcnt(4) lgkmcnt(3)              ← 第一条 MFMA 前，等前两个 scale 和前两个片段
   +6940      0     12   v_mfma ... a[0:3]
   +6952    512    516   global_load_ubyte v58, ...                 ← 又被堵了 512 周期
   ...
   +8060   2016   2016   s_waitcnt vmcnt(0) lgkmcnt(0)              ← 等最后几个 scale，2,016 周期
  +10076      0      8   v_mfma ... a[32:35]
  +10192    196    196   s_barrier                                  ← 第二个 barrier，等其他 wave
```

load 指令上的 stall 是“发不出去”，地址队列满了；`s_waitcnt` 的 duration 是“已经发出去了，在等数据回来”。这是两种不同的等，修法也不同：前者要少发几条、每条发宽一点；后者要早点发、多发几条一起等。

**从记录到汇总。** `att_summary.py` 做的事，就是把所有 wave 的所有记录按指令种类，把 `duration` 加起来：`s_waitcnt` 按它等的计数器分开，记录之间的空隙算“空档”。

“等得最久的源码行”用的是 `s_waitcnt` 自己的源码行。编译器会把它记在要用这份数据的那一行上：拷贝循环的等待落在 193 行（写 LDS 那句），MFMA 前的等待落在 `common.h:34`（`mfma_fp4` 里的那条内建函数）。想知道是哪一处调用的，就看 Source 内联链的下一层，这里是 `common.h:34 -> v0_naive.h:250`。

**读时间线图（图 2）：**

- 第一行是这个 SIMD 的 MFMA 管线，蓝色在算，浅红空着；
- 下面每行是一个同时驻留的 wave；
- 紫色的“梳齿”是拷贝循环的 8 次等待，后面一大段浅紫是 MFMA 前等 scale，红色是 barrier；
- 3 个 wave 的紫色段大量重叠，同一个 SIMD 上找不到能接手的 wave，MFMA 管线就只能空着。

占用率只有 3，同时在等的 wave 又这么多，所以靠换 wave 是藏不住延迟的。

**和计数器对一下：**

| | ATT | 计数器 |
|---|---:|---:|
| wave 寿命（周期） | 378,261 | 374,492 |
| MFMA 忙 | 7.6% | 7.3% |
| 在等（vmcnt、vmcnt + lgkmcnt、barrier、lgkmcnt 加起来） | 71.8% | `SQ_WAIT_ANY` 76% |

对得上，说明追的这一个 CU 有代表性，可以拿它的细节来推整个 kernel。
