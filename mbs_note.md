# MBS 设计思路：它快在哪里

> 记录于 2026-10-24。对象是 `kernel/gemm_a4w4.hip` 里的 `gemm_a4w4_wg_kernel`。
> 寄存器、LDS 用量和指令流来自把内核编译成 `.s` 后的核对；cycle 数和加速比来自 `asm_vs_wg_kernel.md` 里的实测。

MBS 快，是因为它把这个问题真正的瓶颈——矩阵核心（MFMA）的发射——几乎喂满了。在主力的 256×256 tile 上，去掉所有 load 后一个 K=256 的 stage 要 2290 cycle，MBS 现在约 2710，只多 18%。生产 asm 多 12%，最终 MBS 的速度是 asm 的 0.927 倍。

这靠四层设计：

- 每个 wave 算一块很大的输出，让相邻 MFMA 互不依赖。
- 工作组内协作，把每条 MFMA 摊到的数据搬运压到最少。
- 4 级流水，把 load 提前 1 到 2 个 stage 发出。
- 每个 stage 的指令顺序在编译期排定，MFMA 之间只夹少量不会卡住的指令。

下面以 `gemm_a4w4_wg_kernel<16, 4, 4>`（256×256）为例。

## 前提：瓶颈是 MFMA，不是带宽，也不是 occupancy

一个 256×256 工作组每个 stage 搬约 68 KB（A、B 各 32 KB，scale 4 KB），算 33.5 MFLOP，每个 wave 发 128 条 MFMA。去掉所有 load 后，stage 只从约 2710 降到 2290 cycle，说明主循环基本被 MFMA 本身限住。

GPU 一般靠多开 wave 掩盖延迟，这里反过来：`__launch_bounds__(256, 1)`，每个 SIMD 只跑 1 个 wave，独占 512 个寄存器。矩阵核心只要求下一条 MFMA 的操作数已经在寄存器里、且不依赖上一条的结果，一个 wave 自己就能做到。多开 wave 只会把寄存器分薄，tile 变小，复用变差。

还有一个贯穿全篇的硬件事实：一个 wave 的指令按顺序发射。一条 FP4 MFMA 要占矩阵核心十几个 cycle（实测地板约 16.7 cycle/条）。这段时间里同一个 wave 还能发别的指令，所以 MFMA 之间夹一条 load、LDS 读或标量指令，理想情况下几乎不花时间。但只要夹的这条卡住，后面所有 MFMA 都得等；卡住的原因可能是等 `s_waitcnt`、VMEM 队列满，或者前面一串地址计算没算完。后面很多细节都是为了避免这种情况。

## 1. 大块输出，累加器钉在 AGPR

每个 wave 算 256×64 的输出，即 16×4 = 64 个 16×16 子块。每块一个独立的 f32x4 累加器，合计 256 个寄存器，正好放满 AGPR（MFMA 专用的累加寄存器）。

- **没有依赖：** 相邻 MFMA 写的是不同累加器，可以背靠背发射。
- **寄存器级复用：** 一个 A 片段喂 4 条 MFMA，一个 B 片段喂 16 条。
- **inline asm 而不是 builtin：** MFMA 用 inline asm 加 `"+a"` 约束来写。用 builtin 时，编译器会在 K 循环里把累加器在 AGPR 和 VGPR 之间来回搬，每个 stage 几百条拷贝。编译结果是 256 VGPR + 256 AGPR、0 spill，768 条 `v_accvgpr_*` 全在初始化和 epilogue 里，K 循环里一条没有。
- **VGPR 全给流水线：** 两套 B 共 64 个，A 片段双缓冲 128 个，其余是 scale 和地址偏移。

## 2. 搬运最省：A 共享，B 直达，scale 免费

- **A 走 LDS，4 个 wave 共享。** 4 个 wave 沿 N 并排，用的是同一块 256 行的 A。A 每个 stage 只从全局读一次，用 `buffer_load_dwordx4 ... lds` 直接 DMA 进 LDS，不经过 VGPR，也不需要 `ds_write`。全局 A 流量因此是每个 wave 各读一份时的 1/4。
  - 为避免 bank conflict 做了 XOR swizzle。DMA 写进 LDS 的位置由 lane 号固定，所以 swizzle 改的是全局读地址（`a_lane` 里的 `(lane & 7) ^ (lane >> 3)`）。实测 bank conflict 为 0。
- **B 预重排后直接进寄存器。** aiter 的 16×16 preshuffle 已经离线把 B 排成 MFMA 要的布局：一条 MFMA 的 B 操作数（16 行 × 128 个 K）是连续的 1 KB，lane L 读第 L 个 16 字节，正好就是它该拿的那份。所以一个 B 片段就是一条完全合并的 `buffer_load_dwordx4`，不走 LDS，也不用同步。
- **scale 零开销。** 洗牌过的 E8M0 布局里，每个 lane 的一个 dword 恰好装着 4 个 scale 字节，对应相邻两个 16 行子块 × K 的前后两个 128。MFMA 用 `op_sel`/`op_sel_hi` 直接挑字节（`mfma_fp4_sel<OA,OB>`），没有一条解包指令。A 的 scale 也是每个工作组只 DMA 一次进 LDS。

合起来，每个 wave 每个 stage 是 128 条 MFMA，配 20 条全局 load（8 条 B、2 条 B scale、8 条 A DMA、2 条 A scale DMA）和 40 条 LDS 读。

## 3. 流水线：提前 1 到 2 个 stage

- **LDS 环形缓冲：** K 每 256 为一个 stage。LDS 上有 4 个 stage 缓冲，每个 34 KB，共 136 KB（gfx950 每个 CU 有 160 KB）。
- **提前量：** A 提前 2 个 stage 发出，B 提前 1 个。每多提前一个 stage，就要多一套 32 个 VGPR 的 B 寄存器，256×256 放不下第三套。
- **按 4 展开：** 每个 LDS 缓冲和每套 B 寄存器都是编译期固定的变量，没有动态下标。要求 K % 1024 == 0 就是为了这个。
- **persistent + 按 XCD 重排：** grid = min(tiles, 256)，每个 CU 一个工作组轮流取 tile。当前 tile 收尾时，下一个 tile 的前几个 stage 已经在预取（`step` 调用处的 `wrap` 逻辑）。tile 顺序按 XCD 重排，让同一个 XCD 上的工作组在 L2 里共享 A、B。不过实测 persistent 与否、GROUP_M 取 2/8/16，差别都在 ±2% 以内，这一块不是主要收益来源。

## 4. 核心：每个 stage 的指令顺序在编译期排定

一个 stage 分两段，分别算 M 的前 128 行和后 128 行，各 64 条 MFMA。下面是编译后 256×256 主循环里的一个 stage，一个字母代表一条指令（省略了每个 stage 约 60 条地址计算类指令）：

```text
段0  WRBMMMRMMBMRMMMRMMBMRMMMMRBMMMRMMMRBMMMRMMMRBMMMMRMBMMRMMMRMBMMRMMMRbMMMMRMMbMrMMMrMMMrMMMrMMMM
段1  W|RmAMMMRMMmAMRMMMRMMmAMRMMMMRmAMMMRMMMRmAMMMRMMMRmAMMMMRMmAMMRMMMRMmAMMRMMMRmaMMMMRMMmaMrMMMrMMMrMMMrMMMM

M    MFMA（每段 64 条）
R/r  从 LDS 读下一段要用的 A 片段 / A scale
B/b  下一个 stage 的 B / B scale 读进寄存器（只在段0）
A/a  再下一个 stage 的 A / A scale DMA 进 LDS（只在段1）；m 是设置 DMA 目标地址的 s_mov m0
W    s_waitcnt vmcnt(10) lgkmcnt(0)        |  s_barrier
```

这个顺序由 `section`（第 690–734 行）里的编译期公式决定。第 r 次 LDS 读放在第 ⌊64r/20⌋ 条 MFMA 之前，第 k 条全局 load 放在第 ⌊56k/10⌋ 条之前。每放一个元素就插一个 `sched_barrier(0)`，不让编译器重排。

`kernel/gemm_a4w4.hip` 第 703–733 行（节选）：

```cpp
        static_for<TS>([&](auto tc) __attribute__((always_inline)) {
            constexpr int t = decltype(tc)::value;
            // fragment/scale read r goes before MFMA floor(r*TS/NR)
            constexpr int r = (t * NR + TS - 1) / TS;
            if constexpr (r < NR && r * TS / NR == t)
            {
                // ... ds_read 下一段的 A 片段或 A scale ...
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
            // ... 2x2 zig-zag：由 t 算出 (h, mi, ni) ...
            mfma_fp4_sel<2 * h + (ni & 1), 2 * h + (mi & 1)>(acc[mi][ni], bf[ni][h],
                                                             cf[h * MC + ml], bs[ni / 2], cs[ml / 2]);
            __builtin_amdgcn_sched_barrier(0);
        });
```

几条规则和理由：

- **LDS 读提前一整段。** 当前段计算时，把下一段的 16 个 A 片段全部读进另一半缓冲。LDS 延迟有整整 64 条 MFMA 的余量，MFMA 之间不用等 LDS，只在每段开头 `lgkmcnt(0)` 一次。
- **全局 load 一条条撒开，只撒在每段的前 7/8。** 20 条连着发会塞满 VMEM 队列，按序发射的 MFMA 流就会停住。只撒前 7/8，在计时工具里每 stage 快约 2%（端到端不到 1%）。
- **每个 stage 只等两次，每次仍留 10 条 load 在路上。** vmcnt 按发射顺序退役，对应代码里的 `W_START` 和 `W_MID`（第 646–647 行）。队列从来不会被清空：
  - 段0 开头需要本 stage 的 B 已经到达。在它之后发出的只有上个 stage 段1 的 10 条 A DMA，所以等到 vmcnt ≤ 10 就够。
  - 段1 开头需要下个 stage 的 A 已经进了 LDS。在它之后发出的只有本 stage 段0 的 10 条寄存器 load，同样是 ≤ 10。
- **每个 stage 只有一个 barrier。**
  - 段1 必须有：A 的每个 stage 由 4 个 wave 各 DMA 四分之一，每个 wave 先等自己那份落地，再 barrier，之后才能读别人那份。
  - 段0 不需要：将被覆盖的缓冲，所有 wave 在上一个 stage 的 barrier 之前就读完了，4 个缓冲还额外多留了一个 stage 的余量。省掉它，每 stage 少 60 cycle（约 1%）。
- **所有 load 都写成 inline asm，等待全部手写。** 编译器如果看得见 LDS DMA，会在每次无法证明地址不重叠的 LDS 读之前插 `vmcnt(0)`，把预取全部排空。
- **MFMA 顺序照搬 asm 的 2×2 之字形。**

整体上，这相当于用 C++ 模板写 asm：热循环里每条指令的位置由编译期公式决定，编译器基本只负责分配寄存器，同一份代码生成 11 种 tile。

## 5. 最后几个百分点：都是为了不卡住按序发射

- **load 寻址，约 6%。** buffer load 的地址由四部分相加：描述符基址、VGPR 偏移、SGPR 偏移（soffset）、立即数。
  - MBS 的做法：每条 load 的 VGPR 偏移在 kernel 开头算一次就不再变，`opaque()` 防止编译器在每条 load 前重算。soffset 为 0（256 行 tile 的 scale load 例外，用它省 4 个 VGPR）。随 tile/stage 变化的部分加到描述符基址上，每个 stage 每个描述符只花几条标量指令。
  - 实测：非 0 的 soffset 会让发一条 VMEM 指令从 1 cycle 变成 2 cycle；每条 load 前一串标量地址计算也会卡住指令流。改完后每 stage 从 3011 降到 2833 cycle。
  - 越界交给硬件：描述符的范围随基址一起缩小，越界读返回 0，越界写被丢弃，热循环里没有分支。
- **epilogue 写整条 cache line，不经过 LDS。**
  - MFMA 把 B 放在第一个操作数，算的是 C^T，于是每个 lane 拿到同一行的 4 个连续输出。用 `v_cvt_pk_bf16_f32` 打包成 bf16，再用 `v_permlane16_swap` 和 DPP `row_ror:8` 在 lane 之间重组，每条 16 字节的 store 正好覆盖 8 行完整的 128 B。
  - 这样做的原因：实测全 GPU 同时写时，半行 store 要约 7.5k cycle 才被确认，整行约 3.0k；而 vmcnt 按序退役，下一个 tile 的预取会排在这些 store 后面。
  - epilogue 里一条等待都没有。store 固定 32 条（越界的给 0x7fffffff 偏移），下一个 tile 第一步的等待额度里预留了这 32 条。
- **tile 交界处 0 spill。** 之前 256×256 在 epilogue 里有 4 个 VGPR spill，spill 回读带 `vmcnt(0)`，会把下个 tile 已经在路上的预取全部等完。现在三处一起改，spill 归零：
  - lane 号在 asm 里用 `v_mbcnt` 现算；
  - 每次 AGPR 读用空 asm 钉在原位；
  - 累加器清零也做成编译器看不透的形式。

  这一条和整行 store 合起来，14 个 shape 的几何平均快了约 2%。

## tile 选择

启动器在 11 种 tile 里选 `ceil(tiles/256) × (MTILE+64) × NTILE` 最小的那个，也就是“轮数 × 每轮代价”，每轮代价是 tile 面积加约 64 行的固定开销。

例如 M=1024、N=17408：

- 256×256 只有 272 个 tile，第二轮只有 16 个 CU 在干活，代价 163,840。
- 160×256 有 476 个 tile，两轮都比较满，代价 114,688，所以选它。

K 不是 1024 的倍数，或者 tile 少于 64 个时，退回单 wave 内核 `gemm_a4w4_mfma_kernel`：每个 wave 自己从全局读 A 和 B，指令由编译器排。wg 内核比它多出来的，就是上面讲的这些。

## 离地板和 asm 还差多少

- **每 stage 的差距：** 256×256（4096² × 16384）上，地板 2290 cycle，MBS 约 2710，asm 约 2570。MBS 多出的约 420 cycle 主要在两段 body 里：段0 夹带 B load 多 170，段1 夹带 A DMA 多 230。等待一共约 180 cycle，不是差距所在。所以和 asm 的差距在于 load 与 MFMA 怎么交错。
- **epilogue：** 多轮的大 shape 上约占 15%，主要花在 store 发射上，asm 也要付这部分。
- **不是瓶颈的地方：**
  - LDS 读提前量 `FRAG_AHEAD` 从 2 加到 4，差别在 ±2% 内，说明 LDS 延迟不是瓶颈。
  - 两段式循环和 A scale 走 LDS，单独测都是中性的，作用是给后面几项铺路。
- **短板是小 grid：** 1024×1024×5120 只有 64 个 tile，速度是 asm 的 0.53 倍；M=1024、用 96/160 行 tile 的 shape 还慢 4–8%。
