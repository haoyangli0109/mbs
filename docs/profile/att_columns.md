# ATT 的 Hit、Latency、Stall、Idle：单位、含义、怎么读

`rocprofv3 --att` 解码后，`code.json` 给每条指令四个数：Hit、Latency、Stall、Idle。本文回答三个问题：它们的单位是什么，各自是什么意思，对不同的指令该怎么理解。定义先对照 AMD 的官方文档，再用原始 trace 逐行核对。

文中的数字都来自同一次采集：v0（`lab/v0_naive.h`），M = N = 8192，K = 5120，数据在 `/tmp/lab_prof/tutorial/v0_att_now/ui_output_agent_62461_dispatch_3`。`analyze/tmp.txt` 就是用 `analyze/analyze.py` 从这份 `code.json` 打印出来的，下文引用的「第 N 行」指的就是 `tmp.txt` 的行号。

## 0. 结论

1. **单位**：Hit 是次数；Latency、Stall、Idle 是 GPU 时钟周期，也就是常说的 cycle。每一行都是所有被追踪的 wave 加起来的总和。这次追踪了 1 个 CU 上的 64 个 wave，除以 Hit 是每执行一次的平均值，除以 64 是每个 wave 的平均值。
2. **关系**：Latency = Stall + 发射时间。Idle 不在 Latency 里，它是这条指令开始之前、和上一条之间的空档。一个 wave 的寿命等于它所有指令的 Latency 加 Idle。
3. **访存指令的 Latency 只算把请求发出去**，不含数据回来的时间。等数据的时间记在后面的 `s_waitcnt` 上。
4. **`s_waitcnt` 和 `s_barrier` 的 Latency 和 Stall 完全相等**：它们的发射时间是 0，全部时间都在等。
5. **v0 的一个 wave 平均活 377,281 周期**，其中 Stall 占 85.6%，发射占 10.1%，Idle 占 4.2%。光是拷贝循环里的那一条 `s_waitcnt vmcnt(0)` 就占了 44.7%。
6. **同一时刻**：每个 SIMD 的寄存器堆有 512 行，驻留的 wave 各分一段、互不共用。v0 每个 wave 要 144 行，所以每个 SIMD 同时驻留 3 个 wave，它们来自 3 个不同的 workgroup。调度器每 4 个周期轮到这个 SIMD 一次，从准备好的 wave 里挑指令发。见 4.4 到 4.6。

## 1. 数据从哪来

采集命令见 [`naive_gemm_lab.md`](./naive_gemm_lab.md) 第 0 节，要点是：

```bash
rocprofv3 --att --att-library-path $DEC --kernel-include-regex gemm_v0 --kernel-iteration-range 2 \
    -d v0_att -- $EXE v0 --iters 2 --warmup-ms 0 --check 0
```

`--att-library-path` 指向 ROCprof Trace Decoder。`rocprofv3` 跑完程序后会自动调用它，把原始的 `.att` 解成 `ui_output_agent_*_dispatch_*/` 目录下的 JSON，不需要再执行第二条命令。和本文有关的文件有三类：

| 文件 | 内容 |
|---|---|
| `code.json` | 每条指令一行，表头 `ISA, _, LineNumber, Source, Codeobj, Vaddr, Hit, Latency, Stall, Idle`。`analyze.py` 只打印了第 1 列和最后 4 列 |
| `se<SE>_sm<SIMD>_sl<槽位>_wv<第几个>.json` | 每个 wave 一个文件，按时间顺序记录它执行的每一条指令 |
| `filenames.json` | 文件索引。里面的 `"gfxip": 9` 说明 gfx950 按 gfx9 的规则解码 |

ATT 默认只追踪 1 个 CU（这次是 CU 1）上的 4 个 SIMD。v0 每个 SIMD 同时放 3 个 wave，整个 kernel 期间这个 CU 先后跑了 16 个 workgroup，所以一共 64 个 wave，每个 wave 一个文件。

## 2. 官方定义

ROCprofiler-SDK 的文档 [Using thread trace](https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/how-to/using-thread-trace.html) 对这四列的定义：

- **Hitcount**：把所有被追踪的 wave 加在一起，这条指令执行了多少次。
- **Latency**：总周期数。在 gfx9 上定义为 Stall 加发射时间；gfx10 以后是 Stall 加执行时间。
- **Stall**：硬件管线接不住这条指令的总周期，通常是因为执行单元忙，比如访存单元或 LDS 被堵住。
- **Idle**：上一条指令结束到这条指令开始之间的空档。原因可能是发射权被别的 wave 抢走、源或目的寄存器有依赖、指令缓存没命中。

AMD 的博客 [Thread Trace Part 1](https://rocm.blogs.amd.com/software-tools-optimization/thread-trace/README.html) 补充了三点：

1. Stall 是 wave 已经准备好发这条指令，但它需要的管线接不住；Idle 是 wave 还没准备好去发下一条。
2. trace 记录的是指令什么时候**发出去**，不是访存什么时候**完成**。访存在后台继续进行，wave 之后用 `s_waitcnt` 等它。
3. MI200、MI300 最快每 4 个周期发一条指令，所以 trace 里每次执行的时间都是 4 的倍数。gfx950 也是这样：这次采集的 553,600 条记录，时刻、stall、duration 全是 4 的倍数。

## 3. 用原始记录逐行核对

### 3.1 每条记录的格式

wave 文件里，`wave.instructions` 的每一项是一条指令的一次执行：

```text
[cycle, type, stall, duration, 行号]
```

- `cycle`：wave 开始尝试发这条指令的时刻。
- `type`：指令类别，编号含义见第 5 节的表。
- `stall`：等了多少周期才发出去。
- `duration`：从 `cycle` 开始，过多久开始尝试下一条指令。
- `行号`：这条指令在 `code.json` 的 `code` 数组里是第几行。

下面是 SIMD 0 上第一个 wave（`se0_sm0_sl0_wv0.json`）的前 9 条记录。这个 wave 在第 20,712 周期启动。

| 记录 | 指令 | Idle | Stall | 发射 | 说明 |
|---|---|---:|---:|---:|---|
| `[20744, 1, 0, 4, 1]` | `s_load_dwordx2 s[12:13], s[0:1], 0x4` | 32 | 0 | 4 | 第一条，从 wave 启动算起空了 32 周期 |
| `[20748, 6, 0, 8, 2]` | `v_lshrrev_b32 v1, 7, v0` | 0 | 0 | 8 | 20744 + 4 = 20748，紧接着上一条 |
| `[20756, 6, 36, 40, 3]` | `v_bfe_u32 v2, v0, 6, 1` | 0 | 36 | 4 | 等了 36 周期才发出去 |
| `[20796, 6, 0, 4, 4]` | `v_lshlrev_b32 v29, 6, v1` | 0 | 0 | 4 | 20756 + 40 = 20796 |
| `[20800, 6, 0, 4, 5]` | `v_lshlrev_b32 v30, 6, v2` | 0 | 0 | 4 | |
| `[20804, 9, 408, 408, 6]` | `s_waitcnt lgkmcnt(0)` | 0 | 408 | 0 | 等第一条 `s_load` 读回 N 和 K |
| `[21212, 2, 0, 4, 7]` | `s_cmp_gt_i32 s13, 0` | 0 | 0 | 4 | 判断 K > 0 |
| `[21220, 7, 0, 4, 8]` | `s_cbranch_scc1` | 4 | 0 | 4 | 类型 7 表示跳转了 |
| `[21240, 2, 0, 4, 13]` | `s_mov_b64 s[4:5], -1` | 16 | 0 | 4 | 行号从 8 直接到 13，第 9–12 行被跳过；跳转后空了 16 周期 |

表里的「发射」是 `duration − stall`，「Idle」是 `cycle − (上一条的 cycle + 上一条的 duration)`。

### 3.2 四列是怎么加出来的

把 64 个 wave 文件的全部记录按行号加起来，和 `code.json` 的 541 行逐行对比，四列全部相等：

| 列 | 等于 | 核对结果 |
|---|---|---|
| Hit | 这一行的记录条数 | 541/541 行相等 |
| Latency | 所有记录的 `duration` 之和 | 541/541 |
| Stall | 所有记录的 `stall` 之和 | 541/541 |
| Idle | 每条记录前面那段空档之和；第一条指令从 wave 启动算起 | 541/541 |

画在时间线上（issue 就是发射）：

```text
           previous                  this one
  ... --[ stall | issue ]---------[ stall ............ | issue ]---- ...
                        |<-idle-->|<-------- latency --------->|
                        ^         ^
                        |         this.cycle
                        prev.cycle + prev.duration
```

有两点容易搞错：

- Stall 包含在 Latency 里，Idle 不在 Latency 里。
- 空档记在空档**之后**的那条指令上。比如跳转的代价，记在跳过去之后的第一条指令的 Idle 上，不记在分支指令自己身上。

### 3.3 wave 的寿命正好拆成三类

每个 wave 从启动到结束的周期数，等于它所有记录的 Idle 加 Latency（64 个 wave 里 29 个完全相等，35 个差 4 个周期）。所以 wave 活着的每一个周期都恰好落在三类里的一类：发射、Stall、Idle。

`tmp.txt` 里 541 行的 Latency 加 Idle 总和除以 64，就是一个 wave 的平均寿命：

| | 每个 wave（周期） | 占比 |
|---|---:|---:|
| Stall | 323,035 | 85.6% |
| 发射（Latency − Stall） | 38,265 | 10.1% |
| Idle | 15,982 | 4.2% |
| 合计 | 377,281 | 100% |

这和 [`naive_gemm_lab.md`](./naive_gemm_lab.md) 2.5 节里 `att_summary.py` 算出的 37.8 万周期一致。

### 3.4 自己核对

把下面的脚本存成 `att_columns.py`，参数给一个 `ui_output_agent_*_dispatch_*` 目录。它会重算四列并和 `code.json` 对比，再输出寿命拆分、类型编号对应的指令，以及每种指令每执行一次平均的 stall、发射和 idle。第 5 节的表就是用它算出来的。

```python
#!/usr/bin/env python3
"""Rebuild code.json's Hit / Latency / Stall / Idle from the per-wave ATT records.

    python3 att_columns.py <ui_output_agent_*_dispatch_* dir>
"""
import collections, glob, json, os, statistics, sys

ui = sys.argv[1]
code = json.load(open(os.path.join(ui, "code.json")))["code"]
n = len(code)
hit, lat, stall, idle = [0] * n, [0] * n, [0] * n, [0] * n
per_op = collections.defaultdict(list)          # mnemonic -> [(stall, issue, idle)]
types = collections.defaultdict(collections.Counter)
waves, gaps = 0, collections.Counter()
for path in sorted(glob.glob(os.path.join(ui, "se*_sm*_sl*_wv*.json"))):
    w = json.load(open(path))["wave"]
    waves += 1
    prev_end, life = w["begin"], 0
    for cycle, typ, st, dur, idx in w["instructions"]:
        gap = max(0, cycle - prev_end)          # Idle is charged to the instruction after the gap
        hit[idx] += 1; lat[idx] += dur; stall[idx] += st; idle[idx] += gap
        op = code[idx][0].split()[0]
        per_op[op].append((st, dur - st, gap))
        types[typ][op] += 1
        life += gap + dur
        prev_end = cycle + dur
    gaps[(w["end"] - w["begin"]) - life] += 1

cols = (("Hit", hit, 6), ("Latency", lat, 7), ("Stall", stall, 8), ("Idle", idle, 9))
print(f"{waves} waves, {n} rows in code.json")
for name, mine, k in cols:
    print(f"  {name:8s} rows equal to code.json: {sum(mine[i] == code[i][k] for i in range(n))}/{n}")
print(f"  wave lifetime - sum(Idle + Latency): {dict(gaps)}")
L, S, I = sum(lat), sum(stall), sum(idle)
print(f"per wave: {(L + I) / waves:,.0f} cycles = stall {S / waves:,.0f} ({S / (L + I):.1%})"
      f" + issue {(L - S) / waves:,.0f} ({(L - S) / (L + I):.1%}) + idle {I / waves:,.0f} ({I / (L + I):.1%})")
print("type -> instructions:")
for t in sorted(types):
    print(f"  {t:2d}: {', '.join(op for op, _ in types[t].most_common(4))}")
print(f"{'instruction':36s} {'hit':>7s} {'stall':>7s} {'issue':>6s} {'idle':>6s}   (cycles per execution)")
for op, r in sorted(per_op.items(), key=lambda kv: -len(kv[1])):
    m = [statistics.mean(x[j] for x in r) for j in range(3)]
    print(f"{op:36s} {len(r):7d} {m[0]:7.1f} {m[1]:6.1f} {m[2]:6.1f}")
```

## 4. 单位、换算和并发

### 4.1 Hit 是次数，能和代码对上

| Hit | 怎么来的 | 例子 |
|---:|---|---|
| 64 | 64 个 wave，每个执行 1 次 | 开头算 `tid`、清零累加器、收尾写 C |
| 2,560 | 64 个 wave × 40 步 K | MFMA、`s_barrier`、读 scale |
| 20,480 | 64 个 wave × 40 步 × 拷贝循环 8 圈 | 拷贝循环里的 load、`s_waitcnt vmcnt(0)`、`ds_write2st64_b32` |

拷贝循环末尾的 `s_cbranch_execnz` 在 wave 文件里分成两种记录：类型 7（跳了）17,920 次，等于 64 × 40 × 7；类型 8（没跳）2,560 次，等于 64 × 40。也就是每步 K 往回跳 7 次，第 8 圈结束时落空 1 次退出。

### 4.2 Latency、Stall、Idle 是时钟周期，每一行都是总和

- 除以 Hit：每执行一次平均多少周期。RCV 里 Latency 下拉框的 Mean all 就是这样算的，Sum all 是总和。
- 除以 64：每个 wave 平均多少周期。
- 想换成时间，再除以当时的时钟频率，见下一段。

**时钟周期就是常说的 cycle。** GPU 的计算核心有一个时钟，每跳一下就是一个周期，指令的发射和执行都按这个节拍进行。ATT 记的是计算核心的时钟（常叫 shader clock 或 gfx clock），不是显存的时钟。一个周期有多长，取决于当时的频率：

```text
一个周期的时间 = 1 / 频率
```

v0 连续跑时频率约 2.19 GHz（[`naive_gemm_lab.md`](./naive_gemm_lab.md) 2.1 节），一个周期约 0.46 纳秒。按这个频率粗算，拷贝循环里的 `s_waitcnt vmcnt(0)` 每次 527 周期，约 0.24 µs；一个 wave 平均活 377,281 周期，约 172 µs。采 ATT 时的频率不一定是 2.19 GHz，所以 ATT 只给周期数，不给时间。

频率会变：功耗到上限时 GPU 会降频，MBS 跑起来只有约 1.49 GHz。同样的周期数，频率低就要花更多时间。所以比较两个 kernel 本身写得好不好，比周期数更公平；想知道实际耗时，再除以当时的频率。另外，gfx950 每 4 个周期才能发一条指令，所以每次执行的周期数都是 4 的倍数（见第 2 节）。

### 4.3 总和远大于实际经过的时间

这个 CU 从第一个 wave 启动到最后一个 wave 结束只有 2,162,776 周期，但 64 个 wave 的寿命加起来是 24,145,844 周期，平均同时有 11.2 个 wave 在这个 CU 上跑。所以 `s_waitcnt` 上的大数只说明 wave 在等，不等于硬件在空转：一个 wave 在等的时候，同一个 SIMD 上的另外两个 wave 可能正在发指令。为什么能这样，见 4.4 和 4.5。

### 4.4 一个 SIMD 驻留几个 wave，寄存器怎么分

**寄存器堆属于 SIMD。** 每个 SIMD 有一块自己的向量寄存器堆，一个 CU 有 4 块，互相不通。一块是 512 行 × 64 列，每格 32 位，共 512 × 64 × 4 字节 = 128 KB。

- 64 列对应 wave 的 64 个 lane。「每个 lane 512 个寄存器」说的是每一列有 512 格。
- 这 512 行是整个 SIMD 的总量，由驻留在这个 SIMD 上的 wave 分着用，不是每个 wave 各有 512 行。一个 wave 最多能要满 512 行；MBS 就要满了，所以它每个 SIMD 只能放 1 个 wave。
- 执行单元一次实际只处理 16 个 lane（ROCm Compute Profiler 文档称之为 16-wide SIMD），一条向量指令分 4 个周期做完 64 个 lane。这就是每 4 个周期才能发一条的原因。

**wave 是硬件调度和执行的单位。** 写代码时看到的是线程。硬件把 workgroup 里每 64 个连续的线程捆成一个 wave：`tid` 0–63 是一个，64–127 是一个，依此类推。一个 wave 只有一个程序计数器，64 个 lane 总是执行同一条指令，各自用寄存器堆里自己那一列。wave 启动时，硬件在某个 SIMD 上给它分三样东西：

1. 一个槽位：这个 SIMD 的 8 个槽位之一，存它的程序计数器、EXEC 掩码（哪些 lane 有效）、状态和指令缓冲。
2. 一段向量寄存器：连续的若干行，这个 wave 独占。
3. 一段标量寄存器（SGPR）：一个值整个 wave 共用，同样是这个 wave 独占。

workgroup 另外在 CU 上分到一段 LDS，组里 4 个 wave 共用，组内 wave 之间就是靠它交换数据。

所以 wave 更像 CPU 的硬件线程（超线程）：有自己的一套寄存器和程序计数器。不同的是，wave 从启动到结束，寄存器一直留在寄存器堆里，从不搬到内存。调度器在驻留的 wave 之间切换没有任何开销；代价是 512 行要分着用，每个 wave 要得越多，能同时驻留的就越少。

| 资源 | 硬件上有多少 | 分给谁 | v0 用多少 |
|---|---|---|---|
| wave 槽位 | 每个 SIMD 8 个 | 每个 wave 1 个 | 每个 SIMD 3 个 |
| 向量寄存器（VGPR + AGPR） | 每个 SIMD 512 行 × 64 列 | 每个 wave 一段，独占 | 每个 wave 144 行，3 个共 432 行 |
| 标量寄存器（SGPR） | 每个 SIMD 约 800 个 | 每个 wave 一段，独占 | 每个 wave 28 个 |
| LDS | 每个 CU 160 KB | 每个 workgroup 一段，组内 4 个 wave 共用 | 每个 workgroup 16 KB |

一个 SIMD 能驻留几个 wave，由这几样里最先用完的那一样决定。v0 每个 wave 要 144 行：先放 73 个 VGPR；AGPR 必须从 4 的倍数开始，所以从第 76 行起放 64 个 AGPR，到第 139 行；再按 8 行一组取整，得到 144。512 ÷ 144 ≈ 3.6，所以每个 SIMD 最多放 3 个，是被向量寄存器卡住的。

**实测：每个 wave 分到了哪几行。** 硬件给每个 wave 留了两个只读的状态寄存器，wave 可以用 `s_getreg_b32` 读出来：`HW_ID` 记着它在哪个 CU、哪个 SIMD、第几个槽位；`GPR_ALLOC` 记着它的向量寄存器从第几行开始、一共几行。本节末尾的探测程序让每个 wave 读出这两个值，然后空转约 30 µs，让好几个 wave 同时驻留。分别让每个 wave 要 144、96、64、32 行，在一张空闲的卡上各跑一次：

| 每个 wave 要几行 | 一个 SIMD 最多同时驻留 | 分到的行 | 没用上的行 |
|---:|---:|---|---|
| 144（和 v0 一样） | 3 | 0–143、144–287、288–431 | 432–511，共 80 行 |
| 96 | 5 | 0–95、96–191、192–287、288–383、384–479 | 480–511，共 32 行 |
| 64 | 8 | 0–63、64–127、……、448–511 | 无 |
| 32 | 8 | 0–31、32–63、……、224–255 | 256–511。寄存器够放 16 个，被 8 个槽位卡住 |

v0 的情况画出来是这样：

```text
SIMD 的向量寄存器堆：512 行 × 64 列（每格 4 字节，共 128 KB）

  行号        给谁
  0–143      槽位 0 的 wave：它的 v0–v72、空 3 行、a0–a63、空 4 行
  144–287    槽位 1 的 wave（来自另一个 workgroup）
  288–431    槽位 2 的 wave（又一个 workgroup）
  432–511    空着，80 行，不够再放一个
```

测出来的几个事实：

- **各占一段，不重叠。** 把同一个 SIMD 上同时驻留的 wave 两两比较，144 行时有 6,144 对，寄存器段重叠 0 次，占同一个槽位 0 次。另外三种情况也一样。
- **跑完就交接。** 144 行时，槽位 0 的 wave 在 t = 3,000 结束，t = 3,036 进来一个新的 wave（来自另一个 workgroup），还是槽位 0、第 0–143 行。这里的时间单位是 100 MHz 的墙钟，1 个单位 10 ns。ATT 里「slot 00 先后跑了 6 个 wave」就是这个意思。
- **一个 workgroup 的 4 个 wave 在同一个 CU 的 4 个不同 SIMD 上**，1,536 个 workgroup 全部如此。但哪个 wave 去哪个 SIMD 不固定：wave 编号和 SIMD 编号相同的只占约四分之一（每次运行略有不同，例如 1,472/6,144），所以 wave 0 不一定在 SIMD 0 上。

**同一条指令，不同 wave 改的是不同的行。** 比如拷贝循环里的 `v_add_u32 v45, 0x100, v45`：槽位 0 的 wave 执行时改第 45 行，槽位 1 的 wave 执行时改第 144 + 45 = 189 行。硬件用这个 wave 的起始行加上寄存器编号去找；同一个 wave 里，64 个 lane 各改这一行中自己那一列。

**汇编是按一个 wave 写的，所有 wave 执行同一份。** 整个 kernel 只有这一份指令，在显存里也只存一份，所有 CU 上的所有 wave 都从头到尾执行它，每条指令对每个 wave 发一次。`s_` 开头的标量指令一个 wave 只算一个值，64 个 lane 共用，比如 `m0`；`v_` 开头的向量指令同时作用于 64 个 lane，每个 lane 用自己的 VGPR 各算各的，读的时候可以代入其中一个 lane，就像读 C++ kernel 时代入一个线程。不同 wave 算的数据不同，是因为起点不同：启动时，硬件把 `blockIdx.x`、`blockIdx.y` 填进 `s2`、`s3`，把 `threadIdx.x` 填进每个 lane 的 `v0`，后面所有的下标和地址都由它们算出来。比如 workgroup (3, 5) 里 `tid = 165` 的 lane，同一条 `s_lshl_b32 s5, s3, 7` 得到 `m0 = 640`，`v_and_b32 v28, 15, v0` 得到 `r16 = 5`；换成 workgroup (0, 0) 里 `tid = 0` 的 lane，两个结果都是 0。跳转是整个 wave 一起跳；只想让一部分 lane 做事时，就在 `exec` 里把其他 lane 关掉，比如拷贝循环末尾关掉已经搬完的 lane。

**复现。** 把下面的程序存成 `wave_probe.hip`，换一张空闲的卡运行：

```bash
hipcc -O3 -std=c++20 --offload-arch=gfx950 wave_probe.hip -o wave_probe
HIP_VISIBLE_DEVICES=5 ./wave_probe
```

`GPR_ALLOC` 各字段的位置是从实测结果反推的：低 6 位是起始行 ÷ 8，第 6–11 位是行数 ÷ 8 − 1。四种寄存器数都按这个读法对上了。

<details>
<summary>wave_probe.hip</summary>

```cpp
// Every wave reads its own HW_ID (CU / SIMD / slot) and GPR_ALLOC (its VGPR block),
// then spins ~30 us so that several waves are resident on each SIMD at once.
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <tuple>
#include <vector>

#define CHECK(x)                                                              \
    do {                                                                      \
        hipError_t e_ = (x);                                                  \
        if (e_ != hipSuccess) {                                               \
            std::printf("%s: %s\n", #x, hipGetErrorString(e_));               \
            std::exit(1);                                                     \
        }                                                                     \
    } while (0)

struct Rec {
    unsigned hw_id, gpr_alloc, xcc;
    int wg, wave;
    unsigned long long t0, t1;
};

// Writing register REG makes the compiler allocate REG + 1 VGPRs per lane.
// s_getreg ids: 4 = HW_ID, 5 = GPR_ALLOC, 20 = XCC_ID.
#define PROBE(NAME, REG)                                                            \
    __global__ __launch_bounds__(256) void NAME(Rec* out, unsigned long long spin)     \
    {                                                                               \
        asm volatile("v_mov_b32 " REG ", 0" ::: REG);                               \
        const unsigned hw  = __builtin_amdgcn_s_getreg((31 << 11) | 4);             \
        const unsigned gpr = __builtin_amdgcn_s_getreg((31 << 11) | 5);             \
        const unsigned xcc = __builtin_amdgcn_s_getreg((31 << 11) | 20);            \
        const unsigned long long t0 = wall_clock64();                               \
        unsigned long long t;                                                       \
        do { t = wall_clock64(); } while (t - t0 < spin);                           \
        if (threadIdx.x % 64 == 0)                                                  \
            out[blockIdx.x * 4 + threadIdx.x / 64] =                                \
                Rec{hw, gpr, xcc, (int)blockIdx.x, (int)threadIdx.x / 64, t0, t};    \
    }
PROBE(probe144, "v143")
PROBE(probe96, "v95")
PROBE(probe64, "v63")
PROBE(probe32, "v31")

// Field layout as observed on gfx950:
//   HW_ID     [3:0] slot, [5:4] SIMD, [11:8] CU, [12] SH, [15:13] SE
//   GPR_ALLOC [5:0] first VGPR row / 8, [11:6] VGPR rows / 8 - 1
using CuKey = std::tuple<unsigned, unsigned, unsigned, unsigned>;
static unsigned slot(const Rec& r) { return r.hw_id & 15; }
static unsigned simd(const Rec& r) { return (r.hw_id >> 4) & 3; }
static CuKey cu(const Rec& r) { return {r.xcc & 15, (r.hw_id >> 13) & 7, (r.hw_id >> 12) & 1, (r.hw_id >> 8) & 15}; }
static int first_row(const Rec& r) { return (r.gpr_alloc & 63) * 8; }
static int rows(const Rec& r) { return (((r.gpr_alloc >> 6) & 63) + 1) * 8; }

template <class Kernel>
static void run(const char* name, Kernel kernel, int wgs, unsigned long long spin)
{
    Rec* d;
    CHECK(hipMalloc(&d, sizeof(Rec) * wgs * 4));
    kernel<<<wgs, 256>>>(d, spin);
    CHECK(hipDeviceSynchronize());
    std::vector<Rec> r(wgs * 4);
    CHECK(hipMemcpy(r.data(), d, sizeof(Rec) * r.size(), hipMemcpyDeviceToHost));
    CHECK(hipFree(d));

    std::map<std::pair<CuKey, unsigned>, std::vector<const Rec*>> per_simd;
    for (const Rec& x : r) per_simd[{cu(x), simd(x)}].push_back(&x);

    int peak = 0;
    long pairs = 0, overlap = 0, same_slot = 0;
    for (auto& [key, v] : per_simd) {
        std::vector<std::pair<unsigned long long, int>> ev;
        for (const Rec* x : v) {
            ev.push_back({x->t0, 1});
            ev.push_back({x->t1, -1});
        }
        std::sort(ev.begin(), ev.end(), [](auto a, auto b) {
            return a.first != b.first ? a.first < b.first : a.second < b.second;
        });
        int live = 0;
        for (auto& e : ev) peak = std::max(peak, live += e.second);
        for (size_t i = 0; i < v.size(); ++i)
            for (size_t j = i + 1; j < v.size(); ++j) {
                const Rec &a = *v[i], &b = *v[j];
                if (!(a.t0 < b.t1 && b.t0 < a.t1)) continue;
                ++pairs;
                overlap += first_row(a) < first_row(b) + rows(b) && first_row(b) < first_row(a) + rows(a);
                same_slot += slot(a) == slot(b);
            }
    }

    long one_cu = 0, four_simds = 0, wave_is_simd = 0;
    for (int g = 0; g < wgs; ++g) {
        unsigned mask = 0;
        bool same = true;
        for (int w = 0; w < 4; ++w) {
            const Rec& x = r[g * 4 + w];
            same &= cu(x) == cu(r[g * 4]);
            mask |= 1u << simd(x);
            wave_is_simd += unsigned(w) == simd(x);
        }
        one_cu += same;
        four_simds += mask == 15;
    }

    std::printf("== %s, %d workgroups\n", name, wgs);
    std::printf("   most waves resident on one SIMD at once: %d\n", peak);
    std::printf("   co-resident pairs: %ld, VGPR blocks overlapping: %ld, same slot: %ld\n", pairs, overlap, same_slot);
    std::printf("   workgroups on one CU: %ld, on 4 different SIMDs: %ld, waves with wave index == SIMD: %ld of %zu\n",
                one_cu, four_simds, wave_is_simd, r.size());
    std::vector<const Rec*> s = per_simd.begin()->second;
    std::sort(s.begin(), s.end(), [](auto a, auto b) { return a->t0 < b->t0; });
    for (size_t i = 0; i < std::min<size_t>(s.size(), 9); ++i)
        std::printf("   WG %4d wave %d  slot %u  VGPR rows %3d-%3d  t = %5llu .. %5llu\n", s[i]->wg, s[i]->wave,
                    slot(*s[i]), first_row(*s[i]), first_row(*s[i]) + rows(*s[i]) - 1, s[i]->t0 - s[0]->t0,
                    s[i]->t1 - s[0]->t0);
}

int main()
{
    int khz = 0;
    CHECK(hipDeviceGetAttribute(&khz, hipDeviceAttributeWallClockRate, 0));
    const unsigned long long spin = 30ull * khz / 1000;  // 30 us of wall clock
    std::printf("wall clock %d kHz, spin %llu ticks\n", khz, spin);
    run("144 VGPR rows per wave", probe144, 256 * 3 * 2, spin);
    run(" 96 VGPR rows per wave", probe96, 256 * 5 * 2, spin);
    run(" 64 VGPR rows per wave", probe64, 256 * 8 * 2, spin);
    run(" 32 VGPR rows per wave", probe32, 256 * 8 * 2, spin);
    return 0;
}
```

</details>

### 4.5 调度器怎么在 wave 之间挑

每个 wave 自己按程序顺序发指令，卡在一条上，后面的就一条也发不出去。在 wave 之间挑谁来发，是调度器的事。按 ROCm Compute Profiler 文档 [Pipeline descriptions](https://rocm.docs.amd.com/projects/rocprofiler-compute/en/latest/conceptual/cdna/pipeline-descriptions.html) 的描述：

- 调度器每个周期只看一个 SIMD，4 个 SIMD 轮流，所以每个 SIMD 每 4 个周期轮到一次。
- 轮到某个 SIMD 时，从它驻留的 wave 里挑：每个 wave 最多发 1 条，每一类指令最多发 1 条。类别有 5 种：向量运算、向量访存、标量运算和标量访存、LDS、分支。所以一次最多发 5 条，来自不同的 wave。MFMA 从向量运算这一路发出；`s_waitcnt`、`s_barrier` 这类内部指令不进执行单元。

```text
周期:   0      1      2      3      4      5      6   ...
轮到:  SIMD0  SIMD1  SIMD2  SIMD3  SIMD0  SIMD1  SIMD2 ...
```

**发射机会不跨 SIMD 争，执行单元有的要争。** 调度器按固定顺序轮流服务 4 个 SIMD，所以一个 wave 只和同一个 SIMD 上的其他驻留 wave 争发射机会，v0 里是另外 2 个来自别的 workgroup 的 wave。执行单元则分两类：

| 单元 | 有几套 | 谁和谁争 |
|---|---|---|
| 向量寄存器堆、向量运算单元、矩阵单元（MFMA） | 每个 SIMD 一套 | 只有同一个 SIMD 上的 wave |
| 访存单元（含 L1 缓存） | 整个 CU 一套 | 4 个 SIMD 上的所有 wave |
| LDS | 整个 CU 一套，4 个 SIMD 两两一对接入 | 同一对里，同一时刻只能有一个 SIMD 发 LDS 指令 |
| 标量单元、分支单元 | 整个 CU 一套 | 调度器每个周期只服务一个 SIMD，天然轮流，不会撞车 |

ATT 里有两个证据。一是在至少有一个矩阵单元在算的时间里，有 5.1% 的时间 4 个 SIMD 同时在算 MFMA，如果整个 CU 只有一个矩阵单元，这不可能发生。二是一条 load 等待期间，**其他 SIMD** 先交进去的访存越多，它的 Stall 越长：0 条时平均 1.9 周期，3 条时 87，6 条以上时 198；而同一个 SIMD 上的 wave 在这期间插进来的情况很少。

两种「争」记在不同的列：同一个 SIMD 里同类指令抢输了，记成 Idle；共用单元正忙着处理别的 SIMD 的请求而不收，记成 Stall。v0 的 load Stall 主要是后一种（见 5.1）。

**轮流的是发指令的机会，不是轮流独占整个 SIMD。**

- 每个执行单元一次收一条指令。轮到这个 SIMD 时，同一类指令只能发一条，要发同一类的 wave 得争。
- 不同类的指令可以同时进行。比如 wave A 的 MFMA 还在矩阵单元里算（要 16 个周期），这时 wave B 可以发一条向量运算，wave C 的 load 可以交给访存单元。本节末尾的例子里，一个 wave 停在 `s_waitcnt` 等了 576 周期，另外两个 wave 在这段时间里发了 38 条指令，其中有 4 条 MFMA。

**什么叫准备好。** 每个 wave 只能按程序顺序发指令，所以每次只看它的下一条指令。「准备好」就是这条指令现在就能发，要同时满足三个条件：

1. **指令已经取到了。** 刚跳转之后要重新取指令，这时还没准备好。v0 拷贝循环每次跳回开头，都要空十几个周期（`tmp.txt` 第 221 行）。
2. **它要读的寄存器已经写好了。** 比如拷贝循环里的 `global_load_dword`，地址是上一条 `v_lshl_add_u64` 刚算出来的，要等几个周期结果写好才能发（第 224 行）。
3. **它不是在等东西。** 下一条如果是 `s_waitcnt`，要等访存数据回来（第 233 行，每次 527 周期）；如果是 `s_barrier`，要等组里其他 wave 都到齐（第 244 行）。

三条都满足，这个 wave 才算准备好。准备好以后还有两关：

- 同一个 SIMD 上如果还有别的 wave 也要发同一类指令，每类每次只能发一条，抢输了就等下一轮。
- 抢到了，执行单元还得肯收。访存队列满了，或者矩阵管线还在算上一条 MFMA，就只能等。

ROCprofiler-SDK 的 [PC 采样文档](https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/how-to/cdna3-cdna4-pc-sampling.html) 列出了 wave 发不出指令的各种原因，正好对应上面的三个条件和两关。对应到 ATT 的列是这样：

| 原因（官方名称） | 意思 | ATT 里记成 | v0 的例子（`tmp.txt` 行号） |
|---|---|---|---|
| `NO_INSTRUCTION_AVAILABLE` | 下一条还没取进来，比如刚跳转、指令缓存没命中 | Idle | 第 221 行：拷贝循环跳回开头后，每次空约 14 周期 |
| `ALU_DEPENDENCY` | 要用的寄存器还没被前面的指令写好 | Idle | 第 224 行：地址刚算出来，约 8 周期；第 232 行：等 `vcc`，约 7 周期 |
| `WAITCNT` | 停在 `s_waitcnt`，数据还没回来 | `s_waitcnt` 的 Stall | 第 233 行：每次 527 周期 |
| `BARRIER_WAIT` | 同一个 workgroup 的其他 wave 还没到 | `s_barrier` 的 Stall | 第 244 行：每次约 144 周期 |
| `ARBITER_NOT_WIN` | 同一个 SIMD 上另一个 wave 这次也要发同一类，它赢了 | Idle（官方把抢输了列为 Idle 的原因之一） | |
| `ARBITER_WIN_EX_STALL` | 被选中了，但执行单元不收 | 这条指令的 Stall | 5.1 节的 load 排队；第 278 行：连着发的 MFMA，每次 Stall 11.5 |

表里前四种是三个条件没满足：前两种是条件 1、2，等访存和等 barrier 是条件 3。后两种是准备好以后的那两关。

**发的是什么。** 发的是这个 wave 的下一条指令：调度器把它从 wave 的指令缓冲里取出来，读好操作数，交给对应的执行单元，wave 就可以考虑下一条了。执行单元接着干活：向量运算约 4 个周期，MFMA 16 个周期，访存几百个周期，都和 wave 后面的发射重叠进行。

| 类别 | 交给谁 | v0 里的例子 |
|---|---|---|
| 向量运算（VALU） | 本 SIMD 的向量单元，64 个 lane 各算各的 | `v_add_u32`、`v_mad_i64_i32` |
| MFMA | 从 VALU 这一路发出，在单独的矩阵单元上执行 | `v_mfma_scale_f32_16x16x128_f8f6f4` |
| 标量运算、标量访存 | 标量单元，一个值整个 wave 共用 | `s_add_u32`、`s_load_dwordx2` |
| 向量访存 | 访存单元，整个 CU 共用 | `global_load_dword`、`global_store_short` |
| LDS | LDS，整个 CU 共用 | `ds_read_b128`、`ds_write2st64_b32` |
| 分支 | 分支单元 | `s_cbranch_execnz` |
| 内部指令 | 不进执行单元，调度器自己处理 | `s_waitcnt`、`s_barrier`、`s_endpgm` |

**trace 里的例子。**

- **轮流发**：slot 00 的一个 wave 在第 778,608 周期停在 `s_waitcnt vmcnt(0)`，等了 576 周期。这段时间里 slot 01 发了 29 条指令（其中 4 条 MFMA），slot 02 发了 9 条。
- **一起发**：SIMD 0 上有 9,576 个时刻是两个以上的 wave 同时发了指令。最常见的是标量 + 向量（2,807 次）、LDS + 向量（1,309 次）、访存 + 向量（1,292 次）。另有 1,635 次是两条向量运算一起发，PC 采样文档专门说明，CDNA3、CDNA4 在某些条件下可以一次发两条 VALU。

反过来，MBS 每个 SIMD 只放 1 个 wave，没有别的 wave 能和它搭伙一起发，而一个 wave 每次轮到只能发 1 条，所以只能靠它自己把 load 一条条插在 MFMA 之间。

### 4.6 小结

- **SIMD 和它的寄存器堆是固定的硬件**：每个 SIMD 512 行 × 64 列。
- **一个 SIMD 最多放 8 个 wave**（8 个槽位），同时还受寄存器限制。SGPR、LDS 也会限制，但通常是向量寄存器先用完。v0 每个 wave 要 144 行，所以每个 SIMD 放 3 个：寄存器分成 3 段，每个 wave 独占一段，剩下 80 行没人用。
- **wave 不只是软件概念。** 64 个线程捆成一个 wave 是硬件做的。wave 驻留期间，在硬件里实实在在占着一个槽位（存着它的程序计数器和 EXEC 掩码）和自己那一段寄存器。它不是一块固定的电路，最贴切的说法是硬件里的一个执行上下文，类似 CPU 的硬件线程。
- **汇编是按一个 wave 写的**，所有 wave 执行同一份：`s_` 指令一个 wave 算一个值，`v_` 指令 64 个 lane 各算各的。各个 wave 算的数据不同，是因为启动时 `s2`、`s3`（`blockIdx`）和 `v0`（`threadIdx.x`）不同。见 4.4 节「汇编是按一个 wave 写的」那段。
- **3 个 wave 轮流的是发指令的机会**，不是轮流独占整个 SIMD：同一类指令每次只能发一条，要争；不同类的指令可以同时进行。
- **跨 SIMD**：发射机会不争。向量单元、矩阵单元每个 SIMD 一套，不跨 SIMD 争；访存单元、LDS 整个 CU 共用，会被别的 SIMD 的 wave 堵住，堵住的时间记在 Stall。
- **「准备好」是这个 wave 的下一条指令现在就能发**：指令已经取到、要读的寄存器已经写好、不在等访存或 barrier。准备好以后还要抢赢同类、执行单元肯收，才算真正发出去。
- **Stall 是准备好了、也轮到了，但硬件不收的时间**；抢输了记在 Idle。`s_waitcnt`、`s_barrier` 是例外，它们的 Stall 是在等数据、等别的 wave。见第 5 节开头。

## 5. 不同指令怎么读

**先看是哪种指令，再读 Stall。** Stall 在不同指令上意思不一样：

- **普通指令**（向量运算、MFMA、load/store、LDS、标量运算、分支）：这条指令已经准备好了（指令取到了，要读的寄存器也写好了），也轮到它发了，但执行单元或管线暂时不收，只能原地等。比如 `global_load_dword` 每次平均 Stall 23 周期，是访存单元正忙着收别的 wave 的请求；连着发的 MFMA 每次 Stall 约 11.5 周期，是矩阵管线还在算上一条。
- **「抢输了」不算 Stall。** 同一个 SIMD 上另一个 wave 这一轮也要发同一类指令并且抢先了，这种情况按官方定义记在 Idle 里。所以 Stall 更准确地说是「轮到这个 wave 发了，但执行单元不收」，纯粹是硬件那头的原因。
- **`s_waitcnt` 和 `s_barrier` 是例外。** 它们不进任何执行单元，不存在「硬件收不收」的问题，它们的 Stall 是这条指令本身在等：`s_waitcnt` 等前面发出去的访存数据回来，拷贝循环里那条每次 527 周期，意思是数据还没回来；`s_barrier` 等同一个 workgroup 的其他 wave 都到齐。

所以同样是 Stall：在普通指令上表示硬件那头忙，在 `s_waitcnt` 上表示在等数据，在 `s_barrier` 上表示在等别的 wave。

下表是这次采集里，每类指令每执行一次的平均值（第 3.4 节脚本的输出）。「类型」是 wave 记录里第二个字段的编号。

| 类别（类型） | 例子 | 发射 | Stall | Idle | Stall 在等什么 |
|---|---|---:|---:|---:|---|
| 标量运算（2） | `s_or_b64`、`s_andn2_b64` | 4 | 0–0.2 | 0–8 | 几乎不会 stall |
| 向量运算（6） | `v_add_u32`、`v_lshl_add_u64` | 4 | 2–5 | 0–7 | 管线正被别的 wave 占着 |
| MFMA（6） | `v_mfma_scale_f32_16x16x128_f8f6f4` | 4–12 | 0 或约 11.5 | 0 | 矩阵管线还被上一条 MFMA 占着 |
| 全局访存（4） | `global_load_dword`、`global_load_ubyte`、`global_store_short` | 4 | 16–60 | 3–8 | 在排队：整个 CU 共用的访存单元暂时不收，见 5.1 |
| LDS（5） | `ds_read_b128`、`ds_write2st64_b32` | 读 4，写 13 | 1–4 | 0 | 这里很小；队列满时会变大 |
| 标量访存（1） | `s_load_dwordx2` | 4 | 0.2 | 22 | 几乎不会 stall；Idle 大主要是因为其中一条是 wave 的第一条指令 |
| 等待（9） | `s_waitcnt`、`s_barrier` | 0 | 每次几个到几百 | 0 | 等访存完成，或者等同一 workgroup 的其他 wave |
| 分支（7 跳了，8 没跳） | `s_cbranch_execnz` | 4 | 0 | 4 | 不会 stall；跳转的代价记在下一条的 Idle 上 |
| 结束（11） | `s_endpgm` | 536 | 0 | 0 | 例外，见 5.7 |

下面逐类说明。引用的行都在 `tmp.txt` 里，格式是「指令 Hit Latency Stall Idle」。

### 5.1 访存指令：Latency 只是把请求交出去，Stall 是在排队

```text
224: global_load_dword v48, v[46:47], off     20480   559688   473612   161208
227: global_load_dword v0, v[46:47], off      20480   448992   364156   144992
```

一条 load 从准备好到用上数据，分成几段。以第 224 行为例，每执行一次的平均值：

```text
地址还没算好         排队等访存单元收        交进去
|<-- Idle 7.9 -->|<---- Stall 23.1 ---->|<- 发射 4.2 ->|
                                          之后数据在 L2 / 显存里走一个来回，
                                          算在后面的 s_waitcnt vmcnt(0) 上（第 233 行，每次 527）
```

- **Idle**：load 自己还没准备好。地址 `v[46:47]` 是上一条 `v_lshl_add_u64` 刚算出来的，要等结果写好。
- **Stall**：地址已经有了，但访存单元现在不收，只能原地排队。不是在等数据。
- **发射**：把请求交给访存单元，交完 wave 就能往下走。
- 所以这一行的 Latency 只有 27 周期，数据回来要五百多周期，那部分算在等它的 `s_waitcnt` 上（见 5.2）。

**为什么要排队。** 一个 CU 的 4 个 SIMD 共用一个向量访存单元，v0 同时有 12 个 wave 往里交请求。把 64 个 wave 放到同一条时间轴上，统计每条全局访存在等待期间，别的 wave 先交进去了几条：

| 等待期间别的 wave 交了几条 | 次数 | 平均 Stall（周期） |
|---:|---:|---:|
| 0 | 46,354 | 1.9 |
| 1 | 6,132 | 16.4 |
| 3 | 4,065 | 82.8 |
| 8 条以上 | 2,324 | 432.0 |

没人抢时几乎不用等，别人交得越多，它等得越久：它在等的时候，访存单元正忙着收别人的请求。

**一批连着发的 load，靠后的更容易被堵。** 每步 K 读 scale 的 6 条 `global_load_ubyte` 是连着发的。第 245 行是第 1 条，每次平均 Stall 23；第 255 行是第 6 条，平均 189，98% 的时候都要等。scale 的 load 尤其容易堵：一条指令的 64 个 lane 读 16 行、每行 4 个字节，这 16 行离得很远，要碰 16 个 cache line；拷贝循环的 `global_load_dword` 一条只碰 4 行。每条请求处理得慢，入口前的队就排得长。

`ds_read`、`s_load` 也是同一个道理：Latency 只是交出去，读回来的等待在后面的 `s_waitcnt lgkmcnt` 上。

### 5.2 `s_waitcnt` 和 `s_barrier`：全部时间都是 Stall

```text
233: s_waitcnt vmcnt(0)                       20480 10789944 10789944        0
265: s_waitcnt vmcnt(4) lgkmcnt(3)             2560  1299204  1299204        0
282: s_waitcnt vmcnt(0) lgkmcnt(0)             2560  2057548  2057548        0
244: s_barrier                                 2560   367800   367800       64
289: s_barrier                                 2560   408076   408076      104
```

这两种指令不进任何执行单元，发射时间是 0，所以 Latency 和 Stall 完全相等。

- **`s_waitcnt` 的 Stall 是在等访存计数降到要求的值，访存延迟就是在这里显露出来的。** 拷贝循环的 `s_waitcnt vmcnt(0)`（第 233 行）每次 527 周期，大约是一次显存往返（文档里测的 L2 读延迟是 623 周期）。每个 wave 合计 168,593 周期，占寿命的 44.7%。第 265 行每次 508 周期、第 282 行每次 804 周期，是 MFMA 前等 scale 和 LDS 读。
- **`s_barrier` 的 Stall 是在等同一个 workgroup 的其他 wave 都到达。** 两处分别是每次 144 和 159 周期。

博客特别提醒：这类指令看起来很「贵」，一个原因是很多 wave 会同时停在相似的地方，总和就很大。

### 5.3 MFMA：发射约 8 周期，连着发时 Latency 约 16

```text
266: v_mfma_scale... a[0:3],   ... v0,  v62   2560    31764     1296        4
277: v_mfma_scale... a[16:19], ... v66, v62   2560    23016     1040        0
278: v_mfma_scale... a[20:23], ... v66, v63   2560    41428    29492        0
```

- **发射约 8 周期，因为它其实是两条指令。** `v_mfma_scale_f32_16x16x128_f8f6f4` 的编码是 16 字节。前 8 字节单独反汇编是 `v_mfma_ld_scale_b32 v0, v62`，先把两个 scale 送进去；后 8 字节才是 MFMA 本身。
- **连着发时，后一条每次 Latency 16.2、Stall 11.5**（第 278 行）。一条 FP4 的 16×16×128 MFMA 要占矩阵管线 16 个周期，后一条得等前一条让出来。这种 Stall 是好事，说明矩阵管线被喂满了。
- **等完数据后发的第一条，Stall 接近 0**（第 277 行：每次 Latency 9.0、Stall 0.4），说明在它之前矩阵管线一直是空的。
- 第 266 行每次发射约 12 周期，比其他的多 4 个，原因没能从数据里确定。

### 5.4 LDS 写：发射比读久

```text
234: ds_write2st64_b32 v42, v48, v0 offset1:32   20480   333724    71540        0
```

每次 Latency 16.3、Stall 3.5、发射 12.8。`ds_read_b128` 的发射只有 4.2。写指令要把每个 lane 的两个 dword 从寄存器送出去，数据多，发射就久。

### 5.5 标量和向量运算：主要看 Idle

运算指令的发射基本都是 4 个周期，Stall 也很小，值得看的是 Idle，它能指出依赖和跳转：

```text
221: v_mad_i64_i32 v[46:47], s[10:11], v43, s18, v[24:25]   20480   211520   129292   283656
232: s_or_b64 s[2:3], vcc, s[2:3]                            20480    94668     3456   142824
 15: s_mov_b64 s[4:5], -1                                       64      280       12     1008
```

- 第 221 行是拷贝循环的第一条，每次 Idle 13.9。8 圈里有 7 圈是从循环末尾跳回来的，跳转之后要重新取指令，每次大约空 16 周期。
- 第 232 行每次 Idle 7.0。`s_or_b64` 要读 `vcc`，而 `vcc` 是前面的向量指令 `v_cmp_lt_u32` 写的，要等它写好。
- 第 15 行是开头 `s_cbranch_scc1` 跳过去后的第一条，每次 Idle 约 16（1,008 ÷ 64），也是跳转的代价。

### 5.6 分支：代价在下一条

分支指令自己发射 4 周期、Stall 0。跳转的代价不在它身上，而在跳过去之后那条指令的 Idle 上（见 5.5）。wave 记录里类型 7 表示跳了，类型 8 表示没跳，可以用来数循环跑了几圈（见第 4 节）。

### 5.7 `s_endpgm`：例外

```text
542: s_endpgm                                    64    34304        0        0
```

它是 wave 的最后一条，Latency 是从这条指令到 wave 真正结束的时间，这里每次约 536 周期。它不是在发射，不要和其他指令的「发射」放在一起比。

## 6. 读表的步骤

1. **先算三类时间的比例**（3.3 节），知道 wave 主要是在等、在发，还是在空着。v0 是 85.6% 在等。
2. **按 Stall 排序，看它落在哪类指令上。** 不同的位置，改法不同：
   - 落在 `s_waitcnt` 上：访存延迟没藏住。要么早点发 load，要么一次多发几条再一起等。v0 的拷贝循环就属于这种。
   - 落在访存指令自己身上：访存单元接不住。可以少发几条、每条发宽一点（比如 `dwordx4`），或者把它们错开。
   - 落在连着发的 MFMA 上：矩阵管线满了，这正是想要的状态。
   - 落在 `s_barrier` 上：同一个 workgroup 里的 wave 进度不一样，或者其中一个在等访存。
3. **再看 Idle 大的地方**：是不是依赖离得太近，或者跳转太多。RCV 的热点图只统计 Latency，不含 Idle，所以 Idle 要单独看。
4. **不要只看 `s_waitcnt` 的数大，就断定硬件空着。** 硬件有没有活干，要看那段时间同一个 SIMD 上的其他 wave 有没有在发指令，这需要看时间线：RCV 的 Compute Unit 和 Utilization 视图（Utilization 视图不显示 `s_waitcnt` 这类指令，也不显示 stall），或者 `lab/att_summary.py` 算的 MFMA 忙闲比例。

## 7. 参考

- ROCprofiler-SDK：[Using thread trace](https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/how-to/using-thread-trace.html)，`stats_*.csv` 各列的定义。
- AMD ROCm 博客：[Thread Trace Part 1: ROCprof Compute Viewer](https://rocm.blogs.amd.com/software-tools-optimization/thread-trace/README.html)，Stall 和 Idle 的区别、指令类别、RCV 各视图。
- AMD ROCm 博客：[Occupancy Math on the AMD MI355X GPU (CDNA4)](https://rocm.blogs.amd.com/software-tools-optimization/occupancy-math-mi355x/README.html)，寄存器堆、槽位、LDS 的大小和 occupancy 的算法。
- ROCm Compute Profiler：[Pipeline descriptions](https://rocm.docs.amd.com/projects/rocprofiler-compute/en/latest/conceptual/cdna/pipeline-descriptions.html)，调度器每个周期怎么挑 wave、每类指令最多发几条。
- ROCprofiler-SDK：[Analyzing PC sampling data on CDNA3 and CDNA4](https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/how-to/cdna3-cdna4-pc-sampling.html)，wave 发不出指令的各种原因，以及两条 VALU 一起发的情况。
- 本仓库：[`profiling_guide.md`](./profiling_guide.md) 第 7 节（ATT 的采集和输出文件），[`naive_gemm_lab.md`](./naive_gemm_lab.md) 第 2.5 节（v0 的 ATT 汇总）。
