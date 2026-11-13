# 用 profile 优化 HIP kernel：从测时间到看每条指令

这份指南回答 `todel.txt` 里的问题：优化一个 HIP kernel 时先做什么、后做什么，要装哪些工具，每种 profile 怎么看，看到以后怎么改。例子是 MBS 的 `gemm_a4w4_wg_kernel<16, 4, 4>`（256×256 tile）。文中的命令都在这台机器上跑过，数字都是 M = N = 8192、K = 5120 的实测值。

测量环境：AMD Instinct MI350 OAM（即 MI350X，gfx950，256 个 CU，功耗上限 1000 W，最高时钟 2.2 GHz），ROCm 7.2.3，rocprofv3 1.3.2。用 `HIP_VISIBLE_DEVICES=4` 跑，对应 amd-smi 的 7 号卡。这台机器不是 MI355X（1400 W、2.4 GHz），算峰值时要用 MI350X 的数。

## 0. 先说结论

1. **顺序**：测准时间 → 编译期体检 → 算上限 → 硬件计数器 → 指令级 trace（ATT）→ 只改一处，再回到第一步。前一步的数不准，后一步的结论就不可信。
2. **“MFMA 只有 40%”是两个数乘起来的**：实测 3.83 PFLOPS，除以规格峰值 9.2 PFLOPS 等于 41%。它等于 MFMA 管线忙的时间比例（61%）乘以时钟比（68%）：功耗顶在 1000 W，时钟只有 1.49 GHz，而规格峰值是按 2.2 GHz 算的。
3. **数据跟得上**：HBM 带宽只用了 33%。K 循环里 MFMA 空着的时间中，wave 在 `s_waitcnt` 上等数据的只占 1%。
4. **MFMA 空着，是因为 MFMA 发不出来**：K 循环里 MFMA 管线空着的时间占 wave 寿命的 19.6%。这段时间里，30% 是 wave 在发一串标量指令（给 load 算地址和描述符），30% 是 MFMA 自己发不出去（多半是两条 MFMA 紧挨着也要隔 20 周期），16% 是在发 load。
5. **epilogue 占 15%**：每换一次 tile 约 9,600 周期，主要花在写回 C 时被堵，以及从累加器读出结果、再把累加器清零。

## 1. 工具地图：每一层回答什么

![图 1：优化一个 HIP kernel 的六步](./images/profiling_workflow.svg)

图 1：优化的六步。每一步回答一个问题，右列是 MBS 的结果。改完代码回到第 ① 步重新测。

| 层次 | 工具 | 回答什么 | 这台机器上 |
|---|---|---|---|
| 模型、端到端 | `torch.profiler` + TraceLens | 一次推理里哪些 kernel 占时间、离 roofline 多远 | TraceLens 已装 |
| 一次运行的时间线 | `rocprofv3 --kernel-trace --stats` | 每个 kernel 多久、调了几次、kernel 之间有没有空档 | 自带 |
| 一个 kernel 的硬件计数器 | `rocprofv3 --pmc`、`rocprof-compute` | 哪个部件在忙、命中率、带宽、roofline | rocprofv3 自带，rocprof-compute 要装 |
| 指令级 | `rocprofv3 --att` + 解码器、RCV | 每条指令什么时候发、等了多久、卡在哪 | 解码器要装，RCV 装在自己电脑上 |
| 编译期 | `hipcc -S`、`-Rpass-analysis`、`-gline-tables-only` | 寄存器、spill、生成了什么指令 | 自带 |
| 自己插桩 | `-DMBS_TIMING=1`（`s_memtime`） | 某一段代码平均花多少周期 | 代码里已有 |

从上往下，看得越来越细，开销也越来越大。先用上层找到要优化的 kernel 和 shape，再往下找原因。别一上来就看指令。

## 2. 要装什么

### 2.1 已经有的

ROCm 7.2.3 自带：

- `rocprofv3`：kernel trace、硬件计数器、ATT 采集；
- `rocprofv3-avail`：查有哪些计数器、计数器的公式；
- `amd-smi`：看时钟、功耗、温度、利用率。

笔记里的 `rocprof-compute` 在 `/home/hangy/rocprof-compute-build` 下，这台机器上没有这个目录。

### 2.2 ATT 解码器（要看指令级 trace 就必须装）

`rocprofv3 --att` 采到的是二进制的 `.att`，要靠 ROCprof Trace Decoder（一个 `.so`）解成 JSON。ROCm 7.13 起它跟着 rocprofv3 一起装，7.2 要自己装。不装的话 `--att` 只会报错，`asm_vs_wg_kernel.md` 里记的就是这个问题。

不需要 root 的装法（本文就是这样装的）：

```bash
mkdir -p ~/tools/att-decoder && cd ~/tools/att-decoder
curl -L -o dec.tgz https://github.com/ROCm/rocprof-trace-decoder/releases/download/0.1.6/rocprof-trace-decoder-manylinux-2.28-0.1.6-Linux.tar.gz
tar xzf dec.tgz
ls rocprof-trace-decoder-manylinux-2.28-0.1.6-Linux/opt/rocm/lib/librocprof-trace-decoder.so
```

用的时候给 `rocprofv3` 加 `--att-library-path <这个 lib 目录>`，或者设环境变量 `ROCPROF_ATT_LIBRARY_PATH`。有 root 的话，也可以装 release 里的 `rocprof-trace-decoder-ubuntu-22.04-0.1.6-Linux.deb`，库会装到 `/opt/rocm/lib`，之后就不用再指定路径。

### 2.3 rocprof-compute（计数器整理成表，带 roofline）

apt 源里有 3.4.0，和 ROCm 7.2.3 配套，就是笔记里用的版本：

```bash
sudo apt install rocprofiler-compute
python3 -m venv --system-site-packages ~/venv-rpc
~/venv-rpc/bin/pip install -r /opt/rocm/libexec/rocprofiler-compute/requirements.txt "pandas<3"
~/venv-rpc/bin/python /opt/rocm/bin/rocprof-compute --version
```

没有 root 也能用：`apt-get download rocprofiler-compute`，再用 `dpkg -x rocprofiler-compute_*.deb ~/rpc` 解开，用 `~/rpc/opt/rocm-7.2.3/bin/rocprof-compute`。本文就是这样用的。

在这台机器上会碰到几个坑：

- **pandas 版本**：系统里的 pandas 是 3.0.6，rocprof-compute 3.4.0 不兼容，会报 `merge on str and int64 columns for key 'Agent_Id'` 和 `Invalid value '0.0' for dtype 'str'`。解决办法是建一个带 `--system-site-packages` 的 venv，在里面装 `pandas<3`。venv 里的 pandas 2.3.3 会盖住系统的那份；被测的 `python bench_one.py` 不在 venv 的 PATH 里，仍然用系统 python，torch 和 aiter 不受影响。笔记里那句 “python deps live in a --system-site-packages venv” 说的就是这件事。
- **`-k` 和 roofline 图**：`-k gemm_a4w4_wg_kernel` 采集没问题，但最后画 roofline PDF 时会报 “kernels requested with `--kernel` missing”，因为这一步要求完整的 kernel 名。不影响 `analyze`。
- **寄存器数**：rocprof-compute 显示的 VGPR 128、AGPR 384 不对，以编译器报告为准（见 4.1）。
- **耗时**：每采一轮计数器，就要把整个程序重跑一遍，完整 profile 在这里跑了 3 分 16 秒。所以被测程序要短，`bench_one.py` 跑 5 次就够了。

### 2.4 ROCprof Compute Viewer（RCV，看 ATT 的图形界面）

RCV 在 Windows 和 macOS 上有安装包，在 GitHub 的 [ROCm/rocprof-compute-viewer](https://github.com/ROCm/rocprof-compute-viewer) release 页面下载，最新版是 0.2.1。Linux 上要用 Qt 从源码编译，Ubuntu 22.04 官方只标为部分支持。最省事的办法是把 `ui_output_agent_*_dispatch_*` 目录打包拷回自己的电脑，在 RCV 里选 Import > Rocprofv3 UI Output 打开。

### 2.5 TraceLens

TraceLens 已经装好了，版本 0.1.0.dev，命令是 `TraceLens_generate_perf_report_pytorch` 和 `TraceLens_generate_perf_report_rocprof`。quant-perf 固定用 v1.0.0：

```bash
pip install "git+https://github.com/AMD-AGI/TraceLens.git@v1.0.0"
```

### 2.6 可选

- **Perfetto**：浏览器打开 <https://ui.perfetto.dev>，看 `rocprofv3 --output-format pftrace` 生成的时间线，适合找 kernel 之间的空档。
- **rocprof-sys**：CPU 和 GPU 放在一起的系统级时间线，排查 host 端问题（比如 launch 太慢、同步太多）时用。

## 3. 第 1 步：把时间测准

优化的节奏是改一处、测一次、比一比。要看清 2% 的提升，测量误差就得远小于 2%。在 GPU 上，最大的误差来源是时钟。

### 3.1 测得太短会偏慢

同一个 kernel，只改计时的次数（每次都先热身 10 次）：

| `bench_one.py 8192 8192 5120 <次数>` | 每次耗时 | TFLOPS |
|---|---:|---:|
| 50 次 | 209.6 µs | 3,279 |
| 500 次 | 181.9 µs | 3,777 |
| 5,000 次 | 179.4 µs | 3,830 |

原因在 amd-smi 里看得很清楚。连续跑 8 秒时每秒采一次：

```text
 TIMESTAMP  GPU  XCP  POWER  PWR_CAP   GPU_T   MEM_T   GFX_CLK   GFX%
1794324076    7    0  264 W   1000 W   54 °C   42 °C   485 MHz    1 %
1794324077    7    0  990 W   1000 W   55 °C   46 °C  1505 MHz   99 %
1794324078    7    0 1000 W   1000 W   56 °C   48 °C  1494 MHz  100 %
1794324083    7    0 1000 W   1000 W   59 °C   51 °C  1485 MHz  100 %
```

命令是 `amd-smi monitor -g 7 -u -p -t -w 1 -i 22`。从中可以读出两件事：

- 空闲时时钟只有 94 MHz，开始算以后要一段时间才升上来。计时 50 次只有约 10 ms，大部分时间时钟还没到位，所以偏慢 17%。
- 稳态下功耗一直顶在 1000 W，时钟只有 1.49 GHz，离最高的 2.2 GHz 很远。这个 kernel 是功耗受限的：同样的周期数要花多少时间，取决于功耗允许跑多快。所以改完代码要同时看周期数和时间。

### 3.2 profiler 里的时间不能和正常运行比

采计数器时，rocprofv3 一次只放一个 dispatch，前后有停顿，GPU 还没撞到功耗墙。这时单次是 176 µs，按周期换算时钟约 1.62 GHz；rocprof-compute 里是 167 µs，约 1.74 GHz。而正常连续跑是 179 µs，时钟 1.49 GHz。

所以在 profiler 里比较两个版本，要比周期数，不要比微秒。下面这些都是按周期计的，不受时钟影响：

- `GRBM_GUI_ACTIVE / 8`（8 个 XCD 各有一个 GRBM）；
- rocprof-compute 的 `Kernel Time (Cycles)`；
- ATT 里的所有时间。

### 3.3 测量规则

- **用同一块空闲的卡。** 先用 `amd-smi monitor -u -p` 看 GFX% 和功耗。`HIP_VISIBLE_DEVICES` 的编号和 amd-smi 的不一样，这台机器上 HIP 的 4 号是 amd-smi 的 7 号。不确定的话，跑起来看 amd-smi 里哪块卡变忙。
- **先热身，计时要够长。** 每个版本先热身，计时段至少几百毫秒，比如 `bench_one.py` 跑 5,000 次。
- **两个版本交替测。** 按 A B A B 的顺序跑几轮，看差值稳不稳定。差值比来回的波动还小，就当没变。
- **注意 aiter 的 `perftest`。** `test_gemm_a4w4.py` 用的 `perftest` 默认热身 2 次、计时 101 次，还会轮换多份输入，避免命中 cache。它适合在同一进程里横向比较，但单次测量只有 20 ms 左右，同样受时钟爬升影响，排在前面测的 kernel 会吃亏。

## 4. 第 2 步：编译期体检（不需要 GPU）

### 4.1 寄存器、spill、occupancy：寄存器用满没有

```bash
cd mbs
hipcc -O3 -std=c++20 -fPIC --offload-arch=gfx950 -D__HIP_PLATFORM_AMD__=1 -D__Float4_e2m1fn_x2 -Ikernel \
      --cuda-device-only -c -o /dev/null -Rpass-analysis=kernel-resource-usage kernel/gemm_a4w4.hip 2>&1 \
  | grep -A10 "Function Name: .*gemm_a4w4_wg_kernelILi16ELi4ELi4E"
```

```text
remark: Function Name: _ZN12_GLOBAL__N_119gemm_a4w4_wg_kernelILi16ELi4ELi4EEEvPKhS2_S2_S2_P14__hip_bfloat16iiii
remark:     TotalSGPRs: 106
remark:     VGPRs: 256
remark:     AGPRs: 256
remark:     ScratchSize [bytes/lane]: 0
remark:     Dynamic Stack: False
remark:     Occupancy [waves/SIMD]: 1
remark:     SGPRs Spill: 18
remark:     VGPRs Spill: 0
remark:     LDS Size [bytes/block]: 139264
```

怎么读：

- **寄存器用满了吗？** 用满了。gfx950 上每个 lane 最多 512 个寄存器（VGPR 和 AGPR 合计），这里 256 + 256 = 512，所以每个 SIMD 只能放 1 个 wave（Occupancy 1）。对 MBS 来说这是有意的：256 个 AGPR 放累加器，256 个 VGPR 放双缓冲的 A、B 片段和地址。
- **有没有 spill？** `VGPRs Spill: 0` 而且 `ScratchSize` 为 0，才算没有 spill。`SGPRs Spill: 18` 是 SGPR 溢出到 VGPR 的某个 lane 里（`v_writelane` / `v_readlane`），不碰显存，代价很小。VGPR 一旦 spill，代码里就会出现 `scratch_load` / `scratch_store`，在 ATT 里也看得到。
- **spill 发生在哪？** 再加 `-Rpass-missed=regalloc`，编译器会报出 spill 和 reload 所在的源码行。
- **LDS 够不够？** LDS 用了 139,264 B（4 个槽位 × 34,816 B），一个 CU 一共 160 KB，所以一个 CU 只能放 1 个 workgroup。
- **`.s` 里也有：** 用 `-S` 或 `--save-temps` 生成的 `.s` 元数据给的是同样的信息：`.vgpr_count: 512` 是 VGPR 和 AGPR 的合计，`.agpr_count: 256`，`.amdhsa_accum_offset 256` 表示 AGPR 从第 256 个寄存器开始，`.vgpr_spill_count: 0`，`.sgpr_spill_count: 18`，`.private_segment_fixed_size: 0`。
- **别信 profiler 里的寄存器数。** rocprofv3 的 CSV 和 rocprof-compute 第 7.1 节都显示 VGPR 128、AGPR 384。总数 512 是对的，但在 gfx950 上拆分错了。

### 4.2 看生成的指令

```bash
hipcc -O3 -std=c++20 -fPIC --offload-arch=gfx950 -D__HIP_PLATFORM_AMD__=1 -D__Float4_e2m1fn_x2 -Ikernel \
      --cuda-device-only -S -o /tmp/gemm_a4w4.s kernel/gemm_a4w4.hip
```

从 `_ZN12_GLOBAL__N_119gemm_a4w4_wg_kernelILi16ELi4ELi4E...:` 这一行到 `.Lfunc_end` 之间就是主 kernel，一共 2,955 条指令，其中 512 条 MFMA（K 循环展开了 RING = 4 个 stage，每个 stage 128 条）。

看汇编主要是核对“生成的和想的是否一致”：每个 stage 有多少条非 MFMA 指令，`s_waitcnt` 的计数对不对，有没有意外的 `scratch_` 指令，load 的寻址方式对不对。例如数一下 load 的 soffset：

| load | 条数（4 个 stage） | soffset |
|---|---:|---|
| `buffer_load_dwordx4 ... lds`（A 数据 DMA） | 48 | 0 |
| `buffer_load_dwordx4`（B 片段） | 40 | 0 |
| `buffer_load_dword ... lds`（A scale DMA） | 12 | SGPR |
| `buffer_load_dword`（B scale） | 10 | SGPR |

这张表和后面的计数器对得上：soffset 用 SGPR 时，SQ 发一条 VMEM 指令要 2 个周期，否则 1 个周期（`asm_vs_wg_kernel.md` 第 6 节量过）。每个 stage 有 16 条 dwordx4 和 4 条 dword，平均 (16 × 1 + 4 × 2) / 20 = 1.2 周期/条；计数器给出的 `SQ_INST_CYCLES_VMEM_RD / SQ_INSTS_VMEM_RD` = 2,002,944 / 1,669,120，也是 1.20。

ATT 里数出来，一个稳态 stage 有 258 条指令：128 条 MFMA、40 条 `ds_read`、20 条 `buffer_load`、约 60 条标量指令、约 10 条 VALU、2 条 `s_waitcnt`、1 条 `s_barrier`。

### 4.3 让指令能对回源码：用 `-gline-tables-only`，不要用 `-g`

ATT 和 RCV 能把每条指令标回源码行，前提是编译时带了行号信息。三种编法比下来：

| 编译选项 | 主 kernel 指令数 | scratch 指令 | scratch 大小 | 计时（5,000 次） |
|---|---:|---:|---:|---:|
| 不加调试信息 | 2,955 | 0 | 0 | 179.4–180.0 µs |
| `-gline-tables-only` | 2,955，逐条相同 | 0 | 0 | 179.3 µs |
| `-g` | 2,968 | 8 | 20 B/lane | 179.9 µs |

`-g` 要保留变量的位置信息，会改变寄存器分配：从第 353 条指令起寄存器就不一样了，还多了 8 条 scratch 指令。这次它们碰巧不在热循环里（每个 wave 每个 kernel 写 4 次，每个 tile 读 4 次），所以速度没变，但不能指望每次都这样。`-gline-tables-only` 只多了行号表，指令逐条相同，所以平时就可以一直带着它编：

```bash
make gemm HIPCC="hipcc -gline-tables-only"
```

只想另编一份、不动 `build/`，可以加 `BUILD_DIR=/tmp/mbs_glt`，然后用 `PYTHONPATH=/tmp/mbs_glt` 跑。注意 `bench_one.py` 会优先导入 `mbs/build` 里的模块，所以 `mbs/build` 里不能有同名模块。

## 5. 第 3 步：算上限，判断受算力限制还是带宽限制

### 5.1 自己算

- **计算量**：2MNK = 687.2 GFLOP。换成 MFMA 是 M·N·K / (16·16·128) = 10,485,760 条，分给 1,024 个 wave，每个 wave 10,240 条。
- **MFMA 最少要多少时间**：一条 FP4 16×16×128 的 MFMA 占管线 16 个周期（rocprof-compute 11.2.9 实测 16.0），所以每个 wave 最少 163,840 个周期。按 1.49 GHz 算是 110 µs，实测 179 µs，也就是 MFMA 管线只忙了 61% 的时间。
- **规格峰值**：256 个 CU × 每 CU 每周期 16,384 FLOP（4 个 SIMD，每个 SIMD 每 16 周期做 65,536 FLOP）× 2.2 GHz = 9.2 PFLOPS。实测 3.83 PFLOPS 是它的 41%，而 41% ≈ 61% × (1.49 / 2.2)。
- **最少流量**：A、B 各 21 MB，scale 共 2.6 MB，C 是 134 MB（bf16），合计约 179 MB。按 4.84 TB/s 算只要 37 µs，远小于 179 µs，所以带宽不是上限。

### 5.2 用 rocprof-compute 画 roofline

```bash
RPC="$HOME/venv-rpc/bin/python /opt/rocm/bin/rocprof-compute"
$RPC profile --name mbs_8k --roof-only -R FP4 -k gemm_a4w4_wg_kernel -p /tmp/rpc/mbs_8k_roof \
    -- python bench_one.py 8192 8192 5120 20
$RPC analyze -p /tmp/rpc/mbs_8k_roof -b 4
```

第一次 profile 会跑一组微基准，量出这块卡实际能达到的峰值，结果存在 `roofline.csv`。把它拷进新 workload 的目录，下次就不用再跑：

| 项目 | 实测峰值 |
|---|---:|
| FP4 MFMA | 8.09 PFLOPS |
| HBM | 4.84 TB/s |
| MALL（Infinity Cache） | 6.97 TB/s |
| L2 | 31.5 TB/s |
| L1 | 34.8 TB/s |
| LDS | 62.2 TB/s |

MBS 的算术强度：按 HBM 流量算是 2,559 FLOP/B，按 L2 是 798，按 L1 是 345。拐点（峰值算力 ÷ 峰值带宽）在 HBM 上是 8.09 P / 4.84 T = 1,672 FLOP/B，在 L2 上是 257。MBS 在两条线上都在拐点右边，所以受算力限制：要变快，就得让 MFMA 更忙；单纯减少访存，不会直接变快。

实际 HBM 流量是 268 MB，比上面算的最少 179 MB 多。原因是 8 个 XCD 各有自己的 L2，同一块 A、B 会被多个 XCD 各读一遍。

## 6. 第 4 步：硬件计数器

### 6.1 rocprofv3 --pmc

查计数器：

```bash
rocprofv3-avail list --pmc                 # gfx950 上有几百个，一行两个名字
rocprofv3-avail info --pmc MfmaUtil        # 看派生指标的公式
# Expression: reduce(SQ_VALU_MFMA_BUSY_CYCLES,sum)/(reduce(GRBM_GUI_ACTIVE,max)*SIMD_NUM)*100
```

一轮能同时采的计数器有限，所以写一个输入文件分几轮采：`scripts/pmc_mbs.txt` 里每行 `pmc:` 是一轮，rocprofv3 会把程序跑三遍。

```bash
cd mbs
HIP_VISIBLE_DEVICES=4 rocprofv3 -i scripts/pmc_mbs.txt --kernel-include-regex gemm_a4w4_wg_kernel \
    --output-format csv -d /tmp/mbs_pmc -o pmc -- python bench_one.py 8192 8192 5120 5
python3 scripts/pmc_summary.py /tmp/mbs_pmc 8192 8192 5120
```

每一轮的输出目录（`pmc_1/`、`pmc_2/`、`pmc_3/`）里有 `pmc_counter_collection.csv`，一个 dispatch 的一个计数器占一行。每个值都是整块 GPU 的总和。列里还有 `Grid_Size`、`LDS_Block_Size`、`Scratch_Size` 和 dispatch 的起止时间。

| 计数器 | MBS 的值 | 怎么读 |
|---|---:|---|
| `SQ_INSTS_MFMA` | 10,485,760 | 和理论值完全一样，没多也没少 |
| `SQ_INSTS_VALU_MFMA_MOPS_F6F4` | 1,342,177,280 | 等于 2MNK / 512：所有 MFMA 都按 FP4 算 |
| `MfmaUtil` | 54.9% | MFMA 管线忙的周期 ÷（GPU 忙的周期 × 1,024 个 SIMD） |
| `GRBM_GUI_ACTIVE` | 2,281,780 | 8 个 XCD 相加；÷ 8 = 285,222 周期，再 ÷ 176 µs，时钟 1.62 GHz |
| `SQ_WAVES`、`SQ_WAVE_CYCLES` | 1,024、65,989,844 | 后者以 4 周期为单位；× 4 ÷ 1,024 = 每个 wave 活 257,773 周期 |
| `SQ_WAIT_INST_ANY` | 18,700,501 | 也以 4 周期为单位：每个 wave 有 73,049 周期在等发射 |
| `SQ_LDS_BANK_CONFLICT` | 0 | LDS 的 swizzle 没问题 |
| `TCC_HIT_sum`、`TCC_MISS_sum` | 10.36 M、3.15 M | L2 命中 76.7% |
| `TCC_EA0_WRREQ_sum` | 2,097,152 | × 64 B = 128 MiB，正好是 C 的大小，没有多写 |
| `SQ_INSTS_VMEM_WR` | 131,072 | = 1,024 个 wave × 4 个 tile × 32 条 store |
| `SQ_INST_CYCLES_VMEM_RD` ÷ `SQ_INSTS_VMEM_RD` | 1.20 | 和 4.2 里按 soffset 算出的 1.2 一致 |

先用已知量核对，比如 MFMA 条数、写回字节数、store 条数，都对得上了，再去信其他数字。

### 6.2 rocprof-compute：把计数器整理成表

```bash
$RPC profile --name mbs_8k -R FP4 -k gemm_a4w4_wg_kernel -p /tmp/rpc/mbs_8k \
    -- python bench_one.py 8192 8192 5120 5
$RPC analyze -p /tmp/rpc/mbs_8k -b 2 4 7 11 12      # 命令行表格
$RPC analyze -p /tmp/rpc/mbs_8k --gui              # 网页版，默认端口 8050（远程机器要做端口转发）
```

`-b` 选要看的分析块：2 是 System Speed-of-Light，4 是 Roofline，7 是 Wavefront，11 是 Compute Pipeline，12 是 LDS。用 `--list-metrics gfx950` 可以列出全部。MBS 的关键数字：

| 编号 | 指标 | MBS | 怎么读 |
|---|---|---:|---|
| 2.1.7 | MFMA FLOPs (F6F4) | 4.12 PF | profile 模式下时钟高（1.74 GHz），比正常运行快 |
| 2.1.12 | MFMA Utilization | 56.5% | 和 `MfmaUtil` 是同一个公式 |
| 11.2.9 | MFMA Instruction Cycles | 16.0 | 一条 FP4 MFMA 占管线 16 周期 |
| 2.1.11 | VALU Utilization | 35.9% | VALU 不紧张 |
| 2.1.13 | VMEM Utilization | 2.9% | 访存单元平均很闲；后面说的“被堵”是扎堆造成的，不是总量太多 |
| 2.1.16 | IPC | 0.41 | 平均每周期发 0.41 条指令 |
| 4.1.10 | HBM Bandwidth | 1.61 TB/s | 实测峰值的 33% |
| 2.1.22 | L2 Cache Hit Rate | 76.7% | |
| 12.1.3 | LDS Bank Conflict Rate | 0 | |
| 7.1.9 | Scratch Allocation | 0 | |
| 7.2.4–7.2.6 | Dependency Wait / Issue Wait / Active | 14% / 29% / 57% | 每个 wave 分别约 36.9k / 74.0k / 148.4k 周期 |

wavefront 那三项的含义：

- **Dependency Wait**：在等依赖，比如 `s_waitcnt` 等访存结果、`s_barrier` 等其他 wave。
- **Issue Wait**：指令已经准备好了，但要用的执行单元正忙，这里主要是 MFMA 管线忙。对一个受 MFMA 限制的 kernel，这是“好的等待”。
- **Active**：正在发指令。

Speed-of-Light 的读法：找离峰值最近的那一项，它就是瓶颈。MBS 里 MFMA 用到 56%，其他各项都远低于峰值，所以是 MFMA 受限。如果是 HBM 或 L2 接近峰值，那就是带宽受限，应该去减少流量。

### 6.3 几个“MFMA 利用率”为什么不一样

| 数字 | 从哪来 | 分母 |
|---|---|---|
| 54.9% | rocprofv3 `MfmaUtil` | 整个 kernel 期间 GPU 忙的周期，包括开头派发、结尾等最后一批 wave |
| 56.5% | rocprof-compute 2.1.12 | 同上，是另一次运行 |
| 62.9% | ATT，一个 CU 上的 4 个 wave | 这几个 wave 自己的寿命 |
| 41% | 实测 TFLOPS ÷ 规格峰值 | 时间 × 2.2 GHz，把降频也算进去了 |

这几个数都没错，只是回答的问题不同。看优化效果时，固定用同一个数比。

## 7. 第 5 步：ATT，看每一条指令

### 7.1 采集

```bash
cd mbs
make gemm HIPCC="hipcc -gline-tables-only"
DEC=~/tools/att-decoder/rocprof-trace-decoder-manylinux-2.28-0.1.6-Linux/opt/rocm/lib
HIP_VISIBLE_DEVICES=4 rocprofv3 --att --att-library-path $DEC \
    --kernel-include-regex gemm_a4w4_wg_kernel --kernel-iteration-range 6 \
    -d /tmp/mbs_att -- python bench_one.py 8192 8192 5120 1
```

- **只追一次 dispatch**：`--kernel-iteration-range 6` 表示只追踪匹配到的第 6 次 dispatch，跳过前面的热身。一次追很多个，每个都要解码一份。
- **只追一个 CU**：默认只追 SE 0 上的 1 号 CU，四个 SIMD 都追（参数 `--att-target-cu 1`、`--att-shader-engine-mask 0x1`）。MBS 每个 CU 只有一个 workgroup，所以追到的就是这个 workgroup 的 4 个 wave，每个 SIMD 一个，每个 wave 算 4 个 tile。
- **开销**：这一次跑了 8 秒，输出 4.7 MB。
- **笔记里的例子**：笔记里追 `mbs_factor_experiment_kernel` 那条命令也是这么用的，换成自己的 kernel 名和脚本就行。

### 7.2 输出文件和字段

```text
/tmp/mbs_att/
├── stats_ui_output_agent_<A>_dispatch_<D>.csv   每条指令的汇总，和 code.json 是同一份数
├── ui_output_agent_<A>_dispatch_<D>/            给 RCV 和脚本用
│   ├── code.json                每条 ISA 指令：源码行、执行次数、Latency、Stall、Idle
│   ├── se0_sm0_sl0_wv0.json     一个 wave 的逐条指令时间线（sm 是 SIMD，wv 是 wave）
│   ├── se0_sm1_sl0_wv0.json …  另外三个 SIMD 上的 wave
│   ├── occupancy.json、wstates*.json、realtime.json
│   └── snapshots.json、source_*.hip   源码快照
└── <hostname>/                  原始的 .att 和 code object（.out）
```

`code.json` 的表头是 `ISA, _, LineNumber, Source, Codeobj, Vaddr, Hit, Latency, Stall, Idle`。定义见 AMD 的博客 [Thread Trace Part 1](https://rocm.blogs.amd.com/software-tools-optimization/thread-trace/README.html)：

- **Hit**：被追踪的这几个 wave 一共执行了它几次。
- **Stall**：wave 已经准备好发这条指令，但它要用的管线接不住，比如 MFMA 管线正忙、访存队列满了。
- **Idle**：wave 还没准备好发下一条，比如在等前一条 VALU 的结果。
- **Latency**：这条指令占掉的周期，即发射加上 stall。
- **Source**：内联调用链，最里层在前，比如 `gemm_a4w4.hip:306 -> gemm_a4w4.hip:730 -> …`。统计热点时要按最里层那一行；按整条链统计，所有内联进来的代码都会算到最外层的 894 行（那个 `static_for<RING>` 循环）上。

每个 wave 的文件是 `{"begin": …, "end": …, "instructions": [[cycle, type, stall, duration, code 下标], …]}`，含义是：

- wave 在第 `cycle` 周期开始尝试发这条指令，等了 `stall` 周期才发出去，`duration` 周期后开始尝试下一条；
- 时间单位是周期，这份 trace 里的时间都是 4 的倍数。

`type` 的编号是从数据里对出来的：1 是 SMEM，2 是 SALU，3 是 VMEM，4 是 scratch，5 是 LDS，6 是 VALU（MFMA 也算在里面），7 和 8 是分支，9 是 `s_waitcnt`、`s_barrier`、`s_nop` 这类不进执行单元的指令，11 是 `s_endpgm`。

已有的 `scripts/parse_att_trace.py` 可以把 `code.json` 打印成表。

### 7.3 用 RCV 看

把 `ui_output_*` 目录拷回自己电脑，用 Import > Rocprofv3 UI Output 打开。常用的几个视图：

- **Compute Unit**：每个 wave 一行时间线，看哪段在发、哪段 stall、哪段 idle；
- **Utilization**：VALU、MFMA、VMEM、LDS 各条管线忙不忙，空白就是空泡；
- **Instructions**：每条 ISA 花了多少周期；点 `s_waitcnt`，能看到它在等哪些访存；
- **Hotspot**、**Flamegraph**：按指令或源码行汇总。

看 MBS 时，在 Compute Unit 视图里拉到一个 stage，找 MFMA 之间的空白，再点空白前后的指令，看是什么。7.5 的图就是用脚本做的同一件事。

### 7.4 用脚本算一个 wave 的时间预算

```bash
UI=$(ls -d /tmp/mbs_att/ui_output_agent_*_dispatch_*)
python3 scripts/att_budget.py $UI kernel/gemm_a4w4.hip
python3 scripts/draw_profiling_figs.py --att $UI        # 重画图 2、图 3
```

`att_budget.py` 的算法：

1. 每条 MFMA 占管线 16 周期，从“它发出的时刻”和“上一条做完的时刻”中较晚的那个开始算。
2. 如果下一条 MFMA 发出时管线已经空了，中间这段就是一个空泡。
3. 空泡里的每个周期，都记到 wave 那一刻正在发、或者被堵住的那条指令上。中间的指令都发完了、MFMA 自己还发不出去，记作“下一条 MFMA 自己被卡”；那一刻 wave 没有指令在发，记作“空档”。

![图 2：一个 wave 的周期花在哪](./images/profiling_wave_budget.svg)

图 2：一个 wave 的 260,354 个周期花在哪。上面是总账；左下把 K 循环里 MFMA 空着的时间，按 wave 当时在做什么拆开；右下把一次 tile 切换拆开。

| 时间预算（每个 wave） | 周期 | 占比 |
|---|---:|---:|
| MFMA 在算 | 163,840 | 62.9% |
| K 循环里 MFMA 空闲 | 51,027 | 19.6% |
| tile 切换（epilogue 加下一个 tile 的开头） | 28,733 | 11.0% |
| 收尾（最后一个 epilogue） | 10,776 | 4.1% |
| 开头（第一条 MFMA 之前） | 5,978 | 2.3% |

K 循环里 MFMA 空着的 51,027 个周期，wave 当时在做：

| wave 在做什么 | 周期 | 占比 | 说明 |
|---|---:|---:|---|
| 发标量指令（SALU） | 15,344 | 30.1% | 发 load 之前那一串算地址和描述符的指令 |
| 下一条 MFMA 自己被卡 | 15,099 | 29.6% | 四分之三发生在两条 MFMA 紧挨着时，见 7.6 |
| 发 `buffer_load` | 7,978 | 15.6% | 一条 load 本身要发 8–16 周期，有时还被堵 |
| 空档（wave 没在发指令） | 5,733 | 11.2% | 多半是 `v_or_b32` 刚算完 LDS 地址，下一条 `ds_read` 马上要用 |
| 发 `ds_read` | 2,992 | 5.9% | |
| 等其他 wave（`s_barrier`） | 2,943 | 5.8% | barrier 的等待大部分和 MFMA 重叠了 |
| 等数据（`s_waitcnt`） | 528 | 1.0% | 预取把访存延迟全藏住了 |
| 发 VALU | 410 | 0.8% | |

一次 tile 切换平均 9,594 周期，各类指令占的周期如下：

| 指令 | 周期 | 占比 | 说明 |
|---|---:|---:|---|
| `buffer_store` | 3,480 | 36.3% | 32 条 store，其中 3,334 周期是被堵，平均每条约 104 周期 |
| `v_accvgpr_*` | 2,048 | 21.3% | 512 条：256 条读出结果，256 条为下一个 tile 清零 |
| 其他 VALU | 1,463 | 15.3% | 拼 bf16、DPP、选择、算地址 |
| `v_cvt` / `v_permlane` | 1,032 | 10.8% | |
| SALU | 492 | 5.1% | |
| 其他和空隙 | 1,078 | 11.2% | |

### 7.5 看一个 stage 的时间线：可视化地找 gap

![图 3：ATT 里的一个 stage](./images/profiling_att_stage.svg)

图 3：SIMD 0 上那个 wave 的一个稳态 stage，长度是中位数。

- **管线行**：每段第一行是 MFMA 管线，蓝块是一条 MFMA 占的 16 周期，红块是管线空着；红字是这个空泡里占得最多的一两项。
- **其他指令行**：下面几行是同一时刻发出去的其他指令。浅红是指令发射前被堵的时间，紫色是 `s_waitcnt` / `s_barrier` 等待的时间。

图里能直接看出：

- **段 0 很顺。** B 的 load 和 A 片段的 `ds_read` 均匀地插在 MFMA 之间，几乎没有空泡。
- **段 1 开头空了 100 周期。** 这里 barrier 只等了 24 周期，而且是在 MFMA 还在算的时候等的。真正让管线空着的，是 barrier 之后那一串 13 条标量指令：它们在为 stage s+2 的 A DMA 算 buffer 描述符和 LDS 地址（`m0`），算完才发出 `buffer_load ... lds`。
- **段 1 中间那个大空泡（124 周期）。** 先是 11 条标量指令在算 A scale DMA 的地址，然后发一条 `buffer_load_dword ... lds`，接着下一条 MFMA 自己又被卡了 68 周期。
- **段 1 结尾又空了 100 周期。** 其中 84 周期在发 24 条标量指令：给下一个 stage 的两组 load 算描述符（base 和 num_records），外加循环计数和槽位选择。

这就是“可视化看 gap”：先看红块在哪，再看同一时刻 wave 在发什么。

### 7.6 MBS 的结论

1. **数据跟得上。** K 循环里 `s_waitcnt` 只占 MFMA 空闲的 1%，每个 wave 528 周期。PF = 2 的预取把访存延迟藏住了，再加深预取没有用。
2. **标量指令扎堆，占 30%。** 每个 stage 约 60 条标量指令，集中在发 load 之前算描述符和地址。`asm_vs_wg_kernel.md` 第 6 节第 3 条已经处理过一部分（把每个 stage 的偏移挪进描述符基址），剩下的就是这些。
3. **MFMA 自己发不出去，占 30%。**
   - 四分之三发生在两条 MFMA 紧挨着的时候：中间什么指令都没有，发射间隔也只有约一半（52%）是 16 周期，44% 是 20 周期。每个 wave 约 1,744 次，平均每次多 6.5 周期。asm 文档里量到的“不发 load 时每个 stage 2,290 周期”（约 17.9 周期一条 MFMA）也是这个现象。
   - 另外四分之一发生在 `ds_read` 或 `buffer_load` 刚发出之后，一次 23–29 周期。
   - 原因还没查清。
4. **发 load 占 16%。** 一条 `buffer_load` 本身要发 8–16 周期，扎堆时还会被堵。
5. **空档占 11%。** 主要来自源码 662 行的 `scale_at`：
   - 先用 `v_or_b32 v0, 0x10800, v198` 算出 A scale 的 LDS 地址，紧接着 `ds_read_b32` 就要用它，于是要等 VALU 写回；
   - 0x10800 是 1 号槽 scale 区的地址（34,816 + 32,768）。1–3 号槽的 scale 区都在 64 KB 以外，放不进 `ds_read` 的 16 位立即数偏移，所以每次都要多一条 VALU。
6. **barrier 占 6%。** barrier 的等待大部分和 MFMA 重叠了，暴露出来的不多。
7. **epilogue（tile 切换加收尾）占 15%。** 写回 C 被堵，加上 512 条 AGPR 读写。

另外，ATT 的源码热点（按 stall 排序）是：

- 305–320 行的几种 `MBS_MFMA_FP4`：MFMA 在排队，说明管线是忙的，这是好事；
- 838、840 行的两条 `__builtin_amdgcn_raw_buffer_store_b128`：epilogue 写回；
- `amd_device_functions.h` 692 行：`__syncthreads()` 里的 `s_barrier`；
- 662 行：`scale_at`，即上面第 5 条。

## 8. 第 6 步：改代码，再验证

### 8.1 从 profile 到改动：MBS 接下来可以试的方向

下表都是从 trace 推出来的猜想，还没有验证。“能省的上限”是假设这部分完全消失时，每个 wave 省下的周期数（每个 wave 一共约 260k 周期）。

| 方向 | 依据 | 能省的上限 |
|---|---|---:|
| 让 A scale（还有 A 片段）的 LDS 读地址能用立即数偏移，比如把所有槽的 scale 区放到 LDS 前 64 KB，或者每两个槽共用一个基址 VGPR | 7.6 第 5 条 | 空档 5.7k，外加每个 stage 8 条 VALU |
| 把描述符和地址的标量计算提前、打散到 MFMA 之间，或者少算一些（现在每个 stage 都要为每组 load 重新算一遍 base 和 num_records） | 7.6 第 2 条 | 15.3k |
| 查清两条 MFMA 紧挨着为什么要隔 20 周期：写一个只发 MFMA 的小 kernel，换几种寄存器分配，用 ATT 量间隔；再用同样的方法看 asm kernel 的间隔分布 | 7.6 第 3 条 | 11.4k |
| epilogue 不再用 256 条 `v_accvgpr_write` 清零：下一个 tile 第一步的 MFMA 直接用 0 作累加输入（MFMA 的 C 操作数可以是常数 0，要验证 scale 版是否支持） | 7.4 tile 切换表 | 每次切换约 1k |
| epilogue 的写回和下一个 tile 的计算重叠起来 | 32 条 store 每条被堵约 104 周期 | 每次切换约 3.3k |

省下的周期不一定全部变成时间：这个 kernel 受功耗限制，少发指令可能省电、换来更高的时钟，MFMA 更密也可能更费电、让时钟变低。所以每次改完都要按第 3 步重新测时间，同时比周期数。

### 8.2 怎么验证

- **一次只改一处**，按第 3 步的规则交替测时间。在 profiler 里比较同一组数：计数器的 `MfmaUtil` 和 `GRBM_GUI_ACTIVE`，ATT 的时间预算，`att_budget.py` 每一节的数字。
- **把结果记下来。** `asm_vs_wg_kernel.md` 第 6 节就是这样一轮优化的记录，每条都对应一类证据：
  - **load 寻址**：`SQ_INST_CYCLES_VMEM_RD` 量出 SGPR soffset 让每条 VMEM 多花 1 个周期，加上 load 前那串标量指令，改完每个 stage 从 3,011 降到 2,833 周期；
  - **写回整行**：用微基准量出半行写回要约 7.5k 周期才确认，整行只要约 3.0k，和消除 spill 一起，几何平均快了约 2%；
  - **段 0 去掉 barrier**：每个 stage 少 60 周期，约 1%；
  - **`LOAD_WIN`**：load 只放在每段的前 7/8，每个 stage 快约 2%。
- **正确性要一起测。** `test_gemm_a4w4.py` 要跑过所有 shape。性能改动经常引入只在边界 shape 出现的错误。

## 9. `todel.txt` 里的问题，逐条回答

**rocprof-compute 可视化，看性能、找 bound。** 见 5.2 和 6.2。`analyze` 默认输出命令行表格和文本版 roofline；加 `--gui` 是网页版，`--tui` 是终端界面。找 bound 的方法：看 Speed-of-Light（2.1 节）里哪一项离峰值最近，再看 roofline 上的点在拐点左边还是右边。MBS 是 MFMA bound：MFMA 用到 56%，算术强度 2,559 FLOP/B，大于拐点 1,672；HBM 只用了 33%。

**rocprofv3 --att 生成 json，看指令排布，用 rocprof compute viewer 可视化。** 见第 7 节。必须先装解码器（2.2）。JSON 在 `ui_output_*` 目录里：`code.json` 是每条指令的汇总，`se*_wv*.json` 是每个 wave 的时间线。RCV 要装在自己电脑上（2.4）。在服务器上可以直接用 `scripts/att_budget.py` 和 `draw_profiling_figs.py` 看同样的东西。

**从 Python 代码看到 kernel 汇编和源码的对应。** 用 `-gline-tables-only` 编译（4.3），然后照常用 Python 跑，`rocprofv3 --att` 采下来的 `code.json` 里，每条 ISA 都带着源码行和内联链；RCV 的 Instructions 视图可以在源码和 ISA 之间来回跳。不跑 GPU 也可以看：`hipcc -S -gline-tables-only` 生成的 `.s` 里，每段指令前都有 `.loc 文件号 行号 列号`。

**观察 MFMA 的使用率。** 三个来源：`rocprofv3 --pmc MfmaUtil`（54.9%），rocprof-compute 2.1.12（56.5%），ATT 的时间预算（62.9%）。它们为什么不一样，见 6.3。

**issue 过多的加载指令会导致堆积吗？** 会。VMEM 和 LDS 指令发出去以后进队列，队列满了，下一条同类指令就发不出去，在 ATT 里表现为这条指令的 Stall。AMD 的博客里有一个 LDS FIFO 满了的例子。MBS 的情况：

- 在 K 循环里，wave 发 `buffer_load`（包括被堵的时间）占了 MFMA 空闲的 15.6%，每个 wave 约 8k 周期；
- 但 rocprof-compute 的 VMEM Utilization 只有 2.9%，说明问题是扎堆，不是总量太多。MBS 用 `LOAD_WIN` 把每段 10 条 load 摊到前 7/8 的 MFMA 之间，就是为了这个；
- 还有一种堆积是 vmcnt 按顺序退休：epilogue 的 store 和下一个 tile 的 load 共用 vmcnt，load 会排在 store 后面（`asm_vs_wg_kernel.md` 第 6 节第 1 条）。

**MFMA 调用率只有 40%？数据加载根本满足不了计算需求？优化 stage 来加速？**

- **40% 是怎么来的**：实测 TFLOPS 除以规格峰值，等于 41%。它是 MFMA 管线忙的时间比例（61%）和时钟比（1.49 / 2.2 = 68%）的乘积。
- **数据加载跟得上**：带宽只用了 33%，K 循环里等数据只占 MFMA 空闲的 1%。
- **优化 stage 有用，但方向不是加深预取**：要减少、打散每个 stage 里的非 MFMA 指令（标量地址计算、需要 VALU 算地址的 LDS 读），并查清 MFMA 紧挨着也要隔 20 周期的原因（8.1）。
- **另外 15% 在 epilogue**，那是 stage 之外的事。

**barrier 就是 `__syncthreads()` 吗？** 差不多，但不完全一样。

- `__syncthreads()` 展开是“workgroup 范围的 release fence + `s_barrier` + acquire fence”，见 `/opt/rocm/include/hip/amd_detail/amd_device_functions.h` 690–694 行。`s_barrier` 是那条硬件指令：同一个 workgroup 里的 wave 都到了，才一起往下走。fence 规定的是过 barrier 之前，哪些访存必须已经完成、对别的 wave 可见。
- 在 gfx950 上，这个 fence 编出来是 `s_waitcnt lgkmcnt(0)`，并不会把 vmcnt 清零。MBS 在 758 行调用 `__syncthreads()` 之前，还有自己的 `wait_vm_lgkm0`，两者合并以后，汇编里是 `s_waitcnt vmcnt(10) lgkmcnt(0)` 加 `s_barrier`。
- ATT 里，barrier 的 stall 就是在等最慢的那个 wave。在 MBS 里，每个 wave 过 81 次 barrier，平均每次等 43 周期，其中约 61 次超过 8 周期。这些等待大部分和还在算的 MFMA 重叠了，暴露出来的只占 MFMA 空闲的 6%。

**加载数据，系统会有等待吗？** 有两个地方会等：

- **发射时**：队列满了，load 指令自己发不出去（上面说的堆积）；
- **使用时**：load 是异步的，发出去以后 wave 继续往下走，直到碰到 `s_waitcnt vmcnt(N)` 或 `lgkmcnt(N)`。如果还没回来的访存比 N 多，就停下来等。这种等待在 rocprof-compute 里算作 Dependency Wait，在 ATT 里是那条 `s_waitcnt` 的 stall。

MBS 的 `W_START`、`W_MID` 就是这个 N（见 `stage.md`）。稳态下几乎不等，只有每个 wave 第一个 tile 开头，等第一批数据时会真的等上几千周期（时间预算里的“开头”，5,978 周期）。

**研究 rocprof。** 常用的 rocprofv3 选项：

| 选项 | 作用 |
|---|---|
| `--kernel-trace`、`--stats` | 每次 dispatch 的起止时间，以及按 kernel 的汇总（次数、平均、占比） |
| `--pmc A B C`、`-i 文件` | 采硬件计数器；文件里每行 `pmc:` 是一轮 |
| `--att`（加 `--att-library-path`） | 指令级 thread trace |
| `--kernel-include-regex`、`--kernel-exclude-regex` | 只看名字匹配的 kernel |
| `--kernel-iteration-range` | 只看第几次 dispatch |
| `-d 目录`、`-o 前缀` | 输出位置 |
| `--output-format csv / json / pftrace / rocpd` | json 给 TraceLens 用，pftrace 给 Perfetto 用 |
| `--memory-copy-trace`、`--hip-trace` | 拷贝、HIP API 的时间线 |
| `rocprofv3-avail list / info --pmc` | 查计数器和它们的公式 |

例如 `rocprofv3 --kernel-trace --stats --output-format csv -d /tmp/kt -o kt -- python bench_one.py 8192 8192 5120 20` 会给出 `kt_kernel_stats.csv`。这次运行里 GEMM 调了 30 次，占 GPU 时间的 96.5%，其余是生成随机数和量化的 kernel。

**怎么看 kernel 慢在哪里，可视化看 gap。** gap 有两种：

- **kernel 之间的空档**（launch 慢、同步、CPU 在忙）：用 `rocprofv3 --kernel-trace --output-format pftrace`，在 Perfetto 里看时间线，或者看 TraceLens 的 `gpu_timeline` 表。
- **kernel 里面 MFMA 之间的空泡**：用 ATT。在 RCV 的 Compute Unit 和 Utilization 视图里找空白，或者用 `draw_profiling_figs.py` 画出图 3 那样的图，再用 `att_budget.py` 把空泡按原因加总（图 2）。

**编译后的代码跟源码对应起来，用 -g 模式？** 用 `-gline-tables-only`，不要用 `-g`。前者的指令和正常构建逐条相同；`-g` 改了寄存器分配，还多出 scratch（见 4.3）。

**检查寄存器用满没有。** 见 4.1：用 `-Rpass-analysis=kernel-resource-usage` 或 `.s` 的元数据。MBS 是 256 VGPR + 256 AGPR，用满了，每个 SIMD 1 个 wave，没有 VGPR spill，SGPR spill 了 18 个（溢出到 VGPR 的 lane 里）。别用 profiler 显示的寄存器数。

**TraceLens 基于 rocprof 做了哪些事情，对 quant-perf 有什么用？**

- **它是一个分析库，自己不采数据。** 它读 PyTorch profiler、JAX 或 rocprofv3 的 trace，生成 Excel 报告。
- **读 rocprofv3 的 trace 时**，按 kernel 名把 kernel 分类，统计时间，算 GPU 忙和闲的时间线。这次实测：
  ```bash
  rocprofv3 --kernel-trace --memory-copy-trace --output-format json -d /tmp/tl -o run -- python bench_one.py 8192 8192 5120 20
  TraceLens_generate_perf_report_rocprof --profile_json_path /tmp/tl/run_results.json --kernel_details --short_kernel_study
  ```
  生成的 `run_perf_report.xlsx` 里有 `gpu_timeline`、`kernel_summary`、`kernel_summary_by_category`、`kernel_details` 等表，GEMM 一类占 96.3%。要注意两点：一是这个版本 `kernel_summary` 里标着 µs 的列，数值其实是 ns，大了 1000 倍；二是时间线覆盖了整个 Python 进程（含导入 aiter、量化），所以显示 GPU 空闲 99.9%。要看稳态，必须只截一段运行窗口。
- **读 PyTorch trace 时**，它知道每个 op 的 shape，能算每个 GEMM 的 FLOPs、字节数、离 roofline 多远（`Pct Roofline`），判断是算力还是带宽受限。rocprofv3 的 trace 没有这些信息。
- **在 quant-perf 里，它在 PerfOpt 阶段用：**
  1. `perfopt/collect.py` 通过推理服务的 `/start_profile`、`/stop_profile` 采一段 PyTorch trace，再用 TraceLens 的 `split_inference_trace_annotation --find-steady-state` 截出稳态的 decode 步（默认 32 步），把 CUDAGraph 捕获、热身和 prefill 都去掉；
  2. `perfopt/locate.py` 用 `TreePerfAnalyzer`，加上 vLLM、aiter、mxfp4 的“伪算子”映射，生成按 kernel 时间排序、带 roofline 标注的瓶颈表；
  3. 再交给 vendor GEMM 调优或 GEAK 去改 kernel。TraceLens 不可用时，退回到直接汇总 trace 里 kernel 时间的方式。
- **它对 quant-perf 的作用**：在模型层面告诉你优化哪个 kernel、哪个 shape 最值得，避免花力气优化一个只占 2% 端到端时间的 kernel。它管不到 kernel 内部，指令级的问题还得用第 6、7 节的工具。

## 附录 A：命令速查

```bash
cd mbs
export HIP_VISIBLE_DEVICES=4

# 1. 时间（够长）和时钟
python bench_one.py 8192 8192 5120 5000
amd-smi monitor -g 7 -u -p -t -w 1 -i 20

# 2. 编译期
hipcc -O3 -std=c++20 -fPIC --offload-arch=gfx950 -D__HIP_PLATFORM_AMD__=1 -D__Float4_e2m1fn_x2 -Ikernel \
      --cuda-device-only -c -o /dev/null -Rpass-analysis=kernel-resource-usage kernel/gemm_a4w4.hip
make gemm HIPCC="hipcc -gline-tables-only"

# 3. kernel trace
rocprofv3 --kernel-trace --stats --output-format csv -d /tmp/kt -o kt -- python bench_one.py 8192 8192 5120 20

# 4. 计数器
rocprofv3 -i scripts/pmc_mbs.txt --kernel-include-regex gemm_a4w4_wg_kernel --output-format csv \
    -d /tmp/mbs_pmc -o pmc -- python bench_one.py 8192 8192 5120 5
python3 scripts/pmc_summary.py /tmp/mbs_pmc 8192 8192 5120

# 5. rocprof-compute（RPC 见 5.2）
$RPC profile --name mbs_8k -R FP4 -k gemm_a4w4_wg_kernel -p /tmp/rpc/mbs_8k -- python bench_one.py 8192 8192 5120 5
$RPC analyze -p /tmp/rpc/mbs_8k -b 2 4 7 11 12

# 6. ATT
rocprofv3 --att --att-library-path $DEC --kernel-include-regex gemm_a4w4_wg_kernel --kernel-iteration-range 6 \
    -d /tmp/mbs_att -- python bench_one.py 8192 8192 5120 1
UI=$(ls -d /tmp/mbs_att/ui_output_agent_*_dispatch_*)
python3 scripts/att_budget.py $UI kernel/gemm_a4w4.hip
python3 scripts/draw_profiling_figs.py --att $UI

# 7. 插桩计时（插桩版会慢约 10%，只在桶之间比较）
make gemm BUILD_DIR=/tmp/mbs_timing HIPCC="hipcc -DMBS_TIMING=1"
python3 scripts/mbs_timing.py /tmp/mbs_timing 8192 8192 5120
```

## 附录 B：本文用到的脚本

| 脚本 | 做什么 |
|---|---|
| `scripts/pmc_mbs.txt` | 三轮计数器的输入文件 |
| `scripts/pmc_summary.py` | 汇总 `--pmc` 的 CSV，换算时钟、wave 寿命，并按 M、N、K 核对 MFMA 条数和写回字节数 |
| `scripts/att_budget.py` | 从 ATT 算每个 wave 的时间预算、MFMA 空闲的原因、tile 切换的构成、源码热点 |
| `scripts/draw_profiling_figs.py` | 画图 1–3；图 2、图 3 要给 `--att <ui_output 目录>` |
| `scripts/mbs_timing.py` | 读 `-DMBS_TIMING=1` 构建的分段计时 |
| `scripts/parse_att_trace.py` | （原有）把 `code.json` 打印成表 |
| `scripts/parse_kernel_trace.py` | （原有）按 kernel 名汇总 kernel trace 的耗时 |
| `scripts/rocprof-compute.sh` | （原有）笔记里那套 rocprof-compute 命令，路径是别人机器上的 |

`MBS_TIMING` 的几个桶（段 0 等待、段 0 主体、段 1 等待加 barrier、段 1 主体、epilogue……）和 ATT 是互补的：插桩不需要装任何东西，能按代码段直接加总；但每读一次 `s_memtime` 都要 `s_waitcnt lgkmcnt(0)`，会顺带等完还没回来的 LDS 读。这次插桩版每个 stage 约 2,955 周期，ATT 量到的是 2,686 周期。所以插桩的数只能在桶之间互相比较，不要拿去和不插桩的运行比。
