```bash
cd /home/haoyanli/a121_quark/mbs/lab
```

# 汇编，带 .loc 行号

```bash
hipcc -O3 -std=c++20 --offload-arch=gfx950 -gline-tables-only --cuda-device-only -S gemm_lab.hip -o build/gemm_lab.s
```

- `hipcc`：HIP 编译器入口。它自己不编译，只是给 clang++ 补上 `--driver-mode=g++`、`-x hip`、`--hip-link` 再转交。设 `HIPCC_VERBOSE=1` 可以看到实际执行的命令。
- `-O3`：最高档优化，包括循环展开、指令调度、把变量放进寄存器。实测 `-O0` 每个 lane 用掉 720 B scratch，`#pragma unroll` 不生效，慢约 50 倍（50,568 µs 对 1,042 µs）。
- `-std=c++20`：按 C++20 标准编译。这份代码用 C++17 也能编过，写 20 是为了和 MBS 保持一致。
- `--offload-arch=gfx950`：GPU 架构（MI355X），决定能用哪些指令。换成 gfx942 会编译失败，因为带 scale 的 FP4 MFMA 只有 gfx950 才有。可以写多次，同时编多种 GPU。
- `-gline-tables-only`：只生成行号表，记录每条指令来自源码哪一行。`.s` 里的 `.loc`、反汇编里插入的源码行、ATT 里的 `v0_naive.h:197` 都靠它。生成的指令和不加时逐条相同。不要用 `-g`：它会改变寄存器分配，还会多出 scratch（见 `docs/profiling_guide.md` 4.3 节）。
- `--cuda-device-only`：只编 GPU 那一半。一个 `.hip` 文件要编两遍：host 一遍，每个 arch 各一遍。通用写法是 `--offload-device-only`。不加的话，`-S` 会把 gfx950 和 x86 两份汇编打包进同一个文件。
- `-S`：编到汇编就停，不生成目标文件，也不链接。
- `gemm_lab.hip`：源文件，`#include` 了 `v0_naive.h` 和 `common.h`。
- `-o build/gemm_lab.s`：输出路径。

警告 `argument unused during compilation: '--hip-link'`：hipcc 总会加上 `--hip-link`（链接时带上 HIP 运行时），但 `-S` 不链接，用不上它。可以忽略。

# code object，再反汇编并插入源码行

```bash
hipcc -O3 -std=c++20 --offload-arch=gfx950 -gline-tables-only --cuda-device-only --no-gpu-bundle-output -c gemm_lab.hip -o build/gemm_lab.co
/opt/rocm/llvm/bin/llvm-objdump -d --source -l --no-show-raw-insn build/gemm_lab.co > build/gemm_lab_src.txt
```

- `-c`：编到二进制目标文件，也就是 GPU 的 code object，不链接。
- `--no-gpu-bundle-output`：不打包，直接输出 ELF。打包过的文件 `llvm-objdump` 不认。
- `llvm-objdump -d`：反汇编。
- `--source`：在指令之间插入源码行；`-l`：标出文件名和行号。两者都靠 `-gline-tables-only`。
- `--no-show-raw-insn`：不显示指令编码。不过在 AMDGPU 上，行尾注释里仍然会带地址和编码。

`make src` 就是这两条命令，另外加了 `--symbolize-operands`，把跳转目标显示成 `L0`、`L1` 这样的标签，再用 sed 去掉行尾的地址和编码。

# Makefile 目标对照

- `make`：去掉 `--cuda-device-only -S`，host 和 GPU 都编，链接成 `build/gemm_lab`。
- `make asm`：第一条命令。
- `make src`：第二条命令，见上。
- `make resources`：`--cuda-device-only -c -o /dev/null -Rpass-analysis=kernel-resource-usage`，只打印每个 kernel 的寄存器、LDS、scratch 用量。
- `make debug`：用 `-O0 -g` 编出 `build/gemm_lab_dbg` 和 `build/dbg_demo`，给 rocgdb 用，见下一节。

# 用 rocgdb 调试 kernel

rocgdb 是 ROCm 自带的 GDB（这台机器上是 ROCm 7.2，基于 GDB 16.3）。它能在 GPU kernel 里下断点、切换 wave 和 lane、看变量和寄存器、读 LDS。下面的命令都在这台机器上跑过。

## 准备

```bash
cd /home/haoyanli/a121_quark/mbs/lab
make debug                   # build/gemm_lab_dbg（实验室程序）和 build/dbg_demo（练习程序）
export HIP_VISIBLE_DEVICES=5
```

- 调试版用 `-O0 -g`。`-O3` 下很多变量会显示 `<optimized out>`，单步也会按重排后的指令顺序来回跳；`-gline-tables-only` 只有行号，看不了变量。
- 调试版慢约 50 倍，只用来查逻辑，不要拿来测性能。
- 先用 `amd-smi metric --usage` 找一张空闲的卡。HIP 编号和 amd-smi 编号不一样，这台机器上 HIP 0–7 依次是 amd-smi 的 3、0、2、1、7、4、6、5（按 PCI 地址对出来的），`HIP_VISIBLE_DEVICES=5` 就是 amd-smi 4 号卡。
- 规模取小，比如 `v0 256 256 512`。调试器停下时所有 wave 都会停，规模大了一屏全是 wave。

## 基本流程

```bash
rocgdb --args build/gemm_lab_dbg v0 256 256 512 --iters 1 --warmup-ms 0 --check 0
```

```text
(gdb) break v0_naive.h:36          ← 拷贝循环里写 sA 的那一行
(gdb) run
Thread 7 "gemm_v0_naive" hit Breakpoint 1, with lanes [0-63], v0::gemm_v0_naive (p=...) at ./v0_naive.h:36
```

- 程序刚启动时 GPU 代码还没加载，`break` 会提示 “No compiled code for line 36”，问要不要设成 pending 断点，答 `y`。kernel 一启动就会命中。
- rocgdb 里一个 wave 就是一个 thread，wave 里的 64 个线程叫 lane。命中断点时，所有 wave 都会停下。
- 常用的 GDB 命令照样能用：`continue`、`next`、`step`、`stepi`、`finish`、`bt`、`list`、`info break`、`disable`、`delete`、`quit`。
- 想写成脚本反复跑，就把命令写进文件，用 `rocgdb -batch -x cmds.gdb --args ...` 运行。脚本开头先写 `set pagination off` 和 `set breakpoint pending on`。

## GPU 特有的命令

- `info agents`：列出所有 GPU，当前用的那张前面有 `*`。
- `info dispatches`：正在跑的 kernel，带 grid 和 workgroup 大小。grid 按线程数算，比如原版 v0 在 256×256 时显示 `[512,2,1]`。
- `info threads`：列出所有 wave。`AMDGPU Wave 6:1:1:3 (0,0,0)/2` 的意思是：agent 6、queue 1、dispatch 1 里的第 3 个 wave，属于 workgroup (0,0,0)，是这个 workgroup 里的 wave 2。
- `thread 9`：切到 9 号 wave；`lane 37`：切到当前 wave 的 lane 37。之后 `print` 打印的都是这个 lane 的值。
- `info lanes 0-3`：看几个 lane 各自的 threadIdx。
- 便利变量：`$_wave_id`（`"(0,0,0)/2"`）、`$_lane`、`$_lane_workgroup_pos`（就是 threadIdx，比如 `"[165,0,0]"`）、`$_lane_count`。
- 寄存器：`print $s0` 是标量寄存器（一个值），`print $v1` 是向量寄存器（64 个 lane 各一个值），`print/x $exec` 看哪些 lane 是活的，`print $pc` 看 PC。
- 汇编：`x/6i $pc` 从 PC 往后看 6 条指令，`disassemble /s 起始,结束` 带源码看一段，`stepi` 走一条指令。
- 内存：全局内存直接用指针，比如 `x/16xb p.A`；LDS 用变量名（`x/16xb sA`），或者写成地址空间的形式 `x/16xb local#0`。

## 例子一：在 v0 里核对下标和 LDS

行号按你 2026-11-12 的 `v0_naive.h`：第 36 行写 sA，第 48 行算 ra。改了文件以后，先用 `list 30,50` 核对。

**看某个 lane 的下标。** 停在第 36 行以后：

```text
(gdb) lane 37
(gdb) print tid       → 37
(gdb) print r16       → 5
(gdb) print g         → 2
(gdb) print r         → 2
(gdb) print c         → 5
```

在原版 launch（每个 workgroup 256 个线程）下，`thread 9` 再 `lane 37` 就是 wave 2 的 lane 37：`tid` 165、`wm` 1、`wn` 0、`i` 165、`r` 10、`c` 5。

**条件断点加单步。**

```text
(gdb) disable 1                          ← 先关掉 1 号断点，否则单步时别的 wave 会撞上它
(gdb) break v0_naive.h:48 if tid == 37
(gdb) continue
Thread 7 "gemm_v0_naive" hit Breakpoint 2, with lane 37, ...
(gdb) next
(gdb) print ra        → 5
```

条件是按 lane 判断的，停下时当前 lane 会自动切到满足条件的那个。

**核对 LDS 里的数据。**

```text
(gdb) x/16xb sA                           ← LDS 第 0 行的前 16 字节
local#0x0:      0x2f 0xd3 0x26 0x35 0xcc 0x74 0x98 0xf9 ...
(gdb) x/16xb p.A + (size_t)m0 * KB        ← A 第 m0 行的前 16 字节，应该完全一样
0x7ffff4200000: 0x2f 0xd3 0x26 0x35 0xcc 0x74 0x98 0xf9 ...
(gdb) next
(gdb) print/x a                           ← lane 37 从 LDS 读出的片段
(gdb) x/4xw sA + ra * 64 + g * 16         ← 直接看 LDS 里的同一个位置
```

`ROWB`、`BM` 这类 `constexpr` 常量在调试信息里没有，会显示 `<optimized out>`，表达式里要直接写数字。

这一步正好能看出你现在实验版本的问题：launch 只有 64 个线程，拷贝循环只写了第 0–3、16–19、32–35……行。lane 37 要读的第 5 行没人写过，`a` 读到的是 `{0x40800000, 0x40a00000, 0x40c00000, 0}`，也就是 4.0、5.0、6.0、0.0 这样的 float，是之前别的 kernel 留在 LDS 里的旧数据。

## 例子二：定位内存越界

`dbg_demo.hip` 是专门的练习程序。加 `bug` 参数时，第 26 行读 `in[gid + shift]` 的下标多了 2^30，读到了没有映射的地址。直接运行只能看到：

```text
Memory access fault by GPU node-7 (Agent handle: 0x2dc386f0) on address 0x7fc60861a000. Reason: Unknown.
```

看不出是哪一行。用 rocgdb：

```text
rocgdb --args build/dbg_demo bug
(gdb) set amdgpu precise-memory on
(gdb) run
Thread 7 "neighbour_sum" received signal SIGBUS, Bus error.
26	    tile[tid] = in[gid + shift];
(gdb) print shift              → 1073741824
(gdb) print &in[gid + shift]   → 0x8000ee800000，比 in（0x7fffee800000）多出 4 GiB
(gdb) disassemble /s $pc-24,$pc+4
   ...  flat_load_dword v2, v[0:1]       ← 出错的 load
=> ...  v_mov_b64_e32 v[0:1], s[16:17]   ← PC 停在它的下一条
```

- `set amdgpu precise-memory on` 让每条访存指令都等它完成再往下走，所以出错时 PC 正好停在出错指令的下一条。不开的话，wave 会继续往前跑，直到要用这份数据（`s_waitcnt`）才停，rocgdb 会提示 “reported location may not be accurate”。`-O0` 下两种情况还落在同一行；`-O3` 会把 load 提前很多，位置可能差好几行。代价是 kernel 会变慢。
- 越界一点点（比如多写一个元素）通常不会报错：hipMalloc 按页分配，多出来的那点还在同一页里，数据被悄悄写坏。这种问题只能靠对拍发现。

## 不开调试器的办法

- **printf**：kernel 里可以直接 `printf`，但每个线程都会打印一遍。只看一个线程：`if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 37) printf(...);`。
- **ROCm debug agent**：出错时自动打印出错 wave 的寄存器、LDS 和带源码的反汇编：

  ```bash
  ROCM_DEBUG_AGENT_OPTIONS="-p -o /tmp/agent.txt" \
  HSA_TOOLS_LIB=/opt/rocm/lib/librocm-debug-agent.so.2 HSA_ENABLE_DEBUG=1 build/dbg_demo bug
  ```

  `-p` 是精确模式，`-o` 把输出写进文件。每个出错的 wave 都会打印一份。dbg_demo 这种所有 wave 都越界的情况，文件有 200 MB，看前面几百行就够了：找 `wave_…: pc=… (stopped, reason: MEMORY_VIOLATION)`，再看后面的 `Disassembly for function`，`=>` 指着 PC。
- **HIP 运行时日志**：`AMD_LOG_LEVEL=3` 会打出每次 `hipLaunchKernel` 的 grid、block 和 kernel 名，用来核对 launch 配置。`AMD_SERIALIZE_KERNEL=3` 让每次启动 kernel 前后都等 GPU 做完，出错时能对上是哪一次启动。

## 常见的坑

- 单步时被别的 wave 打断：继续执行时所有 wave 一起跑，别的 wave 会撞上同一个断点。单步前先 `disable` 掉断点，或者用条件断点只盯一个 lane。
- `print p` 会把 `uint8_t*` 当字符串打出一大串，用 `print/x p.A`、`print p.K` 看单个字段。
- 改了源码记得重编调试版（`make debug`）。拿不准有没有重编时，用 `make -B debug` 强制重编。
