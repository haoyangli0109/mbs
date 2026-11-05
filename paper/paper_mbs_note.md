# 论文解读：用 OAS 和 MBS 把 MXFP4 的精度拉近 NVFP4（arXiv:2603.08713）

> - 论文：Jatin Chhugani、Geonhwa Jeong 等（Meta），*Unveiling the Potential of Quantization with MXFP4: Strategies for Quantization Error Reduction*，arXiv:2603.08713v1，2026-01-30。原文在同目录的 `2603.08713.pdf`。
> - 第 1–6 节是论文本身的内容。第 7 节是我结合 MI355X 和本仓库代码做的分析，其中的 QSNR 数字是我在 Qwen3.5-9B 上实测的，不是论文的数据。
> - 名字撞车：论文的 MBS 指 Macro Block Scaling（宏块缩放）。本仓库的 kernel 也叫 MBS，但它实现的是标准 MXFP4（每 32 个 K 一个 E8M0 scale），没有论文里的宏块缩放。

## 结论

论文把 MXFP4 精度不如 NVFP4 的原因拆成两项，硬件上只做便宜的那项，贵的那项用软件补：

| 差异 | 对 QSNR 的影响 | 硬件代价 | 论文的处理 |
|---|---|---|---|
| block 大小：32 对 16 | 约 +1 dB | tensor core 面积 +2% | 改成 16 |
| scale 精度：E8M0（只能是 2 的幂）对 E4M3（3 位尾数） | +3–4 dB | tensor core 面积 +12.6% | 保留 E8M0，用 OAS 和 MBS 补 |

- **OAS（Overflow-Aware Scaling）**：换一种 E8M0 指数的取法，让 block 最大值缩放后落在 (3.5, 7]，而不是 (3, 6] 或 [4, 8)。没有任何开销。
- **MBS（Macro Block Scaling）**：每 128 个元素再加一个只有 8 位尾数的缩放因子（E0M8）。量化前先乘上它；GEMM 里每 128 个 K 的部分和，在 FP32 里乘回 `σA ⊗ σB` 再累加。
- **效果**：在 Llama3.1-8B 和 Qwen3-8B 上，平均精度离 NVFP4 不到 1 个点（MXFP4-OCP 落后约 6 个点）。B200 上 GEMM 平均慢 6.2%。

放到我们的环境里，有三点要注意：

1. **OAS 我们已经有了。** aiter 和 Quark 默认的 `even` scale 取整和 OAS 是同一条规则，所以我们现在的 MXFP4 基线比论文里的 "MXFP4-OCP" 强。
2. **MI355X 用不了 1×16。** scaled MFMA 每 32 个 K 只认一个 E8M0。论文没给 1×32 + OAS + MBS 的数据，我在 Qwen3.5-9B 上测了：输出 QSNR 从现状的 17.14 dB 提到 18.76 dB（+1.6 dB），离 NVFP4 的 20.17 dB 还差 1.4 dB。1×16 时只差 0.8 dB。
3. **MBS 在 MI355X 上不便宜。** 一条 16x16x128 的 MFMA 正好覆盖一个 128 宏块。两侧都做 MBS 时，每条 MFMA 要配约 4 条 packed FP32 指令，估计会占满 MFMA 执行期间 VALU 的全部发射能力。更现实的做法有三种：
   - 宏块放大到 256，两侧都做：+1.3 dB，VALU 约占一半。
   - 只对激活做：+0.9 dB，VALU 约占一半。
   - 每行一个因子，只在 epilogue 里乘：+0.6 dB，K 循环零开销。

## 1. 三种 FP4 格式

![三种 FP4 格式（论文 Figure 2）](docs/images/paper_2603.08713_fig2_formats.png)

三者的元素都是 E2M1，能表示的绝对值只有 0、0.5、1、1.5、2、3、4、6，最大非零值和最小非零值之比只有 12。区别全在 scale 上：

| | MXFP4（OCP） | NVFP4 | 论文的 MXFP4 + MBS |
|---|---|---|---|
| 一级 scale | E8M0，每 32 个元素 | E4M3，每 16 个元素 | E8M0，每 16 个元素（性能测试的 kernel 用的是每 32 个） |
| 二级 scale | 无 | FP32，每个 tensor 一个 | E0M8（只有尾数），每 128 个元素 |
| 每元素位宽 | 4.25 bit | 4.5 bit | 4.5625 bit（block 32 时 4.3125 bit） |
| 一级 scale 怎么算 | OCP 规范：`2^(⌊log2 amax⌋ − 2)` | `amax / 6 / tensor_scale`，舍入到 E4M3 | OAS |
| 硬件要求 | MX tensor core | tensor core 要支持浮点 scale | MX tensor core，外加向量单元做缩放 |
| GEMM | 原生 | 原生 | 原生 MX GEMM，每 128 个 K 的部分和乘 `σA ⊗ σB` 后累加 |

还有一点区别：NVFP4 的 per-tensor scale 要先知道整个 tensor 的 amax，对激活来说要么扫两遍，要么用离线校准的静态值。MBS 只看局部的 1×128，一遍就能算完。

## 2. 论文的分析：差距从哪来

论文用 QSNR（量化信噪比）衡量保真度：`QSNR = 10·log10(‖X‖² / ‖X − Q(X)‖²)`，越高越好。分别对输入 tensor 和 matmul 输出计算，在 Llama3.1-8B 和 Qwen3-8B 推理时 dump 的 tensor 里随机抽 1000 个取平均。

### 2.1 block 从 32 改成 16：便宜，约 +1 dB

- FP4 只有 2 位指数，一个 block 内能区分的范围只有 12 倍。block 越大，越容易有小值被量化成 0。
- 激活的 block 从 32 改成 16，量化成 0 的元素从 20% 降到 13%，QSNR 约 +1 dB。
- 硬件代价是 tensor core 面积 +2%。主要来自 A/B 的 SRAM 从每元素 4.25 bit 变成 4.5 bit，以及 block 间的加法树变宽。

### 2.2 scale 从 E8M0 改成 E4M3：贵，+3–4 dB

- **指数位基本是浪费的。** 几乎所有权重 tensor 和 98% 以上的激活 tensor，scale 的范围用 4 位指数就够了。
- **关键在尾数。** E8M0 只能是 2 的幂，block 最大值落在两个 2 的幂之间时对不准。比如落在 4 和 6 之间的 5，只能变成 4 或 6，误差 20%。E4M3 有 3 位尾数，效果和 FP32 scale 只差 0.2–0.3 dB。单这一项贡献 3–4 dB。
- **硬件代价大，因为 block 间的对齐变贵了。**
  - E8M0：每个 block 只需要一次整数加法（把 scale 指数加到 block 内的最大指数上），再在 NumBlocks 个值里找最大指数。
  - E4M3：每个 block 要做浮点乘法，最大指数要在全部 NumBlocks × BlockSize 个值里找。
  - 结果是计算逻辑面积 +21.3%，tensor core 总面积 +12.6%（同为 block 16）。这些数字来自一个量产多格式 tensor core 的面积数据（先进 TSMC 工艺）搭出来的分析模型。

### 2.3 结论

硬件只做便宜的那项（block 16，scale 仍是 E8M0），scale 的精度用 OAS 和 MBS 在软件里补。

## 3. 优化一：OAS

### 3.1 三种指数取法

设 block 内绝对值最大的元素为 `amax`，E8M0 scale 为 `2^e`，量化时元素除以 `2^e` 再舍入到 FP4。`e` 有三种取法：

| 取法 | e | amax 缩放后落在 | 问题 |
|---|---|---|---|
| OCP（floor） | `⌊log2 amax⌋ − 2` | [4, 8) | 超过 6 的部分被截到 6，amax 本身误差最大 25% |
| ceil（论文的 MXFP4-16 基线，也是 NV 的 RCEIL） | `⌈log2(amax / 6)⌉` | (3, 6] | 不截断；但 amax 落在 (3, 3.5] 时，FP4 的 [4, 6] 这一段完全用不上，小值更容易变成 0 |
| **OAS** | 先按 ceil 算；如果 amax 落在 (3, 3.5]，e 再减 1 | (3.5, 7] | — |

### 3.2 为什么 OAS 不会更差

OAS 只在 ceil 取法下 amax 缩放后落在 (3, 3.5] 的 block 上动手，这时 block 里所有元素缩放后都不超过 3.5。scale 减半、元素放大 2 倍后：

- 原来落在 [1, 3.5] 的元素被放大到 [2, 7]。[1, 3] 内的格点 {1, 1.5, 2, 3} 放大后正好对应 {2, 3, 4, 6}，相对位置不变；(3, 3.5) 原来舍入到 3，现在 (6, 7) 截到 6，相对误差也一样（3.3 → 3.0 和 6.6 → 6.0 都是 9%）。
- 原来小于 1 的元素（FP4 在这里只有 0 和 0.5 两个格点）被放大到格点更密的区间，误差变小。

所以 OAS 对每个元素的误差都不超过 ceil。我在 4096×4096 的重尾随机数据上验证过：没有一个元素的误差比 ceil 大。

和 OCP 相比，两者只在 amax 的尾数 ≥ 1.75 时不同：

- OCP 把 amax 放到 [7, 8)，截到 6，误差 14–25%。
- OAS 把它放到 [3.5, 4)，舍入到 4，误差不超过 14.3%。
- 论文的例子：amax 映射到 7.6 时误差是 21%，映射到 3.8 时只有 5.3%。

实现上只需要看 amax 尾数的最高两位，零开销。论文观察到约 15% 的 block 用上了 OAS，QSNR +0.5 dB。我在 Qwen3.5-9B 的 1×32 block 上数到的比例更高一些：激活 22%，权重 24.5%。

### 3.3 aiter 和 Quark 的 `even` 就是 OAS

aiter 的 MXFP4 量化（`aiter/ops/triton/_triton_kernels/quant/quant.py` 第 130–142 行）：

```python
@triton.jit
def _mxfp4_scale_from_amax(amax):
    """Convert an FP32 absolute maximum to E8M0 and its reciprocal."""
    amax = amax.to(tl.int32, bitcast=True)
    amax = (amax + 0x200000).to(tl.uint32, bitcast=True) & 0xFF800000
    amax = amax.to(tl.float32, bitcast=True)
    scale_e8m0_unbiased = tl.log2(amax).floor() - 2
```

- `+ 0x200000` 相当于给 amax 的尾数加 0.25。尾数 ≥ 1.75 时进位到指数，再取 `⌊log2⌋ − 2`，amax 就落在 [3.5, 7)。
- 它和 OAS 只在尾数恰好等于 1.75 时有差别：OAS 把 amax 放到 7 再截到 6，`even` 把它放到 3.5 再舍入到 4，amax 本身的误差一样。
- 在重尾随机数据上实测，两者只在 0.64% 的 block 上不同，QSNR 相差 0.02 dB。

本仓库的 `test_gemm_a4w4.py` 用 `aiter.get_triton_quant(per_1x32)` 量化，最终调到 `aiter/utility/fp4_utils.py` 的 `dynamic_mxfp4_quant(scaling_mode="even")`，第 409–413 行是同样的写法。Quark 也一样：MX 配置默认 `scale_calculation_mode="even"`，未指定时也走 `even_round`（`quark/torch/quantization/observer/observer.py` 第 118 行）。aiter 源码的注释把它叫作 "torchao EVEN / Quark EVEN"，也就是说这条规则在论文之前就已经在用了。

**论文的 "MXFP4-OCP" 基线对应的是 `floor` 模式，不是我们的默认配置。我们现在的基线相当于论文里的 "1×32 + OAS"。**

## 4. 优化二：MBS

### 4.1 想法

- E8M0 只能把整个 block 乘一个 2 的幂，改变不了元素的尾数。所以一个 block 的量化误差完全由元素自己的尾数决定，scale 怎么选，离群值都对不准。
- MBS 给每个 1×128 的宏块加一个尾数因子 `f = 1 + m/256 ∈ [1, 2)`。量化前先把这 128 个元素乘上 `f`，再做普通的 MXFP4（block 16，OAS）。
- `f` 的指数部分会被各子块的 E8M0 吸收，所以只需要存 8 位尾数 `m`，论文叫它 E0M8。
- 反量化：`x ≈ FP4 × 2^e / f`。

为什么选 128（附录 A 的消融）：宏块越大 QSNR 越低，从 32 到 512 输出 QSNR 掉约 1.1 dB。128 保留了宏块 32 时 96% 的输出 QSNR，存储和 GEMM 开销又小。

### 4.2 MBS-S（静态）

```text
m = (bits(6.0 / amax128) & 0x007F8000) >> 15
```

- 取 `6 / amax128` 的 FP32 尾数最高 8 位。效果是每个宏块的最大值缩放后正好落在 6.0 附近，误差不超过 0.3%（我实测最大 0.32%）。
- 代价：量化 kernel 原本每元素约 16.1 次运算，MBS-S 多 2.7 次。量化本身是访存瓶颈，所以对在线量化的激活没有额外耗时。
- 效果：比 1×16 + OAS 高 1.1 dB。

### 4.3 MBS-D（动态）

- 在 16 个候选 `m` 里选一个，让宏块的平方误差和（SSE）最小。
- 为了不在运行时真的对每个候选做一遍量化，预先算好查找表，存每个"值的位置 × 候选"的相对平方误差。次正规值（< 1）和正规值各一张表，每张 64 个值点 × 16 个候选，共 2048 个 FP16，4 KB，不到 B200 shared memory 的 2%。
- 代价：每元素约 32 次运算，比 MBS-S 慢 2.5–3 倍。
- 效果：比 1×16 + OAS 高 1.6 dB。
- 论文没说清 16 个候选怎么选，只说是"一个窄范围内的搜索"。

### 4.4 MBS-H（论文默认）

权重用 MBS-D（离线量化，不占推理时间），激活用 MBS-S（在线量化）。端到端评测都用这个配置。

### 4.5 GEMM 里怎么用

![带 MBS 的矩阵乘（论文 Figure 5）](docs/images/paper_2603.08713_fig5_mbs_gemm.png)

```text
C[m, n] = Σ_kb  σA[m, kb] · σB[n, kb] · P_kb[m, n]
```

- `P_kb` 是第 kb 个 128-K 块上普通 MXFP4 MMA（带 E8M0 scale）的部分和。`σ = 1/f`，以 FP16 存放。
- **σ 随 kb 变化，所以这一步不能只在 epilogue 做一次。** 每个 128-K 块的部分和都要单独乘自己的 `σA ⊗ σB`，再累加到 FP32 寄存器里。论文 4.3.2 节写的是 "in the epilogue"，看附录 E 才知道，指的是每个 128-K 子 tile 的 "epilogue"。
- 结构上和 DeepSeek-V3 的 FP8 1×128 分块 GEMM 一样：每 128 个 K 把部分和提到 FP32，乘 scale 再累加（这个类比是我加的）。

B200 上的实现（CUTLASS 4.3，SM100，附录 E）：

- 在 1×32 的 MXFP4-OCP kernel 上改。4 个 CTA 组成 2×2×1 的 cluster，算 256×256 的输出 tile。leader CTA 发 MMA，peer CTA 负责 TMA multicast。
- K tile 是 256，正好两个 MBS 块。MMA warp 把前 128 个 K 的部分和写进一块 TMEM，然后切到另一块 TMEM 算后 128 个（双缓冲）。
- 主循环期间本来空闲的 epilogue warp（4–7 号）把部分和从 TMEM 读进寄存器，乘 `σA·σB` 后累加。MBS 因子直接从全局内存读。
- 理论开销：每个 128³ tile 多 2·128² 次 FP32 运算，是 FP4 乘法次数的 1.56%（1/64）；额外访存是每 64 KB 输出 tile 512 B（0.78%）。只要向量单元的吞吐超过 tensor core 的 1.56%，就能藏住。

实测 GEMM 开销（论文表 4，M = N = K）：

| shape | MXFP4-OCP（TFLOPS） | MXFP4-MBS-H（TFLOPS） | 开销 |
|---|---|---|---|
| 512 | 35.90 | 30.57 | 14.8% |
| 1024 | 285.04 | 259.34 | 9.0% |
| 2048 | 1669.40 | 1666.63 | 0.2% |
| 4096 | 4462.95 | 4354.87 | 2.4% |
| 8192 | 4745.00 | 4520.53 | 4.7% |

平均 6.2%，而最接近的前人工作 MX+ 是 54%。decode 阶段受权重加载的带宽限制，这点开销可以忽略。

## 5. 实验结果

- 配置：vLLM 加 lm-eval-harness。所有线性层（QKVO、FFN、MoE 的每个 expert）的权重和激活都量化成 FP4。
- 量化方式：直接量化，不用校准数据，不做 GPTQ、旋转或微调。

| 格式 | Llama3.1-8B 平均 | Qwen3-8B 平均 | DeepSeek-R1 MMLU-Pro | Llama4-Maverick MMLU-Pro | Wikitext PPL（Llama / Qwen） |
|---|---|---|---|---|---|
| BF16 | 70.53 | 73.47 | 83.19 | 80.98 | 8.82 / 12.20 |
| MXFP4-OCP | 61.25 | 65.50 | 72.52 | 77.73 | 11.49 / 15.18 |
| MX+ | 62.61 | 67.65 | 79.85 | 78.46 | 10.82 / 14.51 |
| MXFP4-16（ceil） | 60.55 | 66.82 | 76.29 | 78.12 | 11.52 / 15.15 |
| MXFP4-16-OAS | 63.39 | 69.71 | 75.36 | 78.64 | 10.57 / 13.65 |
| MXFP4-MBS-S | 65.77 | 70.50 | 82.37 | 79.38 | 10.04 / 13.09 |
| MXFP4-MBS-H | 66.50 | 70.84 | 82.06 | 79.77 | 9.88 / 13.03 |
| NVFP4 | 67.02 | 71.48 | 82.69 | 80.06 | 9.68 / 12.69 |

"平均"是 MMLU-Pro、GSM8K、HellaSwag、WinoGrande、ARC-C、ARC-E 六项的平均。几个值得注意的地方：

- **只缩小 block 反而更差。** MXFP4-16 用 ceil 取法，在 Llama 上比 block 32 的 OCP 还低（60.55 对 61.25）。可见 scale 取法和 block 大小一样重要；加上 OAS 之后，block 16 的收益才显出来。
- **单项收益最大的是 OAS 和 MBS-S。** OAS 在两个 8B 模型上各 +2.8 和 +2.9 个点；MBS-S 再 +2.4 和 +0.8。MBS-H 比 MBS-S 又多 0.7 和 0.3。
- **MBS-H 离 NVFP4 还差 0.5 和 0.6 个点。** 两个大 MoE 模型上基本追平（MMLU-Pro 差 0.3–0.6 个点），而 MXFP4-OCP 在 DeepSeek-R1 的 MMLU-Pro 上比 BF16 低了 10.7 个点。
- QSNR（论文 Figure 6）：MBS 把权重从 18.6 dB 提到 20.1 dB，激活从 17.4 dB 提到 19.9 dB，离 NVFP4 都不到 1 dB。

## 6. 和 MXFP4、NVFP4 的区别

**和 MXFP4（OCP）比：**

- 硬件和编码完全相同（E2M1 元素，E8M0 scale），区别有三处：
  - block 从 32 改成 16。严格说这已经不是 OCP 标准的 MXFP4，标准规定 block 为 32。
  - E8M0 指数用 OAS 取，不用 OCP 建议的 floor。
  - 多一个每 128 元素的 E0M8 因子，GEMM 里每 128 个 K 要做一次缩放。
- 代价：每元素多 0.31 bit；B200 上 GEMM 平均慢 6.2%。
- 收益：两个 8B 模型上平均各高约 5.3 个点。

**和 NVFP4 比：**

- 两者都是两级 scale，但切法不同：
  - NVFP4：每 16 个元素一个 E4M3，指数和 3 位尾数都在细粒度上，外加一个 FP32 的 per-tensor scale。
  - 论文：指数在细粒度（每 16 个一个 E8M0），尾数在粗粒度（每 128 个一个 8 位尾数）。
- NVFP4 要求 tensor core 支持浮点 scale（面积 +12.6%）；论文的格式只需要 MX tensor core 加向量单元。
- NVFP4 的 per-tensor scale 要先拿到整个 tensor 的 amax；MBS 只看局部 1×128，一遍完成。
- NVFP4 的 GEMM 没有额外步骤；MBS 每 128 个 K 要占用一些向量算力。
- 精度上仍略低于 NVFP4：平均不到 1 个点，输出 QSNR 低约 0.7–1 dB。

**和 MX+（最接近的前人工作）比：** MX+ 给每个 block 的最大元素多存几位尾数，需要额外跑一个稀疏 GEMM，开销最高 54%。论文在精度和开销上都比它好。

## 7. 对 MI355X 和本仓库 kernel 的意义

以下是我的分析和实测，不是论文的内容。

### 7.1 MI355X 只能做 1×32

本仓库的 MFMA 是 `v_mfma_scale_f32_16x16x128_f8f6f4`（`kernel/gemm_a4w4.hip` 第 296–300 行）。每个 lane 提供沿 K 连续的 32 个 FP4 和 1 个 E8M0 字节（用 `op_sel` 选字节），所以 block 固定是 32，scale 只能是 E8M0。

- 论文主推的 1×16 在 MI355X 上没有原生实现。要模拟只能让每个 lane 的 32 个值里一半补 0，MFMA 吞吐减半。
- NVFP4 的 E4M3 scale 同样不支持。
- 能做的只有 1×32 + OAS（也就是现状）再加 MBS。论文恰好没测这个组合。

### 7.2 在 Qwen3.5-9B 上的 QSNR 实测

方法见文末。这里只测了 QSNR，没测下游精度。和论文 Figure 6 里 Qwen3-8B 的数值相比，重合的格式都在 0.5 dB 以内，趋势一致。

| 格式 | block | 激活 | 权重 | 输出 | 输出比现状 |
|---|---|---|---|---|---|
| NVFP4 | 16（E4M3） | 21.87 | 20.49 | 20.17 | +3.03 |
| MXFP4-OCP（floor） | 32 | 17.16 | 18.58 | 16.28 | −0.86 |
| ceil | 32 | 17.15 | 18.41 | 16.55 | −0.59 |
| **OAS（= aiter/Quark `even`，现状）** | 32 | 17.70 | 18.81 | **17.14** | 0 |
| ceil（论文的 MXFP4-16） | 16 | 18.18 | 18.77 | 17.24 | +0.10 |
| OAS | 16 | 18.61 | 19.08 | 17.73 | +0.59 |
| MBS-S | 16 | 20.84 | 19.39 | 19.07 | +1.93 |
| MBS-H（论文默认） | 16 | 20.84 | 20.10 | 19.38 | +2.24 |
| MBS-D | 16 | 21.30 | 20.10 | 19.60 | +2.46 |
| MBS-S | 32 | 19.71 | 19.23 | 18.51 | +1.37 |
| **MBS-H** | 32 | 19.71 | 19.86 | **18.76** | +1.62 |
| MBS-D | 32 | 20.06 | 19.86 | 18.91 | +1.77 |

- 现状比论文的 OCP 基线高 0.86 dB，这部分就是 OAS 的收益。
- 在 1×32 上做 MBS-H 能再提 1.62 dB，但离 NVFP4 还有 1.41 dB；1×16 时只差 0.79 dB。
- 在 block 固定为 32 的硬件上，MBS 比缩小 block 更值：1×32 的 MBS-S（18.51 dB）高于 1×16 的 OAS（17.73 dB）。

### 7.3 MBS 在本 kernel 里的代价

**每条 MFMA 都要跟一次缩放。** 一条 16x16x128 的 MFMA 正好覆盖 K 方向的一个 128 宏块，所以每条 MFMA 的结果都要乘自己的 `σA·σB` 才能累加。

- 每条 MFMA 每个 lane 产出 4 个输出。两侧都有 MBS 时，每个输出要 2 次运算：先算 `σA·σB`，再做一次 FMA。
- 合计每个 lane 8 次 FP32 运算，即 4 条 `v_pk_mul_f32` / `v_pk_fma_f32`。wave64 的一条 packed FP32 指令要 4 个 cycle，4 条就是 16 个 cycle。
- 一条 FP4 MFMA 本身约 16 个 cycle（`mbs_note.md` 实测地板 16.7）。也就是说，VALU 要在 MFMA 执行期间一直满负荷。
- 这和整卡峰值对得上：MI355X 的 FP32 向量峰值 157.3 TFLOPS，除以 MXFP4 峰值 10.1 PFLOPS 正好是 1.56%，就是论文说的"刚好藏得住"的门槛。

**AMD 上没有 B200 那种分工。** B200 可以把 MMA 结果放在 TMEM，交给另一组 warp 去缩放。MI355X 上 MFMA 的结果就在发射它的 wave 的寄存器里，只能由同一个 wave 把这些 VALU 指令和 MFMA 交错发射。本 kernel 的热循环是按 MFMA 发射排满的（见 `mbs_note.md`），每个 stage 加几百条 VALU 指令会彻底改变排布。

**寄存器也要重排。**

- 现在的累加器钉在 AGPR 上（`"+a"` 约束）。VALU 不能直接读写 AGPR，必须用 `v_accvgpr_read/write` 来回搬。
- 256 VGPR + 256 AGPR 已经全部用满，要么把累加器挪到 VGPR、重新安排操作数，要么缩小 tile。
- 一个省寄存器的写法：每个 128-K 块开始前，先把累加器原地乘上相邻两块 scale 的比值 `r`，再让 MFMA 直接累加进去（`acc ← acc·r + P`），最后乘一次最后一块的 scale。这样不需要第二套累加器，但 VALU 运算量不变。

**几种折中方案的收益和代价**（1×32，Qwen3.5-9B 输出 QSNR；VALU 占用按 MFMA 跑满峰值估算）：

| 方案 | 输出 QSNR | 比现状 | 每个输出每 128 K 的 VALU 运算 | 约占 MFMA 期间的 VALU 能力 |
|---|---|---|---|---|
| 现状（OAS） | 17.14 | 0 | 0 | 0 |
| MBS-H，宏块 128 | 18.76 | +1.62 | 2 | ~100% |
| MBS-H，宏块 256（一个 stage） | 18.46 | +1.32 | 1 | ~50% |
| MBS-H，宏块 512 | 18.25 | +1.11 | 0.5 | ~25% |
| MBS-H，宏块 1024 | 18.06 | +0.92 | 0.25 | ~12% |
| 只对激活做 MBS-S，宏块 128 | 18.03 | +0.89 | 1 | ~50% |
| 只对激活做 MBS-S，宏块 256 | 17.90 | +0.76 | 0.5 | ~25% |
| 只对权重做 MBS-D，宏块 128 | 17.72 | +0.58 | 1 | ~50% |
| 只对权重做 MBS-D，宏块 256 | 17.59 | +0.45 | 0.5 | ~25% |
| MBS-H，每行一个因子（宏块 = K） | 17.74 | +0.60 | 0（epilogue 里每个输出乘一次） | ~0 |

我建议的顺序：

1. **先试每行一个因子。** 因子沿 K 不变，`σA[m]·σB[n]` 可以挪到 epilogue 里乘，和现有的 fp32 → bf16 转换放在一起，K 循环一条指令都不加。这相当于把 NVFP4 的 per-tensor FP32 scale 细化到每行，只是只存尾数。权重的因子离线算；激活的因子要在量化 kernel 里先求整行最大值，一个工作组处理一整行就能一遍完成。+0.6 dB 和把 block 从 32 缩到 16 的收益相当。
2. **再考虑宏块 256、两侧都做。** 宏块和 K=256 的 stage 对齐，+1.3 dB，但 VALU 约占一半，而且要重排累加器寄存器。
3. **完整的宏块 128 MBS-H 留到最后。** +1.6 dB，但在 MI355X 上 VALU 很可能成为新瓶颈。

动 kernel 之前，先用 Quark 或 vLLM 的 fake quant 跑下游精度（比如 GSM8K、MMLU-Pro），确认 QSNR 的提升能变成真实的精度收益。

### 7.4 论文本身要注意的地方

- **精度和性能用的不是同一个配置。** 精度实验用的是 1×16，靠 NVIDIA 的 NVFP4 流水线、把 E4M3 scale 限制成 2 的幂来模拟；GEMM 性能用的是 1×32 的 MXFP4-OCP kernel。1×32 + OAS + MBS 的精度论文没给（7.2 节补了 QSNR）。
- **GEMM 开销只测了 5 个方阵。** 8192³ 时基线只有 4745 TFLOPS，离 B200 的 FP4 峰值还有不小距离。基线越接近峰值，MBS 的开销越难藏住。
- **OAS 不算新规则。** 它和 torchao、Quark、aiter 已有的 EVEN 取法等价。论文引用了 torchao，但没有提到这一点。
- **MBS-D 的 16 个候选怎么选没写清。** 我的实测从静态因子开始，在一个倍频程内等比取 16 个候选，按真实 SSE 选；论文用查找表近似 SSE。
- **只做了直接量化。** 加上 GPTQ 或旋转之后，各格式之间的差距可能会缩小。

## 附：实测方法

- **模型**：Qwen3.5-9B（bf16，`/mnt/dcgpuval/haoyanli/Qwen/Qwen3.5-9B`），用 transformers 5.19 的 `Qwen3_5ForCausalLM` 加载。
- **数据**：
  - 取 32 个 decoder 层里全部 248 个线性层，包括 linear attention、full attention 和 MLP；不含 lm_head 和视觉部分。
  - 激活是用论文正文的前 1024 个 token 跑一次前向时，各线性层的输入。
  - 权重抽查了 3 层，和直接从 safetensors 读出的权重算出的 QSNR 完全一致。
- **QSNR**：按论文公式 1、2 计算，对 248 层取平均。输出指 FP32 的 `A @ W^T`。
- **量化模拟**：
  - FP4 用就近偶数舍入，超过 6 截断。
  - NVFP4 是每 16 个一个 E4M3 加 FP32 per-tensor scale。
  - MBS-D 在 16 个候选里按真实 SSE 选。
- **脚本**：在 `/tmp/mbs_paper_sim/`，不在仓库里。`fp4sim.py` 是各格式的参考实现，`sanity.py` 是性质检查，`model_qsnr*.py` 是模型实验。用的是 HIP 4 号卡，测试时空闲。
