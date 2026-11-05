# MXFP4 + MBS（宏块缩放）的 Triton 实现，及 ATOM 在线量化接入

> 2026-11-02。硬件为 MI355X（gfx950），软件为 Triton 3.8（ROCm）、aiter 0.1.24.post1、torch 2.13。
> 论文和格式背景见 [`../paper/paper_mbs_note.md`](../paper/paper_mbs_note.md)。下文的"MBS"都指论文的 Macro Block Scaling，不是仓库里那个 HIP GEMM。

## 结论

**做了什么**

- 用 Triton 实现了论文的 MBS，跑在 MI355X 原生的 MXFP4 MFMA（`tl.dot_scaled`，E8M0 每 32 个 K 一个）上，包括：
  - 量化 kernel：激活用 MBS-S；权重在加载时用 MBS-D，在 16 个候选因子里选平方误差最小的。
  - GEMM：每个宏块的部分和除以 `f_激活 · f_权重` 后累加。宏块可以是 128 / 256 / 512 / 1024，可以两侧都做，也可以只做一侧。
- 接进了 ATOM 的在线量化，新格式为 `mxfp4_mbs[<宏块>][_a|_w]`。
  - 代码在分支 `lhy/mxfp4_mbs`，worktree 是 `/home/haoyanli/a121/lhy_atom/ATOM_mbs`，基于 `lhy/mxfp4_state`，补丁在 [`atom_patch/`](atom_patch/)。
  - Qwen3.5-9B 加载时，128 个 Linear 的 MBS 量化一共 0.6 s。
- 正确性：
  - 关掉 MBS 时，量化结果和 aiter 的 MXFP4 逐位相同。
  - 打开 MBS 时，和 torch 参考实现逐位相同，和笔记里的 `fp4sim.py` 逐元素相同。
  - GEMM 和 fp32 参考的相对误差约 1e-6。
  - ATOM 在 Qwen3.5-9B 上可能用到的 7,560 组"格式 × 形状 × M"全部单独检查过，误差都在 bf16 舍入的水平。

**精度**（Qwen3.5-9B，BF16 源权重，所有 decoder Linear 都量化）

| | BF16 | MXFP4 | MBS 各变体 |
|---|---|---|---|
| 输出 QSNR（dB） | – | 17.08 | 17.64–18.71，两侧 128 最高 |
| wikitext-2 PPL | 8.65 | 10.06 | 8.78–9.52，找回 38–91% 的差距 |
| GSM8K（4 次测量平均） | 86.6 | 72.5 | 比 MXFP4 高 1.2–4.9 个点 |
| AIME 2026（思考模式，每题 5 次采样） | 85.3 | 6.0 | 40.0–62.7，两侧 128 和两侧 512 最高 |

- QSNR 的增益和笔记 7.2 节用 fp4sim 估算的一致，比如两侧 128 是 +1.62 dB，误差在 ±0.03 dB 以内。
- 困惑度上，两侧都做（找回 68–91%）明显好于只做一侧（38–64%）。
- GSM8K 的提升比困惑度小，噪声也大，各变体之间分不出稳定的先后。
- AIME 上差距被放大了：
  - MXFP4 有 94% 的回答写满 81,920 个 token 也没结束。每个 MBS 变体都在 16–24 道题上好于 MXFP4，没有一道更差。
  - 但所有 MBS 变体离 BF16 都还差 23–45 个点，主要原因是推理变长、停不下来，而不是算错。
  - 两侧 128 和两侧 512 并列最好，彼此分不开。两侧 128 显著好于其余四个变体；两侧 512 好于其中三个，对只做激活的优势不显著（p = 0.057）。
  - AIME 的排序和困惑度大体一致，和 GSM8K 相反。

**性能**

| | 宏块 128，两侧 | 宏块 512，两侧 |
|---|---|---|
| GEMM，相对同一 kernel 关闭 MBS，prefill（M ≥ 4096） | 0.53–0.54 | 0.73–0.74 |
| GEMM，相对同一 kernel 关闭 MBS，decode（M ≤ 16） | 0.66–0.72 | 0.94–1.08 |
| ATOM 端到端吞吐，相对 MXFP4（aiter asm） | 0.72–0.81（约等于 BF16） | 0.81–0.92 |

**为什么慢，还能怎么办**

- 开销符合笔记 7.3 节的估算：两侧都做时，每个宏块每个输出要 2 次 FP32 乘法；宏块为 128 时，VALU 时间和 MFMA 时间相当。
- Triton 把这两类指令排成前后两段：一个 K 步里先是 253 条 `v_pk_mul`，再是 32 条 MFMA，没有交错。试过的几种改写和 LLVM 调度选项都改变不了这一点。
- 要把开销藏住，得像仓库里的 HIP kernel 那样手排指令。此外，Triton 版本即使关掉 MBS，prefill 也只有 asm 的 0.66–0.77 倍。

**建议**

- 如果在线推理要用，宏块 512 两侧最划算：困惑度最低（8.78），AIME 和两侧 128 并列最好，吞吐保住 MXFP4 的 81–92%。
- 用于长推理（思考模式）时，即使是最好的变体也比 BF16 低 23 个点，在 32k 的 token 预算下差距更大。这需要另外解决，见"局限和下一步"。
- 其余变体在 AIME 上的差距（比如权重用 MBS-S 比用 MBS-D 低 21 个点）只在这一个模型上测过，还要受"这份权重舍入落在哪里"的影响，暂时不要当作格式本身的结论。

## 用法

```bash
export PYTHONPATH=/home/haoyanli/a121/lhy_atom/ATOM_mbs AITER_LOG_LEVEL=WARNING
python atom_e2e/serve_atom.py --model /mnt/dcgpuval/haoyanli/Qwen/Qwen3.5-9B --trust-remote-code -tp 1 \
  --online_quant_config '{"global_quant_config": "mxfp4_mbs512",
                          "exclude_layer": ["lm_head", "*visual*", "*conv1d*"]}'
```

- `serve_atom.py` 等价于 `python -m atom.entrypoints.openai_server`。这台机器上另一个会话会杀掉命令行里带这个字符串的进程，所以要经过包装。
- 格式字符串：

  | 格式 | 宏块 | 因子在哪一侧 |
  |---|---|---|
  | `mxfp4_mbs` | 128 | 激活和权重，即论文的 MBS-H |
  | `mxfp4_mbs256`、`mxfp4_mbs512`、`mxfp4_mbs1024` | 256 / 512 / 1024 | 激活和权重 |
  | `_a` 后缀 | 同上 | 只有激活 |
  | `_w` 后缀 | 同上 | 只有权重 |

- `ATOM_MBS_WEIGHT_CANDIDATES` 控制权重的候选因子数：默认 16（MBS-D），1 是静态因子（MBS-S），256 是全部尾数。
- 限制：
  - 只支持 dense Linear，权重必须是二维，K 能被 max(宏块, 256) 整除，N 能被 16 整除。
  - Qwen3.5 GDN 的 `conv1d` 不是 GEMM，必须排除；普通 `mxfp4` 也一样。
  - fused MoE 退回普通 MXFP4，会打一条警告。
  - 由 fused norm+quant 产生的激活已经是普通 MXFP4，不带因子。
  - TP > 1 没有测过。

## 文件

| 文件 | 内容 |
|---|---|
| `mxfp4_mbs.py` | 全部 kernel 和 Python 接口，和 ATOM 里的 `atom/model_ops/mxfp4_mbs.py` 逐字节相同 |
| `mbs_gemm_configs.json` | 调优表：6 个变体 × 6 组 (N, K) × 13 档 M，ATOM 里有同一份 |
| `test_mxfp4_mbs.py` | 量化、GEMM 的正确性测试，对照 aiter、torch 参考实现和 fp4sim |
| `test_atom_op.py` | 走 ATOM 的加载后处理和注册好的 custom op，覆盖每种格式字符串 |
| `test_configs.py` | 调优表里每个配置的数值检查：所有格式、Qwen3.5-9B 的全部形状、M 从 1 到 16384 |
| `tune_gemm.py` | 调优脚本。会先在子进程里编译每个候选配置，跳过让 LLVM 崩溃的 |
| `bench_gemm.py`、`bench_baseline.py`、`bench_utils.py` | kernel 级计时，和 aiter asm / Triton 对比 |
| `dump_asm.py` | 热循环的指令构成，以及 MFMA 和 VALU 的排布 |
| `hf_eval.py` | 在 HF transformers 里走真实 kernel：`qsnr` 测各层 QSNR，`ppl` 测 wikitext-2 困惑度 |
| `atom_e2e/e2e.sh` | ATOM 端到端：起服务、冒烟测试、GSM8K、吞吐测试、汇总 |
| `gsm8k_pair.py`、`bench_summary.py` | GSM8K 逐题配对检验，吞吐汇总 |
| `reason_eval.py` | AIME 2026 评测，从 LeapQuant 实验复制，只改了默认路径 |
| `aime_summary.py`、`aime_analyze.py` | AIME 汇总（准确率、区间、逐题配对检验），以及截断、循环和长度分析 |
| `atom_e2e/stop_after_samples.sh` | 前 N 次采样全部做完后停掉评测 |
| `atom_patch/0001-*.patch` | ATOM 改动，`git am` 可以直接打到 `lhy/mxfp4_state`。打到 `main` 时 `envs.py` 有一处上下文冲突 |
| `results/` | 全部原始结果，见文末 |

## 实现

### 格式

每一行 K 个元素：

- fp4（e2m1）码，两个一字节，形状 `[R, K/2]`。
- 每 32 个 K 一个 E8M0 scale，取整规则和 aiter/Quark 的 `even` 相同，形状 `[R, K/32]`。
- 每个宏块一个 8 位尾数 m，f = 1 + m/256，形状 `[R, K/MB]`。

量化时先把宏块乘以 f，再做普通 MXFP4；反量化为 `x ≈ fp4 · 2^(s−127) / f`。m = 0 时就是 aiter 的 MXFP4。

### 量化 kernel（`_mbs_quant_kernel`）

- 一个 program 处理"若干行 × 一个宏块"。
- MBS-S：m 取 `6/amax(宏块)` 的 FP32 尾数高 8 位，即论文公式 3。
- MBS-D 用来量化权重，在 16 个候选里按真实平方误差选，候选按笔记的方法取：从静态因子起，在一个倍频程内等比取。设成 256 时穷举全部尾数。
- FP4 舍入用阈值比较实现，就近偶数、超过 6 截断，和 aiter 的位运算结果逐位相同。
- E8M0 的取法和 aiter 完全相同。scale 可以按行主序写出，也可以直接写成 aiter 的 shuffle 布局。
- 激活量化顺带输出 GEMM 要的两个 fp32 数组：
  - `ratio[kb] = f[kb]/f[kb−1]`。前一块的因子由前一块的 amax 重新算出，所以各 program 之间互不等待。
  - `sigma_last = 1/f`，取最后一个宏块。

### GEMM kernel（`_mbs_gemm_kernel`）

- **布局**：和 ATOM 现有的 MXFP4 路径一致，权重不需要另存一份。
  - A 是行主序，scale 用 shuffle 布局（32 行以下用行主序）。
  - B 用 `shuffle_weight(layout=(16, 16))` 预重排，scale 用 shuffle 布局。
- **累加方式：每进入一个宏块，先缩放累加器。**
  - 设 `D_kb = Σ_{j≤kb} P_j /(fA_j·fB_j)`，累加器保存 `D_kb · fA_kb · fB_kb`。
  - 进入第 kb 块时，累加器先乘 `ratioA ⊗ ratioB`，再由 MFMA 原地累加 `P_kb`；epilogue 最后乘 `sigma_last`。
  - 只需要一套累加器，每个宏块每个输出 2 次乘法；只做一侧时是 1 次。
- **K 的切分**：
  - 宏块小于 BLOCK_K（128 配 256）时，一个 K 步内按宏块切成几次 `dot_scaled`。scale 是交错存放的，用固定的字节偏移挑出其中一半。
  - 宏块大于 BLOCK_K 时，只在宏块的第一个 K 步做缩放，用运行时分支判断。
- **因子的存放**：权重的因子在运行时存成 fp32 的 `ratio [K/MB, N]` 和 `sigma [N]`，而不是 8 位的 E0M8。
  - 原因是 kernel 里现算除法很贵，而且 uint8 读取没法向量化。
  - 代价：宏块为 128 时，比 FP4 权重多占 6.25% 的显存和带宽；宏块为 512 时多 1.6%。

### ATOM 改动（`lhy/mxfp4_mbs`，10 个文件，核心约 300 行）

- **`quant_spec.py`**：新增 `MBSSpec` 和 `LayerQuantConfig.mbs`，`QuarkOnlineParser` 负责解析 `mxfp4_mbs*` 格式。
- **`model_ops/linear.py`**：
  - `online_quantize_weight` 加一条 MBS 分支，把因子转换后存成两个非持久 buffer。
  - 预重排和 scale shuffle 沿用原来的 MXFP4 流程。
  - 前向通过 custom op `gemm_mxfp4_mbs` 调用。这个 op 用 `torch_compile_guard` 注册，兼容 torch.compile 和 CUDA graph。
- **`model_ops/moe.py`**：MoE 层遇到 MBS 格式时打警告。
- **`model_ops/mxfp4_mbs.py`** 和 **`mbs_gemm_configs.json`**：kernel 与调优表。
- **其他**：`envs.py` 新增 `ATOM_MBS_WEIGHT_CANDIDATES`；更新了 `docs/online_quantization_guide.md` 和 `docs/environment_variables.md`；补了 7 个解析器单测。相关的 589 个单测全部通过，black 和 ruff 检查也通过。

## 正确性

- **`test_mxfp4_mbs.py` 全部通过**：
  - 关闭 MBS 时，q 和 scale 与 aiter 的 `dynamic_mxfp4_quant` 逐位相同，包括 shuffle 布局。
  - MBS-S、MBS-D(16)、MBS-D(256) 与 torch 参考实现逐位相同，m 的一致率为 100%。
  - 在重尾随机数据上，三种格式与 fp4sim 逐元素相同：QSNR 分别为 16.704、18.339、18.740 dB，两边一致。
  - GEMM 覆盖 MB = 128/256/512/1024、四种 MBS 开关、M 从 1 到 1024，fp32 输出的相对误差在 9e-7 以内。
  - 关闭 MBS 时，GEMM 和 aiter asm `gemm_a4w4` 逐位相同。
- **`test_atom_op.py` 全部通过**：6 种格式字符串走 ATOM 的实际路径，与参考实现的差距就是 bf16 舍入，约 1.7e-3。
- **`test_configs.py` 全部通过**：调优表里的配置是按耗时选的，调优时没有检查结果，所以单独补了这项检查。
  - 范围：ATOM 可能选到的每个配置，即 7 种格式字符串 × Qwen3.5-9B 的 6 组 (N, K) × 180 个 M（1–160 逐个，加上更大各档边界的两侧，直到 16384），共 7,560 组。
  - 判据：和 fp32 参考比较整体、最差一行、最差一列（M ≥ 32 时），以及最差单个元素的误差。单个元素的误差除以它的典型大小 ‖a_i‖·‖b_j‖/√K。
  - 结果：最差的单元素误差是典型大小的 2.1%，相当于 bf16 舍入；漏掉一个 128 的 K 块会到 110%。
- 端到端冒烟测试正常；128 层全部带上了预期的 MBS 配置，见 `online_quant_info_*.json`。

## 精度

### QSNR（`hf_eval.py qsnr`）

248 个 decoder Linear 的输入取自 wikitext-2 前 1024 个 token 的一次前向（不含 lm_head 和视觉部分）。输出由 MBS GEMM 以 fp32 算出，参考值是 fp32 的 `A @ W^T`；表中是各层平均。

| 格式 | 激活 | 权重 | 输出 | 比 MXFP4 | 笔记 7.2 节（fp4sim） |
|---|---|---|---|---|---|
| MXFP4（`even`） | 17.72 | 18.80 | 17.08 | 0 | 0 |
| 两侧 128，权重 MBS-D16（MBS-H） | 19.82 | 19.86 | 18.70 | +1.62 | +1.62 |
| 两侧 128，权重穷举 256 个候选 | 19.82 | 19.90 | 18.71 | +1.63 | – |
| 两侧 128，权重 MBS-S | 19.82 | 19.23 | 18.46 | +1.38 | +1.37 |
| 两侧 256 | 19.54 | 19.56 | 18.41 | +1.34 | +1.32 |
| 两侧 512 | 19.33 | 19.34 | 18.19 | +1.11 | +1.11 |
| 两侧 1024 | 19.15 | 19.19 | 18.01 | +0.94 | +0.92 |
| 只做激活，128 | 19.82 | 18.80 | 17.98 | +0.90 | +0.89 |
| 只做激活，256 | 19.54 | 18.80 | 17.85 | +0.77 | +0.76 |
| 只做权重，128 | 17.72 | 19.86 | 17.64 | +0.56 | +0.58 |

- 真实 kernel 的增益和笔记里的模拟值相差不到 0.03 dB。
- 权重穷举 256 个候选，只比 16 个多 0.01 dB。
- 按层类型拆开看，每一类层的输出 QSNR 都随宏块增大单调下降。

### wikitext-2 困惑度（`hf_eval.py ppl`，145 个 2048-token 窗口）

所有 decoder Linear 都替换成"量化激活 + MBS GEMM"，bf16 输出；权重按 ATOM 的方式量化一次。

| 格式 | PPL | 找回 MXFP4 掉的部分 |
|---|---|---|
| BF16 | 8.6465 | – |
| MXFP4 | 10.0594 | 0% |
| 两侧 512 | 8.7799 | 91% |
| 两侧 128（MBS-H） | 8.8429 | 86% |
| 两侧 128，权重 MBS-S | 8.8743 | 84% |
| 两侧 1024 | 8.9139 | 81% |
| 两侧 128，权重穷举 256 | 8.9650 | 77% |
| 两侧 256 | 9.1025 | 68% |
| 只做权重，128 | 9.1592 | 64% |
| 只做激活，128 | 9.4990 | 40% |
| 只做激活，256 | 9.5218 | 38% |

- 两侧都做的变体明显好于只做一侧的。
- 两侧变体之间的排序不随宏块单调：256 比 512 差，权重穷举 256 个候选比只搜 16 个差（8.97 对 8.84）。而这两组的 QSNR 几乎相同（18.71 对 18.70），所以 0.1–0.2 量级的差距应当看成"舍入落在哪里"的偶然结果，不能当作格式本身的优劣。
- GDN 的门控 `in_proj_a/b` 保持 BF16 后，各格式的 PPL 变化都在 ±0.06 以内，排序不变（`results/ppl_qwen35_9b_gates_bf16.json`）。所以 MXFP4 的退化不是门控造成的。

### GSM8K（ATOM 端到端，`local-completions`，1319 题，greedy）

| 格式 | 5-shot flexible | 5-shot strict | 3-shot flexible | 3-shot strict | 4 次平均比 MXFP4 |
|---|---|---|---|---|---|
| BF16 | 85.97 | 87.79 | 85.67 | 86.96 | +14.12 |
| MXFP4（asm） | 72.33 | 72.93 | 72.25 | 72.40 | 0 |
| 只做激活，128 | 78.01 | 77.86 | 77.18 | 76.27 | +4.85 |
| 两侧 256 | 75.82 | 76.57 | 75.28 | 75.06 | +3.20 |
| 两侧 128，权重 MBS-S | 74.15 | 74.68 | 75.36 | 76.35 | +2.65 |
| 只做权重，128 | 75.59 | 76.95 | 70.81 | 75.97 | +2.35 |
| 两侧 128（MBS-H） | 73.84 | 75.97 | 71.57 | 74.53 | +1.50 |
| 两侧 512 | 74.37 | 77.10 | 70.58 | 72.55 | +1.18 |

- 单次测量的标准误约 1.2 个点。逐题配对后，换一种格式会有 300 多题的对错发生变化，所以相近的变体分不开。
- 只做激活的 128，在 4 次测量里都显著高于 MXFP4（符号检验 p ≤ 0.003）。
- 两侧 128 和两侧 512 在 5-shot strict 口径下显著（p = 0.028 和 0.002），3-shot 时不再显著：两侧 128 在 strict 口径下 +2.1（p = 0.15），flexible 口径下低 0.7；两侧 512 两种口径分别是 +0.2 和 −1.7。
- GSM8K 和困惑度对变体的排序几乎相反：只做激活在 GSM8K 上最好，在困惑度上最差。这正说明，要在变体之间做选择，需要更大的评测。

### AIME 2026（ATOM 端到端，思考模式，`reason_eval.py`）

设置：

- 题目是 `math-ai/aime26` 的 30 道题，每题独立采样 5 次，共 150 个回答。BF16、两侧 128、两侧 512 跑了 8 次，下面统一只取前 5 次。
- 采样参数用 `reason_eval.py` 的默认值，和 LeapQuant 实验相同：temperature 1.0、top_p 0.95、top_k 20、presence_penalty 0，最多输出 81,920 个 token。
- 服务和 GSM8K 相同（单卡 TP1、CUDA graph、bf16 KV cache、关闭前缀缓存），另加 `--max-model-len 90112`。
- 只认 `</think>` 之后正文里的最后一个 `\boxed{}`。写满 81,920 个 token 还没结束的回答记为错。

| 格式 | 准确率 | 95% 区间 | 平均输出 token | 被截断 | 比两侧 128 好 / 差的题 |
|---|---|---|---|---|---|
| BF16 | 85.3 | 74.7–94.7 | 34.7k | 6.7% | 14 / 1（p = 0.001） |
| 两侧 128（MBS-H） | 62.7 | 48.0–77.3 | 57.4k | 35.3% | – |
| 两侧 512 | 60.0 | 45.3–74.0 | 57.4k | 35.3% | 5 / 8（p = 0.58） |
| 只做激活，128 | 48.7 | 34.0–64.0 | 64.8k | 50.0% | 3 / 12（p = 0.035） |
| 两侧 256 | 46.0 | 32.7–60.0 | 62.0k | 53.3% | 4 / 16（p = 0.012） |
| 两侧 128，权重 MBS-S | 41.3 | 27.3–54.7 | 64.6k | 55.3% | 2 / 17（p = 0.001） |
| 只做权重，128 | 40.0 | 25.3–54.7 | 68.0k | 58.0% | 2 / 18（p < 0.001） |
| MXFP4（asm） | 6.0 | 2.0–11.3 | 80.1k | 94.0% | 0 / 24（p < 0.001） |

- 准确率是 150 个回答的平均。区间是对 30 道题重采样得到的，题少所以很宽。
- 最后一列逐题比较每题 5 次的平均准确率，数出变好和变差的题数，用符号检验。这比区间灵敏得多。
- 和 BF16、MXFP4 比：
  - 每种量化格式都显著差于 BF16（p ≤ 0.001）。
  - 每个 MBS 变体都在 16–24 道题上好于 MXFP4，没有一道更差（p < 0.001）。
- 单独一次 30 题的运行（同一个采样序号），准确率的标准差是 3–8 个点。比如 BF16 的 5 次在 80.0–90.0 之间，两侧 128 在 56.7–70.0 之间，所以只跑一次很容易排错相近的变体。

**输在哪里：想得更久，停不下来**

| 格式 | 被截断 | 截断回答以短循环结尾 | 自然结束的回答：准确率 | 自然结束的回答：长度（BF16 = 1） |
|---|---|---|---|---|
| BF16 | 6.7% | 0% | 91.4 | 1.00 |
| 两侧 128 | 35.3% | 9% | 96.9 | 1.70 |
| 两侧 512 | 35.3% | 30% | 92.8 | 1.64 |
| 只做激活，128 | 50.0% | 37% | 97.3 | 2.03 |
| 两侧 256 | 53.3% | 49% | 98.6 | 1.57 |
| 两侧 128，权重 MBS-S | 55.3% | 41% | 92.5 | 1.78 |
| 只做权重，128 | 58.0% | 22% | 95.2 | 2.13 |
| MXFP4 | 94.0% | 62% | 100（只有 9 个） | 3.15 |

（`python aime_analyze.py --samples 5`。长度是同一道题上自然结束的回答的平均 token 数和 BF16 之比，再对题目取几何平均。）

- 自然结束的回答，准确率和 BF16 一样高，甚至更高。这里有选择效应：能写完的多半是这个模型本来就会做的题。
- 损失几乎全部来自截断：
  - 即使是自然结束的回答，同一道题量化后也要多写 57–113% 的 token，MXFP4 是 3 倍多。
  - MBS 各变体有 35–58% 的回答写满 81,920 个 token 还没有给出答案。
- 截断的回答分两种：
  - 退化循环："短循环"指最后 200 个非空行里不同的行不到一半。MXFP4 的截断回答大多是这种，比如连续几千行 "Wait."，或者三句话轮流重复。
  - 迟迟不收尾：两侧 128 的截断回答只有 9% 是短循环。按中位数，结尾 200 行里有 82% 是前面没写过的内容，模型在反复验算、换方法重做。
- MBS 各变体截断的回答里，有 15–41% 在推理中已经写出过正确答案，只是没有收尾。
- 所以 token 预算越紧，量化的损失越大。下表是在预算以内自然结束并答对的比例，等价于用这个 `max_tokens` 重跑：

  | 格式 | 16k | 32k | 48k | 64k | 81,920 |
  |---|---|---|---|---|---|
  | BF16 | 8.0 | 56.7 | 78.7 | 84.0 | 85.3 |
  | 两侧 128 | 1.3 | 20.0 | 42.0 | 54.0 | 62.7 |
  | 两侧 512 | 3.3 | 20.0 | 40.7 | 52.0 | 60.0 |
  | 只做激活，128 | 0.0 | 8.0 | 31.3 | 39.3 | 48.7 |
  | MXFP4 | 0.0 | 1.3 | 2.7 | 4.7 | 6.0 |

  BF16 到 64k 已基本饱和，MBS 到 81,920 还在上升，给更多 token 还能再找回一些。
- 这也解释了为什么 GSM8K 显示不出这么大的差别：GSM8K 的回答只有几百个 token，误差来不及积累成停不下来的推理。

**和困惑度、GSM8K 的关系**

- AIME 的排序和困惑度大体一致：8 种格式的秩相关是 0.81，只看 6 个 MBS 变体是 0.54。和 GSM8K 相反，6 个变体上是 −0.37。
- 两侧 256 在困惑度和 AIME 上都比 128 和 512 差。两个评测用的是同一份量化权重，所以这反映的是这份权重本身，而不是测量噪声。但它同样不能说明宏块 256 这个格式更差，见困惑度一节关于"舍入落在哪里"的讨论。
- 同理，两侧 128 用 MBS-S 和 MBS-D 量化的两份权重，困惑度几乎相同（8.87 对 8.84），AIME 却差 21 个点。这是不是权重搜索带来的稳定收益，要在更多模型上确认。

**关于 presence_penalty**

- Qwen3.5 模型卡推荐思考模式用 `presence_penalty=1.5`，它可能抑制循环。但 ATOM 的 OpenAI 接口只接收这个参数，并不实现：`protocol.py` 里的注释是 "Accepted for compatibility, not actively used"，`SamplingParams` 里也没有这一项。所以这里只测了 0。
- 最初起过两组 presence_penalty=1.5 的对照，发现参数无效后停掉了，残留结果放在 `results/e2e_aime_pp15/`，不要使用。

**注意**

- 跑到一半时的准确率不能用。短回答先完成，被截断的回答要写满 81,920 个 token 才返回，所以中途的准确率明显偏高。比如只做激活的 128，跑到一半时是 97%，最终是 48.7%。
- 两侧 256、只做权重、权重 MBS-S 这三组是在并发 100 下跑的。其余几组是并发 160，并且按 8 次采样提交，多出来的采样表里不用。并发只影响 KV cache 满了以后的抢占和重算，不改变输出的分布。
- 这些运行用到的每个 GEMM 配置都由 `test_configs.py` 确认过数值正确，所有回答里也没有出现 NaN 解码出来的 `!!!!!!!!`。8 组运行用的是同一份 ATOM 代码和调优表。

## 性能

### kernel 级（`bench_gemm.py`，CUDA graph 计时，L2 热）

对 Qwen3.5-9B 的 7 个 Linear 形状取几何平均。括号里先是相对 asm 的速度，再是相对同一 kernel 关闭 MBS 的速度；数字越大越快。

| M | asm（µs） | 关闭 MBS（µs） | 两侧 128 | 只做激活 128 | 两侧 256 | 两侧 512 |
|---|---|---|---|---|---|---|
| 1 | 9.8 | 7.9（1.24） | 12.0（0.82 / 0.66） | 9.4（1.05 / 0.84） | 9.1（1.08 / 0.87） | 8.4（1.17 / 0.94） |
| 16 | 10.6 | 9.5（1.12） | 13.1（0.81 / 0.72） | 10.6（1.00 / 0.90） | 9.2（1.15 / 1.03） | 8.7（1.21 / 1.08） |
| 64 | 10.9 | 11.9（0.92） | 17.5（0.62 / 0.68） | 14.0（0.78 / 0.85） | 14.0（0.78 / 0.85） | 12.6（0.86 / 0.94） |
| 256 | 14.1 | 20.5（0.69） | 33.6（0.42 / 0.61） | 27.0（0.52 / 0.76） | 26.5（0.53 / 0.77） | 25.4（0.56 / 0.81） |
| 1024 | 30.7 | 46.4（0.66） | 75.1（0.41 / 0.62） | 57.9（0.53 / 0.80） | 59.0（0.52 / 0.79） | 54.0（0.57 / 0.86） |
| 4096 | 93.7 | 127.9（0.73） | 235.3（0.40 / 0.54） | 187.2（0.50 / 0.68） | 183.4（0.51 / 0.70） | 172.7（0.54 / 0.74） |
| 8192 | 189.4 | 244.9（0.77） | 465.6（0.41 / 0.53） | 370.7（0.51 / 0.66） | 362.4（0.52 / 0.68） | 334.4（0.57 / 0.73） |

- MBS 关闭时，大 M 下这个 kernel 达到 2200–3000 TFLOPS，高于 aiter 自带的 Triton preshuffle 版（2000–2700），但仍低于 asm。asm 在多数形状上是 3100–4200，只有 N=12352 时约 2000。
- 激活量化的耗时：M=1 时各宏块都是 2.7–3.0 µs，和普通 MXFP4 一样；M=1024 时从 6.6 µs 变成 8.5–9.1 µs，多出来的是为了算比值而多读的前一个宏块。

### ATOM 端到端吞吐（`e2e.sh bench`）

Qwen3.5-9B，单卡 MI355X，TP1，CUDA graph，关闭前缀缓存，随机数据。单位为输出 tok/s，括号里是相对 MXFP4 的倍数。

| 格式 | 1024→1024，并发 64 | 4096→256，并发 32 | 1024→1024，并发 256 |
|---|---|---|---|
| BF16 | 6993（0.83） | 2008（0.72） | 10886（0.76） |
| MXFP4（asm） | 8395 | 2789 | 14240 |
| 两侧 128（MBS-H） | 6762（0.81） | 1995（0.72） | 11059（0.78） |
| 只做激活，128 | 7501（0.89） | 2244（0.80） | 11764（0.83） |
| 两侧 256 | 7407（0.88） | 2098（0.75） | 11983（0.84） |
| 两侧 512 | 7712（0.92） | 2270（0.81） | 12441（0.87） |

- TTFT 和 TPOT 的完整表用 `python bench_summary.py` 生成。
- MXFP4 在两张卡上各测了一次，结果相差不到 0.5%。

### 为什么慢

以 M=8192、`gate_up_proj` 调出的配置为例（`results/dump_asm_M8192.txt`），一个 K 步（256 个 K）里：

| | tile | MFMA | `v_pk_mul` | 寄存器 |
|---|---|---|---|---|
| 关闭 MBS | 256×256，8 warp | 64 条 | 0 | 244 VGPR，无 spill |
| 两侧 128 | 128×256，4 warp，MFMA 32×32 | 32 条 | 253 条 | 408 个寄存器，无 spill |
| 两侧 256 | 256×128，8 warp | 32 条 | 64 条 | 195 VGPR |

- 253 条 packed FP32 乘法大约占 1000 个周期，和 32 条 32×32×64 MFMA 的时间相当。这和笔记 7.3 节"VALU 基本占满"的估算一致。
- 指令序列是 `VVVV…VVVV` 之后接 `MMMM…`，乘法和 MFMA 完全没有交错，矩阵核心在 VALU 期间空转。
- 关闭 MBS 时这个 kernel 也只用到 MFMA 峰值的约 24%，可这些 VALU 指令同样没有藏进空档，而是直接叠加到了关键路径上。

试过但没有用的办法：

1. **新部分和加 FMA**（论文 Figure 5 的写法）：需要两套累加器，寄存器压力翻倍。同一个 preshuffle kernel、同样 128×128 的 tile 下，这种写法是 930 TFLOPS，缩放累加器是 1216。最早的版本还叠加了精确除法、A 和 B 都经过 LDS，有 234 条 AGPR↔VGPR 搬运和 82 个 VGPR spill，只有 306 TFLOPS。
2. **kernel 里从 uint8 的 m 现算 σ**：`fdiv` 生成的是精确除法，再加上取模寻址没法向量化，导致 207 个 VGPR spill、363 个 SGPR spill，M=1 时慢 2.3 倍。改成读预先算好的 fp32 比值后恢复。
3. **LLVM 调度策略**（`llvm_fn_attrs="amdgpu-sched-strategy=..."`，试了 max-ilp、iterative-ilp、iterative-minreg、max-memory-clause）：耗时变化在 ±7% 以内（测量时卡上还有别的任务，只能看相对值），乘法仍然扎成一串。
4. **奇数宏块单独算部分和，再用一次 FMA 合并**：编出来的调度一样，耗时没有变化。

要把这部分开销藏住，需要在 MFMA 之间逐条插入乘法，正是仓库里 HIP wg kernel 的做法。不过那个 kernel 的累加器钉在 AGPR 上，VALU 读写不了，寄存器要重新规划（见笔记 7.3 节）。

## 局限和下一步

- 端到端只测了 Qwen3.5-9B 一个 dense 模型。MoE 的 expert 没有 MBS（fused MoE kernel 不支持），TP > 1 也没有测过。
- 精度方面，GSM8K 和困惑度不足以在 MBS 各变体之间做选择。AIME 能分开一部分，但只有 30 道题，而且变体之间的差别还受"这份权重舍入落在哪里"的影响。下一步可以：
  - 换更大的评测（MMLU-Pro、多任务平均），或者换更多模型。
  - 或者在困惑度上做多次扰动，估计"舍入偶然性"带来的方差。
- 长推理方面，量化模型主要输在推理变长、停不下来。可以试：
  - 在 ATOM 的采样器里实现 presence_penalty，测模型卡推荐的设置。
  - 到预算上限时强制插入 `</think>`，让模型给出答案。截断回答里已有 15–41% 写出过正确答案。
  - 找出对长推理最敏感的层，让它们保持 BF16。
- 性能方面，在 Triton 里继续优化的空间不大。更值得做的是：
  - 在 HIP kernel 里实现交错发射。
  - 或者先采用宏块 512 这种开销小的配置。
  - 小 M 时还可以加 split-K（M ≤ 64 时 N=4096 的层只有 32–64 个 tile）。

## 复现

在本目录下执行。卡号是 HIP 编号；跑性能要用独占的卡。

```bash
export AITER_LOG_LEVEL=WARNING
HIP_VISIBLE_DEVICES=4 python test_mxfp4_mbs.py
PYTHONPATH=/home/haoyanli/a121/lhy_atom/ATOM_mbs HIP_VISIBLE_DEVICES=4 python test_atom_op.py
HIP_VISIBLE_DEVICES=4 python tune_gemm.py --variants plain mbs128 mbs256 mbs128_a mbs256_a mbs512
HIP_VISIBLE_DEVICES=4 python bench_gemm.py --out results/bench_gemm.json
HIP_VISIBLE_DEVICES=0 python hf_eval.py qsnr --out results/qsnr_qwen35_9b.json
HIP_VISIBLE_DEVICES=0 python hf_eval.py ppl --formats bf16 mxfp4 mbs128 mbs512 --out results/ppl_qwen35_9b.json
bash atom_e2e/e2e.sh run mbs512 7 18300                                       # 起服务，跑 5-shot GSM8K
NUM_FEWSHOT=3 OUT=$PWD/results/e2e_3shot bash atom_e2e/e2e.sh run mbs512 7 18300
SKIP_GSM8K=1 BENCH=1 bash atom_e2e/e2e.sh run mbs512 7 18300                   # 吞吐
bash atom_e2e/e2e.sh summary; python gsm8k_pair.py mxfp4 mbs512; python bench_summary.py
PYTHONPATH=/home/haoyanli/a121/lhy_atom/ATOM_mbs HIP_VISIBLE_DEVICES=4 python test_configs.py   # 约 3 分钟
OUT=$PWD/results/e2e_aime MAX_LEN=90112 SKIP_GSM8K=1 REASON=1 SAMPLES=5 CONC=100 \
  bash atom_e2e/e2e.sh run mbs512 7 18300                                     # AIME 2026，约 2 小时
python aime_summary.py --base bf16 --samples 5; python aime_analyze.py --samples 5
```

- AIME 的环境变量要单独 `export`，或者像上面这样写在命令前面。不要写成 `export ... && (...) &`，那样变量只在后台子 shell 里生效。
- 一组跑了 8 次采样、只想用前 5 次时，`bash atom_e2e/stop_after_samples.sh 5 <格式>...` 会在前 5 次全部做完后停掉评测。

`results/` 里的文件：

| 文件 | 内容 |
|---|---|
| `qsnr_qwen35_9b.json` | 每层、每种格式的 QSNR |
| `ppl_qwen35_9b*.json` | 困惑度 |
| `bench_gemm.json`、`baseline.json` | kernel 计时 |
| `tune_gemm.jsonl`、`bad_configs.json` | 调优记录，以及让编译器崩溃或超出 LDS 的配置 |
| `e2e/<格式>/`、`e2e_3shot/<格式>/` | 服务日志、`online_quant_info`、GSM8K 逐题样本、吞吐 JSON |
| `bench_final_gpu*.log`、`bench_bf16_mxfp4_gpu6.log` | 吞吐测试的日志 |
| `bench_prelim_gpu7.log` | 量化启动优化前的早期结果，已被 `bench_final_*` 取代 |
| `dump_asm_M8192.txt` | 热循环的指令构成 |
| `test_configs.log` | 调优表配置检查的输出 |
| `reason/<格式>/aime26.jsonl`、`aime26_outputs.jsonl` | AIME 每个回答的记录和全文 |
| `e2e_aime/<格式>/` | AIME 的服务日志、评测日志、`online_quant_info` |
| `e2e_aime_pp15/` | 停掉的 presence_penalty=1.5 对照。这个参数在 ATOM 里无效，结果不要使用 |
| `aime_*.log`、`stop_after_samples.log` | AIME 各次启动的日志，以及自动停止的记录 |
