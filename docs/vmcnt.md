# vmcnt：等全局访存的计数器

**vmcnt 是每个 wave 的硬件计数器，记录已经发出、但还没完成的向量访存指令条数。** 发一条 `buffer_load` / `global_load` / `buffer_store`，计数加 1；这条指令完成后，计数减 1。访存是异步的，发出后 wave 继续往下执行。要用 load 回来的数据，先执行 `s_waitcnt vmcnt(N)`：一直等到未完成条数 ≤ N。

| 计数器 | 统计的指令 |
|---|---|
| vmcnt | global / buffer 的 load 和 store |
| lgkmcnt | LDS 读写（`ds_read` / `ds_write`）、标量 load（`s_load`） |
| expcnt | export |

注意：`buffer_load ... lds`（A DMA）虽然把数据写进 LDS，走的仍是向量内存通路，算 vmcnt，不算 lgkmcnt。

## 按发射顺序完成，所以可以不等到 0

vmcnt 按发射顺序递减，所以“剩余 ≤ N 条”就等于“除了最后发出的 N 条，前面的都完成了”。例：依次发出 L1–L5，要用 L2，它后面还有 3 条，写 `vmcnt(3)` 即可，不用等 L3–L5。

kernel 有多级预取，计算 step s 时，后面几个 stage 的 load 还在路上。写 `vmcnt(0)` 会把这些预取也一起等完，流水就断了。所以要算出“要等的那批之后又发了多少条”，按这个数等待。

## 本 kernel 的记账

每个 wave 每个 stage 发 `LOADS_PER_STAGE` = 20 条：编号 [0, `KA`) = [0, 10) 是 B 片段 8 条 + B scale 2 条，在段 0 发；[10, 20) 是 A DMA 8 条 + A scale DMA 2 条，在段 1 末尾发。

- **`W_START`**（step s 第一条 MFMA 前）：要等本 stage 的 B / B scale，它们在 step s−`PF_B` 的段 0 发出。之后发的是那个 step 的 A DMA（20 − 10 条），再加中间 `PF_B` − 1 个 step 的全部指令。
  `W_START = (LOADS_PER_STAGE − KA) + LOADS_PER_STAGE × (PF_B − 1)`，`PF_B` = 1 时为 10。
- **`W_MID`**（段 1 的 barrier 处）：要等 stage s+1 的 A DMA，它在 step s+1−`PF` 的末尾发出。之后发的是 step s+2−`PF` 到 s−1 的全部指令，再加本 step 段 0 的 10 条寄存器 load。
  `W_MID = KA + LOADS_PER_STAGE × (PF − 2)`，`PF` = 2 时为 10。
- **`EPI_STORES`**：每个 tile 的 epilogue 在两个 step 之间发出 `MSUB × NSUB / 2` = 32 条 buffer store，也计入 vmcnt，所以下一个 tile 的第一个 step 要把阈值再放宽 32。

## 编码限制

gfx9 上 vmcnt 只有 6 位，最大 63，所以有 `static_assert(W_MID + EPI_STORES <= 63 && W_START + EPI_STORES <= 63)`。`s_waitcnt` 立即数里，vmcnt 的低 4 位在 [3:0]、高 2 位在 [15:14]，expcnt 在 [6:4]，lgkmcnt 在 [11:8]。`wait_vm_lgkm0<n>` 拼成 `(n & 15) | (7 << 4) | (0 << 8) | ((n >> 4) << 14)`：等到 vmcnt ≤ n，同时等 lgkmcnt = 0，expcnt 填最大值 7 表示不等。
