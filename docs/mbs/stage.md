# 总结：一个 stage 里 A、B 怎么从显存到寄存器

以 256×256 tile（MSUB = 16、NSUB = 4）为例。tile 和 stage 管的是不同方向：tile 是 C 的一块 256（M）× 256（N）输出，stage 是 K 方向上的 256 个 K。一个 tile 要沿 K 走完 K/256 个 stage 才算完（K = 5120 时 20 个）；每个 stage 把这 256 个 K 的贡献累加到全部 256 × 256 个输出上，每个 wave 负责其中 256 行 × 64 列，发 16（mi）× 4（ni）× 2（h）= 128 条 MFMA，整个工作组 512 条。下面说的都是一个 stage 里的事：要读 A 的 256 行 × 256 个 K（32 KB）和 B 的 256 行 × 256 个 K（32 KB，每个 wave 8 KB）。

**A：显存 → LDS → 寄存器。**

- **显存 → LDS：** 4 个 wave 分着搬，wave 0 搬第 0–63 行 × 256 个 K，共 8 KB。每条 `buffer_load_dwordx4 … lds` 搬 1 KB（8 行 × 128 B），发 8 条。
- **LDS → 寄存器：** 每个 wave 要读全部 256 行，不只是自己搬的 64 行。一条 `ds_read_b128`（64 个 lane × 16 B = 1 KB）正好是一条 MFMA 的 A 操作数：子块 mi 的 16 行 × K 半段 h 的 128 个 K。lane L 读第 mi·16 + L%16 行、第 h·4 + L/16 个 32-K 块，LDS 地址按 XOR swizzle 换算：`chk = (h·4 + g) ^ (row & 7)`，地址 = `row·128 + chk·16`（`frag_at`）。例：lane 37、mi = 8、h = 0，行 133，逻辑块 2（K 64–95），物理块 2 ^ 5 = 7，槽内偏移 17136。
- **一个 stage 共多少：** 32 条 `ds_read_b128`（16 个子块 × 2 个半段，32 KB），另外 A scale 有 8 条 `ds_read_b32`。
- **分两段、提前一段读：** M 分成两段，各 8 个子块（MS0 = MS1 = 8）；A 片段双缓冲 `afr0`、`afr1`，每个 16 个片段，每个 lane 占 64 个 VGPR。段 0 算 mi 0–7，同时读本 stage mi 8–15 的片段和 scale 第 4–7 块；段 1 算 mi 8–15，同时从下一个槽读下个 stage mi 0–7 的片段和 scale 第 0–3 块。

**B：显存 → 寄存器。**

一个 tile 在 N 方向有 256 行，先按 wave 分成 4 份，每个 wave 64 行；每个 wave 再把自己的 64 行切成 4 个 16 行的子块，编号 ni = 0–3。所以整个 tile 一共有 16 个 B 子块，`NSUB = 4` 指的是每个 wave 的子块数：

| wave | 负责 tile 里的 N 行 | ni = 0 | ni = 1 | ni = 2 | ni = 3 |
|---|---|---|---|---|---|
| 0 | 0–63 | 0–15 | 16–31 | 32–47 | 48–63 |
| 1 | 64–127 | 64–79 | 80–95 | 96–111 | 112–127 |
| 2 | 128–191 | 128–143 | 144–159 | 160–175 | 176–191 |
| 3 | 192–255 | 192–207 | 208–223 | 224–239 | 240–255 |

wave 之间的区别在描述符基址里：`t.n0` 带着 `wave * WNTILE`（64·wave），`t.b = (t.n0 >> 4)·KT·512` 指向本 wave 的第一个 16 行组；`b_voff[ni]` 只在这个起点上再跳 ni 个 16 行组，所以 4 个 wave 的 `b_voff` 完全一样。

- **wave 0 负责什么：** N 方向第 0–63 行 × 256 个 K = 64 × 128 B = 8 KB。B 已离线预重排，一个 16 行子块 ni 的一个 K 半段 h 在显存里是连续的 1 KB。
- **一条 load 读什么：** 一条 `buffer_load_dwordx4`（不带 `lds`）读这 1 KB，lane L 在偏移 16L，读进来就是 MFMA 的 B 操作数，不用再换形状。例：lane 37、ni = 1、h = 0，读第 n0 + 21 行、K 64–95，在这 1 KB 里的偏移是 592。
- **一个 stage 共多少：** 8 条（4 个子块 × 2 个半段），写进 `bf[ni][h]`，每条占 4 个 VGPR，一套 32 个；另外 B scale 有 2 条 `buffer_load_dword`，写进 `bs[0]`、`bs[1]`。
- **提前多少：** B 提前 1 个 stage（PF_B = 1）：step s 的段 0 里发 stage s+1 的这 10 条（B 8 条 + B scale 2 条），所以同一时刻活着 2 套 B。

B 的 8 条 load 按 `issue_one` 里的编号 k = 0–7 拆成 ni = k/2、h = k%2，第 k 条写进 `bf[ni][h]`。行号是本 wave 那 64 行（N 方向）里的行号，K 是本 stage 内的 K：

| k | 子块 ni | 半段 h | 读的是 | voffset + 立即数 |
|---|---|---|---|---|
| 0 | 0 | 0 | 第 0–15 行，K 0–127 | `b_voff[0]` + 0 |
| 1 | 0 | 1 | 第 0–15 行，K 128–255 | `b_voff[0]` + 1024 |
| 2 | 1 | 0 | 第 16–31 行，K 0–127 | `b_voff[1]` + 0 |
| 3 | 1 | 1 | 第 16–31 行，K 128–255 | `b_voff[1]` + 1024 |
| 4 | 2 | 0 | 第 32–47 行，K 0–127 | `b_voff[2]` + 0 |
| 5 | 2 | 1 | 第 32–47 行，K 128–255 | `b_voff[2]` + 1024 |
| 6 | 3 | 0 | 第 48–63 行，K 0–127 | `b_voff[3]` + 0 |
| 7 | 3 | 1 | 第 48–63 行，K 128–255 | `b_voff[3]` + 1024 |

**A、B 的 scale：一个 lane 读一个 dword。**

scale 每 32 个 K 一个字节，一个 stage 每行 8 个。量化时已经 shuffle 好：每 32 行 × 一个 stage 连成 256 B，第 L 个 dword 正好是 lane L 要的 4 个字节，即第 row16、row16+16 两行（相邻两个 16 行子块）× 第 g、g+4 两个 32-K 块（前后两个 K 半段），字节号 = 2h + 子块奇偶。所以 A、B 的 scale 都是一个 lane 读一个 dword，lane 偏移都是 `s_lane = 4·lane`。例：lane 37 读偏移 148，拿到（第 5 行，块 2）、（第 21 行，块 2）、（第 5 行，块 6）、（第 21 行，块 6）。

- **shuffle 的原理：** 不 shuffle 时 scale 是行主序 `[M, K/32]`，lane 37 要的这 4 个字节在 802、806、3362、3366（scaleN_pad = 160），散在两行里，要读 4 次。量化时传 `shuffle=True`（和 aiter 的 `shuffle_scale` 一样）把它重排，做法和 B 的 shuffle 相同，也是 `view` 拆维度、`permute` 换顺序：

  ```python
  scale = scale.view(sm // 32, 2, 16, sn // 8, 2, 4)   # [32 行组, 子块奇偶, 行%16, stage, h, 块%4]
  scale = scale.permute(0, 3, 5, 2, 4, 1)              # [32 行组, stage, 块%4, 行%16, h, 子块奇偶]
  ```

  于是每个 32 行组 × stage 的 256 B 里，偏移 = 64·g + 4·row16 + 2h + 奇偶 = 4·(16g + row16) + (2h + 奇偶) = `4·lane + 字节号`。一个 lane 要的 4 个字节被挤进同一个 dword，dword 又按 lane 顺序排，64 个 lane 读的是连续的 256 B（两条满的 cache line）。A、B 的 scale 用的是同一种 shuffle。
- **A scale，显存 → LDS → 寄存器：** tile 的 256 行是 8 个 32 行块，每个 stage 共 8 × 256 B = 2 KB，4 个 wave 都要用。和 A 一样先 DMA 进 LDS：每个 wave 发 2 条 `buffer_load_dword … lds`，wave w 搬第 w、w+4 块，放在槽里 A 数据的后面（`STAGE_A + p·256`，`STAGE_A` = 32 KB，所以一个槽是 34 KB）。之后每个 wave 用 8 条 `ds_read_b32` 从 `STAGE_A + p·256 + 4·lane` 读出 8 块里本 lane 的 dword（`scale_at`）。
- **B scale，显存 → 寄存器：** 本 wave 的 64 行是 2 个 32 行块，每个 stage 发 2 条 `buffer_load_dword`，写进 `bs[0]`、`bs[1]`，`bs[p]` 管子块 ni = 2p、2p+1。地址 = 描述符基址 `tob.bs + stb·256` + voffset `4·lane` + soffset `p·32·scaleN_pad`（256 行 tile 打开了 SCALE_SOFF）。
- **MFMA 怎么用：** 一个 dword 管 2 个子块 × 2 个 K 半段，MFMA 用 `op_sel` 直接挑第 2h + 奇偶（mi & 1 或 ni & 1）个字节，不用解包。
- **提前多少、占多少寄存器：** A scale 跟着 A 提前 2 个 stage，在段 1 里和 8 条 A DMA 一起发；从 LDS 读的时机和 A 片段一样分两段，段 0 读本 stage 第 4–7 块，段 1 读下个 stage 第 0–3 块，双缓冲 `asc0`、`asc1` 各 4 个 dword，共 8 个 VGPR。B scale 跟着 B 提前 1 个 stage，在段 0 里和 8 条 B load 一起发，同时活着 2 套，共 4 个 VGPR。

**B 和 scale 在显存里的排法：先按行分组，组内一个 stage 接一个 stage 连续存放。**

B 的数据按 16 行一组重排：一组 16 行 × 5120 个 K，即 16 × 2560 B = 40960 B，在显存里是连续的一段；组内一个 stage 接一个 stage 排，每个 stage（16 行 × 256 个 K）是连续的 2 KB。A、B 的 scale 按 32 行一块重排：一块 32 行 × 160 个 scale，即 5120 B，也是连续的；块内每个 stage（32 行 × 8 个 32-K 块）是连续的 256 B。

| | B 的数据 | A、B 的 scale |
|---|---|---|
| 行怎么分 | 16 行一组，40960 B | 32 行一块，5120 B |
| 组内一个 stage | 2 KB，在 2048·s | 256 B，在 256·s |
| stage 里面 | 前、后两个 K 半段各 1 KB，每 1 KB 里 lane L 在 16·L | lane L 的 dword 在 4·L |
| 一条指令读多少 | 1 KB（一个半段） | 256 B（一整块） |

A 没有重排，是行主序：一行在 stage s 里的 128 B 是连续的，正好一条 cache line；但同一个 stage 的 8 行是 8 段分开的 128 B，彼此隔着一整行。所以 A 要按整行 DMA 进 LDS，再从 LDS 按 MFMA 要的形状读出来。

**`issue_one` 的编号 k：一个 step 里的 20 条全局访存。**

256×256 的 tile 每个 step 一共发 20 条全局访存（第 693 行，`LOADS_PER_STAGE` = 8 + 2 + 8 + 2）。`issue_one` 用编号 k 区分这次发的是哪一条，编号按区间划分（第 694–695 行）：

| k | 是什么 | 由 k 推出的 |
|---|---|---|
| 0–7 | B 片段 | ni = k/2、h = k%2，写进 `bf[ni][h]`，用 `b_voff[ni]`，立即数 `h·1024` |
| 8–9 | B scale | p = k − 8，写进 `bs[p]`，soffset 是 `p·32·scaleN_pad` |
| 10–17 | A 的 DMA | j = k − 10，用 `a_voff[j]`，M0 = `lds_base + lds_wave_off + j·1024` |
| 18–19 | A scale 的 DMA | j = k − 18，搬第 `min(j·4 + wave, 7)` 块 |

段 0 发 k = 0–9，即 B 8 条 + B scale 2 条，共 10 条；段 1 发 k = 10–19，即 A 8 条 + A scale 2 条，共 10 条（第 980 行）。它们不是在段的开头或结尾一次发完，而是一条一条插在这一段的 MFMA 之间，具体位置见下一节。

- **k 怎么变成编译期常量：** 调用时写的是 `issue_one(std::integral_constant<int, KLO + k>{}, …)`（第 1006 行）。参数 `kc` 的类型 `integral_constant<int, N>` 本身就带着 N，函数里用 `decltype(kc)::value` 取出来，得到的 `k` 是 `constexpr`。如果直接传 `int`，它在函数体里就是运行时的值。`static_for`（第 427–437 行）是同一个套路：依次用 `integral_constant<int, 0>`、`integral_constant<int, 1>`……调用函数，每个不同的 k 编出一份单独的代码。
- **为什么一定要编译期常量：** `issue_one` 里是一串 `if constexpr (k < KBS) … else if constexpr (k < KA) …`，编译时就选定分支，每次调用都展开成一条确定的指令，寄存器和立即数都是固定的，运行时没有分支，也没有下标计算。例如 k = 3 推出 ni = 1、h = 1，编成一条读进 `bf[1][1]`、voffset 为 `b_voff[1]`、立即数 1024 的 `buffer_load_dwordx4`。k 还要当 `"i"` 立即数、当寄存器数组的固定下标；寄存器数组如果用运行时下标，编译器只能把它放进 scratch，或者多出一串选择指令。

# 两段流水：一个 stage 的数据什么时候进来、什么时候被用

以 256×256 tile 为例（PF = 2、PF_B = 1）。每个 step 算一个 stage，分成两段：段 0 算 mi 0–7，段 1 算 mi 8–15，每段 64 条 MFMA。下图每行是一种数据，从左到右是时间，格子里写的是 stage 号和写到哪。蓝色是 stage s 自己的数据，灰色是同一时刻别的 stage 在做的事。每段开头的竖线是这段开始前的等待：虚线是 ①，实线是 ②，② 还要过一次 barrier。

![一个 stage 的数据怎么流过两个段](./images/stage_pipeline.svg)

**一段里，load 和 LDS 读什么时候发。** 既不是在段的开头一次发完，也不是等到段的结尾，而是一条一条插在这一段的 64 条 MFMA 之间，每条都用 `sched_barrier` 钉住位置（`section` 的第 986–1006 行）：

| | 每段条数 | 插在第几条 MFMA 前（0–63） |
|---|---|---|
| 全局 load | 10 条：段 0 是 B 8 + B scale 2，段 1 是 A 8 + A scale 2 | 0、5、11、16、22、28、33、39、44、50；前 8 个位置是 B 或 A，第 44、50 条前是 scale |
| LDS 读 | 20 条：A 片段 16 + A scale 4 | 0、3、6、9、…、60，约每 3 条一条；最后 4 个位置（51、54、57、60）是 scale |

第 k 条 load 在第 ⌊k·56/10⌋ 条 MFMA 前，其中 56 = 64 × `LOAD_WIN` / 8（`LOAD_WIN` = 7，第 886 行）；第 r 条 LDS 读在第 ⌊r·64/20⌋ 条前。所以第一条 load 一过等待就发，最后一条在第 50 条 MFMA 前，段末 13 条 MFMA 之间不再发 load。汇编里每一段都是这样。

**stage s 的数据，按时间顺序：**

1. **step s−2 段 1：** 发 stage s 的 A 数据（8 条 `buffer_load_dwordx4 … lds`）和 A scale（2 条 `buffer_load_dword … lds`），共 10 条，写进 LDS 槽 s%4。
2. **step s−1 段 0：** 发 stage s 的 B 片段（8 条 `buffer_load_dwordx4`）和 B scale（2 条 `buffer_load_dword`），共 10 条，写进 `bf[s%4]`、`bs[s%4]`。
3. **step s−1 段 1：** 开头 ② 先等 A(s) 和 A scale(s) 到齐，再过一次 barrier，确认 4 个 wave 搬的都到了。然后从槽 s%4 读 mi 0–7 的 16 个 A 片段和 scale 块 0–3，放进 `afr0`、`asc0`。
4. **step s 段 0：** 开头 ① 等 B(s) 和 B scale(s) 到齐，同时等 `afr0`、`asc0` 读完。64 条 MFMA 算 mi 0–7：A 用 `afr0`，B 用 `bf[s%4]`，scale 用 `asc0`、`bs[s%4]`。算的同时读 mi 8–15 的片段和 scale 块 4–7，放进 `afr1`、`asc1`。
5. **step s 段 1：** 开头 ② 等 `afr1`、`asc1` 读完。64 条 MFMA 算 mi 8–15，A 用 `afr1`，B 还是 `bf[s%4]`。到这里 stage s 就算完了。

**只看 step s，两段各在做什么：**

| | 段 0 | 段 1 |
|---|---|---|
| 开头的等待 | ①：B(s)、B scale(s) 到齐（vmcnt ≤ 10）；afr0、asc0 读完（lgkmcnt = 0） | ②：A(s+1)、A scale(s+1) 到齐（vmcnt ≤ 10）；afr1、asc1 读完（lgkmcnt = 0）；再 barrier |
| MFMA（64 条） | stage s 的 mi 0–7 | stage s 的 mi 8–15 |
| LDS → 寄存器（20 条） | stage s 的 mi 8–15 → afr1（16 条），块 4–7 → asc1（4 条） | stage s+1 的 mi 0–7 → afr0（16 条），块 0–3 → asc0（4 条） |
| 全局 load（10 条） | B(s+1) 8 条 + B scale(s+1) 2 条 → bf、bs[(s+1)%4] | A(s+2) 8 条 + A scale(s+2) 2 条 → 槽 (s+2)%4 |

**对照代码：两处等待。** 参数在第 902–909 行，用在 `step` 的第 1039–1056 行。vmcnt 记的是本 wave 发出去、还没完成的显存访问条数，load 和 store 都算，而且按发射顺序完成（第 891 行的注释）。所以要等某一批 load，只要等到“最多剩它后面发的那几条”。按发射顺序看：

```text
step s−1 段 0：B(s) 8 条 + B scale(s) 2 条        ← ① 要等这 10 条
step s−1 段 1：A(s+1) 8 条 + A scale(s+1) 2 条    ← ② 要等这 10 条
step s   段 0：B(s+1) 8 条 + B scale(s+1) 2 条
step s   段 1：A(s+2) 8 条 + A scale(s+2) 2 条
```

- **`W_START`（第 906 行），① 处。** B(s) 和 B scale(s) 是在 step s−PF_B 的段 0 发的。它们后面发的有：同一个 step 段 1 的 A 类 `LOADS_PER_STAGE − KA` 条（256×256 是 A 8 条 + A scale 2 条 = 10），再加上中间隔着的 `PF_B − 1` 个整 step。256×256 的 PF_B = 1，所以是 10；PF_B = 2 的 tile 中间多隔一整个 step，比如 192×256 是 8 + 18 = 26。
- **`W_MID`（第 908 行），② 处。** A(s+1) 和 A scale(s+1) 是 step s−1 段 1 发的，而且是那个 step 最后发的 10 条。它们后面只发过本 step 段 0 的 B(s+1) 8 条 + B scale(s+1) 2 条（`KA` = 10），PF 更大时再加 `PF − 2` 个整 step，所以是 10。这要求 A 类必须是每个 step 最后发的，段 1 把 k = 10–19 放在最后就是为了这个。
- **`EPI_STORES`（第 902 行）。** 算完一个 tile，epilogue 把 C 写回：每个 wave 256 行 × 64 列 bf16，共 32 KB，每条 store 1 KB，共 32 条，即 `MSUB * (NSUB / 2)`。这些 store 比下一个 tile 已经预取的 B(0)、A(1) 更新，所以下一个 tile 的第一个 step 两处都放宽成 10 + 32 = 42（第 1039、1053 行）。不放宽结果也对，只是要白等 22 条 store 写完；放宽过头才会出错，因为要的数据可能还没到。
- **`static_assert`（第 909 行）。** vmcnt 字段只有 6 位，放宽后的值不能超过 63。11 种 tile 里最大的是 192×256 的 26 + 24 = 50。
- **`wait_vm_lgkm0(n)`（第 913–916 行）生成 `s_waitcnt vmcnt(n) lgkmcnt(0)`。** gfx9 的立即数里，vmcnt 的低 4 位在 [3:0]、高 2 位在 [15:14]；expcnt 在 [6:4]，填 7 表示不等；lgkmcnt 在 [11:8]，填 0 表示等全部，保证上一段读的 `afr`、`asc` 已经进了寄存器。例：n = 10 编成 0x007A，n = 42 编成 0x807A。
- **② 后面的 `__syncthreads()`（第 1056 行）。** A 是 4 个 wave 一起 DMA 进 LDS 的，每个 wave 的 vmcnt 只管自己发的那几条，要靠 barrier 确认别的 wave 也都到了。B 是每个 wave 自己的寄存器，① 处不需要 barrier。这是每个 stage 唯一的 barrier（第 869–871 行的注释）。
- **序幕和收尾。** K 循环开始前（第 1199–1200 行）用 vmcnt(20) 加一次 barrier，等齐 stage 0 的 A。kernel 退出前（第 1240 行）用 `WAIT_ALL` 只等 vmcnt(0)：最后几个 step 还在给凑数的“下一个 tile”发 LDS DMA，要等它们写完，工作组才能结束、交还 LDS。

图由 `scripts/draw_stage_figs.py` 生成，条数和 stage 号都从 kernel 的常量推出来。

# 一段（section）内部：算多大的块，64 条 MFMA 怎么排

以 256×256 tile 为例（MSUB = 16、NSUB = 4、MS0 = MS1 = 8）。

**两段是按 M 切的，不是按 K 切的。** 注释里的"段 0 / 段 1"就是代码里的 `section`：段 0 算 tile 的第 0–127 行（mi = 0..7），段 1 算第 128–255 行（mi = 8..15），每段都把本 stage 的 256 个 K 走完。按 K 切的那一层叫"K 半段"，就是 `h`，在段的里面：每段前 32 条 MFMA 算 K 0–127，后 32 条算 K 128–255。所以一个 step 的顺序是：前 128 行 × K 0–127 → 前 128 行 × K 128–255 → 后 128 行 × K 0–127 → 后 128 行 × K 128–255。

| | 段 0 | 段 1 |
|---|---|---|
| 每个 wave 用的 A | 第 0–127 行 × 256 K | 第 128–255 行 × 256 K |
| 每个 wave 用的 B | 本 wave 的 64 个 N × 256 K，8 份 `bf[ni][h]` 全用 | 同一份，寄存器不变 |
| 每个 wave 累加的 C | 128 × 64 | 128 × 64 |
| 4 个 wave 合起来的 C | 上半块 128 × 256 | 下半块 128 × 256 |
| 每个 wave 的 MFMA | 8（mi）× 4（ni）× 2（h）= 64 条 | 64 条 |

一条 MFMA 算 16 行 × 16 列 × 128 个 K。一块 256×256 tile 有 16 × 16 = 256 个 16×16 子块，K = 5120 时每块沿 K 累加 40 次，共 10240 条：每个 wave 每段 64 条、每个 step 128 条、整个 tile 2560 条；整个工作组每个 step 512 条。

**MS0 为什么是 8。** 一个 M 片段是 16 行（MFMA 一次算 16 行），256 行就是 MSUB = 16 个。`MS0 = (MSUB / 2 + 1) / 2 * 2`（第 883 行）的意思是"取一半，奇数就向上凑成偶数"。要偶数，是因为一个 scale dword 管相邻两个 M 片段，每段必须是整对。MSUB = 16、12、10、6、4 时，MS0 分别是 8、6、6、4、2，MS1 是剩下的。

**为什么按 M 切。** 按 K 切也能算对，每段同样是 64 条 MFMA，同样要预读 16 个 A 片段（64 个 VGPR）。区别在 scale：一个 A scale dword 装的是 2 个子块 × 2 个 K 半段。按 M 切，每段只用 4 个 dword，4 个字节在这一段里全用完，`asc0`、`asc1` 共 8 个 VGPR。按 K 切，每段都要全部 8 个 dword、每个只用一半，它们得跨两段活着，再加上提前读的下一个 stage 的 8 个，共 16 个 VGPR。注释给的理由是照搬 asm kernel，scale 的排法正好和它配套。

**段开头的常量（第 973–991 行）。**

| 常量 | 含义 | 段 0 | 段 1 |
|---|---|---|---|
| `SIG` | 第几段，从 `integral_constant` 取出的编译期值 | 0 | 1 |
| `MC`、`MB` | 本段算几个 M 片段、从第几个开始 | 8、0 | 8、8 |
| `MN`、`MBN` | 给下一段预读几个 M 片段、从第几个开始 | 8、8：本 stage 的 mi 8–15 | 8、0：下一个 stage 的 mi 0–7 |
| `TS` | 本段 MFMA 条数 = 2 × MC × NSUB | 64 | 64 |
| `NF` | 预读几个 A 片段 = 2 × MN | 16 | 16 |
| `NR` | LDS 读的总次数 = NF + MN/2 个 scale dword | 20 | 20 |
| `KLO`、`NL` | 本段发的全局访存从编号几开始、发几条 | 0、10：B 8 + B scale 2 | 10、10：A 8 + A scale 2 |
| `TLW` | 全局访存撒在前几条 MFMA 里 = TS × 7/8，至少 NL | 56 | 56 |

`MC` 和 `MN` 分开写，是因为 MSUB 不能平分时两段不一样大，比如 MSUB = 10 时段 0 算 6 个、预读 4 个。LDS 读都是从 `rd_lds` 指向的槽读进寄存器：段 0 读本 stage 的槽，填 `afr1`、`asc1`；段 1 读下一个 stage 的槽，填 `afr0`、`asc0`。`static_assert(NR <= TS)` 保证每条 MFMA 前最多插一次 LDS 读。

**64 条 MFMA 怎么展开（第 994–1030 行）。** `static_for<TS>` 在编译期展开成 64 段直线代码。每个 t 依次做三件事：可能插一次 LDS 读，可能插一条全局访存，再发第 t 条 MFMA。每件事后面都有 `sched_barrier(0)`，编译器不能挪动。

- **这个 t 插不插 LDS 读：** 第 r 次读放在第 ⌊r·TS/NR⌋ 条 MFMA 前。代码反过来算 `r = ⌈t·NR/TS⌉`，再检查 `r·TS/NR == t`。例：t = 3 时 r = 1，1·64/20 = 3，插；t = 4 时 r = 2，2·64/20 = 6，不插。r < 16 读 A 片段（h = r/8，mi = MBN + r%8），否则读第 MBN/2 + r − 16 块 scale。
- **插不插全局访存：** 算法一样，换成 NL = 10 条撒在前 TLW = 56 条里，落中了就调 `issue_one(KLO + k)`。
- **第 t 条 MFMA 算哪块：** t 拆成 `q = t%4`（2×2 里的位置）、`mp`（第几对 A 片段，每 4 条换一对）、`np`（第几对 B 片段，4 对 A 走完再换）、`h = t/32`（K 半段），由此得到 `ml`（段内的 M 片段号）、`mi = MB + ml`、`ni`。

MFMA 的参数：

| 参数 | 含义 |
|---|---|
| `acc[mi][ni]` | 累加到哪个 16×16 的 C 小块 |
| `bf[ni][h]` | B 片段：第 ni 个 16 N、第 h 个 K 半段 |
| `cf[h·MC + ml]` | A 片段：本段第 ml 个 M 片段、第 h 个 K 半段，下标排法和预读时的 `nf[r]` 一样 |
| `bs[ni/2]`、`cs[ml/2]` | B、A 的 scale dword，一个管两个片段 |
| 模板参数 `2h + (ni&1)`、`2h + (mi&1)` | 从 scale dword 里挑第几个字节 |

B 放在第一个操作数，算出来的是 C^T，lane L 拿到同一行的 4 个连续输出，方便 epilogue 写回。

**一条 MFMA 用哪些数据。** 设 tile 起点 m0，本 wave 的 N 起点 n0（已含 64·wave），stage s：A 是第 m0 + 16·mi 起的 16 行、K 从 256s + 128h 起的 128 个；B 是第 n0 + 16·ni 起的 16 个 N、同样的 K。lane L 只供 32 个 K：A 第 m0 + 16·mi + L%16 行、B 第 n0 + 16·ni + L%16 个 N，K 从 256s + 128h + 32·(L/16) 起。结果 `acc[mi][ni][i]` 是 C[m0 + 16·mi + L%16][n0 + 16·ni + 4·(L/16) + i]（第 1074 行）。例：tile 起点 (1280, 0)、wave 2、stage 3、h = 1、mi = 5、ni = 2、lane 37：A 第 1365 行、B 第 165 个 N，K 960–991，结果累加到 C[1365][168..171]。lane 读的 B 行和写出的列不同，是因为 MFMA 在 64 个 lane 之间交换数据。

段 0 前面几条按代码公式展开的结果：

```text
t= 0: LDS r=0:片段 h=0 mi=8->nf[0] | 全局 k=0 | MFMA acc[0][0] h=0 B=bf[0][0] A=cf[0] 字节B=0 A=0
t= 1: MFMA acc[1][0] h=0 B=bf[0][0] A=cf[1] 字节B=0 A=1
t= 2: MFMA acc[0][1] h=0 B=bf[1][0] A=cf[0] 字节B=1 A=0
t= 3: LDS r=1:片段 h=0 mi=9->nf[1] | MFMA acc[1][1] h=0 B=bf[1][0] A=cf[1] 字节B=1 A=1
t= 4: MFMA acc[2][0] ...                       ← 换下一对 A（mi 2、3）
t=16: LDS r=5 | 全局 k=3 | MFMA acc[0][2] ...  ← 4 对 A 走完，换下一对 B（ni 2、3）
t=32: LDS r=10:片段 h=1 mi=10 | MFMA acc[0][0] h=1 B=bf[0][1] A=cf[8] 字节B=2 A=2   ← 换 K 半段
t=63: MFMA acc[7][3] h=1 B=bf[3][1] A=cf[15] 字节B=3 A=3
```

2×2 之字形有三个好处：相邻 4 条共用同一个 `bs` 和同一个 `cs`，只换字节；每个片段连用两次；相邻的 MFMA 写不同的 `acc`，同一个 `acc[mi][ni]` 隔 32 条才再写，不用等上一条的结果。

**不是每条 MFMA 前都插东西。** 每段只要插 20 次 LDS 读和 10 条全局访存，64 条 MFMA 里：

| 前面插了什么 | 条数 | t |
|---|---|---|
| LDS 读和全局访存都有 | 5 | 0、16、22、28、44 |
| 只有 LDS 读 | 15 | 3、6、9、12、19、25、32、35、38、41、48、51、54、57、60 |
| 只有全局访存 | 5 | 5、11、33、39、50 |
| 什么都不插 | 39 | 其余 |

全局访存只在前 56 条里，最后 8 条只剩 t = 57、60 两次 LDS 读。为什么不集中发，见下一节。

# load 的三个时间：为什么既要提前发，又要穿插发

一条 load 有三个不同的时间，不能混在一起：

| | 发射 | 占用访存单元（或 LDS） | 延迟：数据回来 | 靠什么盖住 |
|---|---|---|---|---|
| B 的 `buffer_load` | 1 cycle | 十几到二十 cycle（实测每条拖慢约 17） | 几百到上千 cycle | 提前 1 个 stage，约 1300–2700 cycle |
| A 的 DMA | 1 cycle | 约二十 cycle（实测约 23） | 几百到上千 cycle | 提前 2 个 stage |
| `ds_read_b128` | 1 cycle | 几个 cycle（无 bank 冲突时 4 拍） | 一百左右 | 提前一整段，约 1000 cycle |

发射的 1 cycle 是实测（soffset 不为 0 时 2 cycle）；占用是从每段多出的 170、230 cycle 折算的；延迟仓库里没有单独测过，是估计。一条 MFMA 约 16.7 cycle。

**提前发，盖的是延迟。** 发出去的 load 在后台跑，wave 不等它，接着发 MFMA。这些 MFMA 用的是更早到的数据，和路上的 load 互不相干。从发出到被用隔着 64–128 条 MFMA，只要延迟比这段短，用的时候数据早就到了。实测每个 stage 约 2710 cycle 里，等待只有约 180 cycle。

**穿插发，盖的是占用。** 发出去不等于出列。wave 发一条 load 只要 1 cycle，但它进的是访存单元门口的队列，要等访存单元处理完才出列。处理就是把 64 个 lane 的地址拆成 cache line 请求、送给 L1，一条 1 KB 的 load 要约二十 cycle。进队 1 cycle 一条，出队约 20 cycle 一条，一个 CU 的 4 个 wave 还共用这个单元，连着发很快就满了。满了以后，下一条 load 发不出去，只能留在 wave 的指令流里原地等。wave 按顺序发射，没有"让后面的 MFMA 先走"的机制，于是 MFMA 全都停下，矩阵单元空转。两条 load 之间隔 5 条左右 MFMA（约 80 cycle），访存单元来得及处理完上一条，队列就不会满。LDS 读同理：两次之间隔约 3 条 MFMA，LDS 处理一次只要几个 cycle。

**两个上限。** 每个 wave 的 `vmcnt` 只有 6 位，在途的向量访存最多 63 条。这个 kernel 在途的一般只有二三十条，碰不到。真正会满的是访存单元的输入队列，AMD 没公开深度。注释里"约 20 条连着发会塞满"是实测出来的量级，4 个 wave 加起来就是 80 条一起涌进来。

**一句话：** 访存单元像一条传送带，约 20 cycle 能放上一个箱子，每个箱子要走几百到上千 cycle 才到头。提前发，保证箱子在被用之前已经走到头；穿插发，保证放箱子的速度不超过传送带接收的速度。剩下的代价是每条 load 仍会拖慢 MFMA 约 20 cycle：每个 stage 比纯 MFMA 的 2290 cycle 多出约 420，主要就是这 20 条的占用。

# epilogue：把一块 tile 写回 C

一块 tile 的 K 循环走完后调用一次 `epilogue(cur)`，把 256×256 个 fp32 累加器转成 bf16 写回 C。每个 wave 只写自己的 256 行 × 64 列，不经过 LDS、不要 barrier，发 32 条 store，每条写 8 行、每行完整的 128 B。

**先分清四个层次。** epilogue 在一块 tile 的 K 循环全部走完之后才执行，这时累加器里已经是最终结果。“256×256”和“16×16”说的是不同的层次：

| 层次 | 范围 | 结果个数 | 放在哪 |
|---|---|---|---|
| tile（整个工作组） | C 的 256 行 × 256 列 | 65536 个 | 4 个 wave 合起来 |
| wave | 256 行 × 本 wave 的 64 列 | 16384 个，即 16 × 4 个子块 | `acc[16][4]`，每个 lane 256 个 AGPR |
| 子块 `acc[mi][ni]` | 16 行 × 16 列 | 256 个 | 64 个 lane 分着拿 |
| lane | 子块里第 L%16 行、第 4·(L/16) 到 +3 列 | 每个子块 4 个 | 一个 `f32x4` |

16×16 是一条 MFMA 输出的形状，但 `acc[mi][ni]` 不是某一条 MFMA 的结果，而是同一个 (mi, ni) 在整个 K 上所有 MFMA 累加进同一组寄存器的和：每个 stage 2 条（h = 0、1），K = 5120 时共 40 条。这些 MFMA 把结果分给 lane 的方式都一样，所以 tile 算完时，每个子块依然是“lane L 拿第 L%16 行、第 4·(L/16) 到 +3 列”。

**在流水里的位置。** MFMA 停了，load 没停。一块 tile 的最后两个 step 已经在发下一块 tile 的 A(0)、A(1)、B(0)，最后一段还把它 stage 0 的 A 片段读进了 `afr0`；epilogue 写回的同时这些 load 在路上。下一块 tile 的第一个 step 两处等待各放宽 32 条（10 + 32 = 42，`after_epi`），不等 store 写完就开算。多轮的形状上 epilogue 约占 kernel 时间的 15%，主要是发 store：每 KB 每个 CU 约 68 cycle，4 个 wave × 32 条约 8700 cycle，一块 tile 的 K 循环约 20 × 2710 ≈ 54000 cycle。

**wave 怎么分。** 4 个 wave 各写各的。例：tile 起点 (1280, 0) 时 wave w 写第 1280–1535 行、第 64w–64w+63 列。wave 内按 mi = 0..15 一个 16 行的行块一个行块地写：每个行块是 4 个累加器 `acc[mi][0..3]`，16 行 × 64 列，2 条 store（前 8 行、后 8 行），共 16 × 2 = 32 条，就是 `EPI_STORES`。n0 是 64 的倍数，每个 wave 的一行（64 列 × 2 B）正好是一条对齐的 cache line。

外层按 mi 走（第 1138 行的 `for`）；每个行块里 `pair(mi, 0)` 管 ni = 0、1，`pair(mi, 1)` 管 ni = 2、3，两个 pair 再用 DPP 拼成整行：

```text
一个 wave 的 256 行 × 64 列（列号相对本 wave 的 n0）

                 ni=0       ni=1        ni=2       ni=3
                列 0–15    列 16–31    列 32–47   列 48–63
mi=0   行 0–15     [   pair(0, 0)   ]  [   pair(0, 1)   ]   → DPP 重组 → 2 条 store
mi=1   行 16–31    [   pair(1, 0)   ]  [   pair(1, 1)   ]   → 2 条 store
 …
mi=15  行 240–255  [   pair(15, 0)  ]  [   pair(15, 1)  ]   → 2 条 store
```

**每个 lane 的 AGPR 里放的是哪些元素。** 一个 wave 的 256×64 是 16（mi）× 4（ni）个 16×16 小块，每个小块是一个累加器 `acc[mi][ni]`。lane L（row16 = L%16，G = L/16）在每个小块里都拿同一位置的 1 行 × 4 列：第 16·mi + row16 行、第 16·ni + 4G 到 +3 列。64 个小块 × 4 = 256 个 fp32，正好 256 个 AGPR；64 个 lane × 256 = 16384 = 256 × 64，每个元素只在一个 lane 里。

![每个 lane 的 AGPR 里放的是哪些元素](./images/epilogue_agpr.svg)

- **具体是哪几个 AGPR：** 从 256×256 实例的汇编看，K 循环里每个累加器固定用 4 个连续的 AGPR，`acc[mi][ni]` 从 a[(128 + 16·mi + 4·ni) mod 256] 开始，第 i 个放那 4 列里的第 i 列；mi = 0..7 在 a128–a255，mi = 8..15 在 a0–a127。
- **例：lane 37**（row16 = 5，G = 2）拿第 5、21、…、245 行，每行第 8–11、24–27、40–43、56–59 列。a128–a131 是 C[5][8..11]，a212–a215 是 C[85][24..27]（相对本 wave 的 m0、n0）。上面示意图里每个 `pair(mi, np)` 格子（16 行 × 32 列）中属于它的是 1 行 × 8 列，即 8 个 AGPR，`pair` 一次正好读出这 8 个。
- **换个方向看：** 固定 AGPR 号看 64 个 lane，a128–a131 合起来就是完整的 `acc[0][0]`，其中 a128 装的是各行的第 0、4、8、12 列。所以一个累加器是 4 个 AGPR × 64 个 lane。

**先看一个子块、一行：`pair` 在做什么。**

![epilogue 的第一步：一个 16×16 子块怎么分给 lane，pair 怎么把一行凑成 16 B](./images/epilogue_pair_row.svg)

- **左图：一个子块怎么分给 lane。** 每一行被 4 个 lane 分掉，lane r、r+16、r+32、r+48 各拿本行连续 4 列，例如第 5 行是 lane 5、21、37、53。
- **为什么要换。** C 在显存里按行存，元素是 bf16。一个 lane 手里的 4 个结果转成 bf16 只有 8 B，而一条 store 每个 lane 最多写 16 B（`buffer_store_dwordx4`）；想一次写满 16 B，lane 手里得有同一行的连续 8 列。一个子块的一行只有 16 列，又分给了 4 个 lane，凑不出来。
- **右图：`pair` 拿两个子块凑 32 列。** `pair(mi, np)` 一次处理相邻两个子块（ni = 2np、2np+1），一行变成 32 列，正好够 4 个 lane 每人 8 列。交换前，lane 5 拿第 0–3 列和第 16–19 列；`permlane16_swap` 让 lane 5 和 lane 21 互换一半：lane 5 把第 16–19 列交给 lane 21，换回第 4–7 列。lane 37 和 lane 53 同样互换。交换只在 lane L 和 L±16 之间，行号不变。
- **返回值。** 换完每个 lane 拿本行连续 8 列 = 8 个 bf16 = 16 B，`pair` 把它作为一个 `u32x4` 返回。`u32x4` 是 `ext_vector_type(4)` 的 4 个 32 位无符号数，每个 lane 占 4 个连续 VGPR，正好是一条 16 字节 store 的数据；每个 dword 是 `pack_bf16x2` 打包的 2 个 bf16。这些值由 `permlane16_swap` 这类 VALU 指令算出，VALU 的结果都写进 VGPR；AGPR 这时已经清零，不再参与。

**`pair` 在寄存器里怎么做。** 以 `pair(mi, 0)`、第 5 行的 lane 5 和 lane 21 为例（已逐 lane 模拟核对）。lane 5 在 `acc[mi][0..3]` 里拿第 0–3、16–19、32–35、48–51 列，lane 21 拿第 4–7、20–23、36–39、52–55 列：一行 64 列里各有 4 段，每段 4 列、隔 16 列，而 store 要的是连续 8 列。

1. **读累加器（每个 lane 只读自己的）：** x = `acc[mi][0]`、y = `acc[mi][1]` 从 AGPR 拷进 VGPR，读完清零。lane 5 的 x 是第 0–3 列、y 是第 16–19 列；lane 21 的 x 是第 4–7 列、y 是第 20–23 列。
2. **转 bf16、两两打包（也是每个 lane 自己做）：**

| | a0 = pack(x[0], x[1]) | a1 = pack(x[2], x[3]) | b0 = pack(y[0], y[1]) | b1 = pack(y[2], y[3]) |
|---|---|---|---|---|
| lane 5 | 列 0、1 | 列 2、3 | 列 16、17 | 列 18、19 |
| lane 21 | 列 4、5 | 列 6、7 | 列 20、21 | 列 22、23 |

3. **交换（两条 `permlane16_swap`）：** lane 5 把 y 部分（b0、b1，第 16–19 列）给 lane 21，换回 lane 21 的 x 部分（a0、a1，第 4–7 列）。`permlane16_swap(a0, b0)`、`permlane16_swap(a1, b1)` 各换一对 dword。
4. **拼成 16 B（第 1121 行，`{s0[0], s1[0], s0[1], s1[1]}`）：**

| | s0[0] | s1[0] | s0[1] | s1[1] | 合起来 |
|---|---|---|---|---|---|
| lane 5 | 列 0、1（自己的） | 列 2、3（自己的） | 列 4、5（lane 21 给的） | 列 6、7（lane 21 给的） | 列 0–7 |
| lane 21 | 列 16、17（lane 5 给的） | 列 18、19（lane 5 给的） | 列 20、21（自己的） | 列 22、23（自己的） | 列 16–23 |

lane 37 和 lane 53 同样互换，拿到第 8–15、24–31 列；其他 15 行也一样，各行的 4 个 lane 是 r、r+16、r+32、r+48。`pair(mi, 1)` 用同样的办法处理 ni = 2、3，lane 5 拿到第 32–39 列。mi = 1 时也是同样的互换，做法和 mi = 0 一模一样，其他 mi 也是，mi 只决定处理的是 tile 里哪 16 行；一个 wave 每块 tile 共 32 次 `pair`、64 条 `permlane16_swap`。

**交换规则。** `permlane16_swap` 把 64 个 lane 分成 4 组，每组 16 个；第 0、1 组之间和第 2、3 组之间，同一位置的 lane 两两交换（L ↔ L+16）。第一个寄存器里偶数组留着自己的值、奇数组换成偶数组的值，第二个寄存器反过来。一条指令每对 lane 换 1 个 dword，两条正好换 8 B。硬件只交换“第一个操作数的奇数组”和“第二个操作数的偶数组”，代码把 `pack(x…)` 放第一个参数、`pack(y…)` 放第二个，所以换的正好是 lane 21 的 x 和 lane 5 的 y，lane 5 的 x 和 lane 21 的 y 不动；两个参数反过来就会换错。

**交换只改 VGPR 里的副本，不改 `acc`。** `acc[mi][ni]` 拷进 x、y 后就清零了；交换在 VGPR 里打包好的 bf16 上做，每个 lane 的 4 个 dword 里变 2 个（偶数组变 b0、b1，奇数组变 a0、a1）。数值本身不变，只是换了 lane，比如 C[5][4] 从 lane 21 搬到了 lane 5。结果放在 `pair` 返回的 `u32x4` 里，不写回 `acc`。

lane 5 一路拿的数据（第 mi 个行块的第 5 行，列号相对本 wave 的 n0）：

| 阶段 | lane 5 手里的数据 |
|---|---|
| MFMA 算完（AGPR，`acc[mi][0..3]`） | 第 0–3、16–19、32–35、48–51 列 |
| `pair(mi, 0)` 之后（VGPR） | 第 0–7 列：自己的 0–3 + lane 21 的 4–7 |
| `pair(mi, 1)` 之后（VGPR） | 第 32–39 列：自己的 32–35 + lane 21 的 36–39 |
| DPP 之后（`va`、`vb`） | `va`：第 5 行第 0–7 列；`vb`：第 13 行第 0–7 列 |

每个 lane 换完只拿 8 列，第 8–15 列不在 lane 5 手里：一个 `pair` 做完，第 5 行的第 0–31 列由 lane 5、37、21、53 分别拿 0–7、8–15、16–23、24–31。DPP 那一步 lane 5 再把第 32–39 列交给 lane 13，换回第 13 行的第 0–7 列，最后写出第 5 行和第 13 行的第 0–7 列，各一条 store。

下面的三步图把一个行块的两个 pair（64 列）整个画出来，第 ③ 步再把它们拼成整行。

**lane 怎么分：一个行块的三步交换。**

![epilogue 里一个行块的 lane 交换](./images/epilogue_lanes.svg)

lane L 记作 (row16, G)，row16 = L%16，G = L/16。

1. **MFMA 之后：** 因为交换了操作数（算 C^T），`acc[mi][ni][i]` 在 lane L 上是第 16mi + row16 行、第 16ni + 4G + i 列。lane 管第 row16 行，在每个累加器里有本行连续 4 列，4 个累加器合起来 16 列，分成 4 段。不交换的话一个 lane 拿的是同一列的 4 行，没法连续写。
2. **`pair(mi, np)`：** 两个相邻累加器（ni = 2np、2np+1，共 32 列）从 AGPR 拷进 VGPR，马上清零留给下一块 tile；`v_cvt_pk_bf16_f32` 两个 fp32 就近舍入成一个 dword 里的两个 bf16；`permlane16_swap` 把第一个寄存器的奇数 lane 组和第二个的偶数 lane 组互换。之后 lane (r, G) 拿第 r 行连续 8 列（16 B），起点 (G&1)·16 + (G>>1)·8，即 G = 0、1、2、3 时为 0、16、8、24。一行的 32 列 = 4 个 lane × 16 B = 半条 cache line。
3. **DPP `row_ror:8` + 选择：** 16 个 lane 里循环右移 8 位，lane (r, G) 和 lane (r+8, G) 互相拿到对方的值。`va = lo8 ? v0 : r1` 拼出第 0–7 行：row16 < 8 的 lane 出左 32 列（自己的 v0），row16 ≥ 8 的 lane 出右 32 列（对面的 v1）；`vb = lo8 ? r0 : v1` 拼出第 8–15 行。每行 = 8 个 lane × 16 B = 128 B。

例：lane 37（row16 = 5，G = 2）。① 拿第 5 行第 8–11、24–27、40–43、56–59 列；② v0 是第 5 行第 8–15 列，v1 是第 40–47 列；③ va 是第 5 行第 8–15 列，vb 是第 13 行第 8–15 列（lane 45 的 v0），而它的 v1 交给 lane 45 去写。tile 起点 (1280, 0)、wave 2、mi = 5 时，va 写 C[1365][136..143]，vb 写 C[1373][136..143]。

**为什么还要做第 ③ 步：直接存和交换后再存。**

![epilogue 第 ③ 步为什么要交换：直接存 vs 交换后再存](./images/epilogue_dpp_stores.svg)

`pair` 之后每个 lane 已经有两份连续的 16 B：`v0` 在本行左半边，`v1` 在同一行右半边。直接存也是对的，NSUB = 6 时最后一对子块就这么存（`vofs_h` 那条路），区别只在每条 store 写出去的形状：

| 写法 | 每个行块几条 store | 每条覆盖 | 每行写多少 | 一条碰几条 cache line |
|---|---|---|---|---|
| 直接存 `v0`、`v1`（左图） | 2 | 16 行 × 32 列 | 64 B，半条 | 16 条，每条写一半 |
| 交换后存 `va`、`vb`（右图） | 2 | 8 行 × 64 列 | 128 B，整条 | 8 条，每条写满 |

一条 store 要写满第 5 行，这一行的 8 段必须在同一个寄存器、8 个不同的 lane 上。直接存时，第 5 行的右半在 lane 5、21、37、53 的 `v1` 里，和左半不在同一个寄存器。第 13 行的 4 个 lane（13、29、45、61）编号正好差 8、列也一样，就让它们替第 5 行写右半，同时把自己的左半交给第 5 行的 lane：粗框里 lane 5 把第 5 行右段交给 lane 13，换回第 13 行左段。lane L 和 L^8 在同一个 16-lane 行里，DPP `row_ror:8` 一条 `mov` 就能完成交换。整条 cache line 的 store 确认得快约 2.5 倍（见下面的 store 一段），代价只是每个行块 8 条 DPP 和 8 条选择指令。

**两条 store 都发了，硬件不会自己凑成整行吗？** 不会。总字节数两种写法一样，但访存单元只在一条指令内部把 64 个 lane 合成 cache line 请求，交出去就走，不会等下一条指令来拼；两条 store 之间还隔着别的指令，到 L2 的时间也不同。所以直接存时，同一条 line 会收到两个来自不同指令的半条写请求：每个行块 32 个 64 B 的请求，交换后是 16 个 128 B 的请求。两半也许最后在 L2 里合到同一条 line，但请求数翻倍、每次都是部分写，实测全 GPU 同时写时确认要约 7.5k cycle，整条约 3.0k；`vmcnt` 按顺序完成，确认慢会拖住下一块 tile step 0 里排在后面的 load。交换每个行块只多 16 条 VALU，十几到几十 cycle，很划算。

**cache line 说的是写到哪，不是数据在哪个寄存器。** 寄存器没有地址。一条 `buffer_store_dwordx4` 里每个 lane 给出 16 B 数据（4 个连续 VGPR）和一个目的地址；访存单元按目的地址把 64 个 lane 归到显存里 128 B 对齐的 cache line 上，每条 line 一个带字节掩码的写请求送往 L2，再由 L2 写回 HBM。所以两次交换各管一层：`pair` 让每个 lane 的 16 B 在显存里连续，这是正确性要求，`dwordx4` 只能往一个地址连续写 16 B；DPP 让一条指令的 64 × 16 B 拼成整条 cache line，这是性能要求。直接存 `v0` 落在 16 条 line 上、每条只写一半；交换后存 `va` 落在 8 条 line 上、每条写满，请求少一半，也不用在 L2 里把半条 line 和其余部分合并，实测确认时间快约 2.5 倍。

**store。** 地址全在 voffset 里：`vofs_f = ((m0 + row16%8)·N + ccol)·2`，`ccol = n0 + (row16/8)·32 + (G&1)·16 + (G>>1)·8`，vb 再加 8 行，每个行块后加 16 行。一条 `buffer_store_dwordx4` = 64 个 lane × 16 B = 8 行 × 128 B，写满 8 条 cache line。要写整行，是因为实测全 GPU 同时写时半条 cache line 的 store 约 7.5k cycle 才被确认，整条约 3.0k；`vmcnt` 按顺序完成，下一块 tile step 0 里的 load 排在这些 store 后面。store 直接从 VGPR 取数据，例如 256×256 实例里的 `buffer_store_dwordx4 v[180:183], v188, s[12:15], 0 offen`：`v[180:183]` 是拼好的 `va`（4 个连续 VGPR，16 B），`v188` 是本 lane 的偏移 `vofs_f`，`s[12:15]` 是 C 的描述符 `o_rsrc`。

**发几条 store，每个 lane 写多少。** 指令按 wave 发：SQ 发一次，64 个 lane 一起执行，每个 lane 贡献 16 B，没有"每个 lane 单独发几次"。

| 层次 | store 条数 | 每条写多少 | 合计 |
|---|---|---|---|
| 每个 lane | 32（跟着 wave 执行） | 16 B = 8 个 bf16 = 4 个 VGPR | 512 B = 256 个 bf16 |
| 每个 wave | 32 = 16 个行块 × 2 | 1 KB = 8 行 × 128 B | 32 KB = 256 × 64 × 2 B |
| 每个工作组 | 128 | 1 KB | 128 KB = 256 × 256 × 2 B |

每个行块 2 条：`va` 写第 0–7 行，`vb` 写第 8–15 行。例：lane 37 在 mi = 0 写第 5 行和第 13 行的第 8–15 列，mi = 1 写第 21、29 行，依此类推。一个 lane 写出的 256 个 bf16 是 256 个 AGPR 转换后的数量，但已经在 lane 之间换过位置，不是它原来那 256 个元素。写之前在寄存器里还有读累加器、转 bf16、`permlane16_swap`、DPP、`cndmask` 这些准备指令（条数见下面的数据流向表），真正写内存的只有这 32 条。

**每条 store 里每个 lane 写哪 1 行 × 8 列。** lane L（row16 = L%16，G = L/16）在第 mi 个行块：`va` 写第 16·mi + row16%8 行，`vb` 写第 16·mi + 8 + row16%8 行；两条都是第 32·(row16/8) + 16·(G&1) + 8·(G>>1) 起的 8 列（相对本 wave 的 m0、n0）。mi = 0 的 `va` 里每格由哪个 lane 写：

| 行 \ 列 | 0–7 | 8–15 | 16–23 | 24–31 | 32–39 | 40–47 | 48–55 | 56–63 |
|---|---|---|---|---|---|---|---|---|
| 0 | 0 | 32 | 16 | 48 | 8 | 40 | 24 | 56 |
| 1 | 1 | 33 | 17 | 49 | 9 | 41 | 25 | 57 |
| 2 | 2 | 34 | 18 | 50 | 10 | 42 | 26 | 58 |
| 3 | 3 | 35 | 19 | 51 | 11 | 43 | 27 | 59 |
| 4 | 4 | 36 | 20 | 52 | 12 | 44 | 28 | 60 |
| 5 | 5 | 37 | 21 | 53 | 13 | 45 | 29 | 61 |
| 6 | 6 | 38 | 22 | 54 | 14 | 46 | 30 | 62 |
| 7 | 7 | 39 | 23 | 55 | 15 | 47 | 31 | 63 |

每行由 8 个 lane 拼成：row16 < 8 的 4 个写左 32 列，row16 ≥ 8 的 4 个写右 32 列，共 128 B，正好一条 cache line。`vb` 的表完全一样，只是行号是第 8–15 行；换行块时行号整体加 16，列不变。所以一个 lane 在 32 条 store 里始终写固定的 8 列、32 个不同的行，共 256 个 bf16。例：lane 37 写第 8–15 列，行依次是 5、13、21、29、…、245、253；lane 45 的行和它一样，列是第 40–47 列。这张表和 `epilogue_lanes.svg` 的 ③ 一致，生成脚本里逐格断言过这个公式。

**注意事项：**

- **不用 LDS 转置：** LDS 的 4 个槽里放着下一块 tile 的 A，用 LDS 还得加 barrier；`permlane16_swap`、DPP 是寄存器里的 lane 交换，只要几条 VALU。
- **一个行块一个行块地读累加器：** 每个 mi 前有 `sched_barrier`，AGPR→VGPR 的拷贝用空 asm 钉住。否则编译器把 256 个累加器的读取都提到最前面，VGPR 放不下就 spill；现在同时只活 4 个累加器（16 个 VGPR）加临时值。
- **清零后加 `"+a"` 的空 asm：** 直接写常量 0 会被合并进 tile 循环里累加器的 phi 节点，寄存器分配器就会搬动累加器并 spill。
- **lane 号用 `v_mbcnt` 现算、偏移只放 voffset：** 不让由 lane 推出的地址被提到 tile 循环外、跨整个 K 循环占着 VGPR / SGPR。`v_mbcnt_lo_u32_b32 %0, -1, 0` 和 `v_mbcnt_hi_u32_b32 %0, -1, %0` 数的是编号比自己小、掩码为 1 的 lane 有几个，低 32 位、高 32 位各数一次；掩码给 −1，数出来就是 lane 号，例如 lane 37 是 32 + 5。写成 `asm volatile`，编译器看不出它就是开头的 `threadIdx.x & 63`，不会合并，也不会挪出循环，每块 tile 只多 2 条指令。
- **store 无条件发：** vmcnt 记账假定每个 epilogue 正好 32 条。N 方向越界给偏移 `0x7fffffff`，M 方向越界超出 `o_rsrc` 的范围（M·N·2 字节），都被硬件丢弃。
- **整行要求对齐：** 每行起点 128 B 对齐，即 N 是 64 的倍数（8192、17408 都是）。NSUB = 6 时子块对是奇数，最后一对只能按半行写（每条 store 16 行 × 64 B）。

**数据流向：从 AGPR 到 HBM。** K 循环的 20 个 stage 里累加器一直在 AGPR，每条 MFMA 读旧值、加上乘积、写回同一组 AGPR；到 epilogue 才往外走。条数是每个 wave 每块 tile，来自 256×256 实例的汇编：

| 步 | 数据流向 | 指令（条数） | 谁来做 |
|---|---|---|---|
| ① | AGPR → VGPR，随后 AGPR 清零 | `v_accvgpr_read_b32`（256）、`v_accvgpr_write_b32`（约 256） | VALU；AGPR、VGPR 在同一个 SIMD 的寄存器堆里，不出 CU |
| ② | VGPR 里 fp32 → bf16 | `v_cvt_pk_bf16_f32`（128） | VALU |
| ③ | VGPR 里 lane 交换，拼成整行 | `v_permlane16_swap_b32`（64）、`v_mov_b32_dpp row_ror:8`（128）、`v_cndmask_b32`（160：128 条拼 va/vb，32 条选越界偏移） | VALU 的跨 lane 通路 |
| ④ | VGPR → 访存单元 → L1 → L2 | `buffer_store_dwordx4`（32） | SQ 发射；访存单元用描述符基址 + voffset 算地址、做越界检查、合成 cache line 请求；L1 直写，L2 收下后确认，`vmcnt` 减 1 |
| ⑤ | L2 → HBM | — | L2 写回；kernel 结束时命令处理器做 release，下一个 kernel 和主机才看得到 |

转换和 lane 交换都是 VALU 指令，只能在 VGPR 上做，所以累加器必须先搬出 AGPR。①–③ 都在寄存器里，每个 wave 七八百条 VALU；大头是 ④，每条 1 KB 的 store 每个 CU 约 68 cycle。kernel 只管到 ④ 的确认（`vmcnt`），数据什么时候落到 HBM 由 L2 和运行时决定。

**表里的几个名词。** 普通 VALU 指令里 64 个 lane 各算各的，只碰自己那份寄存器；DPP 和 permlane 让 lane 拿到别的 lane 的数据，cndmask 让每个 lane 各自二选一，SQ 负责把这些指令发出去。

- **SQ（Sequencer）：** 每个 CU 一个的指令调度器。取指、解码，每个 cycle 挑能发的指令送到 VALU / 矩阵核心、标量单元、向量访存单元、LDS 或分支单元；同一个 wave 按顺序发，"一条 load 发不出去，后面的 MFMA 都被堵住"就发生在这里。`s_waitcnt`、`s_barrier`、`s_nop` 也由它执行；`SQ_INST_CYCLES_VMEM_RD` 是它的性能计数器。
- **DPP（Data Parallel Primitives）：** 挂在 VALU 指令上的修饰，让第一个源操作数按固定模式从别的 lane 读，代价和普通 VALU 一样。模式基本都在 16 个 lane 的一行之内，如 `quad_perm`、`row_shl`/`row_shr`、`row_ror`。epilogue 里的 `v_mov_b32_dpp v181, v192 row_ror:8 row_mask:0xf bank_mask:0xf` 让每个 lane 拿同一行 lane l^8 的值；源码 `__builtin_amdgcn_mov_dpp(v, 0x128, 0xF, 0xF, false)` 里的 0x128 就是 `row_ror:8` 的编码。它对前后指令的读写间隔有要求，编译器会补 `s_nop`。
- **`v_permlane16_swap_b32 vA, vB`：** gfx950 新加的指令。64 个 lane 看成 4 行、每行 16 个，把 vA 的第 1、3 行和 vB 的第 0、2 行整行互换，一条指令改两个寄存器，补上 DPP 跨不出一行的缺口；同系列的 `permlane32_swap` 交换前后 32 个 lane。builtin 返回交换后的两个值。
- **`v_cndmask_b32 vdst, src0, src1, vcc`：** 按 lane 二选一，VCC 是 64 位掩码，每个 lane 一位，为 1 取 src1、为 0 取 src0。汇编里先 `v_cmp_gt_u32_e32 vcc, 8, v218`（v218 = row16）设好 lo8，`va[i] = lo8 ? v0[i] : r1` 就编成一条 cndmask，没有分支。

lane 之间交换数据的几种办法：DPP 只能在一行内、代价同普通 VALU；`permlane*_swap` 是行与行整行交换、一条 VALU；`ds_bpermute_b32` 能任意重排，但借 LDS 的硬件、算 `lgkmcnt`，更慢；写进 LDS 再读出来最灵活，要两次 LDS 访问，还可能要 barrier。epilogue 的两次交换一次在相邻两行之间、一次在行内 r 和 r+8 之间，正好用最便宜的 permlane 和 DPP。

这一节的图都由 `scripts/draw_epilogue_figs.py` 生成，每格的 lane 号都按指令语义逐 lane 模拟得到。

# 每条 load 的 lane 偏移、SCALE_SOFF 和 opaque

## lane 偏移：每条 load 一个 VGPR，只算一次
opaque(v) 原样返回 v，自己不生成任何指令。它的作用是让编译器“看不透”这个值：编译器只知道它在某个 VGPR 里，却不知道它等于什么。这样编译器只能在循环外把它算一次，然后一直留在寄存器里，K 循环里的每条 load 直接拿来用

一条 buffer load 的地址由 4 项相加，硬件在访存单元里一次加好，不用额外的加法指令。4 项按“谁的值不一样、什么时候变”分开放：

| 项 | 放在哪 | 谁的值不一样 | 什么时候变 | 这个 kernel 里装的是 |
|---|---|---|---|---|
| 基址 | SGPR（描述符里） | 整个 wave 相同 | 换 tile、换 stage 时由 `rsrc_at` 挪动 | A / B 的第一个字节 + tile 和 stage 的偏移 |
| voffset | VGPR | 每个 lane 都不一样 | 不变，开头算一次 | 本 lane 那 16 B 的位置（`a_voff`、`b_voff`） |
| soffset | SGPR | 整个 wave 相同 | — | 一般是 0；256 行 tile 的 scale load 用它放块偏移 |
| 立即数 | 编码在指令里 | 编译时定死 | 不变 | 如 B load 的 `h·1024` |

例：B 的 load 是 `buffer_load_dwordx4 %0, %1, %2, 0 offen offset:%3`。工作组 72 的 wave 1、stage 3、ni = 2、h = 1、lane 37：基址 = B + t.b + 3·2048 = B + 1480704，voffset = `b_voff[2]` = 82512，soffset = 0，立即数 = 1024，合计 B + 1564240，正是 B 第 613 行、K 960–991 那 16 B。

tile、stage 的偏移加在基址上，不放进 soffset，有两个原因（第 705–712 行的注释）：硬件的越界检查只看“voffset + 立即数”有没有超出描述符的范围，所以要把偏移加到基址上、同时把范围缩小同样的量；另外 soffset 不为 0 时，发一条访存指令要 2 个 cycle。

voffset 在 kernel 开头算好，之后不再变：

| 数组 | 个数 | 每个是什么 | lane 37 |
|---|---|---|---|
| `a_voff[j]` | `A_LDS_PER_WAVE` = 8 | 第 j 条 A DMA：`a_lane + j·8·Kp`，往下第 j 个 8 行 | 10256、30736、…、153616 |
| `b_voff[ni]` | `NSUB` = 4 | 第 ni 个 B 子块：`b_lane + ni·8K`，第 ni 个 16 行组 | 592、41552、82512、123472 |
| `bs_voff` | 1 或 `NSUB/2` | B scale 的第 p 个 32 行块 | 见下节 |
| `as_voff` | 1 或 `S_LDS_PER_WAVE` | 第 j 条 A scale DMA 搬的那一块 | 见下节 |

## 立即数：写在指令里的常数偏移

立即数是直接编码在指令里的常数，汇编里写作 `offset:N`，不占寄存器。它在指令里只有 12 位，能放 0–4095 字节。用 `llvm-mc` 按 gfx950 汇编可以看到：`offset:1024` 编码的低 12 位是 0x400，`offset:4095` 是 0xfff；`offset:4096` 汇编时不报错，但编码和 `offset:0` 完全一样，溢出成了 0。

它的用处是把编译时就知道的小常数偏移免费加上：不占寄存器，也不多一条加法。B 的每个子块每个 stage 有两条 load（h = 0、1），指令在编译时全部展开，h 是常数，所以两条共用同一个 `b_voff[ni]` 和同一个描述符，只在立即数上差 1024。没有它，要么每个子块多备一个 VGPR（4 个子块就多 4 个），要么每条 load 前多一条 `v_add`。

能放进立即数的偏移必须编译时确定，而且小于 4096：

| 偏移 | 编译时确定 | 大小 | 放在哪 |
|---|---|---|---|
| `h·1024` | 是 | 0 或 1024 | 立即数 |
| `ni·8K` | 是 | 40960 起，超过 4095 | VGPR（`b_voff[ni]`） |
| `stb·2048` | 否，每个 stage 都变 | 最大 2048 × 19 | 描述符基址 |

硬件的越界检查覆盖的是“voffset + 立即数”，所以用立即数不影响越界检查。

## SCALE_SOFF：用发射速度换 VGPR

`SCALE_SOFF = MSUB >= 16`，只有 256 行的 tile 打开。每个 stage 有 4 条 scale load（2 条 B scale、2 条 A scale）：

- **关着（其他 tile）：** 每条 scale load 一个 VGPR，块偏移提前算进去，如 `bs_voff[p] = s_lane + p·32·scaleN_pad`；指令里 soffset 为 0。
- **打开（256 行 tile）：** 4 条共用一个 VGPR，`bs_voff[0] = as_voff[0] = s_lane = 4·lane`（lane 37 是 148）；块偏移 `p·32·scaleN_pad` 改放进指令的 SGPR soffset（p = 1 时是 5120）。

代价是 soffset 不为 0 时，发一条访存指令要 2 个 cycle，平时是 1 个（第 710 行的实测）。好处是按注释能省下 4 个 VGPR。256 行 tile 的累加器最多，不省这几个会在 epilogue 附近 spill；其他 tile 寄存器够用，就用更快的写法。

## opaque：不让编译器改写这些偏移

```cpp
auto opaque = [](int v) { asm volatile("" : "+v"(v)); return v; };
```

不包的话，编译器知道 `a_voff[j] = a_lane + j * a_jstride` 是怎么来的，寄存器紧张时可能不一直占着它，而是每条 load 前用 `a_lane` 重算一遍（多一条 `v_add`），或者把加法挪到别处合并。指令按顺序发射，load 前多出来的计算会卡住后面的指令；注释里实测，这类计算加上非 0 的 soffset，每个 stage 约慢 6%。

这段空 asm 的三部分各管一件事：

- `""`：内容为空，不生成任何指令，没有开销；
- `"+v"(v)`：asm 要读 v、也可能改写 v，而且 v 必须在 VGPR 里；
- `volatile`：不能删掉，也不能挪位置。

所以在这一点之前，编译器必须把 v 算好放进一个 VGPR；在这之后，它只能假设 v 可能被改过，不再知道 v 等于 `a_lane + j * a_jstride`，也就没法重算或合并，只能一直留着这个 VGPR，在 load 里直接当 voffset 用。

一句话：`opaque` 没有新建寄存器，而是把一个算好的值钉在 VGPR 里，不让编译器优化掉；代价只是这个值要一直占着一个 VGPR，这也是 256 行 tile 要靠 `SCALE_SOFF` 再省几个 VGPR 的原因。


int 是 4 字节，也就是 32 位。VGPR 正好是每个 lane 32 位，所以一个 int 占 1 个 VGPR。

要注意，VGPR 是“每个 lane 各一份”：一个 VGPR 对整个 wave 来说是 64 个 lane × 4 B = 256 B 的寄存器空间。这些偏移必须放在 VGPR 里，因为每个 lane 的值都不一样，比如 lane 0 的 a_voff[0] 是 0，lane 37 的是 10256。而 SGPR 整个 wave 只有一份 32 位，只能放所有 lane 都相同的值。块偏移 p·32·scaleN_pad 就是所有 lane 都相同的，所以 SCALE_SOFF 可以把它挪到 SGPR 里。

## spill：寄存器放不下，值被挪去显存

spill（寄存器溢出）：某一处同时“活着”的值比每个 lane 的寄存器还多，编译器只好把一部分临时写进 scratch（每个 lane 私有的一块显存），要用时再读回来。结果不变，只是变慢。

一个值从算出来到最后一次被用，这段时间叫“活着”，期间它必须占一个寄存器。会不会 spill 看的是同一时刻有多少值活着（寄存器压力），不是变量总数；被挑出去的通常是很早算出来、很晚才用的值。

这个 kernel 贴着上限。每个 SIMD 只跑 1 个 wave 时，每个 lane 最多 256 个 VGPR 加 256 个 AGPR。256 行 tile 的累加器是 16×4 = 64 个 f32x4，正好 256 个 AGPR；VGPR 里要放 A、B 片段的双缓冲、scale 和上面这些偏移。压力最高的是 epilogue：累加器要从 AGPR 搬进 VGPR 做转换和打包，下一个 tile 的 B 预取已经在往寄存器里写，所有偏移也还活着。scale 的偏移再各占一个 VGPR 就会超过 256，所以 SCALE_SOFF 要省下那 4 个。

**为什么贵：** scratch 在显存里，命中 cache 也要几百个 cycle。更要命的是 `scratch_load` 和 `buffer_load` 共用 `vmcnt`，而 vmcnt 按发射顺序完成：要等 spill 读回来，就得先等它之前所有还在路上的 load，包括下一个 tile 的预取，流水被排空。`asm_vs_wg_kernel.md` 第 5 节记过一次：256×256 只 spill 了 4 个 VGPR（每个 lane 20 B），但读回后的 `s_waitcnt vmcnt(0)` 让每个 tile 边界的跨 tile 重叠都没了。所以 spill 的代价主要不在那几条读写，而在它打断了流水。

**SGPR 的 spill 便宜得多：** SGPR 溢出不进显存，而是用 `v_writelane_b32` 存进某个 VGPR 的一个 lane（一个 VGPR 能放 64 个标量），用 `v_readlane_b32` 取回，代价是几条 VALU 指令加一个 VGPR。类似地，AGPR 有空时 VGPR 也可以先溢出到 AGPR，不碰内存；大 tile 的 AGPR 是满的，只能进 scratch。

**怎么查：** 编译时加 `-Rpass-analysis=kernel-resource-usage`，每个 kernel 会打出 VGPRs、AGPRs、ScratchSize、VGPRs Spill、SGPRs Spill：

```bash
cd mbs
hipcc -O3 -std=c++20 --offload-arch=gfx950 -D__HIP_PLATFORM_AMD__=1 -D__Float4_e2m1fn_x2 -Ikernel \
      --cuda-device-only -S kernel/gemm_a4w4.hip -o /tmp/gemm_a4w4_gfx950.s \
      -Rpass-analysis=kernel-resource-usage
```

生成的汇编里，元数据 `.vgpr_spill_count`、`.sgpr_spill_count`、`.private_segment_fixed_size` 是同样的信息；正文搜 `Folded Spill` / `Folded Reload` 找 VGPR spill 的位置，搜 `v_writelane` / `v_readlane` 找 SGPR spill。找到后先看位置：在 K 循环里，基本都要消掉；在循环外，就看读回后面的 `s_waitcnt` 会不会排空正在路上的预取。

2026-11-05 编译的结果：

- **wg 256×256：** VGPR spill 0，SGPR spill 18。SGPR 存在 v255 的 lane 里，都在序幕和每个 tile 的首尾，K 循环里没有，可以不管。
- **单 wave 128×128（`gemm_a4w4_mfma_kernel<8, 8>`）：** VGPR spill 1（每个 lane 8 B scratch）。第 236 行的 `g = lane >> 4` 进 K 循环前存进 scratch，到第 338 行算 `om` 时才读回。和第 1035–1038 行是同一个问题，wg kernel 用 `v_mbcnt` 现算 lane 号解决了。这里在循环外，没有下一个 tile 的预取可排空，而且只有 `K % 1024 != 0` 才会选到它，影响很小。

**常见原因和对策**（这份代码里都用过）：

- **需求本身太大：** 缩 tile，或少预取一级。B 每多提前一个 stage 就多占 2·NSUB·4 个 VGPR；256 行 tile 每个 stage 的 MFMA 够多，提前 1 个就够藏延迟，不多花这些 VGPR（第 481–484 行）。
- **occupancy 要求太高：** `__launch_bounds__` 要求每个 SIMD 驻留 2 个 wave，每个 wave 就只剩 256 个寄存器（第 208–212 行）。
- **编译器拉长了生命周期：** 循环不变量被提到循环外、调度器为藏延迟把读取提前，都会让值活得更久。用 `asm volatile` 把计算或拷贝钉在原地（第 1035–1057 行），再加 `sched_barrier`（第 1085–1086 行）。
- **整个 wave 相同的值放进了 VGPR：** 描述符用 `sq`（`readfirstlane`）转成 SGPR，否则约 10 个 VGPR 跨整个循环活着（第 650–652 行）。
- **拿别的资源换寄存器：** SCALE_SOFF 用发射 cycle 换 VGPR（第 717–719 行）。

## 编译器：能控制什么，不能控制什么

编译器做的事，就是把我们写的 C++ 翻译成 GPU 能执行的机器指令。`hipcc` 会把 `gemm_a4w4.hip` 编两遍：一遍是 CPU 上跑的主机代码（launcher、发起 kernel 的调用），一遍是 gfx950 上跑的 kernel。kernel 这一路依次经过：C++ → LLVM IR（和硬件无关的中间表示，内联、循环展开、常量折叠在这一层做）→ AMDGPU 后端（选指令、分配寄存器、排顺序、插 `s_waitcnt`）→ 汇编 → 二进制机器码，最后打包进 `libmbs_gemm_kernel.so`。运行时 launcher 用 `hipLaunchKernelGGL` 发起 kernel，GPU 上每个 wave 按顺序执行这些指令。汇编只是机器码给人看的文本形式，GPU 实际执行的是二进制编码。

inline asm 写的部分会原样贴进汇编，编译器只按约束（`"v"`、`"s"`、`"i"`）填上寄存器号或常数。比如 B 的 load，在 256×256 实例里变成：

```text
buffer_load_dwordx4 v[16:19], v206, s[20:23], 0 offen offset:0
buffer_load_dwordx4 v[0:3],   v206, s[20:23], 0 offen offset:0x400
```

v206 是 `b_voff[ni]`，s[20:23] 是描述符，两条只差立即数 0 和 0x400（1024）。仓库里的生产 asm kernel（`kernel/asm/*.amdgcn`）更彻底：整个用汇编写成，只经过汇编器编码，编译器不参与任何决定。

C++ 只规定“算什么”，“怎么在硬件上执行”由编译器决定：用哪些指令，每个值放 VGPR、SGPR 还是 AGPR，指令怎么排，`s_waitcnt` 插在哪，一个值是一直留着还是用的时候重算，寄存器不够时 spill 谁。它的规则是“结果不变就行”，靠通用经验做决定，不知道我们的性能意图，比如“这个偏移要一直留在寄存器里”“这条 load 要和那批 MFMA 重叠”。所以结果我们能控制，过程只能引导。

引导它有三种办法，这个 kernel 都用过：

| 办法 | 例子 | 作用 |
|---|---|---|
| 告诉它推不出来的事实 | `readfirstlane`、`constexpr` / `static_for`、`__launch_bounds__` | 这个值整个 wave 相同；这些是编译期常量；寄存器预算是多少 |
| 禁止某种改写 | `opaque`、`asm volatile`、`sched_barrier` | 不准重算、合并或挪动 |
| 自己接管 | inline asm 写 load、手写 `vmcnt` 等待 | 指令完全自己定；但编译器看不到 asm 里面，正确性要自己保证，单 wave kernel 必须用两套缓冲就是这个原因 |

写完要看它实际生成了什么：汇编（`scripts/gen_asm.sh` 用的就是 `-S`）和资源报告（查法见上一节）。256×256 实例的报告是 VGPR 256、AGPR 256，都正好用满，VGPR spill 0，SGPR spill 18。这些数都不是我们指定的，而是编译器决定之后读出来的。

这份代码里吃过的亏（来自代码注释）：

- 单 wave kernel 的 `load_frag` 里有分支，编译器在每批预取后插了 `vmcnt(0)`，load 和 MFMA 没能重叠，速度只有 asm 版的一半。wg kernel 改用描述符做越界检查，去掉了这个分支。
- 描述符留在 VGPR 里，256×256 曾因此 spill；改用 `sq` 放进 SGPR。
- 每条 load 前重算地址，每个 stage 约慢 6%；改用 `opaque` 固定偏移，tile、stage 偏移挪到描述符基址上。

所以和编译器打交道是一个循环：用 C++ 写清楚意思，看汇编和资源报告，发现它做了不想要的事，就补信息、加约束；还不行，就在那一小段用汇编接管。