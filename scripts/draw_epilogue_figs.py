#!/usr/bin/env python3
"""Draw the epilogue lane-exchange figure (SVG, Chinese labels).

    python3 scripts/draw_epilogue_figs.py [out_dir]     # default: docs/images

The owner of every cell is obtained by simulating the epilogue of
gemm_a4w4_wg_kernel<16, 4, 4> lane by lane: the MFMA output layout,
v_cvt_pk_bf16_f32, v_permlane16_swap_b32, DPP row_ror:8 and the va / vb select.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from draw_layout_figs import Fig, G, INK, MUTED, RED  # noqa: E402

NSUB = 4


def acc(ni):
    """acc[mi][ni] after the (operand-swapped) MFMA: lane L, element i = (row, col) in the row block."""
    return [[(L % 16, 16 * ni + 4 * (L // 16) + i) for i in range(4)] for L in range(64)]


def permlane16_swap(a, b):
    """Odd 16-lane rows of a <-> even 16-lane rows of b; returns (new a, new b)."""
    na, nb = a[:], b[:]
    for L in range(64):
        if (L // 16) % 2:
            na[L] = b[L - 16]
        else:
            nb[L] = a[L + 16]
    return na, nb


def pair(np_):
    x, y = acc(2 * np_), acc(2 * np_ + 1)
    s0 = permlane16_swap([(x[L][0], x[L][1]) for L in range(64)], [(y[L][0], y[L][1]) for L in range(64)])
    s1 = permlane16_swap([(x[L][2], x[L][3]) for L in range(64)], [(y[L][2], y[L][3]) for L in range(64)])
    return [[s0[0][L], s1[0][L], s0[1][L], s1[1][L]] for L in range(64)]


def ror8(v):
    return [v[(L & ~15) | ((L % 16 + 8) % 16)] for L in range(64)]


def owners():
    """Return three dicts (row, first col of chunk) -> lane: after MFMA, after pair, after DPP."""
    o1 = {}
    for ni in range(NSUB):
        for L, vals in enumerate(acc(ni)):
            r, c = vals[0]
            assert [v for v in vals] == [(r, c + i) for i in range(4)]
            o1[(r, c)] = L
    v = [pair(0), pair(1)]
    o2 = {}
    for np_ in range(2):
        for L in range(64):
            elems = [e for d in v[np_][L] for e in d]
            r, c = elems[0]
            assert elems == [(r, c + i) for i in range(8)]
            o2[(r, c)] = L
    r0 = [ror8([v[0][L][i] for L in range(64)]) for i in range(4)]
    r1 = [ror8([v[1][L][i] for L in range(64)]) for i in range(4)]
    o3 = {}
    for L in range(64):
        lo8 = L % 16 < 8
        va = [v[0][L][i] if lo8 else r1[i][L] for i in range(4)]
        vb = [r0[i][L] if lo8 else v[1][L][i] for i in range(4)]
        for regs in (va, vb):
            elems = [e for d in regs for e in d]
            r, c = elems[0]
            assert elems == [(r, c + i) for i in range(8)]
            o3[(r, c)] = L
    # closed forms used in the text
    for r in range(16):
        for c in range(0, 64, 8):
            gg = [0, 2, 1, 3][(c % 32) // 8]
            assert o2[(r, c)] == 16 * gg + r
            row16 = (r % 8) + 8 * (c // 32)
            assert o3[(r, c)] == 16 * gg + row16
    assert len(o1) == 256 and len(o2) == 128 and len(o3) == 128
    return o1, o2, o3


def fig_epilogue_lanes():
    o1, o2, o3 = owners()
    f = Fig(1090, 548, "epilogue 里一个 16 行行块的 lane 交换。① MFMA 之后，lane (row16, G) 在 4 个累加器里各有本行 4 列；"
                       "② pair 把两个累加器转成 bf16 并用 permlane16_swap 交换，每个 lane 拿本行连续 8 列；"
                       "③ DPP row_ror:8 让第 r 行和第 r+8 行的 lane 互换一半，va 是第 0 到 7 行、vb 是第 8 到 15 行，"
                       "各用一条 store 写出 8 行整 128 字节")
    f.header("epilogue：一个 16 行行块的 64 列，怎么从累加器变成两条整行 store",
             "256×256 tile，一个 wave 的 16 行 × 64 列（4 个累加器）。格子颜色是 lane 组 G = lane/16，"
             "格子里的数是 lane 号，红框是第 5 行")
    X0, PW, GAP = 44, 272, 60
    Y0, RH = 146, 16
    CW = PW / 64                        # px per column
    px = [X0, X0 + PW + GAP, X0 + 2 * (PW + GAP)]

    def grid(x0, own, width, small):
        for (r, c), L in own.items():
            gg, row16 = L // 16, L % 16
            fill = G[gg][1] if row16 < 8 else G[gg][2]
            x, y = x0 + c * CW, Y0 + r * RH
            f.rect(x, y, width * CW, RH, fill=fill, stroke="#ffffff", sw=1)
            f.text(x + width * CW / 2, y + RH / 2 + 0.5, str(L), size=8.2 if small else 9.5, fill=INK, mono=True)
        f.rect(x0, Y0 + 5 * RH, PW, RH, stroke=RED, sw=1.6)
        for c in (0, 16, 32, 48, 64):
            f.line(x0 + c * CW, Y0 + 16 * RH, x0 + c * CW, Y0 + 16 * RH + 4, stroke=MUTED)
            f.text(x0 + c * CW, Y0 + 16 * RH + 12, str(c), size=9, fill=MUTED)

    titles = [("① MFMA 之后：acc[mi][0..3]", "每个累加器里，lane 有本行 4 列"),
              ("② pair()：转 bf16 + permlane16_swap", "每个 lane 拿本行连续 8 列（16 B）"),
              ("③ DPP row_ror:8 + 选择", "第 r 行和第 r+8 行的 lane 互换一半")]
    for p, (t, s) in enumerate(titles):
        f.text(px[p], 74, t, size=12.5, anchor="start", weight=600)
        f.text(px[p], 93, s, size=11, anchor="start", fill=MUTED)
    grid(px[0], o1, 4, True)
    grid(px[1], o2, 8, False)
    grid(px[2], o3, 8, False)

    for ni in range(NSUB):                                   # panel 1: which accumulator
        a, b = px[0] + 16 * ni * CW + 2, px[0] + 16 * (ni + 1) * CW - 2
        f.text((a + b) / 2, 118, f"ni = {ni}", size=10, fill=MUTED)
        f.bracket(a, b, 130)
    for k, s in enumerate(["v0 = pair(mi, 0)", "v1 = pair(mi, 1)"]):   # panel 2: which pair
        a, b = px[1] + 32 * k * CW + 2, px[1] + 32 * (k + 1) * CW - 2
        f.text((a + b) / 2, 118, s, size=10, fill=MUTED)
        f.bracket(a, b, 130)
    for k, s in enumerate(["row16 < 8 的 lane", "row16 ≥ 8 的 lane"]):  # panel 3: which lanes
        a, b = px[2] + 32 * k * CW + 2, px[2] + 32 * (k + 1) * CW - 2
        f.text((a + b) / 2, 118, s, size=10, fill=MUTED)
        f.bracket(a, b, 130)

    for r in range(16):
        f.text(X0 - 8, Y0 + r * RH + RH / 2, str(r), size=9, anchor="end", fill=MUTED)
    f.text(X0 - 8, Y0 - 10, "行", size=9, anchor="end", fill=MUTED)
    for p in range(2):                                       # arrows between panels
        y = Y0 + 8 * RH
        f.line(px[p] + PW + 10, y, px[p + 1] - 10, y, stroke=INK, sw=1.4, arrow=True)

    xb = px[2] + PW + 8                                      # va / vb brackets
    for k, (name, store) in enumerate([("va：第 0–7 行", "store 1"), ("vb：第 8–15 行", "store 2")]):
        ya, yb = Y0 + 8 * k * RH + 2, Y0 + 8 * (k + 1) * RH - 2
        f.path(f"M{xb - 4} {ya}L{xb} {ya}L{xb} {yb}L{xb - 4} {yb}", stroke=INK, sw=1.2)
        f.text(xb + 8, (ya + yb) / 2 - 8, name, size=11, anchor="start", weight=600)
        f.text(xb + 8, (ya + yb) / 2 + 9, store, size=11, anchor="start", fill=MUTED)

    ly = Y0 + 16 * RH + 36                                   # legend
    x = X0
    for gg in range(4):
        f.rect(x, ly - 6, 12, 12, fill=G[gg][1])
        f.rect(x + 13, ly - 6, 12, 12, fill=G[gg][2])
        s = f"G{gg}：lane {16 * gg}–{16 * gg + 15}"
        f.text(x + 31, ly, s, size=11, anchor="start")
        x += 31 + len(s) * 6.4 + 22
    f.text(x, ly, "深色：row16 = lane%16 < 8，浅色：row16 ≥ 8", size=11, anchor="start", fill=MUTED)
    f.lines(X0, ly + 30, [
        "行、列是本 wave 一个 16 行行块里的相对位置：C 的行 = m0 + 16·mi + 行，C 的列 = n0 + 列（n0 已含 64·wave）。底下的数是列号。",
        "③ 里每一行由 8 个 lane 拼成：row16 < 8 的 4 个 lane 出第 0–31 列，row16 ≥ 8 的 4 个 lane 出第 32–63 列，每个 lane 16 B。",
        "va、vb 各用一条 buffer_store_dwordx4 写出：64 个 lane × 16 B = 8 行 × 128 B，每行正好一条整 cache line。",
    ], size=11, anchor="start", fill=INK)
    return f


def fig_pair_row():
    """One accumulator's lane layout, and row 5 of pair(mi, 0) before / after permlane16_swap."""
    blk = {vals[0]: L for L, vals in enumerate(acc(0))}
    before = {vals[0][1]: L for ni in range(2) for L, vals in enumerate(acc(ni)) if vals[0][0] == 5}
    after = {}
    for L, regs in enumerate(pair(0)):
        elems = [e for d in regs for e in d]
        assert elems == [(elems[0][0], elems[0][1] + i) for i in range(8)]
        if elems[0][0] == 5:
            after[elems[0][1]] = L
    assert [before[c] for c in range(0, 32, 4)] == [5, 21, 37, 53, 5, 21, 37, 53]
    assert [after[c] for c in range(0, 32, 8)] == [5, 37, 21, 53]
    moved = [c for c in range(0, 32, 4) if after[c // 8 * 8] != before[c]]
    assert moved == [4, 12, 16, 24]

    f = Fig(790, 512, "epilogue 的第一步。左：一个 16×16 子块 acc[mi][ni]，第 r 行的 16 列分给 lane r、r+16、r+32、r+48，"
                      "每个 lane 拿连续 4 列。右：一个 pair 把相邻两个子块拼成 32 列，第 5 行交换前 lane 5 拿第 0 到 3 列和"
                      "第 16 到 19 列；permlane16_swap 之后 lane 5 拿第 0 到 7 列，lane 37 拿 8 到 15 列，lane 21 拿 16 到 23 列，"
                      "lane 53 拿 24 到 31 列，每个 lane 都是本行连续 8 列，即 16 字节")
    f.header("epilogue 的第一步：一个 16×16 子块怎么分给 lane，pair 怎么把一行凑成 16 B",
             "256×256 tile。格子颜色是 lane 组 G = lane/16，格子里的数是 lane 号，红框是第 5 行")
    X0, Y0, CW, RH = 44, 146, 18, 16
    f.text(X0, 74, "① 一个子块 acc[mi][ni]：16 行 × 16 列", size=12.5, anchor="start", weight=600)
    f.text(X0, 93, "每个 lane 拿本行连续 4 列（4 个 fp32）", size=11, anchor="start", fill=MUTED)
    for (r, c), L in blk.items():
        x, y = X0 + c * CW, Y0 + r * RH
        f.rect(x, y, 4 * CW, RH, fill=G[L // 16][1] if L % 16 < 8 else G[L // 16][2], stroke="#ffffff", sw=1)
        f.text(x + 2 * CW, y + RH / 2 + 0.5, str(L), size=9.5, mono=True)
    f.rect(X0, Y0 + 5 * RH, 16 * CW, RH, stroke=RED, sw=1.6)
    for r in range(16):
        f.text(X0 - 8, Y0 + r * RH + RH / 2, str(r), size=9, anchor="end", fill=MUTED)
    f.text(X0 - 8, Y0 - 10, "行", size=9, anchor="end", fill=MUTED)
    for c in range(0, 17, 4):
        f.line(X0 + c * CW, Y0 + 16 * RH, X0 + c * CW, Y0 + 16 * RH + 4, stroke=MUTED)
        f.text(X0 + c * CW, Y0 + 16 * RH + 12, str(c), size=9, fill=MUTED)

    X1, CW2, SH = 388, 12, 22
    f.text(X1, 74, "② 一个 pair（ni = 2np、2np+1）的第 5 行", size=12.5, anchor="start", weight=600)
    f.text(X1, 93, "permlane16_swap 前后，每个 lane 拿哪几列", size=11, anchor="start", fill=MUTED)
    for k, s in enumerate(["子块 2np：列 0–15", "子块 2np+1：列 16–31"]):
        a, b = X1 + 16 * k * CW2 + 2, X1 + 16 * (k + 1) * CW2 - 2
        f.text((a + b) / 2, 118, s, size=10, fill=MUTED)
        f.bracket(a, b, 130)
    yb, ya = 160, 222
    f.text(X1, yb - 9, "交换前（MFMA 之后）", size=10.5, anchor="start", fill=MUTED)
    f.text(X1, ya - 9, "交换后（pair 返回的 16 B）", size=10.5, anchor="start", fill=MUTED)
    for c, L in before.items():
        f.rect(X1 + c * CW2, yb, 4 * CW2, SH, fill=G[L // 16][1], stroke="#ffffff", sw=1)
        f.text(X1 + (c + 2) * CW2, yb + SH / 2 + 0.5, str(L), size=10, mono=True)
    for c, L in after.items():
        f.rect(X1 + c * CW2, ya, 8 * CW2, SH, fill=G[L // 16][1], stroke="#ffffff", sw=1)
        for h in (c, c + 4):
            if h in moved:
                f.rect(X1 + h * CW2 + 1, ya + 1, 4 * CW2 - 2, SH - 2, stroke=INK, sw=1.8)
                f.text(X1 + (h + 2) * CW2, ya + SH / 2 + 0.5, f"原 {before[h]}", size=9)
            else:
                f.text(X1 + (h + 2) * CW2, ya + SH / 2 + 0.5, str(L), size=10, mono=True)
    for c in range(0, 33, 8):
        f.line(X1 + c * CW2, ya + SH, X1 + c * CW2, ya + SH + 4, stroke=MUTED)
        f.text(X1 + c * CW2, ya + SH + 12, str(c), size=9, fill=MUTED)
    f.lines(X1, ya + SH + 40, [
        "粗框是换了主人的 4 列，“原 21”表示原来在 lane 21 手里：",
        "lane 5 把第 16–19 列给 lane 21，换回第 4–7 列；",
        "lane 37 和 lane 53 同样互换。",
        "只在 lane L 和 L±16 之间换，行号不变。",
        "换完每个 lane 拿本行连续 8 列 = 8 个 bf16 = 16 B，",
        "pair 把它作为一个 u32x4 返回。"], size=10.5, anchor="start", step=19)

    ly = Y0 + 16 * RH + 44
    x = X0
    for gg in range(4):
        f.rect(x, ly - 6, 12, 12, fill=G[gg][1])
        f.rect(x + 13, ly - 6, 12, 12, fill=G[gg][2])
        s = f"G{gg}：lane {16 * gg}–{16 * gg + 15}"
        f.text(x + 31, ly, s, size=11, anchor="start")
        x += 31 + len(s) * 6.4 + 22
    f.text(x, ly, "深色：row16 < 8，浅色：row16 ≥ 8", size=11, anchor="start", fill=MUTED)
    f.lines(X0, ly + 28, [
        "16×16 是一条 MFMA 输出的形状；acc[mi][ni] 是整个 K 上同一 (mi, ni) 的 MFMA 累加的和（K = 5120 时 40 条），分法不变。",
        "一个 lane 只有本行 4 列，bf16 只有 8 B；拼两个子块后一行 32 列，4 个 lane 每人 8 列 = 16 B，才凑成一条 16 B 的 store。",
    ], size=11, anchor="start", step=20)
    return f


def agpr_base(mi, ni):
    """First of the 4 AGPRs holding acc[mi][ni] in the 256x256 instance (read off its asm)."""
    return (128 + 16 * mi + 4 * ni) % 256


def fig_agpr_layout():
    """Which elements of a wave's 256 x 64 sit in one lane's AGPRs."""
    bases = sorted(agpr_base(mi, ni) for mi in range(16) for ni in range(NSUB))
    assert bases == list(range(0, 256, 4))
    blk = {vals[0]: L for L, vals in enumerate(acc(0))}
    assert len(blk) == 64 and set(blk.values()) == set(range(64))
    LANE = 37
    lr, lg = LANE % 16, LANE // 16
    assert blk[(lr, 4 * lg)] == LANE

    f = Fig(1000, 706, "一个 wave 的 256 行 × 64 列由 16 × 4 个 16×16 小块组成，每个小块是一个累加器，占 4 个连续 AGPR。"
                       "lane L 在每个小块里都拿同一位置的 1 行 × 4 列：第 L%16 行、第 4·(L/16) 到 +3 列，"
                       "64 个小块共 256 个数，正好 256 个 AGPR。例如 lane 37 在每块里拿第 5 行第 8 到 11 列，"
                       "acc[0][0] 里这 4 个数在 a128 到 a131")
    f.header("每个 lane 的 AGPR 里放的是哪些元素：16 × 4 个小块，每块同一位置的 1 行 × 4 列",
             "256×256 tile，一个 wave 的 256 行 × 64 列（相对本 wave 的 m0、n0）。AGPR 编号读自这个实例的汇编")

    LX0, LY0, BW, BH = 118, 146, 80, 28
    f.text(16, 74, "① 一个 wave：16 × 4 个 16×16 小块（累加器）", size=12.5, anchor="start", weight=600)
    f.text(16, 93, "格子里是这个累加器占的 4 个 AGPR；红条是 lane 37 在每块里拿的 1 行 × 4 列",
           size=11, anchor="start", fill=MUTED)
    for ni in range(NSUB):
        f.text(LX0 + ni * BW + BW / 2, 116, f"ni = {ni}", size=10, fill=MUTED)
        f.text(LX0 + ni * BW + BW / 2, 131, f"列 {16 * ni}–{16 * ni + 15}", size=10, fill=MUTED)
    for mi in range(16):
        y = LY0 + mi * BH
        f.text(LX0 - 6, y + BH / 2, f"mi={mi} · 行 {16 * mi}–{16 * mi + 15}", size=9.5, anchor="end", fill=MUTED)
        for ni in range(NSUB):
            x, b = LX0 + ni * BW, agpr_base(mi, ni)
            f.rect(x, y, BW, BH, fill="#f3f4f6", stroke="#d1d5db", sw=1)
            f.rect(x + 4 * lg * BW / 16, y + lr * BH / 16, 4 * BW / 16, max(BH / 16, 2.6), fill=RED)
            f.text(x + BW / 2, y + BH * 0.76, f"a{b}–{b + 3}", size=9.5, mono=True)
    f.rect(LX0, LY0, BW, BH, stroke=INK, sw=1.8)
    yb = LY0 + 16 * BH + 10
    for k in range(2):
        a, b = LX0 + 2 * k * BW + 3, LX0 + 2 * (k + 1) * BW - 3
        f.bracket(a, b, yb, up=False)
        f.text((a + b) / 2, yb + 12, f"pair(mi, {k})：ni = {2 * k}、{2 * k + 1}", size=10, fill=MUTED)

    RX0, RY0, CW, RH = 556, 146, 24, 22
    f.text(RX0 - 30, 74, "② 放大一个小块：acc[0][0]，a128–a131", size=12.5, anchor="start", weight=600)
    f.text(RX0 - 30, 93, "64 个 lane 各拿同一位置的 1 行 × 4 列，正好分完 16 × 16", size=11, anchor="start",
           fill=MUTED)
    for gg in range(4):
        f.text(RX0 + (4 * gg + 2) * CW, 131, f"G={gg}：列 {4 * gg}–{4 * gg + 3}", size=9.5, fill=MUTED)
    for (r, c), L in blk.items():
        x, y = RX0 + c * CW, RY0 + r * RH
        for i in range(4):
            f.rect(x + i * CW, y, CW, RH, fill=G[L // 16][1], stroke="#ffffff", sw=1)
        f.text(x + 2 * CW, y + RH / 2 + 0.5, str(L), size=10, mono=True)
    f.rect(RX0 + 4 * lg * CW, RY0 + lr * RH, 4 * CW, RH, stroke=RED, sw=2)
    for r in range(16):
        f.text(RX0 - 6, RY0 + r * RH + RH / 2, str(r), size=9, anchor="end", fill=MUTED)
    f.text(RX0 - 6, RY0 - 10, "行", size=9, anchor="end", fill=MUTED)

    ey = RY0 + 16 * RH + 24
    f.text(RX0, ey, f"lane {LANE} 在这个小块里的 4 个数：第 {lr} 行第 {4 * lg}–{4 * lg + 3} 列，依次放在 4 个 AGPR",
           size=10.5, anchor="start")
    bx, bw = RX0 + 4 * lg * CW + 2 * CW - 2 * 54, 54
    for i in range(4):
        f.rect(bx + i * bw, ey + 14, bw - 4, 26, fill="#fee2e2", stroke=RED, sw=1.2)
        f.text(bx + i * bw + (bw - 4) / 2, ey + 27, f"a{agpr_base(0, 0) + i}", size=10.5, mono=True)
        f.text(bx + i * bw + (bw - 4) / 2, ey + 52, f"列 {4 * lg + i}", size=9.5, fill=MUTED)

    f.lines(16, 654, [
        "lane L =（row16, G），row16 = L%16，G = L/16：在每个小块里都拿第 row16 行、第 4G 到 4G+3 列；16 × 4 个小块 × 4 = 256 个 fp32，正好 256 个 AGPR。",
        "一个累加器 acc[mi][ni] = 4 个连续 AGPR × 64 个 lane，从 a[(128 + 16·mi + 4·ni) mod 256] 开始，第 i 个 AGPR 放那 4 列里的第 i 列。",
        "例：lane 37 在第 5、21、…、245 行各拿第 8–11、24–27、40–43、56–59 列；a128–a131 是 C[5][8..11]，a212–a215 是 C[85][24..27]。",
    ], size=10.5, anchor="start", step=18)
    return f


def fig_dpp_stores():
    """Store coverage of one row block: v0 / v1 stored directly vs va / vb after the DPP exchange."""
    _, o2, o3 = owners()
    S1, S2 = "#bae6fd", "#fed7aa"
    direct = {(r, c): (L, 1 if c < 32 else 2) for (r, c), L in o2.items()}
    swapped = {(r, c): (L, 1 if r < 8 else 2) for (r, c), L in o3.items()}
    for own in (direct, swapped):
        assert len(own) == 128 and all(sum(v[1] == s for v in own.values()) == 64 for s in (1, 2))
    assert all(len({direct[(r, c)][1] for c in range(0, 64, 8)}) == 2 for r in range(16))
    assert all(len({swapped[(r, c)][1] for c in range(0, 64, 8)}) == 1 for r in range(16))

    f = Fig(790, 540, "epilogue 第三步为什么要交换。左：直接存 v0、v1，第 1 条 store 写 16 行的左 32 列，第 2 条写右 32 列，"
                      "每行被两条 store 各写一半。右：DPP 交换后存 va、vb，第 1 条写第 0 到 7 行的整行，第 2 条写第 8 到 15 行的整行。"
                      "lane 5 把第 5 行右段交给 lane 13，换回第 13 行左段")
    f.header("epilogue 第 ③ 步为什么要交换：直接存 vs 交换后再存",
             "256×256 tile，一个 wave 的一个行块（16 行 × 64 列）。颜色是第几条 store，格子里是写这 8 列的 lane，粗框是 lane 5、13")
    CW, RH, Y0 = 40, 18, 136
    panels = [(44, direct, "① 直接存 v0、v1：第 1 条写左半，第 2 条写右半",
               "每行被两条 store 各写一半：一条碰 16 条 cache line"),
              (424, swapped, "② 交换后存 va、vb：第 1 条写上 8 行，第 2 条写下 8 行",
               "每行由一条 store 写满：一条碰 8 条 cache line")]
    for x0, own, title, sub in panels:
        f.text(x0, 74, title, size=12, anchor="start", weight=600)
        f.text(x0, 93, sub, size=10.5, anchor="start", fill=MUTED)
        f.text(x0 - 8, Y0 - 10, "行", size=9, anchor="end", fill=MUTED)
        for k in range(8):
            f.text(x0 + k * CW + CW / 2, Y0 - 10, f"{8 * k}–{8 * k + 7}", size=9, fill=MUTED)
        for r in range(16):
            f.text(x0 - 8, Y0 + r * RH + RH / 2, str(r), size=9, anchor="end", fill=MUTED)
            for k in range(8):
                L, s = own[(r, 8 * k)]
                x, y = x0 + k * CW, Y0 + r * RH
                f.rect(x, y, CW, RH, fill=S1 if s == 1 else S2, stroke="#ffffff", sw=1)
                if L in (5, 13):
                    f.rect(x + 1, y + 1, CW - 2, RH - 2, stroke=INK, sw=1.8)
                f.text(x + CW / 2, y + RH / 2 + 0.5, str(L), size=9.5, mono=True, weight=600 if L in (5, 13) else 400)

    ly = Y0 + 16 * RH + 30
    x = 44
    for fill, s in ((S1, "第 1 条 store（左：v0，右：va）"), (S2, "第 2 条 store（左：v1，右：vb）")):
        f.rect(x, ly - 6, 12, 12, fill=fill)
        f.text(x + 18, ly, s, size=11, anchor="start")
        x += 18 + len(s) * 9.6 + 24
    f.lines(44, ly + 28, [
        "一格 = 8 列 = 16 B，一行 64 列 = 128 B = 一条 cache line（n0 是 64 的倍数）。两种写法都是 2 条 store、每条 64 个 lane × 16 B = 1 KB。",
        "粗框：lane 5 把第 5 行右段（v1）交给 lane 13，换回第 13 行左段（v0）；r0、r1 由 DPP row_ror:8 从 lane L^8 取来，",
        "va = lo8 ? v0 : r1，vb = lo8 ? r0 : v1。",
    ], size=10.5, anchor="start", step=19)
    return f


FIGS = {"epilogue_lanes.svg": fig_epilogue_lanes, "epilogue_pair_row.svg": fig_pair_row,
        "epilogue_agpr.svg": fig_agpr_layout, "epilogue_dpp_stores.svg": fig_dpp_stores}

if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "..", "docs", "images")
    os.makedirs(out, exist_ok=True)
    for name, fn in FIGS.items():
        fn().save(os.path.join(out, name))
