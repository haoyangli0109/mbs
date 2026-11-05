#!/usr/bin/env python3
"""Draw the before / after XCD-remap figures used by docs/M_sub_n_sub.md (SVG).

    python3 scripts/draw_xcd_figs.py [out_dir]     # default: docs/images

Tile positions come from the same tile_off logic as kernel/gemm_a4w4.hip (lines 443-451),
for 8192x8192 with the 256x256 tile: 1024 tiles, 256 workgroups, 4 rounds.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from draw_layout_figs import Fig, G, INK, MUTED, CELL  # noqa: E402

NUM_M = NUM_N = 32
TOTAL, GRID, NXCD, GROUP_M = NUM_M * NUM_N, 256, 8, 4
BLUE, LIGHT, ORANGE = "#2563eb", "#bfdbfe", "#ea580c"
EX_XCD, EX_WG = 1, 9


def tile(r):
    group = r // (GROUP_M * NUM_N)
    first_m = group * GROUP_M
    gsize = min(NUM_M - first_m, GROUP_M)
    return first_m + (r % (GROUP_M * NUM_N)) % gsize, (r % (GROUP_M * NUM_N)) // gsize


def remap(i):
    return (i % NXCD) * (TOTAL // NXCD) + i // NXCD if i < TOTAL // NXCD * NXCD else i


def layout(use_remap):
    """(m, n) -> (workgroup, round) for every tile."""
    owner = {}
    for rnd in range(TOTAL // GRID):
        for b in range(GRID):
            i = b + rnd * GRID
            owner[tile(remap(i) if use_remap else i)] = (b, rnd)
    assert len(owner) == TOTAL
    return owner


def fig_xcd(use_remap):
    owner = layout(use_remap)
    r0 = [mn for mn, (b, rnd) in owner.items() if b % NXCD == EX_XCD and rnd == 0]
    ms, ns = sorted({m for m, _ in r0}), sorted({n for _, n in r0})
    wg = sorted(((rnd, mn) for mn, (b, rnd) in owner.items() if b == EX_WG))
    if use_remap:
        assert (ms, ns) == ([4, 5, 6, 7], list(range(8))) and [mn for _, mn in wg] == [(5, 0), (5, 8), (5, 16), (5, 24)]
    else:
        assert (ms, ns) == ([1, 5], list(range(0, 32, 2))) and [mn for _, mn in wg] == [(1, 2), (9, 2), (17, 2), (25, 2)]

    title = ("优化后：按 XCD 重排编号（第 444–445 行）" if use_remap
             else "优化前：不做 XCD 重排，编号直接进第 446–451 行")
    f = Fig(640, 724, f"{title}。8192×8192 的 C 切成 32×32 个 tile；蓝色是 XCD 1 第 0 轮的 32 个工作组，"
                      f"浅蓝是 XCD 1 后 3 轮的 tile，橙色是 WG 9 的 4 个 tile，格里写第几轮")
    f.header(title, "8192×8192，每格 256×256；工作组 b 在 XCD b%8 上，第 i 轮算编号 b + 256·i")
    legend = [(BLUE, "XCD 1 第 0 轮的 32 个 WG"), (LIGHT, "XCD 1 第 1–3 轮"), (ORANGE, "WG 9 的 4 个 tile（格里是第几轮）")]
    lx = 14
    for col, s in legend:
        f.rect(lx, 66, 11, 11, col)
        f.text(lx + 16, 72, s, size=10.5, anchor="start")
        lx += 30 + len(s) * 10.5 * 0.82

    X0, Y0, P, GAP = 150, 150, 14, 4
    cx = lambda n: X0 + n * P + (n // 8) * GAP
    cy = lambda m: Y0 + m * P + (m // 4) * GAP
    for c in range(4):
        x = cx(8 * c) + 4 * P - P / 2
        if use_remap:
            f.text(x, Y0 - 34, f"第 {c} 轮", size=11.5)
        f.text(x, Y0 - 14, f"n{8 * c}–n{8 * c + 7}", size=10, fill=MUTED)
    for g in range(8):
        y = cy(4 * g) + 2 * P - P / 2
        if use_remap:
            f.text(X0 - 12, y - 8, f"XCD {g}", size=11, anchor="end")
            f.text(X0 - 12, y + 9, f"m{4 * g}–m{4 * g + 3}", size=10, anchor="end", fill=MUTED)
        else:
            f.text(X0 - 12, y, f"m{4 * g}–m{4 * g + 3}", size=10, anchor="end", fill=MUTED)
    if not use_remap:
        for rnd in range(4):
            y0, y1 = cy(8 * rnd), cy(8 * rnd + 7) + P - 1
            f.line(62, y0, 62, y1, stroke=MUTED)
            f.text(48, (y0 + y1) / 2, f"第 {rnd} 轮", size=11, rotate=True)
    rounds_of_wg = {mn: rnd for rnd, mn in wg}
    for (m, n), (b, rnd) in owner.items():
        if (m, n) in rounds_of_wg:
            fill = ORANGE
        elif b % NXCD == EX_XCD:
            fill = BLUE if rnd == 0 else LIGHT
        else:
            fill = CELL
        f.rect(cx(n), cy(m), P - 1, P - 1, fill)
        if (m, n) in rounds_of_wg:
            f.text(cx(n) + (P - 1) / 2, cy(m) + (P - 1) / 2, str(rounds_of_wg[(m, n)]), size=9,
                   fill="#ffffff", weight=600)

    yb = cy(31) + P + 24
    path = " → ".join(f"m{m} n{n}" for _, (m, n) in wg)
    if use_remap:
        lines = [f"XCD 1 第 0 轮的 32 个 WG 挤在 m4–m7 × n0–n7 一块：要读 4 条 A + 8 条 B = 12 条",
                 f"WG 9 四轮都在 m5 这一行，每轮往右移 8 列：{path}",
                 "XCD x 始终负责 m4x–m4x+3；第 i 轮全 GPU 一起算 n8i–n8i+7"]
    else:
        lines = [f"XCD 1 第 0 轮的 32 个 WG 散在 m1、m5 两行 × 隔一列的 16 列：要读 2 条 A + 16 条 B = 18 条",
                 f"WG 9 四轮都在 n2 这一列，每轮往下移 8 行：{path}",
                 "第 i 轮全 GPU 一起算 m8i–m8i+7 这 8 行；XCD x 在每 4 行里只占第 x%4 行，而且隔一列才有一格"]
    f.lines(14, yb, lines, size=11, anchor="start", step=21)
    return f


def tile_g(r, gm):
    group = r // (gm * NUM_N)
    first_m = group * gm
    gsize = min(NUM_M - first_m, gm)
    return first_m + (r % (gm * NUM_N)) % gsize, (r % (gm * NUM_N)) // gsize


def fig_group():
    """XCD 1's band with and without GROUP_M grouping: same 128 tiles, different rounds."""
    RC = [BLUE, "#e5e7eb", "#e5e7eb", "#e5e7eb"]
    f = Fig(760, 516, "分组改的是先后顺序：XCD 1 这条带 m4–m7 × n0–n31 里每个 tile 是第几轮算的。分组时每轮算 4 行 × 8 列，"
                      "第 0 轮要 4 条 A + 8 条 B；不分组时每轮算 1 行 × 32 列，第 0 轮要 1 条 A + 32 条 B")
    f.header("分组改的是先后顺序：XCD 1 这条带（m4–m7 × n0–n31）里每个 tile 是第几轮算的",
             "8192×8192；两种排法 XCD 1 四轮一共都算这 128 块。蓝色是第 0 轮，橙色是 WG 9；左边、上边的深色条是第 0 轮要读的 A、B")
    X0, CW = 104, 19
    for k, (gm, title) in enumerate(((4, "① 分组（GROUP_M = 4，现在的代码）：条带竖着切成 4 块"),
                                     (1, "② 不分组（直接按行排）：条带横着切成 4 条"))):
        Y0 = 132 + k * 214
        rnd = {}
        for i in range(TOTAL // GRID):
            for b in range(EX_XCD, GRID, NXCD):
                rnd[tile_g(remap(b + i * GRID), gm)] = (i, b)
        assert sorted({m for m, _ in rnd}) == [4, 5, 6, 7] and len(rnd) == 128
        r0 = [mn for mn, (i, _) in rnd.items() if i == 0]
        a_rows, b_cols = {m for m, _ in r0}, {n for _, n in r0}
        wg = [mn for mn, (i, b) in sorted(rnd.items(), key=lambda t: t[1][0]) if b == EX_WG]
        f.text(14, Y0 - 52, title, size=12.5, anchor="start", weight=600)
        f.text(X0 - 38, Y0 - 14, "B", size=10, fill=MUTED)
        for n in range(NUM_N):                                   # B strips used in round 0
            f.rect(X0 + n * CW, Y0 - 20, CW - 2, 9, INK if n in b_cols else CELL)
            if n % 8 == 0:
                f.text(X0 + n * CW + CW / 2, Y0 - 30, f"n{n}", size=8.5, fill=MUTED)
        for j, m in enumerate(range(4, 8)):                      # A strips used in round 0
            y = Y0 + j * CW
            f.rect(X0 - 18, y, 9, CW - 2, INK if m in a_rows else CELL)
            f.text(X0 - 24, y + (CW - 2) / 2, f"m{m}", size=9, anchor="end", fill=MUTED)
        for (m, n), (i, b) in rnd.items():
            x, y = X0 + n * CW, Y0 + (m - 4) * CW
            fill = ORANGE if b == EX_WG else RC[i]
            f.rect(x, y, CW - 2, CW - 2, fill)
            f.text(x + (CW - 2) / 2, y + (CW - 2) / 2, str(i), size=9,
                   fill="#ffffff" if fill in (BLUE, ORANGE) else INK, weight=600 if fill != RC[1] else 400)
        na, nb = len(a_rows), len(b_cols)
        shape = f"{na} 行 × {nb} 列"
        share = "每条 B 被同一列的 4 个 WG 共用" if gm == 4 else "每条 B 只有 1 个 WG 用"
        path = " → ".join(f"(m{m}, n{n})" for m, n in wg)
        f.lines(14, Y0 + 4 * CW + 18,
                [f"第 0 轮：{shape} → {na} 条 A + {nb} 条 B = {na + nb} 条，{share}",
                 f"WG 9 依次算 {path}"], size=10.5, anchor="start", step=20)
        if gm == 4:
            assert (na, nb) == (4, 8) and wg == [(5, 0), (5, 8), (5, 16), (5, 24)]
        else:
            assert (na, nb) == (1, 32) and wg == [(4, 1), (5, 1), (6, 1), (7, 1)]
    f.text(14, 498, "下一轮还要的 B 已经隔了一整个 tile 的计算（20 个 stage），早被挤出 L2，所以只有同一轮里的共用才算数",
           size=10.5, anchor="start", fill=MUTED)
    return f


def fig_ids():
    """Tile number id = b + 256*i on the full 32 x 32 grid: plain row-major, remap only, remap + grouping."""
    before = {(i // NUM_N, i % NUM_N): i for i in range(TOTAL)}
    mid = {(remap(i) // NUM_N, remap(i) % NUM_N): i for i in range(TOTAL)}
    after = {tile(remap(i)): i for i in range(TOTAL)}
    assert len(mid) == len(after) == TOTAL
    for (m, n), i in mid.items():
        assert i == 256 * (m % 4) + 8 * n + m // 4
    for (m, n), i in after.items():
        assert i == m // 4 + 8 * (m % 4) + 32 * (n % 8) + 256 * (n // 8)
    assert [before[(0, n)] for n in range(NUM_N)] == list(range(32))
    assert [after[(m, 0)] for m in range(NUM_M)] == [m // 4 + 8 * (m % 4) for m in range(NUM_M)]

    RF = [G[r][2] for r in range(4)]                     # light fill = round id // 256
    title = "tile 编号在重排、分组前后的排法（8192×8192，32 × 32 个 tile）"
    f = Fig(866, 1858, f"{title}。编号 id = b + 256·i 是工作组 b 第 i 轮拿到的编号。① 不重排也不分组，按行主序，"
                       "第 m 行是 32m 到 32m+31；② 只做重排，XCD x 的 128 个编号连成第 4x 到 4x+3 这 4 整行，条带里第 i 行"
                       "是第 i 轮；③ 重排加分组之后，原来的一行变成一列，第 i 轮的 256 个编号占第 8i 到 8i+7 列，每个 XCD "
                       "每轮是一块 4 × 8。颜色是第几轮，深蓝是编号 0 到 31")
    f.header(title, "编号 id = b + 256·i：工作组 b 第 i 轮拿到的编号；颜色是第几轮，深蓝是编号 0–31（原来的第 0 行）")
    lx = 14
    for col, s in ((RF[0], "第 0 轮（编号 0–255）"), (RF[1], "第 1 轮（256–511）"), (RF[2], "第 2 轮（512–767）"),
                   (RF[3], "第 3 轮（768–1023）"), (G[0][0], "编号 0–31")):
        f.rect(lx, 66, 11, 11, col)
        f.text(lx + 16, 72, s, size=10.5, anchor="start")
        lx += 30 + len(s) * 10.5 * 0.82

    LX, CW, CH, GXB, GYB = 64, 24, 14, 6, 4
    cx = lambda n: LX + n * CW + (n // 8) * GXB
    panels = ((before, "① 原来：不重排也不分组，按行主序，tile (m, n) 的编号是 32·m + n"),
              (mid, "② 只做重排（第 444–445 行）：XCD x 的 128 个编号连成 m4x–m4x+3 这 4 整行，再按行主序摆"),
              (after, "③ 重排 + 分组（第 444–451 行）：条带里改成先往下、再往右，每个 XCD 每轮是一块 4 × 8"))
    for k, (grid, head) in enumerate(panels):
        PY = 104 + k * 568
        Y0 = PY + 36
        cy = lambda m: Y0 + m * CH + (m // 4) * GYB
        f.text(14, PY, head, size=12.5, anchor="start", weight=600)
        for c in range(4):
            label = f"第 {c} 轮 · n{8 * c}–n{8 * c + 7}" if k == 2 else f"n{8 * c}–n{8 * c + 7}"
            f.text(cx(8 * c) + 4 * CW - 1, PY + 22, label, size=10, fill=MUTED)
        for g in range(NXCD):
            yc = cy(4 * g) + 2 * CH - 1
            if k:
                f.text(LX - 8, yc - 7, f"XCD {g}", size=9.5, anchor="end")
                f.text(LX - 8, yc + 7, f"m{4 * g}–{4 * g + 3}", size=8.5, anchor="end", fill=MUTED)
            else:
                f.text(LX - 8, yc, f"m{4 * g}–{4 * g + 3}", size=8.5, anchor="end", fill=MUTED)
        for (m, n), i in grid.items():
            hot = i < 32
            x, y = cx(n), cy(m)
            f.rect(x, y, CW - 2, CH - 2, G[0][0] if hot else RF[i // GRID])
            f.text(x + (CW - 2) / 2, y + (CH - 2) / 2, str(i), size=9,
                   fill="#ffffff" if hot else INK, weight=600 if hot else 400)

    f.lines(14, 1780,
            ["① 第 i 轮的 256 个编号占第 8i–8i+7 行；同一个工作组 b 的四个编号 b、b+256、b+512、b+768 在同一列",
             "② id = 256·(m%4) + 8·n + m/4：第 4x–4x+3 行全是 XCD x 的编号，条带里第 i 行就是第 i 轮，每轮是一整行 1 × 32",
             "③ id = m/4 + 8·(m%4) + 32·(n%8) + 256·(n/8)：条带里往下 +8，往右一列 +32，下一个条带 +1，往右 8 列 +256",
             "从 ① 到 ③：原来的一行变成一列；第 i 轮从第 8i–8i+7 行（①），到每个条带的第 i 行（②），再到第 8i–8i+7 列（③）"],
            size=11, anchor="start", step=20)
    return f


FIGS = {
    "M_sub_n_sub_xcd_before.svg": lambda: fig_xcd(False),
    "M_sub_n_sub_xcd_after.svg": lambda: fig_xcd(True),
    "M_sub_n_sub_ids.svg": fig_ids,
    "M_sub_n_sub_group.svg": fig_group,
}

if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "docs", "images")
    for name, fn in FIGS.items():
        fn().save(os.path.join(out, name))
