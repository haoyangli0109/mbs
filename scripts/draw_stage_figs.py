#!/usr/bin/env python3
"""Draw the two-section pipeline figure used by docs/stage.md (SVG).

    python3 scripts/draw_stage_figs.py [out_dir]     # default: docs/images

Counts and stage offsets come from the constants of gemm_a4w4_wg_kernel<16, 4, 4>
(PF, PF_B, LOADS_PER_STAGE, W_START, W_MID, MS0), the same way kernel/gemm_a4w4.hip
derives them.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from draw_layout_figs import Fig, G, INK, MUTED, LINE, text_width  # noqa: E402

MSUB, NSUB, WAVES, PF = 16, 4, 4, 2
PF_B = 2 if (NSUB <= 4 and MSUB < 16) else 1
MTILE, NSB = 16 * MSUB, MSUB // 2
KA = 2 * NSUB + NSUB // 2
A_DMA, AS_DMA = MTILE // 8 // WAVES, (NSB + WAVES - 1) // WAVES
LOADS = KA + A_DMA + AS_DMA
W_START = (LOADS - KA) + LOADS * (PF_B - 1)
W_MID = KA + LOADS * (PF - 2)
MS0 = (MSUB // 2 + 1) // 2 * 2
MS1 = MSUB - MS0
assert (PF_B, KA, LOADS, W_START, W_MID, MS0, MS1) == (1, 10, 20, 10, 10, 8, 8)
# section(): load k goes before MFMA k*TLW/NL, LDS read r before MFMA r*TS/NR
TS, LOAD_WIN, NR = 2 * MS0 * NSUB, 7, 2 * MS0 + MS0 // 2
TLW = max(TS * LOAD_WIN // 8, KA)
LOAD_SLOTS = [k * TLW // KA for k in range(KA)]
assert LOAD_SLOTS == [0, 5, 11, 16, 22, 28, 33, 39, 44, 50] and LOADS - KA == KA


def S(d):
    return "s" if d == 0 else (f"s+{d}" if d > 0 else f"s−{-d}")


def slot(d):
    return "s%4" if d == 0 else f"({S(d)})%4"


def section_cells(t, sec):
    """(row, stage offset, label) of what step s+t does in section `sec`."""
    if sec == 0:
        return [(2, t + PF_B, f"{S(t + PF_B)} → bf[{slot(t + PF_B)}]"),
                (3, t + PF_B, f"{S(t + PF_B)} → bs[{slot(t + PF_B)}]"),
                (4, t, f"{S(t)} · mi {MS0}–{MSUB - 1} → afr1"),
                (5, t, f"{S(t)} · 块 {MS0 // 2}–{NSB - 1} → asc1"),
                (6, t, f"{S(t)} · mi 0–{MS0 - 1}")]
    return [(0, t + PF, f"{S(t + PF)} → 槽 {slot(t + PF)}"),
            (1, t + PF, f"{S(t + PF)} → 槽 {slot(t + PF)}"),
            (4, t + 1, f"{S(t + 1)} · mi 0–{MS0 - 1} → afr0"),
            (5, t + 1, f"{S(t + 1)} · 块 0–{MS0 // 2 - 1} → asc0"),
            (6, t, f"{S(t)} · mi {MS0}–{MSUB - 1}")]


def fig_stage_pipeline():
    cols = [(-2, 1), (-1, 0), (-1, 1), (0, 0), (0, 1)]
    rows = [("A 数据", f"显存 → LDS · {A_DMA} 条"), ("A scale", f"显存 → LDS · {AS_DMA} 条"),
            ("B 片段", f"显存 → bf · {2 * NSUB} 条"), ("B scale", f"显存 → bs · {NSUB // 2} 条"),
            ("A 片段", f"LDS → afr · {2 * MS0} 条"), ("A scale", f"LDS → asc · {MS0 // 2} 条"),
            ("MFMA", f"每段 {2 * MS0 * NSUB} 条")]
    f = Fig(790, 512, "一个 stage 的数据怎么流过两个段，256×256 tile，PF 为 2，PF_B 为 1。每行一种数据，从左到右是时间："
                      "step s−2 段 1 发 stage s 的 A 和 A scale DMA，写进 LDS 槽 s%4；step s−1 段 0 发 stage s 的 B 和 "
                      "B scale，写进 bf[s%4] 和 bs[s%4]；step s−1 段 1 开头等 A(s) 到齐并 barrier，再读 stage s 的 "
                      "mi 0 到 7 和 scale 块 0 到 3，进 afr0 和 asc0；step s 段 0 开头等 B(s)，算 stage s 的 mi 0 到 7，"
                      "同时读 mi 8 到 15 和 scale 块 4 到 7，进 afr1 和 asc1；step s 段 1 算 stage s 的 mi 8 到 15")
    f.header("一个 stage 的数据怎么流过两个段：A、A scale、B、B scale、A 片段、MFMA",
             f"256×256 tile，PF = {PF}、PF_B = {PF_B}。每行一种数据，从左到右是时间；"
             "蓝色是 stage s 自己的数据，灰色是同一时刻别的 stage")
    CX0, P, CW = 132, 128, 118
    RY0, RP, RH = 128, 38, 28
    cx = lambda c: CX0 + c * P
    ry = lambda r: RY0 + r * RP

    f.text(cx(0) + CW / 2, 72, "step s−2", size=12, weight=600)
    f.bracket(cx(0), cx(0) + CW, 84)
    f.text((cx(1) + cx(2) + CW) / 2, 72, "step s−1", size=12, weight=600)
    f.bracket(cx(1), cx(2) + CW, 84)
    f.text((cx(3) + cx(4) + CW) / 2, 72, "step s：算 stage s", size=12, weight=600)
    f.bracket(cx(3), cx(4) + CW, 84)
    for c, (_, sec) in enumerate(cols):
        f.text(cx(c) + CW / 2, 104, f"段 {sec}（mi {'0–7' if sec == 0 else '8–15'}）", size=10.5, fill=MUTED)

    for r, (name, sub) in enumerate(rows):
        f.text(14, ry(r) + 8, name, size=11, anchor="start", weight=600)
        f.text(14, ry(r) + 22, sub, size=9.5, anchor="start", fill=MUTED)

    yt, yb = RY0 - 10, ry(len(rows) - 1) + RH + 8
    for c, (_, sec) in enumerate(cols):
        x = cx(c) - 5
        if sec == 0:
            f.line(x, yt, x, yb, stroke=INK, dash="3 3")
            f.text(x, yb + 12, "①", size=11, weight=600)
        else:
            f.line(x, yt, x, yb, stroke=INK, sw=1.6)
            f.text(x, yb + 12, "② barrier", size=10.5, weight=600)

    for c, (t, sec) in enumerate(cols):
        for r, d, label in section_cells(t, sec):
            mine = d == 0
            assert text_width(label, 10) <= CW - 10, label
            f.rect(cx(c), ry(r), CW, RH, G[0][2] if mine else "#ffffff",
                   G[0][0] if mine else LINE, sw=1.3 if mine else 0.8, rx=3)
            f.text(cx(c) + CW / 2, ry(r) + RH / 2, label, size=10,
                   fill=INK if mine else MUTED, weight=600 if mine else 400)

    f.lines(14, yb + 44,
            [f"① 段 0 开始前：等本 stage 的 B、B scale 到齐（vmcnt ≤ W_START = {W_START}），"
             "afr0、asc0 也已读完（lgkmcnt(0)）",
             f"② 段 1 开始前：等 stage s+1 的 A、A scale DMA 到齐（vmcnt ≤ W_MID = {W_MID}），"
             "afr1、asc1 也已读完，再 __syncthreads()",
             f"每段 {KA} 条 load 插在第 {'、'.join(map(str, LOAD_SLOTS))} 条 MFMA 前（scale 是最后 2 条），"
             f"{NR} 条 LDS 读约每 {TS // NR} 条 MFMA 插一条",
             "格子里写的是 stage 号和写到哪"],
            size=10.5, anchor="start", step=20, fill=MUTED)
    return f


FIGS = {"stage_pipeline.svg": fig_stage_pipeline}

if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "docs", "images")
    for name, fn in FIGS.items():
        fn().save(os.path.join(out, name))
