#!/usr/bin/env python3
"""Draw the 256x256 GEMM tiling and data-movement diagrams described in mynote.md.

    python3 scripts/draw_gemm_figs.py [out_dir]     # default: the mbs directory

The labels are Chinese, so a CJK font is required: set MBS_CJK_FONT to an
.otf/.ttf file, or place NotoSansSC-Regular.otf / NotoSansSC-Bold.otf in
/tmp/fonts (https://github.com/notofonts/noto-cjk, Sans/SubsetOTF/SC).
"""
import glob
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

COL = {
    "A": "#2F6DB5", "A_l": "#D3E3F6",
    "B": "#D9730D", "B_l": "#FBDFC2",
    "S": "#7B52C1", "S_l": "#E6DCF6",
    "O": "#C0392B", "O_l": "#F7D4D0",
    "glob": "#F1F1F1", "lds": "#FFF6D5", "reg": "#E6F3E1",
    "mfma": "#22695B", "ink": "#222222", "grid": "#9A9A9A",
}
MONO = "DejaVu Sans Mono"
WHITE_BG = dict(fc="white", ec="none", pad=1.5, alpha=0.9)


def setup_font():
    paths = [os.environ.get("MBS_CJK_FONT", "")] + sorted(glob.glob("/tmp/fonts/NotoSansSC-*.otf"))
    for p in paths:
        if p and os.path.exists(p):
            fm.fontManager.addfont(p)
    cjk = [f.name for f in fm.fontManager.ttflist
           if any(k in f.name for k in ("Noto Sans SC", "Noto Sans CJK", "Source Han Sans", "WenQuanYi"))]
    if not cjk:
        sys.exit("no CJK font found (see the docstring for how to provide one)")
    plt.rcParams["font.family"] = [cjk[0], "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def canvas(fig, rect, w, h):
    ax = fig.add_axes(rect)
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.set_aspect("equal")
    ax.axis("off")
    return ax


def box(ax, x, y, w, h, fc, ec="#444444", lw=1.0, hatch=None, ls="-", z=2, rnd=0.0):
    if rnd:
        p = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={rnd}",
                           fc=fc, ec=ec, lw=lw, hatch=hatch, ls=ls, zorder=z)
    else:
        p = Rectangle((x, y), w, h, fc=fc, ec=ec, lw=lw, hatch=hatch, ls=ls, zorder=z)
    ax.add_patch(p)
    return p


def text(ax, x, y, s, size=11, ha="center", va="center", color=COL["ink"], weight="normal", z=6, **kw):
    return ax.text(x, y, s, fontsize=size, ha=ha, va=va, color=color, fontweight=weight, zorder=z, **kw)


def arrow(ax, p0, p1, color="#333333", lw=1.8, style="-|>", ms=16, ls="-", rad=0.0, z=5):
    a = FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=ms, color=color, lw=lw, ls=ls,
                        connectionstyle=f"arc3,rad={rad}", zorder=z, shrinkA=0, shrinkB=0)
    ax.add_patch(a)
    return a


def grid(ax, x, y, w, h, nx, ny, color=COL["grid"], lw=0.5, z=3):
    for i in range(1, nx):
        ax.plot([x + i * w / nx] * 2, [y, y + h], color=color, lw=lw, zorder=z)
    for j in range(1, ny):
        ax.plot([x, x + w], [y + j * h / ny] * 2, color=color, lw=lw, zorder=z)


def swatch(ax, x, y, label, fc, ec, hatch=None, ls="-", size=10):
    box(ax, x, y, 0.7, 0.42, fc, ec=ec, hatch=hatch, ls=ls, lw=1.2, z=6)
    text(ax, x + 0.9, y + 0.21, label, size=size, ha="left")


# --------------------------------------------------------------------------- tiling
def draw_tiling(path):
    fig = plt.figure(figsize=(20, 10.5))
    ax = canvas(fig, [0, 0, 1, 1], 40, 21)
    text(ax, 20, 20.25, "MBS 的 GEMM 分块：以 256×256 tile 为例（gemm_a4w4_wg_kernel<16, 4, 4>）",
         size=17, weight="bold")

    # ① the whole GEMM, one cell = 256 x 256
    text(ax, 7.0, 18.8, "① 整个 GEMM：out = A · Bᵀ", size=14, weight="bold")
    u, nk, nm, nn = 0.75, 5, 8, 8
    xA, xC, yCt = 1.5, 5.6, 12.4
    yCb, yBb = yCt - nm * u, yCt + 0.4
    ti, tj, tk = 2, 5, 1  # highlighted tile row / column, current stage
    box(ax, xC, yCb, nn * u, nm * u, "white", lw=1.2)
    grid(ax, xC, yCb, nn * u, nm * u, nn, nm)
    box(ax, xC + tj * u, yCt - (ti + 1) * u, u, u, COL["O"], ec=COL["O"], z=4)
    box(ax, xA, yCb, nk * u, nm * u, "white", lw=1.2)
    box(ax, xA, yCt - (ti + 1) * u, nk * u, u, COL["A_l"], ec=COL["A"], z=3)
    box(ax, xA + tk * u, yCt - (ti + 1) * u, u, u, COL["A"], ec=COL["A"], z=4)
    grid(ax, xA, yCb, nk * u, nm * u, nk, nm, z=5)
    box(ax, xC, yBb, nn * u, nk * u, "white", lw=1.2)
    box(ax, xC + tj * u, yBb, u, nk * u, COL["B_l"], ec=COL["B"], z=3)
    box(ax, xC + tj * u, yBb + (nk - 1 - tk) * u, u, u, COL["B"], ec=COL["B"], z=4)
    grid(ax, xC, yBb, nn * u, nk * u, nn, nk, z=5)
    text(ax, xA + nk * u / 2, yCt + 0.35, "A：M × K（FP4）", size=11)
    text(ax, xC + nn * u / 2, yBb + nk * u + 0.35, "Bᵀ：K × N（FP4）", size=11)
    text(ax, xC + nn * u / 2, yCb - 0.4, "out：M × N（bf16）", size=11)
    text(ax, xA + nk * u / 2, yCb - 0.4, "← K 方向 →", size=10, color="#555555")
    notes = [
        (COL["O"], "红块：一个工作组负责的 256×256 输出块（每个 CU 跑一个工作组）"),
        (COL["A"], "深蓝：这个输出块在当前 stage 要用的 A（256 行 × 256 个 K）"),
        (COL["B"], "深橙：当前 stage 要用的 B（256 列 × 256 个 K）"),
    ]
    for i, (c, s) in enumerate(notes):
        y = 4.9 - i * 0.8
        box(ax, 0.9, y - 0.18, 0.36, 0.36, c, ec=c, z=6)
        text(ax, 1.45, y, s, size=10.5, ha="left")
    text(ax, 0.9, 2.5, "沿 K 一个 stage（256）一个 stage 往下走，K/256 次累加完，这个输出块才算完。",
         size=10.5, ha="left")

    # ② one workgroup tile
    text(ax, 21.0, 18.8, "② 放大红块：4 个 wave 沿 N 并排", size=14, weight="bold")
    S, xT, yTt = 6.4, 19.6, 12.4
    yTb, wv = yTt - S, S / 4
    xAs, wAs = 16.6, 2.4  # A slice of this stage (K drawn compressed)
    yBs, hBs = yTt + 0.6, 2.4
    box(ax, xT, yTb, S, S, "#FBEAE7", ec=COL["O"], lw=1.5)
    for w in range(4):
        grid(ax, xT + w * wv, yTb, wv, S, 4, 16, color="#D9A8A1", lw=0.4)
        if w:
            ax.plot([xT + w * wv] * 2, [yTb, yTt], color=COL["O"], lw=1.5, zorder=4)
        text(ax, xT + (w + 0.5) * wv, yTb - 0.35, f"wave {w}", size=10.5)
        box(ax, xT + w * wv + 0.05, yBs, wv - 0.1, hBs, COL["B_l"], ec=COL["B"], lw=1.2)
        text(ax, xT + (w + 0.5) * wv, yBs + hBs / 2, f"B\n(wave {w})", size=9, color=COL["B"],
             bbox=dict(fc=COL["B_l"], ec="none", pad=1.0))
    ax.plot([xT + 0.05, xT + S - 0.05], [yBs + hBs / 2] * 2, color=COL["B"], lw=0.8, ls="--", zorder=4)
    text(ax, xT + S + 0.2, yBs + hBs * 0.75, "h=0", size=9, ha="left", color="#555555")
    text(ax, xT + S + 0.2, yBs + hBs * 0.25, "h=1", size=9, ha="left", color="#555555")
    text(ax, xT + S / 2, yBs + hBs + 0.45, "每个 wave 自己的 B：64 列 × 256 个 K = 8 KB → 直接读进寄存器", size=10.5)
    ax.plot([xT, xT + S], [yTt - S / 2] * 2, color="#333333", lw=1.8, ls="--", zorder=5)
    text(ax, xT + S + 0.25, yTt - S / 4, "段0\n行 0–127", size=10.5, ha="left")
    text(ax, xT + S + 0.25, yTb + S / 4, "段1\n行 128–255", size=10.5, ha="left")
    mi, ni, w1 = 3, 2, 1
    box(ax, xT + w1 * wv + ni * wv / 4, yTt - (mi + 1) * S / 16, wv / 4, S / 16, COL["O"], ec=COL["O"], z=6)
    box(ax, xAs, yTb, wAs, S, COL["A_l"], ec=COL["A"], lw=1.2)
    ax.plot([xAs + wAs / 2] * 2, [yTb, yTt], color=COL["A"], lw=0.8, ls="--", zorder=4)
    ax.plot([xAs, xAs + wAs], [yTt - S / 2] * 2, color="#333333", lw=1.8, ls="--", zorder=5)
    text(ax, xAs + wAs / 4, yTt + 0.25, "h=0", size=9, color="#555555")
    text(ax, xAs + wAs * 3 / 4, yTt + 0.25, "h=1", size=9, color="#555555")
    a_bg = dict(fc=COL["A_l"], ec="none", pad=1.0)
    text(ax, xAs + wAs / 2, yTt - S / 4, "A\n行 0–127", size=9.5, color=COL["A"], bbox=a_bg)
    text(ax, xAs + wAs / 2, yTb + S / 4, "A\n行 128–255", size=9.5, color=COL["A"], bbox=a_bg)
    text(ax, xAs + wAs / 2, yTb - 1.15, "当前 stage 的 A\n256 行 × 256 个 K\n= 32 KB，放进 LDS\n4 个 wave 共用",
         size=10, color=COL["A"])
    text(ax, xT + S / 2, yTb - 1.25, "每个 wave：16 行 × 4 列 = 64 个 16×16 子块\n= 64 个累加器，正好占满 256 个 AGPR",
         size=10.5)
    text(ax, xT + S / 2, yTb - 2.25, "红色小格：一个子块（放大见 ③）；h=0/1 是 K 的前后两个 128",
         size=10, color="#555555")

    # ③ one MFMA
    text(ax, 34.0, 18.8, "③ 放大一个子块：一条 MFMA", size=14, weight="bold")
    xa, ya, wa, ha = 29.4, 9.6, 4.8, 2.0  # A fragment (K drawn compressed)
    xo = xa + wa + 0.8
    yb = ya + ha + 0.8
    box(ax, xa, ya, wa, ha, COL["A_l"], ec=COL["A"], lw=1.3)
    box(ax, xo, ya, ha, ha, COL["O_l"], ec=COL["O"], lw=1.5)
    box(ax, xo, yb, ha, wa, COL["B_l"], ec=COL["B"], lw=1.3)
    for q in range(4):
        if q:
            ax.plot([xa + q * wa / 4] * 2, [ya, ya + ha], color=COL["A"], lw=0.9, zorder=4)
            ax.plot([xo, xo + ha], [yb + q * wa / 4] * 2, color=COL["B"], lw=0.9, zorder=4)
        box(ax, xa + (q + 0.5) * wa / 4 - 0.16, ya + ha + 0.08, 0.32, 0.32, COL["S"], ec=COL["S"], z=6)
        box(ax, xo - 0.4, yb + (q + 0.5) * wa / 4 - 0.16, 0.32, 0.32, COL["S"], ec=COL["S"], z=6)
        text(ax, xa + (q + 0.5) * wa / 4, ya + ha / 2, f"K {32 * q}–{32 * q + 31}", size=8.5, color=COL["A"])
        text(ax, xo + ha / 2, yb + wa - (q + 0.5) * wa / 4, f"K {32 * q}–{32 * q + 31}", size=8.5,
             color=COL["B"])
    text(ax, xa + wa / 2, ya - 0.4, "A 片段：16 行 × 128 个 FP4", size=10.5)
    text(ax, xo + ha / 2, yb + wa + 0.4, "B 片段：16 列 × 128 个 FP4", size=10.5)
    text(ax, xo + ha / 2, ya + ha / 2, "16×16\n输出", size=10.5, color=COL["O"])
    text(ax, xo + ha / 2, ya - 0.4, "累加进 acc（AGPR）", size=10.5, color=COL["O"])
    text(ax, xa + wa / 2, yb + wa / 2 - 0.2,
         "紫色小方块：每 32 个 K\n一个 E8M0 scale\nMFMA 用 op_sel\n直接挑出要用的字节",
         size=10, color=COL["S"])
    lines = [
        ("指令：", "v_mfma_scale_f32_16x16x128_f8f6f4"),
        ("64 个 lane 各拿 16 字节（32 个 FP4），一条指令算完 16×16×128", None),
        ("一个 stage 的 K=256 = 前后两个 128（h=0/1）", None),
        ("→ 每个子块每个 stage 要 2 条 MFMA", None),
        ("每个 wave 每个 stage：16 × 4 × 2 = 128 条", None),
        ("一个工作组每个 stage：512 条（33.5 MFLOP）", None),
    ]
    for i, (a, mono) in enumerate(lines):
        y = 8.0 - i * 0.72
        text(ax, 29.2, y, a, size=10.5, ha="left")
        if mono:
            text(ax, 30.4, y, mono, size=9.5, ha="left", family=MONO)

    text(ax, 20, 0.9, "每个工作组每个 stage：搬 A 32 KB + B 32 KB + scale 4 KB，做 512 条 MFMA。"
                      "图中 K 方向做了压缩，未按比例。", size=11, color="#444444")
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)


# --------------------------------------------------------------------------- data flow
def draw_dataflow(path):
    fig = plt.figure(figsize=(20, 15.5))
    top = canvas(fig, [0.0, 0.42, 1.0, 0.58], 40, 18)
    bot = canvas(fig, [0.0, 0.0, 1.0, 0.40], 40, 12.4)

    # ---- top: where the data lives and how it moves (one wave, while stage s computes)
    text(top, 20, 17.35, "MBS 的存取逻辑：第 s 个 stage 正在计算时，数据在哪、往哪搬（256×256 tile，一个 wave 的视角）",
         size=16, weight="bold")

    box(top, 0.5, 0.6, 6.6, 15.6, COL["glob"], ec="#999999", rnd=0.2, z=1)
    text(top, 3.8, 15.75, "全局显存（HBM / L2）", size=12.5, weight="bold")
    gx, gw = 0.9, 5.8
    box(top, gx, 12.9, gw, 1.6, COL["A_l"], ec=COL["A"], lw=1.3)
    text(top, gx + gw / 2, 13.7, "A：FP4，行主序\n[M, K/2]", size=11)
    box(top, gx, 11.3, gw, 1.2, COL["S_l"], ec=COL["S"], lw=1.3)
    text(top, gx + gw / 2, 11.9, "A scale：E8M0", size=11)
    box(top, gx, 7.1, gw, 1.4, COL["B_l"], ec=COL["B"], lw=1.3)
    text(top, gx + gw / 2, 7.8, "B：FP4，16×16 预重排\n[N, K/2]", size=11)
    box(top, gx, 5.6, gw, 1.2, COL["S_l"], ec=COL["S"], lw=1.3)
    text(top, gx + gw / 2, 6.2, "B scale：E8M0", size=11)
    box(top, gx, 1.0, gw, 1.6, COL["O_l"], ec=COL["O"], lw=1.3)
    text(top, gx + gw / 2, 1.8, "out：bf16\n[M, N]", size=11)

    box(top, 11.4, 8.9, 7.8, 7.3, COL["lds"], ec="#C9A227", rnd=0.2, z=1)
    text(top, 15.3, 15.75, "LDS（每个 CU 一份，4 个槽共 136 KB）", size=12.5, weight="bold")
    text(top, 14.8, 15.1, "A（32 KB）", size=9, color="#7A5C00")
    text(top, 18.2, 15.1, "scale", size=8.5, color="#7A5C00")
    slots = [
        ("stage s · 正在读", "段0 里读它的后半（段1 用）", dict(fc=COL["A_l"], ec=COL["A"]), COL["S_l"], None),
        ("stage s+1 · 段1 开始前到齐", "段1 里读它的前半（下一步用）", dict(fc=COL["A_l"], ec=COL["A"]), COL["S_l"], None),
        ("stage s+2 · 正在写入", "段1 里 DMA 写进来", dict(fc="white", ec=COL["A"], hatch="///"), "white", "///"),
        ("空闲：上一个 stage 已读完", "下一步轮到它被写", dict(fc="white", ec="#999999", ls="--"), "white", None),
    ]
    slot_y = [13.65, 12.3, 10.95, 9.6]
    for (l1, l2, st, sfc, sh), y in zip(slots, slot_y):
        box(top, 11.8, y, 6.0, 1.1, lw=1.4, z=3, **st)
        box(top, 17.8, y, 0.8, 1.1, sfc, ec=COL["S"] if st.get("ls") != "--" else "#999999",
            hatch=sh, ls=st.get("ls", "-"), lw=1.2, z=3)
        gray = st.get("ls") == "--"
        text(top, 14.8, y + 0.75, l1, size=10, weight="bold", color="#777777" if gray else COL["ink"],
             bbox=WHITE_BG if sh else None)
        text(top, 14.8, y + 0.32, l2, size=9.5, color="#777777" if gray else COL["ink"],
             bbox=WHITE_BG if sh else None)
    text(top, 15.3, 9.25, "每个 step，四个槽的角色整体轮转一格", size=9.5, color="#7A5C00")

    box(top, 22.6, 1.9, 10.8, 14.3, COL["reg"], ec="#5E9E4F", rnd=0.2, z=1)
    text(top, 28.0, 15.75, "寄存器（每个 wave 独享）", size=12.5, weight="bold")
    box(top, 23.0, 5.45, 10.0, 9.65, "white", ec="#5E9E4F", lw=0.8, rnd=0.12, z=2)
    text(top, 25.65, 14.7, "正在装（下一批）", size=10.5, weight="bold")
    text(top, 30.65, 14.7, "正在用（这一批）", size=10.5, weight="bold")
    xl, xr, bw = 23.3, 28.3, 4.7
    rows = [  # y, h, color, light color, left label, right label
        (12.9, 1.4, COL["A"], COL["A_l"], "A 片段 ×16\n下一段要用的", "A 片段 ×16\n这一段在用"),
        (11.75, 0.8, COL["S"], COL["S_l"], "A scale ×4", "A scale ×4"),
        (7.1, 1.4, COL["B"], COL["B_l"], "bf[s+1]：B 片段 ×8\n下一个 stage 的", "bf[s]：B 片段 ×8\n这一个 stage 在用"),
        (5.8, 0.8, COL["S"], COL["S_l"], "bs[s+1]：B scale", "bs[s]：B scale"),
    ]
    for y, h, c, cl, ll, rl in rows:
        box(top, xl, y, bw, h, "white", ec=c, hatch="///", lw=1.3, z=3)
        box(top, xr, y, bw, h, cl, ec=c, lw=1.3, z=3)
        text(top, xl + bw / 2, y + h / 2, ll, size=9.5, bbox=WHITE_BG)
        text(top, xr + bw / 2, y + h / 2, rl, size=9.5)
    text(top, 28.0, 11.35, "A 片段两组（afr0 / afr1，共 128 个）每段互换角色", size=9, color="#555555")
    text(top, 28.0, 9.9, "VGPR 共 256 个：A 片段 128 + B 64\n+ scale 12 + 地址偏移等", size=9.5, color="#3D6B32")
    box(top, 23.0, 2.3, 10.0, 2.7, COL["O_l"], ec=COL["O"], lw=1.3, rnd=0.12, z=2)
    text(top, 28.0, 3.65, "AGPR（256 个）\n64 个累加器 acc[16][4]：这个 wave 负责的 256×64 输出", size=10.5)

    box(top, 35.2, 7.0, 4.3, 4.6, COL["mfma"], ec=COL["mfma"], rnd=0.25, z=2)
    text(top, 37.35, 9.3, "矩阵核心\nMFMA\n每条 16×16×128", size=13, color="white", weight="bold")

    swatch(top, 34.6, 15.0, "实心：正在被读 / 在用", COL["A_l"], COL["A"])
    swatch(top, 34.6, 14.25, "斜线：正在装下一批", "white", "#555555", hatch="///")
    swatch(top, 34.6, 13.5, "虚线：空闲", "white", "#999999", ls="--")
    text(top, 34.6, 12.95, "蓝 A · 橙 B · 紫 scale · 红 输出", size=9.5, ha="left", color="#555555")

    # ① A and A scale -> LDS
    arrow(top, (gx + gw, 13.7), (11.8, 11.65), color=COL["A"], lw=2.2)
    arrow(top, (gx + gw, 11.9), (11.8, 11.25), color=COL["S"], lw=1.8)
    text(top, 9.25, 15.0, "① A、A scale → LDS", size=10.5, weight="bold", color=COL["A"])
    text(top, 9.25, 14.55, "buffer_load … lds", size=9, family=MONO)
    text(top, 9.25, 14.1, "DMA 直写，不经寄存器", size=9.5)
    text(top, 9.25, 13.68, "4 个 wave 各搬 1/4", size=9.5)
    text(top, 9.25, 13.26, "提前 2 个 stage 发出", size=9.5)

    # ② LDS -> fragment registers
    arrow(top, (18.6, 14.2), (xl, 13.75), color=COL["A"], lw=2.0)
    arrow(top, (18.6, 12.85), (xl, 13.3), color=COL["A"], lw=2.0, ls="--")
    text(top, 20.9, 14.35, "段0 时", size=9.5, color=COL["A"])
    text(top, 20.9, 12.6, "段1 时", size=9.5, color=COL["A"])
    text(top, 20.9, 11.75, "② ds_read", size=10.5, weight="bold", color=COL["A"])
    text(top, 20.9, 11.3, "b128 / b32", size=9, family=MONO)
    text(top, 20.9, 10.88, "提前一整段读", size=9.5)
    text(top, 20.9, 10.46, "下一段的 A 片段", size=9.5)

    # ③ B and B scale -> registers, passing under the LDS
    arrow(top, (gx + gw, 7.8), (xl, 7.8), color=COL["B"], lw=2.2)
    arrow(top, (gx + gw, 6.2), (xl, 6.2), color=COL["S"], lw=1.8)
    text(top, 15.0, 8.3, "③ B → 寄存器：buffer_load_dwordx4，已按 16×16 预重排，直接进寄存器（不走 LDS），提前 1 个 stage",
         size=10, color=COL["B"])
    text(top, 15.0, 5.75, "B scale：buffer_load_dword → 寄存器", size=10, color=COL["S"])

    # operands -> MFMA, accumulate in AGPR
    for y, c in ((13.6, COL["A"]), (12.15, COL["S"]), (7.8, COL["B"]), (6.2, COL["S"])):
        arrow(top, (xr + bw, y), (35.2, 9.3 + (y - 9.95) * 0.35), color=c, lw=1.6)
    text(top, 34.1, 9.6, "源操作数", size=9.5, color="#555555")
    arrow(top, (37.35, 7.0), (33.0, 3.65), color=COL["O"], lw=2.0, style="<|-|>", rad=-0.3)
    text(top, 37.4, 4.6, "acc += A·Bᵀ\n读出累加器，加上乘积，写回", size=10, color=COL["O"])

    # epilogue and ④ store
    box(top, 11.4, 1.0, 9.6, 3.6, "#FDECEA", ec=COL["O"], lw=1.3, rnd=0.2, z=2)
    text(top, 16.2, 4.05, "epilogue：这个 tile 的最后一个 stage 算完后", size=10.5, weight="bold", color=COL["O"])
    text(top, 16.2, 3.4, "AGPR → 转 bf16 → permlane16_swap + DPP", size=10)
    text(top, 16.2, 2.6, "每行拼成整条 128 B 再写，不经过 LDS", size=10)
    text(top, 16.2, 1.8, "此时下一个 tile 的前两个 stage 已在预取", size=10, color="#555555")
    arrow(top, (23.0, 3.0), (21.0, 3.0), color=COL["O"], lw=2.0)
    arrow(top, (11.4, 1.8), (gx + gw, 1.8), color=COL["O"], lw=2.2)
    text(top, 9.05, 3.2, "④ 写回", size=10.5, weight="bold", color=COL["O"])
    text(top, 9.05, 2.7, "buffer_store_dwordx4", size=9, family=MONO)
    text(top, 9.05, 1.3, "每条写 8 行 × 128 B", size=9.5)

    # ---- bottom: the order of things inside one step
    text(bot, 20, 11.85, "一个 step 内部的时间顺序（stage s；横轴 = 第几条 MFMA，一个 step 共 128 条）",
         size=14.5, weight="bold")
    x0, dx = 7.2, 0.25
    xs, xe = x0 + 64 * dx, x0 + 128 * dx
    y_m, y_l, y_v = 8.0, 6.3, 4.6
    for y, s in ((10.05, "等待 / 同步"), (y_m, "MFMA（矩阵核心）"), (y_l, "LDS 读 → VGPR"), (y_v, "全局 load（VMEM）")):
        text(bot, 6.9, y, s, size=11, ha="right", weight="bold")
    for x in (x0, xs, xe):
        bot.plot([x, x], [3.7, 10.5], color="#666666", lw=1.2, ls="--", zorder=1)
    text(bot, x0 + 0.15, 10.35, "W: s_waitcnt vmcnt(10) lgkmcnt(0)", size=9.5, ha="left", family=MONO)
    text(bot, x0 + 0.15, 9.85, "等 bf[s] 和这一段的 A 片段到齐", size=9.5, ha="left")
    text(bot, xs + 0.15, 10.35, "W + s_barrier", size=9.5, ha="left", family=MONO)
    text(bot, xs + 0.15, 9.85, "等 stage s+1 的 A 进 LDS、这一段的片段到齐；4 个 wave 在此同步", size=9.5, ha="left")
    text(bot, xe - 0.15, 10.35, "下一个 step 的 W", size=9.5, ha="right")

    for xb, fc, lab in ((x0, COL["O_l"], "段0：M 行 0–127 的 64 条 MFMA（用 afr0、bf[s]）"),
                        (xs, "#FBE7E3", "段1：M 行 128–255 的 64 条 MFMA（用 afr1、bf[s]）")):
        box(bot, xb, y_m - 0.45, 64 * dx, 0.9, fc, ec=COL["O"], lw=1.3)
        for i in range(1, 64):
            bot.plot([xb + i * dx] * 2, [y_m - 0.45, y_m + 0.45], color="white", lw=0.7, zorder=3)
        text(bot, xb + 32 * dx, y_m, lab, size=10, bbox=WHITE_BG)

    # tick positions follow `section`: read r before MFMA floor(r*64/20), load k before floor(k*56/10)
    for xb, lab in ((x0, "读 stage s 段1 的 A 片段和 scale → 装进 afr1"),
                    (xs, "读 stage s+1 段0 的 A 片段和 scale → 装进 afr0")):
        for r in range(20):
            t = r * 64 // 20
            box(bot, xb + t * dx - 0.05, y_l - 0.32, 0.1, 0.64, COL["A"] if r < 16 else COL["S"],
                ec="none", z=4)
        text(bot, xb + 32 * dx, y_l + 0.65, lab, size=9.5)
    for xb, main, lab in ((x0, COL["B"], "stage s+1 的 B（8 条）+ B scale（2 条）→ bf[s+1]"),
                          (xs, COL["A"], "stage s+2 的 A（8 条 DMA）+ A scale（2 条）→ LDS 槽 s+2")):
        for k in range(10):
            t = k * 56 // 10
            box(bot, xb + t * dx - 0.07, y_v - 0.36, 0.14, 0.72, main if k < 8 else COL["S"], ec="none", z=4)
        text(bot, xb + 32 * dx, y_v + 0.68, lab, size=9.5)
        bot.plot([xb, xb + 56 * dx], [3.95, 3.95], color="#555555", lw=1.0, zorder=3)
        for x in (xb, xb + 56 * dx):
            bot.plot([x, x], [3.85, 4.05], color="#555555", lw=1.0, zorder=3)
        text(bot, xb + 28 * dx, 3.6, "只铺在前 7/8（56 条 MFMA）里", size=9, color="#555555")

    swatch(bot, 1.0, 2.55, "A（数据 / 片段）", COL["A"], COL["A"])
    swatch(bot, 6.2, 2.55, "B", COL["B"], COL["B"])
    swatch(bot, 8.4, 2.55, "scale", COL["S"], COL["S"])
    swatch(bot, 11.6, 2.55, "一格 = 一条 MFMA", COL["O_l"], COL["O"])
    text(bot, 1.0, 1.8, "所有 load 和 LDS 读都写进“下一批”缓冲，不碰正在被 MFMA 读的寄存器；它们夹在 MFMA 之间发出，基本不占额外时间。",
         size=10.5, ha="left")
    text(bot, 1.0, 1.15, "整个 step 只等 2 次，每次仍留 10 条 load 在路上（vmcnt 按发射顺序退役），队列从不排空。",
         size=10.5, ha="left")
    text(bot, 1.0, 0.5, "实测：一个 step 约 2710 cycle；把 load 全去掉的地板约 2290 cycle（约 16.7 cycle/条 MFMA）。",
         size=10.5, ha="left")

    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
    setup_font()
    draw_tiling(os.path.join(out, "gemm_tiling.png"))
    draw_dataflow(os.path.join(out, "gemm_dataflow.png"))
    print("wrote", os.path.abspath(os.path.join(out, "gemm_tiling.png")),
          os.path.abspath(os.path.join(out, "gemm_dataflow.png")))


if __name__ == "__main__":
    main()
