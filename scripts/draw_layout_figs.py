#!/usr/bin/env python3
"""Draw the data-layout figures used by docs/kernel_intro.md (SVG, Chinese labels).

    python3 scripts/draw_layout_figs.py [out_dir]     # default: docs/images

Every lane / offset printed in the figures is computed from the same formulas the
kernel uses (flatB, flatS, a_lane, frag_at), not typed in by hand.
"""
import os
import sys
from xml.sax.saxutils import escape

FONT = ("'Noto Sans SC','Noto Sans CJK SC','Source Han Sans SC','PingFang SC',"
        "'Microsoft YaHei',sans-serif")
MONO = "Consolas,'DejaVu Sans Mono',monospace"
INK, MUTED, LINE, CELL, BOX = "#111827", "#6b7280", "#9ca3af", "#e5e7eb", "#f3f4f6"
RED, OK = "#dc2626", "#15803d"
# lane group g = lane / 16: (stroke, medium fill, light fill)
G = [("#2563eb", "#93c5fd", "#dbeafe"), ("#ea580c", "#fdba74", "#ffedd5"),
     ("#16a34a", "#86efac", "#dcfce7"), ("#9333ea", "#d8b4fe", "#f3e8ff")]
SC = ("#b45309", "#fcd34d", "#fef3c7")  # E8M0 scale


def chunk_fill(c):
    """Logical 16-byte chunk c of a 128-byte A row: hue = lane group, shade = K half."""
    return G[c % 4][1 if c < 4 else 2]


def fmt(v):
    return ("%.2f" % v).rstrip("0").rstrip(".")


def text_width(s, size, mono=False):
    if mono:
        return len(s) * size * 0.6
    w = 0.0
    for ch in s:
        if ord(ch) >= 0x2E80 or ch in "，。：；（）、“”！？—…×→":
            w += 1.0
        elif ch in "il.,:;|'!·":
            w += 0.3
        elif ch == " ":
            w += 0.27
        elif ch.isupper() or ch in "mw%":
            w += 0.68
        else:
            w += 0.56
    return w * size


class Fig:
    def __init__(self, W, H, aria):
        self.W, self.H, self.aria = W, H, aria
        self.out, self.markers = [], {}

    def _check(self, x0, y0, x1, y1, what):
        assert -0.5 <= x0 and x1 <= self.W + 0.5 and -0.5 <= y0 and y1 <= self.H + 0.5, \
            f"out of canvas: {what} ({x0:.0f},{y0:.0f})-({x1:.0f},{y1:.0f})"

    def rect(self, x, y, w, h, fill="none", stroke=None, sw=1.0, rx=0, dash=None):
        self._check(x, y, x + w, y + h, "rect")
        a = f'<rect x="{fmt(x)}" y="{fmt(y)}" width="{fmt(w)}" height="{fmt(h)}" fill="{fill}"'
        if stroke:
            a += f' stroke="{stroke}" stroke-width="{fmt(sw)}"'
        if rx:
            a += f' rx="{fmt(rx)}"'
        if dash:
            a += f' stroke-dasharray="{dash}"'
        self.out.append(a + "/>")

    def _marker(self, color):
        if color not in self.markers:
            self.markers[color] = f"arr{len(self.markers)}"
        return self.markers[color]

    def path(self, d, stroke=LINE, sw=1.0, fill="none", dash=None, arrow=False):
        a = f'<path d="{d}" fill="{fill}" stroke="{stroke}" stroke-width="{fmt(sw)}"'
        if dash:
            a += f' stroke-dasharray="{dash}"'
        if arrow:
            a += f' marker-end="url(#{self._marker(stroke)})"'
        self.out.append(a + "/>")

    def line(self, x1, y1, x2, y2, stroke=LINE, sw=1.0, dash=None, arrow=False):
        self._check(min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2), "line")
        self.path(f"M{fmt(x1)} {fmt(y1)}L{fmt(x2)} {fmt(y2)}", stroke, sw, dash=dash, arrow=arrow)

    def text(self, x, y, s, size=12, anchor="middle", fill=INK, weight=400, mono=False,
             rotate=False):
        w = text_width(s, size, mono)
        if rotate:
            self._check(x - size / 2, y - w / 2, x + size / 2, y + w / 2, s)
        else:
            x0 = {"start": x, "middle": x - w / 2, "end": x - w}[anchor]
            self._check(x0, y - size / 2, x0 + w, y + size / 2, s)
        a = (f'<text x="{fmt(x)}" y="{fmt(y)}" font-size="{fmt(size)}" fill="{fill}" '
             f'text-anchor="{anchor}" dominant-baseline="central"')
        if weight != 400:
            a += f' font-weight="{weight}"'
        if mono:
            a += f' font-family="{MONO}"'
        if rotate:
            a += f' transform="rotate(-90 {fmt(x)} {fmt(y)})"'
        self.out.append(a + f">{escape(s)}</text>")

    def lines(self, x, y, rows, size=12, step=None, **kw):
        step = step or size * 1.55
        for i, s in enumerate(rows):
            if s:
                self.text(x, y + i * step, s, size=size, **kw)

    def header(self, title, subtitle):
        self.text(14, 20, title, size=15, anchor="start", weight=600)
        self.text(14, 43, subtitle, size=12, anchor="start", fill=MUTED)

    def bracket(self, x0, x1, y, up=True, color=MUTED):
        t = -4 if up else 4
        self.path(f"M{fmt(x0)} {fmt(y - t)}L{fmt(x0)} {fmt(y)}L{fmt(x1)} {fmt(y)}L{fmt(x1)} {fmt(y - t)}",
                  stroke=color)

    def save(self, path):
        defs = "".join(
            f'<marker id="{mid}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" '
            f'markerHeight="7" orient="auto"><path d="M0 0L10 5L0 10z" fill="{col}"/></marker>'
            for col, mid in self.markers.items())
        head = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.W}" height="{self.H}" '
                f'viewBox="0 0 {self.W} {self.H}" font-family="{FONT}" role="img" '
                f'aria-label="{escape(self.aria)}">',
                f'<rect width="{self.W}" height="{self.H}" fill="#ffffff"/>']
        if defs:
            head.append(f"<defs>{defs}</defs>")
        with open(path, "w", encoding="utf-8") as fp:
            fp.write("\n".join(head + self.out + ["</svg>"]) + "\n")
        print("wrote", path)


# ----------------------------------------------------------------------------- 1. MXFP4
def fig_mxfp4():
    f = Fig(790, 505, "A 的一行在显存里：K 个 fp4 打包成 K/2 字节，每 32 个 fp4（16 字节）一块，每块在 scale "
                      "张量里有一个 E8M0 字节。放大第 0 块：字节 j 的低 4 位是第 2j 个元素，高 4 位是第 2j+1 个。"
                      "例子：字节 0xF3 = 低 4 位 0011（1.5）、高 4 位 1111（-6），scale 126 表示 0.5，"
                      "反量化得到 0.75 和 -3")
    f.header("MXFP4 在显存里的样子：以 A 的一行为例",
             "两个 fp4 打包成 1 字节；沿 K 每 32 个 fp4 共用一个 E8M0 scale，scale 放在单独的张量里")
    # row of A, cut into 32-element blocks
    y1, h1, x0 = 80, 38, 134
    blocks = [("块 0", "k 0–31"), ("块 1", "k 32–63"), ("块 2", "k 64–95"), ("块 3", "k 96–127"),
              None, ("块 K/32−1", "最后 32 个")]
    xs, x = [], x0
    for b in blocks:
        w = 56 if b is None else 100
        xs.append((x, w))
        x += w
    for (x, w), b in zip(xs, blocks):
        if b is None:
            f.text(x + w / 2, y1 + h1 / 2, "…", size=14, fill=MUTED)
            continue
        f.rect(x, y1, w, h1, G[0][2] if b[0] == "块 0" else BOX, INK)
        f.text(x + w / 2, y1 + 13, b[0], size=12, weight=600)
        f.text(x + w / 2, y1 + 28, b[1], size=10, fill=MUTED)
    f.text(x0 - 10, y1 + 12, "A 的第 m 行", anchor="end", weight=600)
    f.text(x0 - 10, y1 + 28, "共 K/2 字节", anchor="end", size=11, fill=MUTED)
    # the scale row
    ys, hs = 148, 24
    for i, ((x, w), b) in enumerate(zip(xs, blocks)):
        if b is None:
            f.text(x + w / 2, ys + hs / 2, "…", size=14, fill=MUTED)
            continue
        cx, sw_ = x + w / 2, (72 if i == 5 else 44)
        f.line(cx, y1 + h1, cx, ys, dash="3 3")
        f.rect(cx - sw_ / 2, ys, sw_, hs, SC[2], SC[0])
        f.text(cx, ys + hs / 2, "s(K/32−1)" if i == 5 else f"s{i}", size=11)
    f.text(x0 - 10, ys + 4, "scale 张量第 m 行", anchor="end", weight=600)
    f.text(x0 - 10, ys + 20, "共 K/32 字节", anchor="end", size=11, fill=MUTED)
    # zoom: block 0 -> its 16 bytes
    sx0, bw, y2, h2 = 59, 42, 228, 34
    bx, bwid = xs[0]
    for xa, xb in ((bx, sx0), (bx + bwid, sx0 + 16 * bw)):
        f.path(f"M{xa} {y1 + h1}L{xa} 184L{xb} {y2 - 14}", dash="4 3")
    f.text(sx0 - 6, y2 - 8, "字节", anchor="end", size=10, fill=MUTED)
    f.text(sx0 - 6, y2 + h2 / 2, "k", anchor="end", size=11, fill=MUTED)
    for j in range(16):
        x = sx0 + j * bw
        f.text(x + bw / 2, y2 - 8, str(j), size=10, fill=MUTED)
        f.rect(x, y2, bw / 2, h2, "#dbeafe")
        f.rect(x + bw / 2, y2, bw / 2, h2, "#bfdbfe")
        f.rect(x, y2, bw, h2, "none", INK)
        f.text(x + bw / 4, y2 + h2 / 2, str(2 * j), size=10.5)
        f.text(x + 3 * bw / 4, y2 + h2 / 2, str(2 * j + 1), size=10.5)
    f.bracket(sx0, sx0 + 16 * bw, y2 + h2 + 8, up=False)
    f.text(395, y2 + h2 + 26, "32 个 fp4 = 16 字节 = 一个 lane 一次读进来的 i32x4，"
                              "也就是 MFMA 里一个 lane 的一份操作数", size=12)
    f.text(395, y2 + h2 + 44, "每个字节：左半格是低 4 位（偶数 k），右半格是高 4 位（奇数 k）",
           size=11, fill=MUTED)

    # bit-level zoom of byte 0 and of s0
    def bits(x, y, val, fills, title):
        f.text(x, y - 34, title, size=12, anchor="start", weight=600)
        for i in range(8):
            f.text(x + 14 + 28 * i, y - 12, str(7 - i), size=9.5, fill=MUTED)
            f.rect(x + 28 * i, y, 28, 28, fills[i], INK)
            f.text(x + 14 + 28 * i, y + 14, str((val >> (7 - i)) & 1), size=13, mono=True)
    yb = 376
    bits(70, yb, 0xF3, ["#bfdbfe"] * 4 + ["#dbeafe"] * 4, "放大字节 0（例：0xF3）")
    f.text(70 + 56, yb + 44, "k = 1：1111 → −6", size=11)
    f.text(70 + 168, yb + 44, "k = 0：0011 → 1.5", size=11)
    f.text(70 + 56, yb + 62, "（高 4 位）", size=10.5, fill=MUTED)
    f.text(70 + 168, yb + 62, "（低 4 位）", size=10.5, fill=MUTED)
    f.text(70 + 112, yb + 82, "每 4 位依次是 符号 s、指数 e e、尾数 m；按位画时高位在左",
           size=10.5, fill=MUTED)
    bits(470, yb, 126, [SC[2]] * 8, "放大 s0（E8M0 = 126）")
    f.text(470 + 112, yb + 44, "2^(126 − 127) = 0.5", size=12)
    f.text(470 + 112, yb + 64, "反量化：1.5 × 0.5 = 0.75，−6 × 0.5 = −3", size=11)
    f.text(395, yb + 112, "这两个数原本是 0.7 和 −3.0：块里最大绝对值是 3，量化器选 scale = 0.5，"
                          "让 3 / 0.5 = 6 正好是 fp4 的最大值；0.7 / 0.5 = 1.4 舍入成 1.5",
           size=11, fill=MUTED)
    return f


# ----------------------------------------------------------------------------- 2. MFMA lanes
def fig_mfma():
    f = Fig(790, 640, "v_mfma_scale_f32_16x16x128_f8f6f4 的 lane 分工：A 操作数 16 行 × 4 个 32-K 块，"
                      "lane L 拿第 L%16 行第 L/16 块；B 操作数排法相同，第 L%16 行对应输出第 L%16 列；"
                      "输出 16×16，lane L 拿第 L%16 列、第 4(L/16) 到 4(L/16)+3 行，放在 acc[0..3]")
    f.header("一条 MFMA（16×16×128）里，每个 lane 拿什么、得什么",
             "格子里的数字是 lane 号，颜色是 lane 组 g = lane / 16")
    for g in range(4):
        x = 14 + 160 * g
        f.rect(x, 61, 12, 12, G[g][1], G[g][0])
        f.text(x + 18, 67, f"g = {g}：lane {16 * g}–{16 * g + 15}", size=11.5, anchor="start")
    AX, KW, H = 64, 40, 20          # A: 4 K-blocks x 16 rows
    CX, W = AX + 4 * KW + 28, 26    # C / B columns
    BY, KH = 124, 22                # B: 4 K-blocks x 16 columns
    AY = BY + 4 * KH + 26
    kl = ["k 0–31", "32–63", "64–95", "96–127"]
    # B (drawn transposed: K down, output column across)
    f.text(CX, 100, "B 操作数：排法和 A 一样。lane L 拿 B 的第 L%16 行（= 输出的第 L%16 列）、第 L/16 块",
           size=12, anchor="start")
    for n in range(16):
        f.text(CX + n * W + W / 2, BY - 8, str(n), size=9.5, fill=MUTED)
    for kb in range(4):
        f.text(CX - 8, BY + kb * KH + KH / 2, kl[kb], size=9.5, anchor="end", fill=MUTED)
        for n in range(16):
            f.rect(CX + n * W, BY + kb * KH, W, KH, G[kb][1], "#ffffff")
            f.text(CX + n * W + W / 2, BY + kb * KH + KH / 2, str(16 * kb + n), size=9.5)
    # A
    f.text(14, 118, "A 操作数", size=12.5, anchor="start", weight=600)
    f.lines(14, 138, ["16 行 × 128 个 K", "每格 32 个 fp4 = 16 B", "lane L：第 L%16 行、", "第 L/16 块"],
            size=11.5, anchor="start", step=18)
    f.text(AX - 6, AY - 10, "行", size=9.5, anchor="end", fill=MUTED)
    for kb in range(4):
        f.text(AX + kb * KW + KW / 2, AY - 10, kl[kb], size=9.5, fill=MUTED)
    for m in range(16):
        f.text(AX - 6, AY + m * H + H / 2, str(m), size=9.5, anchor="end", fill=MUTED)
        for kb in range(4):
            f.rect(AX + kb * KW, AY + m * H, KW, H, G[kb][1], "#ffffff")
            f.text(AX + kb * KW + KW / 2, AY + m * H + H / 2, str(16 * kb + m), size=10.5)
    # C
    for m in range(16):
        g = m // 4
        for n in range(16):
            f.rect(CX + n * W, AY + m * H, W, H, G[g][2], "#ffffff")
            f.text(CX + n * W + W / 2, AY + m * H + H / 2, str(16 * g + n), size=9.5)
        f.text(CX + 16 * W + 6, AY + m * H + H / 2, f"acc[{m % 4}]", size=9.5, anchor="start", fill=MUTED)
    for g in range(4):
        f.text(CX + 16 * W + 46, AY + (4 * g + 2) * H, f"lane {16 * g}–{16 * g + 15}", size=9.5,
               anchor="start", fill=G[g][0])
        if g:
            f.line(CX, AY + 4 * g * H, CX + 16 * W, AY + 4 * g * H, stroke=INK, sw=1.4)
    f.rect(CX, AY, 16 * W, 16 * H, "none", INK)
    f.rect(AX, AY, 4 * KW, 16 * H, "none", INK)
    f.rect(CX, BY, 16 * W, 4 * KH, "none", INK)
    yb = AY + 16 * H
    f.text(CX + 8 * W, yb + 18, "输出 C（16×16，fp32）：lane L 得到第 L%16 列、第 4·(L/16) 到 4·(L/16)+3 行，"
                                "放在 acc[0..3]", size=11.5)
    f.text(14, yb + 46, "scale：每个 lane 另给一个 32 位寄存器，MFMA 用 op_sel 从中挑 1 个 E8M0 字节，"
                        "作用于这个 lane 自己那一格的 32 个 fp4", size=11, anchor="start", fill=MUTED)
    f.text(14, yb + 66, "wg kernel 把 B 片段放在第一个操作数、A 片段放在第二个，算出来的是 C 的转置："
                        "lane L 拿第 L%16 行、第 4·(L/16) 到 +3 列", size=11, anchor="start", fill=MUTED)
    return f


# ----------------------------------------------------------------------------- 3. direct read
def fig_direct_read():
    f = Fig(790, 492, "左：行主序的 A 按 MFMA 的格子直接读，64 个 lane 碰到 16 条 cache line，每条只用前 64 字节，"
                      "16 行彼此隔 K/2 字节。右：预重排的 B，一条 MFMA 的操作数是连续 1 KB，8 条 cache line 全部用满，"
                      "lane L 读偏移 16L")
    f.header("同样读 1 KB：直接读行主序的 A，和读预重排的 B",
             "一条 buffer_load_dwordx4：64 个 lane 各读 16 B。小格 = 16 B，格里是读它的 lane；"
             "一整行 = 一条 128 B 的 cache line")
    CW, CH, P = 40, 15, 17
    # left: row-major A
    LX, Y0 = 64, 108
    f.text(14, 82, "A（行主序）：按 MFMA 要的格子直接读 K 0–127", size=12.5, anchor="start", weight=600)
    for m in range(16):
        y = Y0 + m * P
        f.text(LX - 6, y + CH / 2, f"行 {m}", size=10, anchor="end", fill=MUTED)
        for c in range(8):
            if c < 4:
                f.rect(LX + c * CW, y, CW, CH, G[c][1], "#ffffff")
                f.text(LX + c * CW + CW / 2, y + CH / 2, str(16 * c + m), size=10)
            else:
                f.rect(LX + c * CW, y, CW, CH, CELL, "#ffffff")
        f.rect(LX, y, 8 * CW, CH, "none", INK, sw=0.8)
    yb = Y0 + 16 * P
    f.bracket(LX, LX + 4 * CW, yb + 4, up=False)
    f.bracket(LX + 4 * CW, LX + 8 * CW, yb + 4, up=False)
    f.text(LX + 2 * CW, yb + 18, "本步要用的 64 B", size=10.5)
    f.text(LX + 6 * CW, yb + 18, "下一步（K 128–255）才用", size=10.5, fill=MUTED)
    f.lines(14, yb + 44, ["· 64 个 lane 碰到 16 条 cache line，每条只用到一半",
                          "· 16 行在显存里彼此隔 K/2 字节，只能各占一条 line"], size=11.5, anchor="start")
    # right: preshuffled B
    RX, RY = 470, 150
    f.text(410, 82, "B（预重排）：一条 MFMA 的操作数是连续的 1 KB", size=12.5, anchor="start", weight=600)
    f.text(RX - 6, RY - 12, "偏移", size=10, anchor="end", fill=MUTED)
    for i in range(8):
        y = RY + i * P
        f.text(RX - 6, y + CH / 2, str(128 * i), size=10, anchor="end", fill=MUTED)
        for j in range(8):
            L = 8 * i + j
            f.rect(RX + j * CW, y, CW, CH, G[L // 16][1], "#ffffff")
            f.text(RX + j * CW + CW / 2, y + CH / 2, str(L), size=10)
        f.rect(RX, y, 8 * CW, CH, "none", INK, sw=0.8)
    f.lines(410, RY + 8 * P + 26, ["· 8 条 cache line，全部用满", "· lane L 读偏移 16·L，地址完全连续"],
            size=11.5, anchor="start")
    f.text(14, 476, "回退用的单 wave kernel 读 A 就是左边这种方式；wg kernel 先让 A 按整行进 LDS，"
                    "再从 LDS 按 MFMA 的格子读（第 5 节）", size=11, anchor="start", fill=MUTED)
    return f


# ----------------------------------------------------------------------------- 4. B preshuffle
def flatB(n, kp, KT):
    n_tile, n_in, k_tile, rem = n >> 4, n & 15, kp >> 5, kp & 31
    return n_tile * KT * 512 + k_tile * 512 + (rem >> 4) * 256 + n_in * 16 + (kp & 15)


def fig_b_shuffle():
    K, s = 5120, 3              # any K % 1024 == 0 and any stage s: only the asserts use them
    KT = K // 64
    EX = 37                     # the lane of the worked example in §4.2

    def after(n, kp):           # offset after the shuffle, from where this 16-row group's stage s starts
        return flatB(n, 128 * s + kp, KT) - flatB(0, 128 * s, KT)

    def before(n, kp):          # offset in the plain row-major [N, K/2] B, from where row 0's stage s starts
        return n * (K // 2) + kp

    f = Fig(790, 836, "B 的 16×16 预重排。① ② 按 B 的逻辑形状画，一行是 B 的一行、K 朝右；③ 按重排后的显存地址画，"
                      "每行是连续的 128 B。① 一个 stage 里每个 wave 读自己的 "
                      "64 行 × 256 个 K = 8 KB，切成 4 组 ni × 2 段 h = 8 份 1 KB，按发射顺序 k = 2ni + h 编号。② 重排前 B 是行主序，"
                      "ni = 0 这一组的 2 KB 是 16 条 cache line，彼此隔 K/2 字节，k = 0 这条 load 碰 16 条、每条只用一半。"
                      "③ 重排后同样的 16 B 格子换了先后顺序：把 ② 按列读出来，k = 0 是连续的 1 KB，lane L 在偏移 16L，"
                      "8 条 cache line 全部用满")
    f.header("B 的 16×16 预重排：同一批 16 B 的格子，重排前后在显存里的先后顺序",
             "① ② 按 B 的逻辑形状画（一行是 B 的一行，K 朝右）；③ 按重排后的显存地址画（一行是连续的 128 B）")

    def cell(x, y, w, h, lane, fill, ex):
        f.rect(x, y, w, h, INK if ex else fill, "#ffffff")
        f.text(x + w / 2, y + h / 2 + 0.6, str(lane), size=9, fill="#ffffff" if ex else INK,
               weight=600 if ex else 400)

    # ---- ① what one wave reads in one stage: 4 groups of 16 rows x 2 halves of K
    X0, Y0, CWk, GH = 158, 100, 128, 18
    f.text(14, 74, "① 一个 stage 里每个 wave 读的 B（横向 K，纵向是 B 的行）：64 行 × 256 个 K = 8 KB，切成 8 份",
           size=12.5, anchor="start", weight=600)
    for h in range(2):
        f.text(X0 + h * CWk + CWk / 2, Y0 - 10, f"h = {h}：K {128 * h}–{128 * h + 127}", size=9.5, fill=MUTED)
    f.text(14, Y0 + 2 * GH, "wave 0", size=11, anchor="start", weight=600)
    for ni in range(4):
        y = Y0 + ni * GH
        f.text(X0 - 6, y + GH / 2, f"ni = {ni}：行 {16 * ni}–{16 * ni + 15}", size=9.5, anchor="end")
        for h in range(2):
            x = X0 + h * CWk
            for g in range(4):
                f.rect(x + g * CWk / 4, y, CWk / 4, GH, chunk_fill(4 * h + g), "#ffffff", sw=0.6)
            f.rect(x, y, CWk, GH, "none", INK, sw=2.4 if ni == 0 else 1.0)
            f.rect(x + CWk / 2 - 20, y + 3, 40, GH - 6, "#ffffff", rx=2)
            f.text(x + CWk / 2, y + GH / 2, f"k = {2 * ni + h}", size=9.5, weight=600)
    for w in range(1, 4):
        y = Y0 + 4 * GH + 6 + (w - 1) * 16
        f.rect(X0, y, 2 * CWk, 13, BOX, LINE, sw=0.6)
        f.text(X0 - 6, y + 6.5, f"wave {w}：行 {64 * w}–{64 * w + 63}", size=9, anchor="end", fill=MUTED)
    f.lines(X0 + 2 * CWk + 22, Y0 - 4, ["wave w 读第 64w 到 64w+63 行，4 个 wave 合起来 256 行",
                                        "每份 16 行 × 64 B = 1 KB：一条 MFMA 的 B 操作数，",
                                        "用一条 buffer_load_dwordx4 读进 VGPR",
                                        "k = 2·ni + h 是这 8 条 load 的发射顺序",
                                        "格子颜色 = 一行里的第几个 16 B（深浅分 h）",
                                        "粗框：ni = 0 这一组（k = 0、1），下面放大"],
            size=10.5, anchor="start", step=20)

    # ---- ② ③ the group's 2 KB in memory, before and after; one drawn row = one 128 B cache line
    MT = 270
    f.text(14, MT, "② ③ 放大 ni = 0 这一组的 2 KB：每行画一条 128 B 的 cache line，格里是读这 16 B 的 lane",
           size=12.5, anchor="start", weight=600)
    CW, P, BH = 30, 16, 12
    LX, RX, MY = 100, 478, MT + 86
    f.text(LX + 4 * CW, MT + 26, "② 重排前：行主序 [N, K/2]", size=11.5, weight=600, fill=RED)
    f.text(RX + 4 * CW, MT + 26, "③ 重排后：16×16 预重排", size=11.5, weight=600, fill=OK)
    f.text(LX + 4 * CW, MT + 44, "横向：K，往右一格地址 +16 B；纵向：B 的下一行，地址 +K/2", size=9.5, fill=MUTED)
    f.text(RX + 4 * CW, MT + 44, "按地址画：往右一格 +16 B，往下一行 +128 B", size=9.5, fill=MUTED)
    f.text(LX - 6, MY - 10, "偏移", size=9, anchor="end", fill=MUTED)
    for h in range(2):
        f.text(LX + (4 * h + 2) * CW, MY - 26, f"k = {h}", size=9.5, weight=600)
    for c in range(8):
        f.text(LX + c * CW + CW / 2, MY - 10, f"块 {c}", size=8, fill=MUTED)
    for r in range(16):
        y = MY + r * P
        f.text(14, y + BH / 2, f"行 {r}", size=9, anchor="start", fill=MUTED)
        f.text(LX - 6, y + BH / 2, "0" if r == 0 else "K/2" if r == 1 else f"{r}·K/2", size=9, anchor="end",
               fill=MUTED)
        for c in range(8):
            h, g = c // 4, c % 4
            L = 16 * g + r
            assert before(r, 64 * h + 16 * g) == r * (K // 2) + 16 * c
            cell(LX + c * CW, y, CW, BH, L, chunk_fill(c), h == 0 and L == EX)
        f.rect(LX, y, 8 * CW, BH, "none", INK, sw=0.8)
        f.rect(LX, y, 4 * CW, BH, "none", INK, sw=2.0)
    # reading ② down its columns gives ③'s order
    f.rect(LX - 2, MY - 2, CW + 4, 15 * P + BH + 4, "none", OK, sw=1.8, dash="4 3")
    f.line(LX + CW / 2, MY + 16 * P + 2, LX + CW / 2, MY + 16 * P + 14, stroke=OK, sw=1.8, arrow=True)
    f.text(RX - 6, MY - 10, "偏移", size=9, anchor="end", fill=MUTED)
    f.text(RX - 40, MY - 10, "哪一块", size=9, anchor="end", fill=MUTED)
    for c in range(8):          # after the shuffle, every 2 lines (256 B) hold one K block of all 16 rows
        assert after(0, 16 * c) // 128 == 2 * c
        f.text(RX - 40, MY + (2 * c + 1) * P, f"块 {c}", size=9, anchor="end", fill=MUTED)
    for i in range(16):
        y = MY + i * P
        f.text(RX - 6, y + P / 2, str(128 * i), size=9, anchor="end", fill=MUTED)
        for j in range(8):
            o = 128 * i + 16 * j
            h, g, r = o // 1024, o // 256 % 4, o // 16 % 16
            L = 16 * g + r
            assert L == o // 16 % 64 and after(r, 64 * h + 16 * g) == o
            cell(RX + j * CW, y, CW, P, L, chunk_fill(4 * h + g), h == 0 and L == EX)
    f.rect(RX, MY, 8 * CW, 16 * P, "none", INK)
    f.rect(RX, MY, 8 * CW, 8 * P, "none", INK, sw=2.4)
    for h in range(2):
        x0, y0, y1 = RX + 8 * CW + 6, MY + 8 * h * P + 2, MY + 8 * (h + 1) * P - 2
        f.path(f"M{fmt(x0)} {fmt(y0)}L{fmt(x0 + 4)} {fmt(y0)}L{fmt(x0 + 4)} {fmt(y1)}L{fmt(x0)} {fmt(y1)}",
               stroke=MUTED)
        f.text(x0 + 10, (y0 + y1) / 2, f"k = {h}", size=10, anchor="start", weight=600)
    NY = MY + 16 * P + 30
    f.text(14, NY, "k = 0（粗框）：碰 16 条 cache line，每条只用一半", size=11, anchor="start", weight=600, fill=RED)
    f.lines(14, NY + 19, ["k = 1 再把这 16 条读一遍，用另一半",
                          "相邻两行隔 K/2 字节（K = 5120 时 2560 B）"], size=10.5, anchor="start", step=18,
            fill=MUTED)
    f.text(14, NY + 60, "绿虚框 ↓：把 ② 按列往下读（0、1、2、…、15，再 16、17、…），", size=10.5, anchor="start",
           fill=OK)
    f.text(14, NY + 78, "就是 ③ 里的先后顺序", size=10.5, anchor="start", fill=OK)
    f.text(RX - 50, NY, "k = 0：连续的 1 KB，8 条 cache line 全部用满", size=11, anchor="start", weight=600,
           fill=OK)
    f.lines(RX - 50, NY + 19, ["lane L 在偏移 16·L；k = 1 紧跟着是下一个 1 KB",
                               "16 B 的格子里的内容没变，只是格子的先后换了"], size=10.5, anchor="start", step=18,
            fill=MUTED)

    # ---- the "transpose", as offsets
    TY = NY + 112
    f.text(14, TY, "所谓“转置”：每组 16 行里，B 第 n 行的第 c 格（16 B）放在哪", size=12, anchor="start", weight=600)
    f.text(14, TY + 22, "重排前：n·K/2 + 16·c", size=11, anchor="start", fill=RED)
    f.text(250, TY + 22, "先排完一行的所有格，再排下一行（n 在外层，c 在内层）", size=10.5, anchor="start",
           fill=MUTED)
    f.text(14, TY + 42, "重排后：(n/16)·8·K + 256·c + 16·(n%16)", size=11, anchor="start", fill=OK)
    f.text(250, TY + 42, "先排完一格的 16 行，再排下一格（c 在外层，n 在内层）", size=10.5, anchor="start",
           fill=MUTED)
    g_ex, r_ex = EX // 16, EX % 16
    off_before = before(r_ex, 16 * g_ex) - r_ex * (K // 2)
    assert (off_before, after(r_ex, 16 * g_ex)) == (32, 16 * EX)
    f.text(14, TY + 64, f"深色格是 4.2 节举例的 lane {EX}：重排前在 B 的第 {r_ex} 行、这个 stage 里的第 {off_before} 字节；"
                        f"重排后在这 1 KB 的第 {16 * EX} 字节", size=10.5, anchor="start", fill=MUTED)
    return f

def fig_b_rule():
    K = 512                     # a row is 256 B = 16 cells (two stages); the 16-row group is 4 KB
    C, KT = K // 32, K // 64
    EXN, EXC = 5, 2             # example cell: row 5, cell 2
    f = Fig(790, 848, "16 行一组怎么重排，以 K = 512 为例：一行 256 B = 16 格，16 行共 4 KB。每格 16 B，写着行.格，"
                      "颜色表示第几格；画的每一行是连续的 256 B。重排前一行就是 B 的一行，同一格的 16 行在一列里、"
                      "上下相隔 256 B。重排后一行是一个小块：第 c 格的 16 行。两张格子图互为转置。"
                      "例：第 5 行第 2 格，重排前在 1312，重排后在 592")
    f.header("16 行一组怎么重排：以 K = 512 为例（一行 256 B = 16 格，16 行共 4 KB）",
             "每格 16 B（32 个 K），格里写“行.格”，颜色 = 第几格。画的每一行是连续的 256 B，左边是起始地址")
    CW, CH, X0 = 40, 17, 110

    def grid(y0, slot_of, label_of, frame):
        f.text(X0 - 6, y0 - 8, "地址", size=8.5, anchor="end", fill=MUTED)
        for i in range(16):
            f.text(X0 + i * CW + CW / 2, y0 - 8, f"+{16 * i}", size=8, fill=MUTED)
        for line in range(16):
            y = y0 + line * CH
            f.text(14, y + CH / 2, label_of(line), size=9, anchor="start", fill=MUTED)
            f.text(X0 - 6, y + CH / 2, str(256 * line), size=9, anchor="end", fill=MUTED)
            for i in range(16):
                n, c = slot_of(16 * line + i)
                ex = (n, c) == (EXN, EXC)
                f.rect(X0 + i * CW, y, CW, CH, INK if ex else chunk_fill(c % 8), "#ffffff")
                f.text(X0 + i * CW + CW / 2, y + CH / 2, f"{n}.{c}", size=9, fill="#ffffff" if ex else INK,
                       weight=600 if ex else 400)
        f.rect(X0, y0, 16 * CW, 16 * CH, "none", INK)
        f.rect(*frame, "none", INK, sw=2.4)

    # ① before: line n is row n, cell (n, c) at n*K/2 + 16c
    def before(slot):
        n, c = slot // C, slot % C
        assert n * (K // 2) + 16 * c == 16 * slot
        return n, c

    Y1 = 124
    f.text(14, 76, "① 重排前（行主序）：画的一行就是 B 的一行，16 格连续放", size=12.5, anchor="start", weight=600,
           fill=RED)
    for s in range(2):
        f.text(X0 + (8 * s + 4) * CW, 96, f"stage {s}：K {256 * s}–{256 * s + 255}", size=9.5, fill=MUTED)
        f.bracket(X0 + 8 * s * CW + 2, X0 + 8 * (s + 1) * CW - 2, 104)
    grid(Y1, before, lambda line: f"行 {line}", (X0, Y1, CW, 16 * CH))
    f.line(X0 + 8 * CW, Y1, X0 + 8 * CW, Y1 + 16 * CH, stroke=INK, sw=1.6)
    f.text(14, Y1 + 16 * CH + 18, "粗框是第 0 格（K 0–31）的 16 个：在同一列，上下相隔 256 B", size=10.5, anchor="start",
           fill=MUTED)

    # ② after: line c is block c = cell c of all 16 rows, cell (n, c) at 256c + 16n
    def after(slot):
        n, c = slot % 16, slot // 16
        assert flatB(n, 16 * c, KT) == 16 * slot
        return n, c

    Y2 = Y1 + 16 * CH + 74
    f.text(14, Y2 - 30, "② 重排后：画的一行是一个小块，即第 c 格的 16 行；小块一个接一个放", size=12.5, anchor="start",
           weight=600, fill=OK)
    grid(Y2, after, lambda line: f"块 {line}", (X0, Y2, 16 * CW, CH))
    f.line(X0, Y2 + 8 * CH, X0 + 16 * CW, Y2 + 8 * CH, stroke=INK, sw=1.6)
    for s in range(2):
        f.text(X0 + 16 * CW + 14, Y2 + (8 * s + 4) * CH, f"stage {s}", size=9.5, rotate=True, fill=MUTED)
    f.text(14, Y2 + 16 * CH + 18, "粗框是块 0：第 0 格的 16 行连成 256 B。块 0–7 是 stage 0，块 8–15 是 stage 1",
           size=10.5, anchor="start", fill=MUTED)

    before_off, after_off = EXN * (K // 2) + 16 * EXC, flatB(EXN, 16 * EXC, KT)
    assert (before_off, after_off) == (1312, 592)
    NY = Y2 + 16 * CH + 44
    f.text(14, NY, "① 和 ② 互为转置：① 第 n 行第 c 格的 16 B，原样搬到 ② 的第 c 行第 n 格", size=11, anchor="start",
           weight=600)
    f.text(14, NY + 22, f"深色格：第 {EXN} 行第 {EXC} 格。重排前在 {EXN}·256 + {EXC}·16 = {before_off}，"
                        f"重排后在 {EXC}·256 + {EXN}·16 = {after_off}", size=10.5, anchor="start")
    f.text(14, NY + 42, "K 更大时规则一样：重排前一行 K/2 字节；重排后这一组有 K/32 个小块，第 s 个 stage 是第 8s 到 8s+7 块",
           size=10.5, anchor="start", fill=MUTED)
    return f


def fig_b_blocks():
    N, K = 256, 512
    NG, C, KT = N // 16, K // 32, K // 64
    f = Fig(790, 640, "整个 B 怎么排。每 16 行 × 32 个 K 是一个 256 B 的小块。① N = 256、K = 512 的 B 切成 16 组 × 16 格 = "
                      "256 个小块，格里是小块重排后的序号，地址 = 序号 × 256：序号先沿 K 增加，一组排完再排下一组。"
                      "粗框是 wave 0 在 stage 0 读的 4 组 × 8 块。② 一般的 N × K：小块 (g, c) 在 g·8K + 256c，"
                      "排成的形状是 [N/16, K/32, 16, 16] 字节，重排前是 [N/16, 16, K/32, 16]")
    f.header("整个 B 怎么排：每 16 行 × 32 个 K 是一个 256 B 的小块，小块先沿 K 排、再沿 N 排",
             "① 以 N = 256、K = 512 为例，格里是小块在重排后的序号，地址 = 序号 × 256；② 一般的 N × K")
    BW, BH, GAP, X0, Y0 = 26, 17, 4, 116, 124
    f.text(14, 76, "① N = 256、K = 512：16 组 × 16 格 = 256 个小块，共 64 KB", size=12.5, anchor="start",
           weight=600)
    for s in range(2):
        f.text(X0 + (8 * s + 4) * BW, 96, f"stage {s}：K {256 * s}–{256 * s + 255}", size=9.5, fill=MUTED)
        f.bracket(X0 + 8 * s * BW + 2, X0 + 8 * (s + 1) * BW - 2, 104)
    for c in range(C):
        f.text(X0 + c * BW + BW / 2, Y0 - 8, f"c{c}", size=8, fill=MUTED)

    def gy(g):
        return Y0 + g * BH + (g // 4) * GAP

    for g in range(NG):
        w, y = g // 4, gy(g)
        f.text(X0 - 6, y + BH / 2, f"行 {16 * g}–{16 * g + 15}", size=8.5, anchor="end", fill=MUTED)
        for c in range(C):
            idx = g * C + c
            assert flatB(16 * g, 16 * c, KT) == 256 * idx
            hot = w == 0 and c < 8
            f.rect(X0 + c * BW, y, BW, BH, G[w][1] if hot else G[w][2], "#ffffff", sw=0.6)
            f.text(X0 + c * BW + BW / 2, y + BH / 2, str(idx), size=8, weight=600 if hot else 400)
        if g % 4 == 0:
            f.text(14, (gy(g) + gy(g + 3) + BH) / 2, f"wave {w}", size=10, anchor="start", weight=600,
                   fill=G[w][0])
    f.rect(X0, Y0, C * BW, gy(NG - 1) + BH - Y0, "none", INK)
    f.rect(X0, Y0, 8 * BW, gy(3) + BH - Y0, "none", INK, sw=2.4)
    f.line(X0 + 8 * BW, Y0, X0 + 8 * BW, gy(NG - 1) + BH, stroke=INK, sw=1.4)
    f.lines(X0 + C * BW + 22, Y0 + 2, ["序号沿行增加：先排完一组的 16 个小块",
                                       "（4 KB = 16 × K/2），再排下一组",
                                       "块 (g, c) 的序号是 16g + c，地址 × 256",
                                       "",
                                       "每个小块：16 行 × 16 B，第 r 行在 +16·r",
                                       "（图 5 ② 的一行）",
                                       "",
                                       "粗框：wave 0 在 stage 0 读的 4 组 × 8 块",
                                       "每组 8 块连续 2 KB，组与组相隔 4 KB",
                                       "8 条 load，每条是同一组里相邻 4 块 = 1 KB"],
            size=10.5, anchor="start", step=19)

    # ---- ② general N x K
    PY = gy(NG - 1) + BH + 40
    f.text(14, PY, "② 一般的 N × K：B 被切成 N/16 组 × K/32 格的小块网格", size=12.5, anchor="start", weight=600)
    SX, SY, SW, SH = 116, PY + 40, 74, 24
    cols = ["0", "1", "2", "…", "K/32−1"]
    rows = ["0", "1", "…", "N/16−1"]
    for j, cl in enumerate(cols):
        f.text(SX + j * SW + SW / 2, SY - 9, f"c = {cl}", size=9, fill=MUTED)
    for i, rl in enumerate(rows):
        y = SY + i * SH
        f.text(SX - 6, y + SH / 2, f"g = {rl}", size=9, anchor="end", fill=MUTED)
        for j, cl in enumerate(cols):
            x = SX + j * SW
            if rl == "…" or cl == "…":
                f.text(x + SW / 2, y + SH / 2, "…", size=11, fill=MUTED)
                continue
            f.rect(x, y, SW, SH, BOX, LINE, sw=0.8)
            f.text(x + SW / 2, y + SH / 2, f"({rl}, {cl})", size=9)
    f.lines(SX + 5 * SW + 24, SY + 2, ["小块 (g, c)：第 16g–16g+15 行 × 第 32c–32c+31 个 K",
                                       "地址 = g·8·K + 256·c，块内第 r 行在 +16·r",
                                       "先沿 c（K）排完一组，再排下一组 g"],
            size=10.5, anchor="start", step=20)
    f.text(14, SY + 4 * SH + 26, "所以 [N, K/2] 的字节被排成 [N/16, K/32, 16, 16]：前两维是小块网格，后两维是块内 16 行 × 16 B",
           size=11, anchor="start", weight=600)
    f.text(14, SY + 4 * SH + 48, "重排前同样的字节是 [N/16, 16, K/32, 16]（先行后格），shuffle 只是把中间两维交换",
           size=11, anchor="start")
    return f


def fig_load_vars():
    K = 512                                    # a B row is 256 B = 16 cells of 16 B
    KT, NC = K // 64, K // 32
    EXL, EXKC, NSUB = 37, 1, 4
    exr, exg = EXL & 15, EXL >> 4
    ex_cell = 4 * EXKC + exg
    ex_ktile, ex_kpack = 2 * EXKC + (exg >> 1), exg & 1
    ex_kp = ex_ktile * 32 + ex_kpack * 16
    assert ex_kp == EXKC * 64 + exg * 16 == 16 * ex_cell

    f = Fig(860, 786, "单 wave kernel 的 load 里各变量指向哪些数据，以 K = 512 为例。① 一行 B 上的刻度：kc、stage 和 h、"
                      "g、kblk、ktile、kpack；② 一步里一个 16 行子块是 16 行 × 64 B，lane L 读第 L%16 行的第 L/16 块；"
                      "③ 同一个 lane 在每个子块里都读第 row16 行")
    f.header("单 wave kernel 的 load：每个变量指向哪些数据（以 K = 512 为例）",
             f"B 的一行 256 B，一小格 16 B（32 个 K）。深色是 lane {EXL}（row16 = {exr}，g = {exg}）"
             f"在第 kc = {EXKC} 步读的 16 B")

    # ① one B row with every K-direction variable drawn as a ruler
    X0, CW, RH = 244, 37, 24
    f.text(14, 78, "① 一行 B 上的几种刻度：横向是 K，往右一小格 +16 B", size=12.5, anchor="start", weight=600)
    y = 102
    f.text(X0 - 8, y, "行内字节 kp", size=9.5, anchor="end", fill=MUTED)
    for i in range(NC + 1):
        hit = 16 * i == ex_kp
        f.text(X0 + i * CW, y, str(16 * i), size=9 if hit else 8, fill=INK if hit else MUTED,
               weight=700 if hit else 400)
    y += 12
    alt = lambda i: BOX if i % 2 == 0 else CELL
    rows = [
        ("kc：第几步（64 B，128 个 K）", 4, lambda c: f"kc = {c // 4}", lambda c: alt(c // 4)),
        ("wg kernel 里：stage = kc/2，h = kc%2", 4, lambda c: f"stage {c // 8} · h {c // 4 % 2}",
         lambda c: alt(c // 4)),
        ("g = lane/16：这一步里第几块", 1, lambda c: str(c % 4), lambda c: G[c % 4][2]),
        ("kblk = 4·kc + g：第几个 32-K 块", 1, lambda c: str(c), lambda c: G[c % 4][2]),
        ("ktile = 2·kc + g/2：第几个 K tile（32 B）", 2, lambda c: str(c // 2), lambda c: alt(c // 2)),
        ("kpack = g & 1：K tile 里第几格", 1, lambda c: str(c % 2), lambda c: alt(c)),
    ]
    for label, span, txt, col in rows:
        f.text(X0 - 8, y + RH / 2, label, size=10, anchor="end")
        for c0 in range(0, NC, span):
            hit = c0 <= ex_cell < c0 + span
            dark = hit and span == 1
            fill = INK if dark else (G[exg][1] if hit else col(c0))
            f.rect(X0 + c0 * CW, y, span * CW, RH, fill, "#ffffff")
            f.text(X0 + (c0 + span / 2) * CW, y + RH / 2, txt(c0), size=9.5,
                   fill="#ffffff" if dark else INK, weight=600 if hit else 400)
        y += RH + 6
    f.text(14, y + 8, f"lane {EXL} 在 kc = {EXKC}：kpstart = ktile·32 + kpack·16 = {ex_ktile}·32 + {ex_kpack}·16 = {ex_kp}，"
                      f"也就是 kc·64 + g·16 = {64 * EXKC} + {16 * exg} = {ex_kp}；A 的 aoff 是同一个值",
           size=10.5, anchor="start")
    f.text(14, y + 28, f"kblk = {ex_cell} 用来查这 32 个 K 的 scale；ktile、kpack 就是 4.1 节说的“K tile”和“格”，"
                      f"flatB 会把 kpstart 再拆成这两位", size=10.5, anchor="start", fill=MUTED)

    # ② one step of one 16-row subtile: lane L -> (row16, g)
    Y2 = y + 70
    GX, GW, GH = 74, 46, 16
    f.text(14, Y2, "② 一步里一个子块：16 行 × 64 B（1 KB）", size=12.5, anchor="start", weight=600)
    f.text(14, Y2 + 20, "格里是 lane 号：lane L 读第 row16 = L%16 行、", size=10.5, anchor="start", fill=MUTED)
    f.text(14, Y2 + 38, "第 g = L/16 块（16 B）", size=10.5, anchor="start", fill=MUTED)
    gy = Y2 + 70
    for gg in range(4):
        f.text(GX + gg * GW + GW / 2, gy - 9, f"g = {gg}", size=9, fill=MUTED)
    for r in range(16):
        f.text(GX - 6, gy + r * GH + GH / 2, f"行 {r}", size=8.5, anchor="end", fill=MUTED)
        for gg in range(4):
            L = 16 * gg + r
            hit = L == EXL
            f.rect(GX + gg * GW, gy + r * GH, GW, GH, INK if hit else G[gg][2], "#ffffff")
            f.text(GX + gg * GW + GW / 2, gy + r * GH + GH / 2, str(L), size=8.5,
                   fill="#ffffff" if hit else INK, weight=600 if hit else 400)
    f.rect(GX, gy, 4 * GW, 16 * GH, "none", INK)

    # ③ the wave's B rows: NSUB subtiles of 16 rows; the same lane reads row16 of each
    BX, RW, BW = 400, 4, 64
    f.text(BX - 20, Y2, f"③ 行方向：tile 里的 NSUB 个子块（这里 NSUB = {NSUB}）", size=12.5, anchor="start", weight=600)
    f.text(BX - 20, Y2 + 20, "同一个 lane 在每个子块里都读第 row16 行：", size=10.5, anchor="start", fill=MUTED)
    f.text(BX - 20, Y2 + 38, "bn = n0 + 16·ni + row16", size=10.5, anchor="start", fill=MUTED)
    f.text(BX + BW / 2, gy - 9, "B 的行", size=9, fill=MUTED)
    for ni in range(NSUB):
        y0 = gy + ni * 16 * RW
        f.rect(BX, y0, BW, 16 * RW, G[ni][2], INK)
        f.rect(BX, y0 + exr * RW, BW, RW, INK)
        f.text(BX + BW + 10, y0 + 16 * RW / 2 - 8, f"ni = {ni}：行 n0+{16 * ni} … n0+{16 * ni + 15}", size=10,
               anchor="start")
        f.text(BX + BW + 10, y0 + 16 * RW / 2 + 9, f"lane {EXL} 读 n0+{16 * ni + exr}（深色）", size=9.5,
               anchor="start", fill=MUTED)
    f.text(BX - 20, gy + 64 * RW + 20, "n0 = blockIdx.x·16·NSUB；A 同理：am = m0 + 16·mi + row16，", size=10.5,
           anchor="start")
    f.text(BX - 20, gy + 64 * RW + 40, "m0 = blockIdx.y·16·MSUB", size=10.5, anchor="start")

    n0, ni = 64, 2
    bn = n0 + 16 * ni + exr
    a = flatB(bn, ex_kp, KT)
    assert a == (bn >> 4) * 8 * K + EXKC * 1024 + 16 * EXL
    f.text(14, gy + 64 * RW + 76, f"合起来：flatB(bn, kpstart) = (bn/16)·8K + kc·1024 + 16·lane。例：n0 = {n0}、ni = {ni} → "
                                  f"bn = {bn}，地址 = {bn >> 4}·{8 * K} + {EXKC}·1024 + 16·{EXL} = {a}",
           size=10.5, anchor="start", weight=600)
    return f

# ----------------------------------------------------------------------------- 5. A via LDS
# ds_read_b128 on gfx950 runs in 4 phases of 16 lanes (AMD CDNA4 GEMM blog, "LDS Access")
PHASES = [[*range(0, 4), *range(12, 16), *range(20, 24), *range(24, 28)],
          [*range(32, 36), *range(44, 48), *range(52, 56), *range(56, 60)],
          [*range(4, 8), *range(8, 12), *range(16, 20), *range(28, 32)],
          [*range(36, 40), *range(40, 44), *range(48, 52), *range(60, 64)]]
PHASE_OF = {L: p for p, lanes in enumerate(PHASES) for L in lanes}
assert sorted(PHASE_OF) == list(range(64))
PH = [("#be123c", "#fecdd3"), ("#a16207", "#fde68a"), ("#0e7490", "#a5f3fc"), ("#4338ca", "#c7d2fe")]


def bank_group(addr):
    """Group q (banks 4q .. 4q+3) hit by a 16-byte-aligned LDS read at addr."""
    return addr // 4 % 64 // 4


def frag_addr(L, h, swz):
    """frag_at(lds, h, mi = 0): LDS offset of lane L's 16 bytes, with or without the XOR swizzle."""
    row, c = L & 15, h * 4 + (L >> 4)
    return row * 128 + ((c ^ (row & 7)) if swz else c) * 16


def lanes_str(lanes):
    return "、".join(f"{lanes[i]}–{lanes[i + 3]}" for i in range(0, 16, 4))


def fig_lds_read():
    f = Fig(790, 792, "ds_read_b128 在 gfx950 上分 4 拍，每拍 16 个 lane；一个 lane 读 16 B，占 4 个相邻 bank。"
                      "一拍的 16 个 lane 落在 16 组不同的 bank 上就不冲突。连续读 16L 时每拍正好铺满 64 个 bank。"
                      "A 的片段读：lane L 读第 L%16 行的逻辑块 L/16，不 swizzle 时每拍只用到 4 组 bank、每组 4 个 lane，"
                      "4 路冲突；物理格 = 逻辑块 ^ (行 & 7) 之后每拍 16 组各 1 个，不冲突")
    f.header("ds_read_b128 在 64 个 bank 上怎么取数：分 4 拍，每拍 16 个 lane",
             "bank = (地址 / 4) % 64，每个 bank 4 B。一个 lane 读 16 B，占 4 个相邻的 bank（下面叫一组，共 16 组）")
    CW, CH, X0 = 26, 18, 76
    # ---- ① which lanes share a phase
    Y0 = 96
    f.text(14, 76, "① 硬件把 64 个 lane 分成 4 拍（颜色 = 第几拍）", size=12.5, anchor="start", weight=600)
    for i in range(4):
        f.text(X0 - 6, Y0 + i * CH + CH / 2, f"lane {16 * i}–{16 * i + 15}", size=9, anchor="end", fill=MUTED)
        for j in range(16):
            L = 16 * i + j
            f.rect(X0 + j * CW, Y0 + i * CH, CW, CH, PH[PHASE_OF[L]][1], "#ffffff")
            f.text(X0 + j * CW + CW / 2, Y0 + i * CH + CH / 2, str(L), size=9.5)
    f.rect(X0, Y0, 16 * CW, 4 * CH, "none", INK)
    for p in range(4):
        y = Y0 + p * CH + CH / 2
        f.rect(510, y - 6, 12, 12, PH[p][1], PH[p][0])
        f.text(528, y, f"第 {p + 1} 拍：{lanes_str(PHASES[p])}", size=10.5, anchor="start")
    # ---- ② the simplest pattern: lane L reads 16*L
    Y1 = 222
    f.text(14, Y1 - 30, "② 最简单的读法：lane L 读连续的 16·L，落在第 L%16 组——每拍正好铺满 16 组", size=12.5,
           anchor="start", weight=600)
    f.text(X0 - 6, Y1 - 9, "bank", size=8.5, anchor="end", fill=MUTED)
    for q in range(16):
        f.text(X0 + q * CW + CW / 2, Y1 - 9, f"{4 * q}–{4 * q + 3}", size=7.5, fill=MUTED)
    for p in range(4):
        y = Y1 + p * CH
        f.text(X0 - 6, y + CH / 2, f"第 {p + 1} 拍", size=9, anchor="end", fill=PH[p][0])
        at = {}
        for L in PHASES[p]:
            q = bank_group(16 * L)
            assert q == L % 16 and q not in at
            at[q] = L
        for q in range(16):
            f.rect(X0 + q * CW, y, CW, CH, PH[p][1], "#ffffff")
            f.text(X0 + q * CW + CW / 2, y + CH / 2, str(at[q]), size=9.5)
    f.rect(X0, Y1, 16 * CW, 4 * CH, "none", INK)
    f.lines(510, Y1 + 6, ["格里是读这一组 bank 的 lane", "每拍 16 个 lane 各占一组，", "64 个 bank 各用一次：1 个周期",
                          "4 拍共 4 个周期读完 1 KB"], size=10.5, anchor="start", step=17)
    # ---- ③ the A fragment read, one line per A row, x axis = bank
    Y2, GW, RH = 390, 21, 14
    GY = Y2 + 28
    f.text(14, Y2 - 44, "③ A 的片段读（h = 0、mi = 0）：lane L 读第 L%16 行的逻辑块 L/16（frag_at）", size=12.5,
           anchor="start", weight=600)
    f.text(14, Y2 - 24, "A 的一行 128 B 占半个 bank 行：第 r 行在第 r/2 个 bank 行，偶数行在 bank 0–31，"
                        "奇数行在 bank 32–63。下面每行 A 单独画一行，横轴是 bank", size=10.5, anchor="start", fill=MUTED)
    for PX, swz, title in ((46, False, "不做 swizzle：物理格 = 逻辑块"),
                           (432, True, "XOR swizzle：物理格 = 逻辑块 ^ (行 & 7)")):
        f.text(PX + 8 * GW, Y2, title, size=11.5, weight=600)
        f.text(PX - 4, GY - 8, "行", size=8.5, anchor="end", fill=MUTED)
        for q in range(16):
            f.text(PX + q * GW + GW / 2, GY - 8, str(4 * q), size=7.5, fill=MUTED)
        for r in range(16):
            f.text(PX - 4, GY + r * RH + RH / 2, str(r), size=8.5, anchor="end", fill=MUTED)
            for p in range(8):
                q, c = (r % 2) * 8 + p, (p ^ (r & 7)) if swz else p
                x, y = PX + q * GW, GY + r * RH
                assert bank_group(r * 128 + p * 16) == q
                if c < 4:       # read by this instruction (h = 0)
                    L = 16 * c + r
                    assert frag_addr(L, 0, swz) == r * 128 + p * 16
                    f.rect(x, y, GW, RH, PH[PHASE_OF[L]][1], "#ffffff", sw=0.6)
                    f.text(x + GW / 2, y + RH / 2, str(L), size=8.5, weight=600)
                else:
                    f.rect(x, y, GW, RH, BOX, "#ffffff", sw=0.6)
                    f.text(x + GW / 2, y + RH / 2, str(c), size=7.5, fill=LINE)
        f.rect(PX, GY, 16 * GW, 16 * RH, "none", INK)
        f.line(PX + 8 * GW, GY, PX + 8 * GW, GY + 16 * RH, stroke=INK, sw=1.2)
        p21 = (1 ^ 5) if swz else 1           # lane 21: row 5, logical chunk 1
        f.rect(PX + (8 + p21) * GW, GY + 5 * RH, GW, RH, "none", INK, sw=2.2)
        # per phase: how many lanes hit each bank group
        TY = GY + 16 * RH + 8
        for ph in range(4):
            for h in (0, 1):
                cnt = [0] * 16
                for L in PHASES[ph]:
                    cnt[bank_group(frag_addr(L, h, swz))] += 1
                assert cnt == [1] * 16 if swz else sorted(cnt) == [0] * 12 + [4] * 4
            cnt = [0] * 16
            for L in PHASES[ph]:
                cnt[bank_group(frag_addr(L, 0, swz))] += 1
            f.text(PX - 4, TY + ph * 11 + 5.5, f"第{ph + 1}拍", size=8, anchor="end", fill=PH[ph][0])
            for q in range(16):
                col = RED if cnt[q] > 1 else (OK if cnt[q] == 1 else LINE)
                f.text(PX + q * GW + GW / 2, TY + ph * 11 + 5.5, str(cnt[q]), size=8.5, fill=col, weight=600)
        f.text(PX + 8 * GW, TY + 58, "每拍 16 组各 1 个 lane：不冲突，4 拍共 4 个周期" if swz
               else "每拍只用 4 组、每组 4 个 lane：4 路冲突，4 拍共 16 个周期",
               size=10.5, fill=OK if swz else RED, weight=600)
    # the swizzle only works with this phase grouping: with phases of lanes 0-15, 16-31, ... it is 2-way
    naive = [max(sum(1 for L in range(16 * i, 16 * i + 16) if bank_group(frag_addr(L, 0, True)) == q)
                 for q in range(16)) for i in range(4)]
    assert naive == [2, 2, 2, 2]
    NY = GY + 16 * RH + 8 + 92
    f.lines(14, NY, ["格里的数字是读这一格的 lane，颜色是它在第几拍；灰格是逻辑块 4–7（h = 1 那条指令才读），数字是逻辑块号。",
                     "下面的计数：每一拍里，每组 bank 被几个 lane 访问。h = 1 时结果一样；换一个 mi 只是整体往后挪 2 KB，bank 不变。",
                     "粗框是 lane 21（第 1 拍）：读第 5 行的逻辑块 1，不 swizzle 在 bank 36–39；swizzle 后在物理格 4，即 bank 48–51。"],
            size=10.5, anchor="start", step=19, fill=MUTED)
    return f


def fig_lds_write():
    f = Fig(790, 540, "wave 0 的前两条 buffer_load_dwordx4 lds 把 A 的 16 行写进 LDS。左：显存里 A 的逻辑位置，"
                      "格里是读它的 lane，lane L 读第 8j + L/8 行的逻辑块 (L%8)^(L/8)。右：LDS 按 64 个 bank 画，"
                      "偶数行在 bank 0–31、奇数行在 bank 32–63，lane L 写 M0 + 16L，即第 8j + L/8 行的物理格 L%8，"
                      "所以第 r 行物理格 p 放的是逻辑块 p ^ (r & 7)。例：lane 44 读第 5 行逻辑块 1，写到偏移 704，bank 48–51")
    f.header("A 的 DMA 写入：lane L 写 LDS 的 M0 + 16·L（硬件定死），swizzle 改的是它从显存读哪一块",
             "画的是 wave 0 的前两条 DMA（j = 0、1），一共 16 行，就是一次片段读用到的那 16 行；每格 16 B = 32 个 fp4")
    for c in range(8):
        f.rect(14 + 18 * c, 60, 15, 15, chunk_fill(c), INK, sw=0.6)
        f.text(14 + 18 * c + 7.5, 67.5, str(c), size=9.5)
    f.text(14 + 18 * 8 + 4, 67.5, "颜色 = 逻辑块号（这一行 K 方向的第几块）：0–3 是 stage 的前 128 个 K（h = 0），"
                                  "4–7 是后 128 个（h = 1）", size=11, anchor="start", fill=MUTED)
    CW, CH, GX, GY = 26, 15, 74, 140
    BW, LX = 22, 414
    f.text(14, 100, "显存里的 A（逻辑位置），格里是读它的 lane", size=12, anchor="start", weight=600)
    f.text(LX - 30, 100, "LDS 槽（物理位置），横轴是 64 个 bank，格里是放进来的逻辑块", size=12, anchor="start",
           weight=600)
    for c in range(8):
        f.text(GX + c * CW + CW / 2, GY - 9, f"块{c}", size=8.5, fill=MUTED)
    for q in range(16):
        f.text(LX + q * BW + BW / 2, GY - 9, str(4 * q), size=7.5, fill=MUTED)
    f.text(LX + 4 * BW, GY - 24, "偶数行 → bank 0–31", size=9.5, fill=MUTED)
    f.text(LX + 12 * BW, GY - 24, "奇数行 → bank 32–63", size=9.5, fill=MUTED)
    for r in range(16):
        j, rr = divmod(r, 8)
        y = GY + r * CH
        f.text(GX - 5, y + CH / 2, f"行{r}", size=9, anchor="end", fill=MUTED)
        f.text(LX - 5, y + CH / 2, f"行{r}", size=9, anchor="end", fill=MUTED)
        for c in range(8):
            # a_voff[j] = a_lane + j*8*Kp: lane L fetches row 8j + L/8, logical chunk (L%8) ^ (L/8)
            L = 8 * rr + (c ^ rr)
            assert L >> 3 == rr and (L & 7) ^ (L >> 3) == c
            f.rect(GX + c * CW, y, CW, CH, chunk_fill(c), "#ffffff")
            f.text(GX + c * CW + CW / 2, y + CH / 2, str(L), size=9)
        for p in range(8):
            # M0 = slot + j*1024: lane L writes M0 + 16*L -> row 8j + L/8, physical chunk L%8
            L, c, off = 8 * rr + p, p ^ rr, r * 128 + p * 16
            assert off == j * 1024 + 16 * L and bank_group(off) == (r % 2) * 8 + p
            x = LX + bank_group(off) * BW
            f.rect(x, y, BW, CH, chunk_fill(c), "#ffffff")
            f.text(x + BW / 2, y + CH / 2, str(c), size=9.5, weight=600)
    f.rect(GX, GY, 8 * CW, 16 * CH, "none", INK)
    f.line(GX, GY + 8 * CH, GX + 8 * CW, GY + 8 * CH, stroke=INK, sw=1.4)
    f.rect(LX, GY, 16 * BW, 16 * CH, "none", INK)
    f.line(LX + 8 * BW, GY, LX + 8 * BW, GY + 16 * CH, stroke=INK, sw=1.2)
    f.line(LX, GY + 8 * CH, LX + 16 * BW, GY + 8 * CH, stroke=INK, sw=1.4)
    for j in range(2):
        f.text(14, GY + (8 * j + 4) * CH, f"j = {j}", size=10.5, anchor="start", weight=600)
    # lane 44 of DMA 0: row 5, logical chunk 1 -> physical chunk 4 (bank 48-51)
    assert 8 * 5 + (1 ^ 5) == 44 and bank_group(16 * 44) == 12
    f.rect(GX + CW, GY + 5 * CH, CW, CH, "none", INK, sw=2.2)
    f.rect(LX + 12 * BW, GY + 5 * CH, BW, CH, "none", INK, sw=2.2)
    f.line(GX + 8 * CW + 10, GY + 8 * CH, LX - 40, GY + 8 * CH, stroke=MUTED, sw=1.6, arrow=True)
    f.text((GX + 8 * CW + LX - 30) / 2, GY + 8 * CH - 12, "DMA", size=10.5, fill=MUTED)
    NY = GY + 16 * CH + 30
    f.text(14, NY, "写的地址由硬件定：lane L 写 M0 + 16·L，M0 = lds_base + lds_wave_off + j·1024，"
                   "即第 8j + L/8 行的物理格 L%8", size=11, anchor="start")
    f.text(14, NY + 21, "所以 swizzle 只能做在显存地址上：a_voff[j] = a_lane + j·8·Kp，"
                        "lane L 读第 8j + L/8 行的逻辑块 (L%8) ^ (L/8)", size=11, anchor="start")
    f.text(30, NY + 43, "a_lane = (lane >> 3) * Kp + (((lane & 7) ^ (lane >> 3)) * 16)", size=10.5, anchor="start",
           mono=True)
    f.text(14, NY + 65, "结果：第 r 行的物理格 p 放的是逻辑块 p ^ (r & 7)；读片段时 frag_at 用 "
                        "chk = (h·4 + g) ^ (row & 7) 换回来", size=11, anchor="start")
    f.text(14, NY + 87, "粗框：j = 0 的 lane 44 读显存第 5 行的逻辑块 4 ^ 5 = 1，写到 LDS 偏移 16·44 = 704，"
                        "即第 5 行物理格 4（bank 48–51）", size=11, anchor="start", fill=MUTED)
    f.text(14, NY + 109, "写的时候 lane L 管第 L/8 行（一行 8 个 lane）；读片段时 lane L 管第 L%16 行，"
                         "两边的 lane 号不是一回事", size=11, anchor="start", fill=MUTED)
    return f

# ----------------------------------------------------------------------------- 6. prefetch pipeline
def fig_pipeline():
    # kernel constants of the 256x256 tile, gemm_a4w4_wg_kernel<16, 4, 4>
    MSUB, NSUB, WAVES, RING, PF = 16, 4, 4, 4, 2
    PF_B = 2 if (NSUB <= 4 and MSUB < 16) else 1
    MTILE, NSB = 16 * MSUB, MSUB // 2
    KA = 2 * NSUB + NSUB // 2                       # B + B-scale loads of a step (section 0)
    LOADS = KA + MTILE // 8 // WAVES + (NSB + WAVES - 1) // WAVES
    W_START = (LOADS - KA) + LOADS * (PF_B - 1)
    W_MID = KA + LOADS * (PF - 2)
    MS0 = (MSUB // 2 + 1) // 2 * 2
    SLOT_KB = (MTILE * 128 + NSB * 256) // 1024
    assert (PF_B, KA, LOADS, W_START, W_MID, MS0, SLOT_KB) == (1, 10, 20, 10, 10, 8, 34)
    f = Fig(790, 545, "256×256 tile 的软件流水。step s 算 stage s，分两段：段 0 算 mi 0 到 7，同时从 LDS 读本 stage "
                      "mi 8 到 15 的片段，并发出 stage s+1 的 B load；段 1 算 mi 8 到 15，同时读 stage s+1 mi 0 到 7 "
                      "的片段，并发出 stage s+2 的 A DMA。step 开始前等本 stage 的 B 到齐，段 1 开始前等 stage s+1 的 A "
                      "到齐并做 barrier。LDS 4 个槽依次放 stage s、s+1、s+2，第 4 个槽是已读完的 stage s-1；"
                      "B 寄存器两套，一套在用，一套在写")
    f.header(f"预取：step s 算 stage s，同时搬 stage s+{PF_B} 的 B、stage s+{PF} 的 A",
             f"256×256 tile：PF_B = {PF_B}，PF = {PF}，LDS 环 {RING} 个槽。每个 step 两段，每段 {2 * MS0 * NSUB} "
             "条 MFMA；load 和 LDS 读一条一条插在 MFMA 之间")
    X, SW, GW = (124, 456), 150, 22                 # step starts, section width, gap at the barrier
    RY, RH, RP = 120, 28, 38                        # first row, box height, row pitch

    def S(d):
        return "s" if d == 0 else f"s+{d}"

    for i, name in enumerate(["MFMA", "LDS → A 片段", "B → 寄存器", "A → LDS（DMA）"]):
        f.text(14, RY + i * RP + RH / 2, name, size=11, anchor="start", weight=600)
    yt, yb = RY - 6, RY + 3 * RP + RH + 6
    for si, x0 in enumerate(X):
        x1 = x0 + SW + GW
        f.text(x0 + SW + GW / 2, 70, f"step {S(si)}：算 stage {S(si)}", size=12, weight=600)
        f.bracket(x0, x1 + SW, 82)
        f.text(x0 + SW / 2, 101, f"段 0（mi 0–{MS0 - 1}）", size=10.5, fill=MUTED)
        f.text(x1 + SW / 2, 101, f"段 1（mi {MS0}–{MSUB - 1}）", size=10.5, fill=MUTED)
        for r, x, label, fill in ((0, x0, f"stage {S(si)} · mi 0–{MS0 - 1}", BOX),
                                  (0, x1, f"stage {S(si)} · mi {MS0}–{MSUB - 1}", BOX),
                                  (1, x0, f"stage {S(si)} 的 mi {MS0}–{MSUB - 1} → afr1", G[0][2]),
                                  (1, x1, f"stage {S(si + 1)} 的 mi 0–{MS0 - 1} → afr0", G[0][2]),
                                  (2, x0, f"B({S(si + PF_B)}) → bf[({S(si + PF_B)})%4]", G[1][2]),
                                  (3, x1, f"A({S(si + PF)}) → 槽 ({S(si + PF)})%4", G[2][2])):
            y = RY + r * RP
            f.rect(x, y, SW, RH, fill, INK, sw=0.9, rx=3)
            f.text(x + SW / 2, y + RH / 2, label, size=10.5)
        f.line(x0 - 5, yt, x0 - 5, yb, stroke=INK, dash="3 3")
        f.text(x0 - 5, yb + 12, "①", size=11, weight=600)
        xb = x0 + SW + GW / 2
        f.line(xb, yt, xb, yb, stroke=INK, sw=1.6)
        f.text(xb, yb + 12, "② barrier", size=10.5, weight=600)
    # B(s+1): issued in step s's section 0, needed at the wait in front of step s+1
    yB = RY + 2 * RP + RH / 2 + 4
    f.path(f"M{X[0] + SW + 2} {yB}L{X[1] - 8} {yB}", stroke=G[1][0], sw=1.6, arrow=True)
    f.text(X[0] + SW + GW + SW / 2, yB - 11, "下一步开头就要用：① 处等它", size=9.5, fill=G[1][0])
    # A(s+2): issued in step s's section 1, first read in step s+1's section 1
    yA, yR, xv = RY + 3 * RP + RH / 2 + 4, RY + RP + RH / 2, X[1] + SW + 6
    f.path(f"M{X[0] + 2 * SW + GW + 2} {yA}L{xv} {yA}L{xv} {yR}L{X[1] + SW + GW - 2} {yR}",
           stroke=G[2][0], sw=1.6, arrow=True)
    f.text(X[1] + SW / 2, yA - 11, "下一步段 1 才读：② 处等它", size=9.5, fill=G[2][0])
    f.lines(14, yb + 36, [f"① step 开始前：等本 stage 的 B 和 B scale 到齐（vmcnt ≤ W_START = {W_START}），"
                          "段 0 要用的 A 片段也已读完（lgkmcnt(0)）",
                          f"② 段 1 开始前：等本 wave 为 stage s+1 发的 A DMA 到齐（vmcnt ≤ W_MID = {W_MID}），"
                          "再 __syncthreads()，等 4 个 wave 都到齐"],
            size=10.5, anchor="start", step=20, fill=MUTED)
    # ---- what the buffers hold during step s
    BY = yb + 92
    f.text(14, BY, "step s 期间，各个缓冲里放的是什么", size=12.5, anchor="start", weight=600)
    BX, BWd, BHt = (124, 284, 444, 604), 152, 46

    def boxes(y, items):
        for x, (a, b, fill) in zip(BX, items):
            f.rect(x, y, BWd, BHt, fill, INK, sw=0.9, rx=3)
            f.text(x + BWd / 2, y + 15, a, size=10.5, weight=600)
            f.text(x + BWd / 2, y + 32, b, size=10)

    y1 = BY + 18
    f.text(14, y1 + 15, "LDS 环", size=11, anchor="start", weight=600)
    f.text(14, y1 + 32, f"每槽 {SLOT_KB} KB", size=9.5, anchor="start", fill=MUTED)
    boxes(y1, [("槽 s%4：stage s", f"段 0 里读 mi {MS0}–{MSUB - 1}", G[0][2]),
               ("槽 (s+1)%4：stage s+1", f"② 处到齐，段 1 里读 mi 0–{MS0 - 1}", G[0][2]),
               ("槽 (s+2)%4：stage s+2", "段 1 里 DMA 写进来", G[2][2]),
               ("槽 (s+3)%4：stage s−1", "已读完，空着，下一步重填", "#ffffff")])
    y2 = y1 + BHt + 14
    f.text(14, y2 + 15, "B 寄存器", size=11, anchor="start", weight=600)
    f.text(14, y2 + 32, "8 片段 + 2 scale", size=9.5, anchor="start", fill=MUTED)
    boxes(y2, [("bf[s%4]：stage s", "两段的 MFMA 都在用", G[1][2]),
               ("bf[(s+1)%4]：stage s+1", "段 0 里写进来", G[1][2])])
    f.lines(BX[2], y2 + 15, [f"代码里声明了 {RING} 套（bf[RING]），同一时刻只活 {PF_B + 1} 套，",
                             "编译器复用到 2 套物理寄存器上（PF_B = 2 的 tile 是 3 套）"],
            size=10, anchor="start", fill=MUTED, step=17)
    f.lines(14, y2 + BHt + 26, ["stage 号超出本 tile 时，读的是下一个 tile 的开头：一个 tile 的最后两个 step 已经在搬"
                                "下一个 tile 的 A stage 0、1 和 B stage 0，",
                                "epilogue 写回期间这些 load 都在路上"], size=10.5, anchor="start", step=19)
    return f


# ----------------------------------------------------------------------------- 7. scale shuffle
def flatS(r, b, pad):
    r32, b8 = r & 31, b & 7
    return (r32 >> 4) + (b8 >> 2) * 2 + (r32 & 15) * 4 + (b8 & 3) * 64 + (b >> 3) * 256 + (r >> 5) * 32 * pad


def fig_scale_tile():
    MSUB, NSUB, WAVES, K = 16, 4, 4, 5120          # the 256x256 tile; K only for the example numbers
    pad = (K // 32 + 7) // 8 * 8                   # scaleN_pad
    NSB = MSUB // 2                                # A-scale blocks per stage
    S_LDS_PER_WAVE = (NSB + WAVES - 1) // WAVES
    dma_of = {}                                    # issue_one: block p = min(j*WAVES + wave, NSB-1)
    for w in range(WAVES):
        for j in range(S_LDS_PER_WAVE):
            dma_of.setdefault(min(j * WAVES + w, NSB - 1), w)
    assert sorted(dma_of) == list(range(NSB)) and pad == 160
    for p in range(NSB):                           # block p of stage s = 256 contiguous bytes at (G+p)*32*pad + 256*s
        for s in (0, 3):
            offs = sorted(flatS(32 * p + r, 8 * s + b, pad) for r in range(32) for b in range(8))
            assert offs == list(range(p * 32 * pad + 256 * s, p * 32 * pad + 256 * s + 256))
    f = Fig(790, 686, "一个 stage 的 scale，按 256×256 tile 画。A scale 是 256 行 × 8 个 K 块 = 2 KB，按 32 行切成 8 块，"
                      "每块 256 B；wave w 用 buffer_load_dword lds 搬第 w 和 w+4 块进 LDS，4 个 wave 再各用 ds_read_b32 读全部 8 块。"
                      "B scale 每个 wave 只要自己的 64 行 × 8 = 512 B，2 块，直接读进 VGPR。显存里每 32 行一组，"
                      "组内一个 stage 是连续的 256 B，块与块相隔 32 × scaleN_pad 字节")
    f.header("一个 stage 的 scale：A 要 256 行 × 8 个 K 块，B 每个 wave 要 64 行 × 8 个 K 块",
             "都按 32 行切成 256 B 一块，A、B 的布局完全一样（同一个 flatS）；区别只在谁读、读到哪")
    CW, BH, PIT, Y0 = 20, 24, 29, 124

    def blocks_header(X):
        for h in range(2):
            f.text(X + (4 * h + 2) * CW, 100, f"K 块 {4 * h}–{4 * h + 3}（h = {h}）", size=9.5, fill=MUTED)
            f.bracket(X + 4 * h * CW + 2, X + 4 * (h + 1) * CW - 2, 108)

    def block(X, y, w, label):
        for b in range(8):
            f.rect(X + b * CW, y, CW, BH, SC[1] if b < 4 else SC[2], "#ffffff", sw=0.6)
        f.rect(X, y, 8 * CW, BH, "none", G[w][0], sw=2)
        f.text(X + 4 * CW, y + BH / 2, label, size=9.5, weight=600)

    # ---- ① A scale: 8 blocks, DMA'd by wave p % 4, read by every wave
    AX = 112
    f.text(14, 78, "① A scale：256 行 × 8 个 K 块 = 2 KB，切成 8 块", size=12.5, anchor="start", weight=600)
    blocks_header(AX)
    for p in range(NSB):
        y, w = Y0 + p * PIT, dma_of[p]
        block(AX, y, w, f"p = {p}")
        f.text(AX - 6, y + BH / 2, f"行 {32 * p}–{32 * p + 31}", size=9.5, anchor="end", fill=MUTED)
        f.text(AX + 8 * CW + 6, y + BH / 2, f"mi {2 * p}、{2 * p + 1}", size=9.5, anchor="start")
        f.text(AX + 8 * CW + 56, y + BH / 2, f"wave {w} 搬", size=9.5, anchor="start", fill=G[w][0], weight=600)
    ny = Y0 + NSB * PIT + 12
    f.lines(14, ny, ["每个 wave 用 buffer_load_dword … lds 搬 2 块进 LDS 槽",
                     "（放在 A 数据后面）；4 个 wave 再各用 ds_read_b32",
                     "读全部 8 块，所以 A scale 是共用的"], size=10.5, anchor="start", step=17)
    # ---- ② B scale: each wave only its own 2 blocks, straight into VGPRs
    BX = 524
    f.text(410, 78, "② B scale：每个 wave 64 行 × 8 个 K 块 = 512 B，2 块", size=12.5, anchor="start", weight=600)
    blocks_header(BX)
    for w in range(WAVES):
        for p in range(NSUB // 2):
            y = Y0 + (2 * w + p) * PIT
            block(BX, y, w, f"wave {w} · p = {p}")
            f.text(BX - 6, y + BH / 2, f"n {64 * w + 32 * p}–{64 * w + 32 * p + 31}", size=9.5, anchor="end",
                   fill=MUTED)
            f.text(BX + 8 * CW + 6, y + BH / 2, f"ni {2 * p}、{2 * p + 1} → bs[{p}]", size=9.5, anchor="start")
    f.lines(410, ny, ["每个 wave 只用 buffer_load_dword 读自己的 2 块，",
                      "直接进 VGPR（bs[0]、bs[1]），不经过 LDS；",
                      "4 个 wave 合起来也是 256 行 × 8 = 2 KB"], size=10.5, anchor="start", step=17)
    # ---- ③ where the blocks sit in the scale tensor
    PY, SY, SW, SH, SP = 452, 482, 34, 18, 22
    f.text(14, PY, "③ 显存里：scale 张量按 32 行一组，每组 32 × pad 字节；组内一个 stage 是连续的 256 B（第 s 段）",
           size=12.5, anchor="start", weight=600)
    segs = [(150, "stage 0", False), (184, "stage 1", False), (248, "stage s", True), (312, "最后一个", False)]
    for x0, lab, _ in segs:
        f.text(x0 + SW / 2, SY - 10, lab, size=9, fill=MUTED)
    for p in range(NSB):
        y, w = SY + p * SP, dma_of[p]
        f.text(14, y + SH / 2, f"A 第 p = {p} 块（第 {p} 组）", size=10, anchor="start")
        for x0, _, cur in segs:
            if cur:
                f.rect(x0, y, SW, SH, SC[1], G[w][0], sw=1.8)
                f.text(x0 + SW / 2, y + SH / 2, f"p={p}", size=8.5, weight=600)
            else:
                f.rect(x0, y, SW, SH, BOX, "#ffffff", sw=0.8)
        for xg in (233, 297):
            f.text(xg, y + SH / 2, "…", size=11, fill=MUTED)
    f.lines(380, SY + 2, ["A scale 第 p 块（stage s，lane L 的 dword）：",
                          "AScale + (m0/32 + p)·32·pad + 256·s + 4·L",
                          "8 块各自连续 256 B，块与块相隔 32·pad 字节",
                          f"（K = {K} 时 pad = {pad}：一组 {32 * pad} B = {pad // 8} 段）", "",
                          "B scale 一样，换成本 wave 的 n0：",
                          "BScale + (n0/32 + p)·32·pad + 256·s + 4·L",
                          "每个 wave 只有 p = 0、1 两块"], size=10.5, anchor="start", step=19)
    f.text(14, SY + NSB * SP + 18, "pad = scaleN_pad（K/32 向上取整到 8 的倍数）；图 12 放大其中一块，"
                                   "看 64 个 lane 各拿哪 4 个字节", size=10.5, anchor="start", fill=MUTED)
    return f


def fig_scale():
    f = Fig(790, 610, "scale 的 shuffle 布局：32 行 × 8 个 K 块（两个 16 行子块 × 一个 stage）是连续的 256 字节，"
                      "lane L 读第 L 个 dword。dword 里 4 个字节依次是：子块 2p 的前 128 个 K、子块 2p+1 的前 128 个 K、"
                      "子块 2p 的后 128 个 K、子块 2p+1 的后 128 个 K。例：lane 21 的 dword 在偏移 84 到 87")
    f.header("一块 scale 里怎么排：32 行 × 8 个 K 块是连续的 256 B，lane L 读第 L 个 dword",
             "图 11 里的一块放大（A、B 的 scale 都这样排）。左：这块的 32 行 × 8 个 K 块，格里是 lane 号；右：这 256 B 的顺序")
    GX, GY, CW, CH = 104, 112, 32, 13
    f.text(GX + 2 * CW, 78, "h = 0（K 0–127）", size=10.5)
    f.text(GX + 6 * CW, 78, "h = 1（K 128–255）", size=10.5)
    f.bracket(GX + 2, GX + 4 * CW - 2, 88)
    f.bracket(GX + 4 * CW + 2, GX + 8 * CW - 2, 88)
    for b in range(8):
        f.text(GX + b * CW + CW / 2, GY - 10, f"块{b}", size=9.5, fill=MUTED)
    f.text(30, GY + 8 * CH, "子块 2p（行 0–15）", size=11, rotate=True)
    f.text(30, GY + 24 * CH, "子块 2p+1（行 16–31）", size=11, rotate=True)
    f.text(GX - 5, GY - 10, "行", size=9.5, anchor="end", fill=MUTED)
    for r in range(32):
        y = GY + r * CH
        f.text(GX - 5, y + CH / 2, str(r), size=8, anchor="end", fill=MUTED)
        for b in range(8):
            off = flatS(r, b, 8)
            L, byte = off // 4, off % 4
            assert L == (b % 4) * 16 + r % 16 and byte == 2 * (b // 4) + r // 16
            f.rect(GX + b * CW, y, CW, CH, G[b % 4][1 if b < 4 else 2], "#ffffff", sw=0.6)
            f.text(GX + b * CW + CW / 2, y + CH / 2, str(L), size=8.5)
    for r, b in ((5, 1), (21, 1), (5, 5), (21, 5)):
        f.rect(GX + b * CW, GY + r * CH, CW, CH, "none", INK, sw=2.2)
    f.rect(GX, GY, 8 * CW, 32 * CH, "none", INK)
    f.line(GX, GY + 16 * CH, GX + 8 * CW, GY + 16 * CH, stroke=INK, sw=1.8)
    f.line(GX + 4 * CW, GY, GX + 4 * CW, GY + 32 * CH, stroke=INK, sw=1.8)
    yb = GY + 32 * CH
    f.lines(14, yb + 22, ["四个象限 = dword 里的 4 个字节：左上 0、左下 1、右上 2、右下 3",
                          "即 字节号 = 2h + 子块奇偶（子块对 A 是 mi，对 B 是 ni）",
                          "黑框：lane 21 的 4 个字节"], size=11, anchor="start", fill=MUTED)
    # right: physical order
    RX, SY, SW = 392, 112, 6
    f.text(RX, 74, "显存里的这 256 B：64 个 dword，按 lane 号排", size=12, anchor="start", weight=600)
    for L in range(64):
        f.rect(RX + L * SW, SY - 8, SW, 26, G[L // 16][1], "#ffffff", sw=0.4)
    for g in range(4):
        f.text(RX + 96 * g + 48, SY - 17, f"lane {16 * g}–{16 * g + 15}", size=9.5, fill=G[g][0])
    f.rect(RX, SY - 8, 384, 26, "none", INK)
    f.rect(RX + 21 * SW, SY - 8, SW, 26, "none", INK, sw=2.2)
    for k in range(5):
        f.line(RX + 96 * k, SY + 18, RX + 96 * k, SY + 23, stroke=INK)
        f.text(RX + 96 * k, SY + 32, str(64 * k), size=9.5, fill=MUTED)
    # zoom into lane 21's dword
    ZX, ZY, ZW, ZH = RX, 196, 96, 62
    f.path(f"M{RX + 21 * SW} {SY + 18}L{ZX} {ZY}", dash="4 3")
    f.path(f"M{RX + 22 * SW} {SY + 18}L{ZX + 4 * ZW} {ZY}", dash="4 3")
    f.text(ZX + 2 * ZW, ZY - 26, "放大 lane 21 的 dword（偏移 84–87）", size=11.5)
    for k in range(4):
        h, par = k // 2, k % 2
        r, b = par * 16 + 5, h * 4 + 1
        assert flatS(r, b, 8) == 84 + k
        x = ZX + k * ZW
        f.rect(x, ZY, ZW, ZH, G[1][1 if h == 0 else 2], INK)
        f.text(x + ZW / 2, ZY + 13, f"字节 {k}（偏移 {84 + k}）", size=10, weight=600)
        f.text(x + ZW / 2, ZY + 31, f"行 {r} · 块 {b}", size=10.5)
        f.text(x + ZW / 2, ZY + 49, f"子块 {'2p+1' if par else '2p'}，h = {h}", size=10)
    f.text(RX, 296, "MFMA 用 op_sel 直接挑字节，不用解包", size=12, anchor="start", weight=600)
    f.lines(RX, 318, ["字节号 = op_sel_hi × 2 + op_sel",
                      "一条 buffer_load_dword 读来的 4 个字节，正好喂",
                      "（子块 2p、2p+1）×（h = 0、1）这 4 条 MFMA"], size=11, anchor="start", step=19)
    f.text(RX, 378, "mfma_fp4_sel<2*h + (ni&1), 2*h + (mi&1)>", size=10.5, anchor="start", mono=True)
    f.text(RX, 416, "整个 scale 张量怎么排", size=12, anchor="start", weight=600)
    f.lines(RX, 438, ["1. 行按 32 一组，每组占 32 × scaleN_pad 字节",
                      "2. 组内每 8 个 K 块（一个 stage）是一段 256 B",
                      "3. 段内 dword 号 = (块 % 4) × 16 + 行 % 16 = lane",
                      "4. dword 内字节号 = 2 × (块 / 4 % 2) + 行 / 16 % 2"],
            size=11, anchor="start", step=19)
    ind = RX + text_width("flatS(r, b) = ", 10.5, mono=True)
    f.text(RX, 528, "flatS(r, b) = (r/32)·32·pad + (b/8)·256", size=10.5, anchor="start", mono=True)
    f.text(ind, 546, "+ (b%4)·64 + (r%16)·4", size=10.5, anchor="start", mono=True)
    f.text(ind, 564, "+ ((b%8)/4)·2 + (r%32)/16", size=10.5, anchor="start", mono=True)
    f.text(RX, 588, "pad = scaleN_pad：K/32 向上取整到 8 的倍数", size=10.5, anchor="start", fill=MUTED)
    return f


def fig_scale_shuffle():
    """One scale block (32 rows x 8 K blocks = 256 B): row-major before the shuffle, lane-major dwords after."""
    lane_of = lambda r, b: 16 * (b % 4) + r % 16
    byte_of = lambda r, b: 2 * (b // 4) + r // 16
    new = {}
    for r in range(32):
        for b in range(8):
            off = 4 * lane_of(r, b) + byte_of(r, b)
            assert off == flatS(r, b, 8)
            new[off] = (r, b)
    assert sorted(new) == list(range(256))
    FILL = [G[0][2], G[0][1], G[1][2], G[1][1]]      # byte k = 2*h + parity
    EX = 21
    ex = [new[4 * EX + k] for k in range(4)]
    assert ex == [(5, 1), (21, 1), (5, 5), (21, 5)]
    f = Fig(790, 706, "scale 的 shuffle：一块 32 行 × 8 个 K 块共 256 B。左边是原来按行存的排布，第 r 行第 b 块在偏移 8r + b，"
                      "格子里是 shuffle 之后拿走它的 lane；右边是 shuffle 之后的 256 B，每行 8 个字节、2 个 lane，每 4 格是一个 "
                      "lane 的 dword。块切成 4 个 16 × 4 的小块，lane L 从每个小块的同一位置各拿 1 个字节：左上是字节 0，左下是 1，"
                      "右上是 2，右下是 3。黑框是 lane 21 的 4 个字节，原偏移 41、169、45、173，新偏移 84 到 87")
    f.header("scale 的 shuffle 一步一步看：一块 = 32 行 × 8 个 K 块 = 256 B",
             "左：原来按行存，格子里是 shuffle 之后拿走它的 lane；右：shuffle 之后的 256 B，每 4 格是一个 lane 的 dword")
    lx = 14
    for k, label in enumerate(("字节 0：行 0–15 × b0–b3", "字节 1：行 16–31 × b0–b3",
                               "字节 2：行 0–15 × b4–b7", "字节 3：行 16–31 × b4–b7")):
        f.rect(lx, 65, 11, 11, FILL[k])
        f.text(lx + 17, 70.5, label, size=10.5, anchor="start")
        lx += 17 + text_width(label, 10.5) + 20
    f.rect(lx, 65, 11, 11, "#ffffff", INK, sw=1.6)
    f.text(lx + 17, 70.5, f"lane {EX} 的 4 个字节", size=10.5, anchor="start")

    CW, CH, QG, LX, RX, Y0 = 26, 15, 6, 64, 386, 142

    def cell(x, y, L, k):
        f.rect(x, y, CW - 2, CH - 2, FILL[k], INK if L == EX else None, sw=1.6)
        f.text(x + (CW - 2) / 2, y + (CH - 2) / 2, str(L), size=9.5, weight=600 if L == EX else 400)

    # ① before: row-major 32 x 8, quadrants split by the subtile (rows) and the K half (blocks)
    bx = lambda b: LX + b * CW + (QG if b >= 4 else 0)
    by = lambda r: Y0 + r * CH + (QG if r >= 16 else 0)
    f.text(14, 98, "① 原来：按行存，第 r 行第 b 块在偏移 8r + b", size=12.5, anchor="start", weight=600)
    f.text(bx(0) + 2 * CW - 1, 116, "b0–b3：前 128 个 K", size=10, fill=MUTED)
    f.text(bx(4) + 2 * CW - 1, 116, "b4–b7：后 128 个 K", size=10, fill=MUTED)
    for b in range(8):
        f.text(bx(b) + (CW - 2) / 2, 132, f"b{b}", size=10, fill=MUTED)
    f.text(16, by(0) + 8 * CH, "行 0–15：子块 2p", size=10, fill=MUTED, rotate=True)
    f.text(16, by(16) + 8 * CH, "行 16–31：子块 2p+1", size=10, fill=MUTED, rotate=True)
    for r in range(32):
        f.text(LX - 6, by(r) + (CH - 2) / 2, f"r{r}", size=9, anchor="end", fill=MUTED)
        for b in range(8):
            cell(bx(b), by(r), lane_of(r, b), byte_of(r, b))

    # ② after: the same 256 bytes in memory order, 8 per row = the dwords of lanes 2q and 2q+1
    ax = lambda c: RX + c * CW + (QG if c >= 4 else 0)
    f.text(RX - 40, 98, "② shuffle 后：按 lane 存，第 L 个 dword 在偏移 4L", size=12.5, anchor="start", weight=600)
    f.text(ax(0) + 2 * CW - 1, 116, "lane 2q", size=10, fill=MUTED)
    f.text(ax(4) + 2 * CW - 1, 116, "lane 2q+1", size=10, fill=MUTED)
    f.text(RX - 6, 132, "字节", size=10, anchor="end", fill=MUTED)
    for c in range(8):
        f.text(ax(c) + (CW - 2) / 2, 132, str(c % 4), size=10, fill=MUTED)
    for q in range(32):
        f.text(RX - 6, Y0 + q * CH + (CH - 2) / 2, f"+{8 * q}", size=9, anchor="end", fill=MUTED)
        for c in range(8):
            off = 8 * q + c
            cell(ax(c), Y0 + q * CH, off // 4, off % 4)

    yq = Y0 + (4 * EX // 8) * CH + (CH - 2) / 2
    f.line(ax(7) + CW, yq, ax(7) + CW + 12, yq, stroke=INK, sw=1.2, arrow=True)
    f.text(616, yq - 36, f"lane {EX} 的 dword", size=11, anchor="start", weight=600)
    f.text(616, yq - 18, f"新偏移 {4 * EX}–{4 * EX + 3}，原来散在：", size=9.5, anchor="start", fill=MUTED)
    for k, (r, b) in enumerate(ex):
        f.text(616, yq + k * 17, f"字节 {k} ← (r{r}, b{b})，原偏移 {8 * r + b}", size=9.5, anchor="start")

    f.lines(14, 650,
            ["lane L 要的 4 个字节 = 4 个 16 × 4 小块里同一位置的格子：行 L%16 和 16+L%16，K 块 L/16 和 4+L/16",
             "字节号 = 2 × K 半段 + 子块：蓝色是前 128 个 K，橙色是后 128 个 K；浅色是上面的子块，深色是下面的子块",
             "新偏移 = 4·(16·(b%4) + r%16) + 2·(b/4) + r/16；一条 buffer_load_dword 读连续 256 B，每个 lane 4 B"],
            size=11, anchor="start", step=20)
    return f


# ----------------------------------------------------------------------------- 8. tile order across XCDs
def tile_of(i, num_m, num_n, remap, nxcd=8, group_m=4):
    """(m-tile, n-tile) of logical tile id i. remap=True is tile_off() of the wg kernel;
    remap=False is plain row-major order (no XCD remap, no GROUP_M grouping)."""
    total = num_m * num_n
    i = min(i, total - 1)
    if not remap:
        return i // num_n, i % num_n
    if i < total // nxcd * nxcd:
        i = (i % nxcd) * (total // nxcd) + i // nxcd
    span = group_m * num_n
    first_m = i // span * group_m
    gsize = min(num_m - first_m, group_m)
    return first_m + i % span % gsize, i % span // gsize


def span_str(vals, prefix):
    vals = sorted(vals)
    step = vals[1] - vals[0]
    assert all(b - a == step for a, b in zip(vals, vals[1:]))
    if step == 1:
        return f"{prefix}{vals[0]}–{prefix}{vals[-1]}"
    return f"{prefix}{vals[0]}、{prefix}{vals[1]}、…、{prefix}{vals[-1]}"


def fig_xcd_remap():
    # 8192 x 17408 with the 256x256 tile; one round = the 256 resident workgroups
    NUM_M, NUM_N, GRID, NXCD, GROUP_M = 8192 // 256, 17408 // 256, 256, 8, 4
    assert sorted(tile_of(i, NUM_M, NUM_N, True) for i in range(NUM_M * NUM_N)) == \
        [(m, n) for m in range(NUM_M) for n in range(NUM_N)]
    f = Fig(790, 796, f"XCD 重排前后对比。以 8192×17408、256×256 tile 为例，共 {NUM_M} 行 × {NUM_N} 列个 tile，"
                      f"第 0 轮 {GRID} 个工作组，工作组 b 在第 b % 8 个 XCD 上。重排前 XCD 0 的 32 个 tile 落在 m0 到 m3 行、"
                      "17 个不同的列上，用到 4 条 A 和 17 条 B；重排后集中在 m0 到 m3、n0 到 n7 的 4 × 8 方块，"
                      "只用到 4 条 A 和 8 条 B，8 个 XCD 各占 4 行")
    f.header("tile 的处理顺序：XCD 重排前后（第 0 轮）",
             f"例：8192×17408，256×256 tile，共 {NUM_M} × {NUM_N} = {NUM_M * NUM_N} 个；grid {GRID} 个工作组，"
             f"工作组 b 在第 b % {NXCD} 个 XCD 上，每个 XCD {GRID // NXCD} 个")
    lx, ly = 14, 68
    for fill, w, label in ((G[0][0], 10, "XCD 0 这一轮的 32 个 tile"), (LINE, 10, "其他 7 个 XCD 这一轮的 tile"),
                           (CELL, 10, "后面几轮的 tile"), (G[0][0], 4, "XCD 0 用到的 A 行、B 列")):
        f.rect(lx, ly - 5, w, 10, fill)
        f.text(lx + w + 6, ly, label, size=11, anchor="start")
        lx += w + 6 + text_width(label, 11) + 22
    CS, GX = 8, 76
    for remap, PY in ((False, 92), (True, 444)):
        tiles = [tile_of(b, NUM_M, NUM_N, remap) for b in range(GRID)]
        assert len(set(tiles)) == GRID
        per_xcd = [[t for b, t in enumerate(tiles) if b % NXCD == x] for x in range(NXCD)]
        for blk in per_xcd:
            assert len({m for m, _ in blk}) == 4 and len({n for _, n in blk}) == (8 if remap else 17)
        if remap:
            for x, blk in enumerate(per_xcd):
                assert set(blk) == {(GROUP_M * x + j, n) for j in range(GROUP_M) for n in range(8)}
        rows, cols = {m for m, _ in per_xcd[0]}, {n for _, n in per_xcd[0]}
        state = {t: 1 for t in tiles}
        state.update({t: 2 for t in per_xcd[0]})
        title = ("重排后：XCD 重排 + GROUP_M = 4（tile_off）" if remap
                 else "重排前：工作组 b 直接算第 b 个 tile（按行主序）")
        sub = (f"XCD 0 这一轮用到 {len(rows)} 条 A（{span_str(rows, 'm')}）+ {len(cols)} 条 B"
               f"（{span_str(cols, 'n')}）= {len(rows) + len(cols)} 条")
        if remap:
            sub += "；每个 XCD 都是一块 4 × 8"
        GY = PY + 56
        f.text(14, PY, title, size=12.5, anchor="start", weight=600)
        f.text(14, PY + 18, sub, size=11, anchor="start", fill=MUTED)
        f.text(GX + 3.5, GY - 20, "n0", size=10, fill=MUTED)
        f.text(GX + (NUM_N - 1) * CS + 3.5, GY - 20, f"n{NUM_N - 1}", size=10, fill=MUTED)
        f.text(GX - 14, GY + 3.5, "m0", size=10, anchor="end", fill=MUTED)
        f.text(GX - 14, GY + (NUM_M - 1) * CS + 3.5, f"m{NUM_M - 1}", size=10, anchor="end", fill=MUTED)
        for m in range(NUM_M):
            for n in range(NUM_N):
                f.rect(GX + n * CS, GY + m * CS, CS - 1, CS - 1, (CELL, LINE, G[0][0])[state.get((m, n), 0)])
        for n in cols:
            f.rect(GX + n * CS, GY - 10, CS - 1, 5, G[0][0])
        for m in rows:
            f.rect(GX - 10, GY + m * CS, 5, CS - 1, G[0][0])
        if remap:
            for x in range(NXCD):
                f.text(GX + NUM_N * CS + 8, GY + (GROUP_M * x + 2) * CS - 0.5, f"XCD {x}", size=10,
                       anchor="start", fill=MUTED)
    f.text(14, 780, "下一轮：重排前是编号 256–511 的 tile；重排后 8 个 XCD 整体右移 8 列（第 i 轮是 n8i–n8i+7），"
                    "第 8 轮只剩 n64–n67", size=10.5, anchor="start", fill=MUTED)
    return f


FIGS = {
    "kernel_intro_mxfp4.svg": fig_mxfp4,
    "kernel_intro_mfma.svg": fig_mfma,
    "kernel_intro_direct_read.svg": fig_direct_read,
    "kernel_intro_b_layout.svg": fig_b_shuffle,
    "kernel_intro_b_rule.svg": fig_b_rule,
    "kernel_intro_b_blocks.svg": fig_b_blocks,
    "kernel_intro_load_vars.svg": fig_load_vars,
    "kernel_intro_lds_read.svg": fig_lds_read,
    "kernel_intro_lds_write.svg": fig_lds_write,
    "kernel_intro_pipeline.svg": fig_pipeline,
    "kernel_intro_scale_tile.svg": fig_scale_tile,
    "kernel_intro_scale.svg": fig_scale,
    "kernel_intro_scale_shuffle.svg": fig_scale_shuffle,
    "kernel_intro_xcd_remap.svg": fig_xcd_remap,
}

if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "docs", "images")
    for name, fn in FIGS.items():
        fn().save(os.path.join(out, name))
