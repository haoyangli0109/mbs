#!/usr/bin/env python3
"""Draw the per-version ATT figures of docs/naive_gemm_lab.md (SVG).

    python3 draw_lab_figs.py <version> <ui_output_agent_*_dispatch_* dir> [out_dir]

Writes lab_<version>_wave_time.svg (how the waves' time splits over instruction
kinds) and lab_<version>_timeline.svg (SIMD 0: the MFMA pipe and every resident
wave over about two K steps). out_dir defaults to ../docs/images.
"""
import os
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "scripts"))
from att_summary import MFMA_CYCLES, load, src_line, wave_time  # noqa: E402
from draw_layout_figs import Fig, INK, LINE, MUTED, text_width  # noqa: E402

COLOR = {"MFMA": "#2563eb", "等 vmcnt（全局内存）": "#a78bfa", "等 vmcnt + lgkmcnt": "#c4b5fd",
         "等 lgkmcnt（LDS、标量读）": "#fdba74", "等 barrier": "#f87171", "发全局读": "#374151",
         "发全局写": "#6b7280", "发 LDS 读": "#10b981", "发 LDS 写": "#6ee7b7",
         "向量指令": "#9ca3af", "标量指令": "#d1d5db", "分支": "#e5e7eb", "其他": "#e5e7eb",
         "空档（没有指令可发）": "#ffffff", "开头和结尾": "#ffffff"}
# what a long wait at a source line means, per version
WAIT_LABEL = {
    "v0": {("v0_naive.h", 193): "拷贝循环：等这一次的 load 回来",
           ("common.h", 34): "MFMA 前：等 scale 和 LDS 读"},
}
STEP_MFMAS = {"v0": 16}


def fig_wave_time(ver, waves, kinds):
    total = Counter()
    for _, w in waves:
        total += wave_time(w, kinds)
    life = sum(total.values())
    items = [(k, v) for k, v in total.most_common() if v / life >= 0.002]
    rows = (len(items) + 1) // 2
    f = Fig(790, 150 + rows * 20, f"{ver} 的 wave 时间按指令类别的分配")
    f.header(f"{ver}：每个 wave 的 {life / len(waves):,.0f} 个周期花在哪（ATT，{len(waves)} 个 wave 平均）",
             "每条指令占的周期 = 发射 + 被堵；s_waitcnt 按它等的计数器分开")
    X, W, Y = 24, 742, 70
    cx = X
    for k, v in items:
        w = W * v / life
        f.rect(cx, Y, w, 30, fill=COLOR[k], stroke="#ffffff", sw=1)
        pct = f"{100 * v / life:.0f}%"
        if w > text_width(pct, 12) + 8:
            f.text(cx + w / 2, Y + 15, pct, size=12, fill="#ffffff" if COLOR[k] in ("#2563eb", "#374151") else INK)
        cx += w
    for i, (k, v) in enumerate(items):
        x, y = X + (i // rows) * 380, Y + 54 + (i % rows) * 20
        f.rect(x, y - 5, 10, 10, fill=COLOR[k], stroke=MUTED, sw=0.6)
        f.text(x + 16, y, k, size=12, anchor="start")
        f.text(x + 300, y, f"{v / len(waves):,.0f}", size=12, anchor="end", fill=MUTED)
        f.text(x + 350, y, f"{100 * v / life:.1f}%", size=12, anchor="end")
    return f


def fig_timeline(ver, code, waves, kinds, simd=0, step=10, span=20000):
    ws = sorted((w for s, w in waves if s == simd), key=lambda w: w["begin"])
    focus = ws[len(ws) // 2]
    mf = [i for i in focus["instructions"] if kinds[i[4]] == "MFMA"]
    t0 = mf[step * STEP_MFMAS[ver]][0] - span // 5
    t1 = t0 + span
    rows = [w for w in ws if w["begin"] < t1 and w["end"] > t0]
    issues = sorted(i[0] + i[2] for w in ws for i in w["instructions"] if kinds[i[4]] == "MFMA")
    X0, X1, RY, RH = 120, 772, 92, 30
    sx = (X1 - X0) / span
    X = lambda t: X0 + (t - t0) * sx
    f = Fig(790, RY + (len(rows) + 1) * (RH + 14) + 120,
            f"{ver}：SIMD {simd} 上 MFMA 管线和每个驻留 wave 在约两步 K 里的时间线")
    f.header(f"{ver}：SIMD {simd} 上的 {span:,} 个周期（约两步 K）",
             "第一行是这个 SIMD 的 MFMA 管线（蓝色在算）；下面每行一个驻留的 wave，颜色是它当时在做什么")
    f.text(X0 - 10, RY + RH / 2, "MFMA 管线", size=11, anchor="end")
    f.rect(X0, RY, X1 - X0, RH, fill="#fee2e2")
    finish = -1
    for t in issues:
        start = max(t, finish)
        finish = start + MFMA_CYCLES
        if t0 <= start < t1:
            f.rect(X(start), RY, max(MFMA_CYCLES * sx, 0.8), RH, fill=COLOR["MFMA"])
    labels = WAIT_LABEL.get(ver, {})
    for r, w in enumerate(rows):
        y = RY + (r + 1) * (RH + 14)
        f.text(X0 - 10, y + RH / 2, f"wave {r + 1}" + ("（标注）" if w is focus else ""), size=11,
               anchor="end", fill=INK if w is focus else MUTED)
        f.rect(X0, y, X1 - X0, RH, fill="#ffffff", stroke="#e5e7eb", sw=0.6)
        done = set()
        for t, _, stall, dur, idx in w["instructions"]:
            a, b = max(t, t0), min(t + dur, t1)
            if b <= a:
                continue
            k = kinds[idx]
            wide = k in ("MFMA", "发全局读", "发全局写")
            f.rect(X(a), y, max((b - a) * sx, 0.8 if wide else 0.3), RH, fill=COLOR[k])
            lab = labels.get(src_line(code[idx][3]))
            if lab and w is focus and k.startswith("等") and dur > span // 40 and lab not in done:
                done.add(lab)
                f.line(X(a) + 1, y - 3, X(a) + 1, y, stroke="#5b21b6")
                f.text(min(max(X(a), X0 + 2), X1 - text_width(lab, 10) - 2), y - 8, lab, size=10,
                       anchor="start", fill="#5b21b6")
    ax = RY + (len(rows) + 1) * (RH + 14)
    f.line(X0, ax, X1, ax, stroke=LINE)
    for c in range(0, span + 1, 4000):
        f.line(X0 + c * sx, ax, X0 + c * sx, ax + 4, stroke=LINE)
        f.text(X0 + c * sx, ax + 13, f"+{c:,}", size=10, fill=MUTED)
    keys = ["MFMA", "等 vmcnt（全局内存）", "等 vmcnt + lgkmcnt", "等 lgkmcnt（LDS、标量读）", "等 barrier",
            "发全局读", "发 LDS 读", "发 LDS 写", "向量指令", "标量指令"]
    for i, k in enumerate(keys):
        x, yy = 24 + (i % 5) * 152, ax + 40 + (i // 5) * 20
        f.rect(x, yy - 5, 10, 10, fill=COLOR[k], stroke=MUTED, sw=0.6)
        f.text(x + 16, yy, k, size=11, anchor="start")
    return f


def main():
    ver, ui = sys.argv[1], sys.argv[2]
    out = sys.argv[3] if len(sys.argv) > 3 else os.path.join(HERE, "..", "docs", "images")
    code, kinds, waves = load(ui)
    fig_wave_time(ver, waves, kinds).save(os.path.join(out, f"lab_{ver}_wave_time.svg"))
    fig_timeline(ver, code, waves, kinds).save(os.path.join(out, f"lab_{ver}_timeline.svg"))


if __name__ == "__main__":
    main()
