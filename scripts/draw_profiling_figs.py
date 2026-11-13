#!/usr/bin/env python3
"""Draw the figures used by docs/profiling_guide.md (SVG).

    python3 scripts/draw_profiling_figs.py [--att UI_DIR] [out_dir]     # default: docs/images

profiling_workflow.svg is drawn from the numbers below. profiling_wave_budget.svg
and profiling_att_stage.svg are computed from a rocprofv3 --att ui_output directory
(command in docs/profiling_guide.md, section 7) with scripts/att_budget.py; without
--att they are skipped.
"""
import os
import statistics
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from att_budget import MFMA_CYCLES, MFMA_STALL, NO_INST, PER_STAGE, analyze, load, tile_switch_mix  # noqa: E402
from draw_layout_figs import Fig, INK, LINE, MUTED, text_width  # noqa: E402

RED = "#dc2626"
IDLE_COLOR = {"发标量指令（SALU）": "#a3e635", MFMA_STALL: "#93c5fd", "发 buffer_load": "#fbbf24",
              NO_INST: "#d1d5db", "发 ds_read": "#fde68a", "等其他 wave（s_barrier）": "#f87171",
              "等数据（s_waitcnt）": "#fca5a5", "发向量指令（VALU）": "#fb923c", "其他指令": "#f3f4f6"}
SHORT = {"发标量指令（SALU）": "标量", MFMA_STALL: "MFMA 被卡", "发 buffer_load": "load",
         NO_INST: "空档", "发 ds_read": "ds_read", "等其他 wave（s_barrier）": "barrier",
         "等数据（s_waitcnt）": "等数据", "发向量指令（VALU）": "VALU", "其他指令": "其他"}


# ----------------------------------------------------------------------------- 1. workflow
STEPS = [
    ("① 测准时间", "bench_one.py 跑够久 · amd-smi monitor",
     ["现在到底多快？", "GPU 时钟稳了没有？"],
     ["5000 次平均 179 µs；只跑 50 次是 209 µs，", "时钟还没升上来。功耗顶在 1000 W，1.49 GHz"]),
    ("② 编译期体检", "-Rpass-analysis · .s · -gline-tables-only",
     ["寄存器够不够、有没有 spill？", "生成的指令和想的一样吗？"],
     ["256 VGPR + 256 AGPR 用满，scratch 为 0，", "每 SIMD 1 个 wave；每个 wave 26,592 条指令"]),
    ("③ 算上限", "MFMA 周期数 · rocprof-compute roofline",
     ["受算力限制，还是受带宽限制？", "最快能有多快？"],
     ["算术强度 2,559 FLOP/B，在拐点 1,672 右边，", "受 MFMA 限制；MFMA 一直忙只要 110 µs"]),
    ("④ 硬件计数器", "rocprofv3 --pmc · rocprof-compute",
     ["哪个部件在忙，忙到几成？", "有没有冲突、没命中？"],
     ["MFMA 利用率 55–56%，HBM 带宽只用 33%，", "LDS 没有 bank conflict，L2 命中 77%"]),
    ("⑤ 指令级 trace（ATT）", "rocprofv3 --att · RCV · att_budget.py",
     ["MFMA 空着的时候，", "wave 卡在哪条指令上？"],
     ["K 循环里 MFMA 空 19.6%，tile 切换 11%；", "等数据只占这 19.6% 里的 1%"]),
    ("⑥ 改一处，回到 ①", "A/B 交替测 · 比周期数",
     ["改动真的让它变快了吗？", ""],
     ["每次只改一个地方，", "两个版本交替测，看同一组数字"]),
]


def fig_workflow():
    f = Fig(790, 510, "优化一个 HIP kernel 的六步：测准时间、编译期体检、算上限、硬件计数器、指令级 trace、"
                      "改一处再回到第一步；每一步写了用的工具、回答的问题和在 MBS 上的结果")
    f.header("优化一个 HIP kernel：先测准，再一层层往下找",
             "每一步回答一个问题；改完代码回到第 ① 步重新测。右列是 MBS 在 8192×8192×5120 上的结果")
    X0, BW, QX, RX = 36, 236, 290, 506
    Y0, RH, GAP = 96, 58, 8
    f.text(X0, 76, "步骤 · 工具", size=12, anchor="start", fill=MUTED, weight=600)
    f.text(QX, 76, "这一步回答", size=12, anchor="start", fill=MUTED, weight=600)
    f.text(RX, 76, "在 MBS 上的结果", size=12, anchor="start", fill=MUTED, weight=600)
    for i, (title, tools, q, r) in enumerate(STEPS):
        y = Y0 + i * (RH + GAP)
        f.rect(X0, y, BW, RH, fill="#eff6ff" if i < 5 else "#f0fdf4",
               stroke="#2563eb" if i < 5 else "#15803d", rx=6)
        f.text(X0 + 12, y + 20, title, size=13, anchor="start", weight=600)
        f.text(X0 + 12, y + 41, tools, size=11, anchor="start", fill=MUTED)
        f.lines(QX, y + 20, q, size=12, step=19, anchor="start")
        f.lines(RX, y + 20, r, size=12, step=19, anchor="start")
        if i < len(STEPS) - 1:
            f.line(X0 + BW / 2, y + RH, X0 + BW / 2, y + RH + GAP, stroke=LINE, arrow=True)
    yl = Y0 + 5 * (RH + GAP) + RH / 2
    f.path(f"M{X0} {yl}L20 {yl}L20 {Y0 + RH / 2}L{X0} {Y0 + RH / 2}", stroke="#15803d",
           sw=1.4, arrow=True)
    f.text(14, Y0 + 3 * (RH + GAP), "重新测", size=11, fill="#15803d", rotate=True)
    return f


# ----------------------------------------------------------------------------- 2. budget
def legend(f, x, y, items, total, cols=(300, 350), step=19, chip=10):
    for i, (name, v, color) in enumerate(items):
        yy = y + i * step
        f.rect(x, yy - chip / 2, chip, chip, fill=color, stroke=MUTED, sw=0.6)
        f.text(x + chip + 6, yy, name, size=12, anchor="start")
        f.text(x + cols[0], yy, f"{v:,.0f}", size=12, anchor="end", fill=MUTED)
        f.text(x + cols[1], yy, f"{100 * v / total:.1f}%", size=12, anchor="end")


def bar(f, x, y, w, h, items, total, labels=None):
    """Stacked bar; writes the longest of `labels[i]` (or the percentage) that fits in a
    segment and returns the x range of every segment plus the ones left unlabeled."""
    spans, bare, cx = [], [], x
    for i, (name, v, color) in enumerate(items):
        sw = w * v / total
        f.rect(cx, y, sw, h, fill=color, stroke="#ffffff", sw=1)
        pct = f"{100 * v / total:.0f}%" if labels is None else f"{100 * v / total:.1f}%"
        cands = [(f"{c} {pct}", s) for c in (labels[i] if labels else []) for s in (12, 11)]
        fit = next((c for c in cands + [(pct, 12)] if text_width(c[0], c[1]) + 10 < sw), None)
        if fit:
            f.text(cx + sw / 2, y + h / 2, fit[0], size=fit[1])
        else:
            bare.append((cx, cx + sw, f"{name} {pct}"))
        spans.append((cx, cx + sw))
        cx += sw
    return spans, bare


def fig_wave_budget(res, kinds):
    nw = len(res)
    budget = sum((r["budget"] for r in res), Counter())
    life = sum(budget.values())
    top = [("开头", budget["开头（第一条 MFMA 之前）"] / nw, "#e5e7eb"),
           ("收尾", budget["收尾（最后一个 epilogue）"] / nw, "#d1d5db"),
           ("MFMA 在算", budget["MFMA 在算"] / nw, "#93c5fd"),
           ("K 循环里空闲", budget["K 循环里 MFMA 空闲"] / nw, "#fdba74"),
           ("tile 切换", budget["tile 切换（epilogue + 下一 tile 开头）"] / nw, "#d8b4fe")]
    idle = sum((r["idle"] for r in res), Counter())
    idle_items = [(k, v / nw, IDLE_COLOR[k]) for k, v in idle.most_common()]
    mix, n = Counter(), 0
    for r in res:
        m, c = tile_switch_mix(r, kinds)
        mix += m
        n += c
    win = sum(r["issue"][lo2] - r["issue"][hi - 1] for r in res
              for (_, hi), (lo2, _) in zip(r["tiles"], r["tiles"][1:])) / n
    sw_items = [("buffer_store（写回 C）", mix["buffer_store"] / n, "#f0abfc"),
                ("v_accvgpr_*（读结果、清零）", mix["v_accvgpr_*"] / n, "#d8b4fe"),
                ("其他 VALU（拼数、算地址）", mix["VALU"] / n, "#c4b5fd"),
                ("v_cvt / v_permlane", (mix["v_cvt_*"] + mix["v_permlane*"]) / n, "#a5b4fc"),
                ("SALU", mix["SALU"] / n, "#bfdbfe")]
    sw_items.append(("其他和空隙", win - sum(v for _, v, _ in sw_items), "#f3f4f6"))

    lo, hi = res[0]["tiles"][0]
    rows = max(len(idle_items), len(sw_items))
    BY, HW = 192, 362
    FY = BY + 44 + rows * 19 + 12
    f = Fig(790, FY + 46, "一个 wave 的时间预算：MFMA 在算、K 循环里 MFMA 空闲、tile 切换、收尾、开头；下面把 "
                          "K 循环的空闲按 wave 当时在发的指令拆开，把 tile 切换按指令类别拆开")
    f.header(f"一个 wave 的 {life / nw:,.0f} 个周期花在哪（ATT 实测，{nw} 个 wave 平均）",
             f"8192×8192×5120，每个 wave {len(res[0]['tiles'])} 个 tile × {(hi - lo) // PER_STAGE} 个 stage；"
             "一条 MFMA 占管线 16 周期")
    X, W, Y1 = 24, 742, 90
    names = [[f"{n} {v:,.0f} 周期 ·", n, n.split()[-1]] for n, v, _ in top]
    spans, bare = bar(f, X, Y1, W, 32, top, life / nw, labels=names)
    if bare:
        f.bracket(bare[0][0], bare[-1][1], Y1 + 38, up=False)
        f.text(X, Y1 + 52, "左边小段：" + "，".join(s for _, _, s in bare), size=12, anchor="start")
    for (x0, x1), (bx0, bx1) in ((spans[3], (X, X + HW)), (spans[4], (X + W - HW, X + W))):
        f.path(f"M{x0:.1f} {Y1 + 32}L{x1:.1f} {Y1 + 32}L{bx1:.1f} {BY}L{bx0:.1f} {BY}Z",
               stroke="none", fill="#f3f4f6")
        f.line(x0, Y1 + 32, bx0, BY, stroke=LINE, dash="3 3")
        f.line(x1, Y1 + 32, bx1, BY, stroke=LINE, dash="3 3")
    f.text(X, BY - 14, "K 循环里 MFMA 空着时，wave 在做什么", size=12, anchor="start", weight=600)
    f.text(X + W - HW, BY - 14, f"一次 tile 切换（平均 {win:,.0f} 周期）里各类指令", size=12,
           anchor="start", weight=600)
    idle_total = sum(v for _, v, _ in idle_items)
    bar(f, X, BY, HW, 26, idle_items, idle_total)
    bar(f, X + W - HW, BY, HW, 26, sw_items, win)
    legend(f, X, BY + 44, idle_items, idle_total, cols=(300, HW))
    legend(f, X + W - HW, BY + 44, sw_items, win, cols=(300, HW))
    f.lines(X, FY, ["MFMA 空着的每个周期，记到 wave 那一刻正在发、或者被堵住的那条指令上；“下一条 MFMA 自己被卡”"
                    "是中间的指令都发完了，",
                    "MFMA 还发不出去；“空档”是那一刻 wave 没有指令在发（ATT 的 Idle）。"
                    "数字是每个 wave 的周期数，由 scripts/att_budget.py 算出"],
            size=11, step=17, anchor="start", fill=MUTED)
    return f


# ----------------------------------------------------------------------------- 3. one stage
ROWS = [("MFMA 管线", None), ("buffer_load", ("buffer_load（→LDS）", "buffer_load（→VGPR）")),
        ("ds_read", ("ds_read",)), ("SALU", ("SALU",)), ("VALU", ("VALU",)),
        ("等待", ("s_waitcnt", "s_barrier"))]
ROW_Y = [0, 24, 40, 56, 72, 88]


def pick_stage(a, kinds):
    """A stage of the second tile whose length is the median one and whose barrier waited."""
    ins, mp, issue = a["ins"], a["mp"], a["issue"]
    lo, hi = a["tiles"][1]
    cands = []
    for s in range(1, (hi - lo) // PER_STAGE - 1):
        j0 = lo + s * PER_STAGE
        waited = any(kinds[i[4]] == "s_barrier" and i[3] > 8
                     for i in ins[mp[j0 + 63] + 1:mp[j0 + 64]])
        cands.append((issue[j0 + PER_STAGE] - issue[j0], waited, s, j0))
    med = statistics.median(c[0] for c in cands)
    return min(cands, key=lambda c: (not c[1], abs(c[0] - med)))


def gap_label(parts):
    top2 = parts.most_common(2)
    txt = f"{SHORT[top2[0][0]]} {top2[0][1]}"
    if len(top2) > 1 and top2[1][1] >= 16:
        txt += f" + {SHORT[top2[1][0]]} {top2[1][1]}"
    return txt


def fig_att_stage(res, kinds):
    a = res[0]
    ins, mp, issue, start, gaps = a["ins"], a["mp"], a["issue"], a["start"], a["gaps"]
    length, _, s, j0 = pick_stage(a, kinds)
    f = Fig(790, 500, "ATT 里的一个 stage：上面是段 0，下面是段 1；每段第一行是 MFMA 管线，蓝块是一条 MFMA "
                      "占的 16 周期，红色是管线空着；下面几行是同一时刻发出去的 load、LDS 读、标量、向量指令和等待")
    f.header(f"ATT 里的一个 stage：MFMA 什么时候在算，空着时 wave 在干什么（共 {length:,} 周期）",
             f"SIMD 0 上的 wave，第 2 个 tile 的第 {s + 1} 个 stage。蓝块：一条 MFMA 占管线 16 周期；"
             "红块：管线空着；细竖线：其他指令发出的时刻")
    X0, X1 = 104, 772
    T0 = issue[j0]
    panels = [(0, T0 - 8, start[j0 + 63] + MFMA_CYCLES),
              (1, start[j0 + 63] + MFMA_CYCLES, issue[j0 + PER_STAGE] + 8)]
    sx = (X1 - X0) / max(t1 - t0 for _, t0, t1 in panels)
    stage_idle = Counter()
    for p, t0, t1 in panels:
        PY = 98 + p * 196
        title = ("段 0：算 mi 0–7（MFMA 0–63），同时发 stage s+1 的 B、B scale，读本 stage mi 8–15 的 A 片段"
                 if p == 0 else
                 "段 1：开头 barrier，算 mi 8–15（MFMA 64–127），同时发 stage s+2 的 A DMA，读 s+1 的 A 片段")
        f.text(14, PY - 26, title, size=12, anchor="start", weight=600)
        for (name, _), dy in zip(ROWS, ROW_Y):
            mfma = name == "MFMA 管线"
            f.text(X0 - 10, PY + dy + (8 if mfma else 5), name, size=11, anchor="end",
                   fill=INK if mfma else MUTED)
            f.line(X0, PY + dy + (18 if mfma else 10), X1, PY + dy + (18 if mfma else 10),
                   stroke="#e5e7eb")
        X = lambda t: X0 + (t - t0) * sx
        labels = []
        for j in (range(j0, j0 + 64) if p == 0 else range(j0 + 64, j0 + PER_STAGE + 1)):
            if j in gaps and j > j0:
                g0 = start[j - 1] + MFMA_CYCLES
                if t0 <= g0 < t1:
                    f.rect(X(g0), PY, (issue[j] - g0) * sx, 16, fill=RED)
                    stage_idle += gaps[j]
                    labels.append((issue[j] - g0, X(g0) + (issue[j] - g0) * sx / 2, gaps[j]))
            if j < j0 + PER_STAGE and t0 <= start[j] < t1:
                f.rect(X(start[j]), PY, MFMA_CYCLES * sx - 0.6, 16, fill="#60a5fa")
        placed = []
        for gap, x, parts in sorted(labels, key=lambda l: -l[0])[:4]:
            txt = gap_label(parts)
            w = text_width(txt, 10)
            x = min(max(x, X0 + w / 2), X1 - w / 2)
            if gap >= 12 and all(abs(x - q) > (w + wq) / 2 + 4 for q, wq in placed):
                f.text(x, PY - 9, txt, size=10, fill=RED)
                placed.append((x, w))
        k_lo = mp[j0 + (0 if p == 0 else 63)]
        k_hi = mp[j0 + (64 if p == 0 else PER_STAGE)]
        for t, _, stall, dur, idx in ins[k_lo:k_hi + 1]:
            if not t0 <= t < t1:
                continue
            for (name, ks), dy in zip(ROWS, ROW_Y):
                if not ks or kinds[idx] not in ks:
                    continue
                y = PY + dy
                if name == "等待":
                    f.rect(X(t), y, max(dur * sx, 1.2), 10, fill="#a78bfa")
                    if dur > 16:
                        txt = f"{kinds[idx]} 等了 {dur} 周期"
                        right = X(t) + dur * sx + 4 + text_width(txt, 10) <= X1
                        f.text(X(t) + dur * sx + 4 if right else X(t) - 4, y + 5, txt, size=10,
                               anchor="start" if right else "end", fill="#6d28d9")
                else:
                    if stall:
                        f.rect(X(t), y, stall * sx, 10, fill="#fecaca")
                    f.rect(X(t + stall), y, max(1.2, 4 * sx), 10, fill=INK)
        ax = PY + 108
        f.line(X0, ax, X1, ax, stroke=LINE)
        for c in range(-(-(t0 - T0) // 200) * 200, t1 - T0, 200):
            f.line(X(T0 + c), ax, X(T0 + c), ax + 4, stroke=LINE)
            f.text(X(T0 + c), ax + 13, f"{c:+d}" if c else "0", size=10, fill=MUTED)
    tot = sum(stage_idle.values())
    words = [f"这个 stage：128 条 MFMA 占 {PER_STAGE * MFMA_CYCLES:,} 周期，管线空了 {tot:,} 周期："]
    words += [f"{SHORT[k]} {v}，" for k, v in stage_idle.most_common()]
    words[-1] = words[-1][:-1] + "。"
    words += ["红字是一个空泡里占得最多的一两项；", "浅红是指令发射前被堵的周期，",
              "紫色是 s_waitcnt / s_barrier 等的周期。", "横轴相对这个 stage 第一条 MFMA。"]
    rows = [""]
    for w in words:
        if text_width(rows[-1] + w, 11) > 760:
            rows.append("")
        rows[-1] += w
    f.lines(14, 444, rows, size=11, step=17, anchor="start", fill=MUTED)
    return f


def main():
    args = sys.argv[1:]
    ui = None
    if "--att" in args:
        i = args.index("--att")
        ui = args[i + 1]
        del args[i:i + 2]
    out = args[0] if args else os.path.join(os.path.dirname(__file__), "..", "docs", "images")
    os.makedirs(out, exist_ok=True)
    fig_workflow().save(os.path.join(out, "profiling_workflow.svg"))
    if ui is None:
        print("skip profiling_wave_budget.svg / profiling_att_stage.svg: pass --att <ui_output dir>")
        return
    code, kinds, waves = load(ui)
    res = [analyze(w, kinds) for w in waves]
    fig_wave_budget(res, kinds).save(os.path.join(out, "profiling_wave_budget.svg"))
    fig_att_stage(res, kinds).save(os.path.join(out, "profiling_att_stage.svg"))


if __name__ == "__main__":
    main()
