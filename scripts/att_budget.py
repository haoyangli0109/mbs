#!/usr/bin/env python3
"""Where the cycles of a gemm_a4w4_wg_kernel wave go, from one rocprofv3 --att trace.

    python3 scripts/att_budget.py <ui_output_agent_*_dispatch_* dir> [kernel/gemm_a4w4.hip]

Averaged over the traced waves, prints
  1. the wave's time budget: MFMA busy, MFMA pipe idle inside the K loop, tile
     switches (epilogue + start of the next tile), head and tail;
  2. what the wave was doing while the MFMA pipe sat idle in the K loop;
  3. what the tile-switch windows spend their cycles on;
  4. the source lines with the most stall (innermost line of the inline chain).

Wave records are [cycle, type, stall, duration, code index]: the wave starts trying
the instruction at `cycle`, issues it after `stall` cycles and tries the next one
`duration` cycles later. Pipe model: an FP4 16x16x128 MFMA keeps the matrix pipe
busy for 16 cycles (rocprof-compute 11.2.9 "MFMA Instruction Cycles") and starts at
max(its issue, previous MFMA's finish). Every idle pipe cycle is charged to the
instruction the wave was issuing or stalled on at that cycle; cycles inside the next
MFMA's own stall are charged to that MFMA, and cycles with no instruction in flight
(the trace's Idle) to "wave 没在发指令".
"""
import glob
import json
import os
import sys
from collections import Counter, defaultdict

MFMA_CYCLES = 16
PER_STAGE = 128
CAT = {"s_waitcnt": "等数据（s_waitcnt）", "s_barrier": "等其他 wave（s_barrier）",
       "buffer_load（→LDS）": "发 buffer_load", "buffer_load（→VGPR）": "发 buffer_load",
       "ds_read": "发 ds_read", "SALU": "发标量指令（SALU）", "VALU": "发向量指令（VALU）"}
MFMA_STALL = "下一条 MFMA 自己被卡"
NO_INST = "wave 没在发指令（空档）"


def kind(isa):
    op = isa.split()[0] if isa and not isa.startswith(";") else ""
    if op.startswith("v_mfma"):
        return "MFMA"
    if op in ("s_waitcnt", "s_barrier", "s_nop"):
        return op
    if op.startswith("buffer_load"):
        return "buffer_load（→LDS）" if " lds" in isa else "buffer_load（→VGPR）"
    for prefix, name in (("buffer_store", "buffer_store"), ("ds_", "ds_read"),
                         ("v_accvgpr", "v_accvgpr_*"), ("v_cvt", "v_cvt_*"),
                         ("v_permlane", "v_permlane*"), ("s_", "SALU"), ("v_", "VALU")):
        if op.startswith(prefix):
            return name
    return "其他"


def load(ui):
    code = json.load(open(os.path.join(ui, "code.json")))["code"]
    waves = [json.load(open(p))["wave"]
             for p in sorted(glob.glob(os.path.join(ui, "se*_wv*.json")))]
    return code, [kind(c[0]) for c in code], waves


def overlap(a0, a1, b0, b1):
    return max(0, min(a1, b1) - max(a0, b0))


def charge_gap(ins, k_prev, k_next, g0, g1, kinds):
    """Split the idle pipe interval [g0, g1) between MFMAs ins[k_prev] and ins[k_next]."""
    out = Counter()
    for i in ins[k_prev + 1:k_next]:
        out[CAT.get(kinds[i[4]], "其他指令")] += overlap(g0, g1, i[0], i[0] + i[3])
    m = ins[k_next]
    out[MFMA_STALL] += overlap(g0, g1, m[0], m[0] + m[2])
    out[NO_INST] += (g1 - g0) - sum(out.values())
    return +out


def analyze(wave, kinds):
    """Budget, idle split and per-MFMA pipe timeline of one wave."""
    ins = wave["instructions"]
    mp = [k for k, i in enumerate(ins) if kinds[i[4]] == "MFMA"]
    issue = [ins[k][0] + ins[k][2] for k in mp]
    cuts = [j for j in range(1, len(mp))
            if any(kinds[i[4]] == "buffer_store" for i in ins[mp[j - 1] + 1:mp[j]])]
    tiles = list(zip([0] + cuts, cuts + [len(mp)]))
    budget, idle, start, gaps, ends = Counter(), Counter(), [0] * len(mp), {}, []
    for lo, hi in tiles:
        finish = issue[lo]
        for j in range(lo, hi):
            start[j] = max(issue[j], finish)
            if j > lo and issue[j] > finish:
                gaps[j] = charge_gap(ins, mp[j - 1], mp[j], finish, issue[j], kinds)
                idle += gaps[j]
            finish = start[j] + MFMA_CYCLES
        ends.append(finish)
        budget["MFMA 在算"] += (hi - lo) * MFMA_CYCLES
    budget["K 循环里 MFMA 空闲"] = sum(idle.values())
    budget["tile 切换（epilogue + 下一 tile 开头）"] = sum(
        issue[lo2] - end for end, (lo2, _) in zip(ends, tiles[1:]))
    budget["收尾（最后一个 epilogue）"] = wave["end"] - ends[-1]
    budget["开头（第一条 MFMA 之前）"] = issue[0] - wave["begin"]
    return dict(ins=ins, mp=mp, issue=issue, start=start, tiles=tiles,
                budget=budget, idle=idle, gaps=gaps)


def tile_switch_mix(a, kinds):
    """Cycles per instruction kind inside the windows between two tiles' MFMAs."""
    mix, n = Counter(), 0
    for (_, hi), (lo2, _) in zip(a["tiles"], a["tiles"][1:]):
        n += 1
        for i in a["ins"][a["mp"][hi - 1] + 1:a["mp"][lo2]]:
            mix[kinds[i[4]]] += i[3]
    return mix, n


def hotspots(code, kernel_src=None, top=10):
    by_line = defaultdict(lambda: [0, 0, set()])
    for c in code:
        if not c[6] or not c[8]:
            continue
        f, _, ln = str(c[3]).split(" -> ")[0].rpartition(":")
        if ln.isdigit():
            b = by_line[(os.path.basename(f), int(ln))]
            b[0] += c[8]
            b[1] += c[6]
            b[2].add(c[0].split()[0])
    lines = open(kernel_src).read().split("\n") if kernel_src else []
    base = os.path.basename(kernel_src) if kernel_src else None
    rows = []
    for (f, ln), (stall, hit, ops) in sorted(by_line.items(), key=lambda kv: -kv[1][0])[:top]:
        text = lines[ln - 1].strip() if f == base and ln <= len(lines) else ""
        rows.append((f"{f}:{ln}", stall, hit, ",".join(sorted(ops)), text))
    return rows


def main():
    ui = sys.argv[1]
    code, kinds, waves = load(ui)
    res = [analyze(w, kinds) for w in waves]
    nw = len(res)
    a0 = res[0]
    stages = (a0["tiles"][0][1] - a0["tiles"][0][0]) // PER_STAGE
    print(f"{nw} 个 wave，每个 wave {len(a0['tiles'])} 个 tile × {stages} 个 stage，"
          f"{len(a0['ins']):,} 条指令，{len(a0['mp']):,} 条 MFMA\n")

    budget = sum((r["budget"] for r in res), Counter())
    life = sum(budget.values())
    print("1. 每个 wave 的时间预算（周期，wave 平均）")
    for k, v in budget.most_common():
        print(f"   {k:34s}{v / nw:>10,.0f}  {100 * v / life:5.1f}%")
    print(f"   {'合计 = wave 寿命':34s}{life / nw:>10,.0f}")

    idle = sum((r["idle"] for r in res), Counter())
    total = sum(idle.values())
    print("\n2. K 循环里 MFMA 空着的时候，wave 在做什么")
    for k, v in idle.most_common():
        print(f"   {k:34s}{v / nw:>10,.0f}  {100 * v / total:5.1f}%")

    mix, n = Counter(), 0
    for r in res:
        m, c = tile_switch_mix(r, kinds)
        mix += m
        n += c
    if n:
        win = sum(r["issue"][lo2] - r["issue"][hi - 1] for r in res
                  for (_, hi), (lo2, _) in zip(r["tiles"], r["tiles"][1:]))
        print(f"\n3. tile 切换窗口（每次平均 {win / n:,.0f} 周期）里各类指令占的周期")
        for k, v in mix.most_common(8):
            print(f"   {k:34s}{v / n:>10,.0f}  {100 * v / win:5.1f}%")

    print("\n4. stall 最多的源码行（内联链最里层）")
    for loc, stall, hit, ops, text in hotspots(code, sys.argv[2] if len(sys.argv) > 2 else None):
        print(f"   {loc:28s} stall {stall:>8,d}  hit {hit:>6,d}  {ops[:34]:34s} {text[:56]}")


if __name__ == "__main__":
    main()
