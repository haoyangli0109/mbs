#!/usr/bin/env python3
"""Where the time of every wave goes, for any kernel, from one rocprofv3 --att trace.

    python3 att_summary.py <ui_output_agent_*_dispatch_* dir> [source file ...]

Prints
  1. waves traced, waves resident per SIMD, and how busy each SIMD's MFMA pipe was
     (an FP4 16x16x128 MFMA keeps the pipe busy 16 cycles; all waves on a SIMD share it);
  2. how the waves' time splits over instruction kinds: issue + stall of each
     instruction, s_waitcnt split by the counter it waits for, and holes where the
     wave had nothing to issue;
  3. the source lines whose s_waitcnt / s_barrier waited longest;
  4. the source lines with the most stall.

Wave records are [cycle, type, stall, duration, code index]: the wave starts trying
the instruction at `cycle`, issues it after `stall` cycles and tries the next one
`duration` cycles later.
"""
import glob
import json
import os
import re
import sys
from collections import Counter, defaultdict

MFMA_CYCLES = 16


def kind(isa):
    op = isa.split()[0] if isa and not isa.startswith(";") else ""
    if op.startswith("v_mfma"):
        return "MFMA"
    if op == "s_waitcnt":
        vm, lgkm = "vmcnt" in isa, "lgkmcnt" in isa
        return ("等 vmcnt + lgkmcnt" if vm and lgkm else
                "等 vmcnt（全局内存）" if vm else "等 lgkmcnt（LDS、标量读）")
    if op == "s_barrier":
        return "等 barrier"
    for prefix, name in (("global_load", "发全局读"), ("buffer_load", "发全局读"),
                         ("global_store", "发全局写"), ("buffer_store", "发全局写"),
                         ("ds_read", "发 LDS 读"), ("ds_write", "发 LDS 写"),
                         ("s_cbranch", "分支"), ("s_branch", "分支"),
                         ("s_", "标量指令"), ("v_", "向量指令")):
        if op.startswith(prefix):
            return name
    return "其他"


def src_line(source):
    f, _, ln = str(source).split(" -> ")[0].rpartition(":")
    return (os.path.basename(f), int(ln)) if ln.isdigit() else None


def load(ui):
    code = json.load(open(os.path.join(ui, "code.json")))["code"]
    waves = []
    for p in sorted(glob.glob(os.path.join(ui, "se*_wv*.json"))):
        simd = int(re.search(r"_sm(\d+)_", os.path.basename(p)).group(1))
        waves.append((simd, json.load(open(p))["wave"]))
    return code, [kind(c[0]) for c in code], waves


def wave_time(wave, kinds):
    """Cycles of one wave per kind; holes between instructions count as '空档'."""
    t = Counter()
    ins = wave["instructions"]
    for a, b in zip(ins, ins[1:] + [None]):
        t[kinds[a[4]]] += a[3]
        if b is not None and b[0] > a[0] + a[3]:
            t["空档（没有指令可发）"] += b[0] - a[0] - a[3]
    t["开头和结尾"] += (wave["end"] - wave["begin"]) - sum(t.values())
    return t


def simd_mfma(waves, kinds):
    """Per SIMD: (waves, max resident at once, MFMA count, active span, busy fraction)."""
    out = {}
    for simd in sorted({s for s, _ in waves}):
        ws = [w for s, w in waves if s == simd]
        issues = sorted(i[0] + i[2] for w in ws for i in w["instructions"] if kinds[i[4]] == "MFMA")
        busy, finish = 0, -1
        for t in issues:
            start = max(t, finish)
            busy += MFMA_CYCLES
            finish = start + MFMA_CYCLES
        span = max(w["end"] for w in ws) - min(w["begin"] for w in ws)
        edges = sorted([(w["begin"], 1) for w in ws] + [(w["end"], -1) for w in ws])
        cur = peak = 0
        for _, d in edges:
            cur += d
            peak = max(peak, cur)
        out[simd] = (len(ws), peak, len(issues), span, busy / span)
    return out


def main():
    ui = sys.argv[1]
    sources = {os.path.basename(p): open(p).read().split("\n") for p in sys.argv[2:]}
    code, kinds, waves = load(ui)

    print("1. 每个 SIMD 的 MFMA 管线")
    for simd, (n, peak, nm, span, frac) in simd_mfma(waves, kinds).items():
        print(f"   SIMD {simd}: {n} 个 wave（最多同时 {peak} 个），{nm:,} 条 MFMA，"
              f"活跃 {span:,} 周期，MFMA 忙 {100 * frac:.1f}%")

    total = Counter()
    for _, w in waves:
        total += wave_time(w, kinds)
    life = sum(total.values())
    print(f"\n2. wave 的时间花在哪（{len(waves)} 个 wave 合计，平均每个 wave {life / len(waves):,.0f} 周期）")
    for k, v in total.most_common():
        print(f"   {k:26s}{v / len(waves):>12,.0f}  {100 * v / life:5.1f}%")

    def text(key):
        lines = sources.get(key[0])
        return lines[key[1] - 1].strip()[:64] if lines and key[1] <= len(lines) else ""

    waits = defaultdict(lambda: [0, 0])
    stalls = defaultdict(lambda: [0, 0, set()])
    for c, k in zip(code, kinds):
        key = src_line(c[3])
        if not c[6] or key is None:
            continue
        if k.startswith("等"):
            waits[(k, key)][0] += c[8]
            waits[(k, key)][1] += c[6]
        else:
            stalls[key][0] += c[8]
            stalls[key][1] += c[6]
            stalls[key][2].add(c[0].split()[0])
    print("\n3. 等得最久的 s_waitcnt / s_barrier（按源码行）")
    for (k, key), (cyc, hit) in sorted(waits.items(), key=lambda kv: -kv[1][0])[:6]:
        print(f"   {k:22s} {key[0]}:{key[1]:<5d} {cyc / len(waves):>10,.0f} 周期/wave  {text(key)}")
    print("\n4. stall 最多的其他指令（按源码行）")
    for key, (st, hit, ops) in sorted(stalls.items(), key=lambda kv: -kv[1][0])[:6]:
        print(f"   {key[0]}:{key[1]:<5d} {st / len(waves):>10,.0f} 周期/wave  "
              f"{','.join(sorted(ops))[:40]:40s} {text(key)}")


if __name__ == "__main__":
    main()
