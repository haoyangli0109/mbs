#!/usr/bin/env python3
"""Print the raw ATT records of one K step of one wave, next to its ISA and source line.

    python3 att_dump.py <ui_output dir> [--simd S] [--step K] [--mfma-per-step N] [--waits]

Takes the wave in the middle of SIMD S's wave list (default SIMD 0), cuts its
instruction stream at every N-th MFMA (default 16, one K step of v0) and prints step K
(default 10): cycle relative to the step start, stall, duration, ISA, source line.
--waits keeps only loads, s_waitcnt, s_barrier and MFMAs.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from att_summary import load  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("ui")
ap.add_argument("--simd", type=int, default=0)
ap.add_argument("--step", type=int, default=10)
ap.add_argument("--mfma-per-step", type=int, default=16)
ap.add_argument("--waits", action="store_true")
args = ap.parse_args()

code, kinds, waves = load(args.ui)
ws = sorted((w for s, w in waves if s == args.simd), key=lambda w: w["begin"])
ins = ws[len(ws) // 2]["instructions"]
mp = [k for k, i in enumerate(ins) if kinds[i[4]] == "MFMA"]
n = args.mfma_per_step
lo = mp[n * args.step - 1] + 1 if args.step else 0
hi = mp[n * (args.step + 1) - 1] + 1
t0 = ins[lo][0]
print(f"SIMD {args.simd} 中间那个 wave，第 {args.step} 步：{hi - lo} 条指令，"
      f"{ins[hi][0] - t0 if hi < len(ins) else ins[hi - 1][0] - t0:,} 周期")
print(f"{'周期':>8s} {'stall':>6s} {'dur':>6s}   指令")
for t, _, stall, dur, idx in ins[lo:hi]:
    isa = code[idx][0]
    op = isa.split()[0]
    if args.waits and not (op in ("s_waitcnt", "s_barrier") or "load" in op or op.startswith("v_mfma")):
        continue
    src = str(code[idx][3]).split(" -> ")[0].rsplit("/", 1)[-1]
    print(f"{'+' + str(t - t0):>8s} {stall:6d} {dur:6d}   {isa[:62]:62s} {src}")
