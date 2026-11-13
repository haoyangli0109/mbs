#!/usr/bin/env python3
"""Summarize rocprofv3 --pmc CSVs of one GEMM kernel and check them against M, N, K.

    rocprofv3 -i scripts/pmc_mbs.txt --kernel-include-regex gemm_a4w4_wg_kernel \
        --output-format csv -d /tmp/mbs_pmc -o pmc -- python bench_one.py 8192 8192 5120 5
    python3 scripts/pmc_summary.py /tmp/mbs_pmc 8192 8192 5120

Every counter is summed over the whole GPU for one dispatch; the median over all
dispatches is printed. MI355X has 8 XCDs with one GRBM each, so the GPU-busy cycle
count is GRBM_GUI_ACTIVE / 8. SQ_WAVE_CYCLES, SQ_WAIT_ANY and SQ_WAIT_INST_ANY count in
units of 4 cycles (rocprof-compute multiplies them by 4 as well).
"""
import csv
import glob
import statistics
import sys
from collections import defaultdict

XCDS = 8

root = sys.argv[1]
M, N, K = (int(v) for v in sys.argv[2:5])

vals, durs = defaultdict(list), []
meta = {}
for path in sorted(glob.glob(f"{root}/**/*counter_collection.csv", recursive=True)):
    for r in csv.DictReader(open(path)):
        vals[r["Counter_Name"]].append(float(r["Counter_Value"]))
        durs.append((int(r["End_Timestamp"]) - int(r["Start_Timestamp"])) / 1e3)
        meta = meta or r
med = {k: statistics.median(v) for k, v in vals.items()}
us = statistics.median(durs)

print(f"kernel   {meta['Kernel_Name'][:90]}")
print(f"grid {meta['Grid_Size']}  block {meta['Workgroup_Size']}  LDS {meta['LDS_Block_Size']} B  "
      f"scratch {meta['Scratch_Size']}  (VGPR/AGPR 字段在 gfx950 上拆分不准，以编译器报告为准)")
print(f"每次 dispatch 中位耗时 {us:.1f} us（采计数器时 dispatch 被串行化，比连续跑时快）\n")
for k in sorted(med):
    print(f"  {k:30s}{med[k]:>18,.1f}   (n={len(vals[k])})")


def show(name, value, note=""):
    print(f"  {name:30s}{value:>18s}   {note}")


print("\n派生量")
g = med.get("GRBM_GUI_ACTIVE")
if g:
    show("GPU 忙周期 (GRBM/8)", f"{g / XCDS:,.0f}", f"时钟约 {g / XCDS / us / 1e3:.2f} GHz")
if "SQ_WAVE_CYCLES" in med and "SQ_WAVES" in med:
    waves = med["SQ_WAVES"]
    life = 4 * med["SQ_WAVE_CYCLES"] / waves
    show("每个 wave 的寿命", f"{life:,.0f}", f"周期，{waves:.0f} 个 wave")
    for name, key, note in (("每个 wave 等依赖", "SQ_WAIT_ANY", "s_waitcnt 等数据、s_barrier 等其他 wave"),
                            ("每个 wave 等发射", "SQ_WAIT_INST_ANY", "指令备好了，执行单元接不住")):
        if key in med:
            cyc = 4 * med[key] / waves
            show(name, f"{cyc:,.0f}", f"周期（{key} x 4），占寿命 {100 * cyc / life:.0f}%，{note}")
if "SQ_INSTS_MFMA" in med:
    show("MFMA 条数 / 理论", f"{med['SQ_INSTS_MFMA'] / (M * N * K / (16 * 16 * 128)):.4f}",
         "理论 = M*N*K / (16*16*128)")
if "SQ_INSTS_VALU_MFMA_MOPS_F6F4" in med:
    show("F6F4 MOPS / 理论", f"{med['SQ_INSTS_VALU_MFMA_MOPS_F6F4'] / (2 * M * N * K / 512):.4f}",
         "理论 = 2*M*N*K / 512，等于 1 说明全部 MFMA 都按 FP4 算")
if "TCC_HIT_sum" in med and "TCC_MISS_sum" in med:
    show("L2 命中率", f"{100 * med['TCC_HIT_sum'] / (med['TCC_HIT_sum'] + med['TCC_MISS_sum']):.1f}%")
if "TCC_EA0_WRREQ_sum" in med:
    show("L2 往显存的写请求 x 64 B", f"{med['TCC_EA0_WRREQ_sum'] * 64 / 2**20:,.1f} MiB",
         f"输出 C 是 {M * N * 2 / 2**20:,.1f} MiB")
if "MfmaUtil" in med:
    show("MfmaUtil", f"{med['MfmaUtil']:.1f}%", "MFMA 忙周期 / (GPU 忙周期 x SIMD 数)")
