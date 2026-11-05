"""Compile one GEMM variant/config and summarize its K loop in AMDGCN: the
instruction mix, register use, and the MFMA / VALU interleaving.

  HIP_VISIBLE_DEVICES=4 python dump_asm.py mbs128 '{"BLOCK_M":128,"BLOCK_N":256,"BLOCK_K":256,
      "num_warps":4,"waves_per_eu":1,"num_stages":2,"matrix_instr_nonkdim":32,"GROUP_M":4}' [--save DIR]

Sequence letters: M MFMA, V v_pk_mul, L buffer_load, R ds_read, W ds_write,
w s_waitcnt, | s_barrier.
"""

import argparse
import collections
import json
import os
import re

import torch

import mxfp4_mbs as mm
from bench_utils import graph_time_us, tflops
from tune_gemm import VARIANTS

PREFIXES = ("v_mfma", "buffer_load", "global_load", "ds_read", "ds_write", "v_accvgpr", "s_waitcnt",
            "s_barrier", "v_pk_mul", "v_pk_fma", "v_pk_", "v_fma", "v_mul_f32", "v_cvt", "v_perm",
            "v_mov", "v_rcp", "v_div", "scratch", "s_")
SEQ = {"v_mfma": "M", "v_pk_mul": "V", "buffer_load": "L", "ds_read": "R", "ds_write": "W",
       "s_waitcnt": "w", "s_barrier": "|"}


def loop_body(asm):
    """Lines of the largest block that ends in a backward branch (the K loop)."""
    lines = asm.splitlines()
    labels = {m.group(1): i for i, l in enumerate(lines) if (m := re.match(r"^(\.LBB\S+):", l))}
    best = []
    for i, l in enumerate(lines):
        m = re.search(r"s_cbranch_\w+\s+(\.LBB\S+)", l)
        if m and m.group(1) in labels and labels[m.group(1)] < i and i - labels[m.group(1)] > len(best):
            best = lines[labels[m.group(1)] : i + 1]
    return best


def summarize(asm):
    cnt, seq = collections.Counter(), []
    for l in loop_body(asm):
        t = l.strip().split()
        if not t or t[0].startswith((";", ".", "//")):
            continue
        cnt[next((p + "*" for p in PREFIXES if t[0].startswith(p)), t[0])] += 1
        seq.append(next((c for p, c in SEQ.items() if t[0].startswith(p)), ""))
    meta = dict(re.findall(r"\.(vgpr_count|agpr_count|vgpr_spill_count|sgpr_spill_count):\s+(\d+)", asm))
    return cnt, meta, "".join(seq)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("variant", choices=list(VARIANTS))
    ap.add_argument("config", help="JSON tile config")
    ap.add_argument("--shape", default="8192,12288,4096", help="M,N,K")
    ap.add_argument("--save", default=None)
    args = ap.parse_args()
    torch.set_default_device("cuda")
    mb, a_mbs, b_mbs = VARIANTS[args.variant]
    M, N, K = (int(t) for t in args.shape.split(","))
    x = torch.randn(M, K, dtype=torch.bfloat16)
    w = torch.randn(N, K, dtype=torch.bfloat16)
    A = mm.quant_activation(x, mb=mb, use_mbs=a_mbs)
    W = mm.prepare_weight(w, mb=mb, ncand=1, use_mbs=b_mbs)
    cfg = json.loads(args.config)
    us, _ = graph_time_us(lambda: mm.mbs_gemm(A, W, mb=mb, config=cfg))
    kern = list(mm._mbs_gemm_kernel.device_caches[0][0].values())[-1]
    cnt, meta, seq = summarize(kern.asm["amdgcn"])
    print(f"{args.variant} M={M} N={N} K={K}: {us:.1f} us, {tflops(M, N, K, us):.0f} TFLOPS, {meta}")
    print("  " + "  ".join(f"{k}={v}" for k, v in sorted(cnt.items(), key=lambda kv: -kv[1]) if v >= 2))
    print("  " + seq)
    if args.save:
        os.makedirs(args.save, exist_ok=True)
        for ext in ("amdgcn", "ttgir"):
            open(os.path.join(args.save, f"{args.variant}.{ext}"), "w").write(kern.asm[ext])


if __name__ == "__main__":
    main()
