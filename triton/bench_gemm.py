"""Kernel-level timing: the Triton MBS GEMM (tuned configs) against MXFP4 baselines.

  HIP_VISIBLE_DEVICES=4 python bench_gemm.py --m 1 16 64 256 1024 4096 8192 --out results/bench_gemm.json

Per shape and M, CUDA-graph timed (hot L2):
  asm          aiter gemm_a4w4 (what ATOM's `mxfp4` runs), GEMM only
  asm+q        + aiter's HIP MXFP4 activation quantization (ATOM's online path)
  aiter_tri    aiter Triton gemm_afp4wfp4_preshuffle, GEMM only (M >= 32)
  <variant>    this kernel, GEMM only, and <variant>+q with quant_activation
"""

import argparse
import json
import os

import torch

import aiter
from aiter.ops.triton.gemm.basic.gemm_afp4wfp4 import gemm_afp4wfp4_preshuffle

import mxfp4_mbs as mm
from bench_utils import QWEN35_9B_LINEARS, graph_time_us, tflops
from tune_gemm import VARIANTS

torch.set_default_device("cuda")

SHAPES = QWEN35_9B_LINEARS[:4] + [("gdn.in_proj_qkvzba", 12352, 4096)] + QWEN35_9B_LINEARS[4:]


def run_shape(M, N, K, variants):
    x = torch.randn(M, K, dtype=torch.bfloat16)
    w = torch.randn(N, K, dtype=torch.bfloat16)
    row = dict(M=M, N=N, K=K)
    f4, e8 = torch.float4_e2m1fn_x2, torch.float8_e8m0fnu
    P = mm.prepare_weight(w, use_mbs=False)
    A = mm.quant_activation(x, use_mbs=False, shuffle=True)
    y = torch.empty((M + 31) // 32 * 32, N, dtype=torch.bfloat16)
    hip_quant = aiter.get_hip_quant(aiter.QuantType.per_1x32)
    asm = lambda: aiter.gemm_a4w4(A["q"].view(f4), P["q"].view(f4), A["s"].view(e8), P["s"].view(e8), y)

    def asm_q():
        xq, xs = hip_quant(x, quant_dtype=aiter.dtypes.fp4x2, shuffle=True)
        return aiter.gemm_a4w4(xq, P["q"].view(f4), xs, P["s"].view(e8), y)

    row["asm_us"], _ = graph_time_us(asm)
    row["asm+q_us"], _ = graph_time_us(asm_q)
    if M >= 32:
        tri = lambda: gemm_afp4wfp4_preshuffle(
            A["q"], P["q"].view(N // 16, -1), A["s"].view(-1, A["s"].shape[-1] * 32),
            P["s"].view(-1, P["s"].shape[-1] * 32))
        row["aiter_tri_us"], _ = graph_time_us(tri)
    for v in variants:
        mb, a_mbs, b_mbs = VARIANTS[v]
        W = mm.prepare_weight(w, mb=mb, ncand=16, use_mbs=b_mbs)
        Av = mm.quant_activation(x, mb=mb, use_mbs=a_mbs)
        cfg = mm.select_config(M, N, K, mb, a_mbs, b_mbs)
        row[f"{v}_us"], _ = graph_time_us(lambda: mm.mbs_gemm(Av, W, mb=mb, config=cfg))
        row[f"{v}+q_us"], _ = graph_time_us(lambda: mm.mbs_linear(x, W, mb=mb, act_mbs=a_mbs))
    for k in [k for k in row if k.endswith("_us")]:
        row[k] = round(row[k], 2)
        row[k.replace("_us", "_tflops")] = round(tflops(M, N, K, row[k]), 1)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", type=int, nargs="*", default=[1, 16, 64, 256, 1024, 4096, 8192])
    ap.add_argument("--variants", nargs="*", default=["plain", "mbs128", "mbs128_a", "mbs256", "mbs512"])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    rows = []
    for M in args.m:
        for name, N, K in SHAPES:
            row = run_shape(M, N, K, args.variants)
            row["layer"] = name
            print(json.dumps(row), flush=True)
            rows.append(row)
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        json.dump(rows, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
