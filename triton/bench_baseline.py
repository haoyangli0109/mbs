"""Baseline MXFP4 GEMMs on MI355X: aiter asm `gemm_a4w4` vs aiter Triton.

Usage: HIP_VISIBLE_DEVICES=4 python bench_baseline.py [--m 32 256 1024 8192]
"""

import argparse
import json
import os

import torch

import aiter
from aiter.ops.shuffle import shuffle_weight
from aiter.ops.triton.gemm.basic.gemm_afp4wfp4 import (
    gemm_afp4wfp4,
    gemm_afp4wfp4_preshuffle,
)

from bench_utils import QWEN35_9B_LINEARS, graph_time_us, tflops

torch.set_default_device("cuda")


def make_inputs(M, N, K):
    quant = aiter.get_triton_quant(aiter.QuantType.per_1x32)
    x = torch.randn(M, K, dtype=torch.bfloat16)
    w = torch.randn(N, K, dtype=torch.bfloat16)
    xq, xs = quant(x, shuffle=False)
    wq, ws = quant(w, shuffle=False)
    _, xs_sh = quant(x, shuffle=True)
    _, ws_sh = quant(w, shuffle=True)
    return dict(
        x=x, xq=xq, xs=xs, wq=wq, ws=ws, xs_sh=xs_sh, ws_sh=ws_sh,
        w_sh=shuffle_weight(wq, layout=(16, 16)),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", type=int, nargs="*", default=[32, 256, 1024, 8192])
    ap.add_argument("--out", default="results/baseline.json")
    args = ap.parse_args()
    rows = []
    for M in args.m:
        for name, N, K in QWEN35_9B_LINEARS:
            d = make_inputs(M, N, K)
            Mpad = (M + 31) // 32 * 32
            y_asm = torch.empty(Mpad, N, dtype=torch.bfloat16)
            asm = lambda: aiter.gemm_a4w4(d["xq"], d["w_sh"], d["xs_sh"], d["ws_sh"], y_asm)
            u8 = lambda t: t.view(torch.uint8)
            tri = lambda: gemm_afp4wfp4(u8(d["xq"]), u8(d["wq"]), u8(d["xs"]), u8(d["ws"]))
            xs_p = u8(d["xs_sh"]).view(-1, d["xs_sh"].shape[-1] * 32) if M >= 32 else u8(d["xs"])
            ws_p = u8(d["ws_sh"]).view(-1, d["ws_sh"].shape[-1] * 32)
            w_p = u8(d["w_sh"]).view(N // 16, -1)
            tri_ps = lambda: gemm_afp4wfp4_preshuffle(u8(d["xq"]), w_p, xs_p, ws_p)
            ref = asm()[:M].float()
            row = dict(M=M, N=N, K=K, layer=name)
            for tag, fn in (("asm", asm), ("triton", tri), ("triton_preshuffle", tri_ps)):
                out = fn()[:M].float()
                err = ((out - ref).norm() / ref.norm()).item()
                us, _ = graph_time_us(fn)
                row[f"{tag}_us"] = round(us, 2)
                row[f"{tag}_tflops"] = round(tflops(M, N, K, us), 1)
                row[f"{tag}_relerr_vs_asm"] = err
            print(json.dumps(row), flush=True)
            rows.append(row)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(rows, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
