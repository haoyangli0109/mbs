"""Every tuned GEMM config ATOM can pick, against the fp32 reference.

  PYTHONPATH=/home/haoyanli/a121/lhy_atom/ATOM_mbs HIP_VISIBLE_DEVICES=4 python test_configs.py
  ... python test_configs.py --shapes 16480,5120 5120,6144   # other models' (N, K)

`select_config` picks a config per (variant, N, K, M bucket) from
mbs_gemm_configs.json, and the tuner only timed them. This runs each MBS format
string through ATOM's weight post-processing and the `gemm_mxfp4_mbs` op (as
test_atom_op.py) on the Qwen3.5-9B shapes of the table, for every M from 1 to
160 (decode at concurrency <= 160) and both sides of every larger bucket edge.

Checks: the relative error of the whole output, of its worst row, of its worst
column when M >= 32 (with fewer rows a column's norm can be a cancellation-small
single number), and the worst element against its typical size
|a_i| |b_j| / sqrt(K) of the dequantized operands: bf16 rounding stays near
1e-2 there, a block left out or applied twice is ~0.2.
"""

import argparse
import sys

import torch
from aiter.utility import fp4_utils

from atom.model_ops import mxfp4_mbs as mm
from atom.model_ops.linear import gemm_mxfp4_mbs, quant_weight_online_mbs
from atom.model_ops.utils import shuffle_weights
from atom.quant_spec import QuarkOnlineParser

FORMATS = ("mxfp4_mbs", "mxfp4_mbs_a", "mxfp4_mbs_w", "mxfp4_mbs256", "mxfp4_mbs256_a", "mxfp4_mbs512", "mxfp4_mbs1024")
SHAPES = ((12352, 4096), (12288, 4096), (10240, 4096), (24576, 4096), (4096, 4096), (4096, 12288))
MS = list(range(1, 161)) + [255, 256, 257, 511, 512, 513, 1023, 1024, 1025, 2047, 2048, 2049, 4095, 4096, 4097,
                            8191, 8192, 8193, 16383, 16384]

ap = argparse.ArgumentParser()
ap.add_argument("--shapes", nargs="*", default=None, help="N,K pairs (default: Qwen3.5-9B's)")
args = ap.parse_args()
if args.shapes:
    SHAPES = tuple(tuple(int(v) for v in s.split(",")) for s in args.shapes)

torch.manual_seed(0)
dev = "cuda"
failed = []
worst = (0.0, None)

# The element check must see one macro block of K dropped from an output.
_a = torch.randn(64, 4096, device=dev) * torch.exp(torch.randn(1, 4096, device=dev))
_b = torch.randn(4096, 4096, device=dev)
_drop = (_a[:, :128] @ _b[:, :128].T).abs() / (_a.norm(dim=1)[:, None] * _b.norm(dim=1)[None, :] / 4096**0.5)
assert _drop.max().item() > 5e-2, _drop.max().item()
print(f"element check: a dropped 128-K block shows as up to {_drop.max().item():.2f} (threshold 5e-2)")
del _a, _b, _drop

for fmt in FORMATS:
    spec = QuarkOnlineParser().parse({"global_quant_config": fmt}).global_spec.mbs
    for N, K in SHAPES:
        w = torch.randn(N, K, device=dev).to(torch.bfloat16)
        q, s, m = quant_weight_online_mbs(w, spec)
        weight = torch.nn.Parameter(q.clone(), requires_grad=False)
        shuffle_weights(weight)
        w_scale = fp4_utils.e8m0_shuffle(s)
        ratio = sigma = None
        if m is not None:
            ratio, sigma = mm.mbs_ratios(m)
        zeros_w = torch.zeros(N, K // spec.block, dtype=torch.uint8, device=dev)
        b_deq = mm.mbs_dequant(q.view(torch.uint8), s.view(torch.uint8), m if m is not None else zeros_w, mb=spec.block)
        b_norm = b_deq.norm(dim=1)
        x_all = (torch.randn(max(MS), K, device=dev) * torch.exp(torch.randn(1, K, device=dev))).to(torch.bfloat16)
        for M in MS:
            x = x_all[:M]
            y = gemm_mxfp4_mbs(x, None, weight.data, w_scale, ratio, sigma, torch.bfloat16, spec.block, spec.act)
            qa, sa, ma = mm.mbs_quant(x, mb=spec.block, use_mbs=spec.act)
            if ma is None:
                ma = torch.zeros(M, K // spec.block, dtype=torch.uint8, device=dev)
            a_deq = mm.mbs_dequant(qa, sa, ma, mb=spec.block)
            ref = a_deq @ b_deq.T
            err = y.float() - ref
            rel = (err.norm() / ref.norm()).item()
            row = (err.norm(dim=1) / ref.norm(dim=1)).max().item()
            col = (err.norm(dim=0) / ref.norm(dim=0)).max().item() if M >= 32 else 0.0
            typical = a_deq.norm(dim=1)[:, None] * b_norm[None, :] / K**0.5
            elem = (err.abs() / typical).max().item()
            finite = bool(torch.isfinite(y).all())
            ok = finite and rel < 4e-3 and row < 1e-2 and col < 1e-2 and elem < 5e-2 and y.shape == (M, N)
            if elem > worst[0]:
                worst = (elem, (fmt, M, N, K))
            if not ok:
                cfg = mm.select_config(M, N, K, spec.block, spec.act, ratio is not None)
                failed.append((fmt, M, N, K, rel, row, col, elem, finite, cfg))
                print(f"[FAIL] {fmt} M={M} N={N} K={K} rel {rel:.2e} worst row {row:.2e} col {col:.2e} "
                      f"elem {elem:.2e} finite {finite} cfg {cfg}", flush=True)
            del y, ref, err, a_deq, typical
        print(f"{fmt:15s} N={N:5d} K={K:5d}: {len(MS)} M values done, failures so far {len(failed)}", flush=True)
        del b_deq, b_norm, x_all, weight, w_scale, ratio, sigma
        torch.cuda.empty_cache()

print(f"worst element error / typical size {worst[0]:.2e} at {worst[1]}")
print("ALL PASS" if not failed else f"{len(failed)} FAILURES")
sys.exit(1 if failed else 0)
