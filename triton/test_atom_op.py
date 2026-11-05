"""The ATOM integration path of the MBS GEMM against the reference.

  PYTHONPATH=/home/haoyanli/a121/lhy_atom/ATOM_mbs HIP_VISIBLE_DEVICES=4 python test_atom_op.py

Runs the weight through ATOM's own post-load steps (quant_weight_online_mbs,
shuffle_weights, e8m0_shuffle, mbs_ratios as in LinearBase) and the forward
through the registered custom op `gemm_mxfp4_mbs`, for every MBS format string
the online parser accepts, and compares with dequant(A) @ dequant(W)^T.
"""

import sys

import torch
from aiter.utility import fp4_utils

from atom.model_ops import mxfp4_mbs as mm
from atom.model_ops.linear import gemm_mxfp4_mbs, quant_weight_online_mbs
from atom.model_ops.utils import shuffle_weights
from atom.quant_spec import QuarkOnlineParser

torch.manual_seed(0)
dev = "cuda"
failed = False

for fmt in ("mxfp4_mbs", "mxfp4_mbs_a", "mxfp4_mbs_w", "mxfp4_mbs256", "mxfp4_mbs512", "mxfp4_mbs1024"):
    spec = QuarkOnlineParser().parse({"global_quant_config": fmt}).global_spec.mbs
    for M, N, K in ((1, 12352, 4096), (7, 4096, 12288), (40, 1024, 4096), (300, 2048, 4096), (2048, 4096, 4096)):
        x = (torch.randn(M, K, device=dev) * torch.exp(torch.randn(1, K, device=dev))).to(torch.bfloat16)
        w = torch.randn(N, K, device=dev).to(torch.bfloat16)
        q, s, m = quant_weight_online_mbs(w, spec)
        # shuffle_weights rewrites the parameter's storage in place
        weight = torch.nn.Parameter(q.clone(), requires_grad=False)
        shuffle_weights(weight)
        w_scale = fp4_utils.e8m0_shuffle(s)
        ratio = sigma = None
        if m is not None:
            ratio, sigma = mm.mbs_ratios(m)
        y = gemm_mxfp4_mbs(x, None, weight.data, w_scale, ratio, sigma, torch.bfloat16, spec.block, spec.act)
        qa, sa, ma = mm.mbs_quant(x, mb=spec.block, use_mbs=spec.act)
        zeros = torch.zeros(M, K // spec.block, dtype=torch.uint8, device=dev)
        ref = mm.mbs_gemm_ref((qa, sa, ma if spec.act else zeros),
                              (q.view(torch.uint8), s.view(torch.uint8), m if m is not None else zeros[:1].expand(N, -1)),
                              mb=spec.block)
        rel = ((y.float() - ref).norm() / ref.norm()).item()
        ok = rel < 4e-3 and y.shape == (M, N)
        failed |= not ok
        print(f"[{'PASS' if ok else 'FAIL'}] {fmt:14s} {spec} M={M} N={N} K={K} rel err {rel:.2e}")

print("ALL PASS" if not failed else "SOME CHECKS FAILED")
sys.exit(1 if failed else 0)
