#!/usr/bin/env python3
"""Read the MBS_TIMING per-phase cycle buckets of gemm_a4w4_wg_kernel.

    make gemm BUILD_DIR=/tmp/mbs_timing HIPCC="hipcc -DMBS_TIMING=1"
    HIP_VISIBLE_DEVICES=4 python3 scripts/mbs_timing.py /tmp/mbs_timing 8192 8192 5120 [iters]

The buckets (mbs_timing in kernel/gemm_a4w4.hip) are s_memtime cycles summed over
all waves of all timed launches; they are printed per wave per launch. Every
s_memtime read is followed by s_waitcnt lgkmcnt(0), which also drains in-flight LDS
reads, so the instrumented kernel runs about 10% slower than the plain one: compare
the buckets with each other, not with untimed runs.
"""
import ctypes
import os
import sys

import torch

build = os.path.abspath(sys.argv[1])
M, N, K = (int(v) for v in sys.argv[2:5])
iters = int(sys.argv[5]) if len(sys.argv) > 5 else 20
sys.path.insert(0, build)
import aiter  # noqa: E402
from aiter.ops.shuffle import shuffle_weight  # noqa: E402
import mbs_gemm_a4w4  # noqa: E402

NAMES = ["段 0 等待", "段 0 主体", "段 1 等待 + barrier", "段 1 主体", "epilogue",
         "tile 第 1 步的段 0 等待", "tile 第 2 步的段 0 等待", "这两步的段 1 等待"]

torch.set_default_device("cuda")
quant = aiter.get_triton_quant(aiter.QuantType.per_1x32)
x, xs = quant(torch.randn((M, K), dtype=torch.bfloat16), shuffle=True)
w, ws = quant(torch.randn((N, K), dtype=torch.bfloat16), shuffle=True)
w = shuffle_weight(w, layout=(16, 16))
lib = ctypes.CDLL(os.path.join(build, "libmbs_gemm_kernel.so"))
buf = (ctypes.c_ulonglong * 16)()

for _ in range(10):
    mbs_gemm_a4w4.gemm_a4w4(x, w, xs, ws)
torch.cuda.synchronize()
lib.mbs_timing_read(buf, 1)
for _ in range(iters):
    mbs_gemm_a4w4.gemm_a4w4(x, w, xs, ws)
torch.cuda.synchronize()
lib.mbs_timing_read(buf, 0)

waves = buf[8]
print(f"{waves} 个 wave（{iters} 次 launch 合计），下面是每个 wave 每次 launch 的周期")
for name, cycles in zip(NAMES, buf[:8]):
    print(f"  {name:24s}{cycles / waves:>12,.0f}")
