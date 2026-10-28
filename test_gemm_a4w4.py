# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import argparse
import os
import sys

import pandas as pd
import torch

# Make the freshly built mbs_gemm_a4w4 extension (in this directory) importable.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "build"))

import aiter
from aiter import dtypes
from aiter.ops.shuffle import shuffle_weight
from aiter.test_common import benchmark, checkAllclose, perftest, run_perftest
from aiter.utility import fp4_utils

torch.set_default_device("cuda")
torch.set_printoptions(sci_mode=False)
SCALE_GROUP_SIZE = 32
pd.set_option("display.max_columns", 30)
pd.set_option("display.width", 1000)
pd.set_option("display.max_colwidth", 30)


@perftest(num_iters=5)
def run_torch(x, w, x_scales, w_scales, dtype):
    m, _k = x.shape
    n, _k = w.shape
    # First convert the x and w inputs to f32.
    x_f32 = fp4_utils.mxfp4_to_f32(x)
    w_f32 = fp4_utils.mxfp4_to_f32(w)
    # Next convert the e8m0 scales to f32.
    x_scales = x_scales[:m]
    x_scales = x_scales.repeat_interleave(SCALE_GROUP_SIZE, dim=1)
    x_scales_f32 = fp4_utils.e8m0_to_f32(x_scales)
    x_f32 = x_f32 * x_scales_f32
    w_scales = w_scales[:n]
    w_scales = w_scales.repeat_interleave(SCALE_GROUP_SIZE, dim=1)
    w_scales_f32 = fp4_utils.e8m0_to_f32(w_scales)
    w_f32 = w_f32 * w_scales_f32
    return torch.mm(x_f32, w_f32.T).to(dtype)[:m, :n]


@perftest()
def run_gemm_ck(x, weight, x_scale, w_scale, out):
    return aiter.gemm_a4w4_blockscale(x, weight, x_scale, w_scale, out)


@perftest()
def run_mbs(x, wshuffle, x_scale_shuffle, w_scale_shuffle):
    import mbs_gemm_a4w4

    return mbs_gemm_a4w4.gemm_a4w4(
        x, wshuffle, x_scale_shuffle, w_scale_shuffle
    )


@perftest()
def run_triton(x, w, x_scales, w_scales, out, dtype=dtypes.bf16):
    from aiter.ops.triton.gemm.basic.gemm_afp4wfp4 import gemm_afp4wfp4

    gemm_afp4wfp4(x, w, x_scales, w_scales, dtype, out)
    return out


@perftest()
def run_gemm_asm(
    x,
    weightshuffle,
    x_scale,
    w_scale,
    out,
    kernelName="",
    bias=None,
    dtype=dtypes.bf16,
    bpreshuffle=True,
    log2_k_split=None,
):
    # if log2_k_split is not None and log2_k_split > 0:
    #     out_reset = torch.zeros(
    #         (out.shape[0] + 31) // 32 * 32, out.shape[1], dtype=dtype
    #     )
    #     out = out_reset

    aiter.gemm_a4w4_asm(
        x,
        weightshuffle,
        x_scale,
        w_scale,
        out,
        kernelName,
        bias,
        bpreshuffle=bpreshuffle,
        log2_k_split=log2_k_split,
    )
    return out


@benchmark()
def test_gemm(dtype, M, N, K):
    from aiter.jit.utils.chip_info import get_gfx_runtime as get_gfx

    if get_gfx() not in ["gfx950"]:
        return
    ret = {}
    quant_func = aiter.get_triton_quant(aiter.QuantType.per_1x32)
    x = torch.randn((M, K), dtype=dtype)
    w = torch.randn((N, K), dtype=dtype)
    _, x_scales = quant_func(x, shuffle=False)
    _, w_scales = quant_func(w, shuffle=False)
    x, x_scales_shuffle = quant_func(x, shuffle=True)
    w, w_scales_shuffle = quant_func(w, shuffle=True)
    wshuffle = shuffle_weight(w, layout=(16, 16))
    x_scales = x_scales.view(torch.uint8)
    w_scales = w_scales.view(torch.uint8)
    a, _avg_a = run_torch(x, w, x_scales, w_scales, dtype)
    # out1 = torch.empty(M, N, dtype=dtype)
    # b, avg_b = run_triton(x, w.T, x_scales, w_scales, out1, dtype)
    # b, avg_b = a, 0
    # err_b = checkAllclose(a, b, msg="triton        ")

    c, us = run_perftest(
        aiter.gemm_a4w4,
        x,
        wshuffle,
        x_scales_shuffle,
        w_scales_shuffle,
        bpreshuffle=True,
    )
    # Non-fatal: the asm kernel is catastrophically wrong on some shapes
    # (e.g. N=48); record its error but let the sweep finish and print the table.
    err = checkAllclose(a, c, msg="unified api", catastrophic_check=False)
    ret["us"] = us
    ret["TFLOPS"] = M * N * K * 2 / us / 1e6
    ret["TB/s"] = (x.nbytes + w.nbytes) / us / 1e6
    ret["err"] = err

    # mbs: standalone HIP MXFP4 A4W4 GEMM (same shuffled-input contract).
    f, us = run_mbs(x, wshuffle, x_scales_shuffle, w_scales_shuffle)
    err = checkAllclose(a, f, msg="mbs hip", catastrophic_check=True)
    ret["us mbs"] = us
    ret["TFLOPS mbs"] = M * N * K * 2 / us / 1e6
    ret["TB/s mbs"] = (x.nbytes + w.nbytes) / us / 1e6
    ret["err mbs"] = err

    # kernelName = "" # "_ZN5aiter42f4gemm_bf16_per1x32Fp4_BpreShuffle_128x512E"
    # log2_k_split = 1
    # out2 = torch.empty((M + 31) // 32 * 32, N, dtype=dtype)
    # d, us = run_gemm_asm(
    #     x,
    #     wshuffle,
    #     x_scales_shuffle,
    #     w_scales_shuffle,
    #     out2,
    #     kernelName,
    #     bias_f32,
    #     bpreshuffle=True,
    #     log2_k_split=log2_k_split,
    # )
    # err = checkAllclose(a, d[:M], msg=f"asm {kernelName} log2_k_split_{log2_k_split}")
    # tag = "asm_dbg"
    # ret[f"us {tag}"] = us
    # ret[f"TFLOPS {tag}"] = M * N * K * 2 / us / 1e6
    # ret[f"TB/s {tag}"] = (x.nbytes + w.nbytes) / us / 1e6
    # ret[f"err {tag}"] = err

    # out3 = torch.empty((M + 31) // 32 * 32, N, dtype=dtype)
    # e, us = run_gemm_ck(x, wshuffle, x_scales_shuffle, w_scales_shuffle, out3)
    # err = checkAllclose(a, e[:M], msg="ck            ")
    # tag = "ck"
    # ret[f"us {tag}"] = us
    # ret[f"TFLOPS {tag}"] = M * N * K * 2 / us / 1e6
    # ret[f"TB/s {tag}"] = (x.nbytes + w.nbytes) / us / 1e6
    # ret[f"err {tag}"] = err

    return ret


parser = argparse.ArgumentParser(
    formatter_class=argparse.RawTextHelpFormatter,
    description="config input of test",
)
parser.add_argument(
    "-d",
    "--dtype",
    type=dtypes.str2Dtype,
    nargs="*",
    choices=[dtypes.d_dtypes["bf16"]],
    metavar="{bf16}",
    default=[dtypes.d_dtypes["bf16"]],
    help="""Data type.
    e.g.: -d bf16""",
)
parser.add_argument(
    "-mnk",
    "--shape",
    type=dtypes.str2tuple,
    nargs="*",
    default=[
        # Target model projection shapes, as (M, N, K) with N=out, K=in.
        # Swept at M=1024 and M=8192. Modules sharing an (N, K) are listed once.
        # --- M = 1024 ---
        (1024, 17408, 5120),  # mlp.gate_proj, mlp.up_proj
        (1024, 5120, 17408),  # mlp.down_proj
        (1024, 10240, 5120),  # linear_attn.in_proj_qkv
        (1024, 6144, 5120),   # linear_attn.in_proj_z
        (1024, 5120, 6144),   # linear_attn.out_proj, self_attn.o_proj
        (1024, 12288, 5120),  # self_attn.q_proj
        (1024, 1024, 5120),   # self_attn.k_proj, self_attn.v_proj
        # # --- M = 8192 ---
        (8192, 17408, 5120),  # mlp.gate_proj, mlp.up_proj
        (8192, 5120, 17408),  # mlp.down_proj
        (8192, 10240, 5120),  # linear_attn.in_proj_qkv
        (8192, 6144, 5120),   # linear_attn.in_proj_z
        (8192, 5120, 6144),   # linear_attn.out_proj, self_attn.o_proj
        (8192, 12288, 5120),  # self_attn.q_proj
        (8192, 1024, 5120),   # self_attn.k_proj, self_attn.v_proj
        # # linear_attn.in_proj_a, in_proj_b (N=48). Kept LAST: the asm kernel
        # # (unified api) is catastrophically wrong at N=48 and the accuracy check
        # # raises, aborting the run -- so every other shape finishes first.
        # (1024, 48, 5120),
        # (8192, 48, 5120),
    ],
    help="""Shape of mnk.
    e.g. -mnk 1280,8192,1024""",
)

args = parser.parse_args()

df = []
for dtype in args.dtype:
    for m, n, k in args.shape:
        ret = test_gemm(dtype, m, n, k)
        df.append(ret)
df = pd.DataFrame(df)
df_md = df.to_markdown(index=False)
aiter.logger.info("gemm_a4w4 summary (markdown):\n%s", df_md)
