"""Tune the MBS GEMM's tile config per (variant, M bucket, N, K) and write the
table `mbs_gemm_configs.json` that `mxfp4_mbs.select_config` reads.

  HIP_VISIBLE_DEVICES=4 python tune_gemm.py --variants plain mbs128 mbs256 \
      --m 1 8 16 32 64 128 256 512 1024 2048 4096 8192 16384
"""

import argparse
import concurrent.futures
import json
import os
import subprocess
import sys
import time

import torch

import mxfp4_mbs as mm
from bench_utils import QWEN35_9B_LINEARS, graph_time_us, tflops

torch.set_default_device("cuda")

# variant -> (macro block, A side MBS, B side MBS)
VARIANTS = {
    "plain": (128, False, False),
    "mbs128": (128, True, True),
    "mbs128_a": (128, True, False),
    "mbs128_w": (128, False, True),
    "mbs256": (256, True, True),
    "mbs256_a": (256, True, False),
    "mbs512": (512, True, True),
    "mbs1024": (1024, True, True),
}

# GDN in_proj_qkvzba of a BF16 Qwen3.5 source: q, k, v, z plus the 2 x 32 b/a gates
SHAPES = QWEN35_9B_LINEARS + [("gdn.in_proj_qkvzba", 12352, 4096)]


def candidates(M):
    """(BLOCK_M, BLOCK_N, BLOCK_K, num_warps, waves_per_eu, num_stages, nonkdim, GROUP_M)."""
    if M <= 16:
        out = [(16, bn, bk, nw, wpe, 2, 16, 1) for bn in (32, 64, 128) for bk in (256, 512, 1024)
               for nw in (2, 4) for wpe in (1, 2)]
    elif M <= 32:
        out = [(bm, bn, bk, 4, wpe, 2, 16, 1) for bm in (16, 32) for bn in (32, 64, 128)
               for bk in (256, 512) for wpe in (1, 2)]
    elif M <= 128:
        out = [(bm, bn, bk, 4, wpe, st, 16, gm) for bm in (32, 64) for bn in (64, 128, 256)
               for bk in (256, 512) for wpe in (1, 2) for st in (2,) for gm in (1, 4)]
    elif M <= 512:
        out = [(bm, bn, 256, nw, wpe, st, nk, 4) for bm, bn, nw in ((64, 128, 4), (64, 256, 4), (128, 64, 4),
                                                                   (128, 128, 4), (128, 256, 4), (64, 64, 4))
               for wpe in (1, 2) for st in (2, 3) for nk in (16,)]
        out += [(bm, bn, 512, 4, wpe, 2, 16, 4) for bm, bn in ((64, 128), (128, 128)) for wpe in (1, 2)]
    else:
        out = [(bm, bn, 256, nw, wpe, st, nk, 4) for bm, bn, nw in ((64, 256, 4), (128, 128, 4), (128, 256, 4),
                                                                   (256, 128, 8), (256, 256, 8), (128, 256, 8))
               for wpe in (1, 2) for st in (2, 3) for nk in (16, 32)]
    return [dict(BLOCK_M=c[0], BLOCK_N=c[1], BLOCK_K=c[2], num_warps=c[3], waves_per_eu=c[4],
                 num_stages=c[5], matrix_instr_nonkdim=c[6], GROUP_M=c[7]) for c in out]


_CHECK = """
import sys, json, torch
import mxfp4_mbs as mm
from tune_gemm import VARIANTS
torch.set_default_device("cuda")
v, cfg, M = sys.argv[1], json.loads(sys.argv[2]), int(sys.argv[3])
mb, a_mbs, b_mbs = VARIANTS[v]
x = torch.randn(M, 4096, dtype=torch.bfloat16)
w = torch.randn(256, 4096, dtype=torch.bfloat16)
A = mm.quant_activation(x, mb=mb, use_mbs=a_mbs)
B = mm.prepare_weight(w, mb=mb, ncand=1, use_mbs=b_mbs)
mm.mbs_gemm(A, B, mb=mb, config=cfg)
torch.cuda.synchronize()
"""


def find_bad_configs(variants, ms, cache="results/bad_configs.json", workers=12):
    """Compile every candidate once in a subprocess: some tile configs crash the
    LLVM backend ("Virtual register defs don't dominate all uses"), which aborts
    the process. Also fills Triton's disk cache for the tuning run."""
    here = os.path.dirname(os.path.abspath(__file__))
    bad = json.load(open(cache)) if os.path.exists(cache) else {}
    jobs = []
    for v in variants:
        for M in ms:
            for cfg in candidates(M):
                key = f"{v}|{json.dumps(cfg, sort_keys=True)}"
                if key not in bad and key not in [j[0] for j in jobs]:
                    jobs.append((key, v, cfg, 32 if cfg["BLOCK_M"] >= 32 else 8))

    def run(job):
        key, v, cfg, M = job
        r = subprocess.run([sys.executable, "-c", _CHECK, v, json.dumps(cfg), str(M)], cwd=here,
                           capture_output=True, text=True, timeout=1200)
        return key, r.returncode != 0

    with concurrent.futures.ThreadPoolExecutor(workers) as ex:
        for key, failed in ex.map(run, jobs):
            bad[key] = failed
            if failed:
                print("bad config:", key, flush=True)
    json.dump(bad, open(cache, "w"), indent=1, sort_keys=True)
    return {k for k, failed in bad.items() if failed}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="*", default=["plain", "mbs128", "mbs256"])
    ap.add_argument("--m", type=int, nargs="*", default=[1, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384])
    ap.add_argument("--out", default="mbs_gemm_configs.json")
    ap.add_argument("--log", default="results/tune_gemm.jsonl")
    args = ap.parse_args()
    bad = find_bad_configs(args.variants, args.m)
    print(f"{len(bad)} configs crash the compiler and are skipped", flush=True)
    table = json.load(open(args.out)) if os.path.exists(args.out) else {}
    os.makedirs(os.path.dirname(args.log), exist_ok=True)
    log = open(args.log, "a")
    t0 = time.time()
    for name, N, K in SHAPES:
        w = torch.randn(N, K, dtype=torch.bfloat16)
        for v in args.variants:
            mb, a_mbs, b_mbs = VARIANTS[v]
            B = mm.prepare_weight(w, mb=mb, ncand=1, use_mbs=b_mbs)
            for M in args.m:
                x = torch.randn(M, K, dtype=torch.bfloat16)
                A = mm.quant_activation(x, mb=mb, use_mbs=a_mbs)
                best = None
                for cfg in candidates(M):
                    if K % max(mb, cfg["BLOCK_K"]) or (mb == 128 and cfg["BLOCK_K"] % 256):
                        continue
                    if f"{v}|{json.dumps(cfg, sort_keys=True)}" in bad:
                        continue
                    fn = lambda: mm.mbs_gemm(A, B, mb=mb, config=cfg)
                    try:
                        us, _ = graph_time_us(fn, n_calls=10, n_replays=5, warmup=2)
                    except Exception as e:  # noqa: BLE001
                        print(f"skip {v} M={M} {cfg}: {type(e).__name__} {str(e)[:120]}", flush=True)
                        continue
                    if best is None or us < best[0]:
                        best = (us, cfg)
                key = f"{v}|N={N}|K={K}"
                table.setdefault(key, {})[str(M)] = dict(best[1], us=round(best[0], 2))
                rec = dict(variant=v, layer=name, M=M, N=N, K=K, us=round(best[0], 2),
                           tflops=round(tflops(M, N, K, best[0]), 1), cfg=best[1])
                log.write(json.dumps(rec) + "\n")
                log.flush()
                print(json.dumps(rec), f"[{time.time() - t0:.0f}s]", flush=True)
                json.dump(table, open(args.out, "w"), indent=1, sort_keys=True)


if __name__ == "__main__":
    main()
