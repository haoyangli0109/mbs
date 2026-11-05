"""The Triton quantizer on real checkpoint weights: bitwise against the torch
reference, and the weight QSNR of each format.

  HIP_VISIBLE_DEVICES=4 python check_quant_real.py /data/Qwen/Qwen3.6-27B [--layers 0 31 63]

test_mxfp4_mbs.py checks the quantizer on random data; this reads a few decoder
layers' Linear weights straight from the safetensors (no model load).
"""

import argparse
import fnmatch
import json
import os

import torch
from safetensors import safe_open

import mxfp4_mbs as mm

# name -> (macro block, weight MBS, candidates)
FORMATS = {"mxfp4": (128, False, 1), "mbs-s": (128, True, 1), "mbs-d16": (128, True, 16), "mbs512-d16": (512, True, 16)}


def qsnr(ref, q):
    ref = ref.double()
    return (10 * torch.log10(ref.pow(2).sum() / (ref - q.double()).pow(2).sum())).item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--layers", nargs="*", type=int, default=[0, 1, 2])
    args = ap.parse_args()
    index = json.load(open(os.path.join(args.model, "model.safetensors.index.json")))["weight_map"]
    names = [n for n in index if n.endswith(".weight") and "visual" not in n and "mtp" not in n
             and any(fnmatch.fnmatch(n, f"*layers.{i}.*proj*.weight") for i in args.layers)]
    mismatch = 0
    sums = {f: [] for f in FORMATS}
    for name in sorted(names):
        with safe_open(os.path.join(args.model, index[name]), "pt", device="cuda") as fh:
            w = fh.get_tensor(name).to(torch.bfloat16)
        if w.dim() != 2 or w.shape[1] % 512:
            continue
        row = []
        for f, (mb, use_mbs, ncand) in FORMATS.items():
            q, s, m = mm.mbs_quant(w, mb=mb, ncand=ncand, use_mbs=use_mbs)
            if use_mbs:
                qr, sr, mr = mm.mbs_quant_ref(w, mb=mb, ncand=ncand)
                same = torch.equal(q, qr) and torch.equal(s, sr) and torch.equal(m, mr)
                mismatch += not same
            else:
                same = True
                m = torch.zeros(w.shape[0], w.shape[1] // mb, dtype=torch.uint8, device=w.device)
            v = qsnr(w, mm.mbs_dequant(q, s, m, mb))
            sums[f].append(v)
            row.append(f"{f} {v:6.2f}{'' if same else ' (!= ref)'}")
        print(f"{name:60s} {tuple(w.shape)}  " + "  ".join(row), flush=True)
    print("mean weight QSNR: " + "  ".join(f"{f} {sum(v) / len(v):.2f}" for f, v in sums.items()))
    print("ALL BITWISE EQUAL TO REFERENCE" if not mismatch else f"{mismatch} MISMATCHES")


if __name__ == "__main__":
    main()
