"""Format differences against rounding luck: wikitext-2 PPL of each format over
several quantizations of slightly perturbed weights.

  HIP_VISIBLE_DEVICES=4 python exp_dither.py --model /mnt/dcgpuval/haoyanli/Qwen/Qwen3-8B \
      --out results/dither_qwen3_8b.json

Each seed multiplies every quantized Linear weight by (1 + sigma * N(0, 1))
before quantizing it (sigma = 2%: ~34 dB, far below FP4's ~19 dB, but it moves
many elements and macro-block factors across rounding boundaries). bf16 with
the same perturbation is the control: it shows what the perturbation alone does.
"""

import argparse
import json
import math
import os
import statistics
import time

import torch

import hf_eval


def ppl(model, ids, n_win, seqlen=2048):
    nll, count = 0.0, 0
    with torch.no_grad():
        for i in range(n_win):
            x = ids[:, i * seqlen:(i + 1) * seqlen].cuda()
            logits = model(input_ids=x).logits.float()
            nll += torch.nn.functional.cross_entropy(logits[0, :-1], x[0, 1:], reduction="sum").item()
            count += x.shape[1] - 1
    return math.exp(nll / count)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--formats", nargs="*", default=["bf16", "mxfp4", "mbs128", "mbs128_s", "mbs512", "mbs512_s", "mbs128_a"])
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--sigma", type=float, default=0.02)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    model, tok = hf_eval.load_model(args.model)
    ids = hf_eval.wikitext_ids(tok)
    n_win = ids.shape[1] // 2048
    linears = hf_eval.decoder_linears(model)
    originals = {n: lin.weight.detach().cpu() for n, lin in linears.items()}  # 27B: no room for a GPU copy
    results = json.load(open(args.out)) if os.path.exists(args.out) else {}
    for seed in range(args.seeds):
        g = torch.Generator(device="cuda").manual_seed(seed)
        for n, lin in linears.items():
            w = originals[n].cuda()
            lin.weight.data.copy_((w.float() * (1 + args.sigma * torch.randn(w.shape, generator=g, device="cuda"))).to(w.dtype))
        for f in args.formats:
            key = f"{f}|seed{seed}"
            if key in results:
                continue
            t0 = time.time()
            if f != "bf16":
                mb, a_mbs, w_mbs, ncand = hf_eval.FORMATS[f]
                for n, lin in linears.items():
                    parent = model.get_submodule(n.rsplit(".", 1)[0])
                    setattr(parent, n.rsplit(".", 1)[1], hf_eval.QuantLinear(lin, mb, a_mbs, w_mbs, ncand))
            results[key] = dict(ppl=ppl(model, ids, n_win), sigma=args.sigma, model=args.model)
            for n, lin in linears.items():
                parent = model.get_submodule(n.rsplit(".", 1)[0])
                setattr(parent, n.rsplit(".", 1)[1], lin)
            print(f"seed {seed} {f:10s} ppl {results[key]['ppl']:.4f}  ({time.time() - t0:.0f}s)", flush=True)
            json.dump(results, open(args.out, "w"), indent=1)
    print(f"\n{'format':10s} {'mean':>8s} {'sd':>6s}  per seed")
    for f in args.formats:
        v = [results[f"{f}|seed{s}"]["ppl"] for s in range(args.seeds)]
        print(f"{f:10s} {statistics.mean(v):8.4f} {statistics.stdev(v):6.4f}  " + " ".join(f"{x:.3f}" for x in v))


if __name__ == "__main__":
    main()
