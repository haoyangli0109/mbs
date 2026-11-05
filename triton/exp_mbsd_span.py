"""How wide MBS-D should search: wikitext-2 PPL with the weight candidates
squeezed from one octave below the static factor to a fraction of it.

  HIP_VISIBLE_DEVICES=4 python exp_mbsd_span.py --model /mnt/dcgpuval/haoyanli/Qwen/Qwen3-8B \
      --out results/mbsd_span_qwen3_8b.json

MBS-D tries f0 * 2^(-span * j / 16), j = 0..15, folded into [1, 2), where f0 is
the static (MBS-S) factor; span = 1 is the shipped search. Every format here is
two-sided macro block 128 (activations MBS-S), so only the weight factors
differ. Per span it also reports the weight QSNR and the share of macro blocks
whose largest weight moves by more than 5%, over all quantized Linears.
"""

import argparse
import functools
import json
import math
import os
import time

import torch

import hf_eval
import mxfp4_mbs as mm


def set_span(span):
    mm._cands = functools.cache(
        lambda ncand, device: torch.tensor([2.0 ** (-span * j / ncand) for j in range(ncand)],
                                           dtype=torch.float32, device=device))


def weight_stats(linears, ncand):
    num = den = moved = blocks = 0.0
    for lin in linears.values():
        w = lin.weight.detach()
        if ncand == 0:
            q, s, m = mm.mbs_quant(w, mb=128, use_mbs=False)
            m = torch.zeros(w.shape[0], w.shape[1] // 128, dtype=torch.uint8, device=w.device)
        else:
            q, s, m = mm.mbs_quant(w, mb=128, ncand=ncand)
        wf = w.float()
        d = mm.mbs_dequant(q, s, m, 128)
        num += (wf - d).pow(2).sum().item()
        den += wf.pow(2).sum().item()
        blk = wf.reshape(w.shape[0], -1, 128)
        idx = blk.abs().argmax(-1, keepdim=True)
        wmax = blk.gather(-1, idx)
        dmax = d.reshape(w.shape[0], -1, 128).gather(-1, idx)
        moved += ((dmax - wmax).abs() > 0.05 * wmax.abs()).sum().item()
        blocks += wmax.numel()
    return 10 * math.log10(den / num), 100 * moved / blocks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--spans", nargs="*", type=float, default=[1.0, 0.5, 0.25, 0.125, 0.0625])
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    model, tok = hf_eval.load_model(args.model)
    ids = hf_eval.wikitext_ids(tok)
    seqlen = 2048
    n_win = ids.shape[1] // seqlen
    linears = hf_eval.decoder_linears(model)
    results = json.load(open(args.out)) if os.path.exists(args.out) else {}
    runs = [("mxfp4 weights", 0, None), ("MBS-S weights", 1, None)] + [(f"MBS-D span {s:g}", 16, s) for s in args.spans]
    for name, ncand, span in runs:
        if name in results:
            continue
        set_span(span if span is not None else 1.0)
        t0 = time.time()
        qsnr, moved = weight_stats(linears, ncand)
        for n, lin in linears.items():
            parent = model.get_submodule(n.rsplit(".", 1)[0])
            setattr(parent, n.rsplit(".", 1)[1],
                    hf_eval.QuantLinear(lin, 128, True, ncand > 0, max(ncand, 1)))
        nll, count = 0.0, 0
        with torch.no_grad():
            for i in range(n_win):
                x = ids[:, i * seqlen:(i + 1) * seqlen].cuda()
                logits = model(input_ids=x).logits.float()
                nll += torch.nn.functional.cross_entropy(logits[0, :-1], x[0, 1:], reduction="sum").item()
                count += x.shape[1] - 1
        for n, lin in linears.items():
            parent = model.get_submodule(n.rsplit(".", 1)[0])
            setattr(parent, n.rsplit(".", 1)[1], lin)
        ppl = math.exp(nll / count)
        results[name] = dict(ppl=ppl, weight_qsnr=qsnr, max_moved_pct=moved, span=span, ncand=ncand, model=args.model)
        print(f"{name:20s} ppl {ppl:.4f}  weight QSNR {qsnr:.2f} dB  block max moved >5%: {moved:4.1f}%  "
              f"({time.time() - t0:.0f}s)", flush=True)
        json.dump(results, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
