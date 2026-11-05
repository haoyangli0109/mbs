"""Qwen3.5-9B (or --model) through the real Triton MXFP4/MBS kernels, outside ATOM.

  HIP_VISIBLE_DEVICES=0 python hf_eval.py qsnr --out results/qsnr_qwen35_9b.json
  HIP_VISIBLE_DEVICES=0 python hf_eval.py ppl  --formats bf16 mxfp4 mbs128 --out results/ppl_qwen35_9b.json
  HIP_VISIBLE_DEVICES=0 python hf_eval.py ppl  --model /data/Qwen/Qwen3.6-27B --out results/ppl_qwen36_27b.json

qsnr: every decoder Linear (no lm_head, no vision tower) gets its input from a
      1024-token forward pass over wikitext-2 test; activation, weight and
      output QSNR (paper eq. 1-2) per format, averaged over layers. Outputs
      come from the MBS GEMM in fp32, the reference is fp32 A @ W^T.
ppl:  wikitext-2 test perplexity (2048-token windows) with every decoder
      Linear running quantize(activation) + MBS GEMM (bf16 out), weights
      quantized once like ATOM's online quantization does.
"""

import argparse
import fnmatch
import json
import math
import os
import time

import datasets
import torch
import transformers

import mxfp4_mbs as mm

MODEL = "/mnt/dcgpuval/haoyanli/Qwen/Qwen3.5-9B"

# name -> (macro block, activation MBS, weight MBS, weight candidates)
FORMATS = {
    "mxfp4": (128, False, False, 1),
    "mbs128": (128, True, True, 16),
    "mbs128_w256": (128, True, True, 256),
    "mbs128_s": (128, True, True, 1),
    "mbs128_a": (128, True, False, 1),
    "mbs128_w": (128, False, True, 16),
    "mbs256": (256, True, True, 16),
    "mbs256_s": (256, True, True, 1),
    "mbs256_a": (256, True, False, 1),
    "mbs512": (512, True, True, 16),
    "mbs512_s": (512, True, True, 1),
    "mbs1024": (1024, True, True, 16),
    "mbs1024_s": (1024, True, True, 1),
}


def load_model(path):
    try:
        model = transformers.AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, device_map="cuda")
    except Exception as exc:  # noqa: BLE001  (VLM checkpoint)
        print("CausalLM load failed:", repr(exc)[:200])
        model = transformers.AutoModelForImageTextToText.from_pretrained(path, dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    return model, transformers.AutoTokenizer.from_pretrained(path)


def decoder_linears(model, exclude=()):
    return {n: m for n, m in model.named_modules()
            if isinstance(m, torch.nn.Linear) and ".layers." in n and "visual" not in n
            and not any(fnmatch.fnmatch(n, p) for p in exclude)}


def wikitext_ids(tok):
    text = "\n\n".join(datasets.load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")["text"])
    return tok(text, return_tensors="pt").input_ids


def qsnr(ref, q):
    ref = ref.double()
    return (10 * torch.log10(ref.pow(2).sum() / (ref - q.double()).pow(2).sum())).item()


def run_qsnr(args):
    model, tok = load_model(args.model)
    linears = decoder_linears(model)
    print(f"{len(linears)} decoder linears", flush=True)
    acts = {}

    def hook(name):
        def fn(mod, inp, out):
            acts.setdefault(name, inp[0].detach().reshape(-1, inp[0].shape[-1])[: args.ntok].clone())
        return fn

    hs = [m.register_forward_hook(hook(n)) for n, m in linears.items()]
    with torch.no_grad():
        model(input_ids=wikitext_ids(tok)[:, : args.ntok].cuda())
    for h in hs:
        h.remove()
    res = {f: {"act": [], "w": [], "out": []} for f in args.formats}
    t0 = time.time()
    with torch.no_grad():
        for i, (name, mod) in enumerate(linears.items()):
            a = acts.pop(name)
            w = mod.weight.detach()
            ref = a.float() @ w.float().T
            for f in args.formats:
                mb, a_mbs, w_mbs, ncand = FORMATS[f]
                A = mm.quant_activation(a, mb=mb, use_mbs=a_mbs, keep_m=True)
                W = mm.prepare_weight(w, mb=mb, ncand=ncand, use_mbs=w_mbs)
                zeros = lambda rows: torch.zeros(rows, a.shape[1] // mb, dtype=torch.uint8, device=a.device)
                qa = mm.mbs_quant(a, mb=mb, use_mbs=a_mbs)
                qw = mm.mbs_quant(w, mb=mb, ncand=ncand, use_mbs=w_mbs)
                res[f]["act"].append(qsnr(a, mm.mbs_dequant(qa[0], qa[1], qa[2] if a_mbs else zeros(a.shape[0]), mb)))
                res[f]["w"].append(qsnr(w, mm.mbs_dequant(qw[0], qw[1], qw[2] if w_mbs else zeros(w.shape[0]), mb)))
                cfg = mm.select_config(a.shape[0], w.shape[0], a.shape[1], mb, a_mbs, w_mbs)
                out = mm.mbs_gemm(A, W, mb=mb, out_dtype=torch.float32, config=cfg)
                res[f]["out"].append(qsnr(ref, out))
            if i % 40 == 0:
                print(f"{i}/{len(linears)} {name} {tuple(w.shape)} {time.time() - t0:.0f}s", flush=True)
    summary = {f: {k: sum(v) / len(v) for k, v in r.items()} for f, r in res.items()}
    base = summary.get("mxfp4", {}).get("out")
    print(f"\n{'format':14s} {'act':>7s} {'weight':>7s} {'output':>7s} {'vs mxfp4':>9s}")
    for f, s in summary.items():
        delta = f"{s['out'] - base:+9.2f}" if base is not None else ""
        print(f"{f:14s} {s['act']:7.2f} {s['w']:7.2f} {s['out']:7.2f} {delta}")
    json.dump({"modules": list(linears), "ntok": args.ntok, "per_module": res, "summary": summary},
              open(args.out, "w"), indent=1)


class QuantLinear(torch.nn.Module):
    """nn.Linear replacement: MXFP4/MBS quantized activation x quantized weight."""

    def __init__(self, lin, mb, a_mbs, w_mbs, ncand):
        super().__init__()
        self.mb, self.a_mbs = mb, a_mbs
        self.w = mm.prepare_weight(lin.weight.detach(), mb=mb, ncand=ncand, use_mbs=w_mbs)
        self.bias = lin.bias
        self.n = lin.out_features

    def forward(self, x):
        x2 = x.reshape(-1, x.shape[-1]).contiguous()
        y = mm.mbs_linear(x2, self.w, mb=self.mb, act_mbs=self.a_mbs).view(*x.shape[:-1], self.n)
        return y if self.bias is None else y + self.bias


def run_ppl(args):
    model, tok = load_model(args.model)
    ids = wikitext_ids(tok)
    n_win = ids.shape[1] // args.seqlen
    if args.max_windows:
        n_win = min(n_win, args.max_windows)
    originals = decoder_linears(model, args.exclude)
    print(f"{len(originals)} decoder linears quantized, excluding {args.exclude}", flush=True)
    results = json.load(open(args.out)) if os.path.exists(args.out) else {}
    for f in args.formats:
        t0 = time.time()
        if f != "bf16":
            mb, a_mbs, w_mbs, ncand = FORMATS[f]
            for name, lin in originals.items():
                parent = model.get_submodule(name.rsplit(".", 1)[0])
                setattr(parent, name.rsplit(".", 1)[1], QuantLinear(lin, mb, a_mbs, w_mbs, ncand))
        torch.cuda.synchronize()
        tq = time.time() - t0
        nll, count = 0.0, 0
        with torch.no_grad():
            for i in range(n_win):
                x = ids[:, i * args.seqlen : (i + 1) * args.seqlen].cuda()
                logits = model(input_ids=x).logits.float()
                loss = torch.nn.functional.cross_entropy(logits[0, :-1], x[0, 1:], reduction="sum")
                nll += loss.item()
                count += x.shape[1] - 1
        ppl = math.exp(nll / count)
        results[f] = dict(ppl=ppl, model=args.model, windows=n_win, seqlen=args.seqlen, tokens=count, exclude=args.exclude,
                          quantized_linears=len(originals), quant_seconds=round(tq, 1),
                          eval_seconds=round(time.time() - t0 - tq, 1))
        print(f"{f:12s} ppl {ppl:.4f}  ({n_win} x {args.seqlen} tokens, quant {tq:.1f}s, eval {time.time() - t0 - tq:.0f}s)",
              flush=True)
        json.dump(results, open(args.out, "w"), indent=1)
        for name, lin in originals.items():  # restore BF16 linears
            parent = model.get_submodule(name.rsplit(".", 1)[0])
            setattr(parent, name.rsplit(".", 1)[1], lin)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["qsnr", "ppl"])
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--formats", nargs="*", default=None)
    ap.add_argument("--ntok", type=int, default=1024)
    ap.add_argument("--seqlen", type=int, default=2048)
    ap.add_argument("--max_windows", type=int, default=0)
    ap.add_argument("--exclude", nargs="*", default=[], help="fnmatch patterns of Linears kept in BF16 (ppl)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    if args.formats is None:
        args.formats = list(FORMATS) if args.mode == "qsnr" else ["bf16", "mxfp4", "mbs128", "mbs256", "mbs128_a"]
    run_qsnr(args) if args.mode == "qsnr" else run_ppl(args)


if __name__ == "__main__":
    main()
