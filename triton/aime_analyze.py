"""How the AIME 2026 replies of each tag end: finished or cut, and if cut, looping.

  python aime_analyze.py [--samples 5] [--base bf16] [tags...]

Reads results/reason/<tag>/aime26.jsonl and aime26_outputs.jsonl (reason_eval.py).
Per tag, over the (question, sample) pairs kept by --samples:

  acc        accuracy, cut replies count as wrong (as in aime_summary.py)
  cut        share stopped by the 81,920-token limit
  acc|fin    accuracy of the replies that finished on their own
  tok|fin    mean output tokens of those
  len×       per question, mean tokens of finished replies over --base's, geometric mean
  loop|cut   share of cut replies whose last 200 non-empty lines have < 50% distinct lines
  dup        median share of non-empty lines (≥ 20 chars) that repeat an earlier line
  nan        replies containing "!!!!!!!!" (NaN logits decode as token 0)
"""

import argparse
import collections
import json
import math
import os
import statistics

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "reason")
TAIL = 200


def load(tag, samples):
    d = os.path.join(ROOT, tag)
    meta = {}
    for line in open(os.path.join(d, "aime26.jsonl")):
        r = json.loads(line)
        if samples is None or r["sample"] < samples:
            meta[(r["idx"], r["sample"])] = r
    text = {}
    for line in open(os.path.join(d, "aime26_outputs.jsonl")):
        r = json.loads(line)
        key = (r["idx"], r["sample"])
        if key in meta:
            text[key] = r["reasoning"] + "\n" + r["content"]
    return meta, text


def line_stats(text):
    lines = [s.strip() for s in text.splitlines() if s.strip()]
    tail = lines[-TAIL:]
    distinct_tail = len(set(tail)) / len(tail) if tail else 1.0
    long_lines = [s for s in lines if len(s) >= 20]
    seen, dup = set(), 0
    for s in long_lines:
        dup += s in seen
        seen.add(s)
    return distinct_tail, dup / len(long_lines) if long_lines else 0.0


def finished_tokens(meta):
    per_q = collections.defaultdict(list)
    for r in meta.values():
        if r["finish"] != "length":
            per_q[r["idx"]].append(r["completion_tokens"])
    return {q: statistics.mean(v) for q, v in per_q.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tags", nargs="*")
    ap.add_argument("--samples", type=int, default=None)
    ap.add_argument("--base", default="bf16")
    args = ap.parse_args()
    tags = args.tags or sorted(os.listdir(ROOT))
    base_tok = finished_tokens(load(args.base, args.samples)[0])
    print(f"{'tag':9s} {'n':>4s} {'acc':>5s} {'cut':>5s} {'acc|fin':>7s} {'tok|fin':>7s} "
          f"{'len×':>5s} {'loop|cut':>8s} {'dup':>5s} {'nan':>3s}")
    for t in tags:
        meta, text = load(t, args.samples)
        recs = list(meta.values())
        fin = [r for r in recs if r["finish"] != "length"]
        cut = [k for k, r in meta.items() if r["finish"] == "length"]
        stats = {k: line_stats(text[k]) for k in meta if k in text}
        loops = sum(stats[k][0] < 0.5 for k in cut if k in stats)
        tok = finished_tokens(meta)
        common = tok.keys() & base_tok.keys()
        ratio = math.exp(statistics.mean(math.log(tok[q] / base_tok[q]) for q in common)) if common else float("nan")
        n = len(recs)
        print(f"{t:9s} {n:4d} {100 * sum(r['correct'] for r in recs) / n:5.1f} {100 * len(cut) / n:4.1f}% "
              f"{100 * sum(r['correct'] for r in fin) / max(len(fin), 1):7.1f} "
              f"{statistics.mean(r['completion_tokens'] for r in fin) if fin else 0:7.0f} {ratio:5.2f} "
              f"{100 * loops / max(len(cut), 1):7.0f}% {100 * statistics.median(s[1] for s in stats.values()):4.1f}% "
              f"{sum(r['bang'] for r in recs):3d}")


if __name__ == "__main__":
    main()
