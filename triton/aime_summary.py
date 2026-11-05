"""AIME 2026 table of results/reason/<tag>/aime26.jsonl (from reason_eval.py).

  python aime_summary.py [--base mxfp4] [--samples 5] [tags...]

Per tag: accuracy over all (question, sample) pairs, a 95% interval from
resampling the 30 questions, mean output tokens, share of replies cut at the
token limit (scored wrong unless the cut text already holds a boxed answer),
and, against --base, the questions whose per-question accuracy is lower /
higher with a two-sided sign test on them. --samples N keeps samples 0..N-1
only, so tags that ran different sample counts compare on the same draws.
"""

import argparse
import collections
import json
import math
import os
import random

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "reason")


def load(tag, samples=None):
    path = os.path.join(ROOT, tag, "aime26.jsonl")
    if not os.path.exists(path):
        return None
    recs = [json.loads(line) for line in open(path)]
    return [r for r in recs if samples is None or r["sample"] < samples]


def per_question(recs):
    acc = collections.defaultdict(list)
    for r in recs:
        acc[r["idx"]].append(r["correct"])
    return {i: sum(v) / len(v) for i, v in acc.items()}


def sign_test(wins, losses):
    n = wins + losses
    if n == 0:
        return 1.0
    k = min(wins, losses)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2**n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tags", nargs="*")
    ap.add_argument("--base", default="mxfp4")
    ap.add_argument("--samples", type=int, default=None)
    args = ap.parse_args()
    tags = args.tags or sorted(t for t in os.listdir(ROOT) if load(t))
    base = per_question(load(args.base, args.samples)) if load(args.base) else None
    rng = random.Random(0)
    print(f"{'tag':10s} {'n':>4s} {'acc':>6s} {'95% CI':>15s} {'tokens':>7s} {'cut':>6s}   vs {args.base}: worse/better (p)")
    for t in tags:
        recs = load(t, args.samples)
        pq = per_question(recs)
        qs = sorted(pq)
        acc = 100 * sum(r["correct"] for r in recs) / len(recs)
        boots = sorted(100 * sum(pq[rng.choice(qs)] for _ in qs) / len(qs) for _ in range(2000))
        toks = sum(r["completion_tokens"] for r in recs) / len(recs)
        cut = 100 * sum(r["finish"] == "length" for r in recs) / len(recs)
        cmp = ""
        if base is not None and t != args.base:
            common = pq.keys() & base.keys()
            worse = sum(pq[i] < base[i] for i in common)
            better = sum(pq[i] > base[i] for i in common)
            cmp = f"{worse}/{better} (p={sign_test(better, worse):.3f})"
        print(f"{t:10s} {len(recs):4d} {acc:6.1f} [{boots[50]:5.1f}, {boots[1949]:5.1f}] {toks:7.0f} {cut:5.1f}%   {cmp}")


if __name__ == "__main__":
    main()
