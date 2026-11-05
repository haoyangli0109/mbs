"""Question-by-question agreement between GSM8K runs of <root>/<tag>/gsm8k.

  python gsm8k_pair.py [--root results/e2e_3shot] mxfp4 mbs128 ...   (first tag is the base)
"""
import glob
import json
import math
import sys

ROOT = "results/e2e"


def load_both(tag):
    path = sorted(glob.glob(f"{ROOT}/{tag}/gsm8k/*/samples_gsm8k_*.jsonl"))[-1]
    res = {}
    for line in open(path):
        s = json.loads(line)
        res.setdefault(s["filter"], {})[s["doc_id"]] = bool(s["exact_match"])
    return res


def sign_test_p(wins, losses):
    """Two-sided exact binomial (sign) test on the discordant pairs."""
    n = wins + losses
    if n == 0:
        return 1.0
    k = min(wins, losses)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * p)


args = sys.argv[1:]
if args[0] == "--root":
    ROOT, args = args[1], args[2:]
base_tag, *others = args
base = load_both(base_tag)
for filt in ("flexible-extract", "strict-match"):
    print(f"== {filt}")
    for tag in others:
        o = load_both(tag)[filt]
        b = base[filt]
        ids = b.keys() & o.keys()
        only_b = sum(b[i] and not o[i] for i in ids)
        only_o = sum(o[i] and not b[i] for i in ids)
        print(f"  {tag:10s} vs {base_tag:8s}: n={len(ids)}  {base_tag} {100 * sum(b[i] for i in ids) / len(ids):.2f}  "
              f"{tag} {100 * sum(o[i] for i in ids) / len(ids):.2f}  only {base_tag} {only_b}  only {tag} {only_o}  "
              f"sign-test p={sign_test_p(only_o, only_b):.3f}")
