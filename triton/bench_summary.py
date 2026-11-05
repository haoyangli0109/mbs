"""Tables of the ATOM serving benchmarks in results/e2e/<tag>/bench/.

  python bench_summary.py [tags...]      (default: every tag with results)
"""

import glob
import json
import os
import sys

ROOT = "results/e2e"
SPECS = ["isl1024_osl1024_c64", "isl4096_osl256_c32", "isl1024_osl1024_c256"]

tags = sys.argv[1:] or sorted(t for t in os.listdir(ROOT) if glob.glob(f"{ROOT}/{t}/bench/*.json"))
res = {t: {s: json.load(open(f"{ROOT}/{t}/bench/{s}.json")) for s in SPECS
           if os.path.exists(f"{ROOT}/{t}/bench/{s}.json")} for t in tags}
base = res.get("mxfp4", {})
for key, unit in (("output_throughput", "tok/s"), ("mean_ttft_ms", "ms"), ("mean_tpot_ms", "ms")):
    print(f"== {key} ({unit}); in parentheses mxfp4 / this for times, this / mxfp4 for throughput")
    print(f"{'tag':10s} " + " ".join(f"{s:>26s}" for s in SPECS))
    for t in tags:
        cells = []
        for s in SPECS:
            if s not in res[t]:
                cells.append(f"{'-':>26s}")
                continue
            v = res[t][s][key]
            if s in base:
                b = base[s][key]
                rel = v / b if key == "output_throughput" else b / v
                cells.append(f"{v:17.1f} ({rel:5.2f})")
            else:
                cells.append(f"{v:26.1f}")
        print(f"{t:10s} " + " ".join(cells))
