#!/bin/bash
# Stop the reason_eval.py run of each TAG once samples 0..N-1 of all 30 AIME
# questions are done; e2e.sh then shuts that server down as usual.
#
#   bash stop_after_samples.sh N TAG...
N=$1
shift
RES=$(cd "$(dirname "$0")/.." && pwd)/results/reason
pending=("$@")
while ((${#pending[@]})); do
  left=()
  for tag in "${pending[@]}"; do
    done_n=$(python3 -c "
import json, sys
recs = [json.loads(l) for l in open('$RES/$tag/aime26.jsonl')]
print(len({(r['idx'], r['sample']) for r in recs if r['sample'] < $N}))" 2>/dev/null || echo 0)
    if [ "$done_n" -ge $((30 * N)) ]; then
      pid=$(ps -eo pid,args | awk -v t="$tag" '$2=="python" && $3 ~ /reason_eval\.py$/ && $5==t {print $1}')
      echo "$(date '+%F %T') $tag: samples 0-$((N - 1)) done ($done_n), stopping reason_eval pid ${pid:-none}"
      [ -n "$pid" ] && kill -INT $pid
    else
      left+=("$tag")
    fi
  done
  pending=("${left[@]}")
  ((${#pending[@]})) && sleep 60
done
