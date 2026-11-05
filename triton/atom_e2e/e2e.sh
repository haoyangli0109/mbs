#!/bin/bash
# End-to-end runs of Qwen3.5-9B on ATOM (branch lhy/mxfp4_mbs) with online
# quantization: BF16, MXFP4 (aiter asm GEMM) and MXFP4 + macro block scaling
# (Triton GEMM). One MI355X, TP1, CUDA graphs, no prefix caching.
#
#   bash e2e.sh run TAG [GPU] [PORT]     # serve, smoke test, GSM8K, then stop
#   bash e2e.sh serve TAG [GPU] [PORT]   # serve in the foreground
#   bash e2e.sh gsm8k TAG [PORT]         # GSM8K against a running server
#   bash e2e.sh bench TAG [PORT]         # serving throughput against a running server
#   bash e2e.sh summary                  # GSM8K table of all tags
#   REASON=1 SKIP_GSM8K=1 bash e2e.sh run TAG GPU PORT   # + AIME 2026 (reason_eval.py)
#
# TAG picks the online quantization (see online_cfg). Results go to
# ../results/e2e/<TAG>/; set NUM_FEWSHOT / LIMIT to change the GSM8K run.
set -o pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
export ATOM_ROOT=${ATOM_ROOT:-/home/haoyanli/a121/lhy_atom/ATOM_mbs}
export PYTHONPATH=$ATOM_ROOT${PYTHONPATH:+:$PYTHONPATH}
export AITER_LOG_LEVEL=WARNING
MODEL=${MODEL:-/mnt/dcgpuval/haoyanli/Qwen/Qwen3.5-9B}
OUT=${OUT:-$(dirname "$HERE")/results/e2e}
# conv1d is the GDN's depthwise causal conv (weight [conv_dim, 1, 4]), not a GEMM
EXCLUDE='["lm_head", "*visual*", "*conv1d*"]'

online_cfg() {
  case $1 in
    bf16) echo "" ;;
    mxfp4) echo "{\"global_quant_config\": \"mxfp4\", \"exclude_layer\": $EXCLUDE}" ;;
    mbs128 | mbs128_s) echo "{\"global_quant_config\": \"mxfp4_mbs\", \"exclude_layer\": $EXCLUDE}" ;;
    mbs128_a) echo "{\"global_quant_config\": \"mxfp4_mbs_a\", \"exclude_layer\": $EXCLUDE}" ;;
    mbs128_w) echo "{\"global_quant_config\": \"mxfp4_mbs_w\", \"exclude_layer\": $EXCLUDE}" ;;
    mbs256) echo "{\"global_quant_config\": \"mxfp4_mbs256\", \"exclude_layer\": $EXCLUDE}" ;;
    mbs512) echo "{\"global_quant_config\": \"mxfp4_mbs512\", \"exclude_layer\": $EXCLUDE}" ;;
    *) echo "unknown tag $1" >&2; return 1 ;;
  esac
}

log() { echo "$(date '+%F %T') $*"; }

serve() {
  local tag=$1 gpu=${2:-7} port=${3:-18300} cfg
  cfg=$(online_cfg "$tag") || return 1
  mkdir -p "$OUT/$tag"
  export HIP_VISIBLE_DEVICES=$gpu
  export ATOM_TORCH_PROFILER_DIR=$OUT/$tag   # online_quant_info_*.json lands here
  # mbs128_s: static (MBS-S) factors on the weights too, instead of the 16-candidate search
  [ "$tag" = mbs128_s ] && export ATOM_MBS_WEIGHT_CANDIDATES=1
  local args=(--model "$MODEL" --trust-remote-code -tp 1 --no-enable_prefix_caching
              --hf-overrides '{"mamba_ssm_dtype": "float32"}'
              --server-port "$port" --port $((port + 50))
              --max-num-seqs 256 --max-model-len "${MAX_LEN:-8192}" ${EXTRA_ARGS:-})
  [ -n "$cfg" ] && args+=(--online_quant_config "$cfg")
  exec python "$HERE/serve_atom.py" "${args[@]}"
}

wait_up() {  # PID LOG
  for _ in $(seq 1 480); do
    grep -q "Uvicorn running\|Application startup complete" "$2" 2>/dev/null && return 0
    kill -0 "$1" 2>/dev/null || return 1
    sleep 5
  done
  return 1
}

ptree() { echo "$1"; local c; for c in $(ps -o pid= --ppid "$1" 2>/dev/null); do ptree "$c"; done; }

stop_server() {  # SIGINT for a clean exit; after 3 minutes SIGKILL our own tree
  local pids p
  pids=$(ptree "$1")
  kill -INT "$1" 2>/dev/null
  for _ in $(seq 1 60); do kill -0 "$1" 2>/dev/null || break; sleep 3; done
  for p in $pids; do kill -0 "$p" 2>/dev/null && kill -KILL "$p" 2>/dev/null; done
  wait "$1" 2>/dev/null
  return 0
}

smoke() {  # PORT
  curl -s -m 300 "http://localhost:$1/v1/completions" -H 'Content-Type: application/json' \
    -d "{\"model\": \"$MODEL\", \"prompt\": \"The capital of France is\", \"max_tokens\": 24, \"temperature\": 0}" |
    python -c 'import json, sys; print(repr(json.load(sys.stdin)["choices"][0]["text"]))'
}

gsm8k() {  # TAG PORT
  local tag=$1 port=${2:-18300}
  lm_eval --model local-completions \
    --model_args "model=$MODEL,base_url=http://localhost:$port/v1/completions,num_concurrent=${NUM_CONCURRENT:-64},max_retries=3,timeout=1800,tokenized_requests=False,trust_remote_code=True" \
    --tasks gsm8k --num_fewshot "${NUM_FEWSHOT:-5}" ${LIMIT:+--limit $LIMIT} \
    --log_samples --output_path "$OUT/$tag/gsm8k"
}

bench() {  # TAG PORT: prefill-heavy and decode-heavy serving throughput
  local tag=$1 port=${2:-18300} isl osl c
  mkdir -p "$OUT/$tag/bench"
  for spec in ${BENCH_SPECS:-"1024,1024,64" "4096,256,32" "1024,1024,256"}; do
    IFS=, read -r isl osl c <<<"$spec"
    python -m atom.benchmarks.benchmark_serving \
      --model="$MODEL" --backend=vllm --base-url="http://localhost:$port" \
      --dataset-name=random --random-input-len="$isl" --random-output-len="$osl" \
      --random-range-ratio=1.0 --num-prompts=$((c * 4)) --max-concurrency="$c" \
      --request-rate=inf --ignore-eos --percentile-metrics=ttft,tpot,itl,e2el \
      --save-result --result-dir="$OUT/$tag/bench" --result-filename="isl${isl}_osl${osl}_c${c}.json"
  done
}

summary() {
  python - "$OUT" <<'EOF'
import glob, json, os, sys
root = sys.argv[1]
print(f"{'tag':10s} {'flexible':>9s} {'strict':>9s} {'stderr':>7s}  n")
for tag in sorted(os.listdir(root)):
    files = sorted(glob.glob(f"{root}/{tag}/gsm8k/*/results_*.json"))
    if not files:
        continue
    r = json.load(open(files[-1]))["results"]["gsm8k"]
    n = json.load(open(files[-1]))["n-samples"]["gsm8k"]["effective"]
    print(f"{tag:10s} {100 * r['exact_match,flexible-extract']:9.2f} {100 * r['exact_match,strict-match']:9.2f} "
          f"{100 * r['exact_match_stderr,flexible-extract']:7.2f}  {n}")
EOF
}

case "${1:-}" in
serve) serve "$2" "$3" "$4" ;;
gsm8k) gsm8k "$2" "$3" ;;
bench) bench "$2" "$3" ;;
summary) summary ;;
run)
  tag=$2 gpu=${3:-7} port=${4:-18300}
  mkdir -p "$OUT/$tag"
  slog=$OUT/$tag/server.log
  log "start $tag on GPU $gpu port $port"
  (serve "$tag" "$gpu" "$port") >"$slog" 2>&1 &
  pid=$!
  if ! wait_up "$pid" "$slog"; then log "server $tag failed, see $slog"; stop_server "$pid"; exit 1; fi
  log "up; smoke: $(smoke "$port")"
  grep -h "online-quantized\|Online quantization info" "$slog" | tail -2
  [ "${SKIP_GSM8K:-0}" = 1 ] || gsm8k "$tag" "$port" 2>&1 | tail -8
  if [ "${REASON:-0}" = 1 ]; then
    # thinking mode, model-card sampling, up to 81,920 output tokens (needs MAX_LEN >= 90112)
    # REASON_TAG names the results (default TAG); REASON_SAMPLING overrides sampling (JSON)
    python "$HERE/../reason_eval.py" --tag "${REASON_TAG:-$tag}" --url "http://localhost:$port" --model "$MODEL" \
      --tasks ${REASON_TASKS:-aime26} --samples "${SAMPLES:-8}" --concurrency "${CONC:-160}" \
      --max-tokens "${MAX_TOKENS:-81920}" ${REASON_SAMPLING:+--sampling "$REASON_SAMPLING"} \
      >"$OUT/$tag/reason${REASON_TAG:+_$REASON_TAG}.log" 2>&1
    tail -2 "$OUT/$tag/reason${REASON_TAG:+_$REASON_TAG}.log"
  fi
  if [ "${BENCH:-0}" = 1 ]; then
    bench "$tag" "$port" >"$OUT/$tag/bench.log" 2>&1
    grep -E "Output token throughput|Mean TTFT|Mean TPOT" "$OUT/$tag/bench.log"
  fi
  stop_server "$pid"
  log "done $tag"
  ;;
*) sed -n 2,14p "$0" ;;
esac
