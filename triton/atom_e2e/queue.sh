#!/bin/bash
# Run the jobs of JOBFILE, one shell command per line, on a pool of GPUs: each
# GPU takes the next line when its previous job ends. "{gpu}" and "{port}" in a
# line become that GPU's HIP index and server port. Lines appended to JOBFILE
# while the queue runs are picked up too; a GPU that finds the file empty waits
# QUEUE_IDLE seconds (default 600) for more before it stops.
#
#   bash queue.sh JOBFILE GPU:PORT...      # e.g. bash queue.sh jobs.txt 4:18340 5:18330
JOBS=$(realpath "$1")
shift
IDLE=${QUEUE_IDLE:-600}

next_job() {  # pop the first line of $JOBS under a lock
  flock "$JOBS.lock" bash -c 'line=$(head -n 1 "$1"); [ -n "$line" ] && sed -i 1d "$1"; printf "%s" "$line"' _ "$JOBS"
}

worker() {
  local gpu=${1%:*} port=${1#*:} cmd waited=0
  while ((waited <= IDLE)); do
    cmd=$(next_job)
    if [ -z "$cmd" ]; then
      sleep 30
      waited=$((waited + 30))
      continue
    fi
    waited=0
    cmd=${cmd//\{gpu\}/$gpu}
    cmd=${cmd//\{port\}/$port}
    echo "$(date '+%F %T') GPU $gpu start: $cmd"
    bash -c "$cmd"
    echo "$(date '+%F %T') GPU $gpu done (exit $?): $cmd"
  done
}

for slot in "$@"; do worker "$slot" & done
wait
