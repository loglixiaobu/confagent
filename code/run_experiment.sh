#!/bin/bash
# Start selected local replicas and run the batch runner.
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
GPUS=${1:?"Usage: bash code/run_experiment.sh '0 1' 18000 [batch arguments] [--stop]"}
BASE_PORT=${2:-18000}
shift
if (( $# )); then shift; fi
read -ra GPU_LIST <<< "$GPUS"

STOP_AFTER=0
BATCH_ARGS=()
for arg in "$@"; do
  if [[ "$arg" == "--stop" ]]; then
    STOP_AFTER=1
  else
    BATCH_ARGS+=("$arg")
  fi
done

endpoints=()
for i in "${!GPU_LIST[@]}"; do
  endpoints+=("127.0.0.1:$((BASE_PORT + i))")
done
endpoint_list=$(IFS=,; printf '%s' "${endpoints[*]}")

bash "$ROOT/code/launch_replicas.sh" "$GPUS" "$BASE_PORT"
if (( STOP_AFTER )); then
  trap 'bash "$ROOT/code/stop_replicas.sh" "${GPU_LIST[@]}"' EXIT
fi
cd "$ROOT"
"${CONFAGENT_PYTHON:-python3}" code/run_batch.py --endpoints "$endpoint_list" "${BATCH_ARGS[@]}"
