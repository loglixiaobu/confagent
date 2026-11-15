#!/usr/bin/env bash
# Run each local vLLM replica on one caller-selected GPU.
set -euo pipefail

code_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
project_dir=$(dirname -- "$code_dir")
python_cmd=$(command -v "${CONFAGENT_PYTHON:-python3}") ||
  { echo 'Set CONFAGENT_PYTHON to an interpreter with vLLM installed.' >&2; exit 2; }
export PATH="$(dirname -- "$python_cmd"):$PATH"
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"

model_path=${MODEL_PATH:-"$project_dir/models/Qwen3-8B"}
[[ -d "$model_path" ]] || { echo 'Set MODEL_PATH to a local Qwen3-8B checkpoint.' >&2; exit 2; }
model_path=$(cd -- "$model_path" && pwd -P)
served_name=${SERVED_NAME:-Qwen3-8B}
log_dir=${CONFAGENT_SERVICE_LOG_DIR:-"$project_dir/logs"}
mkdir -p -- "$log_dir"
log_dir=$(cd -- "$log_dir" && pwd -P)

read -r -a gpus <<< "${1:-0 1}"
base_port=${2:-18000}
if [[ ${#gpus[@]} -eq 0 || ! "$base_port" =~ ^[0-9]{1,5}$ ]]; then
  echo 'Usage: bash code/launch_replicas.sh "0 1" 18000' >&2
  exit 2
fi
base_port=$((10#$base_port))
if (( base_port < 1024 || base_port + ${#gpus[@]} - 1 > 65535 )); then
  echo 'Replica ports must be between 1024 and 65535.' >&2
  exit 2
fi

exec 9>"$log_dir/.replicas.lock"
flock -n 9 || { echo 'A launcher already owns this log directory.' >&2; exit 2; }

declare -A seen
for index in "${!gpus[@]}"; do
  gpu=${gpus[$index]}
  if [[ ! "$gpu" =~ ^(0|[1-9][0-9]*)$ || -n "${seen[$gpu]:-}" ]]; then
    echo "Invalid or repeated GPU index: $gpu" >&2
    exit 2
  fi
  seen[$gpu]=1
  port=$((base_port + index))
  "$python_cmd" - "$port" <<'PY'
import socket
import sys

port = int(sys.argv[1])
with socket.socket() as listener:
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        listener.bind(("127.0.0.1", port))
    except OSError as exc:
        raise SystemExit(f"Port {port} is unavailable: {exc}") from None
PY
  compute_pids=$(nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader)
  [[ -z "$compute_pids" ]] || { echo "GPU $gpu already has a compute process." >&2; exit 2; }
  uuid=$(nvidia-smi -i "$gpu" --query-gpu=uuid --format=csv,noheader)
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$gpu" "$python_cmd" - "$uuid" <<'PY'
import sys
import torch

actual = str(torch.cuda.get_device_properties(0).uuid)
expected = sys.argv[1].strip()
actual = actual[4:] if actual.startswith("GPU-") else actual
expected = expected[4:] if expected.startswith("GPU-") else expected
if actual != expected:
    raise SystemExit("CUDA and NVIDIA device indices do not match.")
PY
done

launch_id=$("$python_cmd" -c 'import uuid; print(uuid.uuid4().hex)')
cleanup_failed_launch() {
  if [[ $? -ne 0 ]]; then
    CONFAGENT_PYTHON="$python_cmd" CONFAGENT_SERVICE_LOG_DIR="$log_dir" \
      CONFAGENT_STOP_RUN_ID="$launch_id" bash "$code_dir/stop_replicas.sh" || true
  fi
}
trap cleanup_failed_launch EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
pids=()
for index in "${!gpus[@]}"; do
  gpu=${gpus[$index]}
  port=$((base_port + index))
  state_path="$log_dir/confagent_gpu${gpu}_port${port}.pid"
  log_path="$log_dir/vllm_gpu${gpu}_port${port}.log"
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$gpu" CONFAGENT_SERVICE_ROOT="$project_dir" \
    CONFAGENT_SERVICE_LOG_DIR="$log_dir" CONFAGENT_SERVICE_RUN_ID="$launch_id" SPT_NOENV=1 \
    nohup "$python_cmd" -m vllm.entrypoints.cli.main serve \
      "$model_path" --served-model-name "$served_name" \
      --host 127.0.0.1 --port "$port" --dtype bfloat16 \
      --max-model-len "${MAX_LEN:-8192}" --gpu-memory-utilization "${GMU:-0.85}" \
      --max-num-seqs 2 --enable-auto-tool-choice --tool-call-parser hermes \
      --generation-config vllm > "$log_path" 2>&1 < /dev/null 9>&- &
  pid=$!
  pids+=("$pid")
  printf '%s %s\n' "$pid" "$launch_id" > "$state_path"
  printf 'Started GPU %s on port %s. Log: %s\n' "$gpu" "$port" "$log_path"
done

for ((attempt = 0; attempt < 84; attempt++)); do
  ready=0
  for index in "${!pids[@]}"; do
    port=$((base_port + index))
    if ! kill -0 "${pids[$index]}" 2>/dev/null; then
      echo "Replica on port $port exited. Check its log in $log_dir." >&2
      exit 1
    fi
    if curl --fail --silent --noproxy '*' --max-time 2 "http://127.0.0.1:$port/v1/models" >/dev/null; then
      ready=$((ready + 1))
    fi
  done
  if [[ "$ready" -eq "${#pids[@]}" ]]; then
    printf 'Ready:'
    for index in "${!pids[@]}"; do
      printf ' http://127.0.0.1:%s/v1' "$((base_port + index))"
    done
    printf '\n'
    exit 0
  fi
  sleep 5
done
echo "Replica startup timed out. Check logs in $log_dir." >&2
exit 1
