#!/bin/bash
PROJECT_ROOT="${PROJECT_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
# ============================================================================
# 在指定的 GPU 组上启动多个 vLLM 副本（每个副本独占一张卡，端口递增）
#
# 用法：
#   bash launch_replicas.sh "0 1 2 3"        # GPU 0-3，端口 8000-8003
#   bash launch_replicas.sh "4 5 6 7" 8004   # GPU 4-7，端口 8004-8007
#
# 设计意图（温度管理）：
#   4 张一组轮换使用，另一组 4 张在批次间得到降温时间，避免持续满载过热。
# ============================================================================

set -u

GPUS="${1:-0 1 2 3}"
BASE_PORT="${2:-8000}"

source "${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
conda activate confagent

MODEL_PATH="${MODEL_PATH:-${PROJECT_ROOT}/models/Qwen3-8B}"
# 注意：不要在这里设 CUDA_DEVICE_ORDER=PCI_BUS_ID。
# 2026-10-14 实测：在本机（已有 GPU 掉出总线）设置该变量并不能修正设备映射，
# 反而会引入额外的初始化失败风险。保持默认顺序即可。
SERVED_NAME="${SERVED_NAME:-Qwen3-8B}"
MAX_LEN="${MAX_LEN:-16384}"
GMU="${GMU:-0.90}"
LOGDIR=${PROJECT_ROOT}/logs
mkdir -p "$LOGDIR"

echo "=============================================="
echo " 启动副本   GPU 组: $GPUS   起始端口: $BASE_PORT"
echo " （使用 vllm_serve_patched.py 以容忍故障 GPU）"
echo "=============================================="

i=0
PORTS=""
for g in $GPUS; do
  port=$((BASE_PORT + i))
  PORTS="$PORTS $port"
  log="$LOGDIR/vllm_gpu${g}_port${port}.log"
  echo "  GPU $g -> port $port  (log: $log)"
  CUDA_VISIBLE_DEVICES="$g" nohup python ${PROJECT_ROOT}/code/vllm_serve_patched.py serve "$MODEL_PATH" \
    --served-model-name "$SERVED_NAME" \
    --host 0.0.0.0 --port "$port" \
    --dtype bfloat16 --max-model-len "$MAX_LEN" \
    --gpu-memory-utilization "$GMU" \
    --enable-auto-tool-choice --tool-call-parser hermes \
    > "$log" 2>&1 &
  i=$((i + 1))
  sleep 2
done

echo
echo "等待全部副本就绪（最多 420 秒）..."
for t in $(seq 1 84); do
  ready=0
  for p in $PORTS; do
    if curl -s --max-time 2 "http://localhost:${p}/v1/models" > /dev/null 2>&1; then
      ready=$((ready + 1))
    fi
  done
  if [ "$ready" -eq "$i" ]; then
    echo "全部 $ready 个副本已就绪（约 $((t*5)) 秒）"
    ENDPOINTS=""
    for p in $PORTS; do
      ENDPOINTS="${ENDPOINTS}localhost:${p},"
    done
    ENDPOINTS="${ENDPOINTS%,}"
    echo "ENDPOINTS=$ENDPOINTS"
    echo
    echo "就绪端口:$PORTS"
    exit 0
  fi
  sleep 5
done

echo "[WARN] 超时，已就绪 $ready/$i 个副本。检查日志："
for g in $GPUS; do
  echo "  tail -30 $LOGDIR/vllm_gpu${g}_port*.log"
done
exit 1
