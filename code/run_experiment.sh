#!/bin/bash
PROJECT_ROOT="${PROJECT_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
# ============================================================================
# 一条命令跑一轮实验：指定 GPU 组 -> 启动副本 -> 跑批处理 -> 报告温度
#
# 用法：
#   bash run_experiment.sh "0 1 2 3" 8000 --limit 50 --concurrency 2
#   bash run_experiment.sh "4 5 6 7" 8004 --limit 50 --concurrency 2 --tag seed1
#
# 温度管理建议：
#   第 1 轮用 GPU 0-3，第 2 轮用 GPU 4-7，交替进行，让另一组有时间降温。
#   跑完一轮可加 --stop 参数停掉副本（下一轮换另一组）。
# ============================================================================

set -u

GPUS="${1:?用法: bash run_experiment.sh \"0 1 2 3\" [base_port] [run_batch 参数...]}"
shift
BASE_PORT="${1:-8000}"
shift 2>/dev/null || true

STOP_AFTER=0
ARGS=""
for a in "$@"; do
  if [ "$a" = "--stop" ]; then STOP_AFTER=1; else ARGS="$ARGS $a"; fi
done

CODE=${PROJECT_ROOT}/code
source "${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
conda activate confagent

# 构造端点列表
ENDPOINTS=""
i=0
for g in $GPUS; do
  port=$((BASE_PORT + i))
  ENDPOINTS="${ENDPOINTS}localhost:${port},"
  i=$((i + 1))
done
ENDPOINTS=$(echo "$ENDPOINTS" | sed 's/,$//')

echo "################################################################"
echo "# 实验轮次   GPU 组: $GPUS"
echo "# 端点: $ENDPOINTS"
echo "# 参数:$ARGS"
echo "################################################################"
echo

echo "--- 启动前温度 ---"
nvidia-smi --query-gpu=index,temperature.gpu,memory.used --format=csv,noheader
echo

bash "$CODE/launch_replicas.sh" "$GPUS" "$BASE_PORT" || exit 1
echo

echo "--- 运行中温度 ---"
nvidia-smi --query-gpu=index,temperature.gpu,memory.used,power.draw --format=csv,noheader
echo

cd "$CODE"
python run_batch.py --endpoints "$ENDPOINTS" $ARGS
RC=$?

echo
echo "--- 运行后温度 ---"
nvidia-smi --query-gpu=index,temperature.gpu,memory.used --format=csv,noheader

if [ "$STOP_AFTER" = "1" ]; then
  echo
  echo "--- 停止本组副本（便于下一轮换另一组 GPU） ---"
  bash "$CODE/stop_replicas.sh" $GPUS
fi

exit $RC
