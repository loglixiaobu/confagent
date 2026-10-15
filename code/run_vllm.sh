#!/bin/bash
PROJECT_ROOT="${PROJECT_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
# ============================================================================
# 启动 vLLM 服务（Qwen3-8B，单卡）
#
# 关键点：本项目的置信度信号源依赖 token 级 logprob，
#         而 vLLM 的 OpenAI 兼容接口原生支持 logprobs，这是选本地模型的决定性理由。
#
# 用法：
#   bash run_vllm.sh              # 默认用 GPU 0
#   GPU=3 bash run_vllm.sh        # 指定用 GPU 3
#   PORT=8001 bash run_vllm.sh    # 换端口（多人并行时用）
# ============================================================================

set -e

source "${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
conda activate confagent

MODEL_PATH="${MODEL_PATH:-${PROJECT_ROOT}/models/Qwen3-8B}"
SERVED_NAME="${SERVED_NAME:-Qwen3-8B}"
GPU="${GPU:-0}"
PORT="${PORT:-8000}"
MAX_LEN="${MAX_LEN:-16384}"
GMU="${GMU:-0.90}"
LOG="${PROJECT_ROOT}/logs/vllm_gpu${GPU}_port${PORT}.log"

export CUDA_VISIBLE_DEVICES="${GPU}"

echo "=============================================="
echo " model      : ${MODEL_PATH}"
echo " served as  : ${SERVED_NAME}"
echo " GPU        : ${GPU}   (CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES})"
echo " port       : ${PORT}"
echo " max-len    : ${MAX_LEN}"
echo " log        : ${LOG}"
echo "=============================================="

# --enable-auto-tool-choice + --tool-call-parser 让服务端直接解析 tool_calls
# Qwen3 系列使用 Hermes 风格的工具调用模板
nohup vllm serve "${MODEL_PATH}" \
  --served-model-name "${SERVED_NAME}" \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --dtype bfloat16 \
  --max-model-len "${MAX_LEN}" \
  --gpu-memory-utilization "${GMU}" \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  > "${LOG}" 2>&1 &

echo "PID=$!"
echo
echo "等待服务就绪（最多 300 秒）..."
for i in $(seq 1 60); do
  if curl -s --max-time 3 "http://localhost:${PORT}/v1/models" > /dev/null 2>&1; then
    echo "服务已就绪（约 $((i*5)) 秒）"
    curl -s "http://localhost:${PORT}/v1/models"
    echo
    echo
    echo "下一步："
    echo "  python ${PROJECT_ROOT}/code/verify_logprobs.py --base-url http://localhost:${PORT}/v1"
    exit 0
  fi
  sleep 5
done

echo "[WARN] 300 秒内未就绪，请查看日志：tail -50 ${LOG}"
exit 1
