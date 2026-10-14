#!/bin/bash
# ============================================================================
# 停止 vLLM 副本，并等待显存释放（用于 GPU 组轮换）
#
# 注意：pkill 的匹配串写成 "vllm [s]erve" 这种带字符类的形式，
#       否则 pkill 会匹配到"执行这条命令的 shell 自身"（命令行里含同样的字符串），
#       导致脚本把自己杀掉。
#
# 用法：
#   bash stop_replicas.sh              # 停掉所有副本
#   bash stop_replicas.sh 1 2 3 4      # 只停指定 GPU 上的副本
# ============================================================================

PAT_SERVE='vllm [s]erve'
PAT_PATCHED='vllm_serve_[p]atched'

echo "=== 当前副本进程 ==="
pgrep -af "$PAT_SERVE" || echo "(无 vllm serve)"
pgrep -af "$PAT_PATCHED" || echo "(无 patched wrapper)"

TARGET_GPUS="$*"

if [ -z "$TARGET_GPUS" ]; then
  echo
  echo "停止全部副本..."
  pkill -f "$PAT_SERVE" 2>/dev/null
  pkill -f "$PAT_PATCHED" 2>/dev/null
else
  echo
  echo "只停止 GPU 组 [$TARGET_GPUS] 上的副本..."
  for pid in $(pgrep -f "$PAT_SERVE"; pgrep -f "$PAT_PATCHED"); do
    for g in $TARGET_GPUS; do
      if tr '\0' '\n' < /proc/$pid/environ 2>/dev/null | grep -qx "CUDA_VISIBLE_DEVICES=$g"; then
        echo "  kill $pid (GPU $g)"
        kill "$pid" 2>/dev/null
        break
      fi
    done
  done
fi

echo
echo "等待进程退出与显存释放..."
for t in $(seq 1 30); do
  n=$(pgrep -f "$PAT_SERVE" | wc -l)
  if [ "$n" -eq 0 ]; then
    echo "已全部退出（约 $((t*2)) 秒）"
    break
  fi
  sleep 2
done

echo
echo "=== GPU 状态 ==="
nvidia-smi --query-gpu=index,memory.used,temperature.gpu --format=csv,noheader 2>&1
