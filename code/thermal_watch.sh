#!/bin/bash
# ============================================================================
# GPU 温度监控（后台常驻，超过阈值时写告警日志）
#
# 背景：RTX 3090 的 GDDR6X 显存结温在持续满载下可超过 100°C，而该传感器
#       不通过 nvidia-smi 暴露。GPU 从 PCIe 总线上掉下来（掉卡）通常就是
#       长期高温所致。本脚本持续记录核心温度，供事后判断是否存在过热风险。
#
# 用法：
#   nohup bash thermal_watch.sh > /dev/null 2>&1 &
#   tail -f /home1/lh/confagent/logs/thermal.log
# ============================================================================

INTERVAL="${INTERVAL:-30}"
THRESHOLD="${THRESHOLD:-78}"
LOG=/home1/lh/confagent/logs/thermal.log
mkdir -p "$(dirname "$LOG")"

echo "[$(date '+%F %T')] thermal_watch 启动 | 采样间隔 ${INTERVAL}s | 告警阈值 ${THRESHOLD}°C" >> "$LOG"

while true; do
  TS=$(date '+%F %T')
  LINE=$(nvidia-smi --query-gpu=index,temperature.gpu,utilization.gpu,power.draw \
         --format=csv,noheader,nounits 2>/dev/null | tr '\n' ';')
  MAXT=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits 2>/dev/null \
         | sort -n | tail -1)

  echo "[$TS] max=${MAXT}C | $LINE" >> "$LOG"

  if [ -n "$MAXT" ] && [ "$MAXT" -ge "$THRESHOLD" ] 2>/dev/null; then
    echo "[$TS] !! ALERT 最高温 ${MAXT}°C >= ${THRESHOLD}°C —— 建议暂停任务或降低功耗上限" >> "$LOG"
  fi

  sleep "$INTERVAL"
done
