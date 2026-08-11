#!/usr/bin/env bash
# 启动 SAC 训练，并实时保存到 sac_results/logs/training.log。
set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="$ROOT_DIR/sac_results/logs"
LOG_FILE="$LOG_DIR/training.log"

mkdir -p "$LOG_DIR"
: > "$LOG_FILE"

{
    echo "===== OpenArm SAC 训练日志 ====="
    echo "开始时间: $(date --iso-8601=seconds)"
    echo "工作目录: $ROOT_DIR"
    echo "Python: $(command -v python)"
    echo "命令: python -u scripts/train_sac.py"
    echo "自动续训: 开启（存在兼容检查点时自动读取）"
    echo "日志文件: $LOG_FILE"
    echo "================================"
} | tee -a "$LOG_FILE"

cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
stdbuf -oL -eL python -u scripts/train_sac.py 2>&1 | (trap '' INT; tee -a "$LOG_FILE")
status=${PIPESTATUS[0]}

{
    echo "================================"
    echo "结束时间: $(date --iso-8601=seconds)"
    echo "训练进程退出码: $status"
} | tee -a "$LOG_FILE"

exit "$status"
