#!/usr/bin/env bash
# 启动 Gazebo SAC 仿真，并把本次完整终端输出实时保存到 sac_results/logs/simulation.log。
set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="$ROOT_DIR/sac_results/logs"
LOG_FILE="$LOG_DIR/simulation.log"

mkdir -p "$LOG_DIR"
: > "$LOG_FILE"

if [[ $# -eq 0 ]]; then
    set -- rviz:=false
fi

{
    echo "===== OpenArm SAC 仿真日志 ====="
    echo "开始时间: $(date --iso-8601=seconds)"
    echo "工作目录: $ROOT_DIR"
    echo "命令: ros2 launch openarm_gazebo_hole_search sac_hole_search_sim.launch.py $*"
    echo "日志文件: $LOG_FILE"
    echo "================================"
} | tee -a "$LOG_FILE"

cd "$ROOT_DIR"
stdbuf -oL -eL ros2 launch openarm_gazebo_hole_search \
    sac_hole_search_sim.launch.py "$@" 2>&1 | tee -a "$LOG_FILE"
status=${PIPESTATUS[0]}

{
    echo "================================"
    echo "结束时间: $(date --iso-8601=seconds)"
    echo "仿真进程退出码: $status"
} | tee -a "$LOG_FILE"

exit "$status"
