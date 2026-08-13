"""V1 冻结奖励的无 ROS 纯函数。"""

from __future__ import annotations

import numpy as np


def is_reverse_action(action: np.ndarray, previous_action: np.ndarray) -> bool:
    """仅在两个动作模长都大于 0.2 且夹角余弦小于 -0.5 时判为反向。"""
    current = np.asarray(action, dtype=np.float64)
    previous = np.asarray(previous_action, dtype=np.float64)
    current_norm, previous_norm = np.linalg.norm(current), np.linalg.norm(previous)
    return bool(current_norm > 0.2 and previous_norm > 0.2 and (
        np.dot(current, previous) / (current_norm * previous_norm)) < -0.5)



def is_large_turn_action(action: np.ndarray, previous_action: np.ndarray) -> bool:
    """两个动作模长均 > 0.2，且相邻动作夹角在 60°～120° 时判为中等大转向。"""
    current = np.asarray(action, dtype=np.float64)
    previous = np.asarray(previous_action, dtype=np.float64)

    current_norm = np.linalg.norm(current)
    previous_norm = np.linalg.norm(previous)

    # 动作很小时不根据方向角惩罚，避免微小动作被误判
    if current_norm <= 0.2 or previous_norm <= 0.2:
        return False

    cosine = np.dot(current, previous) / (current_norm * previous_norm)
    cosine = np.clip(cosine, -1.0, 1.0)

    # cos(60°) = +0.5
    # cos(120°) = -0.5
    # 因此 60° <= theta <= 120° 等价于 -0.5 <= cos(theta) <= +0.5
    return bool(-0.5 <= cosine <= 0.5)

def search_step_reward(cell_event: str, action: np.ndarray, previous_action: np.ndarray,
                       near_boundary: bool) -> float:
    """计算一个非终止 SAC 搜索步的冻结奖励。"""
    reward = -0.01
    if cell_event == 'new':
        reward += 0.05
    elif cell_event == 'revisit':
        reward -= 0.02
    # 相邻动作方向平滑性惩罚（互斥分段）
    # theta > 120°：强烈反向，惩罚 -0.10
    # 60° <= theta <= 120°：中等大转向，惩罚 -0.04
    # theta < 60°：不额外惩罚
    if is_reverse_action(action, previous_action):
        reward -= 0.10
    elif is_large_turn_action(action, previous_action):
        reward -= 0.06

    if near_boundary:
        reward -= 0.15
    return reward
