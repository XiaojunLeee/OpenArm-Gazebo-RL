"""V1 冻结的 92 维 SAC 观测构造；不接收孔坐标或接触真值。"""

from __future__ import annotations

import numpy as np


def build_observation(
    delta_xyz_m: np.ndarray, velocity_xyz_m_s: np.ndarray,
    remaining_time_fraction: float, previous_action: np.ndarray,
    fy_fz_command_n: np.ndarray, visit_map_flat: np.ndarray,
    x_scale_m: float, velocity_scale_m_s: float, yz_force_limit_n: float,
) -> np.ndarray:
    """返回顺序固定的 92 维 float32 向量。

    输入不包含真实孔位、到孔误差或虚拟接触反力，因此不会向 Actor 泄漏真值。
    """
    delta = np.asarray(delta_xyz_m, dtype=np.float64)
    velocity = np.asarray(velocity_xyz_m_s, dtype=np.float64)
    action = np.asarray(previous_action, dtype=np.float64)
    force = np.asarray(fy_fz_command_n, dtype=np.float64)
    visit = np.asarray(visit_map_flat, dtype=np.float32)
    if delta.shape != (3,) or velocity.shape != (3,) or action.shape != (2,):
        raise ValueError('位移、速度、上一动作维度分别必须是 3、3、2')
    if force.shape != (2,) or visit.shape != (81,):
        raise ValueError('力与访问图维度分别必须是 2、81')
    result = np.concatenate((
        np.clip(delta / np.array([x_scale_m, 0.004, 0.004]), -1.0, 1.0),
        np.clip(velocity / velocity_scale_m_s, -1.0, 1.0),
        [np.clip(remaining_time_fraction, 0.0, 1.0)],
        np.clip(action, -1.0, 1.0),
        np.clip(force / yz_force_limit_n, -1.0, 1.0),
        visit,
    )).astype(np.float32)
    if result.shape != (92,) or not np.all(np.isfinite(result)):
        raise ValueError('观测必须是有限的 92 维向量')
    return result
