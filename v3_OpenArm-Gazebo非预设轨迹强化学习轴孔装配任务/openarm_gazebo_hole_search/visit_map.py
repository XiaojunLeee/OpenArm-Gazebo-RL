"""SAC 搜索阶段的 9×9 访问历史。

本模块不依赖 ROS 或 Gazebo，便于单元测试。坐标顺序统一为 ``[Y, Z]``；
二维数组第一维是 Y 的网格行，第二维是 Z 的网格列，最后以 C 顺序展平。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np


@dataclass
class VisitMap:
    """记录本 Episode 内已经访问过的搜索网格。"""

    center_yz_m: np.ndarray
    half_range_m: float = 0.004
    cells_per_axis: int = 9
    visited: np.ndarray = field(init=False)
    current_cell: Optional[Tuple[int, int]] = field(default=None, init=False)

    def __post_init__(self) -> None:
        self.center_yz_m = np.asarray(self.center_yz_m, dtype=np.float64)
        if self.center_yz_m.shape != (2,):
            raise ValueError('搜索中心必须是 [Y, Z] 两个坐标')
        if self.half_range_m <= 0.0 or self.cells_per_axis <= 0:
            raise ValueError('搜索范围和网格数量必须为正数')
        self.visited = np.zeros(
            (self.cells_per_axis, self.cells_per_axis), dtype=np.float32)

    def cell_for(self, yz_m: np.ndarray) -> Tuple[int, int]:
        """将范围内坐标映射为确定的 ``(Y行, Z列)``，边界不会产生索引 9。"""
        yz = np.asarray(yz_m, dtype=np.float64)
        normalized = (yz - (self.center_yz_m - self.half_range_m)) / (
            2.0 * self.half_range_m)
        raw = np.floor(normalized * self.cells_per_axis).astype(int)
        clipped = np.clip(raw, 0, self.cells_per_axis - 1)
        return int(clipped[0]), int(clipped[1])

    def enter(self, yz_m: np.ndarray) -> str:
        """登记当前位置，返回 ``start/new/revisit/stay`` 四种事件之一。"""
        cell = self.cell_for(yz_m)
        if self.current_cell == cell:
            return 'stay'
        was_visited = bool(self.visited[cell])
        self.visited[cell] = 1.0
        previous = self.current_cell
        self.current_cell = cell
        if previous is None:
            return 'start'
        return 'revisit' if was_visited else 'new'

    def flattened(self) -> np.ndarray:
        """按固定 C 顺序返回 81 维 float32 访问图。"""
        return self.visited.reshape(-1, order='C').astype(np.float32, copy=True)
