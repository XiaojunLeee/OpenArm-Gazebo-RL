"""冻结孔位数据集的加载与校验。每个元素是相对 SEARCHING 起点的 [Y, Z] 米。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class HoleDataset:
    """只读的固定孔偏移集合。"""

    name: str
    offsets_m: np.ndarray

    def __post_init__(self) -> None:
        values = np.asarray(self.offsets_m, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != 2:
            raise ValueError('孔位数据必须为 shape=(N, 2)')
        if not np.all(np.isfinite(values)) or np.any(np.abs(values) > 0.003):
            raise ValueError('每轴孔位偏移必须是有限值且在 ±3 mm 内')
        object.__setattr__(self, 'offsets_m', values)

    def offset(self, index: int) -> np.ndarray:
        return self.offsets_m[int(index) % len(self.offsets_m)].copy()

    def __len__(self) -> int:
        return len(self.offsets_m)


def load_dataset(path: Path, name: str, expected_size: int) -> HoleDataset:
    """读取一个不可在训练时重生成的 ``.npy`` 固定数据集。"""
    values = np.load(path, allow_pickle=False)
    if values.shape != (expected_size, 2):
        raise ValueError(f'{name} 数据集应为 ({expected_size}, 2)，实际为 {values.shape}')
    return HoleDataset(name=name, offsets_m=values)
