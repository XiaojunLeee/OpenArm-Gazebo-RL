#!/usr/bin/env python3
"""生成 V1 固定训练、验证、测试孔位数据集；训练时不应再次调用本脚本。"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def unique_uniform(seed: int, count: int, used: set[tuple[float, float]]) -> np.ndarray:
    """使用独立 RNG 生成 ±3 mm 数据，并确定性消除集合内/集合间重复。"""
    rng = np.random.default_rng(seed)
    rows: list[np.ndarray] = []
    while len(rows) < count:
        candidate = rng.uniform(-0.003, 0.003, size=2).astype(np.float64)
        key = tuple(candidate.tolist())
        if key not in used:
            used.add(key)
            rows.append(candidate)
    return np.asarray(rows, dtype=np.float64)


def main() -> None:
    package_root = Path(__file__).resolve().parents[1]
    dataset_dir = package_root / 'datasets'
    dataset_dir.mkdir(exist_ok=True)
    used: set[tuple[float, float]] = set()
    specification = {
        'train': (100, 1000, 'train_holes.npy'),
        'validation': (200, 30, 'validation_holes.npy'),
        'test': (300, 100, 'test_holes.npy'),
    }
    metadata = {
        'unit': 'm', 'coordinate_order': ['delta_y', 'delta_z'],
        'range_per_axis_m': [-0.003, 0.003],
        'generator': 'numpy.random.default_rng',
        'generated_at_utc': datetime.now(timezone.utc).isoformat(),
        'sets': {},
    }
    for name, (seed, count, filename) in specification.items():
        values = unique_uniform(seed, count, used)
        np.save(dataset_dir / filename, values, allow_pickle=False)
        metadata['sets'][name] = {'seed': seed, 'count': count, 'file': filename,
                                  'dtype': str(values.dtype)}
    (dataset_dir / 'dataset_metadata.json').write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'固定孔位数据已生成：{dataset_dir}')


if __name__ == '__main__':
    main()
