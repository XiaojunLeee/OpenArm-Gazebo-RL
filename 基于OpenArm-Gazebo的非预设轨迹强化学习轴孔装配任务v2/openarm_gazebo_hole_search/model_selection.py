"""验证集最佳模型的冻结字典序比较。"""

from __future__ import annotations

from typing import Mapping, Tuple


def metric_key(metrics: Mapping[str, float]) -> Tuple[float, float, float, float, float]:
    """数值越大越优；后两项取负号以实现时间/路径越短越优。"""
    search_time = metrics.get('mean_successful_search_time_s')
    path_length = metrics.get('mean_successful_yz_path_length_m')
    return (
        float(metrics['assembly_success_rate']),
        float(metrics['hole_search_success_rate']),
        -float(metrics['fault_rate']),
        -float('inf') if search_time is None else -float(search_time),
        -float('inf') if path_length is None else -float(path_length),
    )


def is_better(candidate: Mapping[str, float], incumbent: Mapping[str, float] | None) -> bool:
    """严格按装配率、找孔率、FAULT率、时间、路径的优先级选择最佳模型。"""
    return incumbent is None or metric_key(candidate) > metric_key(incumbent)
