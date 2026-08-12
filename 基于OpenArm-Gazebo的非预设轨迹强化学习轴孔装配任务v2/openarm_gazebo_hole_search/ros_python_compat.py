"""Conda Python 与 ROS 2 Humble Python 包的兼容导入。

某些终端中 ``install/setup.bash`` 会覆盖 Conda 的 ``PYTHONPATH``。训练脚本
仍须使用 Conda 的 PyTorch/SB3，同时加载系统安装的 ROS 2 Python 3.10 包，
因此在首次缺少 ROS 包时仅补充这两个 Humble 安装目录。
"""

from __future__ import annotations

import sys


def ensure_ros_python_path() -> None:
    """确保 rclpy、std_msgs 等 ROS 2 Python 包可被当前 Conda Python 找到。"""
    for path in (
        '/opt/ros/humble/lib/python3.10/site-packages',
        '/opt/ros/humble/local/lib/python3.10/dist-packages',
    ):
        if path not in sys.path:
            sys.path.insert(0, path)
