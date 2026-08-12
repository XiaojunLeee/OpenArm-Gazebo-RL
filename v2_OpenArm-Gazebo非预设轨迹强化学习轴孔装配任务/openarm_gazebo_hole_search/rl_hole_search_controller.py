#!/usr/bin/env python3
"""SAC 搜索控制入口：只替换 SEARCHING 的 Y/Z 参考，不改变其余状态。"""

from __future__ import annotations

import json
import math
from typing import Optional

import numpy as np
import rclpy
from std_msgs.msg import String

from .spiral_hole_search_controller import SpiralHoleSearchController


class RlHoleSearchController(SpiralHoleSearchController):
    """复用原 500 Hz 力矩控制；SAC 只在 SEARCHING 发出二维位置增量。"""

    def __init__(self) -> None:
        # 父类创建原控制器的力矩循环和状态机，插入及保持逻辑不被覆盖。
        super().__init__()
        self.declare_parameter('sac_action_period', 0.05)
        self.declare_parameter('sac_action_step', 0.0002)
        self.declare_parameter('sac_search_half_range', 0.004)
        self.declare_parameter('sac_state_publish_period', 0.05)
        self.declare_parameter('action_topic', '/sac_hole_search/action')
        self.declare_parameter('state_topic', '/sac_hole_search/state')

        self.pending_action: Optional[np.ndarray] = None
        self.search_center: Optional[np.ndarray] = None
        self.rl_reference_yz: Optional[np.ndarray] = None
        self.last_action_sim_time = -math.inf
        self.low_level_cycles = 0
        self.cycles_since_action = 0
        self.last_state_publish_time = -math.inf
        self.out_of_bounds = False
        self.create_subscription(
            String, str(self.get_parameter('action_topic').value),
            self.action_callback, 10)
        self.state_pub = self.create_publisher(
            String, str(self.get_parameter('state_topic').value), 10)
        self.get_logger().info('RL search interface ready; other states stay rule-based')

    def action_callback(self, msg: String) -> None:
        """缓存一个 ``{"action": [a_y, a_z]}``，不在回调中重复累加。"""
        try:
            action = np.asarray(json.loads(msg.data)['action'], dtype=np.float64)
            if action.shape != (2,) or not np.all(np.isfinite(action)):
                raise ValueError('动作必须是两个有限数')
            if np.any(action < -1.0) or np.any(action > 1.0):
                raise ValueError('动作必须位于 [-1, 1]^2')
            self.pending_action = action
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            self.get_logger().warning(f'忽略无效 SAC 动作：{exc}')

    def sample_spiral(self, elapsed: float):
        """SEARCHING 返回 SAC 参考位置；原螺旋函数只在原入口中继续使用。"""
        del elapsed
        if self.center is None:
            return np.zeros(2), np.zeros(2), np.zeros(2), 0.0
        reference = self.rl_reference_yz
        if reference is None:
            reference = self.center[1:3]
        reference = np.asarray(reference, dtype=np.float64)
        return reference.copy(), np.zeros(2), np.zeros(2), float(
            np.linalg.norm(reference - self.center[1:3]))

    def control_callback(self) -> None:
        """500 Hz 原循环之前，最多每 0.05 个仿真秒应用一次动作。"""
        now = self.get_clock().now().nanoseconds * 1e-9
        previous_state = self.state
        if self.state == 'SEARCHING':
            self._apply_action_once(now)
        super().control_callback()
        self.low_level_cycles += 1
        if self.state == 'SEARCHING':
            self.cycles_since_action += 1
        if previous_state != 'SEARCHING' and self.state == 'SEARCHING':
            self.search_center = np.array(
                self.data.oMf[self.tool_frame_id].translation, dtype=np.float64)
            self.rl_reference_yz = self.search_center[1:3].copy()
            self.out_of_bounds = False
            self.cycles_since_action = 0
            self.get_logger().info('SAC SEARCHING started from the measured tool tip')
        self._publish_state(now)

    def _apply_action_once(self, now: float) -> None:
        """先检查未裁剪候选位置，再一次性更新参考 Y/Z。"""
        if self.pending_action is None or self.search_center is None:
            return
        if now - self.last_action_sim_time < float(
            self.get_parameter('sac_action_period').value):
            return
        action, self.pending_action = self.pending_action, None
        proposed = self.rl_reference_yz + action * float(
            self.get_parameter('sac_action_step').value)
        half_range = float(self.get_parameter('sac_search_half_range').value)
        if np.any(np.abs(proposed - self.search_center[1:3]) > half_range):
            self.out_of_bounds = True
            self.get_logger().warning('SAC candidate reference is out of bounds')
            return
        self.rl_reference_yz = proposed
        self.last_action_sim_time = now
        self.cycles_since_action = 0

    def _publish_state(self, now: float) -> None:
        """仅发布接口调试状态，绝不发布孔真值或虚拟接触反力。"""
        if now - self.last_state_publish_time < float(
            self.get_parameter('sac_state_publish_period').value):
            return
        self.last_state_publish_time = now
        message = String()
        message.data = json.dumps({
            'controller_state': self.state,
            'out_of_bounds': self.out_of_bounds,
            'low_level_cycles': self.low_level_cycles,
            'cycles_since_action': self.cycles_since_action,
            'reference_yz_m': (
                self.rl_reference_yz.tolist()
                if self.rl_reference_yz is not None else None),
        })
        self.state_pub.publish(message)


def main(args=None) -> None:
    """ROS 2 console script 入口。"""
    rclpy.init(args=args)
    node = RlHoleSearchController()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
