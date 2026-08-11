#!/usr/bin/env python3
"""SAC 运行控制器：仅用 SAC 替换 SEARCHING 的 Y/Z 参考。

其余状态一律调用原 ``SpiralHoleSearchController``，因此不会修改原螺旋
入口、接近、插入或插入保持控制代码。
"""
from __future__ import annotations

import json
import math
from typing import Optional

import numpy as np
import pinocchio as pin
import rclpy
from geometry_msgs.msg import TransformStamped
from rclpy.parameter import Parameter
from rclpy.qos import (
    DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy,
)
from ros_gz_interfaces.msg import Entity
from ros_gz_interfaces.srv import DeleteEntity, SpawnEntity
from std_msgs.msg import Float64MultiArray, String
from tf2_ros import TransformBroadcaster

from .spiral_hole_search_controller import SpiralHoleSearchController


class SacRuntimeController(SpiralHoleSearchController):
    """带 Episode 接口的独立 RL 控制器，底层规则控制仍来自父类。"""

    VERSION = 1

    def __init__(self) -> None:
        super().__init__()
        # 冻结 SAC 接口与安全/复位参数。单位均在 YAML 的中文注释中说明。
        for name, value in (
            ('sac_action_period', 0.05), ('sac_action_step', 0.0002),
            ('sac_search_half_range', 0.004), ('sac_max_search_time', 20.0),
            ('sac_max_steps', 400), ('sac_holding_duration', 2.0),
            ('insertion_max_tilt_deg', 1.5),
            ('insertion_depth_tolerance', 0.0001),
            ('insertion_timeout', 60.0), ('reset_settle_time', 0.75),
            ('reset_joint_kp', [30., 30., 30., 30., 20., 20., 15.]),
            ('reset_joint_kd', [2., 2., 1.5, 1.5, .5, .5, .4]),
            ('reset_joint_positions', [0.079866760, 1.132669807,
             -0.046520126, 1.107354402, -0.026191662, 0.016393745,
             0.430011105]),
            ('fixture_model_name', 'hole_fixture'), ('fixture_center_x', .33220),
            ('board_center_y', -0.43800), ('board_center_z', 0.57242),
            ('board_size_y', 0.340), ('board_size_z', 0.620),
            ('board_depth_x', 0.017), ('hole_size_yz', 0.007),
            ('board_transparency', 0.65),
            ('world_name', 'hole_search'), ('action_topic', '/sac_hole_search/action'),
            ('command_topic', '/sac_hole_search/episode_command'),
            ('state_topic', '/sac_hole_search/state'),
            ('sac_state_publish_period', 0.05),
        ):
            self.declare_parameter(name, value)
        latest_reliable_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.create_subscription(
            String, str(self.get_parameter('action_topic').value),
            self._action_cb, latest_reliable_qos)
        self.create_subscription(
            String, str(self.get_parameter('command_topic').value),
            self._command_cb, latest_reliable_qos)
        self.state_pub = self.create_publisher(
            String, str(self.get_parameter('state_topic').value),
            latest_reliable_qos)
        self.tf_broadcaster = TransformBroadcaster(self)
        world_service = f'/world/{self.get_parameter("world_name").value}'
        self.delete_client = self.create_client(
            DeleteEntity, f'{world_service}/remove')
        self.spawn_client = self.create_client(
            SpawnEntity, f'{world_service}/create')

        self.resetting = False
        self.reset_started = -math.inf
        self.delete_future = None
        self.spawn_future = None
        self.board_reset_stage = 'idle'
        self.pending_hole_yz: Optional[np.ndarray] = None
        self.episode_active = False
        self.episode_done = False
        self.session_id = ''
        self.episode_id = -1
        self.reset_attempt = -1
        self.mode = 'training'
        self.dataset_index = -1
        self.offset_yz = np.zeros(2)
        self.pending_action: Optional[tuple[int, np.ndarray]] = None
        self.active_step: Optional[int] = None
        self.last_action_time = -math.inf
        self.search_center: Optional[np.ndarray] = None
        self.reference_yz: Optional[np.ndarray] = None
        self.search_started = -math.inf
        self.insertion_started: Optional[float] = None
        self.holding_started: Optional[float] = None
        self.hole_found = self.assembly_success = self.out_of_bounds = False
        self.timeout = self.insertion_failure = False
        self.low_level_cycles = self.sac_steps = 0
        self.last_publish = -math.inf
        self.last_reset_report = -math.inf
        self.last_fyz = np.zeros(2)
        self.state_sequence = 0
        self.get_logger().info('SAC runtime ready; only SEARCHING is policy-controlled')

    def sample_spiral(self, elapsed: float):
        """父类在 SEARCHING 调用这里；SAC 目标替代螺旋目标。"""
        del elapsed
        if self.reference_yz is None or self.search_center is None:
            base = self.center[1:3] if self.center is not None else np.zeros(2)
            return base.copy(), np.zeros(2), np.zeros(2), 0.0
        return self.reference_yz.copy(), np.zeros(2), np.zeros(2), float(np.linalg.norm(self.reference_yz - self.search_center[1:3]))

    def allow_hole_entry(self) -> bool:
        """至少执行一个SAC动作后才允许SEARCHING转入INSERTING。

        控制器必须先发布一次可被Gym接收的SEARCHING，避免孔恰好位于
        轴尖时在一个500 Hz周期内直接跳过SEARCHING。
        """
        return self.sac_steps >= 1

    def _command_cb(self, msg: String) -> None:
        """接收环境 reset；孔偏移只用于仿真和规则判定，永不发布给策略。"""
        try:
            data = json.loads(msg.data)
            if data.get('version') != self.VERSION or data.get('command') != 'reset':
                raise ValueError('仅支持 version=1 的 reset')
            offset = np.asarray([data['hole_offset_y_m'], data['hole_offset_z_m']], dtype=float)
            if offset.shape != (2,) or np.any(np.abs(offset) > .003):
                raise ValueError('孔偏移必须在每轴 ±3 mm 内')
            requested_episode_id = int(data['episode_id'])
            requested_session_id = str(data['session_id'])
            requested_reset_attempt = int(data.get('reset_attempt', 0))
            if not requested_session_id:
                raise ValueError('session_id 不能为空')
            if requested_reset_attempt < 0:
                raise ValueError('reset_attempt 不能为负数')

            same_episode = (
                requested_session_id == self.session_id
                and requested_episode_id == self.episode_id
            )
            # 环境会每 0.5 s 重发同一次 reset，因此同一 reset_attempt 必须
            # 幂等。若本 Episode 在进入 SEARCHING 前就因 APPROACHING/FAULT
            # 结束，环境会把 reset_attempt 加 1；此时允许同一 Episode 重新
            # 执行完整 reset，而不会被旧的“episode_active”状态永久挡住。
            if same_episode and requested_reset_attempt <= self.reset_attempt:
                return
            if same_episode and requested_reset_attempt > self.reset_attempt:
                self.get_logger().warning(
                    f'RESETTING retry episode={requested_episode_id}, '
                    f'attempt={requested_reset_attempt}, '
                    f'previous_state={self.state}, done={self.episode_done}')

            self.session_id = requested_session_id
            self.episode_id = requested_episode_id
            self.reset_attempt = requested_reset_attempt
            self.mode = str(data.get('mode', 'training'))
            self.dataset_index, self.offset_yz = int(data.get('dataset_index', -1)), offset
            self._begin_reset()
            self.get_logger().info(
                f'RESETTING episode={self.episode_id}, '
                f'dataset_index={self.dataset_index}, '
                f'offset_yz_mm='
                f'({self.offset_yz[0] * 1000.0:.3f}, '
                f'{self.offset_yz[1] * 1000.0:.3f})')
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            self.get_logger().error(f'拒绝 reset：{exc}')

    def _action_cb(self, msg: String) -> None:
        """缓存一个动作；同一动作只在一个 20 Hz 周期起点应用一次。"""
        try:
            data = json.loads(msg.data)
            action = np.asarray(data['action'], dtype=float)
            step = int(data['step_id'])
            if (
                str(data.get('session_id', '')) != self.session_id
                or int(data['episode_id']) != self.episode_id
                or action.shape != (2,)
            ):
                return
            if not np.all(np.isfinite(action)) or np.any(np.abs(action) > 1.0):
                raise ValueError('动作必须是 [-1, 1] 的二维有限向量')
            if self.active_step is None or step > self.active_step:
                self.pending_action = (step, action)
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            self.get_logger().warning(f'忽略无效动作：{exc}')

    def _begin_reset(self) -> None:
        """清空跨 Episode 缓存并进入受控关节复位，不改变父类规则算法。"""
        self.resetting, self.episode_active, self.episode_done = True, False, False
        self.reset_started = self.get_clock().now().nanoseconds * 1e-9
        # 每个 Episode 都删除并重建一次真实方孔板。必须清空两个异步
        # 服务缓存，否则下一回合会误用上一回合已经完成的 future。
        self.delete_future = None; self.spawn_future = None
        self.board_reset_stage = 'idle'; self.pending_hole_yz = None
        self.pending_action = None; self.active_step = None; self.search_center = None
        self.reference_yz = None; self.insertion_started = self.holding_started = None
        self.hole_found = self.assembly_success = self.out_of_bounds = self.timeout = self.insertion_failure = False
        self.low_level_cycles = self.sac_steps = 0
        self.filtered_insert_dq.fill(0.0); self.filtered_hold_dq.fill(0.0)
        self.state = 'RESETTING'

    def control_callback(self) -> None:
        now = self.get_clock().now().nanoseconds * 1e-9
        if self.resetting:
            self._reset_tick(now); self._publish(now); return
        if not self.episode_active:
            self.publish_zero(); self._publish(now); return
        if self.episode_done:
            self.publish_gravity_hold(); self._publish(now); return
        if self.state == 'SEARCHING':
            self._apply_action(now)
            if self.episode_done:
                self.publish_gravity_hold(); self._publish(now); return
        previous = self.state
        super().control_callback()
        self.low_level_cycles += 1
        if previous != 'HOLDING' and self.state == 'HOLDING':
            # 复位关节值已通过逆运动学保证轴尖位置不变且轴线对准孔轴；
            # 这里只打印实测倾角，不在运行中突然修改姿态目标。
            self.get_logger().info(
                f'HOLDING: reset tool tilt={self._tool_tilt_deg():.2f} deg')
        if previous != 'SEARCHING' and self.state == 'SEARCHING':
            self.search_center = self._tool_position()
            self.reference_yz = self.search_center[1:3].copy()
            self.search_started = now; self.sac_steps = 0
            # 状态变化绕过50 ms发布节流，保证Gym reset看见SEARCHING。
            self.last_publish = -math.inf
        # 父类每秒打印的 YZ_ref 在 SEARCHING 中是 SAC 的目标指令，
        # 不是孔坐标。状态刚切到 INSERTING 时额外明确打印孔中心参考，
        # 方便从终端核对真实孔心与工具位置的 Y/Z 误差。
        if previous == 'SEARCHING' and self.state == 'INSERTING':
            tip = self._tool_position()
            tilt_deg = self._tool_tilt_deg()
            hole_y = float(self.get_parameter('hole_center_y').value)
            hole_z = float(self.get_parameter('hole_center_z').value)
            self.get_logger().info(
                'INSERTING handoff: '
                f'HOLE_YZ_ref=({hole_y:.4f}, {hole_z:.4f}) m, '
                f'tool_YZ=({tip[1]:.4f}, {tip[2]:.4f}) m, '
                f'center_error_YZ_mm=({(tip[1] - hole_y) * 1000.0:.3f}, '
                f'{(tip[2] - hole_z) * 1000.0:.3f}), '
                f'tool_tilt={tilt_deg:.2f} deg')
            max_tilt = float(
                self.get_parameter('insertion_max_tilt_deg').value)
            if tilt_deg > max_tilt:
                # 倾斜状态下禁止继续沿 X 硬插。找到孔仍然记为成功，
                # 但本回合完整装配判为 insertion_failure。
                self.hole_found = True
                self.insertion_started = now
                self.insertion_failure = self.episode_done = True
                self.get_logger().warning(
                    f'INSERTING rejected: tool tilt {tilt_deg:.2f} deg '
                    f'exceeds {max_tilt:.2f} deg; Episode will reset')
        if not self.hole_found and self.state == 'INSERTING':
            self.hole_found, self.insertion_started = True, now
        # INSERTING 中允许底层深度/YZ 控制器消除瞬时耦合偏差。父类只有在
        # 深度、速度和每轴 ±0.5 mm 对中同时稳定后才会进入保持；因此这里
        # 不能因单个 500 Hz 采样越界就提前终止，否则控制器没有重新对中的
        # 时间。最终成功条件仍是严格 ±0.5 mm，超时则判定插入失败。
        if self.state == 'INSERT_HOLDING':
            self.holding_started = now if self.holding_started is None else self.holding_started
            if now - self.holding_started >= float(self.get_parameter('sac_holding_duration').value):
                # 不能仅因“保持了 2 秒”就认定装配成功。必须在保持结束时
                # 再次确认轴心仍在孔中心 ±0.5 mm 内。
                geometry = self._insertion_geometry()
                if geometry['valid']:
                    self.assembly_success = self.episode_done = True
                    self.get_logger().info(
                        'INSERT_HOLDING complete: physical geometry '
                        'confirmed; '
                        f'depth={geometry["depth_m"]:.4f} m, '
                        f'tip_error_YZ_mm='
                        f'({geometry["tip_error_yz_mm"][0]:.3f}, '
                        f'{geometry["tip_error_yz_mm"][1]:.3f}), '
                        f'entry_error_YZ_mm='
                        f'({geometry["entry_error_yz_mm"][0]:.3f}, '
                        f'{geometry["entry_error_yz_mm"][1]:.3f}), '
                        f'tilt={geometry["tilt_deg"]:.2f} deg; '
                        'Episode is an assembly success')
                else:
                    self.insertion_failure = self.episode_done = True
                    self.get_logger().warning(
                        'INSERT_HOLDING failed physical geometry check: '
                        f'depth={geometry["depth_m"]:.4f} m, '
                        f'tip_error_YZ_mm='
                        f'({geometry["tip_error_yz_mm"][0]:.3f}, '
                        f'{geometry["tip_error_yz_mm"][1]:.3f}), '
                        f'entry_error_YZ_mm='
                        f'({geometry["entry_error_yz_mm"][0]:.3f}, '
                        f'{geometry["entry_error_yz_mm"][1]:.3f}), '
                        f'tilt={geometry["tilt_deg"]:.2f} deg; '
                        'Episode will reset')
        if self.state == 'FAULT': self.episode_done = True
        if self.insertion_started and not self.episode_done and now - self.insertion_started > float(self.get_parameter('insertion_timeout').value):
            self.insertion_failure = self.episode_done = True
        self._publish(now)

    def _reset_tick(self, now: float) -> None:
        if self.model is None or self.q is None or self.dq is None: return
        target = np.asarray(self.get_parameter('reset_joint_positions').value, dtype=float)
        kp = np.asarray(self.get_parameter('reset_joint_kp').value, dtype=float)
        kd = np.asarray(self.get_parameter('reset_joint_kd').value, dtype=float)
        torque = np.clip(kp * (target - self.q) - kd * self.dq, -np.asarray(self.get_parameter('max_joint_torque').value), np.asarray(self.get_parameter('max_joint_torque').value))
        command = Float64MultiArray(); command.data = torque.tolist(); self.command_pub.publish(command)
        position_error = float(np.max(np.abs(target - self.q)))
        velocity_error = float(np.max(np.abs(self.dq)))
        if position_error > .01 or velocity_error > .10:
            if now - self.last_reset_report >= 1.0:
                self.get_logger().info(
                    f'RESETTING: joint_error={position_error:.4f} rad, '
                    f'joint_velocity={velocity_error:.4f} rad/s')
                self.last_reset_report = now
            return
        if self.board_reset_stage == 'idle':
            # Gazebo 尚在启动时服务可能不存在；保持复位力矩并周期重试。
            if (
                not self.delete_client.service_is_ready()
                or not self.spawn_client.service_is_ready()
            ):
                if now - self.last_reset_report >= 1.0:
                    self.get_logger().info(
                        'RESETTING: waiting for Gazebo remove/create services')
                    self.last_reset_report = now
                return
            tip = self._tool_position()
            self.pending_hole_yz = tip[1:3] + self.offset_yz
            self.set_parameters([
                Parameter('hole_center_y', value=float(
                    self.pending_hole_yz[0])),
                Parameter('hole_center_z', value=float(
                    self.pending_hole_yz[1])),
            ])
            request = DeleteEntity.Request()
            request.entity = Entity(
                name=str(self.get_parameter('fixture_model_name').value),
                type=Entity.MODEL)
            self.delete_future = self.delete_client.call_async(request)
            self.board_reset_stage = 'deleting'
            return

        if self.board_reset_stage == 'deleting':
            if self.delete_future is None or not self.delete_future.done():
                if now - self.last_reset_report >= 1.0:
                    self.get_logger().info(
                        'RESETTING: waiting for old board deletion')
                    self.last_reset_report = now
                return
            try:
                response = self.delete_future.result()
                if response is not None and not response.success:
                    # 模型已不存在时也可以继续生成，不能永久卡在删除阶段。
                    self.get_logger().warning(
                        'RESETTING: old board was already absent; spawning')
            except Exception as exc:
                self.get_logger().warning(
                    f'RESETTING: board deletion returned {exc}; spawning')
            request = SpawnEntity.Request()
            factory = request.entity_factory
            factory.name = str(
                self.get_parameter('fixture_model_name').value)
            factory.allow_renaming = False
            factory.sdf = self._board_sdf(self.pending_hole_yz)
            factory.pose.position.x = float(
                self.get_parameter('fixture_center_x').value)
            factory.pose.position.y = float(
                self.get_parameter('board_center_y').value)
            factory.pose.position.z = float(
                self.get_parameter('board_center_z').value)
            factory.pose.orientation.w = 1.0
            factory.relative_to = 'world'
            self.spawn_future = self.spawn_client.call_async(request)
            self.board_reset_stage = 'spawning'
            return

        if self.board_reset_stage == 'spawning':
            if self.spawn_future is None or not self.spawn_future.done():
                if now - self.last_reset_report >= 1.0:
                    self.get_logger().info(
                        'RESETTING: waiting for rebuilt board response')
                    self.last_reset_report = now
                return
            try:
                response = self.spawn_future.result()
                if response is None or not response.success:
                    raise RuntimeError('Gazebo rejected rebuilt board')
            except Exception as exc:
                self.get_logger().error(
                    f'RESETTING: board rebuild failed: {exc}; retrying')
                self.delete_future = None; self.spawn_future = None
                self.board_reset_stage = 'idle'
                return
            self.board_reset_stage = 'done'
            hole_yz = self.pending_hole_yz
            # ``HOLE_YZ_ref`` 仅用于终端诊断，不进入 Gym 观测。
            self.get_logger().info(
                'RESETTING: 固定外边界板面及真实深方孔已重建，'
                f'WORLD_HOLE_XYZ=('
                f'{float(self.get_parameter("fixture_center_x").value):.5f}, '
                f'{hole_yz[0]:.5f}, {hole_yz[1]:.5f}) m, '
                f'HOLE_YZ_ref=({hole_yz[0]:.4f}, {hole_yz[1]:.4f}) m')

        if self.board_reset_stage != 'done':
            return
        if now - self.reset_started < float(self.get_parameter('reset_settle_time').value): return
        # 清空父类状态，让它从未改变的 HOLDING 规则重新开始。
        self.center = self.center_rotation = self.q_home = None; self.ready_time = self.approach_start_time = self.search_start_time = None
        self.insert_yz = self.q_insert_hold = self.insert_hold_position = None; self.insert_hold_candidate_time = self.mit_hold_start_time = None
        self.contact_samples = 0; self.contact_active = False; self.state = 'WAITING_FOR_JOINTS'
        self.resetting = False; self.episode_active = True
        self.get_logger().info('RESET complete; starting original HOLDING state')

    def _board_sdf(self, hole_yz: np.ndarray) -> str:
        """生成外边界固定、孔心可变的四块板 SDF。所有长度单位为 m。"""
        center_y = float(self.get_parameter('board_center_y').value)
        center_z = float(self.get_parameter('board_center_z').value)
        size_y = float(self.get_parameter('board_size_y').value)
        size_z = float(self.get_parameter('board_size_z').value)
        depth_x = float(self.get_parameter('board_depth_x').value)
        hole_size = float(self.get_parameter('hole_size_yz').value)
        transparency = float(
            self.get_parameter('board_transparency').value)
        offset_y = float(hole_yz[0]) - center_y
        offset_z = float(hole_yz[1]) - center_z
        half_y = 0.5 * size_y; half_z = 0.5 * size_z
        hole_half = 0.5 * hole_size

        # 上下块覆盖全宽；左右块只补齐孔高7 mm的中间条带。
        top_min = offset_z + hole_half
        bottom_max = offset_z - hole_half
        positive_y_min = offset_y + hole_half
        negative_y_max = offset_y - hole_half
        segments = (
            ('top', 0.0, 0.5 * (half_z + top_min),
             size_y, half_z - top_min),
            ('bottom', 0.0, 0.5 * (-half_z + bottom_max),
             size_y, bottom_max + half_z),
            ('left', 0.5 * (half_y + positive_y_min), offset_z,
             half_y - positive_y_min, hole_size),
            ('right', 0.5 * (-half_y + negative_y_max), offset_z,
             negative_y_max + half_y, hole_size),
        )

        parts = []
        for name, y, z, segment_y, segment_z in segments:
            if segment_y <= 0.0 or segment_z <= 0.0:
                raise ValueError('随机孔超出固定板面边界')
            geometry = (
                f'<geometry><box><size>{depth_x:.9f} '
                f'{segment_y:.9f} {segment_z:.9f}</size></box></geometry>')
            pose = f'<pose>0 {y:.9f} {z:.9f} 0 0 0</pose>'
            parts.append(
                f'<collision name="{name}">{pose}{geometry}</collision>'
                f'<visual name="{name}_visual">{pose}{geometry}'
                f'<transparency>{transparency:.6f}</transparency>'
                '<material><diffuse>0.75 0.45 0.15 0.35</diffuse>'
                '</material></visual>')
        return (
            '<sdf version="1.8"><model name="hole_fixture"><static>true</static>'
            '<link name="fixture">' + ''.join(parts)
            + '</link></model></sdf>')

    def _apply_action(self, now: float) -> None:
        if not self.pending_action or self.search_center is None or self.reference_yz is None: return
        if now - self.last_action_time < float(self.get_parameter('sac_action_period').value): return
        step, action = self.pending_action; self.pending_action = None
        proposed = self.reference_yz + action * float(self.get_parameter('sac_action_step').value)
        if np.any(np.abs(proposed - self.search_center[1:3]) > float(self.get_parameter('sac_search_half_range').value)):
            self.out_of_bounds = self.episode_done = True; return
        self.reference_yz = proposed; self.active_step = step; self.last_action_time = now; self.sac_steps += 1
        if now - self.search_started >= float(self.get_parameter('sac_max_search_time').value) or self.sac_steps >= int(self.get_parameter('sac_max_steps').value):
            self.timeout = self.episode_done = True

    def _tool_position(self) -> np.ndarray:
        pin.forwardKinematics(self.model, self.data, self.q, self.dq); pin.updateFramePlacements(self.model, self.data)
        # Pinocchio 的 translation 是内部缓存视图；必须复制。否则保存为
        # search_center 后，它会随下一次正运动学更新而偷偷变化，导致零动作
        # 也被误判越过固定的 ±4 mm 搜索边界。
        return np.asarray(
            self.data.oMf[self.tool_frame_id].translation,
            dtype=float).copy()

    def _tool_tilt_deg(self) -> float:
        """返回轴线相对孔轴（世界 +X）的夹角，单位 deg。"""
        self._tool_position()
        tool_axis = np.asarray(
            self.data.oMf[self.tool_frame_id].rotation[:, 2], dtype=float)
        return math.degrees(math.acos(float(np.clip(tool_axis[0], -1.0, 1.0))))

    def _insertion_geometry(self) -> dict:
        """检查整根轴，而不只是检查轴尖是否落在孔心窗口内。"""
        position = self._tool_position()
        tool_axis = np.asarray(
            self.data.oMf[self.tool_frame_id].rotation[:, 2], dtype=float)
        force_direction = (
            1.0 if float(self.get_parameter('fx').value) >= 0.0 else -1.0)
        axial_component = force_direction * float(tool_axis[0])
        tilt_deg = math.degrees(math.acos(float(np.clip(
            axial_component, -1.0, 1.0))))
        surface_x = float(self.get_parameter('hole_surface_x').value)
        depth = force_direction * (float(position[0]) - surface_x)
        hole_yz = np.array([
            float(self.get_parameter('hole_center_y').value),
            float(self.get_parameter('hole_center_z').value),
        ])
        tip_error = position[1:3] - hole_yz
        if axial_component > 1.0e-6:
            # 沿轴线从轴尖反推到板面 X，得到入口处的真实 Y/Z。
            entry_yz = (
                position[1:3]
                - (depth / axial_component) * tool_axis[1:3]
            )
            entry_error = entry_yz - hole_yz
        else:
            entry_error = np.array([math.inf, math.inf])
        clearance = float(self.get_parameter('hole_center_clearance').value)
        target_depth = float(
            self.get_parameter('insert_target_depth').value)
        depth_tolerance = float(
            self.get_parameter('insertion_depth_tolerance').value)
        max_tilt = float(
            self.get_parameter('insertion_max_tilt_deg').value)
        valid = bool(
            depth >= target_depth - depth_tolerance
            and tilt_deg <= max_tilt
            and np.all(np.abs(tip_error) <= clearance)
            and np.all(np.abs(entry_error) <= clearance)
        )
        return {
            'valid': valid,
            'depth_m': depth,
            'tilt_deg': tilt_deg,
            'tip_error_yz_mm': tip_error * 1000.0,
            'entry_error_yz_mm': entry_error * 1000.0,
        }

    def _tool_inside_hole(self) -> bool:
        """用冻结的每轴 ±0.5 mm 条件校验真实几何，仅用于终止判定。

        孔心不会写入 SAC 观测，也不会发布给策略；该检查与原有底层
        接触/插入规则一样，只属于仿真安全和 Episode 成功标签。
        """
        return bool(self._insertion_geometry()['valid'])

    def _publish(self, now: float) -> None:
        period = float(self.get_parameter('sac_state_publish_period').value)
        if now - self.last_publish < period:
            return
        self.last_publish = now
        self.state_sequence += 1
        tip = (
            self._tool_position()
            if self.model is not None and self.q is not None
            else np.zeros(3)
        )
        velocity = (
            np.zeros(3)
            if self.dq is None or self.model is None
            else pin.computeFrameJacobian(
                self.model, self.data, self.q, self.tool_frame_id,
                pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)[:3] @ self.dq
        )
        payload = {
            'session_id': self.session_id,
            'episode_id': self.episode_id,
            'state_sequence': self.state_sequence,
            'stamp_sim_s': now,
            'controller_state': self.state,
            'tip_xyz_m': tip.tolist(),
            'tip_velocity_m_s': velocity.tolist(),
            'search_start_m': (
                None if self.search_center is None else self.search_center.tolist()),
            'reference_yz_m': (
                None if self.reference_yz is None else self.reference_yz.tolist()),
            'low_level_cycles': self.low_level_cycles,
            'sac_steps': self.sac_steps,
            'active_step': -1 if self.active_step is None else self.active_step,
            'hole_found': self.hole_found,
            'assembly_success': self.assembly_success,
            'fault': self.state == 'FAULT',
            'out_of_bounds': self.out_of_bounds,
            'timeout': self.timeout,
            'insertion_failure': self.insertion_failure,
            'done': self.episode_done,
        }
        message = String()
        message.data = json.dumps(payload)
        self.state_pub.publish(message)


def main(args=None) -> None:
    rclpy.init(args=args); node = SacRuntimeController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except RuntimeError:
        # Humble 在 launch 关闭订阅端的同一时刻可能抛出一次消息转换异常；
        # 仅当 ROS context 仍有效时才把它作为真正运行错误继续抛出。
        if rclpy.ok():
            raise
    finally:
        node.destroy_node()
        # ros2 launch 收到 Ctrl+C 时可能已经关闭全局 context；重复 shutdown
        # 会产生 RCLError。仅在 context 仍有效时主动关闭。
        if rclpy.ok():
            rclpy.shutdown()
