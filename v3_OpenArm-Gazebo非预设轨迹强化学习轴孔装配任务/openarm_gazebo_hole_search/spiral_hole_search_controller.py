#!/usr/bin/env python3

import math
from typing import Dict, Optional

import numpy as np
import pinocchio as pin
import rclpy
from geometry_msgs.msg import TransformStamped
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from ros_gz_interfaces.msg import Contacts
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray, String
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster
from visualization_msgs.msg import Marker, MarkerArray


class SpiralHoleSearchController(Node):
    """Cartesian force controller for a YZ spiral and constant world-X force."""

    def __init__(self) -> None:
        super().__init__('spiral_hole_search_controller')

        self.declare_parameter('control_rate', 100.0)
        self.declare_parameter('start_delay', 0.5)
        self.declare_parameter('r_max', 0.004)
        self.declare_parameter('r_rate', 0.0002)
        self.declare_parameter('omega', 2.0)
        self.declare_parameter('myz', 0.5)
        self.declare_parameter('kyz', 80.0)
        self.declare_parameter('dyz', 30.0)
        self.declare_parameter('fx', 1.0)
        self.declare_parameter('max_yz_force', 4.0)
        self.declare_parameter('orientation_k', 10.0)
        self.declare_parameter('orientation_d', 5.0)
        self.declare_parameter('max_orientation_torque', 3.0)
        self.declare_parameter('use_virtual_plate_contact', True)
        self.declare_parameter('virtual_contact_k', 1000.0)
        self.declare_parameter('virtual_contact_d', 20.0)
        self.declare_parameter('max_virtual_contact_force', 3.0)
        self.declare_parameter('virtual_contact_tolerance', 0.0002)
        self.declare_parameter(
            'max_joint_torque', [12.0, 12.0, 8.0, 8.0, 3.0, 3.0, 3.0])
        self.declare_parameter('gravity_scale', 1.0)
        self.declare_parameter(
            'joint_damping', [2.0, 2.0, 1.5, 1.5, 0.5, 0.5, 0.4])
        self.declare_parameter(
            'joint_stiffness', [4.0, 4.0, 3.0, 3.0, 1.0, 1.0, 0.5])
        self.declare_parameter('max_search_time', 30.0)
        self.declare_parameter('approach_timeout', 8.0)
        self.declare_parameter('contact_confirm_time', 0.30)
        self.declare_parameter('max_x_travel', 0.035)
        self.declare_parameter('hole_entry_depth', 0.003)
        self.declare_parameter('hole_center_y', -0.43800)
        self.declare_parameter('hole_center_z', 0.57242)
        self.declare_parameter('hole_center_clearance', 0.0005)
        self.declare_parameter('hole_inside_hysteresis', 0.00015)
        self.declare_parameter('hole_surface_x', 0.32370)
        self.declare_parameter('insert_target_depth', 0.017)
        self.declare_parameter('insert_fx', 2.0)
        self.declare_parameter('insert_hold_confirm_time', 0.05)
        self.declare_parameter('insert_velocity_filter_tau', 0.05)
        # 接近目标深度后使用 X 向位置-速度伺服，防止持续单向推力越过
        # 17 mm 目标并在横向尚未对中时提前锁住。单位分别为 N/m、N*s/m。
        self.declare_parameter('insert_depth_kx', 600.0)
        self.declare_parameter('insert_depth_dx', 30.0)
        # 仅用于深度控制参考的正向余量，验收深度仍为 17 mm。单位 m。
        self.declare_parameter('insert_depth_reference_margin', 0.0003)
        self.declare_parameter('insert_hold_max_joint_velocity', 3.0)
        self.declare_parameter('insert_hold_max_tool_velocity', 0.25)
        self.declare_parameter('insert_max_depth', 0.020)
        self.declare_parameter('insert_kyz', 200.0)
        self.declare_parameter('insert_dyz', 25.0)
        self.declare_parameter('insert_dx', 10.0)
        self.declare_parameter('max_insert_x_force', 2.5)
        self.declare_parameter('max_insert_yz_force', 4.0)
        self.declare_parameter(
            'mit_hold_kp', [30.0, 30.0, 30.0, 30.0, 20.0, 20.0, 15.0])
        self.declare_parameter(
            'mit_hold_kd', [1.5, 1.5, 1.5, 1.5, 0.5, 0.5, 0.3])
        self.declare_parameter('mit_hold_ramp_time', 0.2)
        self.declare_parameter('mit_hold_safety_delay', 2.0)
        self.declare_parameter('mit_velocity_filter_tau', 0.02)
        self.declare_parameter('mit_hold_max_joint_error', 0.30)
        self.declare_parameter('mit_hold_max_joint_velocity', 4.0)
        self.declare_parameter('mit_hold_fault_confirm_time', 0.25)
        self.declare_parameter('mit_hold_cartesian_kx', 180.0)
        self.declare_parameter('mit_hold_cartesian_dx', 25.0)
        self.declare_parameter('mit_hold_cartesian_kyz', 100.0)
        self.declare_parameter('mit_hold_cartesian_dyz', 15.0)
        self.declare_parameter('mit_hold_max_cartesian_force', 4.0)
        self.declare_parameter('tool_frame', 'openarm_right_search_tool_tip')
        self.declare_parameter('world_frame', 'world')
        self.declare_parameter('hole_frame', 'hole_center')

        self.joint_names = [
            f'openarm_right_joint{i}' for i in range(1, 8)
        ]
        self.model: Optional[pin.Model] = None
        self.data = None
        self.tool_frame_id: Optional[int] = None
        self.q: Optional[np.ndarray] = None
        self.dq: Optional[np.ndarray] = None
        self.q_home: Optional[np.ndarray] = None
        self.center: Optional[np.ndarray] = None
        self.center_rotation: Optional[np.ndarray] = None
        self.start_x = 0.0
        self.ready_time: Optional[float] = None
        self.approach_start_time: Optional[float] = None
        self.search_start_time: Optional[float] = None
        self.contact_x: Optional[float] = None
        self.insert_yz: Optional[np.ndarray] = None
        self.insert_hold_candidate_time: Optional[float] = None
        self.q_insert_hold: Optional[np.ndarray] = None
        self.insert_hold_position: Optional[np.ndarray] = None
        self.mit_hold_start_time: Optional[float] = None
        self.mit_fault_candidate_time: Optional[float] = None
        self.filtered_insert_dq = np.zeros(7)
        self.filtered_hold_dq = np.zeros(7)
        self.inside_hole_latched = False
        self.contact_samples = 0
        self.contact_active = False
        self.last_report_time = -math.inf
        self.state = 'WAITING_FOR_MODEL'
        self.path_points = []

        transient_qos = QoSProfile(depth=1)
        transient_qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
        transient_qos.reliability = ReliabilityPolicy.RELIABLE
        self.create_subscription(
            String, '/robot_description', self.robot_description_callback,
            transient_qos)
        self.create_subscription(
            JointState, '/joint_states', self.joint_state_callback, 10)
        self.create_subscription(
            Contacts, '/tool_contacts', self.contact_callback, 10)

        self.command_pub = self.create_publisher(
            Float64MultiArray, '/arm_effort_controller/commands', 10)
        self.marker_pub = self.create_publisher(
            MarkerArray, '/spiral_hole_search/markers', 10)
        self.static_tf_broadcaster = StaticTransformBroadcaster(self)
        self.publish_hole_transform()

        rate = float(self.get_parameter('control_rate').value)
        if rate <= 0.0:
            raise ValueError('control_rate must be positive')
        self.create_timer(1.0 / rate, self.control_callback)
        self.get_logger().info(
            'Waiting for robot_description and joint states...')

    def publish_hole_transform(self) -> None:
        transform = TransformStamped()
        transform.header.stamp = self.get_clock().now().to_msg()
        transform.header.frame_id = str(
            self.get_parameter('world_frame').value)
        transform.child_frame_id = str(
            self.get_parameter('hole_frame').value)
        transform.transform.translation.x = float(
            self.get_parameter('hole_surface_x').value)
        transform.transform.translation.y = float(
            self.get_parameter('hole_center_y').value)
        transform.transform.translation.z = float(
            self.get_parameter('hole_center_z').value)
        transform.transform.rotation.w = 1.0
        self.static_tf_broadcaster.sendTransform(transform)

    def robot_description_callback(self, msg: String) -> None:
        if self.model is not None:
            return
        try:
            model = pin.buildModelFromXML(msg.data)
            tool_frame = str(self.get_parameter('tool_frame').value)
            if not model.existFrame(tool_frame):
                raise ValueError(f'frame {tool_frame!r} is absent from the model')
            missing = [
                name for name in self.joint_names
                if not model.existJointName(name)
            ]
            if missing:
                raise ValueError(f'missing joints: {missing}')
            self.model = model
            self.data = model.createData()
            self.tool_frame_id = model.getFrameId(tool_frame)
            self.state = 'WAITING_FOR_JOINTS'
            self.get_logger().info(
                f'Loaded model with {model.nq} positions; tool={tool_frame}')
        except Exception as exc:  # pinocchio reports parser failures as RuntimeError
            self.get_logger().error(f'Cannot load robot model: {exc}')

    def joint_state_callback(self, msg: JointState) -> None:
        lookup: Dict[str, int] = {name: i for i, name in enumerate(msg.name)}
        if any(name not in lookup for name in self.joint_names):
            return
        try:
            self.q = np.array(
                [msg.position[lookup[name]] for name in self.joint_names],
                dtype=float)
            if len(msg.velocity) == len(msg.name):
                self.dq = np.array(
                    [msg.velocity[lookup[name]] for name in self.joint_names],
                    dtype=float)
            else:
                self.dq = np.zeros(7)
        except (IndexError, ValueError):
            return

    def contact_callback(self, msg: Contacts) -> None:
        self.contact_active = bool(msg.contacts)

    def sample_spiral(self, elapsed: float):
        r_max = float(self.get_parameter('r_max').value)
        r_rate = float(self.get_parameter('r_rate').value)
        omega = float(self.get_parameter('omega').value)
        growing = r_rate * elapsed < r_max
        radius = min(r_rate * elapsed, r_max)
        radius_dot = r_rate if growing else 0.0
        theta = omega * elapsed
        c = math.cos(theta)
        s = math.sin(theta)
        position = np.array([
            self.center[1] + radius * c,
            self.center[2] + radius * s,
        ])
        velocity = np.array([
            radius_dot * c - radius * omega * s,
            radius_dot * s + radius * omega * c,
        ])
        acceleration = np.array([
            -radius * omega * omega * c - 2.0 * radius_dot * omega * s,
            -radius * omega * omega * s + 2.0 * radius_dot * omega * c,
        ])
        return position, velocity, acceleration, radius

    def allow_hole_entry(self) -> bool:
        """规则入口默认可立即判孔；RL子类会等待首个SAC动作握手。"""
        return True

    @staticmethod
    def clamp_norm(value: np.ndarray, limit: float) -> np.ndarray:
        norm = np.linalg.norm(value)
        if limit <= 0.0 or norm <= limit:
            return value
        return value * limit / norm

    def set_state(self, state: str, text: str) -> None:
        if self.state != state:
            self.state = state
            self.get_logger().info(f'{state}: {text}')

    def control_callback(self) -> None:
        if self.model is None or self.q is None or self.dq is None:
            return
        if not np.all(np.isfinite(self.q)) or not np.all(np.isfinite(self.dq)):
            self.publish_zero()
            self.set_state('FAULT', 'non-finite joint state')
            return

        now = self.get_clock().now().nanoseconds * 1e-9
        pin.forwardKinematics(self.model, self.data, self.q, self.dq)
        pin.updateFramePlacements(self.model, self.data)
        placement = self.data.oMf[self.tool_frame_id]
        position = np.array(placement.translation)
        jacobian = pin.computeFrameJacobian(
            self.model, self.data, self.q, self.tool_frame_id,
            pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
        twist = jacobian @ self.dq
        control_velocity = twist[:3].copy()
        control_period = 1.0 / float(
            self.get_parameter('control_rate').value)
        insert_filter_tau = float(
            self.get_parameter('insert_velocity_filter_tau').value)
        insert_alpha = (
            1.0 if insert_filter_tau <= 0.0
            else control_period / (insert_filter_tau + control_period)
        )
        self.filtered_insert_dq += insert_alpha * (
            self.dq - self.filtered_insert_dq)
        filtered_insert_twist = jacobian @ self.filtered_insert_dq

        if self.state == 'FAULT':
            self.publish_gravity_hold()
            return

        if self.state == 'INSERT_HOLDING':
            if (
                self.q_insert_hold is None
                or self.insert_hold_position is None
                or self.mit_hold_start_time is None
            ):
                self.publish_zero()
                self.set_state('FAULT', 'MIT hold reference is unavailable')
                return

            kp = np.asarray(
                self.get_parameter('mit_hold_kp').value, dtype=float)
            kd = np.asarray(
                self.get_parameter('mit_hold_kd').value, dtype=float)
            limits = np.asarray(
                self.get_parameter('max_joint_torque').value, dtype=float)
            if kp.size != 7 or kd.size != 7 or limits.size != 7:
                self.publish_zero()
                self.set_state(
                    'FAULT',
                    'MIT Kp/Kd and torque-limit arrays must contain 7 values')
                return

            filter_tau = float(
                self.get_parameter('mit_velocity_filter_tau').value)
            control_period = 1.0 / float(
                self.get_parameter('control_rate').value)
            alpha = (
                1.0 if filter_tau <= 0.0
                else control_period / (filter_tau + control_period)
            )
            self.filtered_hold_dq += alpha * (
                self.dq - self.filtered_hold_dq)

            ramp_time = float(
                self.get_parameter('mit_hold_ramp_time').value)
            hold_elapsed = max(0.0, now - self.mit_hold_start_time)
            ramp = (
                1.0 if ramp_time <= 0.0
                else min(1.0, hold_elapsed / ramp_time)
            )
            joint_error = self.q_insert_hold - self.q
            max_joint_error = float(np.max(np.abs(joint_error)))
            max_joint_velocity = float(
                np.max(np.abs(self.filtered_hold_dq)))

            fx_target = float(self.get_parameter('fx').value)
            force_direction = 1.0 if fx_target >= 0.0 else -1.0
            total_x_travel = force_direction * (
                position[0] - self.start_x)
            insertion = force_direction * (
                position[0]
                - float(self.get_parameter('hole_surface_x').value)
            )
            hole_center_y = float(
                self.get_parameter('hole_center_y').value)
            hole_center_z = float(
                self.get_parameter('hole_center_z').value)
            hole_clearance = float(
                self.get_parameter('hole_center_clearance').value)
            # 装配验收始终使用冻结的每轴 ±0.5 mm 条件。这里不扩大
            # 这里只诊断轴尖 Y/Z 是否对中，不代表整根轴已正确插入；
            # RL 最终验收还会检查深度、板面入口误差和轴线倾角。
            diagnostic_clearance = hole_clearance
            inside_hole_geometry = (
                abs(position[1] - hole_center_y) <= diagnostic_clearance
                and abs(position[2] - hole_center_z) <= diagnostic_clearance
            )
            self.inside_hole_latched = inside_hole_geometry
            if (
                abs(position[0] - self.start_x)
                > float(self.get_parameter('max_x_travel').value)
                or insertion
                > float(self.get_parameter('insert_max_depth').value)
            ):
                self.publish_zero()
                self.set_state(
                    'FAULT',
                    f'MIT hold Cartesian safety limit exceeded: '
                    f'dx={total_x_travel:.4f} m, depth={insertion:.4f} m')
                return
            joint_limit_exceeded = (
                hold_elapsed
                >= float(
                    self.get_parameter('mit_hold_safety_delay').value)
                and (
                    max_joint_error
                    > float(
                        self.get_parameter(
                            'mit_hold_max_joint_error').value)
                    or max_joint_velocity
                    > float(
                        self.get_parameter(
                            'mit_hold_max_joint_velocity').value)
                )
            )
            if joint_limit_exceeded:
                if self.mit_fault_candidate_time is None:
                    self.mit_fault_candidate_time = now
                elif (
                    now - self.mit_fault_candidate_time
                    >= float(
                        self.get_parameter(
                            'mit_hold_fault_confirm_time').value)
                ):
                    self.publish_zero()
                    self.set_state(
                        'FAULT',
                        f'MIT hold joint safety limit sustained: '
                        f'error={max_joint_error:.4f} rad, '
                        f'velocity={max_joint_velocity:.4f} rad/s')
                    return
            else:
                self.mit_fault_candidate_time = None

            gravity = pin.computeGeneralizedGravity(
                self.model, self.data, self.q)
            filtered_twist = jacobian @ self.filtered_hold_dq
            position_error = self.insert_hold_position - position
            hold_force = np.array([
                float(self.get_parameter('mit_hold_cartesian_kx').value)
                * position_error[0]
                - float(self.get_parameter('mit_hold_cartesian_dx').value)
                * filtered_twist[0],
                float(self.get_parameter('mit_hold_cartesian_kyz').value)
                * position_error[1]
                - float(self.get_parameter('mit_hold_cartesian_dyz').value)
                * filtered_twist[1],
                float(self.get_parameter('mit_hold_cartesian_kyz').value)
                * position_error[2]
                - float(self.get_parameter('mit_hold_cartesian_dyz').value)
                * filtered_twist[2],
            ])
            hold_force = self.clamp_norm(
                hold_force,
                float(
                    self.get_parameter(
                        'mit_hold_max_cartesian_force').value),
            )
            hold_wrench = np.zeros(6)
            hold_wrench[:3] = hold_force
            torque = (
                ramp * kp * joint_error
                - kd * self.filtered_hold_dq
                + jacobian.T @ hold_wrench
                + float(self.get_parameter('gravity_scale').value) * gravity
            )
            torque = np.clip(torque, -limits, limits)
            command = Float64MultiArray()
            command.data = torque.tolist()
            self.command_pub.publish(command)

            if now - self.last_report_time >= 1.0:
                self.get_logger().info(
                    f'INSERT_HOLDING: MIT joint lock, '
                    f'dx={total_x_travel:.4f} m, '
                    f'depth={insertion:.4f} m, '
                    f'YZ=({position[1]:.4f}, {position[2]:.4f}) m, '
                    f'ramp={ramp:.2f}, '
                    f'q_error_max={max_joint_error:.4f} rad, '
                    f'dq_max={max_joint_velocity:.4f} rad/s, '
                    f'tau_max={float(np.max(np.abs(torque))):.2f} Nm, '
                    f'hold_force_max='
                    f'{float(np.max(np.abs(hold_force))):.2f} N, '
                    f'tip_centered_YZ={inside_hole_geometry}, '
                    f'contact={self.contact_active}')
                self.last_report_time = now
            self.publish_markers(
                position, self.insert_yz, np.zeros(6), 0.0)
            return

        if self.center is None:
            self.center = position.copy()
            self.center_rotation = placement.rotation.copy()
            self.start_x = float(position[0])
            self.q_home = self.q.copy()
            self.ready_time = now
            self.set_state('HOLDING', f'center={self.center}')

        start_delay = float(self.get_parameter('start_delay').value)
        if (
            self.approach_start_time is None
            and now - self.ready_time >= start_delay
        ):
            self.approach_start_time = now
            self.set_state('APPROACHING', 'ramping world-X contact force')

        elapsed = 0.0
        if self.search_start_time is not None:
            elapsed = max(0.0, now - self.search_start_time)

        if (
            self.state == 'INSERTING'
            and self.insert_yz is not None
        ):
            ref_position = self.insert_yz.copy()
            ref_velocity = np.zeros(2)
            ref_acceleration = np.zeros(2)
            radius = 0.0
        else:
            ref_position, ref_velocity, ref_acceleration, radius = (
                self.sample_spiral(elapsed))
        myz = float(self.get_parameter('myz').value)
        if self.state == 'INSERTING':
            kyz = float(self.get_parameter('insert_kyz').value)
            dyz = float(self.get_parameter('insert_dyz').value)
            max_yz_force = float(
                self.get_parameter('max_insert_yz_force').value)
        else:
            kyz = float(self.get_parameter('kyz').value)
            dyz = float(self.get_parameter('dyz').value)
            max_yz_force = float(
                self.get_parameter('max_yz_force').value)
        yz_control_velocity = (
            filtered_insert_twist[1:3]
            if self.state == 'INSERTING'
            else control_velocity[1:3]
        )
        f_yz = (
            myz * ref_acceleration
            + dyz * (ref_velocity - yz_control_velocity)
            + kyz * (ref_position - position[1:3])
        )
        f_yz = self.clamp_norm(f_yz, max_yz_force)
        orientation_error = pin.log3(
            self.center_rotation @ placement.rotation.T)
        orientation_torque = (
            float(self.get_parameter('orientation_k').value)
            * orientation_error
            - float(self.get_parameter('orientation_d').value)
            * twist[3:6]
        )
        orientation_torque = self.clamp_norm(
            orientation_torque,
            float(self.get_parameter('max_orientation_torque').value),
        )

        fx_target = float(self.get_parameter('fx').value)
        force_direction = 1.0 if fx_target >= 0.0 else -1.0
        hole_surface_x = float(
            self.get_parameter('hole_surface_x').value)
        insertion_now = force_direction * (
            position[0] - hole_surface_x)
        hole_center_y = float(
            self.get_parameter('hole_center_y').value)
        hole_center_z = float(
            self.get_parameter('hole_center_z').value)
        hole_clearance = float(
            self.get_parameter('hole_center_clearance').value)
        tool_is_inside_hole_now = (
            abs(position[1] - hole_center_y) <= hole_clearance
            and abs(position[2] - hole_center_z) <= hole_clearance
        )
        if self.approach_start_time is None:
            fx = 0.0
        else:
            approach_elapsed = max(0.0, now - self.approach_start_time)
            fx = fx_target * min(1.0, approach_elapsed / 1.0)
        if self.state == 'INSERTING':
            axial_velocity = (
                force_direction * float(filtered_insert_twist[0]))
            insert_force_target = abs(float(
                self.get_parameter('insert_fx').value))
            feed_force = float(
                insert_force_target
                - float(self.get_parameter('insert_dx').value)
                * axial_velocity)
            target_depth = float(
                self.get_parameter('insert_target_depth').value)
            depth_reference = (
                target_depth
                + float(self.get_parameter(
                    'insert_depth_reference_margin').value))
            depth_error = depth_reference - insertion_now
            depth_servo_force = (
                float(self.get_parameter('insert_depth_kx').value)
                * depth_error
                - float(self.get_parameter('insert_depth_dx').value)
                * axial_velocity
            )
            # 远离目标时仍使用原来的恒力推进；接近目标后自动选取更小
            # 的深度伺服力。超过目标时伺服力允许为负，从而轻微退出，
            # 而不是继续推向 20 mm 安全上限。
            axial_force = float(np.clip(
                min(feed_force, depth_servo_force),
                -float(self.get_parameter('max_insert_x_force').value),
                float(self.get_parameter('max_insert_x_force').value),
            ))
            fx = force_direction * axial_force
        elif bool(
            self.get_parameter('use_virtual_plate_contact').value
        ):
            tolerance = float(
                self.get_parameter('virtual_contact_tolerance').value)
            self.contact_active = (
                not tool_is_inside_hole_now
                and insertion_now >= -tolerance
            )
            if self.contact_active:
                axial_velocity = (
                    force_direction * float(twist[0]))
                reaction = (
                    float(self.get_parameter('virtual_contact_k').value)
                    * max(0.0, insertion_now)
                    + float(self.get_parameter('virtual_contact_d').value)
                    * max(0.0, axial_velocity)
                )
                reaction = float(np.clip(
                    reaction,
                    0.0,
                    float(
                        self.get_parameter(
                            'max_virtual_contact_force').value),
                ))
                fx -= force_direction * reaction
        wrench = np.array([
            fx,
            f_yz[0],
            f_yz[1],
            orientation_torque[0],
            orientation_torque[1],
            orientation_torque[2],
        ])
        gravity = pin.computeGeneralizedGravity(
            self.model, self.data, self.q)
        damping = np.asarray(
            self.get_parameter('joint_damping').value, dtype=float)
        stiffness = np.asarray(
            self.get_parameter('joint_stiffness').value, dtype=float)
        # During insertion the axial force must be free to advance through the
        # full lock depth.  A spring to the pre-insertion joint pose otherwise
        # builds an opposing wrench and creates a false equilibrium after only
        # a few millimetres. Cartesian YZ/orientation control and joint damping
        # remain active, and the seated joint pose is captured at target depth.
        active_stiffness = (
            np.zeros_like(stiffness)
            if self.state == 'INSERTING'
            else stiffness
        )
        torque = (
            jacobian.T @ wrench
            + float(self.get_parameter('gravity_scale').value) * gravity
            - damping * self.dq
            - active_stiffness * (self.q - self.q_home)
        )
        limits = np.asarray(
            self.get_parameter('max_joint_torque').value, dtype=float)
        if damping.size != 7 or stiffness.size != 7 or limits.size != 7:
            self.publish_zero()
            self.set_state(
                'FAULT',
                'torque/stiffness/damping arrays must contain 7 values')
            return
        torque = np.clip(torque, -limits, limits)

        total_x_travel = force_direction * (position[0] - self.start_x)
        insertion = force_direction * (position[0] - hole_surface_x)
        if now - self.last_report_time >= 1.0:
            center_error_mm = (
                position[1:3]
                - np.array([hole_center_y, hole_center_z])) * 1000.0
            self.get_logger().info(
                f'{self.state}: dx={total_x_travel:.4f} m, '
                f'depth={insertion:.4f} m, '
                f'YZ=({position[1]:.4f}, {position[2]:.4f}) m, '
                f'YZ_ref=({ref_position[0]:.4f}, '
                f'{ref_position[1]:.4f}) m, '
                f'center_error_YZ_mm=({center_error_mm[0]:.3f}, '
                f'{center_error_mm[1]:.3f}), '
                f'vx={float(twist[0]):.4f} m/s, '
                f'Fx={fx:.2f} N, '
                f'contact={self.contact_active}')
            self.last_report_time = now
        if total_x_travel > float(self.get_parameter('max_x_travel').value):
            self.publish_zero()
            self.set_state(
                'FAULT', f'max X travel exceeded: {total_x_travel:.4f} m')
            self.publish_markers(position, ref_position, wrench, radius)
            return

        if self.state == 'APPROACHING':
            approach_elapsed = now - self.approach_start_time
            force_is_established = approach_elapsed >= 1.0
            if force_is_established and self.contact_active:
                self.contact_samples += 1
            else:
                self.contact_samples = 0
            required_samples = max(
                1,
                int(
                    float(self.get_parameter('contact_confirm_time').value)
                    * float(self.get_parameter('control_rate').value)
                ),
            )
            if self.contact_samples >= required_samples:
                self.contact_x = float(position[0])
                # Keep the YZ spiral centered on the startup pose. Contact
                # impulses can momentarily deflect the simulated arm by much
                # more than the sub-millimetre physical hole clearance.
                self.search_start_time = now
                self.set_state(
                    'SEARCHING',
                    f'contact at X={self.contact_x:.4f}; YZ spiral started',
                )
            elif approach_elapsed > float(
                self.get_parameter('approach_timeout').value
            ):
                self.publish_zero()
                self.set_state('FAULT', 'contact approach timeout')
                self.publish_markers(position, ref_position, wrench, radius)
                return

        entry_depth = float(self.get_parameter('hole_entry_depth').value)
        hole_center_y = float(
            self.get_parameter('hole_center_y').value)
        hole_center_z = float(
            self.get_parameter('hole_center_z').value)
        hole_clearance = float(
            self.get_parameter('hole_center_clearance').value)
        tool_is_inside_hole = (
            abs(position[1] - hole_center_y) <= hole_clearance
            and abs(position[2] - hole_center_z) <= hole_clearance
        )
        target_depth = float(
            self.get_parameter('insert_target_depth').value)
        max_insert_depth = float(
            self.get_parameter('insert_max_depth').value)
        if not 0.0 < entry_depth <= target_depth < max_insert_depth:
            self.publish_zero()
            self.set_state(
                'FAULT',
                'require 0 < hole_entry_depth <= insert_target_depth '
                '< insert_max_depth',
            )
            self.publish_markers(position, ref_position, wrench, radius)
            return
        virtual_hole_entry = (
            bool(
                self.get_parameter(
                    'use_virtual_plate_contact').value)
            and insertion
            >= -float(
                self.get_parameter(
                    'virtual_contact_tolerance').value)
        )
        if (
            self.state == 'SEARCHING'
            and self.allow_hole_entry()
            and tool_is_inside_hole
            and (
                insertion >= entry_depth
                or virtual_hole_entry
            )
        ):
            # In this deterministic Gazebo world the fixture center is known.
            # Holding the detection sample can leave the pin close to a hole
            # edge, so center it before and during insertion.
            self.insert_yz = np.array([hole_center_y, hole_center_z])
            self.set_state(
                'INSERTING',
                f'hole alignment/entry detected at {insertion:.4f} m; '
                f'centering YZ and inserting to {target_depth:.4f} m',
            )
        if (
            self.state == 'INSERTING'
            and insertion > max_insert_depth
        ):
            self.publish_zero()
            self.set_state(
                'FAULT',
                f'max insertion depth exceeded: {insertion:.4f} m',
            )
            self.publish_markers(position, ref_position, wrench, radius)
            return
        if (
            self.state == 'INSERTING'
            and insertion >= target_depth
            and tool_is_inside_hole
            and float(np.max(np.abs(self.filtered_insert_dq)))
            <= float(
                self.get_parameter(
                    'insert_hold_max_joint_velocity').value)
            and float(np.linalg.norm(filtered_insert_twist[:3]))
            <= float(
                self.get_parameter(
                    'insert_hold_max_tool_velocity').value)
        ):
            if self.insert_hold_candidate_time is None:
                self.insert_hold_candidate_time = now
        elif self.state == 'INSERTING':
            self.insert_hold_candidate_time = None
        if (
            self.state == 'INSERTING'
            and self.insert_hold_candidate_time is not None
            and now - self.insert_hold_candidate_time
            >= float(
                self.get_parameter('insert_hold_confirm_time').value)
        ):
            # Move the null-space posture reference to the inserted
            # configuration so it does not pull the tool back out of the hole.
            self.q_home = self.q.copy()
            self.q_insert_hold = self.q.copy()
            # 保持参考必须是“17 mm 深度处的孔中心”，不能把瞬时偏心
            # 位置保存为目标，否则 INSERT_HOLDING 会稳定地锁在错误位置。
            self.insert_hold_position = position.copy()
            self.insert_hold_position[0] = (
                hole_surface_x + force_direction * target_depth)
            self.insert_hold_position[1:3] = self.insert_yz.copy()
            self.inside_hole_latched = tool_is_inside_hole
            self.mit_hold_start_time = now
            self.mit_fault_candidate_time = None
            self.filtered_hold_dq.fill(0.0)
            self.set_state(
                'INSERT_HOLDING',
                f'target depth confirmed at {insertion:.4f} m; '
                f'joint velocity='
                f'{float(np.max(np.abs(self.filtered_insert_dq))):.3f} '
                f'rad/s; tool velocity='
                f'{float(np.linalg.norm(filtered_insert_twist[:3])):.3f} '
                'm/s; '
                'captured current joints and starting MIT-equivalent hold',
            )
        if (
            self.state == 'SEARCHING'
            and self.search_start_time is not None
            and elapsed > float(self.get_parameter('max_search_time').value)
        ):
            self.publish_zero()
            self.set_state('FAULT', 'search timeout')
            self.publish_markers(position, ref_position, wrench, radius)
            return
        if self.state == 'FAULT':
            self.publish_gravity_hold()
            return

        command = Float64MultiArray()
        command.data = torque.tolist()
        self.command_pub.publish(command)
        self.publish_markers(position, ref_position, wrench, radius)

    def publish_zero(self) -> None:
        command = Float64MultiArray()
        command.data = [0.0] * 7
        self.command_pub.publish(command)

    def publish_gravity_hold(self) -> None:
        gravity = pin.computeGeneralizedGravity(
            self.model, self.data, self.q)
        damping = np.asarray(
            self.get_parameter('joint_damping').value, dtype=float)
        stiffness = np.asarray(
            self.get_parameter('joint_stiffness').value, dtype=float)
        limits = np.asarray(
            self.get_parameter('max_joint_torque').value, dtype=float)
        torque = (
            float(self.get_parameter('gravity_scale').value) * gravity
            - damping * self.dq
            - stiffness * (self.q - self.q_home)
        )
        torque = np.clip(torque, -limits, limits)
        command = Float64MultiArray()
        command.data = torque.tolist()
        self.command_pub.publish(command)

    def publish_markers(
        self, position: np.ndarray, reference_yz: np.ndarray,
        wrench: np.ndarray, radius: float
    ) -> None:
        frame = str(self.get_parameter('world_frame').value)
        stamp = self.get_clock().now().to_msg()
        markers = MarkerArray()

        actual = Marker()
        actual.header.frame_id = frame
        actual.header.stamp = stamp
        actual.ns = 'hole_search'
        actual.id = 0
        actual.type = Marker.SPHERE
        actual.action = Marker.ADD
        actual.pose.position.x = float(position[0])
        actual.pose.position.y = float(position[1])
        actual.pose.position.z = float(position[2])
        actual.pose.orientation.w = 1.0
        actual.scale.x = actual.scale.y = actual.scale.z = 0.014
        actual.color.r = 0.1
        actual.color.g = 0.9
        actual.color.b = 0.2
        actual.color.a = 1.0
        markers.markers.append(actual)

        target = Marker()
        target.header = actual.header
        target.ns = 'hole_search'
        target.id = 1
        target.type = Marker.SPHERE
        target.action = Marker.ADD
        target.pose.position.x = float(self.start_x)
        target.pose.position.y = float(reference_yz[0])
        target.pose.position.z = float(reference_yz[1])
        target.pose.orientation.w = 1.0
        target.scale.x = target.scale.y = target.scale.z = 0.010
        target.color.r = 0.9
        target.color.g = 0.2
        target.color.b = 0.1
        target.color.a = 1.0
        markers.markers.append(target)

        force = Marker()
        force.header = actual.header
        force.ns = 'hole_search'
        force.id = 2
        force.type = Marker.ARROW
        force.action = Marker.ADD
        force.pose.position.x = float(position[0])
        force.pose.position.y = float(position[1])
        force.pose.position.z = float(position[2])
        force.pose.orientation.w = 1.0
        force.scale.x = max(0.01, abs(float(wrench[0])) * 0.025)
        force.scale.y = 0.008
        force.scale.z = 0.012
        if wrench[0] < 0.0:
            force.pose.orientation.z = 1.0
            force.pose.orientation.w = 0.0
        force.color.r = 0.2
        force.color.g = 0.4
        force.color.b = 1.0
        force.color.a = 1.0
        markers.markers.append(force)

        status = Marker()
        status.header = actual.header
        status.ns = 'hole_search'
        status.id = 3
        status.type = Marker.TEXT_VIEW_FACING
        status.action = Marker.ADD
        status.pose.position.x = float(position[0])
        status.pose.position.y = float(position[1])
        status.pose.position.z = float(position[2] + 0.08)
        status.pose.orientation.w = 1.0
        status.scale.z = 0.025
        status.color.r = status.color.g = status.color.b = status.color.a = 1.0
        status.text = f'{self.state}  r={radius * 1000.0:.1f} mm'
        markers.markers.append(status)

        self.marker_pub.publish(markers)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = SpiralHoleSearchController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            if rclpy.ok():
                node.publish_zero()
            node.destroy_node()
            if rclpy.ok():
                rclpy.shutdown()
        except (KeyboardInterrupt, Exception):
            pass


if __name__ == '__main__':
    main()
