"""Gymnasium 环境：通过 JSON 话题驱动独立 SAC 搜索控制器。"""
from __future__ import annotations

import json
import math
import time
import uuid
from pathlib import Path

import gymnasium as gym
import numpy as np
from .ros_python_compat import ensure_ros_python_path

ensure_ros_python_path()
import rclpy
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from std_msgs.msg import String
import yaml

from .observation import build_observation
from .reward import search_step_reward
from .visit_map import VisitMap


class OpenArmHoleSearchEnv(gym.Env):
    """SAC 仅得到 92 维非真值观测；孔偏移只用于 reset 与离线 info。"""

    metadata = {'render_modes': []}

    def __init__(
        self,
        dataset,
        mode: str,
        search_config_path: Path,
        ros_response_timeout_s: float = 30.0,
        ros_reset_timeout_s: float = 180.0,
        ros_terminal_timeout_s: float = 0.0,
        episode_order_seed: int | None = None,
    ):
        super().__init__()
        self.dataset = dataset
        self.mode = mode

        # response timeout 是“连续多久收不到当前 Episode 心跳才报错”，不是
        # INSERTING/INSERT_HOLDING 整段动作必须在多少秒内结束。
        self.response_timeout_s = float(ros_response_timeout_s)
        self.reset_timeout_s = float(ros_reset_timeout_s)
        # <= 0 表示终止阶段不设墙钟总时限，只监视状态心跳；控制器内部仍有
        # insertion_timeout（仿真时间），因此不会因一次正常但较慢的插入而中断。
        terminal = float(ros_terminal_timeout_s)
        self.terminal_timeout_s = None if terminal <= 0.0 else terminal

        cfg = yaml.safe_load(Path(search_config_path).read_text())[
            'rl_hole_search_controller'
        ]['ros__parameters']
        self.cfg = cfg
        self.action_space = gym.spaces.Box(-1.0, 1.0, (2,), np.float32)
        self.observation_space = gym.spaces.Box(-1.0, 1.0, (92,), np.float32)

        if not rclpy.ok():
            rclpy.init()

        # 每次启动训练/验证程序都生成独立会话号。仅靠从 1 开始的 Episode
        # 编号会在“不重启 Gazebo、只重启训练脚本”时与旧回合撞号。
        self.session_id = uuid.uuid4().hex
        self.node = Node(f'sac_env_{self.session_id[:8]}')
        self.state = None
        self.episode_id = 0
        self._state_rx_count = 0

        # 状态只需要最新一帧。depth=1 可避免训练网络更新期间没有 spin 时，
        # 20 Hz 状态在订阅队列里积压，随后环境又逐帧处理旧状态。
        latest_reliable_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.node.create_subscription(
            String, cfg['state_topic'], self._state_cb, latest_reliable_qos
        )
        self.action_pub = self.node.create_publisher(
            String, cfg['action_topic'], latest_reliable_qos
        )
        self.command_pub = self.node.create_publisher(
            String, cfg['command_topic'], latest_reliable_qos
        )

        self.visit = None
        self.previous_action = np.zeros(2, np.float32)
        self.hole_found_rewarded = False
        self.path_m = 0.0
        self.last_tip_yz = None
        self.trajectory = []
        self.current_dataset_index = -1
        self.current_hole_offset = np.zeros(2, dtype=np.float64)

        # 训练时将固定的 5000 个孔位随机打乱后逐个使用。
        self.episode_order = np.arange(len(dataset), dtype=np.int64)
        if episode_order_seed is not None:
            self.episode_order = np.random.default_rng(
                episode_order_seed
            ).permutation(len(dataset))

    def _state_cb(self, msg: String) -> None:
        try:
            self.state = json.loads(msg.data)
            self._state_rx_count += 1
        except json.JSONDecodeError:
            pass

    def _is_current_state(self, state) -> bool:
        """状态必须同时属于本次训练会话和当前 Episode。"""
        return (
            state.get('session_id') == self.session_id
            and state.get('episode_id') == self.episode_id
        )

    def _wait(
        self,
        predicate,
        *,
        description: str,
        overall_timeout_s: float | None,
    ):
        """等待目标状态，并把超时定义为“控制器失联”而非“动作太慢”。

        只要当前 Episode 的状态心跳仍持续到达，就刷新 inactivity deadline。
        这样 INSERTING 运行 30 秒以上不会误判超时；真正断开、Gazebo 暂停
        或控制器崩溃时，仍会在 response_timeout_s 后明确报错。
        """
        start = time.monotonic()
        overall_deadline = (
            math.inf
            if overall_timeout_s is None or overall_timeout_s <= 0.0
            else start + overall_timeout_s
        )
        inactivity_deadline = start + self.response_timeout_s
        seen_rx_count = self._state_rx_count

        while True:
            now = time.monotonic()
            if now >= overall_deadline:
                latest = None if self.state is None else {
                    'session_id': self.state.get('session_id'),
                    'episode_id': self.state.get('episode_id'),
                    'controller_state': self.state.get('controller_state'),
                    'sac_steps': self.state.get('sac_steps'),
                    'done': self.state.get('done'),
                }
                raise TimeoutError(
                    f'{description}总等待超时（{overall_timeout_s:.1f} s）；'
                    f'最新状态={latest}'
                )

            rclpy.spin_once(self.node, timeout_sec=0.05)

            if self._state_rx_count != seen_rx_count:
                seen_rx_count = self._state_rx_count
                if self.state is not None and self._is_current_state(self.state):
                    inactivity_deadline = time.monotonic() + self.response_timeout_s

            if self.state is not None and predicate(self.state):
                return self.state

            if time.monotonic() >= inactivity_deadline:
                latest = None if self.state is None else {
                    'session_id': self.state.get('session_id'),
                    'episode_id': self.state.get('episode_id'),
                    'controller_state': self.state.get('controller_state'),
                    'sac_steps': self.state.get('sac_steps'),
                    'active_step': self.state.get('active_step'),
                    'done': self.state.get('done'),
                }
                raise TimeoutError(
                    f'{description}期间连续 {self.response_timeout_s:.1f} s '
                    f'未收到当前 Episode 的控制器状态；最新状态={latest}'
                )

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        options = dict(options or {})
        reuse_current_episode = bool(
            options.pop('_reuse_current_episode', False)
        )

        # SB3 在终止步返回给 callback 之前，DummyVecEnv 已经自动 reset 到
        # 下一训练 Episode。检查点验证会暂时把同一个 Gazebo 控制器切换到
        # validation 会话；验证结束后必须重发这个已创建训练 Episode 的 reset，
        # 但不能再次递增 episode_id 或消耗下一个训练孔。
        if reuse_current_episode:
            if self.episode_id <= 0 or self.current_dataset_index < 0:
                raise RuntimeError('当前没有可重新接管的训练 Episode')
            index = self.current_dataset_index
            offset = self.current_hole_offset.copy()
        else:
            # 若调用方未指定孔索引（SB3 的训练回合正是这种情况），从固定随机
            # 顺序取下一个孔。若验证/测试显式指定，则完全服从指定的固定索引。
            index = int(
                options.get(
                    'dataset_index',
                    self.episode_order[self.episode_id % len(self.dataset)],
                )
            )
            offset = self.dataset.offset(index)
            self.episode_id += 1
        self.state = None
        self.current_dataset_index = index
        self.current_hole_offset = offset.copy()

        reset_attempt = 0

        def build_reset_message(attempt: int) -> String:
            return String(
                data=json.dumps(
                    {
                        'version': 1,
                        'command': 'reset',
                        'session_id': self.session_id,
                        'episode_id': self.episode_id,
                        'reset_attempt': int(attempt),
                        'mode': self.mode,
                        'dataset_index': index,
                        'hole_offset_y_m': float(offset[0]),
                        'hole_offset_z_m': float(offset[1]),
                    }
                )
            )

        reset_message = build_reset_message(reset_attempt)
        retry_pending = False

        # 普通 topic 不是锁存消息：控制器尚未匹配时第一条 reset 可能丢失，
        # 因此在进入本 Episode 的 SEARCHING 前每 0.5 s 幂等重发。
        # 若控制器在 SEARCHING 之前就进入终态（例如 APPROACHING 超时进入
        # FAULT），则提升 reset_attempt，明确要求控制器重启同一 Episode。
        start = time.monotonic()
        overall_deadline = start + self.reset_timeout_s
        inactivity_deadline = start + self.response_timeout_s
        next_publish = 0.0
        seen_rx_count = self._state_rx_count
        state = None

        while time.monotonic() < overall_deadline:
            now = time.monotonic()
            if now >= next_publish:
                self.command_pub.publish(reset_message)
                next_publish = now + 0.5

            rclpy.spin_once(self.node, timeout_sec=0.05)

            if self._state_rx_count != seen_rx_count:
                seen_rx_count = self._state_rx_count
                if self.state is not None and self._is_current_state(self.state):
                    inactivity_deadline = time.monotonic() + self.response_timeout_s

                    controller_state = self.state.get('controller_state')
                    done = bool(self.state.get('done'))
                    if controller_state == 'RESETTING' and not done:
                        retry_pending = False
                    elif done and controller_state != 'SEARCHING' and not retry_pending:
                        reset_attempt += 1
                        reset_message = build_reset_message(reset_attempt)
                        retry_pending = True
                        self.command_pub.publish(reset_message)
                        next_publish = time.monotonic() + 0.5
                        print(
                            '[复位重试] 当前 Episode 在进入 SEARCHING 前已结束；'
                            f'重新复位 Episode {self.episode_id}，'
                            f'attempt={reset_attempt}，'
                            f'controller_state={controller_state}',
                            flush=True,
                        )

            if (
                self.state is not None
                and self._is_current_state(self.state)
                and self.state.get('controller_state') == 'SEARCHING'
            ):
                state = self.state
                break

            if time.monotonic() >= inactivity_deadline:
                raise TimeoutError(
                    '等待 Gazebo 控制器进入 SEARCHING 时，连续 '
                    f'{self.response_timeout_s:.1f} s 未收到当前 Episode 状态'
                )

        if state is None:
            latest = None if self.state is None else {
                'session_id': self.state.get('session_id'),
                'episode_id': self.state.get('episode_id'),
                'controller_state': self.state.get('controller_state'),
                'done': self.state.get('done'),
                'fault': self.state.get('fault'),
            }
            raise TimeoutError(
                '等待 Gazebo 控制器进入 SEARCHING 总超时：'
                f'{self.reset_timeout_s:.1f} s；'
                f'reset_attempt={reset_attempt}；最新状态={latest}'
            )

        self.visit = VisitMap(np.asarray(state['search_start_m'][1:3]))
        self.visit.enter(state['tip_xyz_m'][1:3])
        self.previous_action.fill(0.0)
        self.hole_found_rewarded = False
        self.path_m = 0.0
        self.last_tip_yz = np.asarray(state['tip_xyz_m'][1:3])
        self.trajectory = []
        return self._observation(state), self._info(state, offset)

    def set_completed_episode_count(self, completed_episodes: int) -> None:
        """设置断点续训时已经完整完成的训练回合数。

        下一次普通 reset 会使用固定打乱顺序中的下一个孔；不会重复已经
        完整保存到检查点中的训练回合。只能在新进程尚未 reset 前调用。
        """
        completed = int(completed_episodes)
        if completed < 0:
            raise ValueError('completed_episodes 不能为负数')
        if self.episode_id != 0 or self.current_dataset_index >= 0:
            raise RuntimeError('只能在环境第一次 reset 之前设置续训回合数')
        self.episode_id = completed

    def reacquire_current_episode(self):
        """验证结束后重新接管 SB3 已自动创建的下一训练回合。

        该操作重发当前 session/episode/孔位的 reset，并重新同步初始观测；
        不增加 Episode 编号，也不跳过训练数据集中的下一个孔。
        """
        return self.reset(options={'_reuse_current_episode': True})

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        before = -1 if self.state is None else int(self.state['sac_steps'])
        requested_step = before + 1

        self.action_pub.publish(
            String(
                data=json.dumps(
                    {
                        'session_id': self.session_id,
                        'episode_id': self.episode_id,
                        'step_id': requested_step,
                        'action': action.tolist(),
                    }
                )
            )
        )

        # 使用 active_step 作为动作确认号，而不是仅凭周期状态的 sac_steps
        # 猜测动作是否已执行。旧控制器状态没有 active_step 时保留兼容路径。
        state = self._wait(
            lambda s: self._is_current_state(s)
            and (
                bool(s.get('done'))
                or s.get('controller_state') != 'SEARCHING'
                or int(s.get('active_step', -1)) >= requested_step
                or int(s.get('sac_steps', -1)) > before
            ),
            description=f'等待 SAC 动作 step_id={requested_step} 确认',
            overall_timeout_s=self.response_timeout_s,
        )

        # SAC 只负责 SEARCHING。找到孔后，该 Gym step 等待规则式
        # INSERTING/INSERT_HOLDING 完整结束。这里采用“心跳超时”，不再把
        # 整个插入过程错误限制为 30 秒墙钟时间。
        if state.get('controller_state') != 'SEARCHING' and not state.get('done'):
            state = self._wait(
                lambda s: self._is_current_state(s) and bool(s.get('done')),
                description='等待 INSERTING/INSERT_HOLDING 完成本 Episode',
                overall_timeout_s=self.terminal_timeout_s,
            )

        reward = 0.0
        terminated = bool(state['done'])
        truncated = bool(state['timeout'])

        if state['controller_state'] == 'SEARCHING' and not terminated:
            event = self.visit.enter(np.asarray(state['tip_xyz_m'][1:3]))
            ref = np.asarray(state['reference_yz_m'])
            center = np.asarray(state['search_start_m'][1:3])
            near = bool(np.any(0.004 - np.abs(ref - center) < 0.0005))
            reward += search_step_reward(
                event, action, self.previous_action, near
            )
            self.previous_action = action.copy()

        if state['hole_found'] and not self.hole_found_rewarded:
            reward += 20.0
            self.hole_found_rewarded = True
        if state['assembly_success']:
            reward += 100.0
        if state['out_of_bounds']:
            reward -= 5.0
        if state['fault']:
            reward -= 30.0
        if state['timeout']:
            reward -= 10.0
            terminated = False
            truncated = True

        tip = np.asarray(state['tip_xyz_m'][1:3])
        self.path_m += float(np.linalg.norm(tip - self.last_tip_yz))
        self.last_tip_yz = tip
        self.trajectory.append(state.copy())
        return (
            self._observation(state),
            reward,
            terminated,
            truncated,
            self._info(state, None, reward),
        )

    def _observation(self, state):
        start = np.asarray(state['search_start_m'])
        tip = np.asarray(state['tip_xyz_m'])
        velocity = np.asarray(state['tip_velocity_m_s'])
        elapsed = (
            0.0
            if state['controller_state'] != 'SEARCHING'
            else state['sac_steps'] * 0.10
        )
        return build_observation(
            tip - start,
            velocity,
            1.0 - elapsed / 40.0,
            self.previous_action,
            np.zeros(2),
            self.visit.flattened(),
            0.035,
            0.25,
            4.0,
        )

    def _info(self, state, offset=None, reward=None):
        del offset
        return {
            'episode_id': self.episode_id,
            'dataset_index': self.current_dataset_index,
            'hole_offset_y_m': float(self.current_hole_offset[0]),
            'hole_offset_z_m': float(self.current_hole_offset[1]),
            'hole_found': state['hole_found'],
            'assembly_success': state['assembly_success'],
            'fault': state['fault'],
            'timeout': state['timeout'],
            'out_of_bounds': state['out_of_bounds'],
            'insertion_failure': state['insertion_failure'],
            'termination_reason': (
                'assembly_success'
                if state['assembly_success']
                else 'fault'
                if state['fault']
                else 'out_of_bounds'
                if state['out_of_bounds']
                else 'timeout'
                if state['timeout']
                else 'insertion_failure'
                if state['insertion_failure']
                else ''
            ),
            'sac_steps': state['sac_steps'],
            'low_level_cycles': state['low_level_cycles'],
            'yz_path_length_m': self.path_m,
            'search_time_s': state['sac_steps'] * 0.10,
            'episode_reward': reward,
        }

    def close(self):
        self.node.destroy_node()
