#!/usr/bin/env python3
"""训练 SAC，支持定期验证、Ctrl+C 安全保存和自动断点续训。"""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import time
from typing import Any

import numpy as np
import yaml
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback

from openarm_gazebo_hole_search.hole_dataset import load_dataset
from openarm_gazebo_hole_search.model_selection import is_better
from openarm_gazebo_hole_search.openarm_hole_search_env import OpenArmHoleSearchEnv


RESUME_FORMAT_VERSION = 1


class EpisodeStopper(BaseCallback):
    """以真实完成 Episode 计数，并记录每一步训练 SAC action。"""

    def __init__(
        self,
        start: int,
        checkpoint_target: int,
        training_total: int,
        action_log_path: Path,
    ):
        super().__init__()
        self.start = int(start)
        self.checkpoint_target = int(checkpoint_target)
        self.training_total = int(training_total)
        self.completed = 0

        # 仅用于调试/分析，不参与 observation、reward 或控制计算。
        self.action_log_path = Path(action_log_path)
        self.action_log_path.parent.mkdir(parents=True, exist_ok=True)
        self.episode_action_step = 0
        self.logged_episode = None
        self.learning_starts_notice_done = False

    @staticmethod
    def _action_direction(action_y: float, action_z: float):
        """返回动作模长、YZ平面方向角和8方向文字标签。

        方向角定义：
          +Y =   0 deg
          +Z = +90 deg
          -Y = +/-180 deg
          -Z = -90 deg
        """
        magnitude = float(np.hypot(action_y, action_z))

        if magnitude < 1.0e-6:
            return magnitude, 0.0, 'STAY'

        angle_deg = float(np.degrees(np.arctan2(action_z, action_y)))

        if -22.5 <= angle_deg < 22.5:
            direction = '+Y'
        elif 22.5 <= angle_deg < 67.5:
            direction = '+Y+Z'
        elif 67.5 <= angle_deg < 112.5:
            direction = '+Z'
        elif 112.5 <= angle_deg < 157.5:
            direction = '-Y+Z'
        elif angle_deg >= 157.5 or angle_deg < -157.5:
            direction = '-Y'
        elif -157.5 <= angle_deg < -112.5:
            direction = '-Y-Z'
        elif -112.5 <= angle_deg < -67.5:
            direction = '-Z'
        else:
            direction = '+Y-Z'

        return magnitude, angle_deg, direction

    def _write_action_log(self, line: str) -> None:
        with self.action_log_path.open('a', encoding='utf-8') as fp:
            fp.write(line + '\n')

    def _on_step(self) -> bool:
        infos = self.locals['infos']
        dones = self.locals['dones']

        # --------------------------------------------------------
        # SAC warm-up -> policy 阶段切换提示。
        #
        # SB3 callback 在 env.step() 完成、num_timesteps 增加之后调用。
        # 因此：
        #   GlobalStep <= learning_starts : WARMUP_RANDOM
        #   GlobalStep >  learning_starts : POLICY_STOCHASTIC
        # --------------------------------------------------------
        learning_starts = int(
            getattr(self.model, 'learning_starts', 0)
        )
        global_step = int(self.num_timesteps)

        if (
            not self.learning_starts_notice_done
            and global_step > learning_starts
        ):
            if global_step == learning_starts + 1:
                transition_text = '刚刚越过'
            else:
                transition_text = '当前已经越过'

            notice = (
                f'[SAC阶段切换] GlobalStep={global_step} '
                f'{transition_text} learning_starts={learning_starts}\n'
                '从现在开始：WARMUP_RANDOM 已结束，'
                '后续训练动作来自 SAC POLICY_STOCHASTIC。\n'
                '注意：刚进入策略阶段时 Actor 尚未充分训练，'
                '动作不一定立即变平滑。'
            )

            print(
                '\n' + '=' * 100 + '\n'
                + notice + '\n'
                + '=' * 100,
                flush=True,
            )

            self._write_action_log('=' * 100)
            self._write_action_log(notice)
            self._write_action_log('=' * 100)

            self.learning_starts_notice_done = True

        # Stable-Baselines3 在 model.learn() 内实际送给环境的动作。
        # 当前工程只有一个环境，因此 actions 通常为 shape=(1, 2)。
        actions = np.asarray(self.locals.get('actions'), dtype=float)
        if actions.ndim == 1:
            actions = actions.reshape(1, -1)

        for env_index, (info, done) in enumerate(zip(infos, dones)):
            current_episode = self.start + self.completed + 1

            if self.logged_episode != current_episode:
                self.logged_episode = current_episode
                self.episode_action_step = 0
                self._write_action_log(
                    '=' * 100
                )
                self._write_action_log(
                    f'[Episode {current_episode} START]'
                )
                self._write_action_log(
                    '字段: Episode | Step | GlobalStep | Source | '
                    'action_y | action_z | magnitude | angle_deg | direction'
                )
                self._write_action_log(
                    '方向角: +Y=0deg, +Z=+90deg, -Y=+/-180deg, -Z=-90deg'
                )
                self._write_action_log(
                    '当前实验 sac_action_step = 0.05 mm；'
                    '实际每轴参考增量 = action × 0.05 mm'
                )
                self._write_action_log('-' * 100)

            self.episode_action_step += 1

            if env_index < len(actions) and actions.shape[1] >= 2:
                action_y = float(actions[env_index, 0])
                action_z = float(actions[env_index, 1])

                magnitude, angle_deg, direction = self._action_direction(
                    action_y, action_z
                )

                learning_starts = int(
                    getattr(self.model, 'learning_starts', 0)
                )
                source = (
                    'WARMUP_RANDOM'
                    if int(self.num_timesteps) <= learning_starts
                    else 'POLICY_STOCHASTIC'
                )

                self._write_action_log(
                    f'Episode={current_episode} | '
                    f'Step={self.episode_action_step} | '
                    f'GlobalStep={int(self.num_timesteps)} | '
                    f'Source={source} | '
                    f'action_y={action_y:+.6f} | '
                    f'action_z={action_z:+.6f} | '
                    f'magnitude={magnitude:.6f} | '
                    f'angle_deg={angle_deg:+.2f} | '
                    f'direction={direction}'
                )

            if not done:
                continue

            self.completed += 1
            number = self.start + self.completed

            # 这里打印的是数据集中的随机孔位置：相对本轮复位轴尖的
            # [Y, Z] 偏移，范围均为 ±3 mm；它不是插入阶段中心误差。
            hole_y_mm = info.get('hole_offset_y_m', 0.0) * 1000.0
            hole_z_mm = info.get('hole_offset_z_m', 0.0) * 1000.0
            monitor_episode = info.get('episode', {})

            self._write_action_log('-' * 100)
            self._write_action_log(
                f'[Episode {number} END] | '
                f'reason={info.get("termination_reason")} | '
                f'hole_found={info.get("hole_found")} | '
                f'assembly_success={info.get("assembly_success")} | '
                f'search_steps={info.get("sac_steps")} | '
                f'search_time_s={info.get("search_time_s", 0.0):.2f} | '
                f'yz_path_mm='
                f'{info.get("yz_path_length_m", 0.0) * 1000.0:.2f}'
            )
            self._write_action_log('=' * 100)

            print(
                f"[训练] Episode {number}/{self.training_total} | "
                f"训练集seed=100 | 孔索引={info.get('dataset_index')} | "
                f"孔位置YZ(相对复位轴尖)=({hole_y_mm:.3f}, "
                f"{hole_z_mm:.3f}) mm | "
                f"找孔={info.get('hole_found')} | "
                f"完整装配={info.get('assembly_success')} | "
                f"结束原因={info.get('termination_reason')} | "
                f"回合奖励={monitor_episode.get('r', float('nan')):.3f} | "
                f"搜索步={info.get('sac_steps')} | "
                f"搜索时间={info.get('search_time_s', 0.0):.2f} s | "
                f"YZ路径={info.get('yz_path_length_m', 0.0) * 1000.0:.2f} mm",
                flush=True,
            )

            if number >= self.checkpoint_target:
                return False

        return True


def evaluate(model: SAC, env: OpenArmHoleSearchEnv, dataset):
    """固定数据集确定性验证；不调用 learn，也不写入回放池。"""
    rows = []
    for index in range(len(dataset)):
        obs, _ = env.reset(options={'dataset_index': index})
        done = truncated = False
        reward = 0.0
        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, step_reward, done, truncated, info = env.step(action)
            reward += step_reward
        info['episode_reward'] = reward
        info['dataset_index'] = index
        rows.append(info)

    found = [row for row in rows if row['hole_found']]
    metrics = {
        'assembly_success_rate': float(
            np.mean([row['assembly_success'] for row in rows])
        ),
        'hole_search_success_rate': float(
            np.mean([row['hole_found'] for row in rows])
        ),
        'fault_rate': float(np.mean([row['fault'] for row in rows])),
        'mean_successful_search_time_s': (
            None
            if not found
            else float(np.mean([row['search_time_s'] for row in found]))
        ),
        'mean_successful_yz_path_length_m': (
            None
            if not found
            else float(np.mean([row['yz_path_length_m'] for row in found]))
        ),
    }
    return metrics, rows


def _model_zip_path(checkpoint_dir: Path) -> Path:
    return checkpoint_dir / 'sac_model.zip'


def _state_path(checkpoint_dir: Path) -> Path:
    return checkpoint_dir / 'training_state.json'


def _checkpoint_is_complete(checkpoint_dir: Path) -> bool:
    return (
        _model_zip_path(checkpoint_dir).is_file()
        and (checkpoint_dir / 'replay_buffer.pkl').is_file()
        and _state_path(checkpoint_dir).is_file()
    )


def _write_training_state(
    checkpoint_dir: Path,
    *,
    completed: int,
    last_validated_episode: int,
    model: SAC,
    reason: str,
) -> dict[str, Any]:
    state = {
        'resume_format_version': RESUME_FORMAT_VERSION,
        'training_episode': int(completed),
        'last_validated_episode': int(last_validated_episode),
        'total_env_steps': int(model.num_timesteps),
        'reason': str(reason),
        'saved_at_unix': time.time(),
        'saved_at_local': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
    }
    _state_path(checkpoint_dir).write_text(
        json.dumps(state, indent=2, ensure_ascii=False),
        encoding='utf-8',
    )
    return state


def save_resumable_checkpoint(
    model: SAC,
    checkpoint_dir: Path,
    *,
    completed: int,
    last_validated_episode: int,
    reason: str,
    root: Path,
) -> dict[str, Any]:
    """保存继续训练所需的模型、回放池、进度和冻结配置。"""
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model.save(checkpoint_dir / 'sac_model')
    model.save_replay_buffer(checkpoint_dir / 'replay_buffer.pkl')
    state = _write_training_state(
        checkpoint_dir,
        completed=completed,
        last_validated_episode=last_validated_episode,
        model=model,
        reason=reason,
    )
    shutil.copy2(
        root / 'config/sac_training.yaml',
        checkpoint_dir / 'resolved_training_config.yaml',
    )
    shutil.copy2(
        root / 'config/sac_search.yaml',
        checkpoint_dir / 'resolved_search_config.yaml',
    )
    return state


def find_latest_resume_checkpoint(results: Path, training_total: int):
    """寻找最新的周期检查点或 Ctrl+C/异常保护检查点。"""
    models_dir = results / 'models'
    candidates = []
    candidates.extend((models_dir / 'checkpoints').glob('episode_*'))
    candidates.append(models_dir / 'interrupt_checkpoint')

    valid = []
    for checkpoint_dir in candidates:
        if not _checkpoint_is_complete(checkpoint_dir):
            continue
        try:
            state = json.loads(
                _state_path(checkpoint_dir).read_text(encoding='utf-8')
            )
        except (OSError, json.JSONDecodeError):
            continue
        if state.get('resume_format_version') != RESUME_FORMAT_VERSION:
            continue
        completed = int(state.get('training_episode', -1))
        if not 0 <= completed <= int(training_total):
            continue
        valid.append(
            (
                completed,
                float(state.get('saved_at_unix', 0.0)),
                checkpoint_dir,
                state,
            )
        )

    if not valid:
        return None
    _, _, checkpoint_dir, state = max(valid, key=lambda item: (item[0], item[1]))
    return checkpoint_dir, state


def create_fresh_model(env, sac_cfg: dict[str, Any], tensorboard_dir: Path) -> SAC:
    return SAC(
        'MlpPolicy',
        env,
        policy_kwargs={'net_arch': sac_cfg['net_arch']},
        learning_rate=sac_cfg['learning_rate'],
        gamma=sac_cfg['gamma'],
        tau=sac_cfg['tau'],
        batch_size=sac_cfg['batch_size'],
        buffer_size=sac_cfg['buffer_size'],
        learning_starts=sac_cfg['learning_starts'],
        train_freq=sac_cfg['train_freq'],
        gradient_steps=sac_cfg['gradient_steps'],
        target_update_interval=sac_cfg['target_update_interval'],
        ent_coef=sac_cfg['ent_coef'],
        target_entropy=sac_cfg['target_entropy'],
        device=sac_cfg['device'],
        tensorboard_log=str(tensorboard_dir),
        verbose=1,
    )


def load_or_create_model(
    env: OpenArmHoleSearchEnv,
    sac_cfg: dict[str, Any],
    results: Path,
    training_total: int,
):
    latest = find_latest_resume_checkpoint(results, training_total)
    if latest is None:
        print('[自动续训] 未找到兼容检查点，将从 Episode 1 开始训练。', flush=True)
        return (
            create_fresh_model(env, sac_cfg, results / 'tensorboard'),
            0,
            0,
            None,
        )

    checkpoint_dir, state = latest
    completed = int(state['training_episode'])
    last_validated = int(state.get('last_validated_episode', 0))

    # 固定训练孔顺序由 episode_id 决定。设置为已经完整完成的回合数后，
    # 新进程第一次 reset 会从下一个尚未完成的训练孔开始。
    env.set_completed_episode_count(completed)
    model = SAC.load(
        checkpoint_dir / 'sac_model',
        env=env,
        device=sac_cfg['device'],
        tensorboard_log=str(results / 'tensorboard'),
    )
    model.load_replay_buffer(checkpoint_dir / 'replay_buffer.pkl')
    print(
        f'[自动续训] 已读取：{checkpoint_dir}\n'
        f'[自动续训] 从 Episode {completed + 1} 继续；'
        f'已完成={completed}/{training_total}，'
        f'总环境步={model.num_timesteps}，'
        f'上次完成验证={last_validated}',
        flush=True,
    )
    return model, completed, last_validated, checkpoint_dir


def load_best_metrics(results: Path):
    info_path = results / 'models/best_model/best_model_info.json'
    if not info_path.is_file():
        return None
    try:
        data = json.loads(info_path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError):
        return None
    required = {
        'assembly_success_rate',
        'hole_search_success_rate',
        'fault_rate',
        'mean_successful_search_time_s',
        'mean_successful_yz_path_length_m',
    }
    return data if required.issubset(data) else None


def reacquire_training_rollout(model: SAC) -> None:
    """验证会话结束后，让训练环境重新拥有 Gazebo 控制器。"""
    vec_env = model.get_env()
    if vec_env is None or not hasattr(vec_env, 'envs') or len(vec_env.envs) != 1:
        raise RuntimeError('当前恢复逻辑要求单环境 DummyVecEnv')

    raw_env = vec_env.envs[0].unwrapped
    obs, _ = raw_env.reacquire_current_episode()
    obs = np.asarray(obs, dtype=np.float32)

    # reset_num_timesteps=False 时，下一次 learn() 会继续使用 _last_obs，
    # 因此必须替换为重新接管后得到的真实初始观测。
    model._last_obs = np.expand_dims(obs, axis=0)
    model._last_episode_starts = np.ones((1,), dtype=bool)
    if hasattr(vec_env, '_save_obs'):
        vec_env._save_obs(0, obs)

    print(
        f'[训练恢复] 已重新接管训练 Episode '
        f'{raw_env.episode_id}，孔索引={raw_env.current_dataset_index}',
        flush=True,
    )


def run_validation(
    model: SAC,
    valid,
    root: Path,
    run_cfg: dict[str, Any],
    results: Path,
    completed: int,
    checkpoint_dir: Path,
    best,
):
    validation_env = OpenArmHoleSearchEnv(
        valid,
        'validation',
        root / 'config/sac_search.yaml',
        ros_response_timeout_s=run_cfg['ros_response_timeout_s'],
        ros_reset_timeout_s=run_cfg.get('ros_reset_timeout_s', 180.0),
        ros_terminal_timeout_s=run_cfg.get('ros_terminal_timeout_s', 0.0),
    )
    try:
        metrics, rows = evaluate(model, validation_env, valid)
    finally:
        validation_env.close()

    (checkpoint_dir / 'validation_metrics.json').write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False),
        encoding='utf-8',
    )
    (checkpoint_dir / 'validation_rows.json').write_text(
        json.dumps(rows, indent=2, ensure_ascii=False),
        encoding='utf-8',
    )
    print(
        f'[检查点] 保存 Episode {completed}：{checkpoint_dir}\n'
        f'[验证] {metrics}',
        flush=True,
    )

    if is_better(metrics, best):
        best = metrics
        best_dir = results / 'models/best_model'
        best_dir.mkdir(parents=True, exist_ok=True)
        model.save(best_dir / 'sac_model')
        (best_dir / 'best_model_info.json').write_text(
            json.dumps(
                {
                    **metrics,
                    'source_training_episode': completed,
                    'total_env_steps': model.num_timesteps,
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding='utf-8',
        )
        print(
            f'[最优模型] 更新为 Episode {completed}，指标：{metrics}',
            flush=True,
        )
    return best


def next_checkpoint_episode(completed: int, interval: int, total: int) -> int:
    next_multiple = ((int(completed) // int(interval)) + 1) * int(interval)
    return min(next_multiple, int(total))


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load(
        (root / 'config/sac_training.yaml').read_text(encoding='utf-8')
    )
    run_cfg, sac_cfg = config['run'], config['sac']
    training_total = int(run_cfg['training_episodes'])
    validation_interval = int(run_cfg['validation_interval'])
    if validation_interval <= 0:
        raise ValueError('validation_interval 必须大于 0')

    results = root / run_cfg['artifact_root']
    results.mkdir(exist_ok=True)
    train = load_dataset(root / 'datasets/train_holes.npy', 'train', 1000)
    valid = load_dataset(
        root / 'datasets/validation_holes.npy', 'validation', 30
    )

    # seed=100 固定打乱 1000 个训练孔的使用顺序。断点续训时通过已完成
    # Episode 数恢复到同一顺序位置，不会重新从第一个训练孔开始。
    env = OpenArmHoleSearchEnv(
        train,
        'training',
        root / 'config/sac_search.yaml',
        ros_response_timeout_s=run_cfg['ros_response_timeout_s'],
        ros_reset_timeout_s=run_cfg.get('ros_reset_timeout_s', 180.0),
        ros_terminal_timeout_s=run_cfg.get('ros_terminal_timeout_s', 0.0),
        episode_order_seed=100,
    )

    model = None
    completed = 0
    last_validated_episode = 0
    active_callback = None
    segment_start = 0

    try:
        model, completed, last_validated_episode, _ = load_or_create_model(
            env, sac_cfg, results, training_total
        )
        best = load_best_metrics(results)

        if completed >= training_total:
            print(
                f'[训练完成] 检查点已记录 {completed}/{training_total} 个回合；'
                '无需继续训练。',
                flush=True,
            )
            return

        # 上一次若在周期验证过程中 Ctrl+C，模型检查点已经保存，但验证文件
        # 尚未完成。新进程先补做该次验证，再开始下一个训练区间。
        if (
            completed > 0
            and completed % validation_interval == 0
            and last_validated_episode < completed
        ):
            checkpoint_dir = (
                results
                / 'models/checkpoints'
                / f'episode_{completed:04d}'
            )
            if not _checkpoint_is_complete(checkpoint_dir):
                save_resumable_checkpoint(
                    model,
                    checkpoint_dir,
                    completed=completed,
                    last_validated_episode=last_validated_episode,
                    reason='periodic_before_validation',
                    root=root,
                )
            print(
                f'[自动续训] Episode {completed} 的验证尚未完成，'
                '先补做本次 30 孔验证。',
                flush=True,
            )
            best = run_validation(
                model,
                valid,
                root,
                run_cfg,
                results,
                completed,
                checkpoint_dir,
                best,
            )
            last_validated_episode = completed
            _write_training_state(
                checkpoint_dir,
                completed=completed,
                last_validated_episode=last_validated_episode,
                model=model,
                reason='periodic_validation_complete',
            )

        while completed < training_total:
            target = next_checkpoint_episode(
                completed, validation_interval, training_total
            )
            segment_start = completed
            active_callback = EpisodeStopper(
                segment_start,
                target,
                training_total,
                action_log_path=results / 'logs' / 'action.log',
            )
            model.learn(
                total_timesteps=10**9,
                callback=active_callback,
                reset_num_timesteps=False,
            )
            completed = segment_start + active_callback.completed
            active_callback = None

            if completed != target:
                raise RuntimeError(
                    f'训练区间应完成到 Episode {target}，实际为 {completed}'
                )

            checkpoint_dir = (
                results
                / 'models/checkpoints'
                / f'episode_{completed:04d}'
            )
            save_resumable_checkpoint(
                model,
                checkpoint_dir,
                completed=completed,
                last_validated_episode=last_validated_episode,
                reason='periodic_before_validation',
                root=root,
            )

            best = run_validation(
                model,
                valid,
                root,
                run_cfg,
                results,
                completed,
                checkpoint_dir,
                best,
            )
            last_validated_episode = completed
            _write_training_state(
                checkpoint_dir,
                completed=completed,
                last_validated_episode=last_validated_episode,
                model=model,
                reason='periodic_validation_complete',
            )

            if completed < training_total:
                reacquire_training_rollout(model)

    except KeyboardInterrupt:
        if model is not None:
            if active_callback is not None:
                completed = segment_start + active_callback.completed
            interrupt_dir = results / 'models/interrupt_checkpoint'
            print(
                '\n[中断保存] 收到 Ctrl+C，正在保存模型和 Replay Buffer；'
                '保存完成前请不要再次按 Ctrl+C……',
                flush=True,
            )
            save_resumable_checkpoint(
                model,
                interrupt_dir,
                completed=completed,
                last_validated_episode=last_validated_episode,
                reason='keyboard_interrupt',
                root=root,
            )
            print(
                f'[中断保存] 已完成：{interrupt_dir}\n'
                f'[中断保存] 下次启动将从 Episode {completed + 1} 自动继续。',
                flush=True,
            )
        else:
            print('\n[中断] 模型尚未创建，没有可保存的训练状态。', flush=True)
    except Exception:
        if model is not None:
            if active_callback is not None:
                completed = segment_start + active_callback.completed
            failure_dir = results / 'models/interrupt_checkpoint'
            print(
                '\n[异常保护] 训练异常，正在保存最近可续训状态……',
                flush=True,
            )
            try:
                save_resumable_checkpoint(
                    model,
                    failure_dir,
                    completed=completed,
                    last_validated_episode=last_validated_episode,
                    reason='exception',
                    root=root,
                )
                print(
                    f'[异常保护] 已保存：{failure_dir}',
                    flush=True,
                )
            except Exception as save_error:
                print(
                    f'[异常保护] 保存失败：{save_error!r}',
                    flush=True,
                )
        raise
    finally:
        env.close()


if __name__ == '__main__':
    main()
