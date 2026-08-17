#!/usr/bin/env python3
"""加载 best_model，在固定 100 孔测试集上确定性推理，不更新网络或回放池。"""

from pathlib import Path
import json

import numpy as np
import yaml
from stable_baselines3 import SAC

from openarm_gazebo_hole_search.hole_dataset import load_dataset
from openarm_gazebo_hole_search.openarm_hole_search_env import OpenArmHoleSearchEnv


def action_direction(action):
    """返回动作模长、YZ 平面方向角和 8 方向文字标签。

    方向角定义：
      +Y =   0 deg
      +Z = +90 deg
      -Y = +/-180 deg
      -Z = -90 deg
    """
    action_y = float(action[0])
    action_z = float(action[1])

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


def adjacent_action_angle(current_action, previous_action):
    """返回当前 action 与上一 action 的最小夹角，范围 0～180 deg。"""
    if previous_action is None:
        return None

    current = np.asarray(current_action, dtype=np.float64)
    previous = np.asarray(previous_action, dtype=np.float64)

    current_norm = float(np.linalg.norm(current))
    previous_norm = float(np.linalg.norm(previous))

    if current_norm < 1.0e-6 or previous_norm < 1.0e-6:
        return None

    cosine = float(
        np.dot(current, previous)
        / (current_norm * previous_norm)
    )
    cosine = float(np.clip(cosine, -1.0, 1.0))

    return float(np.degrees(np.arccos(cosine)))


def write_action_log(path: Path, line: str):
    """向测试动作日志追加一行。"""
    with path.open('a', encoding='utf-8') as fp:
        fp.write(line + '\n')


def main():
    root = Path(__file__).resolve().parents[1]

    cfg = yaml.safe_load(
        (root / 'config/sac_training.yaml').read_text()
    )['run']

    dataset = load_dataset(
        root / 'datasets/test_holes.npy',
        'test',
        100,
    )

    # ============================================================
    # 测试 action.log
    #
    # 每次启动 test_sac.py 都重新创建 action.log，
    # 因此自动覆盖上一次测试留下的 action.log。
    # ============================================================
    test_log_dir = root / cfg['artifact_root'] / 'test'
    test_log_dir.mkdir(parents=True, exist_ok=True)

    action_log_path = test_log_dir / 'action.log'

    action_log_path.write_text(
        'SAC TEST ACTION LOG\n'
        '策略模式: DETERMINISTIC\n'
        '方向角: +Y=0deg, +Z=+90deg, '
        '-Y=+/-180deg, -Z=-90deg\n'
        'turn_angle_deg: 当前 action 与上一 action '
        '之间的最小夹角，范围 0～180deg\n'
        '每个测试 Episode 的 Step 1 因没有上一动作，'
        '因此 turn_angle_deg=N/A\n'
        + '=' * 120 + '\n',
        encoding='utf-8',
    )

    env = OpenArmHoleSearchEnv(
        dataset,
        'test',
        root / 'config/sac_search.yaml',
        ros_response_timeout_s=cfg['ros_response_timeout_s'],
        ros_reset_timeout_s=cfg.get(
            'ros_reset_timeout_s',
            180.0,
        ),
        ros_terminal_timeout_s=cfg.get(
            'ros_terminal_timeout_s',
            0.0,
        ),
    )

    model = SAC.load(
        root / cfg['artifact_root'] / 'models/best_model/sac_model',
        device='auto',
    )

    rows = []

    for i in range(len(dataset)):
        obs, _ = env.reset(
            options={'dataset_index': i}
        )

        done = False
        truncated = False

        action_step = 0
        previous_action = None

        write_action_log(
            action_log_path,
            '=' * 120,
        )
        write_action_log(
            action_log_path,
            f'[Test Episode {i + 1} START]',
        )
        write_action_log(
            action_log_path,
            '字段: TestEpisode | Step | Source | '
            'action_y | action_z | magnitude | '
            'angle_deg | direction | turn_angle_deg',
        )
        write_action_log(
            action_log_path,
            '-' * 120,
        )

        while not (done or truncated):
            # 测试保持原来的 deterministic=True，不改变测试策略。
            action, _ = model.predict(
                obs,
                deterministic=True,
            )

            action = np.asarray(
                action,
                dtype=np.float64,
            ).reshape(-1)

            action_step += 1

            magnitude, angle_deg, direction = action_direction(
                action
            )

            turn_angle_deg = adjacent_action_angle(
                action,
                previous_action,
            )

            turn_angle_text = (
                'N/A'
                if turn_angle_deg is None
                else f'{turn_angle_deg:.2f}'
            )

            write_action_log(
                action_log_path,
                f'TestEpisode={i + 1} | '
                f'Step={action_step} | '
                f'Source=DETERMINISTIC | '
                f'action_y={float(action[0]):+.6f} | '
                f'action_z={float(action[1]):+.6f} | '
                f'magnitude={magnitude:.6f} | '
                f'angle_deg={angle_deg:+.2f} | '
                f'direction={direction} | '
                f'turn_angle_deg={turn_angle_text}',
            )

            previous_action = action.copy()

            obs, _, done, truncated, info = env.step(
                action
            )

        write_action_log(
            action_log_path,
            '-' * 120,
        )

        write_action_log(
            action_log_path,
            f'[Test Episode {i + 1} END] | '
            f'reason={info.get("termination_reason")} | '
            f'hole_found={info.get("hole_found")} | '
            f'assembly_success={info.get("assembly_success")} | '
            f'search_steps={info.get("sac_steps")} | '
            f'search_time_s='
            f'{info.get("search_time_s", 0.0):.2f} | '
            f'yz_path_mm='
            f'{info.get("yz_path_length_m", 0.0) * 1000.0:.2f}',
        )

        write_action_log(
            action_log_path,
            '=' * 120,
        )

        rows.append(info)

        print(
            f'[测试] {i + 1}/100 | '
            f'找孔={info["hole_found"]} | '
            f'装配={info["assembly_success"]} | '
            f'原因={info["termination_reason"]}',
            flush=True,
        )

    out = root / cfg['artifact_root'] / 'testing'
    out.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary = {
        'hole_search_success_rate': float(
            np.mean([r['hole_found'] for r in rows])
        ),
        'assembly_success_rate': float(
            np.mean([r['assembly_success'] for r in rows])
        ),
        'fault_rate': float(
            np.mean([r['fault'] for r in rows])
        ),
    }

    (out / 'test_summary.json').write_text(
        json.dumps(
            summary,
            indent=2,
        )
    )

    print(
        f'[测试汇总] {summary}',
        flush=True,
    )

    print(
        f'[测试动作日志] {action_log_path}',
        flush=True,
    )

    env.close()


if __name__ == '__main__':
    main()
