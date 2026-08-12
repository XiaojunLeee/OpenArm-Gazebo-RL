#!/usr/bin/env python3
"""加载 best_model，在固定 100 孔测试集上确定性推理，不更新网络或回放池。"""
from pathlib import Path
import json
import numpy as np
import yaml
from stable_baselines3 import SAC
from openarm_gazebo_hole_search.hole_dataset import load_dataset
from openarm_gazebo_hole_search.openarm_hole_search_env import OpenArmHoleSearchEnv

def main():
    root = Path(__file__).resolve().parents[1]; cfg = yaml.safe_load((root/'config/sac_training.yaml').read_text())['run']
    dataset = load_dataset(root/'datasets/test_holes.npy', 'test', 100)
    env = OpenArmHoleSearchEnv(
        dataset, 'test', root/'config/sac_search.yaml',
        ros_response_timeout_s=cfg['ros_response_timeout_s'],
        ros_reset_timeout_s=cfg.get('ros_reset_timeout_s', 180.0),
        ros_terminal_timeout_s=cfg.get('ros_terminal_timeout_s', 0.0))
    model = SAC.load(root/cfg['artifact_root']/'models/best_model/sac_model', device='auto'); rows = []
    for i in range(len(dataset)):
        obs, _ = env.reset(options={'dataset_index': i}); done = truncated = False
        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True); obs, _, done, truncated, info = env.step(action)
        rows.append(info); print(f'[测试] {i + 1}/100 | 找孔={info["hole_found"]} | 装配={info["assembly_success"]} | 原因={info["termination_reason"]}', flush=True)
    out = root/cfg['artifact_root']/'testing'; out.mkdir(parents=True, exist_ok=True)
    summary = {'hole_search_success_rate': float(np.mean([r['hole_found'] for r in rows])), 'assembly_success_rate': float(np.mean([r['assembly_success'] for r in rows])), 'fault_rate': float(np.mean([r['fault'] for r in rows]))}
    (out/'test_summary.json').write_text(json.dumps(summary, indent=2)); print(f'[测试汇总] {summary}')
    env.close()
if __name__ == '__main__': main()
