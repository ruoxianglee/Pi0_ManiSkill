#!/usr/bin/env python3
"""
在 ManiSkill 环境中使用预训练的 pi0_base 模型进行推理示例。
将该脚本放到 ManiSkill/examples/baselines/ 下并确保 openpi 和 maniskill2 已安装。
"""
import argparse
import numpy as np
from openpi.policies.policy_config import create_trained_policy
from openpi.training.config import get_config
from maniskill2.env import ManiSkillEnv


def convert_obs(obs: dict) -> dict:
    """
    将 ManiSkill 的 obs 转换为 openpi policy.infer 所需的输入格式。
    假设 env.obs 包含:
      - 'state': 关节状态向量 (n_dim,)
      - 'rgb': HxWx3 uint8 图像
    返回 dict 包括键:
      - 'observation/state'
      - 'observation/image'
      - 'observation/wrist_image'
      - 可选 'prompt'
    """
    state = obs.get('state')
    rgb = obs.get('rgb')
    # 如果没有手腕相机，可用零张量填充
    wrist = np.zeros_like(rgb)
    return {
        'observation/state': state,
        'observation/image': rgb,
        'observation/wrist_image': wrist,
        # 不提供 prompt 时，使用训练时默认 prompt
    }


def main():
    parser = argparse.ArgumentParser(description='ManiSkill 中运行 pi0_base 推理')
    parser.add_argument('--env', type=str, default='LiftCube-v0', help='ManiSkill 环境名称')
    parser.add_argument('--checkpoint', type=str, required=True, help='pi0_base 模型 checkpoint 目录')
    parser.add_argument('--prompt', type=str, default=None, help='默认指令文本')
    parser.add_argument('--max_steps', type=int, default=100, help='最大环境步数')
    args = parser.parse_args()

    # 加载训练配置并创建 policy
    train_cfg = get_config('pi0_aloha')  # 使用 pi0_base 对应的配置
    policy = create_trained_policy(train_cfg, args.checkpoint, default_prompt=args.prompt)

    # 创建 ManiSkill 环境
    env = ManiSkillEnv(args.env, obs_mode='rgb', reward_type='dense')
    obs = env.reset()
    done = False
    steps = 0

    while steps < args.max_steps and not done:
        inp = convert_obs(obs)
        out = policy.infer(inp)
        actions = out['actions']  # 形状 (horizon, action_dim)
        # 将一组动作序列按步执行到环境
        for a in actions:
            obs, rew, done, info = env.step(a)
            steps += 1
            if done or steps >= args.max_steps:
                break

    env.close()


if __name__ == '__main__':
    main() 