#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time

import numpy as np
import torch

from two_mouse_il.env_setup import (
    action_layout_from_space,
    make_env,
    ordered_agent_keys,
    unflatten_action_vector,
)
from two_mouse_il.policy import load_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a behavior cloning checkpoint in a dual-arm ManiSkill task.")
    parser.add_argument("--checkpoint", required=True, help="Path to the saved BC checkpoint.")
    parser.add_argument("--episodes", type=int, default=5, help="Number of evaluation episodes.")
    parser.add_argument("--render", action="store_true", help="Render the environment with a human viewer.")
    parser.add_argument("--seed", type=int, default=1000, help="Starting seed for evaluation episodes.")
    return parser.parse_args()


def _to_scalar(value) -> float:
    if isinstance(value, np.ndarray):
        return float(value.reshape(-1)[0])
    if torch.is_tensor(value):
        return float(value.reshape(-1)[0].item())
    return float(value)


def _to_bool(value) -> bool:
    if isinstance(value, np.ndarray):
        return bool(value.reshape(-1)[0])
    if torch.is_tensor(value):
        return bool(value.reshape(-1)[0].item())
    return bool(value)


def main() -> int:
    args = parse_args()
    checkpoint = load_checkpoint(args.checkpoint)
    env = make_env(
        env_id=checkpoint.metadata["env_id"],
        control_mode=checkpoint.metadata["control_mode"],
        render_mode="rgb_array",
        obs_mode="state",
        reward_mode="normalized_dense",
    )
    agent_keys = ordered_agent_keys(env)
    layout = action_layout_from_space(env.unwrapped.single_action_space, agent_keys)

    success_once_count = 0
    success_at_end_count = 0
    returns = []

    try:
        for episode_idx in range(args.episodes):
            obs, _ = env.reset(seed=args.seed + episode_idx)
            done = False
            episode_return = 0.0
            episode_success_once = False
            last_success = False
            while not done:
                if args.render:
                    env.render_human()
                action_vector = checkpoint.predict(np.asarray(obs, dtype=np.float32).reshape(-1))
                action = unflatten_action_vector(action_vector, layout)
                obs, reward, terminated, truncated, info = env.step(action)
                episode_return += _to_scalar(reward)
                current_success = _to_bool(info.get("success", False))
                episode_success_once = episode_success_once or current_success
                last_success = current_success
                done = _to_bool(terminated) or _to_bool(truncated)
                if args.render:
                    time.sleep(0.02)

            success_once_count += int(episode_success_once)
            success_at_end_count += int(last_success)
            returns.append(episode_return)
            print(
                f"episode={episode_idx} return={episode_return:.4f} success_once={episode_success_once} success_at_end={last_success}"
            )
    finally:
        env.close()

    print(
        f"avg_return={np.mean(returns):.4f} success_once={success_once_count / args.episodes:.4f} success_at_end={success_at_end_count / args.episodes:.4f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
