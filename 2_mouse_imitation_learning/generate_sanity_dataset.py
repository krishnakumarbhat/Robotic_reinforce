#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from two_mouse_il.dataset import DatasetWriter, EpisodeData
from two_mouse_il.env_setup import action_layout_from_space, flatten_multi_agent_action, make_env, ordered_agent_keys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a small scripted dataset to sanity-check the dual-arm IL pipeline.")
    parser.add_argument("--env-id", default="TwoRobotPickCube-v1", help="ManiSkill environment id.")
    parser.add_argument("--dataset", default="data/recordings/two_robot_pick_cube_sanity.h5", help="Output dataset H5 path.")
    parser.add_argument("--metadata", default=None, help="Optional metadata JSON path.")
    parser.add_argument("--episodes", type=int, default=4, help="Number of scripted episodes.")
    parser.add_argument("--steps", type=int, default=30, help="Maximum steps per scripted episode.")
    parser.add_argument("--seed", type=int, default=123, help="Starting seed.")
    return parser.parse_args()


def _scalar(value) -> float:
    if isinstance(value, np.ndarray):
        return float(value.reshape(-1)[0])
    if torch.is_tensor(value):
        return float(value.reshape(-1)[0].item())
    return float(value)


def _as_bool(value) -> bool:
    if isinstance(value, np.ndarray):
        return bool(value.reshape(-1)[0])
    if torch.is_tensor(value):
        return bool(value.reshape(-1)[0].item())
    return bool(value)


def _scripted_action(step_idx: int, phase_shift: float = 0.0) -> np.ndarray:
    phase = 0.15 * step_idx + phase_shift
    x = 0.25 * np.sin(phase)
    y = 0.20 * np.cos(phase)
    z = 0.15 * np.sin(0.5 * phase)
    gripper = -1.0 if step_idx % 20 < 10 else 1.0
    return np.asarray([x, y, z, gripper], dtype=np.float32)


def main() -> int:
    args = parse_args()
    dataset_path = Path(args.dataset)
    metadata_path = Path(args.metadata) if args.metadata else dataset_path.with_suffix(".json")

    env = make_env(
        env_id=args.env_id,
        control_mode="pd_ee_delta_pos",
        obs_mode="state",
        reward_mode="normalized_dense",
        render_mode="rgb_array",
    )
    try:
        agent_keys = ordered_agent_keys(env)
        if len(agent_keys) != 2:
            raise RuntimeError(f"Expected a two-agent environment, got {agent_keys}")
        layout = action_layout_from_space(env.unwrapped.single_action_space, agent_keys)
        dataset_metadata = dict(
            env_id=args.env_id,
            control_mode="pd_ee_delta_pos",
            obs_dim=int(np.prod(env.unwrapped.single_observation_space.shape)),
            action_dim=layout.total_dim,
            agent_keys=list(layout.agent_keys),
            action_slices={key: list(value) for key, value in layout.slices.items()},
            generated_by="generate_sanity_dataset.py",
        )
        writer = DatasetWriter(dataset_path, metadata_path, dataset_metadata)
        try:
            for episode_id in range(args.episodes):
                obs, _ = env.reset(seed=args.seed + episode_id)
                episode = EpisodeData(obs=[], actions=[], rewards=[], successes=[], dones=[])
                for step_idx in range(args.steps):
                    action = {
                        agent_keys[0]: _scripted_action(step_idx, phase_shift=0.0),
                        agent_keys[1]: _scripted_action(step_idx, phase_shift=1.2),
                    }
                    flat_obs = np.asarray(obs, dtype=np.float32).reshape(-1)
                    flat_action = flatten_multi_agent_action(action, layout)
                    next_obs, reward, terminated, truncated, info = env.step(action)
                    reward_value = _scalar(reward)
                    success = _as_bool(info.get("success", False))
                    done = _as_bool(terminated) or _as_bool(truncated)
                    episode.obs.append(flat_obs)
                    episode.actions.append(flat_action)
                    episode.rewards.append(reward_value)
                    episode.successes.append(success)
                    episode.dones.append(done)
                    obs = next_obs
                    if done:
                        break
                writer.write_episode(episode_id=episode_id, seed=args.seed + episode_id, episode=episode)
                print(f"saved episode={episode_id} len={len(episode.obs)} return={sum(episode.rewards):.4f} success={episode.successes[-1] if episode.successes else False}")
        finally:
            writer.close()
    finally:
        env.close()

    print(f"Dataset written to {dataset_path}")
    print(f"Metadata written to {metadata_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())