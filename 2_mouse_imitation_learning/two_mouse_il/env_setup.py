from __future__ import annotations

from dataclasses import dataclass

import gymnasium as gym
import numpy as np
from gymnasium import spaces

import mani_skill.envs  # noqa: F401


@dataclass(frozen=True)
class ActionLayout:
    agent_keys: tuple[str, ...]
    slices: dict[str, tuple[int, int]]
    total_dim: int


def make_env(
    env_id: str = "TwoRobotPickCube-v1",
    control_mode: str = "pd_ee_delta_pos",
    obs_mode: str = "state",
    reward_mode: str = "normalized_dense",
    render_mode: str = "rgb_array",
):
    return gym.make(
        env_id,
        obs_mode=obs_mode,
        reward_mode=reward_mode,
        control_mode=control_mode,
        render_mode=render_mode,
        robot_uids=("panda_wristcam", "panda_wristcam"),
        human_render_camera_configs=dict(shader_pack="default"),
        viewer_camera_configs=dict(shader_pack="default"),
    )


def ordered_agent_keys(env) -> list[str]:
    return list(env.unwrapped.agent.agents_dict.keys())


def action_layout_from_space(action_space: spaces.Dict, agent_keys: list[str]) -> ActionLayout:
    cursor = 0
    slices: dict[str, tuple[int, int]] = {}
    for key in agent_keys:
        subspace = action_space[key]
        if not isinstance(subspace, spaces.Box):
            raise TypeError(f"Expected Box action space for {key}, got {type(subspace)}")
        dim = int(np.prod(subspace.shape))
        slices[key] = (cursor, cursor + dim)
        cursor += dim
    return ActionLayout(agent_keys=tuple(agent_keys), slices=slices, total_dim=cursor)


def flatten_multi_agent_action(action: dict[str, np.ndarray], layout: ActionLayout) -> np.ndarray:
    chunks = []
    for key in layout.agent_keys:
        chunks.append(np.asarray(action[key], dtype=np.float32).reshape(-1))
    return np.concatenate(chunks, axis=0)


def unflatten_action_vector(action_vector: np.ndarray, layout: ActionLayout) -> dict[str, np.ndarray]:
    action_vector = np.asarray(action_vector, dtype=np.float32).reshape(-1)
    actions: dict[str, np.ndarray] = {}
    for key in layout.agent_keys:
        start, end = layout.slices[key]
        actions[key] = action_vector[start:end].astype(np.float32, copy=False)
    return actions
