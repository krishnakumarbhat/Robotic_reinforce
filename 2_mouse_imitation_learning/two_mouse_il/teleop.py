from __future__ import annotations

import signal
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .dataset import DatasetWriter, EpisodeData
from .env_setup import (
    action_layout_from_space,
    flatten_multi_agent_action,
    make_env,
    ordered_agent_keys,
)
from .input_devices import DeviceAccessError, MultiMouseReader, choose_pointer_pair


@dataclass
class TeleopConfig:
    env_id: str
    dataset_path: Path
    metadata_path: Path
    seed: int
    hz: float
    include_touchpad: bool
    left_event_path: str | None
    right_event_path: str | None
    grab_devices: bool
    xy_action_scale: float
    z_action_scale: float
    control_mode: str = "pd_ee_delta_pos"
    reward_mode: str = "normalized_dense"
    open_gripper_action: float = 1.0
    closed_gripper_action: float = -1.0


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


def _new_episode() -> EpisodeData:
    return EpisodeData(obs=[], actions=[], rewards=[], successes=[], dones=[])


def _mouse_frame_to_action(frame, arm_side: str, gripper_closed: bool, config: TeleopConfig) -> np.ndarray:
    x_delta = np.clip(frame.dx * config.xy_action_scale, -1.0, 1.0)
    local_forward = np.clip(-frame.dy * config.xy_action_scale, -1.0, 1.0)
    y_delta = local_forward if arm_side == "left" else -local_forward
    z_delta = np.clip(frame.wheel * config.z_action_scale, -1.0, 1.0)
    gripper = config.closed_gripper_action if gripper_closed else config.open_gripper_action
    return np.asarray([x_delta, y_delta, z_delta, gripper], dtype=np.float32)


def run_dual_mouse_teleop(config: TeleopConfig) -> int:
    left_info, right_info = choose_pointer_pair(
        include_touchpad=config.include_touchpad,
        left_event_path=config.left_event_path,
        right_event_path=config.right_event_path,
    )

    try:
        reader = MultiMouseReader(left_info=left_info, right_info=right_info, grab=config.grab_devices)
    except (PermissionError, DeviceAccessError) as exc:
        print(f"Could not open two input devices: {exc}")
        print("Grant read access to /dev/input/event* or add your user to the input group.")
        return 3

    env = make_env(
        env_id=config.env_id,
        control_mode=config.control_mode,
        obs_mode="state",
        reward_mode=config.reward_mode,
        render_mode="rgb_array",
    )
    agent_keys = ordered_agent_keys(env)
    if len(agent_keys) != 2:
        reader.close()
        env.close()
        raise RuntimeError(f"Expected exactly two agents, got {agent_keys}")
    layout = action_layout_from_space(env.unwrapped.single_action_space, agent_keys)

    dataset_metadata = dict(
        env_id=config.env_id,
        control_mode=config.control_mode,
        obs_dim=int(np.prod(env.unwrapped.single_observation_space.shape)),
        action_dim=layout.total_dim,
        agent_keys=list(layout.agent_keys),
        action_slices={key: list(value) for key, value in layout.slices.items()},
        left_device=dict(path=left_info.event_path, name=left_info.name),
        right_device=dict(path=right_info.event_path, name=right_info.name),
    )
    writer = DatasetWriter(config.dataset_path, config.metadata_path, dataset_metadata)

    stop_requested = False

    def _handle_sigint(_signum, _frame):
        nonlocal stop_requested
        stop_requested = True

    old_handler = signal.signal(signal.SIGINT, _handle_sigint)
    print("Dual-mouse teleoperation started.")
    print(f"Left device:  {left_info.event_path} ({left_info.name})")
    print(f"Right device: {right_info.event_path} ({right_info.name})")
    print("Controls:")
    print("  move mouse: XY for that arm")
    print("  wheel: Z for that arm")
    print("  left click: toggle gripper for that arm")
    print("  left mouse middle click: save episode and reset")
    print("  right mouse middle click: save episode and quit")
    print("  Ctrl+C: quit safely")

    try:
        obs, _ = env.reset(seed=config.seed)
        episode = _new_episode()
        episode_id = 0
        seed = config.seed
        left_gripper_closed = False
        right_gripper_closed = False

        while True:
            loop_start = time.monotonic()
            env.render_human()
            frames = reader.poll()
            left_frame = frames["left"]
            right_frame = frames["right"]

            if left_frame.left_click:
                left_gripper_closed = not left_gripper_closed
            if right_frame.left_click:
                right_gripper_closed = not right_gripper_closed

            should_save_and_reset = left_frame.middle_click
            should_save_and_quit = right_frame.middle_click or stop_requested

            action = {
                agent_keys[0]: _mouse_frame_to_action(left_frame, arm_side="left", gripper_closed=left_gripper_closed, config=config),
                agent_keys[1]: _mouse_frame_to_action(right_frame, arm_side="right", gripper_closed=right_gripper_closed, config=config),
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

            if success:
                print(f"episode={episode_id} success=True reward={reward_value:.4f}")

            if done or should_save_and_reset or should_save_and_quit:
                writer.write_episode(episode_id=episode_id, seed=seed, episode=episode)
                print(
                    f"saved episode={episode_id} len={len(episode.obs)} return={sum(episode.rewards):.4f} success={episode.successes[-1] if episode.successes else False}"
                )
                episode_id += 1
                if should_save_and_quit:
                    break
                episode = _new_episode()
                seed += 1
                obs, _ = env.reset(seed=seed)

            sleep_time = max(0.0, (1.0 / config.hz) - (time.monotonic() - loop_start))
            if sleep_time > 0:
                time.sleep(sleep_time)
    finally:
        signal.signal(signal.SIGINT, old_handler)
        reader.close()
        env.close()
        writer.close()

    print(f"Dataset written to {config.dataset_path}")
    print(f"Metadata written to {config.metadata_path}")
    return 0
