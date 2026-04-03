#!/usr/bin/env python3
from __future__ import annotations

import argparse

import numpy as np
import torch

from two_mouse_il.env_setup import (
    action_layout_from_space,
    make_env,
    ordered_agent_keys,
    unflatten_action_vector,
)
from two_mouse_il.input_devices import describe_devices, list_candidate_pointers


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke-test the two-mouse project: device discovery plus ManiSkill startup.")
    parser.add_argument("--include-touchpad", action="store_true", help="Include touchpads when listing pointer candidates.")
    parser.add_argument("--env-id", default="TwoRobotPickCube-v1", help="Two-arm ManiSkill env to instantiate.")
    parser.add_argument("--steps", type=int, default=5, help="Number of zero-action environment steps to run.")
    return parser.parse_args()


def _success_to_bool(value) -> bool:
    if isinstance(value, np.ndarray):
        return bool(value.reshape(-1)[0])
    if torch.is_tensor(value):
        return bool(value.reshape(-1)[0].item())
    return bool(value)


def main() -> int:
    args = parse_args()
    print("== Pointer Devices ==")
    devices = list_candidate_pointers(include_touchpad=args.include_touchpad)
    print(describe_devices(devices))

    print("\n== ManiSkill Environment ==")
    env = make_env(env_id=args.env_id, control_mode="pd_ee_delta_pos", obs_mode="state", reward_mode="normalized_dense")
    try:
        obs, _ = env.reset(seed=0)
        agent_keys = ordered_agent_keys(env)
        layout = action_layout_from_space(env.unwrapped.single_action_space, agent_keys)
        print(f"env_id={args.env_id}")
        print(f"agent_keys={agent_keys}")
        print(f"action_dim={layout.total_dim}")
        zero_action = unflatten_action_vector(np.zeros(layout.total_dim, dtype=np.float32), layout)
        for step_idx in range(args.steps):
            obs, reward, terminated, truncated, info = env.step(zero_action)
            success = _success_to_bool(info.get("success", False))
            print(f"step={step_idx} reward={float(np.asarray(reward).reshape(-1)[0]):.4f} success={success}")
            if _success_to_bool(terminated) or _success_to_bool(truncated):
                break
    finally:
        env.close()

    print("\nSmoke test completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
