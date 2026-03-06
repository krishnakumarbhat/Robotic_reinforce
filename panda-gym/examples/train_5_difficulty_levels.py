import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import gymnasium as gym
import numpy as np
import panda_gym
from stable_baselines3 import DDPG, PPO, SAC, TD3, HerReplayBuffer
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.her.goal_selection_strategy import GoalSelectionStrategy

try:
    from sb3_contrib import TQC
except ImportError:
    TQC = None


@dataclass(frozen=True)
class AlgoSpec:
    name: str
    use_her: bool = False
    note: str = ""

    @property
    def label(self) -> str:
        return f"{self.name} + HER" if self.use_her else self.name


LEVEL_CONFIG: List[Dict[str, Any]] = [
    {
        "level": 1,
        "task": "Reach",
        "env_id": "PandaReach-v3",
        "env_kwargs": {},
        "algorithms": [AlgoSpec("PPO"), AlgoSpec("DDPG"), AlgoSpec("SAC")],
    },
    {
        "level": 2,
        "task": "Push",
        "env_id": "PandaPush-v3",
        "env_kwargs": {"reward_type": "dense"},
        "algorithms": [AlgoSpec("TD3"), AlgoSpec("SAC")],
    },
    {
        "level": 3,
        "task": "Slide",
        "env_id": "PandaSlide-v3",
        "env_kwargs": {},
        "algorithms": [AlgoSpec("TD3", use_her=True), AlgoSpec("SAC", use_her=True)],
    },
    {
        "level": 4,
        "task": "PickAndPlace",
        "env_id": "PandaPickAndPlace-v3",
        "env_kwargs": {},
        "algorithms": [AlgoSpec("SAC", use_her=True), AlgoSpec("TQC", use_her=True)],
    },
    {
        "level": 5,
        "task": "Stack",
        "env_id": "PandaStack-v3",
        "env_kwargs": {},
        "algorithms": [
            AlgoSpec("TQC", use_her=True),
            AlgoSpec(
                "ImitationLearning",
                note="Requires demonstration data, not trained in this script.",
            ),
        ],
    },
]


def build_model(
    algo: AlgoSpec,
    env,
    n_envs: int,
    seed: int,
):
    if algo.name == "ImitationLearning":
        raise ValueError("Imitation Learning requires demonstration data and a dedicated training pipeline.")

    model_classes = {
        "PPO": PPO,
        "DDPG": DDPG,
        "TD3": TD3,
        "SAC": SAC,
        "TQC": TQC,
    }
    model_class = model_classes.get(algo.name)
    if model_class is None:
        raise ValueError(f"Unsupported algorithm: {algo.name}")
    if algo.name == "TQC" and TQC is None:
        raise ImportError("TQC requested but `sb3_contrib` is not installed.")

    if algo.name == "PPO":
        return model_class(
            policy="MultiInputPolicy",
            env=env,
            verbose=0,
            seed=seed,
        )

    model_kwargs: Dict[str, Any] = {
        "policy": "MultiInputPolicy",
        "env": env,
        "verbose": 0,
        "seed": seed,
        "gamma": 0.95,
        "batch_size": 512,
        "learning_rate": 1e-3,
        "policy_kwargs": {"net_arch": [256, 256, 256]},
        "train_freq": (1, "step"),
        "gradient_steps": n_envs,
    }

    if algo.use_her:
        model_kwargs.update(
            {
                "replay_buffer_class": HerReplayBuffer,
                "replay_buffer_kwargs": {
                    "n_sampled_goal": 4,
                    "goal_selection_strategy": GoalSelectionStrategy.FUTURE,
                },
            }
        )

    return model_class(**model_kwargs)


def evaluate(model, env_id: str, env_kwargs: Dict[str, Any], episodes: int, seed: int) -> Dict[str, float]:
    eval_env = gym.make(env_id, **env_kwargs)
    rewards = []
    successes = []

    for episode_index in range(episodes):
        observation, info = eval_env.reset(seed=seed + episode_index)
        episode_reward = 0.0
        episode_done = False
        last_success = bool(info.get("is_success", False))

        while not episode_done:
            action, _ = model.predict(observation, deterministic=True)
            observation, reward, terminated, truncated, info = eval_env.step(action)
            episode_reward += float(reward)
            last_success = bool(info.get("is_success", False))
            episode_done = terminated or truncated

        rewards.append(episode_reward)
        successes.append(1.0 if last_success else 0.0)

    eval_env.close()
    return {
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "success_rate": float(np.mean(successes)),
    }


def run_combo(
    level_cfg: Dict[str, Any],
    algo: AlgoSpec,
    timesteps: int,
    eval_episodes: int,
    n_envs: int,
    output_dir: Path,
    seed: int,
) -> Dict[str, Any]:
    result = {
        "level": level_cfg["level"],
        "task": level_cfg["task"],
        "env_id": level_cfg["env_id"],
        "algorithm": algo.label,
        "timesteps": timesteps,
        "status": "ok",
        "train_seconds": 0.0,
        "mean_reward": None,
        "std_reward": None,
        "success_rate": None,
        "note": algo.note,
    }

    if algo.name == "ImitationLearning":
        result["status"] = "skipped"
        return result

    train_env = make_vec_env(
        level_cfg["env_id"],
        n_envs=n_envs,
        env_kwargs=level_cfg["env_kwargs"],
        vec_env_cls=DummyVecEnv,
    )

    try:
        model = build_model(algo=algo, env=train_env, n_envs=n_envs, seed=seed)
        train_start = time.time()
        model.learn(total_timesteps=timesteps)
        result["train_seconds"] = round(time.time() - train_start, 2)

        model_name = (
            f"level{level_cfg['level']}_{level_cfg['task'].lower()}_"
            f"{algo.label.lower().replace(' + ', '_').replace(' ', '_')}"
        )
        model_path = output_dir / model_name
        model.save(str(model_path))

        metrics = evaluate(
            model=model,
            env_id=level_cfg["env_id"],
            env_kwargs=level_cfg["env_kwargs"],
            episodes=eval_episodes,
            seed=seed,
        )
        result.update(metrics)
    except Exception as exc:
        result["status"] = "failed"
        result["note"] = str(exc)
    finally:
        train_env.close()

    return result


def format_value(value: Optional[float], precision: int = 3) -> str:
    if value is None:
        return "-"
    return f"{value:.{precision}f}"


def print_results(results: List[Dict[str, Any]]) -> None:
    print("\n===== 5-Level Difficulty Results =====")
    headers = [
        "Level",
        "Task",
        "Algorithm",
        "Status",
        "Success",
        "MeanReward",
        "StdReward",
        "TrainSec",
        "Note",
    ]

    rows = []
    for item in results:
        rows.append(
            [
                str(item["level"]),
                item["task"],
                item["algorithm"],
                item["status"],
                format_value(item["success_rate"]),
                format_value(item["mean_reward"]),
                format_value(item["std_reward"]),
                format_value(item["train_seconds"], precision=2),
                item["note"] or "",
            ]
        )

    widths = [len(header) for header in headers]
    for row in rows:
        for idx, value in enumerate(row):
            widths[idx] = max(widths[idx], len(value))

    def render_row(values: List[str]) -> str:
        return " | ".join(value.ljust(widths[idx]) for idx, value in enumerate(values))

    print(render_row(headers))
    print("-+-".join("-" * width for width in widths))
    for row in rows:
        print(render_row(row))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and evaluate 5 Panda-Gym difficulty levels.")
    parser.add_argument("--timesteps", type=int, default=30_000, help="Training timesteps per algorithm combo.")
    parser.add_argument("--eval-episodes", type=int, default=10, help="Evaluation episodes per trained model.")
    parser.add_argument("--n-envs", type=int, default=2, help="Vectorized training environments.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--output-dir", type=str, default="trained_models", help="Directory to save models.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Starting 5-level curriculum training...")
    print(
        f"timesteps={args.timesteps}, eval_episodes={args.eval_episodes}, "
        f"n_envs={args.n_envs}, seed={args.seed}"
    )

    results: List[Dict[str, Any]] = []
    for level_cfg in LEVEL_CONFIG:
        print(f"\n--- Level {level_cfg['level']}: {level_cfg['task']} ({level_cfg['env_id']}) ---")
        for algo in level_cfg["algorithms"]:
            print(f"Training/Evaluating: {algo.label}")
            combo_result = run_combo(
                level_cfg=level_cfg,
                algo=algo,
                timesteps=args.timesteps,
                eval_episodes=args.eval_episodes,
                n_envs=args.n_envs,
                output_dir=output_dir,
                seed=args.seed,
            )
            results.append(combo_result)

    print_results(results)


if __name__ == "__main__":
    main()