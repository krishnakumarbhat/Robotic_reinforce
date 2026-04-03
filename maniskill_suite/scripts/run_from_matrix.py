#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path


def load_matrix(matrix_path: Path) -> dict:
    return json.loads(matrix_path.read_text())


def task_index(matrix: dict) -> dict:
    return {item["id"]: item for item in matrix["tasks"]}


def build_demo_root() -> str:
    return str(Path(os.environ.get("MANISKILL_DEMO_ROOT", "~/.maniskill/demos")).expanduser())


def build_python_command(suite_root: Path) -> str:
    venv_python = suite_root / ".venv" / "bin" / "python"
    if venv_python.exists():
        return f'env -u PYTHONPATH "{venv_python}"'
    return "python"


def build_raw_traj_path(demo_root: str, task_id: str, replay_source: str, control_mode: str) -> str:
    base_dir = Path(demo_root).expanduser() / task_id / replay_source
    default_path = base_dir / "trajectory.h5"
    if replay_source != "rl" or default_path.exists():
        return str(default_path)

    candidates = [
        base_dir / f"trajectory.none.{control_mode}.physx_cuda.h5",
        base_dir / f"trajectory.none.{control_mode}.physx_cpu.h5",
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return str(default_path)


def build_repo_root(suite_root: Path, explicit_repo: str | None) -> Path:
    if explicit_repo:
        return Path(explicit_repo).expanduser().resolve()
    env_repo = os.environ.get("MANISKILL_REPO")
    if env_repo:
        return Path(env_repo).expanduser().resolve()
    return (suite_root / "external" / "ManiSkill").resolve()


def build_command(combo: dict, task: dict, repo_root: Path, demo_root: str, python_cmd: str) -> str:
    algo = combo["algorithm"]
    task_id = combo["task"]
    control_mode = combo["control_mode"]
    combo_id = combo["id"]

    if algo == "ppo":
        return (
            f'cd "{repo_root / "examples/baselines/ppo"}" && '
            f'{python_cmd} ppo.py --env_id="{task_id}" '
            f'--num_envs={combo["num_envs"]} --num-steps={combo["num_steps"]} '
            f'--update_epochs=8 --num_minibatches=4 --total_timesteps={combo["total_timesteps"]} '
            f'--control-mode="{control_mode}" --exp-name="{combo_id}"'
        )

    if algo == "sac":
        return (
            f'cd "{repo_root / "examples/baselines/sac"}" && '
            f'{python_cmd} sac.py --env_id="{task_id}" '
            f'--num_envs={combo["num_envs"]} --utd={combo["utd"]} '
            f'--buffer_size={combo["buffer_size"]} --total_timesteps={combo["total_timesteps"]} '
            f'--eval_freq=50000 --control-mode="{control_mode}" --exp-name="{combo_id}"'
        )

    replay_source = combo["demo_source"]
    replay_count = combo["replay_count"]
    replay_flag = "--use-env-states" if replay_source == "rl" else "--use-first-env-state"
    raw_path = build_raw_traj_path(demo_root=demo_root, task_id=task_id, replay_source=replay_source, control_mode=control_mode)
    processed_path = f"{demo_root}/{task_id}/{replay_source}/trajectory.state.{control_mode}.physx_cpu.h5"

    if algo == "bc":
        return (
            f'{python_cmd} -m mani_skill.trajectory.replay_trajectory --traj-path "{raw_path}" {replay_flag} '
            f'-c "{control_mode}" -o state --save-traj --count {replay_count} --num-envs 4 -b physx_cpu\n'
            f'cd "{repo_root / "examples/baselines/bc"}" && '
            f'{python_cmd} bc.py --env-id "{task_id}" --demo-path "{processed_path}" '
            f'--control-mode "{control_mode}" --sim-backend "cpu" '
            f'--max-episode-steps {combo["max_episode_steps"]} --total-iters {combo["total_iters"]} '
            f'--exp-name "{combo_id}"'
        )

    if algo == "act":
        return (
            f'{python_cmd} -m mani_skill.trajectory.replay_trajectory --traj-path "{raw_path}" {replay_flag} '
            f'-c "{control_mode}" -o state --save-traj --count {replay_count} --num-envs 4 -b physx_cpu\n'
            f'cd "{repo_root / "examples/baselines/act"}" && '
            f'{python_cmd} train.py --env-id "{task_id}" --demo-path "{processed_path}" '
            f'--control-mode "{control_mode}" --sim-backend "physx_cpu" '
            f'--max_episode_steps {combo["max_episode_steps"]} --total_iters {combo["total_iters"]} '
            f'--exp-name "{combo_id}" --demo_type {replay_source}'
        )

    if algo == "diffusion_policy":
        return (
            f'{python_cmd} -m mani_skill.trajectory.replay_trajectory --traj-path "{raw_path}" {replay_flag} '
            f'-c "{control_mode}" -o state --save-traj --count {replay_count} --num-envs 4 -b physx_cpu\n'
            f'cd "{repo_root / "examples/baselines/diffusion_policy"}" && '
            f'{python_cmd} train.py --env-id "{task_id}" --demo-path "{processed_path}" '
            f'--control-mode "{control_mode}" --sim-backend "physx_cpu" '
            f'--max_episode_steps {combo["max_episode_steps"]} --total_iters {combo["total_iters"]} '
            f'--exp-name "{combo_id}" --demo_type={replay_source}'
        )

    if algo == "rfcl":
        joint_path = f"{demo_root}/{task_id}/{replay_source}/trajectory.state.{control_mode}.physx_cpu.h5"
        return (
            f'{python_cmd} -m mani_skill.trajectory.replay_trajectory --traj-path "{raw_path}" --use-first-env-state '
            f'-c "{control_mode}" -o state --save-traj --count {replay_count} --num-envs 4 -b physx_cpu\n'
            f'cd "{repo_root / "examples/baselines/rfcl"}" && '
            f'XLA_PYTHON_CLIENT_PREALLOCATE=false {python_cmd} train.py configs/base_sac_ms3_sample_efficient.yml '
            f'logger.exp_name={combo_id} seed=42 train.num_demos={replay_count} '
            f'train.steps={combo["train_steps"]} env.env_id={task_id} '
            f'train.dataset_path="{joint_path}" demo_type="{replay_source}" config_type="sample_efficient"'
        )

    if algo == "rlpd":
        joint_path = f"{demo_root}/{task_id}/{replay_source}/trajectory.state.{control_mode}.physx_cpu.h5"
        return (
            f'{python_cmd} -m mani_skill.trajectory.replay_trajectory --traj-path "{raw_path}" {replay_flag} '
            f'-c "{control_mode}" -o state --save-traj --count {replay_count} --num-envs 4 -b physx_cpu\n'
            f'cd "{repo_root / "examples/baselines/rlpd"}" && '
            f'XLA_PYTHON_CLIENT_PREALLOCATE=false {python_cmd} train_ms3.py configs/base_rlpd_ms3_sample_efficient.yml '
            f'logger.exp_name="{combo_id}" seed=42 train.num_demos={replay_count} '
            f'train.steps={combo["train_steps"]} env.env_id={task_id} '
            f'train.dataset_path="{joint_path}" demo_type="{replay_source}" config_type="sample_efficient"'
        )

    raise ValueError(f"Unsupported algorithm: {algo}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Print a ManiSkill baseline command from the experiment matrix.")
    parser.add_argument("--combo", required=True, help="Combination id from experiment_matrix.json")
    parser.add_argument("--matrix", default=None, help="Optional custom matrix path")
    parser.add_argument("--maniskill-repo", default=None, help="Optional override for local ManiSkill clone")
    parser.add_argument("--show-json", action="store_true", help="Print combo metadata before the command")
    args = parser.parse_args()

    suite_root = Path(__file__).resolve().parents[1]
    matrix_path = Path(args.matrix) if args.matrix else suite_root / "experiment_matrix.json"
    matrix = load_matrix(matrix_path)
    tasks = task_index(matrix)
    combos = {item["id"]: item for item in matrix["combinations"]}
    if args.combo not in combos:
        raise SystemExit(f"Unknown combo id: {args.combo}")

    combo = combos[args.combo]
    task = tasks[combo["task"]]
    repo_root = build_repo_root(suite_root, args.maniskill_repo)
    demo_root = build_demo_root()
    python_cmd = build_python_command(suite_root)

    if args.show_json:
        print(json.dumps(combo, indent=2))
        print()

    print(build_command(combo=combo, task=task, repo_root=repo_root, demo_root=demo_root, python_cmd=python_cmd))


if __name__ == "__main__":
    main()