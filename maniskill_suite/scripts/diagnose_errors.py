#!/usr/bin/env python3
"""Quick diagnosis: run each failing algo with tiny budget, capture full error."""
import subprocess, os, pathlib, sys, time

ROOT = pathlib.Path(__file__).resolve().parents[1]
VENV_PY = str(ROOT / ".venv" / "bin" / "python")
BASELINES = ROOT / "external" / "ManiSkill" / "examples" / "baselines"
DEMO_ROOT = pathlib.Path.home() / ".maniskill" / "demos"

env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}

tests = {
    "PPO": {
        "cwd": BASELINES / "ppo",
        "cmd": f'{VENV_PY} ppo.py --env_id=PickCube-v1 --num_envs=2 --num-steps=10 '
               f'--update_epochs=1 --num_minibatches=1 --total_timesteps=100 '
               f'--control-mode=pd_ee_delta_pos --exp-name=diag_ppo --no-capture_video --no-save_model',
        "timeout": 120,
    },
    "SAC": {
        "cwd": BASELINES / "sac",
        "cmd": f'{VENV_PY} sac.py --env_id=PickCube-v1 --num_envs=2 --utd=0.5 '
               f'--buffer_size=1000 --total_timesteps=100 --eval_freq=5000 '
               f'--control-mode=pd_ee_delta_pos --exp-name=diag_sac --no-capture_video --no-save_model',
        "timeout": 120,
    },
    "BC": {
        "cwd": BASELINES / "bc",
        "cmd": f'{VENV_PY} bc.py --env-id StackCube-v1 '
               f'--demo-path {DEMO_ROOT}/StackCube-v1/motionplanning/trajectory.state.pd_ee_delta_pos.physx_cpu.h5 '
               f'--control-mode pd_ee_delta_pos --sim-backend cpu --max-episode-steps 200 '
               f'--total-iters 2 --num-eval-episodes 2 --num-eval-envs 1 --eval_freq 100 --exp-name diag_bc',
        "timeout": 120,
    },
    "ACT": {
        "cwd": BASELINES / "act",
        "cmd": f'{VENV_PY} train.py --env-id PickCube-v1 '
               f'--demo-path {DEMO_ROOT}/PickCube-v1/teleop/trajectory.state.pd_ee_delta_pos.physx_cpu.h5 '
               f'--control-mode pd_ee_delta_pos --sim-backend physx_cpu --max_episode_steps 100 '
               f'--total_iters 2 --num-eval-episodes 2 --num-eval-envs 1 --exp-name diag_act --demo_type teleop',
        "timeout": 120,
    },
    "DiffPol": {
        "cwd": BASELINES / "diffusion_policy",
        "cmd": f'{VENV_PY} train.py --env-id PickCube-v1 '
               f'--demo-path {DEMO_ROOT}/PickCube-v1/teleop/trajectory.state.pd_ee_delta_pos.physx_cpu.h5 '
               f'--control-mode pd_ee_delta_pos --sim-backend physx_cpu --max_episode_steps 100 '
               f'--total_iters 2 --num-eval-episodes 2 --num-eval-envs 1 --exp-name diag_diffpol --demo_type=teleop',
        "timeout": 120,
    },
    "RFCL": {
        "cwd": BASELINES / "rfcl",
        "cmd": f'XLA_PYTHON_CLIENT_PREALLOCATE=false {VENV_PY} train.py '
               f'configs/base_sac_ms3_sample_efficient.yml '
               f'logger.exp_name=diag_rfcl seed=42 train.num_demos=5 train.steps=100 '
               f'env.env_id=PushCube-v1 '
               f'train.dataset_path={DEMO_ROOT}/PushCube-v1/motionplanning/trajectory.state.pd_joint_delta_pos.physx_cpu.h5 '
               f'demo_type=motionplanning config_type=sample_efficient',
        "timeout": 120,
    },
    "RLPD": {
        "cwd": BASELINES / "rlpd",
        "cmd": f'XLA_PYTHON_CLIENT_PREALLOCATE=false {VENV_PY} train_ms3.py '
               f'configs/base_rlpd_ms3_sample_efficient.yml '
               f'logger.exp_name=diag_rlpd seed=42 train.num_demos=5 train.steps=100 '
               f'env.env_id=PushCube-v1 '
               f'train.dataset_path={DEMO_ROOT}/PushCube-v1/rl/trajectory.state.pd_joint_delta_pos.physx_cpu.h5 '
               f'demo_type=rl config_type=sample_efficient',
        "timeout": 120,
    },
}

for name, cfg in tests.items():
    print(f"\n{'='*60}")
    print(f"TESTING: {name}")
    print(f"{'='*60}")
    print(f"CMD: {cfg['cmd'][:200]}")
    t0 = time.time()
    try:
        r = subprocess.run(cfg['cmd'], shell=True, cwd=str(cfg['cwd']), env=env,
                          capture_output=True, text=True, timeout=cfg['timeout'])
        wall = time.time() - t0
        print(f"RC: {r.returncode}  ({wall:.0f}s)")
        output = r.stdout + r.stderr
        if r.returncode != 0:
            print(f"--- LAST 1500 chars ---")
            print(output[-1500:])
        else:
            print("SUCCESS")
            print(output[-300:])
    except subprocess.TimeoutExpired:
        wall = time.time() - t0
        print(f"TIMEOUT after {wall:.0f}s")
    except Exception as e:
        print(f"EXCEPTION: {e}")
    sys.stdout.flush()

print("\n\nDONE - all diagnostics complete")
