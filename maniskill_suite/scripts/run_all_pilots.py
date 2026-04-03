#!/usr/bin/env python3
"""Batch pilot runner - runs all 17 remaining experiments with reduced budgets."""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

SUITE_ROOT = Path(__file__).resolve().parents[1]
VENV_PY = str(SUITE_ROOT / ".venv" / "bin" / "python")
REPO = SUITE_ROOT / "external" / "ManiSkill"
BASELINES = REPO / "examples" / "baselines"
DEMO_ROOT = Path.home() / ".maniskill" / "demos"
EXPERIMENTS = SUITE_ROOT / "experiments"

def run(cmd, cwd=None, timeout=600):
    """Run command, return (returncode, stdout+stderr)."""
    print(f"  CMD: {cmd[:120]}...")
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    try:
        r = subprocess.run(cmd, shell=True, cwd=cwd, env=env,
                          capture_output=True, text=True, timeout=timeout)
        out = r.stdout + r.stderr
        if r.returncode != 0:
            print(f"  [WARN] returncode={r.returncode}")
            print(out[-500:] if len(out) > 500 else out)
        return r.returncode, out
    except subprocess.TimeoutExpired:
        print(f"  [TIMEOUT] after {timeout}s")
        return -1, "TIMEOUT"

def save_metrics(combo_id, status, wall_sec, notes="", extra=None):
    d = EXPERIMENTS / combo_id
    d.mkdir(parents=True, exist_ok=True)
    m = {"combo_id": combo_id, "status": status, "wall_seconds": wall_sec, "notes": notes}
    if extra:
        m.update(extra)
    (d / "metrics.json").write_text(json.dumps(m, indent=2))
    print(f"  -> {combo_id}: {status} ({wall_sec:.0f}s) {notes}")

def replay_demo(task, source, control_mode, count):
    """Replay raw demo to state format if not already done."""
    raw_dir = DEMO_ROOT / task / source
    processed = raw_dir / f"trajectory.state.{control_mode}.physx_cpu.h5"
    if processed.exists():
        print(f"  [skip] {processed.name} exists")
        return True

    # Find raw trajectory
    raw_path = raw_dir / "trajectory.h5"
    if not raw_path.exists():
        for suffix in [f"none.{control_mode}.physx_cuda", f"none.{control_mode}.physx_cpu"]:
            candidate = raw_dir / f"trajectory.{suffix}.h5"
            if candidate.exists():
                raw_path = candidate
                break

    if not raw_path.exists():
        print(f"  [FAIL] No raw trajectory for {task}/{source}")
        return False

    replay_flag = "--use-env-states" if source == "rl" else "--use-first-env-state"
    cmd = (f'"{VENV_PY}" -m mani_skill.trajectory.replay_trajectory '
           f'--traj-path "{raw_path}" {replay_flag} '
           f'-c {control_mode} -o state --save-traj --count {count} --num-envs 4 -b physx_cpu')
    rc, out = run(cmd, timeout=600)
    return processed.exists()

def run_ppo(task, combo_id):
    print(f"\n{'='*60}\nPPO: {combo_id}\n{'='*60}")
    t0 = time.time()
    cmd = (f'"{VENV_PY}" ppo.py --env_id="{task}" '
           f'--num_envs=8 --num-steps=50 '
           f'--update_epochs=4 --num_minibatches=4 '
           f'--total_timesteps=50000 '
           f'--control-mode="pd_ee_delta_pos" '
           f'--exp-name="{combo_id}" '
           f'--no-capture_video --no-save_model')
    rc, out = run(cmd, cwd=str(BASELINES / "ppo"), timeout=600)
    wall = time.time() - t0
    status = "pilot_done" if rc == 0 else "error"
    save_metrics(combo_id, status, wall, f"PPO 50K steps 8envs rc={rc}")

def run_sac(task, combo_id):
    print(f"\n{'='*60}\nSAC: {combo_id}\n{'='*60}")
    t0 = time.time()
    cmd = (f'"{VENV_PY}" sac.py --env_id="{task}" '
           f'--num_envs=8 --utd=0.5 '
           f'--buffer_size=50000 '
           f'--total_timesteps=10000 '
           f'--eval_freq=5000 '
           f'--control-mode="pd_ee_delta_pos" '
           f'--exp-name="{combo_id}" '
           f'--no-capture_video --no-save_model')
    rc, out = run(cmd, cwd=str(BASELINES / "sac"), timeout=1200)
    wall = time.time() - t0
    status = "pilot_done" if rc == 0 else "error"
    save_metrics(combo_id, status, wall, f"SAC 10K steps 8envs rc={rc}")

def run_bc(task, combo_id, demo_path, control_mode, max_ep_steps):
    print(f"\n{'='*60}\nBC: {combo_id}\n{'='*60}")
    t0 = time.time()
    cmd = (f'"{VENV_PY}" bc.py --env-id "{task}" '
           f'--demo-path "{demo_path}" '
           f'--control-mode "{control_mode}" '
           f'--sim-backend "cpu" '
           f'--max-episode-steps {max_ep_steps} '
           f'--total-iters 200 '
           f'--num-eval-episodes 10 --num-eval-envs 4 '
           f'--eval_freq 100 '
           f'--exp-name "{combo_id}"')
    rc, out = run(cmd, cwd=str(BASELINES / "bc"), timeout=600)
    wall = time.time() - t0
    status = "pilot_done" if rc == 0 else "error"
    save_metrics(combo_id, status, wall, f"BC 200 iters rc={rc}")

def run_rfcl(task, combo_id, demo_path, num_demos, demo_type="motionplanning"):
    print(f"\n{'='*60}\nRFCL: {combo_id}\n{'='*60}")
    t0 = time.time()
    cmd = (f'XLA_PYTHON_CLIENT_PREALLOCATE=false '
           f'"{VENV_PY}" train.py configs/base_sac_ms3_sample_efficient.yml '
           f'logger.exp_name={combo_id} seed=42 '
           f'train.num_demos={num_demos} train.steps=10000 '
           f'env.env_id={task} '
           f'train.dataset_path="{demo_path}" '
           f'demo_type="{demo_type}" config_type="sample_efficient"')
    rc, out = run(cmd, cwd=str(BASELINES / "rfcl"), timeout=600)
    wall = time.time() - t0
    status = "pilot_done" if rc == 0 else "error"
    save_metrics(combo_id, status, wall, f"RFCL 10K steps {num_demos} demos rc={rc}")

def run_rlpd(task, combo_id, demo_path, demo_type, num_demos):
    print(f"\n{'='*60}\nRLPD: {combo_id}\n{'='*60}")
    t0 = time.time()
    cmd = (f'XLA_PYTHON_CLIENT_PREALLOCATE=false '
           f'"{VENV_PY}" train_ms3.py configs/base_rlpd_ms3_sample_efficient.yml '
           f'logger.exp_name="{combo_id}" seed=42 '
           f'train.num_demos={num_demos} train.steps=2000 '
           f'env.env_id={task} '
           f'train.dataset_path="{demo_path}" '
           f'demo_type="{demo_type}" config_type="sample_efficient"')
    rc, out = run(cmd, cwd=str(BASELINES / "rlpd"), timeout=2400)
    wall = time.time() - t0
    status = "pilot_done" if rc == 0 else "error"
    save_metrics(combo_id, status, wall, f"RLPD 2K steps {num_demos} demos rc={rc}")

def run_act(task, combo_id, demo_path, control_mode, max_ep_steps, demo_type):
    print(f"\n{'='*60}\nACT: {combo_id}\n{'='*60}")
    t0 = time.time()
    cmd = (f'"{VENV_PY}" train.py --env-id "{task}" '
           f'--demo-path "{demo_path}" '
           f'--control-mode "{control_mode}" '
           f'--sim-backend "physx_cpu" '
           f'--max_episode_steps {max_ep_steps} '
           f'--total_iters 200 '
           f'--num-eval-episodes 10 --num-eval-envs 4 '
           f'--exp-name "{combo_id}" --demo_type {demo_type}')
    rc, out = run(cmd, cwd=str(BASELINES / "act"), timeout=600)
    wall = time.time() - t0
    status = "pilot_done" if rc == 0 else "error"
    save_metrics(combo_id, status, wall, f"ACT 200 iters rc={rc}")

def run_diffpol(task, combo_id, demo_path, control_mode, max_ep_steps, demo_type):
    print(f"\n{'='*60}\nDIFFPOL: {combo_id}\n{'='*60}")
    t0 = time.time()
    cmd = (f'"{VENV_PY}" train.py --env-id "{task}" '
           f'--demo-path "{demo_path}" '
           f'--control-mode "{control_mode}" '
           f'--sim-backend "physx_cpu" '
           f'--max_episode_steps {max_ep_steps} '
           f'--total_iters 200 '
           f'--num-eval-episodes 10 --num-eval-envs 4 '
           f'--exp-name "{combo_id}" --demo_type={demo_type}')
    rc, out = run(cmd, cwd=str(BASELINES / "diffusion_policy"), timeout=600)
    wall = time.time() - t0
    status = "pilot_done" if rc == 0 else "error"
    save_metrics(combo_id, status, wall, f"DiffPol 200 iters rc={rc}")

def already_done(combo_id):
    m = EXPERIMENTS / combo_id / "metrics.json"
    if m.exists():
        d = json.loads(m.read_text())
        if d.get("status") in ("pilot_done", "done"):
            print(f"  [skip] {combo_id} already done")
            return True
    return False

def main():
    total_start = time.time()

    # ============================================================
    # STEP 0: Replay demos for PushCube/StackCube
    # ============================================================
    print("\n" + "="*60 + "\nSTEP 0: REPLAY DEMOS\n" + "="*60)

    replays = [
        ("PushCube-v1", "motionplanning", "pd_ee_delta_pos", 3),
        ("PushCube-v1", "motionplanning", "pd_joint_delta_pos", 5),
        ("PushCube-v1", "rl", "pd_joint_delta_pos", 100),
        ("StackCube-v1", "motionplanning", "pd_ee_delta_pos", 3),
        ("StackCube-v1", "motionplanning", "pd_joint_delta_pos", 5),
    ]
    for task, source, cm, count in replays:
        print(f"\n[Replay] {task}/{source} -> {cm} (count={count})")
        replay_demo(task, source, cm, count)

    # ============================================================
    # STEP 1: PPO (no demos, fastest)
    # ============================================================
    print("\n" + "="*60 + "\nSTEP 1: PPO PILOTS\n" + "="*60)
    for task, combo in [("PickCube-v1", "pickcube_ppo"),
                        ("PushCube-v1", "pushcube_ppo"),
                        ("StackCube-v1", "stackcube_ppo")]:
        if not already_done(combo):
            run_ppo(task, combo)

    # ============================================================
    # STEP 2: SAC (no demos)
    # ============================================================
    print("\n" + "="*60 + "\nSTEP 2: SAC PILOTS\n" + "="*60)
    for task, combo in [("PushCube-v1", "pushcube_sac"),
                        ("StackCube-v1", "stackcube_sac")]:
        if not already_done(combo):
            run_sac(task, combo)

    # ============================================================
    # STEP 3: BC (need replayed demos)
    # ============================================================
    print("\n" + "="*60 + "\nSTEP 3: BC PILOTS\n" + "="*60)
    bc_configs = [
        ("PushCube-v1", "pushcube_bc",
         str(DEMO_ROOT / "PushCube-v1/motionplanning/trajectory.state.pd_ee_delta_pos.physx_cpu.h5"),
         "pd_ee_delta_pos", 100),
        ("StackCube-v1", "stackcube_bc",
         str(DEMO_ROOT / "StackCube-v1/motionplanning/trajectory.state.pd_ee_delta_pos.physx_cpu.h5"),
         "pd_ee_delta_pos", 200),
    ]
    for task, combo, dp, cm, mep in bc_configs:
        if not already_done(combo):
            run_bc(task, combo, dp, cm, mep)

    # ============================================================
    # STEP 4: RFCL
    # ============================================================
    print("\n" + "="*60 + "\nSTEP 4: RFCL PILOTS\n" + "="*60)
    rfcl_configs = [
        ("PushCube-v1", "pushcube_rfcl",
         str(DEMO_ROOT / "PushCube-v1/motionplanning/trajectory.state.pd_joint_delta_pos.physx_cpu.h5"), 5),
        ("StackCube-v1", "stackcube_rfcl",
         str(DEMO_ROOT / "StackCube-v1/motionplanning/trajectory.state.pd_joint_delta_pos.physx_cpu.h5"), 5),
    ]
    for task, combo, dp, nd in rfcl_configs:
        if not already_done(combo):
            run_rfcl(task, combo, dp, nd)

    # ============================================================
    # STEP 5: RLPD
    # ============================================================
    print("\n" + "="*60 + "\nSTEP 5: RLPD PILOTS\n" + "="*60)
    rlpd_configs = [
        ("PushCube-v1", "pushcube_rlpd",
         str(DEMO_ROOT / "PushCube-v1/rl/trajectory.state.pd_joint_delta_pos.physx_cpu.h5"),
         "rl", 100),
        ("StackCube-v1", "stackcube_rlpd",
         str(DEMO_ROOT / "StackCube-v1/motionplanning/trajectory.state.pd_joint_delta_pos.physx_cpu.h5"),
         "motionplanning", 100),
    ]
    for task, combo, dp, dt, nd in rlpd_configs:
        if not already_done(combo):
            run_rlpd(task, combo, dp, dt, nd)

    # ============================================================
    # STEP 6: ACT
    # ============================================================
    print("\n" + "="*60 + "\nSTEP 6: ACT PILOTS\n" + "="*60)
    act_configs = [
        ("PickCube-v1", "pickcube_act",
         str(DEMO_ROOT / "PickCube-v1/teleop/trajectory.state.pd_ee_delta_pos.physx_cpu.h5"),
         "pd_ee_delta_pos", 100, "teleop"),
        ("PushCube-v1", "pushcube_act",
         str(DEMO_ROOT / "PushCube-v1/motionplanning/trajectory.state.pd_ee_delta_pos.physx_cpu.h5"),
         "pd_ee_delta_pos", 100, "motionplanning"),
        ("StackCube-v1", "stackcube_act",
         str(DEMO_ROOT / "StackCube-v1/motionplanning/trajectory.state.pd_ee_delta_pos.physx_cpu.h5"),
         "pd_ee_delta_pos", 200, "motionplanning"),
    ]
    for task, combo, dp, cm, mep, dt in act_configs:
        if not already_done(combo):
            run_act(task, combo, dp, cm, mep, dt)

    # ============================================================
    # STEP 7: DIFFUSION POLICY
    # ============================================================
    print("\n" + "="*60 + "\nSTEP 7: DIFFUSION POLICY PILOTS\n" + "="*60)
    dp_configs = [
        ("PickCube-v1", "pickcube_diffusion_policy",
         str(DEMO_ROOT / "PickCube-v1/teleop/trajectory.state.pd_ee_delta_pos.physx_cpu.h5"),
         "pd_ee_delta_pos", 100, "teleop"),
        ("PushCube-v1", "pushcube_diffusion_policy",
         str(DEMO_ROOT / "PushCube-v1/motionplanning/trajectory.state.pd_ee_delta_pos.physx_cpu.h5"),
         "pd_ee_delta_pos", 100, "motionplanning"),
        ("StackCube-v1", "stackcube_diffusion_policy",
         str(DEMO_ROOT / "StackCube-v1/motionplanning/trajectory.state.pd_ee_delta_pos.physx_cpu.h5"),
         "pd_ee_delta_pos", 200, "motionplanning"),
    ]
    for task, combo, dp, cm, mep, dt in dp_configs:
        if not already_done(combo):
            run_diffpol(task, combo, dp, cm, mep, dt)

    # ============================================================
    # SUMMARY
    # ============================================================
    total_wall = time.time() - total_start
    print(f"\n{'='*60}\nALL PILOTS COMPLETE ({total_wall:.0f}s total)\n{'='*60}")
    metrics_files = sorted(EXPERIMENTS.glob("*/metrics.json"))
    for mf in metrics_files:
        d = json.loads(mf.read_text())
        cid = d.get('combo_id', mf.parent.name)
        st = d.get('status', 'unknown')
        ws = d.get('wall_seconds', 0)
        notes = d.get('notes', '')
        print(f"  {cid:30s} {st:12s} {ws:6.0f}s  {notes}")

if __name__ == "__main__":
    main()
