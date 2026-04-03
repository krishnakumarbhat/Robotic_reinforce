#!/usr/bin/env bash
# ---- Batch Pilot Runner for ManiSkill Suite ----
# Runs all 17 remaining experiments with reduced pilot budgets.
# Already completed: pickcube_bc, pickcube_sac, pickcube_rfcl, pickcube_rlpd
set -euo pipefail

SUITE_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VENV_PY="$SUITE_ROOT/.venv/bin/python"
PY="env -u PYTHONPATH $VENV_PY"
REPO="$SUITE_ROOT/external/ManiSkill"
BASELINES="$REPO/examples/baselines"
DEMO_ROOT="$HOME/.maniskill/demos"
EXPERIMENTS="$SUITE_ROOT/experiments"

log() { echo ""; echo "======== $1 ========"; echo ""; }
ts() { date +%s; }

save_metrics() {
    local combo_id="$1" status="$2" wall_sec="$3" notes="${4:-}"
    local dir="$EXPERIMENTS/$combo_id"
    mkdir -p "$dir"
    cat > "$dir/metrics.json" <<EOF
{
  "combo_id": "$combo_id",
  "status": "$status",
  "wall_seconds": $wall_sec,
  "notes": "$notes"
}
EOF
    echo "  -> Saved $dir/metrics.json (status=$status, ${wall_sec}s)"
}

# ============================================================
# STEP 0: Ensure demos are replayed to state format
# ============================================================
replay_demo() {
    local task="$1" source="$2" control_mode="$3" count="$4"
    local raw_dir="$DEMO_ROOT/$task/$source"
    local processed="$raw_dir/trajectory.state.${control_mode}.physx_cpu.h5"
    
    if [ -f "$processed" ]; then
        echo "  [skip] $processed already exists"
        return 0
    fi

    # Find raw trajectory
    local raw_path="$raw_dir/trajectory.h5"
    if [ ! -f "$raw_path" ]; then
        # Try RL naming convention
        for suffix in "none.${control_mode}.physx_cuda" "none.${control_mode}.physx_cpu"; do
            local candidate="$raw_dir/trajectory.${suffix}.h5"
            if [ -f "$candidate" ]; then
                raw_path="$candidate"
                break
            fi
        done
    fi

    if [ ! -f "$raw_path" ]; then
        echo "  [WARN] No raw trajectory found for $task/$source"
        return 1
    fi

    local replay_flag="--use-first-env-state"
    if [ "$source" = "rl" ]; then
        replay_flag="--use-env-states"
    fi

    echo "  Replaying $raw_path -> state format..."
    $PY -m mani_skill.trajectory.replay_trajectory \
        --traj-path "$raw_path" $replay_flag \
        -c "$control_mode" -o state --save-traj \
        --count "$count" --num-envs 4 -b physx_cpu 2>&1 | tail -5
}

# ============================================================
# STEP 1: REPLAY ALL NEEDED DEMOS
# ============================================================
log "REPLAYING DEMOS"

# PushCube motionplanning -> pd_ee_delta_pos (for BC/ACT/DiffPol)
echo "[PushCube motionplanning -> pd_ee_delta_pos]"
replay_demo PushCube-v1 motionplanning pd_ee_delta_pos 3

# PushCube motionplanning -> pd_joint_delta_pos (for RFCL)
echo "[PushCube motionplanning -> pd_joint_delta_pos]"
replay_demo PushCube-v1 motionplanning pd_joint_delta_pos 5

# PushCube rl -> pd_joint_delta_pos (for RLPD)
echo "[PushCube rl -> pd_joint_delta_pos]"
replay_demo PushCube-v1 rl pd_joint_delta_pos 100

# StackCube motionplanning -> pd_ee_delta_pos (for BC/ACT/DiffPol)
echo "[StackCube motionplanning -> pd_ee_delta_pos]"
replay_demo StackCube-v1 motionplanning pd_ee_delta_pos 3

# StackCube motionplanning -> pd_joint_delta_pos (for RFCL)
echo "[StackCube motionplanning -> pd_joint_delta_pos]"
replay_demo StackCube-v1 motionplanning pd_joint_delta_pos 5

# StackCube motionplanning -> pd_joint_delta_pos (for RLPD - uses motionplanning source)
echo "[StackCube motionplanning -> pd_joint_delta_pos (RLPD)]"
# Already done above

log "DEMO REPLAY COMPLETE"

# ============================================================
# STEP 2: PPO PILOTS (no demos needed)
# ============================================================
run_ppo() {
    local task="$1" combo_id="$2"
    log "PPO: $combo_id"
    local t0; t0=$(ts)
    (
        cd "$BASELINES/ppo"
        $PY ppo.py --env_id="$task" \
            --num_envs=8 --num-steps=50 \
            --update_epochs=4 --num_minibatches=4 \
            --total_timesteps=50000 \
            --control-mode="pd_ee_delta_pos" \
            --exp-name="$combo_id" \
            --capture_video=False \
            --save_model=False 2>&1 | tail -20
    ) || true
    local t1; t1=$(ts)
    save_metrics "$combo_id" "pilot_done" $((t1-t0)) "PPO pilot 50K steps, 8 envs"
}

run_ppo "PickCube-v1" "pickcube_ppo"
run_ppo "PushCube-v1" "pushcube_ppo"
run_ppo "StackCube-v1" "stackcube_ppo"

# ============================================================
# STEP 3: SAC PILOTS (PushCube, StackCube; PickCube done)
# ============================================================
run_sac() {
    local task="$1" combo_id="$2"
    log "SAC: $combo_id"
    local t0; t0=$(ts)
    (
        cd "$BASELINES/sac"
        $PY sac.py --env_id="$task" \
            --num_envs=8 --utd=0.5 \
            --buffer_size=50000 \
            --total_timesteps=10000 \
            --eval_freq=5000 \
            --control-mode="pd_ee_delta_pos" \
            --exp-name="$combo_id" \
            --capture_video=False \
            --save_model=False 2>&1 | tail -20
    ) || true
    local t1; t1=$(ts)
    save_metrics "$combo_id" "pilot_done" $((t1-t0)) "SAC pilot 10K steps, 8 envs"
}

run_sac "PushCube-v1" "pushcube_sac"
run_sac "StackCube-v1" "stackcube_sac"

# ============================================================
# STEP 4: BC PILOTS (PushCube, StackCube; PickCube done)
# ============================================================
run_bc() {
    local task="$1" combo_id="$2" demo_path="$3" control_mode="$4" max_ep_steps="$5"
    log "BC: $combo_id"
    local t0; t0=$(ts)
    (
        cd "$BASELINES/bc"
        $PY bc.py --env-id "$task" \
            --demo-path "$demo_path" \
            --control-mode "$control_mode" \
            --sim-backend "cpu" \
            --max-episode-steps "$max_ep_steps" \
            --total-iters 200 \
            --num-eval-episodes 10 \
            --num-eval-envs 4 \
            --eval_freq 100 \
            --exp-name "$combo_id" 2>&1 | tail -20
    ) || true
    local t1; t1=$(ts)
    save_metrics "$combo_id" "pilot_done" $((t1-t0)) "BC pilot 200 iters"
}

run_bc "PushCube-v1" "pushcube_bc" \
    "$DEMO_ROOT/PushCube-v1/motionplanning/trajectory.state.pd_ee_delta_pos.physx_cpu.h5" \
    "pd_ee_delta_pos" 100

run_bc "StackCube-v1" "stackcube_bc" \
    "$DEMO_ROOT/StackCube-v1/motionplanning/trajectory.state.pd_ee_delta_pos.physx_cpu.h5" \
    "pd_ee_delta_pos" 200

# ============================================================
# STEP 5: RFCL PILOTS (PushCube, StackCube; PickCube done)
# ============================================================
run_rfcl() {
    local task="$1" combo_id="$2" demo_path="$3" num_demos="$4"
    log "RFCL: $combo_id"
    local t0; t0=$(ts)
    (
        cd "$BASELINES/rfcl"
        XLA_PYTHON_CLIENT_PREALLOCATE=false \
        $PY train.py configs/base_sac_ms3_sample_efficient.yml \
            logger.exp_name="$combo_id" seed=42 \
            train.num_demos="$num_demos" \
            train.steps=10000 \
            env.env_id="$task" \
            train.dataset_path="$demo_path" \
            demo_type="motionplanning" \
            config_type="sample_efficient" 2>&1 | tail -20
    ) || true
    local t1; t1=$(ts)
    save_metrics "$combo_id" "pilot_done" $((t1-t0)) "RFCL pilot 10K steps, $num_demos demos"
}

run_rfcl "PushCube-v1" "pushcube_rfcl" \
    "$DEMO_ROOT/PushCube-v1/motionplanning/trajectory.state.pd_joint_delta_pos.physx_cpu.h5" 5

run_rfcl "StackCube-v1" "stackcube_rfcl" \
    "$DEMO_ROOT/StackCube-v1/motionplanning/trajectory.state.pd_joint_delta_pos.physx_cpu.h5" 5

# ============================================================
# STEP 6: RLPD PILOTS (PushCube, StackCube; PickCube done)
# ============================================================
run_rlpd() {
    local task="$1" combo_id="$2" demo_path="$3" demo_type="$4" num_demos="$5"
    log "RLPD: $combo_id"
    local t0; t0=$(ts)
    (
        cd "$BASELINES/rlpd"
        XLA_PYTHON_CLIENT_PREALLOCATE=false \
        $PY train_ms3.py configs/base_rlpd_ms3_sample_efficient.yml \
            logger.exp_name="$combo_id" seed=42 \
            train.num_demos="$num_demos" \
            train.steps=2000 \
            env.env_id="$task" \
            train.dataset_path="$demo_path" \
            demo_type="$demo_type" \
            config_type="sample_efficient" 2>&1 | tail -20
    ) || true
    local t1; t1=$(ts)
    save_metrics "$combo_id" "pilot_done" $((t1-t0)) "RLPD pilot 2K steps, $num_demos demos"
}

run_rlpd "PushCube-v1" "pushcube_rlpd" \
    "$DEMO_ROOT/PushCube-v1/rl/trajectory.state.pd_joint_delta_pos.physx_cpu.h5" \
    "rl" 100

run_rlpd "StackCube-v1" "stackcube_rlpd" \
    "$DEMO_ROOT/StackCube-v1/motionplanning/trajectory.state.pd_joint_delta_pos.physx_cpu.h5" \
    "motionplanning" 100

# ============================================================
# STEP 7: ACT PILOTS (all 3 tasks)
# ============================================================
run_act() {
    local task="$1" combo_id="$2" demo_path="$3" control_mode="$4" max_ep_steps="$5" demo_type="$6"
    log "ACT: $combo_id"
    local t0; t0=$(ts)
    (
        cd "$BASELINES/act"
        $PY train.py --env-id "$task" \
            --demo-path "$demo_path" \
            --control-mode "$control_mode" \
            --sim-backend "physx_cpu" \
            --max_episode_steps "$max_ep_steps" \
            --total_iters 200 \
            --num-eval-episodes 10 \
            --num-eval-envs 4 \
            --exp-name "$combo_id" \
            --demo_type "$demo_type" 2>&1 | tail -20
    ) || true
    local t1; t1=$(ts)
    save_metrics "$combo_id" "pilot_done" $((t1-t0)) "ACT pilot 200 iters"
}

run_act "PickCube-v1" "pickcube_act" \
    "$DEMO_ROOT/PickCube-v1/teleop/trajectory.state.pd_ee_delta_pos.physx_cpu.h5" \
    "pd_ee_delta_pos" 100 "teleop"

run_act "PushCube-v1" "pushcube_act" \
    "$DEMO_ROOT/PushCube-v1/motionplanning/trajectory.state.pd_ee_delta_pos.physx_cpu.h5" \
    "pd_ee_delta_pos" 100 "motionplanning"

run_act "StackCube-v1" "stackcube_act" \
    "$DEMO_ROOT/StackCube-v1/motionplanning/trajectory.state.pd_ee_delta_pos.physx_cpu.h5" \
    "pd_ee_delta_pos" 200 "motionplanning"

# ============================================================
# STEP 8: DIFFUSION POLICY PILOTS (all 3 tasks)
# ============================================================
run_diffpol() {
    local task="$1" combo_id="$2" demo_path="$3" control_mode="$4" max_ep_steps="$5" demo_type="$6"
    log "DIFFUSION POLICY: $combo_id"
    local t0; t0=$(ts)
    (
        cd "$BASELINES/diffusion_policy"
        $PY train.py --env-id "$task" \
            --demo-path "$demo_path" \
            --control-mode "$control_mode" \
            --sim-backend "physx_cpu" \
            --max_episode_steps "$max_ep_steps" \
            --total_iters 200 \
            --num-eval-episodes 10 \
            --num-eval-envs 4 \
            --exp-name "$combo_id" \
            --demo_type="$demo_type" 2>&1 | tail -20
    ) || true
    local t1; t1=$(ts)
    save_metrics "$combo_id" "pilot_done" $((t1-t0)) "DiffPol pilot 200 iters"
}

run_diffpol "PickCube-v1" "pickcube_diffusion_policy" \
    "$DEMO_ROOT/PickCube-v1/teleop/trajectory.state.pd_ee_delta_pos.physx_cpu.h5" \
    "pd_ee_delta_pos" 100 "teleop"

run_diffpol "PushCube-v1" "pushcube_diffusion_policy" \
    "$DEMO_ROOT/PushCube-v1/motionplanning/trajectory.state.pd_ee_delta_pos.physx_cpu.h5" \
    "pd_ee_delta_pos" 100 "motionplanning"

run_diffpol "StackCube-v1" "stackcube_diffusion_policy" \
    "$DEMO_ROOT/StackCube-v1/motionplanning/trajectory.state.pd_ee_delta_pos.physx_cpu.h5" \
    "pd_ee_delta_pos" 200 "motionplanning"

# ============================================================
# SUMMARY
# ============================================================
log "ALL PILOTS COMPLETE"
echo "Experiments completed:"
find "$EXPERIMENTS" -name "metrics.json" -exec echo "  {}" \;
echo ""
echo "Run 'python scripts/summarize_results.py' to update experiment_results.md"
