#!/usr/bin/env bash
# N218 (run 420) RESUME -- the interrupted sweep left 10/12 arms written; this finishes it.
#
# GAP A (Y2' never ran). Every `N218_r420_fixu030_*` header records `pose_noise_cfg 0,0` and
# `compare_env` = AMP+W only; `POSE_FIX_U` is 0.0 in both arm knob blocks, so the held
# +0.030 m u-offset of the pre-registration (equations.md ROW N218, Y2') never reached the rig.
# Those 3 arms are therefore a 100-seed PAIRED REPLICATION of run 419's zero-noise amplitude
# ladder (they are kept as evidence, not overwritten) and the held-offset arms are re-issued here
# with `AEGIS_POSE_FIX_U=0.030` inside `--compare-env`, so arm `b` alone carries the offset and is
# paired against the un-offset frozen champion (arm `a`) on the SAME seeds.
#
# GAP B (Y4 truncated). `N218_r420_n0306_lam060/070.jsonl` stop mid-sweep (756 KiB, no summary /
# compare record) because the C1 assertion raises on nearly every round-face episode; re-issued.
#
# NO rig byte changed. `_coverage_cont` VERBATIM (physics contacts at return time). Same seeds as
# the frozen arms (G4 paired). gate_mode post-hoc. Nothing downloaded.
set -u
cd /media/pope/projecteo/github_proj/a_resume/Robotic_reinforce

run () {  # run <tag> <noise> <amp> <w> [extra compare-env]
  local tag="$1" noise="$2" amp="$3" w="$4" extra="${5:-}"
  local ce="AEGIS_TROCH_AMP_M=${amp},AEGIS_TROCH_W=${w}"
  [ -n "$extra" ] && ce="${ce},${extra}"
  AEGIS_POSE_NOISE="$noise" timeout 1200 python3 experiments/kaggle_aegis_sweep.py \
    --seeds 100 --no-upload --path trochoid --compare trochoid \
    --compare-env "$ce" \
    --suites fixture_A,fixture_B,fixture_R \
    --out "results/aegis_v2/N218_r420_${tag}.jsonl" \
    > "results/aegis_v2/N218_r420_${tag}.log" 2>&1
  echo "done ${tag} rc=$?"
}

# --- GAP A: Y2' held u-offset +0.030 m, ZERO noise, lambda = 0.5 fixed (pure plan-frame error) ---
run fixu030o_A005 0,0 0.005 100.0 AEGIS_POSE_FIX_U=0.030 &
run fixu030o_A023 0,0 0.0225 22.222222222222221 AEGIS_POSE_FIX_U=0.030 &
run fixu030o_A030 0,0 0.030 16.666666666666668 AEGIS_POSE_FIX_U=0.030 &
# --- GAP B: Y4 the C1 wall must be pose-INDEPENDENT (harness counts identical to run 419) ---
run n0306_lam060 0.03,6 0.015 40.0 &
run n0306_lam070 0.03,6 0.015 46.666666666666664 &
wait
pgrep -f "^python3 .*kaggle_aegis_sweep.py" | xargs -r kill
echo ALL_DONE