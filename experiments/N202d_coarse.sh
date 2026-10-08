#!/usr/bin/env bash
# N202d -- the ONE arm the C1 law nominates.  The plan-level scan (N202_kinematic.py,
# the rig's own scrub_uv + max_turn_deg, no physics) shows max_turn_deg at the cusp-safe
# k = w*A = 0.5 falls monotonically with AMPLITUDE (33.4 deg at A=0.060/w=8.33, 45.4 at
# the champion 0.015/33.33, 49.1 at 0.0075/66.67): the phase step per base point is
# w*BASE_DS_M, so a LARGER amplitude at the same k is a FINER offset mesh and a lower
# max heading change.  Coverage is measured FLAT in A over [0.005, 0.015] (bit-identical,
# 0/200 seeds differing), so the amplitude is a purely kinematic knob and this is the
# only arm left that the law predicts should move anything.  Cusp margin is unchanged
# (k = 0.5, min|T| = 0.5), the C1 assert passes with 26 deg more headroom, and the plan
# is 11% shorter.  Paired in-rig against the champion amplitude on the SAME seeds (G4).
set -u
export AEGIS_REG=depth AEGIS_REG_HALF_M=0.6 AEGIS_REG_CASTS=0 AEGIS_REG_DFACT=2.0
export AEGIS_REG_CN=16 AEGIS_REG_N=32
export AEGIS_BASE_SEED=400000 AEGIS_POSE_NOISE="0,56"
AEGIS_TROCH_AMP_M=0.060 AEGIS_TROCH_W=8.333333333333334 timeout 1200 python3 \
  experiments/kaggle_aegis_sweep.py --seeds 100 --no-upload --path trochoid \
  --compare trochoid --compare-env AEGIS_TROCH_AMP_M=0.015 \
  --suites fixture_A,fixture_B,fixture_R \
  --out results/aegis_v2/N202d_r312_s100_amp0p060_w8p33.jsonl \
  > experiments/N202d_r312_s100_amp0p060_w8p33.log 2>&1
echo "candidate rc=$?"
pgrep -f "^python3 .*kaggle_aegis_sweep.py" | xargs -r kill
echo DONE
