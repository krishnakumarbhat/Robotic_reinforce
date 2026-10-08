#!/usr/bin/env bash
# N202e -- close the LEGAL amplitude window.  The C1 law (experiments/N202_kinematic.py,
# calibrated on three rig-asserted points to 0.05 deg) puts the kinematic cliff at
# k = 0.5923, i.e. A_max = 0.01777 m at the champion rate, so A = 0.0165 (k = 0.55,
# max turn 50.9 deg < the 60 deg assert) is the TIGHTEST LEGAL upper arm and
# A = 0.005 (k = 0.1667, max turn 14.9 deg) the loosest tested.  One knob override only
# (AEGIS_TROCH_AMP_M), so the paired baseline arm restores the champion's rate
# AEGIS_TROCH_W=33.333 and the pairing is valid -- unlike N202d, whose baseline arm
# inherited w = 8.333 from the environment and is therefore NOT a champion pairing.
set -u
export AEGIS_REG=depth AEGIS_REG_HALF_M=0.6 AEGIS_REG_CASTS=0 AEGIS_REG_DFACT=2.0
export AEGIS_REG_CN=16 AEGIS_REG_N=32
export AEGIS_BASE_SEED=400000 AEGIS_POSE_NOISE="0,56"
AEGIS_TROCH_AMP_M=0.0165 timeout 1200 python3 experiments/kaggle_aegis_sweep.py \
  --seeds 100 --no-upload --path trochoid --compare trochoid \
  --compare-env AEGIS_TROCH_AMP_M=0.015 --suites fixture_A,fixture_B,fixture_R \
  --out results/aegis_v2/N202e_r312_s100_amp0p0165.jsonl \
  > experiments/N202e_r312_s100_amp0p0165.log 2>&1
echo "rc=$?"
pgrep -f "^python3 .*kaggle_aegis_sweep.py" | xargs -r kill
echo DONE
