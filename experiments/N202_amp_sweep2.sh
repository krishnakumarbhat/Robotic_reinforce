#!/usr/bin/env bash
set -u
export AEGIS_REG=depth AEGIS_REG_HALF_M=0.6 AEGIS_REG_CASTS=0 AEGIS_REG_DFACT=2.0 AEGIS_REG_CN=16 AEGIS_REG_N=32
export AEGIS_BASE_SEED=400000
AEGIS_POSE_NOISE="0,56" AEGIS_TROCH_AMP_M=0.030 timeout 1200 python3 experiments/kaggle_aegis_sweep.py \
  --seeds 100 --no-upload --path trochoid --compare trochoid \
  --compare-env AEGIS_TROCH_AMP_M=0.015 --suites fixture_A,fixture_B,fixture_R \
  --out "results/aegis_v2/N202_r312_s100_amp0p030.jsonl" > "experiments/N202_r312_s100_amp0p030.log" 2>&1
echo "amp=0.030 rc=$?"
pgrep -f "^python3 .*kaggle_aegis_sweep.py" | xargs -r kill
AEGIS_POSE_NOISE="0,56" AEGIS_TROCH_AMP_M=0.015 timeout 1200 python3 experiments/kaggle_aegis_sweep.py \
  --seeds 20 --no-upload --path trochoid --compare trochoid \
  --compare-env AEGIS_TROCH_AMP_M=0.015 --suites fixture_A,fixture_B,fixture_R \
  --out "results/aegis_v2/N202_CONTROL_s20_amp0p015.jsonl" > "experiments/N202_CONTROL_s20_amp0p015.log" 2>&1
echo "control rc=$?"
pgrep -f "^python3 .*kaggle_aegis_sweep.py" | xargs -r kill
