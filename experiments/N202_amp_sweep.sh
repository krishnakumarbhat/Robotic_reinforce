#!/usr/bin/env bash
# N202 -- is the tool offset in N201's residual-yaw law a GEOMETRIC SUPPLY term
# (r_eff + loop amplitude) or a DYNAMIC one (mass/slip)?  Env-only: the loop radius
# AEGIS_TROCH_AMP_M is an existing knob among 51.  Paired in-rig against the champion
# amplitude 0.015 on the SAME seeds (G4).  Fresh seed base 400000 (verified unused).
set -u
export AEGIS_REG=depth AEGIS_REG_HALF_M=0.6 AEGIS_REG_CASTS=0 AEGIS_REG_DFACT=2.0 AEGIS_REG_CN=16 AEGIS_REG_N=32
export AEGIS_BASE_SEED=400000
for amp in 0.005 0.015 0.030; do
  tag=$(echo "$amp" | tr '.' 'p')
  AEGIS_POSE_NOISE="0,56" AEGIS_TROCH_AMP_M="$amp" timeout 1200 python3 experiments/kaggle_aegis_sweep.py \
    --seeds 100 --no-upload --path trochoid --compare trochoid \
    --compare-env AEGIS_TROCH_AMP_M=0.015 --suites fixture_A,fixture_B,fixture_R \
    --out "results/aegis_v2/N202_r312_s100_amp${tag}.jsonl" > "experiments/N202_r312_s100_amp${tag}.log" 2>&1
  echo "amp=$amp rc=$?"
  pgrep -f "^python3 .*kaggle_aegis_sweep.py" | xargs -r kill
done
# harness control: champion amplitude against itself must be a bit-identical no-op
AEGIS_POSE_NOISE="0,56" AEGIS_TROCH_AMP_M=0.015 timeout 1200 python3 experiments/kaggle_aegis_sweep.py \
  --seeds 20 --no-upload --path trochoid --compare trochoid \
  --compare-env AEGIS_TROCH_AMP_M=0.015 --suites fixture_A,fixture_B,fixture_R \
  --out "results/aegis_v2/N202_CONTROL_s20_amp0p015.jsonl" > "experiments/N202_CONTROL_s20_amp0p015.log" 2>&1
echo "control rc=$?"
pgrep -f "^python3 .*kaggle_aegis_sweep.py" | xargs -r kill
