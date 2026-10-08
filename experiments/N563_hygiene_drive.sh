#!/usr/bin/env bash
# N563 HYGIENE -- default-OFF regression on the ONE knob this iteration doses.
# The rig source is byte-unchanged by run 563, so the only thing that needs proving is that
# AEGIS_REG_EST="mean" (the explicit spelling of the frozen default) reproduces the DEFAULT arm
# field-for-field. Two identical invocations, no --compare, same 20 seeds; the diff is done by
# experiments/N563_summarize.py's companion check below.
set -u
cd "$(dirname "$0")/.." || exit 1
OUT=results/aegis_v2
STACK="AEGIS_REG=depth AEGIS_REG_HALF_M=0.6 AEGIS_REG_N=32 AEGIS_ROW_CENTRE=1 AEGIS_REG_CASTS=0 AEGIS_REG_CN=12 AEGIS_REG_DFACT=2.0"

echo "=== N563 H1: AEGIS_REG_EST=mean spelled explicitly, pose_noise 3.20,640 ==="
timeout 1200 env $STACK AEGIS_REG_EST=mean AEGIS_POSE_NOISE="3.20,640" \
    python3 experiments/kaggle_aegis_sweep.py --seeds 20 --no-upload --path trochoid \
      --suites fixture_A,fixture_B,fixture_R --out "$OUT/N563_H1_explicit_mean.jsonl" \
      > "$OUT/N563_H1_explicit_mean.log" 2>&1
echo "  exit=$? bytes=$(stat -c%s "$OUT/N563_H1_explicit_mean.jsonl" 2>/dev/null)"
pgrep -f "^python3 .*kaggle_aegis_sweep.py" | xargs -r kill

echo "=== N563 H2: AEGIS_REG_EST unset (frozen default), pose_noise 3.20,640 ==="
timeout 1200 env $STACK AEGIS_POSE_NOISE="3.20,640" \
    python3 experiments/kaggle_aegis_sweep.py --seeds 20 --no-upload --path trochoid \
      --suites fixture_A,fixture_B,fixture_R --out "$OUT/N563_H2_default.jsonl" \
      > "$OUT/N563_H2_default.log" 2>&1
echo "  exit=$? bytes=$(stat -c%s "$OUT/N563_H2_default.jsonl" 2>/dev/null)"
pgrep -f "^python3 .*kaggle_aegis_sweep.py" | xargs -r kill
echo "N563 HYGIENE DONE"
