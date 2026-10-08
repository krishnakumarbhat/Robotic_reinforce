#!/usr/bin/env bash
# Serialized re-run of the cells destroyed by the cross-driver anchored-kill interference, plus
# the N563 default-OFF hygiene arm that was killed by the same event.
# SERIALIZED on purpose: two concurrent drivers sharing `pgrep -f "^python3 .*kaggle_aegis_sweep.py"
# | xargs -r kill` destroy each other's cells (an orphaned driver truncated its own W2 at 89 of 120
# episodes and killed this run's H2 arm). One driver at a time from here.
set -u
cd "$(dirname "$0")/.." || exit 1
OUT=results/aegis_v2
STACK="AEGIS_REG=depth AEGIS_REG_HALF_M=0.6 AEGIS_REG_N=128 AEGIS_ROW_CENTRE=1 AEGIS_REG_CASTS=0 AEGIS_REG_CN=12 AEGIS_REG_DFACT=2.0"

echo "=== N563 H2: AEGIS_REG_EST unset (frozen default), pose_noise 3.20,640 [RE-RUN] ==="
timeout 1200 env AEGIS_REG=depth AEGIS_REG_HALF_M=0.6 AEGIS_REG_N=32 AEGIS_ROW_CENTRE=1 \
    AEGIS_REG_CASTS=0 AEGIS_REG_CN=12 AEGIS_REG_DFACT=2.0 AEGIS_POSE_NOISE="3.20,640" \
    python3 experiments/kaggle_aegis_sweep.py --seeds 20 --no-upload --path trochoid \
      --suites fixture_A,fixture_B,fixture_R --out "$OUT/N563_H2_default.jsonl" \
      > "$OUT/N563_H2_default.log" 2>&1
echo "  exit=$? bytes=$(stat -c%s "$OUT/N563_H2_default.jsonl" 2>/dev/null)"
pgrep -f "^python3 .*kaggle_aegis_sweep.py" | xargs -r kill

cell () {  # $1=tag  $2=pose_noise   (N563d wall ladder, REG_N=128, pre-registered by the N563d driver)
  echo "=== N563d $1  REG_N=128  pose_noise=$2 [RE-RUN] ==="
  timeout 1200 env $STACK AEGIS_REG_EST=extent AEGIS_POSE_NOISE="$2" \
      python3 experiments/kaggle_aegis_sweep.py --seeds 20 --no-upload --path trochoid \
        --compare trochoid --compare-env "AEGIS_REG_EST=0.0,AEGIS_REG_N=128.0" \
        --suites fixture_A,fixture_B,fixture_R --out "$OUT/N563d_$1.jsonl" \
        > "$OUT/N563d_$1.log" 2>&1
  echo "  exit=$? bytes=$(stat -c%s "$OUT/N563d_$1.jsonl" 2>/dev/null)"
  pgrep -f "^python3 .*kaggle_aegis_sweep.py" | xargs -r kill
}

cell W2_16003200 "16.00,3200"
cell W3_20004000 "20.00,4000"
cell W4_32006400 "32.00,6400"
echo "N563 FINISH ALL DONE"
