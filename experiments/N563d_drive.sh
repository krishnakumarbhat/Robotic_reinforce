#!/usr/bin/env bash
# N563d -- where IS the pose-noise wall on the certified lattice stack? Pre-registered from the
# N563c P3 outcome BEFORE running.
#
# Going in: N563c found fixture_B at 20/20 in BOTH arms at 4.80,960 / 6.40,1280 / 8.00,1600
# with reg_ok 20/20 and coverage_cont identical (985.2/985.2 at 8.00,1600). N562 certified only
# 3.20,640. The estimator never changed the outcome, so the frontier is the LATTICE CONTAINMENT
# (N562 D7) and it has simply never been probed above 3.20,640 to its end.
#
# Question: is the wall FINITE, and where? Two competing readings:
#   (W-FINITE) the lattice containment fails past some pose noise -> a real certified wall.
#   (W-UNBND) containment holds and B stays 20/20 -> the pose-noise axis has no wall at all in
#              this rig and 3.20,640 was only where the last worker stopped looking.
# Candidate arm = certified lattice stack at REG_N=128 + AEGIS_REG_EST=extent.
# Baseline  arm = same stack at REG_N=128, AEGIS_REG_EST=0 (frozen mean). SAME 20 seeds (G4).
# 4 paired invocations x 20 seeds x {A,B,R} x 2 arms = 480 physical episodes.
set -u
cd "$(dirname "$0")/.." || exit 1
OUT=results/aegis_v2
mkdir -p "$OUT"

cell () {  # $1=tag  $2=pose_noise
  local tag="$1" pn="$2"
  echo "=== N563d $tag  REG_N=128  pose_noise=$pn ==="
  timeout 1200 env AEGIS_REG=depth AEGIS_REG_HALF_M=0.6 AEGIS_REG_N=128 \
      AEGIS_ROW_CENTRE=1 AEGIS_REG_CASTS=0 AEGIS_REG_CN=12 AEGIS_REG_DFACT=2.0 \
      AEGIS_REG_EST=extent AEGIS_POSE_NOISE="$pn" \
      python3 experiments/kaggle_aegis_sweep.py --seeds 20 --no-upload \
        --path trochoid --compare trochoid --compare-env "AEGIS_REG_EST=0.0,AEGIS_REG_N=128.0" \
        --suites fixture_A,fixture_B,fixture_R \
        --out "$OUT/N563d_${tag}.jsonl" > "$OUT/N563d_${tag}.log" 2>&1
  echo "  exit=$? -> $OUT/N563d_${tag}.jsonl ($(stat -c%s "$OUT/N563d_${tag}.jsonl" 2>/dev/null) bytes)"
  pgrep -f "^python3 .*kaggle_aegis_sweep.py" | xargs -r kill
}

cell W1_12002400 "12.00,2400"
cell W2_16003200 "16.00,3200"
cell W3_20004000 "20.00,4000"
cell W4_32006400 "32.00,6400"

echo "N563d ALL DONE"