#!/usr/bin/env bash
# N563c -- the P3 DECIDER. Pre-registered from the N563b M-DISCR confirmation BEFORE running.
#
# What is established going in (all paired, 20 seeds, rig keep=false on every cell so far):
#   - N563 P1 REFUTED: at the certified N=32 the extent midpoint does NOT cut reg_err_xy p50 by
#     2x on fixture_B (0.98x..1.56x) and is 2.0-2.5x WORSE on fixture_A.
#   - N563b M-DISCR CONFIRMED: the ratio base/cand GROWS monotonically with REG_N on the two
#     elongated faces -- B 1.224 -> 1.807 -> 2.926 and R 1.393 -> 2.392 -> 3.871 for
#     N = 32 -> 64 -> 128 -- and the scaling exponents separate (mean ~1/N^2, extent ~1/N^1.3-1.6).
#     So the estimator gap is a DISCRETISATION-BIAS floor on the extremal estimator, NOT the
#     truncation bias N562 named, and the frozen N=32 simply cannot see the lever.
#   - At N=128 on fixture_B: reg_err_xy p50 = 2.04 mm (extent) vs 5.98 mm (mean), 2.93x.
#     This is the first cell where the estimator residual is sub-3-mm, i.e. P3's "0 mm residual".
#   - reg_ok is ALREADY 20/20 at every cell in every arm (lattice containment), so the estimator
#     has never been the binder of the pose-noise frontier. N562's D7 already said so.
#
# P3 DECIDER: with the residual now small enough to matter, does it buy a further frontier cell?
# Candidate arm = certified lattice stack at REG_N=128 + AEGIS_REG_EST=extent.
# Baseline  arm = same stack at REG_N=128, AEGIS_REG_EST=0 (frozen mean). SAME 20 seeds (G4).
# 3 paired invocations x 20 seeds x {A,B,R} x 2 arms = 360 physical episodes.
# A KEEP needs rig keep=true (>=20 seeds, B > 0.70, paired p < 0.01, no coverage regression).
set -u
cd "$(dirname "$0")/.." || exit 1
OUT=results/aegis_v2
mkdir -p "$OUT"

cell () {  # $1=tag  $2=pose_noise
  local tag="$1" pn="$2"
  echo "=== N563c $tag  REG_N=128  pose_noise=$pn ==="
  timeout 1200 env AEGIS_REG=depth AEGIS_REG_HALF_M=0.6 AEGIS_REG_N=128 \
      AEGIS_ROW_CENTRE=1 AEGIS_REG_CASTS=0 AEGIS_REG_CN=12 AEGIS_REG_DFACT=2.0 \
      AEGIS_REG_EST=extent AEGIS_POSE_NOISE="$pn" \
      python3 experiments/kaggle_aegis_sweep.py --seeds 20 --no-upload \
        --path trochoid --compare trochoid --compare-env "AEGIS_REG_EST=0.0,AEGIS_REG_N=128.0" \
        --suites fixture_A,fixture_B,fixture_R \
        --out "$OUT/N563c_${tag}.jsonl" > "$OUT/N563c_${tag}.log" 2>&1
  echo "  exit=$? -> $OUT/N563c_${tag}.jsonl ($(stat -c%s "$OUT/N563c_${tag}.jsonl" 2>/dev/null) bytes)"
  pgrep -f "^python3 .*kaggle_aegis_sweep.py" | xargs -r kill
}

cell D1_480960  "4.80,960"
cell D2_6401280 "6.40,1280"
cell D3_8001600 "8.00,1600"

echo "N563c ALL DONE"