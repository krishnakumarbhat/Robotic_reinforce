#!/usr/bin/env bash
# N563b -- the MECHANISM LADDER for the refuted P1. Pre-registered from the P1 refutation:
#
#   P1 REFUTED (run 563): the extent midpoint did NOT cut reg_err_xy p50 >=2x on fixture_B
#   (0.98x .. 1.56x) and made it 2.0-2.5x WORSE on fixture_A. The competing mechanisms are:
#     (M-TRUNC)  truncation bias -- N562 already falsified this: reg_border_frac = 0.0000-0.0008.
#     (M-DISCR)  DISCRETISATION bias -- the extremal estimator reads only 4 support points of a
#                32x32 lattice of pitch 2H/(N-1), so it carries an O(pitch) INWARD bias, while
#                the mean averages ~N^2 points and carries O(pitch/sqrt(N^2)) = O(1/N^2).
#     Falsifier: raise REG_N. Under M-DISCR the extent error falls ~1/N (4x per doubling) and
#     the mean error falls ~1/N^2 (16x per doubling), so the extent/base RATIO must GROW with N
#     and cross 1.0 from below. Under M-TRUNC both are flat in N and the ratio is constant.
#
# Candidate arm = certified lattice stack + AEGIS_REG_EST=extent, REG_N swept.
# Baseline  arm = same stack, REG_N swept, AEGIS_REG_EST=0 (frozen mean). SAME 20 seeds (G4).
# 4 paired invocations x 20 seeds x {A,B,R} x 2 arms = 480 physical episodes.
set -u
cd "$(dirname "$0")/.." || exit 1
OUT=results/aegis_v2
mkdir -p "$OUT"

cell () {  # $1=tag  $2=REG_N  $3=pose_noise
  local tag="$1" n="$2" pn="$3"
  echo "=== N563b $tag  REG_N=$n  pose_noise=$pn ==="
  timeout 1200 env AEGIS_REG=depth AEGIS_REG_HALF_M=0.6 AEGIS_REG_N="$n" \
      AEGIS_ROW_CENTRE=1 AEGIS_REG_CASTS=0 AEGIS_REG_CN=12 AEGIS_REG_DFACT=2.0 \
      AEGIS_REG_EST=extent AEGIS_POSE_NOISE="$pn" \
      python3 experiments/kaggle_aegis_sweep.py --seeds 20 --no-upload \
        --path trochoid --compare trochoid --compare-env "AEGIS_REG_EST=0.0,AEGIS_REG_N=${n}.0" \
        --suites fixture_A,fixture_B,fixture_R \
        --out "$OUT/N563b_${tag}.jsonl" > "$OUT/N563b_${tag}.log" 2>&1
  echo "  exit=$? -> $OUT/N563b_${tag}.jsonl ($(stat -c%s "$OUT/N563b_${tag}.jsonl" 2>/dev/null) bytes)"
  pgrep -f "^python3 .*kaggle_aegis_sweep.py" | xargs -r kill
}

cell N16  16  "3.20,640"
cell N32  32  "3.20,640"
cell N64  64  "3.20,640"
cell N128 128 "3.20,640"

echo "N563b ALL DONE"