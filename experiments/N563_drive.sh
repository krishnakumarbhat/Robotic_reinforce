#!/usr/bin/env bash
# N563 -- AEGIS_REG_EST=extent on the TRUNCATION-FREE lattice stack (the named lever from run 562).
# Pre-registered in autoresearch_research.ideas.md (N563 block) BEFORE any dose.
#
# Candidate arm  = the certified N478/N562 lattice stack VERBATIM + AEGIS_REG_EST=extent.
#                   Only the estimator differs from the paired baseline; no mechanism byte moves.
# Baseline  arm  = --compare trochoid --compare-env AEGIS_REG_EST=0 (the frozen mean), SAME 20
#                   seeds -> same friction / tool / noise / customer (G4).
# 5 paired invocations x 20 seeds x {A,B,R} x 2 arms = 600 physical episodes.
set -u
cd "$(dirname "$0")/.." || exit 1
OUT=results/aegis_v2
mkdir -p "$OUT"

# baseline arm: only AEGIS_REG_EST is written; float 0.0 -> "" -> the frozen mean path
# (the `if REG_EST == "extent"` branch is the only reader, experiments/kaggle_aegis_sweep.py:2536).
BASEENV="AEGIS_REG_EST=0.0"

cell () {  # $1=tag  $2=pose_noise
  local tag="$1" pn="$2"
  echo "=== N563 $tag  pose_noise=$pn  (candidate REG_EST=extent vs paired REG_EST=mean) ==="
  timeout 1200 env AEGIS_REG=depth AEGIS_REG_HALF_M=0.6 AEGIS_REG_N=32 \
      AEGIS_ROW_CENTRE=1 AEGIS_REG_CASTS=0 AEGIS_REG_CN=12 AEGIS_REG_DFACT=2.0 \
      AEGIS_REG_EST=extent AEGIS_POSE_NOISE="$pn" \
      python3 experiments/kaggle_aegis_sweep.py --seeds 20 --no-upload \
        --path trochoid --compare trochoid --compare-env "$BASEENV" \
        --suites fixture_A,fixture_B,fixture_R \
        --out "$OUT/N563_${tag}.jsonl" > "$OUT/N563_${tag}.log" 2>&1
  echo "  exit=$? -> $OUT/N563_${tag}.jsonl ($(stat -c%s "$OUT/N563_${tag}.jsonl" 2>/dev/null) bytes)"
  pgrep -f "^python3 .*kaggle_aegis_sweep.py" | xargs -r kill
}

cell C1_03264   "0.32,64"
cell C2_096192  "0.96,192"
cell C3_320640  "3.20,640"
cell C4_400800  "4.00,800"
cell C5_480960  "4.80,960"

echo "N563 ALL DONE"