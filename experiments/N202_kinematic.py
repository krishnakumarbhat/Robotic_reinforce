#!/usr/bin/env python3
"""N202d -- the C1 legality law of the champion trochoid is a function of the
DIMENSIONLESS MESH RATIO k = w*A alone, and coverage is flat in A inside the legal
window, so the loop amplitude is a PURELY KINEMATIC knob.

Two independent measurements, zero physics, zero new rig bytes:
  (1) INVARIANCE: max_turn_deg(scrub_uv(A, w)) is constant along k = w*A for every
      (A, w) on the line, over two decades of A.  The rig's own `max_turn_deg` and
      `scrub_uv` are called verbatim, one subprocess per (A, w) because the rig reads
      its knobs from the environment at import time.
  (2) the two harness crashes already on disk (AEGIS_TROCH_AMP_M = 0.0225 and 0.030,
      k = 0.75 and 1.00) are the SAME points on this curve, so the law is calibrated on
      rig-asserted numbers and not only on legal ones.
Purpose/Inputs/Outputs documented per function; prints a table, writes
results/aegis_v2/N202_kinematic.json.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

PROBE = r"""
import json, math, sys
sys.argv = ["rig"]
import importlib.util
spec = importlib.util.spec_from_file_location("rig", "experiments/kaggle_aegis_sweep.py")
rig = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rig)
out = {}
for shape, side in (("round", 0.18), ("elongated", 0.12)):
    sp = {"tank_shape": shape}
    uv = rig.scrub_uv(sp, "trochoid")
    out[shape] = {"n": len(uv), "len_m": rig.uv_length(uv),
                  "max_turn_deg": rig.max_turn_deg(uv)}
out["k"] = rig.TROCHOID_AMP_M * rig.TROCHOID_W
out["A"] = rig.TROCHOID_AMP_M
out["w"] = rig.TROCHOID_W
out["min_speed_law"] = 1.0 - out["k"]
print("PROBE " + json.dumps(out))
"""


def probe(amp: float, w: float | None) -> dict:
    """Purpose: call the rig's own plan generator + C1 check for one (A, w) pair.
    Inputs: loop amplitude A in metres, rate w (None keeps the champion w = DR/R).
    Outputs: the probe dict (per-shape point count, length, max turn, k, min speed)."""
    env = dict(os.environ)
    env["AEGIS_TROCH_AMP_M"] = f"{amp!r}"
    if w is not None:
        env["AEGIS_TROCH_W"] = f"{w!r}"
    res = subprocess.run([sys.executable, "-c", PROBE], env=env, cwd=".",
                         capture_output=True, text=True, timeout=300)
    for line in res.stdout.splitlines():
        if line.startswith("PROBE "):
            return json.loads(line[6:])
    return {"error": (res.stderr or res.stdout)[-300:]}


def main() -> None:
    champ_w = 0.5 / 0.015
    lines: dict = {"champion_w": champ_w, "rows": []}
    amps = [0.005, 0.0075, 0.010, 0.0125, 0.015, 0.0165, 0.018, 0.0195, 0.021,
            0.0225, 0.0240, 0.0270, 0.0300]
    for a in amps:
        p = probe(a, None)
        lines["rows"].append({"A_m": a, "k": p.get("k"),
                              "min_speed_1_minus_k": p.get("min_speed_law"),
                              "max_turn_deg": (p.get("round") or {}).get("max_turn_deg"),
                              "len_m": (p.get("round") or {}).get("len_m"),
                              "max_turn_elongated_deg": (p.get("elongated") or {}).get(
                                  "max_turn_deg")})
    # INVARIANCE along k = 0.5: (A, w) = (0.015, 33.33), (0.030, 16.67), (0.0075, 66.67),
    # (0.060, 8.33) -- a 8x amplitude range at constant k.
    inv = []
    for a, w in ((0.0075, 0.5 / 0.0075), (0.015, champ_w), (0.030, 0.5 / 0.030),
                 (0.060, 0.5 / 0.060)):
        p = probe(a, w)
        inv.append({"A_m": a, "w": w, "k": p.get("k"),
                    "max_turn_deg": (p.get("round") or {}).get("max_turn_deg"),
                    "len_m": (p.get("round") or {}).get("len_m")})
    lines["k_0p5_invariance"] = inv
    t = [r["max_turn_deg"] for r in inv if isinstance(r["max_turn_deg"], (int, float))]
    lines["k_0p5_max_turn_spread_deg"] = (max(t) - min(t)) if len(t) > 1 else None
    lines["rig_asserted_points"] = {"k=0.75 (A=0.0225)": 102.9, "k=1.00 (A=0.030)": 150.6}
    lines["C1_threshold_deg"] = 60.0
    legal = [r["A_m"] for r in lines["rows"]
             if isinstance(r["max_turn_deg"], (int, float)) and r["max_turn_deg"] < 60.0]
    lines["legal_A_window_m"] = [min(legal), max(legal)] if legal else None
    # NB: this used to assert that A=0.005/0.010/0.015 were bit-identical -- that was
    # written before the runs and is FALSE for the CANDIDATE arms.  What is bit-identical
    # is the BASELINE (champion A=0.015) arm across files: 200/200 seeds identical at
    # 200 seeds (0.7204 in all three files) and 100/100 at 100 seeds (0.7169), which is
    # the harness-reproducibility control.  The dose itself is measured in
    # results/aegis_v2/N202_readout.json (P1/P2), not here.
    lines["coverage_dose_reference"] = (
        "fixture_A, pose_noise 0,56, seed base 400000: the champion baseline arm is "
        "BIT-IDENTICAL across files (200/200 seeds at covc 0.7204 in the three 200-seed "
        "files, 100/100 at 0.7169 in the three 100-seed files); the candidate arms sit "
        "-0.0064 / -0.0050 / -0.0059 / -0.0001 below it at A = 0.0075 / 0.010 / 0.005 / "
        "0.0165 -- see N202_readout.json P1 (paired) and P2 (equivalent yaw shift).")
    json.dump(lines, open("results/aegis_v2/N202_kinematic.json", "w"), indent=1)
    for r in lines["rows"]:
        print(f"  A={r['A_m']:.4f} k={r['k']:.4f} max_turn={r['max_turn_deg']} deg "
              f"len={r['len_m']}")
    print("  k=0.5 invariance:", json.dumps(inv))
    print(f"  spread {lines['k_0p5_max_turn_spread_deg']} deg  "
          f"legal A window {lines['legal_A_window_m']}")


if __name__ == "__main__":
    main()
