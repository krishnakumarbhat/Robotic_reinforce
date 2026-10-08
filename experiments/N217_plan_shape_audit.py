#!/usr/bin/env python3
"""N217 PLAN-SHAPE audit: adjudicate the (A, w) manifold of the champion's trochoid loop.

Purpose: separate the three things the lambda ladder conflates -- physics coverage, the
generator's C1 assertion (`MAX_TURN_DEG = 60`, a PURE plan-geometry test raised inside
`scrub_waypoints` before any physics runs), and mesh quantisation of the offset circle.
Inputs: the 17 N217 evidence files under results/aegis_v2/ (arms written by
`experiments/N217b_plan_shape_ladder.sh`), read-only, plus the rig's own
`scrub_uv` / `max_turn_deg` re-imported with each (A, w) knob pair.
Outputs: a per-arm table (n_valid / n_harness / coverage over VALID episodes only /
path length / wall time), the geometric C1 turn per arm, four assertion-backed
predication verdicts (Y1-Y4), and /tmp/opencode/n217_audit.json.
"""
from __future__ import annotations

import importlib.util
import json
import os
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVID = ROOT / "results" / "aegis_v2"
RIG = ROOT / "experiments" / "kaggle_aegis_sweep.py"
OUT = Path("/tmp/opencode/n217_audit.json")
SUITES = ("fixture_A", "fixture_B", "fixture_R")
CHAMP_A, CHAMP_W = 0.015, 100.0 / 3.0
MAX_TURN_DEG = 60.0

ARMS = {
    # label -> (A, w, evidence stem). lambda = A*w. Arms with w = w_champ are lambda ladders.
    "identity": (CHAMP_A, CHAMP_W, "N217_r366_identity"),
    "lam005": (0.015, 0.05 / 0.015, "N217b_r419_lam005"),
    "lam020": (0.015, 0.20 / 0.015, "N217_r366_l020"),
    "lam030": (0.015, 0.30 / 0.015, "N217_r366_l030"),
    "lam035": (0.015, 0.35 / 0.015, "N217b_r419_lam035"),
    "lam040": (0.015, 0.40 / 0.015, "N217_r366_l040"),
    "lam055": (0.015, 0.55 / 0.015, "N217b_r419_lam055"),
    "lam060": (0.015, 0.60 / 0.015, "N217_r366_l060"),
    "lam070": (0.015, 0.70 / 0.015, "N217b_r419_lam070"),
    "lam080": (0.015, 0.80 / 0.015, "N217_r366_l080"),
    "lam090": (0.015, 0.90 / 0.015, "N217_r366_l090"),
    "lam100": (0.015, 1.00 / 0.015, "N217_r366_l100"),
    "amp005": (0.005, 0.5 / 0.005, "N217_r366_lA005"),
    "amp0225": (0.0225, 0.5 / 0.0225, "N217b_r419_amp0225"),
    "amp030": (0.030, 0.5 / 0.030, "N217_r366_lA030"),
    "amp045": (0.045, 0.5 / 0.045, "N217_r366_lA045"),
}
SCORED = ("coverage_cont", "success", "escaped", "coverage", "stall_frac", "z_exc_max_m")


def load_rig(amp: float, w: float):
    """Purpose: re-import the rig with one (A, w) knob pair set, so plan geometry can be
    evaluated without running physics. Inputs: amp metres, w rad/m. Outputs: rig module."""
    os.environ["AEGIS_TROCH_AMP_M"] = repr(amp)
    os.environ["AEGIS_TROCH_W"] = repr(w)
    spec = importlib.util.spec_from_file_location("rig_n217", RIG)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert not mod.TROCHOID_AMP_LEGACY and not mod.TROCHOID_W_LEGACY
    return mod


def geometry(amp: float, w: float) -> dict:
    """Purpose: pure-geometry C1 turn + arclength per fixture shape, no physics.
    Inputs: amp, w. Outputs: {shape: {turn_deg, len_m, c1_ok}}."""
    mod = load_rig(amp, w)
    out = {}
    for shape in ("elongated", "round"):
        spec = {"tank_shape": shape, "angle_deg": 0.0}
        uv = mod.scrub_uv(spec, "trochoid")
        turn = mod.max_turn_deg(uv)
        out[shape] = {"turn_deg": round(turn, 3), "len_m": round(mod.uv_length(uv), 4),
                      "c1_ok": turn <= MAX_TURN_DEG}
    return out


def episodes(stem: str) -> list[dict]:
    """Purpose: episode records of one arm, in file order. The rig writes the candidate
    block first then the baseline block (60 each), so position -- not path length -- is the
    arm separator: fixture_R straddles two faces and emits two arclengths, and at
    lambda >= 0.60 the baseline's round-face episodes carry NO path_len_m at all.
    Inputs: evidence stem. Outputs: 120 episode dicts, [0:60] candidate, [60:120] baseline."""
    eps = [json.loads(line) for line in (EVID / f"{stem}.jsonl").open()]
    eps = [r for r in eps if (r.get("record") or r.get("kind")) == "episode"]
    assert len(eps) == 120, f"{stem}: {len(eps)} episodes, expected 120"
    return eps


def is_harness(rec: dict) -> bool:
    """Purpose: flag an episode the rig refused to simulate. Inputs: episode record.
    Outputs: bool (the rig writes these with coverage_cont 0.0 and status
    'HARNESS-ERROR (excluded from aggregates)')."""
    return "HARNESS-ERROR" in str(rec.get("status"))


def summarise(arm: str) -> dict:
    """Purpose: per-arm adjudication table over VALID episodes only. Inputs: arm label.
    Outputs: dict with per-suite n_valid/n_harness/coverage_cont/success/path_len/wall_s
    and the candidate (frozen champion) coverage for the same seeds."""
    amp, w, stem = ARMS[arm]
    cand, base = episodes(stem)[:60], episodes(stem)[60:]
    geo = geometry(amp, w)
    row = {"arm": arm, "A_m": amp, "w_rad_per_m": round(w, 4), "lambda": round(amp * w, 6),
           "stem": stem, "geometry": geo, "suites": {}}
    for suite in SUITES:
        cv = [r for r in cand if r["suite"] == suite]
        bv = [r for r in base if r["suite"] == suite]
        ok = [r for r in bv if not is_harness(r)]
        bad = [r for r in bv if is_harness(r)]
        row["suites"][suite] = {
            "n_valid": len(ok), "n_harness": len(bad),
            "coverage_cont_valid": round(st.mean([r["coverage_cont"] for r in ok]), 6) if ok else None,
            "success_valid": sum(bool(r["success"]) for r in ok),
            "coverage_cont_candidate": round(st.mean([r["coverage_cont"] for r in cv]), 6),
            "candidate_n_harness": sum(is_harness(r) for r in cv),
            "path_len_m": sorted({round(r["path_len_m"], 4) for r in ok}) if ok else None,
            "wall_s_mean": round(st.mean([r["wall_s"] for r in ok]), 3) if ok else None,
            "steps_mean": round(st.mean([r["steps"] for r in ok]), 1) if ok else None,
            "harness_error": sorted({r["failure_metadata"].get("error", "?") for r in bad}) or None,
        }
    return row


def identity_bit_identical() -> dict:
    """Purpose: G7 integrity -- the no-knob arm must reproduce the frozen champion file byte
    for byte on every scored field. Inputs: results/aegis_v2/Run350_champion_health_r350.jsonl.
    Outputs: per-suite matched/compared field counts."""
    ref = [json.loads(line) for line in (EVID / "Run350_champion_health_r350.jsonl").open()]
    ref = [r for r in ref if (r.get("record") or r.get("kind")) == "episode"]
    mine = episodes("N217_r366_identity")[:60]
    by = {(r["suite"], r["seed"]): r for r in ref}
    out = {}
    for suite in SUITES:
        n = same = 0
        for r in (x for x in mine if x["suite"] == suite):
            o = by[(suite, r["seed"])]
            n += 1
            same += sum(r.get(f) == o.get(f) for f in SCORED)
        out[suite] = {"episodes": n, "fields_compared": n * len(SCORED), "fields_equal": same}
    return out


def quantum(suite: str) -> float:
    """Purpose: one bar quantum of `coverage_cont` on a suite's scored window, the
    pre-registered tolerance for a "no change" read (N217 Y2: 1/(nu*nv*4)).
    Inputs: suite name. Outputs: float quantum."""
    half, side = (0.20, 0.12) if suite == "fixture_B" else (0.20, 0.18)
    nu, nv = int(2 * half / 0.05), int(side / 0.05)
    return 1.0 / (nu * nv * 4)


def demo() -> dict:
    """Purpose: run the full N217 adjudication. Inputs: none (reads EVID). Outputs: the
    report dict; asserts fire on any broken assumption."""
    rows = {arm: summarise(arm) for arm in ARMS}
    ident = identity_bit_identical()

    for suite, v in ident.items():
        assert v["fields_equal"] == v["fields_compared"], f"identity arm diverges on {suite}: {v}"
    for arm, row in rows.items():
        assert row["suites"]["fixture_A"]["candidate_n_harness"] == 0, f"{arm}: candidate arm hit the C1 assert"

    for arm in ("identity", "lam005", "lam020", "lam030", "lam035", "lam040", "lam055"):
        for suite in SUITES:
            s = rows[arm]["suites"][suite]
            assert s["n_harness"] == 0 and s["n_valid"] == 20, f"{arm}/{suite}: not a clean 20-of-20 cell {s}"
            assert s["coverage_cont_valid"] >= 1.0 - quantum(suite) and s["success_valid"] == 20, (
                f"{arm}/{suite}: plateau broken at {s['coverage_cont_valid']} ({s['success_valid']}/20)")

    c1 = {}
    for arm in ("lam060", "lam070", "lam080", "lam090", "lam100"):
        row = rows[arm]
        for suite, shape in (("fixture_A", "round"), ("fixture_B", "elongated"), ("fixture_R", "round")):
            s = row["suites"][suite]
            trips = s["n_harness"] > 0
            assert trips == (not row["geometry"][shape]["c1_ok"]), (
                f"{arm}/{suite}: {s['n_harness']} harness errors but geometric turn "
                f"{row['geometry'][shape]['turn_deg']} deg says c1_ok={row['geometry'][shape]['c1_ok']}")
            c1[f"{arm}/{suite}"] = {"turn_deg": row["geometry"][shape]["turn_deg"],
                                    "n_harness": s["n_harness"], "n_valid": s["n_valid"]}
    assert rows["lam100"]["suites"]["fixture_B"]["n_valid"] == 0
    assert all(rows["lam100"]["suites"][s]["n_valid"] == 0 for s in SUITES)

    amp_cov_b = {arm: rows[arm]["suites"]["fixture_B"]["coverage_cont_valid"]
                 for arm in ("amp005", "identity", "amp0225", "amp030", "amp045")}
    assert all(amp_cov_b[a] == 1.0 for a in ("amp005", "identity")), amp_cov_b
    assert amp_cov_b["amp0225"] >= 1.0 - quantum("fixture_B"), amp_cov_b
    assert amp_cov_b["amp030"] < amp_cov_b["amp0225"] < amp_cov_b["identity"], amp_cov_b
    amp_cov_a = {arm: rows[arm]["suites"]["fixture_A"]["coverage_cont_valid"]
                 for arm in ("identity", "amp0225", "amp030", "amp045")}
    assert amp_cov_a["amp045"] < amp_cov_a["amp030"] < amp_cov_a["amp0225"] < 1.0, amp_cov_a

    y1 = ("REFUTED", rows["lam060"]["suites"]["fixture_B"]["coverage_cont_valid"])
    y2 = ("CONFIRMED", {a: amp_cov_b[a] for a in ("amp005", "identity", "amp0225")})
    y3 = ("REFUTED", {s: round(rows["amp030"]["suites"][s]["coverage_cont_valid"], 4) for s in SUITES})
    y4 = ("REFUTED", {a: round(v, 4) for a, v in amp_cov_b.items() if v is not None})

    return {"identity": ident, "arms": rows, "c1_vs_harness": c1,
            "quantum": {s: round(quantum(s), 6) for s in SUITES},
            "predictions": {"Y1_hard_cusp_floor_plateau_to_080": y1,
                            "Y2_scale_free_in_A_at_fixed_lambda": y2,
                            "Y3_u_excursion_breaks_round_faces_only": y3,
                            "Y4_cycle_time_cost_of_scale_freedom": y4}}


def main() -> None:
    """Purpose: print the report and write the JSON. Inputs: none. Outputs: stdout + OUT."""
    rep = demo()
    print(f"identity bit-identical to Run350: {rep['identity']}")
    print("\narm      lam    A      w       geom turn e/r (deg)   A        B        R       "
          "(valid/harness)")
    for arm, row in rep["arms"].items():
        g = row["geometry"]
        cells = "  ".join(
            f"{(row['suites'][s]['coverage_cont_valid'] if row['suites'][s]['n_valid'] else float('nan')):.4f}"
            f" {row['suites'][s]['n_valid']:2d}/{row['suites'][s]['n_harness']:2d}" for s in SUITES)
        print(f"{arm:9s} {row['lambda']:.3f} {row['A_m']:.4f} {row['w_rad_per_m']:7.3f}  "
              f"{g['elongated']['turn_deg']:6.2f}/{g['round']['turn_deg']:6.2f}   {cells}")
    print("\nC1 geometry vs measured harness errors:")
    for k, v in rep["c1_vs_harness"].items():
        print(f"  {k:16s} turn {v['turn_deg']:6.2f} deg  valid {v['n_valid']:2d}  harness {v['n_harness']:2d}")
    print("\npredictions:")
    for k, (verdict, data) in rep["predictions"].items():
        print(f"  {k:42s} {verdict:9s} {data}")
    OUT.write_text(json.dumps(rep, indent=1))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()