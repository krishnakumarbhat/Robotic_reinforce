#!/usr/bin/env python3
"""N206 -- exhaustive G4 keep-feasibility map of the canonical rig's discrete surface.

The keep bar (G2/G4) reads ONE quantity: fixture_B transfer_success from the canonical
rig, must be > 0.70 over >= 20 seeds, and must beat the paired champion significantly
with no coverage regression. Every idea so far has been judged one cell at a time, so
the campaign's real question -- WHICH (path, pose-noise, registration) cells can still
ever return keep=true -- has never been measured. This script answers it by reading the
rig's own `keep` field. It edits no rig byte, adds no flag and computes nothing about
coverage or success itself (G7: every number below is copied verbatim from the rig's
emitted compare record).

Purpose: map keep-availability over {8 path modes} x {5 I5 pose-noise doses} x {reg off/on},
        paired against the frozen champion stack trochoid + AEGIS_REG=depth.
Inputs:  experiments/kaggle_aegis_sweep.py (canonical rig, invoked as a subprocess),
         AEGIS_POSE_NOISE / AEGIS_REG env, --compare + --compare-env for G4 pairing.
Outputs: results/aegis_v2/N206_keep_feasibility_map.jsonl (one row per cell, rig fields
         only) and a markdown table on stdout.

Run:  timeout 1200 python3 experiments/N206_keep_feasibility_map.py
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RIG = REPO / "experiments" / "kaggle_aegis_sweep.py"
OUTDIR = REPO / "results" / "aegis_v2"
MAP = OUTDIR / "N206_keep_feasibility_map.jsonl"
SEEDS = 20
WORKERS = 8
CELL_TIMEOUT_S = 420
# The I5 dose ladder. Levels outside it are not part of the documented surface.
NOISES: tuple[str, ...] = ("0,0", "0.005,1", "0.01,2", "0.02,4", "0.03,6")
CHAMPION = "trochoid"          # frozen champion path (I9: trochoid + REG)
# apply_knobs parses --compare-env values with float(); the string spelling
# "AEGIS_REG=depth" raises ValueError, so the knob is addressed numerically (1 -> depth).
CHAMPION_KNOB = "AEGIS_REG=1"
# The rig hard-refuses `orbit` on fixture_B ("patch circumradius 0.2088 m > face inradius
# 0.14 m; on such a face the yaw must be OBSERVABLE"), i.e. B-ineligible by construction
# (the N198 no-op). Captured at runtime below rather than asserted.
B_INELIGIBLE = ("orbit",)


def rig_path_modes() -> tuple[str, ...]:
    """Purpose: read PATH_MODES out of the rig source so the map cannot drift from it.
    Inputs: experiments/kaggle_aegis_sweep.py text. Outputs: tuple of mode names.
    """
    m = re.search(r"^PATH_MODES\s*=\s*\((.*?)\)\s*$", RIG.read_text(), re.M | re.S)
    if not m:
        raise SystemExit("PATH_MODES not found in rig source -- refusing to guess the surface")
    return tuple(re.findall(r'"([a-z0-9_]+)"', m.group(1)))


def probe_b_ineligible(mode: str) -> str:
    """Purpose: record the rig's own refusal for a mode that cannot run the bar's suite.
    Inputs: mode name. Outputs: the rig's stderr/message proving B-ineligibility.
    """
    env = dict(os.environ, AEGIS_POSE_NOISE="0,0", AEGIS_UPLOAD="0")
    p = subprocess.run([sys.executable, str(RIG), "--seeds", "1", "--no-upload",
                        "--path", mode, "--suites", "fixture_B",
                        "--out", str(OUTDIR / f"_ineligible_{mode}.jsonl")],
                       env=env, cwd=REPO, timeout=CELL_TIMEOUT_S,
                       capture_output=True, text=True)
    msg = (p.stdout or "") + (p.stderr or "")
    for line in msg.splitlines():
        if "rotation-invariant" in line or "inradius" in line:
            return line.strip().split("] ", 1)[-1]
    return (msg.strip().splitlines() or ["<no message>"])[-1][:300]


def build_cell(noise: str, mode: str, reg: int, tag: str = "") -> dict:
    """Purpose: one (pose-noise, path, registration) cell of the keep-feasibility map.
    Inputs: noise "sigma_t,sigma_yaw"; mode path name; reg 1 = AEGIS_REG=depth on the
            candidate arm. Outputs: dict with the exact rig argv/env for the cell.
    """
    env = dict(os.environ, AEGIS_POSE_NOISE=noise, AEGIS_UPLOAD="0")
    if reg:
        env["AEGIS_REG"] = "depth"
    else:
        env.pop("AEGIS_REG", None)
    argv = [sys.executable, str(RIG), "--seeds", str(SEEDS), "--no-upload",
            "--path", mode, "--compare", CHAMPION, "--suites", "fixture_B",
            "--out", str(OUTDIR / f"_cell_{noise}_{mode}_reg{reg}{tag}.jsonl")]
    if not reg:
        # Candidate arm has registration OFF, so the paired champion arm must carry it.
        argv += ["--compare-env", CHAMPION_KNOB]
    return {"pose_noise": noise, "path": mode, "reg": reg, "env": env, "argv": argv}


def run_cell(cell: dict) -> dict:
    """Purpose: invoke the rig for one cell and copy its compare record verbatim.
    Inputs: cell from build_cell. Outputs: one map row (rig fields only, no arithmetic).
    """
    row = {k: cell[k] for k in ("pose_noise", "path", "reg")}
    row["rig_invoked"] = True
    row["seeds_requested"] = SEEDS
    try:
        p = subprocess.run(cell["argv"], env=cell["env"], cwd=REPO, timeout=CELL_TIMEOUT_S,
                           capture_output=True, text=True)
    except subprocess.TimeoutExpired:
        row.update(status="cell_timeout", keep=None)
        return row
    out = Path(cell["argv"][-1])
    recs = []
    if out.exists():
        recs = [json.loads(ln) for ln in out.read_text().splitlines() if ln.strip()]
    cmp_rec = next((r for r in recs if r.get("record") == "compare"), None)
    if p.returncode != 0 or cmp_rec is None:
        row.update(status="cell_crash", keep=None,
                   stderr=(p.stderr or "")[-400:], returncode=p.returncode)
        return row
    fb = cmp_rec.get("fixture_B", {})
    # Pass-through only: these are the rig's own numbers, never recomputed (G7).
    row.update(
        status="mapped",
        candidate=cmp_rec.get("candidate"),
        baseline=cmp_rec.get("baseline"),
        candidate_reg=cmp_rec.get("candidate_knobs", {}).get("AEGIS_REG"),
        baseline_reg=cmp_rec.get("baseline_knobs", {}).get("AEGIS_REG"),
        pose_noise_cfg=cmp_rec.get("pose_noise_cfg") or next(
            (r.get("pose_noise_cfg") for r in recs if r.get("pose_noise_cfg")), None),
        harness_errors=next((r.get("harness_errors") for r in recs
                             if r.get("harness_errors") is not None), None),
        n_b=fb.get("n_b"), succ_a=fb.get("succ_a"), succ_b=fb.get("succ_b"),
        ts_a=fb.get("succ_a", 0) / fb["n_a"] if fb.get("n_a") else None,
        ts_b=fb.get("succ_b", 0) / fb["n_b"] if fb.get("n_b") else None,
        welch_p=fb.get("welch_p"), fisher_p=fb.get("fisher_p"),
        delta_cov=fb.get("delta"), mean_a=fb.get("mean_a"), mean_b=fb.get("mean_b"),
        keep=cmp_rec.get("keep"),
        episodes=sum(1 for r in recs if r.get("suite") == "fixture_B"),
    )
    return row


def validate(row: dict) -> None:
    """Purpose: fail loudly on any cell that cannot support a G4 claim.
    Inputs: one map row. Outputs: None; raises AssertionError on a defect.
    """
    if row["status"] != "mapped":
        return
    assert row["n_b"] == SEEDS, f"G4 needs {SEEDS} B episodes, got {row['n_b']} ({row})"
    assert row["candidate_reg"] == ("depth" if row["reg"] else ""), \
        f"candidate registration mis-routed: {row}"
    assert row["baseline_reg"] == "depth", f"champion arm lost registration: {row}"
    assert row["harness_errors"] == 0, f"harness errors in {row['path']}/{row['pose_noise']}"
    # G7: success/coverage arrive from the rig; the map only routes them.
    assert row["keep"] is not None and row["welch_p"] is not None, f"incomplete record: {row}"


def main() -> int:
    """Purpose: run the whole surface, validate it, write the map, print the table.
    Inputs: none. Outputs: exit code 0 on a clean map, 1 on a cell or validation defect.
    """
    OUTDIR.mkdir(parents=True, exist_ok=True)
    modes = rig_path_modes()
    ineligible = {m: probe_b_ineligible(m) for m in B_INELIGIBLE if m in modes}
    print(f"[n206] B-ineligible by the rig's own guard: {ineligible}")
    modes = tuple(m for m in modes if m not in ineligible)
    cells = [build_cell(nz, md, rg) for nz in NOISES for md in modes for rg in (0, 1)
             if not (rg == 1 and md == CHAMPION)]  # a self-compare carries no signal
    print(f"[n206] surface: {len(modes)} modes x {len(NOISES)} noise doses x 2 reg states "
          f"-> {len(cells)} paired cells, {SEEDS} seeds, fixture_B only (the bar's suite)")

    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        rows = list(ex.map(run_cell, cells))
    for row in rows:
        validate(row)

    rows.sort(key=lambda r: (NOISES.index(r["pose_noise"]), r["path"], r["reg"]))
    with MAP.open("w") as fh:
        fh.write(json.dumps({"record": "map_meta", "suite": "fixture_B", "seeds": SEEDS,
                             "champion": f"{CHAMPION}+{CHAMPION_KNOB}", "modes": list(modes),
                             "noises": list(NOISES), "b_ineligible": ineligible,
                             "cells": len(rows)}, sort_keys=True) + "\n")
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True) + "\n")

    print(f"\n| noise | path | reg | ts_a | ts_b | delta_cov | welch_p | fisher_p | keep |")
    print("|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        if r["status"] != "mapped":
            print(f"| {r['pose_noise']} | {r['path']} | {r['reg']} | {r['status']} |||||||")
            continue
        fmt = lambda v, d=4: ("-" if v is None else f"{v:.{d}f}")  # noqa: E731
        print(f"| {r['pose_noise']} | {r['path']} | {r['reg']} | {fmt(r['ts_a'], 3)} | "
              f"{fmt(r['ts_b'], 3)} | {fmt(r['delta_cov'])} | {fmt(r['welch_p'], 5)} | "
              f"{fmt(r['fisher_p'], 5)} | {r['keep']} |")

    bad = [r for r in rows if r["status"] != "mapped"]
    keeps = [r for r in rows if r.get("keep") is True]
    print(f"\n[n206] cells={len(rows)} unmapped={len(bad)} keep_true={len(keeps)}")
    for r in keeps:
        print(f"[n206] KEEP-CELL {r['pose_noise']} {r['path']} reg={r['reg']} "
              f"ts_b={r['ts_b']:.3f} delta={r['delta_cov']:.4f} "
              f"welch_p={r['welch_p']:.3g} fisher_p={r['fisher_p']:.3g}")
    print(f"[n206] map -> {MAP}")

    # Determinism guard runs AFTER the map is on disk: a failing guard must not cost the sweep.
    first = run_cell(build_cell(NOISES[0], modes[0], 0, tag="_guardA"))
    again = run_cell(build_cell(NOISES[0], modes[0], 0, tag="_guardB"))
    validate(first)
    validate(again)
    for k in ("succ_a", "succ_b", "welch_p", "fisher_p", "keep", "mean_b"):
        assert first[k] == again[k], f"cell not deterministic on {k}: {first[k]} vs {again[k]}"
    print("[n206] determinism guard passed (same seeds -> identical keep record)")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
