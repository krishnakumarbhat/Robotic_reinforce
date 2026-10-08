#!/usr/bin/env python3
"""Purpose: N217 arm summariser. Reduce one rig JSONL to one dense line per suite plus the
paired compare record, so a 30-arm ladder is readable without dumping episodes.
Inputs: rig JSONL paths. Outputs: one table row per file on stdout.
"""
from __future__ import annotations

import json
import sys


def summarise(path: str) -> dict:
    """Purpose: read a rig JSONL, return {header knobs, per-suite, compare}.
    Inputs: path. Outputs: dict.
    """
    hdr: dict = {}
    per_suite: dict = {}
    compare: dict = {}
    arms: dict = {}
    with open(path) as fh:
        for line in fh:
            rec = json.loads(line)
            kind = rec.get("kind") or rec.get("record")
            if kind == "header":
                hdr = rec
            elif kind == "summary":
                per_suite = rec.get("per_suite", {})
            elif kind == "compare":
                compare = rec
            elif kind == "arm_summary":
                arms[rec.get("path_mode", "?")] = rec.get("per_suite", {})
    amp = hdr.get("troch_amp_m")
    w = hdr.get("troch_w_rad_per_m")
    return {"file": path.rsplit("/", 1)[-1],
            "amp": f"{amp:.6f}" if isinstance(amp, (int, float)) else "legacy",
            "w": f"{w:.4f}" if isinstance(w, (int, float)) else "legacy",
            "wA": f"{hdr['troch_wA_dimensionless']:.4f}"
            if isinstance(hdr.get("troch_wA_dimensionless"), (int, float)) else "?",
            "per_suite": per_suite, "compare": compare, "arms": arms, "hdr": hdr}


def main() -> None:
    """Purpose: print one row per input file. Inputs: argv paths."""
    for p in sys.argv[1:]:
        s = summarise(p)
        if not s["per_suite"]:
            continue
        cells = []
        for suite in ("fixture_A", "fixture_B", "fixture_R"):
            d = s["per_suite"].get(suite)
            if not d:
                continue
            cells.append(f"{suite[-1]}={d['mean_coverage_cont']:.4f}/{int(d['success_count']):02d}")
        cmp_ = s["compare"]
        line = (f"{s['file'][:40]:40s} A={s['amp']:>8s} w={s['w']:>8s} wA={s['wA']:>6s} "
                + "  ".join(cells))
        if cmp_:
            b = cmp_.get("fixture_B", {})
            line += (f" | keep={str(cmp_.get('keep')):5s} dB={b.get('delta')!s:>9s} "
                     f"wp={b.get('welch_p')!s:>10s} fp={b.get('fisher_p')!s:>9s}")
        print(line)


if __name__ == "__main__":
    main()
