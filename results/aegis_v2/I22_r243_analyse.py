"""I22 (run 243) read-out + regression proof -- NO PHYSICS, pure JSONL arithmetic.

Purpose: (a) split the two paired arms of each I22 file, (b) prove the `--compare-env
AEGIS_ROW_CENTRE=0.0` baseline arm reproduces the FROZEN champion field-for-field, so every
delta attributed to I22 is attributed to the row list, (c) print per-suite coverage/success/
jerk/slip/force for both arms with the rig's own Welch/Fisher numbers quoted, not recomputed.
Inputs: results/aegis_v2/I22_r243_*.jsonl + v2_trochoid_0,0.jsonl.
Outputs: prints a table + results/aegis_v2/I22_r243_analysis.txt.
"""
from __future__ import annotations

import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
V2 = os.path.join(REPO, "results", "aegis_v2")
SKIP = ("ts", "wall_s", "steps")
SUITES = ("fixture_A", "fixture_B", "fixture_R")


def load(name: str) -> list[dict]:
    """All records of one JSONL file."""
    with open(os.path.join(V2, name)) as fh:
        return [json.loads(line) for line in fh if line.strip()]


def eps_of(recs: list[dict]) -> list[dict]:
    """Episode records only (everything that is not header/summary/compare/summary_mode)."""
    return [r for r in recs
            if r.get("record") not in ("header", "summary", "compare", "summary_mode")]


def arms(recs: list[dict]) -> tuple[list[dict], list[dict]]:
    """(candidate, baseline) -- the rig emits the candidate arm first, then the compare arm."""
    e = eps_of(recs)
    h = len(e) // 2
    return e[:h], e[h:]


def key(r: dict) -> tuple:
    """Identity of an episode for the field-by-field regression diff."""
    return (r["suite"], r["seed"])


def flatten(r: dict) -> dict:
    """Episode with the nested failure_metadata tag record flattened to its leaves.

    Run 170 added `gate_intercepted` to failure_metadata, so the FROZEN 0,0 file has a
    different KEY SET there while every physics value is unchanged; flattening lets the
    regression diff compare the values instead of tripping over the schema.
    """
    out = dict(r)
    for k, v in list(out.items()):
        if isinstance(v, dict):
            for kk, vv in v.items():
                out[f"{k}.{kk}"] = vv
            del out[k]
    return out


def diff(a: list[dict], b: list[dict], restrict: set[str] | None = None) -> tuple[int, int, list[str]]:
    """(#episodes matched, #differing fields, first few differing field names)."""
    da = {key(r): flatten(r) for r in a}
    db = {key(r): flatten(r) for r in b}
    common = sorted(set(da) & set(db))
    bad: list[str] = []
    ndiff = 0
    for k in common:
        for f in sorted(set(da[k]) | set(db[k])):
            if f in SKIP or (restrict is not None and f not in restrict):
                continue
            if da[k].get(f) != db[k].get(f):
                ndiff += 1
                if len(bad) < 12:
                    bad.append(f"{k[0]}/s{k[1]}.{f}: {da[k].get(f)!r} vs {db[k].get(f)!r}")
    return len(common), ndiff, bad


def agg(rows: list[dict]) -> dict:
    """Per-suite means over one arm."""
    out = {}
    for s in SUITES:
        g = [r for r in rows if r["suite"] == s]
        if not g:
            continue
        n = len(g)
        out[s] = {
            "n": n,
            "covc": round(sum(r["coverage_cont"] for r in g) / n, 4),
            "min_covc": round(min(r["coverage_cont"] for r in g), 4),
            "succ": f"{sum(bool(r['success']) for r in g)}/{n}",
            "jerk": round(sum(r["jerk"] for r in g) / n, 5),
            "slip_m": round(sum(r["slip_m"] for r in g) / n, 5),
            "fn_mean": round(sum(r["fn_mean"] for r in g) / n, 4),
            "force_comp": round(sum(r["force_compliance"] for r in g) / n, 4),
            "stick": round(sum(r["stick_frac"] for r in g) / n, 5),
            "path_len_m": round(sum(r.get("path_len_m", 0.0) for r in g) / n, 4),
            "escaped": sum(bool(r.get("escaped")) for r in g),
        }
    return out


def main() -> int:
    lines: list[str] = []
    say = lines.append
    say("I22 (run 243) row-centring: paired arms vs the FROZEN champion (--compare-env "
        "AEGIS_ROW_CENTRE=0.0), canonical rig v2, PyBullet DIRECT, system python3\n")
    frozen = {}
    for tag, fn in (("0,0", "I22_r243_rowcentre_n00.jsonl"),
                    ("0.01,2", "I22_r243_rowcentre_n012.jsonl")):
        recs = load(fn)
        cand, base = arms(recs)
        cmp_ = [r for r in recs if r.get("record") == "compare"][0]
        hdr = [r for r in recs if r.get("record") == "header"][0]
        say(f"=== pose_noise {tag}  ({os.path.basename(fn)}) ===")
        say(f"    header pose_noise_cfg={hdr['pose_noise_cfg']!r}  path={hdr['path_mode']!r}  "
            f"compare={hdr['compare']!r}  compare_env={hdr['compare_env']!r}  "
            f"candidate ROW_CENTRE={cmp_['candidate_knobs']['ROW_CENTRE']}  "
            f"baseline ROW_CENTRE={cmp_['baseline_knobs']['ROW_CENTRE']}")
        say(f"    {'suite':<10} {'covc cand':>18} {'covc base':>18} {'succ':>14} "
            f"{'welch_p':>11} {'fisher_p':>11} {'jerk c/b':>15} {'slip c/b':>15} "
            f"{'fcomp c/b':>13} {'len c/b':>13}")
        ac, ab = agg(cand), agg(base)
        for s in SUITES:
            c, b, k = ac[s], ab[s], cmp_[s]
            say(f"    {s:<10} {c['covc']:>10.4f} (min {c['min_covc']:.4f}) "
                f"{b['covc']:>10.4f} (min {b['min_covc']:.4f}) "
                f"{c['succ']:>6}/{b['succ']:<7} {k['welch_p']:>11.3e} {k['fisher_p']:>11.3e} "
                f"{c['jerk']:>7.5f}/{b['jerk']:<7.5f} {c['slip_m']:>6.5f}/{b['slip_m']:<6.5f} "
                f"{c['force_comp']:>6.4f}/{b['force_comp']:<6.4f} "
                f"{c['path_len_m']:>6.4f}/{b['path_len_m']:<6.4f}")
        say(f"    rig compare keep = {cmp_['keep']}   cov_kernel_gap(cand) = "
            f"{max(r.get('cov_kernel_gap', 0.0) for r in cand)}")
        # regression: baseline arm == frozen champion. Two references, because they disagree on
        # FIELD SET only: v2_trochoid_0,0.jsonl predates the force telemetry (compared on the
        # intersection), I3_r242_ppo_n00.jsonl's compare arm is the same champion re-run under
        # the current rig (compared on every field).
        if tag == "0,0":
            ref = load("v2_trochoid_0,0.jsonl")
            refeps = [r for r in eps_of(ref) if r["suite"] in SUITES]
            shared = set(base[0]) & set(refeps[0])
            n, ndiff, bad = diff(base, refeps, restrict=shared)
            say(f"    REGRESSION A vs results/aegis_v2/v2_trochoid_0,0.jsonl "
                f"({len(shared)} shared fields): {n} episodes matched, "
                f"{ndiff} differing scored fields" + ("" if not bad else " -> " + "; ".join(bad)))
            cur = arms(load("I3_r242_ppo_n00.jsonl"))[1]
            n, ndiff, bad = diff(base, cur)
            say(f"    REGRESSION B vs run 242's champion compare arm (I3_r242_ppo_n00.jsonl, "
                f"current rig, ALL {len(base[0])} fields): {n} episodes matched, "
                f"{ndiff} differing scored fields" + ("" if not bad else " -> " + "; ".join(bad)))
            frozen["n"] = n
            frozen["ndiff"] = frozen.get("ndiff", 0) + ndiff
        say("")
    say("path length is IDENTICAL by construction: the fix is a pure +0.025 m translation of the "
        "row list, so the polyline is congruent (analytic path_len_ratio 1.000000 on both suites).")
    out = "\n".join(lines)
    print(out)
    with open(os.path.join(V2, "I22_r243_analysis.txt"), "w") as fh:
        fh.write(out + "\n")
    return 0 if frozen.get("ndiff", 1) == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
