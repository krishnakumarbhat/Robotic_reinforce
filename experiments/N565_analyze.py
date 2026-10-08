#!/usr/bin/env python3
"""N565 containment-cliff analysis: score the PRE-REGISTERED falsifiers P1-P5.

Reads every N565_*.jsonl and N671_*.jsonl physical cell under results/aegis_v2,
reconstructs the N195.1 lattice law from the rig's OWN logged fields
(`pose_noise`, `reg_casts`, `reg_ok`, `success`, `aegis_reg_dfact`, `aegis_reg_half_m`)
and writes results/aegis_v2/N565_analysis.json.

P1 SUFFICIENCY   : in-window base (z_inf <= W/sigma_t) => fixture_B 20/20.
P2 NECESSITY     : out-of-window base => fails EXACTLY the seeds with z > W/sigma_t.
P3 ATTRIBUTION   : failing seeds identified from rig-logged pose_noise only.
P4 SCALE INVARIANCE: same base => same failing seed set at 4.00,800 and 16.00,3200.
P5 THE QUANTUM   : P(fail) = 1 - (1 - 2(1-Phi(W/sigma)))^n from realised W/sigma_t.
"""
import glob
import json
import math
import os
import sys

V2 = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                  "results", "aegis_v2")
RHO_INF = math.hypot(0.34, 0.14)          # rig: elongated +-inf half-extent, worst yaw
FIXTURES = ("fixture_B",)                 # the only decider suite (G4)


def load(path):
    rows = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return rows


def cell(path):
    """Collapse one jsonl into (header, candidate episodes, baseline episodes)."""
    rows = load(path)
    hdr = next((r for r in rows if r.get("record") == "header"), {})
    eps = [r for r in rows if r.get("record") == "episode"]
    sigma_t, sigma_yaw = (float(x) for x in hdr.get("pose_noise_cfg", "0,0").split(","))
    dfact = float(hdr.get("aegis_reg_dfact", 1.5))
    half = float(hdr.get("aegis_reg_half_m", 0.6))
    a_slack = max(half - RHO_INF, 1e-3)
    d = max(dfact * a_slack, 1e-3)
    k = int(math.ceil(6.0 * sigma_t / d)) + 1 if sigma_t > a_slack else 1
    W = (k - 1) * d / 2.0
    ratio = W / sigma_t if sigma_t > 0 else float("inf")
    cand, base = [], []
    for e in eps:
        casts = int(e.get("reg_casts", 1) or 1)
        if casts > 1:
            row = dict(e)
            row["z"] = max(abs(e["pose_noise"][0]), abs(e["pose_noise"][1])) / sigma_t
            cand.append(row)
        else:
            base.append(e)
    return dict(path=path, header=hdr, sigma_t=sigma_t, sigma_yaw=sigma_yaw,
                n=int(hdr.get("seeds_requested", 0)), dfact=dfact, half=half,
                a_slack=a_slack, d=d, k=k, W=W, ratio=ratio,
                n_casts_logged=sorted({int(e.get("reg_casts") or 0) for e in eps}),
                cand=cand, base=base,
                n=int(hdr.get("seeds_requested", 0)),
                reg_n=float(hdr.get("aegis_reg_n", 32)),
                compare_keep=_keep(rows))


def _keep(rows):
    """`compare.keep` is reported by the rig on its summary record."""
    for r in rows:
        if r.get("record") == "compare" and "keep" in r:
            return bool(r["keep"])
    return None


def score(c):
    """Per-cell P1/P2/P3 verdict on the fixture_B decider."""
    b = [e for e in c["cand"] if e.get("suite") == "fixture_B"]
    if not b:
        return None
    n = len(b)
    out = [e for e in b if e["z"] > c["ratio"]]
    inside = [e for e in b if e["z"] <= c["ratio"]]
    failed = [e for e in b if not e["success"]]
    fail_seeds = sorted(e["seed"] for e in failed)
    out_seeds = sorted(e["seed"] for e in out)
    base_b = [e for e in c["base"] if e.get("suite") == "fixture_B"]
    p = 2.0 * (1.0 - 0.5 * (1.0 + math.erf(c["ratio"] / math.sqrt(2.0))))
    return dict(
        path=os.path.basename(c["path"]), sigma_t=c["sigma_t"], reg_n=c["reg_n"],
        n=n, k=c["k"], W=round(c["W"], 5), ratio=round(c["ratio"], 4),
        n_casts_logged=c["n_casts_logged"], compare_keep=c["compare_keep"],
        z_inf=round(max(e["z"] for e in b), 4),
        in_window=len(inside), out_window=len(out),
        n_fail=len(failed), fail_seeds=fail_seeds, out_seeds=out_seeds,
        b_success=sum(1 for e in b if e["success"]),
        baseline_b_success=sum(1 for e in base_b if e["success"]),
        predicted=20 - len(out),
        p1_ok=all(e["success"] for e in inside),
        p2_ok=(fail_seeds == out_seeds),
        p3_ok=all((not e["reg_ok"]) for e in failed) and all(
            (e["z"] > c["ratio"]) for e in failed),
        p5_p_fail=round(1.0 - (1.0 - p) ** n, 6),
        p5_p_single=round(p, 6),
        fail_reg_err=[None if (e.get("reg_err_xy_m") is None or
                                e.get("reg_err_xy_m") != e.get("reg_err_xy_m"))
                      else round(e["reg_err_xy_m"], 5) for e in failed],
        in_win_reg_err_p50=_p50([e["reg_err_xy_m"] for e in inside]),
        in_win_escaped=sum(1 for e in inside if e.get("escaped")),
        fail_escaped=sum(1 for e in failed if e.get("escaped")),
    )


def _p50(xs):
    xs = sorted(x for x in xs if x == x)
    if not xs:
        return None
    return round(xs[len(xs) // 2], 5)


def main():
    paths = sorted(glob.glob(os.path.join(V2, "N565_*.jsonl")) +
                   glob.glob(os.path.join(V2, "N671_*.jsonl")))
    cells = [cell(p) for p in paths]
    scored = [s for s in (score(c) for c in cells) if s]
    # duplicate-detection: same (sigma, reg_n, base set) appearing twice
    by_key = {}
    for c, s in zip([c for c in cells if score(c)], scored):
        by_key.setdefault((s["sigma_t"], s["reg_n"]), []).append(s["path"])
    dupes = {str(k): v for k, v in by_key.items() if len(v) > 1}

    p1 = [s for s in scored if s["in_window"] > 0]
    p2 = [s for s in scored if s["out_window"] > 0]
    p1_violations = [s["path"] for s in p1 if not s["p1_ok"]]
    p2_violations = [s["path"] for s in p2 if not s["p2_ok"]]
    p3_violations = [s["path"] for s in scored if not s["p3_ok"]]

    # P4: same seed base across cells => identical failing seed set
    by_base = {}
    for s in scored:
        base_key = int(s["path"].split("_base")[-1].split(".jsonl")[0]) \
            if "_base" in s["path"] else None
        if base_key is None:
            continue
        by_base.setdefault((base_key, s["reg_n"]), []).append(s)
    p4 = {}
    p4_ok = True
    for (base_key, reg_n), ss in sorted(by_base.items()):
        if len(ss) < 2:
            continue
        sets = {s["sigma_t"]: s["fail_seeds"] for s in ss}
        agree = len({tuple(v) for v in sets.values()}) == 1
        p4[str(base_key)] = dict(sigmas={str(k): v for k, v in sets.items()},
                                 agree=agree)
        p4_ok = p4_ok and agree

    agg = dict(
        cells=len(scored),
        episodes=sum(s["n"] for s in scored),
        in_window_eps=sum(s["in_window"] for s in scored),
        out_window_eps=sum(s["out_window"] for s in scored),
        fails=sum(s["n_fail"] for s in scored),
        b_success=sum(s["b_success"] for s in scored),
        baseline_b_success=sum(s["baseline_b_success"] for s in scored),
        p50_reg_err_in_window=_p50([s["in_win_reg_err_p50"] for s in scored]),
    )
    verdict = dict(
        P1_sufficiency=dict(violations=p1_violations, verdict="HOLDS" if not p1_violations else "REFUTED"),
        P2_necessity=dict(violations=p2_violations, verdict="HOLDS" if not p2_violations else "REFUTED"),
        P3_attribution=dict(violations=p3_violations, verdict="HOLDS" if not p3_violations else "REFUTED"),
        P4_scale_invariance=dict(detail=p4, verdict="HOLDS" if p4_ok else "REFUTED"),
        P5_quantum=dict(
            note="certificate rule: quote a frontier cell ONLY with realised z_inf and margin 1-z_inf/(W/sigma_t)",
            per_cell_p_fail={s["path"]: s["p5_p_fail"] for s in scored},
        ),
        aggregate=agg,
        duplicate_cells=dupes,
    )
    out = dict(cells=scored, verdict=verdict)
    dest = os.path.join(V2, "N565_analysis.json")
    with open(dest, "w") as fh:
        json.dump(out, fh, indent=2, sort_keys=False)
    for s in scored:
        print(f'{s["path"]:44s} sig={s["sigma_t"]:5.2f} reg_n={s["reg_n"]:5.0f} '
              f'W/s={s["ratio"]:.3f} z_inf={s["z_inf"]:.3f} B={s["b_success"]}/20 '
              f'base={s["baseline_b_success"]}/20 out={s["out_window"]} '
              f'fail={s["n_fail"]} P1={s["p1_ok"]} P2={s["p2_ok"]} P3={s["p3_ok"]} '
              f'keep={s["compare_keep"]}')
    print(json.dumps(verdict, indent=2)[:4000])
    return 0 if all(v["verdict"] == "HOLDS" for k, v in verdict.items()
                    if isinstance(v, dict) and "verdict" in v) else 1


if __name__ == "__main__":
    sys.exit(main())
