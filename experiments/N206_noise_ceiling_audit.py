"""N206 audit — pose-noise ceiling of the trochoid+REG stack, and which error channel
(translation residual vs plan yaw) the coverage loss actually tracks.

Purpose:  N190 claims the depth-sweep registration estimator's ~14 mm p90 resolution is the
          new floor. That claim was never tested against the metric. Runs N206_r352 sweep
          pose noise ABOVE I12's top level (0.03,6) and asks: (a) where does the stack stop
          recovering, (b) is coverage a function of the PLAN'S RESIDUAL TRANSLATION OFFSET
          alone (a single master curve across noise levels, N190 confirmed) or does the noise
          level matter on its own, (c) how much of the loss is the unobservable plan yaw.
Inputs:   results/aegis_v2/N206_r352_n*.jsonl  (canonical rig, 50 seeds x 3 suites x 2 arms
          per level, paired by seed; the REG arm logs reg_err_xy_m/reg_yaw_plan_deg, the
          OFF arm's plan offset IS the prior pose_noise since no correction is applied).
Outputs:  a stdout report + assertions that fail if the pairing or the logged provenance is
          not what the report claims.
"""
from __future__ import annotations

import json
import math
import pathlib
import statistics as st
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
LEVELS = ["0.03_6", "0.05_10", "0.08_16", "0.12_24"]
SUITES = ["fixture_A", "fixture_B", "fixture_R"]
BIN_M = 0.010
OFF_TAG = "AEGIS_REG"


def load(tag: str) -> list[dict]:
    """Purpose: read one level's rig JSONL. Inputs: level tag. Outputs: episode records."""
    path = ROOT / "results" / "aegis_v2" / f"N206_r352_n{tag}.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def phys(recs: list[dict]) -> list[dict]:
    """Purpose: keep only physical episodes. Inputs: records. Outputs: filtered records."""
    return [r for r in recs
            if r.get("record") == "episode"
            and str(r.get("status", "")).startswith("PHYSICAL")]


def residual(rec: dict) -> float:
    """Purpose: the translation offset the PLAN is actually built with.

    The REG arm plans from its estimate, so the residual is reg_err_xy_m; the OFF arm plans
    from the raw prior, whose offset is the logged pose_noise xy magnitude. Returns metres.
    """
    if rec.get("reg_err_xy_m") is not None:
        return float(rec["reg_err_xy_m"])
    return math.hypot(rec["pose_noise"][0], rec["pose_noise"][1])


def yaw_resid(rec: dict) -> float:
    """Purpose: the plan's yaw error in degrees (folded to [0,90] — the patch is a RECT, so
    the coverage is periodic in yaw with period 180 deg and only the folded value is
    physical; N194). Inputs: record. Outputs: degrees."""
    if rec.get("reg_yaw_plan_deg") is not None:
        return float(rec["reg_yaw_plan_deg"])
    return min(abs(math.degrees(rec["pose_noise"][2])) % 180.0, 180.0) % 90.0 or \
        min(abs(math.degrees(rec["pose_noise"][2])) % 180.0, 180.0 - abs(math.degrees(rec["pose_noise"][2])) % 180.0)


def quant(vals: list[float], q: float) -> float:
    """Purpose: linear-interpolated quantile. Inputs: values, q in [0,1]. Outputs: quantile."""
    s = sorted(vals)
    if not s:
        return float("nan")
    i = q * (len(s) - 1)
    lo, hi = int(math.floor(i)), int(math.ceil(i))
    return s[lo] if lo == hi else s[lo] + (s[hi] - s[lo]) * (i - lo)


def pearson(xs: list[float], ys: list[float]) -> float:
    """Purpose: Pearson r. Inputs: paired samples. Outputs: r."""
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = math.sqrt(sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys))
    return num / den if den else float("nan")


def spearman(xs: list[float], ys: list[float]) -> float:
    """Purpose: rank correlation (monotone, no linearity assumption). Inputs: samples.
    Outputs: rho."""
    def rank(v: list[float]) -> list[float]:
        order = sorted(range(len(v)), key=lambda i: v[i])
        out = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            r = (i + j) / 2 + 1
            for k in range(i, j + 1):
                out[order[k]] = r
            i = j + 1
        return out
    return pearson(rank(xs), rank(ys))


def main() -> int:
    """Purpose: run the whole audit. Inputs: none. Outputs: 0 on success."""
    per_level: dict[str, dict] = {}
    rows: list[dict] = []
    for tag in LEVELS:
        recs = load(tag)
        head = recs[0]
        assert head["record"] == "header", f"{tag}: first record is not the header"
        assert head["rig_version"] == 2, f"{tag}: not rig v2/v3 scoring"
        assert head["path_mode"] == "trochoid" and head["compare"] == "trochoid"
        assert head["seeds_requested"] == 50
        eps = phys(recs)
        assert len(eps) == 300, f"{tag}: expected 300 physical episodes, got {len(eps)}"
        # the rig runs the candidate arm first, then the baseline arm, and labels BOTH
        # episodes path_mode="trochoid" (the arm distinction lives in summary_mode), so the
        # arm is identified by emission order, not by the record.
        assert {r["path_mode"] for r in eps} == {"trochoid"}, f"{tag}: unexpected path_mode set"
        half = len(eps) // 2
        reg_eps, off_eps = eps[:half], eps[half:]
        # paired-by-seed: the seed depends only on (suite index, k), so both arms draw the
        # same friction / tool / customer sample
        by: dict = {}
        for r in reg_eps:
            by[(r["suite"], r["seed"])] = [r, None]
        for r in off_eps:
            by[(r["suite"], r["seed"])][1] = r
        assert all(v[1] is not None for v in by.values()), f"{tag}: unpaired seeds"
        for cand, base in by.values():
            assert abs(cand["friction"] - base["friction"]) < 1e-12, f"{tag}: friction drift"
            assert cand["tool_id"] == base["tool_id"], f"{tag}: tool drift"
        assert all(r.get("reg_err_xy_m") is None for r in off_eps), f"{tag}: OFF arm logged REG fields"
        assert all(r.get("reg_ok") for r in reg_eps), f"{tag}: reg_ok False in the REG arm"
        assert all(r["pts_source"] == "physics_contact" for r in eps), f"{tag}: non-physics coverage"
        per_level[tag] = {"head": head, "eps": eps}
        for r, arm in [(x, False) for x in reg_eps] + [(x, True) for x in off_eps]:
            rows.append({"level": tag, "suite": r["suite"], "arm": arm,
                         "res": residual(r), "yaw": yaw_resid(r), "cov": r["coverage_cont"],
                         "succ": bool(r["success"]), "len": r["path_len_m"]})

    print("=" * 78)
    print("N206.1  pose-noise ladder — trochoid + depth REG (candidate) vs REG off (paired)")
    print("=" * 78)
    print(f"{'noise':>9} {'arm':>4} | {'suite':>9} {'covc':>7} {'succ':>6} {'res_mm':>7} "
          f"{'p90':>7} {'yaw_deg':>8}")
    ladder = {}
    for tag in LEVELS:
        for arm in (False, True):
            for s in SUITES:
                sel = [r for r in rows if r["level"] == tag and r["arm"] == arm
                       and r["suite"] == s]
                line = (f"{tag.replace('_', ','):>9} {'OFF' if arm else 'REG':>4} | "
                        f"{s:>9} {st.mean(r['cov'] for r in sel):7.4f} "
                        f"{st.mean(r['succ'] for r in sel):6.3f} "
                        f"{1000 * st.mean(r['res'] for r in sel):7.1f} "
                        f"{1000 * quant([r['res'] for r in sel], 0.9):7.1f} "
                        f"{st.mean(r['yaw'] for r in sel):8.2f}")
                print(line)
                ladder[(tag, arm, s)] = (st.mean(r["cov"] for r in sel),
                                         st.mean(r["succ"] for r in sel))
    print()
    for tag in LEVELS:
        for s in SUITES:
            c, cs = ladder[(tag, False, s)]
            b, bs = ladder[(tag, True, s)]
            print(f"  {tag:>9} {s[-1]}: d(covc)={c - b:+.4f}  d(succ)={cs - bs:+.3f}")

    print()
    print("=" * 78)
    print("N206.2  master transfer curve: coverage_cont vs the PLAN'S RESIDUAL OFFSET")
    print("=" * 78)
    for s in SUITES:
        print(f"-- {s}")
        print(f"   {'bin_mm':>10} {'n':>4} {'covc':>7} {'succ':>6}   (pooled over 4 levels x 2 arms)")
        sel_all = [r for r in rows if r["suite"] == s]
        edges = [i * BIN_M for i in range(0, int(0.30 / BIN_M) + 1)]
        for lo, hi in zip(edges[:-1], edges[1:]):
            sel = [r for r in sel_all if lo <= r["res"] < hi]
            if len(sel) < 8:
                continue
            print(f"   {1000 * lo:5.0f}-{1000 * hi:5.0f} {len(sel):4d} "
                  f"{st.mean(r['cov'] for r in sel):7.4f} "
                  f"{st.mean(r['succ'] for r in sel):6.3f}")
        xs = [r["res"] for r in sel_all]
        print(f"   pearson(res, covc)={pearson(xs, [r['cov'] for r in sel_all]):+.3f} "
              f"spearman={spearman(xs, [r['cov'] for r in sel_all]):+.3f}")
        ys = [r["yaw"] for r in sel_all]
        print(f"   pearson(yaw, covc)={pearson(ys, [r['cov'] for r in sel_all]):+.3f} "
              f"spearman={spearman(ys, [r['cov'] for r in sel_all]):+.3f}")
        # translation-only prediction: does the noise LEVEL add anything once res is known?
        within = []
        for tag in LEVELS:
            for arm in (False, True):
                sel = [r for r in sel_all if r["level"] == tag and r["arm"] == arm
                       and 0.010 <= r["res"] < 0.020]
                if len(sel) >= 8:
                    within.append((tag, arm, len(sel), st.mean(r["cov"] for r in sel)))
        if within:
            vals = [w[3] for w in within]
            print("   res in [10,20) mm, mean covc per (level,arm) cell: "
                  + " ".join(f"{t[:5]}/{'OFF' if a else 'REG'}={v:.4f}(n={n})"
                             for t, a, n, v in within))
            print(f"   -> spread across cells with res held in a 10 mm band: "
                  f"{max(vals) - min(vals):.4f}")

    print()
    print("=" * 78)
    print("N206.3  which channel binds: matched-residual, REG arm only (yaw corrected where")
    print("        observable), split by whether the plan yaw was actually corrected")
    print("=" * 78)
    for s in SUITES:
        sel = [r for r in rows if r["suite"] == s and not r["arm"] and 0.010 <= r["res"] < 0.030]
        if len(sel) < 8:
            continue
        for lo, hi in ((0, 2.0), (2.0, 90.0)):
            sub = [r for r in sel if lo <= r["yaw"] < hi]
            if len(sub) < 5:
                continue
            print(f"  {s[-1]} yaw in [{lo:4.1f},{hi:4.1f}) deg  n={len(sub):3d} "
                  f"covc={st.mean(r['cov'] for r in sub):.4f} "
                  f"succ={st.mean(r['succ'] for r in sub):.3f} "
                  f"res={1000 * st.mean(r['res'] for r in sub):.1f}mm")

    print()
    print("=" * 78)
    print("N206.4  the window question: REG's cast window is +-REG_HALF_M (0.35 m) about the")
    print("        NOISY prior; the elongated top face is +-0.34 m, so containment margin =")
    print("        H - 0.34. Estimator error growth vs that margin:")
    print("=" * 78)
    print(f"{'noise':>9} {'H-0.34_m':>9} {'prior_mm':>9} {'res_mm':>8} {'ratio':>6} "
          f"{'res>2r_eff':>10}")
    for tag in LEVELS:
        sel = [r for r in rows if r["level"] == tag and r["suite"] == "fixture_B" and not r["arm"]]
        off = [r for r in rows if r["level"] == tag and r["suite"] == "fixture_B" and r["arm"]]
        assert off, f"{tag}: no OFF-arm fixture_B rows"
        prior = st.mean(r["res"] for r in off)
        print(f"{tag.replace('_', ','):>9} {0.35 - 0.34:9.3f} {1000 * prior:9.1f} "
              f"{1000 * st.mean(r['res'] for r in sel):8.1f} "
              f"{st.mean(r['res'] for r in sel) / prior:6.3f} "
              f"{sum(1 for r in sel if r['res'] > 0.070) / len(sel):10.3f}")
    print()
    print("OK: pairing, physics-only coverage, reg_ok and OFF-arm provenance all asserted.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
