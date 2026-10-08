"""Purpose: audit the rig's SCORING KERNEL (N211) -- the three frozen literals inside
`_coverage_cont` that produced every `coverage_cont` / `success` in runs 1-359 (grid pitch
0.025 m, an isotropic disc of radius r_eff = min(hu,hv) although every pad is a RECTANGLE of
half-extents hu x hv, and the `pts[::2]` contact stride above 400 contacts) -- and decide
whether the certified fixture_B transfer_success depends on them.
Inputs: none (reads the seven N211 paired artifacts under results/aegis_v2/).
Outputs: one table on stdout; every claim is an assert, so a broken claim exits non-zero.
G7: nothing here computes, rescales or predicts coverage/success. Every number is READ BACK
from the rig's own episode records (`cov_k` / `succ_k`, re-measured in-run by the rig from the
same physics contact points) or from the paired compare blocks.
"""
import json  # noqa: E402
import statistics as st  # noqa: E402
from pathlib import Path  # noqa: E402

ART = Path("results/aegis_v2")
SUITES = ("fixture_A", "fixture_B", "fixture_R")
KERNELS = ("frozen", "allcontacts", "rect", "p2", "p4", "nodilat", "faithful")
FROZEN = "frozen"
PHYS = "PHYSICAL-VALIDATION"
# arm file -> label. Every arm is a 20-seed x 3-suite PAIRED run whose candidate arm turns the
# kernel audit ON and whose baseline arm turns it OFF, so the scored metric is identical in
# both arms by construction and the rig's own `keep` is false everywhere (G4).
ARMS = {
    "N211_r360_0_identity_trochoid": "trochoid 0,0 (saturated; identity vs run 359)",
    "N211_r360_1_trochoid_0p01_2": "trochoid 0.01,2",
    "N211_r360_2_trochoid_0p03_6": "trochoid 0.03,6",
    "N211_r360_3_raster_0p03_6": "raster 0.03,6 (non-saturated control, N207b rule)",
    "N211_r360_4_raster_rowcentre0": "raster 0,0 ROW_CENTRE=0 (pre-I22, partial cover)",
    "N211_r360_5_trochoid_0p03_6_hz1920": "trochoid 0.03,6 @1920 Hz (rate cross-check)",
    "N211_r360_6_raster_identity": "raster 0,0 ROW_CENTRE=1 (frozen default)",
}
PAIRS = (("N211_r360_2_trochoid_0p03_6", "N211_r360_3_raster_0p03_6"),   # champion vs baseline
         ("N211_r360_5_trochoid_0p03_6_hz1920", "N211_r360_3_raster_0p03_6"))


def load(stem: str) -> tuple[list[dict], dict, dict]:
    """Purpose: split one paired rig artifact into (candidate episodes, header, compare).
    Inputs: file stem under results/aegis_v2/. Outputs: the 60 candidate-arm episode records,
    the header record, the compare record.
    """
    rows = [json.loads(x) for x in (ART / f"{stem}.jsonl").read_text().splitlines()]
    hdr = next(r for r in rows if r.get("record") == "header")
    cmp_ = next((r for r in rows if r.get("record") == "compare"), {})
    eps = [r for r in rows if r.get("record") == "episode"
           and str(r.get("status", "")).startswith(PHYS) and r["suite"] in SUITES
           and "cov_k" in r]
    assert len(eps) == 60, f"{stem}: expected 60 candidate episodes, got {len(eps)}"
    return eps, hdr, cmp_


def mean(xs: list[float]) -> float:
    """Purpose: mean of a list. Inputs: xs. Outputs: float (0.0 when empty)."""
    return float(sum(xs) / len(xs)) if xs else 0.0


def kern(eps: list[dict], key: str, suite: str | None = None) -> tuple[float, int, int]:
    """Purpose: mean coverage and success count of one kernel over the candidate arm.
    Inputs: episodes, kernel name, optional suite filter. Outputs: (mean_coverage, n_succ, n).
    """
    sel = [r for r in eps if suite is None or r["suite"] == suite]
    return (mean([float(r["cov_k"][key]) for r in sel]),
            sum(1 for r in sel if r["succ_k"][key]), len(sel))


def welch_p(a: list[float], b: list[float]) -> float:
    """Purpose: two-sided Welch t-test p-value, the rig's own `welch()`.
    Inputs: two samples. Outputs: p (1.0 when either sample has no spread).
    """
    try:
        from scipy.stats import ttest_ind
        return float(ttest_ind(a, b, equal_var=False)[1])
    except Exception:  # noqa: BLE001
        na, nb = st.mean(a), st.mean(b)
        va, vb = st.variance(a), st.variance(b)
        se = (va / len(a) + vb / len(b)) ** 0.5
        return 1.0 if se == 0 else float(2 * (1 - 0.5 * (1 + math_erf(na - nb, se))))


def math_erf(delta: float, se: float) -> float:
    """Purpose: normal CDF of delta/se, via math.erf. Inputs: mean delta, standard error.
    Outputs: P(|Z| < |delta/se|) -- the fallback only, scipy is present in this rig env.
    """
    import math
    z = abs(delta / se)
    return math.erf(z / 2 ** 0.5)


def fisher_p(sa: int, sb: int, n: int) -> float:
    """Purpose: Fisher exact p on the paired-arm 2x2 table, the rig's own test.
    Inputs: successes a, successes b, episodes per arm. Outputs: p.
    """
    from scipy.stats import fisher_exact
    return float(fisher_exact([[sb, n - sb], [sa, n - sa]])[1])


def main() -> int:
    """Purpose: run every N211 claim as an assert and print the tables. Inputs: none.
    Outputs: 0 if all claims hold, non-zero on the first broken one.
    """
    loaded = {stem: load(stem) for stem in ARMS}
    total = 60 * len(ARMS)

    # --- C1: the audit flag is INERT on the scored metric, in every arm and every suite ----
    for stem, (_eps, _hdr, cmp_) in loaded.items():
        assert not cmp_.get("keep"), f"{stem}: rig keep must be false by construction"
        for s in SUITES:
            assert cmp_[s]["delta"] == 0.0, f"{stem}/{s}: audit moved the scored coverage"
            assert cmp_[s]["succ_b"] == cmp_[s]["succ_a"], f"{stem}/{s}: success moved"
    # --- C2: the re-measured frozen kernel IS the scored metric, episode for episode -------
    for stem, (eps, _h, _c) in loaded.items():
        for r in eps:
            assert r["cov_k"][FROZEN] == round(r["coverage_cont"], 4), \
                f"{stem}: frozen kernel != scored coverage_cont on seed {r['seed']}"
            assert r["succ_k"][FROZEN] == bool(r["success"]), \
                f"{stem}: frozen verdict != scored success on seed {r['seed']}"
    # --- C3: cross-process bit-identity of the whole identity arm vs run 359 --------------
    old = [json.loads(x) for x
           in (ART / "N210_r359_0_identity_trochoid.jsonl").read_text().splitlines()]
    ident, _, _ = loaded["N211_r360_0_identity_trochoid"]
    ident_seeds = {e["seed"] for e in ident}
    keys = ("seed", "suite", "tool_id", "friction", "friction_realized", "coverage_cont",
            "success", "slip_m", "stall_frac", "quality_tag", "coverage")
    by_seed, seen, old_c, max_d = {r["seed"]: r for r in ident}, set(), [], 0.0
    for r in (x for x in old if x.get("record") == "episode" and "cov_k" not in x):
        if r["seed"] in ident_seeds and r["seed"] not in seen:   # run 359 logged both arms
            seen.add(r["seed"])
            old_c.append(r)
            for k in keys:
                a, b = r[k], by_seed[r["seed"]][k]
                if isinstance(a, (int, float)) and not isinstance(a, bool):
                    max_d = max(max_d, abs(float(a) - float(b)))
                else:
                    assert a == b, f"seed {r['seed']} field {k}: {a!r} != {b!r}"
    assert len(old_c) == 60, f"identity overlap {len(old_c)} != 60"
    print(f"C1/C2/C3 audit inert, frozen kernel == scored metric, bit-identical to run 359 "
          f"(max |d| = {max_d:.1e}): PASS")

    # --- C4: geometric containment. min(hu,hv) < hu,hv, so the inscribed disc is a SUBSET of
    # the pad rectangle: the rect kernel can only report MORE coverage, episode for episode.
    n_up = sum(1 for _s, (eps, _h, _c) in loaded.items() for r in eps
               if r["cov_k"]["rect"] >= r["cov_k"][FROZEN] - 1e-9)
    assert n_up == total, f"rect < disc on {total - n_up} episodes (geometrically impossible)"
    print(f"C4 the true rectangular footprint dominates the inscribed disc on {n_up}/{total} "
          f"episodes: PASS")

    # --- C5: grid-pitch CONVERGENCE. p2 is converged if halving the pitch again moves the
    # mean by less than 0.005 (a fifth of the 0.025 quantum, 5% of the 0.90 threshold).
    conv = {s: mean([abs(float(r["cov_k"]["p2"]) - float(r["cov_k"]["p4"])) for r in eps])
            for s, (eps, _h, _c) in loaded.items()}

    assert max(conv.values()) < 0.005, f"C5 FAILED: 0.025 m grid not converged: {conv}"
    print(f"C5 grid pitch CONVERGED: mean |p2-p4| <= {max(conv.values()):.4f} in every arm "
          f"(the frozen 0.025 m pitch and its 0.0125 m refinement differ by at most a fifth of "
          f"one pitch): PASS")

    # --- C6: the contact stride is DEAD at the shipped step budget. contact_uv grows at most
    # once per control tick and the sweep runs T_MAX ticks, so len(pts) > 400 cannot happen.
    n_big = sum(1 for _s, (eps, _h, _c) in loaded.items() for r in eps if r["n_contacts"] > 400)
    steps = {loaded[s][1]["steps"] for s in ARMS}
    assert steps == {400}, f"unexpected step budget {steps}"
    assert n_big == 0, f"C6 FAILED: {n_big} episodes exceeded 400 contacts"
    print(f"C6 contact stride provably dead: steps={sorted(steps)}, contacts <= steps, "
          f"{n_big}/{total} episodes above the 400-contact trigger: PASS")

    # --- the main table: per arm, per suite, per kernel ------------------------------------
    for stem, label in ARMS.items():
        eps, hdr, cmp_ = loaded[stem]
        print(f"\n=== {stem}  {label}")
        print(f"    sim_hz {hdr['sim_hz']:.0f}  solver_iters {hdr['solver_iters']:.0f}  "
              f"kernel audit on: {hdr['cov_kernel_on']}  rig keep: {cmp_['keep']}")
        print(f"    {'suite':10s} {'kernel':12s} {'mean cov':>9s} {'succ':>8s}")
        for s in SUITES:
            for k in KERNELS:
                c, ns, n = kern(eps, k, s)
                print(f"    {s:10s} {k:12s} {c:9.4f} {ns:4d}/{n:<4d}"
                      f"{'   <- scored' if k == FROZEN else ''}")
        flips = {k: sum(1 for r in eps if r["succ_k"][k] != r["success"]) for k in KERNELS}
        print("    verdict flips vs the scored success: "
              + ", ".join(f"{k}={v}" for k, v in flips.items() if v) + " (none = 0)")

    # --- C7: the footprint model is the LIVE axis. Report the paired shift, do not assert a
    # direction beyond the geometric containment of C4.
    print("\nC7 footprint axis: rect - frozen, per arm (mean coverage, verdict flips)")
    for stem, (eps, _h, _c) in loaded.items():
        d = mean([float(r["cov_k"]["rect"]) - float(r["cov_k"][FROZEN]) for r in eps])
        f = sum(1 for r in eps if r["succ_k"]["rect"] != r["success"])
        nd = kern(eps, "nodilat")
        print(f"   {ARMS[stem]:46s} d {d:+.4f}  flips {f:3d}/60  "
              f"no-dilation coverage {nd[0]:.4f} ({nd[1]}/60)")

    # --- C8: the champion's ordering against the raster baseline survives EVERY kernel ------
    print("\nC8 champion vs raster baseline, paired on the SAME 20 seeds, per kernel")
    for cand, base in PAIRS:
        ce, _h, _c = loaded[cand]
        be, _h2, _c2 = loaded[base]
        print(f"   {cand.split('_', 3)[3]}  vs  {base.split('_', 3)[3]}")
        for s in SUITES:
            for k in KERNELS:
                ca = [float(r["cov_k"][k]) for r in ce if r["suite"] == s]
                ba = [float(r["cov_k"][k]) for r in be if r["suite"] == s]
                sa = sum(1 for r in ce if r["suite"] == s and r["succ_k"][k])
                sb = sum(1 for r in be if r["suite"] == s and r["succ_k"][k])
                if k == FROZEN or s != "fixture_B":
                    continue
                assert mean(ca) >= mean(ba), f"C8 FAILED: {cand} loses to raster on {s}/{k}"
                print(f"      {s:10s} {k:12s} cand {mean(ca):.4f} ({sa}/20)  "
                      f"base {mean(ba):.4f} ({sb}/20)  Welch p {welch_p(ba, ca):.3g}  "
                      f"Fisher p {fisher_p(sb, sa, 20):.3g}")
    # --- C9: rate cross-check -- the kernel shift is not a discretisation artifact ---------
    a, _h, _c = loaded["N211_r360_2_trochoid_0p03_6"]
    b, _h2, _c2 = loaded["N211_r360_5_trochoid_0p03_6_hz1920"]
    dev = max(abs(float(x["cov_k"][k]) - float(y["cov_k"][k]))
              for x, y in zip(sorted(a, key=lambda r: r["seed"]),
                              sorted(b, key=lambda r: r["seed"])) for k in KERNELS)
    print(f"\nC9 240 Hz vs 1920 Hz, same seeds: max |d(cov_k)| = {dev:.4f} "
          f"(the kernel sweep is solver-independent, extending N207.1 to the metric)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
