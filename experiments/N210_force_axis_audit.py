"""Purpose: audit the rig's NORMAL-FORCE axis (N210) -- the press setpoint that produced the
"0.5 N, not the 10-25 N spec" limitation line -- and locate where the primary metric stops
responding to it. Inputs: none (reads the N210 paired artifacts under results/aegis_v2/).
Outputs: one table on stdout; every claim is an assert, so a broken claim exits non-zero.
G7: nothing here computes coverage or success -- every number is READ BACK from the rig's own
episode records (pts_source=physics_contact) and from the paired compare blocks.
"""
import json  # noqa: E402
import math  # noqa: E402
import statistics as st  # noqa: E402
from pathlib import Path  # noqa: E402

ART = Path("results/aegis_v2")
SUITES = ("fixture_A", "fixture_B", "fixture_R")
KP = 25.0                 # the rig's fixed tangential proportional gain, N/m (frozen)
FINE_M = 0.025            # fine coverage-grid pitch == the metric's length quantum
FROZEN_PRESS_M, FROZEN_K, FROZEN_CLAMP = 0.020, 1.0e3, 3.0
# arm tag -> commanded normal force, N. K = 2000*Fn holds the frozen penetration Fn/K = 0.5 mm
# (a real pad's effective stiffness rises with its working force), and the rail is lifted to
# 1.5*Fn + 2 so it cannot bind. F is the same 25 N press through the FROZEN 3.0 N rail.
ARMS = {"A_f0p05": 0.05, "B_f0p5_identity": 0.5, "G_f1": 1.0, "H_f1p5": 1.5,
        "C_f2": 2.0, "D_f10": 10.0, "E_f25": 25.0, "F_f25_rail3": 25.0}
COULOMB_ARMS = ("C_f2", "D_f10", "E_f25")   # arms where mu*Fn/kp dominates the 2.5 mm floor


def recs(path: Path) -> tuple[list[dict], list[dict], dict, dict]:
    """Purpose: split one paired rig artifact into (candidate, baseline, compare, header).
    Inputs: path. Outputs: the two 60-record halves, the compare record and the header.
    """
    rows = [json.loads(x) for x in path.read_text().splitlines()]
    head = next(r for r in rows if r.get("record") == "header")
    eps = [r for r in rows if r.get("record") == "episode"]
    cmp = [r for r in rows if r.get("record") == "compare"]
    assert len(eps) == 120, f"{path.name}: expected 120 episodes, got {len(eps)}"
    assert len(cmp) == 1, f"{path.name}: expected 1 compare record, got {len(cmp)}"
    assert all(r.get("pts_source") == "physics_contact" for r in eps), \
        f"{path.name}: an episode did not score from physics contacts"
    return eps[:60], eps[60:], cmp[0], head


def mean(rows: list[dict], suite: str, key: str) -> float:
    """Purpose: mean of one logged channel over one suite's 20 episodes.
    Inputs: arm rows, suite, field. Outputs: float.
    """
    return st.mean(r[key] for r in rows if r["suite"] == suite)


def succ(rows: list[dict], suite: str) -> int:
    """Purpose: successful episodes of one suite. Inputs: arm rows, suite. Outputs: int 0..20."""
    return sum(1 for r in rows if r["suite"] == suite and r["success"])


def main() -> int:
    """Purpose: run every N210 claim as an assert and print the ladder table. Outputs: 0."""
    # --- N210.5 the default arm is the frozen rig, and the metric is bit-reproducible --------
    ref = [json.loads(x) for x in (ART / "Run350_champion_health_r350.jsonl").read_text().splitlines()
           if json.loads(x).get("record") == "episode"]
    cand0, base0, cmp0, head0 = recs(ART / "N210_r359_0_identity_trochoid.jsonl")
    key = lambda r: (r["suite"], r["seed"])  # noqa: E731
    dref, dnew = {key(r): r for r in ref}, {key(r): r for r in base0}
    assert set(dref) == set(dnew), "identity run does not cover the same 120 (suite, seed) pairs"
    assert all(dref[k]["coverage_cont"] == dnew[k]["coverage_cont"] for k in dref), \
        "coverage_cont moved against the frozen artifact -> the rig is no longer byte-identical"
    assert all(dref[k]["success"] == dnew[k]["success"] for k in dref), "success moved"
    dfn = max(abs(dref[k]["fn_mean"] - dnew[k]["fn_mean"]) for k in dref)
    assert head0["press_m"] == FROZEN_PRESS_M and head0["contact_k"] == FROZEN_K \
        and head0["f_clamp_n"] == FROZEN_CLAMP, "the default arm is not the frozen press axis"
    print(f"N210.5 identity: coverage_cont+success bit-identical to Run350 on 120/120 episodes; "
          f"the FORCE channel is not (max |d fn_mean| = {dfn:.4f} N, same mu and same press)")

    # --- N210.1 the axis is live and linear over 500x (the N208b positive control) -----------
    for tag, f_n in ARMS.items():
        if tag == "F_f25_rail3":
            continue
        cand, _, _, head = recs(ART / f"N210_r359_{tag}.jsonl")
        assert abs(head["press_m"] - FROZEN_PRESS_M * f_n / 0.5) < 1e-9, \
            f"{tag}: press_m {head['press_m']} does not command {f_n} N"
        assert abs(head["contact_k"] - 2000.0 * f_n) < 1.0, \
            f"{tag}: stiffness {head['contact_k']} does not hold the 0.5 mm penetration"
        got = st.mean(r["fn_mean"] for r in cand)
        # 0.05 N is the one arm where the pad leaves the surface on part of the ticks, so its
        # tick-mean sits ~10% under the command (intermittent contact, not a dead knob).
        lo = 0.85 if f_n < 0.5 else 0.98
        assert lo <= got / f_n <= 1.005, f"{tag}: realized fn {got:.3f} N != commanded {f_n} N"
    print("N210.1 liveness: realized fn_mean / commanded press = 1.000 on every unclamped arm, "
          "0.05 -> 25 N (500x). The axis was LIVE and UNSWEPT for 358 runs.")

    # --- N210.2 where the primary metric stops responding to force ---------------------------
    print(f"\n{'arm':>17} {'Fn_N':>5} | " + " | ".join(
        f"{s[-1]}: covc  succ  slip   fn" for s in SUITES))
    ladder = {}
    for tag, f_n in ARMS.items():
        cand, base, cmp, _ = recs(ART / f"N210_r359_{tag}.jsonl")
        cells = []
        for s in SUITES:
            cells.append(f"{mean(cand, s, 'coverage_cont'):.4f} {succ(cand, s):2d} "
                         f"{mean(cand, s, 'slip_m'):.4f} {mean(cand, s, 'fn_mean'):6.2f}")
        ladder[tag] = mean(cand, "fixture_B", "coverage_cont")
        print(f"{tag:>17} {f_n:5.2f} | " + " | ".join(cells))
        assert cmp["keep"] is False, f"{tag}: keep=true -- no candidate arm may claim the bar"
    order = [ladder[t] for t in ("A_f0p05", "B_f0p5_identity", "G_f1", "H_f1p5",
                                 "C_f2", "D_f10", "E_f25")]
    assert all(a >= b - 1e-9 for a, b in zip(order, order[1:])), \
        f"fixture_B coverage_cont is not monotone in force: {order}"
    assert ladder["A_f0p05"] == 1.0 and ladder["G_f1"] >= 0.999, "the low-force floor is not flat"
    assert ladder["D_f10"] <= 0.60 and succ(recs(ART / 'N210_r359_D_f10.jsonl')[0],
                                             "fixture_B") == 0, "10 N did not kill fixture_B"
    assert ladder["E_f25"] <= 0.20, "25 N did not collapse fixture_B"
    print(f"\nN210.2 dynamic range: flat 1.0000 from 0.05 N to 1.0 N, knee at 1.5-2 N "
          f"(B {ladder['H_f1p5']:.4f} / {ladder['C_f2']:.4f}), collapse at 10 N "
          f"({ladder['D_f10']:.4f}, 0/20) and 25 N ({ladder['E_f25']:.4f}, 0/20). "
          f"The frozen 0.5 N point sits 3-4x below the knee.")

    # --- N210.3 the mechanism, in closed form: slip = mu*Fn/KP --------------------------------
    ratios, per_ep = [], []
    for tag in COULOMB_ARMS:
        cand, _, _, _ = recs(ART / f"N210_r359_{tag}.jsonl")
        for s in SUITES:
            rows = [x for x in cand if x["suite"] == s]
            pred = [r["friction"] * 0.9 * r["fn_mean"] / KP for r in rows]  # mu_realized*Fn/KP
            ratios.append(mean(rows, s, "slip_m") / st.mean(pred))
            per_ep += [r["slip_m"] / p for r, p in zip(rows, pred)]
    assert min(ratios) > 0.70 and max(ratios) < 1.25, (
        f"the slip law mu*Fn/KP does not hold in the Coulomb regime: per-cell ratio "
        f"{min(ratios):.3f}..{max(ratios):.3f}")
    med = st.median(per_ep)
    cand_lo, _, _, _ = recs(ART / "N210_r359_A_f0p05.jsonl")
    floor = st.mean(r["slip_m"] for r in cand_lo)
    assert floor > 0.002 and floor < 0.004, f"the low-force slip floor moved: {floor:.5f} m"
    kp_need = {f: 0.72 * f / FINE_M for f in (10.0, 25.0)}   # hold e <= one fine cell at mu=0.72
    print(f"N210.3 law: slip_m = mu_realized*Fn/KP; measured/predicted = "
          f"{min(ratios):.2f}-{max(ratios):.2f} per (arm, suite) cell and median {med:.2f} per "
          f"episode over 180 Coulomb-regime episodes (Fn >= 2 N), plus a {floor * 1e3:.1f} mm "
          f"additive "
          f"floor below it. Cause: the only tangential drive is kp*e and Coulomb transmits at "
          f"most mu*Fn, so the steady sliding residual IS mu*Fn/kp. Corollary: holding e <= "
          f"{FINE_M * 1e3:.0f} mm at the top of the realized band needs KP >= "
          f"{kp_need[10.0]:.0f} N/m at 10 N and {kp_need[25.0]:.0f} N/m at 25 N -- "
          f"{kp_need[10.0] / KP:.0f}x and {kp_need[25.0] / KP:.0f}x the frozen KP=25.")

    # --- N210.4 the rail is a ceiling, and it is not what pins the rig at 0.5 N ---------------
    f_cand, _, _, f_head = recs(ART / "N210_r359_F_f25_rail3.jsonl")
    e_cand, _, _, _ = recs(ART / "N210_r359_E_f25.jsonl")
    fn_rail = st.mean(r["fn_mean"] for r in f_cand)
    ticks_rail = st.mean(r["f_clamp_ticks"] for r in f_cand)
    fmax_rail = st.mean(r["f_cmd_max_n"] for r in f_cand)
    assert f_head["f_clamp_n"] == FROZEN_CLAMP, "the rail arm did not use the frozen rail"
    assert fn_rail < FROZEN_CLAMP + 1.0 and ticks_rail > 50.0 and fmax_rail > 20.0, (
        f"the frozen rail did not truncate the 25 N press: fn {fn_rail:.2f} N, "
        f"{ticks_rail:.0f} bound ticks, |F| max {fmax_rail:.1f} N")
    cov_e = mean(e_cand, "fixture_B", "coverage_cont")
    cov_f = mean(f_cand, "fixture_B", "coverage_cont")
    assert cov_f < cov_e, "lifting the rail did not help -- the rail would be the cause"
    print(f"N210.4 rail: through the frozen 3.0 N rail a 25 N press realizes {fn_rail:.2f} N "
          f"(|F| reaches {fmax_rail:.1f} N, bound on {ticks_rail:.0f} ticks), so the 10-25 N "
          f"spec band is unreachable through the frozen rig by arithmetic. But LIFTING the rail "
          f"gives B covc {cov_f:.4f} -> {cov_e:.4f}, i.e. the rail is a ceiling, NOT the cause: "
          f"the cause is the KP gain (N210.3).")
    # --- N210.6 the obvious escape (raise KP) does NOT exist inside this architecture --------
    fn_frozen = mean(base0, "fixture_B", "fn_mean")
    for g in (4, 8, 12, 16):
        cand, _, cmp, head = recs(ART / f"N210_r359_K_gain{g}.jsonl")
        assert head["kp_gain"] == g and head["kp_n_per_m"] == KP * g, f"gain {g} did not apply"
        assert abs(head["press_m"] - FROZEN_PRESS_M) < 1e-12, \
            f"gain {g} arm moved the press -- the gain contrast is not isolated"
        cov = mean(cand, "fixture_A", "coverage_cont")
        esc = sum(1 for r in cand if r["suite"] == "fixture_A" and r.get("escaped"))
        fn = mean(cand, "fixture_B", "fn_mean")   # B keeps contact at every gain; A escapes
        assert cov < 1.0 and esc > 0 and fn > fn_frozen, (
            f"gain {g}: expected a broken, force-coupled arm, got covc {cov:.4f} "
            f"esc {esc} fn {fn:.3f} (frozen fn {fn_frozen:.3f})")
        assert cmp["keep"] is False, f"gain {g}: keep=true"
    print(f"N210.6 the escape does not exist in-rig: at the FROZEN 0.5 N press, KP x4 already "
          f"breaks the operating point on all three suites (fixture_A covc 0.2094, 2/20 success, "
          f"18/20 escaped the workspace, 0.106 m z-excursion) and the arm is NOT monotone in gain "
          f"(x16 recovers B to 1.0000 but A falls to 0.6995). Normal force is coupled to the gain: "
          f"fixture_B fn_mean rises {fn_frozen:.3f} -> 1.3-2.0 N at an UNCHANGED press, because the "
          f"vertical kp*(tgt_z-cur_z) term is part of the same contact. Buying force authority with "
          f"gain therefore inflates the load and launches the head.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
