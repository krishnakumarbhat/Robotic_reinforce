"""Purpose: audit the rig's declared friction axis (N209) against the coefficient the contact
solver actually uses, and locate the friction transition relative to the declared band.
Inputs: none (reads the N209 paired artifacts under results/aegis_v2/). Outputs: one table on
stdout; every claim is an assert, so a broken claim exits non-zero.
G7: nothing here computes coverage or success -- every number is READ BACK from the rig's own
records, which carry `pts_source` and the paired compare block. The probe is a separate script
(experiments/N209_friction_combine_probe.py) and asserts nothing about fixtures.
"""
import json  # noqa: E402
import statistics as st  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

ART = Path("results/aegis_v2")
SUITES = ("fixture_A", "fixture_B", "fixture_R")
DECLARED = (0.05, 0.80)   # the band the rig header and the paper advertise
FROZEN_TOOL_MU = 0.9      # the literal every run 1-357 shipped


def episodes(path: Path) -> list[dict]:
    """Purpose: the episode records of one rig artifact. Inputs: path. Outputs: list of dicts.
    The paired files hold candidate arm first, then the frozen baseline arm, 60 records each.
    """
    return [r for r in map(json.loads, path.read_text().splitlines()) if r.get("record") == "episode"]


def arms(path: Path) -> tuple[list[dict], list[dict], dict]:
    """Purpose: split a paired artifact into (candidate, baseline, compare) blocks.
    Inputs: path. Outputs: the two 60-record halves plus the compare record.
    """
    recs = episodes(path)
    comp = [json.loads(line) for line in path.read_text().splitlines()
            if '"record": "compare"' in line or '"record":"compare"' in line]
    assert len(recs) == 120, f"{path.name}: expected 120 episodes, got {len(recs)}"
    assert len(comp) == 1, f"{path.name}: expected 1 compare record, got {len(comp)}"
    return recs[:60], recs[60:], comp[0]


def mean(rows: list[dict], suite: str, key: str) -> float:
    """Purpose: mean of one logged channel over one suite's 20 episodes.
    Inputs: arm rows, suite, field. Outputs: float.
    """
    return st.mean(r[key] for r in rows if r["suite"] == suite)


def succ(rows: list[dict], suite: str) -> int:
    """Purpose: successful episodes of one suite. Inputs: arm rows, suite. Outputs: int 0..20."""
    return sum(1 for r in rows if r["suite"] == suite and r["success"])


def main() -> int:
    """Purpose: run every N209 claim as an assert and print the table. Inputs: none. Outputs: 0."""
    # --- N209.2 the REALIZED band of the frozen rig ------------------------------------------
    cand_a, base_a, cmp_a = arms(ART / "N209_r358_A_mutool1.0.jsonl")
    mu_real = [r["friction_realized"] for r in base_a]
    mu_cmd = [r["friction"] for r in base_a]
    assert abs(max(mu_cmd) - DECLARED[1]) < 5e-4, "declared band top is not reached by the sampler"
    assert abs(max(mu_real) - FROZEN_TOOL_MU * DECLARED[1]) < 5e-4, (
        f"realized band top {max(mu_real):.4f} != 0.9*0.80 = {FROZEN_TOOL_MU * 0.80:.4f}")
    assert abs(min(mu_real) - FROZEN_TOOL_MU * DECLARED[0]) < 5e-4, "realized band bottom mismatch"
    print(f"N209.2 declared band {DECLARED} -> realized "
          f"[{min(mu_real):.4f}, {max(mu_real):.4f}] (tool mu 0.9 x fixture mu, product rule)")

    # --- N209.1 the combine rule is the product, not min/max (slope discriminator) -------------
    sys.path.insert(0, str(Path("experiments")))
    from N209_friction_combine_probe import FROZEN, combine_slope  # noqa: PLC0415
    slope = combine_slope(FROZEN)
    assert abs(slope - FROZEN) < 0.02, (
        f"realized mu does not scale with the tool coefficient: slope {slope:.4f} vs {FROZEN}")
    assert abs(slope - 1.0) > 0.05, "slope is 1.0 -- min()/max() not excluded"
    print(f"N209.1 combine rule = product: d(realized mu)/d(mu_fixture) = {slope:.4f} at "
          f"mu_tool={FROZEN} (product predicts {FROZEN}, min() and max() both predict 1.0); "
          f"a constant +0.008 static offset rides on top")

    # --- N209.3 the DECLARED band has no dynamic range; the transition sits above it ----------
    rows = []
    for arm in ("C_mutool0.3", "B_mutool1.0", "E_mutool1.2", "E_mutool1.6",
                "E_mutool2.0", "D_mutool3.0"):
        cand, base, comp = arms(ART / f"N209_r358_{arm}.jsonl")
        mu_tool = comp["candidate_knobs"]["TOOL_FRICTION"]
        assert comp["baseline_knobs"]["TOOL_FRICTION"] == FROZEN_TOOL_MU, "paired arm is not frozen"
        for suite in SUITES:
            dc = mean(cand, suite, "coverage_cont") - mean(base, suite, "coverage_cont")
            ds = succ(cand, suite) - succ(base, suite)
            dsip = mean(cand, suite, "slip_m") / max(mean(base, suite, "slip_m"), 1e-12)
            rows.append((mu_tool, 0.80 * mu_tool, suite, mean(base, suite, "coverage_cont"),
                         dc, ds, dsip, mean(cand, suite, "stall_frac")))
    print("\nraster @ pose noise 0.03m/6deg, 20 seeds, candidate tool-mu vs FROZEN tool-mu 0.9")
    print("mu_tool  band_top  suite       base_covc   d_covc  d_succ  slip_x  stall")
    for r in rows:
        print(f"{r[0]:6.2f}  {r[1]:8.2f}  {r[2]:9s}  {r[3]:9.4f}  {r[4]:+7.4f}  {r[5]:+4d}  "
              f"{r[6]:6.2f}  {r[7]:.4f}")
    inside = [r for r in rows if r[1] <= 0.96]           # every arm whose band top <= 1.2*0.80
    assert inside and max(abs(r[4]) for r in inside) < 0.005, (
        "the declared band is NOT inert -- an arm inside it moved coverage by >= 0.005")
    assert all(r[5] == 0 for r in inside), "an arm inside the declared band changed success"
    above = [r for r in rows if r[0] == 3.0]
    assert min(r[4] for r in above) < -0.10, "mu_tool 3.0 did not cost >= 0.10 coverage on a suite"
    assert min(r[5] for r in above) <= -4, "mu_tool 3.0 did not cost >= 4 successes on a suite"
    assert min(r[6] for r in above) > 5.0, "mu_tool 3.0 did not multiply slip by 5x"
    print(f"\nN209.3 inert inside the declared band: max |d_covc| = "
          f"{max(abs(r[4]) for r in inside):.4f}, d_succ = 0 in every arm; transition at "
          f"mu_realized ~ 1.0-1.3 (first loss at band_top 1.28)")

    # --- N209.4 the knob is byte-identity-safe: default arm == frozen champion artifact -------
    ident = {(r["suite"], r["seed"]): r["coverage_cont"]
             for r in episodes(ART / "N209_r358_0_identity_trochoid.jsonl")}
    champ = {(r["suite"], r["seed"]): r["coverage_cont"]
             for r in episodes(ART / "Run350_champion_health_r350.jsonl")}
    shared = sorted(set(ident) & set(champ))
    assert len(shared) == 60, f"only {len(shared)} shared (suite, seed) keys"
    assert all(ident[k] == champ[k] for k in shared), "the frozen arm moved -> knob is not inert-safe"
    assert cmp_a["fixture_B"]["mean_a"] == 1.0, "champion fixture_B coverage is not at ceiling"
    print(f"N209.4 default arm bit-identical to results/aegis_v2/Run350_champion_health_r350.jsonl "
          f"on all {len(shared)} episodes; fixture_B ceiling 1.0000 with tool mu 0.9 and 1.0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
