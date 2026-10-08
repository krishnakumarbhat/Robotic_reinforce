"""Purpose: audit whether the segment-15 FORCE results survive integrator refinement.
Inputs: the N207 rig JSONL artifacts under results/aegis_v2/ (physical episodes only).
Outputs: the convergence table + asserts. Run: python3 experiments/N207_solver_audit.py
Every number below is read from a rig artifact; nothing is recomputed from physics here.
"""
import json
import math
import statistics as st
from pathlib import Path

R = Path("results/aegis_v2")
FILES = {120: "N207_r356_simhz120_r356.jsonl", 240: None, 480: "N207_r356_simhz480_r356.jsonl",
         960: "N207_r356_simhz960_r356.jsonl", 1920: "N207_r356c_simhz1920_r356.jsonl"}
SUITES = ("fixture_A", "fixture_B", "fixture_R")


def arms(path: str) -> dict[int, list[dict]]:
    """Split one rig file into (candidate=refined rate, baseline=240 Hz) episode lists.
    The rig runs the candidate arm first, then the --compare-env arm, so the split is positional."""
    eps = [json.loads(l) for l in (R / path).open() if json.loads(l).get("record") == "episode"]
    return {0: eps[:len(eps) // 2], 240: eps[len(eps) // 2:]}


def mean(arm: list[dict], suite: str, key: str) -> float:
    return st.mean(r[key] for r in arm if r["suite"] == suite)


def main() -> None:
    refined: dict[int, list[dict]] = {}
    for hz, fname in FILES.items():
        if hz == 240:
            continue
        a = arms(fname)
        refined[hz] = a[0]
        if hz == 120:
            refined[240] = a[240]
    assert set(refined) == {120, 240, 480, 960, 1920}, sorted(refined)

    print("N207 solver-convergence table (20 seeds x 3 suites, paired on seeds, G4)\n")
    hdr = f"{'rate':>6} {'suite':<10}" + "".join(f"{k:>18}" for k in
                                               ("coverage_cont", "force_compliance", "fn_mean", "fn_std"))
    print(hdr)
    for hz in (120, 240, 480, 960, 1920):
        for s in SUITES:
            print(f"{hz:>6} {s:<10}" + "".join(
                f"{mean(refined[hz], s, k):>18.5f}"
                for k in ("coverage_cont", "force_compliance", "fn_mean", "fn_std")))

    print("\nOrder of convergence (Richardson, log2 of successive-difference ratio):")
    for s in SUITES:
        for k in ("fn_mean", "force_compliance"):
            v = [mean(refined[hz], s, k) for hz in (480, 960, 1920)]
            d1, d2 = v[1] - v[0], v[2] - v[1]
            p = math.log2(abs(d1 / d2)) if d1 and d2 else float("nan")
            print(f"  {s:<10} {k:<18} order~{p:5.2f}   "
                  f"(480={v[0]:.5f} 960={v[1]:.5f} 1920={v[2]:.5f})")

    # --- the three claims this audit makes, as executable asserts ---
    # 1. coverage_cont is solver-INDEPENDENT: an 8x rate range moves it by < 0.001 everywhere,
    #    and success is 20/20 at every rate. So the certified B=1.00 is NOT an artifact.
    for hz in (120, 480, 960, 1920):
        for s in SUITES:
            d = abs(mean(refined[hz], s, "coverage_cont") - mean(refined[240], s, "coverage_cont"))
            assert d < 1e-3, f"coverage moved at {hz} Hz on {s}: {d}"
            assert all(r["success"] for r in refined[hz] if r["suite"] == s), f"seed failed at {hz} {s}"

    # 2. force_compliance is NOT converged at 240 Hz: it rises monotonically with the rate and
    #    the 240 Hz value is far below the 1920 Hz limit (A -0.415, R -0.199 measured).
    gaps = {"fixture_A": 0.415, "fixture_R": 0.199}
    for s in ("fixture_A", "fixture_R"):
        seq = [mean(refined[hz], s, "force_compliance") for hz in (120, 240, 480, 960, 1920)]
        assert all(b > a for a, b in zip(seq, seq[1:])), f"{s} not monotone in rate: {seq}"
        assert seq[-1] - seq[1] > gaps[s] - 0.02, f"{s} 240 Hz too close to the limit: {seq[1]} vs {seq[-1]}"

    # 3. the mechanism: fn_std falls with the rate (contact-force ringing is under-resolved at
    #    240 Hz) while fn_mean rises toward the 0.5 N setpoint. fixture_B is dead flat
    #    (fn_std 0.0089 at every rate) and so is the only suite whose force number was ever valid.
    for s in ("fixture_A", "fixture_R"):
        assert mean(refined[1920], s, "fn_std") < 0.5 * mean(refined[240], s, "fn_std"), s
        assert abs(mean(refined[1920], s, "fn_mean") - 0.5) < 0.02, s
    for hz in (120, 240, 480, 960, 1920):
        assert mean(refined[hz], "fixture_B", "fn_std") < 0.01, f"B not flat at {hz} Hz"
        assert mean(refined[hz], "fixture_B", "force_compliance") == 1.0, f"B not compliant at {hz}"

    print("\nall asserts passed: coverage_cont is solver-independent (claim 1 holds); "
          "force_compliance is NOT converged at 240 Hz (claims 2-3 hold)")


if __name__ == "__main__":
    main()
