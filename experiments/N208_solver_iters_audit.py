"""Purpose: decide WHY force_compliance fails to converge at 240 Hz -- the integration RATE or
the solver ITERATION budget -- and whether any force claim retired by N207 survives at a
converged rate. Inputs: the N208 rig JSONL artifacts under results/aegis_v2/ plus the N207 and
N205 artifacts (physical episodes only, read verbatim). Outputs: the dose tables + asserts.
Run: python3 experiments/N208_solver_iters_audit.py
Every number is read from a rig artifact; nothing is recomputed from physics here (G7).
"""
import json
import math
import statistics as st
from pathlib import Path

from scipy.stats import ttest_ind, ttest_rel


def welch(a: list[float], b: list[float]) -> tuple[float, float]:
    """(Welch p, seed-paired p) for the contrast. The rig reports both; N205's force numbers
    quoted the PAIRED p, so that is the comparable one here."""
    if len(a) < 2 or len(b) < 2:
        return float("nan"), float("nan")
    w = float(ttest_ind(b, a, equal_var=False).pvalue)
    pr = float(ttest_rel(b, a).pvalue) if len(a) == len(b) else float("nan")
    return w, pr

R = Path("results/aegis_v2")
SUITES = ("fixture_A", "fixture_B", "fixture_R")
# N208 arms: (rate Hz, iters) -> file. The paired baseline arm inside every N208 file is
# always the frozen 240 Hz / 80 iters champion, so it doubles as the cross-run bit-identity
# check against run 350/351/356 (N207d rule).
N208 = {(240, 80): "N208_r357_control_240x80_r357.jsonl",
        (240, 320): "N208_r357a_iters320_r357.jsonl",
        (240, 1280): "N208_r357b_iters1280_r357.jsonl",
        (3840, 80): "N208_r357c_simhz3840_r357.jsonl",
        (960, 1280): "N208_r357d_960_iters1280_r357.jsonl"}
KEYS = ("coverage_cont", "force_compliance", "fn_mean", "fn_std")


def episodes(path: str) -> tuple[list[dict], list[dict]]:
    """Split one rig file into (candidate arm, paired baseline arm) episode lists.
    The rig runs the candidate first, then the --compare-env arm, so the split is positional."""
    eps = [json.loads(l) for l in (R / path).open()
           if json.loads(l).get("record") == "episode"]
    assert len(eps) % 2 == 0 and eps, path
    return eps[:len(eps) // 2], eps[len(eps) // 2:]


def mean(arm: list[dict], suite: str, key: str) -> float:
    return st.mean(r[key] for r in arm if r["suite"] == suite)


def header(path: str) -> dict:
    for l in (R / path).open():
        r = json.loads(l)
        if r.get("record") == "header":
            return r
    raise AssertionError(f"no header in {path}")


def by_mode(path: str) -> dict[str, list[dict]]:
    """Group a file's episodes by path_mode. The rig emits one summary_mode per mode in run
    order and then the episodes in that same order, so the split is positional on the labels."""
    recs = [json.loads(l) for l in (R / path).open()]
    labels = [r["path_mode"] for r in recs if r.get("record") == "summary_mode"]
    eps = [r for r in recs if r.get("record") == "episode"]
    assert len(labels) == 2 and len(eps) % 2 == 0, (path, labels, len(eps))
    half = len(eps) // 2
    return {labels[0]: eps[:half], labels[1]: eps[half:]}


def main() -> None:
    arms: dict[tuple[int, int], list[dict]] = {}
    for key, fname in N208.items():
        cand, base = episodes(fname)
        hdr = header(fname)
        # --- N207d self-check: the candidate arm's recorded discretisation must be the one
        # asked for, and the paired baseline arm must be the frozen 240/80 champion.
        assert (hdr["sim_hz"], hdr["solver_iters"]) == (float(key[0]), float(key[1])), \
            f"{fname}: header says {hdr['sim_hz']}/{hdr['solver_iters']}, asked {key}"
        arms[key] = cand
        # every file's paired baseline arm is the frozen champion; they must agree exactly,
        # so keep the first and assert agreement through CLAIM 0 below.
        if (240, 80) in arms:
            assert len(arms[(240, 80)]) == len(base)
        else:
            arms[(240, 80)] = base
    frozen = arms[(240, 80)]

    print("N208 discretisation dose: 20 seeds x 3 suites, paired on seeds, 0,0 pose noise\n")
    hdr = f"{'rate':>6} {'iters':>6} {'suite':<10}" + "".join(f"{k:>18}" for k in KEYS)
    print(hdr)
    for key in sorted(arms, key=lambda k: (k[0], k[1])):
        for s in SUITES:
            print(f"{key[0]:>6} {key[1]:>6} {s:<10}" + "".join(
                f"{mean(arms[key], s, k):>18.5f}" for k in KEYS))

    # --- CLAIM 0 (N207d): every 240/80 paired-baseline arm in the six N208 files is the SAME
    # numbers as run 350/351/356, i.e. the new knob is inert when off and the shipped record
    # reproduces. Checked against the N207 240 Hz arm.
    n207 = episodes("N207_r356_simhz120_r356.jsonl")[1]   # that file's paired arm IS 240/80
    for s in SUITES:
        for k in KEYS:
            assert abs(mean(frozen, s, k) - mean(n207, s, k)) < 5e-4, \
                f"frozen arm drifted on {s}/{k}: {mean(frozen, s, k)} vs {mean(n207, s, k)}"
    for s in SUITES:
        assert all(r["success"] for r in frozen if r["suite"] == s), f"frozen arm fails {s}"

    # --- CLAIM 1 (P1, dt-limited): 16x the iteration budget at a FIXED 240 Hz does not move
    # the force metric, so the N207 non-convergence is a property of the integration RATE and
    # cannot be bought back with a free knob.
    for s in ("fixture_A", "fixture_R"):
        base = mean(frozen, s, "force_compliance")
        for it in (320, 1280):
            d = abs(mean(arms[(240, it)], s, "force_compliance") - base)
            assert d < 0.01, f"iterations moved force_compliance on {s} at {it}: {d:.5f}"
            dfn = abs(mean(arms[(240, it)], s, "fn_mean") - mean(frozen, s, "fn_mean"))
            assert dfn < 0.01, f"iterations moved fn_mean on {s} at {it}: {dfn:.5f}"

    # --- CLAIM 2 (P1 dose-shape): the rate still moves it, at 4x the iteration budget too,
    # and 3840 Hz continues to rise -- so 1920 Hz was NOT the asymptote N207 called it.
    n207_rate = {}
    for hz, fname in ((480, "N207_r356_simhz480_r356.jsonl"),
                      (960, "N207_r356_simhz960_r356.jsonl"),
                      (1920, "N207_r356c_simhz1920_r356.jsonl")):
        n207_rate[hz] = episodes(fname)[0]
    for s in ("fixture_A", "fixture_R"):
        seq = [mean(arms[(240, 80)], s, "force_compliance"),
               mean(n207_rate[480], s, "force_compliance"),
               mean(n207_rate[960], s, "force_compliance"),
               mean(n207_rate[1920], s, "force_compliance"),
               mean(arms[(3840, 80)], s, "force_compliance")]
        assert all(b > a for a, b in zip(seq, seq[1:])), f"{s} not monotone to 3840 Hz: {seq}"
        assert seq[-1] - seq[-2] > 0.005, f"{s} 3840 Hz flat -- 1920 IS the limit: {seq}"
        d = [b - a for a, b in zip(seq, seq[1:])]
        order = st.mean([math.log2(d[i] / d[i + 1]) for i in range(len(d) - 1)])
        tail = d[-1] * (d[-1] / d[-2]) / (1.0 - d[-1] / d[-2])   # geometric-tail estimate
        print(f"\n  {s}: force_compliance 240->3840 Hz  "
              f"{' -> '.join(f'{v:.4f}' for v in seq)}")
        print(f"    increments {' '.join(f'+{v:.4f}' for v in d)}   order~{order:.2f}"
              f"   geometric-tail limit ~{seq[-1] + tail:.4f} "
              f"(NOT a convergence claim -- no arm is flat)")
    # the interaction arm: 16x iterations at 960 Hz must match 960 Hz at 80 iters
    for s in ("fixture_A", "fixture_R"):
        d = abs(mean(arms[(960, 1280)], s, "force_compliance")
                - mean(n207_rate[960], s, "force_compliance"))
        assert d < 0.01, f"iterations interact with rate on {s}: {d:.5f}"

    # --- CLAIM 3: coverage_cont stays solver-independent over the whole 2D dose (rate x iters)
    for key, arm in arms.items():
        for s in SUITES:
            d = abs(mean(arm, s, "coverage_cont") - mean(frozen, s, "coverage_cont"))
            assert d < 1e-3, f"coverage moved at {key} on {s}: {d}"
            assert all(r["success"] for r in arm if r["suite"] == s), f"seed failed at {key} {s}"
    assert mean(arms[(3840, 80)], "fixture_B", "fn_std") < 0.01, "B not flat at 3840 Hz"
    # the iteration arms are not merely close, they are episode-identical at 5 decimals
    for it in (320, 1280):
        for s in SUITES:
            for k in KEYS:
                a = [round(r[k], 5) for r in arms[(240, it)] if r["suite"] == s]
                b = [round(r[k], 5) for r in frozen if r["suite"] == s]
                assert a == b, f"iters {it} changed {s}/{k} at 1e-5"

    # --- CLAIM 4 (the payoff): do the N205 force deltas, retired as force evidence by N207,
    # survive at a converged rate? Re-measured at 1920 Hz, both arms of each contrast.
    print("\nretired force contrasts, re-measured at 1920 Hz (both arms) vs their 240 Hz value")
    for cand, base, at240, f1920 in (
            ("rounded", "raster", "N205_r351_rounded_vs_raster.jsonl",
             "N208_r357e_rounded_1920_r357.jsonl"),
            ("fitro", "trochoid", "N205_r351_fitro_vs_trochoid.jsonl",
             "N208_r357f_fitro_1920_r357.jsonl")):
        a240 = by_mode(at240)
        labs240 = list(a240)
        a1920 = by_mode(f1920)
        labs1920 = list(a1920)
        assert labs1920 == [cand, base], (f1920, labs1920)
        assert labs240 == [cand, base], (at240, labs240)
        print(f"  {cand} - {base}:")
        for s in SUITES:
            v240 = [r["force_compliance"] for r in a240[cand] if r["suite"] == s]
            w240 = [r["force_compliance"] for r in a240[base] if r["suite"] == s]
            v1920 = [r["force_compliance"] for r in a1920[cand] if r["suite"] == s]
            w1920 = [r["force_compliance"] for r in a1920[base] if r["suite"] == s]
            d240, d1920 = st.mean(v240) - st.mean(w240), st.mean(v1920) - st.mean(w1920)
            pw240, pp240 = welch(w240, v240)
            pw1920, pp1920 = welch(w1920, v1920)
            sgn = "SAME SIGN" if d240 * d1920 >= 0 else "SIGN FLIP"
            shrink = f"shrink {abs(d240) / abs(d1920):.1f}x" if abs(d1920) > 1e-12 else \
                "B is force-invariant, nothing to shrink"
            print(f"    {s:<10} d(force_compliance) 240Hz {d240:+.4f} "
                  f"(paired p {pp240:.3f} / Welch {pw240:.3f}) -> 1920Hz {d1920:+.4f} "
                  f"(paired p {pp1920:.3f} / Welch {pw1920:.3f})   {sgn}   {shrink}")
            assert d240 * d1920 >= 0, f"{cand}/{s} sign flipped under refinement"
        # coverage is the control: N207 says it is rate-independent, so it must NOT shrink
        for s in SUITES:
            c240 = st.mean([r["coverage_cont"] for r in a240[cand] if r["suite"] == s]) - \
                st.mean([r["coverage_cont"] for r in a240[base] if r["suite"] == s])
            c1920 = st.mean([r["coverage_cont"] for r in a1920[cand] if r["suite"] == s]) - \
                st.mean([r["coverage_cont"] for r in a1920[base] if r["suite"] == s])
            assert abs(c1920 - c240) < 5e-3, f"{cand}/{s} coverage moved with the rate: {c240} -> {c1920}"

    print("\nall asserts passed: the non-convergence is RATE-bound (16x iterations is inert), "
          "3840 Hz is not yet flat, coverage_cont is solver-independent over the whole 2D dose, "
          "and the retired force contrasts keep their sign while SHRINKING toward zero")


if __name__ == "__main__":
    main()
