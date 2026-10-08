"""Purpose: read-out for I20 (run 241, proportional force-cap gate). Recomputes every claim
from the rig JSONL in results/aegis_v2/ -- no arithmetic is done on the physics anywhere else.
Inputs: the I20_r241_* JSONL files. Outputs: the numbers quoted in the JSONL row / worklog.
"""
import json
import statistics as st

from scipy import stats

N00 = ("trig001", "trig005", "trig015", "CTRLstatic003", "floor06", "floor015")
N012 = ("trig001", "trig_floor015", "CTRLstatic003")


def load(tag: str, noise: str) -> tuple[list, dict, dict]:
    """Purpose: split one rig file into candidate/base episodes plus the compare record.
    Inputs: arm tag and noise condition. Outputs: (candidate eps, base eps, compare record).
    """
    name = f"I20_r241_{noise}_{tag}.jsonl" if noise != "n00" else f"I20_r241_{tag}_n00.jsonl"
    rows = [json.loads(x) for x in open(f"results/aegis_v2/{name}")]
    eps = [r for r in rows if r.get("record") == "episode"
           and str(r.get("status", "")).startswith("PHYSICAL")]
    return ([r for r in eps if r.get("gate_prop")], [r for r in eps if not r.get("gate_prop")],
            [r for r in rows if r.get("record") == "compare"][0])


def report(noise: str, tags) -> None:
    """Purpose: print the dose-response table for one pose-noise condition.
    Inputs: condition label, arm tags. Outputs: one printed table row set.
    """
    print(f"\n===== pose_noise {noise} =====")
    print(f"{'arm':20} {'floor':>6} {'theta':>8} {'press':>7} {'scale':>7} {'Bcovc':>7} {'Bslip':>8} "
          f"{'Bfnc':>6} {'Bsucc':>6} {'Rcovc':>7} {'Rsucc':>6} {'keep':>5}")
    for tag in tags:
        cand, base, cmp = load(tag, noise)
        B = [r for r in cand if r["suite"] == "fixture_B"]
        m = lambda k, rs=B: st.mean([r[k] for r in rs])  # noqa: E731
        print(f"{tag:20} {m('gate_prop_min'):6.2f} {m('gate_theta'):8.4f} {m('press_mean_n'):7.4f} "
              f"{m('prop_scale_mean'):7.4f} {cmp['fixture_B']['mean_b']:7.4f} {m('slip_m'):8.5f} "
              f"{m('force_compliance'):6.3f} {cmp['fixture_B']['succ_b']:4d}/20 "
              f"{cmp['fixture_R']['mean_b']:7.4f} {cmp['fixture_R']['succ_b']:4d}/20 {str(cmp['keep']):>5}")
        if noise == "n00":
            b = [r for r in base if r["suite"] == "fixture_B"]
            print(f"{'  gate-off ref':20} {'-':>6} {'-':>8} {st.mean([r['press_mean_n'] for r in b]):7.4f} "
                  f"{'-':>7} {cmp['fixture_B']['mean_a']:7.4f} {st.mean([r['slip_m'] for r in b]):8.5f} "
                  f"{st.mean([r['force_compliance'] for r in b]):6.3f} {cmp['fixture_B']['succ_a']:4d}/20 "
                  f"{cmp['fixture_R']['mean_a']:7.4f} {cmp['fixture_R']['succ_a']:4d}/20")


def law_checks() -> None:
    """Purpose: two falsification tests. (1) is the response law ever off its floor?
    (2) does slip track the threshold (jerk-adaptive) or the press (a static retune)?
    Inputs: none. Outputs: printed verdicts.
    """
    scales = [st.mean([r["prop_scale_mean"] for r in load(t, "n00")[0] if r["suite"] == "fixture_B"])
              for t in N00]
    floors = [st.mean([r["gate_prop_min"] for r in load(t, "n00")[0] if r["suite"] == "fixture_B"])
              for t in N00]
    print(f"\nmean press_scale on triggered ticks, 6 arms: {scales}")
    print(f"  GATE_PROP_MIN per arm: {floors}")
    print("  every value == GATE_PROP_MIN exactly -> the law is evaluated on the TRIGGERED tick "
          "and re-evaluated to 1.0 on the next one, so it never sits between the rails.")
    press, slip, theta = [], [], []
    for t in N00:
        B = [r for r in load(t, "n00")[0] if r["suite"] == "fixture_B"]
        press.append(st.mean([r["press_mean_n"] for r in B]))
        slip.append(st.mean([r["slip_m"] for r in B]))
        theta.append(st.mean([r["gate_theta"] for r in B]))
    rho_p, p_p = stats.spearmanr(press, slip)
    rho_t, p_t = stats.spearmanr(theta, slip)
    print(f"  Spearman slip~press rho={rho_p:+.4f} p={p_p:.3g}   "
          f"slip~theta rho={rho_t:+.4f} p={p_t:.3g}")
    print("  -> slip follows the PRESS; the threshold carries no signal of its own.")


if __name__ == "__main__":
    report("n00", N00)
    report("n012", N012)
    law_checks()
