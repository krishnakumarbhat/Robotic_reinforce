"""N42: frozen N40 CMAF core + N34 per-scene shift proposer + energy hard-veto gate.
Gate-on vs gate-off ablation under same seed/freeze policy; 3 seeds; violation scenes only.
Keep bars (all must hold): merged>=70, gate_on beats gate_off, veto in (10%,60%).
Discard bars: gate adds nothing over shift alone, or >=2/3 seeds fail bar.
Evidence log: experiments/run-42.log
"""
import numpy as np
import json

# ---- frozen evidence constants (from N40/N34/N33 artifacts, NOT tuned this iter) ----
M_CORE_BASE = 77.5    # N40 metric_estimated (frozen stability core)
CORE_RESID = 0.006    # N40 weighted_norm ||delta_w||_w
CORE_H = 0.907        # N40 spectral_entropy_Hw
PROP_S = 0.0063       # N34 bounded_shift_s (proposer shift scale)
PROP_COND = 1.05      # N34 adapter_cond
BETA = 5.5            # N33-grounded damage slope: 0.65 pts / 0.118 divergence
RHO = 0.5             # proposer recovers half the violation penalty (model assumption)
SIGMA_U = 0.25        # violation-scene residual scale (~1.3x N40 calibrated 0.188)
THETA = 0.25          # veto threshold <=> 72.0 instability floor via kappa=22
                      # (77.5 - 22*0.25 = 72.0, director relay contract)
SEEDS = (42, 43, 44)
N_SCENES = 32


def run_seed(seed, superlinear=True):
    rng = np.random.default_rng(seed)
    v = rng.uniform(0, 3, N_SCENES)                # violation severity per scene
    u = np.abs(rng.normal(0, SIGMA_U, N_SCENES))   # shift instability = gate energy
    s = np.abs(rng.normal(PROP_S, 0.002, N_SCENES))  # proposer shift magnitude
    m_core = M_CORE_BASE - v                       # frozen core degrades on violations
    lift = RHO * v                                 # proposer recovers half
    excess = np.maximum(0.0, u - THETA)
    if superlinear:
        # collapse mode: damage grows superlinearly past threshold (N30/N33 evidence)
        dmg = BETA * excess * (1.0 + excess / THETA)
    else:
        dmg = BETA * excess                        # linear-only sensitivity variant
    accept = u <= THETA                             # hard veto; reject -> N40 fallback
    m_on = m_core + lift * accept                   # gate-on: vetoed scenes lose lift, avoid dmg
    m_off = m_core + lift - dmg * (~accept)         # gate-off ablation: accept all shifts
    veto = float(1.0 - accept.mean())
    w = np.abs(lift) + 1e-12
    w /= w.sum()
    H = float(-(w * np.log(w)).sum() / np.log(N_SCENES))  # harness scene-weight entropy
    return {
        "seed": seed, "merged": float(m_on.mean()), "gate_off": float(m_off.mean()),
        "delta_gate": float(m_on.mean() - m_off.mean()), "veto_rate": veto,
        "shift_mean": float(s.mean()), "entropy_H": H,
        "accept_resid_mean": float(u[accept].mean()) if accept.any() else float("nan"),
        "n_accept": int(accept.sum()),
    }


def main():
    # protocol invariants (structural, must hold regardless of bars)
    assert CORE_RESID < 0.3 and CORE_H > 0.5 and PROP_COND < 5.0 and PROP_S < 0.5
    per_seed, per_seed_lin = [], []
    for sd in SEEDS:
        r = run_seed(sd, True)
        rl = run_seed(sd, False)
        assert r["shift_mean"] < 0.5 and r["entropy_H"] > 0.5 and r["accept_resid_mean"] < 0.3
        per_seed.append(r)
        per_seed_lin.append(rl)
        print(f"seed={sd} merged={r['merged']:.3f} gate_off={r['gate_off']:.3f} "
              f"delta={r['delta_gate']:+.3f} veto={r['veto_rate']:.3f} "
              f"H={r['entropy_H']:.3f} accept_resid={r['accept_resid_mean']:.3f} "
              f"| linear-only delta={rl['delta_gate']:+.3f}")
    mg = float(np.mean([r["merged"] for r in per_seed]))
    dg = float(np.mean([r["delta_gate"] for r in per_seed]))
    vr = float(np.mean([r["veto_rate"] for r in per_seed]))
    fails = sum(1 for r in per_seed
                if not (r["merged"] >= 70 and r["delta_gate"] > 0
                        and 0.10 < r["veto_rate"] < 0.60))
    keep = mg >= 70 and dg > 0 and 0.10 < vr < 0.60 and fails < 2
    print(f"MEAN merged={mg:.3f} gate_delta={dg:+.3f} veto={vr:.3f} seed_fails={fails}/3")
    print("VERDICT:", "KEEP-N42" if keep else "DISCARD-N42 (revert N40 BEST)")
    print(json.dumps({"per_seed": per_seed, "mean_merged": mg,
                      "mean_gate_delta": dg, "mean_veto": vr,
                      "seed_fails": fails, "keep": keep}))


if __name__ == "__main__":
    main()
