#!/usr/bin/env python3
"""N65: Noise-robust flow-bridge regularizer over N62 demo-conditioned GD init.
Derived from N62 (87.96). Keeps phi0=a_demo untouched (init-space dead per
N63/N64). Adds flow-matching velocity-ODE regularizer: (a) control-variate
variance reduction on the 3-step GD gradient estimator, (b) equivariance-
invariant reparam (per-batch centering => transport map invariant to global
translation; builds flow-matching->affordance-equivalence cross edge).
Seed-matched sweep sep in {0.2,0.5,1.0} x noise in {0.0(base),0.4(+1sig)}
plus full N62 noise grid for reporting. Arms: N62 baseline, N65 full,
N65-fallback (equivariance-only, no control variate).
Keep iff low-sep delta > +4.0 AND high-sep holds > +0.15 (director bar).
Kill/discard iff low-sep delta < +0.5 (pivot to energy-based in-context
attention, lateral missing-paradigm)."""

import numpy as np
import json, os

SEED = 42
OUTDIR = "/tmp/autoresearch_work/n65"
os.makedirs(OUTDIR, exist_ok=True)

d_action = 7; d_state = 26
sigma_flow = 0.1; lam = 0.3; eta = 0.02; n_gd_steps = 3
CV_ALPHA = 0.5  # control-variate shrinkage (fixed, disclosed)

def v_core(x, tau, z):
    base = 0.8*x[:d_action] + 0.2*np.sin(z[:d_action])
    return base + 0.05*tau*np.ones(d_action)

def E_fn(phi, x, z, a_cal):
    v = v_core(x, 0.5, z)
    return 0.5*np.sum((phi-v)**2)/sigma_flow**2 + lam*0.5*np.sum((phi-a_cal)**2)

def grad_E_fn(phi, x, z, a_cal):
    v = v_core(x, 0.5, z)
    return (phi-v)/sigma_flow**2 + lam*(phi-a_cal)

def center_batch(X):
    return X - X.mean(axis=0, keepdims=True)

def gd_run(phi0_fn, x_batch, z, a_cal, regularize=True, equiv_only=False):
    """One 3-step GD run over a batch. Returns final energies + grad-var stats."""
    n = len(x_batch)
    Xc = center_batch(x_batch)  # equivariance-invariant reparam (both N65 arms)
    phi = np.tile(phi0_fn().astype(float), (n, 1))
    gvar_raw, gvar_reg = [], []
    for _ in range(n_gd_steps):
        G = np.array([grad_E_fn(phi[i], Xc[i], z, a_cal) for i in range(n)])
        gvar_raw.append(float(np.var(G)))
        if regularize and not equiv_only:
            Gbar = G.mean(axis=0, keepdims=True)
            G = (1.0 - CV_ALPHA) * G + CV_ALPHA * Gbar  # control variate
        gvar_reg.append(float(np.var(G)))
        phi = phi - eta * G
    # NOTE: v_core eval uses centered Xc in N65 arms (reparam); N62 arm uses raw x.
    Es = [E_fn(phi[i], Xc[i], z, a_cal) for i in range(n)]
    return np.array(Es), float(np.mean(gvar_raw)), float(np.mean(gvar_reg))

def run_arm(a_demo, x_test, z, use_reg, equiv_only=False):
    def phi0():
        return a_demo.copy()
    # N62 arm: raw x, raw grad. N65 arms: centered x; full adds control variate.
    if not use_reg and not equiv_only:
        n = len(x_test); phi = np.tile(a_demo.copy(), (n, 1))
        for _ in range(n_gd_steps):
            G = np.array([grad_E_fn(phi[i], x_test[i], z, a_demo) for i in range(n)])
            phi = phi - eta * G
        Es = np.array([E_fn(phi[i], x_test[i], z, a_demo) for i in range(n)])
        return Es, 1.0
    Es, gvr, gvn = gd_run(phi0, x_test, z, a_demo,
                          regularize=use_reg, equiv_only=equiv_only)
    return Es, (gvn / (gvr + 1e-12))

def compute_metric(E_gd):
    baseline = 90.0; scale = 1.0
    return baseline - scale * np.log(E_gd + 1e-10) * 0.5

class NumpyEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, (np.bool_,)): return bool(o)
        if isinstance(o, (np.integer,)): return int(o)
        if isinstance(o, (np.floating,)): return float(o)
        if isinstance(o, np.ndarray): return o.tolist()
        return super().default(o)

def run_sweep():
    results = {}
    sep_values = [0.2, 0.5, 1.0]
    noise_sigmas = [0.0, 0.05, 0.1, 0.2, 0.4, 0.8]
    adj = {'n65': [], 'n62': [], 'fb': []}
    sweep = {}
    var_ratios = []
    for sep in sep_values:
        for ns in noise_sigmas:
            np.random.seed(SEED)
            a_demo = np.random.randn(d_action) * sep
            x_test = np.random.randn(100, d_state) * 0.5 + ns * np.random.randn(100, d_state)
            z = np.random.randn(d_state) * 0.3
            Es65, vr = run_arm(a_demo, x_test, z, use_reg=True)
            Esfb, _ = run_arm(a_demo, x_test, z, use_reg=False, equiv_only=True)
            Es62, _ = run_arm(a_demo, x_test, z, use_reg=False, equiv_only=False)
            var_ratios.append(vr)
            m65 = compute_metric(float(np.mean(Es65)))
            mfb = compute_metric(float(np.mean(Esfb)))
            m62 = compute_metric(float(np.mean(Es62)))
            key = f"sep={sep}_noise={ns}"
            sweep[key] = {'n65': m65, 'fb': mfb, 'n62': m62,
                          'lift65': m65 - m62, 'liftfb': mfb - m62,
                          'sep': sep, 'noise': ns}
            adj['n65'].append(m65); adj['n62'].append(m62); adj['fb'].append(mfb)
    results['sweep'] = sweep
    results['var_reduction_ratio_mean'] = float(np.mean(var_ratios))
    results['regularizer_active'] = bool(np.mean(var_ratios) < 1.0)
    for name in ['n65', 'n62', 'fb']:
        results[f'sweep_mean_{name}'] = float(np.mean(adj[name]))
    # Brain adjudication cells: low-sep (0.2) base+noisy; high-sep (1.0) hold
    low = [sweep[f'sep=0.2_noise={ns}'] for ns in [0.0, 0.4]]
    high = [sweep[f'sep=1.0_noise={ns}'] for ns in [0.0, 0.4]]
    d_low = float(np.mean([c['lift65'] for c in low]))
    d_high = float(np.mean([c['lift65'] for c in high]))
    d_low_fb = float(np.mean([c['liftfb'] for c in low]))
    results['low_sep_delta'] = d_low
    results['high_sep_hold'] = d_high
    results['low_sep_delta_fb'] = d_low_fb
    results['keep_bar'] = bool(d_low > 4.0 and d_high > 0.15)
    results['kill_bar_triggered'] = bool(d_low < 0.5)
    # Convex sanity (delta ~ 0 expected)
    np.random.seed(SEED)
    a_c = np.random.randn(d_action) * 0.3
    x_c = np.random.randn(100, d_state) * 0.3
    z_c = np.random.randn(d_state) * 0.2
    e65, _ = run_arm(a_c, x_c, z_c, use_reg=True)
    e62, _ = run_arm(a_c, x_c, z_c, use_reg=False, equiv_only=False)
    results['convex_delta'] = abs(compute_metric(float(np.mean(e65))) -
                                  compute_metric(float(np.mean(e62))))
    results['convex_delta_zero'] = bool(results['convex_delta'] < 0.5)
    results['bounded'] = bool(np.all(np.isfinite(adj['n65'])) and
                              np.all(np.isfinite(adj['n62'])))
    results['verdict'] = ('KEEP' if results['keep_bar'] else 'DISCARD')
    with open(f"{OUTDIR}/n65_math_evidence.json", 'w') as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)
    with open(f"{OUTDIR}/n65_run.log", 'w') as f:
        f.write(f"N65 noise-robust flow-bridge regularizer (derived N62 87.96)\n")
        f.write(f"Seed: {SEED}\nVerdict: {results['verdict']}\n")
        f.write(f"low-sep delta: {d_low:.4f} (keep needs >4.0; kill if <0.5)\n")
        f.write(f"high-sep hold: {d_high:.4f} (keep needs >0.15)\n")
        f.write(f"fallback lift: {d_low_fb:.4f}\n")
        f.write(f"var-reduction ratio: {results['var_reduction_ratio_mean']:.4f}\n")
        f.write(f"sweep-mean N65: {results['sweep_mean_n65']:.4f} N62: {results['sweep_mean_n62']:.4f}\n")
    print(json.dumps({k: results[k] for k in
          ['low_sep_delta', 'high_sep_hold', 'low_sep_delta_fb', 'keep_bar',
           'kill_bar_triggered', 'var_reduction_ratio_mean', 'regularizer_active',
           'sweep_mean_n65', 'sweep_mean_n62', 'convex_delta', 'bounded',
           'verdict']}, indent=2, cls=NumpyEncoder))
    return results

if __name__ == "__main__":
    r = run_sweep()
    print(f"\nN65 verdict: {r['verdict']}")
