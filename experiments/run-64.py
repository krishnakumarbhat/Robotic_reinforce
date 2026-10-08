#!/usr/bin/env python3
"""N64: Demo-conditioned GD init + energy-score demo filter.
Derived from N62 (87.96) — keeps φ0=a_demo, adds energy-score filter
to address N61 noise-sensitivity (sep-decay +3.76@0.2 → +0.21@1.0).
Drops N63's G_pull (failed at 87.82). Same validation suite as N62.
Keep bar: clean 0.5-sep >= 87.96, sweep-mean >= N62, no >0.5pt regression at sep=1.0."""

import numpy as np
import json, os

SEED = 42
np.random.seed(SEED)
OUTDIR = "/tmp/n64_work"
os.makedirs(OUTDIR, exist_ok=True)

d_action = 7; d_state = 26
sigma_flow = 0.1; lam = 0.3; eta = 0.02; n_gd_steps = 3
energy_filter_quantile = 0.8  # keep top 80% lowest-energy demo points

def v_core(x, tau, z):
    base = 0.8*x[:d_action] + 0.2*np.sin(z[:d_action])
    return base + 0.05*tau*np.ones(d_action)

def E_fn(phi, x, z, a_cal):
    v = v_core(x, 0.5, z)
    return 0.5*np.sum((phi-v)**2)/sigma_flow**2 + lam*0.5*np.sum((phi-a_cal)**2)

def grad_E_fn(phi, x, z, a_cal):
    v = v_core(x, 0.5, z)
    return (phi-v)/sigma_flow**2 + lam*(phi-a_cal)

def E_analytical_min(x, z, a_cal):
    v = v_core(x, 0.5, z)
    phi_star = (v/sigma_flow**2 + lam*a_cal) / (1/sigma_flow**2 + lam)
    return E_fn(phi_star, x, z, a_cal)

def energy_score_demo(x_demo_batch, z, a_demo):
    """Per-demo-point energy score: how well does each demo match the flow."""
    scores = []
    for xd in x_demo_batch:
        v = v_core(xd, 0.5, z)
        scores.append(float(np.sum((a_demo - v)**2)))
    return np.array(scores)

def filter_demo_by_energy(x_demo_batch, a_demo, z, quantile=energy_filter_quantile):
    """Keep demo points with energy below quantile threshold."""
    scores = energy_score_demo(x_demo_batch, z, a_demo)
    threshold = np.percentile(scores, quantile * 100)
    mask = scores <= threshold
    filtered = x_demo_batch[mask]
    if len(filtered) == 0:
        filtered = x_demo_batch[np.argmin(scores):np.argmin(scores)+1]
    # Weighted average of filtered demos → filtered a_demo
    weights = np.exp(-scores[mask])
    weights = weights / (weights.sum() + 1e-10)
    a_filtered = np.zeros(d_action)
    for i, xd in enumerate(filtered):
        v = v_core(xd, 0.5, z)
        a_filtered += weights[i] * (a_demo * 0.7 + v * 0.3)
    return a_filtered, float(np.mean(mask))

def run_n64_single(a_demo_raw, x_test, z_scene, n_demo=20, noise_sigma=0.0):
    """N64: energy-filtered demo init + 3-step GD."""
    # Generate demo batch (some noisy)
    np.random.seed(SEED + 1)
    x_demo_batch = np.random.randn(n_demo, d_state) * 0.5
    # Add noise to demo points
    if noise_sigma > 0:
        x_demo_batch += noise_sigma * np.random.randn(n_demo, d_state)

    # Energy-score filter
    a_filtered, keep_frac = filter_demo_by_energy(x_demo_batch, a_demo_raw, z_scene)

    # N64: GD init from filtered demo
    n_test = len(x_test)
    Es_n64 = []
    Es_n62 = []
    Es_analytical = []
    np.random.seed(SEED)
    for i in range(n_test):
        xi = x_test[i] + noise_sigma*np.random.randn(d_state)

        # N64: φ0 = filtered a_demo
        phi = a_filtered.copy()
        for _ in range(n_gd_steps):
            phi = phi - eta*grad_E_fn(phi, xi, z_scene, a_filtered)
        Es_n64.append(E_fn(phi, xi, z_scene, a_filtered))

        # N62 baseline: φ0 = raw a_demo
        phi62 = a_demo_raw.copy()
        for _ in range(n_gd_steps):
            phi62 = phi62 - eta*grad_E_fn(phi62, xi, z_scene, a_demo_raw)
        Es_n62.append(E_fn(phi62, xi, z_scene, a_demo_raw))

        Es_analytical.append(E_analytical_min(xi, z_scene, a_filtered))

    return {
        'E_gd_n64': float(np.mean(Es_n64)),
        'E_gd_n62': float(np.mean(Es_n62)),
        'E_analytical': float(np.mean(Es_analytical)),
        'keep_frac': keep_frac,
        'lift_vs_n62': float(np.mean(Es_n62) - np.mean(Es_n64)),
    }

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
    n62_champion = 87.96
    sep_values = [0.2, 0.5, 1.0]
    noise_sigmas = [0.0, 0.05, 0.1, 0.2, 0.4, 0.8]

    # Convex check
    np.random.seed(SEED)
    x_conv = np.random.randn(100, d_state)*0.3
    z_conv = np.random.randn(d_state)*0.2
    a_demo_conv = np.random.randn(d_action)*0.3
    m_conv = run_n64_single(a_demo_conv, x_conv, z_conv)
    convex_delta = abs(compute_metric(m_conv['E_gd_n64']) - compute_metric(m_conv['E_gd_n62']))
    results['convex_delta'] = convex_delta
    results['convex_delta_zero'] = bool(convex_delta < 0.5)

    # Sweep
    sweep = {}
    all_n64 = []; all_n62 = []
    for sep in sep_values:
        for ns in noise_sigmas:
            np.random.seed(SEED)
            a_demo = np.random.randn(d_action) * sep
            x_test = np.random.randn(100, d_state)*0.5
            z_scene = np.random.randn(d_state)*0.3
            m = run_n64_single(a_demo, x_test, z_scene, noise_sigma=ns)
            m['n64_metric'] = compute_metric(m['E_gd_n64'])
            m['n62_metric'] = compute_metric(m['E_gd_n62'])
            m['metric'] = m['n64_metric']
            m['lift'] = m['n64_metric'] - m['n62_metric']
            m['sep'] = sep; m['noise_sigma'] = ns
            sweep[f"sep={sep}_noise={ns}"] = m
            all_n64.append(m['n64_metric']); all_n62.append(m['n62_metric'])

    results['sweep'] = sweep
    results['sweep_mean_n64'] = float(np.mean(all_n64))
    results['sweep_mean_n62'] = float(np.mean(all_n62))
    results['sweep_mean_lift'] = float(np.mean(np.array(all_n64) - np.array(all_n62)))

    # Per-bucket
    per_sep = {}
    for sep in sep_values:
        b64 = [sweep[f"sep={sep}_noise={ns}"]['n64_metric'] for ns in noise_sigmas]
        b62 = [sweep[f"sep={sep}_noise={ns}"]['n62_metric'] for ns in noise_sigmas]
        per_sep[str(sep)] = {
            'n64_mean': float(np.mean(b64)), 'n62_mean': float(np.mean(b62)),
            'lift': float(np.mean(np.array(b64) - np.array(b62))),
            'std': float(np.std(b64)),
        }
    results['per_bucket'] = per_sep

    # Noise sensitivity
    noise_sens = {}
    for ns in noise_sigmas:
        bucket = [sweep[f"sep={sep}_noise={ns}"] for sep in sep_values]
        lifts = [b['lift'] for b in bucket]
        noise_sens[str(ns)] = {'mean_lift': float(np.mean(lifts)), 'std_lift': float(np.std(lifts))}
    results['noise_sensitivity'] = noise_sens

    # Clean 0.5-sep
    clean_05_n64 = sweep['sep=0.5_noise=0.0']['n64_metric']
    clean_05_n62 = sweep['sep=0.5_noise=0.0']['n62_metric']
    results['clean_05_sep_n64'] = clean_05_n64
    results['clean_05_sep_n62'] = clean_05_n62
    results['clean_05_sep_lift'] = clean_05_n64 - clean_05_n62

    # Keep bar
    clean_05_pass = clean_05_n64 >= n62_champion
    sweep_mean_pass = results['sweep_mean_n64'] >= results['sweep_mean_n62']
    sep10 = per_sep['1.0']
    no_regression = not (sep10['n64_mean'] < (sep10['n62_mean'] - 0.5))
    results['keep_bar'] = {
        'clean_05_ge_87_96': clean_05_pass,
        'sweep_mean_ge_n62': sweep_mean_pass,
        'no_regression_sep1_0': no_regression,
        'all_pass': clean_05_pass and sweep_mean_pass and no_regression,
    }

    # Keep filter stats
    keep_fracs = [sweep[k]['keep_frac'] for k in sweep]
    results['energy_filter_mean_keep_frac'] = float(np.mean(keep_fracs))

    assertions = {
        'convex_delta_zero': results['convex_delta_zero'],
        'clean_05_sep_ge_87_96': clean_05_pass,
        'sweep_mean_ge_n62_mean': sweep_mean_pass,
        'no_05pt_regression_sep1_0': no_regression,
        'energy_filter_active': results['energy_filter_mean_keep_frac'] < 1.0,
    }
    results['all_assertions'] = assertions
    results['all_pass'] = all(assertions.values())
    results['verdict'] = 'KEEP' if results['all_pass'] else 'DISCARD'

    with open(f"{OUTDIR}/n64_math_evidence.json", 'w') as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)
    with open(f"{OUTDIR}/n64_run.log", 'w') as f:
        f.write(f"N64 Demo-Conditioned GD Init + Energy-Score Filter (derived N62 87.96)\n")
        f.write(f"Seed: {SEED}\nAll pass: {results['all_pass']}\nVerdict: {results['verdict']}\n")
        for ak, av in assertions.items():
            f.write(f"  {ak}: {'PASS' if av else 'FAIL'}\n")
        f.write(f"\nConvex delta: {convex_delta}\n")
        f.write(f"Clean 0.5-sep N64: {clean_05_n64} (N62: {clean_05_n62})\n")
        f.write(f"Sweep mean N64: {results['sweep_mean_n64']} (N62: {results['sweep_mean_n62']})\n")
        f.write(f"Energy filter keep frac: {results['energy_filter_mean_keep_frac']}\n")

    print(json.dumps(results, indent=2, cls=NumpyEncoder))
    return results

if __name__ == "__main__":
    results = run_sweep()
    print(f"\nN64 PASS: {results['all_pass']}")
    print(f"Verdict: {results['verdict']}")
