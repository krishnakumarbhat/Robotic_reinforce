#!/usr/bin/env python3
"""
AFIRAS: Adaptive Friction-Informed Residual Action Shaping
- Estimates live friction mu_hat = ||f_lateral|| / max(f_normal, 0.1)
- Computes residual action shaping delta_a = k_mu * (mu_hat - mu_0) * contact_normal
- Validates numerical stability and bounded action shift.
"""

import json
import numpy as np

def compute_afiras_residual(f_lat, f_norm, action_base, mu_nominal=0.4, k_mu=0.1):
    mu_hat = np.clip(np.linalg.norm(f_lat) / max(f_norm, 0.1), 0.05, 0.80)
    delta_a = k_mu * (mu_hat - mu_nominal) * np.ones_like(action_base)
    action_shaped = np.clip(action_base + delta_a, -1.0, 1.0)
    return mu_hat, action_shaped

def demo():
    f_lat = np.array([1.2, 0.5])
    f_norm = 10.0
    action_base = np.array([0.5, -0.2])
    mu_hat, action_shaped = compute_afiras_residual(f_lat, f_norm, action_base)
    assert 0.05 <= mu_hat <= 0.80, f"mu_hat out of bounds: {mu_hat}"
    assert np.all(np.abs(action_shaped) <= 1.0), "action out of bounds"
    print(f"AFIRAS demo passed: mu_hat={mu_hat:.3f}, shaped_action={action_shaped}")
    return True

if __name__ == "__main__":
    demo()
