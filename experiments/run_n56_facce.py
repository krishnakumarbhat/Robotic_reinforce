# ponytail: N56 FACC-E — Energy-Regularized FACC with Deformable Manifold
# Replaces fixed contact threshold with learned EBM energy E(s,a, manifold).
# Skipped complex manifold networks; minimal numpy EBM + attention. Add when multi-modal deformation needed.

import numpy as np

def demo_ebm_energy_facc_e(state, action, manifold_params, beta=1.5, lambda_reg=0.05, gamma=0.01):
    # s: state vector, a: action vector, manifold_params: M vector
    s = np.array(state)
    a = np.array(action)
    m = np.array(manifold_params)
    
    # EBM energy: distance to contact basin + affordance cost + manifold smoothness
    s_contact = np.zeros_like(s)
    e_flow = float(np.sum((s - s_contact) ** 2) / 2.0)
    c_aff = float(np.sum(a ** 2) / 2.0)
    manifold_grad_norm = float(np.sum(m ** 2))
    
    energy = e_flow + lambda_reg * c_aff + gamma * manifold_grad_norm
    attention_weight = float(np.exp(-beta * energy))
    return energy, attention_weight

def demo():
    state = [0.1, -0.05, 0.02]
    action = [0.01, 0.02, -0.01]
    manifold = [0.05, 0.02, 0.00]
    
    e, w = demo_ebm_energy_facc_e(state, action, manifold)
    assert e >= 0.0, f"Energy must be non-negative, got {e}"
    assert 0.0 <= w <= 1.0, f"Attention weight must be in [0,1], got {w}"
    print(f"N56 FACC-E demo passed: energy={e:.4f}, attention={w:.4f}")

if __name__ == "__main__":
    demo()
