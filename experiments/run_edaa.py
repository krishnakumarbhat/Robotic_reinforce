"""EDAA — Energy-Deformed Affordance Attention (iter 32, director proposal).
Not a controller replacement: lightweight 2k-param EBM E(o,h) from vision+
proprioceptive history outputs an SE(3) equivalence-field used as in-context
attention bias to the frozen action-expert query (pi0 flow-mat frozen).
Only deforms the conditioning manifold.

Validation: deformable / articulated contact-shift split.
Kill if no gain over fixed-affordance pi0 on slip-transfer or EBM collapses
to uniform attention (entropy ~ max).

Ponytail: 2k-param MLP EBM, SE(3) field via exp(-E) attention bias,
frozen flow-mat kept intact. Global lock and O(n^2) scan named explicitly.
"""
from __future__ import annotations

import math
import random

# ponytail: 2k-param add-on; global EBM (per-episode, not per-account); if
# throughput needs scale, upgrade to per-account EBM + thread-local cache.
EBM_HIDDEN = 32   # ~2k params with input=8, output=4
EBM_EPS = 1e-4


class LightEBM:
    """Purpose: minimal energy-based model E(o,h) -> SE(3) equivalence field.
    Inputs: vision+proprio history (8-d: pose noise + friction + phase).
    Outputs: (x, y, z, yaw) SE(3) offset field + energy scalar.
    """

    def __init__(self, seed: int = 42):
        rng = random.Random(seed)
        # 2k param budget: W1 [8x32] + b1 [32] + W2 [32x4] + b2 [4] = 256+32+128+4=420
        # plus energy head W_e [32x1] + b_e [1] = 33 -> total ~453 params (under 2k)
        self.W1 = [[rng.uniform(-0.05, 0.05) for _ in range(EBM_HIDDEN)]
                   for _ in range(8)]
        self.b1 = [0.0] * EBM_HIDDEN
        self.We = [[rng.uniform(-0.05, 0.05) for _ in range(EBM_HIDDEN)]
                   for _ in range(1)]
        self.be = [0.0]
        # equivalence field weights: SE(3) bias projection
        self.Wfield = [[rng.uniform(-0.02, 0.02) for _ in range(EBM_HIDDEN)]
                       for _ in range(4)]  # 4 -> SE(3) (x,y,z,yaw)
        self.bfield = [0.0, 0.0, 0.0, 0.0]

    def energy(self, hist: list[float]) -> float:
        # simple MLP forward
        h = hist[:8] + [0.0] * max(0, 8 - len(hist))
        z = [sum(h[i] * self.W1[i][j] for i in range(8)) + self.b1[j]
             for j in range(EBM_HIDDEN)]
        z = [max(-5.0, min(5.0, val)) for val in z]  # bounded tanh-like
        e = sum(z[i] * self.We[0][i] for i in range(EBM_HIDDEN)) + self.be[0]
        return float(math.tanh(e))  # bounded energy [-1,1]

    def se3_field(self, hist: list[float]) -> tuple[float, float, float, float]:
        # ponytail: lightweight projection; if manifold-slip demands adaptive
        # rotation, upgrade to full SE(3) exponential map.
        h = hist[:8] + [0.0] * max(0, 8 - len(hist))
        z = [sum(h[i] * self.W1[i][j] for i in range(8)) + self.b1[j]
             for j in range(EBM_HIDDEN)]
        z = [max(-5.0, min(5.0, val)) for val in z]
        f = [sum(z[i] * self.Wfield[j][i] for i in range(EBM_HIDDEN))
             + self.bfield[j] for j in range(4)]
        # SE(3) equivalence field: small deformation (scale 0.02 m / 0.05 rad max)
        return (float(math.tanh(f[0]) * 0.02),
                float(math.tanh(f[1]) * 0.02),
                float(math.tanh(f[2]) * 0.005),
                float(math.tanh(f[3]) * 0.05))

    def attention_uniformity(self, hist: list[float]) -> float:
        # ponytail: entropy over 4-D field as proxy for uniform-attention collapse.
        f = self.se3_field(hist)
        # entropy of normalized positive weights
        w = [max(0.0, x) for x in f] + [max(0.0, -x) for x in f]
        s = sum(w) + EBM_EPS
        p = [x / s for x in w]
        entropy = -sum(pi * math.log(pi + EBM_EPS) for pi in p if pi > 0)
        max_ent = math.log(len(p) + 1e-9)
        return float(entropy / max_ent)  # 1.0 = uniform collapse


def edaa_condition(spec: dict, seed: int) -> tuple:
    """Purpose: apply EDAA conditioning manifold deformation to fixture spec.
    Inputs: fixture spec dict, seed. Outputs: (deformed_offset_xyz, field_energy,
    attention_uniformity, se3_equivalence_field).
    """
    # Vision + proprioceptive history proxy: pose noise, friction, phase, tool
    noise_cfg = spec.get("pose_noise_cfg", "0,0")
    st, sr = (float(x) for x in noise_cfg.split(","))
    history = [float(st), float(sr), float(spec.get("friction", 0.35)),
               float(spec.get("offset_cm", 0) / 100), float(spec.get("angle_deg", 0)) /
               float(max(st, 0.01) + 0.01), float(spec.get("seed_offset", seed) % 10) / 10.0,
               float(spec.get("phase", 0)), float(random.Random(seed).uniform(0, 1))]
    ebm = LightEBM(seed=seed)
    field = ebm.se3_field(history)
    energy = ebm.energy(history)
    uniformity = ebm.attention_uniformity(history)
    # Deform the fixture offset (manifold deformation, not teleport)
    original_offset = [float(spec.get("offset_cm", 0)) / 100.0,
                       float(spec.get("offset_y_cm", 0)) / 100.0, 0.0]
    deformed = [original_offset[0] + field[0],
                original_offset[1] + field[1],
                original_offset[2] + field[2]]
    return (deformed, float(energy), float(uniformity), field)
