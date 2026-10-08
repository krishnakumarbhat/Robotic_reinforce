"""N533 PRE-PROBE (not the rig, not a metric source): how does Bullet's `contactDamping`
term move the stick/slip channel at the rig's own servo settings?

This script exists ONLY to (a) fix the SIGN and the useful RANGE of the dose ladder before
any rig run, and (b) reject a knob whose only effect is numerical noise. It replicates the
rig's control channel verbatim (KP=25 N/m, KD=1.9 N s/m, 20 Hz tick = 12 substeps at 1/240 s,
pad mass 0.080 kg, contactStiffness 1e3, frozen press 0.5 N, v_cmd 42.5 mm/s) on a single
static plane, so it is a 1-body caricature of `PyBulletScrub.run`.

No coverage, no success, no p-value is produced or claimed here. Physical truth for the
iteration remains `experiments/kaggle_aegis_sweep.py`.
"""
from __future__ import annotations

import json

import numpy as np
import pybullet as p

KP, KD = 25.0, 1.9
TICK_S = 0.05
SUBSTEPS, HZ = 12, 240.0
DT = TICK_S / SUBSTEPS
PRESS_N = 0.5
V_CMD = 0.0425
CONTACT_K = 1.0e3
MASS, HU, HV, HZ_HALF = 0.080, 0.035, 0.035, 0.010
TICKS = 400
DOSES = [0.0, 2.0e1, 2.0e2, 2.0e3, 2.0e4]
MUS = [0.05, 0.40, 0.80]


def episode(mu: float, damping: float) -> dict[str, float]:
    cid = p.connect(p.DIRECT)
    plane = p.createCollisionShape(p.GEOM_BOX, halfExtents=[2.0, 2.0, 0.05])
    p.createMultiBody(0.0, plane, basePosition=[0.0, 0.0, -0.05], physicsClientId=cid)
    pad = p.createCollisionShape(p.GEOM_BOX, halfExtents=[HU, HV, HZ_HALF])
    body = p.createMultiBody(MASS, pad, basePosition=[0.0, 0.0, HZ_HALF], physicsClientId=cid)
    for b in (plane, body):
        p.changeDynamics(b, -1, lateralFriction=mu, restitution=0.0, spinningFriction=0.0,
                         contactStiffness=CONTACT_K, contactDamping=damping, physicsClientId=cid)
    p.setGravity(0.0, 0.0, -9.81, physicsClientId=cid)
    p.setTimeStep(1.0 / HZ, physicsClientId=cid)
    p.setPhysicsEngineParameter(numSolverIterations=80, physicsClientId=cid)
    z0 = HZ_HALF
    x_cmd = 0.0
    slip, lag_max, fn_sum, z_max = [], 0.0, [], 0.0
    for t in range(TICKS):
        ori = p.getBasePositionAndOrientation(body, physicsClientId=cid)[0]
        vel = p.getBaseVelocity(body, physicsClientId=cid)[0]
        cur = ori
        tgt = np.array([x_cmd, 0.0, z0 - PRESS_N / CONTACT_K])
        force = KP * (tgt - np.array(cur)) - KD * np.array(vel)
        for _ in range(SUBSTEPS):
            p.applyExternalForce(body, -1, force.tolist(), [0.0, 0.0, 0.0], p.LINK_FRAME,
                                 physicsClientId=cid)
            p.stepSimulation(physicsClientId=cid)
        pos = p.getBasePositionAndOrientation(body, physicsClientId=cid)[0]
        contacts = p.getContactPoints(bodyA=body, physicsClientId=cid)
        slip.append(abs(x_cmd - pos[0]))
        lag_max = max(lag_max, abs(x_cmd - pos[0]))
        if contacts:
            fn_sum.append(sum(c[9] for c in contacts))
        z_max = max(z_max, float(pos[2]))
        x_cmd += V_CMD * TICK_S
    p.disconnect(physicsClientId=cid)
    return {
        "damping": damping,
        "mu": mu,
        "slip_final_m": slip[-1],
        "slip_mean_tail_m": float(np.mean(slip[-100:])),
        "lag_max_m": lag_max,
        "fn_mean_n": float(np.mean(fn_sum)) if fn_sum else 0.0,
        "z_max_m": z_max,
    }


def main() -> None:
    rows = []
    for mu in MUS:
        for d in DOSES:
            r = episode(mu, d)
            r["slip_over_cmd"] = r["slip_mean_tail_m"] / (V_CMD * TICKS)
            rows.append(r)
            print(json.dumps(r), flush=True)
    print("\n=== summary: slip_mean_tail (m) by (mu, damping) ===")
    print("mu      " + "".join(f"{d:>12.0e}" for d in DOSES))
    for mu in MUS:
        cells = [next(r["slip_mean_tail_m"] for r in rows if r["mu"] == mu and r["damping"] == d)
                 for d in DOSES]
        print(f"{mu:<8.2f}" + "".join(f"{c:12.6f}" for c in cells))
    print("\n=== summary: fn_mean (N) by (mu, damping) ===")
    print("mu      " + "".join(f"{d:>12.0e}" for d in DOSES))
    for mu in MUS:
        cells = [next(r["fn_mean_n"] for r in rows if r["mu"] == mu and r["damping"] == d)
                 for d in DOSES]
        print(f"{mu:<8.2f}" + "".join(f"{c:12.6f}" for c in cells))


if __name__ == "__main__":
    main()