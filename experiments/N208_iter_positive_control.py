"""Purpose: POSITIVE CONTROL for the N208 result. N208 measured that raising the rig's solver
ITERATION budget 16x is episode-identical, which is only evidence if the parameter actually
reaches the engine and the claim is not just a knob that PyBullet ignores.
Inputs: none. Outputs: the same single-contact pad/fixture drop at 240 Hz solved with
numSolverIterations in {1, 2, 4, 80, 1280}; the per-iteration divergence from the 1280-pass
trajectory, and the contact penetration at the end of a 12-substep tick.
Run: python3 experiments/N208_iter_positive_control.py
"""
import pybullet as p

CONTACT_K, CONTACT_C = 1.0e3, 2.0e2
MASS, PRESS, F_CLAMP = 0.080, 0.5, 3.0


def drop(iters: int, substeps: int = 12) -> list[tuple[float, float]]:
    """One press of the lightest sponge pad onto the fixture top for `substeps` 240 Hz steps.
    Returns [(z, fn)] per substep -- the same contact, pressure and step count as the rig."""
    p.connect(p.DIRECT)
    p.setGravity(0, 0, -9.81, physicsClientId=0)
    p.setTimeStep(1.0 / 240, physicsClientId=0)
    p.setPhysicsEngineParameter(numSolverIterations=iters, physicsClientId=0)
    fixture = p.createMultiBody(0, p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.1, 0.1, 0.05]),
                                basePosition=[0, 0, 0], physicsClientId=0)
    pad = p.createMultiBody(MASS, p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.025, 0.0175, 0.006]),
                            basePosition=[0, 0, 0.05 + 0.006 - 0.0005], physicsClientId=0)  # 0.5 mm sink = Fn/CONTACT_K
    for b in (fixture, pad):
        p.changeDynamics(b, -1, lateralFriction=0.4, restitution=0.0,
                         contactStiffness=CONTACT_K, contactDamping=CONTACT_C,
                         physicsClientId=0)
    out = []
    for _ in range(substeps):
        p.applyExternalForce(pad, -1, [0, 0, PRESS], [0, 0, 0], p.LINK_FRAME, physicsClientId=0)
        p.stepSimulation(physicsClientId=0)
        fn = max((c[9] for c in p.getContactPoints(bodyA=pad, bodyB=fixture, physicsClientId=0)),
                 default=0.0)   # box-on-box emits up to 4 contact points; take the largest
        out.append((round(p.getBasePositionAndOrientation(pad, physicsClientId=0)[0][2], 9),
                    round(fn, 9)))
    p.disconnect(physicsClientId=0)
    return out


def main() -> None:
    iters = (1, 2, 4, 8, 80, 1280)
    traj = {n: drop(n) for n in iters}
    ref = traj[1280]
    print("positive control: one 240 Hz press tick (12 substeps), lightest pad, 0.6 mm sink")
    print(f"{'iters':>6} {'max|dz| vs 1280':>18} {'max|dFn| vs 1280':>18} {'z_end':>12} {'fn_end':>10}")
    for n in iters:
        dz = max(abs(a[0] - b[0]) for a, b in zip(traj[n], ref))
        df = max(abs(a[1] - b[1]) for a, b in zip(traj[n], ref))
        print(f"{n:>6} {dz:>18.3e} {df:>18.3e} {traj[n][-1][0]:>12.9f} {traj[n][-1][1]:>10.6f}")
    # the parameter is live: 1 pass differs from 1280 by a margin no no-op could produce
    d1 = max(abs(a[0] - b[0]) for a, b in zip(traj[1], ref))
    assert d1 > 1e-9, f"numSolverIterations is a no-op in this build (max|dz|={d1})"
    # ...and 80 is already converged relative to 1280, which is WHY the N208 dose is inert
    d80 = max(abs(a[0] - b[0]) for a, b in zip(traj[80], ref))
    assert d80 < 1e-9, f"80 passes is NOT converged against 1280 (max|dz|={d80})"
    print(f"\nASSERT PASSED: numSolverIterations is live (1 pass differs by {d1:.3e} m) and 80 "
          f"passes is already converged ({d80:.3e} m vs 1280).\nN208's inert dose is therefore a "
          f"property of the CONTACT (one soft pad on one face, 2 bodies), not an ignored knob.")


if __name__ == "__main__":
    main()
