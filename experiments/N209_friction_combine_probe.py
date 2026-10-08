"""Purpose: measure how PyBullet COMBINES the two bodies' lateralFriction in a contact pair,
so the rig's declared friction axis can be checked against the coefficient the solver actually
uses. Inputs: none (hard-coded coefficient pairs + a lateral force ramp). Outputs: a printed
table on stdout, no files, no repo imports.
G7: this touches a synthetic 2-body scene, not a fixture episode, and it NEVER touches
coverage or success. It is a property measurement of the contact model, nothing else.
ponytail: the smallest experiment that discriminates the three candidate rules. min() and
max() are separated from the product by a single asymmetric pair (0.8 fixture / 0.1 tool):
min predicts 0.10, max predicts 0.80, product predicts 0.08.
"""
import pybullet as p  # noqa: E402

FN_N = 0.5          # normal load, N -- the rig's own 0.5 N press setpoint
RAMP_N = 0.002      # lateral force increment, N
DT = 1.0 / 1920.0   # refined rate: N208.2 -- the frozen 240 Hz under-resolves this transition
SETTLE = 600        # steps held at each force level before the slide test
SLIDE_X = 0.002     # m of lateral travel counted as a slide


def slide_force(mu_fixture: float, mu_tool: float) -> float:
    """Purpose: the lateral force at which a resting pad starts to slide on the fixture.
    Inputs: the two bodies' lateralFriction. Outputs: the slide force in N, or None.
    """
    p.connect(p.DIRECT)
    p.setGravity(0, 0, 0, physicsClientId=0)
    p.setTimeStep(DT, physicsClientId=0)
    plane = p.createMultiBody(0, p.createCollisionShape(p.GEOM_PLANE), physicsClientId=0)
    p.changeDynamics(plane, -1, lateralFriction=mu_fixture, restitution=0.0, physicsClientId=0)
    pad = p.createMultiBody(0.080, p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.025, 0.0175, 0.006]),
                            basePosition=[0, 0, 0.006], physicsClientId=0)
    p.changeDynamics(pad, -1, lateralFriction=mu_tool, restitution=0.0, spinningFriction=0.0,
                     contactStiffness=1.0e3, contactDamping=2.0e2, physicsClientId=0)
    force = 0.0
    slid = None
    while force <= 2.0:
        p.resetBasePositionAndOrientation(pad, [0, 0, 0.006], [0, 0, 0, 1], physicsClientId=0)
        p.resetBaseVelocity(pad, [0, 0, 0], [0, 0, 0], physicsClientId=0)
        for _ in range(SETTLE):
            p.applyExternalForce(pad, -1, [0, 0, -FN_N], [0, 0, 0], p.LINK_FRAME, physicsClientId=0)
            p.applyExternalForce(pad, -1, [force, 0, 0], [0, 0, 0], p.LINK_FRAME, physicsClientId=0)
            p.stepSimulation(physicsClientId=0)
            if p.getBasePositionAndOrientation(pad, physicsClientId=0)[0][0] > SLIDE_X:
                break
        if p.getBasePositionAndOrientation(pad, physicsClientId=0)[0][0] > SLIDE_X:
            slid = force
            break
        force += RAMP_N
    p.disconnect()
    return slid


FROZEN = 0.9   # the tool lateralFriction every run 1-357 shipped (the factor under audit)
GRID = (0.05, 0.20, 0.35, 0.50, 0.65, 0.80)  # fixture coefficients spanning the declared band


def combine_slope(mu_tool: float) -> float:
    """Purpose: least-squares d(realized mu)/d(mu_fixture) at a fixed tool coefficient -- the
    quantity that separates the three candidate combine rules. Inputs: mu_tool. Outputs: slope.
    min() and max() both predict a slope of exactly 1.0 here (the tool coefficient is the
    larger of the pair across this grid), so a slope near mu_tool can only be the product rule.
    """
    xs = list(GRID)
    ys = [slide_force(mu, mu_tool) / FN_N for mu in xs]  # type: ignore[operator]
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = sum((x - mx) ** 2 for x in xs)
    return num / den


if __name__ == "__main__":
    print(f"ramp probe: fn={FN_N} N, ramp={RAMP_N} N, dt={DT:.6f} s (N208.2 -- the frozen 240 Hz "
          f"under-resolves this transition)")
    print("mu_tool mu_fix  F_slide   realized   product  +offset")
    for mu_t in (0.9, 1.0):
        for mu_f in (0.05, 0.20, 0.35, 0.50, 0.65, 0.80):
            fs = slide_force(mu_f, mu_t)
            print(f"{mu_t:6.2f}  {mu_f:5.2f}  {fs:8.4f}  {fs / FN_N:8.4f}  {mu_f * mu_t:7.4f}  "
                  f"{fs / FN_N - mu_f * mu_t:+7.4f}")
        print(f"  -> slope d(realized)/d(mu_fix) = {combine_slope(mu_t):.4f}; the product rule "
              f"predicts {mu_t}, min() and max() both predict 1.0\n")

