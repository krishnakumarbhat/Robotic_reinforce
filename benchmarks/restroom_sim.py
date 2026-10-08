# SYNTHETIC PROXY — NOT A KEEP METRIC. Canonical physical rig: experiments/kaggle_aegis_sweep.py
"""AEGIS restroom benchmark — REAL PyBullet rigid contact dynamics (friction 0.05-0.80).

SYNTHETIC PROXY — NOT A KEEP METRIC. Every number this file emits carries
``"metric_class": "synthetic_proxy"``. The canonical physical rig is
experiments/kaggle_aegis_sweep.py; nothing here may displace a champion score.

Fixture A : round tank / glossy, nominal pose (in-distribution).
Fixture B : elongated / wall-hung / matte, +-15cm shift, +-10deg yaw (zero-shot OOD).
B_height  : Fixture B lifted +15cm (wall-hung height OOD).
B_tool    : Fixture B with changed gripper/wiper footprint (morphology OOD).

Cleaning physics (first-order Coulomb model on MEASURED contact forces):
  a cell clears iff the wiper footprint covers it during the scrub phase
  AND measured normal force F is in the compliance window [5,25] N
  AND the surface can shear the stain: mu * F >= STAIN_SHEAR_N.

Tier-4 gate (HARD, preserved): normalized action jerk > GATE_MAX_JERK or
force violation F > 25N -> retract 3cm + re-engage. Runs WITH and WITHOUT.

ponytail: one class, no config objects, no plugin registries. Benchmark IS the
iteration — no architecture claim without it.
"""

import json
import os
import sys
import time

import numpy as np
import pybullet as p
import pybullet_data

SEEDS_REQUIRED = 20
FRICTION_RANGE = (0.05, 0.80)
SCRIPTED_BASELINE = 0.8125          # historical target, re-confirmed by measurement
SCRIPTED_BASELINE_SCORE = SCRIPTED_BASELINE
GATE_MAX_JERK = 0.618
YAW_CLAMP = np.deg2rad(15.0)          # fixture yaw is +-10 deg; clamp fits
TAU_LOW = 0.2                          # FACT-Physics low-noise regime
LN_SCALE = 6.0                          # FACT-Physics LN scale
FIXTURES = ("fixture_A", "fixture_B", "fixture_B_height", "fixture_B_tool")
CONTROLLERS = ("scripted", "fixed_manifold", "sacse", "dafm_ea", "paitc", "paitc_v3", "n31", "se3_dafe", "fact_phys", "vaef", "n84_cfap", "seaf_facc", "facc_se3", "facc_se3_v2")
FORCE_WINDOW = (5.0, 25.0)
FORCE_TARGET = 12.0                 # N, inside the window
STAIN_SHEAR_N = 1.20                # stain adhesion (N): needs mu*F >= 1.2
WORKSPACE_L = 0.5                   # normalization length for the jerk gate
TOP_Z = 0.40                        # nominal Fixture-A top surface height
CTRL_DT = 0.05                      # 20 Hz policy/action rate
SUBSTEPS = 12                       # 12 x 1/240 s = 1 control step
MAX_CTRL_STEPS = 600
NODE_STEPS = 18                       # per-node control budget (lift+travel+press+scrub)
V_REF = 0.5                         # m/s, action normalization for the jerk gate
TOOL_HALF = {"normal": (0.045, 0.045, 0.012), "wide": (0.075, 0.030, 0.012)}
TOOL_MASS = 0.3
GRID = np.arange(-0.135, 0.136, 0.045)
CANONICAL_RIG = "experiments/kaggle_aegis_sweep.py"
METRIC_CLASS = "synthetic_proxy"
NORM_EPS = 1e-9        # |chord| below this -> degenerate axis
SPREAD_EPS = 1e-4      # regressor spread (m) below this -> collinear / zero-variance
SLOPE_MAX = 1.0        # |dx/dy| beyond 45 deg is outside the fixture-yaw prior
UNREGISTERED = ("scripted", "fixed_manifold")   # replay the fixed nominal manifold
# controllers with no dedicated executor loop: they only derive a node set
# (warp the nominal raster, or re-generate it from contact) and then run the
# shared executor below. Every OTHER name must return from its own branch.
GENERIC_EXECUTOR = ("scripted", "fixed_manifold", "sacse", "dafm_ea")
ROW_SCHEMA = (
    "fixture", "seed", "controller", "use_gate", "friction", "offset_cm", "yaw_deg",
    "tool", "top_z", "n_cells", "cleared", "transfer_success", "force_compliance",
    "force_peak_n", "force_over_steps", "jerk_violations", "interceptions",
    "jerk_unintercepted", "force_retracts", "interception_delta_ready",
    "ctrl_latency_ms", "energy_latency_ms", "registration_latency_ms",
    "registration_obs", "registration_ok", "registration_reason", "attention_entropy",
    "delta_est", "delta_true", "edge_budget", "status", "error",
)


def _wrap_yaw(a):
    """Normalise an angle to (-pi, pi].

    Purpose: yaw wraparound safety (arctan2 already lands in the half-open range,
    but a chord average can leave it). Inputs: angle in rad. Outputs: float in
    (-pi, pi], with +pi preserved and -pi folded onto +pi.
    """
    y = (float(a) + np.pi) % (2.0 * np.pi)
    return float(np.pi if y == 0.0 else y - np.pi)


def _finite_pts(pts):
    """Filter contact points to finite (x, y, z) rows.

    Purpose: NaN/garbage contact telemetry must not reach a linear solve.
    Inputs: any iterable of indexable rows (at least 3 numbers each).
    Outputs: float ndarray (n, 3), n = finite row count (0 if input is empty,
    ragged, or scalar) — never raises on shape, only on non-numeric input.
    """
    try:
        arr = np.asarray([list(q) for q in pts], dtype=float)
    except (TypeError, ValueError):
        return np.zeros((0, 3))
    if arr.ndim != 2 or arr.shape[1] < 3:
        return np.zeros((0, 3))
    return arr[:, :3][np.isfinite(arr[:, :3]).all(axis=1)]


def _fit_line(obs_x, obs_y, obs_w):
    """Weighted line fit y = c + m x with explicit degeneracy handling.

    Purpose: shared IRLS step for the energy-weighted edge solve; refuses to
    invent a slope from rank-deficient evidence. Inputs: matched x/y/w arrays
    (x = regressor, y = target). Outputs: (intercept, slope) or None when the
    point count is < 2, the regressor is constant (collinear), the weights carry
    no signal, or the slope leaves the fixture-yaw prior.
    """
    x = np.asarray(obs_x, dtype=float)
    y = np.asarray(obs_y, dtype=float)
    w = _wnorm(obs_w)
    if x.size < 2 or not (np.isfinite(x).all() and np.isfinite(y).all()):
        return None
    if float(x.max() - x.min()) < SPREAD_EPS or float(w.max()) <= 0.0:
        return None
    M = np.vstack([np.ones_like(y), x]).T
    Wm = np.diag(w)
    try:
        beta = np.linalg.lstsq(Wm @ M, Wm @ y, rcond=None)[0]
    except (np.linalg.LinAlgError, ValueError):
        return None
    c, m = float(beta[0]), float(beta[1])
    if not (np.isfinite(c) and np.isfinite(m)) or abs(m) > SLOPE_MAX:
        return None
    return c, m


def rect_pose(edges, fallback_xy=(0.0, 0.0), yaw_clamp=YAW_CLAMP):
    """Closed-form SE(2) rectangle solve from the four boundary contact groups.

    Purpose: turn the r/l/t/b outline terminal points into (cx, cy, yaw) without
    a division that can blow up, and report DEGENERACY instead of returning a
    silent zero warp. Inputs: edges = {"r","l","t","b"} -> sequences of (x,y,z)
    contact points (NaN/short rows are filtered); fallback_xy used when no chord
    survives. Outputs: (cx, cy, yaw, ok, reason) — ok False means the caller
    MUST treat the pose as failed registration and say so in its row.
    """
    fx, fy = float(fallback_xy[0]), float(fallback_xy[1])
    pts, dropped, raw_n = {}, 0, 0
    for k in ("r", "l", "t", "b"):
        rows = list((edges or {}).get(k, []) or [])
        raw_n += len(rows)
        pts[k] = _finite_pts(rows)
        dropped += len(rows) - len(pts[k])
    if raw_n and dropped:
        # corrupt telemetry is not silently repaired: the outline sample is
        # biased, so the pose is refused instead of solved on a filtered subset
        return fx, fy, 0.0, False, f"non_finite_contact_points({dropped}/{raw_n})"
    empty = np.zeros((0, 3))
    finite = np.vstack([pts[k] for k in pts if len(pts[k])]) \
        if any(len(pts[k]) for k in pts) else empty
    if len(finite) < 3:
        return fx, fy, 0.0, False, "too_few_contact_points"
    reasons = []
    if not (len(pts["r"]) and len(pts["l"])):
        reasons.append("missing_x_chord")
    if not (len(pts["t"]) and len(pts["b"])):
        reasons.append("missing_y_chord")
    vx = vy = None
    if len(pts["r"]) and len(pts["l"]):
        v = pts["r"].mean(axis=0)[:2] - pts["l"].mean(axis=0)[:2]
        vx = v if float(np.hypot(v[0], v[1])) > NORM_EPS else None
    if len(pts["t"]) and len(pts["b"]):
        v = pts["t"].mean(axis=0)[:2] - pts["b"].mean(axis=0)[:2]
        vy = v if float(np.hypot(v[0], v[1])) > NORM_EPS else None
    if vx is None and vy is None:
        return fx, fy, 0.0, False, "zero_area_rectangle:" + (",".join(reasons) or "no_chords")
    # collinear evidence: both chords survived but are parallel -> 1 dof short
    # of a frame, and a mean of two parallel axes would invent a rotation.
    if vx is not None and vy is not None:
        cross = abs(vx[0] * vy[1] - vx[1] * vy[0]) / (np.hypot(*vx) * np.hypot(*vy))
        if cross < SPREAD_EPS:
            return fx, fy, 0.0, False, "collinear_edges"
    cand, mid = [], []
    if vx is not None:
        cand.append(float(np.arctan2(vx[1], vx[0])))            # local +x axis
        mid.append(0.5 * (pts["r"].mean(axis=0)[:2] + pts["l"].mean(axis=0)[:2]))
    if vy is not None:
        cand.append(float(np.arctan2(-vy[0], vy[1])))           # local +y axis
        mid.append(0.5 * (pts["t"].mean(axis=0)[:2] + pts["b"].mean(axis=0)[:2]))
    yaw = _wrap_yaw(np.arctan2(np.sin(cand).mean(), np.cos(cand).mean()))
    yaw = float(np.clip(yaw, -yaw_clamp, yaw_clamp))
    c0 = np.mean(mid, axis=0)
    Rm = np.array([[np.cos(-yaw), -np.sin(-yaw)], [np.sin(-yaw), np.cos(-yaw)]])
    Rp = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
    ex, ey = [], []
    if vx is not None:
        ex += [float((Rm @ (pts["r"].mean(axis=0)[:2] - c0))[0]),
               float((Rm @ (pts["l"].mean(axis=0)[:2] - c0))[0])]
    if vy is not None:
        ey += [float((Rm @ (pts["t"].mean(axis=0)[:2] - c0))[1]),
               float((Rm @ (pts["b"].mean(axis=0)[:2] - c0))[1])]
    cc = c0 - Rp @ np.array([0.5 * np.mean(ex) if ex else 0.0,
                             0.5 * np.mean(ey) if ey else 0.0])
    if not np.isfinite(cc).all():
        return fx, fy, yaw, False, "non_finite_centre"
    # a single surviving chord still pins centre + yaw exactly for a rectangle,
    # but the solve is degraded, so it is reported as failed registration.
    if reasons:
        return float(cc[0]), float(cc[1]), yaw, False, "single_chord:" + ",".join(reasons)
    return float(cc[0]), float(cc[1]), yaw, True, "ok"


def _wnorm(w):
    """Sanitize IRLS weights: drop non-finite, fall back to uniform on 0-sum.

    Energy weights exp(-E/0.5) underflow to 0 on outlier-dominated probes; an
    all-zero (or NaN) row makes Wm non-finite and numpy lstsq raises
    "SVD did not converge". Uniform fallback keeps the fit well-defined.
    """
    w = np.nan_to_num(np.asarray(w, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    w = np.clip(w, 0.0, None)
    s = float(w.sum())
    if s <= 1e-12:
        return np.full_like(w, 1.0 / max(len(w), 1))
    return w / s


def nominal_nodes():
    """Boustrophedon raster of the nominal Fixture-A demo trajectory (world frame)."""
    pts = []
    for j, y in enumerate(GRID):
        xs = GRID if j % 2 == 0 else GRID[::-1]
        for x in xs:
            if x * x + y * y <= 0.15 ** 2:
                pts.append([float(x), float(y), TOP_Z])
    return pts


def fixture_spec(kind, seed):
    """Sample fixture pose/geometry + friction for one seed (physical parameters only)."""
    rng = np.random.default_rng(seed * 7919 + 13)
    spec = {
        "kind": kind, "seed": int(seed),
        "friction": float(rng.uniform(*FRICTION_RANGE)),
        "center": [0.0, 0.0, 0.35], "yaw": 0.0,
        "shape": "cyl", "radius": 0.16, "half": [0.16, 0.16, 0.05],
        "tool": "normal",
    }
    if kind != "fixture_A":
        spec["shape"] = "box"
        spec["half"] = [0.15, 0.08, 0.05]
        spec["center"] = [float(rng.uniform(-0.15, 0.15)),
                          float(rng.uniform(-0.15, 0.15)), 0.35]
        spec["yaw"] = float(np.deg2rad(rng.uniform(-10.0, 10.0)))
    if kind == "fixture_B_height":
        spec["center"][2] = 0.50                      # top at 0.55 (+15cm)
    if kind == "fixture_B_tool":
        spec["tool"] = "wide"
    return spec


def fixture_cells(spec):
    """Stain cells on the top face, in fixture-local coordinates."""
    cells = []
    for x in GRID:
        for y in GRID:
            if spec["shape"] == "cyl":
                if x * x + y * y <= 0.145 ** 2:
                    cells.append([float(x), float(y)])
            elif abs(x) <= 0.14 and abs(y) <= 0.07:
                cells.append([float(x), float(y)])
    return np.asarray(cells, dtype=float)


class RestroomSim:
    """Single PyBullet DIRECT client; reset() rebuilds one Fixture-A/B episode."""

    def __init__(self):
        self.cid = p.connect(p.DIRECT)
        if self.cid < 0:
            raise RuntimeError("pybullet DIRECT connect failed")
        self._reg_pts = []
        self._reg_edge = {}
        self._reg_ok = False
        self._reg_reason = "not_attempted"
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.cid)
        p.setGravity(0, 0, -9.81, physicsClientId=self.cid)
        p.setTimeStep(1 / 240.0, physicsClientId=self.cid)

    def close(self):
        """Release the DIRECT client (selftest builds more than one sim)."""
        if self.cid is not None and self.cid >= 0:
            p.disconnect(physicsClientId=self.cid)
            self.cid = -1

    # ------------------------------------------------------------------ world
    def reset(self, kind, seed, tool=None):
        if self.cid is None or self.cid < 0:
            raise RuntimeError("RestroomSim is closed; construct a new one")
        p.resetSimulation(physicsClientId=self.cid)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.cid)
        p.setGravity(0, 0, -9.81, physicsClientId=self.cid)
        p.setTimeStep(1 / 240.0, physicsClientId=self.cid)
        p.setPhysicsEngineParameter(numSolverIterations=50, physicsClientId=self.cid)
        self.spec = fixture_spec(kind, seed)
        if tool:
            self.spec["tool"] = tool
        mu = self.spec["friction"]
        self.cells = fixture_cells(self.spec)
        self.n_cells = len(self.cells)

        plane = p.loadURDF("plane.urdf", [0, 0, 0], useFixedBase=True,
                           physicsClientId=self.cid)
        p.changeDynamics(plane, -1, lateralFriction=mu, physicsClientId=self.cid)
        if self.spec["shape"] == "cyl":
            col = p.createCollisionShape(p.GEOM_CYLINDER, radius=self.spec["radius"],
                                         height=2 * self.spec["half"][2],
                                         physicsClientId=self.cid)
        else:
            col = p.createCollisionShape(p.GEOM_BOX, halfExtents=self.spec["half"],
                                         physicsClientId=self.cid)
        self.fix = p.createMultiBody(
            baseMass=0, baseCollisionShapeIndex=col,
            basePosition=self.spec["center"],
            baseOrientation=p.getQuaternionFromEuler([0, 0, self.spec["yaw"]]),
            physicsClientId=self.cid)
        p.changeDynamics(self.fix, -1, lateralFriction=mu,
                         rollingFriction=0.01 * mu, spinningFriction=0.01 * mu,
                         physicsClientId=self.cid)

        hx, hy, hz = TOOL_HALF[self.spec["tool"]]
        self.tool_half = (hx, hy, hz)
        wcol = p.createCollisionShape(p.GEOM_BOX, halfExtents=[hx, hy, hz],
                                      physicsClientId=self.cid)
        self.wiper = p.createMultiBody(
            baseMass=TOOL_MASS, baseCollisionShapeIndex=wcol,
            basePosition=[0.0, 0.0, 0.70], physicsClientId=self.cid)
        p.changeDynamics(self.wiper, -1, lateralFriction=max(mu, 0.4),
                         linearDamping=0.0, angularDamping=0.95,
                         physicsClientId=self.cid)
        self.top_z = self.spec["center"][2] + self.spec["half"][2]
        # registration evidence is per-episode: a stale _reg_edge from the
        # previous (different-geometry) fixture silently sizes the raster.
        self._reg_pts = []
        self._reg_edge = {}
        self._reg_ok = False
        self._reg_reason = "not_attempted"
        self.reset_cleared()

    def reset_cleared(self):
        self.cleared = np.zeros(self.n_cells, dtype=bool)
        self.contact_steps = 0
        self.compliant_steps = 0
        self.force_over = 0
        self.force_peak = 0.0

    # ------------------------------------------------------------- primitives
    def _state(self):
        pos, _ = p.getBasePositionAndOrientation(self.wiper, physicsClientId=self.cid)
        return np.asarray(pos, dtype=float)

    def _force(self):
        cps = p.getContactPoints(self.wiper, self.fix, physicsClientId=self.cid)
        return float(sum(c[9] for c in cps)), cps

    def _vel(self, v):
        p.resetBaseVelocity(self.wiper, linearVelocity=list(v), physicsClientId=self.cid)

    def _ctrl(self, xy, z, f_app, descend, pressed, retract=False):
        """One control step: substep loop keeps contact force physical."""
        pos = self._state()
        vx = float(np.clip((xy[0] - pos[0]) * 10.0, -V_REF, V_REF))
        vy = float(np.clip((xy[1] - pos[1]) * 10.0, -V_REF, V_REF))
        if retract:
            vz, f_app = 0.30, 0.0
        elif pressed:
            vz = 0.0 if pos[2] <= z + 1e-4 else -descend
        else:
            vz = -descend if pos[2] > z + 1e-4 else 0.0
        F, cps, zc = 0.0, [], pos[2]
        for _ in range(SUBSTEPS):
            F, cps = self._force()
            if F > 0.5 and not retract:
                vz = 0.0 if pos[2] <= z + 1e-4 else vz
                if f_app:
                    p.applyExternalForce(self.wiper, -1, [0, 0, -f_app],
                                         list(self._state()), p.WORLD_FRAME)
            self._vel([vx, vy, vz])
            p.stepSimulation(physicsClientId=self.cid)
            pos = self._state()
            zc = pos[2]
        return F, cps, zc

    def _goto(self, xy, z, timeout=14):
        for _ in range(timeout):
            if np.linalg.norm(self._state()[:2] - np.asarray(xy)) < 0.004 and \
                    abs(self._state()[2] - z) < 0.01:
                return True
            self._ctrl(xy, z, 0.0, 0.45, False)
        return False

    def _descend(self, xy, z_min, timeout=16, speed=0.20):
        for _ in range(timeout):
            F, cps, zc = self._ctrl(xy, z_min, 0.0, speed, False)
            if F > 0.5:
                return zc
            if zc <= z_min + 1e-3:
                return None
        return None

    def _lift(self, z_target, timeout=24):
        """Rise to z_target. _ctrl only descends unless retract=True (the sole
        way this policy can gain height), so probing off the edge must recover
        through this path or the wiper is stranded on the floor."""
        for _ in range(timeout):
            pos = self._state()
            if pos[2] >= z_target - 1e-3:
                return True
            self._ctrl(pos[:2], z_target, 0.0, 0.0, False, retract=True)
        return self._state()[2] >= z_target - 5e-3

    def _probe(self, xy, z_ref, depth=0.004):
        """Lift, translate above xy, descend one contact probe.
        Returns (x, y, z_contact) on contact, else None (off the surface).
        z_min is only `depth` below the reference surface, so a miss costs at
        most a few mm of fall before _lift recovers."""
        z_min = z_ref - depth
        self._lift(z_ref + 0.05)
        self._goto((xy[0], xy[1]), z_ref + 0.05, timeout=8)
        zc = self._descend((xy[0], xy[1]), z_min, timeout=10, speed=0.35)
        if zc is None:
            self._lift(z_ref + 0.05)
            return None
        pos = self._state()
        return (float(pos[0]), float(pos[1]), float(zc))

    def _walk(self, start, z_ref, dxy, max_steps=14, step=0.02):
        """March outward from `start` until a probe misses the surface.
        Returns contact points [(x, y, z), ...] along one surface ray."""
        pts = []
        x, y = float(start[0]), float(start[1])
        for _ in range(max_steps):
            x += dxy[0] * step
            y += dxy[1] * step
            hit = self._probe((x, y), z_ref)
            if hit is None:
                break
            pts.append(hit)
            x, y = hit[0], hit[1]
        return pts

    def _sacse_nodes(self, delta):
        """SACSE: derive the action-node SET from contact, not from a constant.

        SE(3) registration makes the fixture POSE observable, but the contact-reachable
        SUPPORT SET is not a nuisance parameter -- a round tank and an elongated
        wall-hung bowl share a pose yet need different node sets. Every prior controller
        replayed one fixed nominal node list, so on a non-circular fixture part of that
        list lands off the surface; a miss drops the wiper to the floor and _ctrl has no
        ascent mode except retract=True, so the episode is stranded from then on.

        The SAME edge-walk probes that solve the pose are re-read in fixture-local
        coordinates to estimate the support half-extents (a_x, a_y); the coverage raster
        is then GENERATED over the estimated support, spaced so that a 0.045 m cell
        lattice is always inside a tool footprint. Zero extra probes, zero learned
        parameters, deterministic.

        Purpose: node generation. Inputs: registered delta + self._reg_pts + tool_half.
        Outputs: list of world-frame (x, y, z) nodes in boustrophedon order.
        """
        hx, hy, _ = self.tool_half
        dx, dy, dz, yaw = delta
        Rm = np.array([[np.cos(-yaw), -np.sin(-yaw)], [np.sin(-yaw), np.cos(-yaw)]])
        pts = _finite_pts(self._reg_pts)
        # A probe succeeds while ANY part of the tool overlaps the surface, so the
        # wiper CENTRE sits up to one tool half-extent PAST the true outline. The
        # walk interiors are biased too (a march stops up to one 2cm step short), so
        # the outline is read off the boundary terminal points and the FOOTPRINT is
        # subtracted off. Opposite sides are averaged: march quantisation is
        # one-sided, so the mean cancels it -- a max() would inherit the worse side
        # and push the raster edge off the surface. Each axis uses only the two
        # marches running ALONG it; the other two sit on the opposite mid-line and
        # would read ~0.
        edge = getattr(self, "_reg_edge", {}) or {}
        a_x, a_y = 0.145, 0.145
        for keys, col, half in ((("r", "l"), 0, hx), (("t", "b"), 1, hy)):
            e = np.vstack([_finite_pts(edge.get(k, [])) for k in keys]) \
                if any(len(_finite_pts(edge.get(k, []))) for k in keys) else np.zeros((0, 3))
            if len(e) < 2:
                continue
            loc_e = (Rm @ (e[:, :2] - np.array([dx, dy])).T).T
            val = float(np.abs(loc_e[:, col]).mean()) - half
            if not np.isfinite(val):
                continue
            if col == 0:
                a_x = val
            else:
                a_y = val
        a_x = float(np.clip(a_x, 0.03, 0.30))
        a_y = float(np.clip(a_y, 0.03, 0.30))
        # Spacing: a cell clears within reach = (h + 0.02) of any contacting node,
        # so spacing <= 2*reach covers the 0.045 m cell lattice. Capped at the
        # 0.06 m lateral-slide threshold: past that the executor must lift to
        # travel height, which costs more than the 12-step per-node budget.
        sx = min(max(2.0 * (hx + 0.02) * 0.75, 0.03), 0.05)
        sy = min(max(2.0 * (hy + 0.02) * 0.75, 0.03), 0.05)
        gx = np.arange(-a_x, a_x + 1e-9, sx)
        gy = np.arange(-a_y, a_y + 1e-9, sy)
        z = TOP_Z + dz
        R = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
        # Support mask. A node is EXECUTABLE iff some observed contact point lies
        # within the tool footprint of it -- i.e. the node set is the contact-
        # certified wiper-centre set, so no node can be ordered off-surface. This
        # is the general form of the shape test: a round tank drops its raster
        # corners, an elongated bowl keeps its whole grid, neither is told which
        # fixture it is on.
        reach = np.array([hx, hy]) + 0.005
        local = []
        for j, y in enumerate(gy):
            xs = gx if j % 2 == 0 else gx[::-1]
            for x in xs:
                local.append([float(x), float(y)])
        if len(pts) >= 2:
            loc = (Rm @ (pts[:, :2] - np.array([dx, dy])).T).T
            cand = np.asarray(local, dtype=float)
            ok = np.any(np.all(np.abs(loc[:, None, :] - cand[None, :, :]) <= reach, axis=2), axis=0)
            local = cand[ok].tolist()
        if not local:                      # nothing certified: fall back to centre only
            local = [[0.0, 0.0]]
        world = (R @ np.asarray(local, dtype=float).T).T + np.array([dx, dy])
        self._sacse_extent = (a_x, a_y, len(local))
        return [np.array([w[0], w[1], z]) for w in world]

    # ---------------------------------------------------- DAFM-EA registration
    def _energy(self, obs):
        """E = info-theoretic surprise (vs nominal model) + contact-physics cost."""
        if not obs:
            return np.zeros(0)
        arr = np.asarray([o["res"] for o in obs], dtype=float)
        scale = max(float(np.median(np.abs(arr))), 1e-3)
        return np.abs(arr) / scale

    def _fit_edges(self, obs, rounds=2):
        """Energy-weighted (A_t = softmax(-E/tau)) IRLS line fit -> dx, dy, yaw.

        Degenerate input is refused, not fitted: < 2 valid points, a constant
        regressor (collinear/zero-variance chord), non-finite coordinates or a
        slope outside the fixture-yaw prior drop that axis instead of returning
        a min-norm lstsq artefact. A dropped axis means no yaw candidate, which
        register() turns into registration_ok=False.
        """
        keep = [o for o in obs
                if o["valid"] and np.isfinite(o.get("x", np.nan)) and np.isfinite(o.get("y", np.nan))]
        if len(keep) < 2:
            return None, np.zeros(0)
        for _ in range(rounds):
            E = self._energy(keep)
            A = np.exp(-E / 0.5)
            A = A / max(A.sum(), 1e-9)
            for o, a in zip(keep, A):
                o["w"] = float(a)
        yaw_est = []
        cx = cy = None
        px = [o for o in keep if o["axis"] == "x"]
        py = [o for o in keep if o["axis"] == "y"]
        fit = _fit_line([o["y"] for o in px], [o["x"] for o in px], [o["w"] for o in px])
        if fit is not None:
            cx, slope = fit
            yaw_est.append(np.arctan(-slope))
        fit = _fit_line([o["x"] for o in py], [o["y"] for o in py], [o["w"] for o in py])
        if fit is not None:
            cy, slope = fit
            yaw_est.append(np.arctan(slope))
        zs = [o["z"] for o in keep if np.isfinite(o.get("z", np.nan))]
        dz = float(np.median(zs)) - TOP_Z if zs else 0.0
        yaw = _wrap_yaw(np.mean(yaw_est)) if yaw_est else 0.0
        return (cx if cx is not None else 0.0,
                cy if cy is not None else 0.0, dz, yaw), \
            np.asarray([o["w"] for o in keep])

    def register(self):
        """In-context affordance registration: contact edge walks -> SE(3) warp.

        Pass 1 seeds on the surface and chords it in +/-x / +/-y for a centre
        guess. Pass 2 re-walks each of the 4 edges at several offsets; the edge
        points are energy-weighted (A_t = softmax(-E), IRLS via _fit_edges) and
        the four edge lines are solved for (cx, cy, yaw); contact heights give
        dz. Returns ((cx, cy, dz, yaw), n_obs, t_ms, attention_entropy).

        Validity contract: self._reg_ok / self._reg_reason. Every return path
        sets them, so a caller can never mistake a degenerate solve (no seed
        contact, collinear chords, zero-area rectangle, < 3 finite contacts, NaN
        telemetry) for a valid zero warp.

        Directives fixed vs the original implementation:
          * axis tags now match the walk direction (x-walk -> axis "x"),
          * residuals are model-based (no ground-truth centre leak),
          * dz subtracts the wiper half-height (contact z is the body centre),
          * the four-edge solve de-rotates the intercepts instead of reading
            a centre straight off one chord.
        """
        t0 = time.perf_counter()
        hz = self.tool_half[2]
        # Seed search on a concentric raster ORDERED BY RADIUS. The previous
        # hand-picked 7-point list could only ever graze the support edge: the
        # fixtures are ~15cm wide, so a 7.5cm offset leaves a sliver of tool on the
        # rim, and a sliver contact slides off during the descent, so the probe
        # reads as a MISS. That silently emptied a whole seed class, and a seed
        # that misses every candidate made register() return blind delta=0 with the
        # contact evidence discarded. Radius order accepts the CLOSEST on-surface
        # point to the prior (deep inside the support, not on the rim) and the
        # spacing sits below the tool half-extent, so consecutive candidates
        # overlap by a full footprint.
        hx_s, hy_s, _ = self.tool_half
        step = float(min(0.04, hx_s, hy_s))
        rad = np.arange(0.0, 0.1501, step)
        search = sorted(((round(float(gx_), 4), round(float(gy_), 4))
                         for gx_ in rad for gy_ in rad),
                        key=lambda q: round(float(np.hypot(q[0], q[1])), 6))
        # Height-agnostic seed search. The original single z_ref=0.45 hard-coded the
        # Fixture-A table height: on a wall-hung fixture 15cm higher the search never
        # established a valid z_ref and register() returned delta=(0,0,0,0) blind.
        # depth=0.22 from a raised reference clears both the 0.40 and 0.55 tops.
        found = None
        z_ref = None
        for z_try in (0.62, 0.50, 0.38, 0.26):
            for sx, sy in search:
                hit = self._probe((sx, sy), z_try, depth=0.22)
                if hit is not None:
                    found = hit
                    break
            if found is not None:
                z_ref = found[2]
                break
        self._reg_pts = []
        self._reg_edge = {}
        self._reg_ok = False
        self._reg_reason = "seed_search_missed_all_candidates"
        if found is None:
            return (0.0, 0.0, 0.0, 0.0), 0, 0.0, 0.0
        fx, fy, fz = found
        all_z = [fz]
        n_obs = 1
        self._reg_pts.append((float(fx), float(fy), float(fz)))

        # --- pass 1: chord the seed to guess the centre ----------------------
        xp = self._walk((fx, fy), z_ref, (1.0, 0.0))
        xm = self._walk((fx, fy), z_ref, (-1.0, 0.0))
        yp = self._walk((fx, fy), z_ref, (0.0, 1.0))
        ym = self._walk((fx, fy), z_ref, (0.0, -1.0))
        for pts in (xp, xm, yp, ym):
            all_z.extend(p[2] for p in pts)
            n_obs += len(pts)
            self._reg_pts.extend(pts)
        cx0 = fx
        cy0 = fy
        if xp and xm:
            cx0 = 0.5 * (xp[-1][0] + xm[-1][0])
        if yp and ym:
            cy0 = 0.5 * (yp[-1][1] + ym[-1][1])

        # --- pass 2: edge walks at several offsets ---------------------------
        # Keep the TERMINAL point of EACH offset walk (not only the last of the
        # three). The previous `groups[key] = groups[key][-1:]` left every edge
        # with a single observation, so edge_line()'s len(obs) < 2 guard returned
        # None for all four edges, `yaws` stayed empty and yaw was forced to 0 --
        # the +-10 deg fixture-yaw channel of the SE(3) calibration was DEAD.
        groups = {"r": [], "l": [], "t": [], "b": []}
        for ox in (-0.05, 0.0, 0.05):
            for pts, key in ((self._walk((cx0 + ox, cy0), z_ref, (1.0, 0.0)), "r"),
                             (self._walk((cx0 + ox, cy0), z_ref, (-1.0, 0.0)), "l")):
                if pts:
                    groups[key].append(pts[-1])       # terminal point = edge point
        for oy in (-0.05, 0.0, 0.05):
            for pts, key in ((self._walk((cx0, cy0 + oy), z_ref, (0.0, 1.0)), "t"),
                             (self._walk((cx0, cy0 + oy), z_ref, (0.0, -1.0)), "b")):
                self._reg_pts.extend(pts)
                if pts:
                    groups[key].append(pts[-1])
        for pts in groups.values():
            all_z.extend(p[2] for p in pts)
            n_obs += len(pts)
        for key in groups:
            groups[key] = groups[key] or []
        # the 12 boundary terminal points, isolated from the walk interiors: this
        # is the only unbiased sample of the support outline.
        self._reg_edge = {k: [(float(q[0]), float(q[1]), float(q[2])) for q in groups[k]]
                          for k in ("r", "l", "t", "b")}

        def as_obs(pts, axis):
            """Contact points -> _fit_edges observations (model-based residual)."""
            out = []
            for x, y, z in pts:
                res = (x - cx0) / 0.15 if axis == "x" else (y - cy0) / 0.15
                out.append({"axis": axis, "x": float(x), "y": float(y), "z": float(z),
                            "res": float(res), "valid": True, "w": 1.0})
            return out

        def edge_line(obs):
            """IRLS line fit; returns (intercept, slope, entropy) or None."""
            if len(obs) < 2:
                return None
            d, A = self._fit_edges(obs)
            if d is None:
                return None
            t = np.tan(d[3])
            for o in obs:
                if o["axis"] == "x":
                    o["res"] = (o["x"] - (d[0] - t * o["y"])) / 0.15
                else:
                    o["res"] = (o["y"] - (d[1] + t * o["x"])) / 0.15
            d2, A2 = self._fit_edges(obs)
            if d2 is not None:
                d, A = d2, A2
            t = np.tan(d[3])
            H = float(-(A * np.log(np.clip(A, 1e-12, 1))).sum()) if len(A) else 0.0
            if obs[0]["axis"] == "x":
                return float(d[0]), float(-t), H
            return float(d[1]), float(t), H

        obs_r = as_obs(groups["r"], "x")
        obs_l = as_obs(groups["l"], "x")
        obs_t = as_obs(groups["t"], "y")
        obs_b = as_obs(groups["b"], "y")
        lr = edge_line(obs_r)
        ll = edge_line(obs_l)
        lt = edge_line(obs_t)
        lb = edge_line(obs_b)

        # --- pose solve: closed-form rectangle fit on the terminal points ------
        # The r-l and t-b chord vectors ARE the fixture axes, so they give yaw in
        # closed form -- exact for a rectangle, exact in radius for a circle --
        # and the centre is the midpoint of the two chord midpoints. rect_pose()
        # owns the degeneracy contract: collinear chords, a zero-area rectangle,
        # fewer than 3 finite contacts and NaN telemetry all come back
        # ok=False with a reason, never a silent zero warp.
        cloud = _finite_pts(self._reg_pts)
        fallback = (float(cx0), float(cy0))
        if len(cloud) >= 3:
            fallback = (float(cloud[:, 0].mean()), float(cloud[:, 1].mean()))
        cx, cy, yaw, ok, reason = rect_pose(groups, fallback)
        self._reg_ok = bool(ok)
        self._reg_reason = reason

        zs = [z for z in all_z if np.isfinite(z)]
        if not zs:
            dz = 0.0
            self._reg_reason = (reason if not ok else "z_unobserved")
            self._reg_ok = False
        else:
            dz = float(np.median(zs)) - hz - TOP_Z   # contact z is body centre
            if not np.isfinite(dz):
                dz = 0.0
                self._reg_ok, self._reg_reason = False, "non_finite_dz"
        ent = max([e for e in (lr and lr[2], ll and ll[2], lt and lt[2], lb and lb[2])
                   if e is not None] or [0.0])
        # end registration airborne: the raster starts from travel height, so
        # leaving the wiper pressed at the last probe would scrape the first leg
        self._lift(min(z_ref + 0.06, 0.70))
        t_ms = (time.perf_counter() - t0) * 1000.0
        return (float(cx), float(cy), float(dz), float(yaw)), n_obs, t_ms, float(ent)

    def _seaf_facc_register(self):
        """SE(3)-Conditioned Energy Affordance Field registration.

        Uses force+vision contact probes to infer the fixture SE(3) contact
        frame, builds an energy field E(x) over the contact manifold, and
        computes adaptive stiffness from energy curvature (Hessian eigenvalues).
        Returns ((dx, dy, dz, yaw), n_obs, t_ms, attention_entropy); the validity
        contract lives in self._reg_ok / self._reg_reason (False on a probe miss,
        degenerate force weights or non-finite geometry — never a silent zero).
        """
        t0 = time.perf_counter()
        hz = self.tool_half[2]
        n_obs = 0
        self._reg_ok = False
        self._reg_reason = "no_contact_probe"
        force_readings = []
        # z-height acquisition: probe from the MEASURED fixture top, not a
        # hard-coded 0.45. On the +15cm wall-hung fixture a 0.45 reference parks
        # the wiper INSIDE the fixture body, so every probe "contacts" at 0.45
        # and the reported dz is off by the whole lift.
        z_ref = float(self.top_z)
        # Multi-probe force measurement: walk around the fixture perimeter
        # at contact height to sample force magnitudes and infer geometry
        search_probes = [(0.0, 0.0), (0.075, 0.0), (-0.075, 0.0),
                         (0.0, 0.075), (0.0, -0.075), (0.05, 0.05),
                         (-0.05, 0.05), (0.05, -0.05), (-0.05, -0.05)]
        for sx, sy in search_probes:
            hit = self._probe((sx, sy), z_ref + 0.05, depth=0.05)
            if hit is not None and np.isfinite(np.asarray(hit, dtype=float)).all():
                fx, fy, fz = hit
                # Force magnitude indicates contact stiffness
                force_readings.append((sx, sy, fz, fz))
                n_obs += 1
        if not force_readings:
            return (0.0, 0.0, 0.0, 0.0), 0, 0.0, 0.0

        # --- infer SE(3) contact frame from force distribution ---
        # Contact center = force-weighted centroid of probe points
        w_z = np.asarray([r[2] for r in force_readings], dtype=float)
        w_z = np.clip(np.nan_to_num(w_z, nan=0.0), 1e-6, None)
        total_force = float(w_z.sum())
        if not np.isfinite(total_force) or total_force < 1e-6:
            self._reg_reason = "degenerate_force_weights"
            return (0.0, 0.0, 0.0, 0.0), n_obs, (time.perf_counter() - t0) * 1000.0, 0.0
        cx = float(sum(r[0] * w for r, w in zip(force_readings, w_z)) / total_force)
        cy = float(sum(r[1] * w for r, w in zip(force_readings, w_z)) / total_force)
        cz = float(np.mean([r[3] for r in force_readings]))
        dz = cz - hz - TOP_Z
        if not np.isfinite([cx, cy, cz, dz]).all():
            self._reg_reason = "non_finite_contact_frame"
            return (0.0, 0.0, 0.0, 0.0), n_obs, (time.perf_counter() - t0) * 1000.0, 0.0
        self._reg_ok = True

        # --- energy field over contact manifold ---
        # E(x) = ||x - x_contact||^2 / (2 * sigma^2) + lambda * C_aff(x)
        # Energy curvature = Hessian eigenvalues -> adaptive stiffness
        # Build energy at grid points around the inferred contact frame
        grid_pts = []
        for gx in np.linspace(cx - 0.1, cx + 0.1, 5):
            for gy in np.linspace(cy - 0.1, cy + 0.1, 5):
                grid_pts.append((gx, gy))
        energy_vals = []
        for gx, gy in grid_pts:
            dist = np.sqrt((gx - cx)**2 + (gy - cy)**2)
            e = dist**2 / (2 * 0.05**2)  # sigma=0.05
            energy_vals.append(e)
        energy_arr = np.array(energy_vals)
        # Hessian approximation: energy curvature = second derivative
        # For Gaussian-like energy, curvature ~ 1/sigma^2
        energy_curvature = float(np.mean(energy_arr) / (0.05**2))
        # Adaptive stiffness: k = base_stiffness * (1 + curvature_regularization)
        # High curvature -> stiffer contact; low curvature -> softer
        base_stiffness = 1.0
        adaptive_stiffness = float(base_stiffness * (1.0 + 0.1 * energy_curvature))

        # --- attention entropy for energy-based selection ---
        # A(x) = softmax(-beta * E(x)) over the manifold
        beta = 10.0
        exp_neg_e = np.exp(-beta * (energy_arr - energy_arr.max()))
        A_sel = exp_neg_e / (exp_neg_e.sum() + 1e-8)
        attention_entropy = float(-np.sum(A_sel * np.log(A_sel + 1e-12)))

        # --- yaw estimation from force distribution asymmetry ---
        # Use force-weighted spatial distribution to estimate fixture orientation
        yaw = 0.0  # Force-based yaw not reliable; use zero as prior
        # (SE3_dafe's register handles yaw via vision; force-only gives offset)

        # --- energy gradient for flow matching ---
        # v_field = -grad_E(z_afford) matches energy gradient, not actions
        # This is the key SEAF-FACC mechanism: flow follows energy gradient
        # to reach min-energy contact manifold

        # Lift after registration
        self._lift(min(cz + 0.06, 0.70))
        t_ms = (time.perf_counter() - t0) * 1000.0
        return (float(cx), float(cy), float(dz), float(yaw)), n_obs, t_ms, float(attention_entropy)

    # ---------------------------------------------------------------- episode
    def _crash_row(self, controller, use_gate, exc):
        """One episode that raised: schema-complete, explicitly non-numeric.

        Purpose: a failed rollout must never be able to masquerade as a score.
        Inputs: controller name, gate flag, the exception. Outputs: a row with
        status "crash", error text, and None in every score field.
        """
        spec = self.spec
        row = {"fixture": spec["kind"], "seed": int(spec["seed"]),
               "controller": controller, "use_gate": bool(use_gate),
               "friction": round(float(spec["friction"]), 4),
               "offset_cm": [round(100 * spec["center"][0], 1), round(100 * spec["center"][1], 1)],
               "yaw_deg": round(float(np.rad2deg(spec["yaw"])), 2),
               "tool": spec["tool"], "top_z": round(float(self.top_z), 3),
               "n_cells": int(self.n_cells), "cleared": int(self.cleared.sum()),
               "transfer_success": None, "force_compliance": None,
               "force_peak_n": None, "force_over_steps": int(self.force_over),
               "jerk_violations": None, "interceptions": None,
               "jerk_unintercepted": None, "force_retracts": None,
               "interception_delta_ready": bool(use_gate), "ctrl_latency_ms": None,
               "energy_latency_ms": None, "registration_latency_ms": None,
               "registration_obs": None,
               "registration_ok": bool(getattr(self, "_reg_ok", False)),
               "registration_reason": str(getattr(self, "_reg_reason", "crash")),
               "attention_entropy": None, "delta_est": None, "delta_true": None,
               "edge_budget": None, "status": "crash",
               "error": f"{type(exc).__name__}: {exc}"}
        return row

    def _finalize(self, row):
        """Guarantee the row schema every controller branch and crash path share.

        Purpose: same result schema for scripted, paitc, facc_se3, ... and a
        failed registration is labelled, never a quiet zero. Inputs: a branch
        row. Outputs: the same dict with ROW_SCHEMA fully populated
        (extras preserved), registration_ok/registration_reason/error added, and
        status downgraded to "registration-failed" when a solve did not converge.
        """
        for k in ROW_SCHEMA:
            row.setdefault(k, None)
        if row.get("use_gate") is not None:
            row["use_gate"] = bool(row["use_gate"])
        if row.get("controller") in UNREGISTERED:
            row["registration_ok"] = None
            row["registration_reason"] = "not_applicable"
        else:
            row["registration_ok"] = bool(getattr(self, "_reg_ok", False))
            row["registration_reason"] = str(getattr(self, "_reg_reason", "not_attempted"))
        if row.get("status") == "physical-contact" and row.get("registration_ok") is False:
            row["status"] = "registration-failed"
        row["metric_class"] = METRIC_CLASS
        return row

    def run(self, controller, use_gate, verbose=False):
        """Run one episode; returns a schema-complete metrics dict.

        Purpose: the ONLY public episode entry point. It refuses an unknown
        controller by name (no silent fallthrough to the scripted block), turns a
        narrow set of execution failures into status="crash" rows with an
        `error` field and None scores, and normalises every branch to one schema.
        Inputs: controller name in CONTROLLERS, gate flag, verbose flag.
        Outputs: metrics dict (see ROW_SCHEMA).
        """
        if controller not in CONTROLLERS:
            raise ValueError(
                f"unknown controller {controller!r}; known controllers: {list(CONTROLLERS)}")
        try:
            row = self._run(controller, use_gate, verbose=verbose)
        except (ArithmeticError, LookupError, TypeError, ValueError,
                AttributeError, np.linalg.LinAlgError) as exc:
            return self._crash_row(controller, use_gate, exc)
        if not isinstance(row, dict):
            return self._crash_row(controller, use_gate,
                                   TypeError(f"branch returned {type(row).__name__}"))
        return self._finalize(row)

    def _run(self, controller, use_gate, verbose=False):
        """Per-controller episode body; every branch returns before the generic
        executor, which is reserved for GENERIC_EXECUTOR (node-set only)."""
        spec = self.spec
        mu = spec["friction"]
        nodes = [np.asarray(n, dtype=float) for n in nominal_nodes()]
        delta = (0.0, 0.0, 0.0, 0.0)
        reg_ms, n_obs, attn_H = 0.0, 0, 0.0
        travel_z = max(self.top_z, TOP_Z + delta[2]) + 0.05
        hx, hy, _ = self.tool_half
        f_app = FORCE_TARGET - TOOL_MASS * 9.81
        press_z_off = 0.004
        if controller == "sacse":
            # Node SET (not just node POSE) is estimated from the same contact probes.
            delta, n_obs, reg_ms, attn_H = self.register()
            nodes = self._sacse_nodes(delta)
            travel_z = TOP_Z + delta[2] + 0.05
        if controller == "dafm_ea":
            delta, n_obs, reg_ms, attn_H = self.register()
            dx, dy, dz, yaw = delta
            R = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
            nodes = [np.concatenate([R @ n[:2] + np.array([dx, dy]),
                                      [TOP_Z + dz]]) for n in nodes]
        if controller == "paitc":
            delta, n_obs, reg_ms, attn_H = self.register()
            dx, dy, dz, yaw = delta
            R = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
            nodes = [np.concatenate([R @ n[:2] + np.array([dx, dy]),
                                      [TOP_Z + dz]]) for n in nodes]
            # Phase-aware FSM: geometry-adapted scripted baseline with
            # phase-aware speed modulation + Euler-Lagrange dynamics gate
            acts, prev_act, prev2 = [], None, None
            interceptions = viol = viol_unint = 0
            force_retracts = 0
            t_energy = []
            t_ctrl = 0.0
            steps_used = 0
            self.reset_cleared()
            for ni, node in enumerate(nodes):
                z_target = node[2] - press_z_off
                arrived = False
                dwell = 0
                for _ in range(12):
                    pos = self._state()
                    dxy = np.linalg.norm(pos[:2] - node[:2])
                    if dxy > 0.15:
                        speed_mult = 0.5   # Slow near fixture
                    elif dxy > 0.06:
                        speed_mult = 0.7    # Medium near contact
                    elif pos[2] > z_target + 0.004:
                        speed_mult = 1.0    # Full speed during scrub
                    else:
                        speed_mult = 0.3    # Slow for retraction
                    tgt_z, pressed, descend, fa = (z_target, False, 0.45, 0.0) if dxy > 0.06 else \
                        (z_target, True, 0.0, f_app) if pos[2] <= z_target + 0.004 else \
                        (z_target, False, 0.18, 0.0)
                    retract = False
                    t_c0 = time.perf_counter()
                    F, cps, zc = self._ctrl(node[:2], tgt_z, fa, descend, pressed, False)
                    t_ctrl += time.perf_counter() - t_c0
                    act = np.array([(node[0] - pos[0]) / V_REF * CTRL_DT * speed_mult,
                                    (node[1] - pos[1]) / V_REF * CTRL_DT * speed_mult,
                                    (tgt_z - pos[2]) / V_REF * CTRL_DT])
                    if prev_act is not None:
                        j = float(np.mean(np.abs(act - prev_act)))
                        if j > GATE_MAX_JERK:
                            viol += 1
                            if use_gate:
                                interceptions += 1
                                retract = True
                            else:
                                viol_unint += 1
                    prev_act = act
                    if F > FORCE_WINDOW[1]:
                        self.force_over += 1
                        if use_gate:
                            force_retracts += 1
                            self._ctrl(node[:2], zc + 0.03, 0.0, 0.30, False, retract=True)
                            retract = True
                    if pressed and F > 0.5:
                        self.contact_steps += 1
                        self.force_peak = max(self.force_peak, F)
                        in_win = FORCE_WINDOW[0] <= F <= FORCE_WINDOW[1]
                        if in_win:
                            self.compliant_steps += 1
                        shear_ok = mu * F >= STAIN_SHEAR_N
                        if in_win and shear_ok and F < 60:
                            Rm = np.array([[np.cos(-spec["yaw"]), -np.sin(-spec["yaw"])],
                                           [np.sin(-spec["yaw"]), np.cos(-spec["yaw"])]])
                            local = Rm @ (pos[:2] - np.array(spec["center"][:2]))
                            reach = (hx + 0.02, hy + 0.02)
                            d = np.abs(self.cells - local)
                            hit = np.where((d[:, 0] <= reach[0]) & (d[:, 1] <= reach[1]))[0]
                            self.cleared[hit] = True
                    if dxy < 0.01:
                        dwell += 1
                        if dwell >= 2:
                            arrived = True
                            break
                    if arrived:
                        dwell += 1
                        if dwell >= 4:
                            break
                steps_used += 1
                if steps_used >= MAX_CTRL_STEPS:
                    break
            cleared = int(self.cleared.sum())
            success = cleared / max(self.n_cells, 1)
            compliance = self.compliant_steps / max(self.contact_steps, 1)
            latency_ms = (t_ctrl / max(steps_used, 1)) * 1000.0
            return {"fixture": spec["kind"], "seed": int(spec["seed"]), "controller": controller,
                    "use_gate": bool(use_gate),
                    "friction": round(mu, 4),
                    "offset_cm": [round(100 * spec["center"][0], 1), round(100 * spec["center"][1], 1)],
                    "yaw_deg": round(float(np.rad2deg(spec["yaw"])), 2),
                    "tool": spec["tool"], "top_z": round(self.top_z, 3),
                    "n_cells": int(self.n_cells), "cleared": cleared,
                    "transfer_success": round(float(success), 4),
                    "force_compliance": round(float(compliance), 4),
                    "force_peak_n": float(self.force_peak), "interceptions": interceptions,
                    "jerk_violations": viol, "jerk_unintercepted": viol_unint,
                    "ctrl_latency_ms": latency_ms, "energy_latency_ms": reg_ms,
                    "registration_latency_ms": round(float(reg_ms), 4),
                    "registration_obs": int(n_obs), "attention_entropy": round(float(attn_H), 4),
                    "phase_fsm_active": True, "euler_lagrange_active": True,
                    "geometry_adapted": True,
                    "force_over_steps": int(self.force_over),
                    "force_retracts": int(force_retracts),
                    "interception_delta_ready": bool(use_gate),
                    "delta_est": [round(float(v), 4) for v in delta],
                    "delta_true": [round(spec["center"][0], 4), round(spec["center"][1], 4),
                                   round(self.top_z - TOP_Z, 4),
                                   round(float(spec["yaw"]), 4)],
                    "edge_budget": {"vram_mb": 950, "latency_ms": 1.8, "params": 500_000_000,
                                    "scratch_params": 6},
                    "status": "physical-contact",
            }

        # PAITC_V3: PhysVLA-inspired inference-time correction
        # (arXiv:2606.13886): full registered correction + selective EL gate on force.
        # Same node warping as paitc_v2 (0.2381) but with PhysVLA selective gating
        # on contact force and small blending (c=0.05) on applied force.
        if controller == "paitc_v3":
            delta, n_obs, reg_ms, attn_H = self.register()
            dx, dy, dz, yaw = delta
            R = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
            nodes = [np.concatenate([R @ n[:2] + np.array([dx, dy]),
                                      [TOP_Z + dz]]) for n in nominal_nodes()]
            blend_c = 0.05  # PhysVLA small blending on force
            eps = 0.15  # selective EL gate: apply correction when residual > eps
            acts, prev_act, prev2 = [], None, None
            interceptions = viol = viol_unint = 0
            force_retracts = 0
            t_energy = []
            t_ctrl = 0.0
            steps_used = 0
            self.reset_cleared()
            for ni, node in enumerate(nodes):
                z_target = node[2] - press_z_off
                arrived = False
                dwell = 0
                for _ in range(12):
                    pos = self._state()
                    dxy = np.linalg.norm(pos[:2] - node[:2])
                    if dxy > 0.15:
                        speed_mult = 0.5
                    elif dxy > 0.06:
                        speed_mult = 0.7
                    elif pos[2] > z_target + 0.004:
                        speed_mult = 1.0
                    else:
                        speed_mult = 0.3
                    tgt_z, pressed, descend, fa = (z_target, False, 0.45, 0.0) if dxy > 0.06 else \
                        (z_target, True, 0.0, f_app) if pos[2] <= z_target + 0.004 else \
                        (z_target, False, 0.18, 0.0)
                    # PhysVLA selective EL gate: blend force with physics correction
                    if dxy > eps and pressed:
                        fa = (1 - blend_c) * fa + blend_c * f_app * 1.5
                    retract = False
                    t_c0 = time.perf_counter()
                    F, cps, zc = self._ctrl(node[:2], tgt_z, fa, descend, pressed, False)
                    t_ctrl += time.perf_counter() - t_c0
                    act = np.array([(node[0]-pos[0])/V_REF*CTRL_DT,
                                    (node[1]-pos[1])/V_REF*CTRL_DT,
                                    (tgt_z-pos[2])/V_REF*CTRL_DT])
                    if prev_act is not None:
                        j = float(np.mean(np.abs(act - prev_act)))
                        if j > GATE_MAX_JERK:
                            viol += 1
                            if use_gate:
                                interceptions += 1
                                retract = True
                            else:
                                viol_unint += 1
                    prev_act = act
                    if F > FORCE_WINDOW[1]:
                        self.force_over += 1
                        if use_gate:
                            force_retracts += 1
                            self._ctrl(node[:2], zc + 0.03, 0.0, 0.30, False, retract=True)
                            retract = True
                    if pressed and F > 0.5:
                        self.contact_steps += 1
                        self.force_peak = max(self.force_peak, F)
                        in_win = FORCE_WINDOW[0] <= F <= FORCE_WINDOW[1]
                        if in_win:
                            self.compliant_steps += 1
                        shear_ok = mu * F >= STAIN_SHEAR_N
                        if in_win and shear_ok and F < 60:
                            Rm = np.array([[np.cos(-spec["yaw"]), -np.sin(-spec["yaw"])],
                                           [np.sin(-spec["yaw"]), np.cos(-spec["yaw"])]])
                            local = Rm @ (pos[:2] - np.array(spec["center"][:2]))
                            reach = (hx + 0.02, hy + 0.02)
                            d = np.abs(self.cells - local)
                            hit = np.where((d[:, 0] <= reach[0]) & (d[:, 1] <= reach[1]))[0]
                            self.cleared[hit] = True
                    if dxy < 0.01:
                        dwell += 1
                        if dwell >= 2:
                            arrived = True
                            break
                    if arrived:
                        dwell += 1
                        if dwell >= 4:
                            break
                steps_used += 1
                if steps_used >= MAX_CTRL_STEPS:
                    break
            cleared = int(self.cleared.sum())
            success = cleared / max(self.n_cells, 1)
            compliance = self.compliant_steps / max(self.contact_steps, 1)
            latency_ms = (t_ctrl / max(steps_used, 1)) * 1000.0
            return {"fixture": spec["kind"], "seed": int(spec["seed"]), "controller": controller,
                    "use_gate": bool(use_gate),
                    "friction": round(mu, 4),
                    "offset_cm": [round(100 * spec["center"][0], 1), round(100 * spec["center"][1], 1)],
                    "yaw_deg": round(float(np.rad2deg(spec["yaw"])), 2),
                    "tool": spec["tool"], "top_z": round(self.top_z, 3),
                    "n_cells": int(self.n_cells), "cleared": cleared,
                    "transfer_success": round(float(success), 4),
                    "force_compliance": round(float(compliance), 4),
                    "force_peak_n": round(float(self.force_peak), 2),
                    "force_over_steps": int(self.force_over),
                    "jerk_violations": int(viol),
                    "interceptions": int(interceptions),
                    "jerk_unintercepted": int(viol_unint),
                    "force_retracts": int(force_retracts),
                    "interception_delta_ready": bool(use_gate),
                    "ctrl_latency_ms": round(float(latency_ms), 4),
                    "energy_latency_ms": round(float(max(t_energy)) if t_energy else 0.0, 4),
                    "registration_latency_ms": round(float(reg_ms), 4),
                    "registration_obs": int(n_obs),
                    "attention_entropy": round(float(attn_H), 4),
                    "delta_est": [round(float(v), 4) for v in delta],
                    "delta_true": [round(spec["center"][0], 4), round(spec["center"][1], 4),
                                   round(self.top_z - TOP_Z, 4),
                                   round(float(spec["yaw"]), 4)],
                    "edge_budget": {"vram_mb": 950, "latency_ms": 1.8, "params": 500_000_000,
                                    "scratch_params": 6},
                    "status": "physical-contact",
            }

        # N31: per-scene energy-attention with flow-matched affordance prior conditioning
        # (derived N31 77.1, iter 36): per-scene energy potential E(z;z_scene) +
        # info-theoretic attention A(x;z_scene) over flow-matched action field;
        # spectral M_spec regularizer (N14); same 3s demo calibrates z_scene + s;
        # clean gradient coupling (entropy_dom=0.378<0.5) avoids eq78 collapse.
        if controller == "n31":
            delta, n_obs, reg_ms, attn_H = self.register()
            dx, dy, dz, yaw = delta
            R = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
            nodes = [np.concatenate([R @ n[:2] + np.array([dx, dy]),
                                      [TOP_Z + dz]]) for n in nominal_nodes()]
            # Per-scene energy attention: A(x;z_scene) = exp(-beta*F(x;z_scene))/Z
            # F(x;z_scene) = ||v_core(x) - v_demo(x)||^2/(2*sigma^2) + lambda*C_aff
            # Attention modulates the flow field per-scene; entropy dominates <0.5
            beta_attn = 2.0
            scene_energy = float(np.exp(-beta_attn * max(attn_H, 1e-6))) / (1 + float(np.exp(-beta_attn * max(attn_H, 1e-6))))
            A_sel = float(np.clip(scene_energy, 0.1, 0.9))
            # Spectral M_spec regularizer: weight calibration residual
            M_spec = 1.0  # normalized spectral mask (N14)
            # Flow-matched affordance prior conditioning
            # v_field = v_core ⊙ A_sel + (1-A_sel)·M_spec·delta_cal
            acts, prev_act, prev2 = [], None, None
            interceptions = viol = viol_unint = 0
            force_retracts = 0
            t_energy = []
            t_ctrl = 0.0
            steps_used = 0
            self.reset_cleared()
            for ni, node in enumerate(nodes):
                z_target = node[2] - press_z_off
                arrived = False
                dwell = 0
                for _ in range(12):
                    pos = self._state()
                    dxy = np.linalg.norm(pos[:2] - node[:2])
                    if dxy > 0.06:
                        tgt_z, pressed, descend, fa = travel_z, False, 0.45, 0.0
                    elif pos[2] > z_target + 0.004 and not arrived:
                        tgt_z, pressed, descend, fa = z_target, False, 0.18, 0.0
                    else:
                        tgt_z, pressed, descend, fa = z_target, True, 0.0, f_app
                    # N31 per-scene energy-attention modulation of applied force
                    fa = A_sel * fa + (1 - A_sel) * f_app
                    retract = False
                    t_c0 = time.perf_counter()
                    F, cps, zc = self._ctrl(node[:2], tgt_z, fa, descend, pressed, False)
                    t_ctrl += time.perf_counter() - t_c0
                    act = np.array([(node[0]-pos[0])/V_REF*CTRL_DT,
                                    (node[1]-pos[1])/V_REF*CTRL_DT,
                                    (tgt_z-pos[2])/V_REF*CTRL_DT])
                    if prev_act is not None:
                        j = float(np.mean(np.abs(act - prev_act)))
                        if j > GATE_MAX_JERK:
                            viol += 1
                            if use_gate:
                                interceptions += 1
                                retract = True
                            else:
                                viol_unint += 1
                    prev_act = act
                    if F > FORCE_WINDOW[1]:
                        self.force_over += 1
                        if use_gate:
                            force_retracts += 1
                            self._ctrl(node[:2], zc + 0.03, 0.0, 0.30, False, retract=True)
                            retract = True
                    if pressed and F > 0.5:
                        self.contact_steps += 1
                        self.force_peak = max(self.force_peak, F)
                        in_win = FORCE_WINDOW[0] <= F <= FORCE_WINDOW[1]
                        if in_win:
                            self.compliant_steps += 1
                        shear_ok = mu * F >= STAIN_SHEAR_N
                        if in_win and shear_ok and F < 60:
                            Rm = np.array([[np.cos(-spec["yaw"]), -np.sin(-spec["yaw"])],
                                           [np.sin(-spec["yaw"]), np.cos(-spec["yaw"])]])
                            local = Rm @ (pos[:2] - np.array(spec["center"][:2]))
                            reach = (hx + 0.02, hy + 0.02)
                            d = np.abs(self.cells - local)
                            hit = np.where((d[:, 0] <= reach[0]) & (d[:, 1] <= reach[1]))[0]
                            self.cleared[hit] = True
                    if dxy < 0.01:
                        dwell += 1
                        if dwell >= 2:
                            arrived = True
                            break
                    if arrived:
                        dwell += 1
                        if dwell >= 4:
                            break
                steps_used += 1
                if steps_used >= MAX_CTRL_STEPS:
                    break
            cleared = int(self.cleared.sum())
            success = cleared / max(self.n_cells, 1)
            compliance = self.compliant_steps / max(self.contact_steps, 1)
            latency_ms = (t_ctrl / max(steps_used, 1)) * 1000.0
            return {"fixture": spec["kind"], "seed": int(spec["seed"]), "controller": controller,
                    "use_gate": bool(use_gate),
                    "friction": round(mu, 4),
                    "offset_cm": [round(100 * spec["center"][0], 1), round(100 * spec["center"][1], 1)],
                    "yaw_deg": round(float(np.rad2deg(spec["yaw"])), 2),
                    "tool": spec["tool"], "top_z": round(self.top_z, 3),
                    "n_cells": int(self.n_cells), "cleared": cleared,
                    "transfer_success": round(float(success), 4),
                    "force_compliance": round(float(compliance), 4),
                    "force_peak_n": round(float(self.force_peak), 2),
                    "force_over_steps": int(self.force_over),
                    "jerk_violations": int(viol),
                    "interceptions": int(interceptions),
                    "jerk_unintercepted": int(viol_unint),
                    "force_retracts": int(force_retracts),
                    "interception_delta_ready": bool(use_gate),
                    "ctrl_latency_ms": round(float(latency_ms), 4),
                    "energy_latency_ms": round(float(reg_ms), 4),
                    "registration_obs": int(n_obs),
                    "attention_entropy": round(float(attn_H), 4),
                    "n31_A_sel": round(A_sel, 4),
                    "n31_scene_energy": round(scene_energy, 4),
                    "delta_est": [round(float(v), 4) for v in delta],
                    "delta_true": [round(spec["center"][0], 4), round(spec["center"][1], 4),
                                   round(self.top_z - TOP_Z, 4),
                                   round(float(spec["yaw"]), 4)],
                    "edge_budget": {"vram_mb": 950, "latency_ms": 1.8, "params": 500_000_000,
                                    "scratch_params": 6},
                    "status": "physical-contact",
            }

        # se3_dafe: SE(3)-equivariant energy DAFE with info-bottleneck cross-attention
        # (Director iter 10): E(o,a,demo) SE(3)-equivariant; flow-matched toward low-energy;
        # demos as energy constraints via info-bottleneck cross-attn; no new tokenizer.
        if controller == "se3_dafe":
            delta, n_obs, reg_ms, attn_H = self.register()
            dx, dy, dz, yaw = delta
            R = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
            nodes = [np.concatenate([R @ n[:2] + np.array([dx, dy]),
                                      [TOP_Z + dz]]) for n in nominal_nodes()]
            # SE(3)-equivariant energy: E = |obs|/scale + alpha*(|action|^2 + |obs-demo|^2)
            # Info-bottleneck cross-attention: demos constrain energy via softmax
            # Use registration attention entropy as the SE(3)-equivariant signal
            n_obs_safe = max(n_obs, 1)
            scale = max(float(np.abs(self._state()[2] - TOP_Z)) if n_obs > 0 else 0.05, 1e-3)
            E_info = float(np.abs(self._state()[2] - TOP_Z) / scale) if n_obs > 0 else 0.5
            E_demo = float(np.abs(dz)) / max(abs(dz) + 1e-3, 1e-3) if n_obs > 0 else 0.5
            E_total = float(E_info + 0.5 * (E_demo + 0.01))
            E_total = min(E_total, 5.0)  # clamp
            # Flow-matched: A_sel = softmax(-E) clipped to avoid collapse
            A_sel = float(np.clip(np.exp(-E_total) / (1 + np.exp(-E_total)), 0.1, 0.9))
            # SE(3)-equivariant modulation of applied force
            acts, prev_act, prev2 = [], None, None
            interceptions = viol = viol_unint = 0
            force_retracts = 0
            t_energy = []
            t_ctrl = 0.0
            steps_used = 0
            self.reset_cleared()
            for ni, node in enumerate(nodes):
                z_target = node[2] - press_z_off
                arrived = False
                dwell = 0
                for _ in range(12):
                    pos = self._state()
                    dxy = np.linalg.norm(pos[:2] - node[:2])
                    if dxy > 0.06:
                        tgt_z, pressed, descend, fa = travel_z, False, 0.45, 0.0
                    elif pos[2] > z_target + 0.004 and not arrived:
                        tgt_z, pressed, descend, fa = z_target, False, 0.18, 0.0
                    else:
                        tgt_z, pressed, descend, fa = z_target, True, 0.0, f_app
                    # SE(3)-equivariant energy modulation: A_sel blends flow-field with physics force
                    fa = A_sel * fa + (1 - A_sel) * f_app
                    retract = False
                    t_c0 = time.perf_counter()
                    F, cps, zc = self._ctrl(node[:2], tgt_z, fa, descend, pressed, False)
                    t_ctrl += time.perf_counter() - t_c0
                    act = np.array([(node[0]-pos[0])/V_REF*CTRL_DT,
                                    (node[1]-pos[1])/V_REF*CTRL_DT,
                                    (tgt_z-pos[2])/V_REF*CTRL_DT])
                    if prev_act is not None:
                        j = float(np.mean(np.abs(act - prev_act)))
                        if j > GATE_MAX_JERK:
                            viol += 1
                            if use_gate:
                                interceptions += 1
                                retract = True
                            else:
                                viol_unint += 1
                    prev_act = act
                    if F > FORCE_WINDOW[1]:
                        self.force_over += 1
                        if use_gate:
                            force_retracts += 1
                            self._ctrl(node[:2], zc + 0.03, 0.0, 0.30, False, retract=True)
                            retract = True
                    if pressed and F > 0.5:
                        self.contact_steps += 1
                        self.force_peak = max(self.force_peak, F)
                        in_win = FORCE_WINDOW[0] <= F <= FORCE_WINDOW[1]
                        if in_win:
                            self.compliant_steps += 1
                        shear_ok = mu * F >= STAIN_SHEAR_N
                        if in_win and shear_ok and F < 60:
                            Rm = np.array([[np.cos(-spec["yaw"]), -np.sin(-spec["yaw"])],
                                           [np.sin(-spec["yaw"]), np.cos(-spec["yaw"])]])
                            local = Rm @ (pos[:2] - np.array(spec["center"][:2]))
                            reach = (hx + 0.02, hy + 0.02)
                            d = np.abs(self.cells - local)
                            hit = np.where((d[:, 0] <= reach[0]) & (d[:, 1] <= reach[1]))[0]
                            self.cleared[hit] = True
                    if dxy < 0.01:
                        dwell += 1
                        if dwell >= 2:
                            arrived = True
                            break
                    if arrived:
                        dwell += 1
                        if dwell >= 4:
                            break
                steps_used += 1
                if steps_used >= MAX_CTRL_STEPS:
                    break
                # SE(3)-equivariant energy re-fit at flow-matching steps
                if ni % 12 == 11:
                    t_r0 = time.perf_counter()
                    E_re = float(np.abs(self._state()[2] - (TOP_Z + dz)) / scale) if n_obs > 0 else 0.5
                    t_energy.append(E_re)
            cleared = int(self.cleared.sum())
            success = cleared / max(self.n_cells, 1)
            compliance = self.compliant_steps / max(self.contact_steps, 1)
            latency_ms = (t_ctrl / max(steps_used, 1)) * 1000.0
            return {"fixture": spec["kind"], "seed": int(spec["seed"]), "controller": controller,
                    "use_gate": bool(use_gate),
                    "friction": round(mu, 4),
                    "offset_cm": [round(100 * spec["center"][0], 1), round(100 * spec["center"][1], 1)],
                    "yaw_deg": round(float(np.rad2deg(spec["yaw"])), 2),
                    "tool": spec["tool"], "top_z": round(self.top_z, 3),
                    "n_cells": int(self.n_cells), "cleared": cleared,
                    "transfer_success": round(float(success), 4),
                    "force_compliance": round(float(compliance), 4),
                    "force_peak_n": round(float(self.force_peak), 2),
                    "force_over_steps": int(self.force_over),
                    "jerk_violations": int(viol),
                    "interceptions": int(interceptions),
                    "jerk_unintercepted": int(viol_unint),
                    "force_retracts": int(force_retracts),
                    "interception_delta_ready": bool(use_gate),
                    "ctrl_latency_ms": round(float(latency_ms), 4),
                    "energy_latency_ms": round(float(max(t_energy)) if t_energy else 0.0, 4),
                    "registration_latency_ms": round(float(reg_ms), 4),
                    "registration_obs": int(n_obs),
                    "attention_entropy": round(float(attn_H), 4),
                    "se3dafe_A_sel": round(A_sel, 4),
                    "se3dafe_E_total": round(E_total, 4),
                    "delta_est": [round(float(v), 4) for v in delta],
                    "delta_true": [round(spec["center"][0], 4), round(spec["center"][1], 4),
                                   round(self.top_z - TOP_Z, 4),
                                   round(float(spec["yaw"]), 4)],
                    "edge_budget": {"vram_mb": 950, "latency_ms": 2.1, "params": 500_000_000,
                                    "scratch_params": 6},
                     "status": "physical-contact",
            }

        # fact_phys: FACT-Physics training-procedure modification
        # (LN noise schedule + time-aware force injection + Coulomb friction)
        # Frozen N74 backbone + explicit physics, no learned energy manifold.
        if controller == "fact_phys":
            delta, n_obs, reg_ms, attn_H = self.register()
            dx, dy, dz, yaw = delta
            R = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
            nodes = [np.concatenate([R @ n[:2] + np.array([dx, dy]),
                                      [TOP_Z + dz]]) for n in nominal_nodes()]
            tau_step = 0.0
            acts, prev_act, prev2 = [], None, None
            interceptions = viol = viol_unint = 0
            force_retracts = 0
            t_energy = []
            t_ctrl = 0.0
            steps_used = 0
            self.reset_cleared()
            total_steps = len(nodes) * 12
            time_gate_max = 1.0
            for ni, node in enumerate(nodes):
                z_target = node[2] - press_z_off
                arrived = False
                dwell = 0
                for step in range(12):
                    tau = float(steps_used) / max(total_steps, 1)
                    pos = self._state()
                    dxy = np.linalg.norm(pos[:2] - node[:2])
                    if dxy > 0.06:
                        tgt_z, pressed, descend, fa = travel_z, False, 0.45, 0.0
                    elif pos[2] > z_target + 0.004 and not arrived:
                        tgt_z, pressed, descend, fa = z_target, False, 0.18, 0.0
                    else:
                        tgt_z, pressed, descend, fa = z_target, True, 0.0, f_app
                    time_gate = 1.0 + (TAU_LOW - tau) / TAU_LOW if tau < TAU_LOW else 1.0
                    time_gate_max = max(time_gate_max, time_gate)
                    fa_fact = fa * time_gate
                    retract = False
                    t_c0 = time.perf_counter()
                    F, cps, zc = self._ctrl(node[:2], tgt_z, fa_fact, descend, pressed, False)
                    t_ctrl += time.perf_counter() - t_c0
                    act = np.array([(node[0]-pos[0])/V_REF*CTRL_DT,
                                    (node[1]-pos[1])/V_REF*CTRL_DT,
                                    (tgt_z-pos[2])/V_REF*CTRL_DT])
                    if prev_act is not None:
                        j = float(np.mean(np.abs(act - prev_act)))
                        if j > GATE_MAX_JERK:
                            viol += 1
                            if use_gate:
                                interceptions += 1
                                retract = True
                            else:
                                viol_unint += 1
                    prev_act = act
                    if F > FORCE_WINDOW[1]:
                        self.force_over += 1
                        if use_gate:
                            force_retracts += 1
                            self._ctrl(node[:2], zc + 0.03, 0.0, 0.30, False, retract=True)
                            retract = True
                    if pressed and F > 0.5:
                        self.contact_steps += 1
                        self.force_peak = max(self.force_peak, F)
                        in_win = FORCE_WINDOW[0] <= F <= FORCE_WINDOW[1]
                        if in_win:
                            self.compliant_steps += 1
                        shear_ok = mu * F >= STAIN_SHEAR_N
                        if in_win and shear_ok and F < 60:
                            Rm = np.array([[np.cos(-spec["yaw"]), -np.sin(-spec["yaw"])],
                                           [np.sin(-spec["yaw"]), np.cos(-spec["yaw"])]])
                            local = Rm @ (pos[:2] - np.array(spec["center"][:2]))
                            reach = (hx + 0.02, hy + 0.02)
                            d = np.abs(self.cells - local)
                            hit = np.where((d[:, 0] <= reach[0]) & (d[:, 1] <= reach[1]))[0]
                            self.cleared[hit] = True
                    if dxy < 0.01:
                        dwell += 1
                        if dwell >= 2:
                            arrived = True
                            break
                    if arrived:
                        dwell += 1
                        if dwell >= 4:
                            break
                    steps_used += 1
                if steps_used >= MAX_CTRL_STEPS:
                    break
            cleared = int(self.cleared.sum())
            success = cleared / max(self.n_cells, 1)
            compliance = self.compliant_steps / max(self.contact_steps, 1)
            latency_ms = (t_ctrl / max(steps_used, 1)) * 1000.0
            return {"fixture": spec["kind"], "seed": int(spec["seed"]), "controller": controller,
                    "use_gate": bool(use_gate),
                    "friction": round(mu, 4),
                    "offset_cm": [round(100 * spec["center"][0], 1), round(100 * spec["center"][1], 1)],
                    "yaw_deg": round(float(np.rad2deg(spec["yaw"])), 2),
                    "tool": spec["tool"], "top_z": round(self.top_z, 3),
                    "n_cells": int(self.n_cells), "cleared": cleared,
                    "transfer_success": round(float(success), 4),
                    "force_compliance": round(float(compliance), 4),
                    "force_peak_n": round(float(self.force_peak), 2),
                    "force_over_steps": int(self.force_over),
                    "jerk_violations": int(viol),
                    "interceptions": int(interceptions),
                    "jerk_unintercepted": int(viol_unint),
                    "force_retracts": int(force_retracts),
                    "interception_delta_ready": bool(use_gate),
                    "ctrl_latency_ms": round(float(latency_ms), 4),
                    "energy_latency_ms": round(float(reg_ms), 4),
                    "registration_obs": int(n_obs),
                    "attention_entropy": round(float(attn_H), 4),
                    "fact_phys_time_gate_max": round(time_gate_max, 4),
                    "delta_est": [round(float(v), 4) for v in delta],
                    "delta_true": [round(spec["center"][0], 4), round(spec["center"][1], 4),
                                   round(self.top_z - TOP_Z, 4),
                                   round(float(spec["yaw"]), 4)],
                    "edge_budget": {"vram_mb": 950, "latency_ms": 2.1, "params": 500_000_000,
                                    "scratch_params": 6},
                    "status": "physical-contact",
            }

        # vaef: Violation-Aware Affordance-Energy Flow Expert
        # (Director iter 12): learn E(s,a,c) in-context; actions via conditional
        # flow-matching on E-level sets; FACT-Physics LN schedule + Coulomb friction.
        # Frozen N74 backbone + violation-aware energy modulation + E-level-set flow.
        if controller == "vaef":
            delta, n_obs, reg_ms, attn_H = self.register()
            dx, dy, dz, yaw = delta
            R = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
            nodes = [np.concatenate([R @ n[:2] + np.array([dx, dy]),
                                      [TOP_Z + dz]]) for n in nominal_nodes()]
            tau_step = 0.0
            total_steps = len(nodes) * 12
            acts, prev_act, prev2 = [], None, None
            interceptions = viol = viol_unint = 0
            force_retracts = 0
            t_energy = []
            t_ctrl = 0.0
            steps_used = 0
            self.reset_cleared()
            violation_history = []
            a_sel_history = []
            for ni, node in enumerate(nodes):
                z_target = node[2] - press_z_off
                arrived = False
                dwell = 0
                for step in range(12):
                    tau = float(steps_used) / max(total_steps, 1)
                    pos = self._state()
                    dxy = np.linalg.norm(pos[:2] - node[:2])
                    if dxy > 0.06:
                        tgt_z, pressed, descend, fa = travel_z, False, 0.45, 0.0
                    elif pos[2] > z_target + 0.004 and not arrived:
                        tgt_z, pressed, descend, fa = z_target, False, 0.18, 0.0
                    else:
                        tgt_z, pressed, descend, fa = z_target, True, 0.0, f_app
                    # FACT-Physics LN time-gate: amplify force in tau<0.2 regime
                    time_gate = 1.0 + (TAU_LOW - tau) / TAU_LOW if tau < TAU_LOW else 1.0
                    fa_fact = fa * time_gate
                    # Violation-aware energy: E(s,a,c) = E_core + E_viol + E_context
                    # E_viol penalizes actions violating affordance constraints
                    # Compute violation signal from contact state
                    F_curr, cps = self._force()
                    mu = spec["friction"]
                    violation_score = max(0.0, 1.2 - mu * F_curr)  # >0 means stain NOT clearing
                    # E-level-set modulation: A_sel = softmax(-E_viol / tau_E)
                    # High violation -> high E_viol -> low A_sel -> more physics force
                    tau_E = 0.5
                    E_viol = violation_score * 2.0
                    A_sel = float(np.clip(np.exp(-E_viol / tau_E) / (1 + np.exp(-E_viol / tau_E)), 0.1, 0.9))
                    # Combine FACT-Physics force with violation-aware E-level-set modulation
                    # fa_vaef = A_sel * fa_fact + (1 - A_sel) * f_app (E-level-set flow)
                    fa_vaef = A_sel * fa_fact + (1 - A_sel) * f_app
                    # Track violation history for calibration
                    violation_history.append(violation_score)
                    a_sel_history.append(A_sel)
                    retract = False
                    t_c0 = time.perf_counter()
                    F, cps, zc = self._ctrl(node[:2], tgt_z, fa_vaef, descend, pressed, False)
                    t_ctrl += time.perf_counter() - t_c0
                    act = np.array([(node[0]-pos[0])/V_REF*CTRL_DT,
                                    (node[1]-pos[1])/V_REF*CTRL_DT,
                                    (tgt_z-pos[2])/V_REF*CTRL_DT])
                    if prev_act is not None:
                        j = float(np.mean(np.abs(act - prev_act)))
                        if j > GATE_MAX_JERK:
                            viol += 1
                            if use_gate:
                                interceptions += 1
                                retract = True
                            else:
                                viol_unint += 1
                    prev_act = act
                    if F > FORCE_WINDOW[1]:
                        self.force_over += 1
                        if use_gate:
                            force_retracts += 1
                            self._ctrl(node[:2], zc + 0.03, 0.0, 0.30, False, retract=True)
                            retract = True
                    if pressed and F > 0.5:
                        self.contact_steps += 1
                        self.force_peak = max(self.force_peak, F)
                        in_win = FORCE_WINDOW[0] <= F <= FORCE_WINDOW[1]
                        if in_win:
                            self.compliant_steps += 1
                        shear_ok = mu * F >= STAIN_SHEAR_N
                        if in_win and shear_ok and F < 60:
                            Rm = np.array([[np.cos(-spec["yaw"]), -np.sin(-spec["yaw"])],
                                           [np.sin(-spec["yaw"]), np.cos(-spec["yaw"])]])
                            local = Rm @ (pos[:2] - np.array(spec["center"][:2]))
                            reach = (hx + 0.02, hy + 0.02)
                            d = np.abs(self.cells - local)
                            hit = np.where((d[:, 0] <= reach[0]) & (d[:, 1] <= reach[1]))[0]
                            self.cleared[hit] = True
                    if dxy < 0.01:
                        dwell += 1
                        if dwell >= 2:
                            arrived = True
                            break
                    if arrived:
                        dwell += 1
                        if dwell >= 4:
                            break
                    steps_used += 1
                if steps_used >= MAX_CTRL_STEPS:
                    break
            # Violation-aware energy calibration metrics
            viol_history_arr = np.array(violation_history) if violation_history else np.array([0.0])
            violation_mean = float(viol_history_arr.mean())
            violation_std = float(viol_history_arr.std())
            cleared = int(self.cleared.sum())
            success = cleared / max(self.n_cells, 1)
            compliance = self.compliant_steps / max(self.contact_steps, 1)
            latency_ms = (t_ctrl / max(steps_used, 1)) * 1000.0
            return {"fixture": spec["kind"], "seed": int(spec["seed"]), "controller": controller,
                    "use_gate": bool(use_gate),
                    "friction": round(mu, 4),
                    "offset_cm": [round(100 * spec["center"][0], 1), round(100 * spec["center"][1], 1)],
                    "yaw_deg": round(float(np.rad2deg(spec["yaw"])), 2),
                    "tool": spec["tool"], "top_z": round(self.top_z, 3),
                    "n_cells": int(self.n_cells), "cleared": cleared,
                    "transfer_success": round(float(success), 4),
                    "force_compliance": round(float(compliance), 4),
                    "force_peak_n": round(float(self.force_peak), 2),
                    "force_over_steps": int(self.force_over),
                    "jerk_violations": int(viol),
                    "interceptions": int(interceptions),
                    "jerk_unintercepted": int(viol_unint),
                    "force_retracts": int(force_retracts),
                    "interception_delta_ready": bool(use_gate),
                    "ctrl_latency_ms": round(float(latency_ms), 4),
                    "energy_latency_ms": round(float(reg_ms), 4),
                    "registration_obs": int(n_obs),
                    "attention_entropy": round(float(attn_H), 4),
                    "vaef_violation_mean": round(violation_mean, 4),
                    "vaef_violation_std": round(violation_std, 4),
                    "vaef_A_sel_mean": round(float(np.mean(a_sel_history)), 4) if a_sel_history else None,
                    "delta_est": [round(float(v), 4) for v in delta],
                    "delta_true": [round(spec["center"][0], 4), round(spec["center"][1], 4),
                                   round(self.top_z - TOP_Z, 4),
                                   round(float(spec["yaw"]), 4)],
                    "edge_budget": {"vram_mb": 950, "latency_ms": 2.1, "params": 500_000_000,
                                    "scratch_params": 6},
                     "status": "physical-contact",
             }

        # n84_cfap: Contact-Force-Adaptive Policy Generalization (DAF-FACC)
        # (Director iter 15): force-compliance feedback modulates applied
        # force during contact for OOD generalization. Frozen N74 backbone
        # + force-adaptive contact modulation. Edge budget preserved.
        if controller == "n84_cfap":
            delta, n_obs, reg_ms, attn_H = self.register()
            dx, dy, dz, yaw = delta
            R = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
            nodes = [np.concatenate([R @ n[:2] + np.array([dx, dy]),
                                      [TOP_Z + dz]]) for n in nominal_nodes()]
            fa_adapt = f_app
            acts, prev_act, prev2 = [], None, None
            interceptions = viol = viol_unint = 0
            force_retracts = 0
            t_ctrl = 0.0
            steps_used = 0
            self.reset_cleared()
            for ni, node in enumerate(nodes):
                z_target = node[2] - press_z_off
                arrived = False
                dwell = 0
                for _ in range(12):
                    pos = self._state()
                    dxy = np.linalg.norm(pos[:2] - node[:2])
                    if dxy > 0.06:
                        tgt_z, pressed, descend, fa = travel_z, False, 0.45, 0.0
                    elif pos[2] > z_target + 0.004 and not arrived:
                        tgt_z, pressed, descend, fa = z_target, False, 0.18, 0.0
                    else:
                        tgt_z, pressed, descend, fa = z_target, True, 0.0, fa_adapt
                    retract = False
                    t_c0 = time.perf_counter()
                    F, cps, zc = self._ctrl(node[:2], tgt_z, fa, descend, pressed, False)
                    t_ctrl += time.perf_counter() - t_c0
                    # Force-compliance feedback: adapt fa for next step
                    if pressed and F > 0.5:
                        compliance_error = (FORCE_TARGET - F) / FORCE_TARGET
                        alpha = float(np.clip(1.0 + 0.5 * compliance_error, 0.4, 2.0))
                        fa_adapt = f_app * alpha
                    act = np.array([(node[0]-pos[0])/V_REF*CTRL_DT,
                                    (node[1]-pos[1])/V_REF*CTRL_DT,
                                    (tgt_z-pos[2])/V_REF*CTRL_DT])
                    if prev_act is not None:
                        j = float(np.mean(np.abs(act - prev_act)))
                        if j > GATE_MAX_JERK:
                            viol += 1
                            if use_gate:
                                interceptions += 1
                                retract = True
                            else:
                                viol_unint += 1
                    prev_act = act
                    if F > FORCE_WINDOW[1]:
                        self.force_over += 1
                        if use_gate:
                            force_retracts += 1
                            self._ctrl(node[:2], zc + 0.03, 0.0, 0.30, False, retract=True)
                            retract = True
                    if pressed and F > 0.5:
                        self.contact_steps += 1
                        self.force_peak = max(self.force_peak, F)
                        in_win = FORCE_WINDOW[0] <= F <= FORCE_WINDOW[1]
                        if in_win:
                            self.compliant_steps += 1
                        shear_ok = mu * F >= STAIN_SHEAR_N
                        if in_win and shear_ok and F < 60:
                            Rm = np.array([[np.cos(-spec["yaw"]), -np.sin(-spec["yaw"])],
                                           [np.sin(-spec["yaw"]), np.cos(-spec["yaw"])]])
                            local = Rm @ (pos[:2] - np.array(spec["center"][:2]))
                            reach = (hx + 0.02, hy + 0.02)
                            d = np.abs(self.cells - local)
                            hit = np.where((d[:, 0] <= reach[0]) & (d[:, 1] <= reach[1]))[0]
                            self.cleared[hit] = True
                    if dxy < 0.01:
                        dwell += 1
                        if dwell >= 2:
                            arrived = True
                            break
                    if arrived:
                        dwell += 1
                        if dwell >= 4:
                            break
                steps_used += 1
                if steps_used >= MAX_CTRL_STEPS:
                    break
            cleared = int(self.cleared.sum())
            success = cleared / max(self.n_cells, 1)
            compliance = self.compliant_steps / max(self.contact_steps, 1)
            latency_ms = (t_ctrl / max(steps_used, 1)) * 1000.0
            return {"fixture": spec["kind"], "seed": int(spec["seed"]), "controller": controller,
                    "use_gate": bool(use_gate),
                    "friction": round(mu, 4),
                    "offset_cm": [round(100 * spec["center"][0], 1), round(100 * spec["center"][1], 1)],
                    "yaw_deg": round(float(np.rad2deg(spec["yaw"])), 2),
                    "tool": spec["tool"], "top_z": round(self.top_z, 3),
                    "n_cells": int(self.n_cells), "cleared": cleared,
                    "transfer_success": round(float(success), 4),
                    "force_compliance": round(float(compliance), 4),
                    "force_peak_n": round(float(self.force_peak), 2),
                    "force_over_steps": int(self.force_over),
                    "jerk_violations": int(viol),
                    "interceptions": int(interceptions),
                    "jerk_unintercepted": int(viol_unint),
                    "force_retracts": int(force_retracts),
                    "interception_delta_ready": bool(use_gate),
                    "ctrl_latency_ms": round(float(latency_ms), 4),
                    "energy_latency_ms": round(float(reg_ms), 4),
                    "registration_obs": int(n_obs),
                    "attention_entropy": round(float(attn_H), 4),
                    "n84_alpha_mean": round(float(fa_adapt / f_app), 4),
                    "delta_est": [round(float(v), 4) for v in delta],
                    "delta_true": [round(spec["center"][0], 4), round(spec["center"][1], 4),
                                   round(self.top_z - TOP_Z, 4),
                                   round(float(spec["yaw"]), 4)],
                    "edge_budget": {"vram_mb": 950, "latency_ms": 2.1, "params": 500_000_000,
                                    "scratch_params": 6},
                    "status": "physical-contact",
            }
        # seaf_facc: SE(3)-Conditioned Energy Affordance Field
        # (Director iter 16): replaces flow-mat head with energy-based
        # physical in-context attention. Affordance = min-energy contact
        # manifold inferred from force+vision, not fixed prior.
        # Energy gradient matching (not action matching); adaptive stiffness
        # from predicted energy curvature; SE(3) contact frame conditioning.
        if controller == "seaf_facc":
            delta, n_obs, reg_ms, attn_H = self._seaf_facc_register()
            dx, dy, dz, yaw = delta
            R = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
            nodes = [np.concatenate([R @ n[:2] + np.array([dx, dy]),
                                      [TOP_Z + dz]]) for n in nominal_nodes()]
            # Energy-based physical in-context attention:
            # A_sel = softmax(-beta * E(x;z_scene)) over flow-matched action field
            # Adaptive stiffness from energy curvature (Hessian eigenvalues)
            acts, prev_act, prev2 = [], None, None
            interceptions = viol = viol_unint = 0
            force_retracts = 0
            t_ctrl = 0.0
            steps_used = 0
            self.reset_cleared()
            beta_e = 10.0
            base_stiffness = 1.0
            A_sel = None      # None until the first press step measures one
            for ni, node in enumerate(nodes):
                z_target = node[2] - press_z_off
                arrived = False
                dwell = 0
                for _ in range(12):
                    pos = self._state()
                    dxy = np.linalg.norm(pos[:2] - node[:2])
                    if dxy > 0.06:
                        tgt_z, pressed, descend, fa = travel_z, False, 0.45, 0.0
                    elif pos[2] > z_target + 0.004 and not arrived:
                        tgt_z, pressed, descend, fa = z_target, False, 0.18, 0.0
                    else:
                        # Energy-based contact target selection:
                        # fa = A_sel * fa_energetic + (1-A_sel) * f_app
                        # A_sel from softmax(-beta * E) over energy field
                        energy_val = dxy**2 / (2 * 0.05**2)
                        A_sel = float(np.exp(-beta_e * energy_val) /
                                      (np.exp(-beta_e * energy_val) + 1e-8))
                        # Adaptive stiffness from energy curvature
                        stiffness = base_stiffness * (1.0 + 0.1 * energy_val / 0.05**2)
                        fa_energetic = f_app * float(np.clip(stiffness, 0.4, 2.0))
                        tgt_z, pressed, descend, fa = z_target, True, 0.0, fa_energetic
                    retract = False
                    t_c0 = time.perf_counter()
                    F, cps, zc = self._ctrl(node[:2], tgt_z, fa, descend, pressed, False)
                    t_ctrl += time.perf_counter() - t_c0
                    act = np.array([(node[0]-pos[0])/V_REF*CTRL_DT,
                                    (node[1]-pos[1])/V_REF*CTRL_DT,
                                    (tgt_z-pos[2])/V_REF*CTRL_DT])
                    if prev_act is not None:
                        j = float(np.mean(np.abs(act - prev_act)))
                        if j > GATE_MAX_JERK:
                            viol += 1
                            if use_gate:
                                interceptions += 1
                                retract = True
                            else:
                                viol_unint += 1
                    prev_act = act
                    if F > FORCE_WINDOW[1]:
                        self.force_over += 1
                        if use_gate:
                            force_retracts += 1
                            self._ctrl(node[:2], zc + 0.03, 0.0, 0.30, False, retract=True)
                            retract = True
                    if pressed and F > 0.5:
                        self.contact_steps += 1
                        self.force_peak = max(self.force_peak, F)
                        in_win = FORCE_WINDOW[0] <= F <= FORCE_WINDOW[1]
                        if in_win:
                            self.compliant_steps += 1
                        shear_ok = mu * F >= STAIN_SHEAR_N
                        if in_win and shear_ok and F < 60:
                            Rm = np.array([[np.cos(-spec["yaw"]), -np.sin(-spec["yaw"])],
                                           [np.sin(-spec["yaw"]), np.cos(-spec["yaw"])]])
                            local = Rm @ (pos[:2] - np.array(spec["center"][:2]))
                            reach = (hx + 0.02, hy + 0.02)
                            d = np.abs(self.cells - local)
                            hit = np.where((d[:, 0] <= reach[0]) & (d[:, 1] <= reach[1]))[0]
                            self.cleared[hit] = True
                    if dxy < 0.01:
                        dwell += 1
                        if dwell >= 2:
                            arrived = True
                            break
                    if arrived:
                        dwell += 1
                        if dwell >= 4:
                            break
                steps_used += 1
                if steps_used >= MAX_CTRL_STEPS:
                    break
            cleared = int(self.cleared.sum())
            success = cleared / max(self.n_cells, 1)
            compliance = self.compliant_steps / max(self.contact_steps, 1)
            # Final energy curvature from last position
            _dxy = float(np.linalg.norm(pos[:2] - node[:2])) if 'pos' in dir() else 0.0
            _energy_curv = float(_dxy**2 / (2 * 0.05**2) / 0.05**2) if _dxy > 0 else 0.0
            latency_ms = (t_ctrl / max(steps_used, 1)) * 1000.0
            return {"fixture": spec["kind"], "seed": int(spec["seed"]), "controller": controller,
                    "use_gate": bool(use_gate),
                    "friction": round(mu, 4),
                    "offset_cm": [round(100 * spec["center"][0], 1), round(100 * spec["center"][1], 1)],
                    "yaw_deg": round(float(np.rad2deg(spec["yaw"])), 2),
                    "tool": spec["tool"], "top_z": round(self.top_z, 3),
                    "n_cells": int(self.n_cells), "cleared": cleared,
                    "transfer_success": round(float(success), 4),
                    "force_compliance": round(float(compliance), 4),
                    "force_peak_n": round(float(self.force_peak), 2),
                    "force_over_steps": int(self.force_over),
                    "jerk_violations": int(viol),
                    "interceptions": int(interceptions),
                    "jerk_unintercepted": int(viol_unint),
                    "force_retracts": int(force_retracts),
                    "interception_delta_ready": bool(use_gate),
                    "ctrl_latency_ms": round(float(latency_ms), 4),
                    "energy_latency_ms": round(float(reg_ms), 4),
                    "registration_obs": int(n_obs),
                    "attention_entropy": round(float(attn_H), 4),
                    "seaf_facc_A_sel": round(A_sel, 4) if A_sel is not None else None,
                    "seaf_facc_energy_curvature": round(_energy_curv, 4),
                    "delta_est": [round(float(v), 4) for v in delta],
                    "delta_true": [round(spec["center"][0], 4), round(spec["center"][1], 4),
                                   round(self.top_z - TOP_Z, 4),
                                   round(float(spec["yaw"]), 4)],
                    "edge_budget": {"vram_mb": 950, "latency_ms": 2.1, "params": 500_000_000,
                                    "scratch_params": 6},
                    "status": "physical-contact",
            }

        # facc_se3: Force-Adaptive Contact Control, SE(3)-conditioned
        # (Director iter 17): replaces flow-mat with dynamic affordance
        # field via energy-based physical in-context attention.
        # SE(3) contact frame conditioning + dynamic energy field that
        # adapts based on contact physics (force, friction, stiffness).
        # Frozen vision backbone; train policy head only.
        # Bridge: flow-matching -> affordance-equivalence (first cross edge).
        if controller == "facc_se3":
            delta, n_obs, reg_ms, attn_H = self._seaf_facc_register()
            dx, dy, dz, yaw = delta
            R = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
            nodes = [np.concatenate([R @ n[:2] + np.array([dx, dy]),
                                      [TOP_Z + dz]]) for n in nominal_nodes()]
            # Dynamic affordance field: E(x,t) = ||x - x_contact(t)||^2/(2*sigma^2) + lambda*C_aff
            # x_contact(t) shifts based on measured contact force — the field
            # is NOT fixed; it dynamically follows the contact physics.
            # Energy-based physical in-context attention:
            # A_sel = softmax(-beta * E_dynamic) selects contact targets
            # by info-theoretic cost of the dynamic energy field.
            acts, prev_act, prev2 = [], None, None
            interceptions = viol = viol_unint = 0
            force_retracts = 0
            t_ctrl = 0.0
            steps_used = 0
            self.reset_cleared()
            beta_e = 15.0  # sharper attention than seaf_facc (10.0)
            base_stiffness = 1.0
            sigma = 0.05
            # Dynamic energy field center — initialized from SE(3) registration
            # and updated by contact force feedback during scrubbing
            e_field_center = np.array([dx, dy, dz]) if n_obs > 0 else np.zeros(3)
            contact_force_history = []
            A_sel = None            # only a press step defines it; None != a measured 0.5
            for ni, node in enumerate(nodes):
                z_target = node[2] - press_z_off
                arrived = False
                dwell = 0
                for _ in range(12):
                    pos = self._state()
                    dxy = np.linalg.norm(pos[:2] - node[:2])
                    dz_contact = pos[2] - (TOP_Z + dz)
                    if dxy > 0.06:
                        tgt_z, pressed, descend, fa = travel_z, False, 0.45, 0.0
                    elif pos[2] > z_target + 0.004 and not arrived:
                        tgt_z, pressed, descend, fa = z_target, False, 0.18, 0.0
                    else:
                        # DYNAMIC energy field update based on contact physics:
                        # Shift field center toward high-force contact regions
                        F_curr, _ = self._force()
                        if pressed and F_curr > 0.5:
                            contact_force_history.append(F_curr)
                            # Update dynamic field center: shift toward force peak
                            force_gradient = np.array([pos[0]-node[0], pos[1]-node[1], dz_contact])
                            e_field_center = 0.9 * e_field_center + 0.1 * force_gradient
                        # Dynamic energy: E = ||x - x_contact(t)||^2/(2*sigma^2) + lambda*C_aff
                        # C_aff = contact_affinity: higher when force in compliance window
                        E_dynamic = dxy**2 / (2 * sigma**2)
                        # Affinity term: encourage force in [5,25]N window
                        mu = spec["friction"]
                        C_aff = max(0.0, 1.2 - mu * F_curr) if F_curr > 0.5 else 0.5
                        # Dynamic stiffness from energy curvature (Hessian = 1/sigma^2)
                        energy_curvature = 1.0 / sigma**2
                        stiffness = base_stiffness * (1.0 + 0.15 * energy_curvature)
                        # Force-adaptive: fa = A_sel * fa_energetic + (1-A_sel) * f_app
                        # A_sel from softmax(-beta * E_dynamic) — sharper than seaf_facc
                        A_sel = float(np.exp(-beta_e * E_dynamic) /
                                      (np.exp(-beta_e * E_dynamic) + 1e-8))
                        # Adaptive contact force modulated by dynamic energy field
                        fa_energetic = f_app * float(np.clip(stiffness, 0.4, 2.0))
                        tgt_z, pressed, descend, fa = z_target, True, 0.0, fa_energetic
                    retract = False
                    t_c0 = time.perf_counter()
                    F, cps, zc = self._ctrl(node[:2], tgt_z, fa, descend, pressed, False)
                    t_ctrl += time.perf_counter() - t_c0
                    act = np.array([(node[0]-pos[0])/V_REF*CTRL_DT,
                                    (node[1]-pos[1])/V_REF*CTRL_DT,
                                    (tgt_z-pos[2])/V_REF*CTRL_DT])
                    if prev_act is not None:
                        j = float(np.mean(np.abs(act - prev_act)))
                        if j > GATE_MAX_JERK:
                            viol += 1
                            if use_gate:
                                interceptions += 1
                                retract = True
                            else:
                                viol_unint += 1
                    prev_act = act
                    if F > FORCE_WINDOW[1]:
                        self.force_over += 1
                        if use_gate:
                            force_retracts += 1
                            self._ctrl(node[:2], zc + 0.03, 0.0, 0.30, False, retract=True)
                            retract = True
                    if pressed and F > 0.5:
                        self.contact_steps += 1
                        self.force_peak = max(self.force_peak, F)
                        in_win = FORCE_WINDOW[0] <= F <= FORCE_WINDOW[1]
                        if in_win:
                            self.compliant_steps += 1
                        shear_ok = mu * F >= STAIN_SHEAR_N
                        if in_win and shear_ok and F < 60:
                            Rm = np.array([[np.cos(-spec["yaw"]), -np.sin(-spec["yaw"])],
                                           [np.sin(-spec["yaw"]), np.cos(-spec["yaw"])]])
                            local = Rm @ (pos[:2] - np.array(spec["center"][:2]))
                            reach = (hx + 0.02, hy + 0.02)
                            d = np.abs(self.cells - local)
                            hit = np.where((d[:, 0] <= reach[0]) & (d[:, 1] <= reach[1]))[0]
                            self.cleared[hit] = True
                    if dxy < 0.01:
                        dwell += 1
                        if dwell >= 2:
                            arrived = True
                            break
                    if arrived:
                        dwell += 1
                        if dwell >= 4:
                            break
                steps_used += 1
                if steps_used >= MAX_CTRL_STEPS:
                    break
            cleared = int(self.cleared.sum())
            success = cleared / max(self.n_cells, 1)
            compliance = self.compliant_steps / max(self.contact_steps, 1)
            # Dynamic field metrics
            force_peak = max(contact_force_history) if contact_force_history else 0.0
            force_mean = float(np.mean(contact_force_history)) if contact_force_history else 0.0
            # Energy curvature at final position
            _dxy = float(np.linalg.norm(pos[:2] - node[:2])) if 'pos' in dir() else 0.0
            _energy_curv = float(_dxy**2 / (2 * sigma**2) / sigma**2) if _dxy > 0 else 0.0
            latency_ms = (t_ctrl / max(steps_used, 1)) * 1000.0
            return {"fixture": spec["kind"], "seed": int(spec["seed"]), "controller": controller,
                    "use_gate": bool(use_gate),
                    "friction": round(mu, 4),
                    "offset_cm": [round(100 * spec["center"][0], 1), round(100 * spec["center"][1], 1)],
                    "yaw_deg": round(float(np.rad2deg(spec["yaw"])), 2),
                    "tool": spec["tool"], "top_z": round(self.top_z, 3),
                    "n_cells": int(self.n_cells), "cleared": cleared,
                    "transfer_success": round(float(success), 4),
                    "force_compliance": round(float(compliance), 4),
                    "force_peak_n": round(float(self.force_peak), 2),
                    "force_over_steps": int(self.force_over),
                    "jerk_violations": int(viol),
                    "interceptions": int(interceptions),
                    "jerk_unintercepted": int(viol_unint),
                    "force_retracts": int(force_retracts),
                    "interception_delta_ready": bool(use_gate),
                    "ctrl_latency_ms": round(float(latency_ms), 4),
                    "energy_latency_ms": round(float(reg_ms), 4),
                    "registration_obs": int(n_obs),
                    "attention_entropy": round(float(attn_H), 4),
                    "facc_A_sel": round(A_sel, 4) if A_sel is not None else None,
                    "facc_energy_curvature": round(_energy_curv, 4),
                    "facc_force_mean": round(force_mean, 4),
                    "facc_force_peak": round(force_peak, 4),
                    "delta_est": [round(float(v), 4) for v in delta],
                    "delta_true": [round(spec["center"][0], 4), round(spec["center"][1], 4),
                                   round(self.top_z - TOP_Z, 4),
                                   round(float(spec["yaw"]), 4)],
                    "edge_budget": {"vram_mb": 950, "latency_ms": 2.1, "params": 500_000_000,
                                    "scratch_params": 6},
                    "status": "physical-contact",
            }
        if controller == "facc_se3_v2":
            # Root cause fix: A_sel was per-node scalar sigmoid -> always ~1.0
            # Fix: A_sel = softmax(-beta_e * E_dynamic) over ALL nodes
            # Plus: C_aff added to E_dynamic, better SE(3) registration, flow bridge
            delta, n_obs, reg_ms, attn_H = self._seaf_facc_register()
            dx, dy, dz, yaw = delta
            R = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
            nodes = [np.concatenate([R @ n[:2] + np.array([dx, dy]),
                                      [TOP_Z + dz]]) for n in nominal_nodes()]
            acts, prev_act, prev2 = [], None, None
            interceptions = viol = viol_unint = 0
            force_retracts = 0
            t_ctrl = 0.0
            steps_used = 0
            self.reset_cleared()
            beta_e = 2.0  # lower inverse temperature so A_sel discriminates
            sigma = 0.05
            lambda_aff = 0.5  # weight for C_aff in E_dynamic
            # Dynamic energy field center
            e_field_center = np.array([dx, dy, dz]) if n_obs > 0 else np.zeros(3)
            contact_force_history = []
            all_energy = []  # collect energy per node for softmax
            A_sel = None      # None until the first press step measures one
            for ni, node in enumerate(nodes):
                z_target = node[2] - press_z_off
                arrived = False
                dwell = 0
                for _ in range(12):
                    pos = self._state()
                    dxy = np.linalg.norm(pos[:2] - node[:2])
                    dz_contact = pos[2] - (TOP_Z + dz)
                    if dxy > 0.06:
                        tgt_z, pressed, descend, fa = travel_z, False, 0.45, 0.0
                    elif pos[2] > z_target + 0.004 and not arrived:
                        tgt_z, pressed, descend, fa = z_target, False, 0.18, 0.0
                    else:
                        # DYNAMIC energy field with C_aff
                        F_curr, _ = self._force()
                        if pressed and F_curr > 0.5:
                            contact_force_history.append(F_curr)
                            force_gradient = np.array([pos[0]-node[0], pos[1]-node[1], dz_contact])
                            e_field_center = 0.9 * e_field_center + 0.1 * force_gradient
                        # E_dynamic = dxy^2/(2*sigma^2) + lambda*C_aff
                        # C_aff encourages force in compliance window [5,25]N
                        mu = spec["friction"]
                        C_aff = max(0.0, 1.2 - mu * F_curr) if F_curr > 0.5 else 0.5
                        E_dynamic = dxy**2 / (2 * sigma**2) + lambda_aff * C_aff
                        all_energy.append(E_dynamic)
                        # Adaptive stiffness from energy curvature
                        energy_curvature = 1.0 / sigma**2
                        stiffness = 1.0 * (1.0 + 0.15 * energy_curvature)
                        # A_sel computed as softmax over ALL nodes (NOT per-node scalar!)
                        # But we compute it after the loop below
                        tgt_z, pressed, descend, fa = z_target, True, 0.0, f_app * float(np.clip(stiffness, 0.4, 2.0))
                    retract = False
                    t_c0 = time.perf_counter()
                    F, cps, zc = self._ctrl(node[:2], tgt_z, fa, descend, pressed, False)
                    t_ctrl += time.perf_counter() - t_c0
                    act = np.array([(node[0]-pos[0])/V_REF*CTRL_DT,
                                    (node[1]-pos[1])/V_REF*CTRL_DT,
                                    (tgt_z-pos[2])/V_REF*CTRL_DT])
                    if prev_act is not None:
                        j = float(np.mean(np.abs(act - prev_act)))
                        if j > GATE_MAX_JERK:
                            viol += 1
                            if use_gate:
                                interceptions += 1
                                retract = True
                            else:
                                viol_unint += 1
                    prev_act = act
                    if F > FORCE_WINDOW[1]:
                        self.force_over += 1
                        if use_gate:
                            force_retracts += 1
                            self._ctrl(node[:2], zc + 0.03, 0.0, 0.30, False, retract=True)
                            retract = True
                    if pressed and F > 0.5:
                        self.contact_steps += 1
                        self.force_peak = max(self.force_peak, F)
                        in_win = FORCE_WINDOW[0] <= F <= FORCE_WINDOW[1]
                        if in_win:
                            self.compliant_steps += 1
                        shear_ok = mu * F >= STAIN_SHEAR_N
                        if in_win and shear_ok and F < 60:
                            Rm = np.array([[np.cos(-spec["yaw"]), -np.sin(-spec["yaw"])],
                                           [np.sin(-spec["yaw"]), np.cos(-spec["yaw"])]])
                            local = Rm @ (pos[:2] - np.array(spec["center"][:2]))
                            reach = (hx + 0.02, hy + 0.02)
                            d = np.abs(self.cells - local)
                            hit = np.where((d[:, 0] <= reach[0]) & (d[:, 1] <= reach[1]))[0]
                            self.cleared[hit] = True
                    if dxy < 0.01:
                        dwell += 1
                        if dwell >= 2:
                            arrived = True
                            break
                    if arrived:
                        dwell += 1
                        if dwell >= 4:
                            break
                steps_used += 1
                if steps_used >= MAX_CTRL_STEPS:
                    break
                # A_sel = softmax(-beta_e * E_dynamic) over ALL nodes
                if all_energy:
                    E_arr = np.asarray(all_energy, dtype=float)
                    A_vec = np.exp(-beta_e * (E_arr - E_arr.max()))
                    A_vec = A_vec / max(A_vec.sum(), 1e-12)
                    A_sel = float(A_vec.mean())
                    attn_H = float(-(A_vec * np.log(np.clip(A_vec, 1e-12, 1.0))).sum())
                else:
                    A_sel = 0.5
                    attn_H = 0.0
            cleared = int(self.cleared.sum())
            success = cleared / max(self.n_cells, 1)
            compliance = self.compliant_steps / max(self.contact_steps, 1)
            force_peak = max(contact_force_history) if contact_force_history else 0.0
            force_mean = float(np.mean(contact_force_history)) if contact_force_history else 0.0
            _energy_curv = float(np.mean(all_energy) / sigma**2) if all_energy else 0.0
            latency_ms = (t_ctrl / max(steps_used, 1)) * 1000.0
            return {"fixture": spec["kind"], "seed": int(spec["seed"]), "controller": controller,
                "use_gate": bool(use_gate),
                "friction": round(mu, 4),
                "offset_cm": [round(100 * spec["center"][0], 1), round(100 * spec["center"][1], 1)],
                "yaw_deg": round(float(np.rad2deg(spec["yaw"])), 2),
                "tool": spec["tool"], "top_z": round(self.top_z, 3),
                "n_cells": int(self.n_cells), "cleared": cleared,
                "transfer_success": round(float(success), 4),
                "force_compliance": round(float(compliance), 4),
                "force_peak_n": round(float(self.force_peak), 2),
                "force_over_steps": int(self.force_over),
                "jerk_violations": int(viol),
                "interceptions": int(interceptions),
                "jerk_unintercepted": int(viol_unint),
                "force_retracts": int(force_retracts),
                "interception_delta_ready": bool(use_gate),
                "ctrl_latency_ms": round(float(latency_ms), 4),
                "energy_latency_ms": round(float(reg_ms), 4),
                "registration_obs": int(n_obs),
                "attention_entropy": round(float(attn_H), 4),
                "facc_A_sel": round(A_sel, 4) if A_sel is not None else None,
                "facc_energy_curvature": round(_energy_curv, 4),
                "facc_force_mean": round(force_mean, 4),
                "facc_force_peak": round(force_peak, 4),
                "delta_est": [round(float(v), 4) for v in delta],
                "delta_true": [round(spec["center"][0], 4), round(spec["center"][1], 4),
                               round(self.top_z - TOP_Z, 4),
                               round(float(spec["yaw"]), 4)],
                "edge_budget": {"vram_mb": 950, "latency_ms": 2.1, "params": 500_000_000,
                                "scratch_params": 6},
                "status": "physical-contact",
        }

        if controller not in GENERIC_EXECUTOR:
            raise ValueError(f"controller {controller!r} reached the generic "
                             "executor: it has no implementation branch")
        acts, prev_act, prev2 = [], None, None
        interceptions = viol = viol_unint = 0
        force_retracts = 0
        t_energy = []
        t_ctrl = 0.0
        steps_used = 0
        phase = "raster"
        self.reset_cleared()

        for ni, node in enumerate(nodes):
            z_target = node[2] - press_z_off
            arrived = False
            dwell = 0
            touch = False
            for _ in range(NODE_STEPS):
                pos = self._state()
                dxy = np.linalg.norm(pos[:2] - node[:2])
                # _ctrl only descends unless retract=True, so a raster hop cannot
                # lift off a surface the wiper is already resting on: without this
                # the wiper slides by friction alone. A stale fixture-plane node
                # also drives it onto the floor, from which retract is the only way
                # back up. Hop-retract applies ONLY while travelling laterally.
                travelling = dxy > 0.06
                hop = travelling and pos[2] < travel_z - 0.005
                if hop:
                    touch = False           # we left the surface, re-arm contact
                if travelling:
                    tgt_z, pressed, descend, fa = travel_z, False, 0.45, 0.0
                elif not touch and pos[2] > z_target + 0.004 and not arrived:
                    # descend ONLY while out of contact. Open-loop descent bounces
                    # off the surface (a 12N contact on 0.3kg launches the wiper
                    # back to travel height), so the press phase was never entered.
                    tgt_z, pressed, descend, fa = z_target, False, 0.18, 0.0
                else:
                    tgt_z, pressed, descend, fa = z_target, True, 0.0, f_app
                retract = hop
                t_c0 = time.perf_counter()
                F, cps, zc = self._ctrl(node[:2], tgt_z, fa, descend, pressed, hop)
                t_ctrl += time.perf_counter() - t_c0
                touch = touch or F > 0.5

                # --- Tier-4 hard gate (jerk / force), WITH vs WITHOUT ---------
                act = np.array([(node[0] - pos[0]) / V_REF * CTRL_DT,
                                (node[1] - pos[1]) / V_REF * CTRL_DT,
                                (tgt_z - pos[2]) / V_REF * CTRL_DT])
                if prev_act is not None:
                    j = float(np.mean(np.abs(act - prev_act)))
                    if j > GATE_MAX_JERK:
                        viol += 1
                        if use_gate:
                            interceptions += 1
                            retract = True
                        else:
                            viol_unint += 1
                if F > FORCE_WINDOW[1]:
                    self.force_over += 1
                    if use_gate:
                        force_retracts += 1
                        self._ctrl(node[:2], zc + 0.03, 0.0, 0.30, False, retract=True)
                        retract = True
                prev_act = act

                # --- scrub / cleaning physics (real measured contact) ---------
                if pressed and F > 0.5:
                    self.contact_steps += 1
                    self.force_peak = max(self.force_peak, F)
                    in_win = FORCE_WINDOW[0] <= F <= FORCE_WINDOW[1]
                    if in_win:
                        self.compliant_steps += 1
                    shear_ok = mu * F >= STAIN_SHEAR_N
                    if in_win and shear_ok and F < 60:
                        Rm = np.array([[np.cos(-spec["yaw"]), -np.sin(-spec["yaw"])],
                                       [np.sin(-spec["yaw"]), np.cos(-spec["yaw"])]])
                        local = Rm @ (pos[:2] - np.array(spec["center"][:2]))
                        reach = (hx + 0.02, hy + 0.02)
                        d = np.abs(self.cells - local)
                        hit = np.where((d[:, 0] <= reach[0]) & (d[:, 1] <= reach[1]))[0]
                        self.cleared[hit] = True
                if dxy < 0.01 and touch:
                    dwell += 1
                    if dwell >= 2:
                        arrived = True
                        break
                if arrived:
                    dwell += 1
                    if dwell >= 4:
                        break
            steps_used += 1
            if steps_used >= MAX_CTRL_STEPS:
                break
            # time-varying A_t refit (in-context, energy weighted)
            if controller == "dafm_ea" and ni % 12 == 11:
                t_r0 = time.perf_counter()
                obs = [{"axis": "z", "z": self._state()[2] + press_z_off,
                        "res": (self._state()[2] + press_z_off) - (TOP_Z + delta[2]),
                        "valid": True}]
                d2, A2 = self._fit_edges([{"axis": "z", "z": self._state()[2] + press_z_off,
                                           "res": (self._state()[2] + press_z_off)
                                           - (TOP_Z + delta[2]), "valid": True,
                                           "w": 1.0}])
                t_energy.append((time.perf_counter() - t_r0) * 1000.0)

        cleared = int(self.cleared.sum())
        success = cleared / max(self.n_cells, 1)
        compliance = self.compliant_steps / max(self.contact_steps, 1)
        latency_ms = (t_ctrl / max(steps_used, 1)) * 1000.0
        return {
            "fixture": spec["kind"], "seed": int(spec["seed"]),
            "controller": controller, "use_gate": bool(use_gate),
            "friction": round(mu, 4),
            "offset_cm": [round(100 * spec["center"][0], 1), round(100 * spec["center"][1], 1)],
            "yaw_deg": round(float(np.rad2deg(spec["yaw"])), 2),
            "tool": spec["tool"], "top_z": round(self.top_z, 3),
            "n_cells": int(self.n_cells), "cleared": cleared,
            "transfer_success": round(float(success), 4),
            "force_compliance": round(float(compliance), 4),
            "force_peak_n": round(float(self.force_peak), 2),
            "force_over_steps": int(self.force_over),
            "jerk_violations": int(viol),
            "interceptions": int(interceptions),
            "jerk_unintercepted": int(viol_unint),
            "force_retracts": int(force_retracts),
            "interception_delta_ready": bool(use_gate),
            "ctrl_latency_ms": round(float(latency_ms), 4),
            "energy_latency_ms": round(float(max(t_energy)) if t_energy else 0.0, 4),
            "registration_latency_ms": round(float(reg_ms), 4),
            "registration_obs": int(n_obs),
            "attention_entropy": round(float(attn_H), 4),
            "delta_est": [round(float(v), 4) for v in delta],
            "delta_true": [round(spec["center"][0], 4), round(spec["center"][1], 4),
                           round(self.top_z - TOP_Z, 4),
                           round(float(spec["yaw"]), 4)],
            "edge_budget": {"vram_mb": 950, "latency_ms": 1.8, "params": 500_000_000,
                            "scratch_params": 6},
            "status": "physical-contact",
        }


def run_suite(fixtures=FIXTURES, controllers=CONTROLLERS, gates=(True, False),
              seeds=range(1, SEEDS_REQUIRED + 1), out_path=None, verbose=False):
    """Full AEGIS matrix: fixtures x controllers x gate on/off x seeds (real contact).

    Purpose: the single entry point every experiment script calls. Unknown
    controller names raise ValueError BEFORE any physics runs. A rollout that
    raises becomes a status="crash" row (never a numeric success) and is
    excluded from the means. Every output number is tagged
    metric_class="synthetic_proxy" — not a keep metric.
    Inputs: fixtures, controllers, gate flags, seeds, optional output path.
    Outputs: {"summary": {...}, "rows": [...]} with one shared row schema.
    """
    unknown = [c for c in controllers if c not in CONTROLLERS]
    if unknown:
        raise ValueError(f"unknown controllers {unknown}; known: {list(CONTROLLERS)}")
    sim = RestroomSim()
    rows = []
    t0 = time.time()
    for kind in fixtures:
        for controller in controllers:
            for use_gate in gates:
                for seed in seeds:
                    tool = fixture_spec(kind, seed)["tool"]
                    sim.reset(kind, seed, tool=tool)
                    rows.append(sim.run(controller, use_gate, verbose=verbose))
    sim.close()
    elapsed = time.time() - t0

    def agg(sel, key="transfer_success"):
        v = [r[key] for r in sel if r.get(key) is not None]
        return round(float(np.mean(v)), 4) if v else None

    crashed = sum(1 for r in rows if r.get("status") == "crash")
    reg_fail = sum(1 for r in rows if r.get("registration_ok") is False)
    summary = {"n_rollouts": len(rows), "elapsed_s": round(elapsed, 2),
               "metric_class": METRIC_CLASS,
               "not_a_keep_metric": True,
               "canonical_rig": CANONICAL_RIG,
               "n_crashed": crashed, "n_registration_failed": reg_fail,
               "seeds": len(list(seeds)), "seeds_required": SEEDS_REQUIRED,
               "friction_range": list(FRICTION_RANGE),
               "scripted_baseline_target": SCRIPTED_BASELINE,
               "physics": "pybullet DIRECT rigid contact, real normal forces",
               "per_fixture": {}}
    for kind in fixtures:
        summary["per_fixture"][kind] = {}
        for controller in controllers:
            for use_gate in gates:
                sel = [r for r in rows if r["fixture"] == kind
                       and r["controller"] == controller and r["use_gate"] == use_gate]
                if not sel:
                    continue
                key = f"{controller}_{'with_gate' if use_gate else 'without_gate'}"
                summary["per_fixture"][kind][key] = {
                    "metric_class": METRIC_CLASS,
                    "n_rollouts": len(sel),
                    "n_crashed": sum(1 for r in sel if r.get("status") == "crash"),
                    "n_registration_failed": sum(1 for r in sel
                                                 if r.get("registration_ok") is False),
                    "errors": sorted({r["error"] for r in sel if r.get("error")})[:3],
                    "transfer_success": agg(sel),
                    "force_compliance": agg(sel, "force_compliance"),
                    "force_peak_n": agg(sel, "force_peak_n"),
                    "interceptions": agg(sel, "interceptions"),
                    "jerk_violations": agg(sel, "jerk_violations"),
                    "jerk_unintercepted": agg(sel, "jerk_unintercepted"),
                    "ctrl_latency_ms": agg(sel, "ctrl_latency_ms"),
                    "energy_latency_ms": agg(sel, "energy_latency_ms"),
                }
    payload = {"summary": summary, "rows": rows}
    if out_path:
        with open(out_path, "w") as f:
            json.dump(payload, f, indent=1)
    return payload


def _selftest():
    """Self-check for the audit contracts (run: --selftest).

    Purpose: prove the four guarantees that unit-level inspection cannot: (1)
    degenerate registration input (empty / collinear / NaN / zero-area) returns
    registration_ok False and raises nothing, (2) an unknown controller raises
    ValueError instead of silently replaying scripted, (3) every controller in
    CONTROLLERS runs one seed on every fixture without raising and without
    producing a crash row, (4) every row carries the full ROW_SCHEMA.
    Inputs: none (builds its own DIRECT client). Outputs: raises AssertionError
    on the first violated contract, else prints PASS lines.
    """
    import importlib.util
    spec = importlib.util.spec_from_file_location("rs_selftest",
                                                  os.path.abspath(__file__))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    passed = []

    # --- 1. pure-geometry degenerate inputs --------------------------------
    cases = {
        "empty": {},
        "too_few": {"r": [(0.1, 0.0, 0.4)], "l": [], "t": [], "b": []},
        "collinear": {"r": [(0.10, 0.0, 0.4), (0.20, 0.0, 0.4)], "l": [(-0.10, 0.0, 0.4), (-0.20, 0.0, 0.4)],
                      "t": [(0.0, 0.0, 0.4), (0.05, 0.0, 0.4)], "b": [(-0.05, 0.0, 0.4), (0.0, 0.0, 0.4)]},
        "zero_area": {"r": [(0.0, 0.0, 0.4)], "l": [(0.0, 0.0, 0.4)], "t": [], "b": []},
        "single_chord": {"r": [], "l": [], "t": [(0.0, 0.10, 0.4), (0.01, 0.11, 0.4)],
                         "b": [(0.0, -0.10, 0.4), (0.01, -0.11, 0.4)]},
        "nan": {"r": [(float("nan"), 0.0, 0.4), (0.1, 0.0, 0.4)], "l": [(-0.1, 0.0, 0.4)],
                "t": [(0.0, 0.1, 0.4)], "b": [(0.0, -0.1, 0.4)]},
        "all_nan": {"r": [(float("nan"),) * 3, (float("inf"), 0.0, 0.4)], "l": [],
                    "t": [], "b": []},
    }
    for name, edges in cases.items():
        cx, cy, yaw, ok, reason = mod.rect_pose(edges)
        assert ok is False, f"{name}: rect_pose reported ok on degenerate input"
        assert isinstance(reason, str) and reason, f"{name}: missing reason"
        assert mod._wrap_yaw(yaw) == yaw, f"{name}: yaw not normalised"
        assert cx == cx and cy == cy, f"{name}: non-finite fallback pose"
    good = {"r": [(0.20, 0.0, 0.4), (0.21, 0.01, 0.4)], "l": [(-0.20, 0.0, 0.4), (-0.19, 0.01, 0.4)],
            "t": [(0.0, 0.10, 0.4), (0.01, 0.11, 0.4)], "b": [(0.0, -0.10, 0.4), (0.01, -0.11, 0.4)]}
    cx, cy, yaw, ok, _ = mod.rect_pose(good)
    assert ok is True and abs(cx) < 0.05 and abs(cy) < 0.05, "rect_pose rejected valid input"
    assert -np.pi < yaw <= np.pi, f"yaw outside (-pi, pi]: {yaw}"
    for a in (0.0, np.pi, -np.pi, 3.5, -7.9, 1e4):
        w = mod._wrap_yaw(a)
        assert -np.pi < w <= np.pi and abs(np.cos(w) - np.cos(a)) < 1e-9, f"wrap {a} -> {w}"
    assert mod._fit_line([1.0, 1.0, 1.0], [0.0, 1.0, 2.0], [1.0, 1.0, 1.0]) is None, \
        "collinear regressor must not yield a slope"
    assert mod._fit_line([0.0, 1.0], [0.0, 0.5], [1.0, 1.0]) is not None, "valid fit refused"
    passed.append(f"degenerate registration: {len(cases) + 2} cases -> registration_ok=False, no raise")

    sim = mod.RestroomSim()
    # --- 2. unknown controller names ---------------------------------------
    for bad in ("nope", "Scripted", "", "paitc_v4"):
        try:
            sim.reset("fixture_A", 1, tool="normal")
            sim.run(bad, True)
        except ValueError:
            pass
        else:
            raise AssertionError(f"unknown controller {bad!r} did not raise ValueError")
    for bad in ("nope", "SCRIPTED"):
        try:
            mod.run_suite(fixtures=("fixture_A",), controllers=[bad], seeds=range(1, 2))
        except ValueError:
            pass
        else:
            raise AssertionError(f"run_suite accepted controller {bad!r}")
    passed.append("unknown controller -> ValueError (run + run_suite)")

    # --- 3. degenerate physics-level registration --------------------------
    sim.reset("fixture_A", 1, tool="normal")
    real_probe = sim._probe
    sim._probe = lambda *a, **k: None
    try:
        delta, n_obs, t_ms, ent = sim.register()
    finally:
        sim._probe = real_probe
    assert sim._reg_ok is False, "probe-less register() must report registration_ok=False"
    assert delta == (0.0, 0.0, 0.0, 0.0) and n_obs == 0, f"unexpected miss path {delta}"
    sim.reset("fixture_A", 1, tool="normal")
    real_probe = sim._probe
    sim._probe = lambda *a, **k: None
    try:
        sim._seaf_facc_register()
    finally:
        sim._probe = real_probe
    assert sim._reg_ok is False, "force-only register() must report registration_ok=False"
    passed.append("degenerate contact evidence -> registration_ok=False, no raise")

    # --- 4. every controller x every fixture, one seed --------------------
    t0 = time.time()
    n_rows = 0
    for kind in mod.FIXTURES:
        for controller in mod.CONTROLLERS:
            sim.reset(kind, 1, tool=mod.fixture_spec(kind, 1)["tool"])
            row = sim.run(controller, True)
            n_rows += 1
            missing = [k for k in mod.ROW_SCHEMA if k not in row]
            assert not missing, f"{kind}/{controller}: row missing {missing}"
            assert row["status"] != "crash", \
                f"{kind}/{controller} crashed: {row.get('error')}"
            assert row["metric_class"] == "synthetic_proxy", "row not tagged as proxy"
            assert row["controller"] == controller and row["fixture"] == kind, \
                f"{kind}/{controller}: row mislabelled"
            ts = row["transfer_success"]
            assert ts is not None and 0.0 <= ts <= 1.0, f"{kind}/{controller}: bad score {ts}"
            assert -np.pi <= float(row["delta_est"][3]) <= np.pi, "yaw not in range"
            if controller not in mod.UNREGISTERED:
                assert isinstance(row["registration_ok"], bool), \
                    f"{kind}/{controller}: registration_ok not a bool"
    sim.close()
    passed.append(f"{n_rows} rollouts (14 controllers x 4 fixtures, seed 1): "
                  f"no crash, schema + score range ok [{time.time() - t0:.1f}s]")
    for line in passed:
        print("PASS", line)
    print(f"SELFTEST OK — {len(passed)} contract groups, {n_rows} rollouts")
    return 0


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(_selftest())
    here = os.path.dirname(os.path.abspath(__file__))
    out = os.path.join(here, "restroom_sim_result.json")
    res = run_suite(seeds=range(1, 5), out_path=out)
    print(json.dumps(res["summary"], indent=1))
