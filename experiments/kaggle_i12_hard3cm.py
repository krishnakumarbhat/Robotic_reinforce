from __future__ import annotations
import os
os.environ.setdefault("AEGIS_SEEDS", "200")
os.environ.setdefault("AEGIS_POSE_FIX_U", "0.03")
os.environ.setdefault("AEGIS_POSE_NOISE", "0,0")
os.environ.setdefault("AEGIS_UPLOAD", "0")
os.environ.setdefault("AEGIS_INSTALL", "1")
"""AEGIS restroom fixture sweep on Kaggle (prep-only; this file is NOT pushed).

Runs the Fixture-A (round / tank / glossy) and Fixture-B (elongated / wall-hung /
matte / +15cm / +10deg) suites against a scripted cleaning controller with REAL rigid
contact physics, sweeping surface friction uniformly over 0.05 (wet soap) .. 0.80
(dry porcelain). Every episode streams one JSONL line to /kaggle/working/aegis_sweep.jsonl
carrying the AEGIS context-conditioning payload {tool_id, se3_offset_xyz,
compliance_mode, quality_tag} plus the Tier-4 jerk gate (max_jerk > 0.618).

Run on Kaggle (zero stdin, zero argv):
    kaggle kernels push -p <kernel_dir>      # kernel-metadata.json points at this file
    AEGIS_SEEDS=20 AEGIS_HOURS=7.0           # optional env overrides
Run locally (physics only, no maniskill needed):
    AEGIS_BACKEND=auto python3 experiments/kaggle_aegis_sweep.py --seeds 3 --dry-run-no-upload

Physics backend: a raw PyBullet scrub rig (CPU). Chosen over ManiSkill deliberately --
maniskill 3.0.1 is installed and probed, but its PickCube-v1 actor exposes NO friction
API (no Actor.set_friction) and there is no public handle to the underlying mjModel, so
the swept coefficient cannot reach the contact pair. Emitting episodes that LOG
friction=0.35 while simulating the 0.5 default would poison the dataset, and friction is
the entire axis of this sweep. The pip header still installs maniskill/mujoco and the
header record reports availability. ~0.05 s/episode on CPU, so a 40-episode sweep needs
seconds of a T4 -- the GPU is not the bottleneck and is not rented for it. If pybullet
cannot be imported the script exits 3 rather than emitting synthetic numbers: the AEGIS
protocol forbids synthetic-only rows from being reported as validation.

The scripted controller is a fixed force-driven position trajectory (spray -> raster
scrub -> rinse -> inspect DAG), so the ONLY thing the physics decides is whether the
tool head follows it: tracks = SUCCESS, skates/sticks = FAILURE_SLIP, loses contact =
OBSTACLE_STALL.

MEASURED RESULT, worth reading before trusting the labels: swept-area coverage is
MONOTONE DECREASING in friction (Fixture A 0.92 at mu=0.05 -> 0.68 at mu=0.80;
Fixture B 0.67 -> 0.56). The intuitive "low friction = slip = fail" does NOT hold here.
The raster reverses direction at every row end, and breaking away from static contact
costs a full mu*N stiction impulse the PD cannot deliver inside one control tick, so
HIGH friction loses cells at the reversals while LOW friction tracks almost perfectly.
That is textbook stick-slip, it is reproducible across all three tool heads, and it is
reported as measured rather than tuned into the expected direction. Fixture B never
reaches the 0.70 keep-bar in this configuration, so the summary verdict is DISCARD.
"""


import argparse
import bisect
import json
import math
import os
import random
import subprocess
import sys
import time

# --- headless / determinism (must precede any heavy import) --------------------
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTHONUNBUFFERED", "1")

# --- config (env-overridable; zero-stdin safe defaults) ------------------------
WORKDIR = os.environ.get("AEGIS_WORKDIR", "/kaggle/working")
if not os.access(WORKDIR, os.W_OK):  # not a Kaggle VM (local dry-run) -> stay usable
    WORKDIR = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "results")
    print(f"[aegis-sweep] /kaggle/working unavailable -> WORKDIR={WORKDIR}", flush=True)
JSONL_PATH = os.path.join(WORKDIR, "aegis_sweep.jsonl")
FIG_PATH = os.path.join(WORKDIR, "aegis_sweep.png")
HF_REPO = os.environ.get("AEGIS_HF_REPO", "krishnah27/aegis-sweep")
N_SEEDS = int(os.environ.get("AEGIS_SEEDS", "20"))
BUDGET_H = float(os.environ.get("AEGIS_HOURS", "7.0"))
HARD_CAP_H = float(os.environ.get("AEGIS_HARD_CAP_HOURS", "8.0"))
UPLOAD = os.environ.get("AEGIS_UPLOAD", "1") not in ("0", "false", "False")
FORCE_BACKEND = os.environ.get("AEGIS_BACKEND", "pybullet")  # pybullet
INSTALL_DEPS = os.environ.get("AEGIS_INSTALL", "1") not in ("0", "false", "False")
BENCH_DIR = os.environ.get("AEGIS_BENCH_DIR", "")  # optional: repo benchmarks/ dir
BASE_SEED = int(os.environ.get("AEGIS_BASE_SEED", "91000"))
T_MAX = int(os.environ.get("AEGIS_STEPS", "400"))
PROBE_EPS = 3  # episodes timed before the quota projection is trusted

# --- AEGIS spec bindings, mirrored from benchmarks/restroom_sim.py -------------
try:  # reuse the benchmark constants when the repo is on the path (no drift)
    if BENCH_DIR:
        sys.path.insert(0, os.path.dirname(BENCH_DIR.rstrip("/")))
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "benchmarks"))
    from restroom_sim import FRICTION_RANGE, GATE_MAX_JERK  # type: ignore
    from restroom_sim import FIXTURES as _FIXTURES  # type: ignore
    FIXTURES = {k: dict(v) for k, v in _FIXTURES.items()}
except Exception:  # noqa: BLE001 -- Kaggle has no repo checkout; keep the mirror honest
    FIXTURES = {
        "fixture_A": {"tank_shape": "round", "surface": "glossy",
                      "offset_cm": 0, "angle_deg": 0},
        "fixture_B": {"tank_shape": "elongated", "surface": "matte",
                      "offset_cm": 15, "angle_deg": 10},
    }
    FRICTION_RANGE = [0.05, 0.80]
    GATE_MAX_JERK = 0.618

# glossy porcelain sheds water -> less grip; matte holds the tool head -> more grip.
SURFACE_FRICTION_GAIN = {"glossy": 0.75, "matte": 1.25}
SCRIPTED_BASELINE_SCORE = 0.8125  # from restroom_sim; reported, never used as a row
WALL_HUNG_LIFT_M = 0.15           # Fixture B tank hangs this high off its pedestal
CELL_M = 0.05                     # swept-area cell pitch == raster pass pitch
# I8 curved bowl (realism): concave paraboloid liner on the fixture top face.
SURFACE = os.environ.get("AEGIS_SURFACE", "flat")   # flat | bowl
BOWL_K = float(os.environ.get("AEGIS_BOWL_K", "0.03"))  # bowl depth scale (m)
# I13 quasi-static speed scheduling (I8 unblock 1). Commanded speed v(s) = v0/(1+alpha*kappa(s)),
# kappa = |d theta| / ds of the HORIZONTAL heading of the commanded polyline (1/m). Realised as a
# TIME RE-PARAMETERISATION of the tick -> arclength map: more ticks where kappa is high, fewer
# where it is low, TOTAL TICK BUDGET UNCHANGED (a slow curve must be paid for somewhere).
# No force gain, solver or scoring term is touched; coverage_cont/success still come only from
# physics contacts. Default 0.0 = OFF, which is the frozen rig's own uniform map -- the I13 spec
# default (8.0) is deliberately NOT the module default: that would silently re-time every other
# idea's episode. Pass it explicitly, or pair an arm against alpha=0 with --compare-env.
SPEED_ALPHA = float(os.environ.get("AEGIS_SPEED_ALPHA", "0.0"))
# I15 fn setpoint press regulation (I8 unblock 3 of 3). A PI loop on the MEASURED contact
# normal force replaces the constant press KP_PRESS*kp*PRESS_M during the scrub phase. This is
# a CONTROLLER, not an admission gate: it never decides success/coverage, it only sets a force.
# Default 0 = OFF, i.e. the frozen rig's own constant press term is evaluated unchanged.
# AEGIS_FN_SET is the rig-unit setpoint: 0.5 N is the flat-face operating point (measured
# fn_mean 0.4966) -- rig units, NOT the 10-25 N AEGIS spec band (see ideas.md rig limits).
FORCE_PI = int(os.environ.get("AEGIS_FORCE_PI", "0"))
FN_SET_N = float(os.environ.get("AEGIS_FN_SET", "0.5"))
FN_KP = float(os.environ.get("AEGIS_FN_KP", "0.0"))
FN_KI = float(os.environ.get("AEGIS_FN_KI", "0.0"))
PRESS_MAX_N = float(os.environ.get("AEGIS_PRESS_MAX_N", "1.2"))   # rail on the commanded press
# I14 patch-inset containment (I8 unblock 3 of 3). The commanded scrub path is clamped into
# an inset of the patch rect, and a SOFT WALL -- a PD pull toward that rect, applied as a
# FORCE inside the physics loop, NEVER a teleport (G7) -- keeps the head inside it. Default
# 0.0 = OFF, so every other idea's episode is the frozen rig, bit-identical. AEGIS_WALL_ONLY=1
# keeps the wall and drops the command clamp, which separates the two halves: the clamp
# shrinks the swept band (it costs coverage by construction), the wall only contains.
INSET_M = float(os.environ.get("AEGIS_INSET_M", "0.0"))
WALL_ONLY = int(os.environ.get("AEGIS_WALL_ONLY", "0"))
WALL_KP = float(os.environ.get("AEGIS_WALL_KP", "25.0"))  # N/m; 25 == KP, the servo gain
# I3 residual policy. The commanded scrub target is offset by a 2-D learned residual in the
# FIXTURE frame, clipped to RESIDUAL_CLIP_M, for the duration of the scrub phase. The offset
# goes into the SERVO COMMAND, never into the body state: no teleport, and success/coverage
# are still computed from physics contacts at return time (G7). AEGIS_RESIDUAL_ACTIVE defaults
# to 0 = OFF, which leaves the original `tgt` expression byte-for-byte unchanged, and
# AEGIS_RESIDUAL_ACTIVE is a compare knob so an ON arm can be paired against an OFF arm on the
# SAME seeds (G4). The policy itself is a callable object injected by the trainer
# (experiments/run_i3_ppo.py) through RESIDUAL_HOOK_FN; the rig never learns anything itself.
RESIDUAL_ACTIVE = int(os.environ.get("AEGIS_RESIDUAL_ACTIVE", "0"))
RESIDUAL_CLIP_M = float(os.environ.get("AEGIS_RESIDUAL_CLIP_M", "0.02"))
RESIDUAL_HOOK_FN = None          # set by the trainer: () -> hook object with .trace
LAUNCH_Z_M = 0.02              # I8/I13 diagnostic: scrub-phase z rise above the chased point
                                # that counts as a ballistic launch, not a surface follow
SWEEP_TOL_M = 0.035               # "in contact / on path" radius
# Anchored to the tool's own geometry, not to the chart: the head footprint half-width is
# 0.025-0.045 m, so a 0.012 m mean tracking residual means the head is no longer seated
# on its commanded cell (27-48% of its own support radius). Measured residuals split
# cleanly either side of it -- healthy 0.004-0.010, stiction tail 0.012-0.016 -- so this
# is the natural gap, not a threshold chosen to colour the plot.
SLIP_TOL_M = 0.012                # > this tracking residual = FAILURE_SLIP / stall
STALL_FRAC = 0.40                 # > this contact-free fraction = OBSTACLE_STALL
FINE_M = CELL_M / 2.0             # I0: fine scoring pitch (coverage_cont); raster pitch unchanged
# N211: the MEASUREMENT axis. Every run 1-359 scored the SAME `coverage_cont` from three frozen
# literals inside `_coverage_cont` -- the grid pitch (FINE_M = CELL_M/2 = 0.025 m), the footprint
# model (an ISOTROPIC DISC of radius r_eff = min(hu,hv) although every pad is a RECTANGLE of
# half-extents hu x hv), and the contact stride (`pts[::2] if len(pts) > 400`, i.e. HALF the
# measured contact set is discarded on exactly the longest, highest-coverage episodes). The
# segment's entire evidence base is `coverage_cont >= 0.90` on that kernel, so the kernel is a
# factor that was never probed -- the last frozen term in the whole measurement chain after
# N207 (rate), N208 (iterations), N209 (friction combine) and N210 (normal force).
# 0 = off (the default; no extra field, records byte-identical to runs 1-359). 1 = also
# RE-MEASURE the same physics contacts under alternative kernels, logged as diagnostics in
# `cov_k` / `succ_k`. The SCORED coverage_cont / success are never touched by this flag: the
# frozen expression in `_coverage_cont` is left verbatim and the audit's only claim is about
# how much the verdict depends on the kernel (G7: no arithmetic on the reported metric).
COV_KERNEL = float(os.environ.get("AEGIS_COV_KERNEL", "0"))
# I1 path modes. raster = legacy boustrophedon (180 deg reversals -> v_t crosses 0 ->
# static-friction re-stick). rounded/trochoid/spiral are C1 (max heading change per
# dense segment <= MAX_TURN_DEG) so tangential velocity never reverses.
PATH_MODES = ("raster", "rounded", "trochoid", "spiral", "fitted", "fitro", "orbit", "auto")
PATH_MODE = os.environ.get("AEGIS_PATH", "raster")
MAX_TURN_DEG = 60.0               # C1 assertion: per-segment heading change < 60 deg keeps
                                  # >= cos(60)=50% of velocity along the old heading (raster: 90+90)
TROCHOID_R_M = float(os.environ.get("AEGIS_TROCH_R_M", "0.015"))
                                   # circular-scrub loop radius superimposed on rows
                                   # (Run 28: scaling-invariance sweep; default 0.015 is
                                   # the champion value, byte-identical path)
# d/R of the superimposed trochoid: w = s*d/R / R, so the tool offset circle of radius
# R rotates at rate 1/d while the path arclength runs at 1 -> |dT/ds| = |1 - d/R|.
# d/R=1 is the exact cusp (cycloid, speed 0 -> re-stick), d/R<1 loops, d/R>1 curtate.
# 0.5 is the CHAMPION I1 value (never cusps, min speed 0.5). Run 26 measured the rest.
TROCHOID_DR = float(os.environ.get("AEGIS_TROCH_DR", "0.5"))
# Resample step of the trochoid path. Run 28 sweeps it against R to separate a MESH
# artefact of the C1 turn from intrinsic path curvature. Default 0.004 = champion.
TROCHOID_DS_M = float(os.environ.get("AEGIS_TROCH_DS_M", "0.004"))
# Base-row chord of the rounded/trochoid/fitted paths. This is the REAL mesh of the
# superimposed trochoid: the offset circle is sampled once per base point, so its angular
# step is ds_base*(d/R)/R. Run 28 sweeps it to test mesh convergence. Default 0.01 = champion.
BASE_DS_M = float(os.environ.get("AEGIS_BASE_DS_M", "0.01"))
# R29: Run 26-28 called TROCHOID_DR a "shape parameter d/R", but the generator ties the
# offset AMPLITUDE to R and the offset RATE to DR/R, so d/R is really the DIMENSIONLESS
# OFFSET SPEED w = (DR/R)*R = DR and every "scale" sweep moved amplitude and rate together.
# These two knobs decouple them; <= 0 means "use the coupled legacy value", which keeps the
# champion path bit-identical (the legacy expression is evaluated, not re-derived).
# General law: min|T| = 1 - w*A, so the cusp is w*A = 1 for ANY (A, w) -- not d/R = 1.
#
# N217: the (A, w) MANIFOLD of the champion's own loop, the last un-audited frozen family.
# Runs 26-29 introduced these two knobs precisely because every earlier sweep moved amplitude
# and rate TOGETHER, and then never swept them: every run 1-364 evaluated the champion at the
# single point A = 0.015 m, w = 33.3333 rad/m, i.e. the dimensionless product w*A = 0.500000,
# exactly HALF the cusp. N214 audited the pad, N215 the face, N216 the patch; the PLAN's own
# shape has never been a controlled dose on the rig-v2 measurement chain, so the champion's
# shape is certified at a point and nowhere else.
# The manifold has two independent coordinates and one candidate invariant:
#   lambda = w*A  (dimensionless; min|T| = 1 - lambda, cusp at 1, champion 0.5)
#   A      (metres; the plan's u-excursion is +2A, which N216 measured as the FACE-margin term)
# If lambda alone sets the outcome, the shape is scale-free in A and the champion sits on a
# plateau whose width is measured here. If A enters separately, the two knobs are NOT
# interchangeable and the champion's A is an independent, unclaimed design choice.
# PRE-REGISTERED (written into this source before the first N217 run; equations.md ROW N217):
#   Y1 the cusp is a HARD floor in lambda and the champion has a wide plateau: fixture_B
#      trochoid stays 1.0000/20-of-20 for every lambda <= 0.80 and the first loss appears in
#      (0.80, 1.00]. Refuted if any lambda <= 0.80 loses coverage on any suite, or if the arm is
#      already degraded at 0.80.
#   Y2 INVARIANCE: at FIXED lambda = 0.5, three (A, w) pairs spanning a 4x range in A
#      ((0.005, 100), (0.010, 50), (0.015, 33.3333)) give coverage_cont within one bar quantum
#      (1/(nu*nv*4) = 0.015625 elongated) of each other on all three suites -- the loop is
#      scale-free in A at fixed lambda. Refuted if any pair differs by more than one quantum.
#   Y3 the u-excursion term is REAL and is the one that makes A non-free: doubling A at fixed
#      lambda (A 0.015 -> 0.030, w 33.3333 -> 16.6667) pushes the plan reach +2A from +0.0298 m
#      to +0.0596 m, which N216 showed is the FACE-margin term, so on the round faces (whose
#      binder is the plan CORNER, N216) this pair must lose coverage while the elongated anchor
#      -- whose face half is 0.34 m and whose margin is 1.52x -- need not. This is the SPLIT
#      test: a shape term that is invisible on B and visible on A/R is a face-geometry effect,
#      not a contact effect, and it is exactly where a re-anchored bar would gain dynamic range.
#   Y4 CYCLE TIME is a real cost of Y2: at fixed lambda the path arclength grows with A, so a
#      4x-A reduction is a path-length reduction and the achievable v_cmd at a fixed tick budget
#      rises. If coverage is invariant (Y2) then the SHORTEST path that certifies is the better
#      product and that is a claim the segment can make. Refuted if coverage moves with A.
# Every arm is paired in-rig against the frozen champion on the SAME 20 seeds via
# --compare-env AEGIS_TROCH_AMP_M / AEGIS_TROCH_W; the identity arm (no --compare-env) must be
# bit-identical to run 350. The scored pair stays a pure function of the physics contacts
# (`_coverage_cont` VERBATIM): only the PLAN moves. No force, gate, kernel or window byte moves.
TROCHOID_AMP_ENV = float(os.environ.get("AEGIS_TROCH_AMP_M", "0"))
TROCHOID_W_ENV = float(os.environ.get("AEGIS_TROCH_W", "0"))
# *_LEGACY stays True until a knob is set, so the untouched champion evaluates the ORIGINAL
# expressions `s*DR/R` and `amp = R` (bit-identical, not a re-derived equivalent).
TROCHOID_AMP_LEGACY = TROCHOID_AMP_ENV <= 0.0
TROCHOID_W_LEGACY = TROCHOID_W_ENV <= 0.0
TROCHOID_AMP_M = TROCHOID_AMP_ENV if not TROCHOID_AMP_LEGACY else TROCHOID_R_M
TROCHOID_W = TROCHOID_W_ENV if not TROCHOID_W_LEGACY else TROCHOID_DR / TROCHOID_R_M
# I17: alternate the loop winding per row. Every row currently winds the same way, so the
# per-row residual of a non-integer number of turns (rate*L = 2.12 turns at the champion)
# shares a sign and biases the path laterally. <= 0 keeps the champion expression verbatim.
TROCH_ALT = float(os.environ.get("AEGIS_TROCH_ALT", "0"))
# I22: place the scrub rows at the CENTRES of the coarse cells they are meant to cover.
# `rows = [v0 + r*CELL_M]` from v0 = -side/2 puts every row on a cell's LOWER EDGE, so the
# band sits exactly CELL_M/2 low in v for every patch size and nv, and the top v row of the
# patch is covered only by the loop amplitude's r_eff dilation. Adding 0.5*CELL_M is the
# one-term fix and is a PLATEAU, not a tuned offset (measured +-0.010 on both suites, all
# three pads). 0 = the pre-I22 expression verbatim, so `--compare-env AEGIS_ROW_CENTRE=0`
# reproduces the frozen champion. `spiral` derives its own centre from v0 and is untouched.
ROW_CENTRE = float(os.environ.get("AEGIS_ROW_CENTRE", "1"))
# N206: the OVER-COVERAGE MARGIN. `scrub_grid` re-uses the same half/side constants as
# `scrub_uv`, so the scored patch IS the planned rect: a planning-pose offset e leaves a bare
# strip of width |e| on one side of the band, and N206 measured coverage_cont as a function of
# that offset ALONE (pearson -0.88, and only 0.017 of spread across noise levels once the
# offset is held in a 10 mm band). The margin over-sweeps the band by MARGIN_M on every side,
# which contains the patch for |e| <= MARGIN_M; the in-patch rows keep their phase (re-anchoring
# them to the inflated side would shift the row grid and cost fine coverage). It is PLAN
# GEOMETRY -- it moves no force, admits nothing and vetoes nothing; every contact/coverage
# number still comes from the physics loop. Default 0.0 = OFF = the frozen rig byte-for-byte,
# so `--compare-env AEGIS_MARGIN_M=0.0` pairs it against the champion on the SAME seeds (G4).
# `fitted`/`fitro` build their own row plan and ignore it.
MARGIN_M = float(os.environ.get("AEGIS_MARGIN_M", "0.0"))
# I12: Tier-2 depth registration as a PAIRED KNOB (R299). It is a PLANNER-side estimator --
# it rewrites the scrub plan from a 32x32 ray grid, exactly like pose noise rewrites it, and
# every contact/coverage number still comes from the physics loop. Default "" = OFF, which is
# the pre-I12 rig byte-for-byte; the two os.environ reads below became this global so that
# `--compare-env AEGIS_REG=1` can pair registration ON against OFF on the SAME seeds (G4).
REG_MODE = os.environ.get("AEGIS_REG", "")
# N190: the I9 registration ESTIMATOR is the pose-noise floor, not its ray-grid resolution
# (measured: 32 -> 64 -> 128 -> 181 rays leave reg_err_xy p90 at 14.6/18.4/15.8/15.0 mm on
# fixture_B, while the +-0.35 m window TRUNCATES the top face once the planning pose is off
# and the mean-of-hits bias grows to 118.8 mm p90 at 16 cm / 32 deg). Two planner-side knobs:
#   REG_HALF_M  window half-extent; 0.35 = the frozen I9 value (byte-identical default)
#   REG_EST     "mean" (frozen) | "extent" (prior-yaw-frame extent midpoint, N190)
# No force, solver or scoring term is touched; every contact/coverage number still comes from
# the physics loop, and the tool is never teleported. All default to the frozen expression, so
# every other idea's episode is unchanged, and `--compare-env` pairs them on the SAME seeds.
REG_HALF_M = float(os.environ.get("AEGIS_REG_HALF_M", "0.35"))
REG_N = int(float(os.environ.get("AEGIS_REG_N", "32")))
REG_EST = os.environ.get("AEGIS_REG_EST", "mean")
if REG_EST not in ("mean", "extent"):
    raise SystemExit(f"AEGIS_REG_EST must be mean|extent, got {REG_EST!r}")
# N192: the pose-noise floor is the ray window's CONTAINMENT of the top face (N190.3), and no
# single width fixes it (H=0.9 loses to H=0.6, N190.6) -- so the SENSOR changes instead: a
# k x k lattice of +-H windows whose union covers the plausible-centre set, of which the ONE
# cast that best contains the face is kept. 1 = the frozen single cast (byte-identical);
# 0 = derive k from AEGIS_POSE_NOISE (k=1 whenever sigma <= H - rho_inf, i.e. the dose is
# inert below the containment slack). No force, solver or scoring term is touched.
REG_CASTS = float(os.environ.get("AEGIS_REG_CASTS", "1"))
# N195: the lattice PITCH is a free parameter that the shipped N192 value never varied. The
# guarantee is only that SOME lattice offset lands in the containment set
# {o : max(|f_x-o_x|,|f_y-o_y|) <= a_slack}, a square of half-width a_slack, so pitch d needs
# d/2 <= a_slack (d <= 2a) -- the shipped d = 1.5a (d/2 = 0.75a) is CONSERVATIVE and pays for
# it in k^2 rays. A coarser pitch buys a smaller k at equal coverage; the rays it saves can be
# spent on n (the ray pitch 2H/(n-1) IS the parity floor of the kept cast). 1.5 = the shipped
# N192 expression verbatim, so the default arm is byte-identical; --compare-env pairs it.
REG_DFACT = float(os.environ.get("AEGIS_REG_DFACT", "1.5"))
# N195: the k^2 casts exist only to answer "which offset contains the face", but each was cast
# at the FULL n x n resolution, so the sensor pays k^2 * n^2 rays for a k^2-way argmax and then
# throws k^2-1 of them away. The error that survives is NOT the selection -- it is the parity
# term of the ONE kept cast, 2H/(n-1) = 38.71 mm pitch (N194), i.e. resolution, not search.
# So: run the SELECTION stage on a coarse n_c x n_c grid, then cast the winner once at full n.
# REG_CN = coarse selection resolution; 0 = OFF (the shipped N192 expression verbatim, so the
# default arm is byte-identical). n_c >= n is a no-op by construction, so it is clamped.
REG_CN = float(os.environ.get("AEGIS_REG_CN", "0"))
# R29: env knob -> module global, so --compare-env can re-point the paired baseline arm
# without a second process. Only geometry knobs; force/solver/scoring are untouchable.
KNOB_GLOBALS = {"AEGIS_TROCH_R_M": "TROCHOID_R_M", "AEGIS_TROCH_DS_M": "TROCHOID_DS_M",
                "AEGIS_TROCH_DR": "TROCHOID_DR", "AEGIS_BASE_DS_M": "BASE_DS_M",
                "AEGIS_TROCH_AMP_M": "TROCHOID_AMP_M", "AEGIS_TROCH_W": "TROCHOID_W",
                "AEGIS_TROCH_ALT": "TROCH_ALT", "AEGIS_ROW_CENTRE": "ROW_CENTRE",
                "AEGIS_MARGIN_M": "MARGIN_M",
                "AEGIS_SPEED_ALPHA": "SPEED_ALPHA", "AEGIS_FORCE_PI": "FORCE_PI",
                "AEGIS_FN_SET": "FN_SET_N", "AEGIS_FN_KP": "FN_KP", "AEGIS_FN_KI": "FN_KI",
                "AEGIS_INSET_M": "INSET_M", "AEGIS_WALL_ONLY": "WALL_ONLY",
                "AEGIS_WALL_KP": "WALL_KP",                 "AEGIS_RESIDUAL_ACTIVE": "RESIDUAL_ACTIVE",
                 "AEGIS_RESIDUAL_CLIP_M": "RESIDUAL_CLIP_M",
                 "AEGIS_GATE_ADVISORY": "GATE_ADVISORY",
                 "AEGIS_GATE_PROP": "GATE_PROP", "AEGIS_GATE_TRIG": "GATE_TRIG",
                 "AEGIS_GATE_THETA": "GATE_THETA", "AEGIS_GATE_PROP_MIN": "GATE_PROP_MIN",
                 "AEGIS_REG_HALF_M": "REG_HALF_M", "AEGIS_REG_N": "REG_N",
                 "AEGIS_REG_CASTS": "REG_CASTS", "AEGIS_REG_DFACT": "REG_DFACT",
                 "AEGIS_REG_CN": "REG_CN",
                 # N207: the SOLVER rate is the one axis every run 1-355 held fixed. It is
                 # registered here (not just read from env) purely so `--compare-env
                 # AEGIS_SIM_HZ=240` can pair an arbitrary rate against the frozen champion
                 # on the SAME seeds. No force, scoring or geometry term is touched by it.
                 "AEGIS_SIM_HZ": "SIM_HZ",
                 # N208: solver ITERATIONS are the OTHER half of the discretisation and were
                 # frozen at the literal 80 with no knob at all, so the N207 force
                 # non-convergence had two indistinguishable causes: the integration RATE
                 # (dt too coarse for the contact) or the sequential-impulse ITERATION count.
                 # Registered so `--compare-env AEGIS_SOLVER_ITERS=80` pairs any iteration
                 # budget against the frozen one on the SAME seeds; rate-independent by
                 # construction (the control tick, the phase labels and the press law are
                 # untouched), so the two axes are separable.
                 "AEGIS_SOLVER_ITERS": "SOLVER_ITERS",
                 # N209: the tool head's OWN lateralFriction. Bullet combines the two bodies'
                 # coefficients multiplicatively (measured, N209.1), so this constant is a
                 # second factor on the swept axis: at the frozen 0.9 the REALIZED contact
                 # friction is 0.9*mu, i.e. the band is [0.045, 0.720] and not the declared
                 # [0.05, 0.80]. Registered so `--compare-env AEGIS_TOOL_FRICTION=0.9` pairs
                 # any tool coefficient against the frozen one on the SAME seeds (G4).
                 "AEGIS_TOOL_FRICTION": "TOOL_FRICTION",
                 # N210: the NORMAL-FORCE axis -- the press setpoint, the pad stiffness that
                 # fixes its penetration, and the workspace rail that caps it. All three were
                 # bare literals (0.020 / 1e3 / 3.0) in every run 1-358, so the headline
                 # "0.5 N, not the 10-25 N spec" limitation had never been probed. Registered
                 # so `--compare-env AEGIS_PRESS_M=0.020` pairs any force against the frozen
                 # one on the SAME seeds (G4). Defaults are the frozen literals, so every other
                 # arm stays byte-identical.
                "AEGIS_PRESS_M": "PRESS_M", "AEGIS_CONTACT_K": "CONTACT_K",
                "AEGIS_F_CLAMP_N": "F_CLAMP_N", "AEGIS_KP_GAIN": "KP_GAIN",
                # N211: the SCORING-KERNEL axis. Purely diagnostic -- the scored
                # coverage_cont/success are produced by the untouched `_coverage_cont`; this
                # flag only adds the alternative re-measurements of the same physics contacts.
                "AEGIS_COV_KERNEL": "COV_KERNEL",
                # N212: the TIME-DOSE axis. `T_MAX = 400` was a bare literal in every run
                # 1-360 (all 560 archived rig headers record steps=400), and it is not a
                # neutral bookkeeping constant: `self.steps = round(T_MAX * max(1, cur/ref))`
                # and `v_cmd = total_len_m / (n * TICK_S)`, so the tick budget IS the
                # commanded scrub speed -- 400 ticks x 0.05 s = 20 s over the 0.9098 m
                # fixture_B pass, i.e. 45.5 mm/s, the paper's cycle-time number. The whole
                # segment is certified at ONE point in speed space and that point was never
                # varied. Registered (env + --compare-env) so any dose pairs against the
                # frozen 400 on the SAME seeds (G4). No force, scoring, geometry or phase
                # term is touched: phases come from PATH POSITION, so the scrub FRACTION is
                # dose-invariant and only the per-tick arclength step changes.
                #
                # PRE-REGISTERED PREDICTIONS (stated before any run; see equations.md N212):
                #  D1 at pose noise 0,0 fixture_B stays coverage_cont 1.0000 / 20-of-20 for
                #     every n in [100, 1600] -- the certification is dose-robust over a 16x
                #     cycle-time band.
                #  D2 the first loss is CROSS-TRACK PD lag under the trochoid loop curvature,
                #     e_lat = m v^2 / (R KP) reaching half the pad radius r_eff/2 = 0.0175 m
                #     with m=0.080 kg, R=TROCHOID_R_M=0.015 m, KP=25 N/m -> v_crit=0.286 m/s
                #     -> n_crit = 0.77 / (TICK_S * v_crit) ~= 54 ticks. So n>=100 unchanged,
                #     n=50 marginal, n=25 and n=12 degraded. D2 is REFUTED if coverage falls
                #     at n>=200.
                #  D3 contact SAMPLING is not the binder: contact spacing is L_scrub/n =
                #     0.0019 m at n=400 and only exceeds the 2*r_eff = 0.07 m reach at
                #     n <~ 11, i.e. below every dose in the ladder.
                #  D4 the slow end keeps coverage (N210.3's slip law e = mu*Fn/KP is
                #     speed-INDEPENDENT) and pays only in stick_frac and cycle time.
                "AEGIS_STEPS": "T_MAX",
                # N213: the FIXED planning offset (see POSE_FIX_U/V). Registered so any held
                # offset pairs against the frozen champion on the SAME seeds (G4); defaults are
                # 0.0, the frozen behaviour.
                "AEGIS_POSE_FIX_U": "POSE_FIX_U", "AEGIS_POSE_FIX_V": "POSE_FIX_V",
                # N214: the TOOL-BODY axis. Absolute in-plane half-extents (m) and absolute
                # mass (kg); 0.0 = the frozen per-tool value. `tool_id = seed % 3` changes the
                # footprint, the mass AND the thickness at once and is aliased with the seed,
                # so the head has never been a controlled dose -- yet r_eff is the term every
                # coverage law in the segment is written in (the N211 kernel dilation, the N212
                # frontier s <= 0.486 * 2 r_eff, the "raster fails = tool 0" claim). Only the
                # collision box's in-plane extents and the body mass move; thickness (hence
                # `lift`), the press law, the path and the scoring kernel are untouched.
                "AEGIS_PAD_HU_M": "PAD_HU_M", "AEGIS_PAD_HV_M": "PAD_HV_M",
                "AEGIS_PAD_MASS": "PAD_MASS",
                # N215: the FIXTURE-FACE axis. The two top faces are bare literals in
                # `_build_fixture` (box halfExtents [0.34, 0.14, 0.12], cylinder radius 0.32)
                # and no run 1-363 has ever moved them -- `tank_shape` is a two-valued coin and
                # it aliases with the seed. Yet "zero-shot transfer to a NEW fixture" is a claim
                # about the face, and N213 measured the face boundary as the place a plan-frame
                # offset launches the head. Absolute in-plane half-extents (m); 0.0 = frozen.
                # Registered so `--compare-env AEGIS_FACE_HU_M=0.34` pairs any face against the
                # frozen champion on the SAME seeds (G4). Height, patch, plan, press law and the
                # scoring kernel are untouched.
                "AEGIS_FACE_HU_M": "FACE_HU_M", "AEGIS_FACE_HV_M": "FACE_HV_M",
                "AEGIS_FACE_R_M": "FACE_R_M",
                # N216: the SCORED-WINDOW axis -- the metric's DENOMINATOR. `half = 0.20`,
                # `side = 0.12/0.18` were bare literals in four places for all 457 logged
                # rows, and `scrub_grid` floors them to `[nu*CELL_M, nv*CELL_M]` cells, so
                # every coverage number in the segment is a fraction of an unreported
                # (and v-truncated) window. Absolute metres; 0.0 = the frozen literal.
                # Registered so `--compare-env AEGIS_PATCH_HU_M=0.2,AEGIS_PATCH_SIDE_M=0`
                # pairs any patch against the frozen champion on the SAME seeds (G4). The
                # patch is the TASK (plan rows + scored window together, from the single
                # `patch_extents` source); `_coverage_cont` and the press law are untouched.
                "AEGIS_PATCH_HU_M": "PATCH_HU_M", "AEGIS_PATCH_SIDE_M": "PATCH_SIDE_M"}
# N212: T_MAX is a COUNT, and the header/summary record it as an int; a float knob write is
# coerced back so the frozen arm's records stay byte-identical to runs 1-360.
KNOB_INTS = {"T_MAX"}
# string-typed knobs (a float value maps 1->enabled flag string, 0->off)
KNOB_STR_GLOBALS = {"AEGIS_REG": "REG_MODE", "AEGIS_REG_EST": "REG_EST"}
KNOB_STR_ON = {"REG_MODE": "depth", "REG_EST": "extent"}
# a knob write retires the coupled form it replaces, else the value would be ignored
KNOB_LEGACY = {"AEGIS_TROCH_AMP_M": "TROCHOID_AMP_LEGACY", "AEGIS_TROCH_W": "TROCHOID_W_LEGACY"}


def apply_knobs(spec: str) -> dict:
    """Purpose: override the geometry knobs of the currently-running arm.
    Inputs: "KEY=FLOAT[,KEY=FLOAT]". Outputs: the applied {global_name: value} map.
    Only keys in KNOB_GLOBALS are accepted; an unknown key is a hard error (G7: never
    silently swap a scored quantity). Run 170: AEGIS_GATE is REJECTED (in-loop control
    deleted; post-hoc tag only).
    """
    out: dict = {}
    for part in (p for p in spec.split(",") if p.strip()):
        key, _, val = part.partition("=")
        key, val = key.strip(), val.strip()
        if key not in KNOB_GLOBALS and key not in KNOB_STR_GLOBALS:
            raise SystemExit(f"--compare-env: unknown knob {key!r}; known: "
                             f"{sorted(KNOB_GLOBALS) + sorted(KNOB_STR_GLOBALS)}")
        if key in KNOB_STR_GLOBALS:
            name = KNOB_STR_GLOBALS[key]
            globals()[name] = KNOB_STR_ON[name] if float(val) != 0.0 else ""  # noqa: PLW0603
            out[name] = float(val)
            log(f"knob override {key}={val} -> {name}={globals()[name]!r}")
            continue
        globals()[KNOB_GLOBALS[key]] = float(val)  # noqa: PLW0603 -- deliberate arm switch
        if KNOB_GLOBALS[key] in KNOB_INTS:
            globals()[KNOB_GLOBALS[key]] = max(1, int(round(float(val))))  # noqa: PLW0603
        if key in KNOB_LEGACY:
            globals()[KNOB_LEGACY[key]] = False  # noqa: PLW0603 -- coupled form retired
        out[KNOB_GLOBALS[key]] = float(val)
        log(f"knob override {key}={val} -> {KNOB_GLOBALS.get(key, key)}")
    return out


def knob_snapshot() -> dict:
    """Purpose: record every geometry knob + gate flag in force, so a results header/compare record
    can be checked against a claim. Inputs: none. Outputs: {name: value}.
    Run 170: gate is POST-HOC tag only (in-loop veto deleted); flag is constant."""
    base = {n: globals()[n] for n in KNOB_GLOBALS.values()}
    base["AEGIS_REG"] = REG_MODE
    base["AEGIS_REG_HALF_M"] = REG_HALF_M
    base["AEGIS_REG_N"] = REG_N
    base["AEGIS_REG_EST"] = REG_EST
    base["AEGIS_REG_CASTS"] = REG_CASTS
    base["AEGIS_REG_DFACT"] = REG_DFACT
    base["AEGIS_REG_CN"] = REG_CN
    base["AEGIS_GATE"] = ("proportional" if GATE_PROP else
                          "advisory" if GATE_ADVISORY else "post-hoc")
    return base
# I5: planning-pose error (Tier-2 registration noise). "sigma_t_m,sigma_yaw_deg".
POSE_NOISE = os.environ.get("AEGIS_POSE_NOISE", "0,0")
# N213: the POSE-NOISE AXIS' OWN DISTRIBUTION -- the last declared factor never probed, and the
# only one the paper's "Tier-2 accuracy requirement" is written in. `POSE_NOISE` is a per-axis
# sigma of an UNBOUNDED Gaussian draw, so every labelled level in runs 1-361 ("sigma_t =
# 0.03 m") is a MIXTURE over realized offsets spanning 0 .. ~3 sigma with n = 20 draws, and the
# offset that actually moves coverage_cont has never been held at a controlled magnitude. These
# two knobs add a FIXED planning offset to the drawn noise, so the offset is held exactly at a
# chosen (u, v) while the seed still varies friction / tool / customer -- the axis turned from a
# random variable into a controlled dose, which is what the mechanism needs.
# Default 0.0 = the frozen behaviour (no byte of the plan changes), registered in KNOB_GLOBALS so
# `--compare-env AEGIS_POSE_FIX_U=0` pairs any offset against the frozen champion on the SAME
# seeds (G4). PLAN ONLY: `scrub_grid` and the scoring frame are untouched, so coverage_cont and
# success are still computed from physics contacts against the TRUE patch (G7) and the tool is
# never teleported -- this is the same kind of plan-frame error pose_noise already injects.
POSE_FIX_U = float(os.environ.get("AEGIS_POSE_FIX_U", "0"))
POSE_FIX_V = float(os.environ.get("AEGIS_POSE_FIX_V", "0"))
# N214: the TOOL-BODY FOOTPRINT axis -- the last unvaried term in the measurement chain. Every
# run 1-362 varies the head only through `tool_id = seed % 3`, which changes the footprint
# (r_eff = 0.035 / 0.040 / 0.050), the MASS (0.080 / 0.105 / 0.092) and the THICKNESS
# (0.012 / 0.030 / 0.006) AT ONCE, and it is aliased with the seed, so the footprint has never
# been a controlled dose. r_eff is the term every coverage law in the segment is written in: the
# N211 kernel dilation, the N212 contact-spacing frontier s <= 0.486 * 2 r_eff, and the oldest
# claim in the backlog (raster "fails = tool 0 (r_eff 3.5 cm) only"). These three knobs give the
# footprint and the mass as ABSOLUTE quantities in metres / kilograms, so any pad is reachable and
# the head's plan-frame behaviour is unchanged.
# Default 0.0 = "use the frozen per-tool value", so the default arm is byte-identical to run 350.
# Only the collision box's in-plane half-extents and the body mass are touched: the THICKNESS
# (half[2], hence `lift`) stays per-tool, so the approach height, the press law, `scrub_grid`,
# `scrub_waypoints` and `_coverage_cont` are all untouched, coverage_cont and success are still
# computed from physics contacts against the true patch, and the tool is never teleported.
PAD_HU_M = float(os.environ.get("AEGIS_PAD_HU_M", "0"))
PAD_HV_M = float(os.environ.get("AEGIS_PAD_HV_M", "0"))
PAD_MASS = float(os.environ.get("AEGIS_PAD_MASS", "0"))
# N215: the FIXTURE-FACE axis -- the last unvaried literal group, and the one the segment's
# headline claim is written in. `_build_fixture` hard-codes the two faces (elongated box
# halfExtents [0.34, 0.14, 0.12], round cylinder radius 0.32, height 0.44) and NOTHING in
# runs 1-363 has ever changed them: `tank_shape` is a two-valued coin, and it is the FACE, not
# the plan or the head, that a "zero-shot transfer to a new fixture" claim is about. N213 already
# showed the face boundary is where a plan-frame offset launches the head; this axis asks how much
# face there has to BE. Absolute in-plane extents in metres, 0.0 = the frozen literal.
# Default 0.0 everywhere => the default arm is byte-identical to run 350. ONLY the collision
# shape's in-plane extents move: the height (hence the top-face plane and `lift`), the scrub
# patch (`scrub_grid`), the plan (`scrub_waypoints`), the press law and `_coverage_cont` are all
# untouched, so coverage_cont and success are still computed from physics contacts at return time
# and the tool is never teleported (G7). The floor plane is untouched too, so a head that runs off
# a shrunken face still has somewhere to fall -- that is the measurement, not a rescue.
FACE_HU_M = float(os.environ.get("AEGIS_FACE_HU_M", "0"))
FACE_HV_M = float(os.environ.get("AEGIS_FACE_HV_M", "0"))
FACE_R_M = float(os.environ.get("AEGIS_FACE_R_M", "0"))
# N216: the SCORED-WINDOW axis -- the last unvaried literal group, and it is the metric's
# DENOMINATOR. The scrub patch is the literal `half = 0.20`, `side = 0.12 (elongated) / 0.18
# (round)` in FOUR places (bowl_au_bv, scrub_uv, scrub_grid, and the I14 inset guard) for all
# 457 logged rows, and `scrub_grid` turns it into the cell counts `_coverage_cont` divides by:
# `nu = int(2*half/CELL_M)`, `nv = int(side/CELL_M)`, so the scored window is
# [nu*CELL_M, nv*CELL_M] -- an integer number of 0.05 m cells that starts at -side/2 and is
# therefore TRUNCATED in v (0.10 of the declared 0.12; 0.15 of 0.18) and asymmetric about the
# patch centre. No rig header in any run records it. Every coverage number in the segment is a
# FRACTION OF AN UNREPORTED WINDOW, and "transfer to a new fixture" is a claim about a task
# that has never been resized. Absolute metres; 0.0 = the frozen literal, so the default arm is
# byte-identical to run 350. One helper `patch_extents()` is the single source for all four
# sites, so the plan and the metric cannot drift apart. `_coverage_cont` is left VERBATIM: the
# scored pair stays a pure function of the physics contacts; the patch dose changes the TASK
# (plan and scored window together), never the kernel expression (G7).
#
# PRE-REGISTERED (written into this source before the first N216 run; equations.md ROW N216):
#   X1 scale covariance: trochoid coverage_cont = 1.0000 for every SMALLER patch, and the u
#      cliff is bracketed in (0.30, 0.32] m, i.e. where the plan's own reach half + 2*amp
#      (amp = TROCHOID_R_M = 0.015) passes N215's face half 0.34 -- N215's plan-reach law
#      re-tested at a second, independent operating point.
#   X2 the raster discriminator: raster has NO loop excursion (its reach is half exactly), so
#      its cliff must sit 2*amp = 0.030 m LATER, in (0.34, 0.36]. If both modes cliff at the same
#      half, the binder is the task, not the plan's reach, and X1 is refuted.
#   X3 the truncation: the declared-patch kernel must read coverage_cont = 1.0000 on all three
#      suites at 0,0 (the unscored v rim is inside the loop excursion + r_eff), so the frozen
#      certificate is CONSERVATIVE -- it holds on 120% of the declared patch area and scoring the
#      declared patch would not move any verdict. Refuted if the declared kernel reads < 1.0 on
#      any suite, which would make the reported metric optimistic by the truncation.
#   X4 similarity: at x1.5 similarity (patch + face + pad footprint + pad mass) coverage_cont =
#      1.0000 on fixture_B -- the certificate is a GEOMETRIC one, and the frozen numbers are one
#      representative point of a scale-free region. Refuted if it drops below the frozen 1.0000.
#   X5 the pad ratio: at a fixed enlarged patch a 1.5x pad does not move the cliff, so the
#      coverage binder is a RATIO (footprint / window), not an absolute length -- which would
#      re-express N214's absolute floor r_eff >= 0.0275 m in window units. Refuted if the cliff
#      moves with the pad.
#   X6 metric granularity: at a non-saturated arm (raster, 0,0) coverage_cont as a function of
#      `side` jumps UP by one row of cells exactly where nv = int(side/CELL_M) increments
#      (side 0.149 -> 0.150), because the scored window and the plan row count step together. The
#      bar's own quantum is 1/(nu*nv*4), so it is a function of the unreported window.
PATCH_HU_M = float(os.environ.get("AEGIS_PATCH_HU_M", "0"))
PATCH_SIDE_M = float(os.environ.get("AEGIS_PATCH_SIDE_M", "0"))


def patch_extents(spec: dict) -> tuple[float, float]:
    """Purpose: N216 -- the single source for the scrub patch's half-extent (u) and side (v),
    the literal pair that is simultaneously the TASK (scrub_uv rows) and the metric's
    denominator (scrub_grid cell counts). 0.0 on a knob keeps the frozen literal.
    Inputs: fixture spec dict. Outputs: (half_u, side_v) in metres.
    """
    shape = spec.get("tank_shape", "round")
    half = 0.20 if PATCH_HU_M <= 0.0 else PATCH_HU_M
    side = (0.12 if shape == "elongated" else 0.18) if PATCH_SIDE_M <= 0.0 else PATCH_SIDE_M
    return half, side


def patch_window(spec: dict) -> tuple[list, int, int]:
    """Purpose: N216 -- the SCORED window as the metric actually sees it, plus the truncation
    arithmetic against the DECLARED patch. Read-only bookkeeping for the header: the scored
    coverage_cont comes from `_coverage_cont` on the untouched frozen expression.
    Inputs: fixture spec dict. Outputs: (origin [u,v], n_u, n_v).
    """
    half, side = patch_extents(spec)
    nu = max(1, int(2 * half / CELL_M))
    nv = max(1, int(side / CELL_M))
    return [-half, -side * 0.5], nu, nv


def patch_report() -> dict:
    """Purpose: N216 -- per-shape bookkeeping for the header: the declared patch, the SCORED
    window the metric actually divides by, its fine cell count (the bar's own quantum is
    1/(that count)) and the v truncation `1 - nv*CELL_M/side`. Read-only.
    Inputs: none (reads the knobs and CELL_M). Outputs: {shape: {...}}.
    """
    out = {}
    for shape in ("elongated", "round"):
        half, side = patch_extents({"tank_shape": shape})
        nu, nv = max(1, int(2 * half / CELL_M)), max(1, int(side / CELL_M))
        out[shape] = {"half_u_m": half, "side_v_m": side,
                      "scored_window_u_m": nu * CELL_M, "scored_window_v_m": nv * CELL_M,
                      "scored_cells_coarse": nu * nv, "scored_cells_fine": 4 * nu * nv,
                      "v_truncation_frac": round(1.0 - nv * CELL_M / side, 4)}
    return out


def face_half_extents(spec: dict) -> tuple[float, float]:
    """Purpose: the fixture TOP FACE's in-plane half-extents (m) -- the N215 dose, and the one
    place the frozen literals live. Round face = (radius, radius); elongated face = (hu, hv).
    0.0 on a knob keeps the frozen literal, so the default arm is byte-identical.
    Inputs: fixture spec dict. Outputs: (half_u, half_v) in metres.
    """
    if spec.get("tank_shape") == "elongated":
        return (0.34 if FACE_HU_M <= 0.0 else FACE_HU_M,
                0.14 if FACE_HV_M <= 0.0 else FACE_HV_M)
    r = 0.32 if FACE_R_M <= 0.0 else FACE_R_M
    return (r, r)
# N207: SOLVER RESOLUTION. Every run in this segment (1..355) used exactly ONE integrator
# setting, so the certified fixture_B transfer_success 1.00 is a statement about ONE point in
# discretisation space and has never been refined. This knob changes ONLY the integration
# rate; the CONTROL tick stays 20 Hz, so the commanded arclength speed, the phase labels, the
# press law and the tick budget are all bit-identical across rates. 240 = the frozen value
# (12 substeps x 1/240 s = 0.05 s tick), so the default arm is byte-identical and
# `--compare-env AEGIS_SIM_HZ=240` pairs any rate against the champion on the SAME seeds (G4).
# Stiffness criterion, stated so the sweep is falsifiable rather than arbitrary: the explicit
# contact integrator is stable while sqrt(contactStiffness/m)*dt stays near 1, and this rig
# runs at 0.466 for its lightest pad (m=0.080 kg, CONTACT_K=1e3) at 240 Hz. So 120 Hz is
# predicted to be the LAST rate that still launches cleanly and 480/960 Hz to be the CONVERGED
# limit. If coverage_cont moves materially with the rate, the certification is a discretisation
# artifact; if it is flat, the certification is solver-independent and the claim is stronger.
SIM_HZ = float(os.environ.get("AEGIS_SIM_HZ", "240"))
TICK_S = 0.05   # control period (20 Hz), rate-INDEPENDENT by design: only the integrator moves
# N208: solver ITERATION budget (sequential-impulse passes per step). This was the literal 80
# in every run 1-356, so N207's "force_compliance is not converged at 240 Hz" had two
# indistinguishable causes -- the integration RATE or the ITERATION count -- and they have
# opposite engineering readings (8x the wall clock, or a free knob). 80 = the frozen value.
# Orthogonal to SIM_HZ by construction: it changes how well each fixed dt is solved, not dt.
SOLVER_ITERS = float(os.environ.get("AEGIS_SOLVER_ITERS", "80"))


def substeps_for(sim_hz: float | None = None) -> int:
    """Purpose: physics substeps per 20 Hz control tick at a given integrator rate, so a tick
    always advances 0.05 s of simulated time. Inputs: sim_hz (Hz, None = the current global).
    Outputs: substep count. 240 Hz -> 12 (the frozen value), so the default arm is
    byte-identical to runs 1-355.
    ponytail: the global is read at CALL time, never as a default argument -- a bound default
    silently ignores --compare-env (measured: the 240 Hz paired arm kept the 480 Hz substep
    count and read coverage_cont 0.3625 on fixture_A instead of 1.0000).
    """
    hz = SIM_HZ if sim_hz is None else sim_hz
    return max(1, int(round(TICK_S * hz)))


def solver_iters_for(iters: float | None = None) -> int:
    """Purpose: the sequential-impulse iteration budget for one step, as an int.
    Inputs: iters (count, None = the current global). Outputs: int >= 1.
    80 -> the frozen value used by every run 1-356, so the default arm is byte-identical.
    ponytail: same N207d rule -- the global is read at CALL time, never bound as a default
    argument, so --compare-env cannot be silently ignored.
    """
    return max(1, int(round(SOLVER_ITERS if iters is None else iters)))
# N209: TOOL lateral friction -- the second factor of Bullet's multiplicative friction combine.
# Measured with a ramp probe (experiments/N209_friction_combine_probe.py): the realized contact
# coefficient is mu_tool * mu_fixture + ~0.016 (statically discriminated against min and max:
# 0.8/0.1 and 0.1/0.8 BOTH slide at 0.088, so neither min nor max). The literal 0.9 in every
# run 1-357 therefore scaled the declared friction band [0.05, 0.80] down to [0.045, 0.720] --
# the axis was swept, but its labels were wrong, and no run recorded the factor.
# 0.9 = the frozen value (default arm byte-identical); 1.0 makes realized == commanded, which
# is the only setting in which mu means what the paper says it means.
TOOL_FRICTION = float(os.environ.get("AEGIS_TOOL_FRICTION", "0.9"))


def tool_friction_for(mu: float | None = None) -> float:
    """Purpose: the tool head's lateralFriction for one episode, read at CALL time.
    Inputs: optional explicit value. Outputs: float > 0.
    ponytail: same N207d rule as the two discretisation knobs -- a default argument would bind
    at def time and make --compare-env silently inert.
    """
    return TOOL_FRICTION if mu is None else float(mu)
# ponytail, three measured rig constraints (each cost a debugging round):
#  1. the head is FORCE-driven, never velocity-driven. resetBaseVelocity overwrites the
#     contact solver, so a velocity servo pins the head to the command and the swept
#     friction coefficient has literally zero effect (measured: identical residual at
#     mu=0.05 and mu=0.80).
#  2. pybullet clears applyExternalForce on every stepSimulation, so the force MUST be
#     re-applied inside the substep loop -- once per tick delivers 1/12 of the intended
#     force and the head simply stalls (measured: 0.01 m/s vs a 0.085 m/s target).
#  3. contact integration is stable only while sqrt(contactStiffness/m)*dt stays near 1.
#     At the 1e4 default with a 0.018 kg head that is 3.1 -> the head launches (measured
#     500 m excursions). Soft foam-pad masses (0.08-0.105 kg) + 1e3 stiffness gives 0.47.
#     The mass also sets the FLOOR on normal force via the gravity comp (N >= m*g), so a
#     light head is what lets mu*Fn fall below the lateral demand somewhere inside 0.05-0.80.
# N210: the NORMAL-FORCE axis. In every run 1-358 the commanded press was the bare product
# KP_PRESS*KP*PRESS_M = 1.0*25*0.020 = 0.500 N, i.e. the "0.5 N operating point" in the paper's
# limitation list is KP_PRESS*KP*PRESS_M with NO knob: the only force-ish knob that existed,
# FN_SET_N, is inert at the frozen FORCE_PI=0 (it only sets the force_compliance BAND edges), so
# the normal force was a constant, never a swept factor. 0.020 = the frozen value (default arm
# byte-identical). Physical read: a real pad's effective stiffness scales with its working
# force, so a force ladder must carry CONTACT_K with it to hold penetration at the frozen
# Fn/K = 0.5 mm -- hence AEGIS_CONTACT_K below.
KP_PRESS = 1.0
PRESS_M = float(os.environ.get("AEGIS_PRESS_M", "0.020"))  # press setpoint -- a commanded
                                   # offset, NOT a real sink: actual penetration is Fn/CONTACT_K
CONTACT_K = float(os.environ.get("AEGIS_CONTACT_K", "1.0e3"))   # ~0.6 mm at the frozen 0.5 N
CONTACT_C = 2.0e2
# N210: the workspace rail is a HARD CEILING on the total commanded force, so at the frozen
# 3.0 N it caps the achievable normal force at 3 N whatever the press asks for -- i.e. the
# AEGIS spec band 10-25 N is unreachable by arithmetic, not by tuning (the N199 shape of
# argument). 3.0 = the frozen value; the rail is logged and its binding counted per episode.
F_CLAMP_N = float(os.environ.get("AEGIS_F_CLAMP_N", "3.0"))  # workspace rail: a real cell has
                                   # a force limit
WS_LIMIT_M = 0.60                  # head may not stray this far from the fixture axis
# ponytail: the actuator is FIXED across every episode. An earlier version derived the PD
# gains from compliance_mode, which made the controller an oracle that pre-compensated for
# the very friction being swept -- measured 100% success from mu=0.05 to mu=0.80, i.e. a
# dead sweep. compliance_mode is now a LOGGED context token only (the AEGIS spec asks for
# it as metadata, not as a control input). With kp fixed, lateral grip is ~mu*kp*PRESS_M
# and the pass/fail transition lands inside the 0.05-0.80 band.
KP, KD = 25.0, 1.9
# N210: the actuator GAIN, as a multiplier on both terms so the damping ratio KD/KP is preserved
# (a proportional-only change would silently also change the controller's character). The
# N210.3 law says the tracking residual is e = mu*Fn/KP, so gain is the knob that buys force
# operating point: 1.0 = the frozen value, byte-identical.
KP_GAIN = float(os.environ.get("AEGIS_KP_GAIN", "1.0"))
SCRIPTED_QUALITY_GATE = 0.70      # AEGIS keep-bar for Fixture-B zero-shot (transfer_success)
CLEAN_FRAC = 0.90                 # episode success: >90% of patch area swept in contact
FIELDS = ("record", "ts", "suite", "seed", "backend", "friction", "tool_id",
           "se3_offset_xyz", "se3_offset_rpy", "compliance_mode", "quality_tag",
           "failure_metadata", "success", "jerk", "coverage", "slip_m", "stall_frac",
           "steps", "wall_s", "status", "gate_intercepted", "gate_on", "vetoes",
           "vetoed_ticks")

def is_gate_active() -> bool:
    """Purpose: Run 170 revert -- the Tier-4 gate is a POST-HOC tag only (classify(),
    fixed threshold GATE_MAX_JERK=0.618, score weight only, never control). The in-loop
    veto + soft threshold + contact override (runs 167-169) destroyed contact under
    pose noise, so all in-loop control is deleted; this always returns False.
    Run 240 (I19): the only in-loop gate that survives is the ADVISORY one -- same
    per-tick jerk trigger as run 167, but the response is a commanded-speed HALVING
    (no retraction, no position jump). Off unless AEGIS_GATE_ADVISORY=1, so every other
    arm stays bit-identical to the frozen rig.
    Run 241 (I20): the PROPORTIONAL press-scale gate -- a continuous, no-bang-bang
    response on the SAME per-tick force-magnitude second difference. Off unless
    AEGIS_GATE_PROP=1. AEGIS_GATE_TRIG=1 restricts the trigger to the scrub phase
    while the head is IN CONTACT (the I19 class-level finding: the legacy trigger never
    fires there, being a units mismatch with the episode jerk_proxy it was copied from);
    AEGIS_GATE_THETA replaces GATE_MAX_JERK as the threshold in that mode."""
    return bool(GATE_ADVISORY or GATE_PROP)

GATE_ADVISORY = float(os.environ.get("AEGIS_GATE_ADVISORY", "0"))  # 1 = I19 advisory gate ON
GATE_SLOW_SCALE = 0.5     # I19: commanded arclength-speed multiplier inside an advisory window
GATE_SLOW_TICKS = 8       # I19: advisory window length (ideas.md I19 spec: 8 ticks)
# I20: proportional press-scale gate. press_scale = clip(1 - max(0, j - 0.7*theta)/(0.3*theta),
# GATE_PROP_MIN, 1) applied to the SCRUB press only -- no retraction, no position jump, no
# teleport (G7). GATE_TRIG=0 -> the legacy all-phase trigger at theta=GATE_MAX_JERK (the
# I19 record run, expected inert); GATE_TRIG=1 -> the NEW trigger: scrub phase AND in
# contact, at theta=GATE_THETA (calibrated from the measured distribution, see run 241).
GATE_PROP = float(os.environ.get("AEGIS_GATE_PROP", "0"))
GATE_TRIG = float(os.environ.get("AEGIS_GATE_TRIG", "0"))
GATE_THETA = float(os.environ.get("AEGIS_GATE_THETA", "0.618"))
GATE_PROP_MIN = float(os.environ.get("AEGIS_GATE_PROP_MIN", "0.3"))
GATE_MAX_JERK = 0.618       # Tier-4 threshold, FIXED (Run 170: never scaled, never control)

T0 = time.time()


def log(msg: str) -> None:
    """Purpose: single stdout sink with elapsed seconds (Kaggle log scraping).
    Inputs: message string. Outputs: one flushed line.
    """
    print(f"[aegis-sweep {time.time() - T0:8.1f}s] {msg}", flush=True)


# --- install header (Kaggle T4 image ships torch+CUDA; install only the gaps) ---
def ensure_deps() -> list[str]:
    """Purpose: pip-install maniskill/mujoco/pybullet/hf_hub only when missing, so a
    warm kernel costs zero seconds and a cold one is self-sufficient.
    Inputs: none. Outputs: list of pip specs actually installed.
    """
    import importlib.util
    if not INSTALL_DEPS:
        return []
    want, done = [], []
    for mod, spec in (("mujoco", "mujoco>=3.1.6"),
                      ("pybullet", "pybullet>=3.2.6"),
                      ("mani_skill", "mani_skill>=3.0.0b20"),
                      ("huggingface_hub", "huggingface_hub>=0.24"),
                      ("torch", None)):  # torch last: it is the slowest and usually present
        if importlib.util.find_spec(mod) is not None:
            continue
        if spec is None:
            spec = "torch"
        want.append(spec)
    if not want:
        return done
    log(f"pip install {' '.join(want)}")
    r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", *want],
                       capture_output=True, text=True)
    if r.returncode != 0:
        log(f"pip rc={r.returncode}: {r.stderr[-400:]}")
    done = [s for s, m in zip(want, ("mujoco", "pybullet", "mani_skill",
                                     "huggingface_hub", "torch"))
            if importlib.util.find_spec(m) is not None]
    log(f"deps ready: {done}")
    return done


# --- geometry ------------------------------------------------------------------
def fixture_offset(spec: dict) -> list[float]:
    """Purpose: translate a fixture spec into a metric SE(3) xyz offset, matching
    restroom_sim.offset_cm semantics (15 cm lateral, one budget, not two).
    Inputs: fixture spec dict. Outputs: [x, y, z] metres.
    """
    return [round(float(spec.get("offset_cm", 0)) / 100.0, 4),
            round(float(spec.get("offset_y_cm", 0)) / 100.0, 4), 0.0]


def fixture_rpy(spec: dict) -> list[float]:
    """Purpose: yaw-only rotation for the fixture body.
    Inputs: fixture spec dict. Outputs: [roll, pitch, yaw] radians.
    """
    return [0.0, 0.0, round(math.radians(float(spec.get("angle_deg", 0))), 5)]


def fixture_pose(spec: dict) -> tuple[list[float], list[float]]:
    """Purpose: world position/orientation of the fixture body: lateral offset from
    offset_cm, plus the wall-hung lift for the elongated Fixture B.
    Inputs: fixture spec dict. Outputs: (position[3], quaternion[4]).
    """
    pos = fixture_offset(spec)
    if spec.get("tank_shape") == "elongated":
        pos[2] += WALL_HUNG_LIFT_M
    return pos, quat_from_rpy(*fixture_rpy(spec))


def bowl_au_bv(spec):
    # I8: paraboloid semi-axes for the bowl liner (match the scrub patch). N216: the patch
    # pair comes from the single source `patch_extents`, so the bowl cannot disagree with it.
    half, side = patch_extents(spec)
    return half, side * 0.5


def bowl_z(spec, u, v):
    # I8: bowl surface height above the flat top face at fixture (u, v). >= 0.
    if SURFACE != "bowl":
        return 0.0
    a, b = bowl_au_bv(spec)
    return BOWL_K * ((u / a) ** 2 + (v / b) ** 2)


def bowl_n(spec, u, v):
    # I8: outward surface normal in the FIXTURE frame. Unit (nx, ny, nz).
    if SURFACE != "bowl":
        return (0.0, 0.0, 1.0)
    a, b = bowl_au_bv(spec)
    gu, gv = 2 * BOWL_K * u / a ** 2, 2 * BOWL_K * v / b ** 2
    n = math.sqrt(gu * gu + gv * gv + 1.0)
    return (-gu / n, -gv / n, 1.0 / n)


def quat_from_rpy(roll: float, pitch: float, yaw: float) -> list[float]:
    """Purpose: ZYX euler -> xyzw quaternion (MuJoCo/PyBullet convention).
    Inputs: three radians floats. Outputs: [x, y, z, w].
    """
    cy, sy = math.cos(yaw * 0.5), math.sin(yaw * 0.5)
    cr, sr = math.cos(roll * 0.5), math.sin(roll * 0.5)
    cp, sp = math.cos(pitch * 0.5), math.sin(pitch * 0.5)
    return [sr * cp * cy - cr * sp * sy, cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy, cr * cp * cy + sr * sp * sy]


def scrub_uv(spec: dict, mode: str = "raster") -> list[tuple[float, float]]:
    """Purpose: the scrub-phase path in the FIXTURE frame (u along the long axis, v across).
    raster   -- legacy boustrophedon, rows at CELL_M pitch, 180 deg reversals at row ends.
    rounded  -- SAME rows, each reversal replaced by a semicircle of radius CELL_M/2
                (C1: isolates the reversal/stiction effect, nothing else changes).
    trochoid -- rounded rows + superimposed circular scrub loops of radius TROCHOID_R_M
                (brush-like "spiral pattern"; speed never reaches zero).
    spiral   -- inward stadium (offset-rectangle) spiral, corner radius <= CELL_M/2.
    Inputs: fixture spec, path mode. Outputs: list of (u, v) points, <=0.01 m apart
    (densify later makes spacing uniform). Raises ValueError on unknown mode.
    """
    if mode not in PATH_MODES:
        raise ValueError(f"unknown path mode {mode!r}; valid: {PATH_MODES}")
    shape = spec.get("tank_shape", "round")
    half, side = patch_extents(spec)   # N216: single source (frozen literals 0.20 / 0.12|0.18)
    if INSET_M > 0.0 and not WALL_ONLY:
        # I14 containment PLAN: rows, C1 turns and the trochoid loop offsets are all
        # generated inside the wall rect, so the plan is C1 by construction. Clamping the
        # finished polyline instead shaves the C1 turns flat into 180 deg reversals
        # (measured: max turn 101-180 deg -> the C1 assertion fires). 2*R covers the
        # u offset (u + amp*cos(w) - amp reaches u-2R) and R the v offset (v + amp*sin(w)).
        pad = INSET_M + 2.0 * TROCHOID_R_M
        half, side = max(0.0, half - pad), max(0.0, side - 2.0 * pad)
    nu = max(1, int(2 * half / CELL_M))
    nv = max(1, int(side / CELL_M))
    v0 = -side * 0.5
    rows = [v0 + (r + 0.5 * ROW_CENTRE) * CELL_M for r in range(nv)]   # I22: cell centres, not edges
    if MARGIN_M > 0.0:                          # N206 over-coverage margin
        half += MARGIN_M
        k = max(1, int(math.ceil(MARGIN_M / CELL_M + 0.5)))   # outermost row >= MARGIN_M out
        rows = ([v0 - (j + 0.5) * CELL_M for j in range(k - 1, -1, -1)] + rows
                + [v0 + side + (j + 0.5) * CELL_M for j in range(k)])
        nu = max(1, int(2 * half / CELL_M))
    uv: list[tuple[float, float]] = []
    if mode == "raster":
        for r, v in enumerate(rows):
            us = np_linspace(-half, half, nu + 1) if r % 2 == 0 else np_linspace(half, -half, nu + 1)
            uv += [(float(u), v) for u in us]
        return uv
    if mode == "auto":
        # N193: the DEPLOYABLE form of the orbit dose -- one conditional, and not a tuned
        # knob: take the rotation-invariant plan exactly when the registration cannot observe
        # the yaw (the round top face -> `yaw_est = None` in run_episode, which keeps the FULL
        # prior yaw error in the plan), else the rect row plan whose frame the PCA corrects.
        # `spec["tank_shape"]` is the rig's stand-in for that same observable; on an elongated
        # face the disc does not fit (R > inradius, orbit raises) so the rect plan is the only
        # option anyway. This is the whole policy change: nothing else moves.
        mode = "orbit" if shape == "round" else "trochoid"
    if mode == "orbit":
        # N193: the ROTATION-INVARIANT plan. `coverage_cont` scores a RECT patch fixed in
        # the TRUE fixture frame, so a row plan built in a frame whose yaw is wrong misses
        # the corners by exactly the rect-vs-rect overlap -- and on a rotationally symmetric
        # top face the yaw is UNOBSERVABLE (yaw_est = None, registration keeps the prior), so
        # that error cannot be estimated away: it is a floor, not a noise level. The minimal
        # rotation-invariant set containing the patch is the DISC of radius
        # R = hypot(half, side/2): any rotation-invariant set holding a point at radius R must
        # hold its whole orbit, hence the whole disc. It exists inside the face iff the face's
        # inradius >= R -- round 0.32 >= 0.2193 YES, elongated 0.14 < 0.2088 NO (and there the
        # yaw IS observable from PCA, so no invariant plan is needed). Path = Archimedean
        # spiral, turn pitch CELL_M (adjacent turns 0.05 m <= 2*r_eff for all three pads) and
        # r0 = CELL_M/4 (see below), sampled at TROCHOID_DS_M so the inner turn stays under
        # the C1 assertion (measured max turn 17.9 deg).
        # N215: the inradius now reads the SAME face_half_extents() the collision shape is built
        # from, so the orbit guard cannot disagree with the geometry it is guarding.
        r_in = min(face_half_extents(spec))      # top-face inradius, see _build_fixture
        R = math.hypot(half, side * 0.5)                    # patch circumradius
        if R > r_in + 1e-9:
            raise SystemExit(
                f"orbit: no rotation-invariant plan fits a {shape!r} face -- patch "
                f"circumradius {R:.4f} m > face inradius {r_in:.2f} m; on such a face the "
                f"yaw must be OBSERVABLE (registration PCA) instead")
        # r0 = CELL_M/4 is MEASURED on the metric's own fine cells (0.025 m pitch, 96 cells
        # on fixture_A), not tuned: the max distance from a target cell to the spiral is
        # 0.0249 m at r0 = p/4, 0.0248 m at r0 = 0 and 0.0369 m at r0 = p/2 (p = CELL_M),
        # against the SMALLEST pad's r_eff = 0.035 m. So p/2 does NOT cover the patch with
        # the smallest tool head, while p/4 does with 0.0101 m of margin (r0 = 0 ties on
        # margin but starts the sweep on a zero-radius turn, which the C1 sampling dislikes).
        # Cost of yaw ignorance, stated: the disc is 2.1x the patch area and 3.025 m vs the
        # rect plan's 1.412 m of commanded arclength on fixture_A (+114% = cycle time,
        # 931 vs 434 ticks at the same commanded speed).
        r0, c = 0.25 * CELL_M, CELL_M / (2.0 * math.pi)
        uv, phi, r = [], 0.0, r0
        while True:
            uv.append((r * math.cos(phi), r * math.sin(phi)))
            if r >= R - 1e-12:
                break
            phi += TROCHOID_DS_M / math.hypot(r, c)
            r = min(r0 + c * phi, R)
        return uv
    rad = CELL_M / 2.0
    if mode in ("rounded", "trochoid"):
        n_line = max(2, int(2 * half / BASE_DS_M) + 1)
        rows_of: list[int] = []                         # I17: row index per base point
        for r, v in enumerate(rows):
            fwd = r % 2 == 0
            us = np_linspace(-half, half, n_line) if fwd else np_linspace(half, -half, n_line)
            uv += [(float(u), v) for u in us]
            rows_of += [r] * (n_line + 1)
            if r < len(rows) - 1:                      # C1 semicircle to the next row
                cx, cv = (half if fwd else -half), v + rad
                n_arc = 16
                for k in range(1, n_arc):
                    th = -math.pi / 2 + math.pi * k / n_arc
                    uv.append((cx + (rad * math.cos(th) if fwd else -rad * math.cos(th)),
                               cv + rad * math.sin(th)))
                    rows_of.append(r)
        if mode == "trochoid":                          # superimpose scrub loops
            return _loops_offset(uv, rows_of)
        return uv
    # fitted: footprint-aware row pitch (I10) — ponytail: analytic, no learn
    #
    # I10 v2 (iter 29). The first `fitted` was broken twice over, both measured:
    #  1. row COUNT was `int(side/pitch)` starting at v0, so it swept a 0.05 m band of a
    #     0.12 m patch and left the far rim bare (fixture-B coverage_cont 0.743 vs the
    #     champion 0.945). The champion trochoid only clears that rim because its
    #     +-R loop leaks lateral coverage, not because it is geometrically correct.
    #  2. it INSET every row by r_eff, which double-counts the footprint: `_coverage_cont`
    #     already inflates each contact by r_eff, so an inset can only lose rim cells.
    # Correct rule, from the scoring definition: a fine cell is hit iff some contact lies
    # within r_eff of its centre, so consecutive rows need pitch <= 2*r_eff and the band
    # must SPAN the patch. Minimum row count n = ceil(side / (2*r_eff)) placed at the band
    # centres, no inset in u or v. Analytic coverage_cont = 1.0000 for r_eff in
    # {0.035, 0.04, 0.05} on both fixtures, at a path length <= the champion's.
    # I18 `fitro` = THIS row plan + the champion trochoid loops (the composition of the two
    # best-understood pieces). Same operator, same knobs, only the row plan differs.
    if mode in ("fitted", "fitro"):
        r_eff = float(spec.get("r_eff", 0.035))
        n_rows = max(1, int(math.ceil(side / (2.0 * r_eff))))
        pitch = side / n_rows
        rows = [-side * 0.5 + side * (k + 0.5) / n_rows for k in range(n_rows)]
        uv = []
        n_line = max(2, int(2 * half / BASE_DS_M) + 1)
        for r, v in enumerate(rows):
            fwd = r % 2 == 0
            us = np_linspace(-half, half, n_line) if fwd else np_linspace(half, -half, n_line)
            uv += [(float(u), v) for u in us]
            if r < n_rows - 1:                  # C1 semicircle of radius pitch/2 to the next row
                cx, cv = (half if fwd else -half), v + pitch * 0.5
                rad = pitch * 0.5
                n_arc = 16
                for k in range(1, n_arc):
                    th = -math.pi / 2 + math.pi * k / n_arc
                    uv.append((cx + (rad * math.cos(th) if fwd else -rad * math.cos(th)),
                               cv + rad * math.sin(th)))
        if mode == "fitro":
            return _loops_offset(uv)
        return _resample_uv(uv, 0.004)
    # spiral: nested rounded rectangles, inset CELL_M per loop, centred on the patch
    cu, cv = 0.0, v0 + (nv - 1) * CELL_M * 0.5
    hu, hv = half, (nv - 1) * CELL_M * 0.5
    loops = 0
    while hv >= 0.0 and hu > 0.0 and loops < 50:
        rc = min(rad, hv) if hv > 0 else 0.0
        loop = _rounded_rect(cu, cv, hu, hv, rc)
        uv += loop
        hu -= CELL_M
        hv -= CELL_M
        loops += 1
    return _resample_uv(uv, 0.01)


def _rounded_rect(cu: float, cv: float, hu: float, hv: float, rc: float) -> list:
    """Purpose: one clockwise loop of a rectangle (half extents hu,hv) with corner radius rc.
    Degenerate hv==0 -> a straight out-and-back is avoided: returns a single line.
    Inputs: centre, half extents, corner radius. Outputs: list of (u, v).
    """
    if hv <= 1e-9:
        return [(float(u), cv) for u in np_linspace(cu - hu, cu + hu, 20)]
    pts = []
    corners = ((cu + hu - rc, cv + hv - rc, 0.0), (cu + hu - rc, cv - hv + rc, -math.pi / 2),
               (cu - hu + rc, cv - hv + rc, -math.pi), (cu - hu + rc, cv + hv - rc, math.pi / 2))
    start = (cu - hu + rc, cv + hv)
    pts.append(start)
    for (x, y, a0) in corners:
        a_start = a0 + math.pi / 2
        for k in range(9):
            a = a_start - (math.pi / 2) * k / 8
            pts.append((x + rc * math.cos(a), y + rc * math.sin(a)))
    return pts


def _close_loop_remove_dup(uv: list) -> list:
    """Purpose: FIX2 — arc-length normalize, close loop if near-duplicate at ends,
    remove duplicate consecutive points (std-lib only). Inputs: (u,v) points.
    Outputs: cleaned list.
    """
    if len(uv) < 2:
        return list(uv)
    # Remove exact consecutive duplicates
    cleaned = [uv[0]]
    for a, b in zip(uv, uv[1:]):
        if math.hypot(a[0] - b[0], a[1] - b[1]) > 1e-6:
            cleaned.append(b)
    # Close loop: if last point near first, replace last with first to close
    if len(cleaned) > 1 and math.hypot(cleaned[-1][0] - cleaned[0][0], cleaned[-1][1] - cleaned[0][1]) < 0.002:
        cleaned[-1] = cleaned[0]
    return cleaned


def _resample_uv(uv: list, ds: float) -> list:
    """Purpose: uniform-arclength resample of a (u,v) polyline (stdlib only).
    Inputs: points, spacing. Outputs: resampled points.
    """
    if len(uv) < 2:
        return list(uv)
    out = [uv[0]]
    acc = 0.0
    for (a, b) in zip(uv, uv[1:]):
        seg = math.hypot(b[0] - a[0], b[1] - a[1])
        if seg <= 0:
            continue
        t = ds - acc
        while t <= seg:
            out.append((a[0] + (b[0] - a[0]) * t / seg, a[1] + (b[1] - a[1]) * t / seg))
            t += ds
        acc = seg - (t - ds)
    if out[-1] != uv[-1]:
        out.append(uv[-1])
    return out


def _loops_offset(uv: list, rows_of: list | None = None) -> list:
    """Purpose: superimpose the champion trochoid scrub loops on a C1 base polyline.
    I18: the ONLY difference between the `trochoid` and `fitro` paths is the row plan
    (CELL_M rows vs the fitted footprint-aware pitch), so the loop operator is written
    once here and both paths call it verbatim -- no re-derivation of the champion.
    Inputs: base (u,v) polyline; optional per-point row index (I17 alternating winding,
    None = the champion's same-sense winding). Outputs: closed, arc-length-normalised path.
    """
    out, s, w = [], 0.0, 0.0
    rate = TROCHOID_DR / TROCHOID_R_M if TROCHOID_W_LEGACY else TROCHOID_W
    for i, (u, v) in enumerate(uv):
        ds = 0.0
        if i:
            ds = math.hypot(u - uv[i - 1][0], v - uv[i - 1][1])
            s += ds
        # FIX1: correct trochoid param (curtate/prolate branch, cusp continuity)
        # Run26: d/R is now explicit (TROCHOID_DR). Champion 0.5 == old s/(2R).
        # R29: amplitude A and rate w are INDEPENDENT; the coupled legacy form is
        # kept verbatim when both knobs are unset, so the champion path is unchanged.
        if TROCH_ALT <= 0.0:
            w = (s * TROCHOID_DR / TROCHOID_R_M if TROCHOID_W_LEGACY else s * TROCHOID_W)
        elif i:
            # I17: SIGNED accumulator -- the phase stays continuous and the winding
            # rate flips on odd rows. Negating the phase itself (the literal spec)
            # would jump the offset by up to 2*amp = 0.03 m at every row turn, which
            # is a teleport, not a path.
            w += rate * ds * (1.0 if (rows_of[i] if rows_of else 0) % 2 == 0 else -1.0)
        amp = TROCHOID_R_M if TROCHOID_AMP_LEGACY else TROCHOID_AMP_M
        out.append((u + amp * math.cos(w) - amp,
                    v + amp * math.sin(w)))
    # FIX2: arc-length normalize + close loop, remove duplicate points
    out = _close_loop_remove_dup(out)
    # densify enough for curvature: loop circumference ~0.094 m
    return _resample_uv(out, TROCHOID_DS_M)


def max_turn_deg(uv: list) -> float:
    """Purpose: largest heading change between consecutive segments (C1 check).
    Inputs: (u,v) points. Outputs: degrees.
    """
    worst = 0.0
    prev = None
    for a, b in zip(uv, uv[1:]):
        d = (b[0] - a[0], b[1] - a[1])
        if math.hypot(*d) < 1e-9:
            continue
        if prev is not None:
            c = (prev[0] * d[0] + prev[1] * d[1]) / (math.hypot(*prev) * math.hypot(*d))
            worst = max(worst, math.degrees(math.acos(max(-1.0, min(1.0, c)))))
        prev = d
    return worst


def uv_length(uv: list) -> float:
    """Purpose: polyline length in metres. Inputs: points. Outputs: float."""
    return sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(uv, uv[1:]))


def scrub_waypoints(spec: dict, lift: float, mode: str = "raster",
                    noise: tuple[float, float, float] = (0.0, 0.0, 0.0)) -> list[list[float]]:
    """Purpose: the ONE commanded trajectory both backends follow: hover -> scrub path
    (mode, see scrub_uv) -> rinse -> inspect, mapped from the fixture frame to world.
    `noise` = (dx, dy, dyaw_rad) planning-pose error (I5): the path is planned in a
    WRONG fixture frame while physics + scoring keep the true one.
    ponytail: an earlier diagonal path covered only 0.67 of the grid BY CONSTRUCTION, so
    the coverage metric could not respond to friction no matter how the physics moved.
    Inputs: fixture spec, tool-head half-height, path mode, pose noise. Outputs:
    ([x,y,z] list, [nx,ny,nz] world-normal list, aligned).
    """
    shape = spec.get("tank_shape", "round")
    ox, oy, oz = fixture_pose(spec)[0]
    ox, oy = ox + noise[0] + POSE_FIX_U, oy + noise[1] + POSE_FIX_V
    # scrub plane == the fixture's TOP face: round tank is 0.44 tall (centre at 0.22),
    # elongated wall-hung box is 0.24 thick (centre at 0.15, so top at 0.27).
    plane = oz + (0.12 if shape == "elongated" else 0.22) + lift
    yaw = math.radians(float(spec.get("angle_deg", 0))) + noise[2]
    ca, sa = math.cos(yaw), math.sin(yaw)

    def pt(u: float, v: float, z: float) -> list[float]:
        return [ox + u * ca - v * sa, oy + u * sa + v * ca, z]

    def wn(u: float, v: float) -> list[float]:
        # I8: surface normal in WORLD frame (fixture normal rotated by planned yaw).
        fu, fv, fz = bowl_n(spec, u, v)
        return [fu * ca - fv * sa, fu * sa + fv * ca, fz]

    uv = scrub_uv(spec, mode)
    if mode != "raster":
        turn = max_turn_deg(uv)
        assert turn <= MAX_TURN_DEG, f"{mode} path not C1: max turn {turn:.1f} deg"
    wp = [pt(0.0, 0.0, plane + bowl_z(spec, 0.0, 0.0) + 0.05)]  # spray: hover at centre
    wnorm = [wn(0.0, 0.0)]
    for (u, v) in uv:
        wp.append(pt(u, v, plane + bowl_z(spec, u, v)))
        wnorm.append(wn(u, v))
    wp.append(pt(0.0, 0.0, plane + bowl_z(spec, 0.0, 0.0) + 0.06))  # rinse: lift
    wnorm.append(wn(0.0, 0.0))
    wp.append(pt(0.0, 0.0, plane + bowl_z(spec, 0.0, 0.0) + 0.05))  # inspect
    wnorm.append(wn(0.0, 0.0))
    return wp, wnorm


def np_linspace(lo: float, hi: float, n: int):
    """Purpose: stdlib-free linspace for the raster (avoids importing numpy at module
    import time, which the pip-install header runs before). Inputs: bounds, count.
    Outputs: list of floats.
    """
    if n <= 1:
        return [lo]
    step = (hi - lo) / (n - 1)
    return [lo + i * step for i in range(n)]


def densify(wp: list[list[float]], ds: float = 0.02, arc_cols: int = 3) -> list[list[float]]:
    """Purpose: resample the sparse waypoint polyline at fixed arclength. Without this
    the servo chases 0.5 m teleports and the per-tick residual measures schedule
    coarseness, not friction. After densification the commanded step is <= ds, so the
    residual IS slip. arc_cols = leading columns used for arclength (extra columns,
    e.g. normals, are interpolated but excluded from the distance).
    Inputs: waypoints, arclength spacing. Outputs: dense waypoint list.
    """
    import numpy as np
    P = np.asarray(wp, dtype=np.float64)
    seg = np.linalg.norm(np.diff(P[:, :arc_cols], axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    total = float(s[-1])
    n = max(2, int(total / ds) + 1)
    t = np.linspace(0.0, total, n)
    out = [np.interp(t, s, P[:, k]) for k in range(P.shape[1])]
    return np.stack(out, axis=1).tolist()


def scrub_grid(spec: dict, cell: float = CELL_M) -> tuple[list[float], int, int]:
    """Purpose: the swept-area target. Cleaning succeeds when the head has TOUCHED every
    cell of the scrub patch while in contact -- not when it merely tracks the path. This
    is the metric that makes the friction sweep causal: a slipping head retraces its own
    line, leaves cells untouched, and coverage falls. Tracking error alone scored a
    sliding head as a success.
    Inputs: fixture spec, cell size. Outputs: (local origin [u, v], n_u, n_v).
    N216: the patch pair comes from `patch_extents` (the N216 dose) and the window is
    `[nu*cell, nv*cell]` -- TRUNCATED in v, because `nv = int(side/cell)` floors the
    declared side to a whole number of cells. That truncation is the metric's denominator and
    is reported in the header; the frozen scoring expression below is unchanged.
    """
    half, side = patch_extents(spec)
    return [-half, -side * 0.5], int(2 * half / cell), int(side / cell)


def phase_of(i: int, n: int) -> str:
    """Purpose: DAG phase label for step i of n (spray/scrub/rinse/inspect).
    Inputs: step index, episode length. Outputs: phase string.
    """
    f = i / max(1, n)
    if f < 0.15:
        return "spray"
    if f < 0.80:
        return "scrub"
    if f < 0.92:
        return "rinse"
    return "inspect"


def sample_friction(rng: random.Random, surface: str) -> float:
    """Purpose: draw a surface friction coefficient uniformly over the AEGIS band,
    rescaled by the finish gain and clipped back into the band.
    Inputs: RNG, surface finish. Outputs: friction coefficient in [0.05, 0.80].
    """
    lo, hi = FRICTION_RANGE
    f = rng.uniform(lo, hi) * SURFACE_FRICTION_GAIN.get(surface, 1.0)
    return round(min(hi, max(lo, f)), 4)


def compliance_mode(friction: float) -> str:
    """Purpose: name the end-effector compliance regime implied by available grip --
    the physical justification for why compliance_mode is a context token.
    Inputs: friction coefficient. Outputs: "rigid" | "compliant" | "damped".
    """
    if friction < 0.20:
        return "compliant"   # wet soap: force control, let the tool slide
    if friction > 0.55:
        return "damped"      # dry porcelain: stiff + high damping, avoid snagging
    return "rigid"


# --- quality tagging (Tier-4 jerk gate is HARD) --------------------------------
def jerk_proxy(acts) -> float:
    """Purpose: Tier-4 jerk statistic, identical formula to restroom_sim.Tier4Gate
    (mean squared second difference of the action stream).
    Inputs: sequence of action vectors. Outputs: float >= 0.
    """
    import numpy as np
    a = np.asarray(acts, dtype=np.float64)
    if len(a) < 3 or a.size < a.shape[0]:
        return 0.0
    return float(np.mean(np.diff(a, n=2, axis=0) ** 2))


def classify(success: bool, jerk: float, slip_m: float, stall_frac: float,
              friction: float, gate_intercepted: bool = False) -> tuple[str, dict]:
    """Purpose: assign the AEGIS quality_tag and its failure_metadata from the physics.
    SUCCESS wins outright; a contact-starved episode is OBSTACLE_STALL (wrong failure
    mode to train on); everything else is FAILURE_SLIP. gate_ok is the Tier-4 admission
    bit (max_jerk <= 0.618) consumed by colab_aegis_ft.split_pools. gate_intercepted
    is always False (Run 170: post-hoc tag only, no in-loop interception).
    Inputs: success flag, jerk, worst slip (m), contact-free fraction, friction, gate status.
    Outputs: (quality_tag, failure_metadata dict).
    """
    gate_ok = bool(jerk <= GATE_MAX_JERK)
    if success:
        tag = "SUCCESS"
    elif stall_frac > STALL_FRAC or (slip_m > SLIP_TOL_M and friction > 0.20):
        tag = "OBSTACLE_STALL"
    else:
        tag = "FAILURE_SLIP"
    meta = {"tag": "none" if tag == "SUCCESS" else tag,
            "max_jerk": round(jerk, 5), "gate_ok": gate_ok,
            "gate_max_jerk": GATE_MAX_JERK,
            "gate_intercepted": gate_intercepted,
            "slip_m": round(slip_m, 4), "stall_frac": round(stall_frac, 4),
            "friction": friction, "compliance_mode": compliance_mode(friction)}
    return tag, meta


# --- PyBullet scrub rig (default CPU backend) ----------------------------------
class PyBulletScrub:
    """Purpose: minimal real-contact scrub fixture. A static plane carries the swept
    friction, a static fixture body carries the Fixture-A/B geometry, and a
    force-driven tool head is commanded along `scrub_waypoints` by a velocity servo --
    the physics decides whether it tracks (grip), slides, or bounces off.
    Inputs: fixture spec, friction, tool_id, seed. Outputs: step()/metrics() API.
    ponytail: a velocity-servoed tool head, not a full Panda IK arm. The swept
    variable is surface friction under a scripted path; arm dynamics would only add
    solver cost without changing the slip/stall decision. Add a ManiSkill arm-coupled
    variant only once its friction API exists; until then it cannot sweep the axis.
    """

    # ponytail: all three heads are FLAT pads. A cylinder standing on its end rocks on
    # its rim under a lateral force and rolls out of the cell (measured: every mu), which
    # swamped slip with 0.6 m excursions. tool_id then varies the footprint and mass --
    # the axes that actually change contact pressure.
    TOOL_SHAPES = (("box", [0.05, 0.035, 0.012], 0.080),  # 0 sponge head
                   ("box", [0.04, 0.04, 0.030], 0.105),   # 1 brush head (tall, narrow)
                   ("box", [0.09, 0.05, 0.006], 0.092))  # 2 flat mop head

    def __init__(self, spec: dict, friction: float, tool_id: int, steps: int,
                 mode: str = "raster", noise: tuple = (0.0, 0.0, 0.0),
                 residual_hook=None):
        import pybullet as p
        import pybullet_data
        self.p = p
        self.spec, self.steps = spec, steps
        self.friction = friction  # set BEFORE bodies: fixture + bowl liner need it
        self.substeps = substeps_for()   # N207: read per episode so --compare-env switches rate
        self.iters = solver_iters_for()  # N208: same call-time rule, the other discretisation axis
        p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=0)
        p.setGravity(0, 0, -9.81, physicsClientId=0)
        p.setTimeStep(1.0 / SIM_HZ, physicsClientId=0)          # N207: 240 = frozen value
        p.setPhysicsEngineParameter(numSolverIterations=self.iters, physicsClientId=0)
        self.plane = p.createMultiBody(0, p.createCollisionShape(p.GEOM_PLANE),
                                       basePosition=[0, 0, 0], physicsClientId=0)
        p.changeDynamics(self.plane, -1, lateralFriction=friction,
                         restitution=0.0, physicsClientId=0)
        self.fixture = self._build_fixture(spec)
        # CRITICAL: the swept friction must land on the FIXTURE, not the floor plane. The
        # head only ever touches the fixture, so setting it on the plane alone left the
        # contact pair at the 0.5 default and the sweep was bit-identical from mu=0.05 to
        # mu=0.80 (measured). Set both -- the plane matters if the head falls.
        p.changeDynamics(self.fixture, -1, lateralFriction=friction,
                         restitution=0.0, physicsClientId=0)
        kind, half, mass = self.TOOL_SHAPES[tool_id % 3]
        self.tool_id = tool_id
        # N214: absolute footprint / mass overrides. 0.0 keeps the frozen per-tool value, so
        # the default arm is byte-identical. half[2] (the pad's THICKNESS, and hence `lift` and
        # the approach height) is never overridden -- only the in-plane extents and the mass.
        half = [half[0] if PAD_HU_M <= 0.0 else PAD_HU_M,
                half[1] if PAD_HV_M <= 0.0 else PAD_HV_M, half[2]]
        if PAD_MASS > 0.0:
            mass = PAD_MASS
        self.mass = float(mass)
        # N211: the pad's TRUE footprint half-extents, so the kernel audit can score the same
        # contacts with the rectangle the head actually has instead of its inscribed disc.
        self.pad_half = (float(half[0]), float(half[1])) if kind == "box" else None
        cid = (p.createCollisionShape(p.GEOM_CYLINDER, radius=half[0], height=half[1])
               if kind == "cyl" else
               p.createCollisionShape(p.GEOM_BOX, halfExtents=half))
        self.lift = half[2] + (half[1] * 0.5 if kind == "cyl" else 0.0)
        self.tool = p.createMultiBody(mass, cid, basePosition=[0, 0, 0.5],
                                      physicsClientId=0)
        p.changeDynamics(self.tool, -1, lateralFriction=tool_friction_for(), restitution=0.0,
                         spinningFriction=0.0, contactStiffness=CONTACT_K,
                         contactDamping=CONTACT_C, physicsClientId=0)
        spec_in = dict(spec); spec_in["r_eff"] = float(min(half[0], half[1]))
        sparse, snorm = scrub_waypoints(spec_in, self.lift, mode, noise)
        both = [list(p) + list(n) for p, n in zip(sparse, snorm)]
        dense6 = densify(both)
        self.wp = [d[:3] for d in dense6]
        self.wn = [d[3:] for d in dense6]
        # Phase labels come from PATH POSITION, not tick fraction (rig v2 fix): the old
        # phase_of(i/n) left the first ~15% of the scrub path un-pressed ("spray") and
        # the last ~12% as "rinse", capping coverage for every mode by construction.
        import numpy as _np
        _P = _np.asarray(sparse, dtype=_np.float64)
        _cum = _np.concatenate([[0.0], _np.cumsum(_np.linalg.norm(_np.diff(_P, axis=0), axis=1))])
        s_a, s_b, tot = float(_cum[1]), float(_cum[-3]), float(_cum[-1])
        _sd = _np.linspace(0.0, tot, len(self.wp))
        self.phases = ["spray" if x < s_a - 1e-9 else "scrub" if x <= s_b + 1e-9
                       else ("rinse" if x <= float(_cum[-2]) + 1e-9 else "inspect")
                       for x in _sd]
        self.total_len_m = tot
        # I13: cumulative COMMANDED-TIME table along the dense waypoint polyline (None = OFF,
        # so the frozen uniform tick -> index map is literally the original expression).
        self.ptime = self._warp_table() if SPEED_ALPHA > 0.0 else None
        # Speed normalisation (I1 confound guard): every path mode is driven at the SAME
        # commanded arclength speed as raster, so a longer path gets proportionally more
        # ticks. Otherwise "better path" and "slower head" would be indistinguishable.
        ref = uv_length(scrub_uv(spec, "raster"))
        cur = uv_length(scrub_uv(spec, mode))
        self.steps = int(round(steps * max(1.0, cur / max(ref, 1e-9))))
        self.path_mode, self.path_len_m = mode, round(cur, 4)
        # I0: tool footprint radius for continuous coverage (inscribed circle of the pad)
        self.r_eff = float(min(half[0], half[1]))
        self.mode = compliance_mode(friction)
        # I3: residual policy hook. INACTIVE (None) unless the trainer injected one, so every
        # other idea's episode is the frozen rig with the original `tgt` expression.
        self.residual = residual_hook if (residual_hook is not None and RESIDUAL_ACTIVE) else None

    def _warp_table(self) -> list:
        """Purpose: I13 cumulative commanded-time table along the dense waypoint polyline.
        Segment j is weighted by 1 + alpha*kappa_j (kappa = |d theta|/ds of the horizontal
        heading, so straight rows get kappa ~ 0 and the C1 turns get the large values), i.e.
        v(s) = v0/(1+alpha*kappa): MORE ticks on curves, fewer on straights, same total ticks.
        Curvature uses the 2-D heading, not the 3-D one: the launch failure is a LATERAL
        acceleration demand, and on a bowl the vertical bend is handled by the normal press.
        Inputs: none (uses self.wp). Outputs: increasing path-time list, len == len(self.wp).
        """
        import numpy as np
        P = np.asarray(self.wp, dtype=np.float64)
        d = np.diff(P[:, :2], axis=0)
        ds = np.linalg.norm(d, axis=1)
        th = np.unwrap(np.arctan2(d[:, 1], d[:, 0]))
        dth = np.abs(np.concatenate([[0.0], np.diff(th)]))
        kap = np.zeros(len(self.wp))
        kap[1:] = dth / np.maximum(ds, 1e-9)
        w = 1.0 + SPEED_ALPHA * kap
        dt = np.concatenate([[0.0], np.maximum(ds, 1e-9) * 0.5 * (w[1:] + w[:-1])])
        return np.cumsum(dt).tolist()

    def _warp_index(self, tick: float, n: int) -> float:
        """Purpose: fractional waypoint index chased at a given tick. OFF -> the original
        uniform map tick/(n-1)*(len(wp)-1); ON -> uniform in commanded PATH TIME (I13), so
        arclength advances slowest where curvature is highest.
        Inputs: tick (int or float), tick count. Outputs: fractional index in [0, len(wp)-1].
        """
        m = len(self.wp) - 1
        x = tick / max(1, n - 1)
        if self.ptime is None:
            return x * m
        t = x * float(self.ptime[-1])
        i = min(max(int(bisect.bisect_right(self.ptime, t)) - 1, 0), m - 1)
        dt = max(1e-12, float(self.ptime[i + 1]) - float(self.ptime[i]))
        return min(float(m), i + (t - float(self.ptime[i])) / dt)

    def _build_fixture(self, spec: dict) -> int:
        """Purpose: build the static Fixture-A/B body (round tank vs elongated
        wall-hung) with the spec's offset and yaw. I8 bowl mode adds a concave
        paraboloid heightfield liner on the top face (same frame); it carries the
        swept friction (the head touches IT, not the flat top).
        Inputs: spec. Outputs: body id (fixture base).
        """
        p = self.p
        pos, quat = fixture_pose(spec)
        fhu, fhv = face_half_extents(spec)
        if spec.get("tank_shape") == "elongated":
            # 0.68 x 0.28 m top face: wide enough that the +/-0.06 m lateral scrub
            # amplitude plus the widest tool half-width (0.05) stays ON the fixture.
            # N215: fhu/fhv are the frozen literals unless the face axis overrides them.
            cid = p.createCollisionShape(p.GEOM_BOX, halfExtents=[fhu, fhv, 0.12])
        else:
            cid = p.createCollisionShape(p.GEOM_CYLINDER, radius=fhu, height=0.44)
        base = p.createMultiBody(0, cid, basePosition=pos, baseOrientation=quat,
                                 physicsClientId=0)
        if SURFACE == "bowl":
            import numpy as _np
            # pybullet centers heightfields: vertex (i,j) sits at
            # ((i-(N-1)/2)*sx, (j-(M-1)/2)*sy) + body position, so meshScale
            # entries are per-index STEPS and the body goes at the fixture centre.
            N = M = 48
            sx, sy = 0.9 / (N - 1), 0.7 / (M - 1)
            xs = [(i - (N - 1) / 2) * sx for i in range(N)]
            ys = [(j - (M - 1) / 2) * sy for j in range(M)]
            # pybullet renders heightfield data RELATIVE TO (max+min)/2 (verified:
            # uniform data sits at base z; checker 0/0.5 spans base+-0.25; a bowl
            # grid hit exactly base+(data-mid)). So pass zero-mid data and lift base.
            raw = [float(bowl_z(spec, x, y)) for y in ys for x in xs]
            mu = (max(raw) + min(raw)) / 2.0
            data = [z - mu for z in raw]
            hcid = p.createCollisionShape(
                p.GEOM_HEIGHTFIELD, meshScale=[sx, sy, 1.0],
                heightfieldData=data, numHeightfieldRows=N,
                numHeightfieldColumns=M, physicsClientId=0)
            top = pos[2] + (0.12 if spec.get("tank_shape") == "elongated" else 0.22)
            hb = p.createMultiBody(0, hcid, basePosition=[pos[0], pos[1], top + mu],
                                   baseOrientation=quat, physicsClientId=0)
            p.changeDynamics(hb, -1, lateralFriction=self.friction,
                             restitution=0.0, contactStiffness=CONTACT_K,
                             contactDamping=CONTACT_C, physicsClientId=0)
            self.bowl_id = hb
        else:
            self.bowl_id = -1
        return base

    def run(self) -> dict:
        """Purpose: execute the scripted trajectory and measure grip vs slip vs stall.
        The head is force-driven (PD + gravity comp + press), so the contact solver --
        not the controller -- decides whether it tracks. Slip is the per-tick residual
        (moved minus commanded); lag is held ~1 mm by KP, so the residual IS slip.
        Run 170: Tier-4 jerk gate is POST-HOC TAG ONLY (fixed threshold 0.618 in
        classify(); score weight, never control). No veto, no retraction, no override.
        Inputs: none (uses ctor state). Outputs: metrics dict.
        """
        import numpy as np
        p, wp, n = self.p, self.wp, self.steps
        kp, kd = KP * KP_GAIN, KD * KP_GAIN   # N210: gain scales both terms (damping ratio fixed)
        fpos = fixture_pose(self.spec)[0]
        axis = np.asarray(fpos[:2], dtype=np.float64)
        yaw = math.radians(float(self.spec.get("angle_deg", 0)))
        ca, sa = math.cos(-yaw), math.sin(-yaw)
        org, nu, nv = scrub_grid(self.spec)
        cell = CELL_M
        p.resetBasePositionAndOrientation(self.tool, [0.0, 0.0, wp[0][2]],
                                           [0, 0, 0, 1], physicsClientId=0)
        acts, err, contacts, touched = [], [], [], set()
        contact_uv: list[tuple[float, float]] = []
        scrub_speeds: list[float] = []
        normal_f: list[float] = []
        fn_base_f: list[float] = []
        fn_bowl_f: list[float] = []
        press_n: list[float] = []
        in_tol = n_scrub = 0
        escaped = False
        z_exc_max = 0.0
        launch_ticks = 0
        # I15: integral state and the causal force feedback. fn_prev starts AT the setpoint so the
        # loop begins neutral (no initial kick from fn=0), and it is the PREVIOUS tick's measured
        # normal force -- causal, so the loop cannot use future physics.
        press_int = 0.0
        fn_prev = FN_SET_N
        # I14 containment geometry, in the TRUE fixture frame (INSET_M<=0 -> wall inert).
        # N216: the patch pair comes from the single source, so the wall cannot be built for a
        # different patch than the plan is.
        _half0, side = patch_extents(self.spec)
        self.inset_hu = max(0.0, _half0 - INSET_M)
        self.inset_hv = max(0.0, side * 0.5 - INSET_M)
        wall_ticks = 0
        wall_pen_max = 0.0
        # --- I19 advisory gate: same per-tick jerk trigger as run 167, response = slow down ----
        # force_mags: commanded force magnitudes (N) for the |2nd difference| trigger. The tool is
        # NEVER teleported (G7): the window only slows the CHASED WAYPOINT (path_progress), so the
        # head decelerates into the transient instead of being retracted off the surface.
        gate_active = is_gate_active()
        force_mags: list[float] = []
        vetoes = 0
        veto_scrub = 0          # vetoes raised while the tool is chasing the scrub patch
        veto_scrub_contact = 0  # ... and while it is actually IN contact (the only ones
                                # a response can act on: no contact, nothing to preserve)
        slow_left = 0
        # N210: the workspace rail is a hard ceiling on the TOTAL commanded force, so these two
        # counters are the only way to tell "the press asked for 25 N and got 3" (rail bound)
        # from "the press asked for 25 N and the contact pushed back" (force law). Logging only.
        f_cmd_max = 0.0
        f_clamp_ticks = 0
        slow_ticks_total = 0
        jerk_tick_max = 0.0
        # --- I20 proportional press-scale gate (continuous response, scrub phase only) ---
        # press_scale: the multiplier the PREVIOUS tick's trigger computed; applied to the
        # press one tick later, so the loop never reads its own current force. 1.0 = inert.
        press_scale, prop_scale_next = 1.0, 1.0
        prop_trig = prop_trig_scrub = prop_trig_contact = prop_ticks = 0
        prop_scale_min_seen = 1.0
        prop_scale_sum = 0.0
        j_sc_list: list[float] = []      # trigger quantity, scrub phase AND in contact
        # --- I3 residual policy: incremental fine-grid coverage tracker ----------------
        # Same kernel as _coverage_cont (fine pitch FINE_M, radius r_eff inflation), updated
        # incrementally from the physics contact of each scrub tick. It supplies the policy's
        # 5x5 LOCAL patch and its per-tick reward. The REPORTED coverage_cont is still
        # self._coverage_cont(...) over the whole contact list at return time (G7), and the two
        # are cross-checked at the end (cov_kernel_gap) so a training/eval kernel mismatch can
        # never hide.
        res = self.residual
        res_n = 0
        res_sq = 0.0
        res_cells = 0
        cy_w, sy_w = math.cos(yaw), math.sin(yaw)
        if res is not None:
            fu, fv = 2 * nu, 2 * nv
            res_ncell = fu * fv
            RG = np.stack(np.meshgrid(org[0] + (np.arange(fu) + 0.5) * FINE_M,
                                      org[1] + (np.arange(fv) + 0.5) * FINE_M,
                                      indexing="ij"), axis=-1).reshape(-1, 2)
            res_cov = np.zeros(res_ncell, dtype=bool)
            res.trace = []
        tick = 0
        path_progress = 0.0  # float arclength position along waypoint polyline
        while tick < n:
            if escaped:
                if self._phase(min(tick, n - 1), n) == "scrub":
                    n_scrub += 1
                    contacts.append(False)
                    err.append(WS_LIMIT_M)
                tick += 1
                continue
            f = self._warp_index(path_progress, n)
            lo = int(f); hi = min(int(f) + 1, len(wp) - 1)
            tgt = np.asarray(wp[lo]) * (1 - (f - lo)) + np.asarray(wp[hi]) * (f - lo)
            # I8: surface normal at the chased waypoint (flat mode: [0,0,1]).
            nrm = np.asarray(self.wn[lo]) * (1 - (f - lo)) + np.asarray(self.wn[hi]) * (f - lo)
            nrm = nrm / max(1e-9, float(np.linalg.norm(nrm)))
            ph = self._phase(min(tick, n - 1), n)
            cur = np.asarray(p.getBasePositionAndOrientation(
                self.tool, physicsClientId=0)[0], dtype=np.float64)
            if float(np.linalg.norm(cur[:2] - axis)) > WS_LIMIT_M:
                escaped = True
                tick += 1
                continue
            vel = np.asarray(p.getBaseVelocity(self.tool, physicsClientId=0)[0])
            # I3: learned residual on the COMMAND (never on the body state -- G7 no teleport).
            # (u,v) is the head's position in the TRUE fixture frame and (tu,tv) the chased
            # waypoint's, so the observation is invariant to the fixture's world SE(3) offset and
            # yaw; the inverse map du,dv -> world d is the transpose rotation by +yaw.
            pend = None
            if res is not None and ph == "scrub":
                du0, dv0 = cur[0] - axis[0], cur[1] - axis[1]
                cu_, cv_ = du0 * ca - dv0 * sa, du0 * sa + dv0 * ca
                tu_, tv_ = (tgt[0] - axis[0]) * ca - (tgt[1] - axis[1]) * sa, \
                           (tgt[0] - axis[0]) * sa + (tgt[1] - axis[1]) * ca
                vu_, vv_ = vel[0] * ca - vel[1] * sa, vel[0] * sa + vel[1] * ca
                iu = min(fu - 1, max(0, int((cu_ - org[0]) / FINE_M)))
                iv = min(fv - 1, max(0, int((cv_ - org[1]) / FINE_M)))
                patch = np.zeros(25)
                for a_ in range(5):
                    jv = iv - 2 + a_
                    if 0 <= jv < fv:
                        for b_ in range(5):
                            ju = iu - 2 + b_
                            if 0 <= ju < fu:
                                patch[a_ * 5 + b_] = 1.0 if res_cov[jv * fu + ju] else 0.0
                pend = {"eu": cu_ - tu_, "ev": cv_ - tv_, "vu": vu_, "vv": vv_,
                        "fn": fn_prev, "patch": patch, "prog": path_progress / max(1, n),
                        "friction": self.friction, "tool": self.tool_id}
                a_res = res.act(pend)
                du = max(-RESIDUAL_CLIP_M, min(RESIDUAL_CLIP_M, float(a_res[0])))
                dv = max(-RESIDUAL_CLIP_M, min(RESIDUAL_CLIP_M, float(a_res[1])))
                tgt[0] += du * cy_w - dv * sy_w
                tgt[1] += du * sy_w + dv * cy_w
                res_n += 1
                res_sq += du * du + dv * dv
            F = kp * (tgt - cur) - kd * vel
            F[2] += p.getDynamicsInfo(self.tool, -1, physicsClientId=0)[0] * 9.81
            if INSET_M > 0.0:
                # I14 soft wall: pull back along the fixture-frame axes the head has crossed.
                # (u,v) is the same TRUE-frame map the coverage grid uses, so the wall is
                # pinned to the real fixture, not to the (possibly noisy) planned one.
                du0, dv0 = cur[0] - axis[0], cur[1] - axis[1]
                cu = du0 * ca - dv0 * sa
                cv = du0 * sa + dv0 * ca
                pen_u = max(0.0, abs(cu) - self.inset_hu)
                pen_v = max(0.0, abs(cv) - self.inset_hv)
                if pen_u > 0.0 or pen_v > 0.0:
                    # rotate the fixture-frame pull back to world by +yaw
                    F[0] += WALL_KP * (np.sign(cu) * pen_u * ca + np.sign(cv) * pen_v * sa)
                    F[1] += WALL_KP * (-np.sign(cu) * pen_u * sa + np.sign(cv) * pen_v * ca)
                    wall_ticks += 1
                    wall_pen_max = max(wall_pen_max, pen_u, pen_v)
            if ph == "scrub":                          # deliberate press into the surface
                press = KP_PRESS * kp * PRESS_M        # frozen constant press (FORCE_PI=0)
                if FORCE_PI:
                    # I15: PI regulator on the MEASURED normal force (N), same units as the
                    # press, rails [0, PRESS_MAX_N] -- a scrubbing pad presses, it never pulls.
                    press_int = min(PRESS_MAX_N, max(0.0, press_int
                                                     + FN_KI * (FN_SET_N - fn_prev) * TICK_S))
                    press = min(PRESS_MAX_N, max(0.0, press + FN_KP * (FN_SET_N - fn_prev)
                                                  + press_int))
                # I20: the trigger raised LAST tick scales this tick's press. The response is
                # CONTINUOUS (no veto, no retraction, no position jump, no teleport -- G7) and
                # it is strictly causal: it uses only forces already commanded.
                press *= press_scale
                if press_scale < 1.0:
                    prop_ticks += 1
                    prop_scale_sum += press_scale
                F -= nrm * press                         # I8: along the surface normal
                press_n.append(press)
            mag = float(np.linalg.norm(F))
            f_cmd_max = max(f_cmd_max, mag)
            if mag > F_CLAMP_N:
                f_clamp_ticks += 1
                F *= F_CLAMP_N / mag
            # The trigger quantity and its calibration distribution are logged UNCONDITIONALLY
            # (force_mags is a local list: no physics reads it), so theta can be calibrated on
            # the frozen champion arm itself and every arm is audited with the same instrument.
            if len(force_mags) >= 2:
                j = abs(mag - 2.0 * force_mags[-1] + force_mags[-2])
                if ph == "scrub" and contacts and contacts[-1]:
                    j_sc_list.append(j)
                if GATE_ADVISORY:
                    jerk_tick_max = max(jerk_tick_max, j)
                    if j > GATE_MAX_JERK:
                        vetoes += 1
                        if ph == "scrub":
                            veto_scrub += 1
                            if contacts and contacts[-1]:
                                veto_scrub_contact += 1
                        slow_left = GATE_SLOW_TICKS
                elif GATE_PROP:
                    theta = GATE_THETA if GATE_TRIG else GATE_MAX_JERK
                    fired = j > theta
                    if GATE_TRIG and not (ph == "scrub" and contacts and contacts[-1]):
                        fired = False       # NEW trigger: scrub phase AND in contact
                    if fired:
                        prop_trig += 1
                        if ph == "scrub":
                            prop_trig_scrub += 1
                            if contacts and contacts[-1]:
                                prop_trig_contact += 1
                        # clip(x, lo, hi) == max(lo, min(hi, x)): GATE_PROP_MIN is a FLOOR
                        # (the pad keeps pressing), 1.0 the ceiling. Written the other way
                        # round it becomes a 0.3 CEILING and the gate stops the press dead.
                        prop_scale_next = max(GATE_PROP_MIN, min(
                            1.0, 1.0 - (j - 0.7 * theta) / (0.3 * theta)))
                        prop_scale_min_seen = min(prop_scale_min_seen, prop_scale_next)
            force_mags.append(mag)
            press_scale, prop_scale_next = prop_scale_next, 1.0
            # Normal trajectory tick: record action and simulate
            acts.append(F.copy())
            for _ in range(self.substeps):
                # pybullet clears external forces on every step -> re-apply each substep
                p.applyExternalForce(self.tool, -1, F.tolist(), [0, 0, 0],
                                      p.LINK_FRAME, physicsClientId=0)
                p.stepSimulation(physicsClientId=0)
            new = np.asarray(p.getBasePositionAndOrientation(
                self.tool, physicsClientId=0)[0], dtype=np.float64)
            if ph == "scrub":
                n_scrub += 1
                e = float(np.linalg.norm(new - tgt))
                err.append(min(e, WS_LIMIT_M))
                z_exc = float(new[2] - tgt[2])
                z_exc_max = max(z_exc_max, z_exc)
                launch_ticks += z_exc > LAUNCH_Z_M
                # I15 measurement fix: on the curved surface the head rides the HEIGHTFIELD
                # liner, not the base body, so querying the base alone under-counted the contact
                # (measured: 0 base contacts in 288 bowl scrub ticks, all of them on the liner).
                # The tool's normal force IS the sum over every body it touches. On the flat
                # surface bowl_id == -1 and this is the original query, bit-identical.
                cps = p.getContactPoints(bodyA=self.tool, bodyB=self.fixture,
                                         physicsClientId=0)
                fn_base = float(sum(c[9] for c in cps))
                fn_bowl = 0.0
                if self.bowl_id != -1:
                    cpb = p.getContactPoints(bodyA=self.tool, bodyB=self.bowl_id,
                                              physicsClientId=0)
                    fn_bowl = float(sum(c[9] for c in cpb))
                    cps = list(cps) + list(cpb)
                hit = len(cps) > 0
                fn_t = fn_base + fn_bowl
                normal_f.append(fn_t)                # contact normal force, N
                fn_base_f.append(fn_base)
                fn_bowl_f.append(fn_bowl)
                fn_prev = fn_t
                contacts.append(hit)
                in_tol += e <= SWEEP_TOL_M
                if hit:                    # mark the swept cell in the fixture frame
                    d = new[:2] - axis
                    u = d[0] * ca - d[1] * sa
                    v = d[0] * sa + d[1] * ca
                    if res is not None:     # I3: incremental fine-grid hit set
                        dd = np.linalg.norm(RG - np.array([[u, v]]), axis=1) <= self.r_eff
                        res_cells += int((~res_cov & dd).sum())
                        res_cov |= dd
                    iu = min(nu - 1, max(0, int((u - org[0]) / cell)))
                    iv = min(nv - 1, max(0, int((v - org[1]) / cell)))
                    touched.add((iu, iv))
                    contact_uv.append((float(u), float(v)))
                scrub_speeds.append(float(np.linalg.norm(
                    np.asarray(p.getBaseVelocity(self.tool, physicsClientId=0)[0])[:2])))
                if res is not None and pend is not None:
                    res.trace.append({**pend, "fn_t": fn_t, "cells": res_cells,
                                      "ncell": res_ncell, "hit": bool(hit)})
            path_progress += 1.0
            if slow_left > 0:      # I19: halve the COMMANDED arclength speed for the window
                path_progress -= 0.5 * (1.0 - GATE_SLOW_SCALE)
                slow_left -= 1
                slow_ticks_total += 1
            tick += 1
        # stall = contact-free fraction DURING SCRUB only. Counting the hover phases
        # makes stall_frac ~0.35 by construction and swamps the 0.40 discriminator.
        stall = 1.0 - (sum(contacts) / max(1, len(contacts)))
        coverage = len(touched) / max(1, nu * nv)     # swept-area, not tracking
        coverage_cont = self._coverage_cont(contact_uv, org, nu, nv)
        v_cmd = self.total_len_m / max(1e-9, n * TICK_S)   # constant arclength speed
        sp = scrub_speeds[3:] if len(scrub_speeds) > 3 else scrub_speeds
        stick_frac = (sum(1 for x in sp if x < 0.1 * v_cmd) / len(sp)) if sp else 1.0
        slip = float(np.percentile(err, 90)) if err else 0.0
        ok = (not escaped) and coverage_cont >= CLEAN_FRAC
        repair_ticks = 0
        coverage_cont_pre = round(coverage_cont, 4)
        if os.environ.get("AEGIS_REPAIR") == "1":
            raise NotImplementedError("I9 repair not implemented in the physics loop yet")
        coverage_cont = self._coverage_cont(contact_uv, org, nu, nv)
        ok = (not escaped) and coverage_cont >= CLEAN_FRAC
        # N211: diagnostic re-measurements of the SAME contacts. Appended to the return dict
        # (which run_episode splats into the record) only when the knob is on, so the default
        # arm's records are byte-identical to runs 1-359.
        kern_out: dict = {}
        if COV_KERNEL:
            kern_out = self._kernel_audit(contact_uv, org, nu, nv, escaped)
            assert kern_out["cov_k"]["frozen"] == round(coverage_cont, 4), (
                "N211: the re-measured frozen kernel must equal the scored coverage_cont")
        self.close()
        fn_arr = np.asarray(normal_f, dtype=np.float64) if normal_f else np.zeros(1)
        lo_b, hi_b = 0.5 * FN_SET_N, 1.5 * FN_SET_N
        fn_comp = float(np.mean((fn_arr >= lo_b) & (fn_arr <= hi_b))) if normal_f else 0.0
        return {"success": bool(ok), "jerk": jerk_proxy(acts), "slip_m": slip,
                "stall_frac": stall, "coverage": round(coverage, 4),
                "coverage_cont": round(coverage_cont, 4), "stick_frac": round(stick_frac, 4),
                **kern_out,
                "fn_mean": round(float(np.mean(normal_f)) if normal_f else 0.0, 4),
                "fn_p95": round(float(np.percentile(normal_f, 95)) if normal_f else 0.0, 4),
                "fn_std": round(float(np.std(fn_arr)), 4),
                "fn_set_n": FN_SET_N,
                "force_compliance": round(fn_comp, 4),
                "fn_base_mean": round(float(np.mean(fn_base_f)) if fn_base_f else 0.0, 4),
                "fn_bowl_mean": round(float(np.mean(fn_bowl_f)) if fn_bowl_f else 0.0, 4),
                "press_mean_n": round(float(np.mean(press_n)), 4) if press_n else 0.0,
                "press_max_n": round(float(np.max(press_n)), 4) if press_n else 0.0,
                "f_cmd_max_n": round(f_cmd_max, 4), "f_clamp_ticks": f_clamp_ticks,
                "f_clamp_n": F_CLAMP_N, "press_m": PRESS_M, "contact_k": CONTACT_K,
                "kp_n_per_m": round(kp, 4), "kd": round(kd, 4), "kp_gain": KP_GAIN,
                "force_pi": bool(FORCE_PI), "fn_kp": FN_KP, "fn_ki": FN_KI,
                "pts_source": "physics_contact",
                "path_mode": self.path_mode, "path_len_m": self.path_len_m,
                "track_tol_frac": round(in_tol / max(1, n_scrub), 4),
                "steps": n, "escaped": bool(escaped),
                "repair_ticks": repair_ticks, "coverage_cont_pre": coverage_cont_pre,
                "z_exc_max_m": round(z_exc_max, 4),
                "launch_frac": round(launch_ticks / max(1, n_scrub), 4),
                "speed_alpha": SPEED_ALPHA,
                "inset_m": INSET_M, "wall_only": bool(WALL_ONLY), "wall_kp": WALL_KP,
                "wall_ticks": wall_ticks, "wall_pen_max_m": round(wall_pen_max, 4),
                "residual_active": res is not None, "residual_ticks": res_n,
                "residual_rms_m": round(math.sqrt(res_sq / max(1, res_n) / 2.0), 5),
                "residual_clip_m": RESIDUAL_CLIP_M,
                "cov_kernel_gap": (round(abs(res_cells / max(1, res_ncell) - coverage_cont), 4)
                                   if res is not None else 0.0),
                "gate_intercepted": bool(gate_active and (vetoes > 0 or prop_trig > 0)),
                "gate_on": bool(gate_active), "vetoes": vetoes,
                "vetoed_ticks": slow_ticks_total,
                "veto_scrub": veto_scrub, "veto_scrub_contact": veto_scrub_contact,
                "jerk_tick_max": round(jerk_tick_max, 5),
                "gate_slow_scale": GATE_SLOW_SCALE, "gate_slow_ticks": GATE_SLOW_TICKS,
                "gate_prop": bool(GATE_PROP), "gate_trig_scrub": bool(GATE_TRIG),
                "gate_theta": GATE_THETA, "gate_prop_min": GATE_PROP_MIN,
                "prop_trig": prop_trig, "prop_trig_scrub": prop_trig_scrub,
                "prop_trig_contact": prop_trig_contact, "prop_ticks": prop_ticks,
                "prop_scale_min": round(prop_scale_min_seen, 4),
                "prop_scale_mean": round(prop_scale_sum / max(1, prop_ticks), 4),
                "j_sc_n": len(j_sc_list),
                "j_sc_p50": round(float(np.percentile(j_sc_list, 50)), 5) if j_sc_list else 0.0,
                "j_sc_p90": round(float(np.percentile(j_sc_list, 90)), 5) if j_sc_list else 0.0,
                "j_sc_p99": round(float(np.percentile(j_sc_list, 99)), 5) if j_sc_list else 0.0,
                "j_sc_max": round(float(np.max(j_sc_list)), 5) if j_sc_list else 0.0}

    def _phase(self, i: int, n: int) -> str:
        """Purpose: phase of tick i = phase of the waypoint the servo is chasing.
        Inputs: tick, tick count. Outputs: spray|scrub|rinse|inspect.
        """
        return self.phases[min(int(self._warp_index(i, n)), len(self.phases) - 1)]

    def _coverage_cont(self, pts: list, org: list, nu: int, nv: int) -> float:
        """Purpose: I0 fine-resolution coverage. Fraction of FINE_M cells (same patch as
        the coarse grid) whose centre lies within the pad footprint radius r_eff of any
        in-contact head position. Continuous-ish, so sub-cell gains are observable.
        Inputs: contact (u,v) list, grid origin, coarse counts. Outputs: [0, 1].
        """
        import numpy as np
        fu, fv = 2 * nu, 2 * nv
        if pts is None or not pts:
            return 0.0
        cu = org[0] + (np.arange(fu) + 0.5) * FINE_M
        cv = org[1] + (np.arange(fv) + 0.5) * FINE_M
        G = np.stack(np.meshgrid(cu, cv, indexing="ij"), axis=-1).reshape(-1, 2)
        P = np.asarray(pts[::2] if len(pts) > 400 else pts, dtype=np.float64)
        hit = np.zeros(len(G), dtype=bool)
        for k in range(0, len(P), 256):
            d = np.linalg.norm(G[:, None, :] - P[None, k:k + 256, :], axis=-1)
            hit |= (d <= self.r_eff).any(axis=1)
        return float(hit.mean())

    def _coverage_kernel(self, pts: list, org: list, nu: int, nv: int, pitch_m: float,
                         foot: str = "disc", stride: int = 1) -> float:
        """Purpose: N211 -- RE-MEASURE the same physics contact set under an alternative
        scoring kernel. Diagnostic only: the scored `coverage_cont`/`success` come from
        `_coverage_cont`, which is left verbatim.
        Inputs: contact (u,v) list, grid origin, coarse counts, grid pitch (m), footprint
        model ("disc" = the frozen inscribed radius r_eff, "rect" = the pad's true half-extents,
        "cell" = no footprint inflation, a contact marks its own cell), contact stride.
        Outputs: [0, 1] fraction of grid-cell centres inside the footprint of a kept contact.
        ponytail: the frozen expression stays duplicated on purpose -- sharing one body would put
        the scored metric one refactor away from the audit it is auditing.
        """
        import numpy as np
        if pts is None or not pts:
            return 0.0
        fu = max(1, int(round(nu * CELL_M / pitch_m)))
        fv = max(1, int(round(nv * CELL_M / pitch_m)))
        cu = org[0] + (np.arange(fu) + 0.5) * pitch_m
        cv = org[1] + (np.arange(fv) + 0.5) * pitch_m
        G = np.stack(np.meshgrid(cu, cv, indexing="ij"), axis=-1).reshape(-1, 2)
        P = np.asarray(pts[::max(1, int(stride))], dtype=np.float64)
        if foot == "rect" and self.pad_half is not None:
            hu, hv = self.pad_half
        else:
            hu = hv = self.r_eff
        hit = np.zeros(len(G), dtype=bool)
        for k in range(0, len(P), 256):
            d = np.abs(G[:, None, :] - P[None, k:k + 256, :])
            if foot == "rect" and self.pad_half is not None:
                hit |= ((d[:, :, 0] <= hu) & (d[:, :, 1] <= hv)).any(axis=1)
            elif foot == "cell":
                hit |= ((d[:, :, 0] <= 0.5 * pitch_m) &
                        (d[:, :, 1] <= 0.5 * pitch_m)).any(axis=1)
            else:
                hit |= (np.linalg.norm(d, axis=-1) <= hu).any(axis=1)
        return float(hit.mean())

    def _coverage_declared(self, pts: list, pitch_m: float = FINE_M) -> float:
        """Purpose: N216 -- RE-MEASURE the same physics contacts over the DECLARED patch
        (the full [half, side] rect) instead of the scored window `[nu*CELL_M, nv*CELL_M]`,
        which `nv = int(side/CELL_M)` truncates. Diagnostic only: the scored
        `coverage_cont`/`success` come from `_coverage_cont`, left verbatim (G7).
        Inputs: contact (u,v) list, grid pitch (m). Outputs: [0, 1].
        """
        import numpy as np
        if pts is None or not pts:
            return 0.0
        half, side = patch_extents(self.spec)
        fu = max(1, int(round(2 * half / pitch_m)))
        fv = max(1, int(round(side / pitch_m)))
        cu = -half + (np.arange(fu) + 0.5) * pitch_m
        cv = -side * 0.5 + (np.arange(fv) + 0.5) * pitch_m
        G = np.stack(np.meshgrid(cu, cv, indexing="ij"), axis=-1).reshape(-1, 2)
        P = np.asarray(pts, dtype=np.float64)
        hit = np.zeros(len(G), dtype=bool)
        for k in range(0, len(P), 256):
            d = np.linalg.norm(G[:, None, :] - P[None, k:k + 256, :], axis=-1)
            hit |= (d <= self.r_eff).any(axis=1)
        return float(hit.mean())

    def _kernel_audit(self, contact_uv: list, org: list, nu: int, nv: int,
                      escaped: bool) -> dict:
        """Purpose: N211 -- the full kernel sweep for one episode, from the SAME contacts.
        Inputs: contact (u,v) list, grid origin, coarse counts, escape flag.
        Outputs: {"cov_k": {kernel: coverage}, "succ_k": {kernel: success}, ...} with the
        frozen kernel first and its coverage equal to the scored one by construction.
        Every kernel is scored with the same `coverage >= CLEAN_FRAC` rule the rig uses, so a
        verdict flip is a measured disagreement between definitions, not arithmetic on the
        reported metric.
        """
        kern = {
            "frozen": (FINE_M, "disc", 2),          # the scored one: pitch, inscribed disc,
            "allcontacts": (FINE_M, "disc", 1),     # stride 2 above 400 contacts, else 1
            "rect": (FINE_M, "rect", 1),            # the pad's TRUE rectangular footprint
            "p2": (0.5 * FINE_M, "disc", 1),        # grid-pitch convergence
            "p4": (0.25 * FINE_M, "disc", 1),
            "nodilat": (FINE_M, "cell", 1),         # no footprint inflation: pure path cover
            "faithful": (0.5 * FINE_M, "rect", 1),  # true footprint, all contacts, finer grid
            "declared": (FINE_M, "disc", 1),        # N216: the FULL DECLARED patch, not the
                                                    # truncated scored window [nu,nv]*CELL_M
        }
        cov, succ = {}, {}
        for name, (pitch, foot, stride) in kern.items():
            st = stride if len(contact_uv) > 400 and name == "frozen" else 1
            if name == "frozen":
                c = self._coverage_cont(contact_uv, org, nu, nv)
            elif name == "declared":
                c = self._coverage_declared(contact_uv, pitch)
            else:
                c = self._coverage_kernel(contact_uv, org, nu, nv, pitch, foot, st)
            cov[name] = round(c, 4)
            succ[name] = bool((not escaped) and c >= CLEAN_FRAC)
        return {"cov_k": cov, "succ_k": succ, "n_contacts": len(contact_uv),
                "n_contacts_scored": len(contact_uv[::2]) if len(contact_uv) > 400
                else len(contact_uv),
                "r_eff_m": self.r_eff, "pad_half_m": list(self.pad_half) if self.pad_half
                else None}

    def close(self) -> None:
        """Purpose: tear down the DIRECT client (leaking it wedges the next episode).
        Inputs: none. Outputs: None.
        """
        try:
            self.p.disconnect(physicsClientId=0)
        except Exception:  # noqa: BLE001
            pass


# --- streaming ------------------------------------------------------------------
def emit(fh, rec: dict) -> dict:
    """Purpose: write one JSONL record and flush it (a kernel that dies at minute 40
    must still leave a valid partial sweep on disk).
    Inputs: open file handle, record dict. Outputs: the record (timestamped).
    """
    rec = {"ts": round(time.time() - T0, 3), **rec}
    fh.write(json.dumps(rec, default=float) + "\n")
    fh.flush()
    return rec


def sample_customer_fixture(rng: random.Random) -> dict:
    """Purpose: fixture_R -- a held-out "new customer" drawn from the AEGIS variation
    envelope (shape, finish, +-15 cm x/y placement, +-10 deg yaw). Zero-shot test set:
    NEVER tune on it; report it. Inputs: RNG. Outputs: fixture spec dict.
    """
    return {"tank_shape": rng.choice(["round", "elongated"]),
            "surface": rng.choice(["glossy", "matte"]),
            "offset_cm": round(rng.uniform(-15, 15), 2),
            "offset_y_cm": round(rng.uniform(-15, 15), 2),
            "angle_deg": round(rng.uniform(-10, 10), 2)}


def pose_noise(rng: random.Random) -> tuple[float, float, float]:
    """Purpose: I5 planning-pose error from AEGIS_POSE_NOISE="sigma_t_m,sigma_yaw_deg".
    Inputs: RNG. Outputs: (dx, dy, dyaw_rad).
    """
    st, sr = (float(x) for x in POSE_NOISE.split(","))
    if st <= 0 and sr <= 0:
        return (0.0, 0.0, 0.0)
    return (rng.gauss(0, st), rng.gauss(0, st), math.radians(rng.gauss(0, sr)))


def run_episode(suite: str, spec: dict, seed: int, backend: str,
                mode: str = "raster") -> dict:
    """Purpose: one full fixture episode: build world, sweep friction, roll the scripted
    controller, tag quality, return a record. A harness failure is recorded as a
    HARNESS-ERROR row (excluded from aggregates) instead of aborting the sweep, so a
    7h kernel never dies on one bad episode.
    Inputs: suite name, fixture spec, seed, backend name. Outputs: record dict.
    """
    rng = random.Random(seed)
    if suite == "fixture_R":                     # new-customer draw (seeded)
        spec = sample_customer_fixture(random.Random(seed * 7919 + 1))
    friction = sample_friction(rng, spec.get("surface", "glossy"))
    noise = pose_noise(random.Random(seed * 104729 + 3))
    tool_id = seed % 3
    off, rpy = fixture_offset(spec), fixture_rpy(spec)
    base = {"suite": suite, "seed": seed, "friction": friction, "tool_id": tool_id,
            "se3_offset_xyz": off, "se3_offset_rpy": rpy,
            # N209: the friction the CONTACT actually saw. Bullet multiplies the two bodies'
            # coefficients, so `friction` alone has never been the realized value. Logged (not
            # scored): every friction-binned claim in runs 1-357 must be re-read on this axis.
            "friction_realized": round(tool_friction_for() * friction, 6),
            "compliance_mode": compliance_mode(friction), "path_mode": mode,
            "pose_noise": [round(x, 5) for x in noise], "pose_noise_cfg": POSE_NOISE,
            "fixture_spec": spec}
    t_ep = time.time()
    hook = RESIDUAL_HOOK_FN() if (RESIDUAL_ACTIVE and RESIDUAL_HOOK_FN) else None
    try:
        rig = PyBulletScrub(spec, friction, tool_id, T_MAX, mode, noise, residual_hook=hook)
        assert abs(rig.friction - friction) < 1e-9, "friction not applied to contact pair"
        # --- I9 depth-registration (min patch,G7-compliant; no teleport, no arithmetic coverage) ---
        if REG_MODE == "depth":
            import numpy as np, math
            p = rig.p
            # true fixture centre from spec
            true_pos, _ = fixture_pose(spec)
            true_yaw = math.radians(float(spec.get("angle_deg", 0)))
            true_pert = (noise[0], noise[1], noise[2])  # the error we must estimate
            # ray grid(s) from 0.8 m above the NOISY planned centre (approximated by noise+true).
            # The ray SEGMENT must bracket the top face: from well above to BELOW it
            # (an earlier version ended above the face, so rays never intersected).
            cx = true_pos[0] + noise[0]; cy = true_pos[1] + noise[1]
            z_hi, z_lo = true_pos[2] + 0.8, true_pos[2] - 0.05
            # --- N192: the SENSOR, not the estimator. One +-H window CONTAINS the top face
            # only while the planning error is inside the slack a = H - rho_inf (N190.3); past
            # that the missing crescent biases the centroid, and widening H alone is a trade
            # (H=0.9 loses to H=0.6, N190.6). So cast a k x k LATTICE of +-H windows whose
            # UNION covers the plausible-centre set (half-extent 3 sigma, the shipped
            # convention) and keep only the ONE cast that best contains the face. That cast is
            # truncation-free, so the frozen mean estimator and the frozen PCA yaw are reused
            # verbatim and the ray pitch stays at 2H/(n-1). WHICH cast contains the face is
            # read off the top-point COUNT (maximal exactly when the window holds the whole
            # face); ties go to the cast nearest the prior, so k=1 is bit-identical to shipped.
            rho_inf = math.hypot(0.34, 0.14)   # +-inf half-extent of the top face, worst yaw
            a_slack = max(REG_HALF_M - rho_inf, 1e-3)
            sig_t = float(POSE_NOISE.split(",")[0])
            # N195.1 the guarantee is only that SOME lattice offset lands inside the containment
            # set C(f) = f + [-a,a]^2 (the offsets whose window holds the WHOLE top face), so
            # L(d) cap C(f) != {} for every |e|_inf <= E iff  d/2 <= a  AND  (k-1)d/2 >= E.
            # N192 fixed d = 1.5a and then derived k from a alone (k = ceil(4 sigma/a)+1), which
            # is a d-LAW: at fixed E = 3 sigma the two conditions fix k = ceil(2E/d)+1, so the
            # shipped k is only the minimum for d = 1.5a and 1.78x the ray count of the tightest
            # legal pitch d = 2a. Deriving k from d makes the budget an explicit knob and leaves
            # the shipped expression EXACT: ceil(2*3 sigma/(1.5a)) + 1 = ceil(4 sigma/a) + 1.
            # k: REG_CASTS=0 derives it from the pose-noise scale, else the value is forced.
            d = max(REG_DFACT * a_slack, 1e-3)   # lattice pitch; covering radius d/2 <= a_slack
            k = (int(math.ceil(6.0 * sig_t / d)) + 1 if (REG_CASTS == 0 and sig_t > a_slack)
                 else max(int(REG_CASTS), 1))
            cast_offs = [((i - (k - 1) / 2.0) * d, (j - (k - 1) / 2.0) * d)
                         for i in range(k) for j in range(k)]

            def _cast(ox, oy, nn):
                """One +-H ray grid at offset (ox,oy); return its top-face points."""
                # N196: --compare-env writes every knob as a FLOAT (apply_knobs), and numpy
                # >= 1.20 rejects a float `num` in linspace -> the whole baseline arm died with
                # "TypeError: 'float' object cannot be interpreted as an integer". Coerce once
                # here; REG_N itself is unchanged, so the default arm is untouched.
                nn = int(nn)
                xs = np.linspace(cx - REG_HALF_M + ox, cx + REG_HALF_M + ox, nn)
                ys = np.linspace(cy - REG_HALF_M + oy, cy + REG_HALF_M + oy, nn)
                X, Y = np.meshgrid(xs, ys)
                from_xyz = np.stack([X.ravel(), Y.ravel(), np.full(X.size, float(z_hi))],
                                    axis=-1).tolist()
                to_xyz = np.stack([X.ravel(), Y.ravel(), np.full(X.size, float(z_lo))],
                                  axis=-1).tolist()
                res = []
                for _i in range(0, len(from_xyz), 1024):  # pybullet caps one batch at 1024
                    res.extend(p.rayTestBatch(from_xyz[_i:_i + 1024], to_xyz[_i:_i + 1024],
                                              physicsClientId=0))
                # pybullet rayTestBatch format here: (objId, frac, 1.0, (hx,hy,hz), (nx,ny,nz))
                # FIX: exclude the tool head (it sits ABOVE the fixture and poisons the
                # max-z mode) and the floor plane (vast, dominates any mode statistic).
                hits = [(i, r[3][0], r[3][1], r[3][2]) for i, r in enumerate(res)
                        if r[0] != -1 and r[0] != rig.tool and r[0] != rig.plane]
                z_vals = [h[3] for h in hits]
                # keep points within 1 cm of max-height mode (top face)
                return ([(h[1], h[2], h[3]) for h in hits if abs(h[3] - max(z_vals)) < 0.01]
                        if z_vals else [])

            # N195.2 coarse-to-fine: the k^2 casts exist ONLY to answer a k^2-way argmax, but
            # N192 cast all of them at the FULL n x n resolution, so the sensor paid k^2 * n^2
            # rays to keep one cast's points. Two changes make the stage cheap AND safe:
            # (a) COARSE-TO-FINE: run the argmax on an n_c x n_c grid over the SAME window extent
            #     and cast the winner once at full n. cn >= n makes stage 1 identical to the
            #     shipped single-resolution loop, so the dose is a strict special case and
            #     REG_CN=0 is the frozen expression verbatim.
            # (b) BORDER-MARGIN score instead of the point count. The count is only a proxy for
            #     the intersection area and ALIASES on a coarse grid: measured at 0.64,128, the
            #     count argmax at n_c=8 (pitch 171 mm > a_slack 232 mm) picked a CUT window and
            #     doubled reg_err_xy (43.2 vs 1.9 mm) while at n_c=16 the count margin was 0 on
            #     1 of 3 seeds, i.e. the coarse grid could not separate the candidates at all.
            #     The border margin is the quantity the containment condition actually constrains:
            #     a window holds the WHOLE face iff min over the four sides of the gap between the
            #     point bbox and the window border is >= a_slack, and that gap is a DISTANCE, so
            #     it survives a coarse grid exactly as long as the coarse pitch < a_slack.
            cn = min(int(REG_CN), REG_N) if REG_CN > 0 else REG_N
            coarse_margin = None
            best_off, top_pts, cast_npts = (0.0, 0.0), [], []
            if k == 1 or cn >= REG_N:
                for (ox, oy) in cast_offs:
                    cast_pts = _cast(ox, oy, REG_N)
                    cast_npts.append(len(cast_pts))
                    if len(cast_pts) > len(top_pts):   # strict: ties keep prior-nearest
                        top_pts, best_off = cast_pts, (ox, oy)
            else:
                def _border_margin(pts, ox, oy):
                    """Smallest gap between the point bbox and the window border (metres)."""
                    if not pts:
                        return -1.0
                    x_lo, x_hi = cx - REG_HALF_M + ox, cx + REG_HALF_M + ox
                    y_lo, y_hi = cy - REG_HALF_M + oy, cy + REG_HALF_M + oy
                    xs_ = [q[0] for q in pts]
                    ys_ = [q[1] for q in pts]
                    return min(min(xs_) - x_lo, x_hi - max(xs_),
                               min(ys_) - y_lo, y_hi - max(ys_))
                scored = [_border_margin(_cast(ox, oy, cn), ox, oy)
                          for (ox, oy) in cast_offs]
                sel = max(range(len(cast_offs)), key=lambda i: (scored[i], -i))
                # Falsifier readout, in metres: the WINNER'S OWN border margin. A containing
                # window has margin >= a_slack by construction (N195.1), so any winner with
                # margin < 0 kept a CUT window and reg_err_xy must show it. The gap to the
                # runner-up is the wrong statistic: at d = 2a two lattice offsets can be
                # equidistant from the containment square and BOTH contain the face.
                coarse_margin = round(scored[sel], 6)
                best_off = cast_offs[sel]
                top_pts = _cast(best_off[0], best_off[1], REG_N)
            reg_ok = len(top_pts) >= 30
            if reg_ok:
                pts = np.array(top_pts)
                est = pts[:, :2].mean(axis=0)
                if REG_EST == "extent":
                    # N190: the estimator, not the resolution, is the floor. The mean of a
                    # ray grid that TRUNCATES the top face is biased by the missing crescent,
                    # and the bias grows with the planning-pose error that moved the window
                    # (measured: p90 14.6 mm at 1 cm/2 deg -> 118.8 mm at 16 cm/32 deg, while
                    # 32 -> 128 -> 181 rays are indistinguishable). The extent midpoint in the
                    # PRIOR-YAW frame is unbiased under truncation of a CONVEX top face, and
                    # the prior enters only as the measurement frame -- exactly the role the
                    # frozen PCA already gives it for its pi-ambiguity branch. 0 = the frozen
                    # mean, so the default rig is unchanged.
                    _c, _s = math.cos(-(true_yaw + true_pert[2])), math.sin(-(true_yaw + true_pert[2]))
                    _R = np.array([[_c, -_s], [_s, _c]])
                    _q = pts[:, :2] @ _R.T
                    est = (0.5 * (_q.max(axis=0) + _q.min(axis=0))) @ _R
                # N191 (run 302): a THIRD estimator was built and MEASURED, then deleted. The
                # opposite-interior-support midpoint is exact per world axis whenever both
                # supports of that axis lie inside the ray window (the window is axis-aligned
                # and known, so a hit on its boundary is a bound of the WINDOW, not of the
                # face) and falls back to the mean per axis when truncated. It is unbiased and
                # it LOSES: 0.28 m/56 deg, 100 paired seeds, B transfer_success 0.82 vs 0.84
                # and coverage_cont 0.9125 vs 0.9153 (Welch p 0.928, Fisher p 0.851,
                # keep=false); at 0.03 m/6 deg it is strictly WORSE, B coverage_cont 0.9933 vs
                # 1.0000 (Welch p 8.28e-05) and reg_err_xy median 11.63 vs 7.13 mm. Mechanism:
                # the mean of a symmetric ray grid is already unbiased whenever the window
                # CONTAINS the face, while a 2-point extremal estimator carries the full ray
                # pitch (2*0.6/31 = 38.7 mm) of independent noise on each support, ~13.7 mm at
                # the midpoint. Unbiasedness does not beat a 120-point average. Not shipped.
                if spec.get("tank_shape") == "elongated" and pts.shape[0] > 2:
                    # yaw from PCA of top points; elongated box has a clear major axis
                    cov = np.cov(pts[:, 0], pts[:, 1])
                    vals, vecs = np.linalg.eigh(cov)
                    if vals[1] < 1e-9 or vals[0] / max(vals[1], 1e-12) > 0.8:
                        yaw_est = None  # near-circular spread: yaw unobservable
                    else:
                        axis = vecs[:, 1] if vals[1] >= vals[0] else vecs[:, 0]
                        yaw_est = math.atan2(axis[1], axis[0])
                        # PCA axis has pi ambiguity: pick the branch nearer the prior
                        prior = true_yaw + true_pert[2]
                        while yaw_est - prior > math.pi / 2:
                            yaw_est -= math.pi
                        while yaw_est - prior < -math.pi / 2:
                            yaw_est += math.pi
                else:
                    yaw_est = None  # round fixture: rotation-symmetric, yaw irrelevant
                reg_err_xy = float(np.linalg.norm(est - np.array([true_pos[0], true_pos[1]])))
                if yaw_est is None:
                    dyaw_corr, reg_err_yaw = true_pert[2], 0.0
                else:
                    dyaw_corr = yaw_est - true_yaw
                    while dyaw_corr > math.pi:
                        dyaw_corr -= 2 * math.pi
                    while dyaw_corr < -math.pi:
                        dyaw_corr += 2 * math.pi
                    reg_err_yaw = math.degrees(abs(dyaw_corr - true_pert[2]))
                # N191: reg_err_yaw_deg above is |correction - original|, i.e. the AMOUNT OF
                # YAW ERROR REMOVED, not the error the plan is built with -- it is large
                # exactly when the estimator works, and it is 0.0 by construction whenever the
                # PCA test declares the yaw unobservable and the raw prior is kept. The honest
                # readout is |dyaw_corr| = |yaw_est - true_yaw|, which is the plan's yaw error
                # after correction. reg_yaw_plan_deg supersedes it; both are logged (G7).
                # N194: the scrub patch is a RECT, so the plan's coverage is periodic in the
                # yaw with period 180 deg and only the FOLDED residual is physical. Unfolded,
                # this field reads p90 = 179.5 deg at EVERY pose-noise level (run 305) because
                # the PCA pi-branch legitimately lands 180 deg away half the time -- a large
                # number here is not a large plan error. Fold to [0, 90]; strictly a LOG fix,
                # noise (and therefore every contact/coverage number) is untouched.
                reg_yaw_plan = abs(math.degrees(dyaw_corr) % 180.0)
                reg_yaw_plan = min(reg_yaw_plan, 180.0 - reg_yaw_plan)
                noise = (float(est[0] - true_pos[0]), float(est[1] - true_pos[1]), dyaw_corr)
                # rebuild rig with corrected noise (no teleport; same fixtures)
                p.removeBody(rig.fixture, physicsClientId=0)
                p.removeBody(rig.tool, physicsClientId=0)
                p.removeBody(rig.plane, physicsClientId=0)
                rig = PyBulletScrub(spec, friction, tool_id, T_MAX, mode, noise,
                                    residual_hook=hook)
                # inject log fields via attribute for output
                rig._reg_ok = True; rig._reg_err_xy = reg_err_xy; rig._reg_err_yaw = reg_err_yaw
                rig._reg_yaw_plan = reg_yaw_plan
                rig._reg_casts = len(cast_offs); rig._reg_cast_win = best_off
                rig._reg_cast_pts = len(top_pts); rig._reg_cast_npts = list(cast_npts)
                rig._reg_cn_margin = coarse_margin
            else:
                rig._reg_ok = False; rig._reg_err_xy = float('nan'); rig._reg_err_yaw = float('nan')
                rig._reg_yaw_plan = float('nan')
                rig._reg_casts = len(cast_offs); rig._reg_cast_win = best_off
                rig._reg_cast_pts = len(top_pts); rig._reg_cast_npts = list(cast_npts)
                rig._reg_cn_margin = coarse_margin
        # --- end I9 ---
        out = rig.run()
        tag, meta = classify(out["success"], out["jerk"], out["slip_m"],
                             out["stall_frac"], friction, out.get("gate_intercepted", False))
        return {**base, **out, "record": "episode", "backend": "pybullet",
                "quality_tag": tag, "failure_metadata": meta,
                "wall_s": round(time.time() - t_ep, 3), "status": "PHYSICAL-VALIDATION",
                **({"reg_ok": getattr(rig,"_reg_ok",None),
                    "reg_err_xy_m": getattr(rig,"_reg_err_xy",None),
                    "reg_err_yaw_deg": getattr(rig,"_reg_err_yaw",None),
                    "reg_yaw_plan_deg": getattr(rig,"_reg_yaw_plan",None),
                    "reg_casts": getattr(rig,"_reg_casts",None),
                    "reg_cast_win_m": list(getattr(rig,"_reg_cast_win",(0.0,0.0))),
                    "reg_cast_pts": getattr(rig,"_reg_cast_pts",None),
                    "reg_cn_margin": getattr(rig,"_reg_cn_margin",None)}
                   if REG_MODE=="depth" else {})}
    except Exception as exc:  # noqa: BLE001
        log(f"episode failed {suite}/seed{seed}: {type(exc).__name__}: {str(exc)[:140]}")
        return {**base, "record": "episode", "backend": "none", "quality_tag":
                "OBSTACLE_STALL", "failure_metadata": {"tag": "HARNESS", "max_jerk": 0.0,
                "gate_ok": False, "error": f"{type(exc).__name__}: {str(exc)[:200]}"},
                "success": False, "jerk": 0.0, "slip_m": 0.0, "stall_frac": 1.0,
                "coverage": 0.0, "coverage_cont": 0.0, "stick_frac": 1.0,
                "steps": 0, "wall_s": round(time.time() - t_ep, 3),
                "status": "HARNESS-ERROR (excluded from aggregates)"}


def project(seeds: int, times: list[float], done: int) -> tuple[int, float]:
    """Purpose: quota-aware projection. After PROBE_EPS timed episodes, linearly
    extrapolate; if the projection would breach BUDGET_H, shrink seeds-per-suite to
    the largest count that fits (floor 3) so the kernel exits clean under 8h.
    Inputs: seeds/suite, per-episode wall times, episodes done. Outputs: (seeds, est_h).
    """
    per = (sum(times) / max(1, len(times))) if times else 1.0
    est_h = per * 2.0 * seeds / 3600.0
    if est_h > BUDGET_H:
        scaled = max(3, int(seeds * BUDGET_H / est_h))
        est_h = per * 2.0 * scaled / 3600.0
        log(f"quota guard: {est_h:.2f}h > {BUDGET_H}h -> seeds/suite {seeds} -> {scaled}")
        seeds = scaled
    return seeds, est_h


# --- exports --------------------------------------------------------------------
def hub_push(paths: list[str]) -> dict:
    """Purpose: push the sweep artifacts to the HF dataset repo, skipping silently
    when HF_TOKEN is absent or rejected (401) so a 7h kernel never dies on auth.
    Inputs: list of local file paths. Outputs: status dict (uploaded | skipped | error).
    """
    tok = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not tok:
        return {"status": "skipped", "why": "no HF_TOKEN"}
    try:
        from huggingface_hub import HfApi
        api = HfApi(token=tok)
        api.create_repo(HF_REPO, repo_type="dataset", exist_ok=True, private=True)
        up = []
        for pth in paths:
            if os.path.exists(pth):
                api.upload_file(path_or_fileobj=pth, path_in_repo=os.path.basename(pth),
                                repo_id=HF_REPO, repo_type="dataset")
                up.append(os.path.basename(pth))
        return {"status": "uploaded", "files": up, "url": f"https://huggingface.co/datasets/{HF_REPO}"}
    except Exception as exc:  # noqa: BLE001
        s = f"{type(exc).__name__}: {str(exc)[:200]}"
        if "401" in s or "Unauthorized" in s or "Bad credentials" in s:
            log(f"HF 401 -> graceful skip ({s})")
            return {"status": "skipped", "why": "401"}
        log(f"HF upload failed: {s}")
        return {"status": "error", "why": s}


def summarize(records: list[dict], meta: dict) -> dict:
    """Purpose: aggregate the sweep the way the AEGIS directive reads it: Fixture-B
    zero-shot transfer_success over >=20 seeds vs the scripted baseline 0.8125, the
    keep-bar, the per-tag counts, and the gate's interception delta.
    Inputs: episode records, run meta. Outputs: summary dict (also written as JSONL).
    """
    def agg(rows):
        n = len(rows)
        s = [r for r in rows if r["quality_tag"] == "SUCCESS"]
        gated = [r for r in rows if r["quality_tag"] != "SUCCESS"
                 and r["failure_metadata"].get("gate_ok")]
        return {"n": n, "transfer_success": round(len(s) / n, 4) if n else 0.0,
                "success_count": len(s), "gate_admitted_failures": len(gated),
                "mean_jerk": round(sum(r["jerk"] for r in rows) / n, 5) if n else 0.0,
                "mean_coverage": round(sum(r["coverage"] for r in rows) / n, 4) if n else 0.0,
                "mean_coverage_cont": round(_mean([r["coverage_cont"] for r in rows]), 4),
                "std_coverage_cont": round(_std([r["coverage_cont"] for r in rows]), 4),
                "mean_stick_frac": round(_mean([r.get("stick_frac", 1.0) for r in rows]), 4),
                "mean_slip_m": round(sum(r["slip_m"] for r in rows) / n, 4) if n else 0.0,
                "escaped_frac": round(sum(bool(r.get("escaped")) for r in rows) / n, 4) if n else 0.0,
                "mean_launch_frac": round(_mean([r.get("launch_frac", 0.0) for r in rows]), 4),
                "mean_z_exc_max_m": round(_mean([r.get("z_exc_max_m", 0.0) for r in rows]), 4),
                "mean_fn_mean": round(_mean([r.get("fn_mean", 0.0) for r in rows]), 4),
                "mean_fn_std": round(_mean([r.get("fn_std", 0.0) for r in rows]), 4),
                "mean_fn_base": round(_mean([r.get("fn_base_mean", 0.0) for r in rows]), 4),
                "mean_fn_bowl": round(_mean([r.get("fn_bowl_mean", 0.0) for r in rows]), 4),
                "mean_force_compliance": round(_mean([r.get("force_compliance", 0.0) for r in rows]), 4),
                "mean_press_n": round(_mean([r.get("press_mean_n", 0.0) for r in rows]), 4),
                "force_pi": bool(rows[0].get("force_pi", False)) if rows else False,
                "residual_active": bool(rows[0].get("residual_active", False)) if rows else False,
                "residual_rms_m": _mean([r.get("residual_rms_m", 0.0) for r in rows]),
                "cov_kernel_gap": _mean([r.get("cov_kernel_gap", 0.0) for r in rows]),
                "speed_alpha": rows[0].get("speed_alpha", 0.0) if rows else 0.0}

    good = [r for r in records if str(r.get("status", "")).startswith("PHYSICAL")]
    per = {s: agg([r for r in good if r.get("suite") == s])
           for s in sorted({r.get("suite") for r in records if r.get("suite")})}
    tags = {}
    for r in good:            # harness errors are NOT quality tags -- keep them separate
        tags[r["quality_tag"]] = tags.get(r["quality_tag"], 0) + 1
    n_err = len(records) - len(good)
    fb = per.get("fixture_B", {}).get("transfer_success", 0.0)
    n_b = per.get("fixture_B", {}).get("n", 0)
    keep = fb > SCRIPTED_QUALITY_GATE and n_b >= 20
    if not good:
        verdict = "HARNESS-FAILURE (no physical episodes)"
    elif n_b < 20:
        verdict = "PENDING-VALIDATION (need >=20 seeds/suite)"
    else:
        verdict = "KEEP" if keep else "DISCARD+REVERT"
    return {"record": "summary", "run": meta, "per_suite": per, "tag_counts": tags,
            "harness_errors": n_err,
            "scripted_baseline_score": SCRIPTED_BASELINE_SCORE,
            "fixture_B_keep_bar_met": bool(keep),
            "gate_max_jerk": GATE_MAX_JERK,
            "verdict": verdict, "status": "PHYSICAL-VALIDATION"}


def _mean(x: list) -> float:
    return sum(x) / len(x) if x else 0.0


def _std(x: list) -> float:
    if len(x) < 2:
        return 0.0
    m = _mean(x)
    return (sum((a - m) ** 2 for a in x) / (len(x) - 1)) ** 0.5


def welch(a: list, b: list) -> dict:
    """Purpose: Welch's t-test (unequal variance) b vs a, two-sided p, plus the paired
    t-test on seed-aligned pairs (same seeds -> same friction/tool). scipy if present,
    else a normal approximation (flagged). Inputs: two samples. Outputs: stats dict.
    """
    out = {"n_a": len(a), "n_b": len(b), "mean_a": round(_mean(a), 4),
           "mean_b": round(_mean(b), 4), "delta": round(_mean(b) - _mean(a), 4)}
    try:
        from scipy import stats
        w = stats.ttest_ind(b, a, equal_var=False)
        out["welch_t"], out["welch_p"] = round(float(w.statistic), 4), float(w.pvalue)
        if len(a) == len(b) and len(a) > 1:
            pr = stats.ttest_rel(b, a)
            out["paired_t"], out["paired_p"] = round(float(pr.statistic), 4), float(pr.pvalue)
        out["p_method"] = "scipy"
    except Exception:  # noqa: BLE001
        va, vb = _std(a) ** 2, _std(b) ** 2
        se = ((va / max(1, len(a))) + (vb / max(1, len(b)))) ** 0.5
        t = (out["delta"] / se) if se > 0 else 0.0
        out["welch_t"] = round(t, 4)
        out["welch_p"] = math.erfc(abs(t) / math.sqrt(2))
        out["p_method"] = "normal-approx (scipy missing)"
    return out


def plot(rows: list[dict]) -> str | None:
    """Purpose: optional grip-curve figure (success vs friction, split by fixture).
    Matplotlib Agg only, never fatal -- a missing font or backend must not cost 7h.
    Inputs: records. Outputs: figure path or None.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 4))
        for suite, mk in (("fixture_A", "o"), ("fixture_B", "s")):
            rs = [r for r in rows if r.get("suite") == suite and r.get("friction")]
            if not rs:
                continue
            ax.scatter([r["friction"] for r in rs],
                       [1.0 if r["quality_tag"] == "SUCCESS" else 0.0 for r in rs],
                       marker=mk, s=18, label=suite, alpha=0.75)
        ax.axvline(GATE_MAX_JERK, ls=":", c="k", lw=1)
        ax.set_xlabel("friction (0.05 wet .. 0.80 dry)")
        ax.set_ylabel("episode success")
        ax.set_title(f"AEGIS restroom sweep (gate max_jerk={GATE_MAX_JERK})")
        ax.legend()
        fig.tight_layout()
        fig.savefig(FIG_PATH, dpi=110)
        plt.close(fig)
        return FIG_PATH
    except Exception as exc:  # noqa: BLE001
        log(f"figure skipped ({type(exc).__name__}: {str(exc)[:120]})")
        return None


# --- main -----------------------------------------------------------------------
def main() -> int:
    """Purpose: orchestrate the whole sweep -- deps, backend probe, quota projection,
    per-episode streaming, summary, optional figure + Hub push. Clean exit under 8h.
    Inputs: CLI args / env. Outputs: process exit code (0 ok, 3 no physics backend).
    """
    global T_MAX  # noqa: PLW0603 -- argparse overrides the module default once
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=N_SEEDS)
    ap.add_argument("--steps", type=int, default=T_MAX)
    ap.add_argument("--backend", default=FORCE_BACKEND, choices=["pybullet"])
    ap.add_argument("--no-upload", action="store_true")
    ap.add_argument("--path", default=PATH_MODE, choices=list(PATH_MODES),
                    help="I1 scrub path mode (candidate)")
    ap.add_argument("--compare", default="", choices=[""] + list(PATH_MODES),
                    help="also run this BASELINE mode on the SAME seeds and emit a "
                         "Welch/paired test record (G4)")
    ap.add_argument("--suites", default="fixture_A,fixture_B",
                    help="comma list; fixture_R = held-out random new-customer draws")
    ap.add_argument("--out", default="", help="JSONL path (default results/aegis_sweep.jsonl)")
    ap.add_argument("--compare-env", default="",
                    help="comma list KEY=FLOAT applied to the --compare BASELINE arm ONLY "
                         "(e.g. AEGIS_TROCH_R_M=0.015). R29: pairs a knob variant against "
                         "the champion knob set on the SAME seeds (G4) -- the path mode is "
                         "identical in both arms, so --compare alone cannot express it.")
    args = ap.parse_args()
    T_MAX = args.steps
    global JSONL_PATH, FIG_PATH  # noqa: PLW0603
    if args.out:
        JSONL_PATH = args.out
        FIG_PATH = os.path.splitext(args.out)[0] + ".png"
    suites = [x for x in args.suites.split(",") if x]
    for sname in suites:
        if sname not in FIXTURES and sname != "fixture_R":
            log(f"unknown suite {sname!r}")
            return 2

    os.makedirs(WORKDIR, exist_ok=True)
    ensure_deps()
    backend = args.backend
    import importlib
    try:                       # honest probe: the module must really import
        importlib.import_module("pybullet")
    except Exception as exc:   # noqa: BLE001
        log(f"pybullet unavailable ({type(exc).__name__}) -- AEGIS forbids synthetic "
            f"rows standing in for physical validation, refusing to run")
        return 3
    avail = {m: bool(importlib.util.find_spec(m))
             for m in ("pybullet", "mujoco", "mani_skill", "torch")}
    log(f"backend={backend} seeds/suite={args.seeds} steps={T_MAX} "
        f"budget={BUDGET_H}h hard_cap={HARD_CAP_H}h jsonl={JSONL_PATH}")

    seeds = args.seeds
    records: list[dict] = []
    times: list[float] = []
    est_h = 0.0
    with open(JSONL_PATH, "w") as fh:
        emit(fh, {"record": "header", "status": "PHYSICAL-VALIDATION",
                  "backend": backend, "seeds_requested": seeds, "steps": T_MAX,
                  "fixtures": FIXTURES, "friction_range": FRICTION_RANGE,
                   "gate_max_jerk": GATE_MAX_JERK,
                    "jerk_definition": "mean squared 2nd difference of the commanded force "
                                       "vector per control tick (jerk_proxy). Gate is a POST-HOC "
                                       "tag only (Run 170): fixed threshold 0.618 in classify(), "
                                       "score weight, never control. No veto, no retraction.",
"gate_mode": ("proportional" if GATE_PROP else "advisory" if GATE_ADVISORY else "post-hoc"),
                    "contact_override": False,
                    "gate_prop_theta": GATE_THETA, "gate_prop_min": GATE_PROP_MIN,
                    "gate_trig_scrub": bool(GATE_TRIG),
                    "aegis_reg": REG_MODE, "aegis_reg_half_m": REG_HALF_M,
                    "aegis_margin_m": MARGIN_M,
                    "aegis_reg_n": REG_N, "aegis_reg_est": REG_EST,
                    "aegis_reg_casts": REG_CASTS,
                    "calibration_jsonl": os.environ.get("AEGIS_CALIBRATION_JSONL", ""),
                  "fields": list(FIELDS), "path_mode": args.path, "compare": args.compare,
                  "compare_env": args.compare_env,
                  "suites": suites, "pose_noise_cfg": POSE_NOISE, "rig_version": 2,
                  # N207/N208: the discretisation in force, in the header (G7 -- a claim
                  # about force_compliance is meaningless without the solver it was measured at)
                  "sim_hz": SIM_HZ, "substeps": substeps_for(),
                  "solver_iters": solver_iters_for(),
                  "troch_dr": TROCHOID_DR, "troch_R_m": TROCHOID_R_M,
                  "troch_ds_m": TROCHOID_DS_M, "base_ds_m": BASE_DS_M,
                  "troch_amp_m": TROCHOID_AMP_M, "troch_w_rad_per_m": TROCHOID_W,
                  "troch_amp_legacy": TROCHOID_AMP_LEGACY, "troch_w_legacy": TROCHOID_W_LEGACY,
                  "troch_wA_dimensionless": round(TROCHOID_W * TROCHOID_AMP_M, 6),
                  "surface": SURFACE, "bowl_k_m": BOWL_K, "speed_alpha": SPEED_ALPHA,
                  "launch_z_m": LAUNCH_Z_M,
                  "force_pi": bool(FORCE_PI), "fn_set_n": FN_SET_N, "fn_kp": FN_KP,
                  "fn_ki": FN_KI, "press_max_n": PRESS_MAX_N,
                  # N210: the normal-force axis, in the header next to the solver (G7 -- a
                  # claim about force is meaningless without the force it was measured at)
                  "press_m": PRESS_M, "press_n_frozen": KP_PRESS * KP * PRESS_M,
                  "f_clamp_n": F_CLAMP_N, "contact_k": CONTACT_K, "contact_c": CONTACT_C,
                  "kp_n_per_m": KP * KP_GAIN, "kp_gain": KP_GAIN,
                  # N211: the scoring kernel in the header beside the solver -- a coverage
                  # claim is meaningless without the kernel that produced it
                  "cov_kernel_on": bool(COV_KERNEL), "fine_m": FINE_M,
                  "clean_frac": CLEAN_FRAC,
                  "frozen_kernel": "fine pitch 0.025 m, isotropic disc r_eff = min(hu,hv), "
                                   "contact stride 2 above 400 contacts",
                  # N214: the tool body in force -- a coverage claim is meaningless without the
                  # footprint it was measured with, exactly as the kernel sits beside the solver
                  "pad_hu_m": PAD_HU_M, "pad_hv_m": PAD_HV_M, "pad_mass_kg": PAD_MASS,
                  "tool_shapes_frozen": "0 sponge 0.05x0.035x0.012 0.080 kg | "
                                        "1 brush 0.04x0.04x0.030 0.105 kg | "
                                        "2 mop 0.09x0.05x0.006 0.092 kg (tool_id = seed % 3)",
                  "pad_hu_realized": [round(s[1][0] if PAD_HU_M <= 0.0 else PAD_HU_M, 4)
                                      for s in PyBulletScrub.TOOL_SHAPES],
                  "pad_hv_realized": [round(s[1][1] if PAD_HV_M <= 0.0 else PAD_HV_M, 4)
                                      for s in PyBulletScrub.TOOL_SHAPES],
                  "pad_mass_realized": [round(PAD_MASS if PAD_MASS > 0.0 else s[2], 4)
                                        for s in PyBulletScrub.TOOL_SHAPES],
                  "pad_r_eff_realized": [round(min(s[1][0] if PAD_HU_M <= 0.0 else PAD_HU_M,
                                                  s[1][1] if PAD_HV_M <= 0.0 else PAD_HV_M), 4)
                                         for s in PyBulletScrub.TOOL_SHAPES],
                  # N215: the fixture face in force -- a transfer claim is meaningless without
                  # the face it was measured on, beside the head and the kernel
                  "face_hu_m": FACE_HU_M, "face_hv_m": FACE_HV_M, "face_r_m": FACE_R_M,
                  "face_realized": {"round": list(face_half_extents(
                      {"tank_shape": "round"})),
                      "elongated": list(face_half_extents(
                          {"tank_shape": "elongated"}))},
                  "face_frozen": "round cylinder r=0.32 (h 0.44) | elongated box "
                                 "halfExtents [0.34, 0.14, 0.12]; scrub patch fixed at "
                                 "half=0.20 side=0.18/0.12, so the head runs OFF the face "
                                 "once the face shrinks below patch + r_eff",
                   "inset_m": INSET_M, "wall_only": bool(WALL_ONLY), "wall_kp": WALL_KP,
                  # N216: the SCORED WINDOW in the header beside the kernel -- a coverage
                  # number is meaningless without the fraction it is a fraction OF
                  "patch_hu_m": PATCH_HU_M, "patch_side_m": PATCH_SIDE_M,
                  "patch_frozen": "half_u 0.20 m, side_v 0.12 m (elongated) / 0.18 m "
                                  "(round), cell pitch CELL_M 0.05 m",
                  "patch_realized": patch_report(),
                  "patch_v_truncation_frac": {k: v["v_truncation_frac"]
                                              for k, v in patch_report().items()},
                  "residual_active": bool(RESIDUAL_ACTIVE), "residual_clip_m": RESIDUAL_CLIP_M,
                   "residual_policy": os.environ.get("AEGIS_RESIDUAL_W", ""),

                  "contact_bodies": "fixture+bowl_liner" if SURFACE == "bowl" else "fixture",
                  "scripted_baseline_score": SCRIPTED_BASELINE_SCORE,
                  "keep_bar": SCRIPTED_QUALITY_GATE, "quota_budget_h": BUDGET_H,
                  "torch_cuda": _cuda(), "deps_available": avail,
                  "backend_note": "pybullet only: maniskill 3.0.1 PickCube-v1 has no "
                                  "actor friction API and no public mjModel handle, so "
                                  "the swept mu cannot reach its contact pair",
                  "python": sys.version.split()[0],
                  "time_budget_s": BUDGET_H * 3600,
                  "provenance": {"argv": sys.argv, "cwd": os.getcwd(),
                                 "git_sha": _git_sha(), "argv0": sys.argv[0]}})
        # labels must stay distinct even when the baseline shares the candidate's PATH MODE
        # and differs only in a geometry knob (R29), so by_mode cannot key on the mode name.
        if args.compare and args.compare_env:
            modes = [(args.path, False), (f"{args.compare}[{args.compare_env}]", True)]
        elif args.compare and args.compare != args.path:
            modes = [(args.path, False), (args.compare, True)]
        else:
            modes = [(args.path, False)]
        by_mode: dict = {label: [] for label, _ in modes}
        cand_knobs = knob_snapshot()
        base_knobs = knob_snapshot()
        # N212: the baseline arm's --compare-env REWRITES T_MAX below, so meta["steps"] would
        # report the baseline dose on both summaries. Pin the candidate dose here; each arm's
        # own realised tick count is in its episode records ("steps": n) and each arm's dose
        # is in the label and in the compare record's *_knobs, so nothing is mislabelled.
        cand_steps = T_MAX
        for mode, is_base in modes:
            knob_out = apply_knobs(args.compare_env) if (is_base and args.compare_env) else knob_snapshot()
            if is_base:
                base_knobs = {**knob_snapshot(), **knob_out}
            for si, suite in enumerate(suites):
                spec = FIXTURES.get(suite, {})
                for k in range(seeds):
                    if (time.time() - T0) > BUDGET_H * 3600:
                        log("budget exhausted -> clean stop (results already flushed)")
                        break
                    # seed depends only on (suite index, k): identical across path modes
                    # -> paired comparison (same friction, tool, noise, customer draw).
                    seed = BASE_SEED + 1000 * (si * seeds) + k
                    rec = run_episode(suite, spec, seed, backend,
                                      args.compare if is_base else args.path)
                    records.append(emit(fh, rec))
                    by_mode[mode].append(rec)
                    times.append(rec["wall_s"])
                    log(f"{mode}/{suite} seed={rec['seed']} f={rec['friction']:.2f} "
                        f"tool={rec['tool_id']} tag={rec['quality_tag']} "
                        f"cov={rec['coverage']:.2f} covc={rec.get('coverage_cont', 0):.3f} "
                        f"stick={rec.get('stick_frac', 1):.2f} slip={rec['slip_m']:.3f} "
                        f"{rec['wall_s']:.2f}s")
                    if len(times) == PROBE_EPS:
                        seeds, est_h = project(seeds, times, len(times))
                        log(f"time estimate: {est_h:.2f}h ({sum(times) / len(times):.2f}s/ep)")
        meta = {"backend": backend, "seeds_per_suite": seeds, "episodes": len(records),
                "steps": cand_steps, "candidate_steps": cand_steps, "elapsed_h": round((time.time() - T0) / 3600, 3),
                "est_h_after_probe": round(est_h, 3), "budget_h": BUDGET_H,
                "hard_cap_h": HARD_CAP_H, "seeds_required": 20,
                "jsonl": JSONL_PATH, "jsonl_bytes": os.path.getsize(JSONL_PATH)}
        for mode, _ in modes:
            emit(fh, {**summarize(by_mode[mode], {**meta, "path_mode": mode}),
                      "record": "summary_mode", "path_mode": mode})
        if len(modes) == 2:
            cand, base_m = modes[0][0], modes[1][0]
            cmp = {"record": "compare", "candidate": cand, "baseline": base_m,
                   "candidate_knobs": cand_knobs, "baseline_knobs": base_knobs,
                   "pose_noise_cfg": POSE_NOISE,
                   "metric": "coverage_cont", "keep_rule": ">=20 seeds AND B transfer_success > 0.70 AND (Welch p<0.01 coverage OR Fisher p<0.01 success) AND no coverage regression"}
            for suite in suites:
                a = [r["coverage_cont"] for r in by_mode[base_m]
                     if r.get("suite") == suite and str(r.get("status", "")).startswith("PHYSICAL")]
                b = [r["coverage_cont"] for r in by_mode[cand]
                     if r.get("suite") == suite and str(r.get("status", "")).startswith("PHYSICAL")]
                cmp[suite] = welch(a, b)
                sa = [bool(r["success"]) for r in by_mode[base_m] if r.get("suite") == suite
                      and str(r.get("status", "")).startswith("PHYSICAL")]
                sb = [bool(r["success"]) for r in by_mode[cand] if r.get("suite") == suite
                      and str(r.get("status", "")).startswith("PHYSICAL")]
                cmp[suite]["succ_a"], cmp[suite]["succ_b"] = sum(sa), sum(sb)
                try:  # success is binary -> Fisher exact on the 2x2 table
                    from scipy.stats import fisher_exact
                    cmp[suite]["fisher_p"] = float(fisher_exact(
                        [[sum(sb), len(sb) - sum(sb)], [sum(sa), len(sa) - sum(sa)]])[1])
                except Exception:  # noqa: BLE001
                    cmp[suite]["fisher_p"] = None
            fbc = cmp.get("fixture_B", {})
            # keep (G2/G4): >=20 seeds AND B transfer_success > 0.70 AND a significant
            # improvement over the paired baseline (coverage Welch p<0.01 OR success
            # Fisher p<0.01), never a regression in mean coverage.
            n_b = fbc.get("n_b", 0)
            ts_b = (fbc.get("succ_b", 0) / n_b) if n_b else 0.0
            sig = (fbc.get("welch_p", 1.0) < 0.01 and fbc.get("delta", 0) > 0) or \
                  ((fbc.get("fisher_p") or 1.0) < 0.01 and fbc.get("succ_b", 0) > fbc.get("succ_a", 0))
            cmp["keep"] = bool(n_b >= 20 and ts_b > SCRIPTED_QUALITY_GATE and sig
                               and fbc.get("delta", -1) >= -0.005)
            emit(fh, cmp)
            log("COMPARE " + json.dumps(cmp))
        summ = emit(fh, summarize(by_mode[modes[0][0]], {**meta, "path_mode": modes[0][0]}))
        log(json.dumps({k: summ[k] for k in ("per_suite", "tag_counts",
                                              "harness_errors", "verdict")}, indent=1))
        log(f"wrote {len(records)} episodes -> {JSONL_PATH} "
            f"({meta['jsonl_bytes'] / 1024:.1f} KiB) in {meta['elapsed_h']:.2f}h")

    arts = [JSONL_PATH]
    fig = plot(records)
    if fig:
        arts.append(fig)
    if not args.no_upload and UPLOAD:
        log(f"hub: {hub_push(arts)}")
    else:
        log("hub: skipped (--no-upload / AEGIS_UPLOAD=0)")
    log("AEGIS_SWEEP_DONE")
    return 0


def _git_sha() -> str | None:
    """Purpose: short git sha of the working tree, for run provenance.
    Inputs: none. Outputs: 8-char sha, or None if unavailable (not a repo, no git).
    """
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=10)
        return out.stdout.strip() or None
    except Exception:  # noqa: BLE001
        return None


def _cuda() -> bool:
    """Purpose: record whether the run used the GPU (affects quota accounting).
    Inputs: none. Outputs: bool.
    """
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:  # noqa: BLE001
        return False


if __name__ == "__main__":
    raise SystemExit(main())
