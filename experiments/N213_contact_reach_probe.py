#!/usr/bin/env python3
"""N213 probe: how far does a pad's contact cloud actually reach, per axis, per head?

The pre-registered closed form in equations.md ROW N213 assumed the covered reach per scrub row
is `+-2 hy` in v and `+-2 hx` in u (box-on-plane contacts at the pad corners, dilated by
r_eff = min(hx, hy)). The ladder falsified the ABSOLUTE scale of that assumption while keeping
its anisotropy, so the reach has to be MEASURED rather than re-derived. This probe wraps
`_coverage_cont` (the rig file is NOT edited) to capture the physics contact cloud of one
0,0-offset episode per head and reports the reach against the scored patch.

Purpose: measure the per-axis, per-head coverage reach of the frozen rig.
Inputs: none (env-pinned to the frozen 0,0 arm, system python3, PyBullet DIRECT).
Outputs: printed table + assertions that the reach is ordered by the pad geometry.
"""
import os
import sys

os.environ.setdefault("AEGIS_POSE_NOISE", "0,0")
os.environ.setdefault("AEGIS_POSE_FIX_U", "0")
os.environ.setdefault("AEGIS_POSE_FIX_V", "0")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "benchmarks"))

import importlib.util  # noqa: E402

_SPEC = importlib.util.spec_from_file_location(
    "rig", os.path.join(os.path.dirname(os.path.abspath(__file__)), "kaggle_aegis_sweep.py"))
assert _SPEC is not None and _SPEC.loader is not None
rig = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(rig)

CAPTURED: list = []
_orig = rig.PyBulletScrub._coverage_cont


def spy(self, pts, org, nu, nv):
    """Capture the contact cloud the scored kernel is handed, then score it unchanged."""
    CAPTURED.append([tuple(p) for p in pts])
    return _orig(self, pts, org, nu, nv)


rig.PyBulletScrub._coverage_cont = spy

rows = []
for tool_id in (0, 1, 2):
    for shape in ("round", "elongated"):
        side = 0.18 if shape == "round" else 0.12
        kind, half, mass = rig.PyBulletScrub.TOOL_SHAPES[tool_id]
        specd = {"tank_shape": shape, "surface": "matte", "offset_cm": 0, "angle_deg": 0}
        CAPTURED.clear()
        r = rig.PyBulletScrub(specd, 0.5, tool_id, 400, "trochoid", (0.0, 0.0, 0.0))
        rec = r.run()
        pts = CAPTURED[-1]
        us = [p[0] for p in pts]
        vs = [p[1] for p in pts]
        rows.append((tool_id, shape, half[0], half[1], min(us), max(us), min(vs), max(vs),
                     -side / 2, side / 2, rec["coverage_cont"]))
        print(f"tool {tool_id} {shape:9s} pad hx={half[0]:.3f} hy={half[1]:.3f} | "
              f"contact u [{min(us):+.4f},{max(us):+.4f}] v [{min(vs):+.4f},{max(vs):+.4f}] | "
              f"patch u [-0.2000,+0.2000] v [{-side/2:+.4f},{side/2:+.4f}] | "
              f"v reach beyond patch {(-side/2 - min(vs)) * 1e3:+.1f} / "
              f"{(max(vs) - side/2) * 1e3:+.1f} mm | covc {rec['coverage_cont']:.4f} "
              f"n_contact {len(pts)}")

# --- asserts: the mechanism the falsified closed form needed, measured instead -------------
# 1. the u reach is symmetric about 0 and exceeds the patch half-width (the plan is 0.40 m + arcs)
for tool_id, shape, hx, hy, umin, umax, vmin, vmax, lo, hi, _covc in rows:
    assert abs(umax + umin) < 5e-3, f"u reach not centred: {umin} {umax}"
    assert umax > 0.20, f"u reach inside the patch on tool {tool_id} {shape}: {umax}"
# 2. the v reach is ASYMMETRIC: the plan band is anchored at v0 = -side/2, so the far (+v) edge
#    has strictly less slack than the near edge, on both shapes and all three heads.
for tool_id, shape, hx, hy, umin, umax, vmin, vmax, lo, hi, _covc in rows:
    assert (hi - vmax) < (vmin - lo), f"v slack not tighter at +v: tool {tool_id} {shape}"
# 3. the v reach is what separates the heads, and it is ordered by hy (the narrow pad is the
#    brittle one) -- the sign the closed form got backwards.
vreach = {(t, s): hi - vmax for t, s, hx, hy, umin, umax, vmin, vmax, lo, hi, _c in rows}
for shape in ("round", "elongated"):
    a, b, c = (vreach[(t, shape)] for t in (0, 1, 2))
    assert a < b < c, f"v reach not ordered by pad hy on {shape}: {a} {b} {c}"
print("\nN213.4 ASSERTIONS PASS: u reach symmetric and wider than the patch, v reach "
      "asymmetric with +v the tight sign on every head and shape, v reach ordered by hy.")
