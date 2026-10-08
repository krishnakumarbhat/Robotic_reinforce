"""I12 (run 246) force-compliance cause probe: WHY is fc 0.478 on round and 1.000 on elongated?

Purpose: the 200-seed matrix shows force_compliance separated perfectly by `tank_shape`
(round 0.478, elongated 1.000, 400/400 episodes, zero overlap). The strategy graph called
this an "I11 mass/stiffness question". This probe reads the per-tick normal force of ONE
episode of each shape from the rig's own class, importing the canonical module (no edit to
it, no re-implementation of the physics) and adding no scoring. It answers: is the round
fixture's shortfall a CONTACT-COUNT artefact (one contact vs four, so each carries more
force) or a genuine pressure excess?
Inputs: none. Outputs: per-tick fn summary + contact-count summary for round vs elongated.
"""

from __future__ import annotations

import importlib.util
import json
import statistics
import sys


def load_rig() -> object:
    """Purpose: import the canonical rig module by path without executing its main().
    Inputs: none. Outputs: the module object."""
    path = "experiments/kaggle_aegis_sweep.py"
    spec = importlib.util.spec_from_file_location("aegis_rig", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["aegis_rig"] = mod
    spec.loader.exec_module(mod)  # noqa: E402
    return mod


def main() -> int:
    """Purpose: run one probe episode per tank shape with the per-tick fn captured."""
    mod = load_rig()
    import numpy as np
    import pybullet as p

    rows = []
    for shape, spec in (
        ("round", {"tank_shape": "round", "surface": "glossy",
                   "offset_cm": 0, "angle_deg": 0, "friction": 0.30}),
        ("elongated", {"tank_shape": "elongated", "surface": "matte",
                       "offset_cm": 15, "angle_deg": 10, "friction": 0.30}),
    ):
        for tool_id in (0, 1, 2):
            scrub = mod.PyBulletScrub(spec, 0.30, tool_id, 400, "trochoid")
            # re-run the loop with the per-tick fn captured by monkeypatching the list
            fns: list[float] = []
            ncontacts: list[int] = []
            real_close = scrub.close

            # capture via the contact query the rig already performs
            real_gcp = p.getContactPoints

            def spy(bodyA=None, bodyB=None, **kw):  # noqa: ANN001, ANN202
                cps = real_gcp(bodyA=bodyA, bodyB=bodyB, **kw)
                if bodyB is not None:
                    f = float(sum(c[9] for c in cps))
                    ncontacts.append(len(cps))
                    fns.append(f)
                return cps

            p.getContactPoints = spy  # type: ignore[assignment]
            try:
                out = scrub.run()
            finally:
                p.getContactPoints = real_gcp  # type: ignore[assignment]
                real_close()
            arr = np.asarray(fns)
            band_lo, band_hi = 0.5 * mod.FN_SET_N, 1.5 * mod.FN_SET_N
            in_band = float(np.mean((arr >= band_lo) & (arr <= band_hi))) if arr.size else 0.0
            rows.append({
                "shape": shape, "tool_id": tool_id,
                "n_ticks_captured": int(arr.size),
                "fn_mean": round(float(arr.mean()), 4) if arr.size else None,
                "fn_p50": round(float(np.percentile(arr, 50)), 4) if arr.size else None,
                "fn_p95": round(float(np.percentile(arr, 95)), 4) if arr.size else None,
                "fn_max": round(float(arr.max()), 4) if arr.size else None,
                "in_band_frac_recomputed": round(in_band, 4),
                "rig_force_compliance": out.get("force_compliance"),
                "mean_contact_points": round(statistics.fmean(ncontacts), 3) if ncontacts else None,
                "mode_contact_points": max(set(ncontacts), key=ncontacts.count) if ncontacts else None,
                "frac_ticks_zero_contact": round(
                    sum(1 for x in ncontacts if x == 0) / len(ncontacts), 4) if ncontacts else None,
                "r_eff": round(scrub.r_eff, 4),
                "tool_half": list(mod.PyBulletScrub.TOOL_SHAPES[tool_id % 3][1]),
            })
            print(f"{shape:10s} tool{tool_id} n={rows[-1]['n_ticks_captured']:4d} "
                  f"fn_mean {rows[-1]['fn_mean']} p95 {rows[-1]['fn_p95']} "
                  f"max {rows[-1]['fn_max']} inband(recomp) {in_band:.4f} "
                  f"rig_fc {out.get('force_compliance')} "
                  f"contacts(mean/mode) {rows[-1]['mean_contact_points']}/"
                  f"{rows[-1]['mode_contact_points']} "
                  f"zero-contact-frac {rows[-1]['frac_ticks_zero_contact']} "
                  f"r_eff {rows[-1]['r_eff']}")
    print("\nI12_FC_PROBE_JSON")
    print(json.dumps(rows, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
