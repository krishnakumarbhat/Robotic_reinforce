"""N206 self-check — the over-coverage margin is geometric, not decorative.

Purpose:  `scrub_grid` re-uses `scrub_uv`'s half/side, so the scored patch IS the planned
          rect. A margin is only a mechanism if (a) the swept band actually contains the
          patch for a plan offset up to the margin, (b) the in-patch rows keep their phase
          (re-anchoring to the inflated side would shift the row grid and lose fine
          coverage), and (c) MARGIN_M=0 reproduces the frozen plan bit for bit.
Inputs:   none (imports the rig module and inspects the plan geometry).
Outputs:  0 on success; an AssertionError naming the broken invariant otherwise.
"""
from __future__ import annotations

import importlib.util
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]


def load_rig():
    """Purpose: import the sweep rig as a module. Inputs: none. Outputs: the module."""
    spec = importlib.util.spec_from_file_location(
        "rig_under_test", ROOT / "experiments" / "kaggle_aegis_sweep.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["rig_under_test"] = mod
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    """Purpose: assert the three margin invariants. Inputs: none. Outputs: exit code."""
    rig = load_rig()
    specs = {"elongated": {"tank_shape": "elongated"}, "round": {"tank_shape": "round"}}

    for name, spec in specs.items():
        side = 0.12 if name == "elongated" else 0.18
        patch = (-0.20, 0.20, -side / 2, side / 2)

        rig.MARGIN_M = 0.0
        frozen = {m: rig.scrub_uv(spec, m) for m in ("rounded", "trochoid", "raster")}

        rig.MARGIN_M = 0.05
        for m in ("rounded", "trochoid", "raster"):
            plan = rig.scrub_uv(spec, m)
            us = [u for u, _ in plan]
            vs = [v for _, v in plan]
            # (a) symmetric containment: the swept band brackets the patch shifted by +-margin
            for e in (0.0, 0.05, -0.05):
                assert min(us) <= patch[0] + e + 1e-9, (name, m, "u_lo", min(us), e)
                assert max(us) >= patch[1] + e - 1e-9, (name, m, "u_hi", max(us), e)
                assert min(vs) <= patch[2] + e + 1e-9, (name, m, "v_lo", min(vs), e)
                assert max(vs) >= patch[3] + e - 1e-9, (name, m, "v_hi", max(vs), e)
            # (b) the in-patch rows keep their original phase and the added ones are OUTSIDE
            base_rows = [-side / 2 + (r + 0.5) * rig.CELL_M
                         for r in range(max(1, int(side / rig.CELL_M)))]
            lows = [v for v in vs if v < patch[2] - 1e-9]
            highs = [v for v in vs if v > patch[3] + 1e-9]
            assert lows and highs, (name, m, "no margin rows added")
            assert min(lows) < patch[2] and max(highs) > patch[3]
            # every in-patch row of the frozen plan is still swept by the margined plan.
            # `trochoid` displaces every point by the loop offset, so the exact row value is
            # not a sample; its span invariant is (a) plus the length growth asserted below.
            if m != "trochoid":
                for r in base_rows:
                    assert any(abs(v - r) < 1e-9 for v in vs), (name, m, "row lost", r)
            # (c) OFF is the frozen plan
            rig.MARGIN_M = 0.0
            assert rig.scrub_uv(spec, m) == frozen[m], (name, m, "margin 0 not frozen")
            rig.MARGIN_M = 0.05
            assert len(plan) > len(frozen[m]), (name, m, "margin added no geometry")

    # the margin must be a no-op when the plan is centred, i.e. it never changes the patch
    rig.MARGIN_M = 0.0
    print("OK: margin contains the patch for |e| <= MARGIN_M, keeps the in-patch row phase,")
    print("    adds geometry, and MARGIN_M=0 is bit-identical to the frozen plan.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
