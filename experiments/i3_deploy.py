"""I3 deploy wrapper — run the canonical rig with a residual policy injected through
RESIDUAL_HOOK_FN, so the I3 decider is the RIG'S OWN paired --compare record (G4).

WHY A WRAPPER AND NOT A RIG EDIT: `kaggle_aegis_sweep.RESIDUAL_HOOK_FN` is the documented
injection point (rig lines 1347 / 882). Setting it here leaves the rig file untouched, so
every other idea's episode is still the frozen rig, and the flag pair
`--compare trochoid --compare-env AEGIS_RESIDUAL_ACTIVE=0.0` disables the residual on the
baseline arm only (same 20 seeds -> same friction / tool / customer / pose noise).

MODES (`--i3-mode`, the first positional-free flag):
  npz    : evaluate the PPO weights exported by experiments/run_i3_ppo.py (DECIDER arm)
  null   : hook returns [0, 0] — identical to the OFF arm, used as a mechanism control
  const  : hook returns a constant offset (--i3-const DU,DV) — control for "any offset"
  jitter : zero-mean tanh(0.5*N(0,1)) sample per tick, RMS ~7 mm — the I3 net's own
           action distribution at initialisation. Deterministic: the RNG is re-seeded from
           a fixed per-episode counter, so two arms with the same seed stream match.

Run (decider):
  AEGIS_POSE_NOISE=0.01,2 AEGIS_REG=depth timeout 1200 python3 experiments/i3_deploy.py \
      --npz results/aegis_v2/I3_r241_ppo_net.npz --i3-mode npz \
      --seeds 20 --no-upload --path trochoid --compare trochoid \
      --compare-env AEGIS_RESIDUAL_ACTIVE=0.0 --suites fixture_A,fixture_B,fixture_R \
      --out results/aegis_v2/I3_r241_residual_n012.jsonl
"""
from __future__ import annotations

import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(REPO, "results", "aegis_v2")
sys.path.insert(0, os.path.join(REPO, "experiments"))
os.environ.setdefault("AEGIS_UPLOAD", "0")
os.environ.setdefault("AEGIS_INSTALL", "0")

import numpy as np  # noqa: E402

import kaggle_aegis_sweep as rig  # noqa: E402

# Read ONCE here: the rig reads the same env at import time, so an operator can only change it
# before this process starts. Asserted in factory() so a candidate arm can never silently run the
# frozen champion (an inert decider scored keep=false with delta 0.0000 and looked like a result).
RESIDUAL_ACTIVE = int(os.environ.get("AEGIS_RESIDUAL_ACTIVE", "0"))

OBS_SCALE = np.array([0.05, 0.05, 0.2, 0.2, 0.5, 1.0], dtype=np.float64)  # = run_i3_ppo.OBS_SCALE
CLIP_M = rig.RESIDUAL_CLIP_M


def obs_of(state: dict) -> np.ndarray:
    """Purpose: 31-d observation, byte-identical to run_i3_ppo.obs_of.
    Inputs: rig hook state dict. Outputs: float64 vector."""
    return np.concatenate([
        np.array([state["eu"], state["ev"], state["vu"], state["vv"], state["fn"],
                  state["prog"]], dtype=np.float64) / OBS_SCALE,
        np.asarray(state["patch"], dtype=np.float64)])


class NpzHook:
    """Purpose: pure-numpy eval of the trained residual (deployed forward = 3 matmuls).
    Inputs: weight dict from I3_r*_ppo_net.npz. Outputs: .act(state) -> [du, dv] in metres."""

    def __init__(self, w: dict) -> None:
        self.w, self.trace = w, []

    def act(self, state: dict) -> list:
        o = obs_of(state)
        h = np.tanh(o @ self.w["w1"].T + self.w["b1"])
        h = np.tanh(h @ self.w["w2"].T + self.w["b2"])
        mu = h @ self.w["w3"].T + self.w["b3"]
        return np.clip(CLIP_M * np.tanh(mu), -CLIP_M, CLIP_M).tolist()


class ConstHook:
    """Purpose: constant-offset control arm. Inputs: (du, dv) metres. Outputs: .act()."""

    def __init__(self, du: float, dv: float) -> None:
        self.du, self.dv, self.trace = du, dv, []

    def act(self, state: dict) -> list:
        return [self.du, self.dv]


class JitterHook:
    """Purpose: zero-mean exploration control with the I3 net's INITIAL action
    distribution, tanh(0.5 * N(0,1)) * CLIP_M (RMS ~7 mm). Deterministic per episode:
    the RNG is re-seeded from a global episode counter, so the same seed stream replays
    the same action sequence in both arms. Inputs: none. Outputs: .act()."""

    def __init__(self, seed: int) -> None:
        self.rng = np.random.default_rng(seed)
        self.trace = []

    def act(self, state: dict) -> list:
        raw = 0.5 * self.rng.standard_normal(2)
        a = CLIP_M * np.tanh(raw)
        return np.clip(a, -CLIP_M, CLIP_M).tolist()


def main(argv: list[str]) -> int:
    """Purpose: build the hook named by --i3-mode/--npz, install it in the rig module and
    hand control to the rig's own main() with the remaining argv. Inputs: CLI. Outputs: rig
    exit code. Every argument except --i3-mode/--npz/--i3-const is the rig's own."""
    mode, npz, const = "npz", "", "0.0,0.0"
    rest: list[str] = []
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--i3-mode":
            mode = argv[i + 1]; i += 2
        elif a == "--npz":
            npz = argv[i + 1]; i += 2
        elif a == "--i3-const":
            const = argv[i + 1]; i += 2
        else:
            rest.append(a); i += 1
    counter = {"ep": 0}

    def factory():                       # rig calls this once per episode
        counter["ep"] += 1
        if mode == "const":
            du, dv = (float(x) for x in const.split(","))
            return ConstHook(du, dv)
        if mode == "jitter":
            return JitterHook(90000 + counter["ep"])
        if not RESIDUAL_ACTIVE:
            raise SystemExit("AEGIS_RESIDUAL_ACTIVE is not 1: the rig ignores the hook and both "
                             "arms would run the frozen champion. Export AEGIS_RESIDUAL_ACTIVE=1.")
        d = np.load(npz)
        if "obs_scale" in d:
            assert np.allclose(d["obs_scale"], OBS_SCALE), "npz obs_scale != wrapper OBS_SCALE"
        return NpzHook({k: d[k] for k in ("w1", "b1", "w2", "b2", "w3", "b3")})

    rig.RESIDUAL_HOOK_FN = factory
    rig.log = lambda *a, **k: None       # keep the rig's per-episode log off stdout
    sys.argv = [sys.argv[0]] + rest
    print(f"[i3] mode={mode} npz={npz or '-'} residual_active={rig.RESIDUAL_ACTIVE} "
          f"clip={CLIP_M} pose_noise={rig.POSE_NOISE}", flush=True)
    return rig.main()


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
