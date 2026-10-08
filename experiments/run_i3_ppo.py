"""I3 — residual PPO on the champion controller (ideas.md §2, lowest-numbered queued idea).

WHAT: a = a_trochoid + clip(pi(o), +-0.02 m) in the FIXTURE frame, obs = (pose error,
velocity, contact normal force, local 5x5 continuous-coverage patch, path phase), and
r = delta coverage_cont - 0.01|fn - f*| - 1[veto] exactly as the spec line reads.

WHY IT IS NOT A GATE (G7): the residual is added to the SERVO COMMAND as a bounded target
offset. The body state is never written, the tool is never teleported, and success/coverage
are still computed by the rig from physics contacts at return time. The Tier-4 gate stays the
post-hoc tag (run 170) and is untouched.

PRE-REGISTERED (do not move after seeing the curve):
  * training suites = fixture_A, fixture_B ONLY. The spec line says A+B+R; the rig's own rule
    is "fixture_R: NEVER tune on it; report it", so R is held out here and becomes a genuine
    zero-shot number. DECLARED DEVIATION, and it makes the I3 bar (R >= 0.95) harder.
  * training seeds 300000+ are disjoint from the decider seeds 91000..99000.
  * training pose noise = 1 cm / 2 deg (the I3 spec), with AEGIS_REG=depth (the frozen
    champion stack per ideas.md §0) so training and the decider see the same pipeline.
  * ABORT: gain < 0.02 in mean training coverage_cont between the first and last 20 episodes
    -> DISCARD without spending a 20-seed decider. Cap: 200k scrub ticks (the spec's step
    budget) and a 900 s wall clock.

  BUG FOUND + FIXED (run 241's trainer, this file's GAE): the advantage trace was
  `A_t = (r_t - V_t) + (GAMMA + LAM) * A_{t+1}`, i.e. the lambda-trace term carried NO gamma.
  With GAMMA 0.99 and LAM 0.95 that multiplier is 1.94 > 1, so the recursion DIVERGED: over a
  1222-transition batch the advantage reached inf, the normalised advantage was NaN, the very
  first optimiser step wrote NaN into every weight, and the "trained" net deployed as a CONSTANT
  +0.02 m in BOTH fixture axes (np.clip(NaN, lo, hi) -> hi in the rig's max/min/max(NaN) chain).
  run 241's curve, its gain=0.004 and its ABORT_NO_GAIN verdict are therefore an artefact of a
  broken instrument, not evidence about PPO; `pi_loss` is NaN in episode 1 of that log.
  The fix is the textbook GAE(lambda): `delta_t = r_t + GAMMA*V_{t+1} - V_t`,
  `A_t = delta_t + GAMMA*LAM*A_{t+1}` (GAMMA*LAM = 0.9405 < 1, stable), plus a post-step
  finiteness assertion so a non-finite update aborts loudly instead of silently deploying.
  * DECIDER (if the abort does not fire): 20 seeds x 3 suites, --compare trochoid with
    --compare-env AEGIS_RESIDUAL_ACTIVE=0.0 -> the rig's own paired record (Welch p on
    coverage_cont, Fisher p on success, keep field). Bar: B transfer_success > 0.70 and
    p < 0.01, plus the I3 spec's own R >= 0.95 under the same noise.

EDGE BUDGET: the deployed policy is the pure-numpy forward below (3 matmuls); the torch net
exists only to train it. Params and per-tick latency are measured and logged.

Run: AEGIS_POSE_NOISE=0.01,2 AEGIS_REG=depth timeout 900 python3 experiments/run_i3_ppo.py
"""
from __future__ import annotations

import json
import math
import os
import statistics
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(REPO, "results", "aegis_v2")
os.environ.setdefault("AEGIS_POSE_NOISE", "0.01,2")
# AEGIS_REG is deliberately NOT set. Tier-2 depth registration (I9) was in the earlier draft of
# this trainer, but experiments/i3_deploy.py never enabled it, so the policy was trained on
# registered fixtures and would have been scored on unregistered ones -- the coverage that
# training drove to 1.0000 was the registration's, not the policy's. Train and decider now share
# ONE pipeline: trochoid + 1 cm / 2 deg pose noise, no registration, which is also the pipeline
# every headroom probe on disk used.
os.environ.setdefault("AEGIS_UPLOAD", "0")
os.environ.setdefault("AEGIS_INSTALL", "0")
os.environ.setdefault("AEGIS_RESIDUAL_ACTIVE", "1")
sys.path.insert(0, os.path.join(REPO, "experiments"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

import kaggle_aegis_sweep as rig  # noqa: E402  -- the canonical rig, imported not copied

TRAIN_SEED0 = 300000          # disjoint from the decider's 91000..99000
TRAIN_SUITES = ("fixture_A", "fixture_B")
EPISODES_PER_UPDATE = 16
SCRUB_STEP_CAP = 200_000      # the I3 spec's abort budget
WALL_CAP_S = 900.0
CLIP_M = rig.RESIDUAL_CLIP_M
FN_STAR = rig.FN_SET_N         # 0.5 N rig units, the flat operating point
OBS_SCALE = np.array([0.05, 0.05, 0.2, 0.2, 0.5, 1.0], dtype=np.float64)
GAMMA, LAM = 0.99, 0.95
TAG = os.environ.get("I3_TAG", "I3_r242")   # run-241 outputs stay as the NaN-bug evidence


class Net(nn.Module):
    """Purpose: the residual policy + value trunk. Inputs: 31-d normalised observation.
    Outputs: (mean action in raw tanh units, state value). tanh keeps the action inside
    [-1, 1] and the rig clips the metres, so the policy can never leave its authority bound.
    """

    def __init__(self, obs_dim: int = 31, hidden: int = 64) -> None:
        super().__init__()
        self.body = nn.Sequential(nn.Linear(obs_dim, hidden), nn.Tanh(),
                                  nn.Linear(hidden, hidden), nn.Tanh())
        self.pi = nn.Linear(hidden, 2)
        self.v = nn.Linear(hidden, 1)
        self.log_std = nn.Parameter(torch.full((2,), math.log(0.5)))
        for m in list(self.body) + [self.pi, self.v]:
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, 1.0)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.body(x)
        return self.pi(h), self.v(h).squeeze(-1)


def obs_of(state: dict) -> np.ndarray:
    """Purpose: 31-d observation = 4 pose/vel + fn + phase + 5x5 coverage patch.
    Inputs: rig hook state dict. Outputs: float64 vector (patch row-major, v outer)."""
    return np.concatenate([
        np.array([state["eu"], state["ev"], state["vu"], state["vv"], state["fn"],
                  state["prog"]], dtype=np.float64) / OBS_SCALE,
        np.asarray(state["patch"], dtype=np.float64),
    ])


class Hook:
    """Purpose: the object the rig calls once per scrub tick. .act() returns the 2-D
    residual in metres; .roll keeps (obs, action, logp, value) so the trainer can pair them
    with the rig's .trace (same length: both are appended on the same scrub tick)."""

    def __init__(self, net: Net | None, weights: dict | None, collect: bool) -> None:
        self.net, self.w, self.collect = net, weights, collect
        self.trace: list = []
        self.roll: list = []

    def _numpy_forward(self, o: np.ndarray) -> np.ndarray:
        h = np.tanh(o @ self.w["w1"].T + self.w["b1"])
        h = np.tanh(h @ self.w["w2"].T + self.w["b2"])
        return h @ self.w["w3"].T + self.w["b3"]        # the deployed path: numpy only

    def act(self, state: dict):
        o = obs_of(state)
        if self.w is not None:                        # EVAL: no torch, no grad, pure numpy
            mu = self._numpy_forward(o)
            a = np.clip(CLIP_M * np.tanh(mu), -CLIP_M, CLIP_M)
            return a.tolist()
        ot = torch.as_tensor(o, dtype=torch.float32)
        with torch.no_grad():
            mu, v = self.net(ot)
            std = self.net.log_std.clamp(-2.0, 0.0).exp()
            eps = torch.randn(2)
            raw = mu + std * eps
            logp = (-0.5 * (eps ** 2) - self.net.log_std.clamp(-2.0, 0.0)).sum()
        self.roll.append((o, raw.numpy(), float(logp), float(v)))
        return (CLIP_M * torch.tanh(raw)).tolist()


def rollout(net: Net, ep_seed: int) -> tuple[list, float, dict]:
    """Purpose: one training episode per suite, both arms driven by the SAME hook object.
    Inputs: net, episode index. Outputs: (transitions, coverage_cont, episode record)."""
    transitions = []
    covs = []
    recs = []
    for j, suite in enumerate(TRAIN_SUITES):
        hook = Hook(net, None, True)
        rig.RESIDUAL_HOOK_FN = lambda h=hook: h
        seed = TRAIN_SEED0 + 137 * ep_seed + 1000 * j
        rec = rig.run_episode(suite, rig.FIXTURES[suite], seed, "pybullet", "trochoid")
        recs.append(rec)
        covs.append(rec["coverage_cont"])
        tr, roll = hook.trace, hook.roll
        if len(tr) != len(roll):
            raise RuntimeError("trace/roll length mismatch")
        prev_cells = 0
        block = []
        for st, (o, raw, logp, val) in zip(tr, roll):
            r = (st["cells"] - prev_cells) / st["ncell"] - 0.01 * abs(st["fn_t"] - FN_STAR)
            prev_cells = st["cells"]
            block.append((o, raw, logp, val, r, 0.0))
        if block:                                     # last tick of an episode is terminal
            block[-1] = (*block[-1][:5], 1.0)
        transitions += block
    return transitions, statistics.fmean(covs), recs


def ppo_update(net: Net, opt: torch.optim.Optimizer, batch: list) -> dict:
    """Purpose: clipped-surrogate PPO with GAE(0.95, 0.99) over one collected batch.
    Inputs: net, optimiser, [(obs, raw_action, logp, value, reward, done)]. Outputs: stats."""
    O = torch.as_tensor(np.stack([b[0] for b in batch]), dtype=torch.float32)
    A = torch.as_tensor(np.stack([b[1] for b in batch]), dtype=torch.float32)
    LP = torch.as_tensor([b[2] for b in batch], dtype=torch.float32)
    VL = torch.as_tensor([b[3] for b in batch], dtype=torch.float32)
    R = torch.as_tensor([b[4] for b in batch], dtype=torch.float32)
    D = torch.as_tensor([b[5] for b in batch], dtype=torch.float32)
    with torch.no_grad():
        adv = torch.zeros_like(R)
        # Textbook GAE(lambda). run 241 used A_t = (r_t - V_t) + (GAMMA + LAM)*A_{t+1}: the
        # lambda term carried no gamma, so the multiplier was 1.94 > 1, the recursion diverged
        # to inf, the normalised advantage was NaN and the FIRST update wrote NaN into every
        # weight. GAMMA*LAM = 0.9405 < 1 is the stability condition.
        vnext = torch.cat([VL[1:], torch.zeros(1)])      # V_{t+1}; 0 at the terminal
        last = torch.zeros((), dtype=torch.float32)
        for t in range(len(R) - 1, -1, -1):
            nonterm = 1.0 - D[t]
            delta = R[t] + GAMMA * vnext[t] * nonterm - VL[t]
            last = delta + (GAMMA * LAM) * last * nonterm
            adv[t] = last
        ret = adv + VL
        if not (torch.isfinite(adv).all() and torch.isfinite(ret).all()):
            raise FloatingPointError(f"GAE diverged (adv finite={bool(torch.isfinite(adv).all())})")
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    stats = {"pi_loss": 0.0, "v_loss": 0.0, "kl": 0.0, "n": len(batch)}
    idx = np.arange(len(batch))
    for _ in range(4):                               # 4 epochs
        np.random.shuffle(idx)
        for k in range(0, len(idx), 1024):
            mb = idx[k:k + 1024]
            mu, v = net(O[mb])
            std = net.log_std.clamp(-2.0, 0.0).exp()
            logp = (-0.5 * ((A[mb] - mu) / std) ** 2 - net.log_std.clamp(-2.0, 0.0)).sum(-1)
            ratio = (logp - LP[mb]).exp()
            l1 = -adv[mb] * ratio
            l2 = -adv[mb] * ratio.clamp(0.8, 1.2)
            pi = torch.max(l1, l2).mean()
            vloss = ((v - ret[mb]) ** 2).mean()
            ent = net.log_std.clamp(-2.0, 0.0).sum()
            loss = pi + 0.5 * vloss - 0.01 * ent
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 0.5)
            opt.step()
            if not all(torch.isfinite(p).all() for p in net.parameters()):
                raise FloatingPointError("non-finite parameters after opt.step() -- trainer diverged")
            with torch.no_grad():
                stats["kl"] += float(((LP[mb] - logp) ** 2).mean() / 2) / max(1, len(idx) / 1024)
            stats["pi_loss"] += float(pi)
            stats["v_loss"] += float(vloss)
    nmb = 4 * max(1, math.ceil(len(batch) / 1024))
    for k in ("pi_loss", "v_loss", "kl"):
        stats[k] /= nmb
    return stats


def export_npz(net: Net, path: str, meta: dict) -> None:
    """Purpose: dump the DEPLOYED numpy weights (trunk + policy head) plus the obs scale.
    Inputs: trained net, output path, meta dict. Outputs: file written."""
    w = {k: v.detach().numpy().astype(np.float64) for k, v in net.state_dict().items()}
    np.savez(path, w1=w["body.0.weight"], b1=w["body.0.bias"],
             w2=w["body.2.weight"], b2=w["body.2.bias"],
             w3=w["pi.weight"], b3=w["pi.bias"], obs_scale=OBS_SCALE,
             meta=json.dumps(meta))
    print(f"[i3] exported {path}", flush=True)


def load_npz(path: str) -> dict:
    """Purpose: load the deployed numpy weights. Inputs: npz path. Outputs: weight dict."""
    d = np.load(path)
    return {k: d[k] for k in ("w1", "b1", "w2", "b2", "w3", "b3")}


def main() -> int:
    """Purpose: train the residual PPO policy on A+B under 1cm/2deg noise, log the learning
    curve and the edge budget, export the numpy weights, and apply the pre-registered abort.
    Inputs: none (env knobs). Outputs: results/aegis_v2/$I3_TAG_ppo_{curve.json,net.npz}."""
    os.makedirs(RESULTS, exist_ok=True)
    rig.log = lambda *a, **k: None                    # the rig's per-episode log is noise here
    torch.manual_seed(0)
    np.random.seed(0)
    net = Net()
    opt = torch.optim.Adam(net.parameters(), lr=3e-4)
    n_params = int(sum(p.numel() for p in net.parameters()))
    print(f"[i3] params={n_params} clip={CLIP_M} m noise={rig.POSE_NOISE} reg=depth "
          f"train_suites={TRAIN_SUITES}", flush=True)
    curve, t0, steps, ep = [], time.time(), 0, 0
    first20, last20 = [], []
    while steps < SCRUB_STEP_CAP and time.time() - t0 < WALL_CAP_S:
        batch, cov, recs = rollout(net, ep)
        ep += 1
        steps += len(batch)
        curve.append({"ep": ep, "scrub_steps": len(batch), "cov": round(cov, 4),
                      "success": sum(1 for r in recs if r["success"]),
                      "res_rms_mm": round(1000 * statistics.fmean(
                          [r.get("residual_rms_m", 0.0) for r in recs]), 3),
                      "cov_gap": round(statistics.fmean(
                          [r.get("cov_kernel_gap", 0.0) for r in recs]), 4),
                      "t": round(time.time() - t0, 1)})
        first20 += [cov]
        last20 += [cov]
        stats = ppo_update(net, opt, batch)
        curve[-1].update({k: round(v, 5) for k, v in stats.items() if k != "n"})
        if ep % 4 == 0 or ep == 1:
            print(f"[i3] ep={ep} steps={steps} cov={cov:.4f} "
                  f"succ={curve[-1]['success']}/2 res_rms={curve[-1]['res_rms_mm']}mm "
                  f"pi={stats['pi_loss']:.4f} v={stats['v_loss']:.4f} "
                  f"kl={stats['kl']:.5f} t={curve[-1]['t']}s", flush=True)
    head = statistics.fmean(first20[:20])
    tail = statistics.fmean(last20[-20:])
    gain = tail - head
    # edge budget: the deployed numpy forward, per control tick
    w = {"w1": net.body[0].weight.detach().numpy().astype(np.float64),
         "b1": net.body[0].bias.detach().numpy().astype(np.float64),
         "w2": net.body[2].weight.detach().numpy().astype(np.float64),
         "b2": net.body[2].bias.detach().numpy().astype(np.float64),
         "w3": net.pi.weight.detach().numpy().astype(np.float64),
         "b3": net.pi.bias.detach().numpy().astype(np.float64)}
    hook = Hook(None, w, False)
    o = np.zeros(31)
    t = time.perf_counter()
    for _ in range(2000):
        hook._numpy_forward(o)
    lat_ms = (time.perf_counter() - t) / 2000 * 1e3
    verdict = "TRAINED" if gain >= 0.02 else "ABORT_NO_GAIN"
    meta = {"episodes": ep, "scrub_steps": steps, "cov_first20": round(head, 4),
            "cov_last20": round(tail, 4), "gain": round(gain, 4), "verdict": verdict,
            "params": n_params, "latency_ms_per_tick": round(lat_ms, 5),
            "clip_m": CLIP_M, "pose_noise": rig.POSE_NOISE, "train_suites": list(TRAIN_SUITES),
            "train_seed0": TRAIN_SEED0, "wall_s": round(time.time() - t0, 1)}
    out = os.path.join(RESULTS, f"{TAG}_ppo_net.npz")
    export_npz(net, out, meta)
    with open(os.path.join(RESULTS, f"{TAG}_ppo_curve.json"), "w") as fh:
        json.dump({"meta": meta, "curve": curve}, fh, indent=1)
    print("[i3] RESULT " + json.dumps(meta), flush=True)
    return 0 if verdict == "TRAINED" else 3


if __name__ == "__main__":
    raise SystemExit(main())
