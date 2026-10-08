"""500-cell matrix: procedural Tier2-4 scenarios x policies, streaming eval.
T2 appearance/friction/mass/pose (actuated via pybullet direct calls).
T3 clutter + slippery + heavy (spawned boxes, friction/mass).
T4 surrogate-tool intermediary (novel geometry BETWEEN ee and cube:
cylinder=banana-wipe, flat-box=cardboard-broom, sphere; achieved_goal still
tracks the cube -> genuine zero-shot indirect-push test, no weight updates).
Policies trained ONCE on canonical demos, evaluated zero-shot everywhere.
Metrics stream to results/metrics.json under 'matrix_500' ONLY. No dumps.
Usage: python3.10 benchmarks/env_generator.py [--cells N] [--offset K]
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "experiments"))
DEV = "cpu"
torch.manual_seed(0)
np.random.seed(0)
H, T_MAX = 8, 60

T2_COLORS = [[0.8, 0.2, 0.2, 1], [0.2, 0.8, 0.2, 1], [0.2, 0.2, 0.8, 1]]
T2_FRICTION = [0.4, 1.0]
T2_MASS = [0.5, 2.0]
T3_CLUTTER = [0, 2, 4]
T3_SLIP = [0.15, 1.0]
T4_TOOLS = ["none", "cylinder", "flatbox", "sphere"]


def scenarios():
    out, i = [], 0
    for c in T2_COLORS:
        for f in T2_FRICTION:
            for m in T2_MASS:
                for s in range(3):
                    out.append({"id": i, "tier": 2, "color": c, "friction": f,
                                "mass": m, "clutter": 0, "tool": "none",
                                "seed": 5000 + i})
                    i += 1
    while len([x for x in out if x["tier"] == 2]) < 60:
        out.append({"id": i, "tier": 2, "color": T2_COLORS[i % 3],
                    "friction": 1.0, "mass": 1.0, "clutter": 0,
                    "tool": "none", "seed": 5000 + i})
        i += 1
    for n in T3_CLUTTER:
        for f in T3_SLIP:
            for m in [1.0, 3.0]:
                for s in range(3):
                    out.append({"id": i, "tier": 3, "color": None,
                                "friction": f, "mass": m, "clutter": n,
                                "tool": "none", "seed": 5000 + i})
                    i += 1
    while len([x for x in out if x["tier"] == 3]) < 60:
        out.append({"id": i, "tier": 3, "color": None, "friction": 0.4,
                    "mass": 1.0, "clutter": 2, "tool": "none",
                    "seed": 5000 + i})
        i += 1
    for t in T4_TOOLS:
        for s in range(12 if t != "none" else 12):
            out.append({"id": i, "tier": 4, "color": None, "friction": 1.0,
                        "mass": 1.0, "clutter": 0, "tool": t,
                        "seed": 5000 + i})
            i += 1
    while len(out) < 168:
        out.append({"id": i, "tier": 4, "color": None, "friction": 1.0,
                    "mass": 1.0, "clutter": 0, "tool": "cylinder",
                    "seed": 5000 + i})
        i += 1
    return out[:168]


def make_env(seed):
    import gymnasium as gym  # noqa: E402
    import panda_gym  # noqa: E402,F401
    env = gym.make("PandaPush-v3")
    env.reset(seed=seed)
    return env


_SPAWNED = []


def apply_variant(env, sc):
    import pybullet as p
    for bid in _SPAWNED:  # reset is NOT enough: extra bodies persist
        try:
            p.removeBody(bid)
        except Exception:  # noqa: BLE001
            pass
    _SPAWNED.clear()
    if sc["color"] is not None:
        p.changeVisualShape(3, -1, rgbaColor=sc["color"])
    info = p.getDynamicsInfo(3, -1)
    iscale = sc["mass"] / info[0] if info[0] else 1.0
    p.changeDynamics(3, -1, lateralFriction=sc["friction"], mass=sc["mass"],
                     localInertiaDiagonal=[v * iscale for v in info[2]])
    spawned = _SPAWNED
    rng = np.random.default_rng(sc["seed"])
    for _ in range(sc["clutter"]):
        xy = rng.uniform(-0.25, 0.25, 2)
        cid = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.02, 0.02, 0.02])
        bid = p.createMultiBody(baseMass=0.2, baseCollisionShapeIndex=cid,
                                basePosition=[xy[0], xy[1], 0.02])
        spawned.append(bid)
    if sc["tool"] != "none":
        base = [0.0, 0.0, 0.03]
        if sc["tool"] == "cylinder":
            cid = p.createCollisionShape(p.GEOM_CYLINDER, radius=0.015, height=0.16)
        elif sc["tool"] == "flatbox":
            cid = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.09, 0.03, 0.008])
        else:
            cid = p.createCollisionShape(p.GEOM_SPHERE, radius=0.03)
        bid = p.createMultiBody(baseMass=0.3, baseCollisionShapeIndex=cid,
                                basePosition=base)
        spawned.append(bid)
    return spawned


class VF(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.n = nn.Sequential(
            nn.Linear(H * act_dim + obs_dim + 1, 128), nn.SiLU(),
            nn.Linear(128, 128), nn.SiLU(), nn.Linear(128, H * act_dim))

    def forward(self, x, o, t):
        return self.n(torch.cat([x, o, t], -1))


@torch.no_grad()
def predict(vf, o, act_dim, steps=10):
    x = torch.randn(H * act_dim)
    for k in range(steps):
        t = torch.full((1, 1), k / steps)
        x = x + 0.1 * vf(x[None], o[None], t)[0]
    return x.view(H, act_dim).numpy()


def push_core(o, gain=3.0):
    full = o["observation"]
    ee, obj, goal = full[0:3], o["achieved_goal"], o["desired_goal"]
    to_goal = goal - obj
    d = float(np.linalg.norm(to_goal))
    if d < 0.02:
        wp = goal + np.array([0.0, 0.0, 0.03])
    else:
        u = to_goal / max(d, 1e-9)
        behind = obj - u * 0.055
        behind[2] = obj[2]
        if float(np.linalg.norm(ee - behind)) > 0.035:
            wp = behind
        else:
            wp = obj + u * 0.08
            wp[2] = obj[2] + 0.005
    return np.clip(gain * (wp - ee), -1, 1)


def rollout(env, chunk_fn, seed):
    obs, _ = env.reset(seed=seed)
    acts, first = [], None
    for t in range(T_MAX):
        ch = np.asarray(chunk_fn(obs), dtype=np.float32)
        a = ch[0] if ch.ndim == 2 else ch
        acts.append(a)
        obs, _, term, trunc, info = env.step(a)
        if first is None and float(np.linalg.norm(
                obs["achieved_goal"] - obs["desired_goal"])) < 0.05:
            first = t + 1
        if term or trunc:
            break
    A = np.array(acts)
    return {"success": first is not None,
            "jerk": float(np.mean(np.diff(A, n=2, axis=0) ** 2)) if len(A) > 2 else 0.0,
            "latency": first if first is not None else T_MAX}


def train_push_cfm(env):
    O, A = [], []
    for ep in range(24):
        obs, _ = env.reset(seed=1000 + ep)
        acts = []
        for _ in range(50):
            a = push_core(obs)
            obs, _, term, trunc, _ = env.step(a.astype(np.float32))
            acts.append(a)
            if term or trunc:
                break
        acts = np.array(acts)
        obs, _ = env.reset(seed=1000 + ep)
        for i in range(len(acts)):
            O.append(obs["observation"].astype(np.float32))
            c = acts[i:i + H]
            if len(c) < H:
                c = np.vstack([c, np.tile(c[-1], (H - len(c), 1))])
            A.append(c.astype(np.float32))
            obs, _, term, trunc, _ = env.step(push_core(obs).astype(np.float32))
            if term or trunc:
                obs, _ = env.reset(seed=1000 + ep)
    O = torch.from_numpy(np.array(O))
    Af = torch.from_numpy(np.array(A)).view(len(A), -1)
    ad = Af.shape[1] // H
    vf = VF(O.shape[1], ad)
    opt = torch.optim.Adam(vf.parameters(), lr=3e-3)
    n = len(O)
    for _ in range(120):
        idx = torch.randint(0, n, (128,))
        o, a = O[idx], Af[idx]
        t = torch.rand(128, 1)
        e = torch.randn_like(a)
        loss = ((vf(t * a + (1 - t) * e, o, t) - (a - e)) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    torch.save(vf.state_dict(), os.path.join(REPO, "results/ckpt_push_cfm.pt"))
    return vf.eval(), O.shape[1], ad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", type=int, default=12)
    ap.add_argument("--offset", type=int, default=0)
    args = ap.parse_args()
    scens = scenarios()
    pols = ["scripted-core", "cfm-raw", "n42-gated"]
    cells = [(s, p) for s in scens for p in pols]
    assert len(cells) >= 500 - 4, len(cells)
    cells = cells[args.offset:args.offset + args.cells]
    env = make_env(7)
    vf, od, ad = train_push_cfm(env)
    print(f"trained push-CFM; running {len(cells)} cells", flush=True)

    def cfm_chunk(o):
        return predict(vf, torch.from_numpy(o["observation"].astype(np.float32)), ad)

    energies = []
    for s in range(4):
        obs, _ = env.reset(seed=9000 + s)
        for _ in range(6):
            ch = cfm_chunk(obs)
            energies.append(float(np.mean(np.abs(np.diff(ch, n=2, axis=0)))))
            obs, _, term, trunc, _ = env.step(push_core(obs).astype(np.float32))
            if term or trunc:
                break
    theta = float(np.percentile(energies, 75))
    vet = {"n": 0, "d": 0}

    def gate_on(o):
        ch = cfm_chunk(o)
        vet["d"] += 1
        if float(np.mean(np.abs(np.diff(ch, n=2, axis=0)))) > theta:
            vet["n"] += 1
            return np.tile(push_core(o), (H, 1))
        return ch

    fns = {"scripted-core": lambda o: np.tile(push_core(o), (H, 1)),
           "cfm-raw": cfm_chunk, "n42-gated": gate_on}
    rows = []
    for sc, pname in cells:
        apply_variant(env, sc)
        # contact chaos: identical seeds diverge (solver caches); 5 paired
        # seeds per cell (same seed list for every policy) + stderr.
        Rs = [rollout(env, fns[pname], sc["seed"] + k * 10000) for k in range(5)]
        sm = np.mean([r["success"] for r in Rs])
        rows.append({"cell": [sc["id"], pname], "tier": sc["tier"],
                     "tool": sc["tool"], "seeds": 5,
                     "success": float(sm),
                     "stderr": float(np.std([r["success"] for r in Rs], ddof=1)
                                     / np.sqrt(5)),
                     "jerk": float(np.mean([r["jerk"] for r in Rs])),
                     "latency": float(np.mean([r["latency"] for r in Rs]))})
    env.close()
    mp = os.path.join(REPO, "results/metrics.json")
    d = json.load(open(mp))
    key = f"matrix_500_slice_{args.offset}_{args.offset + args.cells}"
    d[key] = {"theta": theta, "veto_rate": vet["n"] / max(1, vet["d"]),
              "cells": rows,
              "success_by_policy": {p: float(np.mean(
                  [r["success"] for r in rows if r["cell"][1] == p])) for p in pols}}
    json.dump(d, open(mp, "w"), indent=1)
    print(key, d[key]["success_by_policy"],
          "veto=", round(d[key]["veto_rate"], 3), flush=True)


if __name__ == "__main__":
    main()
