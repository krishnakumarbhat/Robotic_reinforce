"""Real-physics comparative test (COMPLETION track): scripted vs CFM vs Diffusion
vs N42-gated-CFM on PandaReach-v3 (existing ckpts) + PandaPush-v3 (trained twins).
CPU-only (no GPU contention with background loop). PyBullet DIRECT.
Metrics -> results/metrics.json ONLY: success, contact-force stability,
trajectory jerk (L2), recovery latency (time-to-first-success).
N42 gate (real instantiation): CFM proposes chunk; energy E = predicted chunk
jerk + drift penalty; veto if E > theta (theta = 75th pct on calib split);
fallback = scripted core (N40-analog). Gate-off arm = raw CFM chunk.
"""
import json
import os
import time
import numpy as np
import torch
import torch.nn as nn

DEV = "cpu"  # ponytail: tiny MLPs, CPU deterministic, zero GPU contention
torch.manual_seed(0)
np.random.seed(0)
H = 8
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def make_env(task, seed):
    import gymnasium as gym  # noqa: E402
    import panda_gym  # noqa: E402,F401
    env = gym.make(task)
    env.reset(seed=seed)
    return env


class VF(nn.Module):
    def __init__(self, obs_dim):
        super().__init__()
        self.n = nn.Sequential(nn.Linear(H * 3 + obs_dim + 1, 128), nn.SiLU(),
                               nn.Linear(128, 128), nn.SiLU(), nn.Linear(128, H * 3))

    def forward(self, x, o, t):
        return self.n(torch.cat([x, o, t], -1))


@torch.no_grad()
def predict_cfm(vf, o, steps=10):
    x = torch.randn(H * 3)
    for k in range(steps):
        t = torch.full((1, 1), k / steps)
        x = x + (1 / steps) * vf(x[None], o[None], t)[0]
    return x.view(H, 3).numpy()


def train_cfm(obs_dim, O, A, epochs=120):
    vf = VF(obs_dim)
    opt = torch.optim.Adam(vf.parameters(), lr=3e-3)
    O = torch.from_numpy(O)
    Af = torch.from_numpy(A).view(len(A), -1)
    n = len(O)
    for _ in range(epochs):
        idx = torch.randint(0, n, (128,))
        o, a = O[idx], Af[idx]
        t = torch.rand(128, 1)
        e = torch.randn_like(a)
        xt = t * a + (1 - t) * e
        loss = ((vf(xt, o, t) - (a - e)) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return vf.eval()


# ---- task adapters: obs vector, scripted core, success check ----
def reach_obs(obs):
    ee, goal = obs["achieved_goal"], obs["desired_goal"]
    return np.concatenate([ee, goal - ee]).astype(np.float32)


def reach_core(obs):
    ee, goal = obs["achieved_goal"], obs["desired_goal"]
    return np.clip(5.0 * (goal - ee), -1, 1)


def push_obs(obs):
    return obs["observation"].astype(np.float32)


def push_core(o):
    ee = o["observation"][0:3]
    obj, goal = o["achieved_goal"], o["desired_goal"]
    to_goal = goal - obj
    d = np.linalg.norm(to_goal)
    if d < 0.02:
        wp = goal + np.array([0.0, 0.0, 0.03])
    else:
        u = to_goal / max(d, 1e-6)
        behind = obj - u * 0.055
        behind[2] = obj[2]
        if np.linalg.norm(ee - behind) > 0.035:
            wp = behind
        else:
            wp = obj + u * 0.08
            wp[2] = obj[2] + 0.005
    return np.clip(3.0 * (wp - ee), -1, 1)


TASKS = {
    "PandaReach-v3": {"obs": reach_obs, "core": reach_core, "obs_dim": 6, "train": False},
    "PandaPush-v3": {"obs": push_obs, "core": push_core, "obs_dim": 18, "train": True},
}


def contact_force(env):
    """Sum of contact normal forces this step; None if API unavailable."""
    try:
        import pybullet as p
        sim = env.unwrapped.sim
        cid = getattr(sim, "physics_client_id", getattr(sim, "_client", 0))
        pts = p.getContactPoints(physicsClientId=cid)
        return float(sum(pt[9] for pt in pts))
    except Exception:
        return None


@torch.no_grad()
def rollout(env, act_fn, seed, max_steps=60):
    obs, _ = env.reset(seed=seed)
    acts, forces, first_succ = [], [], None
    for t in range(max_steps):
        ch = np.asarray(act_fn(obs), dtype=np.float32)
        a = ch[0] if ch.ndim == 2 else ch
        acts.append(a)
        obs, _, term, trunc, _ = env.step(a)
        f = contact_force(env)
        forces.append(f if f is not None else 0.0)
        if first_succ is None and np.linalg.norm(
                obs["achieved_goal"] - obs["desired_goal"]) < 0.05:
            first_succ = t + 1
        if term or trunc:
            break
    A = np.array(acts)
    succ = first_succ is not None
    jerk = float(np.mean(np.diff(A, n=2, axis=0) ** 2)) if len(A) > 2 else 0.0
    F = np.array(forces)
    stab = float(1.0 / (1.0 + np.std(F))) if np.any(F) else None
    return {"success": succ, "jerk": jerk, "contact_stability": stab,
            "mean_force": float(np.mean(F)) if np.any(F) else None,
            "latency": first_succ if succ else max_steps}


def collect_demos(task, core, obs_fn, demos=24, steps=50):
    O, A, env = [], [], make_env(task, 0)
    for ep in range(demos):
        obs, _ = env.reset(seed=1000 + ep)
        acts = []
        for _ in range(steps):
            a = core(obs)
            obs, _, term, trunc, _ = env.step(a.astype(np.float32))
            acts.append(a)
            if term or trunc:
                break
        acts = np.array(acts)
        obs, _ = env.reset(seed=1000 + ep)
        for _ in range(len(acts)):
            O.append(obs_fn(obs))
            c = acts[_:_ + H]
            if len(c) < H:
                c = np.vstack([c, np.tile(c[-1], (H - len(c), 1))])
            A.append(c.astype(np.float32))
            obs, _, term, trunc, _ = env.step(core(obs).astype(np.float32))
            if term or trunc:
                obs, _ = env.reset(seed=1000 + ep)
    env.close()
    return np.array(O), np.array(A)


def main():
    out = {"updated": time.strftime("%F %T"), "device": DEV, "tasks": {}}
    for task, cfg in TASKS.items():
        obs_fn, core, d = cfg["obs"], cfg["core"], cfg["obs_dim"]
        if cfg["train"]:
            O, A = collect_demos(task, core, obs_fn)
            vf = train_cfm(d, O, A)
            print(f"{task}: trained push-CFM on {len(O)} pairs", flush=True)
        else:
            vf = VF(d)
            vf.load_state_dict(torch.load(os.path.join(REPO, "results/ckpt_cfm.pt"),
                                          map_location="cpu", weights_only=True))
            vf.eval()
        env = make_env(task, 7)

        def cfm_chunk(o):
            return predict_cfm(vf, torch.from_numpy(obs_fn(o)))

        # calibrate gate theta on 4 seeds (75th pct of proposal energy)
        energies = []
        for s in range(4):
            obs, _ = env.reset(seed=9000 + s)
            for _ in range(6):
                ch = cfm_chunk(obs)
                ej = float(np.mean(np.abs(np.diff(ch, n=2, axis=0))))
                ee = obs["achieved_goal"]
                drift = float(np.linalg.norm(ee + ch.sum(0) - obs["desired_goal"])
                              - np.linalg.norm(ee - obs["desired_goal"]))
                energies.append(ej + max(0.0, drift))
                obs, _, term, trunc, _ = env.step(core(obs).astype(np.float32))
                if term or trunc:
                    break
        theta = float(np.percentile(energies, 75))
        stats = {"vetoes": 0, "proposals": 0}

        def gate_on(o):
            ch = cfm_chunk(o)
            ej = float(np.mean(np.abs(np.diff(ch, n=2, axis=0))))
            ee = o["achieved_goal"]
            drift = float(np.linalg.norm(ee + ch.sum(0) - o["desired_goal"])
                          - np.linalg.norm(ee - o["desired_goal"]))
            stats["proposals"] += 1
            if ej + max(0.0, drift) > theta:
                stats["vetoes"] += 1
                return core(o)[None].repeat(H, 0)
            return ch

        pols = {"scripted-core": lambda o: core(o)[None].repeat(H, 0),
                "cfm-raw": cfm_chunk, "n42-gated-cfm": gate_on}
        n_seed = (int(os.environ.get("REACH_SEEDS", "20")) if not cfg["train"]
                  else int(os.environ.get("PUSH_SEEDS", "10")))
        res = {}
        for name, fn in pols.items():
            R = [rollout(env, fn, 5000 + s) for s in range(n_seed)]
            res[name] = {"success": float(np.mean([r["success"] for r in R])),
                         "jerk": float(np.mean([r["jerk"] for r in R])),
                         "contact_stability": float(np.mean(
                             [r["contact_stability"] for r in R
                              if r["contact_stability"] is not None]))
                         if any(r["contact_stability"] is not None for r in R) else None,
                         "latency_steps": float(np.mean([r["latency"] for r in R]))}
            print(f"{task} {name}: succ={res[name]['success']:.2f} "
                  f"jerk={res[name]['jerk']:.4f} lat={res[name]['latency_steps']:.0f}",
                  flush=True)
        res["n42_gate"] = {"theta": theta,
                           "veto_rate": stats["vetoes"] / max(1, stats["proposals"]),
                           "gate_delta": res["n42-gated-cfm"]["success"]
                           - res["cfm-raw"]["success"]}
        print(f"{task} gate: theta={theta:.4f} "
              f"veto={res['n42_gate']['veto_rate']:.3f} "
              f"delta={res['n42_gate']['gate_delta']:+.3f}", flush=True)
        env.close()
        out["tasks"][task] = res
    json.dump(out, open(os.path.join(REPO, "results/metrics.json"), "w"), indent=1)
    print("wrote results/metrics.json")


if __name__ == "__main__":
    main()
