"""Adaptation-fluency + learning-speed benchmark (mission priority 1+2, 2026-09-20).
ADAPT: canonical (fric 1.0/mass 1.0/clean) vs NOVEL (T4 cylinder intermediary +
slippery 0.4) x {scripted, push-CFM, n42-gated} x 6 paired seeds -> adapt gap.
LEARN: push-CFM trained k=48 demos E60 vs existing k=24 ckpt -> data-efficiency.
CPU-only (no GPU thrash). Streams to results/metrics.json key 'adapt_benchmark'.
"""
import json
import os
import time
import numpy as np
import torch
import torch.nn as nn

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEV = "cpu"
torch.manual_seed(0)
np.random.seed(0)
H, T_MAX, N_SEED = 8, 60, 8
K48, EPOCHS = int(os.environ.get("ADAPT_K", "48")), 120
SEED_BASE = int(os.environ.get("ADAPT_SEEDS", "5000"))


def make_env(seed):
    import gymnasium as gym  # noqa: E402
    import panda_gym  # noqa: E402,F401
    env = gym.make("PandaPush-v3")
    env.reset(seed=seed)
    return env


class VF(nn.Module):
    def __init__(self, od, ad):
        super().__init__()
        self.n = nn.Sequential(nn.Linear(H * ad + od + 1, 128), nn.SiLU(),
                               nn.Linear(128, 128), nn.SiLU(), nn.Linear(128, H * ad))

    def forward(self, x, o, t):
        return self.n(torch.cat([x, o, t], -1))


@torch.no_grad()
def predict(vf, o, ad, steps=10):
    x = torch.randn(H * ad)
    for k in range(steps):
        t = torch.full((1, 1), k / steps)
        x = x + (1 / steps) * vf(x[None], o[None], t)[0]
    return x.view(H, ad).numpy()


def push_core(o):
    ee = o["observation"][0:3]
    obj, goal = o["achieved_goal"], o["desired_goal"]
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
    return np.clip(3.0 * (wp - ee), -1, 1)


def apply_novel(env):
    import pybullet as p
    p.changeDynamics(3, -1, lateralFriction=0.4)
    cid = p.createCollisionShape(p.GEOM_CYLINDER, radius=0.015, height=0.16)
    bid = p.createMultiBody(baseMass=0.3, baseCollisionShapeIndex=cid,
                            basePosition=[0.0, 0.0, 0.03])
    return [bid]


def rollout(env, chunk_fn, seed):
    obs, _ = env.reset(seed=seed)
    acts, first = [], None
    for t in range(T_MAX):
        ch = np.asarray(chunk_fn(obs), dtype=np.float32)
        a = ch[0] if ch.ndim == 2 else ch
        acts.append(a)
        obs, _, term, trunc, _ = env.step(a)
        if first is None and float(np.linalg.norm(
                obs["achieved_goal"] - obs["desired_goal"])) < 0.05:
            first = t + 1
        if term or trunc:
            break
    A = np.array(acts)
    return {"success": first is not None,
            "jerk": float(np.mean(np.diff(A, n=2, axis=0) ** 2)) if len(A) > 2 else 0.0,
            "latency": first if first is not None else T_MAX}


def train_cfm(env, demos=24, epochs=120):
    O, A = [], []
    for ep in range(demos):
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
    last = None
    for _ in range(epochs):
        idx = torch.randint(0, n, (128,))
        o, a = O[idx], Af[idx]
        t = torch.rand(128, 1)
        e = torch.randn_like(a)
        loss = ((vf(t * a + (1 - t) * e, o, t) - (a - e)) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        last = float(loss.detach())
    print(f"train done: pairs={n} final_loss={last:.4f}", flush=True)
    return vf.eval(), O.shape[1], ad


def main():
    t0 = time.time()
    env = make_env(7)
    # LEARN: k=48 fresh train vs existing k=24 ckpt
    vf48, od, ad = train_cfm(env, demos=K48, epochs=EPOCHS)
    torch.save(vf48.state_dict(), os.path.join(REPO, "results/ckpt_push_cfm_k48.pt"))
    print(f"trained k={K48} ({time.time()-t0:.0f}s)", flush=True)
    vf24 = VF(od, ad)
    vf24.load_state_dict(torch.load(os.path.join(REPO, "results/ckpt_push_cfm.pt"),
                                    map_location="cpu", weights_only=True))
    vf24.eval()

    def chunk(vf):
        return lambda o: predict(vf, torch.from_numpy(
            o["observation"].astype(np.float32)), ad)

    out = {"seeds": N_SEED, "adapt": {}, "learn": {}}
    import pybullet as p
    for cond, setup in [("canonical", lambda: []),
                        ("novel", lambda: apply_novel(env))]:
        bids = setup()
        r = {}
        for name, fn in [("scripted", lambda o: np.tile(push_core(o), (H, 1))),
                         ("cfm24", chunk(vf24)), ("cfm48", chunk(vf48))]:
            S = [rollout(env, fn, SEED_BASE + s)["success"] for s in range(N_SEED)]
            r[name] = float(np.mean(S))
        # gated arm with fixed theta from prior calibration
        def gate_on(o, th=2.8):
            ch = chunk(vf24)(o)
            ej = float(np.mean(np.abs(np.diff(ch, n=2, axis=0))))
            return np.tile(push_core(o), (H, 1)) if ej > th else ch
        S = [rollout(env, gate_on, SEED_BASE + s)["success"] for s in range(N_SEED)]
        r["gated"] = float(np.mean(S))
        out["adapt"][cond] = r
        print(cond, r, flush=True)
        for b in bids:
            try:
                p.removeBody(b)
            except Exception:
                pass
    c, n = out["adapt"]["canonical"], out["adapt"]["novel"]
    out["adapt_gap"] = {k: round(c[k] - n[k], 3) for k in c}
    out["learn"] = {"k24_canon": out["adapt"]["canonical"]["cfm24"],
                    "k48_canon": out["adapt"]["canonical"]["cfm48"],
                    "k_demos": K48}
    env.close()
    mp = os.path.join(REPO, "results/metrics.json")
    d = json.load(open(mp))
    d["adapt_benchmark"] = out
    json.dump(d, open(mp, "w"), indent=1)
    print("adapt_gap:", out["adapt_gap"], "wrote metrics.json")


if __name__ == "__main__":
    main()
