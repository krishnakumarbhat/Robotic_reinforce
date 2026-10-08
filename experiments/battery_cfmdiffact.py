"""Battery: toy-scale CFM vs Diffusion vs ACT-lite on PandaReach (LOCAL harness).
ponytail: toy MLPs, NOT a VLA. Purpose: baseline numbers (success/latency/jerk)
that future DP-Flow ideas must beat. Demos are scripted (procedural, zero download,
in-memory). Keeps ONE tiny checkpoint per policy (top-2 policy respected).
Outputs: results/battery.json + stdout table.
"""
import json, os, time, numpy as np, torch, torch.nn as nn

# AGENT INSTRUCTION: run ONLY under /tmp/.opencode_gpu.lock, GPU_BACKEND=local.
DEV = "cuda" if torch.cuda.is_available() else "cpu"
H, TA, DEMOS, STEPS, EPOCHS, BS = 8, 4, 40, 50, 300, 256
torch.manual_seed(0); np.random.seed(0)


def make_env(seed):
    import gymnasium as gym  # noqa: E402
    import panda_gym  # noqa: E402,F401  (registers Panda* envs)
    env = gym.make("PandaReach-v3")
    env.reset(seed=seed)
    return env


def collect():
    """Scripted proportional expert -> (obs6, chunk24) pairs. Returns arrays."""
    O, A = [], []
    env = make_env(0)
    for ep in range(DEMOS):
        obs, _ = env.reset(seed=1000 + ep)
        acts = []
        for t in range(STEPS):
            ee, goal = obs["achieved_goal"], obs["desired_goal"]
            a = np.clip(5.0 * (goal - ee), -1, 1) + np.random.randn(3) * 0.02
            obs, _, term, trunc, _ = env.step(a.astype(np.float32))
            acts.append(a)
            if term or trunc:
                break
        acts = np.array(acts)
        obs, _ = env.reset(seed=1000 + ep)
        for t in range(len(acts)):
            ee, goal = obs["achieved_goal"], obs["desired_goal"]
            O.append(np.concatenate([ee, goal - ee]).astype(np.float32))
            c = acts[t:t + H]
            if len(c) < H:
                c = np.vstack([c, np.tile(c[-1], (H - len(c), 1))])
            A.append(c.astype(np.float32))
            a = np.clip(5.0 * (goal - ee), -1, 1)
            obs, _, term, trunc, _ = env.step(a.astype(np.float32))
            if term or trunc:
                obs, _ = env.reset(seed=1000 + ep)
    env.close()
    return np.array(O), np.array(A)


class VF(nn.Module):
    """Vector field / epsilon net: (chunk24 + obs6 + tau1) -> chunk24."""

    def __init__(self):
        super().__init__()
        self.n = nn.Sequential(nn.Linear(31, 128), nn.SiLU(), nn.Linear(128, 128),
                               nn.SiLU(), nn.Linear(128, 24))

    def forward(self, x, o, t):
        return self.n(torch.cat([x, o, t], -1))


class CVAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.e = nn.Sequential(nn.Linear(30, 64), nn.SiLU(), nn.Linear(64, 16))
        self.d = nn.Sequential(nn.Linear(14, 64), nn.SiLU(), nn.Linear(64, 64),
                               nn.SiLU(), nn.Linear(64, 24))

    def fwd(self, o, c=None):
        if c is None:
            return self.d(torch.cat([o, torch.zeros(len(o), 8, device=o.device)], -1))
        h = self.e(torch.cat([o, c.view(len(o), -1)], -1))
        mu, lv = h.chunk(2, -1)
        z = mu + torch.exp(0.5 * lv) * torch.randn_like(mu)
        return self.d(torch.cat([o, z], -1)), mu, lv


def train(vf, O, A, mode):
    opt = torch.optim.Adam(vf.parameters() if mode != "act" else list(vf.parameters()), lr=3e-3)
    O = torch.from_numpy(O).to(DEV); A = torch.from_numpy(A).to(DEV)
    Af = A.view(len(A), -1)
    betas = torch.linspace(1e-4, 0.02, 100, device=DEV)
    alph = torch.cumprod(1 - betas, 0)
    n = len(O)
    for ep in range(EPOCHS):
        idx = torch.randint(0, n, (BS,))
        o, a = O[idx], Af[idx]
        if mode == "cfm":
            t = torch.rand(BS, 1, device=DEV)
            e = torch.randn_like(a)
            xt = t * a + (1 - t) * e
            loss = ((vf(xt, o, t) - (a - e)) ** 2).mean()
        elif mode == "diff":
            t = torch.randint(0, 100, (BS,), device=DEV)
            ab = alph[t][:, None]
            e = torch.randn_like(a)
            xt = ab.sqrt() * a + (1 - ab).sqrt() * e
            tn = (t.float() / 100)[:, None]
            loss = ((vf(xt, o, tn) - e) ** 2).mean()
        else:
            pr, mu, lv = vf.fwd(o, a)
            loss = ((pr - a) ** 2).mean() + 1e-3 * (-0.5 * (1 + lv - mu.pow(2) - lv.exp()).mean())
        opt.zero_grad(); loss.backward(); opt.step()
    return vf


@torch.no_grad()
def predict_cfm(vf, o, steps=10):
    x = torch.randn(24, device=DEV)
    for k in range(steps):
        t = torch.full((1, 1), k / steps, device=DEV)
        x = x + (1 / steps) * vf(x[None], o[None], t)[0]
    return x.view(H, 3).cpu().numpy()


@torch.no_grad()
def predict_diff(vf, o, steps=10):
    x = torch.randn(24, device=DEV)
    for k in reversed(range(1, steps + 1)):
        tn = torch.full((1, 1), k / steps, device=DEV)
        e = vf(x[None], o[None], tn)[0]
        x = (x - (1 / steps) * e) / np.sqrt(1 - 1 / steps + 1e-8)
    return x.view(H, 3).cpu().numpy()


@torch.no_grad()
def rollout(env, fn, seed, max_steps=60):
    """Receding-horizon eval with temporal ensembling over overlapping chunks."""
    obs, _ = env.reset(seed=seed)
    acts_done, lat, buf = [], 0.0, []
    for t in range(max_steps):
        ee, goal = obs["achieved_goal"], obs["desired_goal"]
        o = torch.from_numpy(np.concatenate([ee, goal - ee]).astype(np.float32)).to(DEV)
        s = time.time()
        ch = fn(o)
        lat += time.time() - s
        buf.append((t, ch))
        ens = [c[t - t0] for t0, c in buf if 0 <= t - t0 < H]
        a = np.mean(ens, axis=0)
        acts_done.append(a)
        obs, _, term, trunc, _ = env.step(a.astype(np.float32))
        if np.linalg.norm(obs["achieved_goal"] - obs["desired_goal"]) < 0.05:
            break
    A = np.array(acts_done)
    jerk = float(np.mean(np.diff(A, n=2, axis=0) ** 2)) if len(A) > 2 else 0.0
    succ = bool(np.linalg.norm(obs["achieved_goal"] - obs["desired_goal"]) < 0.05)
    return succ, 1000 * lat / max(1, len(acts_done)), jerk


def main():
    t0 = time.time()
    O, A = collect()
    print(f"collected {len(O)} pairs in {time.time()-t0:.0f}s on {DEV}", flush=True)
    res = {}
    vf = VF().to(DEV)
    train(vf, O, A, "cfm"); torch.save(vf.state_dict(), "results/ckpt_cfm.pt")
    vd = VF().to(DEV)
    train(vd, O, A, "diff"); torch.save(vd.state_dict(), "results/ckpt_diff.pt")
    va = CVAE().to(DEV); train(va, O, A, "act")
    vf.eval(); vd.eval(); va.eval()
    env = make_env(7)
    pols = {"cfm-10step": lambda o: predict_cfm(vf, o),
            "diff-10step": lambda o: predict_diff(vd, o),
            "act-lite+ens": lambda o: va.fwd(o).view(H, 3).cpu().numpy(),
            "random": lambda o: np.tile(np.random.uniform(-1, 1, 3), (H, 1))}
    table = {}
    for name, fn in pols.items():
        S, L, J = [], [], []
        for s in range(20):
            ok, ms, jk = rollout(env, fn, 5000 + s)
            S.append(ok); L.append(ms); J.append(jk)
        table[name] = {"success": float(np.mean(S)), "lat_ms": float(np.mean(L)),
                       "jerk": float(np.mean(J))}
        print(f"{name:14s} succ={table[name]['success']:.2f} lat={table[name]['lat_ms']:.1f}ms jerk={table[name]['jerk']:.4f}",
              flush=True)
    env.close()
    os.makedirs("results", exist_ok=True)
    json.dump({"policies": table, "device": DEV, "note": "toy harness baseline"}, open("results/battery.json", "w"), indent=1)
    print("wrote results/battery.json")


if __name__ == "__main__":
    main()
