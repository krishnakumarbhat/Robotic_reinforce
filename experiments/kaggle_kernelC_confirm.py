"""Kaggle kernel C: CONFIRMATION queue v2 (bigger N, shifted seeds) — scripted-core vs CFM-raw vs N42-gated
on PushCube-v1 (T4). Self-debugging: discovers EE pose API, ladders scripted
gain until demos succeed (>=3/8) else aborts arms with logged reason, trains
CFM on-device, calibrates gate theta, evaluates 3 arms x 8 eps. All evidence
 dumped to JSON (never silent, never fabricated).
"""
import json
import subprocess
import sys
import time

import numpy as np

t0 = time.time()
r = subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                    "mani_skill>=3.0.0b20"], capture_output=True, text=True)
print("pip rc:", r.returncode, flush=True)
if r.returncode != 0:
    print(r.stderr[-800:])
    raise SystemExit(2)

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import gymnasium as gym  # noqa: E402
import mani_skill  # noqa: E402,F401

DEV = "cuda" if torch.cuda.is_available() else "cpu"
print("dev:", DEV, torch.__version__, flush=True)
H, N_DEMO, N_EVAL, T_MAX = 8, 24, 12, 100
out: dict = {"stages": {}}


def flat(o):
    return np.asarray(o, dtype=np.float32).reshape(-1)


def find_ee(uw):
    """Return fn() -> ee xyz, trying known ManiSkill TCP accessors."""
    cands = [("agent.tcp.pose.p", lambda: uw.agent.tcp.pose.p),
             ("agent.robot.tcp.pose.p", lambda: uw.agent.robot.tcp.pose.p),
             ("agent.ee_pose.p", lambda: uw.agent.ee_pose.p),
             ("tcp.pose.p", lambda: uw.tcp.pose.p)]
    for name, fn in cands:
        try:
            p = np.asarray(fn(), dtype=np.float64).reshape(-1)[:3]
            if p.shape == (3,) and np.all(np.isfinite(p)):
                return name, fn
        except Exception:  # noqa: BLE001
            continue
    return None, None


def main():
    env = gym.make("PushCube-v1", obs_mode="state",
                   control_mode="pd_ee_target_delta_pos", render_mode="none")
    uw = env.unwrapped
    obs_dim = int(np.asarray(env.reset(seed=0)[0]).size)
    act_dim = int(np.asarray(env.action_space.sample()).size)
    out["dims"] = {"obs": obs_dim, "act": act_dim, "dev": DEV}
    ee_name, ee_fn = find_ee(uw)
    out["stages"]["ee_discovery"] = {"winner": ee_name}
    print("ee accessor:", ee_name, flush=True)
    if ee_fn is None:
        out["abort"] = "no EE pose accessor; see ee_discovery"
        raise SystemExit(3)

    def poses():
        return (np.asarray(ee_fn(), dtype=np.float64).reshape(-1)[:3],
                np.asarray(uw.obj.pose.p, dtype=np.float64).reshape(-1)[:3],
                np.asarray(uw.goal_region.pose.p, dtype=np.float64).reshape(-1)[:3])

    def core(o, gain=6.0):
        ee, cube, goal = poses()
        to_goal = goal - cube
        d = float(np.linalg.norm(to_goal))
        if d < 0.025:
            wp = goal + np.array([0.0, 0.0, 0.05])
        else:
            u = to_goal / max(d, 1e-9)
            behind = cube - u * 0.05
            behind[2] = cube[2]
            if float(np.linalg.norm(ee - behind)) > 0.03:
                wp = behind
            else:
                wp = cube + u * 0.08
                wp[2] = cube[2] + 0.005
        a = np.clip(gain * (wp - ee), -1, 1)
        return np.concatenate([a[:3], [0.0]])[:act_dim].astype(np.float32)

    def rollout(policy, seed):
        obs, _ = env.reset(seed=seed)
        acts, first = [], None
        for t in range(T_MAX):
            a = np.asarray(policy(flat(obs)), dtype=np.float32)
            acts.append(a)
            obs, _, term, trunc, info = env.step(a)
            if first is None and bool(info.get("success", False)):
                first = t + 1
            if term or trunc:
                break
        A = np.array(acts)
        jerk = float(np.mean(np.diff(A, n=2, axis=0) ** 2)) if len(A) > 2 else 0.0
        return {"success": first is not None, "jerk": jerk,
                "latency": first if first is not None else T_MAX}

    # gain ladder: first gain with demo succ >= 3/8 wins
    best, demo_ok = None, 0
    for g in [3.0, 6.0, 12.0]:
        s = sum(rollout(lambda o, gg=g: core(o, gg), 9500 + i)["success"]
                for i in range(8))
        print(f"gain {g}: demo succ {s}/8", flush=True)
        if s >= 3:
            best, demo_ok = g, s
            break
    out["stages"]["gain_ladder"] = {"gain": best, "demo_succ": demo_ok}
    if best is None:
        out["abort"] = "scripted core incompetent at all gains; arms skipped honestly"
        raise SystemExit(4)

    # demos + on-device CFM
    O, A = [], []
    for ep in range(N_DEMO):
        obs, _ = env.reset(seed=1500 + ep)
        acts = []
        for _ in range(60):
            a = core(flat(obs), best)
            obs, _, term, trunc, _ = env.step(a)
            acts.append(a)
            if term or trunc:
                break
        acts = np.array(acts)
        obs, _ = env.reset(seed=1500 + ep)
        for i in range(len(acts)):
            O.append(flat(obs))
            c = acts[i:i + H]
            if len(c) < H:
                c = np.vstack([c, np.tile(c[-1], (H - len(c), 1))])
            A.append(c.astype(np.float32))
            obs, _, term, trunc, _ = env.step(core(flat(obs), best))
            if term or trunc:
                obs, _ = env.reset(seed=1500 + ep)
    O = torch.from_numpy(np.array(O)).to(DEV)
    Af = torch.from_numpy(np.array(A)).view(len(A), -1).to(DEV)
    D = O.shape[1]

    class VF(nn.Module):
        def __init__(self):
            super().__init__()
            self.n = nn.Sequential(nn.Linear(H * act_dim + D + 1, 128), nn.SiLU(),
                                   nn.Linear(128, 128), nn.SiLU(),
                                   nn.Linear(128, H * act_dim))

        def forward(self, x, o, t):
            return self.n(torch.cat([x, o, t], -1))

    vf = VF().to(DEV)
    opt = torch.optim.Adam(vf.parameters(), lr=3e-3)
    n = len(O)
    for _ in range(150):
        idx = torch.randint(0, n, (128,))
        o, a = O[idx], Af[idx]
        t = torch.rand(128, 1, device=DEV)
        e = torch.randn_like(a)
        loss = ((vf(t * a + (1 - t) * e, o, t) - (a - e)) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    vf.eval()
    out["stages"]["train"] = {"pairs": n, "final_loss": float(loss.detach())}
    print("trained on", n, "pairs loss", round(float(loss.detach()), 4), flush=True)

    @torch.no_grad()
    def cfm_chunk(o):
        x = torch.randn(H * act_dim, device=DEV)
        ot = torch.from_numpy(o).to(DEV)
        for k in range(10):
            t = torch.full((1, 1), k / 10, device=DEV)
            x = x + 0.1 * vf(x[None], ot[None], t)[0]
        return x.view(H, act_dim).cpu().numpy()

    energies = []
    for s in range(4):
        obs, _ = env.reset(seed=9600 + s)
        for _ in range(6):
            ch = cfm_chunk(flat(obs))
            ej = float(np.mean(np.abs(np.diff(ch, n=2, axis=0))))
            energies.append(ej)
            obs, _, term, trunc, _ = env.step(core(flat(obs), best))
            if term or trunc:
                break
    theta = float(np.percentile(energies, 75))
    vet = {"n": 0, "d": 0}

    def gate_on(o):
        ch = cfm_chunk(o)
        vet["d"] += 1
        if float(np.mean(np.abs(np.diff(ch, n=2, axis=0)))) > theta:
            vet["n"] += 1
            return np.tile(core(o, best), (H, 1))
        return ch

    res = {}
    for name, fn in [("scripted-core", lambda o: np.tile(core(o, best), (H, 1))),
                     ("cfm-raw", cfm_chunk), ("n42-gated", gate_on)]:
        R = []
        for s in range(N_EVAL):
            # receding horizon: fresh chunk per step, chunk[0] executes
            obs, _ = env.reset(seed=6000 + s)
            acts2, first2 = [], None
            for t in range(T_MAX):
                ch = fn(flat(obs))
                a = np.asarray(ch[0] if ch.ndim == 2 else ch, dtype=np.float32)
                acts2.append(a)
                obs, _, term, trunc, info = env.step(a)
                if first2 is None and bool(info.get("success", False)):
                    first2 = t + 1
                if term or trunc:
                    break
            A2 = np.array(acts2)
            R.append({"success": first2 is not None,
                      "jerk": float(np.mean(np.diff(A2, n=2, axis=0) ** 2))
                      if len(A2) > 2 else 0.0,
                      "latency": first2 if first2 is not None else T_MAX})
        res[name] = {"success": float(np.mean([r["success"] for r in R])),
                     "jerk": float(np.mean([r["jerk"] for r in R])),
                     "latency": float(np.mean([r["latency"] for r in R]))}
        print(name, res[name], flush=True)
    res["gate"] = {"theta": theta, "veto_rate": vet["n"] / max(1, vet["d"]),
                   "delta": res["n42-gated"]["success"] - res["cfm-raw"]["success"]}
    out["results"] = res
    env.close()


try:
    main()
except SystemExit as e:
    out["exit"] = e.code
out["wall_s"] = round(time.time() - t0, 1)
print(json.dumps({k: v for k, v in out.items() if k != "stages"},
                 indent=1)[:1500])
with open("/kaggle/working/kernelC.json", "w") as f:
    json.dump(out, f)
print("KERNELC_DONE")
