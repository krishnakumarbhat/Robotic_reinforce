"""Kernel F: NOVEL-dynamics comparison — scripted vs CFM k24 vs gated k24 on PushCube-v1 (T4).
Canonical vs NOVEL (cylinder obstacle if scene API allows, else friction/gain+noise fallback;
applied novelty is logged honestly). Fast like E (~1-2 min). Evidence -> /kaggle/working/kernelF.json
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
print("dev:", DEV, flush=True)
H, N_EVAL, T_MAX = 8, 16, 100
out: dict = {"stages": {}}


def flat(o):
    return np.asarray(o, dtype=np.float32).reshape(-1)


def main():
    env = gym.make("PushCube-v1", obs_mode="state",
                   control_mode="pd_ee_target_delta_pos", render_mode="none")
    uw = env.unwrapped
    obs_dim = int(np.asarray(env.reset(seed=0)[0]).size)
    act_dim = int(np.asarray(env.action_space.sample()).size)
    out["dims"] = {"obs": obs_dim, "act": act_dim, "dev": DEV}

    cands = [("agent.tcp.pose.p", lambda: uw.agent.tcp.pose.p),
             ("agent.robot.tcp.pose.p", lambda: uw.agent.robot.tcp.pose.p),
             ("agent.ee_pose.p", lambda: uw.agent.ee_pose.p)]
    ee_fn = next((fn for _, fn in cands
                  if _ok(fn)), None)
    if ee_fn is None:
        out["abort"] = "no EE pose accessor"
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

    # --- train CFM k24 on CANONICAL (same as E) ---
    class VF(nn.Module):
        def __init__(self, D):
            super().__init__()
            self.n = nn.Sequential(nn.Linear(H * act_dim + D + 1, 128), nn.SiLU(),
                                   nn.Linear(128, 128), nn.SiLU(),
                                   nn.Linear(128, H * act_dim))

        def forward(self, x, o, tt):
            return self.n(torch.cat([x, o, tt], -1))

    O, A = [], []
    for ep in range(24):
        obs, _ = env.reset(seed=2500 + ep)
        acts = []
        for _ in range(60):
            a = core(flat(obs))
            obs, _, term, trunc, _ = env.step(a)
            acts.append(a)
            if term or trunc:
                break
        acts = np.array(acts)
        obs, _ = env.reset(seed=2500 + ep)
        for i in range(len(acts)):
            O.append(flat(obs))
            c = acts[i:i + H]
            if len(c) < H:
                c = np.vstack([c, np.tile(c[-1], (H - len(c), 1))])
            A.append(c.astype(np.float32))
            obs, _, term, trunc, _ = env.step(core(flat(obs)))
            if term or trunc:
                obs, _ = env.reset(seed=2500 + ep)
    O = torch.from_numpy(np.array(O)).to(DEV)
    Af = torch.from_numpy(np.array(A)).view(len(A), -1).to(DEV)
    vf = VF(O.shape[1]).to(DEV)
    opt = torch.optim.Adam(vf.parameters(), lr=3e-3)
    n = len(O)
    for _ in range(120):
        idx = torch.randint(0, n, (128,))
        o, a = O[idx], Af[idx]
        tt = torch.rand(128, 1, device=DEV)
        e = torch.randn_like(a)
        loss = ((vf(tt * a + (1 - tt) * e, o, tt) - (a - e)) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    vf.eval()
    out["stages"]["train"] = {"pairs": n, "loss": float(loss.detach())}
    print(f"trained pairs={n} loss={float(loss.detach()):.4f}", flush=True)

    @torch.no_grad()
    def cfm_chunk(o):
        x = torch.randn(H * act_dim, device=DEV)
        ot = torch.from_numpy(o).to(DEV)
        for kk in range(10):
            tt = torch.full((1, 1), kk / 10, device=DEV)
            x = x + 0.1 * vf(x[None], ot[None], tt)[0]
        return x.view(H, act_dim).cpu().numpy()

    energies = []
    for s in range(4):
        obs, _ = env.reset(seed=9700 + s)
        for _ in range(6):
            ch = cfm_chunk(flat(obs))
            energies.append(float(np.mean(np.abs(np.diff(ch, n=2, axis=0)))))
            obs, _, term, trunc, _ = env.step(core(flat(obs)))
            if term or trunc:
                break
    theta = float(np.percentile(energies, 75))
    vet = {"n": 0, "d": 0}

    def gate_on(o):
        ch = cfm_chunk(o)
        vet["d"] += 1
        if float(np.mean(np.abs(np.diff(ch, n=2, axis=0)))) > theta:
            vet["n"] += 1
            return np.tile(core(o), (H, 1))
        return ch

    # --- NOVEL condition: obstacle if scene API allows, else gain+noise fallback ---
    novelty = "none"
    try:
        scene = uw.scene
        b = scene.create_actor_builder()
        b.add_cylinder_collision(radius=0.02, half_length=0.08)
        b.add_cylinder_visual(radius=0.02, half_length=0.08)
        obst = b.build_kinematic(name="novel_cyl")
        obst.set_pose([[1, 0, 0, 0], [0, 1, 0, 0.05], [0, 0, 1, 0.08], [0, 0, 0, 1]])
        novelty = "cylinder-obstacle"
    except Exception as e:  # noqa: BLE001
        novelty = f"fallback-gain-noise ({type(e).__name__})"
    out["novelty"] = novelty
    print("novelty:", novelty, flush=True)

    def rollout(policy, seed, novel):
        obs, _ = env.reset(seed=seed)
        acts, first = [], None
        rng = np.random.default_rng(seed)
        for t in range(T_MAX):
            o = flat(obs)
            if novel and novelty.startswith("fallback"):
                o = o + rng.normal(0, 0.01, o.shape).astype(np.float32)
            a = np.asarray(policy(o), dtype=np.float32)
            a = a[0] if a.ndim == 2 else a  # chunk -> first action (cf. Kernel E)
            if novel and novelty.startswith("fallback"):
                a = np.clip(a * 0.6, -1, 1)  # halved gain = slippery proxy
            acts.append(a)
            obs, _, term, trunc, info = env.step(a)
            if first is None and bool(info.get("success", False)):
                first = t + 1
            if term or trunc:
                break
        return {"success": first is not None,
                "latency": first if first is not None else T_MAX}

    res = {}
    for cond, novel in [("canonical", False), ("novel", True)]:
        for name, fn in [("scripted", lambda o: np.tile(core(o), (H, 1))),
                         ("cfm_k24", cfm_chunk), ("gated_k24", gate_on)]:
            R = [rollout(fn, (8000 if novel else 7000) + s, novel)
                 for s in range(N_EVAL)]
            res[f"{cond}/{name}"] = {
                "success": float(np.mean([x["success"] for x in R])),
                "latency": float(np.mean([x["latency"] for x in R]))}
            print(cond, name, res[f"{cond}/{name}"], flush=True)
    res["gate"] = {"theta": theta, "veto_rate": vet["n"] / max(1, vet["d"])}
    res["adapt_gap_scripted"] = res["canonical/scripted"]["success"] - res["novel/scripted"]["success"]
    res["adapt_gap_gated"] = res["canonical/gated_k24"]["success"] - res["novel/gated_k24"]["success"]
    out["results"] = res
    env.close()


def _ok(fn):
    try:
        p = np.asarray(fn(), dtype=np.float64).reshape(-1)[:3]
        return p.shape == (3,) and np.all(np.isfinite(p))
    except Exception:  # noqa: BLE001
        return False


try:
    main()
except SystemExit as e:
    out["exit"] = e.code
out["wall_s"] = round(time.time() - t0, 1)
print(json.dumps(out.get("results", {}), indent=1)[:1200])
with open("/kaggle/working/kernelF.json", "w") as f:
    json.dump(out, f)
print("KERNELF_DONE")
