"""Kaggle kernel A: ManiSkill API introspection + scripted/random baselines.
Proves task API mapping (actors, goals, obs/action dims) before the full
gated-policy confirmation kernel B. Tasks: PushCube-v1 + PickCube-v1.
All findings dumped to JSON (diagnosable, never silent).
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
import gymnasium as gym  # noqa: E402
import mani_skill  # noqa: E402,F401

print("torch:", torch.__version__, "cuda:", torch.cuda.is_available(), flush=True)
out = {"tasks": {}}

for task in ["PushCube-v1", "PickCube-v1"]:
    d: dict = {"introspection": {}}
    try:
        env = gym.make(task, obs_mode="state", control_mode="pd_ee_target_delta_pos",
                       render_mode="none")
        d["introspection"]["obs_space"] = str(env.observation_space)
        d["introspection"]["act_space"] = str(env.action_space)
        uw = env.unwrapped
        attrs = [a for a in dir(uw) if not a.startswith("_")]
        d["introspection"]["unwrapped_attrs"] = attrs[:60]
        found = {}
        for name in ["cube", "goal_site", "goal_region", "target", "obj", "goal"]:
            if hasattr(uw, name):
                try:
                    o = getattr(uw, name)
                    p = getattr(getattr(o, "pose", o), "p", None)
                    found[name] = (np.asarray(p).tolist()
                                   if p is not None else type(o).__name__)
                except Exception as e:  # noqa: BLE001
                    found[name] = "ERR:" + str(e)[:80]
        d["introspection"]["goal_actors"] = found
        o0, info0 = env.reset(seed=0)
        d["introspection"]["reset_info_keys"] = list(info0.keys())
        d["introspection"]["obs_dim"] = int(np.asarray(o0).size)
        d["introspection"]["act_dim"] = int(np.asarray(env.action_space.sample()).size)

        # random baseline, 6 eps x 80 steps
        succ, n = 0, 6
        for ep in range(n):
            obs, info = env.reset(seed=8000 + ep)[0], {}
            for _ in range(80):
                obs, _, term, trunc, info = env.step(env.action_space.sample())
                if term or trunc:
                    break
            succ += bool(info.get("success", False))
        d["random"] = {"success": succ / n}
        env.close()
    except Exception as e:  # noqa: BLE001
        import traceback
        d["error"] = traceback.format_exc()[-600:]
    out["tasks"][task] = d

out["wall_s"] = round(time.time() - t0, 1)
print(json.dumps(out, indent=1)[:3000])
with open("/kaggle/working/kernelA.json", "w") as f:
    json.dump(out, f)
print("KERNELA_DONE")
