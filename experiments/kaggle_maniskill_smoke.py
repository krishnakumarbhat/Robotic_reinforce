"""Kaggle T4 smoke: prove the remote ManiSkill pipeline end-to-end.
Installs mani_skill, runs random-action baseline on PickCube-v1 headless
(repo-documented pattern: demo_random_action --render-mode none),
writes JSON metrics. Follow-up iters port the gated policies here.
"""
import json
import subprocess
import sys
import time

t0 = time.time()
r = subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                    "mani_skill>=3.0.0b20"], capture_output=True, text=True)
print("pip maniskill rc:", r.returncode, flush=True)
if r.returncode != 0:
    print(r.stderr[-1000:])
    raise SystemExit(2)

import torch
import gymnasium as gym  # noqa: E402
import mani_skill  # noqa: E402,F401

print("torch:", torch.__version__, "cuda:", torch.cuda.is_available(), flush=True)
env = gym.make("PickCube-v1", obs_mode="state",
               control_mode="pd_ee_target_delta_pose", render_mode="none")
N, succ, steps = 10, 0, 0
for ep in range(N):
    obs, _ = env.reset(seed=7000 + ep)
    done, term, info = False, False, {}
    while not done:
        obs, _, term, trunc, info = env.step(env.action_space.sample())
        steps += 1
        done = bool(term or trunc)
    if bool(info.get("success", term)):
        succ += 1
env.close()
out = {"task": "PickCube-v1", "policy": "random",
       "episodes": N, "success": succ / N, "steps": steps,
       "wall_s": round(time.time() - t0, 1),
       "torch_cuda": bool(torch.cuda.is_available())}
print(json.dumps(out, indent=1))
with open("/kaggle/working/smoke.json", "w") as f:
    json.dump(out, f)
print("SMOKE_DONE")
