"""Colab lane: LIBERO eval of v19 20k-ckpt (standard suites + noise-perturb subset).
Waits for krishnah27/smolvla-pusht-v20k on the Hub. Run via: colab exec -s research -f THIS --timeout 2700
Evidence printed as JSON (copy to results/libero_v19.json).
"""
import json
import os
import subprocess
import sys
import time

t0 = time.time()
REPO = "krishnah27/smolvla-pusht-v20k"
out: dict = {"repo": REPO, "suites": {}}


def log(m):
    print(m, flush=True)


r = subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                    "lerobot[dataset]", "libero", "mujoco"],
                   capture_output=True, text=True)
log(f"pip rc={r.returncode}")
if r.returncode != 0:
    log((r.stdout + r.stderr)[-500:])

from huggingface_hub import snapshot_download  # noqa: E402
try:
    snap = snapshot_download(REPO)
    log(f"ckpt OK: {snap}")
    out["ckpt"] = True
except Exception as e:  # noqa: BLE001
    out["ckpt"] = False
    out["abort"] = f"ckpt download: {type(e).__name__}: {str(e)[:200]}"
    log(out["abort"])

if out.get("ckpt"):
    import torch  # noqa: E402
    log(f"cuda={torch.cuda.is_available()}")
    for suite in ["libero_spatial", "libero_object", "libero_goal"]:
        cmd = [sys.executable, "-m", "lerobot.scripts.lerobot_eval",
               f"--policy.path={snap}", "--env.type=libero",
               f"--env.task={suite}", "--eval.batch_size=1",
               "--eval.n_episodes=10"]
        log(f"eval {suite} ...")
        try:
            p = subprocess.run(cmd, capture_output=True, text=True, timeout=1500)
            tail = (p.stdout + p.stderr)[-600:]
            out["suites"][suite] = {"rc": p.returncode, "tail": tail}
            log(f"{suite} rc={p.returncode} :: {tail[-200:]}")
        except subprocess.TimeoutExpired:
            out["suites"][suite] = {"rc": "timeout"}
            log(f"{suite} TIMEOUT")
out["wall_s"] = round(time.time() - t0, 1)
print("LIBERO_V19 " + json.dumps(out)[:3000], flush=True)
