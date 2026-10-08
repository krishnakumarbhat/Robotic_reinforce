"""Kernel G3S: SCREENING eval - 3 suites x tasks[0,1,2] x 3 eps = 27 rollouts.
Full 100-ep protocol costs ~15h/suite (measured 8.8min/rollout) - exceeds quota.
Screening gives directional signal in ~2.5h; full protocol goes to paid tier.
Evidence -> /kaggle/working/kernelG3S.json
"""
import json
import os
import re
import subprocess
import sys
import time

t0 = time.time()
WORK = "/kaggle/working" if os.path.exists("/kaggle") else "."
REPO = "krishnah27/smolvla-libero-g3-10k"
RENAME = ("{\"observation.images.image\": \"observation.images.camera1\", "
          "\"observation.images.image2\": \"observation.images.camera2\"}")
TASKS = "all10"
N_EPS = 1  # 10 tasks x 1 ep per suite = 30 rollouts ~4.5h (no task_id flag needed)
out: dict = {"repo": REPO, "n_eps": N_EPS, "suites": {}}


def log(m):
    print(m, flush=True)


def pip(pkgs):
    r = subprocess.run([sys.executable, "-m", "pip", "install", "-q"] + pkgs,
                       capture_output=True, text=True)
    return r.returncode


log(f"base pip rc={pip(['lerobot[dataset]', 'av', 'num2words', 'libero'])}")
log(f"pin mujoco316 rc={pip(['mujoco==3.1.6'])}")

from huggingface_hub import snapshot_download  # noqa: E402
try:
    snap = snapshot_download(REPO)
    out["ckpt"] = True
    log(f"ckpt ok: {snap}")
except Exception as e:  # noqa: BLE001
    out["ckpt"] = False
    out["abort"] = f"ckpt: {type(e).__name__}: {str(e)[:150]}"
    log(out["abort"])

if out.get("ckpt"):
    for suite in ["libero_spatial", "libero_object", "libero_goal"]:
        cmd = [sys.executable, "-m", "lerobot.scripts.lerobot_eval",
               f"--policy.path={snap}", "--env.type=libero",
               f"--env.task={suite}", "--eval.batch_size=1",
               f"--eval.n_episodes={N_EPS}", f"--rename_map={RENAME}"]
        try:
            p = subprocess.run(cmd, input="N\n", capture_output=True,
                               text=True, timeout=8000)
            full = p.stdout + p.stderr
            with open(os.path.join(WORK, f"screen_{suite}.txt"), "w") as f:
                f.write(full)
            m = re.findall(r"'pc_success': ([0-9.]+)", full)
            out["suites"][suite] = {"rc": p.returncode,
                                    "pc_success": m[-1] if m else None}
            log(f"{suite}: {out['suites'][suite]}")
        except subprocess.TimeoutExpired:
            out["suites"][suite] = {"rc": "timeout"}
            log(f"{suite}: TIMEOUT")
out["wall_s"] = round(time.time() - t0, 1)
log(f"G3S_DONE wall={out['wall_s']}")
with open(os.path.join(WORK, "kernelG3S.json"), "w") as f:
    json.dump(out, f)
