"""Kernel G1: LIBERO eval of OUR v19 20k-ckpt (3 suites x 10 eps, T4).
Ckpt from HF krishnah27/smolvla-pusht-v20k (no gates). standart suites only;
perturbations = separate LIBERO-PRO kernel. Evidence -> /kaggle/working/kernelG1.json
"""
import json
import os
import subprocess
import sys
import time

t0 = time.time()
WORK = "/kaggle/working" if os.path.exists("/kaggle") else "."
REPO = "krishnah27/smolvla-pusht-v20k"
out: dict = {"repo": REPO, "suites": {}}


def log(m):
    print(m, flush=True)


r = subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                    "lerobot[dataset]", "libero", "mujoco", "av", "num2words"],
                   capture_output=True, text=True)
log(f"pip rc={r.returncode}")

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
               "--eval.n_episodes=10"]
        log(f"eval {suite} ...")
        try:
            # LIBERO first-import prompts "custom path? (Y/N)"; answer N headlessly
            p = subprocess.run(cmd, input="N\n", capture_output=True,
                               text=True, timeout=2400)
            tail = (p.stdout + p.stderr)[-800:]
            # harvest success numbers from tail
            import re
            nums = re.findall(r"(\d+\.?\d*)\s*%", tail)
            out["suites"][suite] = {"rc": p.returncode, "pct_hits": nums[-4:],
                                    "tail": tail[-400:]}
            log(f"{suite} rc={p.returncode} pct={nums[-4:]}")
        except subprocess.TimeoutExpired:
            out["suites"][suite] = {"rc": "timeout"}
            log(f"{suite} TIMEOUT")
out["wall_s"] = round(time.time() - t0, 1)
log(f"G1_DONE wall={out['wall_s']}")
with open(os.path.join(WORK, "kernelG1.json"), "w") as f:
    json.dump(out, f)
