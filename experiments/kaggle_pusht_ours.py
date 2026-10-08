"""Kernel G2: PushT eval of OUR v19 20k-ckpt (in-distribution check, 20 eps, T4).
G1 proved LIBERO-zero-shot is invalid for a PushT-tuned policy (embodiment
mismatch: 1cam/s2/a2 vs 2cam/s8/a7) — so evaluate where it trained.
Evidence -> /kaggle/working/kernelG2.json
"""
import json
import os
import re
import subprocess
import sys
import time

t0 = time.time()
WORK = "/kaggle/working" if os.path.exists("/kaggle") else "."
REPO = "krishnah27/smolvla-pusht-v20k"
out: dict = {"repo": REPO}


def log(m):
    print(m, flush=True)


r = subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                    "lerobot[dataset]", "av", "num2words", "gym-pusht",
                    "pymunk<7"],  # gym-pusht needs pymunk 6.x Space.add_collision_handler
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
    cmd = [sys.executable, "-m", "lerobot.scripts.lerobot_eval",
           f"--policy.path={snap}", "--env.type=pusht",
           "--eval.batch_size=1", "--eval.n_episodes=20",
           "--rename_map={\"observation.image\": \"observation.images.camera1\"}"]
    log("eval pusht x20 ...")
    try:
        p = subprocess.run(cmd, input="N\n", capture_output=True,
                           text=True, timeout=2400)
        full = p.stdout + p.stderr
        with open(os.path.join(WORK, "eval_full.txt"), "w") as f:
            f.write(full)
        tail = full[-1000:]
        nums = re.findall(r"(\d+\.?\d*)\s*%", tail)
        succ = re.findall(r"[Ss]uccess[^0-9]{0,20}(\d+\.?\d*)", full)
        out["eval"] = {"rc": p.returncode, "pct_hits": nums[-6:],
                       "succ_hits": succ[-6:], "tail": tail[-500:]}
        log(f"rc={p.returncode} pct={nums[-6:]}")
    except subprocess.TimeoutExpired:
        out["eval"] = {"rc": "timeout"}
        log("TIMEOUT")
out["wall_s"] = round(time.time() - t0, 1)
log(f"G2_DONE wall={out['wall_s']}")
with open(os.path.join(WORK, "kernelG2.json"), "w") as f:
    json.dump(out, f)
