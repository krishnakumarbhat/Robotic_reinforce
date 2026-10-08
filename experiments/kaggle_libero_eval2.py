"""Kernel G3E2: full LIBERO single-suite eval with KNOWN-GOOD pin (mujoco==3.1.6).
v35 proved winner=mujoco316, spatial probe 7/20=35%. This kernel does ONE suite
x10eps (100 rollouts, ~3h) with 8500s timeout. SUITE env var selects suite.
Evidence -> /kaggle/working/kernelG3E2_<suite>.json
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
SUITE = os.environ.get("SUITE", "libero_spatial")
N_EPS = int(os.environ.get("N_EPS", "10"))
RENAME = ("{\"observation.images.image\": \"observation.images.camera1\", "
          "\"observation.images.image2\": \"observation.images.camera2\"}")
out: dict = {"repo": REPO, "suite": SUITE, "n_eps": N_EPS}


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
    cmd = [sys.executable, "-m", "lerobot.scripts.lerobot_eval",
           f"--policy.path={snap}", "--env.type=libero",
           f"--env.task={SUITE}", "--eval.batch_size=1",
           f"--eval.n_episodes={N_EPS}", f"--rename_map={RENAME}"]
    log(f"eval {SUITE} x{N_EPS} (~3h) ...")
    try:
        p = subprocess.run(cmd, input="N\n", capture_output=True,
                           text=True, timeout=8500)
        full = p.stdout + p.stderr
        with open(os.path.join(WORK, f"eval_{SUITE}_full.txt"), "w") as f:
            f.write(full)
        m = re.findall(r"'pc_success': ([0-9.]+)", full)
        out["eval"] = {"rc": p.returncode,
                       "pc_success": m[-1] if m else None}
        log(f"{SUITE}: {out['eval']}")
    except subprocess.TimeoutExpired:
        out["eval"] = {"rc": "timeout"}
        log("TIMEOUT")
out["wall_s"] = round(time.time() - t0, 1)
log(f"G3E2_DONE wall={out['wall_s']}")
with open(os.path.join(WORK, f"kernelG3E2_{SUITE}.json"), "w") as f:
    json.dump(out, f)
