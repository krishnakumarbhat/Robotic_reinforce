"""Local RTX LIBERO screening v2: per-suite subprocess evals (10tx1ep each).
Each suite = own process (fresh CUDA space) + own outputs + JSONL stream.
Zero quota. Run with venv python, detached.
"""
import json
import os
import re
import subprocess
import sys
import time

SNAP = "checkpoints/g3_10k_local"
RENAME = ("{\"observation.images.image\": \"observation.images.camera1\", "
          "\"observation.images.image2\": \"observation.images.camera2\"}")
OUT = "results/libero_local.jsonl"
SUITES = ["libero_spatial", "libero_object", "libero_goal"]


def log(m):
    print(m, flush=True)


env = dict(os.environ)
env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

for suite in SUITES:
    cmd = [sys.executable, "-m", "lerobot.scripts.lerobot_eval",
           f"--policy.path={SNAP}", "--env.type=libero",
           f"--env.task={suite}", "--eval.batch_size=1",
           "--eval.n_episodes=1", f"--rename_map={RENAME}"]
    log(f"eval {suite} ...")
    try:
        p = subprocess.run(cmd, input="N\n", capture_output=True,
                           text=True, timeout=7000, env=env)
        full = p.stdout + p.stderr
        with open(f"results/eval_{suite}_full.txt", "w") as f:
            f.write(full)
        m = re.findall(r"'pc_success': ([0-9.]+)", full)
        rec = {"suite": suite, "rc": p.returncode,
               "pc_success": m[-1] if m else None, "t": time.time()}
        log(f"{suite}: {rec}")
    except subprocess.TimeoutExpired:
        rec = {"suite": suite, "rc": "timeout", "t": time.time()}
        log(f"{suite}: TIMEOUT")
    with open(OUT, "a") as f:
        f.write(json.dumps(rec) + "\n")
log("LOCAL_SCREEN_V2_DONE")
