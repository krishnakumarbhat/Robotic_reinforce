"""Local RTX PushT eval of v15k ckpt (20 eps, streaming JSONL). Zero quota.
Uses local checkpoints/v15k_local (already compat-patched). Needs gym-pusht.
Run with the venv python AFTER libero screening finishes (one GPU job at a time).
"""
import json
import os
import re
import subprocess
import sys
import time

SNAP = "checkpoints/v15k_local"
OUT = "results/pusht_local.jsonl"
RENAME = "{\"observation.image\": \"observation.images.camera1\"}"


def log(m):
    print(m, flush=True)


cmd = [sys.executable, "-m", "lerobot.scripts.lerobot_eval",
       f"--policy.path={SNAP}", "--env.type=pusht",
       "--eval.batch_size=1", "--eval.n_episodes=20",
       f"--rename_map={RENAME}"]
log("eval pusht x20 ...")
try:
    p = subprocess.run(cmd, input="N\n", capture_output=True,
                       text=True, timeout=12000)
    full = p.stdout + p.stderr
    with open("results/pusht_local_full.txt", "w") as f:
        f.write(full)
    m = re.findall(r"'pc_success': ([0-9.]+)", full)
    mx = re.findall(r"'avg_max_reward': ([0-9.]+)", full)
    rec = {"rc": p.returncode,
           "pc_success": m[-1] if m else None,
           "avg_max_reward": mx[-1] if mx else None, "t": time.time()}
    log(f"PUSHT: {rec}")
except subprocess.TimeoutExpired:
    rec = {"rc": "timeout", "t": time.time()}
    log("TIMEOUT")
with open(OUT, "a") as f:
    f.write(json.dumps(rec) + "\n")
log("PUSHT_LOCAL_DONE")
