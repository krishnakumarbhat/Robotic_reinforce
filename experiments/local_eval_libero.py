"""Local RTX LIBERO screening: 3 suites x 10tx1ep via lerobot-eval subprocesses.
Zero quota, unlimited wall. Streams results to results/libero_local.jsonl.
Ckpt: krishnah27/smolvla-libero-g3-10k. mujoco==3.1.6 (proven winner pin).
Run with the venv python.
"""
import json
import os
import re
import subprocess
import sys
import time

REPO = "krishnah27/smolvla-libero-g3-10k"
RENAME = ("{\"observation.images.image\": \"observation.images.camera1\", "
          "\"observation.images.image2\": \"observation.images.camera2\"}")
OUT = "results/libero_local.jsonl"
SUITES = ["libero_spatial", "libero_object", "libero_goal"]


def log(m):
    print(m, flush=True)


from huggingface_hub import snapshot_download  # noqa: E402
from lerobot.configs.policies import PreTrainedConfig  # noqa: E402
from lerobot.policies import factory as _factory  # noqa: E402,F401 registers choice classes

snap = snapshot_download(REPO, local_dir="checkpoints/g3_10k_local")
log(f"snapshot: {snap}")
for _ in range(10):  # drop new-lerobot config keys old draccus rejects
    try:
        PreTrainedConfig.from_pretrained(snap)
        break
    except Exception as e:  # noqa: BLE001
        m = re.search(r"The fields `([^`]+)` are not valid", str(e))
        if not m:
            raise
        cp = os.path.join(snap, "config.json")
        d = json.load(open(cp))
        for k in m.group(1).split(", "):
            d.pop(k.strip("` "), None)
        json.dump(d, open(cp, "w"), indent=1)
        log(f"dropped config keys: {m.group(1)}")
POLICY = snap
cp = os.path.join(snap, "config.json")  # ckpt stores remote train path; localize
d = json.load(open(cp))
if "pretrained_path" in d:
    d["pretrained_path"] = snap
    json.dump(d, open(cp, "w"), indent=1)
    log("localized pretrained_path")


for suite in SUITES:
    cmd = [sys.executable, "-m", "lerobot.scripts.lerobot_eval",
           f"--policy.path={POLICY}", "--env.type=libero",
           f"--env.task={suite}", "--eval.batch_size=1",
           "--eval.n_episodes=1", f"--rename_map={RENAME}"]
    log(f"eval {suite} x10tx1ep ...")
    try:
        p = subprocess.run(cmd, input="N\n", capture_output=True,
                           text=True, timeout=12000)
        full = p.stdout + p.stderr
        m = re.findall(r"'pc_success': ([0-9.]+)", full)
        rec = {"suite": suite, "rc": p.returncode,
               "pc_success": m[-1] if m else None, "t": time.time()}
        log(f"{suite}: {rec}")
    except subprocess.TimeoutExpired:
        rec = {"suite": suite, "rc": "timeout", "t": time.time()}
        log(f"{suite}: TIMEOUT")
    with open(OUT, "a") as f:
        f.write(json.dumps(rec) + "\n")
log("LOCAL_SCREEN_DONE")
