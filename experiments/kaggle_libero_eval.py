"""Kernel G3E: LIBERO eval of OUR G3 ckpt with self-healing mujoco pins.
Tries pin sets in order; first env that constructs wins; then full 3-suite eval.
Ckpt from HF krishnah27/smolvla-libero-g3 (uploaded post-bundle).
Evidence -> /kaggle/working/kernelG3E.json
"""
import json
import os
import re
import subprocess
import sys
import time

t0 = time.time()
WORK = "/kaggle/working" if os.path.exists("/kaggle") else "."
REPO = "krishnah27/smolvla-libero-g3-10k"  # 15k-total LIBERO steps
RENAME = ("{\"observation.images.image\": \"observation.images.camera1\", "
          "\"observation.images.image2\": \"observation.images.camera2\"}")
out: dict = {"repo": REPO, "suites": {}}

PINS = [
    ("latest", []),
    ("mujoco316", ["mujoco==3.1.6"]),
    ("mujoco237", ["mujoco==2.3.7"]),
    ("rs150_mj316", ["robosuite==1.5.0", "mujoco==3.1.6"]),
    ("rs141_mj237", ["robosuite==1.4.1", "mujoco==2.3.7"]),
]


def log(m):
    print(m, flush=True)


def pip(pkgs):
    r = subprocess.run([sys.executable, "-m", "pip", "install", "-q"] + pkgs,
                       capture_output=True, text=True)
    return r.returncode


log(f"base pip rc={pip(['lerobot[dataset]', 'av', 'num2words', 'libero', 'mujoco'])}")

from huggingface_hub import snapshot_download  # noqa: E402
try:
    snap = snapshot_download(REPO)
    out["ckpt"] = True
    log(f"ckpt ok: {snap}")
except Exception as e:  # noqa: BLE001
    out["ckpt"] = False
    out["abort"] = f"ckpt: {type(e).__name__}: {str(e)[:150]}"
    log(out["abort"])


def try_eval(suite, n_eps):
    cmd = [sys.executable, "-m", "lerobot.scripts.lerobot_eval",
           f"--policy.path={snap}", "--env.type=libero",
           f"--env.task={suite}", "--eval.batch_size=1",
           f"--eval.n_episodes={n_eps}", f"--rename_map={RENAME}"]
    try:
        p = subprocess.run(cmd, input="N\n", capture_output=True,
                           text=True, timeout=2400)
        full = p.stdout + p.stderr
        m = re.findall(r"'pc_success': ([0-9.]+)", full)
        return {"rc": p.returncode, "pc_success": m[-1] if m else None,
                "err": full[-300:] if p.returncode != 0 else ""}
    except subprocess.TimeoutExpired:
        return {"rc": "timeout"}


winner = None
if out.get("ckpt"):
    for name, pins in PINS:
        if pins:
            log(f"pin set {name}: pip rc={pip(pins)}")
        r = try_eval("libero_spatial", 2)
        log(f"probe {name}: {r}")
        out[f"probe_{name}"] = r
        if r["rc"] == 0:
            winner = name
            break
    out["winner"] = winner
    if winner:
        for suite in ["libero_spatial", "libero_object", "libero_goal"]:
            out["suites"][suite] = try_eval(suite, 10)
            log(f"{suite}: {out['suites'][suite]}")
out["wall_s"] = round(time.time() - t0, 1)
log(f"G3E_DONE wall={out['wall_s']}")
with open(os.path.join(WORK, "kernelG3E.json"), "w") as f:
    json.dump(out, f)
