"""Kernel G4: FULL SmolVLA PushT post-train (batch 32 x 25k steps, ~5h, T4)
+ eval TUNED + eval BASE-control (discriminates incomplete vs broken).
v19 was 80x under paper sample budget (80k vs 6.4M) — this closes 10x of that.
Evidence -> /kaggle/working/kernelG4.json
"""
import json
import os
import re
import subprocess
import sys
import time

t0 = time.time()
WORK = "/kaggle/working" if os.path.exists("/kaggle") else "."
STEPS, BATCH, ACCUM = 25000, 8, 4  # eff 32 via accum: v19-proven memory, same math as b32
out: dict = {"stages": {}, "steps": STEPS, "batch": BATCH}


def log(m):
    print(m, flush=True)


def run(cmd, timeout=20000):
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    return p


r = run([sys.executable, "-m", "pip", "install", "-q",
         "lerobot[dataset]", "av", "num2words", "libero", "mujoco", "gym-pusht",
         "pymunk<7"])
log(f"pip rc={r.returncode}")

import torch  # noqa: E402
log(f"cuda={torch.cuda.is_available()}")

from huggingface_hub import snapshot_download  # noqa: E402
snap = snapshot_download("lerobot/smolvla_base",
                         local_dir=os.path.join(WORK, "smolvla_base"))
cfg_p = os.path.join(snap, "config.json")
cfg = json.load(open(cfg_p))
cfg["push_to_hub"] = False
json.dump(cfg, open(cfg_p, "w"), indent=1)
log("base patched")

# (train_cmd defined below with accum support)


def train_cmd(batch, steps, job, accum=1, outdir="smolvla_full"):
    cmd = [sys.executable, "-m", "lerobot.scripts.lerobot_train",
           f"--policy.path={snap}",
           "--dataset.repo_id=lerobot/pusht_image",
           f"--batch_size={batch}", f"--steps={steps}",
           f"--output_dir={WORK}/{outdir}", f"--job_name={job}",
           "--policy.device=cuda", "--save_freq=5000",
           "--rename_map={\"observation.image\": \"observation.images.camera1\"}"]
    if accum > 1 and ACCUM_FLAG:
        cmd.append(f"{ACCUM_FLAG}={accum}")
    return cmd


help_p = run([sys.executable, "-m", "lerobot.scripts.lerobot_train", "--help"],
             timeout=120)
help_txt = help_p.stdout + help_p.stderr
m_acc = re.search(r"--[\w.]*grad[\w.]*accum[\w.]*", help_txt)
ACCUM_FLAG = m_acc.group(0) if m_acc else ""
log(f"accum flag: {ACCUM_FLAG or 'NONE -> plain batch8'}")


def smoke(batch, accum=1):
    """20-step VRAM/step-time probe in ISOLATED dir (never collide with full run)."""
    try:
        p = run(train_cmd(batch, 20, "smoke", accum,
                          outdir="smolvla_smoke"), timeout=1200)
        tail = (p.stdout + p.stderr)[-1500:]
        mem = re.findall(r"mem_gb:([0-9.]+)", tail)
        return (p.returncode == 0 and "End of training" in tail,
                mem[-1] if mem else None)
    except subprocess.TimeoutExpired:
        return (False, None)


ok8, mem8 = smoke(8, 4 if ACCUM_FLAG else 1)
log(f"smoke b8{'xa4' if ACCUM_FLAG else ''}: ok={ok8} mem={mem8}")
BATCH_EFF = 8
if not ok8:
    out["abort"] = "batch-8 smoke failed"
    log(out["abort"])
out["stages"]["smoke"] = {"b8": [ok8, mem8], "accum_flag": ACCUM_FLAG,
                          "batch_eff": BATCH_EFF}

if "abort" not in out:
    cmd = train_cmd(BATCH_EFF, STEPS, "smolvla_g4",
                    4 if ACCUM_FLAG else 1)
    log("launch full train")
    try:
        p = run(cmd)
        full = p.stdout + p.stderr
        with open(os.path.join(WORK, "train_full.txt"), "w") as f:
            f.write(full)  # never lose the traceback again
        losses = re.findall(r"loss:([0-9.]+)", full)
        tb = full[-600:] if p.returncode != 0 else ""
        out["stages"]["train"] = {"rc": p.returncode, "loss_tail": losses[-4:],
                                  "err_tail": tb}
        log(f"train rc={p.returncode} loss_tail={losses[-4:]}")
    except subprocess.TimeoutExpired:
        out["stages"]["train"] = {"rc": "timeout"}
        log("train TIMEOUT")
else:
    out["stages"]["train"] = {"rc": "skipped"}


def do_eval(name, path):
    cmd = [sys.executable, "-m", "lerobot.scripts.lerobot_eval",
           f"--policy.path={path}", "--env.type=pusht",
           "--eval.batch_size=1", "--eval.n_episodes=20",
           "--rename_map={\"observation.image\": \"observation.images.camera1\"}"]
    try:
        p = subprocess.run(cmd, input="N\n", capture_output=True,
                           text=True, timeout=2400)
        full = p.stdout + p.stderr
        with open(os.path.join(WORK, f"eval_{name}.txt"), "w") as f:
            f.write(full)
        m = re.findall(r"'pc_success': ([0-9.]+)", full)
        mx = re.findall(r"'avg_max_reward': ([0-9.]+)", full)
        return {"rc": p.returncode, "pc_success": m[-1] if m else None,
                "avg_max_reward": mx[-1] if mx else None}
    except subprocess.TimeoutExpired:
        return {"rc": "timeout"}


import glob
ckpts = sorted(glob.glob(os.path.join(WORK, "smolvla_full", "checkpoints", "*")))
out["stages"]["ckpts"] = [os.path.basename(c) for c in ckpts]
log(f"ckpts: {out['stages']['ckpts']}")
if ckpts:
    out["eval_tuned"] = do_eval("tuned", os.path.join(ckpts[-1], "pretrained_model"))
    log(f"TUNED: {out['eval_tuned']}")
out["eval_base"] = do_eval("base", snap)
log(f"BASE: {out['eval_base']}")

out["wall_s"] = round(time.time() - t0, 1)
log(f"G4_DONE wall={out['wall_s']}")
with open(os.path.join(WORK, "kernelG4.json"), "w") as f:
    json.dump(out, f)
