"""Kernel G3: SmolVLA contender fine-tune ON lerobot/libero (20k steps, T4).
Apples-to-apples vs pi05_libero / GR00T-N1.7-LIBERO. Rename both LIBERO cams
into policy camera1/2 (subset rule); paired rename on train AND eval.
Smoke-then-commit in isolated dir; full stderr persisted; tuned+base evals.
Evidence -> /kaggle/working/kernelG3.json
"""
import glob
import json
import os
import re
import subprocess
import sys
import time

t0 = time.time()
WORK = "/kaggle/working" if os.path.exists("/kaggle") else "."
STEPS, BATCH = 15000, 8  # continue 5k->20k (chain via Hub; LIBERO ~4s/step)
DS = "lerobot/libero"
START_REPO = "krishnah27/smolvla-libero-g3-5k"  # chain from G3-5k ckpt
RENAME = ("{\"observation.images.image\": \"observation.images.camera1\", "
          "\"observation.images.image2\": \"observation.images.camera2\"}")
out: dict = {"stages": {}, "dataset": DS, "steps": STEPS, "batch": BATCH}


def log(m):
    print(m, flush=True)


def run(cmd, timeout=30000):
    # stream output incrementally so kills/timeouts keep evidence
    t0 = time.time()
    with open(os.path.join(WORK, "train_stream.txt"), "a") as f:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True,
                                bufsize=1)
        assert proc.stdout is not None
        for line in proc.stdout:
            f.write(line)
            f.flush()
            if time.time() - t0 > timeout:
                proc.kill()
                raise subprocess.TimeoutExpired(cmd, timeout)
        proc.wait()
        out = open(os.path.join(WORK, "train_stream.txt")).read()
    return subprocess.CompletedProcess(cmd, proc.returncode, out, "")


r = run([sys.executable, "-m", "pip", "install", "-q",
         "lerobot[dataset]", "av", "num2words", "libero", "mujoco"])
log(f"pip rc={r.returncode}")

import torch  # noqa: E402
log(f"cuda={torch.cuda.is_available()}")

from huggingface_hub import snapshot_download  # noqa: E402
snap = snapshot_download(START_REPO,
                         local_dir=os.path.join(WORK, "smolvla_g3start"))
cfg_p = os.path.join(snap, "config.json")
cfg_p = os.path.join(snap, "config.json")
cfg = json.load(open(cfg_p))
cfg["push_to_hub"] = False
json.dump(cfg, open(cfg_p, "w"), indent=1)
log("base patched")


def train_cmd(batch, steps, job, outdir):
    return [sys.executable, "-m", "lerobot.scripts.lerobot_train",
            f"--policy.path={snap}", f"--dataset.repo_id={DS}",
            f"--batch_size={batch}", f"--steps={steps}",
            f"--output_dir={WORK}/{outdir}", f"--job_name={job}",
            "--policy.device=cuda", "--save_freq=5000",
            f"--rename_map={RENAME}"]


try:
    p = run(train_cmd(BATCH, 20, "smoke", "smolvla_smoke"), timeout=1200)
    tail = (p.stdout + p.stderr)[-1500:]
    mem = re.findall(r"mem_gb:([0-9.]+)", tail)
    ok = p.returncode == 0 and "End of training" in tail
    out["stages"]["smoke"] = {"ok": ok, "mem": mem[-1] if mem else None}
    log(f"smoke: ok={ok} mem={mem[-1] if mem else None}")
except subprocess.TimeoutExpired:
    out["stages"]["smoke"] = {"ok": False}
    log("smoke TIMEOUT")

if out["stages"]["smoke"].get("ok"):
    log("launch full train")
    try:
        p = run(train_cmd(BATCH, STEPS, "smolvla_g3", "smolvla_g3"))
        full = p.stdout + p.stderr
        with open(os.path.join(WORK, "train_g3.txt"), "w") as f:
            f.write(full)
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
    log("smoke failed, train skipped")


def do_eval(name, path, suite):
    cmd = [sys.executable, "-m", "lerobot.scripts.lerobot_eval",
           f"--policy.path={path}", "--env.type=libero",
           f"--env.task={suite}", "--eval.batch_size=1",
           "--eval.n_episodes=10", f"--rename_map={RENAME}"]
    try:
        p = subprocess.run(cmd, input="N\n", capture_output=True,
                           text=True, timeout=2400)
        full = p.stdout + p.stderr
        with open(os.path.join(WORK, f"eval_{name}_{suite}.txt"), "w") as f:
            f.write(full)
        m = re.findall(r"'pc_success': ([0-9.]+)", full)
        return {"rc": p.returncode, "pc_success": m[-1] if m else None}
    except subprocess.TimeoutExpired:
        return {"rc": "timeout"}


ckpts = sorted(glob.glob(os.path.join(WORK, "smolvla_g3", "checkpoints", "*")))
out["stages"]["ckpts"] = [os.path.basename(c) for c in ckpts]
log(f"ckpts: {out['stages']['ckpts']}")
if ckpts:
    last = os.path.join(ckpts[-1], "pretrained_model")
    for suite in ["libero_spatial", "libero_object", "libero_goal"]:
        out[f"eval_{suite}"] = do_eval("tuned", last, suite)
        log(f"TUNED {suite}: {out[f'eval_{suite}']}")
out["wall_s"] = round(time.time() - t0, 1)
log(f"G3_DONE wall={out['wall_s']}")
with open(os.path.join(WORK, "kernelG3.json"), "w") as f:
    json.dump(out, f)
