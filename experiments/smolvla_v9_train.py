"""Kernel v9: REAL SmolVLA post-train on pusht_image, Kaggle T4.
Official lerobot recipe only (no exotic flags). 2000-step smoke; ckpt + scalars to /kaggle/working.
Evidence -> /kaggle/working/kernelV9.json
"""
import json
import os
import subprocess
import sys
import time

t0 = time.time()
WORK = "/kaggle/working" if os.path.exists("/kaggle") else "."
STEPS, BATCH = 20000, 4  # full run (smoke 2000 done v16: loss 0.435)
out = {"stages": {}, "steps": STEPS, "batch": BATCH}


def log(m):
    print(m, flush=True)


log("pip install av + lerobot[dataset]")
r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "av", "num2words"],
                   capture_output=True, text=True)
log(f"pip av rc: {r.returncode} tail: {(r.stdout + r.stderr)[-300:]}")
r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "lerobot[dataset]"],
                   capture_output=True, text=True)
log(f"pip rc: {r.returncode}")
out["stages"]["pip"] = {"rc": r.returncode}
if r.returncode != 0:
    out["abort"] = "lerobot pip failed"
    log(r.stderr[-1000:])

if "abort" not in out:
    try:
        import av  # noqa: F401
        out["stages"]["av"] = {"ok": True, "ver": av.__version__}
    except Exception as e:  # noqa: BLE001
        out["abort"] = f"av missing: {e}"
        log(out["abort"])

if "abort" not in out:
    import torch  # noqa: E402
    log(f"cuda: {torch.cuda.is_available()}")
    out["stages"]["cuda"] = {"avail": torch.cuda.is_available(),
                             "name": torch.cuda.get_device_name(0)
                             if torch.cuda.is_available() else None}
    help_p = subprocess.run([sys.executable, "-m", "lerobot.scripts.lerobot_train", "--help"],
                            capture_output=True, text=True, timeout=120)
    help_txt = help_p.stdout + help_p.stderr
    out["stages"]["help"] = {"rc": help_p.returncode, "len": len(help_txt)}
    with open(os.path.join(WORK, "v9_help.txt"), "w") as f:
        f.write(help_txt)
    log(f"help rc={help_p.returncode} len={len(help_txt)}")
    from huggingface_hub import snapshot_download  # noqa: E402
    snap = snapshot_download("lerobot/smolvla_base",
                             local_dir=os.path.join(WORK, "smolvla_base"))
    cfg_p = os.path.join(snap, "config.json")
    cfg = json.load(open(cfg_p))
    cfg["push_to_hub"] = False  # base ships push:true/repo:null; CLI has no override
    json.dump(cfg, open(cfg_p, "w"), indent=1)
    out["stages"]["ckpt_patch"] = {"push_to_hub": False}
    log("patched smolvla_base config: push_to_hub=false")
    cmd = [sys.executable, "-m", "lerobot.scripts.lerobot_train",
           f"--policy.path={snap}",
           "--dataset.repo_id=lerobot/pusht_image",
           f"--batch_size={BATCH}", f"--steps={STEPS}",
           f"--output_dir={WORK}/smolvla_out", "--job_name=smolvla_v9",
           "--policy.device=cuda", "--save_freq=10000",
           "--rename_map={\"observation.image\": \"observation.images.camera1\"}"]
    log("launch: local-patched policy + pusht_image")
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=18000)
        out["stages"]["train"] = {"rc": p.returncode,
                                  "tail": (p.stdout + p.stderr)[-1500:]}
        log(f"train rc: {p.returncode}")
        log((p.stdout + p.stderr)[-800:])
    except subprocess.TimeoutExpired:
        out["stages"]["train"] = {"rc": "timeout"}
        log("train TIMEOUT after 5h")

ckpts = []
for dp, _, fns in os.walk(os.path.join(WORK, "smolvla_out")):
    ckpts += [os.path.join(dp, f) for f in fns if f.endswith((".safetensors", ".pt", ".bin"))]
out["checkpoints"] = ckpts[:10]
out["wall_s"] = round(time.time() - t0, 1)
log(f"V9_DONE wall={out['wall_s']}s ckpts={len(ckpts)}")
with open(os.path.join(WORK, "kernelV9.json"), "w") as f:
    json.dump(out, f)
