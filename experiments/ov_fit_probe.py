"""Kernel OV-FIT: OpenVLA-7B 4-bit fit + smoke on T4 (llama gate now open).
HF_TOKEN via env (injected at push, never committed). Load -> VRAM -> 50-step smoke.
Evidence -> /kaggle/working/kernelOV.json
"""
import json
import os
import subprocess
import sys
import time

t0 = time.time()
WORK = "/kaggle/working" if os.path.exists("/kaggle") else "."
out: dict = {"stages": {}}


def log(m):
    print(m, flush=True)


tok = os.environ.get("HF_TOKEN", "")
if tok:
    try:
        from huggingface_hub import login  # noqa: E402
        login(token=tok)
        log("hf login ok")
        out["stages"]["hf"] = True
    except Exception as e:  # noqa: BLE001
        out["stages"]["hf"] = f"FAIL {type(e).__name__}"
        log(out["stages"]["hf"])
else:
    out["stages"]["hf"] = "no-token"
    log("no HF_TOKEN")

r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "fastvla"],
                   capture_output=True, text=True)
log(f"pip fastvla rc={r.returncode}")

import torch  # noqa: E402
log(f"cuda={torch.cuda.is_available()}")
try:
    from fastvla import FastVLAModel  # noqa: E402
    m0 = time.time()
    m = FastVLAModel.from_pretrained("openvla-7b", load_in_4bit=True, use_peft=True)
    n = sum(p.numel() for p in m.parameters())
    vram = {"alloc": round(torch.cuda.memory_allocated() / 1e6, 1),
            "reserved": round(torch.cuda.memory_reserved() / 1e6, 1)}
    out["stages"]["load"] = {"params": n, "load_s": round(time.time() - m0, 1), **vram,
                             "FIT": vram["reserved"] < 15000}
    log(f"openvla-7b-4bit params={n} vram={vram} load ok")
    from fastvla.training import FastVLATrainer  # noqa: E402
    tr = FastVLATrainer(model=m, dataset="pusht", max_steps=50, batch_size=2,
                        output_dir=os.path.join(WORK, "ov_smoke"),
                        save_steps=100000, logging_steps=10, use_wandb=False)
    tr.train()
    hist = getattr(tr, "training_history", [])[-3:]
    out["stages"]["smoke"] = {"rc": 0, "last": hist}
    log(f"SMOKE_DONE last={hist[-1] if hist else None}")
except Exception as e:  # noqa: BLE001
    import traceback
    out["stages"]["smoke"] = {"rc": 1, "err": f"{type(e).__name__}: {str(e)[:250]}"}
    log(f"FAIL: {type(e).__name__}: {str(e)[:200]}")
    log(traceback.format_exc()[-600:])

out["wall_s"] = round(time.time() - t0, 1)
log(f"OVFIT_DONE wall={out['wall_s']}")
with open(os.path.join(WORK, "kernelOV.json"), "w") as f:
    json.dump(out, f)
