"""Kernel PIQ: FastVLA SmolVLA 4-bit QLoRA REAL train on pusht, Kaggle T4.
Preset verified FIT by PI-FIT (3.6GB/16GB). 10k steps ~30min. Ungated deps only.
Evidence -> /kaggle/working/kernelPIQ.json
"""
import json
import os
import subprocess
import sys
import time

t0 = time.time()
WORK = "/kaggle/working" if os.path.exists("/kaggle") else "."
STEPS, BATCH, ACCUM = 10000, 4, 8
out: dict = {"stages": {}, "steps": STEPS, "batch": BATCH, "accum": ACCUM}


def log(m):
    print(m, flush=True)


r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "fastvla"],
                   capture_output=True, text=True)
log(f"pip fastvla rc={r.returncode}")
out["stages"]["pip"] = {"rc": r.returncode}

import torch  # noqa: E402
log(f"cuda={torch.cuda.is_available()}")
try:
    from fastvla import FastVLAModel  # noqa: E402
    from fastvla.training import FastVLATrainer  # noqa: E402
    m = FastVLAModel.from_pretrained("smolvla", load_in_4bit=True, use_peft=True)
    n = sum(p.numel() for p in m.parameters())
    vram = {"alloc": round(torch.cuda.memory_allocated() / 1e6, 1),
            "reserved": round(torch.cuda.memory_reserved() / 1e6, 1)}
    out["stages"]["load"] = {"params": n, **vram}
    log(f"loaded smolvla-4bit params={n} vram={vram}")
    tr = FastVLATrainer(model=m, dataset="pusht", max_steps=STEPS,
                        batch_size=BATCH, gradient_accumulation_steps=ACCUM,
                        output_dir=os.path.join(WORK, "piq_out"),
                        save_steps=2000, logging_steps=500, use_wandb=False)
    tr.train()
    hist = getattr(tr, "training_history", [])[-5:]
    out["stages"]["train"] = {"rc": 0, "last": hist}
    log(f"TRAIN_DONE last={hist[-1] if hist else None}")
except Exception as e:  # noqa: BLE001
    import traceback
    out["stages"]["train"] = {"rc": 1, "err": f"{type(e).__name__}: {str(e)[:300]}"}
    log(f"TRAIN_FAIL: {type(e).__name__}: {str(e)[:300]}")
    log(traceback.format_exc()[-800:])

out["wall_s"] = round(time.time() - t0, 1)
log(f"PIQ_DONE wall={out['wall_s']}")
with open(os.path.join(WORK, "kernelPIQ.json"), "w") as f:
    json.dump(out, f)
