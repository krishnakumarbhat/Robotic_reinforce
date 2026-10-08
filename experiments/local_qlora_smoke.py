"""Local RTX: FastVLA smolvla-4bit QLoRA 100-step smoke (batch 1). venv python.
Proves or refutes local VLA training on 4GB. Streams loss.
"""
import json
import time

import torch

OUT = "results/qlora_local.jsonl"


def log(m):
    print(m, flush=True)


t0 = time.time()
out: dict = {}
try:
    from fastvla import FastVLAModel
    from fastvla.training import FastVLATrainer
    m = FastVLAModel.from_pretrained("smolvla", load_in_4bit=True,
                                     use_peft=True)
    log(f"loaded vram_mb={torch.cuda.memory_reserved()/1e6:.0f}")
    tr = FastVLATrainer(model=m, dataset="pusht", max_steps=100,
                        batch_size=1, output_dir="results/qlora_smoke",
                        save_steps=100000, logging_steps=25, use_wandb=False)
    tr.train()
    hist = getattr(tr, "training_history", [])[-3:]
    out = {"rc": 0, "last": hist, "wall_s": round(time.time() - t0, 1)}
    log(f"TRAIN_OK last={hist[-1] if hist else None}")
except Exception as e:  # noqa: BLE001
    import traceback
    out = {"rc": 1, "err": f"{type(e).__name__}: {str(e)[:250]}",
           "wall_s": round(time.time() - t0, 1)}
    log(traceback.format_exc()[-700:])
with open(OUT, "a") as f:
    f.write(json.dumps(out) + "\n")
log(f"QLORA_DONE {out}")
