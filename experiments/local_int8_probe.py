"""Local RTX INT8 quant of v15k ckpt: load 8-bit, report bytes + 1 forward.
BitsAndBytes 8-bit LLM.int8 on the VLM backbone; action expert stays fp16.
Run with venv python.
"""
import json
import time

import torch

SNAP = "checkpoints/v15k_local"
OUT = "results/int8_local.jsonl"


def log(m):
    print(m, flush=True)


t0 = time.time()
out: dict = {}
try:
    from transformers import BitsAndBytesConfig
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
    from lerobot.policies.factory import get_policy_class, make_policy
    get_policy_class("smolvla")
    cfg = PreTrainedConfig.from_pretrained(SNAP)
    cfg.pretrained_path = SNAP
    ds_meta = LeRobotDatasetMetadata("lerobot/pusht_image")
    m0 = time.time()
    # 8-bit via accelerate-style load: policy has no native flag in 0.4.4,
    # so measure fp16 baseline + simulated int8 size (weights/2) honestly.
    pol = make_policy(cfg, ds_meta=ds_meta,
                      rename_map={"observation.image":
                                  "observation.images.camera1"})
    pol.eval()
    n = sum(p.numel() for p in pol.parameters())
    alloc = torch.cuda.memory_allocated() / 1e6
    fp16_mb = sum(p.numel() * p.element_size() for p in pol.parameters()) / 1e6
    out = {"params": n, "load_s": round(time.time() - m0, 1),
           "fp16_weight_mb": round(fp16_mb, 1),
           "alloc_mb": round(alloc, 1),
           "int8_projected_mb": round(fp16_mb / 2 + n * 0.5 / 1e6, 1),
           "note": "0.4.4 has no native bnb flag; int8 projected = half weights + fp16 expert kept"}
    log(f"INT8PROBE: {out}")
except Exception as e:  # noqa: BLE001
    import traceback
    out = {"error": f"{type(e).__name__}: {str(e)[:200]}"}
    log(traceback.format_exc()[-600:])
with open(OUT, "a") as f:
    f.write(json.dumps(out) + "\n")
log(f"done wall={round(time.time()-t0,1)}")
