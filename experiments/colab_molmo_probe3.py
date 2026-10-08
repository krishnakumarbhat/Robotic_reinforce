"""Colab lane PHASE3: MolmoAct2-SO100_101 load via factory config + VRAM verdict."""
import json
import time
import traceback

t0 = time.time()
out: dict = {"stages": {}}


def log(m):
    print(m, flush=True)


import torch  # noqa: E402

REPO = "allenai/MolmoAct2-SO100_101"
try:
    from lerobot.policies.factory import (  # noqa: E402
        get_policy_class, make_policy, make_policy_config)
    cfg = make_policy_config("molmoact2")
    log(f"base config: {type(cfg).__name__}")
    cfg.pretrained_path = REPO
    m0 = time.time()
    pol = make_policy(cfg, ds_meta=None)
    n = sum(p.numel() for p in pol.parameters())
    out["stages"]["load"] = {
        "params": n, "load_s": round(time.time() - m0, 1),
        "alloc_mb": round(torch.cuda.memory_allocated() / 1e6, 1),
        "reserved_mb": round(torch.cuda.memory_reserved() / 1e6, 1)}
except Exception:  # noqa: BLE001
    out["stages"]["load"] = "FAIL"
    log(traceback.format_exc()[-900:])
log(f"load: {out['stages']['load']}")
out["wall_s"] = round(time.time() - t0, 1)
print("MOLMO_P3 " + json.dumps(out)[:1500], flush=True)
