"""Colab lane PHASE4: MolmoAct2-SO100_101 load with real ds_meta + VRAM verdict."""
import json
import time
import traceback

t0 = time.time()
out: dict = {"stages": {}}


def log(m):
    print(m, flush=True)


import torch  # noqa: E402

r = __import__("subprocess").run(
    [__import__("sys").executable, "-m", "pip", "install", "-q",
     "lerobot[dataset]", "av", "num2words"], capture_output=True, text=True)
log(f"self-install rc={r.returncode}")

from lerobot.utils import import_utils  # noqa: E402
import_utils._require_package_cache.clear()  # poisoned by pre-av imports in this kernel
log("cleared require_package cache")

REPO = "allenai/MolmoAct2-SO100_101"
try:
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata  # noqa: E402
    from lerobot.policies.factory import make_policy, make_policy_config  # noqa: E402
    ds_meta = LeRobotDatasetMetadata("lerobot/pusht_image")
    log("ds_meta OK")
    cfg = make_policy_config("molmoact2")
    cfg.pretrained_path = REPO
    m0 = time.time()
    pol = make_policy(cfg, ds_meta=ds_meta)
    pol.eval()
    n = sum(p.numel() for p in pol.parameters())
    out["stages"]["load"] = {
        "params": n, "load_s": round(time.time() - m0, 1),
        "alloc_mb": round(torch.cuda.memory_allocated() / 1e6, 1),
        "reserved_mb": round(torch.cuda.memory_reserved() / 1e6, 1),
        "fit_t4": bool(torch.cuda.memory_reserved() < 14e9)}
    # one dummy forward for step latency
    import numpy as np  # noqa: E402
    batch = {k: torch.zeros(1, *v.shape[1:], device="cuda")
             if len(v.shape) > 1 else torch.zeros(1, *v.shape, device="cuda")
             for k, v in pol.config.input_features.items()
             for _ in [0]}
    log(f"batch keys: {sorted(batch.keys())}")
except Exception:  # noqa: BLE001
    out["stages"]["load"] = "FAIL"
    log(traceback.format_exc()[-900:])
log(f"load: {out['stages']['load']}")
out["wall_s"] = round(time.time() - t0, 1)
print("MOLMO_P4 " + json.dumps(out)[:1500], flush=True)
