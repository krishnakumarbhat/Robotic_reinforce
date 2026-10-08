"""Colab PHASE6a: background MolmoAct2 load. Returns instantly; poll /root/molmo_load.json."""
import json
import os
import threading

STATUS = "/root/molmo_load.json"


def log(m):
    print(m, flush=True)


def _load():
    try:
        import torch
        from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
        from lerobot.policies.factory import make_policy, make_policy_config
        from lerobot.utils import import_utils
        import_utils._require_package_cache.clear()
        ds_meta = LeRobotDatasetMetadata("lerobot/pusht_image")
        cfg = make_policy_config("molmoact2")
        cfg.pretrained_path = "allenai/MolmoAct2-SO100_101"
        import time
        m0 = time.time()
        pol = make_policy(cfg, ds_meta=ds_meta)
        pol.eval()
        n = sum(p.numel() for p in pol.parameters())
        json.dump({"done": True, "params": n, "load_s": round(time.time() - m0, 1),
                   "alloc_mb": round(torch.cuda.memory_allocated() / 1e6, 1),
                   "reserved_mb": round(torch.cuda.memory_reserved() / 1e6, 1)},
                  open(STATUS, "w"))
    except Exception as e:  # noqa: BLE001
        import traceback
        json.dump({"done": False, "err": f"{type(e).__name__}: {str(e)[:250]}",
                   "tb": traceback.format_exc()[-800:]}, open(STATUS, "w"))


if os.path.exists(STATUS) and json.load(open(STATUS)).get("done"):
    log("already loaded")
else:
    json.dump({"done": False, "err": "running"}, open(STATUS, "w"))
    threading.Thread(target=_load, daemon=True).start()
    log("load launched in background")
print("MOLMO_P6A launched", flush=True)
