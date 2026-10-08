"""Colab PHASE-G2: download GR00T-N1.7-LIBERO/libero_spatial suite + config inspect (background)."""
import json
import os
import threading

STATUS = "/root/groot_dl.json"


def log(m):
    print(m, flush=True)


def _dl():
    try:
        from huggingface_hub import snapshot_download
        p = snapshot_download("nvidia/GR00T-N1.7-LIBERO",
                              local_dir="/root/groot_libero",
                              allow_patterns=["libero_spatial/*"])
        cfg = json.load(open(os.path.join(p, "libero_spatial", "config.json")))
        json.dump({"done": True, "path": p,
                   "cfg_keys": sorted(cfg.keys())[:20]}, open(STATUS, "w"))
    except Exception as e:  # noqa: BLE001
        import traceback
        json.dump({"done": False, "err": f"{type(e).__name__}: {str(e)[:200]}",
                   "tb": traceback.format_exc()[-500:]}, open(STATUS, "w"))


if os.path.exists(STATUS) and json.load(open(STATUS)).get("done"):
    log("already downloaded")
else:
    json.dump({"done": False, "err": "running"}, open(STATUS, "w"))
    threading.Thread(target=_dl, daemon=True).start()
    log("GR00T download launched in background")
print("GROOT_DL launched", flush=True)
