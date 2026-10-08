"""Colab PHASE5a: background snapshot download of MolmoAct2-SO100_101. Returns instantly."""
import json
import os
import threading

STATUS = "/root/molmo_dl.json"


def log(m):
    print(m, flush=True)


def _dl():
    try:
        from huggingface_hub import snapshot_download
        p = snapshot_download("allenai/MolmoAct2-SO100_101",
                              local_dir="/root/molmo_so100")
        json.dump({"done": True, "path": p}, open(STATUS, "w"))
    except Exception as e:  # noqa: BLE001
        json.dump({"done": False, "err": f"{type(e).__name__}: {str(e)[:200]}"},
                  open(STATUS, "w"))


if os.path.exists(STATUS) and json.load(open(STATUS)).get("done"):
    log("already downloaded")
else:
    json.dump({"done": False, "err": "running"}, open(STATUS, "w"))
    threading.Thread(target=_dl, daemon=True).start()
    log("download launched in background")
print("MOLMO_P5A launched", flush=True)
