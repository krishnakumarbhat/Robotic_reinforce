"""Colab PHASE7a: MolmoAct2 via transformers sharded loader (background). Poll /root/molmo_t.json."""
import json
import os
import threading

STATUS = "/root/molmo_t.json"


def log(m):
    print(m, flush=True)


def _load():
    try:
        import torch
        from transformers import AutoModel
        import time
        m0 = time.time()
        m = AutoModel.from_pretrained("allenai/MolmoAct2-SO100_101",
                                      trust_remote_code=True,
                                      torch_dtype=torch.bfloat16,
                                      low_cpu_mem_usage=True)
        m.eval()
        n = sum(p.numel() for p in m.parameters())
        json.dump({"done": True, "params": n, "load_s": round(time.time() - m0, 1),
                   "alloc_gb": round(torch.cuda.memory_allocated() / 1e9, 2),
                   "reserved_gb": round(torch.cuda.memory_reserved() / 1e9, 2)},
                  open(STATUS, "w"))
    except Exception as e:  # noqa: BLE001
        import traceback
        json.dump({"done": False, "err": f"{type(e).__name__}: {str(e)[:250]}",
                   "tb": traceback.format_exc()[-600:]}, open(STATUS, "w"))


if os.path.exists(STATUS) and json.load(open(STATUS)).get("done"):
    log("already loaded")
else:
    json.dump({"done": False, "err": "running"}, open(STATUS, "w"))
    threading.Thread(target=_load, daemon=True).start()
    log("transformers load launched in background")
print("MOLMO_P7A launched", flush=True)
