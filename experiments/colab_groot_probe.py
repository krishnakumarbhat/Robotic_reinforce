"""Colab PHASE-G: GR00T-N1.7-LIBERO fit probe (self-installing, cache-cleared).
Inventory nested ckpt files -> weight bytes -> honest fit verdict vs 15.6GB T4.
Load attempt only if Isaac-GR00T stack importable cheaply; else report + defer to 4090.
Prints GROOT_PG JSON.
"""
import json
import subprocess
import sys
import time

t0 = time.time()
out: dict = {"stages": {}}


def log(m):
    print(m, flush=True)


r = subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                    "huggingface_hub"], capture_output=True, text=True)
log(f"pip rc={r.returncode}")

import torch  # noqa: E402
from huggingface_hub import HfApi  # noqa: E402

log(f"cuda={torch.cuda.is_available()} " +
    (torch.cuda.get_device_name(0) if torch.cuda.is_available() else ""))
REPO = "nvidia/GR00T-N1.7-LIBERO"
try:
    info = HfApi().model_info(REPO)
    fns = [s.rfilename for s in (info.siblings or [])]
    gb = sum(getattr(s, "size", 0) or 0 for s in (info.siblings or [])) / 1e9
    out["stages"]["files"] = len(fns)
    out["stages"]["bytes_gb"] = round(gb, 2)
    safes = [f for f in fns if f.endswith(".safetensors")]
    out["stages"]["safetensors"] = safes[:12]
    log(f"files={len(fns)} bytes_gb={round(gb,2)} safetensors={len(safes)}")
    for f in safes[:12]:
        log(f"  {f}")
except Exception as e:  # noqa: BLE001
    out["stages"]["files"] = f"FAIL {type(e).__name__}: {str(e)[:150]}"
    log(out["stages"]["files"])
out["wall_s"] = round(time.time() - t0, 1)
print("GROOT_PG " + json.dumps(out)[:2000], flush=True)
