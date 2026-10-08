"""Kernel PI-FIT: quant-fit probe for best open VLA on free T4 16GB.
Tries FastVLA 4-bit presets (pi0-base, smolvla, openvla-7b): load -> VRAM -> 100-step smoke.
NO full train (quota discipline). Evidence -> /kaggle/working/kernelPI.json
"""
import json
import os
import subprocess
import sys
import time

t0 = time.time()
WORK = "/kaggle/working" if os.path.exists("/kaggle") else "."
out: dict = {"stages": {}}


def log(m):
    print(m, flush=True)


r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "fastvla"],
                   capture_output=True, text=True)
log(f"pip fastvla rc={r.returncode}")
out["stages"]["pip"] = {"rc": r.returncode}

import torch  # noqa: E402
log(f"cuda={torch.cuda.is_available()} " +
    (torch.cuda.get_device_name(0) if torch.cuda.is_available() else ""))


def vram():
    if torch.cuda.is_available():
        return {"alloc": round(torch.cuda.memory_allocated() / 1e6, 1),
                "reserved": round(torch.cuda.memory_reserved() / 1e6, 1)}
    return {}


res = {}
try:
    import fastvla  # noqa: E402
    out["stages"]["fastvla"] = {"ver": getattr(fastvla, "__version__", "?"),
                                "attrs": [a for a in dir(fastvla) if not a.startswith("_")][:20]}
    log(f"fastvla attrs: {out['stages']['fastvla']['attrs']}")
except Exception as e:  # noqa: BLE001
    out["abort"] = f"fastvla import: {type(e).__name__}: {e}"
    log(out["abort"])

if "abort" not in out:
    loaders = []
    try:
        from fastvla import FastVLAModel  # noqa: E402
        loaders.append(("fastvla.FastVLAModel", FastVLAModel.from_pretrained))
    except Exception as e:  # noqa: BLE001
        log(f"FastVLAModel: {e}")
    try:
        from fastvla.models import FastVLAModel as F2  # noqa: E402
        loaders.append(("fastvla.models.FastVLAModel", F2.from_pretrained))
    except Exception as e:  # noqa: BLE001
        log(f"models.FastVLAModel: {e}")
    out["stages"]["loaders"] = [n for n, _ in loaders]
    for preset in ["pi0-base", "smolvla", "openvla-7b"]:
        for lname, fn in loaders:
            try:
                m0 = time.time()
                m = fn(preset, load_in_4bit=True, use_peft=True)
                dt = round(time.time() - m0, 1)
                v = vram()
                n = sum(p.numel() for p in m.parameters())
                res[f"{preset}@{lname}"] = {"params": n, "load_s": dt, **v, "FIT": v.get("reserved", 1e9) < 15000}
                log(f"{preset}@{lname}: params={n} {v} load={dt}s")
                del m
                torch.cuda.empty_cache()
                break
            except Exception as e:  # noqa: BLE001
                res[f"{preset}@{lname}"] = {"ERR": f"{type(e).__name__}: {str(e)[:200]}"}
                log(f"{preset}@{lname} FAIL: {type(e).__name__}: {str(e)[:150]}")
out["results"] = res
out["wall_s"] = round(time.time() - t0, 1)
log(f"PIFIT_DONE wall={out['wall_s']}")
with open(os.path.join(WORK, "kernelPI.json"), "w") as f:
    json.dump(out, f)
