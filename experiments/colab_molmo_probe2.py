"""Colab lane PHASE2: MolmoAct2-SO100_101 file inventory + load attempt."""
import json
import subprocess
import sys
import time

t0 = time.time()
out: dict = {"stages": {}}


def log(m):
    print(m, flush=True)


r = subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                    "lerobot[dataset]", "av", "num2words"],
                   capture_output=True, text=True)
log(f"pip rc={r.returncode}")

import torch  # noqa: E402
from huggingface_hub import HfApi, hf_hub_download  # noqa: E402

REPO = "allenai/MolmoAct2-SO100_101"
info = HfApi().model_info(REPO)
fns = [s.rfilename for s in (info.siblings or [])]
out["stages"]["files"] = fns[:30]
out["stages"]["n_files"] = len(fns)
gb = sum(getattr(s, "size", 0) or 0 for s in (info.siblings or [])) / 1e9
out["stages"]["total_gb"] = round(gb, 2)
log(f"files={len(fns)} total_gb={round(gb,2)}")
for f in fns[:30]:
    log(f"  {f}")

try:
    from lerobot.policies.factory import get_policy_class  # noqa: E402
    cls = get_policy_class("molmoact2")
    out["stages"]["policy_class"] = f"OK {cls.__name__}"
except Exception as e:  # noqa: BLE001
    out["stages"]["policy_class"] = f"MISS {type(e).__name__}: {str(e)[:200]}"
log(f"policy_class: {out['stages']['policy_class']}")

if out["stages"]["policy_class"].startswith("OK"):
    try:
        from lerobot.configs.policies import PreTrainedConfig  # noqa: E402
        from lerobot.policies.factory import make_policy  # noqa: E402
        cfg_o = PreTrainedConfig.from_pretrained(REPO)
        m0 = time.time()
        pol = make_policy(cfg_o, ds_meta=None)
        n = sum(p.numel() for p in pol.parameters())
        out["stages"]["load"] = {
            "params": n, "load_s": round(time.time() - m0, 1),
            "alloc_mb": round(torch.cuda.memory_allocated() / 1e6, 1),
            "reserved_mb": round(torch.cuda.memory_reserved() / 1e6, 1)}
    except Exception as e:  # noqa: BLE001
        import traceback
        out["stages"]["load"] = f"FAIL {type(e).__name__}: {str(e)[:250]}"
        log(traceback.format_exc()[-700:])
    log(f"load: {out['stages']['load']}")
out["wall_s"] = round(time.time() - t0, 1)
print("MOLMO_P2 " + json.dumps(out)[:3000], flush=True)
