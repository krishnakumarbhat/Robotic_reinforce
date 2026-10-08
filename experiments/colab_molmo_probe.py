"""Colab lane PHASE1: MolmoAct2-SO100_101 fit probe (no training, bounded).
pip lerobot -> check molmoact2 policy present -> download weights -> config bytes
-> attempt fp16 load + VRAM report. Prints MOLMO_P1 JSON.
"""
import json
import subprocess
import sys
import time

t0 = time.time()
out: dict = {"stages": {}}


def log(m):
    print(m, flush=True)


r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "lerobot"],
                   capture_output=True, text=True)
log(f"pip lerobot rc={r.returncode}")
try:
    import lerobot
    out["stages"]["lerobot_ver"] = getattr(lerobot, "__version__", "?")
except Exception as e:  # noqa: BLE001
    out["stages"]["lerobot_ver"] = f"FAIL {e}"
log(f"lerobot: {out['stages']['lerobot_ver']}")

import torch  # noqa: E402
log(f"cuda={torch.cuda.is_available()} " +
    (torch.cuda.get_device_name(0) if torch.cuda.is_available() else ""))
tot = torch.cuda.get_device_properties(0).total_memory / 1e9 if torch.cuda.is_available() else 0
out["stages"]["gpu_gb"] = round(tot, 1)

try:
    from lerobot.policies.factory import get_policy_class  # noqa: E402
    cls = get_policy_class("molmoact2")
    out["stages"]["policy_class"] = f"OK {cls.__name__}"
except Exception as e:  # noqa: BLE001
    out["stages"]["policy_class"] = f"MISS {type(e).__name__}: {str(e)[:150]}"
log(f"policy_class: {out['stages']['policy_class']}")

REPO = "allenai/MolmoAct2-SO100_101"
try:
    from huggingface_hub import hf_hub_download  # noqa: E402
    cfg_p = hf_hub_download(REPO, "config.json")
    cfg = json.load(open(cfg_p))
    out["stages"]["config"] = {"keys": sorted(cfg.keys())[:15]}
    log(f"config keys: {out['stages']['config']['keys']}")
except Exception as e:  # noqa: BLE001
    out["stages"]["config"] = f"FAIL {type(e).__name__}: {str(e)[:150]}"
    log(out["stages"]["config"])

# weight bytes via repo file sizes (no full download yet)
try:
    from huggingface_hub import HfApi  # noqa: E402
    info = HfApi().model_info(REPO)
    gb = sum(getattr(s, "size", 0) or 0 for s in (info.siblings or [])
             if (s.rfilename or "").endswith(".safetensors")) / 1e9
    out["stages"]["weights_gb"] = round(gb, 2)
    out["stages"]["n_files"] = len(info.siblings or [])
except Exception as e:  # noqa: BLE001
    out["stages"]["weights_gb"] = f"FAIL {str(e)[:120]}"
log(f"weights_gb: {out['stages']['weights_gb']}")

if out["stages"].get("policy_class", "").startswith("OK") and isinstance(out["stages"].get("weights_gb"), float):
    try:
        from lerobot.policies.factory import make_policy  # noqa: E402
        from lerobot.configs.policies import PreTrainedConfig  # noqa: E402
        cfg_o = PreTrainedConfig.from_pretrained(REPO)
        pol = make_policy(cfg_o, ds_meta=None)
        n = sum(p.numel() for p in pol.parameters())
        out["stages"]["load"] = {
            "params": n,
            "alloc_mb": round(torch.cuda.memory_allocated() / 1e6, 1),
            "reserved_mb": round(torch.cuda.memory_reserved() / 1e6, 1)}
    except Exception as e:  # noqa: BLE001
        import traceback
        out["stages"]["load"] = f"FAIL {type(e).__name__}: {str(e)[:200]}"
        log(traceback.format_exc()[-600:])
    log(f"load: {out['stages']['load']}")
out["wall_s"] = round(time.time() - t0, 1)
print("MOLMO_P1 " + json.dumps(out)[:2500], flush=True)
