"""Local RTX 3050: first real forward pass of OUR tuned ckpt (zero quota).
Loads krishnah27/smolvla-pusht-v15k-b8, dummy PushT obs, times select_action.
Evidence printed as JSON.
"""
import json
import os
import time

import numpy as np
import torch

REPO = "krishnah27/smolvla-pusht-v15k-b8"
out: dict = {"repo": REPO}


def log(m):
    print(m, flush=True)


log(f"cuda={torch.cuda.is_available()} " +
    (torch.cuda.get_device_name(0) if torch.cuda.is_available() else ""))

# ponytail: shim for transformers-4.57 x hub-0.35 skew — list_repo_templates
# misses RemoteEntryNotFoundError (only chat-template bonus listing; safe []).
try:
    import transformers.utils.hub as _thub
    _orig_lrt = _thub.list_repo_templates

    def _safe_lrt(*a, **k):
        try:
            return _orig_lrt(*a, **k)
        except Exception:
            return []

    _thub.list_repo_templates = _safe_lrt
    try:
        import transformers.processing_utils as _pu
        _pu.list_repo_templates = _safe_lrt
    except Exception:
        pass
    log("chat-template shim on")
except Exception as e:  # noqa: BLE001
    log(f"shim skipped: {e}")
t0 = time.time()
try:
    from huggingface_hub import snapshot_download
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
    from lerobot.policies.factory import make_policy
    from lerobot.configs.policies import PreTrainedConfig
    snap = snapshot_download(REPO, local_dir="checkpoints/v15k_local")
    log(f"snapshot: {snap}")
    # compat: new-lerobot ckpt configs carry fields old draccus rejects;
    # drop offenders iteratively (max 10) against the live decoder
    import re as _re
    cfg = None
    for _ in range(10):
        try:
            cfg = PreTrainedConfig.from_pretrained(snap)
            break
        except Exception as e:  # noqa: BLE001
            m = _re.search(r"The fields `([^`]+)` are not valid", str(e))
            if not m:
                raise
            bad = m.group(1).split(", ")
            cp = os.path.join(snap, "config.json")
            d = json.load(open(cp))
            for k in bad:
                d.pop(k.strip("` "), None)
            json.dump(d, open(cp, "w"), indent=1)
            log(f"dropped config keys: {bad}")
    log(f"config: {type(cfg).__name__}")
    cfg.pretrained_path = snap  # ckpt stores remote train path; point at local copy
    try:
        import sys as _sys
        _o, _s = globals()["_orig_lrt"], globals()["_safe_lrt"]
        for _mod in list(_sys.modules.values()):
            try:
                if getattr(_mod, "list_repo_templates", None) is _o:
                    setattr(_mod, "list_repo_templates", _s)
            except Exception:
                pass
        try:
            import transformers.models.smolvlm.processing_smolvlm as _sp
            _sp.list_repo_templates = _s
        except Exception:
            pass
        log("namespace sweep done")
    except KeyError:
        log("shim names missing, sweep skipped")
    ds_meta = LeRobotDatasetMetadata("lerobot/pusht_image")
    from lerobot.policies.factory import make_policy as _mp
    import inspect as _insp
    _kw = {"rename_map": {"observation.image": "observation.images.camera1"}}
    if "rename_map" not in _insp.signature(_mp).parameters:
        _kw = {}
        log("make_policy lacks rename_map; trying raw")
    m0 = time.time()
    pol = _mp(cfg, ds_meta=ds_meta, **_kw)
    pol.eval()
    n = sum(p.numel() for p in pol.parameters())
    out["load"] = {"params": n, "load_s": round(time.time() - m0, 1),
                   "alloc_mb": round(torch.cuda.memory_allocated() / 1e6, 1),
                   "reserved_mb": round(torch.cuda.memory_reserved() / 1e6, 1)}
    log(f"loaded: {out['load']}")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(
        "HuggingFaceTB/SmolVLM2-500M-Video-Instruct", trust_remote_code=True)
    enc = tok(["push the block"], return_tensors="pt", padding=True,
              truncation=True, max_length=64)
    lang_mask = enc["attention_mask"]
    # eager attention path wants bool masks (preprocessor normally casts)
    if lang_mask.dtype != torch.bool:
        lang_mask = lang_mask.bool()
    batch = {
        "observation.images.camera1": torch.zeros(1, 3, 256, 256),
        "observation.state": torch.zeros(1, 2),
        "observation.language.tokens": enc["input_ids"],
        "observation.language.attention_mask": lang_mask,
        "task": ["push the block"],
    }
    # warmup + timed
    dev = next(pol.parameters()).device
    batch = {k: (v.to(dev) if isinstance(v, torch.Tensor) else v)
             for k, v in batch.items()}
    with torch.no_grad():
        a = pol.select_action(batch)
        t1 = time.time()
        for _ in range(5):
            a = pol.select_action(batch)
        dt = (time.time() - t1) / 5 * 1000
    out["infer"] = {"action_shape": list(a.shape),
                    "ms_per_call": round(dt, 1),
                    "action_mean": round(float(a.mean()), 4)}
    log(f"infer: {out['infer']}")
except Exception as e:  # noqa: BLE001
    import traceback
    out["error"] = f"{type(e).__name__}: {str(e)[:300]}"
    log(traceback.format_exc()[-1200:])
out["wall_s"] = round(time.time() - t0, 1)
print("LOCAL_INFER " + json.dumps(out)[:1200], flush=True)
