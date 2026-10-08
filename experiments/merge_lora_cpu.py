"""Merge step10607 LoRA adapter onto step917 full weights (raw tensors, CPU).
W' = W + (alpha/r) * (B @ A), alpha/r = 2.0; modules_to_save copied verbatim.
Key rule: adapter 'base_model.model.<k>' == base '<k>' (verified by assertion).
Verifies: every mapped key exists in base; reports delta norms; writes merged.
"""
import torch
from safetensors import safe_open
from safetensors.torch import save_file

BASE = "connect_gpu/outputs/merge_tmp/model.safetensors"
AD = "connect_gpu/outputs/aegis_ft/ckpt_step10607/adapter_model.safetensors"
OUT = "connect_gpu/outputs/smolvla-aegis-ft-step10607-merged.safetensors"
PREFIX = "base_model.model."
SCALE = 32.0 / 16.0

print("loading base...", flush=True)
base = {}
with safe_open(BASE, framework="pt") as f:
    for k in f.keys():
        base[k] = f.get_tensor(k)
print(f"base: {len(base)} tensors", flush=True)

print("loading adapter...", flush=True)
ad = {}
with safe_open(AD, framework="pt") as f:
    for k in f.keys():
        ad[k] = f.get_tensor(k)
print(f"adapter: {len(ad)} tensors", flush=True)

# group LoRA pairs
pairs = {}
full = {}
for k, v in ad.items():
    if ".lora_A.weight" in k:
        stem = k.replace(".lora_A.weight", "")
        pairs.setdefault(stem, {})["A"] = v
    elif ".lora_B.weight" in k:
        stem = k.replace(".lora_B.weight", "")
        pairs.setdefault(stem, {})["B"] = v
    else:
        full[k] = v
assert all(set(p) == {"A", "B"} for p in pairs.values()), "unpaired LoRA!"
print(f"lora targets: {len(pairs)}, full tensors: {len(full)}", flush=True)

missing, dn = [], []
merged = dict(base)
for stem, p in pairs.items():
    assert stem.startswith(PREFIX), stem
    bk = stem[len(PREFIX):] + ".weight"  # LoRA targets the module; base stores the .weight tensor
    if bk not in base:
        missing.append(bk)
        continue
    d = (p["B"].float() @ p["A"].float()) * SCALE
    assert d.shape == base[bk].shape, (bk, d.shape, base[bk].shape)
    merged[bk] = (base[bk].float() + d).to(base[bk].dtype)
    dn.append(float(d.norm()))
n_full, n_full_missing = 0, []
for k, v in full.items():
    assert k.startswith(PREFIX), k
    bk = k[len(PREFIX):]
    if bk not in base:
        n_full_missing.append(bk)
        continue
    merged[bk] = v.to(base[bk].dtype) if v.dtype != base[bk].dtype else v
    n_full += 1
print(f"merged-lora: {len(dn)}, missing-lora: {len(missing)}")
print(f"copied-full: {n_full}, missing-full: {len(n_full_missing)}")
for m in (missing + n_full_missing)[:5]:
    print("  MISSING:", m)
assert not missing and not n_full_missing, "key mismatch -- abort, no file written"
import statistics
print(f"mean|delta|={statistics.mean(dn):.6f} max={max(dn):.6f}")
save_file(merged, OUT)
print("wrote", OUT, flush=True)
