"""I4 step A2 — is the 54.9 ms VLM prefix reducible, or is the 25 ms budget structurally dead?

I4 step A (results/I4_r244_stepA_split.json) measured 1-NFE chunk = 82.09 ms, of which the
VLM prefix (embed images+lang, 16 VLM decoder layers, KV cache filled once) is 54.87 ms and
the amortised flow-expert step is 11.4 ms. The prefix is paid ONCE per chunk and no expert-side
shortcut can touch it, so 1-NFE is falsified on its own pre-registered bar (<=25 ms).

The open question this answers: what actually sets the prefix cost? If it scales with image
tokens, the lever is camera count / input resolution, NOT the number of flow steps -- which
redirects the edge-latency line away from shortcut models entirely. If it is flat, the 25 ms
budget is dead for this architecture class and the whole premise needs restating.

Inputs: checkpoints/g3_10k_local, local fp16 cuda:0, dummy obs, offline HF.
Outputs: results/I4_r244_stepA_prefix_scaling.json
"""
import json
import os
import statistics
import time

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CKPT = os.path.join(REPO, "checkpoints", "g3_10k_local")
OUT = os.path.join(REPO, "results", "I4_r244_stepA_prefix_scaling.json")
N_WARMUP, N_ITERS = 5, 15

import torch
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy


def build_batch(resize, n_cams):
    """Purpose: dummy obs at a given resize and camera count -> preprocessed fp16 batch.
    Inputs: resize [h,w] or None (ckpt default), int n_cams. Outputs: dict[str, Tensor]."""
    from lerobot.policies.factory import make_pre_post_processors

    policy = SmolVLAPolicy.from_pretrained(CKPT, local_files_only=True)
    if resize is not None:
        policy.config.resize_imgs_with_padding = list(resize)
        policy.model.config.resize_imgs_with_padding = list(resize)
    pre, _ = make_pre_post_processors(
        policy.config, CKPT, preprocessor_overrides={"device_processor": {"device": "cuda:0"}}
    )
    frame = {
        "observation.state": torch.zeros(8),
        "task": "clean the restroom fixture",
        "robot_type": "panda",
    }
    for i in range(n_cams):
        frame[f"observation.images.image" if i == 0 else f"observation.images.image{i + 1}"] = (
            torch.zeros(3, 256, 256, dtype=torch.uint8)
        )
    batch = pre(frame)
    out = {k: v for k, v in batch.items() if torch.is_tensor(v)}
    for k, v in out.items():
        if v.dtype == torch.uint8:
            out[k] = (v.float() / 255.0).half()
        elif torch.is_floating_point(v):
            out[k] = v.half()
    del policy
    torch.cuda.empty_cache()
    return out


def prefix_only(policy, batch):
    """Purpose: run the shipped prefix path (embed + VLM forward + KV fill) and time it.
    Inputs: policy, batch. Outputs: (median_ms, prefix_token_len)."""
    model = policy.model
    orig = model.embed_prefix
    holder = {}

    def wrapped(*a, **k):
        out = orig(*a, **k)
        holder["len"] = int(out[0].shape[1])
        return out

    model.embed_prefix = wrapped
    try:
        with torch.no_grad():
            for _ in range(N_WARMUP):
                policy.predict_action_chunk(batch)
            torch.cuda.synchronize()
            ts = []
            for _ in range(N_ITERS):
                torch.cuda.synchronize()
                t = time.perf_counter()
                policy.predict_action_chunk(batch)
                torch.cuda.synchronize()
                ts.append((time.perf_counter() - t) * 1000.0)
    finally:
        model.embed_prefix = orig
    return round(statistics.median(ts), 2), holder.get("len")


def main():
    """Purpose: prefix cost vs camera count and input resolution.
    Inputs: none. Outputs: results/I4_r244_stepA_prefix_scaling.json (stdout echo)."""
    torch.cuda.empty_cache()
    policy = SmolVLAPolicy.from_pretrained(CKPT, local_files_only=True)
    policy = policy.half().to("cuda:0").eval()

    cases = [
        ("2cam_512", [512, 512], 2),  # shipped configuration
        ("1cam_512", [512, 512], 1),
        ("2cam_256", [256, 256], 2),
        ("1cam_256", [256, 256], 1),
    ]
    autocast = torch.autocast(device_type="cuda", dtype=torch.float16)
    rows = []
    for name, resize, ncams in cases:
        batch = build_batch(resize, ncams)
        with torch.no_grad(), autocast:
            total, tlen = prefix_only(policy, batch)
        rows.append(
            {
                "case": name,
                "resize": resize,
                "n_cams": ncams,
                "chunk_total_ms_10step": total,
                "prefix_tokens": tlen,
                "hits_25ms_1nfe": False,
            }
        )
        print(name, total, "tokens", tlen, flush=True)

    shipped = next(r for r in rows if r["case"] == "2cam_512")
    best = min(rows, key=lambda r: r["chunk_total_ms_10step"])
    out = {
        "idea": "I4",
        "step": "A2",
        "device": torch.cuda.get_device_name(0),
        "rows": rows,
        "shipped_total_ms": shipped["chunk_total_ms_10step"],
        "best_case": best["case"],
        "best_total_ms": best["chunk_total_ms_10step"],
        "note": (
            "chunk_total_ms_10step is the FULL 10-NFE chunk, not the prefix alone; the "
            "prefix share of the shipped case is 54.87/184.99 = 29.7% (run A). Halving "
            "resolution or dropping a camera is compared against the 25 ms budget here."
        ),
    }
    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
