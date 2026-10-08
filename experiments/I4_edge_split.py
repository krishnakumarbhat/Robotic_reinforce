"""I4 step A — design input for the 1-step expert: prefix / one expert step / full chunk.

Purpose: split the shipped SmolVLA chunk into (a) the VLM-prefix pass that runs ONCE per
chunk (SigLIP + lang + state embed, then the 16-layer VLM decoder filling the KV cache),
(b) ONE flow-matching denoise step that reuses that cache, (c) the full 10-step chunk.
(a) is a hard floor for any 1-NFE scheme, so it decides whether 1 step alone reaches 25 ms.

Inputs: checkpoints/g3_10k_local (fp16, cuda:0, batch 1, 2x256px + 8-dim state), dummy obs.
Outputs: results/edge_split.json

ponytail: single-shot timing script, no CLI flags, no training, no Colab.
"""
import json
import os
import statistics
import time

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CKPT = os.path.join(REPO, "checkpoints", "g3_10k_local")
OUT = os.path.join(REPO, "results", "edge_split.json")
N_WARMUP, N_ITERS = 5, 20
BUDGET_MS, PREFIX_FLOOR_MS = 25.0, 20.0

import torch
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy


def build_batch(policy):
    """Purpose: dummy obs in ckpt-native feature naming -> preprocessed fp16 cuda batch.
    Inputs: loaded policy. Outputs: dict[str, Tensor]."""
    frame = {
        "observation.state": torch.zeros(8),
        "observation.images.image": torch.zeros(3, 256, 256, dtype=torch.uint8),
        "observation.images.image2": torch.zeros(3, 256, 256, dtype=torch.uint8),
        "task": "clean the restroom fixture",
        "robot_type": "panda",
    }
    pre, _ = make_pre_post_processors(
        policy.config, CKPT, preprocessor_overrides={"device_processor": {"device": "cuda:0"}}
    )
    batch = pre(frame)
    batch = {k: v for k, v in batch.items() if torch.is_tensor(v)}
    for k, v in batch.items():
        if v.dtype == torch.uint8:
            batch[k] = (v.float() / 255.0).half()
        elif torch.is_floating_point(v):
            batch[k] = v.half()
    return batch


def split_timed(policy, batch):
    """Purpose: time one real chunk, charging embed, KV-fill VLM forward and denoise_step
    separately; each region is fenced by torch.cuda.synchronize() so the medians are real
    kernel time, not queue latency.
    Inputs: loaded policy, preprocessed batch. Outputs: dict of median ms + raw rows."""
    model = policy.model
    acc = {"prefix": 0.0, "step": 0.0, "calls": 0, "t0": 0.0}
    orig_embed, orig_step = model.embed_prefix, model.denoise_step
    orig_fwd = model.vlm_with_expert.forward

    def timed_embed(*a, **k):
        """Window opener for (a): embed (SigLIP + lang + state) -> KV-fill forward."""
        torch.cuda.synchronize()
        acc["t0"] = time.perf_counter()
        return orig_embed(*a, **k)

    def timed_fwd(*a, **k):
        """Window closer for (a): only the once-per-chunk prefix KV fill ends the window;
        the 10 suffix forwards in denoise_step pass fill_kv_cache=False and are skipped."""
        out = orig_fwd(*a, **k)
        if k.get("fill_kv_cache"):
            torch.cuda.synchronize()
            acc["prefix"] += (time.perf_counter() - acc["t0"]) * 1000.0
        return out

    def timed_step(*a, **k):
        """(b): one flow-expert denoise step, fence-synced both sides."""
        torch.cuda.synchronize()
        t = time.perf_counter()
        out = orig_step(*a, **k)
        torch.cuda.synchronize()
        acc["step"] += (time.perf_counter() - t) * 1000.0
        acc["calls"] += 1
        return out

    model.embed_prefix, model.denoise_step = timed_embed, timed_step
    model.vlm_with_expert.forward = timed_fwd
    try:
        with torch.no_grad():
            for _ in range(N_WARMUP):
                policy.predict_action_chunk(batch)
            torch.cuda.synchronize()
            rows = []
            for _ in range(N_ITERS):
                for k in acc:
                    acc[k] = 0
                torch.cuda.synchronize()
                t = time.perf_counter()
                policy.predict_action_chunk(batch)
                torch.cuda.synchronize()
                rows.append(
                    {
                        "total": (time.perf_counter() - t) * 1000.0,
                        "prefix": acc["prefix"],
                        "step": acc["step"],
                        "calls": acc["calls"],
                    }
                )
    finally:
        model.embed_prefix, model.denoise_step = orig_embed, orig_step
        model.vlm_with_expert.forward = orig_fwd

    med = lambda k: round(statistics.median(r[k] for r in rows), 2)  # noqa: E731
    return {
        "total_ms": med("total"),
        "prefix_ms": med("prefix"),
        "step_ms": med("step"),
        "step_calls": int(statistics.median(r["calls"] for r in rows)),
        "all_total_ms": [round(r["total"], 2) for r in rows],
    }


def main():
    """Purpose: emit results/edge_split.json with the 3 medians, peak VRAM and the verdict.
    Inputs: none. Outputs: results/edge_split.json (stdout echo)."""
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    policy = SmolVLAPolicy.from_pretrained(CKPT, local_files_only=True)
    policy = policy.half().to("cuda:0").eval()
    batch = build_batch(policy)
    model = policy.model
    num_steps = model.config.num_steps
    autocast = torch.autocast(device_type="cuda", dtype=torch.float16)

    with torch.no_grad(), autocast:
        full = split_timed(policy, batch)  # (a) prefix + (c) 10-step chunk
        model.config.num_steps = 1
        one = split_timed(policy, batch)  # (b) a single expert step
        model.config.num_steps = num_steps

    vram = torch.cuda.max_memory_allocated() / (1024**2)
    prefix_ms, step_ms, full_ms = full["prefix_ms"], one["step_ms"], full["total_ms"]
    verdict = (
        f"1-step alone CANNOT hit {BUDGET_MS:.0f}ms: prefix floor {prefix_ms}ms > "
        f"{PREFIX_FLOOR_MS}ms, needs prefix KV-cache reuse across chunks + fewer VLM layers"
        if prefix_ms > PREFIX_FLOOR_MS
        else f"1-step suffices: prefix {prefix_ms}ms <= {PREFIX_FLOOR_MS}ms, "
        f"{prefix_ms}+{step_ms}={round(prefix_ms + step_ms, 2)}ms vs {BUDGET_MS:.0f}ms budget"
    )
    out = {
        "vlm_prefix_ms": prefix_ms,
        "expert_step_ms": step_ms,
        "full_chunk_ms": full_ms,
        "vram_mib": round(vram, 1),
        "verdict": verdict,
    }
    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))
    print("num_steps", num_steps, "step_calls_full", full["step_calls"],
          "full_all", full["all_total_ms"])
    print("one_all", one["all_total_ms"], "one_prefix", one["prefix_ms"])


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback

        traceback.print_exc()
        raise
