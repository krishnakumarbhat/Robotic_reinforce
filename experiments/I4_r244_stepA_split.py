"""I4 step A — split the 285 ms SmolVLA chunk into VLM-prefix vs flow-expert-step.

Purpose: decide whether the 1-NFE shortcut (I4) can reach the <=25 ms Jetson budget at
all. The prefix pass (image+lang embedding + 16 VLM decoder layers, KV cache filled once)
runs ONCE per chunk; the expert step runs num_steps times. A 1-NFE model removes the
repeated expert steps but CANNOT remove the prefix, so the prefix time is a hard floor.
If floor > 25 ms the idea is falsified with zero GPU spend.

Inputs: checkpoints/g3_10k_local (local, fp16, cuda:0), dummy obs, offline HF.
Outputs: results/I4_r244_stepA_split.json

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
OUT = os.path.join(REPO, "results", "I4_r244_stepA_split.json")
N_WARMUP, N_ITERS = 5, 20

import torch
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAConfig, SmolVLAPolicy


def build_batch(policy):
    """Purpose: dummy obs in the ckpt's native feature naming -> preprocessed fp16 batch.
    Inputs: loaded policy. Outputs: dict[str, Tensor] ready for predict_action_chunk."""
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


def timed(fn, n=N_ITERS, warmup=N_WARMUP):
    """Purpose: median wall ms of fn() over n iterations after warmup.
    Inputs: callable fn, int n, int warmup. Outputs: float median ms."""
    with torch.no_grad():
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        ts = []
        for _ in range(n):
            torch.cuda.synchronize()
            t = time.perf_counter()
            fn()
            torch.cuda.synchronize()
            ts.append((time.perf_counter() - t) * 1000.0)
    return round(statistics.median(ts), 2), [round(x, 2) for x in ts]


def split_timed(policy, batch, n=N_ITERS, warmup=N_WARMUP):
    """Purpose: time one real chunk while charging embed_prefix and denoise_step separately.
    Inputs: policy, preprocessed batch, int n, int warmup.
    Outputs: dict with median total/prefix/step-expert ms and the step-call count.

    The prefix is the 16-layer VLM pass over image+language tokens whose KV cache is filled
    ONCE per chunk; each denoise_step is one flow-matching NFE that reuses that cache. The
    split is measured by wrapping the two real methods, so no prefix argument reconstruction
    is needed and the numbers are the ones the shipped inference path actually pays.
    """
    model = policy.model
    acc = {"prefix": 0.0, "step": 0.0, "calls": 0}
    orig_prefix, orig_step = model.embed_prefix, model.denoise_step

    def timed_prefix(*a, **k):
        torch.cuda.synchronize()
        t = time.perf_counter()
        out = orig_prefix(*a, **k)
        torch.cuda.synchronize()
        acc["prefix"] += (time.perf_counter() - t) * 1000.0
        return out

    def timed_step(*a, **k):
        torch.cuda.synchronize()
        t = time.perf_counter()
        out = orig_step(*a, **k)
        torch.cuda.synchronize()
        acc["step"] += (time.perf_counter() - t) * 1000.0
        acc["calls"] += 1
        return out

    model.embed_prefix, model.denoise_step = timed_prefix, timed_step
    try:
        with torch.no_grad():
            for _ in range(warmup):
                policy.predict_action_chunk(batch)
            torch.cuda.synchronize()
            rows = []
            for _ in range(n):
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
                        "step_expert": acc["step"],
                        "calls": acc["calls"],
                    }
                )
    finally:
        model.embed_prefix, model.denoise_step = orig_prefix, orig_step

    med = lambda k: round(statistics.median(r[k] for r in rows), 2)  # noqa: E731
    return {
        "total_ms": med("total"),
        "prefix_ms": med("prefix"),
        "step_expert_ms": med("step_expert"),
        "step_calls": int(statistics.median(r["calls"] for r in rows)),
        "all_total_ms": [round(r["total"], 2) for r in rows],
    }


def main():
    """Purpose: measure prefix / expert-step / total split and the implied 1-NFE floor.
    Inputs: none. Outputs: results/I4_r244_stepA_split.json (stdout echo)."""
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    policy = SmolVLAPolicy.from_pretrained(CKPT, local_files_only=True)
    policy = policy.half().to("cuda:0").eval()
    batch = build_batch(policy)

    model = policy.model
    num_steps = model.config.num_steps
    autocast = torch.autocast(device_type="cuda", dtype=torch.float16)

    with torch.no_grad(), autocast:
        ten = split_timed(policy, batch)
        vram = torch.cuda.max_memory_allocated() / (1024**2)
        model.config.num_steps = 1
        one = split_timed(policy, batch)
        model.config.num_steps = num_steps

    out = {
        "idea": "I4",
        "step": "A",
        "device": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "precision": "fp16",
        "ckpt": CKPT,
        "num_steps_orig": num_steps,
        "chunk_size": model.config.chunk_size,
        "num_expert_layers": model.config.num_expert_layers,
        "num_vlm_layers": model.config.num_vlm_layers,
        "ten_step": ten,
        "one_step": one,
        "expert_step_amortised_ms": round(
            ten["step_expert_ms"] / max(1, ten["step_calls"]), 2
        ),
        "vram_mib": round(vram, 1),
        "budget_ms": 25.0,
        "prefix_alone_hits_25ms": ten["prefix_ms"] <= 25.0,
        "measured_1nfe_hits_25ms": one["total_ms"] <= 25.0,
        "speedup_vs_10step": round(ten["total_ms"] / max(1e-9, one["total_ms"]), 2),
        "share_of_total_prefix": round(ten["prefix_ms"] / max(1e-9, ten["total_ms"]), 4),
        "fits_1p5GB": vram <= 1536,
    }
    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
