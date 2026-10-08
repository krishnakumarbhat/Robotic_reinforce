# Edge-budget measurement: local SmolVLA ckpt, fp16, cuda:0, 20 timed forward passes.
# ponytail: single-shot script, no CLI flags; dummy obs only (no real env).
import json
import os
import statistics
import sys
import time

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CKPT = os.path.join(REPO, "checkpoints", "g3_10k_local")
OUT = os.path.join(REPO, "results", "edge_budget_local.json")
N_WARMUP, N_ITERS = 5, 20

import torch

from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy


def build_dummy_frame(cfg):
    """Purpose: build one raw obs frame in the ckpt's native feature naming (image/image2, 8-dim state).
    Inputs: policy cfg. Outputs: dict frame."""
    # ponytail: ckpt normalizer stats are keyed observation.images.image{,2} + state(8),
    # preprocessor renames image->camera1/image2->camera2; camera3 has no stats and is skipped.
    frame = {
        "observation.state": torch.zeros(8),
        "observation.images.image": torch.zeros(3, 256, 256, dtype=torch.uint8),
        "observation.images.image2": torch.zeros(3, 256, 256, dtype=torch.uint8),
        "task": "clean the restroom fixture",
        "robot_type": "panda",
    }
    return frame


def main():
    """Purpose: time fp16 inference and emit edge-budget JSON. Inputs: none. Outputs: results/edge_budget_local.json."""
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    policy = SmolVLAPolicy.from_pretrained(CKPT, local_files_only=True)
    load_s = time.perf_counter() - t0
    policy = policy.half().to("cuda:0").eval()

    pre, _post = make_pre_post_processors(
        policy.config, CKPT, preprocessor_overrides={"device_processor": {"device": "cuda:0"}}
    )
    batch = pre(build_dummy_frame(policy.config))
    batch = {k: v for k, v in batch.items() if torch.is_tensor(v)}
    for k, v in batch.items():
        if v.dtype == torch.uint8:  # images arrive 0-255; model expects [0,1]
            batch[k] = (v.float() / 255.0).half()
        elif torch.is_floating_point(v):
            batch[k] = v.half()

    # ponytail: weights are fp16; the flow-matching state (x_t) is created in fp32,
    # so autocast keeps every matmul in fp16 instead of patching lerobot internals.
    autocast = torch.autocast(device_type="cuda", dtype=torch.float16)
    with torch.no_grad(), autocast:
        for _ in range(N_WARMUP):
            policy.predict_action_chunk(batch)
        torch.cuda.synchronize()

        times_ms = []
        for _ in range(N_ITERS):
            torch.cuda.synchronize()
            t = time.perf_counter()
            policy.predict_action_chunk(batch)
            torch.cuda.synchronize()
            times_ms.append((time.perf_counter() - t) * 1000.0)

    vram_mib = torch.cuda.max_memory_allocated() / (1024**2)
    median_ms = statistics.median(times_ms)
    out = {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(0),
        "precision": "fp16",
        "ckpt": CKPT,
        "cameras": [k for k in batch if k.startswith("observation.images")],
        "warmup": N_WARMUP,
        "iters": N_ITERS,
        "load_s": round(load_s, 2),
        "smolvla_ms_median": round(median_ms, 2),
        "smolvla_ms_all": [round(x, 2) for x in times_ms],
        "smolvla_vram_mib": round(vram_mib, 1),
        "fits_1p5GB": vram_mib <= 1536,
        "hits_25ms": median_ms <= 25.0,
    }
    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception as e:  # noqa: BLE001 - surface full trace for triage
        import traceback

        traceback.print_exc()
        sys.exit(1)
