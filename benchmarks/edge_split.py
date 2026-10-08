# I4 Step A: split SmolVLA chunk latency into VLM-prefix (1x) vs expert per-step (xN).
# Method: time full chunk at num_steps=10 (default) and num_steps=1; with linearity,
# expert_per_step = (t10 - t1) / 9, prefix = t1 - expert_per_step. Verdict for I4:
# if prefix alone > 20 ms, 1-step flow cannot reach 25 ms without prefix KV-cache/layer cuts.
import json
import os
import statistics
import sys
import time

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CKPT = os.path.join(REPO, "checkpoints", "g3_10k_local")
OUT = os.path.join(REPO, "results", "edge_split.json")
N_WARMUP, N_ITERS = 3, 10

import torch

from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy


def timed(policy, batch, n, label):
    policy.config.num_steps = n
    autocast = torch.autocast(device_type="cuda", dtype=torch.float16)
    with torch.no_grad(), autocast:
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
    m = statistics.median(ts)
    print(f"{label}: num_steps={n} median={m:.1f}ms", flush=True)
    return m


def main():
    torch.cuda.empty_cache()
    policy = SmolVLAPolicy.from_pretrained(CKPT, local_files_only=True)
    policy = policy.half().to("cuda:0").eval()
    pre, _post = make_pre_post_processors(
        policy.config, CKPT, preprocessor_overrides={"device_processor": {"device": "cuda:0"}})
    frame = {
        "observation.state": torch.zeros(8),
        "observation.images.image": torch.zeros(3, 256, 256, dtype=torch.uint8),
        "observation.images.image2": torch.zeros(3, 256, 256, dtype=torch.uint8),
        "task": "clean the restroom fixture",
        "robot_type": "panda",
    }
    batch = pre(frame)
    batch = {k: v for k, v in batch.items() if torch.is_tensor(v)}
    for k, v in batch.items():
        if v.dtype == torch.uint8:
            batch[k] = (v.float() / 255.0).half()
        elif torch.is_floating_point(v):
            batch[k] = v.half()
    t10 = timed(policy, batch, 10, "full")
    t1 = timed(policy, batch, 1, "single")
    per_step = (t10 - t1) / 9.0
    prefix = t1 - per_step
    out = {"t10_ms": round(t10, 1), "t1_ms": round(t1, 1),
           "expert_per_step_ms": round(per_step, 1), "vlm_prefix_ms": round(prefix, 1),
           "verdict_1step_ms": round(prefix + per_step, 1)}
    json.dump(out, open(OUT, "w"), indent=1)
    print(json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
