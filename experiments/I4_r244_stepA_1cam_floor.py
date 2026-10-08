"""I4 step A3 — the prefix floor with the cheapest possible input (1 camera).

Step A measured the shipped 2-cam configuration: 1-NFE chunk 82.09 ms, prefix 54.87 ms
(= 29.7% of the 184.99 ms 10-NFE chunk), amortised expert step 11.4 ms. I4's pre-registered
bar is a 1-NFE chunk <= 25 ms, so the expert side can afford 25 - 54.87 = -29.9 ms: already
impossible before any shortcut model is trained. This run measures the same split with the
cheapest input the stack can take (1 camera, the smallest supported token count) to state how
far the architecture class is from the budget, i.e. whether "one NFE" was ever the binding
constraint or whether the prefix was.

Inputs: checkpoints/g3_10k_local, local fp16 cuda:0, dummy obs, offline HF.
Outputs: results/I4_r244_stepA_1cam_floor.json
"""
import json
import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from I4_r244_stepA_split import CKPT, split_timed  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, "results", "I4_r244_stepA_1cam_floor.json")

import torch  # noqa: E402
from lerobot.policies.factory import make_pre_post_processors  # noqa: E402
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy  # noqa: E402


def one_cam_batch(policy):
    """Purpose: single-camera dummy obs in the ckpt's native naming -> fp16 batch.
    Inputs: policy. Outputs: dict[str, Tensor]."""
    frame = {
        "observation.state": torch.zeros(8),
        "observation.images.image": torch.zeros(3, 256, 256, dtype=torch.uint8),
        "task": "clean the restroom fixture",
        "robot_type": "panda",
    }
    pre, _ = make_pre_post_processors(
        policy.config, CKPT, preprocessor_overrides={"device_processor": {"device": "cuda:0"}}
    )
    batch = {k: v for k, v in pre(frame).items() if torch.is_tensor(v)}
    for k, v in batch.items():
        if v.dtype == torch.uint8:
            batch[k] = (v.float() / 255.0).half()
        elif torch.is_floating_point(v):
            batch[k] = v.half()
    return batch


def main():
    """Purpose: prefix / expert split for the 1-camera input at 10 and 1 NFE.
    Inputs: none. Outputs: results/I4_r244_stepA_1cam_floor.json (stdout echo)."""
    torch.cuda.empty_cache()
    policy = SmolVLAPolicy.from_pretrained(CKPT, local_files_only=True)
    policy = policy.half().to("cuda:0").eval()
    batch = one_cam_batch(policy)
    num_steps = policy.model.config.num_steps

    autocast = torch.autocast(device_type="cuda", dtype=torch.float16)
    with torch.no_grad(), autocast:
        ten = split_timed(policy, batch)
        policy.model.config.num_steps = 1
        one = split_timed(policy, batch)
        policy.model.config.num_steps = num_steps

    out = {
        "idea": "I4",
        "step": "A3",
        "device": torch.cuda.get_device_name(0),
        "n_cams": 1,
        "one_step": one,
        "ten_step": ten,
        "budget_ms": 25.0,
        "prefix_alone_hits_25ms": ten["prefix_ms"] <= 25.0,
        "measured_1nfe_hits_25ms": one["total_ms"] <= 25.0,
    }
    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
