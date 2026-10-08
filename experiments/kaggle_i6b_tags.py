"""Kernel I6b (v48): does the FT adapter CONDITION on quality tags? (static probe)

v47 died: in-process mujoco+torch segfault at LIBERO env build (the CLI path in
kaggle_i6_eval.py works; in-process does not). Lazy fix that answers the SAME
first-order question without any simulator: hold observation FIXED (zero images,
zero state), vary only the task text, and measure how much the action chunk moves.
Identical input across conditions = perfectly controlled contrast.
  - ||a(SUCCESS) - a(FAILURE)|| ~ 0  => policy IGNORES the tag => kill the idea.
  - delta >> across-instruction delta => policy reads the tag => then buy rollouts.
No mujoco, no LIBERO assets, ~5 min.  Evidence -> /kaggle/working/kernelI6b.json
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

t0 = time.time()
WORK = "/kaggle/working" if os.path.exists("/kaggle") else "."
ADAPTER = "krishnah27/smolvla-aegis-ft-step917"
INSTR = [
    "pick up the black bowl between the plate and the ramekin and place it on the plate",
    "pick up the black bowl next to the ramekin and place it on the plate",
    "pick up the black bowl on the wooden cabinet and place it on the plate",
    "pick up the black bowl on the wooden cabinet and place it on the ramekin",
    "pick up the black bowl between the ramekin and the cup and place it on the ramekin",
]
TAGS = {"none": "{t}", "succ": "[Q:SUCCESS] {t}", "fail": "[Q:FAILURE_SLIP] {t}",
        "neutral": "[Q:NEUTRAL] {t}", "xyz": "[Q:ZZQX] {t}"}
out = {"adapter": ADAPTER, "n_instr": len(INSTR), "tags": list(TAGS), "deltas": []}


def log(m):
    print(m, flush=True)


r = subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                    "lerobot[smolvla,dataset]==0.4.4", "av", "num2words"],
                   capture_output=True, text=True)
log(f"pip rc={r.returncode}")

import torch  # noqa: E402
from huggingface_hub import snapshot_download  # noqa: E402
from lerobot.configs.policies import PreTrainedConfig  # noqa: E402
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata  # noqa: E402
from lerobot.policies.factory import get_policy_class, make_policy  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

get_policy_class("smolvla")
snap = snapshot_download(ADAPTER)
cfg = PreTrainedConfig.from_pretrained(snap)
cfg.pretrained_path = Path(snap)
ds_meta = LeRobotDatasetMetadata("lerobot/libero")
RENAME = {"observation.images.image": "observation.images.camera1",
          "observation.images.image2": "observation.images.camera2"}
pol = make_policy(cfg, ds_meta=ds_meta, rename_map=RENAME)
pol.eval()
dev = next(pol.parameters()).device
log(f"policy ok params={sum(p.numel() for p in pol.parameters())} dev={dev}")
tok = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
                                    trust_remote_code=True)


def act(task_text):
    """One select_action on a FIXED dummy observation; only the text changes.

    v50 fix (run-169-class): select_action pops an internal queue of maxlen
    n_action_steps and only recomputes the chunk when EMPTY -- so v48/v49 returned
    successive actions of the FIRST chunk and never saw the tag at all. Also,
    flow-matching samples noise per call, which made runs drift. Fix: reset the
    queue and pin the RNG before every call, so every condition recomputes its own
    chunk from the same obs with the same noise. Verified by a repeat check below.
    """
    pol.reset()
    torch.manual_seed(0)
    enc = tok([task_text], return_tensors="pt", padding=True, truncation=True,
              max_length=64)
    mask = enc["attention_mask"]
    if mask.dtype != torch.bool:
        mask = mask.bool()
    batch = {
        "observation.images.camera1": torch.zeros(1, 3, 256, 256, device=dev),
        "observation.state": torch.zeros(1, 8, device=dev),
        "observation.language.tokens": enc["input_ids"].to(dev),
        "observation.language.attention_mask": mask.to(dev),
        "task": [task_text],
    }
    with torch.no_grad():
        a = pol.select_action(batch)
    return a.float().cpu().flatten()


acts = {}
for i, t in enumerate(INSTR):
    for name, tpl in TAGS.items():
        acts[(i, name)] = act(tpl.format(t=t))
    log(f"instr{i} done")

# reproducibility check: same condition twice must be bit-close, else the deltas are noise
rep_a = act(TAGS["succ"].format(t=INSTR[0]))
rep_b = act(TAGS["succ"].format(t=INSTR[0]))
rep_gap = float((rep_a - rep_b).norm())
log(f"repeat-gap={rep_gap:.8f}")

rows = []
for i in range(len(INSTR)):
    a = {k: acts[(i, k)] for k in TAGS}
    scale = float(a["none"].norm())
    rows.append({
        "instr": i,
        "norm_action": round(scale, 4),
        "d_succ_vs_none": round(float((a["succ"] - a["none"]).norm()), 5),
        "d_fail_vs_none": round(float((a["fail"] - a["none"]).norm()), 5),
        "d_succ_vs_fail": round(float((a["succ"] - a["fail"]).norm()), 5),
        # semantic controls: same shape/length tags that carry NO quality meaning
        "d_succ_vs_neutral": round(float((a["succ"] - a["neutral"]).norm()), 5),
        "d_fail_vs_neutral": round(float((a["fail"] - a["neutral"]).norm()), 5),
        "d_succ_vs_xyz": round(float((a["succ"] - a["xyz"]).norm()), 5),
        "d_xyz_vs_neutral": round(float((a["xyz"] - a["neutral"]).norm()), 5),
        "cos_succ_fail": round(float(torch.nn.functional.cosine_similarity(
            a["succ"].unsqueeze(0), a["fail"].unsqueeze(0)).item()), 6),
    })
out["deltas"] = rows


def m(k):
    return round(sum(r[k] for r in rows) / len(rows), 6)


# scale reference: how much actions differ across DIFFERENT instructions (no tag)
cross = []
for i in range(len(INSTR)):
    for j in range(i + 1, len(INSTR)):
        cross.append(float((acts[(i, "none")] - acts[(j, "none")]).norm()))
out["summary"] = {
    "mean_norm_action": m("norm_action"),
    "mean_d_succ_fail": m("d_succ_vs_fail"),
    "mean_d_succ_none": m("d_succ_vs_none"),
    "mean_d_succ_neutral": m("d_succ_vs_neutral"),
    "mean_d_succ_xyz": m("d_succ_vs_xyz"),
    "mean_d_xyz_neutral": m("d_xyz_vs_neutral"),
    # >1 => the QUALITY WORD carries information beyond a same-shape tag;
    # ~1  => pure token-perturbation, policy ignores quality semantics.
    "semantic_ratio_succ_fail_over_succ_xyz": round(
        (sum(r["d_succ_vs_fail"] for r in rows) / len(rows))
        / max(sum(r["d_succ_vs_xyz"] for r in rows) / len(rows), 1e-9), 4),
    "semantic_ratio_succ_fail_over_succ_neutral": round(
        (sum(r["d_succ_vs_fail"] for r in rows) / len(rows))
        / max(sum(r["d_succ_vs_neutral"] for r in rows) / len(rows), 1e-9), 4),
    "mean_cross_instruction_d": round(sum(cross) / len(cross), 5),
    "ratio_tag_delta_over_cross": round(
        (sum(r["d_succ_vs_fail"] for r in rows) / len(rows))
        / max(sum(cross) / len(cross), 1e-9), 4),
    "mean_cos_succ_fail": m("cos_succ_fail"),
    "repeat_gap": round(rep_gap, 8),
}
out["wall_s"] = round(time.time() - t0, 1)
log("SUMMARY " + json.dumps(out["summary"]))
with open(os.path.join(WORK, "kernelI6b.json"), "w") as f:
    json.dump(out, f, indent=1)
log("I6B_DONE")