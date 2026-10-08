from __future__ import annotations
import os
os.environ.setdefault("AEGIS_WORK", "/kaggle/working/aegis_ft")
os.environ.setdefault("AEGIS_HOURS", "8")
os.environ.setdefault("AEGIS_CKPT_MIN", "60")
os.environ.setdefault("AEGIS_BASE", "krishnah27/smolvla-aegis-ft-step917")
os.environ.setdefault("AEGIS_OUT_REPO", "krishnah27/smolvla-aegis-ft")
os.environ.setdefault("AEGIS_HF_TOKEN", "__HF_TOKEN__")
# B4: expert-FT continuation from step917 (same preprocessor lineage). No context
# sidecar on Kaggle -> script logs synthetic fallback honestly; the loss-descent
# + checkpoint science stands alone (context conditioning already answered by I6b).
"""Colab-exec-ready Aegis burst: SmolVLA-500M, LoRA on the VLM + FULL flow expert.

Run (zero stdin, no argv):
    source /media/pope/projecteo/connect/connect_gpu/use-colab.sh pro
    colab --config "$COLAB_CONFIG" exec -s aegis-ft \
        -f experiments/colab_aegis_ft.py --timeout 18000

AEGIS spec bindings
  context conditioning : per-episode {tool_id token, se3_offset, failure_metadata tag}
                        composed into the task string, then tokenized by lerobot's own
                        pipeline (observation.language_tokens). No new architecture.
  mixed-quality batch  : every micro-batch draws (1-FAIL_FRAC) from the SUCCESS pool and
                        FAIL_FRAC from the GATED FAILURE_SLIP pool, then per-sample
                        flow-matching loss is weighted FAIL_W (RA-BC style).
  gate                 : AEGIS_GATE=1 admits FAILURE_SLIP rows only when the Tier-4 jerk
                        gate (max_jerk<=0.618) passed; AEGIS_GATE=0 admits them all.
                        The pool-size delta IS the interception delta, both reported.
  edge budget          : vram_mb + step_latency_ms on every checkpoint + final report.

A100-40GB headroom: frozen VLM (bf16 autocast, no master-weight copy), LoRA r=16 on VLM
attention+MLP projections, flow expert full-FT in fp32 with fused AdamW, gradient
accumulation to effective batch 64. Micro-batch halves itself on OOM.

Everything is native lerobot 0.4.4 API (make_policy / make_pre_post_processors /
wrap_with_peft) so save_pretrained + processor stats stay consistent.
"""


import json
import math
import os
import random
import re
import subprocess
import sys
import time
from collections import deque

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

LEROBOT_PIN = os.environ.get("AEGIS_LEROBOT", "lerobot[smolvla,dataset]==0.4.4")
BASE_CANDIDATES = [b for b in os.environ.get(
    "AEGIS_BASE", "lerobot/smolvla_base,krishnah27/smolvla-libero-g3-5k").split(",") if b]
DATASET_CANDIDATES = [d for d in os.environ.get(
    "AEGIS_DATASET", "lerobot/libero,lerobot/libero_10_image").split(",") if d]
OUT_REPO = os.environ.get("AEGIS_OUT_REPO", "krishnah27/smolvla-aegis-ft")
WORK = os.environ.get("AEGIS_WORK", "/content/aegis_ft")
CONTEXT_JSON = os.environ.get("AEGIS_CONTEXT_JSON", "/content/aegis_context.json")
CONTEXT_REPO = os.environ.get("AEGIS_CONTEXT_REPO", "")

EFFECTIVE_BATCH = int(os.environ.get("AEGIS_EFF_BATCH", "64"))
MICRO_BATCH = int(os.environ.get("AEGIS_MICRO_BATCH", "16"))
LR_LORA = float(os.environ.get("AEGIS_LR_LORA", "1e-4"))
LR_EXPERT = float(os.environ.get("AEGIS_LR_EXPERT", "1e-5"))
WD = float(os.environ.get("AEGIS_WD", "0.01"))
WARMUP_FRAC = float(os.environ.get("AEGIS_WARMUP", "0.03"))
GRAD_CLIP = float(os.environ.get("AEGIS_CLIP", "1.0"))
LORA_R = int(os.environ.get("AEGIS_LORA_R", "16"))
LORA_ALPHA = int(os.environ.get("AEGIS_LORA_ALPHA", "32"))
FAIL_FRAC = float(os.environ.get("AEGIS_FAIL_FRAC", "0.20"))
FAIL_W = float(os.environ.get("AEGIS_FAIL_W", "1.0"))
GATE_ON = os.environ.get("AEGIS_GATE", "1") not in ("0", "false", "False")
GATE_MAX_JERK = float(os.environ.get("AEGIS_GATE_MAX_JERK", "0.618"))
TIME_BUDGET_S = float(os.environ.get("AEGIS_HOURS", "3.5")) * 3600
CKPT_EVERY_S = float(os.environ.get("AEGIS_CKPT_MIN", "30")) * 60
SCAN_FRAMES = int(os.environ.get("AEGIS_SCAN_FRAMES", "20000"))
SEED = int(os.environ.get("AEGIS_SEED", "0"))
SELFTEST_ONLY = os.environ.get("AEGIS_SELFTEST", "0") == "1"

# LoRA on the VLM trunk only; the flow expert (lm_expert) is excluded on purpose and is
# instead passed to PEFT as a modules_to_save (full-FT) module.
VLM_TARGET_RE = (r".*vlm_with_expert\.vlm\..*\."
                 r"(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)")
FULL_FT_MODULES = ["lm_expert", "state_proj", "action_in_proj", "action_out_proj",
                   "action_time_mlp_in", "action_time_mlp_out"]

T0 = time.time()
REPORT: dict = {"env": {}, "checkpoints": []}


def log(msg: str) -> None:
    print(f"[aegis {time.time() - T0:8.1f}s] {msg}", flush=True)


def env_snapshot() -> dict:
    """Purpose: record the knobs actually in force (goes into the report + model card).
    Inputs: none (reads module config). Outputs: JSON-able dict. No secrets.
    """
    return {"lerobot_pin": LEROBOT_PIN, "base_candidates": BASE_CANDIDATES,
            "dataset_candidates": DATASET_CANDIDATES, "out_repo": OUT_REPO,
            "effective_batch": EFFECTIVE_BATCH, "micro_batch_start": MICRO_BATCH,
            "lr_lora": LR_LORA, "lr_expert": LR_EXPERT, "wd": WD,
            "warmup_frac": WARMUP_FRAC, "grad_clip": GRAD_CLIP,
            "lora_r": LORA_R, "lora_alpha": LORA_ALPHA,
            "vlm_target_re": VLM_TARGET_RE, "full_ft_modules": FULL_FT_MODULES,
            "fail_frac": FAIL_FRAC, "fail_weight": FAIL_W, "gate_on": GATE_ON,
            "gate_max_jerk": GATE_MAX_JERK, "time_budget_s": TIME_BUDGET_S,
            "ckpt_every_s": CKPT_EVERY_S, "seed": SEED}


# --- Hub token (read from the VM, never printed) ------------------------------
def _read_kv(path: str, key: str | None) -> str | None:
    """Purpose: pull one value out of a .env-style file.
    Inputs: path, key (None = whole first line). Outputs: stripped value or None.
    """
    try:
        with open(path) as fh:
            for line in fh:
                if key is None:
                    return line.strip()
                if line.startswith(key + "="):
                    return line.split("=", 1)[1]
    except OSError:
        return None
    return None


def resolve_token() -> str | None:
    """Purpose: find a Hub token on the VM. Inputs: env + well-known paths.
    Outputs: token string or None (caller degrades to local-only checkpoints).
    ponytail: set AEGIS_HF_TOKEN, or `colab upload .../huggingface/.env /content/aegis.env`.
    """
    for cand in (os.environ.get("AEGIS_HF_TOKEN"),
                 _read_kv("/content/aegis.env", "HF_TOKEN"),
                 _read_kv(os.path.expanduser("~/.cache/huggingface/token"), None)):
        if cand and len(cand.strip().strip('"')) > 20:
            return cand.strip().strip('"')
    return None


# --- Aegis context conditioning ----------------------------------------------
def build_task_text(task: str, ctx: dict) -> str:
    """Purpose: carry {tool_id, se3_offset, failure_metadata} into the VLM's language
    channel by composing them onto the task string (tokenized downstream).
    Inputs: original task, per-episode context dict.
    Outputs: composite conditioning string.
    """
    off = (ctx.get("se3_offset") or [0.0] * 6)[:6]
    off_s = " ".join(f"{float(v):+.3f}" for v in off)
    fail = (ctx.get("failure_metadata") or {}).get("tag", "none")
    return (f"{task.strip()} | tool: {ctx.get('tool_id', 'none')} | "
            f"se3_offset: {off_s} | failure: {fail}")


def _unit(*parts) -> float:
    """Deterministic [0,1) draw from a string (reproducible synthetic context)."""
    return (abs(hash("|".join(str(p) for p in parts))) % 10 ** 8) / 10 ** 8


def synth_context(ep: int) -> dict:
    """Fallback per-episode context when no sidecar/columns exist.
    ponytail: hash-derived stand-in so an unattended burst still runs; results tagged
    context_source=synthetic are NOT evidence. Upload AEGIS_CONTEXT_JSON to make it real.
    """
    return {"tool_id": f"tool_{int(_unit('t', ep) * 4)}",
            "se3_offset": [round(0.25 * (_unit("o", ep, k) - 0.5) * 2, 3) for k in range(6)],
            "failure_metadata": {"tag": "none", "max_jerk": 0.0, "gate_ok": True}}


def _load_context_sidecar() -> dict | None:
    """Purpose: real Aegis context from a JSON sidecar. None if absent/unreadable.
    Accepts a single JSON object OR JSONL (header + episode rows + summary):
    episode rows (those with fixture/suite + coverage) are collected into
    {"episodes": [...]}, header/summary rows ignored.
    """
    import json as _json

    def _parse(text):
        text = text.strip()
        if not text:
            return None
        try:
            return _json.loads(text)
        except _json.JSONDecodeError:
            table = {}
            for line in text.splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    row = _json.loads(line)
                except _json.JSONDecodeError:
                    continue
                if isinstance(row, dict) and ("coverage" in row or "suite" in row
                                             or "fixture" in row):
                    v = dict(row)
                    v["failure_metadata"] = (v.get("failure_metadata")
                                             or {"tag": v.get("quality_tag", "none"),
                                                 "max_jerk": v.get("jerk", 0.0)})
                    table[len(table)] = v
            return {"episodes": table} if table else None

    if CONTEXT_REPO:
        from huggingface_hub import hf_hub_download

        p = hf_hub_download(CONTEXT_REPO, "aegis_context.json", repo_type="dataset",
                            token=resolve_token())
        with open(p) as fh:
            return _parse(fh.read())
    if os.path.exists(CONTEXT_JSON):
        with open(CONTEXT_JSON) as fh:
            return _parse(fh.read())
    return None


def load_context_table(ds) -> tuple[dict, str]:
    """Purpose: episode -> context mapping. Sources in order: JSON sidecar (hub/file),
    dataset reward/jerk columns, synthetic fallback.
    Inputs: LeRobotDataset. Outputs: ({episode_index: ctx}, source_tag).
    """
    raw = None
    src = "synthetic"
    try:
        raw = _load_context_sidecar()
        if raw:
            src = "sidecar-hub" if CONTEXT_REPO else "sidecar-file"
    except Exception as exc:  # noqa: BLE001
        log(f"context sidecar unusable ({type(exc).__name__}: {str(exc)[:120]}) -> dataset cols")
    if raw:
        table = {}
        for k, v in (raw.get("episodes", raw)).items():
            v = dict(v)
            v["failure_metadata"] = v.get("failure_metadata") or {"tag": v.get("tag", "none")}
            table[int(k)] = v
        log(f"context table: {len(table)} episodes from {src}")
        return table, src

    cols = set(ds.meta.features.keys())
    has_rew = bool(cols & {"reward", "next.reward"})
    has_jerk = bool(cols & {"max_jerk", "jerk"})
    if not (has_rew or has_jerk):
        log("context table: no reward/jerk columns -> synthetic")
        return synth_table(ds), "synthetic"
    agg: dict[int, dict] = {}
    seen = 0
    try:
        n_frames = getattr(ds.meta, "total_frames", getattr(ds.meta, "num_frames", len(ds)))
        for i in range(min(SCAN_FRAMES, n_frames)):
            f = ds[i]
            a = agg.setdefault(int(f["episode_index"]), {"rew": 0.0, "jerk": 0.0})
            a["rew"] = max(a["rew"], _num(f, "reward", "next.reward"))
            a["jerk"] = max(a["jerk"], _num(f, "max_jerk", "jerk"))
            seen += 1
    except Exception as exc:  # noqa: BLE001
        log(f"scan stopped at {seen} frames: {type(exc).__name__}: {str(exc)[:120]}")
    table = {}
    for ep, a in agg.items():
        table[ep] = {"tool_id": f"tool_{int(_unit('t', ep) * 4)}",
                     "se3_offset": [round(0.25 * (_unit("o", ep, k) - 0.5) * 2, 3)
                                    for k in range(6)],
                     "failure_metadata": {
                         "tag": "none" if a["rew"] > 0 else "FAILURE_SLIP",
                         "max_jerk": a["jerk"],
                         "gate_ok": (a["jerk"] <= GATE_MAX_JERK) if has_jerk else True},
                     "success": a["rew"] > 0}
    log(f"context table: {len(table)} episodes from dataset (reward={has_rew} "
        f"jerk={has_jerk}, scanned {seen} frames)")
    return (table, "dataset") if table else (synth_table(ds), "synthetic")


def synth_table(ds) -> dict:
    """Synthetic context for every episode in the dataset metadata."""
    n = int(getattr(ds.meta, "num_episodes", 0) or 512)
    return {ep: {**synth_context(ep),
                 "success": _unit("f", ep) <= 0.6,
                 "failure_metadata": {
                     "tag": "none" if _unit("f", ep) <= 0.6 else "FAILURE_SLIP",
                     "max_jerk": round(_unit("j", ep), 3),
                     "gate_ok": _unit("g", ep) > 0.2}} for ep in range(n)}


def _num(frame: dict, *keys) -> float:
    for k in keys:
        v = frame.get(k)
        if v is None:
            continue
        try:
            return float(v)
        except (TypeError, ValueError):
            continue
    return 0.0


def split_pools(table: dict) -> tuple[list[int], list[int], dict]:
    """Purpose: mixed-quality pools, with the Tier-4 jerk gate applied to FAILURE_SLIP.
    Inputs: context table. Outputs: (success_eps, failure_eps, stats).
    """
    succ, fail, dropped = [], [], 0
    for ep, ctx in table.items():
        fm = ctx.get("failure_metadata") or {}
        tag = fm.get("tag", "none")
        if tag == "none" or ctx.get("success") is True:
            succ.append(ep)
        elif tag == "FAILURE_SLIP":
            if GATE_ON and not fm.get("gate_ok", True):
                dropped += 1
            else:
                fail.append(ep)
        else:
            succ.append(ep)
    return succ, fail, {"success_pool": len(succ), "failure_pool": len(fail),
                        "gate_dropped_failures": dropped, "gate_on": GATE_ON,
                        "interception_delta": dropped}


def episode_frames(ds) -> dict[int, tuple[int, int]]:
    """Purpose: map episode -> [first frame, last frame) so a batch can sample a RANDOM
    frame per episode (first-frame-only would train on episode prefixes).
    Inputs: LeRobotDataset. Outputs: {episode_index: (start, end)}.
    """
    out: dict[int, tuple[int, int]] = {}
    try:
        eps = ds.meta.episodes
        for i in range(len(eps)):
            row = eps[i]
            ep = int(row["episode_index"] if "episode_index" in row else i)
            out[ep] = (int(row["dataset_from_index"]), int(row["dataset_to_index"]))
    except Exception as exc:  # noqa: BLE001
        log(f"episode ranges unavailable ({type(exc).__name__}) -> frame==episode fallback")
    return out


def draw_batch(succ, fail, mb, rng, succ_q, fail_q) -> tuple[list[int], list[bool]]:
    """Purpose: one mixed-quality micro-batch of episodes.
    Inputs: pools, micro-batch size, rng, rolling shuffled deques.
    Outputs: (episode list, per-sample is_failure flags).
    ponytail: two deques instead of a custom Sampler; upgrade if per-epoch shuffling
    semantics ever matter.
    """

    def refill(pool, q):
        if not pool:
            return
        order = list(pool)
        rng.shuffle(order)
        q.extend(order)

    n_fail = int(round(mb * FAIL_FRAC)) if fail else 0
    if not fail:
        n_fail = 0
    if not succ:
        n_fail = mb
    n_succ = mb - n_fail
    while len(succ_q) < n_succ:
        refill(succ, succ_q)
    while len(fail_q) < n_fail:
        refill(fail, fail_q)
    eps = [succ_q.popleft() for _ in range(n_succ)] + [fail_q.popleft() for _ in range(n_fail)]
    flags = [False] * n_succ + [True] * n_fail
    pairs = list(zip(eps, flags))
    rng.shuffle(pairs)
    return [e for e, _ in pairs], [f for _, f in pairs]


def build_batch(ds, ep_indices, is_failure, ctx_table, ep_ranges, device, rng,
                chunk_size=50):
    """Purpose: stack frames and inject per-episode context into the task string.
    Flow-matching suffix masks hardcode config.chunk_size, so actions MUST be
    (B, H, A) windows, not single frames (single-step actions desync att/pad
    masks: the 227/193 kill). Images/state/task come from the window's first
    frame; short episodes pad by repeating the last frame.
    Inputs: dataset, episode indices, failure flags, context table, episode frame ranges.
    Outputs: raw batch dict on `device` (lerobot preprocessor consumes it next).
    """
    import torch

    frames = []
    for ep, is_fail in zip(ep_indices, is_failure):
        span = ep_ranges.get(ep)
        if span and span[1] - span[0] >= chunk_size:
            s0 = rng.randrange(span[0], span[1] - chunk_size + 1)
        elif span:
            s0 = span[0]
        else:
            s0 = ep
        idxs = [min(s0 + h, (span[1] - 1) if span else len(ds) - 1)
                for h in range(chunk_size)]
        f = dict(ds[idxs[0]])
        # Dataloader fix: ds[i] decodes video frames; for a 50-step action window that
        # is 50 video decodes per sample (~9 s/step on A100). Actions live in the raw
        # parquet table with no video attached: read them directly.
        try:
            ds._ensure_hf_dataset_loaded()
            acts = torch.stack([torch.as_tensor(ds.hf_dataset[i]["action"]) for i in idxs])
        except Exception:
            acts = torch.stack([torch.as_tensor(ds[i]["action"]) for i in idxs])
        f["action"] = acts
        ctx = ctx_table.get(ep) or synth_context(ep)
        f["task"] = build_task_text(str(f.get("task", "")), ctx)
        f["_aegis_failure"] = bool(is_fail)
        frames.append(f)

    batch: dict = {}
    for key in frames[0]:
        if key.startswith("_aegis"):
            continue
        vals = [f[key] for f in frames]
        if isinstance(vals[0], torch.Tensor):
            try:
                batch[key] = torch.stack(vals).to(device, non_blocking=True)
            except RuntimeError:  # ragged -> leave for the preprocessor
                batch[key] = [v.to(device) for v in vals]
        elif isinstance(vals[0], (int, float, bool)):
            batch[key] = torch.as_tensor(vals, device=device)
        else:
            batch[key] = vals
    return batch


# --- policy / peft -----------------------------------------------------------
def build_policy(ds, snap, repo_id):
    """Purpose: load the base policy through lerobot's own factory so feature shapes,
    normalization and processor stats all stay consistent.
    Inputs: dataset (for ds_meta), local snapshot path, output repo id.
    Outputs: (peft-wrapped policy, base policy, policy_cfg).
    """
    from lerobot.policies.factory import make_policy
    # Import registers SmolVLAConfig as a draccus choice for the dispatch below.
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig  # noqa: F401
    from lerobot.configs.policies import PreTrainedConfig

    # Config-version skew: the Hub fork's config.json carries `pretrained_revision`
    # (written by another lerobot), which 0.4.4's strict decode rejects. `type`
    # MUST stay: the native loader pops it for subclass dispatch (stripping it
    # KeyErrors), and nested structures (input_features) must deserialize via
    # draccus (dict-surgery yields dicts that crash make_policy). So: drop ONLY
    # the offender from the file, then use lerobot's own native loader.
    cfg_json = os.path.join(snap, "config.json")
    d = json.load(open(cfg_json))
    if "pretrained_revision" in d:
        del d["pretrained_revision"]
        json.dump(d, open(cfg_json, "w"), indent=2)
        log("config skew: dropped ['pretrained_revision'] from file")
        REPORT["config_skew_dropped"] = ["pretrained_revision"]
    cfg = PreTrainedConfig.from_pretrained(snap)
    cfg.pretrained_path = snap
    cfg.push_to_hub = False
    cfg.repo_id = repo_id
    cfg.device = "cuda"
    if hasattr(cfg, "load_vlm_weights"):
        cfg.load_vlm_weights = True  # a fork shipping this False would train from scratch
    REPORT["policy_cfg"] = {k: str(getattr(cfg, k, None)) for k in
                            ("type", "vlm_model_name", "load_vlm_weights", "max_action_dim",
                             "chunk_size", "n_action_steps", "num_steps")}
    # Feature keys: trust each base's NATIVE pipeline (its config +
    # pretrained preprocessor rename as a unit). The G3-5k fork fight (re-keying
    # cfg fought its own preprocessor: batch came out camera-keyed while the
    # re-keyed policy wanted image keys) proved config surgery loses. So: try
    # native first; only on a LOUD feature-mismatch ValueError, retry once with
    # an auto-built rename_map (dataset visuals zipped into policy slots).
    from lerobot.utils.constants import OBS_IMAGES
    try:
        policy = make_policy(cfg=cfg, ds_meta=ds.meta)
        log("make_policy native OK (no rename)")
        REPORT["rename_map"] = {}
    except ValueError as exc:
        if "rename_map" not in str(exc).lower() and "mismatch" not in str(exc).lower():
            raise
        # v58 post-mortem (Kaggle T4 OOM, log truncated mid-retry): the failed native
        # attempt leaves a half-loaded model pinned by the exception traceback frames,
        # so the retry loads a SECOND copy -> 16GB OOM. Drop the traceback refs and
        # purge the CUDA cache before retrying (A100 survived this; T4 does not).
        err = str(exc)
        del exc
        import gc as _gc
        _gc.collect()
        try:
            import torch as _torch
            if _torch.cuda.is_available():
                _torch.cuda.empty_cache()
        except Exception:
            pass
        log(f"native failed ({err[:120]}); cache purged, retrying with rename_map")
        ds_visuals = sorted(k for k in ds.meta.features if k.startswith(OBS_IMAGES + "."))
        pol_visuals = sorted(k for k in cfg.input_features
                             if k.startswith(OBS_IMAGES + "."))
        rename_map = dict(zip(ds_visuals, pol_visuals))
        log(f"make_policy native failed; retrying with rename_map={rename_map}")
        REPORT["rename_map"] = rename_map
        policy = make_policy(cfg=cfg, ds_meta=ds.meta, rename_map=rename_map)
    base = policy
    NO_LORA = os.environ.get("AEGIS_NO_LORA", "0") == "1"
    if NO_LORA:
        # Expert-only full fine-tune: freeze the VLM trunk entirely, train the
        # flow expert + projections. Zero peft code in the loop (the VM-only
        # LoRA-layer recursion never reproduced locally despite identical pins;
        # freezing the backbone is also the cleaner scientific story: generalist
        # VLM frozen, sanitation/flow expertise distilled into the expert).
        import re as _re
        keep = tuple(FULL_FT_MODULES)
        n_train, n_tot = 0, 0
        for n, p in policy.named_parameters():
            # Param tree is model.vlm_with_expert.{vlm,lm_expert} + top-level
            # projs: match SUBSTRING (startswith misses the model. prefix).
            train_it = ("lm_expert" in n) or any(
                k in n for k in ("state_proj", "action_in_proj", "action_out_proj",
                                 "action_time_mlp_in", "action_time_mlp_out"))
            p.requires_grad_(train_it)
            n_tot += p.numel()
            if train_it:
                n_train += p.numel()
        log(f"no-lora expert-only FT: trainable={n_train / 1e6:.1f}M/{n_tot / 1e6:.1f}M")
        REPORT["mode"] = "expert-only-full-ft"
        if n_train == 0:
            raise RuntimeError("expert-only matched zero params")
        return policy, base, cfg
    if not hasattr(policy, "wrap_with_peft"):
        raise RuntimeError("this lerobot has no wrap_with_peft; install " + LEROBOT_PIN)
    policy = policy.wrap_with_peft(peft_cli_overrides={
        "method_type": "LORA", "target_modules": VLM_TARGET_RE,
        "full_training_modules": FULL_FT_MODULES, "r": LORA_R, "lora_alpha": LORA_ALPHA,
    })
    stats = {"trainable": sum(p.numel() for p in policy.parameters() if p.requires_grad),
             "total": sum(p.numel() for p in policy.parameters()),
             "lora_tensors": sum(1 for n, _ in policy.named_parameters() if "lora_" in n),
             "full_ft_tensors": sum(1 for n, p in policy.named_parameters()
                                    if p.requires_grad and "lora_" not in n)}
    log(f"peft: lora_tensors={stats['lora_tensors']} full_ft_tensors={stats['full_ft_tensors']} "
        f"trainable={stats['trainable'] / 1e6:.1f}M/{stats['total'] / 1e6:.1f}M")
    if stats["trainable"] == 0:
        raise RuntimeError("peft produced zero trainable params -> target regex missed")
    return policy, base, cfg


def build_preprocessor(cfg, snap, ds):
    """Purpose: lerobot's canonical preprocessor (rename -> normalize -> tokenize), so our
    composed task string becomes observation.language_tokens.
    Inputs: policy cfg, snapshot path, dataset. Outputs: callable batch -> batch.
    """
    from lerobot.policies.factory import make_pre_post_processors

    pre, _ = make_pre_post_processors(policy_cfg=cfg, pretrained_path=snap,
                                      dataset_stats=ds.meta.stats)
    return pre


# --- optimisation ------------------------------------------------------------
def build_optimizer(policy):
    """Purpose: two param groups -- LoRA adapters (high lr) vs full-FT expert (low lr).
    Inputs: peft-wrapped policy. Outputs: (optimizer, base_lrs).
    """
    import torch

    lora_p = [p for n, p in policy.named_parameters() if p.requires_grad and "lora_" in n]
    full_p = [p for n, p in policy.named_parameters()
              if p.requires_grad and "lora_" not in n]
    groups = []
    if lora_p:
        groups.append({"params": lora_p, "lr": LR_LORA, "weight_decay": WD})
    if full_p:
        groups.append({"params": full_p, "lr": LR_EXPERT, "weight_decay": WD})
    if not groups:
        raise RuntimeError("no trainable params")
    try:
        opt = torch.optim.AdamW(groups, betas=(0.9, 0.95), eps=1e-8, fused=True)
    except (RuntimeError, TypeError):
        opt = torch.optim.AdamW(groups, betas=(0.9, 0.95), eps=1e-8)
    return opt, [g["lr"] for g in opt.param_groups]


def lr_scale(step: int, total: int) -> float:
    """Warmup then cosine to 10% of peak (burst schedule)."""
    warm = max(1, int(total * WARMUP_FRAC))
    if step < warm:
        return (step + 1) / warm
    prog = min(1.0, (step - warm) / max(1, total - warm))
    return 0.1 + 0.45 * (1 + math.cos(math.pi * prog))


def weighted_loss(base, batch, w, torch):
    """Purpose: flow-matching loss with per-sample quality weights.
    SmolVLAPolicy.forward(reduction='none') returns (per_sample (B,), loss_dict).
    Falls back to the scalar mean if that kwarg is missing.
    Inputs: base policy, preprocessed batch, per-sample weights (B,), torch.
    Outputs: scalar loss tensor.
    """
    # NOTE: debug shape-hooks were removed 2026-09-27. They re-wrapped
    # vlm.embed_image on EVERY forward (wrapper-of-wrapper chain), which is what caused
    # the "VM-only RecursionError" at step ~870 -- not PEFT/LoRA. Never monkeypatch
    # per-call; hook once or use torch forward hooks.
    try:
        per_sample, _ = base(batch, reduction="none")
        wt = w.to(per_sample.dtype)
        return (per_sample * wt).sum() / wt.sum().clamp_min(1e-8)
    except TypeError:
        loss, _ = base(batch)
        return loss


# --- training ----------------------------------------------------------------
def train(policy, base, pre, ds, ctx_table, succ, fail, torch, log_every=20):
    """Purpose: the burst. Grad-accum to EFFECTIVE_BATCH, OOM-adaptive micro-batch,
    30-minute checkpoint stream to the Hub, hard wall-clock stop.
    Inputs: peft policy, base policy, preprocessor, dataset, context table, pools, torch.
    Outputs: training stats dict.
    """
    rng = random.Random(SEED)
    torch.manual_seed(SEED)
    ep_ranges = episode_frames(ds)
    opt, base_lrs = build_optimizer(policy)
    succ_q, fail_q = deque(), deque()
    mb = max(1, min(MICRO_BATCH, EFFECTIVE_BATCH))
    accum = max(1, EFFECTIVE_BATCH // mb)
    params = [p for p in policy.parameters() if p.requires_grad]
    step, oom, hist, marks = 0, 0, [], []
    mix = {"succ": 0, "fail": 0}
    peak, last_ckpt, t_end = 0, time.time(), time.time() + TIME_BUDGET_S
    use_per_sample = True
    key_remap = None
    expected_visuals = list(policy.config.image_features)

    while time.time() < t_end:
        eps, flags = draw_batch(succ, fail, mb, rng, succ_q, fail_q)
        try:
            raw = build_batch(ds, eps, flags, ctx_table, ep_ranges, "cuda", rng,
                                chunk_size=int(getattr(policy.config, "chunk_size", 50)))
            if step == 0:
                # Schema/data drift guard: lerobot/libero meta.features can name
                # keys the rows don't carry (v2/v3 migration). Self-heal once by
                # zipping raw visual-ish keys into expected slots by order.
                log(f"first-batch keys={sorted(raw.keys())}")
                log(f"expected visuals={expected_visuals}")
                REPORT["first_batch_keys"] = sorted(raw.keys())
                REPORT["expected_visuals"] = expected_visuals
                present = [k for k in expected_visuals if k in raw]
                if not present:
                    cands = sorted(k for k in raw
                                   if "image" in k.lower() or "camera" in k.lower()
                                   or "pixel" in k.lower())
                    if not cands:
                        raise RuntimeError(
                            f"no visual keys in batch (keys={sorted(raw.keys())})")
                    key_remap = dict(zip(cands, sorted(expected_visuals)))
                    log(f"key-remap engaged: {key_remap}")
                    REPORT["key_remap"] = key_remap
            if key_remap:
                for old_k, new_k in key_remap.items():
                    if old_k in raw:
                        raw[new_k] = raw.pop(old_k)
            batch = pre(raw)
            if step == 0:
                log(f"processed-batch keys={sorted(batch.keys())}")
                log(f"policy image_features={sorted(policy.config.image_features)}")
                REPORT["processed_batch_keys"] = sorted(batch.keys())
            w = torch.tensor([FAIL_W if f else 1.0 for f in flags], device="cuda")
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = weighted_loss(base, batch, w, torch) / accum
            loss.backward()
            mix["succ"] += sum(1 for f in flags if not f)
            mix["fail"] += sum(1 for f in flags if f)
        except (torch.OutOfMemoryError, RuntimeError) as exc:
            if not isinstance(exc, torch.OutOfMemoryError) and \
                    "out of memory" not in str(exc).lower():
                raise
            oom += 1
            opt.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            if mb <= 2:
                log("OOM at micro-batch 2 -> stopping burst, last checkpoint is kept")
                break
            mb = max(2, mb // 2)
            accum = max(1, EFFECTIVE_BATCH // mb)
            log(f"OOM -> micro_batch={mb} accum={accum} (eff {mb * accum})")
            continue

        if (step + 1) % accum == 0:
            total_steps = max(accum, int((t_end - time.time()) /
                                         max(1e-3, _dt(marks, 1.0) * accum)))
            sc = lr_scale(step, total_steps)
            for g, b in zip(opt.param_groups, base_lrs):
                g["lr"] = b * sc
            gn = torch.nn.utils.clip_grad_norm_(params, GRAD_CLIP)
            opt.step()
            opt.zero_grad(set_to_none=True)
            hist.append(float(loss) * accum)
            if step % log_every == 0:
                log(f"step {step} loss {hist[-1]:.4f} gnorm {float(gn):.2f} "
                    f"lr {opt.param_groups[0]['lr']:.2e} mix {flags.count(True)}/{mb} "
                    f"vram {torch.cuda.max_memory_allocated() / 2 ** 20:.0f}MB")

        step += 1
        torch.cuda.synchronize()
        peak = max(peak, torch.cuda.max_memory_allocated())
        if step % 10 == 0:
            marks.append(time.time())
            marks[:] = marks[-10:]
        if time.time() - last_ckpt >= CKPT_EVERY_S:
            last_ckpt = time.time()
            checkpoint(policy, base, step, hist, mix, peak, marks, mb, accum, cfg_note="mid")

    opt.zero_grad(set_to_none=True)
    del use_per_sample
    return {"steps": step, "micro_batch": mb, "accum": accum, "eff_batch": mb * accum,
            "oom_retries": oom, "loss_first3": [round(x, 5) for x in hist[:3]],
            "loss_tail8": [round(x, 5) for x in hist[-8:]],
            "mixed_succ_frames": mix["succ"], "mixed_fail_frames": mix["fail"],
            "peak_vram_mb": round(peak / 2 ** 20, 1),
            "step_latency_ms": _dt(marks, None),
            "wall_s": round(time.time() - T0, 1)}


def _dt(marks: list[float], default):
    """Seconds between the first and last of the last <=10 step marks (None if <2)."""
    if len(marks) < 2:
        return default
    return round((marks[-1] - marks[0]) / (len(marks) - 1) * 1000.0, 1)


# --- checkpointing -----------------------------------------------------------
def save_local(policy, base, d: str) -> None:
    """Purpose: adapter + policy config + pre/post processors on local disk (works with
    no Hub token). The processors are REQUIRED for eval (missing policy_preprocessor.json
    killed two eval bookings); always persist them alongside weights.
    Inputs: peft policy, base policy, target dir. Outputs: files written.
    """
    os.makedirs(d, exist_ok=True)
    inner = policy.get_base_model() if hasattr(policy, "get_base_model") else policy
    if hasattr(policy, "save_pretrained") and hasattr(policy, "peft_config"):
        policy.save_pretrained(d)  # PeftModel.save_pretrained -> adapter only
    else:
        inner.save_pretrained(d)
    try:
        (inner if hasattr(inner, "config") else base).config.save_pretrained(d)
    except Exception as exc:  # noqa: BLE001
        log(f"config save skipped: {type(exc).__name__}: {str(exc)[:120]}")
    try:
        from lerobot.policies.factory import make_pre_post_processors
        _cfg = getattr(inner, "config", getattr(base, "config", None))
        if _cfg is not None:
            _pre, _post = make_pre_post_processors(policy_cfg=_cfg)
            _pre.save_pretrained(d)
            _post.save_pretrained(d)
            log("processors saved alongside checkpoint")
    except Exception as exc:  # noqa: BLE001
        log(f"processor save skipped: {type(exc).__name__}: {str(exc)[:120]}")


def hub_push(folder: str, repo: str) -> dict:
    """Purpose: one-commit upload of a checkpoint folder. Raises on auth failure so the
    caller can keep going (never let the Hub kill a 3.5h burst).
    Inputs: local dir, target repo id. Outputs: {"repo", "url"}.
    """
    from huggingface_hub import HfApi

    tok = resolve_token()
    if not tok:
        raise RuntimeError("no HF token on VM (AEGIS_HF_TOKEN or upload /content/aegis.env)")
    api = HfApi(token=tok)
    api.create_repo(repo, repo_type="model", exist_ok=True, private=False)
    api.upload_folder(repo_id=repo, folder_path=folder,
                      allow_patterns=["*.safetensors", "*.json", "*.md", "*.pt"])
    return {"repo": repo, "url": f"https://huggingface.co/{repo}"}


def _sanitize_cfg_for_hub(folder: str, repo: str) -> None:
    """Purpose: never bake the trainer VM's local cache path into a pushed config.
    v63 post-mortem: cfg.pretrained_path (a /root/.cache/... snapshot path from the
    warm-start) was uploaded inside config.json; newer huggingface_hub validates it
    as a repo ID at eval time -> HFValidationError. Rewrite it to the checkpoint's
    own repo ID (self-resolving) before upload.
    """
    p = os.path.join(folder, "config.json")
    try:
        d = json.load(open(p))
        if str(d.get("pretrained_path", "")).startswith("/"):
            d["pretrained_path"] = repo
            json.dump(d, open(p, "w"), indent=2)
            log(f"config sanitized: pretrained_path -> {repo}")
    except OSError:
        pass
    # v65 post-mortem: adapter_config.json carries the same poison in
    # base_model_name_or_path (peft reads it at from_pretrained -> same death).
    # v66 post-mortem: pointing it at SELF is a loop (adapter repos have no
    # model.safetensors; the base must be the warm-start source with FULL weights,
    # e.g. step917). Rule: config.pretrained_path -> own repo; adapter base ->
    # the AEGIS_BASE warm-start repo.
    q = os.path.join(folder, "adapter_config.json")
    try:
        d = json.load(open(q))
        if str(d.get("base_model_name_or_path", "")).startswith("/"):
            d["base_model_name_or_path"] = os.environ.get(
                "AEGIS_BASE", "krishnah27/smolvla-aegis-ft-step917").split(",")[0]
            json.dump(d, open(q, "w"), indent=2)
            log(f"adapter config sanitized: base_model -> {d['base_model_name_or_path']}")
    except OSError:
        pass


def checkpoint(policy, base, step, hist, mix, peak, marks, mb, accum, cfg_note=""):
    """Purpose: crash insurance every AEGIS_CKPT_MIN: local adapter + Hub push.
    Inputs: live training state. Outputs: info dict (appended to REPORT['checkpoints']).
    """
    tag = f"{OUT_REPO}-step{step}" if cfg_note == "mid" else OUT_REPO
    d = os.path.join(WORK, f"ckpt_step{step}")
    info = {"step": step, "local": d, "note": cfg_note}
    try:
        save_local(policy, base, d)
    except Exception as exc:  # noqa: BLE001
        info["local_error"] = f"{type(exc).__name__}: {str(exc)[:160]}"
    meta = {"step": step, "loss_tail": [round(x, 5) for x in hist[-5:]],
            "micro_batch": mb, "accum": accum, "eff_batch": mb * accum,
            "vram_mb": round(peak / 2 ** 20, 1), "step_latency_ms": _dt(marks, None),
            "mixed": dict(mix), "wall_s": round(time.time() - T0, 1)}
    try:
        json.dump(meta, open(os.path.join(d, "ckpt_meta.json"), "w"), indent=1)
        _sanitize_cfg_for_hub(d, tag)
        info.update(hub_push(d, tag))
    except Exception as exc:  # noqa: BLE001
        info["hub_error"] = f"{type(exc).__name__}: {str(exc)[:160]}"
        log(f"hub push failed: {info['hub_error']} (local copy kept at {d})")
    log(f"ckpt step={step} vram={meta['vram_mb']}MB lat={meta['step_latency_ms']}ms "
        f"hub={info.get('repo', 'SKIPPED')}")
    REPORT["checkpoints"].append(info)
    return info


# --- export ------------------------------------------------------------------
def export_final(policy, base, train_stats) -> dict:
    """Purpose: final adapter + report + model card locally, then one Hub push.
    Inputs: trained policy, base policy, train stats. Outputs: paths/urls.
    """
    d = os.path.join(WORK, "export")
    out = {"local": d}
    try:
        save_local(policy, base, d)
    except Exception as exc:  # noqa: BLE001
        out["local_error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
    json.dump({"env": env_snapshot(), "report": REPORT, "train": train_stats},
              open(os.path.join(d, "aegis_report.json"), "w"), indent=1, default=str)
    open(os.path.join(d, "README.md"), "w").write(
        f"# {OUT_REPO}\n\nAEGIS SmolVLA-500M burst fine-tune (LoRA r={LORA_R} on VLM, "
        f"full flow expert).\n\n"
        f"- base: `{REPORT.get('base')}` | dataset: `{REPORT.get('dataset')}`\n"
        f"- context conditioning: tool_id + se3_offset + failure tag composed into the "
        f"task string (source: {REPORT.get('context', {}).get('source')})\n"
        f"- mixed batches: {train_stats.get('mixed_succ_frames')} success / "
        f"{train_stats.get('mixed_fail_frames')} gated-failure frames, "
        f"gate_on={GATE_ON}, interception_delta="
        f"{REPORT.get('context', {}).get('interception_delta')}\n"
        f"- eff batch {train_stats.get('eff_batch')}, peak VRAM "
        f"{train_stats.get('peak_vram_mb')}MB, step {train_stats.get('step_latency_ms')}ms\n\n"
        f"**AEGIS status: validated-candidate-predicted ONLY** -- no physical 20-seed "
        f"validation, not a champion.\n")
    try:
        out.update(hub_push(d, OUT_REPO))
    except Exception as exc:  # noqa: BLE001
        out["hub_error"] = f"{type(exc).__name__}: {str(exc)[:160]}"
        log(f"final hub push failed: {out['hub_error']}")
    return out


# --- self-test (CPU, <2s): the logic that can silently rot --------------------
def selftest() -> None:
    """Purpose: prove the Aegis-specific logic before spending Pro units.
    Checks: conditioning text, gate ON/OFF pools, mixed-batch composition,
    mb->accum budget, and loss weighting arithmetic.
    """
    ctx = {"tool_id": "tool_2", "se3_offset": [0.1, -0.2, 0.0, 0.0, 0.0, 0.05],
           "failure_metadata": {"tag": "FAILURE_SLIP"}}
    s = build_task_text("pick up the mug", ctx)
    assert "tool_2" in s and "FAILURE_SLIP" in s and "+0.100" in s, s

    tbl = {0: {"failure_metadata": {"tag": "FAILURE_SLIP", "gate_ok": True}},
           1: {"failure_metadata": {"tag": "FAILURE_SLIP", "gate_ok": False}},
           2: {"failure_metadata": {"tag": "none"}}}
    global GATE_ON
    GATE_ON = True
    _, f_on, st_on = split_pools(tbl)
    GATE_ON = False
    _, f_off, st_off = split_pools(tbl)
    assert f_on == [0] and sorted(f_off) == [0, 1], (f_on, f_off)
    assert st_on["interception_delta"] == 1 and st_off["interception_delta"] == 0

    rng = random.Random(0)
    sq, fq = deque(), deque()
    for _ in range(25):
        eps, flags = draw_batch([10, 11, 12], [20, 21], 8, rng, sq, fq)
        assert len(eps) == 8 and sum(flags) == 2, (eps, flags)
    eps, flags = draw_batch([], [20, 21], 4, rng, sq, fq)
    assert len(eps) == 4 and all(flags), (eps, flags)  # success pool empty -> degrade

    for want in (1, 2, 4, 8, 16, 32, 64, 128):
        mb = max(1, min(want, EFFECTIVE_BATCH))
        assert mb * max(1, EFFECTIVE_BATCH // mb) <= EFFECTIVE_BATCH, want

    import torch

    per_sample = torch.tensor([1.0, 3.0])
    w = torch.tensor([1.0, 2.0])
    got = float((per_sample * w).sum() / w.sum())
    assert abs(got - (1.0 + 6.0) / 3.0) < 1e-6, got
    log("SELFTEST PASS (conditioning, gate on/off, mixed batch, accum, loss weighting)")


# --- orchestration -----------------------------------------------------------
def ensure_deps() -> None:
    """Purpose: pin lerobot so the peft/preprocessor API cannot drift under us.
    Idempotent; quiet unless something is actually missing.
    """
    need = []
    try:
        import lerobot  # noqa: F401

        import peft  # noqa: F401
    except ImportError:
        # Pin torch+transformers as a PAIR: Colab image torch 2.11 removed
        # torch.ao.quantization.CUSTOM_KEY, breaking transformers import
        # (bursts v2/v3). torch==2.10.0 + transformers==4.57.6 proven green
        # on T4 probe AND local venv. 4.57.6 proven with lerobot 0.4.4 locally.
        need = [LEROBOT_PIN, "torch==2.10.0", "peft==0.21.0", "transformers==4.57.6",
                "huggingface_hub==0.35.3", "safetensors==0.8.0", "num2words"]
    # v59 post-mortem (Kaggle: torchao 0.10.0 present, peft>=0.17 dispatches LoRA
    # through torchao and raises on <0.16.0; the image has deps "present" so the
    # pin block above never runs). v60 post-mortem: UPGRADING torchao on Kaggle
    # yields an internally inconsistent install (prototype/ vs utils.py mismatch)
    # that breaks peft's dispatch import. Nothing in the stack needs torchao:
    # absent -> peft uses the classic LoRA path (the path every green burst used).
    # So REMOVE old torchao instead of upgrading it. Absent is fine; only
    # present-but-old is fatal.
    try:
        import torchao as _tc
        _ver = tuple(int(x) for x in str(_tc.__version__).split(".")[:2])
        if _ver < (0, 16):
            log(f"removing stale torchao {_tc.__version__} (peft classic path)")
            _r = subprocess.run(
                [sys.executable, "-m", "pip", "uninstall", "-y", "-q", "torchao"],
                capture_output=True, text=True, timeout=600)
            log(f"torchao uninstall rc={_r.returncode}")
            import importlib as _il
            _il.invalidate_caches()
    except ImportError:
        pass
    if not need:
        log("deps present")
        return
    log(f"installing {need}")
    r = subprocess.run([sys.executable, "-m", "pip", "install", "-q"] + need,
                       capture_output=True, text=True, timeout=2400)
    log(f"pip rc={r.returncode} {(r.stdout + r.stderr)[-300:]}")
    # NOTE: never os.execv here — under `colab exec` the cell owns the process
    # and re-exec kills it silently (burst v5). Isolation is the launcher's job:
    # run this script under a venv python (see colab_aegis_boot.py).
    if r.returncode != 0:
        raise RuntimeError("dependency install failed -- see log above")


def fetch_snapshot(repo_id: str) -> str:
    """Purpose: materialise a Hub repo locally. Inputs: repo id. Outputs: local path.
    Falls through the candidate list so a 404 fork still lets the burst run.
    """
    from huggingface_hub import snapshot_download

    tok = resolve_token()
    errs = []
    for cand in BASE_CANDIDATES:
        try:
            log(f"fetching base {cand}")
            p = snapshot_download(cand, token=tok)
            log(f"base ok: {cand}")
            REPORT["base"] = cand
            return p
        except Exception as exc:  # noqa: BLE001
            errs.append(f"{cand}: {type(exc).__name__}: {str(exc)[:120]}")
    raise RuntimeError("no base reachable: " + " | ".join(errs))


def load_dataset(torch):
    """Purpose: open the training dataset, trying each candidate repo.
    Inputs: torch (unused, keeps the signature honest for future seeding).
    Outputs: LeRobotDataset.
    """
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    errs = []
    for cand in DATASET_CANDIDATES:
        try:
            ds = LeRobotDataset(repo_id=cand)
            REPORT["dataset"] = cand
            meta = ds.meta
            n_frames = getattr(meta, "total_frames", getattr(meta, "num_frames", len(ds)))
            n_eps = getattr(meta, "total_episodes", getattr(meta, "num_episodes", "?"))
            log(f"dataset {cand} frames={n_frames} eps={n_eps}")
            return ds
        except Exception as exc:  # noqa: BLE001
            errs.append(f"{cand}: {type(exc).__name__}: {str(exc)[:120]}")
    raise RuntimeError("no dataset reachable: " + " | ".join(errs))


def main() -> None:
    """Purpose: single entry point -- zero-arg, zero-stdin, safe to re-exec.
    Writes REPORT to /content/aegis_ft_report.json and prints the teardown hint.
    """
    log("env " + json.dumps(env_snapshot()))
    selftest()
    if SELFTEST_ONLY:
        return

    state = os.path.join(WORK, "state.json")
    if os.path.exists(state):
        try:
            prev = json.load(open(state))
        except (OSError, ValueError):
            prev = {}
        if prev.get("phase") == "done":
            log("burst already completed; re-printing report, not re-running")
            print("AEGIS_FT_REPORT " + json.dumps(prev, default=str)[:4000], flush=True)
            print(STOP_HINT, flush=True)
            return
    os.makedirs(WORK, exist_ok=True)

    ensure_deps()
    import torch

    if not torch.cuda.is_available():
        log("FATAL no CUDA on this VM -- provision with --gpu A100")
        return
    torch.backends.cuda.matmul.allow_tf32 = True
    REPORT["env"].update({"torch": torch.__version__,
                          "gpu": torch.cuda.get_device_name(0),
                          "vram_gb": round(torch.cuda.get_device_properties(0).total_memory
                                           / 2 ** 30, 1)})
    log(f"gpu {REPORT['env']['gpu']} {REPORT['env']['vram_gb']}GB torch {torch.__version__}")

    snap = fetch_snapshot(REPORT.get("base", BASE_CANDIDATES[0]))
    ds = load_dataset(torch)
    policy, base, cfg = build_policy(ds, snap, OUT_REPO)
    pre = build_preprocessor(cfg, snap, ds)
    log("preprocessor ready (rename -> normalize -> tokenize)")

    ctx_table, src = load_context_table(ds)
    succ, fail, pool = split_pools(ctx_table)
    REPORT["context"] = {"source": src, "episodes": len(ctx_table), **pool}
    log(f"pools success={len(succ)} failure={len(fail)} gate_on={GATE_ON} "
        f"dropped={pool['gate_dropped_failures']}")
    if not succ and not fail:
        log("FATAL both pools empty")
        return

    policy.train()
    REPORT["train"] = train(policy, base, pre, ds, ctx_table, succ, fail, torch)
    log("train " + json.dumps(REPORT["train"]))
    REPORT["export"] = export_final(policy, base, REPORT["train"])
    REPORT["phase"] = "done"
    REPORT["status"] = "validated-candidate-predicted ONLY (physical 20-seed pending)"
    json.dump(REPORT, open(state, "w"), indent=1, default=str)
    json.dump(REPORT, open("/content/aegis_ft_report.json", "w"), indent=1, default=str)
    print("AEGIS_FT_REPORT " + json.dumps(REPORT, default=str)[:4000], flush=True)
    print(STOP_HINT, flush=True)


STOP_HINT = """
=== TEARDOWN (run LOCALLY, not in the VM) ===
colab --config "$COLAB_CONFIG" stop -s aegis-ft
colab --config "$COLAB_CONFIG" sessions          # expect: No active sessions
colab --config "$COLAB_CONFIG" download -s aegis-ft /content/aegis_ft_report.json ./results/aegis_ft_report.json
"""

main()
