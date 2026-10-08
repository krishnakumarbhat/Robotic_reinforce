"""I6 — quality-tag conditioning on the edge-scale FT expert: is the tag READ at all?

Why this runs: I6's own spec is "Evaluate the FT expert with task prefix [Q:SUCCESS] vs
[Q:FAILURE_SLIP] vs none" and its ABORT rule is "||a_S - a_F|| < 1e-3 (tag ignored)". The
success bar (LIBERO 30 eps) needs `LeRobotDatasetMetadata("lerobot/libero")`, a download the
segment forbids, so the decider that CAN run is the abort rule, and it is the load-bearing one:
if the tag does not move the action chunk then the whole pi0.7 diverse-context-conditioning
premise (failure-metadata / quality-token conditioning, 4-tier stack T1 -> T3) is unsupported at
500M on this checkpoint and no rollouts are worth booking.

The comparison is exactly paired: for observation i the SAME flow noise tensor is fed to every
arm and the SAME images are used, so any difference between arms is the language string alone.
`noise=` is passed explicitly instead of seeding, which removes sampler stochasticity from the
contrast entirely (a seeded control asserts bit-identity).

Inputs : checkpoints/g3_10k_local (local fp16, cuda:0), offline HF, synthetic paired images.
Outputs: results/I6_r245_quality_tag.json

ponytail: images are synthetic; the design is paired, so realism moves absolute magnitudes but
not the contrast, and d_obs (action spread across observations) is reported as the scale.
"""
import json
import os
import statistics
import time

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CKPT = os.path.join(REPO, "checkpoints", "g3_10k_local")
OUT = os.path.join(REPO, "results", "I6_r245_quality_tag.json")
N_OBS, N_WARMUP = 32, 3
BASE = "clean the restroom fixture"

# arms: name -> task string. Only the string differs; obs and noise are shared.
ARMS = {
    "none": BASE,
    "Q_SUCCESS": BASE + " [Q:SUCCESS]",
    "Q_PASS": BASE + " [Q:PASS]",
    "Q_FAILURE_SLIP": BASE + " [Q:FAILURE_SLIP]",
    "Q_SLIP": BASE + " [Q:SLIP]",
    "Q_SUCCESS_PRE": "[Q:SUCCESS] " + BASE,
    "IRRELEVANT": "fold a shirt",
}

import numpy as np
import torch
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAConfig, SmolVLAPolicy


def load_policy() -> tuple:
    """Purpose: load the local FT expert fp16 on cuda with its shipped preprocessors.
    Inputs: none (offline). Outputs: (policy, preprocessor)."""
    pol = SmolVLAPolicy.from_pretrained(CKPT, local_files_only=True)
    pol = pol.half().to("cuda:0").eval()
    pre, _ = make_pre_post_processors(
        pol.config, CKPT, preprocessor_overrides={"device_processor": {"device": "cuda:0"}}
    )
    return pol, pre


_STATS = None


def synth_obs(i: int) -> dict:
    """Purpose: one in-distribution pseudo-observation (state drawn in the ckpt's own normalised
    coordinates, two structured 256x256 renders). Inputs: int seed. Outputs: raw frame dict."""
    global _STATS
    if _STATS is None:
        from safetensors.torch import load_file

        d = load_file(os.path.join(CKPT, "policy_preprocessor_step_5_normalizer_processor.safetensors"))
        _STATS = (d["observation.state.mean"].float(), d["observation.state.std"].float())
    m, s = _STATS
    rng = np.random.default_rng(1000 + i)
    u = torch.from_numpy(rng.uniform(-1.0, 1.0, size=tuple(m.shape)).astype("float32"))
    state = m + s * u * (0.55 + 0.45 * ((i % 8) / 7.0))

    def render(seed: int) -> torch.Tensor:
        r = np.random.default_rng(seed)
        yy, xx = np.mgrid[0:256, 0:256].astype("float32") / 255.0
        out = np.zeros((3, 256, 256), dtype="float32")
        for c in range(3):
            out[c] = 0.35 + 0.30 * np.sin(6.0 * (xx + 0.13 * i) + c) * np.cos(4.0 * (yy - 0.07 * i))
        out[:, 60:180, 40:210] += 0.25  # a hard-edged "fixture face"
        out += 0.05 * r.standard_normal(out.shape)
        return torch.from_numpy(np.clip(out * 255, 0, 255).astype("uint8"))

    return {
        "observation.state": state,
        "observation.images.image": render(2000 + i),
        "observation.images.image2": render(3000 + i),
        "task": BASE,
        "robot_type": "panda",
    }


def batch_for(pre, frame: dict, task: str) -> dict:
    """Purpose: run the shipped preprocessor (tokenisation + normalisation) on one frame.
    Inputs: preprocessor, frame dict, task string. Outputs: fp16 cuda batch dict."""
    f = dict(frame)
    f["task"] = task
    b = pre(f)
    b = {k: v for k, v in b.items() if torch.is_tensor(v)}
    for k, v in b.items():
        if v.dtype == torch.uint8:
            b[k] = (v.float() / 255.0).half().to("cuda:0")
        elif torch.is_floating_point(v):
            b[k] = v.half().to("cuda:0")
    return b


def chunk(pol, b: dict, noise: torch.Tensor) -> torch.Tensor:
    """Purpose: one deterministic action chunk in NORMALISED action space.
    Inputs: policy, preprocessed batch, explicit flow noise. Outputs: (chunk, action_dim) tensor."""
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
        a = pol.predict_action_chunk(b, noise=noise)
    return a.float().cpu()


def stats(xs: list) -> dict:
    """Purpose: mean/sd/bootstrap-CI of a paired per-observation distance list.
    Inputs: list[float]. Outputs: dict."""
    rng = np.random.default_rng(7)
    a = np.asarray(xs)
    boot = [float(a[rng.integers(0, len(a), len(a))].mean()) for _ in range(4000)]
    return {
        "mean": round(float(a.mean()), 6),
        "sd": round(float(a.std(ddof=1)), 6),
        "ci95": [round(float(np.percentile(boot, 2.5)), 6), round(float(np.percentile(boot, 97.5)), 6)],
        "min": round(float(a.min()), 6),
        "max": round(float(a.max()), 6),
    }


def mdist(A: torch.Tensor, B: torch.Tensor) -> float:
    """Purpose: per-timestep mean L2 between two (T, A) normalised action chunks.
    Inputs: two chunk tensors. Outputs: float."""
    return float(torch.linalg.norm(A - B, dim=1).mean())



def main() -> int:
    """Purpose: run the paired arm sweep and write the JSON verdict.
    Inputs: none. Outputs: 0 on success."""
    pol, pre = load_policy()
    names = list(ARMS)
    chunks: dict = {n: [] for n in names}
    lat: dict = {n: [] for n in names}
    ntok: dict = {}
    g = torch.Generator(device="cuda:0").manual_seed(4242)
    noise_proto = torch.randn(1, pol.config.chunk_size, pol.config.max_action_dim,
                              generator=g, device="cuda:0", dtype=torch.float16)

    # warmup + determinism control (same batch + same noise must be bit-identical)
    b0 = batch_for(pre, synth_obs(0), BASE)
    for _ in range(N_WARMUP):
        chunk(pol, b0, noise_proto)
    a1, a2 = chunk(pol, b0, noise_proto), chunk(pol, b0, noise_proto)
    assert torch.equal(a1, a2), "noise= is not deterministic; the paired contrast is invalid"

    for i in range(N_OBS):
        frame = synth_obs(i)
        noise = noise_proto  # SAME noise for every arm at this observation
        for n in names:
            b = batch_for(pre, frame, ARMS[n])
            if i == 0:
                ntok[n] = {
                    "padded": int(b["observation.language.tokens"].shape[1]),
                    "real": int(b["observation.language.attention_mask"].sum()),
                }
            torch.cuda.synchronize()
            t = time.perf_counter()
            chunks[n].append(chunk(pol, b, noise))
            torch.cuda.synchronize()
            lat[n].append((time.perf_counter() - t) * 1000.0)

    pairs = [
        ("Q_SUCCESS", "Q_FAILURE_SLIP"),   # PRIMARY, pre-registered abort arm (threshold 1e-3)
        ("Q_SUCCESS", "Q_PASS"),            # synonym null (same meaning, different surface)
        ("Q_FAILURE_SLIP", "Q_SLIP"),      # synonym null
        ("none", "Q_SUCCESS"),              # tag vs no tag
        ("none", "Q_FAILURE_SLIP"),
        ("none", "IRRELEVANT"),             # positive control: is conditioning alive at all
        ("Q_SUCCESS", "Q_SUCCESS_PRE"),     # position control (prefix vs suffix)
    ]
    pd = {}
    for a, b in pairs:
        d = [mdist(chunks[a][i], chunks[b][i]) for i in range(N_OBS)]
        key = f"{a}__{b}"
        pd[key] = stats(d)
        # Wilcoxon signed-rank on the paired list vs a zero-difference null (tag ignored)
        try:
            from scipy.stats import wilcoxon

            pd[key]["wilcoxon_p"] = float(wilcoxon(d, alternative="greater").pvalue)
        except Exception as exc:  # noqa: BLE001 -- scipy absent is not fatal, log it
            pd[key]["wilcoxon_p"] = f"unavailable: {exc}"

    # scale of legitimate action variation: different observation, same arm
    d_obs = [mdist(chunks["none"][i], chunks["none"][j])
             for i in range(N_OBS) for j in range(i + 1, N_OBS)]
    d_noise = [mdist(chunks["none"][i], chunks["none"][i]) for i in range(N_OBS)]

    # directionality: is a_F - a_S a consistent direction across observations, or noise?
    delta = [(chunks["Q_FAILURE_SLIP"][i] - chunks["Q_SUCCESS"][i]).flatten() for i in range(N_OBS)]
    delta = [x / torch.linalg.norm(x).clamp_min(1e-12) for x in delta]
    cos = [float(delta[i] @ delta[j]) for i in range(N_OBS) for j in range(i + 1, N_OBS)]
    rng = np.random.default_rng(11)
    cos_null = [float(delta[int(i)] @ delta[int(j)])
                for i, j in zip(rng.integers(0, N_OBS, len(cos)), rng.integers(0, N_OBS, len(cos)))]

    primary = pd["Q_SUCCESS__Q_FAILURE_SLIP"]
    abort = primary["mean"] < 1e-3
    verdict = {
        "idea": "I6",
        "run": 245,
        "metric_axis": "quality-tag action sensitivity (normalised action space, paired, same noise)",
        "n_obs": N_OBS,
        "arms": ARMS,
        "prefix_tokens_per_arm": ntok,
        "lang_token_budget": int(pol.config.tokenizer_max_length),
        "chunk_ms_median": {n: round(statistics.median(lat[n]), 2) for n in names},
        "determinism_bit_identical": True,
        "pairwise": pd,
        "scale_d_obs_mean": round(statistics.fmean(d_obs), 6),
        "scale_d_same_obs": max(d_noise),
        "frac_of_d_obs_primary": round(primary["mean"] / statistics.fmean(d_obs), 4),
        "delta_direction_mean_cosine": round(statistics.fmean(cos), 4),
        "delta_direction_null_mean_cosine": round(statistics.fmean(cos_null), 4),
        "cos_wilcoxon_p": None,
        "pre_registered_abort_threshold": 1e-3,
        "pre_registered_abort_triggered": bool(abort),
        "verdict": "TAG IGNORED — pre-registered abort triggered" if abort else "tag moves the chunk",
    }
    try:
        from scipy.stats import wilcoxon

        verdict["cos_wilcoxon_p"] = float(wilcoxon(cos, cos_null, alternative="greater").pvalue)
    except Exception as exc:  # noqa: BLE001
        verdict["cos_wilcoxon_p"] = f"unavailable: {exc}"

    with open(OUT, "w") as fh:
        json.dump(verdict, fh, indent=1)
    print(json.dumps({k: v for k, v in verdict.items() if k != "arms"}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
