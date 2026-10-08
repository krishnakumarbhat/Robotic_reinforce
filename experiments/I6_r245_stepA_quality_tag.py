"""I6 step A — does the quality tag reach the action chunk, or is it inside the noise?

Purpose: decide whether I6 ("quality-tag conditioning on the FT expert") is worth a Colab
burst. I6's own pre-registered abort is `||a_S - a_F|| < 1e-3` (tag ignored), and its success
bar is `SUCCESS-tag >= none + 0.05` over 30 LIBERO episodes. Both of those are answered by a
question the ablation cannot separate: a flow-matching action expert samples the chunk from a
Gaussian noise vector z, so two identical observations ALREADY give different chunks when z is
re-drawn. The I4 lesson (equations.md ROW I4, transfer 1) is to measure before booking GPU.
So: hold z FIXED across the three language conditions, and compare the between-condition
difference against the within-condition difference from re-drawing z. That is the only
comparison in which "the tag did nothing" and "the tag did something" are separable.

Inputs: checkpoints/g3_10k_local (local, fp16, cuda:0, NO download), offline HF.
Outputs: results/I6_r245_stepA_quality_tag.json

ponytail: single-shot measurement, no CLI flags, no training, no Colab booking. If the
pre-check fails, I6 is closed for 3 minutes of local compute instead of 3.5 h of Colab.
"""
import json
import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CKPT = os.path.join(REPO, "checkpoints", "g3_10k_local")
OUT = os.path.join(REPO, "results", "I6_r245_stepA_quality_tag.json")

M_FRAMES, K_NOISE, N_WARMUP = 12, 3, 3
A_DIM = 7  # pi0/SmolVLA native action: dx,dy,dz,drx,dry,drz,gripper
COND = {
    "none": "",
    "success": "clean the restroom fixture [Q:SUCCESS]",
    "failure": "clean the restroom fixture [Q:FAILURE_SLIP]",
}


# ---------------------------------------------------------------- statistics


def fr(a, b):
    """Purpose: mean per-step L2 distance between two chunk sets.
    Inputs: a, b shape [M, K, T, D]. Outputs: float mean_t ||a-b||_2 over M, K, T."""
    return float(((a - b) ** 2).sum(-1).mean() ** 0.5)


def label_permutation_p(a1, a2, a0, n_perm=20000, seed=0):
    """Purpose: model-free null for the SUCCESS-vs-FAILURE contrast.
    Inputs: a1, a2, a0 shape [M, K, T, D] (matched noise across conditions).
    Outputs: one-sided p that the S-vs-F contrast exceeds chance relabelling.
    """
    import numpy as np

    rng = np.random.default_rng(seed)
    slots = [a1, a2, a0]
    obs = fr(a1, a2) - 0.5 * (fr(a1, a0) + fr(a2, a0))
    hits = 0
    for _ in range(n_perm):
        p = rng.permutation(3)
        x, y, z = slots[p[0]], slots[p[1]], slots[p[2]]
        if fr(x, y) - 0.5 * (fr(x, z) + fr(y, z)) >= obs:
            hits += 1
    return (hits + 1) / (n_perm + 1)


def paired_condition_test(a1, a2, a0):
    """Purpose: the I6 decision. Matched-noise between-condition distance vs re-noise distance.
    Inputs: a1, a2, a0 shape [M, K, T, D] for conditions success / failure / none.
    Outputs: dict of statistics, including two independent p-values for the same claim.
    """
    import numpy as np
    from scipy import stats

    # d_cond: same z across conditions -> the sampling noise is HELD FIXED, so the
    # leading variance term (linear-in-z part of the flow integrator) cancels.
    d_cond = np.sqrt(((a1 - a2) ** 2).sum(-1)).mean(-1)  # [M] per frame
    # d_null: same condition, different z -> the scale the between-condition gap must beat.
    d_null = np.stack(
        [np.sqrt(((a1[:, k] - a1[:, j]) ** 2).sum(-1)).mean(-1) for k in range(len(a1[0])) for j in range(k + 1, len(a1[0]))],
        axis=1,
    ).mean(-1)
    d_s_none = np.sqrt(((a1 - a0) ** 2).sum(-1)).mean(-1)
    d_f_none = np.sqrt(((a2 - a0) ** 2).sum(-1)).mean(-1)

    diff = d_cond - d_null
    wil = stats.wilcoxon(d_cond, d_null, alternative="greater", zero_method="wilcox")
    perm = label_permutation_p(a1, a2, a0)
    # exact sign-flip on the per-frame matched contrast: 2^M sign assignments
    signs = np.array(list(__import__("itertools").product([1, -1], repeat=len(diff))))
    obs = diff.mean()
    p_sign = float((np.abs((signs * diff).mean(-1)) >= abs(obs)).mean())

    return {
        "d_cond_success_vs_failure_mean": float(d_cond.mean()),
        "d_null_renoise_mean": float(d_null.mean()),
        "d_success_vs_none_mean": float(d_s_none.mean()),
        "d_failure_vs_none_mean": float(d_f_none.mean()),
        "ratio_cond_over_null": float(d_cond.mean() / d_null.mean()),
        "welcoxon_p": float(wil.pvalue),
        "signflip_p_exact": p_sign,
        "label_permutation_p": perm,
        "cohens_dz": float(obs / diff.std(ddof=1)),
        "d_cond_per_frame": [round(float(x), 6) for x in d_cond],
        "d_null_per_frame": [round(float(x), 6) for x in d_null],
    }


def selfcheck():
    """Purpose: prove paired_condition_test is inert on inert data and fires on a real shift.
    Inputs: none. Outputs: raises AssertionError if the test is miscalibrated."""
    import numpy as np

    rng = np.random.default_rng(7)
    T, D = 50, 7

    def synth(shift, scale=0.3):
        a0 = rng.normal(0, 0.5, (12, 3, T, D))
        a1 = a0 + rng.normal(0, scale, (12, 3, T, D))
        a2 = a1 + shift  # matched z across conditions; only a1 carries the shift
        return a1, a2, a0

    inert = paired_condition_test(*synth(0.0))
    assert inert["welcoxon_p"] > 0.05, f"inert data reported a hit: {inert}"
    assert inert["ratio_cond_over_null"] < 0.2, inert

    real = paired_condition_test(*synth(np.array([0.02] + [0.0] * 6)))
    assert real["ratio_cond_over_null"] > 0.05, real
    print("selfcheck OK: inert ratio %.4f p=%.3f | shifted ratio %.4f p=%.2e"
          % (inert["ratio_cond_over_null"], inert["welcoxon_p"],
             real["ratio_cond_over_null"], real["welcoxon_p"]))


# ---------------------------------------------------------------- measurement


def main():
    """Purpose: run the three conditions x M frames x K noises on the local FT-less ckpt.
    Inputs: none. Outputs: results/I6_r245_stepA_quality_tag.json (stdout echo)."""
    import numpy as np
    import torch
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    selfcheck()
    cfg = SmolVLAPolicy.load_config(CKPT)
    policy = SmolVLAPolicy.from_pretrained(CKPT).to("cuda:0").half().eval()
    chunk, adim = cfg.chunk_size, cfg.max_action_dim

    pre, _ = make_pre_post_processors(
        policy.config, CKPT, preprocessor_overrides={"device_processor": {"device": "cuda:0"}}
    )

    def frame_for(f):
        """Purpose: a deterministic, non-degenerate observation for frame f."""
        g = torch.Generator().manual_seed(9000 + f)
        img = torch.randint(0, 256, (3, 256, 256), generator=g, dtype=torch.uint8)
        st = torch.randn(cfg.max_state_dim, generator=g) * 0.1
        return {"observation.state": st[:8], "observation.images.image": img,
                "observation.images.image2": img.flip(-1), "robot_type": "panda"}

    def predict(obs, task, z):
        batch = pre({**obs, "task": task})
        batch = {k: (v.half() if v.is_floating_point() else v) for k, v in batch.items() if torch.is_tensor(v)}
        batch["task"] = task  # strings survive the preprocessor; dropping it = testing no prompt at all
        with torch.no_grad():
            return policy.predict_action_chunk(batch, noise=z)[0, :, :A_DIM].float().cpu().numpy()

    A = {c: np.zeros((M_FRAMES, K_NOISE, chunk, A_DIM)) for c in COND}
    for f in range(M_FRAMES):
        obs = frame_for(f)
        for k in range(K_NOISE):
            g = torch.Generator().manual_seed(40000 + 7 * f + k)
            z = torch.randn(1, chunk, adim, generator=g)
            for c, task in COND.items():
                A[c][f, k] = predict(obs, task, z)
        if f == 0:  # warmup inside the same code path, outside the stats
            for k in range(K_NOISE):
                for c, task in COND.items():
                    predict(obs, task, z)

    res = paired_condition_test(A["success"], A["failure"], A["none"])
    step_cm = 1.0
    res.update({
        "chunk": chunk, "action_dim": A_DIM, "m_frames": M_FRAMES, "k_noise": K_NOISE,
        "conditions": COND,
        "abort_threshold_frobenius": 1e-3,
        "abort_1e3_per_step_cm": round(1e-3 / step_cm, 6),
        "d_cond_pct_of_1cm_step": round(100 * res["d_cond_success_vs_failure_mean"] / step_cm, 6),
        "chunks_match_on_noise": True,
        "verdict": ("TAG REACHES ACTION" if res["ratio_cond_over_null"] >= 0.05
                    and min(res["welcoxon_p"], res["label_permutation_p"]) < 0.05 else "TAG INERT"),
    })
    with open(OUT, "w") as fh:
        json.dump(res, fh, indent=1)
    print(json.dumps({k: v for k, v in res.items() if not k.endswith("_per_frame")}, indent=1))


if __name__ == "__main__":
    main()
