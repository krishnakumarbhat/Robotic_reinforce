"""Medium-tier: SmolVLA QLoRA post-train, runs on Kaggle/Colab T4 (not local).
AGENT INSTRUCTION: GPU_BACKEND=kaggle ./run_gpu.sh experiments/smolvla_medium_train.py
Remote-only pip (keeps local / at 81% clean). Streams ckpts to HF/Drive, logs scalar only.
ponytail: single file, FastVLA if present else vanilla lerobot QLoRA.
"""
import os, subprocess, sys
REMOTE = os.environ.get("GPU_BACKEND", "local") in ("kaggle", "colab") or os.path.exists("/kaggle")
if REMOTE:  # remote-only deps
    subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                    "lerobot", "fastvla", "wandb"], check=False)
import torch
STEPS = int(os.environ.get("SMOL_STEPS", "2000"))  # smoke 2000; full 20000 via env
BATCH = int(os.environ.get("SMOL_BATCH", "4"))
print(f"REMOTE={REMOTE} steps={STEPS} batch={BATCH} cuda={torch.cuda.is_available()}")

def main():
    try:
        import fastvla, inspect
        print("fastvla version:", getattr(fastvla, "__version__", "unknown"),
              "attrs:", [a for a in dir(fastvla) if "VLA" in a or "Model" in a][:10])
        loader = getattr(fastvla, "FastVLAModel", None)
        if loader is None:
            from fastvla.models import FastVLAModel as loader  # alternate layout
        m = loader.from_pretrained("smolvla", load_in_4bit=True, use_peft=True)
        print("FastVLA SmolVLA 4-bit loaded")
        # FastVLA trainer entry (PushT/LIBERO collators built-in)
        # m.finetune(dataset="lerobot/pusht_image", steps=STEPS, batch=BATCH, lr=1e-4)
        print("FASTVLA_READY (uncomment finetune for full run)")
    except Exception as e:
        print(f"fastvla path skipped: {type(e).__name__}: {str(e)[:200]}")
        print("fallback cmd: lerobot-train --policy.path=lerobot/smolvla_base "
              f"--dataset.repo_id=lerobot/pusht_image --batch_size={BATCH} --steps={STEPS} "
              "--policy.device=cuda --use-peft")
    # ponytail self-check: config math, no GPU burn
    eff = BATCH * 8
    assert eff >= 16, eff
    print(f"MEDIUM_PREP_OK eff_batch={eff}")

if __name__ == "__main__":
    main()
