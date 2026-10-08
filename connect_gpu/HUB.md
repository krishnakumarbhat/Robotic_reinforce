# HuggingFace Hub registry — krishnah27

Profile: <https://huggingface.co/krishnah27> · 31 models, 0 datasets (Oct 2026).
All Aegis fine-tunes are LoRA-adapter repos (227 MB each): `adapter_model.safetensors`
+ full preprocessor chain (`policy_preprocessor.json` required for eval).

## Base models (warm-start sources)

| Repo | What | Use for |
|---|---|---|
| `lerobot/smolvla_base` (upstream) | SmolVLA 450M base | control arm in every eval; never train from here directly |
| `krishnah27/smolvla-libero-g3-5k` | base + 5k LIBERO steps | superseded by g3-10k |
| `krishnah27/smolvla-libero-g3-10k` | base + 10k LIBERO steps | LIBERO-domain warm start |

## Champion lineage (current): 917 → 10607

Expert FT continued on Kaggle T4 (v61, 8 h, loss ~0.55 → ~0.28, 917 → 12,119 steps).

| Repo | Verdict |
|---|---|
| `krishnah27/smolvla-aegis-ft-step917` | LIBERO 0.0 = base 0.0 on all 6 cells (buried as policy, kept as warm start) |
| `...-step1518` … `...-step10607` (7 ckpts) | v61 ladder; step10607 under eval (v64) |
| `krishnah27/smolvla-aegis-ft` | parent repo: `aegis_report.json` (loss curve, ckpt index) |
| `krishnah27/smolvla-aegis-ft-step10607` (+`model.safetensors`) | SELF-CONTAINED full ckpt (merged 2026-10-05, `merge_lora_cpu.py`): loads with no peft |

## Older lineages (superseded, kept for provenance)

- 1198-lineage: step207 … step1830 (steps 207/304/309/305/432/551/587/600/613/649/866/899/1198/1220/1497/1524/1796/1830) — T4 shakedown runs, Sep 2026.
- Read-only: never delete; paper provenance cites step917 + step10607 only.

## Known artifact defect (fixed for future pushes)

Checkpoints ≤ step10607 bake the trainer VM's local `pretrained_path`
(`/root/.cache/...`) into `config.json`; hubs newer than 0.35.3 reject it at eval
time (v62/v63). Evals pin `huggingface_hub==0.35.3`; the trainer now sanitizes
before upload (`_sanitize_cfg_for_hub`).

## Conventions for agents

- New checkpoints: one repo per step (`{OUT_REPO}-step{step}`), public, with
  `ckpt_meta.json` (loss tail, wall, vram).
- Eval kernels resolve repos by ID (`--policy.path=<repo>`), never by local
  snapshot path; keep `snapshot_download` as cache warm-up only.
- Token: `huggingface/.env` (`HF_TOKEN`, git-ignored). Bake into kernel push
  copies only; `git checkout` the push dir immediately after.
