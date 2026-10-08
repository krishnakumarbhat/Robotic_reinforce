# Free GPU Research Stack - Agent Guide

> SHARED SOURCE OF TRUTH: `/media/pope/projecteo/connect/connect_gpu/` owns the
> account setup (`use-colab.sh`, cascade rules). This project delegates Colab
> identity to it (see `run_gpu.sh`). If this file and the shared copies disagree,
> the shared copies win. Quotas: Kaggle 45h/wk, Colab pro + old accounts.

Three free GPU providers, one terminal workflow. Any AI agent reading this file
can operate the full stack. Secrets live in each folder's `.env` (never commit,
never print values).

## Layout

```
connect/
├── AGENTS.md          <- you are here (agent instructions)
├── .env               <- root backend switcher (GPU_BACKEND=local|kaggle|colab)
├── run_gpu.sh         <- dispatcher: ./run_gpu.sh script.py [args]
├── colab/             <- Google Colab CLI  (T4/A100, ~12h sessions)
│   └── .env           <- COLAB_* vars
├── kaggle/            <- Kaggle kernels    (T4x2/P100, 30h/week)
│   ├── .env           <- KAGGLE_* vars
│   └── kernel-metadata.json
└── huggingface/       <- HF Hub CLI        (model/dataset storage + ZeroGPU Spaces)
    └── .env           <- HF_TOKEN
└── HUB.md             <- model registry: krishnah27 profile (31 models, lineages, verdicts)
```

## Provider Cheat Sheet

| Command | Purpose | Quota |
|---|---|---|
| `colab new -s research --gpu T4` / `colab exec -s research -f train.py` / `colab download` / `colab stop` | Interactive GPU VM from terminal | Free tier compute units, ~12h/session |
| `colab run --gpu A100 train.py` | Ephemeral one-shot job, auto-teardown | same |
| `kaggle kernels push -p kaggle` | Headless GPU training job | 30h/week |
| `hf download <repo>` / `hf upload <repo> <path>` | Store datasets/models/checkpoints | free |

## Auth Status & One-Time Setup

| Provider | Status check | Login command | Browser step |
|---|---|---|---|
| Colab | `colab sessions` | `colab login` | OAuth popup -> allow |
| Kaggle | `ls ~/.kaggle/kaggle.json` | none needed | Create token at kaggle.com/settings/api |
| HF | `hf auth whoami` | `hf auth login` (paste token) | Create token at hf.co/settings/tokens (Write) |

## MCP (for AI agent integration)

Colab ships an official MCP server: https://github.com/googlecolab/colab-mcp
Add to any MCP client config:

```json
{
  "mcpServers": {
    "colab": {
      "command": "npx",
      "args": ["-y", "@google/colab-mcp"]
    }
  }
}
```

## Rules for agents

1. Load env before running: `set -a; source <folder>/.env; set +a`
2. Long jobs: save checkpoints every N steps to Drive (`colab drivemount`) or upload to HF Hub — sessions die randomly.
3. Always tear down: `colab stop -s NAME` when done. Kaggle kernels stop on their own.
4. Check quota before big jobs: Kaggle 30h/wk resets Friday night PT; Colab shows units at colab.research.google.com.
5. Never echo `.env` contents into logs or commits.
