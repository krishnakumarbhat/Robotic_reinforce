# GPU RULES (Kaggle 45h + Colab Pro + Colab free + Local RTX) — MANDATORY for all agents

> Canonical rules live at `/media/pope/projecteo/connect/connect_gpu/FREE_TIER_GPU_RULES.md`.
> Cascade: local → kaggle (45h) → colab-pro → colab-old. `COLAB_ACCOUNT=auto` falls pro→old.

You are on FREE tier. Reckless usage burns the owner's weekly quota.
Follow this file exactly. When in doubt, use LESS GPU, not more.

## 1. Backend priority (always in this order)

1. **local** — RTX 3050 4GB (`GPU_BACKEND=local ./run_gpu.sh script.py`)
   - If the task FITS in 4GB VRAM and runs fine locally → RUN IT LOCALLY. Never rent remote GPU for a local-capable job.
   - If local is BUSY (see §4 check) → go to step 2, don't queue behind it.
2. **kaggle** — free T4/P100, 30h/week (`GPU_BACKEND=kaggle ./run_gpu.sh script.py`)
   - First remote choice. Cheaper quota, headless, auto-stops.
3. **colab** — free T4/A100, ~12h/session, compute-unit gated (`GPU_BACKEND=colab ./run_gpu.sh script.py`)
   - LAST resort. Only if local busy AND Kaggle quota/queue blocked.

## 2. Hard quota guardrails (never violate)

- **Kaggle: 30h/week resets Friday night PT. USE MAX 29h. KEEP 1h EMERGENCY RESERVE.**
  - Before any push, estimate runtime and confirm headroom. No estimate → no push.
  - One long job > many restarts (each restart re-queues and wastes time).
  - Save checkpoints every N steps to HF Hub / Drive — sessions die randomly.
- **Colab: free compute units, ~12h/session cap, idle VMs burn units.**
  - `colab new` = billable from second 1. `colab stop -s <name>` the moment work ends.
  - Never leave an IDLE session alive "for later". Later re-creates in seconds.
- **Emergency reserve is NOT yours to spend.** If only ~1h Kaggle (or Colab units nearly out) remains, STOP the loop, log state, and report to the user instead of launching.

## 3. Start → utilize → STOP protocol (every remote session)

```
colab sessions                        # / kaggle kernels list -m (pre-check)
GPU_BACKEND=kaggle ./run_gpu.sh job.py  # or colab
# ... utilize fully while alive: batch work, don't idle ...
colab stop -s <name>                  # MANDATORY when done (kaggle auto-stops)
colab sessions                        # verify: "No active sessions"
```

- Parallel sessions ARE allowed (2-3 independent jobs) to maximize throughput — but EVERY session must be stopped when its job ends.
- `colab run --gpu T4 job.py` (one-shot: new+exec+stop) is preferred over `new`+`exec` for single jobs — it self-cleans even on error.

## 4. Pre-flight (before every dispatch)

1. `nvidia-smi` — local free? Task fits 4GB? → run local, stop here.
2. Local busy + task needs >4GB → Kaggle first.
3. Kaggle quota low/blocked → Colab last.
4. After finish: stop session, verify stopped, log usage in worklog.

## 5. Forbidden (reckless) behavior

- Leaving IDLE Colab sessions running.
- Spending the last 1h Kaggle reserve.
- Renting T4/A100 for a job the RTX 3050 can do.
- Spamming `colab new` in a retry loop on 400/quota errors — back off, fall back to CPU/local.
- Touching `.env` secrets into logs/commits.

Dispatcher lives here: `connect_gpu/run_gpu.sh`. Auth: Kaggle `~/.kaggle/kaggle.json`, Colab OAuth (`colab whoami` to verify).
