"""Kernel I6a: adapter-vs-base LIBERO eval on Kaggle GPU (self-contained; no repo imports).
Compares the Aegis FT adapter (krishnah27/smolvla-aegis-ft-step917, expert-only full-FT)
against its own base (lerobot/smolvla_base): 3 suites x 10 eps = 60 rollouts.
Quality-tag probe (I6b) follows ONLY if adapter >= base (else tags test a broken policy).
Evidence -> /kaggle/working/kernelI6a.json (streamed per suite).
Proven template: G3E2 (mujoco==3.1.6 pin, rename map, N-stdin fix).
"""
import json
import os
import re
import subprocess
import sys
import time

t0 = time.time()
WORK = "/kaggle/working" if os.path.exists("/kaggle") else "."
REPOS = ["krishnah27/smolvla-aegis-ft-step10607", "lerobot/smolvla_base"]
SUITES = ["libero_object", "libero_goal"]  # v62: spatial excluded (timed out 3x); object+goal discriminate first
N_EPS = int(os.environ.get("N_EPS", "4"))  # v63: baked (kernel env isolation eats shell env); 4 cells x ~35min fits
RENAME = ("{\"observation.images.image\": \"observation.images.camera1\", "
          "\"observation.images.image2\": \"observation.images.camera2\"}")
out = {"repos": REPOS, "suites": SUITES, "results": {}}


def log(m):
    print(m, flush=True)


def pip(pkgs):
    r = subprocess.run([sys.executable, "-m", "pip", "install", "-q"] + pkgs,
                       capture_output=True, text=True)
    return r.returncode


log(f"base pip rc={pip(['lerobot[dataset]', 'av', 'num2words', 'libero'])}")
# v68: the image ships torchao 0.10.0 and peft dispatches LoRA through it ->
# ImportError. Only ADAPTER repos hit this (base has no peft path, which is why
# base always passed while the adapter died in v62-v67). Upgrading torchao
# yields an inconsistent install (v60); REMOVAL -> peft classic path (proven).
_r = subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "-q", "torchao"],
                    capture_output=True, text=True)
log(f"rm torchao rc={_r.returncode}")
# v65: hub pin REMOVED (0.35.3 broke the image's transformers: no is_offline_mode).
# No longer needed: all v61 step-repo configs sanitized server-side (local
# pretrained_path -> own repo ID, 2026-10-04), so the new hub validates cleanly.
log(f"pin mujoco316 rc={pip(['mujoco==3.1.6'])}")

from huggingface_hub import snapshot_download  # noqa: E402

for repo in REPOS:
    for suite in SUITES:
        key = f"{repo.split('/')[-1]}::{suite}"
        try:
            snap = snapshot_download(repo)
        except Exception as e:  # noqa: BLE001
            out["results"][key] = {"ckpt": False, "err": f"{type(e).__name__}"}
            continue
        # v63: pass the REPO ID, not the snapshot path. Newer huggingface_hub runs
        # validate_repo_id on policy.path and rejects local cache paths
        # (v62: HFValidationError on /root/.cache/... for both adapter cells).
        # snapshot_download stays as a cache warm-up only.
        cmd = [sys.executable, "-m", "lerobot.scripts.lerobot_eval",
               f"--policy.path={repo}", "--env.type=libero",
               f"--env.task={suite}", "--eval.batch_size=1",
               f"--eval.n_episodes={N_EPS}", f"--rename_map={RENAME}"]
        log(f"eval {key} (~1h) ...")
        try:
            p = subprocess.run(cmd, input="N\n", capture_output=True,
                               text=True, timeout=4200)
            full = p.stdout + p.stderr
            with open(os.path.join(WORK, f"eval_{key.replace('/', '_')}.txt"), "w") as f:
                f.write(full[-20000:])
            m = re.findall(r"'pc_success': ([0-9.]+)", full)
            out["results"][key] = {"rc": p.returncode,
                                   "pc_success": m[-1] if m else None}
        except subprocess.TimeoutExpired:
            out["results"][key] = {"rc": "timeout"}
        log(f"{key}: {out['results'][key]}")
        with open(os.path.join(WORK, "kernelI6a.json"), "w") as f:
            json.dump(out, f)
out["wall_s"] = round(time.time() - t0, 1)
log(f"I6A_DONE wall={out['wall_s']}")
with open(os.path.join(WORK, "kernelI6a.json"), "w") as f:
    json.dump(out, f)
