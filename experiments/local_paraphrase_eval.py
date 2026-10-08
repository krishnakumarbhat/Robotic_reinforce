"""Local RTX: instruction-robustness (LIBERO-PRO language dimension).
Variants: orig / v1 synonyms / v2 messy-tokens. Manual loop with task override.
30 eps/variant. Streams JSONL. Zero quota. venv python, detached.
"""
import json
import os
import time

import gymnasium as gym
import numpy as np
import torch

SNAP = "checkpoints/g3_10k_local"
RENAME = {"observation.images.image": "observation.images.camera1",
          "observation.images.image2": "observation.images.camera2"}
VARIANT = os.environ.get("VARIANT", "v1")
OUT = f"results/paraphrase_{VARIANT}.jsonl"
SUITES = ["libero_spatial", "libero_object", "libero_goal"]
SEED = 7000


def v1_syn(s):
    r = [("place it on", "set it on"), ("place it in", "set it into"),
         ("pick up", "grab"), ("put ", "place "), ("open the", "pull open the"),
         ("turn on", "switch on"), ("push the", "slide the")]
    for a, b in r:
        s = s.replace(a, b)
    return s


def v2_messy(s):
    s = s.lower()
    for w in [" the ", " a ", " an "]:
        s = s.replace(w, " ")
    return " ".join(s.split())


def paraphrase(s):
    return {"orig": s, "v1": v1_syn(s), "v2": v2_messy(s)}[VARIANT]


def log(m):
    print(m, flush=True)


t0 = time.time()
from lerobot.configs.policies import PreTrainedConfig  # noqa: E402
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata  # noqa: E402
from lerobot.envs.configs import LiberoEnv as LiberoEnvConfig  # noqa: E402
from lerobot.envs.factory import make_env_pre_post_processors  # noqa: E402
from lerobot.envs.libero import create_libero_envs  # noqa: E402
from lerobot.policies.factory import (  # noqa: E402
    get_policy_class, make_policy, make_pre_post_processors)
from lerobot.scripts.lerobot_eval import (  # noqa: E402
    add_envs_task, preprocess_observation)
from lerobot.utils.constants import ACTION  # noqa: E402

get_policy_class("smolvla")
cfg = PreTrainedConfig.from_pretrained(SNAP)
cfg.pretrained_path = SNAP
ds_meta = LeRobotDatasetMetadata("lerobot/libero")
pol = make_policy(cfg, ds_meta=ds_meta, rename_map=RENAME)
pol.eval()
log("policy ok")
ov = {"device_processor": {"device": str(pol.config.device)},
      "rename_observations_processor": {"rename_map": RENAME}}
preprocessor, postprocessor = make_pre_post_processors(
    policy_cfg=cfg, pretrained_path=SNAP, preprocessor_overrides=ov)
env_cfg = LiberoEnvConfig(task="libero_spatial")
env_pre, env_post = make_env_pre_post_processors(env_cfg=env_cfg,
                                                 policy_cfg=cfg)

# show paraphrases once
from libero.libero import benchmark as _bm
_s0 = _bm.get_benchmark_dict()["libero_spatial"]()
log(f"ex orig: {_s0.get_task(0).language}")
log(f"ex {VARIANT}: {paraphrase(_s0.get_task(0).language)}")

for si, suite in enumerate(SUITES):
    for tid in range(10):
        vec = create_libero_envs(task=suite, n_envs=1,
                                 env_cls=gym.vector.SyncVectorEnv,
                                 gym_kwargs={"task_ids": [tid],
                                             "obs_type": "pixels_agent_pos"}
                                 )[suite][tid]
        try:
            pol.reset()  # clear action queue (else systematic zeros!)
            obs, info = vec.reset(seed=[SEED + si * 100 + tid])
            max_steps = vec.call("_max_episode_steps")[0]
            done, step, succ = np.array([False]), 0, False
            with torch.inference_mode():
                while not bool(done[0]) and step < max_steps:
                    o = preprocess_observation(obs)
                    o = add_envs_task(vec, o)
                    o["task"] = [paraphrase(o["task"][0])]
                    o = env_pre(o)
                    o = preprocessor(o)
                    a = pol.select_action(o)
                    a = postprocessor(a)
                    a = env_post({ACTION: a})[ACTION]
                    a_np = a.to("cpu").numpy()
                    obs, reward, term, trunc, info = vec.step(a_np)
                    if "final_info" in info and isinstance(info["final_info"], dict):
                        f = info["final_info"].get("is_success", [False])
                        try:
                            if bool(np.asarray(f).reshape(-1)[0]):
                                succ = True
                        except Exception:
                            pass
                    done = np.array([bool(term[0] or trunc[0]) or succ])
                    step += 1
            rec = {"variant": VARIANT, "suite": suite, "task": tid,
                   "success": succ, "steps": step, "t": time.time()}
        except Exception as e:  # noqa: BLE001
            import traceback
            rec = {"variant": VARIANT, "suite": suite, "task": tid,
                   "error": f"{type(e).__name__}: {str(e)[:200]}",
                   "t": time.time()}
            log(traceback.format_exc()[-400:])
        finally:
            try:
                vec.close()
            except Exception:
                pass
        log(f"{suite}/{tid}: succ={rec.get('success')}")
        with open(OUT, "a") as f:
            f.write(json.dumps(rec) + "\n")
log(f"PARAPHRASE_{VARIANT}_DONE wall={round(time.time()-t0,1)}")
