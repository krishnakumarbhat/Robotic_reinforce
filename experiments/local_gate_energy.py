"""Gate-as-predictor: rerun 30 LIBERO eps logging per-step chunk energy
(mean abs 2nd-diff = jerk proxy, same family as ManiSkill gate) + outcome.
Then: does energy predict failure? (precision/recall over thresholds).
Zero quota. venv python, detached.
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
OUT = "results/gate_energy.jsonl"
SUITES = ["libero_spatial", "libero_object", "libero_goal"]
SEED = int(os.environ.get("GATE_SEED", "7000"))


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
    add_envs_task, eval_one, preprocess_observation)

get_policy_class("smolvla")
cfg = PreTrainedConfig.from_pretrained(SNAP)
cfg.pretrained_path = SNAP
ds_meta = LeRobotDatasetMetadata("lerobot/libero")
pol = make_policy(cfg, ds_meta=ds_meta, rename_map=RENAME)
pol.eval()
log("policy ok")

# wrap select_action to record chunk jerk per call
energies: list = []
_orig_sa = pol.select_action


def logged_sa(obs):
    a = _orig_sa(obs)
    ch = getattr(pol, "_last_chunk", None)
    return a


pol.select_action = logged_sa
ov = {"device_processor": {"device": str(pol.config.device)},
      "rename_observations_processor": {"rename_map": RENAME}}
preprocessor, postprocessor = make_pre_post_processors(
    policy_cfg=cfg, pretrained_path=SNAP, preprocessor_overrides=ov)

# energy = executed-action deltas; recorded via wrapper around select_action
from lerobot.scripts.lerobot_eval import eval_one  # noqa: E402

_jerks: list = []
_prev = None
_orig2 = pol.select_action


def rec_sa(obs):
    global _prev
    a = _orig2(obs)
    import numpy as _np
    import torch as _t
    an = a.detach().to("cpu").numpy()
    if _prev is not None:
        _jerks.append(float(_np.mean(abs(an - _prev))))
    _prev = an.copy()
    return a


pol.select_action = rec_sa

for si, suite in enumerate(SUITES):
    env_cfg = LiberoEnvConfig(task=suite)
    env_pre, env_post = make_env_pre_post_processors(env_cfg=env_cfg,
                                                     policy_cfg=cfg)
    for tid in range(10):
        _jerks.clear()
        _prev = None
        vec = create_libero_envs(task=suite, n_envs=1,
                                 env_cls=gym.vector.SyncVectorEnv,
                                 gym_kwargs={"task_ids": [tid],
                                             "obs_type": "pixels_agent_pos"}
                                 )[suite][tid]
        try:
            res = eval_one(vec, policy=pol,
                           env_preprocessor=env_pre,
                           env_postprocessor=env_post,
                           preprocessor=preprocessor,
                           postprocessor=postprocessor,
                           n_episodes=1, max_episodes_rendered=0,
                           videos_dir=None, return_episode_data=False,
                           start_seed=SEED + si * 100 + tid)
            import numpy as _np2
            rec = {"suite": suite, "task": tid, "seed": SEED,
                   "success": bool(res["successes"][0]),
                   "mean_jerk": round(float(_np2.mean(_jerks)), 4)
                   if _jerks else None,
                   "max_jerk": round(float(_np2.max(_jerks)), 4)
                   if _jerks else None,
                   "t": time.time()}
        except Exception as e:  # noqa: BLE001
            import traceback
            rec = {"suite": suite, "task": tid,
                   "error": f"{type(e).__name__}: {str(e)[:200]}",
                   "t": time.time()}
            log(traceback.format_exc()[-400:])
        finally:
            try:
                vec.close()
            except Exception:
                pass
        log(f"{suite}/{tid}: succ={rec.get('success')} mjerk={rec.get('mean_jerk')}")
        with open(OUT, "a") as f:
            f.write(json.dumps(rec) + "\n")
log(f"GATE_DONE wall={round(time.time()-t0,1)}")
