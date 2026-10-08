"""Local RTX PushT eval of v15k ckpt (20 eps, streaming JSONL). Zero quota.
Same custom-loop pattern as LIBERO v3 (proven). Run with venv python.
"""
import json
import time

import gymnasium as gym
import torch

SNAP = "checkpoints/v15k_local"
RENAME = {"observation.image": "observation.images.camera1"}
OUT = "results/pusht_local.jsonl"
N_EPS = 20
SEED = 9000


def log(m):
    print(m, flush=True)


t0 = time.time()
from lerobot.configs.policies import PreTrainedConfig  # noqa: E402
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata  # noqa: E402
from lerobot.envs.configs import PushtEnv as PushtEnvConfig  # noqa: E402
from lerobot.envs.factory import make_env, make_env_pre_post_processors  # noqa: E402
from lerobot.policies.factory import (  # noqa: E402
    get_policy_class, make_policy, make_pre_post_processors)
from lerobot.scripts.lerobot_eval import eval_one  # noqa: E402

get_policy_class("smolvla")
cfg = PreTrainedConfig.from_pretrained(SNAP)
cfg.pretrained_path = SNAP
ds_meta = LeRobotDatasetMetadata("lerobot/pusht_image")
pol = make_policy(cfg, ds_meta=ds_meta, rename_map=RENAME)
pol.eval()
log(f"policy ok, vram_mb={torch.cuda.memory_allocated()/1e6:.0f}")

preprocessor_overrides = {
    "device_processor": {"device": str(pol.config.device)},
    "rename_observations_processor": {"rename_map": RENAME},
}
preprocessor, postprocessor = make_pre_post_processors(
    policy_cfg=cfg, pretrained_path=SNAP,
    preprocessor_overrides=preprocessor_overrides)
env_cfg = PushtEnvConfig()
env_pre, env_post = make_env_pre_post_processors(env_cfg=env_cfg,
                                                 policy_cfg=cfg)
vec = make_env(env_cfg, n_envs=1, use_async_envs=False)["pusht"][0]
log("env ok")
try:
    res = eval_one(vec, policy=pol,
                   env_preprocessor=env_pre, env_postprocessor=env_post,
                   preprocessor=preprocessor, postprocessor=postprocessor,
                   n_episodes=N_EPS, max_episodes_rendered=0,
                   videos_dir=None, return_episode_data=False,
                   start_seed=SEED)
    rec = {"successes": [bool(s) for s in res["successes"]],
           "pc": round(sum(1 for s in res["successes"] if s) /
                       len(res["successes"]), 3),
           "t": time.time()}
except Exception as e:  # noqa: BLE001
    import traceback
    rec = {"error": f"{type(e).__name__}: {str(e)[:200]}", "t": time.time()}
    log(traceback.format_exc()[-600:])
finally:
    try:
        vec.close()
    except Exception:
        pass
log(f"PUSHT: {rec}")
with open(OUT, "a") as f:
    f.write(json.dumps(rec) + "\n")
log(f"PUSHT_LOCAL_DONE wall={round(time.time()-t0,1)}")
