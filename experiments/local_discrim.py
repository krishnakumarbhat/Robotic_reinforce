"""Discriminator: eval_one path, spatial tasks 0-2, stream results."""
import json
import time

import gymnasium as gym
import torch

SNAP = "checkpoints/g3_10k_local"
RENAME = {"observation.images.image": "observation.images.camera1",
          "observation.images.image2": "observation.images.camera2"}
OUT = "results/discrim.jsonl"


def log(m):
    print(m, flush=True)


from lerobot.configs.policies import PreTrainedConfig  # noqa: E402
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata  # noqa: E402
from lerobot.envs.configs import LiberoEnv as LiberoEnvConfig  # noqa: E402
from lerobot.envs.factory import make_env_pre_post_processors  # noqa: E402
from lerobot.envs.libero import create_libero_envs  # noqa: E402
from lerobot.policies.factory import (  # noqa: E402
    get_policy_class, make_policy, make_pre_post_processors)
from lerobot.scripts.lerobot_eval import eval_one  # noqa: E402

get_policy_class("smolvla")
cfg = PreTrainedConfig.from_pretrained(SNAP)
cfg.pretrained_path = SNAP
ds_meta = LeRobotDatasetMetadata("lerobot/libero")
pol = make_policy(cfg, ds_meta=ds_meta, rename_map=RENAME)
pol.eval()
ov = {"device_processor": {"device": str(pol.config.device)},
      "rename_observations_processor": {"rename_map": RENAME}}
pre, post = make_pre_post_processors(policy_cfg=cfg, pretrained_path=SNAP,
                                     preprocessor_overrides=ov)
env_cfg = LiberoEnvConfig(task="libero_spatial")
env_pre, env_post = make_env_pre_post_processors(env_cfg=env_cfg,
                                                 policy_cfg=cfg)
for tid in [0, 1, 2]:
    vec = create_libero_envs(task="libero_spatial", n_envs=1,
                             env_cls=gym.vector.SyncVectorEnv,
                             gym_kwargs={"task_ids": [tid],
                                         "obs_type": "pixels_agent_pos"}
                             )["libero_spatial"][tid]
    try:
        res = eval_one(vec, policy=pol,
                       env_preprocessor=env_pre, env_postprocessor=env_post,
                       preprocessor=pre, postprocessor=post,
                       n_episodes=1, max_episodes_rendered=0,
                       videos_dir=None, return_episode_data=False,
                       start_seed=7000 + tid)
        rec = {"task": tid, "success": bool(res["successes"][0]),
               "maxr": round(float(res["max_rewards"][0]), 2)}
    except Exception as e:  # noqa: BLE001
        rec = {"task": tid, "error": f"{type(e).__name__}: {str(e)[:150]}"}
    finally:
        try:
            vec.close()
        except Exception:
            pass
    log(f"task {tid}: {rec}")
    with open(OUT, "a") as f:
        f.write(json.dumps(rec) + "\n")
log("DISCRIM_DONE")
