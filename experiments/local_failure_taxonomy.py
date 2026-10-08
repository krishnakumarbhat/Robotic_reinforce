"""H1 failure taxonomy: rerun all 30 LIBERO screening eps with videos +
episode data. Saves first/mid/last frames per episode for visual inspection.
Streams JSONL. Zero quota. venv python, detached.
"""
import json
import os
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

SNAP = "checkpoints/g3_10k_local"
RENAME = {"observation.images.image": "observation.images.camera1",
          "observation.images.image2": "observation.images.camera2"}
OUT = "results/failure_taxonomy.jsonl"
VIDDIR = "results/tax_videos"
SUITES = ["libero_spatial", "libero_object", "libero_goal"]
SEED = 7000


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
from lerobot.scripts.lerobot_eval import eval_one  # noqa: E402

get_policy_class("smolvla")
cfg = PreTrainedConfig.from_pretrained(SNAP)
cfg.pretrained_path = SNAP
ds_meta = LeRobotDatasetMetadata("lerobot/libero")
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

os.makedirs(VIDDIR, exist_ok=True)
for si, suite in enumerate(SUITES):
    env_cfg = LiberoEnvConfig(task=suite)
    env_pre, env_post = make_env_pre_post_processors(env_cfg=env_cfg,
                                                     policy_cfg=cfg)
    for tid in range(10):
        vdir = os.path.join(VIDDIR, f"{suite}_{tid}")
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
                           n_episodes=1, max_episodes_rendered=1,
                           videos_dir=Path(vdir), return_episode_data=False,
                           start_seed=SEED + si * 100 + tid)
            rec = {"suite": suite, "task": tid,
                   "success": bool(res["successes"][0]),
                   "sum_reward": round(float(res["sum_rewards"][0]), 2),
                   "max_reward": round(float(res["max_rewards"][0]), 2),
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
        log(f"{suite}/{tid}: success={rec.get('success')} "
            f"maxr={rec.get('max_reward')}")
        with open(OUT, "a") as f:
            f.write(json.dumps(rec) + "\n")
log(f"TAXONOMY_DONE wall={round(time.time()-t0,1)}")
