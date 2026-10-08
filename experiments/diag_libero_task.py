"""Diagnose KeyError observation.state: single LIBERO task, full traceback."""
import traceback

import gymnasium as gym

from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.envs.configs import LiberoEnv as LiberoEnvConfig
from lerobot.envs.factory import make_env_pre_post_processors
from lerobot.envs.libero import create_libero_envs
from lerobot.policies.factory import (
    get_policy_class,
    make_policy,
    make_pre_post_processors,
)
from lerobot.scripts.lerobot_eval import eval_one

RENAME = {"observation.images.image": "observation.images.camera1",
          "observation.images.image2": "observation.images.camera2"}
SNAP = "checkpoints/g3_10k_local"
print("loading policy...", flush=True)
cfg = PreTrainedConfig.from_pretrained(SNAP)
cfg.pretrained_path = SNAP
ds_meta = LeRobotDatasetMetadata("lerobot/libero")
pol = make_policy(cfg, ds_meta=ds_meta, rename_map=RENAME)
pol.eval()
print("policy ok", flush=True)
preprocessor_overrides = {
    "device_processor": {"device": str(pol.config.device)},
    "rename_observations_processor": {"rename_map": RENAME},
}
preprocessor, postprocessor = make_pre_post_processors(
    policy_cfg=cfg, pretrained_path=SNAP,
    preprocessor_overrides=preprocessor_overrides)
env_cfg = LiberoEnvConfig(task="libero_spatial")
env_pre, env_post = make_env_pre_post_processors(env_cfg=env_cfg,
                                                 policy_cfg=cfg)
vec = create_libero_envs(task="libero_spatial", n_envs=1,
                         env_cls=gym.vector.SyncVectorEnv,
                         gym_kwargs={"task_ids": [0]})["libero_spatial"][0]
print("env ok, running 1 ep...", flush=True)
try:
    res = eval_one(vec, policy=pol,
                   env_preprocessor=env_pre, env_postprocessor=env_post,
                   preprocessor=preprocessor, postprocessor=postprocessor,
                   n_episodes=1, max_episodes_rendered=0,
                   videos_dir=None, return_episode_data=False,
                   start_seed=7000)
    print("RESULT:", res, flush=True)
except Exception:
    traceback.print_exc()
finally:
    try:
        vec.close()
    except Exception:
        pass
print("DIAG_DONE", flush=True)
