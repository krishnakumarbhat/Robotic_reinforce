"""Trace obs keys through each preprocessor stage for one LIBERO reset."""
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
print("env_cfg obs_type:", env_cfg.obs_type, flush=True)
env_pre, env_post = make_env_pre_post_processors(env_cfg=env_cfg,
                                                 policy_cfg=cfg)
print("env_pre steps:", [type(s).__name__ for s in env_pre.steps], flush=True)
print("pre steps:", [type(s).__name__ for s in preprocessor.steps], flush=True)
vec = create_libero_envs(task="libero_spatial", n_envs=1,
                         env_cls=gym.vector.SyncVectorEnv,
                         gym_kwargs={"task_ids": [0]})["libero_spatial"][0]
obs, info = vec.reset(seed=7000)
print("raw obs keys:", sorted(obs.keys()), flush=True)
from lerobot.scripts.lerobot_eval import preprocess_observation, add_envs_task
o1 = preprocess_observation(obs)
print("after preprocess_observation:", sorted(o1.keys()), flush=True)
o1 = add_envs_task(vec, o1)
o2 = env_pre(o1)
print("after env_pre:", sorted(o2.keys()), flush=True)
o3 = preprocessor(o2)
print("after preprocessor:", sorted(o3.keys()), flush=True)
vec.close()
print("TRACE_DONE", flush=True)
