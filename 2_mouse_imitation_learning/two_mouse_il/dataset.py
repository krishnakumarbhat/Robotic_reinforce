from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np


@dataclass
class EpisodeData:
    obs: list[np.ndarray]
    actions: list[np.ndarray]
    rewards: list[float]
    successes: list[bool]
    dones: list[bool]


class DatasetWriter:
    def __init__(self, h5_path: Path, metadata_path: Path, dataset_metadata: dict):
        self.h5_path = h5_path
        self.metadata_path = metadata_path
        self.h5_path.parent.mkdir(parents=True, exist_ok=True)
        self.metadata_path.parent.mkdir(parents=True, exist_ok=True)
        self.h5 = h5py.File(self.h5_path, "w")
        self.metadata = dict(dataset_metadata)
        self.metadata["episodes"] = []

    def write_episode(self, episode_id: int, seed: int, episode: EpisodeData) -> None:
        if not episode.obs:
            return
        group = self.h5.create_group(f"episode_{episode_id:05d}")
        group.create_dataset("obs", data=np.asarray(episode.obs, dtype=np.float32), compression="gzip")
        group.create_dataset("actions", data=np.asarray(episode.actions, dtype=np.float32), compression="gzip")
        group.create_dataset("rewards", data=np.asarray(episode.rewards, dtype=np.float32), compression="gzip")
        group.create_dataset("success", data=np.asarray(episode.successes, dtype=np.bool_), compression="gzip")
        group.create_dataset("done", data=np.asarray(episode.dones, dtype=np.bool_), compression="gzip")
        episode_return = float(np.sum(episode.rewards))
        final_success = bool(episode.successes[-1])
        self.metadata["episodes"].append(
            dict(
                episode_id=episode_id,
                seed=seed,
                length=len(episode.obs),
                return_value=episode_return,
                success=final_success,
            )
        )

    def close(self) -> None:
        self.h5.flush()
        self.h5.close()
        self.metadata_path.write_text(json.dumps(self.metadata, indent=2) + "\n", encoding="utf-8")


def load_behavior_cloning_arrays(h5_path: str | Path) -> tuple[np.ndarray, np.ndarray, dict]:
    h5_path = Path(h5_path)
    metadata_path = h5_path.with_suffix(".json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    observations = []
    actions = []
    with h5py.File(h5_path, "r") as h5:
        for episode_name in sorted(h5.keys()):
            group = h5[episode_name]
            observations.append(np.asarray(group["obs"], dtype=np.float32))
            actions.append(np.asarray(group["actions"], dtype=np.float32))
    if not observations:
        return np.empty((0, 0), dtype=np.float32), np.empty((0, 0), dtype=np.float32), metadata
    return np.concatenate(observations, axis=0), np.concatenate(actions, axis=0), metadata
