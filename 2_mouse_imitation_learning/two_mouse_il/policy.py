from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn


class BehaviorCloningMLP(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_dims: tuple[int, ...] = (256, 256)):
        super().__init__()
        layers: list[nn.Module] = []
        input_dim = obs_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, action_dim))
        layers.append(nn.Tanh())
        self.network = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.network(obs)


def save_checkpoint(
    checkpoint_path: str | Path,
    model: BehaviorCloningMLP,
    obs_mean: torch.Tensor,
    obs_std: torch.Tensor,
    metadata: dict,
    hidden_dims: tuple[int, ...],
) -> None:
    checkpoint_path = Path(checkpoint_path)
    checkpoint = dict(
        model_state_dict=model.state_dict(),
        obs_mean=obs_mean.cpu(),
        obs_std=obs_std.cpu(),
        metadata=metadata,
        hidden_dims=hidden_dims,
    )
    torch.save(checkpoint, checkpoint_path)


@dataclass
class LoadedCheckpoint:
    model: BehaviorCloningMLP
    obs_mean: torch.Tensor
    obs_std: torch.Tensor
    metadata: dict

    def predict(self, obs: np.ndarray) -> np.ndarray:
        self.model.eval()
        obs_tensor = torch.from_numpy(np.asarray(obs, dtype=np.float32)).reshape(1, -1)
        normalized_obs = (obs_tensor - self.obs_mean) / self.obs_std
        with torch.no_grad():
            action = self.model(normalized_obs)
        return action.cpu().numpy().reshape(-1).astype(np.float32)


def load_checkpoint(checkpoint_path: str | Path) -> LoadedCheckpoint:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    metadata = checkpoint["metadata"]
    obs_dim = int(metadata["obs_dim"])
    action_dim = int(metadata["action_dim"])
    hidden_dims = tuple(checkpoint["hidden_dims"])
    model = BehaviorCloningMLP(obs_dim=obs_dim, action_dim=action_dim, hidden_dims=hidden_dims)
    model.load_state_dict(checkpoint["model_state_dict"])
    return LoadedCheckpoint(
        model=model,
        obs_mean=checkpoint["obs_mean"],
        obs_std=checkpoint["obs_std"],
        metadata=metadata,
    )
