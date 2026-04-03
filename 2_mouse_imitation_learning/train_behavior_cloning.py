#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from tqdm import tqdm

from two_mouse_il.dataset import load_behavior_cloning_arrays
from two_mouse_il.policy import BehaviorCloningMLP, save_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a simple behavior cloning model from dual-mouse demonstrations.")
    parser.add_argument("--dataset", required=True, help="Path to the recorded H5 dataset.")
    parser.add_argument("--output", required=True, help="Checkpoint output path.")
    parser.add_argument("--epochs", type=int, default=40, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=256, help="Training batch size.")
    parser.add_argument("--learning-rate", type=float, default=3e-4, help="Optimizer learning rate.")
    parser.add_argument("--hidden-dims", type=int, nargs="+", default=[256, 256], help="Hidden layer widths.")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Validation split ratio.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    obs, actions, metadata = load_behavior_cloning_arrays(args.dataset)
    if len(obs) == 0:
        raise SystemExit("Dataset is empty. Record demonstrations first.")

    obs_tensor = torch.from_numpy(obs)
    action_tensor = torch.from_numpy(actions)
    obs_mean = obs_tensor.mean(dim=0)
    obs_std = obs_tensor.std(dim=0).clamp_min(1e-6)
    normalized_obs = (obs_tensor - obs_mean) / obs_std

    dataset = TensorDataset(normalized_obs, action_tensor)
    val_size = max(1, int(len(dataset) * args.val_ratio))
    train_size = max(1, len(dataset) - val_size)
    if train_size + val_size > len(dataset):
        val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BehaviorCloningMLP(obs_dim=obs.shape[1], action_dim=actions.shape[1], hidden_dims=tuple(args.hidden_dims)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    loss_fn = nn.MSELoss()

    best_val_loss = float("inf")
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        train_count = 0
        for batch_obs, batch_actions in tqdm(train_loader, desc=f"train {epoch}/{args.epochs}", leave=False):
            batch_obs = batch_obs.to(device)
            batch_actions = batch_actions.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(batch_obs)
            loss = loss_fn(pred, batch_actions)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(batch_obs)
            train_count += len(batch_obs)

        model.eval()
        val_loss = 0.0
        val_count = 0
        with torch.no_grad():
            for batch_obs, batch_actions in val_loader:
                batch_obs = batch_obs.to(device)
                batch_actions = batch_actions.to(device)
                pred = model(batch_obs)
                loss = loss_fn(pred, batch_actions)
                val_loss += loss.item() * len(batch_obs)
                val_count += len(batch_obs)

        train_loss = train_loss / max(train_count, 1)
        val_loss = val_loss / max(val_count, 1)
        print(f"epoch={epoch:03d} train_loss={train_loss:.6f} val_loss={val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                checkpoint_path=output_path,
                model=model,
                obs_mean=obs_mean,
                obs_std=obs_std,
                metadata=metadata,
                hidden_dims=tuple(args.hidden_dims),
            )

    print(f"Saved best checkpoint to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
