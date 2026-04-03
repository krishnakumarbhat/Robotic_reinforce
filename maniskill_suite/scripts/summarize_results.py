#!/usr/bin/env python3
import json
from pathlib import Path


def load_matrix(suite_root: Path) -> dict:
    return json.loads((suite_root / "experiment_matrix.json").read_text())


def load_metrics(combo_dir: Path) -> dict:
    metrics_path = combo_dir / "metrics.json"
    if metrics_path.exists():
        return json.loads(metrics_path.read_text())
    return {}


def as_text(value) -> str:
    if value is None:
        return "-"
    return str(value)


def main() -> None:
    suite_root = Path(__file__).resolve().parents[1]
    matrix = load_matrix(suite_root)
    lines = [
        "# Experiment Results",
        "",
        "This file is generated from the experiment matrix and any `metrics.json` files found in combo folders.",
        "",
        "| Combo | Status | Success Once | Success At End | Return | Train Steps | Wall Clock Min | Notes |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]

    for combo in matrix["combinations"]:
        combo_dir = suite_root / combo["folder"]
        metrics = load_metrics(combo_dir)
        status = metrics.get("status", combo.get("status", "planned"))
        line = (
            f"| {combo['id']} | {status} | {as_text(metrics.get('success_once'))} | "
            f"{as_text(metrics.get('success_at_end'))} | {as_text(metrics.get('return'))} | "
            f"{as_text(metrics.get('train_steps'))} | {as_text(metrics.get('wall_clock_minutes'))} | "
            f"{as_text(metrics.get('notes'))} |"
        )
        lines.append(line)

    output_path = suite_root / "reports" / "experiment_results.md"
    output_path.write_text("\n".join(lines) + "\n")
    print(output_path)


if __name__ == "__main__":
    main()