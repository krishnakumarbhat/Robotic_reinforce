#!/usr/bin/env python3
import json
from pathlib import Path


def main() -> None:
    suite_root = Path(__file__).resolve().parents[1]
    matrix = json.loads((suite_root / "experiment_matrix.json").read_text())

    headers = ["combo", "task", "algo", "demo_source", "budget", "folder"]
    rows = []
    for combo in matrix["combinations"]:
      rows.append(
          [
              combo["id"],
              combo["task"],
              combo["algorithm"],
              combo["demo_source"],
              combo["pilot_budget"],
              combo["folder"],
          ]
      )

    widths = [len(header) for header in headers]
    for row in rows:
        for index, value in enumerate(row):
            widths[index] = max(widths[index], len(str(value)))

    def render(values: list[str]) -> str:
        return " | ".join(str(value).ljust(widths[index]) for index, value in enumerate(values))

    print(render(headers))
    print("-+-".join("-" * width for width in widths))
    for row in rows:
        print(render(row))


if __name__ == "__main__":
    main()