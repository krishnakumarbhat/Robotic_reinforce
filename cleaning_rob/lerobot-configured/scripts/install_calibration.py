#!/usr/bin/env python3
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Install the calibration files versioned in this repo into LeRobot's calibration cache.

LeRobot only reads calibration from ``HF_LEROBOT_CALIBRATION`` (by default
``~/.cache/huggingface/lerobot/calibration``). This script copies every
``calibration/**/*.json`` from the repo root to the same relative path there.

Existing files that differ are left untouched unless ``--force`` is given, in
which case they are backed up to ``<name>.json.bak-<timestamp>`` first (no
backup is made with ``--export``, since the repo copy is tracked by git).

With ``--export`` the direction is reversed: the machine's calibration cache is
copied into the repo's ``calibration/`` dir, so re-calibrated values can be
committed.

Usage:
    uv run python scripts/install_calibration.py            # install into cache
    uv run python scripts/install_calibration.py --dry-run  # show what would happen
    uv run python scripts/install_calibration.py --force    # overwrite (with backup)
    uv run python scripts/install_calibration.py --export   # cache -> repo
"""

import argparse
import filecmp
import shutil
import sys
import time
from pathlib import Path

from lerobot.utils.constants import HF_LEROBOT_CALIBRATION

REPO_CALIBRATION = Path(__file__).resolve().parents[1] / "calibration"


def sync(src_root: Path, dst_root: Path, force: bool, dry_run: bool, backup: bool = True) -> int:
    """Copy every ``*.json`` under ``src_root`` to ``dst_root``. Returns the number of unresolved conflicts."""
    files = sorted(p for p in src_root.rglob("*.json") if p.is_file())
    if not files:
        print(f"No calibration files found in {src_root}")
        return 0

    prefix = "[dry-run] " if dry_run else ""
    stamp = time.strftime("%Y%m%d-%H%M%S")
    conflicts = 0
    for src in files:
        rel = src.relative_to(src_root)
        dst = dst_root / rel

        if dst.exists() and filecmp.cmp(src, dst, shallow=False):
            print(f"{prefix}up to date  {rel}")
            continue

        if dst.exists():
            if not force:
                print(f"{prefix}CONFLICT    {rel} (differs from {dst}; use --force to overwrite)")
                conflicts += 1
                continue
            if backup:
                backup_path = dst.with_name(f"{dst.name}.bak-{stamp}")
                print(f"{prefix}overwrite   {rel} (backup: {backup_path.name})")
                if not dry_run:
                    shutil.copy2(dst, backup_path)
            else:
                print(f"{prefix}overwrite   {rel}")
        else:
            print(f"{prefix}install     {rel}")

        if not dry_run:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)

    return conflicts


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--export",
        action="store_true",
        help="Copy this machine's calibration cache into the repo instead of the other way around.",
    )
    parser.add_argument(
        "--force", action="store_true", help="Overwrite differing files (a timestamped backup is kept)."
    )
    parser.add_argument("--dry-run", action="store_true", help="Print what would happen without writing.")
    args = parser.parse_args()

    src, dst = (
        (HF_LEROBOT_CALIBRATION, REPO_CALIBRATION)
        if args.export
        else (REPO_CALIBRATION, HF_LEROBOT_CALIBRATION)
    )
    if not src.is_dir():
        print(f"Source directory does not exist: {src}", file=sys.stderr)
        return 1

    print(f"{src} -> {dst}")
    # The repo copy is already versioned by git, so only back up when writing into the cache.
    conflicts = sync(src, dst, force=args.force, dry_run=args.dry_run, backup=not args.export)
    if conflicts:
        print(f"{conflicts} file(s) not copied because they differ. Re-run with --force to overwrite.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
