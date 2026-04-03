#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import time

from two_mouse_il.input_devices import (
    DeviceAccessError,
    MultiMouseReader,
    choose_pointer_pair,
    describe_devices,
    list_candidate_pointers,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detect and optionally open two independent mouse devices.")
    parser.add_argument("--include-touchpad", action="store_true", help="Allow touchpads or trackpads to count as pointer candidates.")
    parser.add_argument("--left-device", default=None, help="Explicit event device path for the left-arm mouse, e.g. /dev/input/event8.")
    parser.add_argument("--right-device", default=None, help="Explicit event device path for the right-arm mouse, e.g. /dev/input/event12.")
    parser.add_argument("--open", action="store_true", help="Attempt to open the selected devices for live per-device input.")
    parser.add_argument("--watch-seconds", type=float, default=0.0, help="If opening devices, poll them for this many seconds and print deltas.")
    parser.add_argument("--grab", action="store_true", help="Grab the devices exclusively while watching them.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    devices = list_candidate_pointers(include_touchpad=args.include_touchpad)
    print(describe_devices(devices))
    if len(devices) < 2 and args.left_device is None and args.right_device is None:
        print("\nFewer than two pointer candidates were found. Plug in a second mouse or retry with --include-touchpad.")
        return 2

    if not args.open:
        return 0

    try:
        left_info, right_info = choose_pointer_pair(
            include_touchpad=args.include_touchpad,
            left_event_path=args.left_device,
            right_event_path=args.right_device,
        )
        reader = MultiMouseReader(left_info=left_info, right_info=right_info, grab=args.grab)
    except (RuntimeError, DeviceAccessError, PermissionError) as exc:
        print(f"\nCould not open two mouse devices: {exc}")
        print("On Linux you usually need read access to /dev/input/event*. Add your user to the input group or install a suitable udev rule.")
        return 3

    print(f"\nOpened left device:  {left_info.event_path} ({left_info.name})")
    print(f"Opened right device: {right_info.event_path} ({right_info.name})")
    if args.watch_seconds > 0:
        print(f"Watching input for {args.watch_seconds:.1f} seconds. Move each mouse to verify independent streams.")
        deadline = time.monotonic() + args.watch_seconds
        try:
            while time.monotonic() < deadline:
                frames = reader.poll()
                left_frame = frames["left"]
                right_frame = frames["right"]
                if left_frame.any_activity or right_frame.any_activity:
                    print(
                        "left=",
                        left_frame,
                        " right=",
                        right_frame,
                        sep="",
                    )
                time.sleep(0.02)
        finally:
            reader.close()
    else:
        reader.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
