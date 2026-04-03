from __future__ import annotations

import glob
import os
from dataclasses import dataclass
from pathlib import Path

from evdev import InputDevice, ecodes


TOUCHPAD_HINTS = ("touchpad", "trackpad")


class DeviceAccessError(RuntimeError):
    pass


@dataclass(frozen=True)
class PointerDeviceInfo:
    event_path: str
    name: str
    by_id_path: str | None
    by_path_path: str | None
    accessible: bool
    is_touchpad: bool


@dataclass
class MouseFrame:
    dx: int = 0
    dy: int = 0
    wheel: int = 0
    left_click: bool = False
    right_click: bool = False
    middle_click: bool = False

    @property
    def any_activity(self) -> bool:
        return any(
            (
                self.dx,
                self.dy,
                self.wheel,
                self.left_click,
                self.right_click,
                self.middle_click,
            )
        )


def _parse_proc_bus_input() -> dict[str, dict[str, object]]:
    text = Path("/proc/bus/input/devices").read_text(encoding="utf-8", errors="ignore")
    event_map: dict[str, dict[str, object]] = {}
    for block in text.strip().split("\n\n"):
        name = "unknown"
        handlers: list[str] = []
        for line in block.splitlines():
            if line.startswith("N: Name="):
                name = line.split("=", 1)[1].strip().strip('"')
            elif line.startswith("H: Handlers="):
                handlers = line.split("=", 1)[1].split()
        for handler in handlers:
            if handler.startswith("event"):
                event_map[f"/dev/input/{handler}"] = {"name": name, "handlers": handlers}
    return event_map


def list_candidate_pointers(include_touchpad: bool = False) -> list[PointerDeviceInfo]:
    proc_map = _parse_proc_bus_input()
    by_id = {os.path.realpath(path): path for path in glob.glob("/dev/input/by-id/*event-mouse")}
    by_path = {os.path.realpath(path): path for path in glob.glob("/dev/input/by-path/*event-mouse")}

    devices: list[PointerDeviceInfo] = []
    for event_path, meta in proc_map.items():
        handlers = meta["handlers"]
        if not any(handler.startswith("mouse") for handler in handlers):
            continue
        name = str(meta["name"])
        is_touchpad = any(token in name.lower() for token in TOUCHPAD_HINTS)
        if is_touchpad and not include_touchpad:
            continue
        devices.append(
            PointerDeviceInfo(
                event_path=event_path,
                name=name,
                by_id_path=by_id.get(event_path),
                by_path_path=by_path.get(event_path),
                accessible=os.access(event_path, os.R_OK),
                is_touchpad=is_touchpad,
            )
        )

    devices.sort(
        key=lambda item: (
            item.by_id_path is None,
            item.is_touchpad,
            item.name.lower(),
            item.event_path,
        )
    )
    return devices


def describe_devices(devices: list[PointerDeviceInfo]) -> str:
    if not devices:
        return "No pointer devices were found."
    lines = [
        "idx | accessible | type      | event path         | by-id symlink                              | name",
        "----+------------+-----------+--------------------+--------------------------------------------+------------------------------",
    ]
    for index, device in enumerate(devices):
        lines.append(
            f"{index:>3} | {str(device.accessible):<10} | "
            f"{('touchpad' if device.is_touchpad else 'mouse'):<9} | "
            f"{device.event_path:<18} | {str(device.by_id_path or '-'): <42} | {device.name}"
        )
    return "\n".join(lines)


def _device_info_for_path(path: str, include_touchpad: bool) -> PointerDeviceInfo:
    for device in list_candidate_pointers(include_touchpad=include_touchpad):
        if device.event_path == path:
            return device
    raise RuntimeError(f"Pointer device {path} was not found in the current candidate list.")


def choose_pointer_pair(
    include_touchpad: bool = False,
    left_event_path: str | None = None,
    right_event_path: str | None = None,
) -> tuple[PointerDeviceInfo, PointerDeviceInfo]:
    if left_event_path and right_event_path:
        return (
            _device_info_for_path(left_event_path, include_touchpad=include_touchpad),
            _device_info_for_path(right_event_path, include_touchpad=include_touchpad),
        )

    candidates = list_candidate_pointers(include_touchpad=include_touchpad)
    if len(candidates) < 2:
        raise RuntimeError("Need at least two pointer devices to build a left/right pair.")
    return candidates[0], candidates[1]


def _open_device(info: PointerDeviceInfo, grab: bool) -> InputDevice:
    try:
        device = InputDevice(info.event_path)
        device.set_nonblocking(True)
        if grab:
            device.grab()
        return device
    except PermissionError as exc:
        raise DeviceAccessError(f"Permission denied opening {info.event_path} ({info.name})") from exc


class MultiMouseReader:
    def __init__(self, left_info: PointerDeviceInfo, right_info: PointerDeviceInfo, grab: bool = False):
        self.left_info = left_info
        self.right_info = right_info
        self.left_device = _open_device(left_info, grab=grab)
        self.right_device = _open_device(right_info, grab=grab)
        self.grab = grab

    def close(self) -> None:
        for device in (self.left_device, self.right_device):
            try:
                if self.grab:
                    device.ungrab()
            except OSError:
                pass
            device.close()

    def _poll_device(self, device: InputDevice) -> MouseFrame:
        frame = MouseFrame()
        while True:
            try:
                events = device.read()
            except BlockingIOError:
                break
            for event in events:
                if event.type == ecodes.EV_REL:
                    if event.code == ecodes.REL_X:
                        frame.dx += int(event.value)
                    elif event.code == ecodes.REL_Y:
                        frame.dy += int(event.value)
                    elif event.code == ecodes.REL_WHEEL:
                        frame.wheel += int(event.value)
                elif event.type == ecodes.EV_KEY and event.value == 1:
                    if event.code == ecodes.BTN_LEFT:
                        frame.left_click = True
                    elif event.code == ecodes.BTN_RIGHT:
                        frame.right_click = True
                    elif event.code == ecodes.BTN_MIDDLE:
                        frame.middle_click = True
        return frame

    def poll(self) -> dict[str, MouseFrame]:
        return {
            "left": self._poll_device(self.left_device),
            "right": self._poll_device(self.right_device),
        }
