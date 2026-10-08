from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from realsense.slam.ASMTCDR import (
    IntegratedMappingAndMonitoringSystem,
    MapUpdate,
)


class RoomMapLogger:
    """Handles storage and persistence of real-time mapping updates."""

    def __init__(self, output_dir: Optional[Path] = None) -> None:
        self.output_dir = output_dir or Path("artifacts/room_map")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._map_updates: List[Dict[str, Any]] = []
        self._change_events: List[Dict[str, Any]] = []

    def handle_map_update(self, update: MapUpdate) -> None:
        payload = update.as_dict()
        self._map_updates.append(payload)
        print(
            f"[room_map] Stored map update #{len(self._map_updates)} "
            f"at pose {payload['pose']} (RGB {payload['rgb_resolution']})"
        )

    def handle_change_event(self, change_summary: Dict[str, Any]) -> None:
        self._change_events.append(change_summary)
        print(
            f"[room_map] Change event logged (count={change_summary.get('change_count', 'n/a')})."
        )

    def persist_session(self, session_summary: Dict[str, Any]) -> Path:
        timestamp = int(time.time())
        payload = {
            "session": session_summary,
            "map_updates": self._map_updates,
            "change_events": self._change_events,
        }

        output_path = self.output_dir / f"session_{timestamp}.json"
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"[room_map] Session summary written to {output_path}.")
        return output_path


def run_autonomous_room_mapping(
    duration_s: int = 60,
    output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Runs the integrated SLAM pipeline and stores results in `room_map.py`."""

    system = IntegratedMappingAndMonitoringSystem()
    logger = RoomMapLogger(output_dir=output_dir)

    session_summary = system.start_autonomous_operation(
        duration_s=duration_s,
        on_map_update=logger.handle_map_update,
        on_change_detected=logger.handle_change_event,
    )

    logger.persist_session(session_summary)
    return session_summary


if __name__ == "__main__":
    run_autonomous_room_mapping(duration_s=10)