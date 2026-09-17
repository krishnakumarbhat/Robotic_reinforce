"""
SO-100/101 follower calibration (SO-ARM low-cost Feetech STS3215 bus).
Uses the lerobot FeetechMotorsBus already vendored at
../Robotic_reinforce/lerobot_rl/lerobot/src — no new disk footprint.
"""
import json
import os
import sys
from pathlib import Path

LEROBOT_SRC = "/media/pope/projecteo/github_proj/a_resume/Robotic_reinforce/lerobot_rl/lerobot/src"
sys.path.insert(0, LEROBOT_SRC)

from lerobot.motors.feetech import FeetechMotorsBus
from lerobot.motors.motors_bus import Motor, MotorNormMode
from lerobot.motors.feetech.feetech import OperatingMode

PORT = "/dev/ttyACM0"
ID = "so_follower"
OUT = Path(__file__).resolve().parent.parent / "calibration" / f"{ID}.json"
OUT.parent.mkdir(parents=True, exist_ok=True)

MOTORS = {
    "shoulder_pan":  (1, "sts3215", MotorNormMode.DEGREES),
    "shoulder_lift": (2, "sts3215", MotorNormMode.DEGREES),
    "elbow_flex":    (3, "sts3215", MotorNormMode.DEGREES),
    "wrist_flex":    (4, "sts3215", MotorNormMode.DEGREES),
    "wrist_roll":    (5, "sts3215", MotorNormMode.DEGREES),
    "gripper":       (6, "sts3215", MotorNormMode.RANGE_0_100),
}


def main() -> int:
    bus = FeetechMotorsBus(
        port=PORT,
        motors={name: Motor(idx, model, norm) for name, (idx, model, norm) in MOTORS.items()},
    )
    bus.connect(handshake=True)
    print(f"connected to {PORT}; bus motors: {list(bus.motors)}")

    if OUT.exists():
        ans = input(f"calibration file exists ({OUT}). Re-run? [y/N]: ").strip().lower()
        if ans != "y":
            cached = json.loads(OUT.read_text())
            bus.write_calibration(cached)
            print("loaded cached calibration; exiting")
            return 0

    bus.disable_torque()
    for name in bus.motors:
        bus.write("Operating_Mode", name, OperatingMode.POSITION.value)

    input("Move arm to the MIDDLE of its range and press ENTER...")
    homing = bus.set_half_turn_homings()
    print("homing offsets:", homing)

    full_turn = "wrist_roll"
    others = [m for m in bus.motors if m != full_turn]
    print(f"Sweep all joints except '{full_turn}' through their full range; ENTER to stop...")
    bus.record_ranges_of_motion(motors=others)

    cal = {m: {
        "id": MOTORS[m][0], "model": MOTORS[m][1], "norm_mode": MOTORS[m][2].name,
        "homing_offset": homing[m].homings[0] if hasattr(homing[m], "homings") else homing[m],
        "range_min": bus.calibration[m].range_min if m in bus.calibration else 0,
        "range_max": bus.calibration[m].range_max if m in bus.calibration else 4095,
    } for m in bus.motors}

    OUT.write_text(json.dumps(cal, indent=2))
    bus.write_calibration({m: bus.calibration[m] for m in bus.motors})
    print(f"saved -> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
