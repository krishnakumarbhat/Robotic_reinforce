"""
00_diag.py — fast sanity check for the SO-ARM bus.

Tries to ping all 6 expected Feetech motors on PORT. If any reply, tries
a quick torque enable + read on motor 1 so the user can SEE the bus is alive
(LEDs blink on STS3215 torque-on). Exits non-zero if no motors found.
"""
import sys
import time

LEROBOT_SRC = "/media/pope/projecteo/github_proj/a_resume/Robotic_reinforce/lerobot_rl/lerobot/src"
sys.path.insert(0, LEROBOT_SRC)

from lerobot.motors.feetech import FeetechMotorsBus
from lerobot.motors.motors_bus import Motor, MotorNormMode

PORT = "/dev/ttyACM0"
EXPECTED = {1: "sts3215", 2: "sts3215", 3: "sts3215", 4: "sts3215", 5: "sts3215", 6: "sts3215"}


def main() -> int:
    bus = FeetechMotorsBus(
        port=PORT,
        motors={f"m{i}": Motor(i, EXPECTED[i], MotorNormMode.DEGREES) for i in EXPECTED},
    )
    try:
        bus.connect(handshake=True)
    except Exception as e:
        print(f"FAIL connect({PORT}): {e}")
        return 2
    found = bus.ping()
    print(f"ping result: {found}")
    if not found:
        print("FAIL: no Feetech servos answering. Bus bridge firmware missing or DIP switch wrong.")
        return 1
    bus.enable_torque()
    time.sleep(0.2)
    pos = bus.sync_read("Present_Position", None)
    print(f"present_position: {pos}")
    bus.disable_torque()
    print("OK: bus alive, torque blinked.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
