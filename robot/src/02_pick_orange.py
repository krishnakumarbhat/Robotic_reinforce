"""
Visual-servo orange pick with SO-100/101 follower. FAST version.

Loop runs as fast as camera + bus allow (no waitKey, no sleeps in the loop).
Use 'p' in the preview window to trigger a pick; the in-pick servo runs at
~100Hz and only settles when the orange is centered.
"""
import json
import sys
import time
from pathlib import Path

LEROBOT_SRC = "/media/pope/projecteo/github_proj/a_resume/Robotic_reinforce/lerobot_rl/lerobot/src"
sys.path.insert(0, LEROBOT_SRC)

import cv2
import numpy as np
from lerobot.motors.feetech import FeetechMotorsBus
from lerobot.motors.motors_bus import Motor, MotorNormMode
from lerobot.motors.feetech.feetech import OperatingMode

PORT = "/dev/ttyACM0"
CAM_INDEX = 0
ID = "so_follower"
CALIB = Path(__file__).resolve().parent.parent / "calibration" / f"{ID}.json"

MOTOR_ORDER = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
MOTOR_MODELS = {m: "sts3215" for m in MOTOR_ORDER}
NORM = {**{m: MotorNormMode.DEGREES for m in MOTOR_ORDER}, "gripper": MotorNormMode.RANGE_0_100}
IDX  = {m: i + 1 for i, m in enumerate(MOTOR_ORDER)}

HSV_LO = np.array([5, 110, 90], dtype=np.uint8)
HSV_HI = np.array([25, 255, 255], dtype=np.uint8)

# FAST: higher gains, larger per-step deltas, no human pausing.
PAN_K   = 0.12
LIFT_K  = 0.10
ELBOW_K = 0.08
MAX_STEP_DEG = 8.0          # cap per-joint step so big pixel errors don't snap
PIX_TOL = 10
SETTLE_FRAMES = 3            # need this many "centered" frames in a row
DESCEND_DEG = 25.0           # how much to lower the arm before closing
LIFT_DEG    = 30.0           # how much to raise after grip

HOME = {
    "shoulder_pan": 0.0, "shoulder_lift": -90.0, "elbow_flex": 90.0,
    "wrist_flex": -30.0, "wrist_roll": 0.0, "gripper": 80.0,
}


def find_orange(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, HSV_LO, HSV_HI)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None, mask
    c = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(c) < 400:
        return None, mask
    (x, y), r = cv2.minEnclosingCircle(c)
    return (float(x), float(y), float(r)), mask


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def main() -> int:
    bus = FeetechMotorsBus(
        port=PORT,
        motors={m: Motor(IDX[m], MOTOR_MODELS[m], NORM[m]) for m in MOTOR_ORDER},
    )
    bus.connect(handshake=True)
    if CALIB.exists():
        bus.write_calibration(json.loads(CALIB.read_text()))
        print(f"loaded calibration: {CALIB}")
    for m in MOTOR_ORDER:
        bus.write("Operating_Mode", m, OperatingMode.POSITION.value)
    bus.enable_torque()
    bus.sync_write("Goal_Position", HOME)
    time.sleep(0.5)
    print(f"connected {PORT}; at HOME; servoing fast (no human prompts)")

    cap = cv2.VideoCapture(CAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)        # ponytail: drop stale frames
    if not cap.isOpened():
        print("camera open failed"); return 1
    cv2.namedWindow("cam", cv2.WINDOW_NORMAL)
    cv2.namedWindow("mask", cv2.WINDOW_NORMAL)

    pick_now = False
    last_write = 0.0
    WRITE_HZ = 0.01                            # 100Hz cap on bus writes
    pick_running = False
    pick_state = "approach"                    # approach -> descend -> grip -> lift
    settled = 0

    try:
        while True:
            t0 = time.perf_counter()
            ok, frame = cap.read()
            if not ok:
                continue
            blob, mask = find_orange(frame)
            vis = frame.copy()
            if blob is not None:
                x, y, r = blob
                cv2.circle(vis, (int(x), int(y)), int(r), (0, 255, 0), 2)
                cv2.putText(vis, f"orange r={r:.0f}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            else:
                cv2.putText(vis, "no orange", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.putText(vis, f"pick:{pick_state if pick_running else 'idle'}",
                        (10, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            cv2.imshow("cam", vis); cv2.imshow("mask", mask)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("p") and not pick_running:
                pick_running = True
                pick_state = "approach"
                settled = 0  # noqa: F841 (kept explicit for clarity)

            if not pick_running and (t0 - last_write) >= WRITE_HZ:
                # idle: hold HOME so the arm doesn't drift
                bus.sync_write("Goal_Position", HOME)
                last_write = t0
            elif pick_running and blob is not None and (t0 - last_write) >= WRITE_HZ:
                h, w = frame.shape[:2]
                cx, cy, r = blob
                dx, dy = cx - w / 2.0, cy - h / 2.0
                cur = bus.sync_read("Present_Position", MOTOR_ORDER)
                tgt = dict(HOME)

                if pick_state == "approach":
                    tgt["shoulder_pan"]  = float(cur["shoulder_pan"])  - clamp(PAN_K * dx,   -MAX_STEP_DEG, MAX_STEP_DEG)
                    tgt["shoulder_lift"] = float(cur["shoulder_lift"]) - clamp(LIFT_K * dy,  -MAX_STEP_DEG, MAX_STEP_DEG)
                    tgt["elbow_flex"]    = float(cur["elbow_flex"])    + clamp(ELBOW_K * dy, -MAX_STEP_DEG, MAX_STEP_DEG)
                    tgt["wrist_flex"]    = float(cur["wrist_flex"])    - clamp(ELBOW_K * dy, -MAX_STEP_DEG, MAX_STEP_DEG) * 0.5
                    tgt["gripper"] = 80.0
                    if abs(dx) < PIX_TOL and abs(dy) < PIX_TOL:
                        settled += 1
                        if settled >= SETTLE_FRAMES:
                            pick_state = "descend"; settled = 0
                    else:
                        settled = 0
                elif pick_state == "descend":
                    tgt["shoulder_lift"] = float(cur["shoulder_lift"]) + DESCEND_DEG
                    tgt["gripper"] = 80.0
                    pick_state = "grip"
                elif pick_state == "grip":
                    tgt["gripper"] = 0.0
                    pick_state = "lift"
                elif pick_state == "lift":
                    tgt["shoulder_lift"] = float(cur["shoulder_lift"]) - LIFT_DEG
                    tgt["gripper"] = 0.0
                    pick_running = False

                bus.sync_write("Goal_Position", tgt)
                last_write = t0
    finally:
        bus.disable_torque(); bus.disconnect()
        cap.release(); cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
