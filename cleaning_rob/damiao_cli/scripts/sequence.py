#!/usr/bin/env python3
"""
Runs a fixed 12-step move sequence across motors 0x01, 0x02, 0x03, with 0x04
held in station throughout. Every motor holds its last commanded position for
the whole run; 2-second hold between steps. After the final step all motors hold
until Ctrl+C.

Loaded joints are held with the feedforward torque their last move measured, so
they sit ON the commanded position rather than load/kp_hold below it.

Single-threaded by design. One MotorBus serves exactly one thread: _send_raw is
a send-then-receive transaction over a shared socket that DISCARDS any frame it
did not ask for, so a second thread streaming the idle motors silently eats the
move loop's feedback. The move then sees no replies, decides the motor never
moved, gives up on the stall guard and returns the position it started from --
which the next hold actively drives the motor back to. That is the "jump a bit,
fall back down" failure this script used to have.

So idle motors are fed from *inside* the move loop instead, through
move_by_offset's on_tick callback, on the one thread. In MIT mode they need it:
there is no setpoint latch, so a motor with no frame for long enough trips its
comms-loss watchdog and drops out of enable mode. In POS_VEL the firmware
latches the position setpoint and holds internally, so no streaming is needed.
"""

import argparse
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# The 5 ms streaming feedback wait. send_command's 50 ms default would stall
# these loops badly with four sends per tick, and any missed reply would park
# the bus for the full 50 ms; missing a sample is much cheaper than starving the
# other motors' command stream.
from motor import MotorBus, _STREAM_RECV_TIMEOUT

MODE_MAP = {"mit": MotorBus.MODE_MIT, "pos_vel": MotorBus.MODE_POS_VEL}

HOLD_SECS   = 2.0
LOOP_PERIOD = 0.005  # seconds per control tick

SEQUENCE = [
    (0x02,  -90),
    (0x03,  -90),
    (0x01,  90),
    (0x04, -90),
    (0x01, -180),
    (0x04, 45),
    (0x01,  90),
    # (0x03,  -90),
    # (0x02,  -45),
    # (0x01,  45),
    # (0x01, -90),
    # (0x01,  45),
    (0x03, 85),
    (0x02, 90),
    (0x04, 45),
]

# 0x04 is held-only: it never appears in SEQUENCE, but it is on the bus and has
# to keep station like the rest. All of these must answer at startup.
MOTOR_IDS = [0x01, 0x02, 0x03, 0x04]


def hold(bus, positions, motor_ids):
    """Send one hold frame to each of `motor_ids`. Gains come from params.

    The torque is the static load the motor's last settle measured (load_ff).
    Without it a loaded joint sits at load/kp_hold BELOW the commanded position
    -- a PD law makes torque only from error -- which is the sag that used to
    appear the moment a move handed off to the hold.
    """
    for mid in motor_ids:
        bus.send_command(mid, positions[mid], vel=0.0,
                         torque=bus.load_ff(mid),
                         recv_timeout=_STREAM_RECV_TIMEOUT)


def hold_all(bus, positions, duration, mode):
    """Hold every motor for `duration`.

    In POS_VEL the motor firmware latches its position setpoint and holds
    internally, so there is nothing to stream and this just waits.
    NOTE / revisit: if the comms-loss watchdog disables a motor on timeout (see
    memory damiao-motor-watchdog), reinstate POS_VEL resend-holding here. MIT
    mode has no latch, so it must keep streaming the hold command.
    """
    if mode != bus.MODE_MIT:
        time.sleep(duration)  # motors hold internally
        return
    end = time.monotonic() + duration
    while time.monotonic() < end:
        hold(bus, positions, MOTOR_IDS)
        time.sleep(LOOP_PERIOD)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["mit", "pos_vel"], default="mit",
                        help="Control mode (default: mit)")
    parser.add_argument("--params", default=None, metavar="PATH",
                        help="Tuning params YAML (default: bundled params.yaml)")
    args = parser.parse_args()
    mode = MODE_MAP[args.mode]

    bus = MotorBus(params_path=args.params)
    positions = {}

    try:
        bus.init_motors(MOTOR_IDS, mode)  # set control mode before enabling

        print("Entering motor mode...")
        for mid in MOTOR_IDS:
            pos = bus.enter_motor_mode(mid)
            if pos is None:
                raise RuntimeError(
                    f"Motor 0x{mid:02X} is listed in MOTOR_IDS but did not "
                    f"respond. Power it up, or drop it from MOTOR_IDS if it is "
                    f"not on the bus.")
            positions[mid] = pos

        for step, (motor_id, degrees) in enumerate(SEQUENCE, 1):
            delta = math.radians(degrees)
            print(f"Step {step:2d}: motor 0x{motor_id:02X}  {degrees:+.0f} deg")
            hold_all(bus, positions, HOLD_SECS, mode)  # wait before continuing

            # Keep the idle motors alive on this same thread, one hold frame per
            # control tick, interleaved with the moving motor's own frames.
            # Doing this from a second thread is what broke this script: see the
            # module docstring and MotorBus._send_raw.
            idle = [mid for mid in MOTOR_IDS if mid != motor_id]
            on_tick = ((lambda: hold(bus, positions, idle))
                       if mode == bus.MODE_MIT else None)

            positions[motor_id] = bus.move_by_offset(
                motor_id, delta, start_pos=positions[motor_id], mode=mode,
                on_tick=on_tick,
            )

        print("Sequence complete. Holding all motors. Ctrl+C to exit.")
        while True:
            hold_all(bus, positions, HOLD_SECS, mode)

    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        for mid in MOTOR_IDS:
            bus.exit_motor_mode(mid)
        bus.shutdown()


if __name__ == "__main__":
    main()
