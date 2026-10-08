#!/usr/bin/env python3
"""
MIT motor controller CLI.

Usage examples:
  python run.py move_by_offset 0x03 20
  python run.py move_to_pos    0x03 -45.0
  python run.py continuous_movement_at_vel 0x03 10.0
  python run.py set_zero 0x03
  python run.py set_id   0x01 0x03
  python run.py query    0x03
  python run.py limits   0x03
"""

import argparse
import math
import sys

from motor import ERR_CODES, MotorBus


def _hex_int(s):
    return int(s, 0)


def _state(fb):
    """Feedback status as a readable word ('enabled', 'overload', ...)."""
    return fb.error or ERR_CODES.get(fb.status, f"status 0x{fb.status:X}")


def _raw_note(args):
    """Tag a raw move in the log line, so a captured session says which control
    stack produced the trace."""
    return "  (raw: no profile/gain ramp/settle)" if args.raw else ""


MODE_MAP = {"mit": MotorBus.MODE_MIT, "pos_vel": MotorBus.MODE_POS_VEL}


# ------------------------------------------------------------------ #
# Command handlers                                                     #
# ------------------------------------------------------------------ #

def cmd_move_by_offset(bus, args):
    motor_id = args.motor_id
    mode = MODE_MAP[args.mode]
    bus.init_motors([motor_id], mode)  # set control mode before enabling
    start = bus.enter_motor_mode(motor_id)
    if start is None:
        return
    try:
        delta = math.radians(args.degrees)
        print(f"Moving {args.degrees:+.1f} deg from {math.degrees(start):.2f} deg"
              f"{_raw_note(args)}...")
        target = bus.move_by_offset(motor_id, delta, start_pos=start,
                                    velocity=args.velocity, mode=mode,
                                    kp=args.kp, kd=args.kd, raw=args.raw)
        print(f"Reached {math.degrees(target):.2f} deg. Holding... Ctrl+C to stop.")
        bus.hold_position(motor_id, target, mode=mode)
    except KeyboardInterrupt:
        print("\nInterrupted!")
    finally:
        bus.exit_motor_mode(motor_id)


def cmd_move_to_pos(bus, args):
    motor_id = args.motor_id
    mode = MODE_MAP[args.mode]
    bus.init_motors([motor_id], mode)  # set control mode before enabling
    start = bus.enter_motor_mode(motor_id)
    if start is None:
        return
    try:
        target_rad = math.radians(args.degrees)
        print(f"Moving to {args.degrees:.2f} deg from {math.degrees(start):.2f} deg"
              f"{_raw_note(args)}...")
        reached = bus.move_to_pos(motor_id, target_rad, start_pos=start,
                                  velocity=args.velocity, mode=mode,
                                  kp=args.kp, kd=args.kd, raw=args.raw)
        print(f"Reached {math.degrees(reached):.2f} deg. Holding... Ctrl+C to stop.")
        bus.hold_position(motor_id, reached, mode=mode)
    except KeyboardInterrupt:
        print("\nInterrupted!")
    finally:
        bus.exit_motor_mode(motor_id)


def cmd_set_zero(bus, args):
    # set_zero does not require motor mode
    bus.set_zero(args.motor_id)


def cmd_set_id(bus, args):
    new_mst = args.new_mst_id if args.new_mst_id is not None else None
    bus.set_id(args.motor_id, args.new_id, new_mst_id=new_mst)


def cmd_continuous_vel(bus, args):
    motor_id = args.motor_id
    # This command streams MIT velocity frames; force MIT mode in case the motor
    # was left in POS_VEL by a previous run.
    bus.init_motors([motor_id], MotorBus.MODE_MIT)
    start = bus.enter_motor_mode(motor_id)
    if start is None:
        return
    try:
        print(f"Running at {args.velocity:.2f} rad/s. Ctrl+C to stop.")
        bus.continuous_movement_at_vel(motor_id, args.velocity)
    finally:
        bus.exit_motor_mode(motor_id)


def cmd_query(bus, args):
    fb = bus.query(args.motor_id)
    if fb is None:
        print(f"Motor 0x{args.motor_id:02X}: no response.")
        return
    print(f"Motor 0x{args.motor_id:02X}:  pos={math.degrees(fb.pos):.2f} deg  "
          f"vel={fb.vel:.3f} rad/s  torque={fb.torque:.3f} Nm  "
          f"T_mos={fb.t_mos:.0f}C  T_rotor={fb.t_rotor:.0f}C  [{_state(fb)}]")


def cmd_scan(bus, args):
    ids = range(args.start, args.end + 1)
    results = bus.scan(ids)
    print(f"\n{'ID':<6} {'Position (deg)':>14} {'Velocity (rad/s)':>17} {'Torque (Nm)':>12}  {'State':<12}")
    print("-" * 68)
    for mid, fb in results.items():
        if fb is None:
            print(f"0x{mid:02X}   {'–':>14} {'–':>17} {'–':>12}  {'–':<12}")
        else:
            print(f"0x{mid:02X}   {math.degrees(fb.pos):>13.2f}° {fb.vel:>16.3f}  "
                  f"{fb.torque:>11.3f}  {_state(fb):<12}")
    print()


def cmd_limits(bus, args):
    """Show the MIT scaling limits a motor reports and the tuning they select.

    Worth checking whenever a motor behaves oddly: if these are wrong, every
    velocity and torque value on the wire is silently mis-scaled.
    """
    motor_id = args.motor_id
    limits, model = bus.resolve(motor_id, force=True)
    params = bus.params_for(motor_id)
    print(f"\nMotor 0x{motor_id:02X}")
    print(f"  model      {model or 'unknown (no vendor row matches)'}")
    print(f"  PMAX       ±{limits.p_max:g} rad  (±{math.degrees(limits.p_max):.0f}°)")
    print(f"  VMAX       ±{limits.v_max:g} rad/s")
    print(f"  TMAX       ±{limits.t_max:g} Nm")
    print(f"  move       kp={params.kp_move:g}  kd={params.kd_move:g}  "
          f"velocity={params.velocity:g} rad/s  accel={params.accel:g} rad/s²")
    print(f"  hold       kp={params.kp_hold:g}  kd={params.kd_hold:g}")
    print(f"  bilateral  kp={params.kp_bilateral:g}  kd={params.kd_bilateral:g}"
          f"   (as leader: kp={params.kp_leader:g}  kd={params.kd_leader:g})")
    saturation = math.degrees(limits.t_max / params.kp_move) if params.kp_move else float('inf')
    print(f"  kp_move saturates TMAX at {saturation:.1f}° of position error\n")


_COMMANDS = [
    ("move_by_offset <id> <deg> [--mode mit|pos_vel] [--raw] [--kp K] [--kd K]",
     "Move a motor by a relative offset in degrees from its current position, then hold."),
    ("move_to_pos <id> <deg> [--mode mit|pos_vel] [--raw] [--kp K] [--kd K]",
     "Move a motor to an absolute position in degrees, then hold."),
    ("continuous_movement_at_vel <id> <rad/s>",
     "Spin a motor at a constant velocity until Ctrl+C; ramps down to zero on stop."),
    ("set_zero <id>",
     "Set the motor's current position as its zero reference (persists across power cycles)."),
    ("set_id <id> <new_id>",
     "Change the motor's CAN ID via register write; power-cycle to apply."),
    ("query <id>",
     "Read and print a single motor's position, velocity, torque, temps and state."),
    ("scan [--start <id>] [--end <id>]",
     "Query all motors from 0x01 to 0x07 (or a custom range) and print a summary table."),
    ("limits <id>",
     "Show the motor's PMAX/VMAX/TMAX, the model they identify, and the gains selected for it."),
    ("help",
     "Show this command reference."),
]

def cmd_help(bus, args):
    print("\nMIT motor controller — available commands:\n")
    for name, desc in _COMMANDS:
        print(f"  {name}")
        print(f"      {desc}\n")
    print("Global options: --channel (default: vcan0)  --interface (default: socketcan)  "
          "--params (default: params.yaml)\n")
    print("  --raw commands the target position directly: no S-curve profile, no ramp\n"
          "  into the hold gains, no end-of-move settle. The stall/timeout guards stay.\n"
          "  MIT mode only. Useful as the baseline to compare a tuned move against.\n")


# ------------------------------------------------------------------ #
# Argument parsing                                                     #
# ------------------------------------------------------------------ #

def _add_move_args(p, degrees_help):
    """Arguments shared by move_to_pos and move_by_offset. The two differ only in
    what `degrees` means, so they are built from one place."""
    p.add_argument("motor_id", type=_hex_int)
    p.add_argument("degrees",  type=float, help=degrees_help)
    p.add_argument("--velocity", type=float, default=4.0, metavar="RAD/S",
                   help="S-curve cruise speed (ignored with --raw)")
    p.add_argument("--mode", choices=["mit", "pos_vel"], default="mit",
                   help="Control mode (default: mit)")
    p.add_argument("--raw", action="store_true",
                   help="Command the target directly: no S-curve profile, no gain "
                        "ramp, no settle. Stall/timeout guards stay. MIT mode only.")
    p.add_argument("--kp", type=float, default=None, metavar="K",
                   help="Move position gain (default: the model's kp_move). Setting "
                        "kp/kd also disables the ramp to the hold gains, so the move "
                        "runs at these gains all the way to the end.")
    p.add_argument("--kd", type=float, default=None, metavar="K",
                   help="Move damping gain (default: the model's kd_move)")


def build_parser():
    parser = argparse.ArgumentParser(
        description="MIT motor controller CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--channel",   default="vcan0",     help="CAN channel (default: vcan0)")
    parser.add_argument("--interface", default="socketcan", help="CAN interface (default: socketcan)")
    parser.add_argument("--params",    default=None, metavar="PATH",
                        help="Tuning params YAML (default: bundled params.yaml)")

    sub = parser.add_subparsers(dest="command", required=True)

    # move_by_offset
    p = sub.add_parser("move_by_offset", help="Move relative to current position, then hold")
    _add_move_args(p, "Offset in degrees (positive = forward)")

    # move_to_pos
    p = sub.add_parser("move_to_pos", help="Move to absolute position, then hold")
    _add_move_args(p, "Target position in degrees")

    # set_zero
    p = sub.add_parser("set_zero", help="Set current position as zero (persists across power cycles)")
    p.add_argument("motor_id", type=_hex_int)

    # set_id
    p = sub.add_parser("set_id", help="Change motor CAN ID (power-cycle to apply)")
    p.add_argument("motor_id", type=_hex_int, help="Current ESC CAN ID")
    p.add_argument("new_id",   type=_hex_int, help="New ESC CAN ID")
    p.add_argument("--new-mst-id", type=_hex_int, default=None,
                   dest="new_mst_id",
                   help="New master/feedback CAN ID (default: new_id | 0x10)")

    # continuous_movement_at_vel
    p = sub.add_parser("continuous_movement_at_vel",
                       help="Spin at constant velocity until Ctrl+C (ramps down on stop)")
    p.add_argument("motor_id", type=_hex_int)
    p.add_argument("velocity", type=float, help="Velocity in rad/s (negative = reverse)")

    # query
    p = sub.add_parser("query", help="Read motor position, velocity, and torque")
    p.add_argument("motor_id", type=_hex_int)

    # scan
    p = sub.add_parser("scan", help="Read positions of all motors in a range (default 0x01–0x07)")
    p.add_argument("--start", type=_hex_int, default=0x01, help="First motor ID (default: 0x01)")
    p.add_argument("--end",   type=_hex_int, default=0x07, help="Last motor ID (default: 0x07)")

    # limits
    p = sub.add_parser("limits", help="Show a motor's MIT scaling limits, model and gains")
    p.add_argument("motor_id", type=_hex_int)

    # help
    sub.add_parser("help", help="Show all available commands and their purpose")

    return parser


DISPATCH = {
    "move_by_offset":            cmd_move_by_offset,
    "move_to_pos":               cmd_move_to_pos,
    "set_zero":                  cmd_set_zero,
    "set_id":                    cmd_set_id,
    "continuous_movement_at_vel": cmd_continuous_vel,
    "query":                     cmd_query,
    "scan":                      cmd_scan,
    "limits":                    cmd_limits,
    "help":                      cmd_help,
}


def main():
    parser = build_parser()
    args = parser.parse_args()
    # Rejected rather than ignored: in POS_VEL the motor's own firmware runs the
    # position loop, so "raw" would silently mean something different from what
    # it means in MIT -- and the trace would be attributed to the wrong stack.
    if getattr(args, "raw", False) and args.mode != "mit":
        parser.error("--raw applies to MIT mode only: in --mode pos_vel the motor's "
                     "firmware runs the position loop, so there are no host-side "
                     "helpers to strip.")
    bus = MotorBus(channel=args.channel, interface=args.interface,
                   params_path=args.params)
    try:
        DISPATCH[args.command](bus, args)
    finally:
        bus.shutdown()


if __name__ == "__main__":
    main()
