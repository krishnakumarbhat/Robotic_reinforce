#!/usr/bin/env python3
"""
Bilateral coupling test for two motors on the same bus (default 0x01 and 0x04).

The pair is joined by a virtual spring/damper and holds a fixed positional
relationship. Backdrive either motor by hand and the other follows; hold one
still and you feel the reaction force build at the other; let go and the pair
settles back to equilibrium. The coupling is symmetric -- neither motor is the
leader -- so force reflects in both directions.

Equilibrium is whatever offset the two motors are at when the loop engages, so
nothing moves at startup. Pass --offset to converge on a different one.

--leader names the motor you drive by hand. It keeps its full spring stiffness
but runs much lower damping, because kd works against the follower's velocity
lag and so reads as viscous drag on whoever is doing the moving. Force
reflection is unaffected: block the follower and the leader still pushes back at
full kp. This is safe because kd acts on the pair's *relative* velocity -- both
motors damp the same coordinate, so the follower's kd keeps anchoring the pair.

    equilibrium:  p1 = s*p2 + offset          (s = -1 with --mirror)

    p1_des = s*p2 + offset      v1_des = s*v2
    p2_des = s*(p1 - offset)    v2_des = s*v1

MIT mode only. The motor firmware evaluates tau = KP*(p_des - p) + KD*(v_des - v)
against its own encoder, so feeding each motor its partner's position and
velocity puts the coupling law on-board at the motor's own loop rate -- the host
just relays state between the two. POS_VEL runs a stiff firmware position loop
with no compliance and cannot do this.

Usage:
    python scripts/bilateral.py
    python scripts/bilateral.py --motors 0x01 0x04 --kp1 6 --kp2 3
    python scripts/bilateral.py --mirror --max-error 10
"""

import argparse
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# The 5 ms streaming feedback wait. send_command's 50 ms default would stall
# this loop to ~10 Hz with two sends per tick; missing a sample is cheaper than
# holding a stale setpoint while the partner keeps moving.
from motor import MotorBus, _STREAM_RECV_TIMEOUT

# No feedback from a motor for this long means it is gone (unplugged, faulted
# out of enable mode, bus down). Commanding the survivor from its partner's
# frozen last-known position would drive a growing spring error into a motor
# that is no longer answering, so give up instead.
STALE_ABORT   = 0.5   # s
STATUS_PERIOD = 0.1   # s between status-line repaints


def _hex_int(s):
    return int(s, 0)


def _clamp(value, lo, hi):
    return max(min(value, hi), lo)


class _Side:
    """One motor of the coupled pair: its gains, its latest state, and the
    send that commands it from the partner's state."""

    def __init__(self, bus, motor_id, kp, kd, vel_alpha):
        self.bus       = bus
        self.id        = motor_id
        self.kp        = kp
        self.kd        = kd
        self.alpha     = vel_alpha
        self.pos       = 0.0
        self.vel       = 0.0
        self.vel_f     = 0.0   # filtered; this is what the partner tracks
        self.torque    = 0.0
        self.last_heard = 0.0

    def prime(self, pos):
        self.pos = pos
        self.last_heard = time.monotonic()

    def step(self, p_des, v_des, kp_scale, max_err):
        """Send one coupling frame and fold in whatever feedback comes back.

        p_des is clamped to within max_err of this motor's own measured
        position, which turns the spring into a saturating one: the coupling
        torque can never exceed kp*max_err no matter how far the partner is
        dragged away, so yanking one motor cannot command a violent snap-back.
        """
        p_des = _clamp(p_des, self.pos - max_err, self.pos + max_err)
        fb = self.bus.send_command(self.id, p_des, vel=v_des,
                                   kp=self.kp * kp_scale, kd=self.kd,
                                   recv_timeout=_STREAM_RECV_TIMEOUT)
        if fb is not None:
            self.pos, self.vel, self.torque = fb.pos, fb.vel, fb.torque
            # Feedback velocity is 12-bit quantized over +/-VMAX. It is fed
            # straight into the partner's kd term, where the noise would come
            # back out as torque ripple, so smooth it first.
            self.vel_f += self.alpha * (fb.vel - self.vel_f)
            self.last_heard = time.monotonic()
        return fb


class StaleFeedback(RuntimeError):
    pass


def _gains_for(params, is_leader, leader_scale, kp_override, kd_override):
    """(kp, kd) for one motor: role gains, scaled, then explicit overrides.

    The leader keeps its stiffness but runs the softer bilateral.leader damping.
    An explicit --kpN/--kdN is taken literally -- it bypasses both the role gains
    and --leader-scale, so a number on the command line is the number that
    reaches the motor.
    """
    if is_leader:
        kp = params.kp_leader * leader_scale
        kd = params.kd_leader * leader_scale
    else:
        kp, kd = params.kp_bilateral, params.kd_bilateral
    return (kp if kp_override is None else kp_override,
            kd if kd_override is None else kd_override)


def _schedule(elapsed, offset0, offset_target, engage, converge):
    """(kp_scale, offset) for a given time since engage.

    kp ramps 0 -> full over `engage` so the coupling comes up without a jolt;
    kd is at full value from the very first frame, because the dangerous way to
    engage a spring is undamped. Only once the spring is fully up does the
    equilibrium offset ramp toward an explicitly requested one.
    """
    kp_scale = 1.0 if engage <= 0 else _clamp(elapsed / engage, 0.0, 1.0)
    if offset_target is None:
        return kp_scale, offset0
    if converge <= 0:
        frac = 1.0
    else:
        frac = _clamp((elapsed - engage) / converge, 0.0, 1.0)
    return kp_scale, offset0 + (offset_target - offset0) * frac


def _tick(a, b, s, offset, kp_scale, max_err, couple_vel):
    """One coupling update: command each motor from the other's state.

    Both setpoints are computed from the same snapshot, before either send.
    Sending a first and then deriving b's target from a's freshly-updated
    position makes b see an error a has already started correcting, so a
    consistently does more of the work -- a half-tick asymmetry that compounds
    into the pair creeping along the one direction the coupling doesn't
    constrain. Snapshotting keeps the two commands consistent with one instant.
    """
    pa, va = a.pos, a.vel_f
    pb, vb = b.pos, b.vel_f
    a.step(s * pb + offset, s * vb if couple_vel else 0.0, kp_scale, max_err)
    b.step(s * (pa - offset), s * va if couple_vel else 0.0, kp_scale, max_err)


def _check_alive(a, b, now):
    for side in (a, b):
        if now - side.last_heard > STALE_ABORT:
            raise StaleFeedback(
                f"motor 0x{side.id:02X} stopped answering for "
                f"{now - side.last_heard:.2f}s")


def _status(a, b, s, offset, hz, leader=None):
    err = math.degrees(a.pos - s * b.pos - offset)
    # '*' marks the leader, so which motor is which role stays obvious while
    # your hands are on the hardware and not on the startup banner.
    ma = "*" if a.id == leader else " "
    mb = "*" if b.id == leader else " "
    sys.stdout.write(
        f"\r  0x{a.id:02X}{ma}{math.degrees(a.pos):+8.2f}° {a.torque:+6.2f}Nm   "
        f"0x{b.id:02X}{mb}{math.degrees(b.pos):+8.2f}° {b.torque:+6.2f}Nm   "
        f"err {err:+7.2f}°   {hz:4.0f}Hz ")
    sys.stdout.flush()


def _release(a, b, s, offset, args, period):
    """Ramp the spring out before disabling. Cutting torque instantly while the
    coupling is stretched would let a loaded joint drop; decaying kp to zero
    leaves the motors damped and limp instead."""
    if args.release <= 0:
        return
    t0 = time.monotonic()
    max_err = math.radians(args.max_error)
    while True:
        frac = (time.monotonic() - t0) / args.release
        if frac >= 1.0:
            break
        _tick(a, b, s, offset, 1.0 - frac, max_err, not args.no_vel_couple)
        time.sleep(period)


def build_parser():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--motors", type=_hex_int, nargs=2, default=[0x01, 0x04],
                   metavar=("ID_A", "ID_B"),
                   help="The two CAN IDs to couple (default: 0x01 0x04)")
    p.add_argument("--mirror", action="store_true",
                   help="Couple in opposite directions, for motors mounted "
                        "facing each other (p1 = -p2 + offset)")
    p.add_argument("--offset", type=float, default=None, metavar="DEG",
                   help="Equilibrium separation to converge on. Default: hold "
                        "whatever separation the motors are at on startup, so "
                        "engaging the loop moves nothing.")
    p.add_argument("--leader", type=_hex_int, default=None, metavar="ID",
                   help="Treat this motor as the leader: the one driven by hand. "
                        "It uses its model's bilateral.leader gains (same spring, "
                        "much less damping) so it feels light to move. Omit for a "
                        "symmetric pair.")
    p.add_argument("--leader-scale", type=float, default=1.0, metavar="F",
                   dest="leader_scale",
                   help="Multiply the leader's kp and kd by F, for tuning the "
                        "feel live without editing params.yaml (default: 1.0)")
    p.add_argument("--kp1", type=float, default=None,
                   help="Coupling stiffness for the first motor. Overrides the "
                        "role gains and --leader-scale outright.")
    p.add_argument("--kd1", type=float, default=None,
                   help="Coupling damping for the first motor")
    p.add_argument("--kp2", type=float, default=None,
                   help="Coupling stiffness for the second motor")
    p.add_argument("--kd2", type=float, default=None,
                   help="Coupling damping for the second motor")
    p.add_argument("--max-error", type=float, default=20.0, metavar="DEG",
                   dest="max_error",
                   help="Saturate the spring past this much coupling error, "
                        "capping the reaction torque at kp*max_error "
                        "(default: 20)")
    p.add_argument("--engage", type=float, default=1.0, metavar="SECS",
                   help="Ramp the coupling in over this long (default: 1.0)")
    p.add_argument("--converge", type=float, default=2.0, metavar="SECS",
                   help="Ramp to an explicit --offset over this long "
                        "(default: 2.0)")
    p.add_argument("--release", type=float, default=0.5, metavar="SECS",
                   help="Ramp the coupling back out on exit (default: 0.5)")
    p.add_argument("--rate", type=float, default=200.0, metavar="HZ",
                   help="Target control loop rate (default: 200)")
    p.add_argument("--vel-filter", type=float, default=0.3, metavar="ALPHA",
                   dest="vel_filter",
                   help="One-pole filter on the coupled velocity, 0-1; lower "
                        "is smoother and laggier (default: 0.3)")
    p.add_argument("--no-vel-couple", action="store_true", dest="no_vel_couple",
                   help="Damp against each motor's own velocity instead of the "
                        "pair's relative velocity. Falls back to a stiffer, "
                        "more viscous feel; try it if the pair buzzes.")
    p.add_argument("--params", default=None, metavar="PATH",
                   help="Tuning params YAML (default: bundled params.yaml)")
    p.add_argument("--channel", default="vcan0")
    p.add_argument("--interface", default="socketcan")
    return p


def main():
    args = build_parser().parse_args()
    id_a, id_b = args.motors
    if id_a == id_b:
        sys.exit("--motors needs two different CAN IDs")
    if args.rate <= 0:
        sys.exit("--rate must be positive")
    if args.max_error <= 0:
        sys.exit("--max-error must be positive (it caps the coupling torque)")
    if args.leader is not None and args.leader not in (id_a, id_b):
        sys.exit(f"--leader 0x{args.leader:02X} is not one of the coupled motors "
                 f"(0x{id_a:02X}, 0x{id_b:02X})")
    if args.leader_scale <= 0:
        sys.exit("--leader-scale must be positive")
    s       = -1.0 if args.mirror else 1.0
    period  = 1.0 / args.rate
    max_err = math.radians(args.max_error)

    bus = MotorBus(channel=args.channel, interface=args.interface,
                   params_path=args.params)
    entered = []
    # Bound up front so the handlers below are safe if we fail during setup,
    # before the pair exists.
    a = b = None
    offset = 0.0
    try:
        # Sets CTRL_MODE and reads each motor's own PMAX/VMAX/TMAX. Must happen
        # before enter_motor_mode: register access needs the motor disabled, and
        # the MIT scaling is per-motor, so the limits have to be known before the
        # first frame is encoded.
        bus.init_motors([id_a, id_b], MotorBus.MODE_MIT)

        sides = []
        for mid, kp, kd in ((id_a, args.kp1, args.kd1),
                            (id_b, args.kp2, args.kd2)):
            params = bus.params_for(mid)     # resolved by pinned/detected model
            side_kp, side_kd = _gains_for(params, mid == args.leader,
                                          args.leader_scale, kp, kd)
            sides.append(_Side(bus, mid, side_kp, side_kd,
                               _clamp(args.vel_filter, 0.0, 1.0)))
        a, b = sides

        print("\nCoupling:")
        for side in sides:
            t_max = bus.limits_for(side.id).t_max
            peak  = side.kp * max_err
            note  = "  (clipped by TMAX)" if peak > t_max else ""
            role  = "leader  " if side.id == args.leader else \
                    ("follower" if args.leader is not None else "        ")
            print(f"  0x{side.id:02X} {role}  kp={side.kp:g} kd={side.kd:g}  "
                  f"peak {peak:.2f} Nm at {args.max_error:g}° error "
                  f"(TMAX {t_max:g} Nm){note}")

        for side in sides:
            pos = bus.enter_motor_mode(side.id)
            if pos is None:
                raise RuntimeError(f"Motor 0x{side.id:02X} did not respond")
            entered.append(side.id)
            side.prime(pos)

        # Equilibrium is the separation they are already at, so engaging the
        # coupling produces no motion -- you test it by disturbing it.
        offset0 = a.pos - s * b.pos
        offset_target = None if args.offset is None else math.radians(args.offset)
        print(f"\nEquilibrium offset {math.degrees(offset0):+.2f}°"
              + ("" if offset_target is None
                 else f" -> {args.offset:+.2f}° over {args.converge:g}s")
              + f"{'  (mirrored)' if args.mirror else ''}")
        print("Coupled. Backdrive either motor by hand. Ctrl+C to stop.\n")

        offset    = offset0
        t0        = time.monotonic()
        next_tick = t0
        last_status, ticks = t0, 0

        while True:
            now = time.monotonic()
            kp_scale, offset = _schedule(now - t0, offset0, offset_target,
                                         args.engage, args.converge)
            _tick(a, b, s, offset, kp_scale, max_err, not args.no_vel_couple)
            _check_alive(a, b, time.monotonic())

            ticks += 1
            if now - last_status >= STATUS_PERIOD:
                _status(a, b, s, offset, ticks / (now - last_status), args.leader)
                last_status, ticks = now, 0

            # Both motors get a frame every tick, so the MIT comms-loss watchdog
            # is fed from this single thread. sequence.py reaches the same end by
            # a different route -- it feeds its idle motors through move_*'s
            # on_tick. Either way it must stay on one thread: a second thread
            # sharing a MotorBus eats the first one's feedback (_send_raw).
            next_tick += period
            sleep = next_tick - time.monotonic()
            if sleep > 0:
                time.sleep(sleep)
            else:
                next_tick = time.monotonic()   # fell behind; resync

    except KeyboardInterrupt:
        print("\nReleasing...")
        if a and b:
            _release(a, b, s, offset, args, period)
    except StaleFeedback as exc:      # subclass of RuntimeError; must come first
        print(f"\n!! {exc} -- releasing.")
        if a and b:
            _release(a, b, s, offset, args, period)
    except RuntimeError as exc:
        print(f"\n!! {exc}")
    finally:
        for mid in entered:
            bus.exit_motor_mode(mid)
        bus.shutdown()


if __name__ == "__main__":
    main()
