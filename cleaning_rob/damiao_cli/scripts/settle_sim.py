#!/usr/bin/env python3
"""Hardware-free simulation of the end-of-move settle (motor.py MotorBus._settle).

Drives the REAL _vel_move / _settle / _settle_release code against a simulated
joint, so this is a regression check on the shipped logic rather than a
re-implementation of it. Use it to re-tune `settle:` in params.yaml without
touching the hardware, and to confirm the anti-windup guard still protects a
jammed joint after any change.

Why the settle exists: tau = kp*(p_des - p) + kd*(v_des - v) makes torque only
from error, so holding against a load tau_L REQUIRES a standing error of
tau_L/kp. A loaded joint parks below its target and stays there however the
trajectory is shaped -- the S-curve profile is already exact at its endpoint.
The settle walks a feedforward torque up until it carries the load, which puts
the equilibrium on the target itself.

    python scripts/settle_sim.py
    python scripts/settle_sim.py --model DM4310 --loads 0.05 0.1 0.3
    python scripts/settle_sim.py --ki 150 --ff-max 10
"""

import argparse
import math
import os
import sys
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:                                  # pyusb is only needed by MotorBus.__init__,
    import usb.core                   # which this simulation never calls.
except ImportError:                   # pragma: no cover - convenience for dev boxes
    _usb, _core = types.ModuleType("usb"), types.ModuleType("usb.core")
    _core.find = lambda **kw: None
    _usb.core = _core
    sys.modules["usb"], sys.modules["usb.core"] = _usb, _core

import motor as motor_mod
from motor import Feedback, MotorBus, MotorLimits, TuningConfig, vendor_limits_for

MOTOR_ID = 0x01
TICK = 0.005


class _Clock:
    """Virtual clock. The move and settle loops are written against wall time and
    time.sleep(); driving them from here makes the simulation deterministic and
    roughly 300x faster than real time."""

    def __init__(self): self.t = 0.0
    def monotonic(self): return self.t
    def sleep(self, dt): self.t += dt


class SimBus(MotorBus):
    """MotorBus with the CAN layer replaced by a rigid joint under constant load.

    Everything above send_command -- the profile, the gain ramp, the stall
    guards, the settle and its anti-windup -- is the real implementation.
    """

    def __init__(self, clock, model, params_path=None,
                 inertia=0.05, damping=0.4, load=0.0, hard_stop=None):
        self.config = TuningConfig.from_yaml(params_path or MotorBus.DEFAULT_PARAMS_PATH)
        limits = (self.config.limits.get(model) or vendor_limits_for(model)
                  or MotorLimits())
        self._limits = {MOTOR_ID: limits}
        self._models = {MOTOR_ID: model}
        self._guessed = {MOTOR_ID: False}
        self._warned, self._warned_gains, self._last_status = set(), set(), {}
        self._load_ff = {}
        self.clock, self.J, self.b, self.load = clock, inertia, damping, load
        self.hard_stop, self.t_max = hard_stop, limits.t_max
        self.p = self.v = 0.0
        self._last_t = 0.0
        self.peak_torque = 0.0

    def send_command(self, motor_id, pos, vel=0.0, kp=None, kd=None, torque=0.0,
                     recv_timeout=None):
        params = self.params_for(motor_id)
        if kp is None: kp = params.kp_hold
        if kd is None: kd = params.kd_hold
        tau = kp * (pos - self.p) + kd * (vel - self.v) + torque
        tau = max(-self.t_max, min(self.t_max, tau))
        self.peak_torque = max(self.peak_torque, abs(tau))
        dt, self._last_t = max(self.clock.t - self._last_t, TICK), self.clock.t
        for _ in range(20):                       # substep for numerical stability
            h = dt / 20
            self.v += (tau - self.load - self.b * self.v) / self.J * h
            self.p += self.v * h
            if self.hard_stop is not None and self.p > self.hard_stop:
                self.p, self.v = self.hard_stop, 0.0
        return Feedback(motor_id=motor_id, status=1, pos=self.p, vel=self.v,
                        torque=tau, t_mos=30.0, t_rotor=30.0)

    def steady_torque(self, setpoint):
        """Torque the motor would hold at, at rest, for this setpoint."""
        p = self.params_for(MOTOR_ID)
        return p.kp_hold * (setpoint - self.p) + self.load_ff(MOTOR_ID)


def run(model, load, target_deg, hard_stop_deg=None, hold_secs=2.0, params=None):
    clock = _Clock()
    real_time, motor_mod.time = motor_mod.time, clock
    try:
        bus = SimBus(clock, model, params_path=params, load=load,
                     hard_stop=None if hard_stop_deg is None else math.radians(hard_stop_deg))
        setpoint = bus.move_by_offset(MOTOR_ID, math.radians(target_deg), start_pos=0.0,
                                      mode=MotorBus.MODE_MIT)
        ff = bus.load_ff(MOTOR_ID)
        for _ in range(int(hold_secs / TICK)):    # sequence.py's hold_all()
            bus.send_command(MOTOR_ID, setpoint, vel=0.0, torque=ff)
            clock.sleep(TICK)
        return dict(final=math.degrees(bus.p), err=math.degrees(bus.p) - target_deg,
                    ff=ff, peak=bus.peak_torque, secs=clock.t,
                    steady=bus.steady_torque(setpoint))
    finally:
        motor_mod.time = real_time


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="DM4340", help="params.yaml gain block (default: DM4340)")
    ap.add_argument("--loads", type=float, nargs="+",
                    default=[0.0, 0.15, 0.30, 0.60, 1.20, 2.50, 5.00],
                    help="static load torques to sweep, Nm")
    ap.add_argument("--degrees", type=float, default=90.0, help="move size (default: 90)")
    ap.add_argument("--hard-stop", type=float, default=45.0, metavar="DEG",
                    help="jam the joint here for the safety case (default: 45)")
    ap.add_argument("--tolerance", type=float, default=0.5, metavar="DEG",
                    help="max acceptable final error under load (default: 0.5)")
    ap.add_argument("--params", default=None, help="params YAML (default: bundled)")
    args = ap.parse_args()

    print(f"\nModel {args.model}, {args.degrees:g} deg move, settle from params.yaml\n")
    print(f"{'load':>7} {'final':>10} {'error':>9} {'tau_ff':>8} {'peak tau':>9} {'sim t':>7}")
    print("-" * 56)
    bad = []
    for load in args.loads:
        r = run(args.model, load, args.degrees, params=args.params)
        flag = "" if abs(r["err"]) <= args.tolerance else "  <-- OVER TOLERANCE"
        if flag: bad.append((load, r["err"]))
        print(f"{load:7.2f} {r['final']:9.3f}° {r['err']:8.3f}° {r['ff']:7.2f}N "
              f"{r['peak']:8.2f}N {r['secs']:6.2f}s{flag}")

    print(f"\nHard stop at {args.hard_stop:g} deg, commanded to {args.degrees:g} deg "
          f"-- must relax, not push:\n")
    h = run(args.model, 0.30, args.degrees, hard_stop_deg=args.hard_stop, params=args.params)
    print(f"  rests at {h['final']:.2f}°   tau_ff kept {h['ff']:.2f} Nm   "
          f"peak {h['peak']:.2f} Nm   steady {h['steady']:.2f} Nm")
    safe = abs(h["ff"]) < 1e-6 and abs(h["steady"]) < 3.0
    print(f"  -> {'SAFE: gave up and relaxed' if safe else 'UNSAFE: still loading the stop'}")

    ok = not bad and safe
    print(f"\n{'PASS' if ok else 'FAIL'}"
          + ("" if ok else f"  ({len(bad)} load(s) over tolerance)" if bad else "  (hard stop unsafe)"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
