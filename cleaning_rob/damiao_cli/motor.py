"""
MIT-mode motor library for Damiao DM-series over SocketCAN.

Control law (MIT): τ = KP*(p_des - p) + KD*(v_des - v) + τ_ff
Register protocol (0x7FF broadcast) is used for set_id, control mode, and for
reading each motor's MIT full-scale limits.
"""

import os
import struct
import math
import time
import threading
from dataclasses import dataclass

import usb.core
import can
import yaml

# KP/KD command ranges. Unlike the position/velocity/torque limits below, these
# are fixed by the MIT frame format and are identical across every DM model.
_KP_MIN, _KP_MAX = 0.0, 500.0
_KD_MIN, _KD_MAX = 0.0,   5.0

# Feedback wait for the 200 Hz streaming loops. These loops must keep the frame
# rate up more than they need every reply: blocking long enough to catch a
# dropped one stalls the command stream, the motor holds a stale setpoint while
# the reference trajectory keeps advancing, and the next frame lands as a step
# change -- a torque kick that shows up as buzzing. Missing a sample is cheaper.
_STREAM_RECV_TIMEOUT = 0.005


@dataclass(frozen=True)
class MotorLimits:
    """Full-scale ranges used to (de)quantize the MIT frame, per motor.

    The MIT frame carries position/velocity/torque as fixed-point fractions of
    each motor's configured full scale, so host and firmware must agree on the
    scale or every command is silently mis-scaled. These are NOT constants of
    the protocol -- they live in the motor's PMAX/VMAX/TMAX registers (RID
    21/22/23) and differ per model (and per unit, if reconfigured in Damiao's
    Debug Assistant). Read them off the motor with MotorBus.read_limits().
    """
    p_max: float = 12.5   # rad
    v_max: float = 30.0   # rad/s
    t_max: float = 10.0   # Nm


# Vendor Limit_Param table (Damiao DM_CAN.py / damiao.h), [PMAX, VMAX, TMAX].
# Used to name a motor from the limits it reports, which in turn selects its
# gain block in params.yaml -- a 10:1 and a 40:1 gearbox need very different
# gains for the same physical behaviour.
#
# A LIST of (name, limits), not a dict, because one model name legitimately owns
# several rows: the 24 V and 48 V builds of the same gearbox report different
# VMAX but want the same gains. Keyed by name these collapsed silently to the
# last row -- Python keeps only the final duplicate -- so a 24 V DM4310 (VMAX 30)
# matched nothing, identify() returned None, and the motor quietly ran on the
# `defaults` gains instead of its tuned block.
VENDOR_LIMITS = [
    ("DM4310",   MotorLimits(12.5,  30.0,  10.0)),   # 24 V
    ("DM4310",   MotorLimits(12.5,  50.0,  10.0)),   # 48 V
    ("DM4340",   MotorLimits(12.5,   8.0,  28.0)),   # 24 V
    ("DM4340",   MotorLimits(12.5,  10.0,  28.0)),   # 48 V
    ("DM6006",   MotorLimits(12.5,  45.0,  20.0)),
    ("DM8006",   MotorLimits(12.5,  45.0,  40.0)),
    ("DM8009",   MotorLimits(12.5,  45.0,  54.0)),
    ("DM10010L", MotorLimits(12.5,  25.0, 200.0)),
    ("DM10010",  MotorLimits(12.5,  20.0, 200.0)),
    ("DMH3510",  MotorLimits(12.5, 280.0,   1.0)),
    ("DMH6215",  MotorLimits(12.5,  45.0,  10.0)),
    ("DMG6220",  MotorLimits(12.5,  45.0,  10.0)),
]


def vendor_limits_for(model):
    """First vendor row named `model`, or None.

    Several rows can share a name (the voltage variants of one gearbox). This is
    only a fallback for when the motor's own registers can't be read, so the
    first row is the best guess on offer; give a variant its real numbers under
    `models.<NAME>.limits` in params.yaml when that guess isn't good enough.
    """
    for name, limits in VENDOR_LIMITS:
        if name == model:
            return limits
    return None

# Feedback status nibble (data[0] >> 4). 0/1 are normal; >= 8 is a fault that
# drops the motor out of enable mode.
ERR_CODES = {
    0x0: "disabled",
    0x1: "enabled",
    0x8: "overvoltage",
    0x9: "undervoltage",
    0xA: "overcurrent",
    0xB: "MOSFET overheat",
    0xC: "motor coil overheat",
    0xD: "communication loss",
    0xE: "overload",
}

# Motor-mode magic bytes
_CMD_ENTER = bytes([0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFC])
_CMD_EXIT  = bytes([0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFD])
_CMD_ZERO  = bytes([0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFE])

# Register protocol opcodes (byte 2 of the 0x7FF frame).
_OP_READ  = 0x33
_OP_WRITE = 0x55
_OP_STORE = 0xAA

# Register IDs for set_id
_REG_MST_ID = 7
_REG_ESC_ID = 8

# Per-motor MIT full-scale limits.
_REG_PMAX = 21
_REG_VMAX = 22
_REG_TMAX = 23

# CTRL_MODE register (RID 10): selects which command frame the motor obeys.
# Writing it takes effect immediately (no power cycle); flash save only persists
# it across power cycles.
_REG_CTRL_MODE = 10
MODE_MIT     = 1
MODE_POS_VEL = 2
MODE_VEL     = 3

# Raw moves strip the host-side control stack, all of which lives on the MIT
# path; there is nothing to strip in POS_VEL, so asking for it is a mistake
# worth naming rather than quietly ignoring.
_RAW_POSVEL_ERR = (
    "raw applies to MIT mode only: in POS_VEL the motor's firmware runs the "
    "position loop, so there are no host-side helpers to strip."
)


def _reg_is_int(rid):
    """True if RID holds a uint32, False if it holds a float32.

    Mirrors is_in_ranges() in Damiao's DM_CAN.py: IDs/versions/baud are integers,
    everything else (including PMAX/VMAX/TMAX) is IEEE-754 float.
    """
    return (7 <= rid <= 10) or (13 <= rid <= 16) or (35 <= rid <= 36)


def _float_to_uint(x, x_min, x_max, bits):
    x = max(min(x, x_max), x_min)
    return int((x - x_min) / (x_max - x_min) * ((1 << bits) - 1))


def _uint_to_float(raw, x_min, x_max, bits):
    return raw / float((1 << bits) - 1) * (x_max - x_min) + x_min


@dataclass
class Feedback:
    """One decoded feedback frame."""
    motor_id: int
    status:   int    # data[0] >> 4; see ERR_CODES
    pos:      float  # rad
    vel:      float  # rad/s
    torque:   float  # Nm
    t_mos:    float  # drive MOSFET temperature, degC
    t_rotor:  float  # rotor coil temperature, degC

    @property
    def error(self):
        """Human-readable fault name, or None when enabled/disabled normally."""
        if self.status < 0x8:
            return None
        return ERR_CODES.get(self.status, f"unknown status 0x{self.status:X}")


def encode_mit_command(pos, vel, kp, kd, torque, limits):
    p      = _float_to_uint(pos,    -limits.p_max, limits.p_max, 16)
    v      = _float_to_uint(vel,    -limits.v_max, limits.v_max, 12)
    kp_int = _float_to_uint(kp,     _KP_MIN,       _KP_MAX,      12)
    kd_int = _float_to_uint(kd,     _KD_MIN,       _KD_MAX,      12)
    t      = _float_to_uint(torque, -limits.t_max, limits.t_max, 12)

    data = bytearray(8)
    data[0] = (p >> 8) & 0xFF
    data[1] = p & 0xFF
    data[2] = (v >> 4) & 0xFF
    data[3] = ((v & 0xF) << 4) | ((kp_int >> 8) & 0xF)
    data[4] = kp_int & 0xFF
    data[5] = (kd_int >> 4) & 0xFF
    data[6] = ((kd_int & 0xF) << 4) | ((t >> 8) & 0xF)
    data[7] = t & 0xFF
    return bytes(data)


def decode_response(data, limits):
    # data[0] packs the feedback status in the high nibble and the source
    # motor ID in the low nibble: (status << 4) | id.
    status     = data[0] >> 4
    motor_id   = data[0] & 0x0F
    pos_raw    = (data[1] << 8) | data[2]
    vel_raw    = (data[3] << 4) | (data[4] >> 4)
    torque_raw = ((data[4] & 0xF) << 8) | data[5]
    return Feedback(
        motor_id = motor_id,
        status   = status,
        pos      = _uint_to_float(pos_raw,    -limits.p_max, limits.p_max, 16),
        vel      = _uint_to_float(vel_raw,    -limits.v_max, limits.v_max, 12),
        torque   = _uint_to_float(torque_raw, -limits.t_max, limits.t_max, 12),
        t_mos    = float(data[6]) if len(data) > 6 else 0.0,
        t_rotor  = float(data[7]) if len(data) > 7 else 0.0,
    )


@dataclass
class _SCurveProfile:
    """Velocity-limited S-curve trapezoidal motion profile from `start` to `target`.

    The velocity magnitude smoothly ramps 0->V_pk (smoothstep corner, so
    acceleration is continuous), cruises at V_pk, then ramps back to 0. Only the
    cruise duration depends on distance -- the accel/decel corners have a fixed
    shape -- so the motion looks identical for short and long moves. This makes
    the trajectory distance-independent in shape.

    Parametrized by elapsed time `t` (seconds) via `sample(t)`. Pure/standalone:
    no bus access, so it can be unit-tested without hardware.

    V is the cruise velocity (rad/s), A the accel/decel limit (rad/s^2); ramp
    time T_a = V/A. If the distance is too short to reach V, it falls back to a
    triangular profile with a lower peak (V_pk = sqrt(D*A))."""
    start:  float
    target: float
    V:      float  # cruise velocity (rad/s)
    A:      float  # accel/decel limit (rad/s^2)

    def __post_init__(self):
        dist = self.target - self.start
        self.D    = abs(dist)
        self.sign = math.copysign(1.0, dist) if dist != 0.0 else 1.0
        if self.D == 0.0 or self.V <= 0.0 or self.A <= 0.0:
            # Degenerate move: nothing to do, snap to target.
            self.V_pk = 0.0
            self.T_a = self.T_c = self.T = 0.0
            return
        D_full = self.V * self.V / self.A      # distance needing full accel+decel
        if self.D >= D_full:
            self.V_pk = self.V
            self.T_a  = self.V / self.A
            self.T_c  = (self.D - D_full) / self.V
        else:
            # Triangular: never reaches cruise velocity.
            self.V_pk = math.sqrt(self.D * self.A)
            self.T_a  = self.V_pk / self.A
            self.T_c  = 0.0
        self.T = 2.0 * self.T_a + self.T_c

    def sample(self, t):
        """Return (p_ref, v_ref, done) at elapsed time `t`. p_ref is the position
        setpoint (rad), v_ref the velocity feedforward (rad/s, signed)."""
        if t >= self.T or self.T_a <= 0.0:
            return self.target, 0.0, True
        Ta, Tc, Vp = self.T_a, self.T_c, self.V_pk
        d_ramp = Vp * Ta / 2.0
        if t < Ta:                              # accel corner
            u = t / Ta
            x = Vp * Ta * (u**3 - 0.5 * u**4)
            v = Vp * (3.0 * u**2 - 2.0 * u**3)
        elif t < Ta + Tc:                       # cruise
            x = d_ramp + Vp * (t - Ta)
            v = Vp
        else:                                   # decel corner (mirror of accel)
            a = (t - Ta - Tc) / Ta              # 0 -> 1 across the decel phase
            w = 1.0 - a
            x = d_ramp + Vp * Tc + Vp * Ta * (0.5 - (w**3 - 0.5 * w**4))
            v = Vp * (3.0 * w**2 - 2.0 * w**3)
        p_ref = self.start + self.sign * x
        v_ref = self.sign * v
        return p_ref, v_ref, False


@dataclass
class MotorParams:
    """Control-tuning values for one motor model. Thresholds are stored in
    radians. Loaded via TuningConfig, which layers per-model overrides on top of
    a shared defaults block."""
    kp_move:      float = 10.0   # position stiffness during moves
    kd_move:      float = 3.5    # velocity damping during moves (protocol max: 5.0)
    torque_ff:    float = 0.0    # Nm feedforward during moves
    velocity:     float = 32.0   # rad/s default cruise speed / pos_vel limit
    accel:        float = math.pi  # rad/s^2 accel/decel limit for the S-curve profile
    kp_hold:      float = 20.0
    kd_hold:      float = 3.5
    # Virtual spring/damper used when two motors are coupled to each other
    # (scripts/bilateral.py). Separate from move/hold because the target is
    # different: coupling wants to stay backdrivable by hand, where a move wants
    # to win against the load.
    kp_bilateral: float = 5.0
    kd_bilateral: float = 2.0
    # Same coupling, but for the motor driving the pair by hand. Typically the
    # same kp (force reflection is unchanged) with much less kd: the damping
    # term works against the follower's velocity lag, so it reads as viscous
    # drag on whoever is doing the moving.
    kp_leader:    float = 5.0
    kd_leader:    float = 0.6
    # End-of-move settle. The MIT law makes torque only from error, so holding
    # against a load torque tau_L REQUIRES a standing error of tau_L/kp -- a
    # loaded joint parks below its target and no trajectory shaping changes that.
    # The settle walks a feedforward torque up until it carries the load, which
    # puts the equilibrium on the target itself. See MotorBus._settle.
    settle_enabled:    bool  = True
    settle_ki:         float = 50.0   # Nm per rad-second
    settle_ff_max:     float = 3.0    # Nm ceiling on the learned feedforward
    settle_max_err:    float = math.radians(5.0)   # setpoint clamp vs measured pos
    settle_tol:        float = math.radians(0.1)   # settle arrival tolerance
    settle_vel:        float = 0.05   # rad/s below which the joint is "at rest"
    settle_max_secs:   float = 1.5    # cap on the settle window
    settle_stall_secs: float = 0.3    # give up this long after ff hits its ceiling
    # Move give-up guards: a joint that can't reach target (mechanical limit,
    # gravity/load steady-state error) must not spin a move loop forever.
    stop_thresh:  float = math.radians(1.0)  # arrival tolerance
    stall_thresh: float = math.radians(0.2)  # min progress to count as "still moving"
    stall_secs:   float = 0.4    # no progress for this long -> accept and continue
    move_timeout: float = 8.0    # hard cap on a single move (s)

    def merged(self, block):
        """Return a copy with the move/hold/bilateral/settle/guards keys in
        `block` applied over self. Used to layer a per-model override on top of
        the defaults, so a model only has to list what it actually changes."""
        move   = block.get("move", {})
        hold   = block.get("hold", {})
        bilat  = block.get("bilateral", {})
        lead   = bilat.get("leader", {})
        settle = block.get("settle", {})
        guards = block.get("guards", {})
        kp_bi  = bilat.get("kp", self.kp_bilateral)
        kd_bi  = bilat.get("kd", self.kd_bilateral)
        # A model that sets `bilateral:` but no `leader:` falls back to *its own*
        # coupling gains rather than the defaults' leader gains -- inheriting a
        # leader stiffness tuned for a different gearbox is exactly the failure
        # this layering is meant to prevent. Only a block with no `bilateral:`
        # at all inherits the defaults' leader values.
        kp_ld  = lead.get("kp", kp_bi if bilat else self.kp_leader)
        kd_ld  = lead.get("kd", kd_bi if bilat else self.kd_leader)
        return MotorParams(
            kp_move      = move.get("kp",        self.kp_move),
            kd_move      = move.get("kd",        self.kd_move),
            torque_ff    = move.get("torque_ff", self.torque_ff),
            velocity     = move.get("velocity",  self.velocity),
            accel        = move.get("accel",     self.accel),
            kp_hold      = hold.get("kp",        self.kp_hold),
            kd_hold      = hold.get("kd",        self.kd_hold),
            kp_bilateral = kp_bi,
            kd_bilateral = kd_bi,
            kp_leader    = kp_ld,
            kd_leader    = kd_ld,
            settle_enabled    = settle.get("enabled",    self.settle_enabled),
            settle_ki         = settle.get("ki",         self.settle_ki),
            settle_ff_max     = settle.get("ff_max",     self.settle_ff_max),
            settle_max_err    = math.radians(settle.get("max_err_deg",
                                             math.degrees(self.settle_max_err))),
            settle_tol        = math.radians(settle.get("tol_deg",
                                             math.degrees(self.settle_tol))),
            settle_vel        = settle.get("vel_thresh", self.settle_vel),
            settle_max_secs   = settle.get("max_secs",   self.settle_max_secs),
            settle_stall_secs = settle.get("stall_secs", self.settle_stall_secs),
            stop_thresh  = math.radians(guards.get("stop_thresh_deg",
                                                   math.degrees(self.stop_thresh))),
            stall_thresh = math.radians(guards.get("stall_thresh_deg",
                                                   math.degrees(self.stall_thresh))),
            stall_secs   = guards.get("stall_secs",   self.stall_secs),
            move_timeout = guards.get("move_timeout", self.move_timeout),
        )


@dataclass
class TuningConfig:
    """params.yaml, parsed. Gains are per motor *model* because they depend on
    the gearbox: reflected inertia scales with the reduction ratio squared, so a
    10:1 J4310 and a 40:1 J4340 need very different kp/kd for the same feel."""
    defaults:   MotorParams
    models:     dict   # {model_name: MotorParams}
    limits:     dict   # {model_name: MotorLimits}  -- fallback when auto-read fails
    pinned:     dict   # {motor_id: model_name}     -- manual override
    auto_read:  bool = True

    @classmethod
    def from_yaml(cls, path):
        with open(path) as f:
            data = yaml.safe_load(f) or {}

        # A file with no `defaults:`/`models:` is the old flat layout; treat its
        # top level as the defaults block so custom --params files keep working.
        base_block = data.get("defaults")
        if base_block is None:
            base_block = data if ("move" in data or "hold" in data) else {}
        defaults = MotorParams().merged(base_block)

        models, limits = {}, {}
        for name, block in (data.get("models") or {}).items():
            block = block or {}
            models[name] = defaults.merged(block)
            lim = block.get("limits")
            if lim:
                limits[name] = MotorLimits(
                    p_max = lim.get("p_max", 12.5),
                    v_max = lim.get("v_max", 30.0),
                    t_max = lim.get("t_max", 10.0),
                )
            elif vendor_limits_for(name):
                limits[name] = vendor_limits_for(name)

        pinned = {int(k, 0) if isinstance(k, str) else int(k): v
                  for k, v in (data.get("motors") or {}).items()}

        return cls(
            defaults  = defaults,
            models    = models,
            limits    = limits,
            pinned    = pinned,
            auto_read = (data.get("limits") or {}).get("auto_read", True),
        )


class MotorBus:
    # Control mode flag: move_to_pos/move_by_offset use this native mode unless a
    # different mode is passed. Switch the default by changing DEFAULT_MODE.
    MODE_MIT     = MODE_MIT
    MODE_POS_VEL = MODE_POS_VEL
    MODE_VEL     = MODE_VEL
    DEFAULT_MODE = MODE_MIT
    _REG_CTRL_MODE = _REG_CTRL_MODE

    # Tuning gains/guards live in TuningConfig (loaded from params.yaml on init).
    DEFAULT_PARAMS_PATH = os.path.join(os.path.dirname(__file__), "params.yaml")

    def __init__(self, channel='vcan0', interface='socketcan',
                 usb_vendor=0x04d8, usb_product=0x0053,
                 params=None, params_path=None):
        if params is None:
            path = params_path or self.DEFAULT_PARAMS_PATH
            params = (TuningConfig.from_yaml(path) if os.path.exists(path)
                      else TuningConfig(MotorParams(), {}, {}, {}))
        self.config = params
        # Resolved per motor ID, populated lazily (or eagerly by init_motors).
        self._limits  = {}  # {motor_id: MotorLimits}
        self._models  = {}  # {motor_id: model_name or None}
        self._guessed = {}  # {motor_id: True if limits are a fallback, not read}
        self._warned  = set()   # motor IDs already warned about guessed limits
        self._warned_gains = set()  # motor IDs already warned about a missing gain block
        self._last_status = {}  # {motor_id: status} for edge-triggered fault logs
        self._load_ff = {}  # {motor_id: Nm} static load learned by the last settle
        dev = usb.core.find(idVendor=usb_vendor, idProduct=usb_product)
        if dev is None:
            raise RuntimeError(f"USB-CAN adapter {usb_vendor:#06x}:{usb_product:#06x} not found")
        for iface in (0, 1):
            if dev.is_kernel_driver_active(iface):
                dev.detach_kernel_driver(iface)
        self._bus  = can.Bus(interface=interface, channel=channel)
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ #
    # Per-motor limits, model identification and tuning                    #
    # ------------------------------------------------------------------ #

    def read_limits(self, motor_id):
        """Read PMAX/VMAX/TMAX (RID 21/22/23) off the motor. Returns MotorLimits
        or None if the motor doesn't answer. Motor must NOT be in motor mode."""
        vals = []
        for rid in (_REG_PMAX, _REG_VMAX, _REG_TMAX):
            v = self._reg_read(motor_id, rid)
            if v is None:
                return None
            vals.append(v)
        return MotorLimits(*vals)

    @staticmethod
    def identify(limits):
        """Name the motor model matching `limits`, or None. PMAX is 12.5 on every
        model, so v_max/t_max are what actually discriminate."""
        for name, ref in VENDOR_LIMITS:
            if (math.isclose(limits.v_max, ref.v_max, rel_tol=1e-3) and
                    math.isclose(limits.t_max, ref.t_max, rel_tol=1e-3)):
                return name
        return None

    def resolve(self, motor_id, force=False, announce=True):
        """Determine and cache (limits, model_name) for a motor.

        The limits always come from the motor's own registers when they can be
        read -- they are per-unit and drive MIT scaling, so nothing may override
        them. The *model name* only selects a gain block, and there an explicit
        `motors:` pin in params.yaml wins: identify() matches on VMAX/TMAX, so a
        unit whose VMAX was reconfigured in Damiao's Debug Assistant lands on
        another variant's row and is named wrongly with full confidence. The pin
        is the human saying which gearbox this actually is.

        `announce` reports the fallback/disagreement immediately -- suppressed on
        the lazy path, where a motor that is simply absent would otherwise
        produce a warning per scanned ID; there the warning is deferred until the
        motor actually answers (_warn_if_guessed).
        """
        if not force and motor_id in self._limits:
            return self._limits[motor_id], self._models[motor_id]

        limits   = self.read_limits(motor_id) if self.config.auto_read else None
        detected = self.identify(limits) if limits else None
        pinned   = self.config.pinned.get(motor_id)
        # The pin applies whatever detection said, and regardless of `announce`
        # -- gating it on the print is why pinning silently did nothing on the
        # params_for()/limits_for() path.
        model    = pinned or detected
        guessed  = limits is None

        if guessed:
            if pinned and pinned in self.config.limits:
                limits = self.config.limits[pinned]
            else:
                limits = MotorLimits()
        elif announce and pinned and detected and pinned != detected:
            print(f"Motor 0x{motor_id:02X}: registers look like {detected}; using "
                  f"pinned {pinned} gains (MIT scaling still +/-{limits.v_max:g} "
                  f"rad/s / +/-{limits.t_max:g} Nm from the motor).")
        elif announce and detected is None:
            # Limits read fine but match no vendor row (reconfigured unit). The
            # scaling is still correct; only the gain-block lookup is affected.
            print(f"Motor 0x{motor_id:02X}: limits {limits.v_max:g} rad/s / "
                  f"{limits.t_max:g} Nm match no known model"
                  + (f"; using pinned {pinned} gains." if pinned else "; using default gains."))

        self._limits[motor_id]  = limits
        self._models[motor_id]  = model
        self._guessed[motor_id] = guessed
        if guessed and announce:
            self._warn_if_guessed(motor_id)
        return limits, model

    def _warn_if_guessed(self, motor_id):
        """Warn once that a motor's MIT scaling is a guess. Wrong limits silently
        mis-scale every velocity and torque on the wire, so this must not be
        quiet -- but it is only worth saying about a motor that actually exists."""
        if not self._guessed.get(motor_id) or motor_id in self._warned:
            return
        self._warned.add(motor_id)
        limits = self._limits[motor_id]
        model  = self._models[motor_id]
        if model:
            print(f"Motor 0x{motor_id:02X}: limits not readable; using pinned {model} "
                  f"(+/-{limits.v_max:g} rad/s, +/-{limits.t_max:g} Nm).")
        else:
            print(f"WARNING motor 0x{motor_id:02X}: could not read PMAX/VMAX/TMAX and "
                  f"no model pinned in params.yaml -- assuming +/-{limits.v_max:g} rad/s "
                  f"/ +/-{limits.t_max:g} Nm. Commands are mis-scaled if that is wrong.")

    def limits_for(self, motor_id):
        return self.resolve(motor_id, announce=False)[0]

    def params_for(self, motor_id):
        """Tuning for this motor, selected by its resolved model."""
        model = self.resolve(motor_id, announce=False)[1]
        if model in self.config.models:
            return self.config.models[model]
        # Falling through to defaults silently is how a whole tuning session can
        # go into a model block that nothing ever reads. Say it once per motor.
        if model and motor_id not in self._warned_gains:
            self._warned_gains.add(motor_id)
            d = self.config.defaults
            print(f"Motor 0x{motor_id:02X}: no `{model}` block in params.yaml -- using "
                  f"defaults (move kp={d.kp_move:g}, hold kp={d.kp_hold:g}, "
                  f"bilateral kp={d.kp_bilateral:g}). Add a block for {model}, or "
                  f"pin the right model for this ID under `motors:`.")
        return self.config.defaults

    # ------------------------------------------------------------------ #
    # Low-level                                                            #
    # ------------------------------------------------------------------ #

    def _report_fault(self, fb):
        """Print a fault the first time it appears. A faulted motor drops out of
        enable mode, so silently discarding this makes a dead motor look like a
        tuning problem. Edge-triggered: the send loops run at 200 Hz."""
        if self._last_status.get(fb.motor_id) == fb.status:
            return
        self._last_status[fb.motor_id] = fb.status
        if fb.error:
            print(f"  !! motor 0x{fb.motor_id:02X} fault: {fb.error} "
                  f"(T_mos={fb.t_mos:.0f}C T_rotor={fb.t_rotor:.0f}C)")

    def _send_raw(self, arb_id, data, recv_timeout=1.0, motor_id=None):
        """Send one frame and wait for that motor's reply.

        ONE THREAD PER BUS. This is a send-then-receive transaction over a single
        shared socket with no per-motor demultiplexing: frames from other motors
        are read and DISCARDED (see the ID-nibble test below), and a discarded
        frame is gone for good. Two threads sharing a MotorBus therefore eat each
        other's feedback -- each one's replies vanish into the other's discard
        loop, its move sees `resp is None` forever, and it concludes the motor
        never moved. To drive several motors at once, round-robin them from a
        single loop (scripts/bilateral.py) or use move_*'s `on_tick`
        (scripts/sequence.py).
        """
        if motor_id is None:
            motor_id = arb_id & 0xFF     # POS_VEL/VEL frames are 0x100/0x200 + id
        # Resolve before taking the lock: this may itself do bus I/O and
        # self._lock is not reentrant.
        limits = self.limits_for(motor_id)
        msg = can.Message(arbitration_id=arb_id, data=data, is_extended_id=False)
        with self._lock:
            self._bus.send(msg)
            deadline = time.monotonic() + recv_timeout
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                resp = self._bus.recv(timeout=remaining)
                if resp is None:
                    return None
                # Discard stale frames from other motors (concurrent threads
                # can leave responses in the kernel CAN buffer); match by the
                # source ID embedded in the feedback payload. data[0] is
                # (status << 4) | id, so compare the ID nibble only.
                if resp.data and (resp.data[0] & 0x0F) == (motor_id & 0x0F):
                    fb = decode_response(resp.data, limits)
                    # The motor is definitely present, so if its limits were a
                    # guess that is now worth saying out loud (prints once).
                    self._warn_if_guessed(motor_id)
                    self._report_fault(fb)
                    return fb

    def send_command(self, motor_id, pos, vel=0.0, kp=None, kd=None, torque=0.0,
                     recv_timeout=0.05):
        params = self.params_for(motor_id)
        if kp is None: kp = params.kp_hold
        if kd is None: kd = params.kd_hold
        data = encode_mit_command(pos=pos, vel=vel, kp=kp, kd=kd, torque=torque,
                                  limits=self.limits_for(motor_id))
        return self._send_raw(motor_id, data, recv_timeout=recv_timeout)

    def send_posvel_command(self, motor_id, pos, vel_limit):
        """Native position-velocity frame: motor firmware runs the position loop.
        Sent on 0x100 + id as float32 position + float32 velocity limit (LE).
        Feedback comes back in the standard format, keyed to the motor ID rather
        than the 0x100-offset arbitration ID."""
        data = struct.pack('<ff', pos, abs(vel_limit))
        return self._send_raw(0x100 + motor_id, data, recv_timeout=0.05,
                              motor_id=motor_id)

    # ------------------------------------------------------------------ #
    # Motor mode                                                           #
    # ------------------------------------------------------------------ #

    def enter_motor_mode(self, motor_id):
        resp = self._send_raw(motor_id, _CMD_ENTER)
        if resp:
            print(f"Motor 0x{motor_id:02X} active. pos={math.degrees(resp.pos):.2f} deg")
            return resp.pos
        print(f"Motor 0x{motor_id:02X} no response to enter_motor_mode!")
        return None

    def exit_motor_mode(self, motor_id):
        self._send_raw(motor_id, _CMD_EXIT)
        print(f"Motor 0x{motor_id:02X} exited motor mode.")

    # ------------------------------------------------------------------ #
    # Configuration commands (no motor mode required)                      #
    # ------------------------------------------------------------------ #

    def set_zero(self, motor_id):
        """Set current position as the zero reference. Persists across power cycles."""
        resp = self._send_raw(motor_id, _CMD_ZERO)
        if resp:
            print(f"Motor 0x{motor_id:02X} zero set.")
        else:
            print(f"Motor 0x{motor_id:02X} no response to set_zero.")

    def _reg_write(self, current_id, reg, value):
        """Write a 32-bit register via the 0x7FF broadcast (0x55 = write)."""
        id_l, id_h = current_id & 0xFF, (current_id >> 8) & 0xFF
        payload = (struct.pack('<I', int(value)) if _reg_is_int(reg)
                   else struct.pack('<f', float(value)))
        data = bytes([id_l, id_h, _OP_WRITE, reg]) + payload
        msg = can.Message(arbitration_id=0x7FF, data=data, is_extended_id=False)
        with self._lock:
            self._bus.send(msg)
            time.sleep(0.05)
            return self._bus.recv(timeout=0.2)

    def _reg_read(self, current_id, reg, timeout=0.2):
        """Read a register via the 0x7FF broadcast (0x33 = read).

        Returns the decoded value (int or float depending on the RID) or None if
        the motor doesn't answer. Motor must NOT be in motor mode.
        """
        id_l, id_h = current_id & 0xFF, (current_id >> 8) & 0xFF
        data = bytes([id_l, id_h, _OP_READ, reg, 0x00, 0x00, 0x00, 0x00])
        msg = can.Message(arbitration_id=0x7FF, data=data, is_extended_id=False)
        with self._lock:
            self._bus.send(msg)
            deadline = time.monotonic() + timeout
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                resp = self._bus.recv(timeout=remaining)
                if resp is None:
                    return None
                # The register reply shares the 0x7FF arbitration ID with every
                # other motor's, so match on the echoed motor ID and RID.
                d = resp.data
                if (d is None or len(d) < 8 or d[0] != id_l or d[1] != id_h
                        or d[3] != reg):
                    continue
                if d[2] not in (_OP_READ, _OP_WRITE):
                    continue
                raw = bytes(d[4:8])
                return (struct.unpack('<I', raw)[0] if _reg_is_int(reg)
                        else struct.unpack('<f', raw)[0])

    def _reg_store(self, current_id):
        """Persist register changes to flash via the 0x7FF broadcast (0xAA = store)."""
        id_l, id_h = current_id & 0xFF, (current_id >> 8) & 0xFF
        data = bytes([id_l, id_h, _OP_STORE, 0x01, 0x00, 0x00, 0x00, 0x00])
        msg = can.Message(arbitration_id=0x7FF, data=data, is_extended_id=False)
        with self._lock:
            self._bus.send(msg)
            time.sleep(0.1)
            return self._bus.recv(timeout=0.2)

    def set_control_mode(self, motor_id, mode, persist=False):
        """Set CTRL_MODE (MIT/POS_VEL/VEL). Takes effect immediately; persist=True
        also stores to flash so it survives a power cycle. Motor must NOT be in
        motor mode (register writes need it disabled)."""
        self._reg_write(motor_id, self._REG_CTRL_MODE, mode)
        if persist:
            self._reg_store(motor_id)

    def init_motors(self, motor_ids, mode=None):
        """Configure each motor's CTRL_MODE to `mode` (defaults to DEFAULT_MODE)
        and learn its MIT scaling limits.

        Call BEFORE enter_motor_mode: register access needs the motor disabled,
        and the limits must be known before the first frame is encoded or
        decoded. No flash save: the mode is set fresh on every run, so it is
        always correct and the flash isn't worn.
        """
        if mode is None:
            mode = self.DEFAULT_MODE
        for mid in motor_ids:
            self.set_control_mode(mid, mode)
            limits, model = self.resolve(mid, force=True)
            print(f"Motor 0x{mid:02X}: {model or 'unknown model'} "
                  f"(+/-{limits.p_max:g} rad, +/-{limits.v_max:g} rad/s, "
                  f"+/-{limits.t_max:g} Nm)")

    def set_id(self, motor_id, new_esc_id, new_mst_id=None):
        """
        Change the motor's CAN IDs via register writes (broadcast on 0x7FF).
        new_mst_id defaults to new_esc_id | 0x10 (Damiao convention).
        Motor must NOT be in MIT motor mode. Power-cycle after to apply.
        """
        if new_mst_id is None:
            new_mst_id = new_esc_id | 0x10

        print(f"Writing MST_ID = 0x{new_mst_id:02X}...")
        self._reg_write(motor_id, _REG_MST_ID, new_mst_id)

        print(f"Writing ESC_ID = 0x{new_esc_id:02X}...")
        self._reg_write(motor_id, _REG_ESC_ID, new_esc_id)
        # Motor now responds to new_esc_id from this point on

        print(f"Storing to flash (addressing 0x{new_esc_id:02X})...")
        self._reg_store(new_esc_id)

        print(f"Done. Power-cycle the motor to apply new ID 0x{new_esc_id:02X}.")

    def query(self, motor_id):
        """Enter motor mode, read state, exit. Returns a Feedback (pos in rad) or
        None. Callers wanting degrees should convert at the display layer."""
        resp = self._send_raw(motor_id, _CMD_ENTER)
        if resp is None:
            return None
        self._send_raw(motor_id, _CMD_EXIT, recv_timeout=0.2)
        return resp

    def scan(self, motor_ids=range(0x01, 0x08)):
        """Query every motor in motor_ids. Returns dict {id: Feedback or None}."""
        return {mid: self.query(mid) for mid in motor_ids}

    # ------------------------------------------------------------------ #
    # Motion                                                               #
    # ------------------------------------------------------------------ #

    def move_to_pos(self, motor_id, target_rad, start_pos, velocity=None,
                    mode=None, kp=None, kd=None, on_tick=None, raw=False):
        """Absolute position move. In POS_VEL mode the motor's firmware drives the
        position loop; in MIT mode the host runs the closed loop with gains from
        params (overridable via kp/kd). mode defaults to DEFAULT_MODE.

        `raw` commands the target itself instead of running the control stack: no
        S-curve profile, no ramp into the hold gains, no end-of-move settle (so
        `velocity` is unused and load_ff is left at zero). The stall/timeout
        guards stay, so a raw move still can't spin forever on a jammed joint.
        MIT only -- raw with any other mode is a ValueError.

        `on_tick` is called once per control tick, after this motor's frame is
        sent, and is how a caller keeps OTHER motors on the bus alive during a
        move -- see scripts/sequence.py. It runs on the calling thread on
        purpose: this bus is single-threaded (see _send_raw), so a second thread
        streaming the idle motors would eat this loop's feedback. Keep the
        callback short; every millisecond it spends is a millisecond this motor
        goes uncommanded.
        """
        if velocity is None:
            velocity = self.params_for(motor_id).velocity
        if mode is None:
            mode = self.DEFAULT_MODE
        if mode != MODE_MIT:
            if raw:
                raise ValueError(_RAW_POSVEL_ERR)
            return self._posvel_move(motor_id, start_pos, target_rad, velocity,
                                     on_tick=on_tick)
        return self._vel_move(motor_id, start_pos, target_rad, velocity, kp, kd,
                              on_tick=on_tick, raw=raw)

    def move_by_offset(self, motor_id, delta_rad, start_pos, velocity=None,
                       mode=None, kp=None, kd=None, on_tick=None, raw=False):
        """Relative move from start_pos by delta_rad. See move_to_pos for mode/gain,
        on_tick and raw semantics."""
        if velocity is None:
            velocity = self.params_for(motor_id).velocity
        if mode is None:
            mode = self.DEFAULT_MODE
        if mode != MODE_MIT:
            if raw:
                raise ValueError(_RAW_POSVEL_ERR)
            return self._posvel_move(motor_id, start_pos, start_pos + delta_rad,
                                     velocity, on_tick=on_tick)
        return self._vel_move(motor_id, start_pos, start_pos + delta_rad, velocity,
                              kp, kd, on_tick=on_tick, raw=raw)

    @staticmethod
    def _report_give_up(motor_id, reason, target, current, heard):
        """Explain why a move loop stopped before reaching target.

        `heard == 0` is the case worth separating out: no feedback arrived at
        all, so `current` is still the caller's start_pos and the motor's true
        position is simply unknown. Reporting that as a mechanical stall is how a
        stale position gets recorded and then actively enforced by the next hold
        -- the motor lurches off and is driven straight back. Name it for what it
        is instead.
        """
        if heard:
            print(f"  motor 0x{motor_id:02X} {reason} "
                  f"{math.degrees(abs(target - current)):.1f} deg short; continuing")
        else:
            print(f"  !! motor 0x{motor_id:02X} {reason} having received NO feedback "
                  f"frames -- its position is unknown, so the move is reporting the "
                  f"position it started from ({math.degrees(current):.1f} deg). Check "
                  f"the motor is powered and answering, and that nothing else is "
                  f"sharing this MotorBus from another thread.")

    def load_ff(self, motor_id):
        """Static load (Nm) the last settle learned for this motor.

        This is how much torque the joint needs just to stay put. Holding with it
        puts the equilibrium on the commanded position instead of tau_load/kp
        below it. Zero until a move has settled, and reset to zero whenever a
        settle gives up -- a joint we could not converge is one whose load we do
        not actually know.
        """
        return self._load_ff.get(motor_id, 0.0)

    def _settle_release(self, motor_id, pos, ff, params, on_tick=None):
        """Bleed the feedforward back out rather than cutting it dead.

        Dropping a loaded torque instantly lets the joint fall; decaying it
        leaves the motor damped and holding roughly where it actually is. The
        commanded position follows the measurement down, so nothing is fighting.
        """
        end = time.monotonic() + 0.25
        while time.monotonic() < end:
            ff *= 0.85
            resp = self.send_command(motor_id, pos, vel=0.0,
                                     kp=params.kp_hold, kd=params.kd_hold,
                                     torque=ff, recv_timeout=_STREAM_RECV_TIMEOUT)
            if resp:
                pos = resp.pos
            if on_tick is not None:
                on_tick()
            time.sleep(0.005)
        return pos

    def _settle(self, motor_id, target, current, params, on_tick=None):
        """Drive the residual position error out after the profile has finished.

        tau = kp*(p_des - p) + kd*(v_des - v) means torque comes only from error,
        so holding against a load tau_L REQUIRES a standing error of tau_L/kp.
        A loaded joint therefore parks below its target and stays there however
        the trajectory is shaped -- this is not something a stiffer profile can
        fix. Here an integral term walks the MIT frame's feedforward torque up
        until it carries the load, which moves the equilibrium onto the target.

        Two things stop this from fighting a mechanical hard stop:

        * The commanded position is clamped to within `settle_max_err` of the
          measured one, so the kp term can never exceed kp_hold*settle_max_err
          however far away the target is -- the saturating-spring trick from
          scripts/bilateral.py. Without it a joint jammed 45 deg short of target
          draws the full TMAX (28 Nm in simulation) instead of about 9.
        * The integral gives up only once it is OUT OF AUTHORITY: at its ceiling
          and still buying no progress. Aborting on "no progress" alone would
          false-fire on exactly the heavy joints this exists for, because they
          legitimately need a large feedforward and take time to build it.

        Returns (pos, tau_ff, ok). ok=False means it gave up, tau_ff has been bled
        to zero, and pos is wherever the joint actually came to rest.
        """
        ff  = 0.0
        pos = current
        vel = 0.0
        best = abs(target - pos)
        t0 = last_t = last_progress = time.monotonic()
        while True:
            now = time.monotonic()
            # Cap dt so a scheduling hiccup can't kick the integrator.
            dt, last_t = min(now - last_t, 0.05), now
            if now - t0 > params.settle_max_secs:
                return pos, ff, True        # window expired; keep what we earned
            err = target - pos
            ff  = max(-params.settle_ff_max,
                      min(params.settle_ff_max, ff + params.settle_ki * err * dt))
            p_cmd = max(pos - params.settle_max_err,
                        min(pos + params.settle_max_err, target))
            resp = self.send_command(motor_id, p_cmd, vel=0.0,
                                     kp=params.kp_hold, kd=params.kd_hold,
                                     torque=ff, recv_timeout=_STREAM_RECV_TIMEOUT)
            if resp:
                pos, vel = resp.pos, resp.vel
            if on_tick is not None:
                on_tick()
            if best - abs(target - pos) > params.stall_thresh:
                best, last_progress = abs(target - pos), now
            if abs(target - pos) <= params.settle_tol and abs(vel) < params.settle_vel:
                return pos, ff, True
            if (abs(ff) >= 0.95 * params.settle_ff_max
                    and now - last_progress > params.settle_stall_secs):
                short = math.degrees(abs(target - pos))
                pos = self._settle_release(motor_id, pos, ff, params, on_tick)
                print(f"  motor 0x{motor_id:02X} settle gave up {short:.1f} deg short "
                      f"at the {params.settle_ff_max:g} Nm feedforward ceiling -- "
                      f"treating this as a mechanical stop and relaxing.")
                return pos, 0.0, False
            time.sleep(0.005)

    def _vel_move(self, motor_id, start_pos, target, velocity, kp=None, kd=None,
                  on_tick=None, raw=False):
        params = self.params_for(motor_id)
        # Ramp to the hold gains at the end of the move only when the gains came
        # from params; an explicit kp/kd from the caller is honoured all the way.
        ramp = kp is None and kd is None
        if kp is None: kp = params.kp_move
        if kd is None: kd = params.kd_move
        current = start_pos
        # Distance-independent reference trajectory: stream intermediate (p_ref,
        # v_ref) waypoints from an S-curve profile instead of the final target,
        # so the tracking error -- and thus the torque kick -- stays bounded and
        # the motion looks the same regardless of distance.
        # `raw` strips those helpers and commands the target itself. A degenerate
        # profile (V=0) is exactly that: __post_init__ leaves T = T_a = 0, so
        # sample() returns (target, 0.0, True) at every tick, and the gain ramp
        # below -- already gated on prof.T_a > 0 -- switches itself off. Nothing
        # else changes, so the stall/timeout guards and the give-up reporting
        # still cover a raw move.
        prof = _SCurveProfile(start_pos, target, 0.0 if raw else velocity,
                              params.accel)
        # A long cruise can legitimately exceed the fixed move_timeout; allow the
        # profile's own duration (plus margin) so it isn't cut off mid-move.
        timeout = max(params.move_timeout, prof.T + 1.0)
        t_start = last_progress = time.monotonic()
        last_pos = start_pos
        heard = 0   # feedback frames received; 0 means `current` is still start_pos
        gave_up = None
        while abs(target - current) > params.stop_thresh:
            now = time.monotonic()
            elapsed = now - t_start
            p_ref, v_ref, _ = prof.sample(elapsed)
            kp_t, kd_t = kp, kd
            if ramp and prof.T_a > 0.0:
                # Blend to the hold gains across the decel corner so the move ENDS
                # at hold stiffness. Otherwise the handoff steps the gains and a
                # loaded joint visibly re-settles at the new, softer equilibrium.
                f = min(1.0, max(0.0, (elapsed - prof.T_a - prof.T_c) / prof.T_a))
                kp_t = kp + (params.kp_hold - kp) * f
                kd_t = kd + (params.kd_hold - kd) * f
            resp = self.send_command(motor_id, p_ref, vel=v_ref,
                                     kp=kp_t, kd=kd_t,
                                     torque=params.torque_ff,
                                     recv_timeout=_STREAM_RECV_TIMEOUT)
            if resp:
                heard += 1
                current = resp.pos
                if abs(current - last_pos) > params.stall_thresh:
                    last_pos, last_progress = current, now
            if on_tick is not None:
                on_tick()
            # Stalled: position hasn't advanced for STALL_SECS -> the joint
            # can't get closer (hard stop / load). Hand it to the settle, which
            # can tell those two apart; see below.
            if now - last_progress > params.stall_secs:
                gave_up = "stalled"
                break
            if now - t_start > timeout:
                gave_up = "move timed out"
                break
            time.sleep(0.005)

        # The settle deliberately runs even after the stall guard fired. Above
        # kp_move*stop_thresh of load the droop alone exceeds the arrival
        # tolerance -- roughly 0.35 Nm on a J4340, 0.05 Nm on a J4310 -- so a
        # loaded joint ALWAYS stalls short and would otherwise never be
        # compensated. "Merely heavy" and "genuinely jammed" look identical in the
        # feedback (both park with velocity ~0 under torque); the settle's own
        # authority guard is what actually distinguishes them.
        # `not raw`: the settle is the biggest of the helpers raw exists to remove.
        # Skipping it falls through to load_ff = 0 below, which is right -- a raw
        # move measures no load, and the caller's hold must not carry a value some
        # earlier tuned move learned.
        if heard and params.settle_enabled and not raw:
            current, ff, ok = self._settle(motor_id, target, current, params, on_tick)
            self._load_ff[motor_id] = ff
            # Converged: hold the COMMANDED target. Returning the measurement is
            # what let the caller re-anchor its hold at the drooped position, so
            # the joint sagged a second time by tau_load/kp_hold below it.
            return target if ok else current

        self._load_ff[motor_id] = 0.0
        if gave_up is not None:
            self._report_give_up(motor_id, gave_up, target, current, heard)
            return current
        return target

    def _posvel_move(self, motor_id, start_pos, target, velocity, on_tick=None):
        """Native POS_VEL move: resend the position+vel_limit frame and poll
        feedback until arrival. Same stall/timeout escapes as _vel_move so a joint
        that can't reach target (hard stop / load) doesn't spin forever.

        No settle phase here: the firmware runs its own position loop and the
        POS_VEL frame carries no gain or feedforward fields, so there is nothing
        for the host to compensate with. Sag under load in this mode is the
        motor's on-board tuning (see memory posvel-choppy-onboard-tuning)."""
        params   = self.params_for(motor_id)
        current  = start_pos
        t_start = last_progress = time.monotonic()
        last_pos = start_pos
        heard = 0   # feedback frames received; 0 means `current` is still start_pos
        while abs(target - current) > params.stop_thresh:
            now  = time.monotonic()
            resp = self.send_posvel_command(motor_id, target, velocity)
            if resp:
                heard += 1
                current = resp.pos
                if abs(current - last_pos) > params.stall_thresh:
                    last_pos, last_progress = current, now
            if on_tick is not None:
                on_tick()
            if now - last_progress > params.stall_secs:
                self._report_give_up(motor_id, "stalled", target, current, heard)
                break
            if now - t_start > params.move_timeout:
                self._report_give_up(motor_id, "move timed out", target, current, heard)
                break
            time.sleep(0.005)
        return current

    def hold_position(self, motor_id, pos_rad, torque_ff=None, mode=None):
        """Block and hold position at pos_rad until KeyboardInterrupt.

        `torque_ff=None` uses the static load the last settle learned for this
        motor (load_ff), which is what keeps a loaded joint on its target instead
        of tau_load/kp_hold below it. Pass a number to override, or 0.0 for none.

        In POS_VEL mode the motor's firmware latches the commanded position, so we
        rely on its internal hold and just idle here without streaming frames.
        NOTE: if the motors' comms-loss watchdog disables output on a timeout
        (see memory damiao-motor-watchdog), this hold could drop -- revisit and
        resend the POS_VEL frame here if testing shows that. MIT mode has no such
        latch, so it keeps streaming the hold command."""
        params = self.params_for(motor_id)
        if torque_ff is None:
            torque_ff = self.load_ff(motor_id)
        if mode is None:
            mode = self.DEFAULT_MODE
        if mode != MODE_MIT:
            while True:
                time.sleep(0.1)
        while True:
            self.send_command(motor_id, pos_rad, vel=0.0,
                              kp=params.kp_hold, kd=params.kd_hold,
                              torque=torque_ff,
                              recv_timeout=_STREAM_RECV_TIMEOUT)
            time.sleep(0.005)

    def continuous_movement_at_vel(self, motor_id, vel_rad_s):
        """
        Spin at constant velocity until KeyboardInterrupt, then ramp down to zero.
        Uses KP=0 (pure velocity mode).
        """
        kd = self.params_for(motor_id).kd_move
        try:
            while True:
                self.send_command(motor_id, 0.0, vel=vel_rad_s, kp=0.0, kd=kd,
                                  recv_timeout=_STREAM_RECV_TIMEOUT)
                time.sleep(0.005)
        except KeyboardInterrupt:
            print("\nRamping down...")
            steps = 5
            for i in range(steps - 1, -1, -1):
                ramp_vel = vel_rad_s * (i / steps)
                for _ in range(20):
                    self.send_command(motor_id, 0.0, vel=ramp_vel, kp=0.0, kd=kd,
                                      recv_timeout=_STREAM_RECV_TIMEOUT)
                    time.sleep(0.005)

    # ------------------------------------------------------------------ #

    def shutdown(self):
        self._bus.shutdown()
