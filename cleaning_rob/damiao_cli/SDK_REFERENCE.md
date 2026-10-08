# Damiao Motor SDK — Agent Reference

## Summary

A small Python SDK for driving **Damiao DM-series motors** (DM-J4310-2EC and
DM-J4340-2EC in this setup) over a **CAN bus** exposed as a SocketCAN interface.
It speaks the motor's **MIT control protocol** for command/feedback and the
**0x7FF register protocol** for configuration (IDs, control mode, limits, flash
storage). The core is a single class, `MotorBus` (in `motor.py`), wrapped by a
CLI (`run.py`) and a demo sequence runner (`scripts/sequence.py`).

The two motors are **not interchangeable in software**. They differ in gearbox
and in the MIT full-scale limits their firmware uses:

| | Reduction | Rated / peak torque | VMAX | TMAX |
|---|---|---|---|---|
| DM-J4310-2EC | 10:1 | 3 / 7 Nm | 30 rad/s | 10 Nm |
| DM-J4340-2EC | 40:1 | 9 / 27 Nm | 8 rad/s | 28 Nm |

Both use PMAX 12.5 rad. Because reflected inertia scales with the reduction
ratio squared, the 4310 sees ~16× less than the 4340 — so they need separate
gains, and `params.yaml` keys tuning by model.

The motor accepts three on-board control modes — **MIT**, **POS_VEL**, and
**VEL** — selected by writing the `CTRL_MODE` register. In MIT mode the *host*
runs the closed loop (`τ = KP·(p_des−p) + KD·(v_des−v) + τ_ff`); in POS_VEL mode
the *motor firmware* runs the position loop and the host just streams a target +
velocity limit.

## Layout

| File | Role |
|------|------|
| `motor.py` | Core SDK: `MotorBus`, `MotorParams`, encode/decode helpers, `_SCurveProfile`. |
| `run.py` | CLI front-end (`move_to_pos`, `move_by_offset`, `query`, `scan`, `set_id`, `set_zero`, `continuous_movement_at_vel`, `help`). |
| `scripts/sequence.py` | Demo: fixed 12-step move sequence over 0x01–0x03 (0x04 held only), single-threaded, idle motors fed via `on_tick`. |
| `scripts/bilateral.py` | Bilateral test: couples two motors with a virtual spring/damper so they hold a positional equilibrium. |
| `scripts/settle_sim.py` | Hardware-free simulation of the end-of-move settle; regression check + tuning tool for the `settle:` block. |
| `params.yaml` | Control tuning (gains, velocity, accel, settle, stall/timeout guards). Loaded on `MotorBus` init. |
| `socketcan_virtual_converter.py` | Bridges a Waveshare CANalyst-II USB adapter (`canalystii`, ch 1, 1 Mbit) to a `vcan0` SocketCAN interface so the rest of the stack can use plain SocketCAN. |

## Hardware / runtime assumptions

- USB-CAN adapter at vendor `0x04d8`, product `0x0053` (Waveshare CANalyst-II).
  `MotorBus` detaches the kernel driver and opens a `python-can` bus.
- Default CAN channel `vcan0`, interface `socketcan`. CAN bitrate 1 Mbit.
- Dependencies: `pyusb` (`usb.core`), `python-can` (`can`), `pyyaml`.
- To use real hardware behind `vcan0`, run `socketcan_virtual_converter.py` first
  (after `sudo ip link add dev vcan0 type vcan && sudo ip link set up vcan0`).

## CLI usage (`run.py`)

```bash
python run.py move_to_pos    0x03 -45.0 [--mode mit|pos_vel] [--velocity RAD/S] [--raw] [--kp K] [--kd K]
python run.py move_by_offset 0x03  20   [--mode mit|pos_vel] [--velocity RAD/S] [--raw] [--kp K] [--kd K]
python run.py move_to_pos    0x03 -45.0 --raw         # no profile/gain ramp/settle (MIT only)
python run.py continuous_movement_at_vel 0x03 10.0   # spin until Ctrl+C, ramps down
python run.py query 0x03                              # pos/vel/torque/temps/state
python run.py scan  [--start 0x01] [--end 0x07]       # table of all motors in range
python run.py limits 0x03                             # PMAX/VMAX/TMAX, model, gains
python run.py set_zero 0x03                           # current pos -> zero (persists)
python run.py set_id   0x01 0x03 [--new-mst-id 0x13]  # change CAN ID (power-cycle to apply)
python run.py help
```

Global options: `--channel` (default `vcan0`), `--interface` (default `socketcan`),
`--params PATH` (default bundled `params.yaml`). Motor IDs accept hex (`0x03`).
Move/positions are in **degrees** at the CLI; the library works in **radians**.

> Note: CLI `--mode` defaults to `mit`, but the *library* `MotorBus.DEFAULT_MODE`
> is `POS_VEL`. `move`/`scan` `--velocity` CLI default is `4.0` rad/s.

## Scripts (`scripts/`)

```bash
python scripts/sequence.py  [--mode mit|pos_vel]     # 12-step multi-motor demo
python scripts/settle_sim.py [--model DM4340] [--loads Nm ...] [--degrees D]
                             [--hard-stop DEG] [--tolerance DEG]   # no hardware
python scripts/bilateral.py [--motors 0x01 0x04] [--mirror] [--offset DEG]
                            [--leader ID] [--leader-scale F]
                            [--kp1 K] [--kd1 K] [--kp2 K] [--kd2 K]
                            [--max-error DEG] [--engage S] [--release S]
                            [--rate HZ] [--vel-filter A] [--no-vel-couple]
```

`bilateral.py` couples two motors with a virtual spring/damper and holds a fixed
separation between them: backdrive either one and the other follows, hold one
and the reaction force builds at the other. The coupling is symmetric — neither
motor leads — and each motor is commanded from its partner's live feedback every
tick, so `p1_des = s·p2 + offset`, `v1_des = s·v2` and vice versa (`s = −1` with
`--mirror`, for motors mounted facing each other).

Equilibrium is whatever separation the pair is at when the loop engages, so
startup moves nothing; `--offset` ramps to a different one. `--max-error`
saturates the spring, capping the reaction torque at `kp·max_error` so yanking
one motor can't command a violent snap-back.

`--leader ID` names the motor you drive by hand. It keeps its full spring
stiffness but takes the softer `bilateral.leader` damping, because kd works
against the *follower's velocity lag* and so reads as viscous drag on whoever is
moving. Force reflection is unchanged — block the follower and the leader still
pushes back at full `kp·error`. The cut is safe because kd acts on the pair's
**relative** velocity: both motors damp the same coordinate, so the follower's kd
keeps anchoring the pair. Simulation puts the effect at ~55% less leader torque
under 1 Hz hand motion, and none at all at constant velocity (once the follower
matches speed the relative velocity vanishes and steady drag is the `kp·lag`
term). `--leader-scale F` scales the leader's kp and kd together for live tuning;
an explicit `--kpN`/`--kdN` is taken literally and bypasses both. MIT only — POS_VEL's firmware
position loop has no compliance. Both motors get a frame every tick, so this
loop feeds the comms-loss watchdog from a single thread — the same reason
`sequence.py` interleaves its idle holds through `on_tick` rather than a second thread.

## Library usage (`motor.py`)

```python
from motor import MotorBus
import math

bus = MotorBus(channel='vcan0')          # loads params.yaml automatically
ids = [0x01]

bus.init_motors(ids, MotorBus.MODE_MIT)  # set CTRL_MODE *before* enter_motor_mode
start = bus.enter_motor_mode(0x01)       # returns current pos (rad) or None
try:
    target = bus.move_to_pos(0x01, math.radians(-45), start_pos=start,
                             mode=MotorBus.MODE_MIT)
    bus.hold_position(0x01, target, mode=MotorBus.MODE_MIT)   # blocks until Ctrl+C
except KeyboardInterrupt:
    pass
finally:
    bus.exit_motor_mode(0x01)
    bus.shutdown()
```

### Key `MotorBus` methods

- **Lifecycle:** `enter_motor_mode(id)` → pos | None, `exit_motor_mode(id)`, `shutdown()`.
- **Config (motor mode OFF):** `init_motors(ids, mode)`, `set_control_mode(id, mode, persist=False)`,
  `set_id(id, new_esc_id, new_mst_id=None)`, `set_zero(id)`.
- **Read:** `query(id)` → `Feedback` | None, `scan(ids)` → `{id: Feedback | None}`.
  `Feedback` carries `pos` (**rad**), `vel`, `torque`, `status`, `error`, `t_mos`, `t_rotor`.
- **Limits/model:** `read_limits(id)` → `MotorLimits` | None, `resolve(id)` →
  `(limits, model_name)`, `limits_for(id)`, `params_for(id)`.
- **Motion:** `move_to_pos(id, target_rad, start_pos, ...)`, `move_by_offset(id, delta_rad, start_pos, ...)`,
  `hold_position(id, pos_rad, ...)`, `continuous_movement_at_vel(id, vel_rad_s)`.
  In MIT mode a move ends with a **settle** phase (see the `settle` block below) and
  returns the *commanded* target once it converges — not the drooped measurement — so a
  caller's hold doesn't re-anchor low. On a genuine stall it still returns the measured
  position. `load_ff(id)` gives the static load the last settle learned;
  `hold_position(..., torque_ff=None)` uses it by default, which is what keeps a loaded
  joint on target instead of `τ_load/kp_hold` below it.
  Both moves take **`raw=False`**. `raw=True` strips the host-side stack — no S-curve
  profile, no ramp into the hold gains, no settle — and simply commands the target, so
  `velocity` is unused and `load_ff` stays 0 (the following hold carries no feedforward).
  The stall/timeout guards still apply, so a raw move can't spin forever on a jam. It is
  **MIT only**: `raw=True` with any other mode raises `ValueError`, because in POS_VEL the
  firmware runs the position loop and there is nothing host-side to strip. Use it as the
  baseline to compare a tuned move against — expect the step to saturate `TMAX` on the
  first frame and the joint to park at `τ_load/kp` short.
  Both moves take **`on_tick=None`** — a zero-arg callable invoked once per control
  tick, after this motor's frame is sent. It is how you keep *other* motors on the bus
  alive during a move without a second thread (see the threading rule below and
  `scripts/sequence.py`). Keep it short: every millisecond it spends is a millisecond
  the moving motor goes uncommanded.
- **Low-level:** `send_command(id, pos, vel, kp, kd, torque)` (MIT frame on arb id = motor id),
  `send_posvel_command(id, pos, vel_limit)` (frame on `0x100 + id`, float32 pos + vel).

### Control modes (`CTRL_MODE` register, RID 10)

| Constant | Value | Loop runs on | Command frame |
|----------|-------|--------------|---------------|
| `MODE_MIT` | 1 | Host | MIT (8 packed bytes) on arb id `= motor id` |
| `MODE_POS_VEL` | 2 | Motor firmware | `0x100 + id`, float32 pos + float32 vel limit |
| `MODE_VEL` | 3 | Motor firmware | (velocity frame; not wired into high-level moves) |

`set_control_mode` writes take effect immediately; `persist=True` also stores to
flash. `init_motors` re-sets the mode fresh each run (no flash wear) **and** reads
each motor's limits, so it must run before `enter_motor_mode`: register access
requires the motor **not** be in motor mode, and the limits must be known before
the first frame is encoded or decoded.

## Tuning (`params.yaml` → `TuningConfig` / `MotorParams`)

Gains are keyed by motor **model**, since they depend on the gearbox. Layout:

- `limits.auto_read` — read PMAX/VMAX/TMAX off each motor at init (default true).
- `defaults` — the base `move` / `hold` / `guards` block.
- `models.<NAME>` — per-model override, shallow-merged over `defaults`, so a block
  only lists what it changes. An optional `limits:` sub-key is the fallback used
  when auto-read is unavailable; it never overrides a successful read.
- `motors` — `{can_id: MODEL}` pin. **Overrides auto-detection** for gain selection;
  the limits read off the motor still drive MIT scaling. Needed whenever a unit's
  VMAX/TMAX have been reconfigured off the vendor defaults.

Within a block: `move` has `kp`, `kd`, `torque_ff`, `velocity` (cruise / pos_vel
limit, rad/s), `accel` (rad/s², S-curve ramp); `hold` has `kp`, `kd` used by
`hold_position` and idle holds; `bilateral` has `kp`, `kd` for the virtual
spring/damper in `scripts/bilateral.py` (softer than `hold` -- the pair has to
stay backdrivable by hand) plus an optional `leader: {kp, kd}` sub-block used for
the `--leader` motor, which falls back to that block's own `kp`/`kd` when absent;
`settle` configures the end-of-move convergence (below);
`guards` has `stop_thresh_deg` (arrival tol),
`stall_thresh_deg` (min progress to count as moving), `stall_secs` (no-progress
give-up), `move_timeout` (hard cap/move, s).

### The `settle` block — why a loaded joint needs it

`τ = kp·(p_des − p) + kd·(v_des − v) + τ_ff` makes torque **only from error**, so
holding against a load torque `τ_L` *requires* a standing position error of `τ_L/kp`.
A loaded joint therefore parks below its target and stays there — and no amount of
trajectory shaping changes that, because `_SCurveProfile` already lands exactly on
target with zero velocity and zero acceleration. The lag is in the motor, not the
reference.

Worse, the move used to return that **measured** drooped position, which the caller
then stored as its hold setpoint, so the joint sagged a *second* time by `τ_L/kp_hold`
below it. That was the visible end-of-move drop on loaded joints.

After the profile finishes, `_settle` walks a feedforward torque up until it carries
the load, putting the equilibrium on the target itself. It runs **even after the stall
guard fires** — above `kp_move · stop_thresh` of load (≈0.35 Nm on a J4340, ≈0.05 Nm on
a J4310) the droop alone exceeds the arrival tolerance, so a loaded joint always stalls
short and would otherwise never be compensated.

Keys: `enabled`, `ki` (Nm per rad-second), `ff_max` (Nm ceiling on the learned
feedforward), `max_err_deg` (setpoint clamp vs. the measured position),
`tol_deg`, `vel_thresh` (rad/s), `max_secs`, `stall_secs`.

Two guards keep it off a mechanical hard stop, and **both matter**:

- The commanded position is clamped to within `max_err_deg` of the measured one, so the
  kp term can never exceed `kp_hold · max_err` however far away the target is — the
  saturating-spring trick from `bilateral.py`. Without it a joint jammed 45° short draws
  the full TMAX (28 Nm simulated) instead of about 9.
- The integral gives up only once it is **out of authority**: at `ff_max` and still
  buying no progress for `stall_secs`. Aborting on "no progress" alone false-fires on
  exactly the heavy joints this exists for, since they legitimately need a large
  feedforward and take time to build it. On give-up it bleeds `τ_ff` back to zero,
  keeps the measured position, and says so.

`ff_max` doubles as that give-up threshold, so oversizing it means a jammed joint pushes
harder for longer before relaxing. Set it a little above the heaviest static load the
joint carries. Re-tune with `python scripts/settle_sim.py --model DM4340`.

Override per-call via `kp=`/`kd=`/`velocity=`/`mode=` kwargs, or load a custom file
with `MotorBus(params_path=...)` / `--params`. A file with no `defaults:`/`models:`
is read as the old flat layout and becomes the defaults block.

Check what a motor resolved to with `python run.py limits 0x01` — it prints the
limits, the detected model, the selected gains, and the position error at which
`kp_move` saturates the motor's peak torque.

## Feedback status codes (`data[0] >> 4`)

`0` disabled · `1` enabled · `8` overvoltage · `9` undervoltage · `A` overcurrent ·
`B` MOSFET overheat · `C` motor coil overheat · `D` communication loss · `E` overload

Anything ≥ 8 is a fault that drops the motor out of enable mode. `MotorBus` prints
these edge-triggered (once per change, not once per 200 Hz frame); `Feedback.error`
exposes the name, and `query`/`scan`/`limits` show it.

## Feature list

- MIT-protocol command encoding & feedback decoding, scaled by **each motor's own**
  PMAX/VMAX/TMAX read from RID 21/22/23 at init (KP 0–500 and KD 0–5 are fixed by
  the frame format and identical on every model).
- Automatic motor-model identification from those limits, which selects the model's
  gain block in `params.yaml`.
- Fault reporting: the feedback status nibble and both temperatures are decoded and
  surfaced instead of discarded.
- Three switchable on-board control modes (MIT host-loop, POS_VEL & VEL firmware-loop).
- **Distance-independent S-curve trajectory** (`_SCurveProfile`): smoothstep accel/decel
  corners + cruise, with triangular fallback for short moves — bounds tracking error so
  MIT moves stay smooth; pure/unit-testable, no bus access.
- Absolute and relative moves, blocking hold, constant-velocity spin with ramp-down.
- **Stall + timeout escapes** on every move loop so a loaded/limited joint never spins forever.
- **Raw mode** (`raw=True`, CLI `--raw`): the same move with the profile, gain ramp and
  settle removed, as an A/B baseline for tuning. Implemented as a degenerate `V=0`
  profile, so it reuses the real move loop and its guards rather than duplicating them.
- **End-of-move settle with load feedforward**: gains ramp to the hold values across the
  decel corner, then an integral term in the MIT `τ_ff` field carries the joint's static
  load so it ends *on* target rather than `τ_load/kp` below it (measured 8–58× less
  end-of-move error under load), with a saturating-spring clamp and an out-of-authority
  give-up so it can't fight a hard stop.
- Register protocol for CAN-ID reassignment, control-mode select, zero-set, flash store.
- Serialized bus access (`threading.Lock`) with ID-nibble feedback matching to discard
  frames from other motors. Note this makes each *frame exchange* atomic; it does **not**
  make a `MotorBus` usable from two threads — see the threading rule below.
- Single-motor `query` and multi-motor `scan` diagnostics.
- YAML-configurable tuning, per-call overrides, custom CAN channel/interface.
- CLI (`run.py`) and scripted multi-motor sequence example.
- **Symmetric bilateral coupling** (`scripts/bilateral.py`): two motors joined by a
  virtual spring/damper holding a positional equilibrium, with a saturating error
  clamp, engage/release ramps and a feedback-staleness abort.

## Protocol gotchas (from project memory)

- **MIT scaling is per-motor, not per-protocol.** The frame carries position/velocity/
  torque as fractions of each motor's PMAX/VMAX/TMAX. Hardcoding the MIT mini-cheetah
  values (±45 rad/s, ±18 Nm) silently sent a J4310 67% of every requested velocity and
  a J4340 only 18%, and overstated feedback by 1.5×/5.6×. Position happened to survive
  because PMAX is 12.5 on both. Read the registers; don't assume.
- **Register RIDs 21/22/23 are float32, not uint32.** Only RIDs 7–10, 13–16 and 35–36
  are integers (`_reg_is_int`); unpacking a float register as `<I` yields garbage.
- **Gains don't transfer between models.** They're a property of the gearbox — see the
  reduction-ratio table above.
- **`identify()` can name the wrong variant, confidently.** It matches on VMAX/TMAX, so
  a unit whose VMAX was reconfigured in Damiao's Debug Assistant lands on another row:
  a 24V DM4310 reading VMAX 50 is named as whichever row matches. `VENDOR_LIMITS` is a
  **list** of `(name, limits)`, not a dict, precisely so one name can own several rows
  (the 24 V and 48 V builds of a gearbox want the same gains) — as a dict the duplicate
  keys collapsed to the last row and a 24 V DM4310 at VMAX 30 matched *nothing*, so
  `params_for()` fell back to `defaults` and every per-model gain silently did nothing.
  Rows can still collide the other way: `DMH6215` and `DMG6220` report identical limits
  and are indistinguishable on the wire. A `motors:` pin **overrides** detection for gain selection (the read
  limits still drive MIT scaling); `params_for()` warns when a resolved model has no
  block. TMAX alone identifies the family (28 → 4340, 10 → 4310) when in doubt.
- **One thread per `MotorBus`.** `_send_raw` is a send-then-receive transaction over a
  single shared socket with no per-motor demultiplexing: frames from other motors are
  read and **discarded**, and a discarded frame is gone. Two threads sharing a bus
  therefore eat each other's feedback — each one's replies vanish into the other's
  discard loop, its move loop sees `resp is None` forever, concludes the motor never
  moved, gives up on the stall guard and returns its *starting* position. The caller
  records that stale value and the next hold actively drives the motor back to it: the
  motor lurches off and is yanked home, every step. Drive multiple motors by
  round-robin from one loop (`bilateral.py`) or via `on_tick` (`sequence.py`).
- **Watchdog:** DM motors in MIT mode drop the last command without continuous frames,
  so idle motors need a frame every tick (POS_VEL latches internally and does not).
  Send those from the *same* thread as the move — a hold thread is the trap above.
  A faulted motor (status ≥ 8) also drops out of enable mode; watch for the printed fault.
- **Streaming loops shouldn't block long on feedback.** A 50 ms wait for a dropped reply
  stalls the command stream at 200 Hz, and the resulting setpoint jump reads as buzzing
  (`_STREAM_RECV_TIMEOUT`).
- **Feedback byte0:** `data[0] = (status << 4) | id`; match the low nibble or every reply
  looks like "no response".
- **Stall escapes:** closed-loop position waits must have stall/timeout escapes; joints
  settle outside a tight threshold under load.
- **A PD joint sags under load, and that is the control law, not the trajectory.** Torque
  comes only from error, so holding `τ_L` needs a standing error of `τ_L/kp`. Symptoms
  that look like a bad motion profile — "drops at the end of travel", drift accumulating
  across steps — are usually this. The cure is feedforward torque (the MIT frame carries
  it), not a stiffer or differently-shaped profile: `_SCurveProfile` already ends exactly
  on target with zero velocity *and* zero acceleration. Never return a measured position
  as the next hold's setpoint under load; it compounds the sag every step.
- **POS_VEL choppiness:** DM-J4340 POS_VEL stutter comes from on-board ACC/DEC/MAX_SPD/
  loop-gain registers, not the CAN frame; MIT is smooth.
