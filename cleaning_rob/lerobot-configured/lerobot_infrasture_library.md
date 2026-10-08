# LeRobot Infrastructure — CAN/Damiao Follower + Feetech Leader

Scope of this document: the parts of the LeRobot codebase relevant to an
implementation where the **follower robot uses Damiao motors over a CAN bus**
and the **leader teleoperator uses Feetech motors over serial**. Generic
capabilities are mentioned only where they frame the specific path. Every
statement below is derived from the files cited; nothing is asserted that was
not read directly from the codebase.

All paths are relative to `src/lerobot/` unless noted.

---

## 1. Layered structure (only the layers on this path)

```
scripts/lerobot_record.py        CLI: lerobot-record  → record() / record_loop()
scripts/lerobot_teleoperate.py   CLI: lerobot-teleoperate → teleoperate() / teleop_loop()
scripts/lerobot_setup_can.py     CLI: lerobot-setup-can (bring up / test CAN interfaces)
        │
robots/                          Robot abstraction (follower)
  robot.py                       Robot ABC
  openarm_follower/              reference Damiao-over-CAN follower
  config.py                      RobotConfig base (draccus ChoiceRegistry)
  utils.py                       make_robot_from_config(), ensure_safe_goal_position()
teleoperators/                   Teleoperator abstraction (leader)
  teleoperator.py                Teleoperator ABC
  so_leader/                     SO-100/101 leader (Feetech)
  config.py                      TeleoperatorConfig base (draccus ChoiceRegistry)
  utils.py                       make_teleoperator_from_config()
motors/
  motors_bus.py                  MotorsBusBase ABC, Motor, MotorCalibration, MotorNormMode
  damiao/                        DamiaoMotorsBus (CAN, python-can) + tables.py
  feetech/                       FeetechMotorsBus (serial) + tables.py
cameras/                         Camera backends (opencv/realsense/zmq), make_cameras_from_configs()
```

The control scripts (`record_loop`, `teleop_loop`) are hardware-agnostic: they
only call methods on the `Robot` / `Teleoperator` objects and pass around flat
`{"<key>": value}` dicts. The CAN/serial distinction lives entirely inside the
motor bus classes. (See `scripts/lerobot_record.py:217` `record_loop`,
`scripts/lerobot_teleoperate.py:132` `teleop_loop`.)

---

## 2. Motor bus interface (`motors/motors_bus.py`)

### `MotorsBusBase` (abstract, line 57)

The minimal interface all buses implement, regardless of transport. Abstract
methods:

- `connect(handshake: bool = True)`
- `disconnect(disable_torque: bool = True)`
- `is_connected` (property)
- `read(data_name: str, motor: str) -> Value`
- `write(data_name: str, motor: str, value: Value)`
- `sync_read(data_name: str, motors=None) -> dict[str, Value]`
- `sync_write(data_name: str, values: dict[str, Value])`
- `enable_torque(motors=None, num_retry=0)` / `disable_torque(...)`
- `read_calibration() -> dict[str, MotorCalibration]`
- `write_calibration(calibration_dict, cache=True)`

`__init__` stores `port`, `motors: dict[str, Motor]`, and `calibration`
(defaults to `{}`). `type Value = int | float`; `type NameOrID = str | int`.

`DamiaoMotorsBus` subclasses `MotorsBusBase` **directly**
(`motors/damiao/damiao.py:76`). `FeetechMotorsBus` subclasses
`SerialMotorsBus` (`motors/feetech/feetech.py:89`).

### `Motor` dataclass (line 183)

```python
@dataclass
class Motor:
    id: int
    model: str
    norm_mode: MotorNormMode
    motor_type_str: str | None = None   # used by Damiao to select MotorType
    recv_id: int | None = None          # used by Damiao for CAN reply ID
```

`motor_type_str` and `recv_id` exist specifically for the CAN path; the serial
(Feetech) path uses `id` + `model` + `norm_mode`.

### `MotorNormMode` (line 168)

`RANGE_0_100`, `RANGE_M100_100`, `DEGREES`.

### `MotorCalibration` dataclass (line 174)

```python
@dataclass
class MotorCalibration:
    id: int
    drive_mode: int
    homing_offset: int
    range_min: int
    range_max: int
```

Calibration is persisted as JSON per device id by the `Robot`/`Teleoperator`
base classes (`_save_calibration` / `_load_calibration`), under
`HF_LEROBOT_CALIBRATION`.

---

## 3. `DamiaoMotorsBus` — the CAN follower bus (`motors/damiao/damiao.py`)

> Class docstring (line 77): "The Damiao implementation for a MotorsBus using
> CAN bus communication. This class uses python-can for CAN bus communication
> with Damiao motors." Portions derived from `DM_Control_Python` (MIT license,
> noted in the file header).

### Construction (line 92)

```python
DamiaoMotorsBus(
    port: str,                       # CAN channel, e.g. "can0"; "/dev/..." for serial-style
    motors: dict[str, Motor],
    calibration: dict[str, MotorCalibration] | None = None,
    can_interface: str = "auto",     # "auto" | "socketcan" | "slcan"
    use_can_fd: bool = True,
    bitrate: int = 1_000_000,        # nominal bitrate
    data_bitrate: int | None = 5_000_000,  # CAN FD data bitrate; ignored if use_can_fd False
)
```

- `require_package("python-can", extra="damiao", import_name="can")` is called
  first — `python-can` is an optional dependency behind the `damiao` extra
  (line 114). The module guards the import via `_can_available` (lines 29-35).
- Each `Motor` must have `motor_type_str` set, or `__init__` raises
  `ValueError("Motor '<name>' is missing required 'motor_type'")` (line 130-131).
- `motor_type_str` is resolved to a `MotorType` enum via
  `getattr(MotorType, motor.motor_type_str.upper().replace("-", "_"))` (line 132).
- Each motor's `recv_id` is mapped in `_recv_id_to_motor` for response filtering
  (line 135-136).
- A per-motor state cache `_last_known_states` (position/velocity/torque/
  temp_mos/temp_rotor) is initialized to zeros (line 139).
- A per-motor gains cache `_gains` is initialized to `{"kp": 10.0, "kd": 0.5}`
  (line 152). These defaults are what `Goal_Position` writes use unless `Kp`/`Kd`
  are written first.

### Connection (`connect`, line 159)

- `can_interface="auto"` resolves to `"slcan"` when `port` starts with `/dev/`,
  otherwise `"socketcan"` (lines 170-176).
- Builds `can.interface.Bus(channel=port, bitrate=bitrate, interface=can_interface)`.
  When `can_interface == "socketcan"` **and** `use_can_fd` **and**
  `data_bitrate is not None`, it adds `data_bitrate=...` and `fd=True`
  (lines 185-193). Only these kwargs are forwarded to python-can.
- `handshake=True` (default) pings every motor: sends the enable frame
  (`[0xFF]*7 + [CAN_CMD_ENABLE]`), waits up to 0.1 s for a reply on the motor's
  `recv_id`, and raises `ConnectionError` listing any non-responding motors
  ("Check power (24V) and CAN wiring.") (`_handshake`, line 204).

### Simple command frames (`_send_simple_command`, line 279)

Sends `[0xFF]*7 + [command_byte]` to the motor's send id. Command bytes
(`tables.py`): `CAN_CMD_ENABLE=0xFC`, `CAN_CMD_DISABLE=0xFD`,
`CAN_CMD_SET_ZERO=0xFE`, `CAN_CMD_REFRESH=0xCC`. Used by:

- `enable_torque` / `disable_torque` (with optional `num_retry`) — lines 296/309.
- `torque_disabled()` context manager — disables then re-enables (line 322).
- `set_zero_position(motors=None)` — sets current pos as zero (line 335).
- `configure_motors()` — just enables each motor; "Damiao motors don't require
  much configuration in MIT mode" (line 271).

### MIT control — the 5-variable frame

This is the core of the Damiao path. The full MIT control frame
**(position, velocity, kp, kd, torque feed-forward)** is encoded in
`_encode_mit_packet` (line 429):

```python
position_rad = radians(position_degrees)
velocity_rad = radians(velocity_deg_per_sec)
pmax, vmax, tmax = MOTOR_LIMIT_PARAMS[motor_type]
kp_uint  = float_to_uint(kp,          *MIT_KP_RANGE, 12)   # 12-bit, range (0.0, 500.0)
kd_uint  = float_to_uint(kd,          *MIT_KD_RANGE, 12)   # 12-bit, range (0.0, 5.0)
q_uint   = float_to_uint(position_rad, -pmax, pmax, 16)    # 16-bit
dq_uint  = float_to_uint(velocity_rad, -vmax, vmax, 12)    # 12-bit
tau_uint = float_to_uint(torque,       -tmax, tmax, 12)    # 12-bit (torque feed-forward)
# packed into 8 bytes, layout at lines 454-462
```

- `_float_to_uint` clamps to `[x_min, x_max]` before encoding (line 530).
- Inputs/outputs use **degrees** externally; the encoder converts to radians.
- `MOTOR_LIMIT_PARAMS[motor_type]` provides `(pmax, vmax, tmax)` per model, so
  the uint scaling is per-motor-type correct (`tables.py:98`).

Two senders:

- `_mit_control(motor, kp, kd, position_deg, velocity_deg_s, torque)` (line 465)
  — single motor; sends one MIT frame then reads one reply and updates the
  cache.
- `_mit_control_batch(commands)` (line 492) — `commands` maps motor →
  `(kp, kd, position_deg, velocity_deg_s, torque)`. Sends **all** frames first,
  then collects replies via `_recv_all_responses`. This is the low-latency
  multi-motor path used by the reference follower.

> Both senders accept the full 5-tuple. To stream velocity and/or torque
> feed-forward you call `_mit_control` / `_mit_control_batch` directly with
> nonzero `velocity_deg_s` / `torque`. See §6 for how the high-level
> `sync_write`/`send_action` path uses these.

### Reads / state refresh

- `_refresh_motor` / `_batch_refresh` send the refresh frame
  (`arbitration_id=CAN_PARAM_ID=0x7FF`, data `[id&0xFF, id>>8, CAN_CMD_REFRESH, 0,0,0,0,0]`)
  and update the cache from replies (lines 342, 676).
- `_decode_motor_state` unpacks reply bytes into
  `(position_deg, velocity_deg_s, torque, temp_mos, temp_rotor)` (line 543).
- On a missing reply, `_batch_refresh` logs `"Packet drop: <motor> ... Using
  last known state."` and keeps the cached value (line 702) — reads never throw
  on a single dropped packet (but `read()` for a single motor **does** raise
  `ConnectionError` if the refresh returns nothing, line 595).

### High-level `read` / `write` / `sync_read` / `sync_write`

`data_name` string → behavior mapping (this is the contract the `Robot` sees):

| `data_name` | `read` / `sync_read` source | `write` / `sync_write` behavior |
|---|---|---|
| `Present_Position` | cached `position` (deg) | — |
| `Present_Velocity` | cached `velocity` (deg/s) | — |
| `Present_Torque`   | cached `torque` | — |
| `Temperature_MOS`  | cached `temp_mos` | — |
| `Temperature_Rotor`| cached `temp_rotor` | — |
| `Kp` | — | stores into `_gains[motor]["kp"]` (not sent immediately) |
| `Kd` | — | stores into `_gains[motor]["kd"]` (not sent immediately) |
| `Goal_Position` | — | sends MIT frame using cached `kp`/`kd`, **velocity=0.0, torque=0.0** |

(`read` line 586, `write` line 618, `sync_read` line 639, `sync_write` line 704.)

So through the generic bus contract, `Goal_Position` = position control with
the currently-cached stiffness/damping; velocity and torque feed-forward are
hardcoded to `0.0` on this path (lines 635, 728). Any other `data_name` to
`write` raises `ValueError("Writing <name> not supported in MIT mode")`.

- `sync_read_all_states(motors=None)` (line 655) — one refresh cycle, returns
  `{motor: {"position","velocity","torque","temp_mos","temp_rotor"}}`. Used by
  the reference follower's `get_observation` to read pos/vel/torque in a single
  CAN cycle instead of three separate reads.

### Calibration on Damiao

- `read_calibration` returns the in-memory `self.calibration` (or `{}`);
  `write_calibration` only caches in memory — "Damiao motors don't store
  calibration internally" (lines 747-758).
- `record_ranges_of_motion(motors=None)` (line 760) — disables torque, streams
  live positions (deg) while you move joints by hand, returns `(mins, maxes)`;
  raises if any joint's range `< 5` degrees. Re-enables torque at the end.
- `is_calibrated` is `bool(self.calibration)` (line 857).

### Motor tables (`motors/damiao/tables.py`)

- `MotorType` enum (line 21): `DM3507, DM4310, DM4310_48V, DM4340, DM4340_48V,
  DM6006, DM8006, DM8009, DM10010L, DM10010, DMH3510, DMH6215, DMG6220`.
- `MOTOR_LIMIT_PARAMS` (line 98): per-type `(PMAX rad, VMAX rad/s, TMAX N·m)`.
  E.g. `DM4310 = (12.5, 30, 10)`, `DM4340 = (12.5, 8, 28)`,
  `DM8009 = (12.5, 45, 54)`.
- `MIT_KP_RANGE = (0.0, 500.0)`, `MIT_KD_RANGE = (0.0, 5.0)` (lines 196-197).
- `AVAILABLE_BAUDRATES` 125 kbps … 5 Mbps; `DEFAULT_BAUDRATE = 1_000_000`
  (lines 149-161).
- `CAN_PARAM_ID = 0x7FF` (line 209); command bytes listed at lines 200-206.
- `ControlMode` enum exists (MIT=1, POS_VEL=2, VEL=3, TORQUE_POS=4, line 38) but
  the bus implementation sends MIT-mode frames; it does not switch control mode
  in code on the read/write path.
- `tables.py` also includes OpenArms reference constants (`OPENARMS_ARM_MOTOR_IDS`,
  `OPENARMS_DEFAULT_MOTOR_TYPES`, send/recv id pairs) at lines 169-193.

### Public export

`motors/damiao/__init__.py` exports `DamiaoMotorsBus` (and the table constants
via `tables`). Import as `from lerobot.motors.damiao import DamiaoMotorsBus`.

---

## 4. `lerobot-setup-can` (`scripts/lerobot_setup_can.py`)

Helper CLI for socketcan interfaces. `CANSetupConfig` fields: `mode`
("setup"/"test"/"speed"), `interfaces` (comma-separated, e.g. `"can0,can1"`),
`bitrate=1_000_000`, `data_bitrate=5_000_000`, `use_fd=True`,
`motor_ids=range(0x01,0x09)`, `timeout`, `speed_iterations`.

- `--mode=setup` runs `sudo ip link set <if> down`, then
  `sudo ip link set <if> type can bitrate <b> [dbitrate <db> fd on]`, then
  `up` (`setup_interface`, line 96). Requires the `ip` command and sudo.
- `--mode=test` opens `can.interface.Bus(channel=<if>, interface="socketcan",
  bitrate=..., [fd, data_bitrate])`, sends each motor an enable frame
  (`data=[0xFF]*7+[0xFC]`) and reports which respond (`test_interface`, line 162).
- `--mode=speed` measures enable→reply latency over `speed_iterations`
  (line 213).
- Guarded by `_can_available`; exits with an install hint if `python-can` is
  missing (line 340).

This script uses `socketcan` explicitly, so it is the tool to validate a real
or virtual CAN interface is up before running record/teleop.

---

## 5. Reference Damiao follower: `OpenArmFollower`

`robots/openarm_follower/openarm_follower.py` is the in-repo example of a
`Robot` built on `DamiaoMotorsBus`. Class docstring (line 39): "OpenArms
Follower Robot which uses CAN bus communication to control 7 DOF arm with a
gripper. The arm uses Damiao motors in MIT control mode." Use it as the template.

### Building the bus from config (`__init__`, line 48)

```python
motors = {}
for motor_name, (send_id, recv_id, motor_type_str) in config.motor_config.items():
    motor = Motor(send_id, motor_type_str, MotorNormMode.DEGREES)  # note: model field = motor_type_str
    motor.recv_id = recv_id
    motor.motor_type_str = motor_type_str
    motors[motor_name] = motor

self.bus = DamiaoMotorsBus(
    port=config.port,
    motors=motors,
    calibration=self.calibration,
    can_interface=config.can_interface,
    use_can_fd=config.use_can_fd,
    bitrate=config.can_bitrate,
    data_bitrate=config.can_data_bitrate if config.use_can_fd else None,
)
```

Note the exact construction used here (line 55-60): `Motor(send_id,
motor_type_str, MotorNormMode.DEGREES)` passes `motor_type_str` positionally as
the `model` field, then sets `motor.motor_type_str` and `motor.recv_id`
explicitly afterward. Damiao motors are always constructed with
`MotorNormMode.DEGREES`.

### Features (`_motors_ft`, line 90)

For each motor it always exposes `"<motor>.pos": float`. When
`config.use_velocity_and_torque` is `True`, it also exposes `"<motor>.vel"` and
`"<motor>.torque"`. `observation_features = {**_motors_ft, **_cameras_ft}`;
`action_features = _motors_ft` (lines 113-121). The dataset schema in recording
is derived from these (see §7), so enabling `use_velocity_and_torque` records
vel/torque as observation features.

### Connect / calibrate / configure

- `connect` (line 128): `bus.connect()` → optional `calibrate()` → connect
  cameras → `configure()` → if calibrated, `bus.set_zero_position()` →
  `bus.enable_torque()`.
- `calibrate` (line 165): sets current pose as zero (`bus.set_zero_position()`),
  then writes a default `MotorCalibration` per motor with `range_min=-90`,
  `range_max=90`, `homing_offset=0`, `drive_mode=0`, and saves it. (Damiao
  calibration is in-memory/JSON only, per §3.)
- `configure` (line 216): `with bus.torque_disabled(): bus.configure_motors()`.
- `setup_motors` raises `NotImplementedError` — "Motor ID configuration is
  typically done via manufacturer tools for CAN motors." (line 222).

### `get_observation` (line 227)

Calls `bus.sync_read_all_states()` once, then fills `"<motor>.pos"` (and
`.vel`/`.torque` if enabled) from the returned states, then reads cameras.

### `send_action` (line 267) — how the 5-tuple is actually populated

```python
def send_action(self, action, custom_kp=None, custom_kd=None):
    goal_pos = {k.removesuffix(".pos"): v for k, v in action.items() if k.endswith(".pos")}
    # 1. clip each joint to config.joint_limits[motor] (min,max in degrees)
    # 2. if config.max_relative_target is not None: read Present_Position and
    #    cap via ensure_safe_goal_position()
    # 3. build MIT commands: kp/kd from custom_* or config.position_kp/position_kd
    #    (per-joint list indexed by motor_index), velocity=0.0, torque=0.0
    commands[motor_name] = (kp, kd, position_degrees, 0.0, 0.0)
    self.bus._mit_control_batch(commands)
    return {f"{m}.pos": v for m, v in goal_pos.items()}
```

Key facts for this implementation:

- The action dict it consumes is `{"<motor>.pos": float}` only.
- It calls `bus._mit_control_batch` directly (not `sync_write`), so it controls
  kp/kd per joint from config, but still passes **velocity=0.0, torque=0.0**
  (line 340).
- `custom_kp` / `custom_kd` are optional per-call overrides, but the standard
  `record_loop`/`teleop_loop` call `send_action(action)` with no extra args
  (see §7/§8), so those overrides are not exercised by the stock loops.
- Joint-limit clipping (`config.joint_limits`) happens before sending; then the
  optional relative-target cap via `ensure_safe_goal_position`
  (`robots/utils.py`, caps `|goal-present|` per motor).

### `OpenArmFollowerConfig` (`config_openarm_follower.py`)

Registered as `openarm_follower` via `@RobotConfig.register_subclass`. Fields
relevant to the CAN/Damiao path (with their in-repo defaults):

- `port: str` (CAN interface, e.g. `"can1"`)
- `side: str | None` ("left"/"right" load preset `joint_limits`, else CLI)
- `can_interface: str = "socketcan"`
- `use_can_fd: bool = True`, `can_bitrate: int = 1_000_000`,
  `can_data_bitrate: int = 5_000_000`
- `disable_torque_on_disconnect: bool = True`
- `use_velocity_and_torque: bool = False`
- `max_relative_target: float | dict[str, float] | None = None`
- `cameras: dict[str, CameraConfig] = {}`
- `motor_config: dict[str, tuple[int,int,str]]` — maps each joint to
  `(send_can_id, recv_can_id, motor_type_str)`; default is the 7-DOF + gripper
  OpenArms map, e.g. `"joint_1": (0x01, 0x11, "dm8009")`,
  `"joint_5": (0x05, 0x15, "dm4310")`, `"gripper": (0x08, 0x18, "dm4310")`.
- `position_kp: list[float]` and `position_kd: list[float]` — 8-element lists
  indexed `[joint_1..joint_7, gripper]`; defaults `position_kp = [240,240,240,
  240,24,31,25,25]`, `position_kd = [5,5,3,5,0.3,0.3,0.3,0.3]`.
- `joint_limits: dict[str, tuple[float,float]]` — per-joint (min,max) degrees;
  small defaults for safety, overridable by `side` or CLI.

This config is the concrete pattern for declaring CAN ids, motor types, and MIT
gains for a Damiao follower.

---

## 6. Driving the full 5-variable frame (velocity / torque feed-forward)

Summary of what the codebase does today, so the implementation choice is clear:

- The encoder and transport for all five MIT variables exist and are exercised:
  `_encode_mit_packet` packs `(kp, kd, position, velocity, torque)` and
  `_mit_control` / `_mit_control_batch` send them (`damiao.py:429/465/492`).
- The two high-level paths that the stock control loops reach —
  `DamiaoMotorsBus.sync_write("Goal_Position", ...)` and
  `OpenArmFollower.send_action(...)` — both build the tuple with
  **velocity=0.0 and torque=0.0** (`damiao.py:728`, `openarm_follower.py:340`),
  with kp/kd from the gains cache or config.
- There is no `data_name` in `DamiaoMotorsBus.write`/`sync_write` that carries
  velocity or torque feed-forward; only `Kp`, `Kd`, and `Goal_Position` are
  handled, anything else raises (`damiao.py:636`).

Therefore, to command nonzero velocity / torque feed-forward per tick you must
call `bus._mit_control(...)` or `bus._mit_control_batch({motor: (kp, kd, pos,
vel, tau)})` directly — e.g. from a custom `Robot.send_action` that expands an
action dict carrying those fields. (The stock `OpenArmFollower` is the template;
it already calls `_mit_control_batch`, only with the last two terms zeroed.)

---

## 7. Recording path (`scripts/lerobot_record.py`)

CLI entry: `lerobot-record` → `main()` → `record()` (line 347).

`RecordConfig` (line 163): `robot: RobotConfig`, `dataset: DatasetRecordConfig`,
`teleop: TeleoperatorConfig | None` (a teleop is **required** —
`__post_init__` raises if `None`, line 182), plus `display_data`, `play_sounds`,
`resume`, etc.

`record()` flow:

1. `robot = make_robot_from_config(cfg.robot)`,
   `teleop = make_teleoperator_from_config(cfg.teleop)` (lines 364-365).
2. If no processors are passed, falls back to identity-style defaults from
   `make_default_processors()` (line 373).
3. Builds `dataset_features` by running `robot.action_features` and
   `robot.observation_features` through the processor pipelines
   (`aggregate_pipeline_dataset_features`, lines 378-391). **The dataset schema
   is derived from the robot's feature dicts** — so for a Damiao follower the
   recorded action/observation keys are exactly the `"<motor>.pos"` (and
   `.vel`/`.torque` if `use_velocity_and_torque`) keys from §5.
4. Creates or resumes a `LeRobotDataset`, `robot.connect()`,
   `teleop.connect()`, starts a keyboard listener producing an `events` dict
   (`exit_early`, `rerecord_episode`, `stop_recording`) (lines 423-444).
5. Loops over `cfg.dataset.num_episodes`, calling `record_loop` to record, then
   again (without a dataset) for the reset window, handling re-record and
   `save_episode()` (lines 451-498).
6. `finally`: disconnect robot/teleop, finalize/optionally push dataset.

### `record_loop` (line 217)

Per-tick data flow (documented in the file at lines 191-214):

```
robot.get_observation()                  -> obs
robot_observation_processor(obs)         -> obs_processed   (default identity)
teleop.get_action()                      -> act             (leader: see §9)
teleop_action_processor((act, obs))      -> action_values   (saved as ACTION)
robot_action_processor((act, obs))       -> robot_action_to_send
robot.send_action(robot_action_to_send)  -> robot executes  (Damiao: see §5/§6)
dataset.add_frame({**observation_frame, **action_frame, "task": single_task})
precise_sleep(control_interval - dt)     -> hold fps
```

- `control_interval = 1 / fps`; `dataset.fps` must equal `fps` or it raises
  (line 238).
- It warns if a loop iteration runs slower than the target fps (line 337).
- `single_task` is stored on every frame.
- Note `record_loop` calls `robot.send_action(robot_action_to_send)` with the
  single positional arg — the Damiao follower's `custom_kp`/`custom_kd` are not
  supplied by this loop.

---

## 8. Teleoperation path (`scripts/lerobot_teleoperate.py`)

CLI entry: `lerobot-teleoperate` → `teleoperate()` (line 213). Same read→process
→send loop as recording but with **no dataset** — useful to verify the
leader→follower link before recording.

`TeleoperateConfig` (line 114): `teleop: TeleoperatorConfig`,
`robot: RobotConfig`, `fps: int = 60`, `teleop_time_s`, `display_data`, etc.

`teleop_loop` (line 132) per tick:

```
obs = robot.get_observation()
raw_action = teleop.get_action()
teleop_action = teleop_action_processor((raw_action, obs))
robot_action_to_send = robot_action_processor((teleop_action, obs))
robot.send_action(robot_action_to_send)
precise_sleep(max(1/fps - dt, 0))
```

Processors come from `make_default_processors()` (line 227). The leader's
`get_action()` keys must line up with the follower's `send_action` expectations
(both sides use `"<motor>.pos"`); see §9 and §10.

---

## 9. Leader teleoperator: Feetech (`teleoperators/so_leader/so_leader.py`)

The leader arm uses Feetech motors over serial. `SOLeader` (line 33; aliases
`SO100Leader`, `SO101Leader`, line 166) is the in-repo reference.

### Bus construction (line 39)

```python
norm_mode_body = MotorNormMode.DEGREES if config.use_degrees else MotorNormMode.RANGE_M100_100
self.bus = FeetechMotorsBus(
    port=config.port,
    motors={
        "shoulder_pan":  Motor(1, "sts3215", norm_mode_body),
        "shoulder_lift": Motor(2, "sts3215", norm_mode_body),
        "elbow_flex":    Motor(3, "sts3215", norm_mode_body),
        "wrist_flex":    Motor(4, "sts3215", norm_mode_body),
        "wrist_roll":    Motor(5, "sts3215", norm_mode_body),
        "gripper":       Motor(6, "sts3215", MotorNormMode.RANGE_0_100),
    },
    calibration=self.calibration,
)
```

Contrast with Damiao: Feetech motors use `id` + `model` ("sts3215") +
`norm_mode`; no `motor_type_str` / `recv_id` (those are CAN-only).

### Interface

- `action_features` / `feedback_features` = `{"<motor>.pos": float}` (line 56).
- `get_action()` (line 145): `bus.sync_read("Present_Position")` →
  `{f"{motor}.pos": val}`. The leader's torque is left off so it can be moved
  by hand (the bus is configured with `Operating_Mode = POSITION` and torque
  disabled in `configure`/`calibrate`).
- `connect` (line 68): `bus.connect()` → optional `calibrate()` → `configure()`.
- `calibrate` (line 84): disables torque, sets `Operating_Mode=POSITION` per
  motor, `bus.set_half_turn_homings()`, `bus.record_ranges_of_motion(...)`
  (treats `wrist_roll` as a full-turn motor, range 0..4095), builds a
  `MotorCalibration` per motor, writes + saves it.
- `configure` (line 127): disable torque, `bus.configure_motors()`,
  `Operating_Mode=POSITION` per motor.
- `send_feedback` (line 154): `bus.sync_write("Goal_Position", goals)` — only
  meaningful if the leader has torque and you drive it; not used by the stock
  record/teleop loops for SO leader.

### `SOLeaderConfig` (`config_so_leader.py`)

Registered as `so101_leader` / `so100_leader`. Fields: `port: str`,
`use_degrees: bool = True`.

### Feetech bus (`motors/feetech/feetech.py`)

`FeetechMotorsBus(SerialMotorsBus)` (line 89). Methods the leader relies on,
present here: `configure_motors` (line 209), `is_calibrated` (line 228),
`write_calibration` (line 268), `disable_torque` (line 291), plus the inherited
serial `connect` / `sync_read` / `sync_write` / `set_half_turn_homings` /
`record_ranges_of_motion`. `default_baudrate` is set on the class (line 97).

---

## 10. Leader→Follower key alignment

- Feetech leader `get_action()` emits `{"shoulder_pan.pos", "shoulder_lift.pos",
  ..., "gripper.pos"}` (6 keys, SO arm).
- Damiao follower `send_action()` consumes `{"<motor>.pos": ...}` and ignores
  keys that don't end in `.pos`, mapping the rest to motor names by stripping
  `.pos` (`openarm_follower.py:288`).
- For direct teleop the **motor names must match** between leader and follower
  (the follower looks up each stripped name in `config.joint_limits` /
  `motor_index` / `bus.motors`). The SO leader uses names
  `shoulder_pan/.../gripper`; the OpenArm follower uses `joint_1.../gripper`.
  If the leader and follower joint names differ, a processor step (the
  `teleop_action_processor` / `robot_action_processor` in the loops) or matching
  motor naming is required to bridge them. This naming bridge is the integration
  point for a Feetech-leader → Damiao-follower setup.

---

## 11. Config & instantiation (draccus)

- `RobotConfig` (`robots/config.py`) and `TeleoperatorConfig`
  (`teleoperators/config.py`) are `draccus.ChoiceRegistry` ABCs. Subclasses
  register a name via `@RobotConfig.register_subclass("openarm_follower")` /
  `@TeleoperatorConfig.register_subclass("so101_leader")`. `config.type`
  returns that registered name.
- CLI flags map onto config fields by dotted path, e.g.
  `--robot.type=openarm_follower --robot.port=can0 --robot.use_can_fd=true
  --teleop.type=so101_leader --teleop.port=/dev/ttyACM0`.
- `RobotConfig.__post_init__` requires `width`/`height`/`fps` on any configured
  camera (`robots/config.py`).
- `make_robot_from_config` (`robots/utils.py:25`) dispatches on `config.type`;
  `openarm_follower` → `OpenArmFollower`. `make_teleoperator_from_config`
  (`teleoperators/utils.py:36`) does the same for teleop. Unknown types fall
  back to a plugin loader (`make_device_from_device_class`).
- A new robot/teleop type is registered by importing its module in the relevant
  factory and in the script import blocks (`lerobot_record.py` / `lerobot_teleoperate.py`
  import each robot/teleop subpackage so draccus sees the registered subclasses).

---

## 12. Abstractions a custom device must implement

### `Robot` ABC (`robots/robot.py`)

Set `config_class` and `name`. Implement: `observation_features`,
`action_features`, `is_connected`, `connect(calibrate=True)`, `is_calibrated`,
`calibrate`, `configure`, `get_observation() -> RobotObservation`,
`send_action(action) -> RobotAction`, `disconnect`. Base provides context-
manager connect/disconnect, `__del__` safety net, and JSON calibration
load/save keyed by `id`.

### `Teleoperator` ABC (`teleoperators/teleoperator.py`)

Set `config_class` and `name`. Implement: `action_features`,
`feedback_features`, `is_connected`, `connect(calibrate=True)`,
`is_calibrated`, `calibrate`, `configure`, `get_action() -> RobotAction`,
`send_feedback(feedback)`, `disconnect`. Same base-class lifecycle/calibration
behavior as `Robot`.

---

## 13. Quick reference — key files

| Concern | File |
|---|---|
| CAN bus, MIT frames, Damiao | `motors/damiao/damiao.py` |
| Damiao motor types / limits / CAN constants | `motors/damiao/tables.py` |
| Reference Damiao-over-CAN follower | `robots/openarm_follower/openarm_follower.py` |
| Damiao follower config (CAN ids, gains, limits) | `robots/openarm_follower/config_openarm_follower.py` |
| CAN interface setup/test CLI | `scripts/lerobot_setup_can.py` |
| Feetech leader | `teleoperators/so_leader/so_leader.py` |
| Feetech serial bus | `motors/feetech/feetech.py` |
| Motor / calibration / bus base types | `motors/motors_bus.py` |
| Recording loop | `scripts/lerobot_record.py` |
| Teleoperation loop | `scripts/lerobot_teleoperate.py` |
| Robot / Teleoperator ABCs | `robots/robot.py`, `teleoperators/teleoperator.py` |
| Config bases + factories | `robots/config.py`, `robots/utils.py`, `teleoperators/config.py`, `teleoperators/utils.py` |
