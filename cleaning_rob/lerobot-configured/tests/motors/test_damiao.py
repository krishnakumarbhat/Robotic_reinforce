"""Minimal test script for Damiao motor with ID 3."""

import time

import pytest

from lerobot.utils.import_utils import _can_available

if not _can_available:
    pytest.skip("python-can not available", allow_module_level=True)

from lerobot.motors import Motor
from lerobot.motors.damiao import DamiaoMotorsBus


@pytest.mark.skip(reason="Requires physical Damiao motor and CAN interface")
def test_damiao_motor():
    motors = {
        "joint_3": Motor(
            id=0x03,
            model="damiao",
            norm_mode="degrees",
            motor_type_str="dm4310",
            recv_id=0x13,
        ),
    }

    bus = DamiaoMotorsBus(port="can0", motors=motors)

    try:
        print("Connecting...")
        bus.connect()
        print("✓ Connected")

        print("Enabling torque...")
        bus.enable_torque()
        print("✓ Torque enabled")

        print("Reading all states...")
        states = bus.sync_read_all_states()
        print(f"✓ States: {states}")

        print("Reading position...")
        positions = bus.sync_read("Present_Position")
        print(f"✓ Position: {positions}")

        print("Testing MIT control batch...")
        current_pos = states["joint_3"]["position"]
        commands = {"joint_3": (10.0, 0.5, current_pos, 0.0, 0.0)}
        bus._mit_control_batch(commands)
        print("✓ MIT control batch sent")

        print("Disabling torque...")
        bus.disable_torque()
        print("✓ Torque disabled")

        print("Setting zero position...")
        bus.set_zero_position()
        print("✓ Zero position set")

    finally:
        print("Disconnecting...")
        bus.disconnect(disable_torque=True)
        print("✓ Disconnected")


if __name__ == "__main__":
    test_damiao_motor()


class _FakeDamiaoMotors:
    """Answers refresh and enable requests on a python-can virtual bus, like Damiao firmware would.

    Each motor replies on its own feedback id after `delay_s`, with data[0] = (status << 4) | id.
    """

    def __init__(
        self, channel: str, feedback_ids: dict[int, int], positions_raw: dict[int, int], delay_s: float
    ):
        import threading

        import can

        self.bus = can.interface.Bus(channel=channel, interface="virtual")
        self.feedback_ids = feedback_ids
        self.positions_raw = positions_raw
        self.delay_s = delay_s
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _reply(self, motor_id: int, status: int) -> None:
        import can

        q = self.positions_raw[motor_id]
        data = [(status << 4) | motor_id, q >> 8, q & 0xFF, 0x7F, 0xF7, 0xFF, 0x1F, 0x1E]
        time.sleep(self.delay_s)
        self.bus.send(
            can.Message(arbitration_id=self.feedback_ids[motor_id], data=data, is_extended_id=False)
        )

    def _run(self) -> None:
        while not self._stop.is_set():
            msg = self.bus.recv(timeout=0.01)
            if msg is None:
                continue
            if msg.arbitration_id == 0x7FF and msg.data[2] == 0xCC and msg.data[0] in self.feedback_ids:
                self._reply(msg.data[0], status=0)
            elif msg.arbitration_id in self.feedback_ids and msg.data[7] in (0xFC, 0xFD):
                self._reply(msg.arbitration_id, status=1 if msg.data[7] == 0xFC else 0)

    def close(self) -> None:
        self._stop.set()
        self._thread.join()
        self.bus.shutdown()


def _make_bus(channel: str, response_timeout_s: float | None) -> DamiaoMotorsBus:
    motors = {
        name: Motor(id=i, model="damiao", norm_mode="degrees", motor_type_str="dm4310", recv_id=0x10 | i)
        for i, name in ((1, "joint_1"), (2, "joint_2"))
    }
    return DamiaoMotorsBus(
        port=channel,
        motors=motors,
        can_interface="virtual",
        use_can_fd=False,
        response_timeout_s=response_timeout_s,
    )


def test_feedback_on_unconfigured_id_is_matched_by_id_nibble():
    # joint_1 answers on the factory-default Master ID 0x000 rather than its configured 0x11.
    fake = _FakeDamiaoMotors(
        "nibble", feedback_ids={1: 0x000, 2: 0x012}, positions_raw={1: 0x9000, 2: 0x7000}, delay_s=0.0
    )
    bus = _make_bus("nibble", response_timeout_s=0.2)
    try:
        bus.connect()
        positions = bus.sync_read("Present_Position")
    finally:
        if bus.is_connected:
            bus.disconnect(disable_torque=False)
        fake.close()
    assert positions["joint_1"] > 0 > positions["joint_2"]


def test_command_frames_from_another_controller_are_not_feedback():
    import can

    bus = _make_bus("foreign", response_timeout_s=None)
    # An MIT command to motor 1 from another process: data[0] is a position byte, not an id.
    command = can.Message(arbitration_id=0x001, data=[0x01, 0, 0, 0, 0, 0, 0, 0], is_extended_id=False)
    refresh = can.Message(arbitration_id=0x7FF, data=[0x01, 0, 0xCC, 0, 0, 0, 0, 0], is_extended_id=False)
    assert bus._response_motor(command) is None
    assert bus._response_motor(refresh) is None


def test_response_timeout_waits_for_slow_replies():
    delay_s = 0.03  # longer than the built-in 10 ms batch window
    fake = _FakeDamiaoMotors(
        "slow", feedback_ids={1: 0x011, 2: 0x012}, positions_raw={1: 0x9000, 2: 0x7000}, delay_s=delay_s
    )
    bus = _make_bus("slow", response_timeout_s=0.5)
    try:
        bus.connect()
        fake.positions_raw[1] = 0x6000
        positions = bus.sync_read("Present_Position")
    finally:
        if bus.is_connected:
            bus.disconnect(disable_torque=False)
        fake.close()
    # The reading reflects the latest position, not one left over from the handshake.
    assert positions["joint_1"] < 0
