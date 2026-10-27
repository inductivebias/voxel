import asyncio
from types import SimpleNamespace
from typing import Any

import pytest

pytest.importorskip("zaber_motion", reason="needs the vxl-drivers `zaber` extra")

from vxl_drivers.axes import zaber
from vxl_drivers.axes.zaber_probe import probe_zaber
from zaber_motion.exceptions import CommandFailedException, CommandFailedExceptionData, RequestTimeoutException

from rigup import DeviceConfig, Rig, RigConfig
from vxl.devices.axes.continuous import StepMode, TriggerMode, TTLStepperConfig
from vxl.devices.axes.continuous.handle import ContinuousAxisHandle

NATIVE_PER_UM = 10
TTL_CONFIG = {"ttl_input": 1, "ttl_trigger": 1, "ttl_status_output": 1, "ttl_status_trigger": 2}
RELATIVE = TTLStepperConfig(step_mode=StepMode.RELATIVE)
NATIVE = zaber.Units.NATIVE
S = zaber.SettingConstants
# Settings the fake stores in micrometres (or per second); natively they are NATIVE_PER_UM times larger.
# The rest are unitless and stored natively.
SCALED = {
    S.POS,
    S.LIMIT_MIN,
    S.LIMIT_MAX,
    S.LIMIT_HOME_PRESET,
    S.MAXSPEED,
    S.ACCEL,
    S.MOTION_ACCELONLY,
    S.MOTION_DECELONLY,
}


class FakeSettings:
    def __init__(self, values: dict[str, float]) -> None:
        self.values = values
        self.set_calls: list[tuple[str, float, object]] = []
        self.get_calls: list[str] = []
        self.conversions: list[tuple[str, float, object]] = []
        self.accel_sets_decel = True
        self.fail_set: set[str] = set()

    def get(self, setting: str, unit: object = NATIVE) -> float:
        self.get_calls.append(setting)
        # Like the controller, the accel alias reads back acceleration only.
        value = self.values[S.MOTION_ACCELONLY if setting == S.ACCEL else setting]
        return value * NATIVE_PER_UM if unit is NATIVE and setting in SCALED else value

    def set(self, setting: str, value: float, unit: object = NATIVE) -> None:
        self.set_calls.append((setting, value, unit))
        if setting in self.fail_set:
            raise RuntimeError(f"{setting} write failed")
        stored = value / NATIVE_PER_UM if unit is NATIVE and setting in SCALED else value
        if setting == S.ACCEL:
            self.values[S.MOTION_ACCELONLY] = stored
            if self.accel_sets_decel:
                self.values[S.MOTION_DECELONLY] = stored
        else:
            self.values[setting] = stored

    def has_setting(self, setting: str) -> bool:
        return setting in self.values

    def convert_to_native_units(self, setting: str, value: float, unit: object, **options: bool) -> float:
        self.conversions.append((setting, value, unit))
        native = value * NATIVE_PER_UM
        return float(round(native)) if options.get("round") else native

    def convert_from_native_units(self, setting: str, value: float, unit: object) -> float:
        del setting, unit
        return value / NATIVE_PER_UM


class FakeTrigger:
    def __init__(self) -> None:
        self.enabled = False
        self.condition: tuple[object, ...] | None = None
        self.actions: dict[object, tuple[int, str]] = {}
        self.cleared: list[object] = []
        self.fail_disable = False

    def disable(self) -> None:
        if self.fail_disable:
            raise RuntimeError("trigger disable failed")
        self.enabled = False

    def enable(self, count: int = 0) -> None:
        assert count == 0, "steps must repeat until disarmed"
        self.enabled = True

    def fire_when_io(self, port_type: object, channel: int, condition: object, value: float) -> None:
        self.condition = ("io", port_type, channel, condition, value)

    def fire_when_setting(self, axis: int, setting: str, condition: object, value: float) -> None:
        self.condition = ("setting", axis, setting, condition, value)

    def on_fire(self, action: object, axis: int, command: str) -> None:
        self.actions[action] = (axis, command)

    def clear_action(self, action: object) -> None:
        self.cleared.append(action)
        if action is zaber.TriggerAction.ALL:
            self.actions.clear()
        else:
            self.actions.pop(action, None)


class FakeTriggers:
    def __init__(self, count: int = 6) -> None:
        self.by_number = {number: FakeTrigger() for number in range(1, count + 1)}

    def get_number_of_triggers(self) -> int:
        return len(self.by_number)

    def get_trigger(self, number: int) -> FakeTrigger:
        return self.by_number[number]

    def get_enabled_states(self) -> list[SimpleNamespace]:
        return [SimpleNamespace(enabled=trigger.enabled) for trigger in self.by_number.values()]


class FakeIO:
    def __init__(self) -> None:
        self.digital_inputs = [False, False, False, False]
        self.digital_outputs: dict[int, object] = {}

    def get_channels_info(self) -> SimpleNamespace:
        return SimpleNamespace(number_digital_inputs=4, number_digital_outputs=4)

    def get_digital_input(self, channel: int) -> bool:
        return self.digital_inputs[channel - 1]

    def set_digital_output(self, channel: int, value: object) -> None:
        self.digital_outputs[channel] = value


class FakeAxis:
    def __init__(self, axis_number: int, position: float = 25_000.0) -> None:
        self.axis_number = axis_number
        self.axis_type = zaber.AxisType.LINEAR
        self.device: FakeDevice | None = None
        self.identity = SimpleNamespace(peripheral_name="X-LSM100A", peripheral_serial_number=axis_number + 1000)
        self.settings = FakeSettings(
            {
                zaber.SettingConstants.LIMIT_MIN: 0.0,
                zaber.SettingConstants.LIMIT_MAX: 100_000.0,
                zaber.SettingConstants.MAXSPEED: 10_000.0,
                zaber.SettingConstants.MOTION_ACCELONLY: 20_000.0,
                zaber.SettingConstants.MOTION_DECELONLY: 20_000.0,
                zaber.SettingConstants.LIMIT_HOME_PRESET: 0.0,
                zaber.SettingConstants.POS: position,
                zaber.SettingConstants.ENCODER_PORT_DEFAULT: 2,
                zaber.SettingConstants.CLOOP_ENABLE: 1,
                zaber.SettingConstants.MOTION_ACCEL_RAMPTIME: 0,
                "encoder.2.out.enable": 0,
                "encoder.2.out.interpolation": 4,
                "encoder.2.out.width": 25,
            }
        )
        self.position = position
        self.encoder_error = 0.0  # Measured encoder position minus the planned position.
        self.busy = False
        self.homed = True
        self.flags: set[str] = set()
        self.warnings = SimpleNamespace(get_flags=lambda: set(self.flags))
        self.move_absolute_calls: list[tuple[float, object, bool]] = []
        self.move_relative_calls: list[tuple[float, object, bool]] = []
        self.home_calls: list[bool] = []
        self.stop_calls: list[bool] = []

    def get_position(self, unit: object) -> float:
        del unit
        return self.position

    def generic_command(self, command: str) -> SimpleNamespace:
        measured = self.position + self.encoder_error
        positions = {"pos": self.position, "encoder.pos": measured, "encoder.2.pos": measured}
        names = command.removeprefix("get ").split()
        return SimpleNamespace(data=" ; ".join(str(round(positions[name] * NATIVE_PER_UM)) for name in names))

    def _reject_outside_limits(self, command: str, target: float) -> None:
        limits = self.settings.values
        if limits[zaber.SettingConstants.LIMIT_MIN] <= target <= limits[zaber.SettingConstants.LIMIT_MAX]:
            return
        details = CommandFailedExceptionData(
            command=command,
            response_data="BADDATA",
            reply_flag="RJ",
            status="IDLE",
            warning_flag="--",
            device_address=1,
            axis_number=self.axis_number,
            id=0,
        )
        raise CommandFailedException(f"Command {command} rejected: BADDATA", details)

    def move_absolute(self, position: float, unit: object, wait_until_idle: bool) -> None:
        self._reject_outside_limits("move abs", position)
        self.move_absolute_calls.append((position, unit, wait_until_idle))
        self.position = position
        self.busy = not wait_until_idle

    def move_relative(self, delta: float, unit: object, wait_until_idle: bool) -> None:
        self._reject_outside_limits("move rel", self.position + delta)
        self.move_relative_calls.append((delta, unit, wait_until_idle))
        self.position += delta
        self.busy = not wait_until_idle

    def home(self, wait_until_idle: bool) -> None:
        self.home_calls.append(wait_until_idle)
        self.position = self.settings.values[zaber.SettingConstants.LIMIT_HOME_PRESET]
        self.busy = not wait_until_idle

    def stop(self, wait_until_idle: bool) -> None:
        self.stop_calls.append(wait_until_idle)
        self.busy = False

    def is_busy(self) -> bool:
        return self.busy

    def is_homed(self) -> bool:
        return self.homed


class FakeDevice:
    def __init__(self, serial: int, axes: list[FakeAxis]) -> None:
        self.serial_number = serial
        self.name = "X-MCC4"
        self.firmware_version = "7.42"
        self.axis_count = len(axes)
        self._axes = {axis.axis_number: axis for axis in axes}
        for axis in axes:
            axis.device = self
        self.triggers = FakeTriggers()
        self.io = FakeIO()

    def get_axis(self, axis_number: int) -> FakeAxis:
        return self._axes[axis_number]


class FakeConnection:
    def __init__(self, devices: list[FakeDevice]) -> None:
        self.devices = devices
        self.default_request_timeout = 1000
        self.closed = False

    def detect_devices(self) -> list[FakeDevice]:
        return self.devices

    def close(self) -> None:
        self.closed = True


def install_connection(monkeypatch: pytest.MonkeyPatch, connection: FakeConnection) -> None:
    def open_connection(**_kwargs: object) -> FakeConnection:
        return connection

    monkeypatch.setattr(zaber, "open_zaber_connection", open_connection)


def test_probe_zaber_reads_identity_without_axis_writes_and_closes_connection(monkeypatch: pytest.MonkeyPatch) -> None:
    axis = FakeAxis(1)
    connection = FakeConnection([FakeDevice(12345, [axis])])
    install_connection(monkeypatch, connection)
    result = probe_zaber(serial=12345, host="controller.invalid")
    assert result.serial == 12345
    assert len(result.axes) == 1
    assert result.axes[0].position_um == 25000
    assert not axis.settings.set_calls
    assert not axis.move_absolute_calls
    assert not axis.move_relative_calls
    assert not axis.home_calls
    assert connection.closed


def test_probe_zaber_wrong_identity_fails_and_closes_connection(monkeypatch: pytest.MonkeyPatch) -> None:
    connection = FakeConnection([FakeDevice(12345, [FakeAxis(1)])])
    install_connection(monkeypatch, connection)
    with pytest.raises(zaber.ZaberDiscoveryError, match="was not detected"):
        probe_zaber(serial=67890, host="controller.invalid")
    assert connection.closed


def zaber_rig_config(**axes: dict[str, Any]) -> RigConfig:
    """Build a rig with one serial ZaberHub and one ZaberLinearAxis per keyword on device 12345."""
    devices = {
        "zaber_controller": DeviceConfig(
            target="vxl_drivers.axes.zaber.ZaberHub",
            init={"port": "/dev/serial/by-id/zaber"},
        )
    }
    for uid, init in axes.items():
        devices[uid] = DeviceConfig(
            target="vxl_drivers.axes.zaber.ZaberLinearAxis",
            init={"hub": "zaber_controller", "device_serial": 12345, **init},
        )
    return RigConfig(devices=devices)


@pytest.fixture
def controller(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[zaber.ZaberHub, FakeConnection, FakeAxis, FakeAxis]:
    x_axis = FakeAxis(1)
    y_axis = FakeAxis(2, position=40_000.0)
    connection = FakeConnection([FakeDevice(12345, [x_axis, y_axis])])
    install_connection(monkeypatch, connection)
    hub = zaber.ZaberHub("zaber_controller", port="/dev/serial/by-id/zaber", timeout_s=2.5)
    return hub, connection, x_axis, y_axis


def test_zaber_hub_detects_devices_by_serial_and_reserves_each_axis_once(
    controller: tuple[zaber.ZaberHub, FakeConnection, FakeAxis, FakeAxis],
) -> None:
    hub, connection, x_axis, _y_axis = controller

    assert hub.device_serials == (12345,)
    assert connection.default_request_timeout == 2500
    assert hub.reserve_axis(12345, 1) is x_axis
    with pytest.raises(ValueError, match="already reserved"):
        hub.reserve_axis(12345, 1)
    with pytest.raises(zaber.ZaberDiscoveryError, match="device serial 99999"):
        hub.reserve_axis(99999, 1)
    with pytest.raises(zaber.ZaberDiscoveryError, match="axis 3"):
        hub.reserve_axis(12345, 3)

    hub.release_axis(12345, 1)
    assert hub.reserve_axis(12345, 1) is x_axis
    hub.release_axis(12345, 1)
    hub.close()
    assert connection.closed


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({}, "exactly one of port or host"),
        (
            {"port": "/dev/zaber", "host": "zaber-12345.local"},
            "exactly one of port or host",
        ),
        ({"port": ""}, "port must not be empty"),
        ({"host": ""}, "host must not be empty"),
        ({"host": "zaber-12345.local", "tcp_port": 0}, "tcp_port must be in"),
        ({"port": "/dev/zaber", "timeout_s": 0}, "timeout_s must be positive"),
    ],
)
def test_zaber_hub_invalid_connection_configuration_fails_before_open(
    monkeypatch: pytest.MonkeyPatch, kwargs: dict[str, Any], message: str
) -> None:
    def fail_open(**_kwargs: object) -> None:
        raise AssertionError("invalid configuration opened the SDK")

    monkeypatch.setattr(zaber, "open_zaber_connection", fail_open)

    with pytest.raises(ValueError, match=message):
        zaber.ZaberHub("zaber_controller", **kwargs)


def test_zaber_hub_opens_serial_or_device_only_tcp_connection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    opened: list[tuple[object, ...]] = []

    def opener(kind: str) -> Any:
        def open_connection(*args: object, **kwargs: object) -> FakeConnection:
            opened.append((kind, *args, *kwargs.items()))
            return FakeConnection([FakeDevice(12345, [FakeAxis(1)])])

        return open_connection

    monkeypatch.setattr(zaber.Connection, "open_serial_port", opener("serial"))
    monkeypatch.setattr(zaber.Connection, "open_tcp", opener("tcp"))

    zaber.ZaberHub("serial_hub", port="/dev/zaber").close()
    zaber.ZaberHub("tcp_hub", host="zaber-12345.local").close()

    assert opened == [
        ("serial", "/dev/zaber", ("test_port", True)),
        ("tcp", "zaber-12345.local", zaber.Connection.TCP_PORT_DEVICE_ONLY),
    ]


def test_zaber_hub_defers_connection_close_until_reserved_axes_close(
    controller: tuple[zaber.ZaberHub, FakeConnection, FakeAxis, FakeAxis],
) -> None:
    hub, connection, _x_axis, _y_axis = controller
    axis = zaber.ZaberLinearAxis("x_axis", hub=hub, device_serial=12345, axis_number=1)

    hub.close()
    assert not connection.closed

    axis.close()
    assert connection.closed


def test_zaber_axis_reads_state_without_changing_hardware(
    controller: tuple[zaber.ZaberHub, FakeConnection, FakeAxis, FakeAxis],
) -> None:
    hub, _connection, sdk_axis, _y_axis = controller
    axis = zaber.ZaberLinearAxis("x_axis", hub=hub, device_serial=12345)

    assert axis.position == 25_000.0
    assert (axis.lower_limit, axis.upper_limit) == (0.0, 100_000.0)
    assert (axis.speed, axis.acceleration, axis.home) == (10_000.0, 20_000.0, 0.0)
    assert (axis.device_name, axis.firmware_version) == ("X-MCC4", "7.42")
    assert axis.peripheral_name == "X-LSM100A"
    assert axis.is_homed

    sdk_axis.position = 101_000.0
    sdk_axis.homed = False
    # Voxel's generic numeric position descriptor clamps its presentation to
    # the declared limits; the diagnostic property must retain the device value.
    assert axis.position == 100_000.0
    assert axis.device_position == 101_000.0
    assert not axis.position_within_limits
    assert not axis.is_homed
    assert not sdk_axis.move_absolute_calls
    assert not sdk_axis.move_relative_calls
    assert not sdk_axis.home_calls
    assert not sdk_axis.settings.set_calls

    axis.close()
    hub.close()


def test_zaber_axis_rereads_limits_at_most_once_a_second_except_after_its_own_writes(
    controller: tuple[zaber.ZaberHub, FakeConnection, FakeAxis, FakeAxis],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hub, _connection, sdk_axis, _y_axis = controller
    axis = zaber.ZaberLinearAxis("x_axis", hub=hub, device_serial=12345)
    now = [100.0]
    monkeypatch.setattr(zaber.time, "monotonic", lambda: now[0])
    reads = sdk_axis.settings.get_calls

    for _ in range(5):  # Five stream ticks: position bounds plus the limit properties.
        _ = (axis.position, axis.upper_limit, axis.lower_limit, axis.speed, axis.acceleration)
    assert len(reads) == 4, "each slow setting is read once per second"

    sdk_axis.settings.values[zaber.SettingConstants.MAXSPEED] = 5_000.0  # e.g. Zaber Launcher
    assert axis.speed == 10_000.0
    now[0] += zaber._SETTINGS_TTL_S
    assert axis.speed == 5_000.0
    axis.upper_limit = 90_000.0
    assert axis.upper_limit == 90_000.0, "a write through the driver is visible at once"

    axis.close()
    hub.close()


def test_zaber_linear_axis_rejects_non_linear_peripheral(
    controller: tuple[zaber.ZaberHub, FakeConnection, FakeAxis, FakeAxis],
) -> None:
    hub, _connection, sdk_axis, _y_axis = controller
    sdk_axis.axis_type = zaber.AxisType.ROTARY

    with pytest.raises(zaber.ZaberDiscoveryError, match="expected a linear axis"):
        zaber.ZaberLinearAxis("x_axis", hub=hub, device_serial=12345, axis_number=1)

    sdk_axis.axis_type = zaber.AxisType.LINEAR
    axis = zaber.ZaberLinearAxis("x_axis", hub=hub, device_serial=12345, axis_number=1)
    axis.close()
    hub.close()


def test_zaber_axis_moves_nonblocking_and_enforces_timeout_and_limits(
    controller: tuple[zaber.ZaberHub, FakeConnection, FakeAxis, FakeAxis],
) -> None:
    hub, _connection, sdk_axis, _y_axis = controller
    axis = zaber.ZaberLinearAxis("x_axis", hub=hub, device_serial=12345, axis_number=1)

    axis.move_abs(30_000.0)
    assert sdk_axis.move_absolute_calls[-1][0] == 30_000.0
    assert sdk_axis.move_absolute_calls[-1][2] is False
    with pytest.raises(TimeoutError, match="did not stop within 0s"):
        axis.await_movement(0)
    axis.halt()

    with pytest.raises(ValueError, match=r"rejected move to 100001 um.*outside the controller"):
        axis.move_abs(100_001.0)
    with pytest.raises(ValueError, match=r"rejected move by -30001 um.*outside the controller"):
        axis.move_rel(-30_001.0)
    assert len(sdk_axis.move_absolute_calls) == 1
    assert not sdk_axis.move_relative_calls
    assert sdk_axis.position == 30_000.0

    axis.close()
    hub.close()


@pytest.mark.parametrize("timeout_s", [-1.0, float("nan"), float("inf")])
def test_zaber_axis_invalid_wait_timeout_rejected_before_motion(
    controller: tuple[zaber.ZaberHub, FakeConnection, FakeAxis, FakeAxis],
    monkeypatch: pytest.MonkeyPatch,
    timeout_s: float,
) -> None:
    hub, _connection, sdk_axis, _y_axis = controller
    axis = zaber.ZaberLinearAxis("x_axis", hub=hub, device_serial=12345)
    # Complete any incorrectly issued move immediately so a NaN timeout cannot hang the test.
    monkeypatch.setattr(sdk_axis, "is_busy", lambda: False)
    try:
        for command, args in (
            (axis.move_abs, (30_000.0,)),
            (axis.move_rel, (100.0,)),
            (axis.go_home, ()),
        ):
            with pytest.raises(ValueError, match="timeout_s"):
                command(*args, wait=True, timeout_s=timeout_s)
        with pytest.raises(ValueError, match="timeout_s"):
            axis.await_movement(timeout_s)
        assert not sdk_axis.move_absolute_calls
        assert not sdk_axis.move_relative_calls
        assert not sdk_axis.home_calls
    finally:
        axis.close()
        hub.close()


def test_zaber_axis_writes_settings_in_axis_units_and_halts_on_close(
    controller: tuple[zaber.ZaberHub, FakeConnection, FakeAxis, FakeAxis],
) -> None:
    hub, _connection, sdk_axis, _y_axis = controller
    axis = zaber.ZaberLinearAxis("x_axis", hub=hub, device_serial=12345, axis_number=1)

    axis.speed = 12_000.0
    axis.acceleration = 24_000.0
    axis.lower_limit = 100.0
    axis.upper_limit = 99_000.0
    axis.home = 500.0
    with pytest.raises(ValueError, match="outside limits"):
        axis.home = 99_001.0
    axis.set_logical_position(750.0)
    axis.close()

    assert [(setting, value) for setting, value, _unit in sdk_axis.settings.set_calls] == [
        (zaber.SettingConstants.MAXSPEED, 12_000.0),
        (zaber.SettingConstants.ACCEL, 24_000.0),
        (zaber.SettingConstants.LIMIT_MIN, 100.0),
        (zaber.SettingConstants.LIMIT_MAX, 99_000.0),
        (zaber.SettingConstants.LIMIT_HOME_PRESET, 500.0),
        (zaber.SettingConstants.POS, 750.0),
    ]
    assert sdk_axis.stop_calls == [False]
    hub.close()


def ttl_axis(hub: zaber.ZaberHub, **overrides: Any) -> tuple[zaber.ZaberLinearAxis, zaber.ZaberTTLStepper]:
    config = {**TTL_CONFIG, **overrides}
    axis = zaber.ZaberLinearAxis("z_axis", hub=hub, device_serial=12345, axis_number=1, **config)
    stepper = axis.get_ttl_stepper()
    assert isinstance(stepper, zaber.ZaberTTLStepper)
    return axis, stepper


def test_zaber_ttl_stepper_disarms_persisted_and_armed_triggers_on_open_reset_and_close(
    controller: tuple[zaber.ZaberHub, FakeConnection, FakeAxis, FakeAxis],
) -> None:
    hub, connection, _z_sdk, _y_sdk = controller
    device = connection.devices[0]
    step, status = device.triggers.by_number[1], device.triggers.by_number[2]
    step.enabled = status.enabled = True  # Left armed by an earlier session; triggers persist.

    axis, stepper = ttl_axis(hub)
    assert not step.enabled, "a persisted step trigger must not survive opening the axis"
    assert not status.enabled
    assert device.io.digital_outputs[1] is zaber.DigitalOutputAction.OFF
    plain = zaber.ZaberLinearAxis("y_axis", hub=hub, device_serial=12345, axis_number=2)
    assert plain.get_ttl_stepper() is None, "axes without TTL settings keep Voxel's no-stepper path"

    stepper.configure(RELATIVE)
    stepper.queue_relative_move(1.0)
    assert step.enabled
    assert status.enabled
    stepper.reset()
    assert not step.enabled, "reset runs before preview and after each stack; it must disarm"
    assert not status.enabled
    assert step.actions == {}

    stepper.configure(RELATIVE)
    stepper.queue_relative_move(1.0)
    axis.close()
    plain.close()
    assert not step.enabled, "closing the axis must disarm an armed step"
    hub.close()
    assert connection.closed


def test_zaber_ttl_stepper_steps_queued_distance_on_each_input_rising_edge(
    controller: tuple[zaber.ZaberHub, FakeConnection, FakeAxis, FakeAxis],
) -> None:
    hub, connection, z_sdk, _y_sdk = controller
    device = connection.devices[0]
    step, status = device.triggers.by_number[1], device.triggers.by_number[2]
    axis, stepper = ttl_axis(hub)

    stepper.configure(RELATIVE)
    stepper.queue_relative_move(2.5)

    assert step.condition == ("io", zaber.IoPortType.DIGITAL_INPUT, 1, zaber.TriggerCondition.EQ, 1)
    assert step.actions == {zaber.TriggerAction.A: (1, "move rel 25")}
    assert zaber.TriggerAction.B in step.cleared, "the unused action must not run a stale command"
    assert z_sdk.settings.conversions == [("pos", 2.5, zaber.Units.LENGTH_MICROMETRES)]
    assert status.condition == ("setting", 1, "motion.busy", zaber.TriggerCondition.EQ, 0)
    assert status.actions == {zaber.TriggerAction.A: (0, "io set do 1 1 schedule 1 0")}
    assert step.enabled
    assert status.enabled

    stepper.configure(TTLStepperConfig(step_mode=StepMode.RELATIVE, pulse_after_move=False))
    stepper.queue_relative_move(-1.0)
    assert step.actions == {zaber.TriggerAction.A: (1, "move rel -10")}
    assert not status.enabled, "no status pulse was requested"
    axis.close()
    hub.close()


def test_zaber_ttl_stepper_refuses_steps_it_cannot_make_safely(
    controller: tuple[zaber.ZaberHub, FakeConnection, FakeAxis, FakeAxis],
) -> None:
    hub, connection, _z_sdk, _y_sdk = controller
    device = connection.devices[0]
    step = device.triggers.by_number[1]
    axis, stepper = ttl_axis(hub)

    with pytest.raises(RuntimeError, match="configure the Zaber TTL stepper"):
        stepper.queue_relative_move(1.0)
    with pytest.raises(NotImplementedError, match="relative steps only"):
        stepper.configure(TTLStepperConfig(step_mode=StepMode.ABSOLUTE))
    with pytest.raises(NotImplementedError, match="input edges only"):
        stepper.configure(TTLStepperConfig(step_mode=StepMode.RELATIVE, trigger_mode=TriggerMode.REPEATING))
    with pytest.raises(NotImplementedError, match="relative steps only"):
        stepper.queue_absolute_move(10.0)

    stepper.configure(RELATIVE)
    for delta in (0.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="finite and nonzero"):
            stepper.queue_relative_move(delta)
    with pytest.raises(ValueError, match="native resolution"):
        stepper.queue_relative_move(0.01)
    with pytest.raises(ValueError, match=r"would move 1\. .* such as 1 or 1\.1"):
        stepper.queue_relative_move(1.03)
    device.io.digital_inputs[0] = True
    with pytest.raises(RuntimeError, match="already high"):
        stepper.queue_relative_move(1.0)
    assert not step.enabled, "no refused request may arm the step trigger"

    device.io.digital_inputs[0] = False
    stepper.queue_relative_move(1.0)
    with pytest.raises(RuntimeError, match="repeats one step"):
        stepper.queue_relative_move(2.0)
    assert step.actions[zaber.TriggerAction.A] == (1, "move rel 10")
    axis.close()
    hub.close()


def test_zaber_ttl_stepper_configure_pulse_without_status_output_rejected(
    controller: tuple[zaber.ZaberHub, FakeConnection, FakeAxis, FakeAxis],
) -> None:
    hub, connection, _z_sdk, _y_sdk = controller
    axis, stepper = ttl_axis(hub, ttl_status_output=None, ttl_status_trigger=None)

    with pytest.raises(ValueError, match="pulse_after_move requires ttl_status_output"):
        stepper.configure(RELATIVE)
    with pytest.raises(RuntimeError, match="configure the Zaber TTL stepper"):
        stepper.queue_relative_move(1.0)

    stepper.configure(TTLStepperConfig(step_mode=StepMode.RELATIVE, pulse_after_move=False))
    stepper.queue_relative_move(1.0)
    assert connection.devices[0].triggers.by_number[1].enabled
    axis.close()
    hub.close()


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"ttl_trigger": None}, "ttl_input and ttl_trigger together"),
        ({"ttl_status_trigger": None}, "ttl_status_output and ttl_status_trigger together"),
        ({"ttl_input": None, "ttl_trigger": None}, "requires ttl_input"),
        ({"ttl_status_trigger": 1}, "different triggers"),
        ({"ttl_input": 5}, "TTL input 5 is outside 1..4"),
        ({"ttl_trigger": 7}, "TTL trigger 7 is outside 1..6"),
        ({"ttl_status_output": 5}, "TTL status output 5 is outside 1..4"),
    ],
)
def test_zaber_linear_axis_invalid_ttl_config_rejected_without_leaking_the_axis(
    controller: tuple[zaber.ZaberHub, FakeConnection, FakeAxis, FakeAxis],
    overrides: dict[str, Any],
    message: str,
) -> None:
    hub, connection, z_sdk, _y_sdk = controller

    with pytest.raises(ValueError, match=message):
        ttl_axis(hub, **overrides)

    assert hub.reserve_axis(12345, 1) is z_sdk, "a rejected configuration must release its axis"
    hub.release_axis(12345, 1)
    assert not any(trigger.enabled for trigger in connection.devices[0].triggers.by_number.values())
    hub.close()


def test_zaber_ttl_stepper_reset_failure_still_disarms_the_rest_and_releases_the_axis(
    controller: tuple[zaber.ZaberHub, FakeConnection, FakeAxis, FakeAxis],
) -> None:
    hub, connection, z_sdk, _y_sdk = controller
    device = connection.devices[0]
    step, status = device.triggers.by_number[1], device.triggers.by_number[2]
    axis, stepper = ttl_axis(hub)
    stepper.configure(RELATIVE)
    stepper.queue_relative_move(1.0)
    step.fail_disable = True

    with pytest.raises(RuntimeError, match="could not disarm Zaber TTL stepping"):
        axis.close()

    assert not status.enabled, "one failed disarm must not stop the others"
    assert device.io.digital_outputs[1] is zaber.DigitalOutputAction.OFF
    assert hub.reserve_axis(12345, 1) is z_sdk, "close must release the axis even when disarm fails"
    hub.release_axis(12345, 1)
    hub.close()


def test_zaber_ttl_stepper_reset_retries_a_transient_disarm_timeout(
    controller: tuple[zaber.ZaberHub, FakeConnection, FakeAxis, FakeAxis],
) -> None:
    hub, connection, _z_sdk, _y_sdk = controller
    step = connection.devices[0].triggers.by_number[1]
    axis, stepper = ttl_axis(hub)
    stepper.configure(RELATIVE)
    stepper.queue_relative_move(1.0)
    attempts: list[bool] = []
    real_disable = step.disable

    def disable_after_one_timeout() -> None:
        attempts.append(True)
        if len(attempts) == 1:
            raise RuntimeError("RequestTimeoutException: Device has not responded in given timeout")
        real_disable()

    step.disable = disable_after_one_timeout  # type: ignore[method-assign]

    stepper.reset()

    assert len(attempts) == 2
    assert not step.enabled
    axis.close()
    hub.close()


def test_zaber_axes_share_one_rig_hub_and_step_through_voxel_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    connection = FakeConnection([FakeDevice(12345, [FakeAxis(1), FakeAxis(2, position=40_000.0)])])
    step = connection.devices[0].triggers.by_number[1]
    install_connection(monkeypatch, connection)
    config = zaber_rig_config(z_axis={"axis_number": 1, **TTL_CONFIG}, y_axis={"axis_number": 2})

    async def run() -> None:
        rig = Rig(config, name="zaber-ttl-test")
        await rig.open()
        try:
            assert not rig.build_errors, f"device build failed: {rig.build_errors}"
            z_axis = ContinuousAxisHandle.wrap(rig.devices["z_axis"])
            y_axis = ContinuousAxisHandle.wrap(rig.devices["y_axis"])
            assert await z_axis.position.get() == 25_000.0
            assert await y_axis.position.get() == 40_000.0
            # The calls Instrument._capture_volume makes around each stack, then preview's reset.
            await z_axis.configure_ttl_stepper(RELATIVE)
            await z_axis.queue_relative_move(1.0)
            assert step.enabled
            await z_axis.reset_ttl_stepper()
            assert not step.enabled
        finally:
            await rig.close()

    asyncio.run(asyncio.wait_for(run(), timeout=5.0))
    assert not step.enabled
    assert connection.closed


SCAN_CONFIG = {
    "scan_encoder": 2,
    "scan_interpolation": 4,
    "scan_output_width_ns": 1000,
    "scan_scale_period": 20.0,
    "scan_acceleration": 50_000.0,
}
SCAN_START = 25_000.0  # FakeAxis(1)'s position.
SCAN_TARGET = 30_000.0
VELOCITY_UM = zaber.Units.VELOCITY_MICROMETRES_PER_SECOND
ACCELERATION_UM = zaber.Units.ACCELERATION_MICROMETRES_PER_SECOND_SQUARED


def scan_axis(
    hub: zaber.ZaberHub, monkeypatch: pytest.MonkeyPatch, **overrides: Any
) -> tuple[zaber.ZaberLinearAxis, zaber.ZaberContinuousScanner]:
    monkeypatch.setattr(zaber, "_SCAN_OUTPUT_SETTLE_S", 0.0)
    config = {**SCAN_CONFIG, **overrides}
    axis = zaber.ZaberLinearAxis("x_axis", hub=hub, device_serial=12345, axis_number=1, **config)
    scanner = axis.get_continuous_scanner()
    assert isinstance(scanner, zaber.ZaberContinuousScanner)
    return axis, scanner


def test_zaber_continuous_scanner_forces_encoder_output_off_on_open_and_close(
    controller: tuple[zaber.ZaberHub, FakeConnection, FakeAxis, FakeAxis],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hub, _connection, x_sdk, y_sdk = controller
    x_sdk.settings.values["encoder.2.out.enable"] = 1  # Left on by an earlier session.
    y_sdk.settings.values["encoder.2.out.enable"] = 1

    axis, scanner = scan_axis(hub, monkeypatch)
    assert x_sdk.settings.values["encoder.2.out.enable"] == 0, "opening the axis must turn the output off"
    plain = zaber.ZaberLinearAxis("y_axis", hub=hub, device_serial=12345, axis_number=2)
    assert plain.get_continuous_scanner() is None, "axes without scan settings keep Voxel's no-scanner path"
    assert not y_sdk.settings.set_calls

    scanner.prepare(start=SCAN_START, speed=2_000.0)
    scanner.start(SCAN_TARGET)
    axis.close()
    plain.close()
    assert x_sdk.stop_calls[0] is False, "closing mid-scan halts the axis before restoring its settings"
    assert x_sdk.settings.values["encoder.2.out.enable"] == 0, "closing the axis must turn the output off"
    assert x_sdk.settings.values[S.MAXSPEED] == 10_000.0
    hub.close()


def test_zaber_continuous_scan_set_verifies_its_settings_then_restores_them_output_first(
    controller: tuple[zaber.ZaberHub, FakeConnection, FakeAxis, FakeAxis],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hub, _connection, x_sdk, _y_sdk = controller
    axis, scanner = scan_axis(hub, monkeypatch)
    settings = x_sdk.settings
    # Interpolation 4 of a 20 um period: 5 um X4 counts, 20 um per A edge, and 5 m/s at 1000 ns.
    assert (scanner.info.edge_spacing, scanner.info.acceleration) == (20.0, 50_000.0)
    assert scanner.info.max_speed == pytest.approx(0.9 * 5e6)
    assert axis.speed == 10_000.0  # Cached for a second.
    settled: list[tuple[float, float]] = []
    monkeypatch.setattr(zaber, "_SCAN_OUTPUT_SETTLE_S", 0.2)
    monkeypatch.setattr(
        zaber.time, "sleep", lambda seconds: settled.append((seconds, settings.values["encoder.2.out.enable"]))
    )
    settings.set_calls.clear()

    state = scanner.prepare(start=SCAN_START, speed=2_000.0)

    assert state["position"] == state["encoder_position"] == state["exported_encoder_position"] == SCAN_START
    assert state["origin"] == state["exported_encoder_position"], "a DAQ counts the exported encoder from here"
    assert [(name, value, unit) for name, value, unit in settings.set_calls] == [
        ("encoder.2.out.interpolation", 4, NATIVE),
        ("encoder.2.out.width", 1000, NATIVE),
        (S.MAXSPEED, 2_000.0, VELOCITY_UM),
        (S.ACCEL, 50_000.0, ACCELERATION_UM),
        ("encoder.2.out.enable", 1, NATIVE),
    ]
    assert settled == [(0.2, 1)], "the output settles after it is enabled"
    assert axis.speed == 2_000.0, "the scan's speed write must not hide behind the settings cache"
    with pytest.raises(RuntimeError, match="already prepared"):
        scanner.prepare(start=SCAN_START, speed=2_000.0)

    scanner.start(SCAN_TARGET)
    assert x_sdk.move_absolute_calls == [(SCAN_TARGET, zaber.Units.LENGTH_MICROMETRES, False)]
    with pytest.raises(RuntimeError, match="already started"):
        scanner.start(SCAN_TARGET)
    assert scanner.is_scanning()
    x_sdk.busy = False
    assert not scanner.is_scanning()
    settings.set_calls.clear()

    final = scanner.finish(SCAN_TARGET)

    assert final["position"] == SCAN_TARGET
    assert final["status_poll_timeouts"] == 0
    assert settings.set_calls == [
        ("encoder.2.out.enable", 0, NATIVE),
        ("encoder.2.out.interpolation", 4, NATIVE),
        ("encoder.2.out.width", 25, NATIVE),
        (S.MAXSPEED, 100_000.0, NATIVE),
        (S.MOTION_ACCELONLY, 200_000.0, NATIVE),
        (S.MOTION_DECELONLY, 200_000.0, NATIVE),
    ]
    assert axis.speed == 10_000.0
    axis.close()
    hub.close()


def _enable_trigger(_sdk: FakeAxis, device: FakeDevice) -> None:
    device.triggers.by_number[3].enabled = True


@pytest.mark.parametrize(
    ("change", "speed", "message"),
    [
        (lambda sdk, _device: setattr(sdk, "homed", False), 2_000.0, "homed"),
        (lambda sdk, _device: setattr(sdk, "busy", True), 2_000.0, "idle"),
        (lambda sdk, _device: sdk.flags.add("FS"), 2_000.0, "warning-free"),
        (
            lambda sdk, _device: setattr(sdk, "position", SCAN_START + 1.5),
            2_000.0,
            r"position 25001\.5 um is not within",
        ),
        (lambda sdk, _device: setattr(sdk, "encoder_error", -1.5), 2_000.0, r"encoder position 24998\.5 um"),
        (_enable_trigger, 2_000.0, "disable every Zaber trigger"),
        (lambda sdk, _device: sdk.settings.values.update({"encoder.2.out.enable": 1}), 2_000.0, "already on"),
        (lambda sdk, _device: sdk.settings.values.update({S.ENCODER_PORT_DEFAULT: 1}), 2_000.0, "follows encoder 1"),
        (lambda sdk, _device: sdk.settings.values.update({S.CLOOP_ENABLE: 0}), 2_000.0, "closed-loop"),
        (lambda sdk, _device: sdk.settings.values.update({S.MOTION_ACCEL_RAMPTIME: 10}), 2_000.0, "ramptime"),
        (lambda sdk, _device: sdk.settings.values.update({S.LIMIT_MIN: 26_000.0}), 2_000.0, "outside limits"),
        (lambda _sdk, _device: None, 4.6e6, r"outside \(0, 4\.5e\+06\]"),
        (lambda _sdk, _device: None, float("nan"), "outside"),
    ],
)
def test_zaber_continuous_scan_prepare_refusal_changes_no_setting_and_moves_nothing(
    controller: tuple[zaber.ZaberHub, FakeConnection, FakeAxis, FakeAxis],
    monkeypatch: pytest.MonkeyPatch,
    change: Any,
    speed: float,
    message: str,
) -> None:
    hub, connection, x_sdk, _y_sdk = controller
    axis, scanner = scan_axis(hub, monkeypatch)
    change(x_sdk, connection.devices[0])
    x_sdk.settings.set_calls.clear()

    with pytest.raises((RuntimeError, ValueError), match=message):
        scanner.prepare(start=SCAN_START, speed=speed)

    assert not x_sdk.settings.set_calls
    assert not x_sdk.move_absolute_calls
    with pytest.raises(RuntimeError, match="prepare the continuous scan first"):
        scanner.start(SCAN_TARGET)
    axis.close()
    hub.close()


def test_zaber_continuous_scan_decelonly_mismatch_refused_and_restored(
    controller: tuple[zaber.ZaberHub, FakeConnection, FakeAxis, FakeAxis],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hub, _connection, x_sdk, _y_sdk = controller
    axis, scanner = scan_axis(hub, monkeypatch)
    original = dict(x_sdk.settings.values)
    x_sdk.settings.accel_sets_decel = False

    with pytest.raises(RuntimeError, match=r"motion\.decelonly reads back 200000, not 500000"):
        scanner.prepare(start=SCAN_START, speed=2_000.0)

    assert x_sdk.settings.values == original, "a failed prepare restores everything it changed"
    assert not x_sdk.move_absolute_calls
    x_sdk.settings.accel_sets_decel = True
    scanner.prepare(start=SCAN_START, speed=2_000.0)
    scanner.reset()
    axis.close()
    hub.close()


@pytest.mark.parametrize("dropped", [3, 4])
def test_zaber_continuous_scan_status_poll_tolerates_three_consecutive_timeouts(
    controller: tuple[zaber.ZaberHub, FakeConnection, FakeAxis, FakeAxis],
    monkeypatch: pytest.MonkeyPatch,
    dropped: int,
) -> None:
    hub, _connection, x_sdk, _y_sdk = controller
    axis, scanner = scan_axis(hub, monkeypatch)
    scanner.prepare(start=SCAN_START, speed=2_000.0)
    scanner.start(SCAN_TARGET)
    x_sdk.busy = False
    remaining = [dropped]
    real_is_busy = x_sdk.is_busy

    def is_busy() -> bool:
        if remaining[0]:
            remaining[0] -= 1
            raise RequestTimeoutException("Device has not responded in given timeout")
        return real_is_busy()

    monkeypatch.setattr(x_sdk, "is_busy", is_busy)
    if dropped > 3:
        with pytest.raises(RequestTimeoutException):
            scanner.is_scanning()
        scanner.reset()
    else:
        assert not scanner.is_scanning()
        assert scanner.finish(SCAN_TARGET)["status_poll_timeouts"] == 3
    axis.close()
    hub.close()


def test_zaber_continuous_scan_finish_requires_the_axis_settled_at_target(
    controller: tuple[zaber.ZaberHub, FakeConnection, FakeAxis, FakeAxis],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hub, _connection, x_sdk, _y_sdk = controller
    axis, scanner = scan_axis(hub, monkeypatch)
    scanner.prepare(start=SCAN_START, speed=2_000.0)
    scanner.start(SCAN_TARGET)

    with pytest.raises(RuntimeError, match="must be idle"):
        scanner.finish(SCAN_TARGET)
    x_sdk.busy = False
    x_sdk.encoder_error = 1.5
    with pytest.raises(RuntimeError, match=r"encoder position 30001\.5 um is not within 1 um"):
        scanner.finish(SCAN_TARGET)
    x_sdk.encoder_error = 0.0
    x_sdk.flags.add("FS")
    with pytest.raises(RuntimeError, match="warning-free"):
        scanner.finish(SCAN_TARGET)
    assert x_sdk.settings.values["encoder.2.out.enable"] == 1, "an unsettled finish leaves restoration to reset"

    x_sdk.flags.clear()
    scanner.finish(SCAN_TARGET)
    assert x_sdk.settings.values["encoder.2.out.enable"] == 0
    axis.close()
    hub.close()


def test_zaber_continuous_scan_reset_attempts_every_step_and_is_idempotent(
    controller: tuple[zaber.ZaberHub, FakeConnection, FakeAxis, FakeAxis],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hub, _connection, x_sdk, _y_sdk = controller
    axis, scanner = scan_axis(hub, monkeypatch)
    settings = x_sdk.settings
    scanner.prepare(start=SCAN_START, speed=2_000.0)
    scanner.start(SCAN_TARGET)
    settings.fail_set = {"encoder.2.out.width"}
    settings.set_calls.clear()

    with pytest.raises(RuntimeError, match=r"restore: could not restore .*encoder\.2\.out\.width: .*write failed"):
        scanner.reset()

    assert x_sdk.stop_calls == [False], "reset halts the moving axis first"
    assert [name for name, _value, _unit in settings.set_calls] == [
        "encoder.2.out.enable",
        "encoder.2.out.interpolation",
        "encoder.2.out.width",
        S.MAXSPEED,
        S.MOTION_ACCELONLY,
        S.MOTION_DECELONLY,
        "encoder.2.out.enable",
    ], "one failed restore must not stop the others, and the output is forced off last"
    assert settings.values[S.MAXSPEED] == 10_000.0
    assert settings.values["encoder.2.out.enable"] == 0

    settings.fail_set.clear()
    scanner.reset()
    assert settings.values["encoder.2.out.width"] == 25, "a later reset retries what failed"
    settings.set_calls.clear()
    scanner.reset()
    assert settings.set_calls == [("encoder.2.out.enable", 0, NATIVE)]
    assert x_sdk.stop_calls == [False]
    axis.close()
    hub.close()


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"scan_acceleration": None}, "configure scan_encoder, .* together"),
        ({"scan_interpolation": 6}, "6 is not a Zaber encoder output interpolation"),
        ({"scan_encoder": 0}, "scan_encoder must be positive"),
        ({"scan_output_width_ns": 0}, "scan_output_width_ns must be positive"),
        ({"scan_scale_period": -20.0}, "scan_scale_period must be finite and positive"),
        ({"scan_acceleration": float("nan")}, "scan_acceleration must be finite and positive"),
    ],
)
def test_zaber_linear_axis_invalid_scan_config_rejected_without_leaking_the_axis(
    controller: tuple[zaber.ZaberHub, FakeConnection, FakeAxis, FakeAxis],
    monkeypatch: pytest.MonkeyPatch,
    overrides: dict[str, Any],
    message: str,
) -> None:
    hub, _connection, x_sdk, _y_sdk = controller

    with pytest.raises(ValueError, match=message):
        scan_axis(hub, monkeypatch, **overrides)

    assert hub.reserve_axis(12345, 1) is x_sdk, "a rejected configuration must release its axis"
    hub.release_axis(12345, 1)
    hub.close()
