import asyncio
from types import SimpleNamespace
from typing import Any

import pytest

pytest.importorskip("zaber_motion", reason="needs the vxl-drivers `zaber` extra")

from vxl_drivers.axes import zaber
from zaber_motion.exceptions import CommandFailedException, CommandFailedExceptionData

from rigup import DeviceConfig, Rig, RigConfig
from vxl.devices.axes.continuous import StepMode, TriggerMode, TTLStepperConfig
from vxl.devices.axes.continuous.handle import ContinuousAxisHandle

NATIVE_PER_UM = 10
TTL_CONFIG = {"ttl_input": 1, "ttl_trigger": 1, "ttl_status_output": 1, "ttl_status_trigger": 2}
RELATIVE = TTLStepperConfig(step_mode=StepMode.RELATIVE)


class FakeSettings:
    def __init__(self, values: dict[str, float]) -> None:
        self.values = values
        self.set_calls: list[tuple[str, float, object]] = []
        self.get_calls: list[str] = []
        self.conversions: list[tuple[str, float, object]] = []

    def get(self, setting: str, unit: object) -> float:
        del unit
        self.get_calls.append(setting)
        return self.values[setting]

    def set(self, setting: str, value: float, unit: object) -> None:
        self.values[setting] = value
        self.set_calls.append((setting, value, unit))

    def has_setting(self, setting: str) -> bool:
        return setting in self.values

    def convert_to_native_units(self, setting: str, value: float, unit: object, **options: bool) -> float:
        self.conversions.append((setting, value, unit))
        native = value * NATIVE_PER_UM
        return float(round(native)) if options.get("round") else native


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
                zaber.SettingConstants.ACCEL: 20_000.0,
                zaber.SettingConstants.LIMIT_HOME_PRESET: 0.0,
                zaber.SettingConstants.POS: position,
            }
        )
        self.position = position
        self.busy = False
        self.homed = True
        self.move_absolute_calls: list[tuple[float, object, bool]] = []
        self.move_relative_calls: list[tuple[float, object, bool]] = []
        self.home_calls: list[bool] = []
        self.stop_calls: list[bool] = []

    def get_position(self, unit: object) -> float:
        del unit
        return self.position

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
