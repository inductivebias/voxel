"""Zaber continuous-axis drivers using the Zaber Motion Library ASCII API."""

import math
import threading
import time
from collections.abc import Callable
from typing import Any

from zaber_motion import Units
from zaber_motion.ascii import (
    Axis,
    AxisType,
    Connection,
    DigitalOutputAction,
    IoPortType,
    SettingConstants,
    TriggerAction,
    TriggerCondition,
)
from zaber_motion.ascii import Device as ZaberDevice
from zaber_motion.exceptions import CommandFailedException, RequestTimeoutException

from rigup import Device, describe
from vxl.devices.axes.continuous import (
    ContinuousAxis,
    ContinuousScanInfo,
    ContinuousScanner,
    StepMode,
    TriggerMode,
    TTLStepper,
    TTLStepperConfig,
)
from vxl.devices.base import DeviceType

DEFAULT_TIMEOUT_S = 2.0
_POLL_INTERVAL_S = 0.05
# Voxel streams every axis property each 0.05 s, and each position read also checks the limits.
# Limits, speed, and acceleration rarely change, so read each at most once a second; writes made
# through this driver take effect at once.
_SETTINGS_TTL_S = 1.0
_RESET_ATTEMPTS = 2
# Largest relative difference between a requested TTL step and the whole native steps it becomes.
_STEP_TOLERANCE = 1e-3
# Long enough for the return interface and an NI PFI input to register the pulse.
_STATUS_PULSE_MS = 1
# encoder.<n>.out.interpolation values Zaber controllers accept.
# fmt: off
_OUTPUT_INTERPOLATIONS = (
    4, 8, 16, 20, 32, 40, 64, 80, 100, 128, 160, 200,
    256, 400, 500, 512, 800, 1000, 1024, 2000, 2048, 4000, 4096,
)
# fmt: on
# A closed-loop stage's velocity ripples around the commanded speed, and the output cannot follow
# transitions closer together than its minimum width, so scans stay 10% below that rate limit.
_SCAN_SPEED_MARGIN = 0.9
# Lets the output-enable transition settle before a DAQ takes its zero-count baseline.
_SCAN_OUTPUT_SETTLE_S = 0.2
# Consecutive unanswered status polls tolerated during a scan. The DAQ, not this poll, records the
# scan, so one lost reply on a lossy controller link must not abort it.
_SCAN_POLL_RETRIES = 3
_SCAN_STOP_TIMEOUT_S = 5.0
# How far the planned and measured positions may be from a scan's start or target: 1 µm in each unit.
_SCAN_POSITION_TOLERANCE = {"nm": 1000.0, "um": 1.0, "mm": 0.001}

# Position, velocity, and acceleration units for each configurable length unit.
_UNITS = {
    "nm": (
        Units.LENGTH_NANOMETRES,
        Units.VELOCITY_NANOMETRES_PER_SECOND,
        Units.ACCELERATION_NANOMETRES_PER_SECOND_SQUARED,
    ),
    "um": (
        Units.LENGTH_MICROMETRES,
        Units.VELOCITY_MICROMETRES_PER_SECOND,
        Units.ACCELERATION_MICROMETRES_PER_SECOND_SQUARED,
    ),
    "mm": (
        Units.LENGTH_MILLIMETRES,
        Units.VELOCITY_MILLIMETRES_PER_SECOND,
        Units.ACCELERATION_MILLIMETRES_PER_SECOND_SQUARED,
    ),
}


class ZaberDiscoveryError(ConnectionError):
    """A configured Zaber connection, device, or axis was not available."""


def open_zaber_connection(*, port: str | None, host: str | None, tcp_port: int) -> Connection:
    """Open one Zaber ASCII connection over serial or TCP."""
    if host is not None:
        return Connection.open_tcp(host, tcp_port)
    if port is None:
        raise ValueError("provide a Zaber port or host")
    return Connection.open_serial_port(port, test_port=True)


def _index_devices(devices: list[ZaberDevice]) -> dict[int, ZaberDevice]:
    if not devices:
        raise ZaberDiscoveryError("no Zaber devices detected")
    indexed: dict[int, ZaberDevice] = {}
    for device in devices:
        serial = device.serial_number
        if serial in indexed:
            raise ZaberDiscoveryError(f"Zaber device serial {serial} was detected more than once")
        indexed[serial] = device
    return indexed


def _validate_timeout(timeout_s: float | None) -> None:
    if timeout_s is not None and (not math.isfinite(timeout_s) or timeout_s < 0):
        raise ValueError(f"timeout_s must be finite and nonnegative, got {timeout_s}")


def _validate_ttl_config(
    input_channel: int | None,
    step_trigger: int | None,
    status_output: int | None,
    status_trigger: int | None,
) -> None:
    if (input_channel is None) != (step_trigger is None):
        raise ValueError("configure ttl_input and ttl_trigger together")
    if (status_output is None) != (status_trigger is None):
        raise ValueError("configure ttl_status_output and ttl_status_trigger together")
    if status_output is not None and input_channel is None:
        raise ValueError("ttl_status_output requires ttl_input")
    if step_trigger is not None and step_trigger == status_trigger:
        raise ValueError("ttl_trigger and ttl_status_trigger must be different triggers")


def _validate_scan_config(*values: float | None) -> None:
    if any(value is None for value in values) and any(value is not None for value in values):
        raise ValueError(
            "configure scan_encoder, scan_interpolation, scan_output_width_ns, scan_scale_period "
            "and scan_acceleration together"
        )


def _check_channel(name: str, number: int, count: int) -> None:
    if not 1 <= number <= count:
        raise ValueError(f"{name} {number} is outside 1..{count} on this Zaber device")


class ZaberTTLStepper(TTLStepper):
    """Repeat one queued relative move each time a Zaber digital input rises.

    Owns trigger ``step_trigger``, which runs ``move rel`` on the axis when ``input_channel``
    goes high, and optionally trigger ``status_trigger``, which pulses ``status_output`` whenever
    the axis becomes idle. The status output also pulses once when armed, because the axis is
    idle then. Only relative, TTL-triggered stepping is supported.
    """

    def __init__(
        self,
        *,
        lock: threading.RLock,
        axis: Axis,
        position_unit: Units,
        input_channel: int,
        step_trigger: int,
        status_output: int | None = None,
        status_trigger: int | None = None,
    ) -> None:
        self._lock = lock
        self._device = axis.device
        self._axis = axis
        self._axis_number = axis.axis_number
        self._unit = position_unit
        self._input_channel = input_channel
        self._step_trigger = step_trigger
        self._status_output = status_output
        self._status_trigger = status_trigger
        self._pulse_after_move = False
        self._configured = False
        self._armed = False
        with lock:
            trigger_count = self._device.triggers.get_number_of_triggers()
            channels = self._device.io.get_channels_info()
        _check_channel("TTL input", input_channel, channels.number_digital_inputs)
        _check_channel("TTL trigger", step_trigger, trigger_count)
        if status_output is not None:
            _check_channel("TTL status output", status_output, channels.number_digital_outputs)
        if status_trigger is not None:
            _check_channel("TTL status trigger", status_trigger, trigger_count)
        # Triggers are non-volatile: never inherit a step armed by an earlier session.
        self.reset()

    def configure(self, cfg: TTLStepperConfig | Any) -> None:
        config = TTLStepperConfig.model_validate(cfg)
        if config.step_mode is not StepMode.RELATIVE:
            raise NotImplementedError("Zaber TTL stepping supports relative steps only")
        if config.trigger_mode is not TriggerMode.TTL:
            raise NotImplementedError("Zaber TTL stepping moves on input edges only")
        if config.pulse_after_move and self._status_output is None:
            raise ValueError("pulse_after_move requires ttl_status_output and ttl_status_trigger on this axis")
        with self._lock:
            self.reset()
            self._pulse_after_move = config.pulse_after_move
            self._configured = True

    def queue_absolute_move(self, position: float) -> None:
        del position
        raise NotImplementedError("Zaber TTL stepping supports relative steps only")

    def queue_relative_move(self, delta: float) -> None:
        if not math.isfinite(delta) or delta == 0:
            raise ValueError(f"TTL step must be finite and nonzero, got {delta}")
        with self._lock:
            if not self._configured:
                raise RuntimeError("configure the Zaber TTL stepper before queueing a step")
            if self._armed:
                raise RuntimeError("Zaber TTL stepping repeats one step; reset before queueing another")
            exact = self._axis.settings.convert_to_native_units(SettingConstants.POS, delta, self._unit)
            native = round(exact)
            if native == 0:
                raise ValueError(f"TTL step {delta} is below the axis's native resolution")
            # The controller moves whole native steps, so a rounded step silently changes the spacing
            # Voxel records: 1 µm on an axis with 0.09525 µm steps moves 0.9525 µm.
            if abs(native - exact) > _STEP_TOLERANCE * abs(exact):
                step = delta / exact
                below, above = sorted((math.floor(exact) * step, math.ceil(exact) * step))
                raise ValueError(
                    f"TTL step {delta:g} is {exact:.4f} native steps; the axis would move {native * step:.6g}. "
                    f"Use a whole number of steps, such as {below:.6g} or {above:.6g}"
                )
            # A trigger whose condition already holds fires as soon as it is enabled.
            if self._device.io.get_digital_input(self._input_channel):
                raise RuntimeError(
                    f"Zaber digital input {self._input_channel} is already high; arming now would step immediately"
                )
            triggers = self._device.triggers
            if self._pulse_after_move and self._status_trigger is not None:
                status = triggers.get_trigger(self._status_trigger)
                status.fire_when_setting(self._axis_number, "motion.busy", TriggerCondition.EQ, 0)
                status.on_fire(
                    TriggerAction.A,
                    0,
                    f"io set do {self._status_output} 1 schedule {_STATUS_PULSE_MS} 0",
                )
                status.clear_action(TriggerAction.B)
                status.enable()
            step = triggers.get_trigger(self._step_trigger)
            step.fire_when_io(IoPortType.DIGITAL_INPUT, self._input_channel, TriggerCondition.EQ, 1)
            step.on_fire(TriggerAction.A, self._axis_number, f"move rel {int(native)}")
            step.clear_action(TriggerAction.B)
            step.enable()
            self._armed = True

    def _disarm_once(self) -> list[BaseException]:
        failures: list[BaseException] = []
        for number in (self._step_trigger, self._status_trigger):
            if number is None:
                continue
            try:
                trigger = self._device.triggers.get_trigger(number)
                trigger.disable()
                trigger.clear_action(TriggerAction.ALL)
            except Exception as error:
                failures.append(error)
        if self._status_output is not None:
            try:
                self._device.io.set_digital_output(self._status_output, DigitalOutputAction.OFF)
            except Exception as error:
                failures.append(error)
        return failures

    def reset(self) -> None:
        """Disarm both triggers and lower the status output, attempting every step."""
        failures: list[BaseException] = []
        with self._lock:
            # Every step is idempotent, so one full retry rides out a transient request timeout.
            for _ in range(_RESET_ATTEMPTS):
                failures = self._disarm_once()
                if not failures:
                    break
            # Re-arming requires configure(), which repeats this disarm and must succeed first.
            self._configured = False
            self._armed = False
        if failures:
            raise RuntimeError(
                f"could not disarm Zaber TTL stepping on axis {self._axis_number}: {failures[0]}"
            ) from failures[0]


class ZaberContinuousScanner(ContinuousScanner):
    """Move a Zaber axis once at constant speed while its controller exports encoder ``encoder`` as TTL A/B.

    Zaber's output interpolation counts X4 quadrature transitions per ``scale_period`` of the analog
    encoder, so the axis travels ``4 * scale_period / interpolation`` per A rising edge. Successive A/B
    transitions are at least ``output_width_ns`` apart, which caps the speed at
    ``scale_period / (interpolation * output_width_ns)``; ``info.max_speed`` keeps a margin below that.
    The output is forced off when the scanner is created and is on only from a successful ``prepare``
    until ``finish`` or ``reset``.
    """

    def __init__(
        self,
        *,
        lock: threading.RLock,
        axis: Axis,
        units: str,
        encoder: int,
        interpolation: int,
        output_width_ns: int,
        scale_period: float,
        acceleration: float,
        invalidate_settings: Callable[[], None],
    ) -> None:
        if encoder < 1:
            raise ValueError(f"scan_encoder must be positive, got {encoder}")
        if interpolation not in _OUTPUT_INTERPOLATIONS:
            raise ValueError(
                f"scan_interpolation {interpolation} is not a Zaber encoder output interpolation; "
                f"use one of {', '.join(map(str, _OUTPUT_INTERPOLATIONS))}"
            )
        if output_width_ns < 1:
            raise ValueError(f"scan_output_width_ns must be positive, got {output_width_ns}")
        for name, value in (("scan_scale_period", scale_period), ("scan_acceleration", acceleration)):
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive, got {value}")
        self._lock = lock
        self._axis = axis
        self._device = axis.device
        self._units = units
        self._position_unit, self._velocity_unit, self._acceleration_unit = _UNITS[units]
        self._tolerance = _SCAN_POSITION_TOLERANCE[units]
        self._encoder = encoder
        self._enable = f"encoder.{encoder}.out.enable"
        self._interpolation = interpolation
        self._output_width_ns = output_width_ns
        self._acceleration = acceleration
        self._invalidate_settings = invalidate_settings
        rate_limit = scale_period / (interpolation * output_width_ns * 1e-9)
        self._info = ContinuousScanInfo(
            edge_spacing=4 * scale_period / interpolation,
            acceleration=acceleration,
            max_speed=_SCAN_SPEED_MARGIN * rate_limit,
        )
        # Output enable first, so restoring in order turns the output off before any other change.
        self._changed = (
            self._enable,
            f"encoder.{encoder}.out.interpolation",
            f"encoder.{encoder}.out.width",
            SettingConstants.MAXSPEED,
            SettingConstants.MOTION_ACCELONLY,
            SettingConstants.MOTION_DECELONLY,
        )
        self._originals: dict[str, float] | None = None  # Native values, held from prepare until restored.
        self._started = False
        self._status_timeouts = 0
        # Never inherit an output that an earlier session left enabled.
        self.reset()

    @property
    def info(self) -> ContinuousScanInfo:
        return self._info

    def _verify_native(self, name: str, expected: float) -> float:
        with self._lock:
            actual = self._axis.settings.get(name)
        if actual != expected:
            raise RuntimeError(f"Zaber {name} reads back {actual:g}, not {expected:g}")
        return actual

    def _set_verified(self, name: str, value: float, unit: Units = Units.NATIVE) -> float:
        """Write one setting and require the native value the controller should store for it."""
        settings = self._axis.settings
        with self._lock:
            expected = value
            if unit is not Units.NATIVE:
                expected = settings.convert_to_native_units(name, value, unit, round=True)
            try:
                settings.set(name, value, unit)
            finally:
                self._invalidate_settings()
            return self._verify_native(name, expected)

    def _state(self) -> dict[str, Any]:
        # One grouped get samples the planned, measured and exported encoder positions together.
        names = (SettingConstants.POS, SettingConstants.ENCODER_POS, f"encoder.{self._encoder}.pos")
        with self._lock:
            reply = self._axis.generic_command(f"get {' '.join(names)}").data
            try:
                native = [float(value) for value in reply.split(";")]
            except ValueError:
                native = []
            if len(native) != len(names) or not all(math.isfinite(value) for value in native):
                raise RuntimeError(f"Zaber returned an invalid position reply {reply!r}")
            position, encoder_position, exported_position = (
                self._axis.settings.convert_from_native_units(name, value, self._position_unit)
                for name, value in zip(names, native, strict=True)
            )
            return {
                "position": position,
                "encoder_position": encoder_position,
                "exported_encoder_position": exported_position,
                "moving": self._axis.is_busy(),
                "homed": self._axis.is_homed(),
                "warnings": sorted(self._axis.warnings.get_flags()),
            }

    def _check_within_limits(self, name: str, position: float) -> None:
        settings = self._axis.settings
        with self._lock:
            lower = settings.get(SettingConstants.LIMIT_MIN, self._position_unit)
            upper = settings.get(SettingConstants.LIMIT_MAX, self._position_unit)
        if not lower <= position <= upper:
            raise ValueError(f"scan {name} {position:g} {self._units} is outside limits [{lower:g}, {upper:g}]")

    def _check_settled(self, state: dict[str, Any], name: str, position: float) -> None:
        if state["moving"] or not state["homed"] or state["warnings"]:
            raise RuntimeError(
                f"the axis must be idle, homed and warning-free at the scan {name}; "
                f"moving={state['moving']}, homed={state['homed']}, warnings={state['warnings']}"
            )
        for key in ("position", "encoder_position"):
            if abs(state[key] - position) > self._tolerance:
                raise RuntimeError(
                    f"{key.replace('_', ' ')} {state[key]:g} {self._units} is not within "
                    f"{self._tolerance:g} {self._units} of the scan {name} {position:g}"
                )

    def _check_ready(self, start: float) -> dict[str, Any]:
        """Read-only checks before prepare changes anything; returns the axis state."""
        with self._lock:
            if any(trigger.enabled for trigger in self._device.triggers.get_enabled_states()):
                raise RuntimeError("disable every Zaber trigger before scanning; reset any TTL stepper first")
            state = self._state()
            self._check_settled(state, "start", start)
            settings = self._axis.settings
            if settings.get(self._enable) != 0:
                raise RuntimeError(f"{self._enable} is already on; another scan or tool is using the output")
            default = settings.get(SettingConstants.ENCODER_PORT_DEFAULT)
            if default != self._encoder:
                raise RuntimeError(f"the axis follows encoder {default:g}, not scan encoder {self._encoder}")
            if settings.get(SettingConstants.CLOOP_ENABLE) != 1:
                raise RuntimeError("closed-loop control must be on to scan")
            # The scan's lead-in and stopping distance assume a plain trapezoidal profile.
            if settings.get(SettingConstants.MOTION_ACCEL_RAMPTIME) != 0:
                raise RuntimeError("motion.accel.ramptime must be 0 to scan")
            self._check_within_limits("start", start)
        return state

    def prepare(self, *, start: float, speed: float) -> dict[str, Any]:
        if not math.isfinite(start):
            raise ValueError(f"scan start must be finite, got {start}")
        if not 0 < speed <= self._info.max_speed:
            raise ValueError(
                f"scan speed {speed:g} {self._units}/s is outside (0, {self._info.max_speed:g}]; "
                "the encoder output cannot follow faster motion"
            )
        with self._lock:
            if self._originals is not None:
                raise RuntimeError("a continuous scan is already prepared; finish or reset it first")
            state = self._check_ready(start)
            self._originals = {name: self._axis.settings.get(name) for name in self._changed}
            self._status_timeouts = 0
            try:
                self._set_verified(f"encoder.{self._encoder}.out.interpolation", self._interpolation)
                self._set_verified(f"encoder.{self._encoder}.out.width", self._output_width_ns)
                self._set_verified(SettingConstants.MAXSPEED, speed, self._velocity_unit)
                # The accel alias writes acceleration and deceleration but reads back acceleration only.
                accel = self._set_verified(SettingConstants.ACCEL, self._acceleration, self._acceleration_unit)
                self._verify_native(SettingConstants.MOTION_DECELONLY, accel)
                self._set_verified(self._enable, 1)
            except BaseException as error:
                try:
                    self.reset()
                except Exception as cleanup_error:
                    error.add_note(f"restoring the axis after the failed prepare also failed: {cleanup_error}")
                raise
        time.sleep(_SCAN_OUTPUT_SETTLE_S)
        # A DAQ counting the exported quadrature counts from the exported encoder's own position.
        return {**state, "origin": state["exported_encoder_position"]}

    def start(self, target: float) -> None:
        if not math.isfinite(target):
            raise ValueError(f"scan target must be finite, got {target}")
        with self._lock:
            if self._originals is None:
                raise RuntimeError("prepare the continuous scan first")
            if self._started:
                raise RuntimeError("the continuous scan has already started")
            self._check_within_limits("target", target)
            self._started = True
            # A lost reply may still have started the move, so the command is never resent.
            self._axis.move_absolute(target, self._position_unit, wait_until_idle=False)

    def is_scanning(self) -> bool:
        if self._originals is None:
            raise RuntimeError("prepare the continuous scan first")
        consecutive = 0
        while True:
            try:
                with self._lock:
                    busy, flags = self._axis.is_busy(), self._axis.warnings.get_flags()
                break
            except RequestTimeoutException:
                self._status_timeouts += 1
                consecutive += 1
                if consecutive > _SCAN_POLL_RETRIES:
                    raise
        if flags:
            raise RuntimeError(f"Zaber reported {', '.join(sorted(flags))} during the continuous scan")
        return busy

    def _restore(self) -> None:
        """Restore every snapshotted setting in order, attempting all; raises naming each failure."""
        if self._originals is None:
            return
        failures: list[tuple[str, Exception]] = []
        for name, value in self._originals.items():
            try:
                self._set_verified(name, value)
            except Exception as error:
                failures.append((name, error))
        if failures:
            details = "; ".join(f"{name}: {error}" for name, error in failures)
            raise RuntimeError(f"could not restore Zaber scan settings ({details})") from failures[0][1]
        self._originals = None
        self._started = False

    def finish(self, target: float) -> dict[str, Any]:
        with self._lock:
            if self._originals is None:
                raise RuntimeError("prepare the continuous scan first")
            state = self._state()
            self._check_settled(state, "target", target)
            self._restore()
        return {**state, "status_poll_timeouts": self._status_timeouts}

    def _stop(self) -> None:
        """Halt the axis if it is moving and wait for it to stop."""
        deadline = time.monotonic() + _SCAN_STOP_TIMEOUT_S
        halted = False
        while True:
            with self._lock:
                if not self._axis.is_busy():
                    return
                if not halted:
                    self._axis.stop(wait_until_idle=False)
                    halted = True
            if time.monotonic() >= deadline:
                raise RuntimeError(f"the axis did not stop within {_SCAN_STOP_TIMEOUT_S:g}s")
            time.sleep(_POLL_INTERVAL_S)

    def reset(self) -> None:
        failures: list[tuple[str, Exception]] = []
        if self._originals is not None:
            try:
                # Restoring speed and acceleration mid-move would change that move.
                self._stop()
            except Exception as error:
                failures.append(("stop", error))
            else:
                try:
                    self._restore()
                except Exception as error:
                    failures.append(("restore", error))
        try:
            self._set_verified(self._enable, 0)
        except Exception as error:
            failures.append(("output off", error))
        if failures:
            details = "; ".join(f"{step}: {error}" for step, error in failures)
            raise RuntimeError(
                f"could not reset the Zaber continuous scan on axis {self._axis.axis_number} ({details})"
            ) from failures[0][1]


class ZaberHub(Device[DeviceType]):
    """Own one Zaber connection and reserve its axes by stable device serial."""

    def __init__(
        self,
        uid: str,
        *,
        port: str | None = None,
        host: str | None = None,
        tcp_port: int = Connection.TCP_PORT_DEVICE_ONLY,
        timeout_s: float = DEFAULT_TIMEOUT_S,
    ) -> None:
        if (port is None) == (host is None):
            raise ValueError("provide exactly one of port or host")
        if port is not None and not port.strip():
            raise ValueError("Zaber port must not be empty")
        if host is not None and not host.strip():
            raise ValueError("Zaber host must not be empty")
        if not 1 <= tcp_port <= 65535:
            raise ValueError(f"Zaber tcp_port must be in 1..65535, got {tcp_port}")
        if not math.isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError(f"Zaber timeout_s must be positive, got {timeout_s}")

        super().__init__(uid=uid)
        self._lock = threading.RLock()
        self._reservation_lock = threading.Lock()
        self._reserved: set[tuple[int, int]] = set()
        self._close_requested = False
        self._connection_closed = False
        self._connection = open_zaber_connection(port=port, host=host, tcp_port=tcp_port)
        try:
            self._connection.default_request_timeout = max(1, round(timeout_s * 1000))
            self._devices = _index_devices(self._connection.detect_devices())
        except Exception:
            self._connection.close()
            raise

        self.log.info(
            "Initialized Zaber connection with devices %s",
            ", ".join(str(serial) for serial in self.device_serials),
        )

    @property
    def lock(self) -> threading.RLock:
        return self._lock

    @property
    @describe(label="Device Serials")
    def device_serials(self) -> tuple[int, ...]:
        return tuple(sorted(self._devices))

    def reserve_axis(self, device_serial: int, axis_number: int) -> Axis:
        device = self.device(device_serial)
        if not 1 <= axis_number <= device.axis_count:
            raise ZaberDiscoveryError(
                f"Zaber device serial {device_serial} has no axis {axis_number}; valid axes are 1..{device.axis_count}"
            )

        key = (device_serial, axis_number)
        with self._reservation_lock:
            if self._close_requested:
                raise RuntimeError(f"Zaber hub {self.uid} is closing")
            if key in self._reserved:
                raise ValueError(f"Zaber device serial {device_serial} axis {axis_number} is already reserved")
            axis = device.get_axis(axis_number)
            self._reserved.add(key)
            return axis

    def device(self, device_serial: int) -> ZaberDevice:
        try:
            return self._devices[device_serial]
        except KeyError as error:
            found = ", ".join(str(serial) for serial in self.device_serials)
            raise ZaberDiscoveryError(
                f"Zaber device serial {device_serial} was not detected; found: {found}"
            ) from error

    def release_axis(self, device_serial: int, axis_number: int) -> None:
        with self._reservation_lock:
            self._reserved.discard((device_serial, axis_number))
        self._close_connection_when_idle()

    def close(self) -> None:
        # Keep the connection alive even when a caller closes the shared hub before its axes.
        with self._reservation_lock:
            self._close_requested = True
        self._close_connection_when_idle()

    def _close_connection_when_idle(self) -> None:
        with self._reservation_lock:
            if not self._close_requested or self._reserved or self._connection_closed:
                return
            self._connection_closed = True
        self._connection.close()


class ZaberLinearAxis(ContinuousAxis):
    """One linear Zaber axis selected by device serial and one-based axis number.

    Setting ``ttl_input`` and ``ttl_trigger`` lets a DAQ pulse on that controller digital input
    step the axis during Voxel acquisitions (see ``ZaberTTLStepper``). ``ttl_status_output`` and
    ``ttl_status_trigger`` optionally pulse a controller digital output after each step.

    Setting all of ``scan_encoder``, ``scan_interpolation``, ``scan_output_width_ns``,
    ``scan_scale_period`` (the analog encoder period, in axis units) and ``scan_acceleration`` lets the
    axis make constant-speed scans while the controller exports that encoder as TTL quadrature (see
    ``ZaberContinuousScanner``).
    """

    def __init__(
        self,
        uid: str,
        *,
        hub: ZaberHub,
        device_serial: int,
        axis_number: int = 1,
        units: str = "um",
        ttl_input: int | None = None,
        ttl_trigger: int | None = None,
        ttl_status_output: int | None = None,
        ttl_status_trigger: int | None = None,
        scan_encoder: int | None = None,
        scan_interpolation: int | None = None,
        scan_output_width_ns: int | None = None,
        scan_scale_period: float | None = None,
        scan_acceleration: float | None = None,
    ) -> None:
        _validate_ttl_config(ttl_input, ttl_trigger, ttl_status_output, ttl_status_trigger)
        _validate_scan_config(
            scan_encoder, scan_interpolation, scan_output_width_ns, scan_scale_period, scan_acceleration
        )
        super().__init__(uid=uid, units=units)
        self.hub = hub
        self.device_serial = device_serial
        self.axis_number = axis_number
        self._position_unit, self._velocity_unit, self._acceleration_unit = _UNITS[units]
        self._closed = False
        self._settings: dict[str, tuple[float, float]] = {}  # name -> (read time, value)
        self._axis = hub.reserve_axis(device_serial, axis_number)
        try:
            if self._axis.axis_type is not AxisType.LINEAR:
                raise ZaberDiscoveryError(  # noqa: TRY301 - release the reservation on validation failure.
                    f"Zaber device serial {device_serial} axis {axis_number} is "
                    f"{self._axis.axis_type.name.lower()}; expected a linear axis"
                )
            self._device = hub.device(device_serial)
            identity = self._axis.identity
            self._peripheral_name = identity.peripheral_name
            self._peripheral_serial_number = identity.peripheral_serial_number
            self._ttl_stepper: ZaberTTLStepper | None = None
            if ttl_input is not None and ttl_trigger is not None:
                self._ttl_stepper = ZaberTTLStepper(
                    lock=hub.lock,
                    axis=self._axis,
                    position_unit=self._position_unit,
                    input_channel=ttl_input,
                    step_trigger=ttl_trigger,
                    status_output=ttl_status_output,
                    status_trigger=ttl_status_trigger,
                )
            self._continuous_scanner: ZaberContinuousScanner | None = None
            if (
                scan_encoder is not None
                and scan_interpolation is not None
                and scan_output_width_ns is not None
                and scan_scale_period is not None
                and scan_acceleration is not None
            ):
                self._continuous_scanner = ZaberContinuousScanner(
                    lock=hub.lock,
                    axis=self._axis,
                    units=units,
                    encoder=scan_encoder,
                    interpolation=scan_interpolation,
                    output_width_ns=scan_output_width_ns,
                    scale_period=scan_scale_period,
                    acceleration=scan_acceleration,
                    invalidate_settings=self._settings.clear,
                )
        except Exception:
            hub.release_axis(device_serial, axis_number)
            raise

        self.log.info(
            "Initialized Zaber %s serial %s axis %s (%s)",
            self._device.name,
            device_serial,
            axis_number,
            self._peripheral_name,
        )

    def _setting(self, name: str, unit: Units) -> float:
        now = time.monotonic()
        cached = self._settings.get(name)
        if cached is not None and now - cached[0] < _SETTINGS_TTL_S:
            return cached[1]
        with self.hub.lock:
            value = self._axis.settings.get(name, unit)
        self._settings[name] = (now, value)
        return value

    def _set_setting(self, name: str, value: float, unit: Units, *, positive: bool = False) -> None:
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite, got {value}")
        if positive and value <= 0:
            raise ValueError(f"{name} must be positive, got {value}")
        with self.hub.lock:
            self._settings.clear()
            self._axis.settings.set(name, value, unit)

    def _start(self, request: str, start: Callable[[], None], wait: bool, timeout_s: float | None) -> None:
        """Run ``start`` under the hub lock, then optionally wait for the axis to stop."""
        if wait:
            _validate_timeout(timeout_s)
        try:
            with self.hub.lock:
                start()
        except CommandFailedException as error:
            # The controller, not the host, enforces limit.min/limit.max on every move.
            if error.details.response_data != "BADDATA":
                raise
            raise ValueError(
                f"Zaber serial {self.device_serial} axis {self.axis_number} rejected {request}; "
                "the target is outside the controller's limits"
            ) from error
        if wait:
            self.await_movement(timeout_s)

    def move_abs(self, position: float, *, wait: bool = False, timeout_s: float | None = None) -> None:
        if not math.isfinite(position):
            raise ValueError(f"position must be finite, got {position}")
        self._start(
            f"move to {position:g} {self.units}",
            lambda: self._axis.move_absolute(position, self._position_unit, wait_until_idle=False),
            wait,
            timeout_s,
        )

    def move_rel(self, delta: float, *, wait: bool = False, timeout_s: float | None = None) -> None:
        if not math.isfinite(delta):
            raise ValueError(f"delta must be finite, got {delta}")
        self._start(
            f"move by {delta:g} {self.units}",
            lambda: self._axis.move_relative(delta, self._position_unit, wait_until_idle=False),
            wait,
            timeout_s,
        )

    def go_home(self, *, wait: bool = False, timeout_s: float | None = None) -> None:
        self._start("home", lambda: self._axis.home(wait_until_idle=False), wait, timeout_s)

    def halt(self) -> None:
        with self.hub.lock:
            self._axis.stop(wait_until_idle=False)

    def await_movement(self, timeout_s: float | None = None) -> None:
        _validate_timeout(timeout_s)
        deadline = time.monotonic() + timeout_s if timeout_s is not None else None
        while self.is_moving:
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError(
                    f"Zaber serial {self.device_serial} axis {self.axis_number} did not stop within {timeout_s:g}s"
                )
            time.sleep(_POLL_INTERVAL_S)

    def _get_position(self) -> float:
        with self.hub.lock:
            return self._axis.get_position(self._position_unit)

    @property
    # Not streamed: Voxel's streamed position already reads the controller each tick.
    @describe(label="Device Position", desc="Unclamped position reported by the controller.")
    def device_position(self) -> float:
        return self._get_position()

    @property
    @describe(label="Position Within Limits")
    def position_within_limits(self) -> bool:
        position = self.device_position
        return self.lower_limit <= position <= self.upper_limit

    @property
    def is_moving(self) -> bool:  # pyrefly: ignore [bad-override]
        # Voxel's abstract property is wrapped by @describe, which Pyrefly models
        # as a method even though its runtime contract and every driver use a property.
        with self.hub.lock:
            return self._axis.is_busy()

    @property
    @describe(label="Is Homed", stream=True)
    def is_homed(self) -> bool:
        with self.hub.lock:
            return self._axis.is_homed()

    def set_zero_here(self) -> None:
        self.set_logical_position(0.0)

    def set_logical_position(self, position: float) -> None:
        self._set_setting(SettingConstants.POS, position, self._position_unit)

    @property
    def upper_limit(self) -> float:
        return self._setting(SettingConstants.LIMIT_MAX, self._position_unit)

    @upper_limit.setter
    def upper_limit(self, value: float) -> None:
        if value <= self.lower_limit:
            raise ValueError("upper limit must be greater than lower limit")
        self._set_setting(SettingConstants.LIMIT_MAX, value, self._position_unit)

    @property
    def lower_limit(self) -> float:
        return self._setting(SettingConstants.LIMIT_MIN, self._position_unit)

    @lower_limit.setter
    def lower_limit(self, value: float) -> None:
        if value >= self.upper_limit:
            raise ValueError("lower limit must be less than upper limit")
        self._set_setting(SettingConstants.LIMIT_MIN, value, self._position_unit)

    @property
    def speed(self) -> float:
        return self._setting(SettingConstants.MAXSPEED, self._velocity_unit)

    @speed.setter
    def speed(self, value: float) -> None:
        self._set_setting(SettingConstants.MAXSPEED, value, self._velocity_unit, positive=True)

    @property
    def acceleration(self) -> float:
        return self._setting(SettingConstants.ACCEL, self._acceleration_unit)

    @acceleration.setter
    def acceleration(self, value: float) -> None:
        self._set_setting(SettingConstants.ACCEL, value, self._acceleration_unit, positive=True)

    @property
    def backlash(self) -> None:
        return None

    @backlash.setter
    def backlash(self, value: float) -> None:
        del value
        raise NotImplementedError("Zaber ASCII axes do not expose generic backlash compensation")

    def _has_home_preset(self) -> bool:
        with self.hub.lock:
            return self._axis.settings.has_setting(SettingConstants.LIMIT_HOME_PRESET)

    @property
    def home(self) -> float | None:
        if not self._has_home_preset():
            return None
        return self._setting(SettingConstants.LIMIT_HOME_PRESET, self._position_unit)

    @home.setter
    def home(self, position: float) -> None:
        if not self._has_home_preset():
            raise NotImplementedError("this Zaber axis does not expose a home preset")
        lower, upper = self.lower_limit, self.upper_limit
        if not lower <= position <= upper:
            raise ValueError(f"home {position} {self.units} is outside limits [{lower}, {upper}]")
        self._set_setting(SettingConstants.LIMIT_HOME_PRESET, position, self._position_unit)

    @property
    @describe(label="Device Serial")
    def serial_number(self) -> int:
        return self.device_serial

    @property
    @describe(label="Device Name")
    def device_name(self) -> str:
        return self._device.name

    @property
    @describe(label="Firmware Version")
    def firmware_version(self) -> str:
        return str(self._device.firmware_version)

    @property
    @describe(label="Peripheral Name")
    def peripheral_name(self) -> str:
        return self._peripheral_name

    @property
    @describe(label="Peripheral Serial")
    def peripheral_serial_number(self) -> int:
        return self._peripheral_serial_number

    def get_ttl_stepper(self) -> TTLStepper | None:
        return self._ttl_stepper

    def get_continuous_scanner(self) -> ContinuousScanner | None:
        return self._continuous_scanner

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            try:
                if self._continuous_scanner is not None:
                    self._continuous_scanner.reset()
            finally:
                if self._ttl_stepper is not None:
                    self._ttl_stepper.reset()
        finally:
            try:
                self.halt()
            finally:
                self.hub.release_axis(self.device_serial, self.axis_number)
