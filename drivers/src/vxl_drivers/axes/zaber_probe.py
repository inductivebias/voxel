"""Read Zaber identity, telemetry, and axis state without moving or un-homing axes."""

import json
import math
import os
import time
from dataclasses import asdict, dataclass
from typing import Any

import click
from vxl_drivers.axes.zaber import ZaberHub
from zaber_motion import Units
from zaber_motion.ascii import Connection, SettingConstants

_RATED_VOLTAGE_MIN_V = 24.0
_RATED_VOLTAGE_MAX_V = 48.0
_ADC_ANOMALY_MAX_V = 5.0
_DEFAULT_WARNING_SETTLE_S = 0.5


@dataclass(frozen=True)
class AxisStatus:
    number: int
    position_um: float
    lower_um: float
    upper_um: float
    homed: bool
    peripheral_serial: int


@dataclass(frozen=True)
class ZaberStatus:
    serial: int
    model: str
    axes: tuple[AxisStatus, ...]


@dataclass(frozen=True)
class AxisDiagnostic:
    number: int
    peripheral_name: str
    peripheral_serial: int
    position_um: float
    encoder_position_um: float | None
    following_error_um: float | None
    lower_um: float
    upper_um: float
    homed: bool
    busy: bool
    driver_enabled: bool | None
    speed_um_s: float | None
    accel_um_s2: float | None
    decel_um_s2: float | None
    ramptime_ms: float | None
    warnings_before: tuple[str, ...]
    warnings_after: tuple[str, ...]


@dataclass(frozen=True)
class ZaberDiagnostic:
    serial: int
    model: str
    firmware: str
    tcp_port: int | None
    system_voltage_before_v: float | None
    system_voltage_after_v: float | None
    system_current_a: float | None
    system_temperature_c: float | None
    device_warnings_before: tuple[str, ...]
    device_warnings_after: tuple[str, ...]
    warnings_cleared: bool
    wv_status: str
    diagnosis: str
    axes: tuple[AxisDiagnostic, ...]


def _optional_float(settings: Any, name: str, unit: Units = Units.NATIVE) -> float | None:
    if settings is None:
        return None
    try:
        value = float(settings.get(name, unit))
    except Exception:
        return None
    return value if math.isfinite(value) else None


def _open_diagnostic_hub(
    *,
    host: str | None,
    port: str | None,
    tcp_port: int | None,
) -> tuple[ZaberHub, int | None]:
    if host is None:
        return ZaberHub("zaber-diagnose", port=port), None
    ports = (tcp_port,) if tcp_port is not None else (Connection.TCP_PORT_DEVICE_ONLY, Connection.TCP_PORT_CHAIN)
    last_error: Exception | None = None
    for candidate in ports:
        try:
            return ZaberHub("zaber-diagnose", host=host, tcp_port=candidate), candidate
        except Exception as error:
            last_error = error
    if last_error is not None:
        raise last_error
    raise ConnectionError("could not open Zaber connection")


def _classify_wv(
    *,
    had_wv: bool,
    has_wv: bool,
    voltage_v: float | None,
    warnings_cleared: bool,
) -> tuple[str, str]:
    v_text = f"{voltage_v:.2f} V" if voltage_v is not None else "unavailable"
    if not had_wv and not has_wv:
        return (
            "none",
            f"No WV warning present (system.voltage = {v_text}; rated 24-48 V DC).",
        )
    if had_wv and not has_wv:
        msg = (
            "WV was a stale latched warning from an earlier move/halt and stayed clear "
            f"after `warnings clear` (system.voltage = {v_text})."
        )
        return ("cleared_stale", msg)
    if not warnings_cleared:
        msg = (
            f"WV is currently asserted (system.voltage = {v_text}); re-run with --clear-warnings "
            "while axes are idle to test whether it is a stale flag or actively re-asserted."
        )
        return ("active_untested", msg)
    if voltage_v is not None and voltage_v < _ADC_ANOMALY_MAX_V:
        msg = (
            f"WV re-asserted immediately while idle and `system.voltage` reads {v_text} "
            "(< 24 V while controller/stages are powered). This matches the X-MCC voltage-sense "
            "ADC / 2-pin power terminal contact anomaly: inspect and re-seat the removable 2-pin "
            "24-48 V DC terminal block on the controller."
        )
        return ("persistent_adc_anomaly", msg)
    if voltage_v is not None and not (_RATED_VOLTAGE_MIN_V <= voltage_v <= _RATED_VOLTAGE_MAX_V):
        msg = (
            f"WV re-asserted immediately while idle; `system.voltage` reads {v_text}, "
            "outside the rated 24-48 V DC window."
        )
        return ("persistent_out_of_range", msg)
    return (
        "persistent_nominal_voltage",
        f"WV re-asserted while idle despite `system.voltage` reading {v_text}.",
    )


def probe_zaber(*, serial: int, host: str | None = None, port: str | None = None) -> ZaberStatus:
    """Require the selected controller and read every stationary, valid axis without writes."""
    hub = ZaberHub("identity-probe", host=host, port=port)
    try:
        device = hub.device(serial)
        if device.axis_count < 1:
            raise RuntimeError("Zaber controller has no axes")
        axes = []
        for number in range(1, device.axis_count + 1):
            axis = device.get_axis(number)
            position, lower, upper = (
                axis.settings.get(name, Units.LENGTH_MICROMETRES)
                for name in (SettingConstants.POS, SettingConstants.LIMIT_MIN, SettingConstants.LIMIT_MAX)
            )
            if (
                not all(math.isfinite(value) for value in (position, lower, upper))
                or not lower <= position <= upper
                or lower >= upper
            ):
                raise RuntimeError(f"Zaber axis {number} has invalid position or limits")
            if axis.is_busy() or axis.warnings.get_flags():
                raise RuntimeError(f"Zaber axis {number} is moving or reports warnings")
            if not axis.identity.peripheral_name or axis.identity.peripheral_serial_number < 1:
                raise RuntimeError(f"Zaber axis {number} has no peripheral identity")
            axes.append(
                AxisStatus(number, position, lower, upper, axis.is_homed(), axis.identity.peripheral_serial_number)
            )
        return ZaberStatus(serial, device.name, tuple(axes))
    finally:
        hub.close()


def diagnose_zaber(
    *,
    serial: int | None = None,
    host: str | None = None,
    port: str | None = None,
    tcp_port: int | None = None,
    clear_warnings: bool = True,
    settle_s: float = _DEFAULT_WARNING_SETTLE_S,
) -> ZaberDiagnostic:
    """Inspect controller voltage, warnings, and axis motion settings without moving or un-homing any stage."""
    hub, connected_tcp_port = _open_diagnostic_hub(host=host, port=port, tcp_port=tcp_port)
    try:
        if serial is None:
            if len(hub.device_serials) != 1:
                found = ", ".join(str(s) for s in hub.device_serials)
                raise ValueError(f"multiple Zaber devices detected ({found}); specify --serial")
            serial = hub.device_serials[0]
        device = hub.device(serial)
        if device.axis_count < 1:
            raise RuntimeError("Zaber controller has no axes")

        dev_settings = getattr(device, "settings", None)
        dev_warnings_obj = getattr(device, "warnings", None)
        voltage_before = _optional_float(dev_settings, SettingConstants.SYSTEM_VOLTAGE)
        current_a = _optional_float(dev_settings, SettingConstants.SYSTEM_CURRENT)
        temperature_c = _optional_float(dev_settings, SettingConstants.SYSTEM_TEMPERATURE)
        dev_flags_before = tuple(sorted(dev_warnings_obj.get_flags())) if dev_warnings_obj is not None else ()

        axis_objs = [device.get_axis(number) for number in range(1, device.axis_count + 1)]
        axes_before: list[dict[str, Any]] = []
        any_busy = False
        for axis in axis_objs:
            pos = axis.settings.get(SettingConstants.POS, Units.LENGTH_MICROMETRES)
            enc = _optional_float(axis.settings, SettingConstants.ENCODER_POS, Units.LENGTH_MICROMETRES)
            lower = axis.settings.get(SettingConstants.LIMIT_MIN, Units.LENGTH_MICROMETRES)
            upper = axis.settings.get(SettingConstants.LIMIT_MAX, Units.LENGTH_MICROMETRES)
            driver_en = _optional_float(axis.settings, SettingConstants.DRIVER_ENABLED)
            busy = axis.is_busy()
            any_busy = any_busy or busy
            axes_before.append(
                {
                    "number": axis.axis_number,
                    "peripheral_name": axis.identity.peripheral_name,
                    "peripheral_serial": axis.identity.peripheral_serial_number,
                    "position_um": pos,
                    "encoder_position_um": enc,
                    "following_error_um": (enc - pos) if enc is not None else None,
                    "lower_um": lower,
                    "upper_um": upper,
                    "homed": axis.is_homed(),
                    "busy": busy,
                    "driver_enabled": None if driver_en is None else bool(round(driver_en)),
                    "speed_um_s": _optional_float(
                        axis.settings, SettingConstants.MAXSPEED, Units.VELOCITY_MICROMETRES_PER_SECOND
                    ),
                    "accel_um_s2": _optional_float(
                        axis.settings,
                        SettingConstants.MOTION_ACCELONLY,
                        Units.ACCELERATION_MICROMETRES_PER_SECOND_SQUARED,
                    ),
                    "decel_um_s2": _optional_float(
                        axis.settings,
                        SettingConstants.MOTION_DECELONLY,
                        Units.ACCELERATION_MICROMETRES_PER_SECOND_SQUARED,
                    ),
                    "ramptime_ms": _optional_float(axis.settings, SettingConstants.MOTION_ACCEL_RAMPTIME),
                    "warnings_before": tuple(sorted(axis.warnings.get_flags())),
                }
            )

        did_clear = clear_warnings and not any_busy
        if did_clear:
            for axis in axis_objs:
                axis.warnings.clear_flags()
            if dev_warnings_obj is not None:
                dev_warnings_obj.clear_flags()
            if settle_s > 0:
                time.sleep(settle_s)

        voltage_after = _optional_float(dev_settings, SettingConstants.SYSTEM_VOLTAGE) if did_clear else voltage_before
        dev_flags_after = (
            tuple(sorted(dev_warnings_obj.get_flags()))
            if (did_clear and dev_warnings_obj is not None)
            else dev_flags_before
        )
        axis_diagnostics: list[AxisDiagnostic] = []
        for axis, info in zip(axis_objs, axes_before, strict=True):
            flags_after = tuple(sorted(axis.warnings.get_flags())) if did_clear else info["warnings_before"]
            axis_diagnostics.append(AxisDiagnostic(**info, warnings_after=flags_after))

        had_wv = "WV" in dev_flags_before or any("WV" in a.warnings_before for a in axis_diagnostics)
        has_wv = "WV" in dev_flags_after or any("WV" in a.warnings_after for a in axis_diagnostics)
        wv_status, diagnosis = _classify_wv(
            had_wv=had_wv,
            has_wv=has_wv,
            voltage_v=voltage_after,
            warnings_cleared=did_clear,
        )
        return ZaberDiagnostic(
            serial=serial,
            model=device.name,
            firmware=str(getattr(device, "firmware_version", "unknown")),
            tcp_port=connected_tcp_port,
            system_voltage_before_v=voltage_before,
            system_voltage_after_v=voltage_after,
            system_current_a=current_a,
            system_temperature_c=temperature_c,
            device_warnings_before=dev_flags_before,
            device_warnings_after=dev_flags_after,
            warnings_cleared=did_clear,
            wv_status=wv_status,
            diagnosis=diagnosis,
            axes=tuple(axis_diagnostics),
        )
    finally:
        hub.close()


def _format_diagnostic_report(report: ZaberDiagnostic) -> str:
    port_info = f" (TCP port {report.tcp_port})" if report.tcp_port is not None else ""
    dev_before = list(report.device_warnings_before)
    dev_after = list(report.device_warnings_after)
    lines = [
        f"Zaber {report.model} serial {report.serial} firmware {report.firmware}{port_info}",
        (
            f"  system.voltage: before={report.system_voltage_before_v} V, "
            f"after={report.system_voltage_after_v} V (rated 24.0-48.0 V DC)"
        ),
        f"  system.current: {report.system_current_a} A | system.temperature: {report.system_temperature_c} C",
        f"  device warnings: before={dev_before} -> after={dev_after}",
        f"  WV status [{report.wv_status}]: {report.diagnosis}",
    ]
    for axis in report.axes:
        lines.append(
            f"  Axis {axis.number} ({axis.peripheral_name} #{axis.peripheral_serial}): "
            f"pos={axis.position_um:.3f} um, enc={axis.encoder_position_um} um, "
            f"homed={axis.homed}, busy={axis.busy}, driver_enabled={axis.driver_enabled}, "
            f"speed={axis.speed_um_s} um/s, accel={axis.accel_um_s2} um/s^2, "
            f"ramptime={axis.ramptime_ms} ms, "
            f"warnings={list(axis.warnings_before)} -> {list(axis.warnings_after)}"
        )
    return "\n".join(lines)


@click.command()
@click.option("--host", default=lambda: os.environ.get("VOXEL_ZABER_HOST"))
@click.option("--port", default=lambda: os.environ.get("VOXEL_ZABER_PORT"))
@click.option("--serial", type=click.IntRange(min=1), required=False)
@click.option("--tcp-port", type=click.IntRange(min=1, max=65535), default=None)
@click.option("--diagnose", is_flag=True, help="Read system.voltage, motion settings, and test WV warning persistence.")
@click.option(
    "--clear-warnings/--no-clear-warnings",
    default=True,
    help="Clear warnings while idle during --diagnose to test whether WV immediately re-asserts.",
)
@click.option("--json-output", "as_json", is_flag=True, help="Emit JSON output in --diagnose mode.")
def main(
    host: str | None,
    port: str | None,
    serial: int | None,
    tcp_port: int | None,
    diagnose: bool,
    clear_warnings: bool,
    as_json: bool,
) -> None:
    if serial is None and (env_serial := os.environ.get("VOXEL_ZABER_DEVICE_SERIAL")):
        serial = int(env_serial)
    try:
        if diagnose:
            diag = diagnose_zaber(
                serial=serial,
                host=host,
                port=port,
                tcp_port=tcp_port,
                clear_warnings=clear_warnings,
            )
            if as_json:
                click.echo(json.dumps(asdict(diag), indent=2))
            else:
                click.echo(_format_diagnostic_report(diag))
                click.echo(json.dumps(asdict(diag)))
            return
        if serial is None:
            raise click.UsageError("Missing option '--serial'.")
        result = probe_zaber(serial=serial, host=host, port=port)
    except (RuntimeError, ValueError, ConnectionError) as error:
        raise click.ClickException(str(error)) from error
    click.echo(json.dumps(asdict(result)))


@click.command()
@click.option("--host", default=lambda: os.environ.get("VOXEL_ZABER_HOST"))
@click.option("--port", default=lambda: os.environ.get("VOXEL_ZABER_PORT"))
@click.option("--serial", type=click.IntRange(min=1), required=False)
@click.option("--tcp-port", type=click.IntRange(min=1, max=65535), default=None)
@click.option(
    "--clear-warnings/--no-clear-warnings",
    default=True,
    help="Clear warnings while idle to test whether WV is stale or actively re-asserted.",
)
@click.option("--json", "as_json", is_flag=True, help="Emit JSON only.")
def diagnose_main(
    host: str | None,
    port: str | None,
    serial: int | None,
    tcp_port: int | None,
    clear_warnings: bool,
    as_json: bool,
) -> None:
    """Inspect Zaber system.voltage, axis motion settings, and WV warning persistence without moving any stage."""
    if serial is None and (env_serial := os.environ.get("VOXEL_ZABER_DEVICE_SERIAL")):
        serial = int(env_serial)
    try:
        diag = diagnose_zaber(
            serial=serial,
            host=host,
            port=port,
            tcp_port=tcp_port,
            clear_warnings=clear_warnings,
        )
    except (RuntimeError, ValueError, ConnectionError) as error:
        raise click.ClickException(str(error)) from error
    if as_json:
        click.echo(json.dumps(asdict(diag), indent=2))
    else:
        click.echo(_format_diagnostic_report(diag))
        click.echo(json.dumps(asdict(diag)))


if __name__ == "__main__":
    main()
