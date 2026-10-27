"""Read Zaber identity and axis state without constructing motion-capable axes."""

import json
import math
from dataclasses import asdict, dataclass

import click
from vxl_drivers.axes.zaber import ZaberHub
from zaber_motion import Units
from zaber_motion.ascii import SettingConstants


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


@click.command()
@click.option("--host")
@click.option("--port")
@click.option("--serial", type=click.IntRange(min=1), required=True)
def main(host: str | None, port: str | None, serial: int) -> None:
    try:
        result = probe_zaber(serial=serial, host=host, port=port)
    except (RuntimeError, ValueError, ConnectionError) as error:
        raise click.ClickException(str(error)) from error
    click.echo(json.dumps(asdict(result)))


if __name__ == "__main__":
    main()
