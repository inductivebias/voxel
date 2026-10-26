"""Resolve USB serial numbers to stable serial-port paths."""

import os
from collections.abc import Iterable

from serial.tools import list_ports
from serial.tools.list_ports_common import ListPortInfo


def describe_port(port: ListPortInfo) -> str:
    """Summarize a serial port's path and USB identity for error messages."""
    identity = ":".join(f"{value:04x}" for value in (port.vid, port.pid) if value is not None)
    fields = [port.device]
    if port.serial_number:
        fields.append(f"serial={port.serial_number}")
    if identity:
        fields.append(f"usb={identity}")
    if port.product:
        fields.append(f"product={port.product}")
    return " ".join(fields)


def discover_serial_port(
    serial: str,
    *,
    label: str,
    error: type[Exception],
    ports: Iterable[ListPortInfo] | None = None,
) -> str:
    """Resolve one USB serial number to a unique physical serial port.

    Aliases of the same physical device (e.g. ``/dev/ttyUSB0`` and its
    ``/dev/serial/by-id`` link) count as one match; the stable by-id path wins.

    Args:
        serial: USB serial number to find.
        label: Names the serial in messages, e.g. ``"FW212C serial"``.
        error: Exception type raised when the serial matches zero or several devices.
        ports: Candidate ports; defaults to the host's current serial ports.

    Raises:
        ValueError: ``serial`` is blank.
    """
    if not serial.strip():
        raise ValueError(f"{label} must not be empty")

    candidates = list(list_ports.comports() if ports is None else ports)
    matches = [port for port in candidates if port.serial_number == serial]
    by_physical_path: dict[str, list[ListPortInfo]] = {}
    for port in matches:
        by_physical_path.setdefault(os.path.realpath(port.device), []).append(port)

    if not by_physical_path:
        available = "; ".join(describe_port(port) for port in candidates) or "none"
        raise error(f"no serial port found for {label} {serial!r}; available serial devices: {available}")
    if len(by_physical_path) > 1:
        found = "; ".join(describe_port(port) for port in matches)
        raise error(f"{label} {serial!r} matched multiple physical ports: {found}")

    aliases = next(iter(by_physical_path.values()))
    return min(aliases, key=lambda port: ("/dev/serial/by-id/" not in port.device, port.device)).device
