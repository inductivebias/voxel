from pathlib import Path

import pytest
from serial.tools.list_ports_common import ListPortInfo
from vxl_drivers.serial_discovery import discover_serial_port


class DiscoveryError(ConnectionError):
    pass


def make_port(device: str, serial: str | None) -> ListPortInfo:
    port = ListPortInfo(device)
    port.serial_number = serial
    port.vid = 0x0403
    port.pid = 0x6001
    port.product = "USB Serial Port"
    return port


def discover(ports: list[ListPortInfo], serial: str = "TARGET") -> str:
    return discover_serial_port(serial, label="Test serial", error=DiscoveryError, ports=ports)


def test_discover_serial_port_exact_match_returns_device() -> None:
    ports = [make_port("/dev/ttyUSB0", "OTHER"), make_port("/dev/ttyUSB1", "TARGET")]

    assert discover(ports) == "/dev/ttyUSB1"


def test_discover_serial_port_missing_match_reports_candidates() -> None:
    ports = [make_port("/dev/ttyUSB0", "OTHER")]

    with pytest.raises(DiscoveryError, match=r"Test serial 'TARGET'.*serial=OTHER usb=0403:6001"):
        discover(ports)


def test_discover_serial_port_blank_serial_rejected() -> None:
    with pytest.raises(ValueError, match="Test serial must not be empty"):
        discover([], serial=" ")


def test_discover_serial_port_multiple_physical_matches_rejected() -> None:
    ports = [make_port("/dev/ttyUSB0", "TARGET"), make_port("/dev/ttyUSB1", "TARGET")]

    with pytest.raises(DiscoveryError, match="multiple physical ports"):
        discover(ports)


def test_discover_serial_port_aliases_same_device_prefers_by_id(tmp_path: Path) -> None:
    physical = tmp_path / "ttyUSB0"
    physical.touch()
    raw_alias = tmp_path / "dev" / "ttyUSB0"
    stable_alias = tmp_path / "dev" / "serial" / "by-id" / "wheel"
    raw_alias.parent.mkdir()
    stable_alias.parent.mkdir(parents=True)
    raw_alias.symlink_to(physical)
    stable_alias.symlink_to(physical)
    ports = [make_port(str(raw_alias), "TARGET"), make_port(str(stable_alias), "TARGET")]

    assert discover(ports) == str(stable_alias)
