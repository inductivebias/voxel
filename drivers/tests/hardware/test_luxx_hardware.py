import os

import pytest
from vxl_drivers.lasers.luxx import LuxxProtocol, LuxxSource, discover_serial_port

USB_SERIAL_VAR = "VOXEL_OMICRON_USB_SERIAL"
WAVELENGTH_VAR = "VOXEL_OMICRON_WAVELENGTH_NM"

pytestmark = [
    pytest.mark.hardware,
    pytest.mark.skipif(
        not os.getenv(USB_SERIAL_VAR) or not os.getenv(WAVELENGTH_VAR),
        reason=f"set {USB_SERIAL_VAR} and {WAVELENGTH_VAR} to read one LuxX laser",
    ),
]


def test_luxx_hardware_reads_identity_status_and_capabilities_without_control() -> None:
    usb_serial = os.environ[USB_SERIAL_VAR]
    expected_wavelength = int(os.environ[WAVELENGTH_VAR])
    port = discover_serial_port(usb_serial)
    protocol = LuxxProtocol(port, timeout_s=1.0)
    try:
        wavelength, specified_power = protocol.specification
        assert "LuxX" in protocol.firmware
        assert protocol.serial_number
        assert wavelength == expected_wavelength
        assert specified_power > 0
        assert protocol.working_hours >= 0
        assert protocol.diode_temperature_c > 0
        assert protocol.ambient_temperature_c > 0
        print(
            f"LuxX usb_serial={usb_serial} laser_serial={protocol.serial_number} "
            f"wavelength_nm={wavelength} specified_power_mw={specified_power} "
            f"enabled={protocol.enabled} measured_power_mw={protocol.measured_power_mw} "
            f"fault=0x{protocol.fault_code:04X} last_fault=0x{protocol.last_fault_code:04X}"
        )
    finally:
        protocol.close()


@pytest.mark.skipif(
    os.getenv("VOXEL_OMICRON_ALLOW_CONTROL") != "YES",
    reason="set VOXEL_OMICRON_ALLOW_CONTROL=YES to verify the driver opens and closes with emission off",
)
def test_luxx_hardware_driver_opens_and_closes_with_emission_disabled() -> None:
    source = LuxxSource(
        "luxx_hardware_test",
        usb_serial=os.environ[USB_SERIAL_VAR],
        wavelength_nm=int(os.environ[WAVELENGTH_VAR]),
    )
    try:
        assert not source.is_source_enabled
        assert source.fault_code == 0
    finally:
        source.close()
