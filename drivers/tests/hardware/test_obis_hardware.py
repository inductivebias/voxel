import os

import pytest
from vxl_drivers.lasers.obis import ObisProtocol, discover_obis_port

SERIAL_VAR = "VOXEL_OBIS_SERIAL"
WAVELENGTH_VAR = "VOXEL_OBIS_WAVELENGTH_NM"

pytestmark = pytest.mark.hardware


@pytest.mark.skipif(
    not os.getenv(SERIAL_VAR) or not os.getenv(WAVELENGTH_VAR),
    reason=f"set {SERIAL_VAR} and {WAVELENGTH_VAR} to read one OBIS module",
)
def test_obis_hardware_reads_identity_status_and_capabilities_without_control() -> None:
    serial = os.environ[SERIAL_VAR]
    expected_wavelength = int(os.environ[WAVELENGTH_VAR])
    port = discover_obis_port(serial, timeout_s=1.0)
    protocol = ObisProtocol(port, timeout_s=1.0)
    try:
        assert "OBIS" in protocol.model
        assert protocol.serial_number == serial
        assert protocol.wavelength_nm == expected_wavelength
        assert protocol.specified_power_mw > 0
        print(
            f"OBIS model={protocol.model!r} serial={serial} firmware={protocol.firmware} "
            f"wavelength_nm={protocol.wavelength_nm} "
            f"specified_power_mw={protocol.specified_power_mw} enabled={protocol.enabled} "
            f"setpoint_mw={protocol.power_setpoint_mw} measured_power_mw={protocol.measured_power_mw} "
            f"temperature_c={protocol.temperature_c} fault={protocol.fault!r}"
        )
    finally:
        protocol.close()
