"""Omicron LuxX source driver for LightHUB systems.

``LuxxSource`` is a Voxel ``LaserSource``: a LuxX has one emission switch (``LOn``/``LOf``) and no
separate output stage, so register it next to Voxel's ``SourceOnlyLaser`` and inject its UID::

    laser_488_source:
      target: vxl_drivers.lasers.luxx.LuxxSource
      init: { usb_serial: ABCD1234, wavelength_nm: 488 }
    laser_488:
      target: vxl.devices.laser.source_only.SourceOnlyLaser
      init: { source: laser_488_source }

Modulation modes map to the LuxX control mode: ``software`` is APC and ``digital`` is ACC, the mode
in which the LightHUB digital input gates emission. Voxel selects ``digital`` before preview and
acquisition. The setpoint is the 12-bit ``SLP`` level scaled by rated power, as commissioned; in
ACC that level sets diode current, so the mW value is nominal rather than regulated optical power.
"""

import math
from collections.abc import Iterable
from typing import override

from rigup.device.props import NumericFloat
from serial.tools.list_ports_common import ListPortInfo

from rigup import describe
from vxl.devices.laser.source import LaserSource, ModulationMode
from vxl_drivers import serial_discovery
from vxl_drivers.serial import SerialTransport

BAUD_RATE = 500_000
DEFAULT_TIMEOUT_S = 1.0
TERMINATOR = b"\r"
LEVEL_MAX = 0xFFF
MAX_PENDING_STATUS_REPLIES = 8
# Bit 8 of the LuxX operating mode selects automatic power control (APC, set) over automatic
# current control (ACC, clear). A LightHUB digital modulation input drives the LuxX laser-enable
# line, which gates emission only in ACC or analogue mode (LightHUB Ultra manual, Table 3), so
# DAQ gating needs ACC. Commissioned lasers reported F918: APC.
APC_MODE_BIT = 1 << 8
MODULATION_TO_CONTROL_MODE = {ModulationMode.SOFTWARE: "APC", ModulationMode.DIGITAL: "ACC"}
CONTROL_TO_MODULATION_MODE = {mode: key for key, mode in MODULATION_TO_CONTROL_MODE.items()}


class LuxxDiscoveryError(ConnectionError):
    """The configured LuxX laser could not be resolved or identified."""


class LuxxCommunicationError(ConnectionError):
    """Communication with a LuxX laser failed."""


class LuxxProtocolError(RuntimeError):
    """A LuxX reply was missing, malformed, or reported an error."""


def discover_serial_port(
    usb_serial: str,
    ports: Iterable[ListPortInfo] | None = None,
) -> str:
    """Resolve an Omicron USB serial number to one physical serial port."""
    return serial_discovery.discover_serial_port(
        usb_serial, label="LuxX USB serial", error=LuxxDiscoveryError, ports=ports
    )


def frame_command(command: str) -> bytes:
    """Frame one case-sensitive LuxX command."""
    if not command or not command.isascii() or not command.isalnum():
        raise ValueError(f"invalid LuxX command {command!r}")
    return f"?{command}\r".encode("ascii")


def parse_reply(command: str, raw: bytes) -> str:
    """Validate a LuxX reply and return its payload."""
    raw = raw.lstrip(b"\x00")
    if not raw.endswith(TERMINATOR):
        raise LuxxProtocolError(f"LuxX reply to {command!r} was incomplete: {raw!r}")
    reply = raw[:-1].decode("latin-1")

    if reply == "!UK":
        raise LuxxProtocolError(f"LuxX does not recognize command {command!r}")
    prefix = f"!{command[:3]}"
    if not reply.startswith(prefix):
        raise LuxxProtocolError(f"unexpected LuxX reply to {command!r}: {raw!r}")
    payload = reply.removeprefix(prefix)
    if payload.startswith("x"):
        raise LuxxProtocolError(f"LuxX rejected command {command!r}: {payload}")
    return payload


def validate_status_frame(raw: bytes) -> None:
    """Validate an unsolicited LuxX status frame before discarding it."""
    if not 5 <= len(raw) <= 64 or not raw.endswith(TERMINATOR):
        raise LuxxProtocolError(f"invalid LuxX status frame: {raw!r}")
    if raw.startswith(b"$") and raw[1:4].isalpha() and all(32 <= byte <= 126 for byte in raw[4:-1]):
        return
    if len(raw) != 9 or not raw.startswith(b"!GAS"):
        raise LuxxProtocolError(f"invalid LuxX status frame: {raw!r}")
    try:
        int(raw[4:-1], 16)
    except ValueError as error:
        raise LuxxProtocolError(f"invalid LuxX status frame: {raw!r}") from error


class LuxxProtocol:
    """Locked request-response protocol for one LuxX USB connection."""

    def __init__(self, port: str, timeout_s: float = DEFAULT_TIMEOUT_S) -> None:
        if not math.isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError("LuxX timeout_s must be finite and positive")
        self.port = port
        self._transport = SerialTransport(port, baud=BAUD_RATE, timeout=timeout_s)
        self._closed = False

    def query(self, command: str) -> str:
        if self._closed:
            raise LuxxCommunicationError(f"LuxX port {self.port!r} is closed")
        try:
            with self._transport.transaction() as serial_port:
                self._transport.drain_input()
                serial_port.write(frame_command(command))
                serial_port.flush()
                for _ in range(MAX_PENDING_STATUS_REPLIES + 1):
                    raw = serial_port.read_until(TERMINATOR).lstrip(b"\x00")
                    if raw.startswith(b"$") or (command != "GAS" and raw.startswith(b"!GAS")):
                        # The module can report its status after LOf, even after acknowledging
                        # that command. Do not mistake that delayed status for the next reply.
                        validate_status_frame(raw)
                        continue
                    if raw.endswith(TERMINATOR) and not raw.startswith((b"!", b"$")):
                        # The tail of an unsolicited status frame whose head drain_input consumed
                        # before the line went quiet (e.g. b"DP0.59\r" from b"$MDP0.59\r").
                        continue
                    return parse_reply(command, raw)
                raise LuxxProtocolError(  # noqa: TRY301 - report an unmatched bounded transaction.
                    f"LuxX sent too many status replies while waiting for {command!r}"
                )
        except Exception as error:
            if isinstance(error, LuxxProtocolError | ValueError):
                raise
            raise LuxxCommunicationError(f"LuxX transaction failed on {self.port!r} for {command!r}") from error

    @property
    def firmware(self) -> str:
        return self.query("GFw")

    @property
    def serial_number(self) -> str:
        return self.query("GSN")

    @property
    def specification(self) -> tuple[int, float]:
        reply = self.query("GSI")
        fields = reply.split("§")
        if len(fields) != 2:
            raise LuxxProtocolError(f"LuxX returned invalid specification {reply!r}")
        try:
            return int(fields[0]), float(fields[1])
        except ValueError as error:
            raise LuxxProtocolError(f"LuxX returned invalid specification {reply!r}") from error

    @property
    def working_hours(self) -> float:
        return self._query_float("GWH")

    @property
    def enabled(self) -> bool:
        return bool(self._query_hex("GAS") & (1 << 1))

    @property
    def operating_mode(self) -> int:
        mode = self._query_hex("GOM")
        if not 0 <= mode <= 0xFFFF:
            raise LuxxProtocolError(f"LuxX returned an operating mode outside 16 bits: {mode:#x}")
        return mode

    @property
    def control_mode(self) -> str:
        return "APC" if self.operating_mode & APC_MODE_BIT else "ACC"

    def set_control_mode(self, control_mode: str) -> None:
        """Select APC or ACC, writing every other operating-mode bit back unchanged."""
        if control_mode not in CONTROL_TO_MODULATION_MODE:
            raise ValueError(f"LuxX control mode must be APC or ACC, got {control_mode!r}")
        mode = self.operating_mode
        wanted = mode | APC_MODE_BIT if control_mode == "APC" else mode & ~APC_MODE_BIT
        if wanted != mode:
            self.query(f"SOM{wanted:04X}")

    @property
    def fault_code(self) -> int:
        return self._query_hex("GFB")

    @property
    def last_fault_code(self) -> int:
        return self._query_hex("GLF")

    def power_setpoint_mw(self, maximum_mw: float) -> float:
        """Read the setpoint, scaling its 12-bit level by the module's ``maximum_mw``."""
        return self._query_hex("GLP") * maximum_mw / LEVEL_MAX

    @property
    def measured_power_mw(self) -> float:
        return self._query_float("MDP")

    @property
    def diode_temperature_c(self) -> float:
        return self._query_float("MTD")

    @property
    def ambient_temperature_c(self) -> float:
        return self._query_float("MTA")

    def set_power_setpoint(self, power_mw: float, maximum_mw: float) -> None:
        if not math.isfinite(power_mw) or not 0 <= power_mw <= maximum_mw:
            raise ValueError(f"LuxX power must be within 0..{maximum_mw:g} mW, got {power_mw}")
        level = round(power_mw * LEVEL_MAX / maximum_mw)
        self.query(f"SLP{level:03X}")

    def enable(self) -> None:
        self.query("LOn")
        if not self.enabled:
            raise LuxxProtocolError(f"LuxX on {self.port!r} remained off after LOn")

    def disable(self) -> None:
        self.query("LOf")

    def _query_float(self, command: str) -> float:
        reply = self.query(command)
        try:
            return float(reply)
        except ValueError as error:
            raise LuxxProtocolError(f"LuxX returned a non-numeric value for {command!r}: {reply!r}") from error

    def _query_hex(self, command: str) -> int:
        reply = self.query(command)
        try:
            return int(reply, 16)
        except ValueError as error:
            raise LuxxProtocolError(f"LuxX returned a non-hex value for {command!r}: {reply!r}") from error

    def close(self) -> None:
        if not self._closed:
            self._transport.close()
            self._closed = True


class LuxxSource(LaserSource):
    """One USB-controlled Omicron LuxX module in a LightHUB, as a Voxel laser source.

    Construction identifies the module and turns emission off; the control mode is left as found.
    The laser keeps its mode after this driver closes; Voxel sets digital (ACC) before imaging.
    """

    def __init__(
        self,
        uid: str,
        *,
        usb_serial: str,
        wavelength_nm: int,
        timeout_s: float = DEFAULT_TIMEOUT_S,
    ) -> None:
        if wavelength_nm <= 0:
            raise ValueError("LuxX wavelength_nm must be positive")
        port = discover_serial_port(usb_serial)
        protocol = LuxxProtocol(port, timeout_s)
        try:
            firmware = protocol.firmware
            serial_number = protocol.serial_number
            actual_wavelength, specified_power = protocol.specification
            if "LuxX" not in firmware:
                raise LuxxDiscoveryError(  # noqa: TRY301 - close on failed identification.
                    f"device on {port!r} identified as {firmware!r}, not a LuxX"
                )
            if actual_wavelength != wavelength_nm:
                raise LuxxDiscoveryError(  # noqa: TRY301 - close on failed identification.
                    f"LuxX USB serial {usb_serial!r} is {actual_wavelength} nm; expected {wavelength_nm} nm"
                )
            if not math.isfinite(specified_power) or specified_power <= 0:
                raise LuxxProtocolError(  # noqa: TRY301 - close on invalid capabilities.
                    f"LuxX {serial_number} reports invalid specified power {specified_power} mW"
                )
            protocol.disable()
            setpoint = protocol.power_setpoint_mw(specified_power)
        except Exception:
            protocol.close()
            raise

        self._usb_serial = usb_serial
        self._protocol = protocol
        self._firmware = firmware
        self._serial_number = serial_number
        self._wavelength = actual_wavelength
        self._specified_power_mw = specified_power
        self._setpoint_mw = setpoint
        self._closed = False
        super().__init__(uid=uid)
        self.log.info(
            "Initialized LuxX %s (%s nm, %s mW) on %s with emission disabled",
            serial_number,
            actual_wavelength,
            specified_power,
            port,
        )

    @property
    @override
    def wavelength(self) -> float:
        return self._wavelength

    @override
    def source_enable(self, *, wait: bool = False) -> None:
        """Turn emission on (gated by the LightHUB input in digital mode) and verify the read-back."""
        del wait  # LuxX reports enable at once; no startup check beyond the read-back.
        self._ensure_open()
        if self._protocol.enabled:
            return
        try:
            self._protocol.enable()
        except Exception as error:
            try:
                self._protocol.disable()
            except Exception as cleanup_error:
                raise ExceptionGroup("LuxX enable and inhibition both failed", [error, cleanup_error]) from None
            raise

    @override
    def source_disable(self) -> None:
        if not self._closed:
            self._protocol.disable()

    @property
    @override
    def is_source_enabled(self) -> bool:
        return not self._closed and self._protocol.enabled

    @property
    @override
    def source_power_setpoint_mw(self) -> float:
        """The last requested setpoint, so it compares exactly with what Voxel asked for."""
        self._ensure_open()
        return NumericFloat(self._setpoint_mw, minimum=0.0, maximum=self._specified_power_mw)

    @source_power_setpoint_mw.setter
    @override
    def source_power_setpoint_mw(self, value: float) -> None:
        self._ensure_open()
        self._protocol.set_power_setpoint(value, self._specified_power_mw)
        actual = self._protocol.power_setpoint_mw(self._specified_power_mw)
        # The 12-bit level quantizes the request; allow one level of difference.
        if abs(actual - value) > self._specified_power_mw / LEVEL_MAX:
            raise LuxxProtocolError(f"LuxX {self._serial_number} setpoint reads {actual:.3f} mW; requested {value} mW")
        self._setpoint_mw = float(value)

    @property
    @override
    def source_power_mw(self) -> float:
        self._ensure_open()
        return self._protocol.measured_power_mw

    @property
    @override
    def temperature_c(self) -> float:
        self._ensure_open()
        return self._protocol.diode_temperature_c

    @override
    def _get_modulation_modes(self) -> tuple[ModulationMode, ...]:
        return tuple(MODULATION_TO_CONTROL_MODE)

    @override
    def _read_modulation_mode(self) -> ModulationMode:
        self._ensure_open()
        return CONTROL_TO_MODULATION_MODE[self._protocol.control_mode]

    @override
    def _configure_modulation_mode(self, mode: ModulationMode) -> None:
        self._ensure_open()
        self._protocol.disable()
        control_mode = MODULATION_TO_CONTROL_MODE[mode]
        self._protocol.set_control_mode(control_mode)
        actual = self._protocol.control_mode
        if actual != control_mode:
            raise LuxxProtocolError(f"LuxX {self._serial_number} remained in {actual}; expected {control_mode}")

    @property
    @describe(label="Ambient Temperature", units="°C")
    def ambient_temperature_c(self) -> float:
        return self._protocol.ambient_temperature_c

    @property
    @describe(label="USB Serial")
    def usb_serial(self) -> str:
        return self._usb_serial

    @property
    @describe(label="Laser Serial")
    def serial_number(self) -> str:
        return self._serial_number

    @property
    @describe(label="Firmware")
    def firmware(self) -> str:
        return self._firmware.replace("§", " ")

    @property
    @describe(label="Specified Power", units="mW")
    def specified_power_mw(self) -> float:
        return self._specified_power_mw

    @property
    @describe(
        label="Control Mode",
        desc="APC holds optical power; ACC holds diode current and lets the LightHUB gate emission.",
    )
    def control_mode(self) -> str:
        return self._protocol.control_mode

    @property
    @describe(label="Working Hours", units="h")
    def working_hours(self) -> float:
        return self._protocol.working_hours

    @property
    @describe(label="Fault Code")
    def fault_code(self) -> int:
        return self._protocol.fault_code

    @property
    @describe(label="Last Fault Code")
    def last_fault_code(self) -> int:
        return self._protocol.last_fault_code

    @override
    def close(self) -> None:
        if self._closed:
            return
        try:
            self._protocol.disable()
        finally:
            self._protocol.close()
            self._closed = True

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError(f"LuxX source {self.uid!r} is closed")
