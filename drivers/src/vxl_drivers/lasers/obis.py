"""Coherent OBIS source driver for modules installed in a LightHUB.

``ObisSource`` is a Voxel ``LaserSource``: an OBIS has one emission switch (``SOUR:AM:STAT``) and no
separate output stage, so register it next to Voxel's ``SourceOnlyLaser`` and inject its UID::

    laser_594_source:
      target: vxl_drivers.lasers.obis.ObisSource
      init: { serial_number: "123456", wavelength_nm: 594, port: /dev/serial/by-path/... }
    laser_594:
      target: vxl.devices.laser.source_only.SourceOnlyLaser
      init: { source: laser_594_source }

Modulation modes map to OBIS operating modes: ``software`` is internal CWP (constant power) and
``digital`` is external DIGITAL, where the LightHUB SMA gate switches emission between 0 mW and the
setpoint. Voxel selects ``digital`` before preview and acquisition.

After power-up an OBIS warms up (``SYST:STAT?`` bit 8) and silently ignores ``SOUR:AM:STAT ON``
until it is ready, so ``source_enable`` verifies the read-back state, and ``wait=True`` first waits
for warm-up to finish.
"""

import math
import time
from collections.abc import Iterable
from typing import override

from rigup.device.props import NumericFloat
from serial.tools import list_ports
from serial.tools.list_ports_common import ListPortInfo

from rigup import describe
from vxl.devices.laser.source import LaserSource, ModulationMode
from vxl_drivers.serial import SerialTransport

BAUD_RATE = 115_200
DEFAULT_TIMEOUT_S = 1.0
# Bounded wait for a module still warming up after power-up. Observed warm-ups ended within
# a minute; the wait blocks that laser's device thread, so keep it short.
DEFAULT_STARTUP_TIMEOUT_S = 120.0
STARTUP_POLL_S = 1.0
# SYST:STAT? bit 8: the module is warming up and ignores SOUR:AM:STAT ON.
WARM_UP_STATUS_BIT = 1 << 8
# Setpoints are written in watts with six decimals; allow 10 µW for the module's own rounding while
# still catching a write that was acknowledged but not applied.
SETPOINT_TOLERANCE_MW = 0.01
OBIS_USB_ID = (0x0D4D, 0x003B)
# CWP/CWC are internal; DIGITAL is external, gated by the LightHUB SMA digital input.
OPERATING_MODES = ("CWP", "CWC", "DIGITAL")
MODULATION_TO_OPERATING_MODE = {ModulationMode.SOFTWARE: "CWP", ModulationMode.DIGITAL: "DIGITAL"}
OPERATING_TO_MODULATION_MODE = {mode: key for key, mode in MODULATION_TO_OPERATING_MODE.items()}


class ObisDiscoveryError(ConnectionError):
    """The configured OBIS laser could not be found or identified."""


class ObisCommunicationError(ConnectionError):
    """Communication with an OBIS laser failed."""


class ObisProtocolError(RuntimeError):
    """An OBIS reply was missing, malformed, or reported an error."""


class ObisProtocol:
    """Locked SCPI protocol for one Coherent OBIS USB connection."""

    def __init__(self, port: str, timeout_s: float = DEFAULT_TIMEOUT_S) -> None:
        if not math.isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError("OBIS timeout_s must be finite and positive")
        self.port = port
        self._transport = SerialTransport(port, baud=BAUD_RATE, timeout=timeout_s)
        self._closed = False

    def _exchange(self, command: str) -> str | None:
        if self._closed:
            raise ObisCommunicationError(f"OBIS port {self.port!r} is closed")
        if not command or not command.isascii() or "\r" in command or "\n" in command:
            raise ValueError(f"invalid OBIS command {command!r}")
        try:
            with self._transport.transaction() as serial_port:
                self._transport.drain_input()
                serial_port.write(f"{command}\r\n".encode("ascii"))
                serial_port.flush()
                raw = serial_port.readline()
        except Exception as error:
            raise ObisCommunicationError(f"OBIS transaction failed on {self.port!r} for {command!r}") from error
        if not raw:
            return None
        if not raw.endswith(b"\r\n"):
            raise ObisProtocolError(f"OBIS returned an incomplete reply to {command!r}: {raw!r}")
        try:
            reply = raw.decode("ascii").strip()
        except UnicodeDecodeError as error:
            raise ObisProtocolError(f"OBIS returned a non-ASCII reply to {command!r}") from error
        if reply.startswith("ERR"):
            raise ObisProtocolError(f"OBIS rejected {command!r}: {reply}")
        if not reply:
            raise ObisProtocolError(f"OBIS returned an empty reply to {command!r}")
        return reply

    def query(self, command: str) -> str:
        reply = self._exchange(command)
        if reply is None or reply == "OK":
            raise ObisProtocolError(f"OBIS returned no value to {command!r}")
        return reply

    def command(self, command: str) -> None:
        # Handshaking is a persistent device setting: support both silence and OK without changing it.
        reply = self._exchange(command)
        if reply is not None and reply != "OK":
            raise ObisProtocolError(f"unexpected OBIS acknowledgement to {command!r}: {reply!r}")

    @property
    def model(self) -> str:
        return self.query("SYST:INF:MOD?")

    @property
    def serial_number(self) -> str:
        return self.query("SYST:INF:SNUM?")

    @property
    def firmware(self) -> str:
        return self.query("SYST:INF:FVER?")

    @property
    def wavelength_nm(self) -> int:
        return round(self._query_float("SYST:INF:WAV?"))

    @property
    def specified_power_mw(self) -> float:
        power = self._query_float("SOUR:POW:NOM?") * 1000
        if not math.isfinite(power) or power <= 0:
            raise ObisProtocolError(f"OBIS reports invalid specified power {power} mW")
        return power

    @property
    def enabled(self) -> bool:
        state = self.query("SOUR:AM:STAT?").upper()
        if state not in {"ON", "OFF"}:
            raise ObisProtocolError(f"OBIS reports invalid emission state {state!r}")
        return state == "ON"

    @property
    def operating_mode(self) -> str:
        return self.query("SOUR:AM:SOUR?").upper()

    @property
    def power_setpoint_mw(self) -> float:
        return self._query_float("SOUR:POW:LEV:IMM:AMPL?") * 1000

    @property
    def measured_power_mw(self) -> float:
        return self._query_float("SOUR:POW:LEV?") * 1000

    @property
    def temperature_c(self) -> float:
        return self._query_float("SOUR:TEMP:BAS?", unit="Cc")

    @property
    def fault(self) -> str:
        return self.query("SYST:FAUL?")

    @property
    def status(self) -> int:
        reply = self.query("SYST:STAT?")
        try:
            return int(reply, 16)
        except ValueError as error:
            raise ObisProtocolError(f"OBIS returned a non-hex status {reply!r}") from error

    @property
    def warming_up(self) -> bool:
        return bool(self.status & WARM_UP_STATUS_BIT)

    def set_power_setpoint(self, power_mw: float, maximum_mw: float) -> None:
        if not math.isfinite(power_mw) or not 0 <= power_mw <= maximum_mw:
            raise ValueError(f"OBIS power must be within 0..{maximum_mw:g} mW, got {power_mw}")
        self.command(f"SOUR:POW:LEV:IMM:AMPL {power_mw / 1000:.6f}")

    def enable(self) -> None:
        """Turn emission on and verify it; a warming-up module ignores the command."""
        self.command("SOUR:AM:STAT ON")
        if not self.enabled:
            detail = " while warming up" if self.warming_up else ""
            raise ObisProtocolError(f"OBIS on {self.port!r} ignored SOUR:AM:STAT ON{detail}")

    def disable(self) -> None:
        self.command("SOUR:AM:STAT OFF")
        if self.enabled:
            raise ObisProtocolError(f"OBIS on {self.port!r} remained enabled after shutdown")

    def set_operating_mode(self, mode: str) -> None:
        """Select an internal (CWP, CWC) or external (DIGITAL) mode; the laser stores it in NVM."""
        if mode not in OPERATING_MODES:
            raise ValueError(f"OBIS operating mode must be one of {OPERATING_MODES}, got {mode!r}")
        self.command(f"SOUR:AM:{'EXT' if mode == 'DIGITAL' else 'INT'} {mode}")

    def _query_float(self, command: str, *, unit: str = "") -> float:
        """Query a number, stripping any trailing characters in ``unit`` (e.g. ``"25.0C"``)."""
        reply = self.query(command)
        try:
            return float(reply.rstrip(unit))
        except ValueError as error:
            raise ObisProtocolError(f"OBIS returned a non-numeric value for {command!r}: {reply!r}") from error

    def close(self) -> None:
        if not self._closed:
            self._transport.close()
            self._closed = True


def discover_obis_port(
    serial_number: str,
    timeout_s: float,
    ports: Iterable[ListPortInfo] | None = None,
) -> str:
    """Find an OBIS module by querying internal serials on OBIS USB interfaces."""
    candidates = [
        port for port in (list_ports.comports() if ports is None else ports) if (port.vid, port.pid) == OBIS_USB_ID
    ]
    matches: list[str] = []
    failures: list[str] = []
    for port in candidates:
        protocol = None
        try:
            protocol = ObisProtocol(port.device, timeout_s)
            model = protocol.model
            if "OBIS" in model and protocol.serial_number == serial_number:
                matches.append(port.device)
        except Exception as error:
            failures.append(f"{port.device}: {error}")
        finally:
            if protocol is not None:
                protocol.close()
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise ObisDiscoveryError(f"OBIS serial {serial_number!r} matched multiple ports: {matches}")
    detail = f"; probe failures: {'; '.join(failures)}" if failures else ""
    raise ObisDiscoveryError(f"no OBIS module found with serial {serial_number!r}{detail}")


class ObisSource(LaserSource):
    """One Coherent OBIS module installed in an Omicron LightHUB, as a Voxel laser source.

    Construction identifies the module and turns emission off. A module in a mode Voxel cannot
    express (CWC, ANALOG, MIXED) is switched to CWP with emission off; CWP and DIGITAL are kept, so
    opening does not rewrite the mode the laser stores in non-volatile memory.
    """

    def __init__(
        self,
        uid: str,
        *,
        serial_number: str,
        wavelength_nm: int,
        port: str | None = None,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        startup_timeout_s: float = DEFAULT_STARTUP_TIMEOUT_S,
    ) -> None:
        if not serial_number.strip():
            raise ValueError("OBIS serial_number must not be empty")
        for name, value in (("timeout_s", timeout_s), ("startup_timeout_s", startup_timeout_s)):
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"OBIS {name} must be finite and positive")
        if wavelength_nm <= 0:
            raise ValueError("OBIS wavelength_nm must be positive")
        if port is not None and not port.strip():
            raise ValueError("OBIS port must not be empty")
        selected_port = port or discover_obis_port(serial_number, timeout_s)
        protocol = ObisProtocol(selected_port, timeout_s)
        try:
            model = protocol.model
            actual_serial = protocol.serial_number
            firmware = protocol.firmware
            actual_wavelength = protocol.wavelength_nm
            specified_power = protocol.specified_power_mw
            if "OBIS" not in model or actual_serial != serial_number:
                raise ObisDiscoveryError(  # noqa: TRY301 - close on failed identification.
                    f"unexpected OBIS identity {model!r} {actual_serial!r}"
                )
            if actual_wavelength != wavelength_nm:
                raise ObisDiscoveryError(  # noqa: TRY301 - close on failed identification.
                    f"OBIS serial {serial_number!r} is {actual_wavelength} nm; expected {wavelength_nm} nm"
                )
            protocol.disable()
            if protocol.operating_mode not in OPERATING_TO_MODULATION_MODE:
                _set_operating_mode(protocol, "CWP")
            setpoint = min(max(protocol.power_setpoint_mw, 0.0), specified_power)
        except Exception:
            protocol.close()
            raise
        self._protocol = protocol
        self._serial_number = serial_number
        self._model = model
        self._firmware = firmware
        self._wavelength = actual_wavelength
        self._specified_power_mw = specified_power
        self._setpoint_mw = setpoint
        self._startup_timeout_s = startup_timeout_s
        self._closed = False
        super().__init__(uid=uid)

    @property
    @override
    def wavelength(self) -> float:
        return self._wavelength

    @override
    def source_enable(self, *, wait: bool = False) -> None:
        """Turn emission on (gated by the LightHUB input in digital mode) and verify the read-back.

        With ``wait=True``, first wait up to ``startup_timeout_s`` for warm-up to finish.
        """
        self._ensure_open()
        if self._protocol.enabled:
            return
        if wait:
            deadline = time.monotonic() + self._startup_timeout_s
            while self._protocol.warming_up:
                if time.monotonic() >= deadline:
                    raise ObisProtocolError(
                        f"OBIS {self._serial_number} was still warming up after {self._startup_timeout_s:g} s"
                    )
                time.sleep(STARTUP_POLL_S)
        try:
            self._protocol.enable()
        except Exception as error:
            try:
                self._protocol.disable()
            except Exception as cleanup_error:
                raise ExceptionGroup("OBIS enable and inhibition both failed", [error, cleanup_error]) from None
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
        actual = self._protocol.power_setpoint_mw
        if not math.isclose(actual, value, rel_tol=0, abs_tol=SETPOINT_TOLERANCE_MW):
            raise ObisProtocolError(f"OBIS {self._serial_number} setpoint reads {actual} mW; requested {value} mW")
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
        return self._protocol.temperature_c

    @override
    def _get_modulation_modes(self) -> tuple[ModulationMode, ...]:
        return tuple(MODULATION_TO_OPERATING_MODE)

    @override
    def _read_modulation_mode(self) -> ModulationMode:
        self._ensure_open()
        operating_mode = self._protocol.operating_mode
        if operating_mode not in OPERATING_TO_MODULATION_MODE:
            raise ObisProtocolError(f"OBIS {self._serial_number} is in {operating_mode!r}, set outside this driver")
        return OPERATING_TO_MODULATION_MODE[operating_mode]

    @override
    def _configure_modulation_mode(self, mode: ModulationMode) -> None:
        self._ensure_open()
        self._protocol.disable()
        _set_operating_mode(self._protocol, MODULATION_TO_OPERATING_MODE[mode])

    @property
    @describe(label="Serial Number")
    def serial_number(self) -> str:
        return self._serial_number

    @property
    @describe(label="Model")
    def model(self) -> str:
        return self._model

    @property
    @describe(label="Firmware")
    def firmware(self) -> str:
        return self._firmware

    @property
    @describe(label="Specified Power", units="mW")
    def specified_power_mw(self) -> float:
        return self._specified_power_mw

    @property
    @describe(label="Operating Mode", desc="CWP is software modulation; DIGITAL is the LightHUB gate.")
    def operating_mode(self) -> str:
        return self._protocol.operating_mode

    @property
    @describe(label="Warming Up", desc="The module ignores enable until warm-up finishes.")
    def warming_up(self) -> bool:
        return self._protocol.warming_up

    @property
    @describe(label="Fault")
    def fault(self) -> str:
        return self._protocol.fault

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
            raise RuntimeError(f"OBIS source {self.uid!r} is closed")


def _set_operating_mode(protocol: ObisProtocol, mode: str) -> None:
    """Select a mode with emission off and require the module to report it."""
    protocol.set_operating_mode(mode)
    actual = protocol.operating_mode
    if actual != mode:
        raise ObisProtocolError(f"OBIS on {protocol.port!r} remained in {actual!r}; expected {mode!r}")
