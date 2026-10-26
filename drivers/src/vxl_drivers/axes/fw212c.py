"""Thorlabs FW212C motorized filter-wheel driver."""

import math
import os
import threading
from collections.abc import Iterable

from serial.tools import list_ports
from serial.tools.list_ports_common import ListPortInfo
from vxl_drivers.serial import SerialTransport
from vxl_drivers.serial_discovery import describe_port, discover_serial_port

from rigup import describe
from vxl.devices.axes.discrete import DiscreteAxis
from vxl.devices.axes.discrete.base import DiscreteAxisState

BAUD_RATE = 115_200
SLOT_COUNT = 12
DEFAULT_TIMEOUT_S = 5.0
DISCOVERY_PROBE_TIMEOUT_S = 0.5
PROMPT = b">"
FW212C_USB_IDS = frozenset({(0x0403, 0x6001)})


class FW212CDiscoveryError(ConnectionError):
    """The configured FW212C could not be resolved to exactly one serial port."""


class FW212CCommunicationError(ConnectionError):
    """Communication with an FW212C failed."""


class FW212CProtocolError(RuntimeError):
    """An FW212C reply was missing, malformed, or reported an error."""


def _check_timeout(name: str, timeout_s: float) -> None:
    if not math.isfinite(timeout_s) or timeout_s <= 0:
        raise ValueError(f"FW212C {name} must be finite and positive")


def parse_reply(command: str, raw: bytes) -> tuple[str, ...]:
    """Split a prompt-terminated reply into nonempty lines, dropping any command echo."""
    try:
        text = raw.removesuffix(PROMPT).decode("ascii")
    except UnicodeDecodeError as error:
        raise FW212CProtocolError(f"FW212C returned a non-ASCII reply to {command!r}") from error

    lines = [line.strip().removeprefix(">").strip() for line in text.splitlines()]
    lines = [line for line in lines if line]
    if lines and lines[0].casefold() == command.casefold():
        lines.pop(0)

    for line in lines:
        if line.casefold().startswith(("command error", "error")):
            raise FW212CProtocolError(f"FW212C rejected {command!r}: {line}")
    return tuple(lines)


def resolve_serial_port(
    serial: str,
    timeout_s: float,
    ports: Iterable[ListPortInfo] | None = None,
) -> str:
    """Resolve by USB serial, falling back to a bounded read-only protocol probe."""
    candidates = list(list_ports.comports() if ports is None else ports)
    try:
        return discover_serial_port(serial, label="FW212C serial", error=FW212CDiscoveryError, ports=candidates)
    except FW212CDiscoveryError as error:
        if any(port.serial_number == serial for port in candidates):
            raise
        discovery_error = error

    # A missing USB serial descriptor is recoverable, but a conflicting serial is
    # authoritative. Restrict probing to the commissioned FTDI VID/PID so built-in
    # UARTs, the Zaber controller, and unrelated USB serial devices are untouched.
    probe_ports: dict[str, ListPortInfo] = {}
    for port in candidates:
        if not port.serial_number and (port.vid, port.pid) in FW212C_USB_IDS:
            probe_ports.setdefault(os.path.realpath(port.device), port)

    verified: list[ListPortInfo] = []
    failures: list[str] = []
    probe_timeout_s = min(timeout_s, DISCOVERY_PROBE_TIMEOUT_S)
    for port in probe_ports.values():
        try:
            protocol, _identity = _open_verified(port.device, serial, probe_timeout_s)
        except Exception as error:
            failures.append(f"{port.device}: {error}")
            continue
        protocol.close()
        verified.append(port)

    if len(verified) == 1:
        return verified[0].device
    if len(verified) > 1:
        found = "; ".join(describe_port(port) for port in verified)
        raise FW212CDiscoveryError(f"FW212C serial {serial!r} matched multiple verified FW212C ports: {found}")

    detail = f"; probe failures: {'; '.join(failures)}" if failures else ""
    raise FW212CDiscoveryError(f"{discovery_error}{detail}") from discovery_error


class FW212CProtocol:
    """Locked prompt-terminated command protocol for one FW212C connection."""

    def __init__(self, port: str, timeout_s: float = DEFAULT_TIMEOUT_S) -> None:
        _check_timeout("timeout_s", timeout_s)
        self.port = port
        self._transport = SerialTransport(port, baud=BAUD_RATE, timeout=timeout_s)
        self._closed = False

    def _transact(self, command: str, timeout_s: float | None = None) -> tuple[str, ...]:
        if self._closed:
            raise FW212CCommunicationError(f"FW212C port {self.port!r} is closed")
        if timeout_s is not None:
            _check_timeout("transaction timeout_s", timeout_s)

        try:
            with self._transport.transaction() as serial_port:
                previous_timeout = serial_port.timeout
                try:
                    if timeout_s is not None:
                        serial_port.timeout = timeout_s
                    self._transport.drain_input()
                    serial_port.write(command.encode("ascii") + b"\r")
                    serial_port.flush()
                    raw = serial_port.read_until(PROMPT)
                finally:
                    serial_port.timeout = previous_timeout
        except Exception as error:
            raise FW212CCommunicationError(f"FW212C transaction failed on {self.port!r} for {command!r}") from error
        # read_until returns whatever arrived when the serial timeout expires.
        if not raw.endswith(PROMPT):
            raise TimeoutError(f"FW212C did not return the ready prompt for {command!r} on {self.port!r}: {raw!r}")
        return parse_reply(command, raw)

    def query(self, command: str) -> str:
        """Run a query and return its final nonempty response line."""
        lines = self._transact(command)
        if not lines:
            raise FW212CProtocolError(f"FW212C returned no value for query {command!r}")
        return lines[-1]

    @property
    def identity(self) -> str:
        return self.query("*idn?")

    @property
    def position_count(self) -> int:
        return self._query_int("pcount?")

    @property
    def native_position(self) -> int:
        return self._query_int("pos?")

    def move_native(self, position: int, timeout_s: float | None = None) -> None:
        """Command a one-based position; the reply arrives once the wheel settles."""
        self._transact(f"pos={position}", timeout_s)

    def _query_int(self, command: str) -> int:
        reply = self.query(command)
        try:
            return int(reply)
        except ValueError as error:
            raise FW212CProtocolError(f"FW212C returned a non-integer value for {command!r}: {reply!r}") from error

    def close(self) -> None:
        if not self._closed:
            self._transport.close()
            self._closed = True


def _validate_device(identity: str, position_count: int, serial: str, port: str) -> None:
    if "FW212C" not in identity.upper():
        raise FW212CProtocolError(f"serial {serial!r} on {port!r} identified as {identity!r}, not an FW212C")
    if position_count != SLOT_COUNT:
        raise FW212CProtocolError(f"FW212C {serial!r} reports {position_count} positions; expected {SLOT_COUNT}")


def _open_verified(port: str, serial: str, timeout_s: float) -> tuple[FW212CProtocol, str]:
    """Open ``port`` and return the protocol and identity once it proves to be an FW212C."""
    protocol = FW212CProtocol(port, timeout_s)
    try:
        identity = protocol.identity
        _validate_device(identity, protocol.position_count, serial, port)
    except Exception:
        protocol.close()
        raise
    return protocol, identity


class FW212C(DiscreteAxis):
    """Thorlabs FW212C twelve-position motorized filter wheel."""

    def __init__(
        self,
        uid: str,
        serial: str,
        slots: dict[int | str, str | None],
        timeout_s: float = DEFAULT_TIMEOUT_S,
    ) -> None:
        # Reject before discovery, whose probe would otherwise report it as a probe failure.
        _check_timeout("timeout_s", timeout_s)
        super().__init__(uid=uid, slots=slots, slot_count=SLOT_COUNT)

        port = resolve_serial_port(serial, timeout_s)
        self._protocol, self._identity = _open_verified(port, serial, timeout_s)
        self._serial = serial
        self._movement_lock = threading.Lock()
        self._movement_done = threading.Event()
        self._movement_done.set()
        self._target: int | None = None
        self.log.info("Initialized FW212C %s on %s: %s", serial, port, self._identity)

    @property
    @describe(label="Serial Number")
    def serial(self) -> str:
        return self._serial

    @property
    @describe(label="Identity")
    def identity(self) -> str:
        return self._identity

    @property
    def state(self) -> DiscreteAxisState:  # pyrefly: ignore [bad-override]
        # Voxel's @describe-wrapped abstract property is modeled as a method by Pyrefly.
        # A move holds the serial transaction until the wheel settles, so report an unresolved
        # position while moving instead of blocking on a position query.
        if not self._movement_done.is_set():
            return DiscreteAxisState(position=None, target=self._target, is_moving=True)
        native = self._protocol.native_position
        if not 1 <= native <= SLOT_COUNT:
            raise FW212CProtocolError(f"FW212C returned position outside 1..{SLOT_COUNT}: {native}")
        return DiscreteAxisState(position=native - 1, target=self._target, is_moving=False)

    def move(self, slot: int, *, wait: bool = False, timeout: float | None = None) -> None:
        """Move to a zero-based Voxel slot (FW212C position ``slot + 1``).

        The FW212C replies only after the wheel settles, so every move is synchronous;
        ``timeout`` replaces the protocol timeout when ``wait`` is true.
        """
        if not 0 <= slot < SLOT_COUNT:
            raise ValueError(f"FW212C slot must be in range 0..{SLOT_COUNT - 1}, got {slot}")
        with self._movement_lock:
            self._movement_done.clear()
            self._target = slot
            try:
                self._protocol.move_native(slot + 1, timeout if wait else None)
            finally:
                self._movement_done.set()

    def home(self, *, wait: bool = False, timeout: float | None = None) -> None:
        del wait, timeout
        raise NotImplementedError("the documented FW212C protocol does not provide a home command")

    def halt(self) -> None:
        raise NotImplementedError("the documented FW212C protocol does not provide a halt command")

    def await_movement(self, timeout: float | None = None) -> None:
        if not self._movement_done.wait(timeout):
            raise TimeoutError(f"FW212C {self._serial!r} did not finish moving within {timeout}s")

    def close(self) -> None:
        """Release the connection after any bounded in-flight transaction ends."""
        with self._movement_lock:
            self._protocol.close()
