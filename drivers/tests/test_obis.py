from collections import deque
from collections.abc import Iterator
from contextlib import contextmanager

import pytest
from serial.tools.list_ports_common import ListPortInfo
from vxl_drivers.lasers import obis

from vxl.devices.laser.source import ModulationMode
from vxl.devices.laser.source_only import SourceOnlyLaser

READY = b"00001008\r\n"
WARMING_UP = b"00001108\r\n"


class FakeTransport:
    def __init__(self, *, handshake: bool = True) -> None:
        self.handshake = handshake
        self.replies = {
            "SYST:INF:MOD?": b"OBIS LS 561-200\r\n",
            "SYST:INF:SNUM?": b"123456\r\n",
            "SYST:INF:FVER?": b"V1.203\r\n",
            "SYST:INF:WAV?": b"561\r\n",
            "SOUR:POW:NOM?": b"0.200\r\n",
            "SOUR:AM:STAT?": b"ON\r\n",
            "SOUR:AM:SOUR?": b"MIXED\r\n",
            "SOUR:POW:LEV:IMM:AMPL?": b"0.010\r\n",
            "SOUR:POW:LEV?": b"0.0095\r\n",
            "SOUR:TEMP:BAS?": b"25.0C\r\n",
            "SYST:FAUL?": b"0\r\n",
            "SYST:STAT?": READY,
        }
        # Status polls that still report warm-up; a warming module ignores SOUR:AM:STAT ON.
        self.warm_up_polls = 0
        self.pending: deque[bytes] = deque()
        self.writes: list[str] = []
        self.close_calls = 0

    @contextmanager
    def transaction(self) -> Iterator["FakeTransport"]:
        yield self

    def write(self, payload: bytes) -> None:
        assert payload.endswith(b"\r\n")
        command = payload[:-2].decode("ascii")
        self.writes.append(command)
        if command == "SYST:STAT?" and self.warm_up_polls:
            self.warm_up_polls -= 1
            self.pending.append(WARMING_UP)
        elif command == "SOUR:AM:STAT ON" and self.warm_up_polls:
            pass  # Ignored, as a warming module does, but still acknowledged below.
        elif command in self.replies:
            self.pending.append(self.replies[command])
        else:
            setting, value = command.split(" ")
            mode_commands = {"SOUR:AM:INT", "SOUR:AM:EXT"}
            query = "SOUR:AM:SOUR?" if setting in mode_commands else f"{setting}?"
            self.replies[query] = f"{value}\r\n".encode("ascii")
        if self.handshake:
            self.pending.append(b"OK\r\n")

    def flush(self) -> None:
        pass

    def readline(self) -> bytes:
        return self.pending.popleft() if self.pending else b""

    def drain_input(self) -> int:
        count = sum(map(len, self.pending))
        self.pending.clear()
        return count

    def close(self) -> None:
        self.close_calls += 1


@pytest.mark.parametrize("handshake", [True, False])
def test_obis_protocol_handshake_modes_preserve_queries_and_commands(
    monkeypatch: pytest.MonkeyPatch, handshake: bool
) -> None:
    transport = FakeTransport(handshake=handshake)
    monkeypatch.setattr(obis, "SerialTransport", lambda *_args, **_kwargs: transport)
    protocol = obis.ObisProtocol("/dev/test")

    assert protocol.model == "OBIS LS 561-200"
    protocol.disable()
    assert not protocol.enabled
    assert protocol.serial_number == "123456"
    protocol.enable()
    assert protocol.enabled
    protocol.close()
    assert transport.close_calls == 1


@pytest.mark.parametrize("reply", [b"", b"\r\n", b"OFF", b"\xff\r\n", b"ERR-100\r\n", b"OK\r\n", b"UNKNOWN\r\n"])
def test_obis_protocol_malformed_state_reply_is_not_reported_as_disabled(
    monkeypatch: pytest.MonkeyPatch, reply: bytes
) -> None:
    transport = FakeTransport()
    transport.replies["SOUR:AM:STAT?"] = reply
    monkeypatch.setattr(obis, "SerialTransport", lambda *_args, **_kwargs: transport)

    with pytest.raises(obis.ObisProtocolError):
        _ = obis.ObisProtocol("/dev/test").enabled


@pytest.mark.parametrize(
    ("query", "attribute"),
    [
        ("SYST:INF:WAV?", "wavelength_nm"),
        ("SOUR:POW:NOM?", "specified_power_mw"),
        ("SOUR:POW:LEV:IMM:AMPL?", "power_setpoint_mw"),
        ("SOUR:POW:LEV?", "measured_power_mw"),
        ("SOUR:TEMP:BAS?", "temperature_c"),
    ],
)
def test_obis_protocol_non_numeric_reply_raises_protocol_error(
    monkeypatch: pytest.MonkeyPatch, query: str, attribute: str
) -> None:
    transport = FakeTransport()
    transport.replies[query] = b"garbage\r\n"
    monkeypatch.setattr(obis, "SerialTransport", lambda *_args, **_kwargs: transport)

    with pytest.raises(obis.ObisProtocolError, match="non-numeric"):
        getattr(obis.ObisProtocol("/dev/test"), attribute)


@pytest.mark.parametrize("reply", [b"unexpected\r\n", b"OK", b"\xff\r\n", b"ERR-100\r\n"])
def test_obis_protocol_malformed_command_acknowledgement_is_rejected(
    monkeypatch: pytest.MonkeyPatch, reply: bytes
) -> None:
    transport = FakeTransport()
    transport.replies["SOUR:AM:STAT OFF"] = reply
    monkeypatch.setattr(obis, "SerialTransport", lambda *_args, **_kwargs: transport)

    with pytest.raises(obis.ObisProtocolError):
        obis.ObisProtocol("/dev/test").disable()


def test_discover_obis_port_probes_only_obis_and_closes_every_connection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transports = {"/dev/a": FakeTransport(), "/dev/b": FakeTransport()}
    transports["/dev/a"].replies["SYST:INF:SNUM?"] = b"OTHER\r\n"
    monkeypatch.setattr(obis, "SerialTransport", lambda port, **_kwargs: transports[port])
    ports = [ListPortInfo(device) for device in ("/dev/a", "/dev/b", "/dev/unrelated")]
    ports[0].vid, ports[0].pid = obis.OBIS_USB_ID
    ports[1].vid, ports[1].pid = obis.OBIS_USB_ID
    ports[2].vid, ports[2].pid = 0x1234, 0x5678

    assert obis.discover_obis_port("123456", 1.0, ports) == "/dev/b"
    for transport in transports.values():
        assert transport.close_calls == 1
        assert all(command.endswith("?") for command in transport.writes)


def open_source(monkeypatch: pytest.MonkeyPatch, transport: FakeTransport, **kwargs: object) -> obis.ObisSource:
    monkeypatch.setattr(obis, "SerialTransport", lambda *_args, **_kwargs: transport)
    monkeypatch.setattr(obis.list_ports, "comports", lambda: pytest.fail("scanned ports"))
    init: dict = {"serial_number": "123456", "wavelength_nm": 561, "port": "/dev/test", **kwargs}
    return obis.ObisSource("laser_561_source", **init)


def test_obis_source_properties_and_shutdown_follow_voxel_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transport = FakeTransport()
    transport.replies["SOUR:AM:SOUR?"] = b"CWP\r\n"
    source = open_source(monkeypatch, transport)

    assert source.wavelength == 561
    assert source.specified_power_mw == 200.0
    assert not source.is_source_enabled
    assert source.modulation_mode == "software"
    assert source.source_power_setpoint_mw == pytest.approx(10.0)
    source.source_power_setpoint_mw = 25.0
    assert source.source_power_setpoint_mw == 25.0
    assert source.source_power_mw == 9.5
    assert source.temperature_c == 25.0
    assert source.fault == "0"
    assert not source.warming_up
    source.source_enable()
    assert source.is_source_enabled
    source.source_enable()
    assert transport.writes.count("SOUR:AM:STAT ON") == 1, "enable must be idempotent"
    source.close()
    source.close()
    assert transport.replies["SOUR:AM:STAT?"] == b"OFF\r\n"
    assert transport.close_calls == 1


@pytest.mark.parametrize(
    ("query", "reply", "error"),
    [
        ("SYST:INF:SNUM?", b"OTHER\r\n", obis.ObisDiscoveryError),
        ("SYST:INF:WAV?", b"594\r\n", obis.ObisDiscoveryError),
        ("SOUR:POW:NOM?", b"nan\r\n", obis.ObisProtocolError),
        ("SOUR:POW:NOM?", b"inf\r\n", obis.ObisProtocolError),
        ("SOUR:POW:NOM?", b"0\r\n", obis.ObisProtocolError),
    ],
)
def test_obis_source_invalid_identity_closes_without_control(
    monkeypatch: pytest.MonkeyPatch, query: str, reply: bytes, error: type[Exception]
) -> None:
    transport = FakeTransport()
    transport.replies[query] = reply

    with pytest.raises(error):
        open_source(monkeypatch, transport)

    assert transport.close_calls == 1
    assert all(command.endswith("?") for command in transport.writes)


def test_obis_source_failed_shutdown_closes_connection_and_reports_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transport = FakeTransport()
    source = open_source(monkeypatch, transport)
    source.source_enable()
    transport.replies["SOUR:AM:STAT OFF"] = b"OK\r\n"  # Acknowledged, but still emitting.

    with pytest.raises(obis.ObisProtocolError, match="enabled"):
        source.close()

    assert transport.close_calls == 1
    source.close()
    assert transport.close_calls == 1


@pytest.mark.parametrize(("stored", "rewritten"), [("CWP", False), ("DIGITAL", False), ("MIXED", True)])
def test_obis_source_keeps_a_supported_stored_mode_and_replaces_others_with_cwp(
    monkeypatch: pytest.MonkeyPatch, stored: str, rewritten: bool
) -> None:
    transport = FakeTransport()
    transport.replies["SOUR:AM:SOUR?"] = f"{stored}\r\n".encode()

    source = open_source(monkeypatch, transport)

    assert ("SOUR:AM:INT CWP" in transport.writes) is rewritten
    assert source.operating_mode == ("CWP" if rewritten else stored)
    if rewritten:
        assert transport.writes.index("SOUR:AM:STAT OFF") < transport.writes.index("SOUR:AM:INT CWP"), (
            "emission must be off before the mode changes"
        )


def test_obis_source_digital_mode_hands_emission_to_the_gate_only_after_readback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transport = FakeTransport()
    transport.replies["SOUR:AM:SOUR?"] = b"CWP\r\n"
    source = open_source(monkeypatch, transport)
    transport.writes.clear()

    source.set_modulation_mode(ModulationMode.DIGITAL)

    assert source.modulation_mode == "digital"
    assert source.operating_mode == "DIGITAL"
    assert transport.writes.index("SOUR:AM:STAT OFF") < transport.writes.index("SOUR:AM:EXT DIGITAL"), (
        "emission must be off before the mode changes"
    )
    assert set(source.modulation_mode.options) == {"software", "digital"}

    # A module that acknowledges but stays in CW would emit whenever enabled, ignoring the gate.
    source.set_modulation_mode(ModulationMode.SOFTWARE)
    transport.replies["SOUR:AM:EXT DIGITAL"] = b"OK\r\n"
    with pytest.raises(obis.ObisProtocolError, match="remained in 'CWP'; expected 'DIGITAL'"):
        source.set_modulation_mode(ModulationMode.DIGITAL)
    assert "SOUR:AM:STAT ON" not in transport.writes


def test_obis_source_enable_during_warm_up_fails_instead_of_reporting_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transport = FakeTransport()
    source = open_source(monkeypatch, transport)
    transport.warm_up_polls = 100

    with pytest.raises(obis.ObisProtocolError, match=r"ignored SOUR:AM:STAT ON while warming up"):
        source.source_enable()

    assert not source.is_source_enabled
    assert transport.writes[-1] == "SOUR:AM:STAT?"
    assert "SOUR:AM:STAT OFF" in transport.writes[-4:], "a failed enable must inhibit again"


def test_obis_source_enable_with_wait_lets_warm_up_finish_first(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transport = FakeTransport()
    source = open_source(monkeypatch, transport)
    monkeypatch.setattr(obis.time, "sleep", lambda _s: None)
    transport.warm_up_polls = 3

    source.source_enable(wait=True)

    assert source.is_source_enabled
    assert transport.writes.count("SYST:STAT?") == 4
    assert transport.writes.count("SOUR:AM:STAT ON") == 1


def test_obis_source_enable_with_wait_gives_up_after_the_startup_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transport = FakeTransport()
    source = open_source(monkeypatch, transport, startup_timeout_s=5.0)
    clock = iter(range(0, 100, 2))
    monkeypatch.setattr(obis.time, "monotonic", lambda: next(clock))
    monkeypatch.setattr(obis.time, "sleep", lambda _s: None)
    transport.warm_up_polls = 100

    with pytest.raises(obis.ObisProtocolError, match="still warming up after 5 s"):
        source.source_enable(wait=True)

    assert "SOUR:AM:STAT ON" not in transport.writes


def test_obis_source_setpoint_must_read_back(monkeypatch: pytest.MonkeyPatch) -> None:
    transport = FakeTransport()
    source = open_source(monkeypatch, transport)
    transport.replies["SOUR:POW:LEV:IMM:AMPL 0.030000"] = b"OK\r\n"  # Acknowledged, not applied.

    with pytest.raises(obis.ObisProtocolError, match=r"setpoint reads 10\.0 mW"):
        source.source_power_setpoint_mw = 30.0
    with pytest.raises(ValueError, match="OBIS power"):
        source.source_power_setpoint_mw = 200.1

    assert source.source_power_setpoint_mw == pytest.approx(10.0)


def test_source_only_laser_drives_the_obis_through_voxels_laser_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transport = FakeTransport()
    transport.replies["SOUR:AM:SOUR?"] = b"CWP\r\n"
    source = open_source(monkeypatch, transport)
    monkeypatch.setattr(obis.time, "sleep", lambda _s: None)
    laser = SourceOnlyLaser("laser_561", source)
    laser.source_power_setpoint_mw = 5.0
    transport.warm_up_polls = 2

    laser.set_modulation_mode("digital")
    laser.source_enable(wait=True)
    assert not source.is_source_enabled, "source permission alone must not emit"
    laser.output_enable(wait=True)

    assert source.is_source_enabled
    assert source.operating_mode == "DIGITAL"
    assert transport.replies["SOUR:POW:LEV:IMM:AMPL?"] == b"0.005000\r\n"
    laser.output_level_percent = 20.0
    assert source.source_power_setpoint_mw == pytest.approx(1.0)
    laser.output_disable()
    assert not source.is_source_enabled
    laser.close()
    source.close()
    assert transport.close_calls == 1


def test_obis_source_invalid_startup_timeout_rejected_before_open(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(obis, "SerialTransport", lambda *_args, **_kwargs: pytest.fail("opened serial port"))
    with pytest.raises(ValueError, match="startup_timeout_s"):
        obis.ObisSource(
            "laser",
            serial_number="123456",
            wavelength_nm=561,
            port="/dev/test",
            startup_timeout_s=0,
        )


@pytest.mark.parametrize("timeout_s", [0.0, float("nan")])
def test_invalid_serial_timeout_rejected_before_open(monkeypatch: pytest.MonkeyPatch, timeout_s: float) -> None:
    monkeypatch.setattr(obis, "SerialTransport", lambda *_args, **_kwargs: pytest.fail("opened serial port"))
    with pytest.raises(ValueError, match="timeout_s"):
        obis.ObisProtocol("/dev/test", timeout_s=timeout_s)
