import asyncio
from collections.abc import Iterator
from contextlib import contextmanager

import pytest
from vxl_drivers.lasers import luxx

from rigup import DeviceConfig, Rig, RigConfig
from vxl.devices.laser.source import ModulationMode
from vxl.devices.laser.source_only import SourceOnlyLaser

# Commands absent from a reply table are acknowledged, and these update what the laser reports.
EMISSION_STATUS = {"LOn": b"!GAS02C2\r", "LOf": b"!GAS02C0\r"}
SETTING_QUERIES = {"SLP": "GLP", "SOM": "GOM"}


class FakeSerial:
    def __init__(self, replies: dict[bytes, bytes | list[bytes]]) -> None:
        self.replies = replies
        self.writes: list[bytes] = []
        self.pending: list[bytes] = []

    def write(self, payload: bytes) -> None:
        self.writes.append(payload)
        reply = self.replies.get(payload)
        if reply is None:
            reply = self._acknowledge(payload[1:-1].decode("ascii"))
        self.pending.extend(reply if isinstance(reply, list) else [reply])

    def _acknowledge(self, command: str) -> bytes:
        name, value = command[:3], command[3:]
        if name in EMISSION_STATUS:
            self.replies[b"?GAS\r"] = EMISSION_STATUS[name]
        elif name in SETTING_QUERIES:
            query = SETTING_QUERIES[name]
            self.replies[f"?{query}\r".encode()] = f"!{query}{value}\r".encode()
        else:
            raise AssertionError(f"unexpected LuxX command {command!r}")
        return f"!{name}>\r".encode()

    def flush(self) -> None:
        pass

    def read_until(self, expected: bytes) -> bytes:
        assert expected == luxx.TERMINATOR
        return self.pending.pop(0)


class FakeTransport:
    def __init__(self, replies: dict[bytes, bytes | list[bytes]]) -> None:
        self.serial = FakeSerial(replies)
        self.close_calls = 0

    @contextmanager
    def transaction(self) -> Iterator[FakeSerial]:
        yield self.serial

    def drain_input(self) -> int:
        return 0

    def close(self) -> None:
        self.close_calls += 1


def luxx_488_replies() -> dict[bytes, bytes | list[bytes]]:
    """Replies of an emitting 200 mW, 488 nm LuxX in APC."""
    return {
        b"?GFw\r": b"!GFwLuxX\xa74\xa72.30\r",
        b"?GSN\r": b"!GSN2606-53\r",
        b"?GSI\r": b"!GSI488\xa7200\r",
        b"?GAS\r": b"!GAS02C2\r",
        b"?GOM\r": b"!GOMF918\r",
        b"?GLP\r": b"!GLP0CD\r",
        b"?MDP\r": b"!MDP9.64\r",
        b"?MTD\r": b"!MTD25.0\r",
        b"?MTA\r": b"!MTA23.8\r",
    }


def open_fake_luxx(
    monkeypatch: pytest.MonkeyPatch,
    transport: FakeTransport,
    *,
    wavelength_nm: int = 488,
) -> luxx.LuxxSource:
    install_fake_transport(monkeypatch, transport)
    monkeypatch.setattr(luxx, "discover_serial_port", lambda *_args, **_kwargs: "/dev/test")
    return luxx.LuxxSource(f"laser_{wavelength_nm}_source", usb_serial="TARGET", wavelength_nm=wavelength_nm)


def install_fake_transport(
    monkeypatch: pytest.MonkeyPatch,
    transport: FakeTransport,
) -> None:
    monkeypatch.setattr(luxx, "SerialTransport", lambda *_args, **_kwargs: transport)


@pytest.mark.parametrize("power_mw", [-0.1, 200.1, float("nan")])
def test_luxx_protocol_set_power_setpoint_outside_maximum_rejected_before_write(
    monkeypatch: pytest.MonkeyPatch, power_mw: float
) -> None:
    transport = FakeTransport({})
    install_fake_transport(monkeypatch, transport)

    with pytest.raises(ValueError, match="LuxX power"):
        luxx.LuxxProtocol("/dev/test").set_power_setpoint(power_mw, 200.0)

    assert transport.serial.writes == []


@pytest.mark.parametrize(
    ("command", "raw", "expected"),
    [
        ("GFw", b"!GFwLuxX\xa74\xa72.30\r", "LuxX§4§2.30"),
        ("GSI", b"!GSI488\xa7200\r", "488§200"),
        ("LOf", b"!LOf>\r", ">"),
        ("GSI", b"\x00!GSI488\xa7200\r", "488§200"),
    ],
)
def test_parse_reply_returns_command_payload(command: str, raw: bytes, expected: str) -> None:
    assert luxx.parse_reply(command, raw) == expected


@pytest.mark.parametrize("raw", [b"", b"!GFwLuxX", b"!UK\r", b"!GFwxfailure\r"])
def test_parse_reply_missing_unknown_or_error_reply_rejected(raw: bytes) -> None:
    with pytest.raises(luxx.LuxxProtocolError):
        luxx.parse_reply("GFw", raw)


def test_protocol_queries_and_commands_use_luxx_frames(monkeypatch: pytest.MonkeyPatch) -> None:
    replies: dict[bytes, bytes | list[bytes]] = {
        b"?GFw\r": b"!GFwLuxX\xa74\xa72.30\r",
        b"?GSN\r": b"!GSN2606-53\r",
        b"?GSI\r": b"!GSI488\xa7200\r",
        b"?GWH\r": b"!GWH65\r",
        b"?GAS\r": b"!GAS02C2\r",
        b"?GFB\r": b"!GFB0000\r",
        b"?GLF\r": b"!GLF0000\r",
        b"?GLP\r": b"!GLP0CD\r",
        b"?MDP\r": b"!MDP9.64\r",
        b"?MTD\r": b"!MTD25.0\r",
        b"?MTA\r": b"!MTA23.8\r",
        b"?SLP800\r": b"!SLP>\r",
        b"?LOn\r": b"!LOn>\r",
        b"?LOf\r": b"!LOf>\r",
    }
    transport = FakeTransport(replies)
    install_fake_transport(monkeypatch, transport)
    protocol = luxx.LuxxProtocol("/dev/test", timeout_s=1.0)

    assert protocol.firmware == "LuxX§4§2.30"
    assert protocol.serial_number == "2606-53"
    assert protocol.specification == (488, 200.0)
    assert protocol.working_hours == 65.0
    assert protocol.enabled
    assert protocol.fault_code == 0
    assert protocol.last_fault_code == 0
    assert protocol.power_setpoint_mw(200.0) == pytest.approx(10.0122, abs=0.001)
    assert protocol.measured_power_mw == 9.64
    assert protocol.diode_temperature_c == 25.0
    assert protocol.ambient_temperature_c == 23.8
    protocol.set_power_setpoint(200.0 * 0x800 / 0xFFF, 200.0)
    protocol.enable()
    protocol.disable()
    protocol.close()

    assert transport.serial.writes == [
        b"?GFw\r",
        b"?GSN\r",
        b"?GSI\r",
        b"?GWH\r",
        b"?GAS\r",
        b"?GFB\r",
        b"?GLF\r",
        b"?GLP\r",
        b"?MDP\r",
        b"?MTD\r",
        b"?MTA\r",
        b"?SLP800\r",
        b"?LOn\r",
        b"?GAS\r",
        b"?LOf\r",
    ]
    assert transport.close_calls == 1


@pytest.mark.parametrize("status", [b"$GAS02C0\r", b"!GAS02C0\r", b"$MDP0.00\r", b"$GOMF818\r"])
def test_query_skips_delayed_status_after_laser_off(monkeypatch: pytest.MonkeyPatch, status: bytes) -> None:
    transport = FakeTransport(
        {
            b"?LOf\r": b"!LOf>\r",
            b"?GOM\r": [status, b"!GOMF918\r"],
        }
    )
    install_fake_transport(monkeypatch, transport)
    protocol = luxx.LuxxProtocol("/dev/test")

    protocol.disable()
    assert protocol.control_mode == "APC"
    assert transport.serial.writes == [b"?LOf\r", b"?GOM\r"]


def test_mode_change_waits_for_reply_after_unsolicited_power_and_mode_frames(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transport = FakeTransport({b"?SOMF818\r": [b"$MDP0.00\r", b"$GOMF818\r", b"!SOM>\r"]})
    install_fake_transport(monkeypatch, transport)

    assert luxx.LuxxProtocol("/dev/test").query("SOMF818") == ">"


@pytest.mark.parametrize("fragment", [b"DP0.59\r", b"GAS02C0\r", b"\r"])
def test_query_skips_status_tail_split_by_input_drain(monkeypatch: pytest.MonkeyPatch, fragment: bytes) -> None:
    transport = FakeTransport({b"?GAS\r": [fragment, b"!GAS0002\r"]})
    install_fake_transport(monkeypatch, transport)

    assert luxx.LuxxProtocol("/dev/test").enabled is True


def test_query_rejects_unrelated_reply(monkeypatch: pytest.MonkeyPatch) -> None:
    transport = FakeTransport({b"?GOM\r": b"!GSI488\xa7200\r"})
    install_fake_transport(monkeypatch, transport)

    with pytest.raises(luxx.LuxxProtocolError, match="unexpected LuxX reply"):
        luxx.LuxxProtocol("/dev/test").query("GOM")


@pytest.mark.parametrize(
    ("current", "requested", "written"),
    [
        (b"F918", "ACC", b"?SOMF818\r"),
        (b"0000", "APC", b"?SOM0100\r"),
        (b"FFFF", "ACC", b"?SOMFEFF\r"),
        (b"F818", "ACC", None),
        (b"F918", "APC", None),
    ],
)
def test_set_control_mode_rewrites_only_the_apc_bit_and_only_when_it_differs(
    monkeypatch: pytest.MonkeyPatch, current: bytes, requested: str, written: bytes | None
) -> None:
    replies: dict[bytes, bytes | list[bytes]] = {b"?GOM\r": b"!GOM" + current + b"\r"}
    if written is not None:
        replies[written] = b"!SOM>\r"
    transport = FakeTransport(replies)
    install_fake_transport(monkeypatch, transport)

    luxx.LuxxProtocol("/dev/test").set_control_mode(requested)

    assert transport.serial.writes == [b"?GOM\r", *([written] if written else [])]


def test_set_control_mode_operating_mode_wider_than_16_bits_is_not_written_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transport = FakeTransport({b"?GOM\r": b"!GOM1F918\r"})
    install_fake_transport(monkeypatch, transport)

    with pytest.raises(luxx.LuxxProtocolError, match="16 bits"):
        luxx.LuxxProtocol("/dev/test").set_control_mode("ACC")

    assert transport.serial.writes == [b"?GOM\r"]


def test_source_validates_identity_then_disables_emission_on_open(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transport = FakeTransport(luxx_488_replies())

    source = open_fake_luxx(monkeypatch, transport)

    assert source.wavelength == 488
    assert source.serial_number == "2606-53"
    assert source.specified_power_mw == 200.0
    assert source.source_power_setpoint_mw == pytest.approx(10.0122, abs=0.001)
    assert transport.serial.writes == [b"?GFw\r", b"?GSN\r", b"?GSI\r", b"?LOf\r", b"?GLP\r"], (
        "opening must leave the stored control mode alone"
    )
    assert not source.is_source_enabled
    source.close()
    assert transport.serial.writes.count(b"?LOf\r") == 2
    assert transport.close_calls == 1


def test_source_wrong_wavelength_closes_without_sending_control_command(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transport = FakeTransport(luxx_488_replies())

    with pytest.raises(luxx.LuxxDiscoveryError, match="expected 660 nm"):
        open_fake_luxx(monkeypatch, transport, wavelength_nm=660)

    assert transport.serial.writes == [b"?GFw\r", b"?GSN\r", b"?GSI\r"]
    assert transport.close_calls == 1


@pytest.mark.parametrize(
    ("mode", "control_mode", "written"),
    [(ModulationMode.DIGITAL, "ACC", b"?SOMF818\r"), (ModulationMode.SOFTWARE, "APC", None)],
)
def test_source_modulation_modes_select_the_control_mode_with_emission_off(
    monkeypatch: pytest.MonkeyPatch, mode: ModulationMode, control_mode: str, written: bytes | None
) -> None:
    transport = FakeTransport(luxx_488_replies())
    source = open_fake_luxx(monkeypatch, transport)
    transport.serial.writes.clear()

    source.set_modulation_mode(mode)

    assert source.modulation_mode == mode.value
    assert source.control_mode == control_mode
    writes = transport.serial.writes
    if written is not None:
        assert writes.index(b"?LOf\r") < writes.index(written), "mode must change with emission off"
    assert set(source.modulation_mode.options) == {"software", "digital"}


def test_source_mode_that_does_not_read_back_fails_with_emission_off(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A LuxX left in APC ignores the LightHUB gate, so the DAQ could not switch it off.
    transport = FakeTransport({**luxx_488_replies(), b"?SOMF818\r": b"!SOM>\r"})
    source = open_fake_luxx(monkeypatch, transport)

    with pytest.raises(luxx.LuxxProtocolError, match="remained in APC; expected ACC"):
        source.set_modulation_mode("digital")
    assert b"?LOn\r" not in transport.serial.writes


def test_source_enable_reads_back_and_disables_when_emission_stays_off(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transport = FakeTransport(luxx_488_replies())
    source = open_fake_luxx(monkeypatch, transport)

    source.source_enable(wait=True)
    assert source.is_source_enabled
    source.source_enable()
    assert transport.serial.writes.count(b"?LOn\r") == 1, "enable must be idempotent"
    source.source_disable()

    transport.serial.replies[b"?LOn\r"] = b"!LOn>\r"  # Acknowledged without emitting.
    with pytest.raises(luxx.LuxxProtocolError, match="remained off"):
        source.source_enable()
    assert transport.serial.writes[-1] == b"?LOf\r"


def test_source_setpoint_keeps_the_request_and_checks_the_12_bit_readback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transport = FakeTransport(luxx_488_replies())
    source = open_fake_luxx(monkeypatch, transport)

    source.source_power_setpoint_mw = 25.0

    assert source.source_power_setpoint_mw == 25.0
    assert [w for w in transport.serial.writes if w.startswith(b"?SLP")] == [b"?SLP200\r"]
    assert source.source_power_mw == 9.64
    assert source.temperature_c == 25.0
    with pytest.raises(ValueError, match="LuxX power"):
        source.source_power_setpoint_mw = 200.1

    transport.serial.replies[b"?SLP100\r"] = b"!SLP>\r"  # Acknowledged but not applied.
    with pytest.raises(luxx.LuxxProtocolError, match="setpoint reads"):
        source.source_power_setpoint_mw = 12.5
    assert source.source_power_setpoint_mw == 25.0


def test_source_only_laser_drives_the_luxx_through_voxels_laser_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transport = FakeTransport(luxx_488_replies())
    source = open_fake_luxx(monkeypatch, transport)
    laser = SourceOnlyLaser("laser_488", source)
    laser.source_power_setpoint_mw = 20.0
    laser.output_level_percent = 50.0

    laser.set_modulation_mode("digital")
    laser.source_enable(wait=True)
    assert not source.is_source_enabled, "source permission alone must not emit"
    laser.output_enable(wait=True)

    assert source.is_source_enabled
    assert source.control_mode == "ACC"
    assert source.source_power_setpoint_mw == pytest.approx(10.0)
    laser.output_disable()
    assert not source.is_source_enabled
    laser.close()
    source.close()
    assert transport.close_calls == 1


def test_source_and_adapter_build_through_rigup_and_close_emission(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transport = FakeTransport(luxx_488_replies())
    install_fake_transport(monkeypatch, transport)
    monkeypatch.setattr(luxx, "discover_serial_port", lambda *_args, **_kwargs: "/dev/test")
    config = RigConfig(
        devices={
            "laser_488_source": DeviceConfig(
                target="vxl_drivers.lasers.luxx.LuxxSource",
                init={"usb_serial": "TARGET", "wavelength_nm": 488},
            ),
            "laser_488": DeviceConfig(
                target="vxl.devices.laser.source_only.SourceOnlyLaser",
                init={"source": "laser_488_source"},
            ),
        }
    )

    async def run() -> None:
        rig = Rig(config, name="luxx-test")
        await rig.open()
        try:
            assert not rig.build_errors
            laser = rig.devices["laser_488"]
            assert await laser.props.get_value("wavelength") == 488
            assert await laser.props.get_value("is_output_enabled") is False
            assert await rig.devices["laser_488_source"].props.get_value("is_source_enabled") is False
        finally:
            await rig.close()

    asyncio.run(run())
    assert transport.serial.writes[-1] == b"?LOf\r"
    assert transport.close_calls == 1


@pytest.mark.parametrize("timeout_s", [0.0, float("nan")])
def test_invalid_serial_timeout_rejected_before_open(monkeypatch: pytest.MonkeyPatch, timeout_s: float) -> None:
    monkeypatch.setattr(luxx, "SerialTransport", lambda *_args, **_kwargs: pytest.fail("opened serial port"))
    with pytest.raises(ValueError, match="timeout_s"):
        luxx.LuxxProtocol("/dev/test", timeout_s=timeout_s)
