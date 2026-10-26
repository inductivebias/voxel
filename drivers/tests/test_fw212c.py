import asyncio
from collections.abc import Callable, Iterator
from contextlib import contextmanager

import pytest
from serial.tools.list_ports_common import ListPortInfo
from vxl_drivers.axes import fw212c

from rigup import DeviceConfig, Rig, RigConfig
from vxl.devices.axes.discrete.base import DiscreteAxisState


class FakeSerial:
    def __init__(self, replies: dict[bytes, bytes]) -> None:
        self.replies = replies
        self.timeout: float | None = 5.0
        self.writes: list[bytes] = []
        self.flush_calls = 0

    def write(self, payload: bytes) -> None:
        self.writes.append(payload)

    def flush(self) -> None:
        self.flush_calls += 1

    def read_until(self, expected: bytes) -> bytes:
        assert expected == fw212c.PROMPT
        return self.replies[self.writes[-1]]


class FakeTransport:
    def __init__(self, replies: dict[bytes, bytes]) -> None:
        self.serial = FakeSerial(replies)
        self.close_calls = 0
        self.drain_calls = 0

    @contextmanager
    def transaction(self) -> Iterator[FakeSerial]:
        yield self.serial

    def drain_input(self) -> int:
        self.drain_calls += 1
        return 0

    def close(self) -> None:
        self.close_calls += 1


class FakeProtocol:
    def __init__(
        self,
        identity: str = "THORLABS FW102C/FW212C Filter Wheel version 1.07",
        position_count: int = 12,
        native_position: int = 1,
    ) -> None:
        self.identity = identity
        self.position_count = position_count
        self.native_position = native_position
        self.moves: list[tuple[int, float | None]] = []
        self.close_calls = 0

    def move_native(self, position: int, timeout_s: float | None = None) -> None:
        self.moves.append((position, timeout_s))
        self.native_position = position

    def close(self) -> None:
        self.close_calls += 1


def make_port(
    device: str,
    *,
    serial: str | None,
    vid: int | None = 0x0403,
    pid: int | None = 0x6001,
) -> ListPortInfo:
    port = ListPortInfo(device)
    port.serial_number = serial
    port.vid = vid
    port.pid = pid
    port.product = "USB Serial Port"
    return port


@pytest.mark.parametrize(
    ("devices", "identity", "outcome"),
    [
        (["/dev/ttyUSB0"], FakeProtocol().identity, "/dev/ttyUSB0"),
        (["/dev/ttyUSB0"], "Unrelated FTDI device", "probe failures"),
        (["/dev/ttyUSB0", "/dev/ttyUSB1"], FakeProtocol().identity, "multiple verified FW212C"),
    ],
)
def test_resolve_serial_port_probes_fw212c_interfaces_missing_usb_serial(
    monkeypatch: pytest.MonkeyPatch, devices: list[str], identity: str, outcome: str
) -> None:
    opened: list[tuple[str, float, FakeProtocol]] = []

    def open_protocol(port: str, timeout: float) -> FakeProtocol:
        opened.append((port, timeout, FakeProtocol(identity=identity)))
        return opened[-1][2]

    monkeypatch.setattr(fw212c, "FW212CProtocol", open_protocol)
    ports = [make_port(device, serial=None) for device in devices]

    if outcome.startswith("/dev/"):
        assert fw212c.resolve_serial_port("TARGET", timeout_s=5.0, ports=ports) == outcome
    else:
        with pytest.raises(fw212c.FW212CDiscoveryError, match=outcome):
            fw212c.resolve_serial_port("TARGET", timeout_s=5.0, ports=ports)
    probe_timeout = fw212c.DISCOVERY_PROBE_TIMEOUT_S
    assert [(port, timeout) for port, timeout, _ in opened] == [(d, probe_timeout) for d in devices]
    assert all(protocol.close_calls == 1 for *_, protocol in opened)


def test_resolve_serial_port_never_probes_non_fw212c_usb_interfaces(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ports = [
        make_port("/dev/ttyS0", serial=None, vid=None, pid=None),
        make_port("/dev/ttyACM0", serial="123456", vid=0x2939, pid=0x49C3),
    ]

    def fail_open(_port: str, _timeout: float) -> FakeProtocol:
        raise AssertionError("an unrelated serial interface was probed")

    monkeypatch.setattr(fw212c, "FW212CProtocol", fail_open)

    with pytest.raises(fw212c.FW212CDiscoveryError, match="no serial port found"):
        fw212c.resolve_serial_port("TARGET", timeout_s=5.0, ports=ports)


@pytest.mark.parametrize(
    ("command", "raw", "expected"),
    [
        ("pos?", b"1\r>", ("1",)),
        ("pos?", b"pos?\r1\r\n>", ("1",)),
        ("pcount?", b" pcount?\r12\r>", ("12",)),
        ("pos=2", b"pos=2\r>", ()),
    ],
)
def test_parse_reply_prompt_and_echo_variants(
    command: str,
    raw: bytes,
    expected: tuple[str, ...],
) -> None:
    assert fw212c.parse_reply(command, raw) == expected


@pytest.mark.parametrize(
    ("raw", "message"),
    [(b"\xff>", "non-ASCII"), (b"Command error CMD_NOT_DEFINED\r>", "CMD_NOT_DEFINED")],
)
def test_parse_reply_non_ascii_or_device_error_rejected(raw: bytes, message: str) -> None:
    with pytest.raises(fw212c.FW212CProtocolError, match=message):
        fw212c.parse_reply("bad", raw)


def test_protocol_queries_and_commands_use_documented_frames(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    replies = {
        b"*idn?\r": b"*idn?\rTHORLABS FW102C/FW212C Filter Wheel version 1.07\r>",
        b"pcount?\r": b" pcount?\r12\r>",
        b"pos?\r": b" pos?\r7\r>",
        b"pos=12\r": b">",
    }
    transport = FakeTransport(replies)
    install_fake_transport(monkeypatch, transport)
    protocol = fw212c.FW212CProtocol("/dev/test", timeout_s=5.0)

    assert protocol.identity == "THORLABS FW102C/FW212C Filter Wheel version 1.07"
    assert protocol.position_count == 12
    assert protocol.native_position == 7
    protocol.move_native(12, timeout_s=1.5)

    assert transport.serial.writes == list(replies)
    assert transport.serial.flush_calls == 4
    assert transport.drain_calls == 4
    assert transport.serial.timeout == 5.0, "a move's temporary timeout must be restored"
    protocol.close()
    protocol.close()
    assert transport.close_calls == 1
    with pytest.raises(fw212c.FW212CCommunicationError, match="closed"):
        _ = protocol.native_position


@pytest.mark.parametrize(
    ("reply", "error", "message"),
    [
        (b"unknown\r>", fw212c.FW212CProtocolError, "non-integer"),
        (b" pos?\r", TimeoutError, "ready prompt"),  # cut short by the serial timeout
    ],
)
def test_protocol_bad_position_reply_rejected(
    monkeypatch: pytest.MonkeyPatch, reply: bytes, error: type[Exception], message: str
) -> None:
    install_fake_transport(monkeypatch, FakeTransport({b"pos?\r": reply}))

    with pytest.raises(error, match=message):
        _ = fw212c.FW212CProtocol("/dev/test").native_position


def install_fake_transport(
    monkeypatch: pytest.MonkeyPatch,
    transport: FakeTransport,
) -> None:
    def make_transport(_port: str, **_kwargs: object) -> FakeTransport:
        return transport

    monkeypatch.setattr(fw212c, "SerialTransport", make_transport)


def install_fake_protocol(
    monkeypatch: pytest.MonkeyPatch,
    protocol: FakeProtocol,
) -> None:
    def open_protocol(_port: str, _timeout: float) -> FakeProtocol:
        return protocol

    monkeypatch.setattr(fw212c.list_ports, "comports", lambda: [make_port("/dev/test", serial="TARGET")])
    monkeypatch.setattr(fw212c, "FW212CProtocol", open_protocol)


@pytest.mark.parametrize(
    ("protocol", "message"),
    [
        (FakeProtocol(identity="Unrelated serial device"), "not an FW212C"),
        (FakeProtocol(position_count=6), "reports 6 positions"),
    ],
)
def test_driver_constructor_wrong_device_rejected_and_closed(
    monkeypatch: pytest.MonkeyPatch, protocol: FakeProtocol, message: str
) -> None:
    install_fake_protocol(monkeypatch, protocol)

    with pytest.raises(fw212c.FW212CProtocolError, match=message):
        fw212c.FW212C("wheel", "TARGET", {0: "open"})

    assert protocol.close_calls == 1


@pytest.mark.parametrize(
    ("slots", "label", "slot"),
    [
        ({0: "open", 11: "far-red"}, "far-red", 11),
        # YAML maps load with string keys.
        ({"0": "open", "1": "bench-01"}, "bench-01", 1),
    ],
)
def test_driver_select_label_moves_to_zero_based_slot(
    monkeypatch: pytest.MonkeyPatch, slots: dict[int | str, str | None], label: str, slot: int
) -> None:
    protocol = FakeProtocol(native_position=1)
    install_fake_protocol(monkeypatch, protocol)
    wheel = fw212c.FW212C("wheel", "TARGET", slots)

    assert wheel.state == DiscreteAxisState(position=0, target=None, is_moving=False)
    wheel.select(label, wait=True, timeout=2.0)

    assert wheel.state == DiscreteAxisState(position=slot, target=slot, is_moving=False)
    assert protocol.moves == [(slot + 1, 2.0)]


@pytest.mark.parametrize(
    ("request_move", "error", "message"),
    [
        (lambda wheel: wheel.select("missing"), KeyError, "Label not found"),
        (lambda wheel: wheel.move(12), ValueError, "slot must be in range"),
        (lambda wheel: wheel.move(-1), ValueError, "slot must be in range"),
    ],
)
def test_driver_invalid_request_never_reaches_protocol(
    monkeypatch: pytest.MonkeyPatch,
    request_move: Callable[[fw212c.FW212C], None],
    error: type[Exception],
    message: str,
) -> None:
    protocol = FakeProtocol()
    install_fake_protocol(monkeypatch, protocol)
    wheel = fw212c.FW212C("wheel", "TARGET", {0: "open"})

    with pytest.raises(error, match=message):
        request_move(wheel)

    assert protocol.moves == []


def test_driver_builds_and_reads_position_through_rigup(monkeypatch: pytest.MonkeyPatch) -> None:
    protocol = FakeProtocol(native_position=3)
    install_fake_protocol(monkeypatch, protocol)
    config = RigConfig(
        devices={
            "wheel": DeviceConfig(
                target="vxl_drivers.axes.fw212c.FW212C",
                init={"serial": "TARGET", "slots": {0: "open", 2: "gfp"}},
            )
        }
    )

    async def run() -> None:
        rig = Rig(config, name="fw212c-test")
        await rig.open()
        try:
            assert not rig.build_errors
            state = DiscreteAxisState.model_validate(await rig.devices["wheel"].props.get_value("state"))
            assert state == DiscreteAxisState(position=2, target=None, is_moving=False)
        finally:
            await rig.close()

    asyncio.run(run())
    assert protocol.close_calls == 1


def test_state_reports_target_without_querying_position_while_moving(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    protocol = FakeProtocol(native_position=1)
    install_fake_protocol(monkeypatch, protocol)
    wheel = fw212c.FW212C("wheel", "TARGET", {0: "open", 4: "gfp"})
    during: list[DiscreteAxisState] = []
    move_native = protocol.move_native

    def observe_then_move(position: int, timeout_s: float | None = None) -> None:
        during.append(wheel.state)
        move_native(position, timeout_s)

    monkeypatch.setattr(protocol, "move_native", observe_then_move)
    wheel.move(4)

    assert during == [DiscreteAxisState(position=None, target=4, is_moving=True)]
    assert wheel.state == DiscreteAxisState(position=4, target=4, is_moving=False)


@pytest.mark.parametrize("native_position", [0, fw212c.SLOT_COUNT + 1])
def test_state_native_position_outside_wheel_rejected(monkeypatch: pytest.MonkeyPatch, native_position: int) -> None:
    install_fake_protocol(monkeypatch, FakeProtocol(native_position=native_position))
    wheel = fw212c.FW212C("wheel", "TARGET", {0: "open"})

    with pytest.raises(fw212c.FW212CProtocolError, match="returned position outside"):
        _ = wheel.state


@pytest.mark.parametrize("timeout_s", [0.0, float("nan")])
def test_invalid_timeout_rejected_before_open(monkeypatch: pytest.MonkeyPatch, timeout_s: float) -> None:
    monkeypatch.setattr(fw212c, "SerialTransport", lambda *_args, **_kwargs: pytest.fail("opened serial port"))
    with pytest.raises(ValueError, match="timeout_s"):
        fw212c.FW212CProtocol("/dev/test", timeout_s=timeout_s)
