import threading
from unittest.mock import Mock, patch

import pytest
from vxl_drivers.serial import SerialTransport


def test_transaction_holds_exclusive_access_across_write_and_read() -> None:
    port = Mock()
    with patch("vxl_drivers.serial.serial.Serial", return_value=port):
        transport = SerialTransport("test")

    started = threading.Event()
    completed = threading.Event()

    def write_concurrently() -> None:
        started.set()
        transport.write(b"other")
        completed.set()

    with transport.transaction() as transaction_port:
        assert transaction_port is port
        thread = threading.Thread(target=write_concurrently)
        thread.start()
        assert started.wait(timeout=1)
        assert not completed.wait(timeout=0.05)

        transaction_port.write(b"query")
        transaction_port.readline()

    thread.join(timeout=1)
    assert completed.is_set()
    assert port.method_calls == [
        ("write", (b"query",), {}),
        ("readline", (), {}),
        ("write", (b"other",), {}),
    ]


def test_drain_input_continuous_bytes_raises_and_restores_timeout() -> None:
    port = Mock(timeout=1.0, baudrate=115200, in_waiting=1_048_576)
    port.read.return_value = b"x" * 1_048_576
    with patch("vxl_drivers.serial.serial.Serial", return_value=port):
        transport = SerialTransport("test")
    with pytest.raises(TimeoutError, match="drain limit"):
        transport.drain_input()
    assert port.timeout == 1.0
