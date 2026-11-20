"""Stage jog updates on the station state WebSocket reach the instrument; malformed ones are ignored."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any
from uuid import UUID, uuid4

import msgpack

from vxl.web.realtime import Realtime


class FakeInstrument:
    def __init__(self) -> None:
        self.jogs: list[tuple[str, int, tuple[float, float, float]]] = []

    async def jog_stage(self, client: str, seq: int, velocity: tuple[float, float, float]) -> None:
        self.jogs.append((client, seq, velocity))


class FakeStation:
    def __init__(self, session_id: UUID, instrument: FakeInstrument) -> None:
        self._session_id = session_id
        self._instrument = instrument

    @asynccontextmanager
    async def instrument(self, session_id: UUID) -> AsyncIterator[FakeInstrument]:
        if session_id != self._session_id:
            raise RuntimeError("no such session")
        yield self._instrument


class FakeWebSocket:
    def __init__(self, messages: list[dict[str, Any]]) -> None:
        self._messages = [{"type": "websocket.receive", "bytes": msgpack.packb(m)} for m in messages]
        self._messages.append({"type": "websocket.disconnect"})

    async def receive(self) -> dict[str, Any]:
        return self._messages.pop(0)


async def test_jog_updates_are_routed_and_invalid_ones_ignored() -> None:
    session_id = uuid4()
    instrument = FakeInstrument()
    realtime = Realtime.__new__(Realtime)
    realtime._station = FakeStation(session_id, instrument)  # type: ignore[assignment]
    sessions: set[UUID] = set()
    websocket = FakeWebSocket(
        [
            {"action": "stage.jog", "session_id": str(session_id), "seq": 1, "velocity": [0.5, 0.0, -1.0]},
            {"action": "stage.jog", "session_id": str(session_id), "seq": 2, "velocity": [0.5, 0.0]},
            {"action": "stage.jog", "session_id": str(session_id), "seq": 3, "velocity": [float("nan"), 0.0, 0.0]},
            {"action": "stage.jog", "session_id": str(uuid4()), "seq": 3, "velocity": [1.0, 0.0, 0.0]},
        ]
    )

    await realtime._receive(websocket, "client-1", sessions)  # type: ignore[arg-type]

    assert instrument.jogs == [("client-1", 1, (0.5, 0.0, -1.0))]
    assert session_id in sessions
