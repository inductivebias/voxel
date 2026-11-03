import asyncio
import sys
import threading
from pathlib import Path
from typing import ClassVar

import pytest
from rigup.config import NodeConfig
from rigup.node import LocalNode, SubprocessNode

from rigup import Device, DeviceConfig, Rig, RigConfig


class _ClosingDevice(Device):
    def __init__(self, uid: str, fail_close: bool = False) -> None:
        super().__init__(uid)
        self.fail_close = fail_close
        self.closed = False

    def close(self) -> None:
        if self.fail_close:
            self.fail_close = False
            raise OSError("close failed once")
        self.closed = True


async def test_rig_close_failure_attempts_other_devices_and_retains_failed_owner() -> None:
    rig = Rig(
        RigConfig(
            devices={
                "good": DeviceConfig(target=f"{__name__}._ClosingDevice"),
                "bad": DeviceConfig(target=f"{__name__}._ClosingDevice", init={"fail_close": True}),
            }
        )
    )
    await rig.open()
    with pytest.raises(ExceptionGroup, match="Rig shutdown failed"):
        await rig.close()
    assert set(rig.nodes["local"].devices) == {"bad"}, "successful owners close even after a failure"
    await rig.close()
    assert not rig.nodes


class _ConstructingDevice(Device):
    instances: ClassVar[list["_ConstructingDevice"]] = []

    def __init__(self, uid: str, started: threading.Event, release: threading.Event) -> None:
        super().__init__(uid)
        started.set()
        if not release.wait(timeout=5):
            raise TimeoutError("constructor did not finish")
        self.closed = False
        self.instances.append(self)

    def close(self) -> None:
        self.closed = True


async def test_build_repeated_cancellation_closes_late_constructor_result() -> None:
    started, release = threading.Event(), threading.Event()
    node = LocalNode()
    config = DeviceConfig(target=f"{__name__}._ConstructingDevice", init={"started": started, "release": release})
    building = asyncio.create_task(node.build_devices({"slow": config}))
    try:
        assert await asyncio.to_thread(started.wait, 2)
        for _ in range(2):
            building.cancel()
            await asyncio.sleep(0)
    finally:
        release.set()
    with pytest.raises(asyncio.CancelledError):
        await building
    assert _ConstructingDevice.instances[-1].closed
    assert not node.devices
    await node.close()


async def test_subprocess_node_open_fails_fast_and_cleans_up_when_child_exits_early(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_exec = asyncio.create_subprocess_exec

    async def crashing_exec(*_args: object, **_kwargs: object) -> asyncio.subprocess.Process:
        return await real_exec(sys.executable, "-c", "raise SystemExit(2)")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", crashing_exec)
    node = SubprocessNode("crash", NodeConfig(kind="subprocess"))
    with pytest.raises(RuntimeError, match="exited during startup"):
        await asyncio.wait_for(node.open(), timeout=5.0)
    assert node._process is None
    assert node._ipc_dir is None or not Path(node._ipc_dir).exists()
