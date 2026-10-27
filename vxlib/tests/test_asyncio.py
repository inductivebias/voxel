import asyncio

import pytest
from vxlib.asyncio import settle


async def test_settle_repeated_cancellation_preserves_owned_work_until_complete() -> None:
    work = asyncio.get_running_loop().create_future()
    waiter = asyncio.create_task(settle(work))
    await asyncio.sleep(0)
    for _ in range(2):
        waiter.cancel()
        await asyncio.sleep(0)
        assert not work.cancelled()
        assert not waiter.done()
    work.set_result("finished")
    with pytest.raises(asyncio.CancelledError):
        await waiter
    assert work.result() == "finished"


async def test_settle_owned_failure_is_reported() -> None:
    work = asyncio.get_running_loop().create_future()
    work.set_exception(OSError("write failed"))
    with pytest.raises(OSError, match="write failed"):
        await settle(work)
