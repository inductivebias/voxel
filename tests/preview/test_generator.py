"""Tests for raw overview and viewport generation."""

import asyncio
import threading
from collections.abc import Callable
from typing import Any

import numpy as np
import pytest
from ome_zarr_writer import FrameLease

from vxl.preview import LatestFrameQueue, PreviewFrame, PreviewGenerator, PreviewLayer, PreviewViewport
from vxl.preview.generator import RENDER_CAP


def _frame(w: int = 2000, h: int = 1600) -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.integers(0, 65535, size=(h, w), dtype=np.uint16)


def _discard(_frame: PreviewFrame) -> None:
    pass


def _gen(sink: Callable[[PreviewFrame], None] = _discard, **kwargs) -> PreviewGenerator:
    kwargs.setdefault("uid", "camera")
    return PreviewGenerator(sink=sink, **kwargs)


def test_expanded_grows_and_clamps() -> None:
    ex = PreviewViewport(x=0.4, y=0.4, w=0.2, h=0.2).expanded(0.25)
    assert ex.w > 0.2
    assert ex.h > 0.2
    assert ex.w <= 1.0
    assert ex.h <= 1.0
    assert ex.x >= 0.0
    assert ex.x + ex.w <= 1.0 + 1e-9
    assert PreviewViewport().expanded(0.25).w == 1.0  # a full viewport stays full


async def test_viewport_emitted_when_zoomed() -> None:
    captured: list[PreviewFrame] = []
    gen = _gen(sink=captured.append, viewport=PreviewViewport(x=0.25, y=0.25, w=0.5, h=0.5))
    try:
        gen.submit_frame(_frame(), 1)
        assert gen._viewport_task is not None
        await gen._viewport_task
    finally:
        gen.close()

    view = next(frame for frame in captured if frame.header.layer is PreviewLayer.VIEWPORT)
    assert view.header.layer is PreviewLayer.VIEWPORT
    assert view.header.source_rect_px.width > 1000  # includes overscan beyond the requested half-sensor viewport
    assert view.header.width <= RENDER_CAP
    assert view.decode().shape == (view.header.height, view.header.width)


async def test_no_viewport_frame_at_full_viewport() -> None:
    captured: list[PreviewFrame] = []
    gen = _gen(sink=captured.append)
    try:
        gen.submit_frame(_frame(), 1)
        assert gen._viewport_task is None
        assert gen._overview_future is not None
        await gen._overview_future
    finally:
        gen.close()

    assert all(frame.header.layer is not PreviewLayer.VIEWPORT for frame in captured)


async def test_latest_frame_queue_replaces_only_the_matching_pending_stream() -> None:
    queue = LatestFrameQueue()
    overview = ("488", PreviewLayer.OVERVIEW)
    viewport = ("488", PreviewLayer.VIEWPORT)

    queue.put(overview, b"sending")
    assert await queue.get() == (overview, b"sending")

    queue.put(overview, b"stale")
    queue.put(overview, b"latest")
    queue.put(viewport, b"viewport")

    assert await queue.get() == (overview, b"latest")
    assert await queue.get() == (viewport, b"viewport")


async def test_overview_and_viewport_share_capture_identity_and_resize_exactly() -> None:
    captured: list[PreviewFrame] = []
    released: list[bool] = []
    frame = _frame(w=230, h=180)
    gen = _gen(
        sink=captured.append,
        uid="camera-1",
        target_width=204,
        viewport=PreviewViewport(x=0.25, y=0.25, w=0.5, h=0.5),
    )
    try:
        gen.submit_frame(FrameLease(frame, lambda: released.append(True)), 42)
        assert gen._overview_future is not None
        assert gen._viewport_task is not None
        await gen._overview_future
        await gen._viewport_task
    finally:
        gen.close()

    assert released == [True]
    by_layer = {source.header.layer: source for source in captured}
    overview = by_layer[PreviewLayer.OVERVIEW]
    viewport = by_layer[PreviewLayer.VIEWPORT]
    assert overview.header.camera_id == viewport.header.camera_id == "camera-1"
    assert overview.header.source_stream_id == viewport.header.source_stream_id
    assert overview.header.frame_idx == viewport.header.frame_idx == 42
    assert overview.header.width == 204
    assert overview.header.source_rect_px.model_dump() == {"x": 0, "y": 0, "width": 230, "height": 180}


async def test_reset_stream_changes_capture_identity() -> None:
    captured: list[PreviewFrame] = []
    gen = _gen(sink=captured.append, uid="camera-1")
    try:
        gen.submit_frame(_frame(w=32, h=24), 0)
        assert gen._overview_future is not None
        await gen._overview_future
        await asyncio.sleep(0)  # allow the future's publish callback to run
        first = captured[-1]
        source_stream_id = gen.reset_stream()
        gen.submit_frame(_frame(w=32, h=24), 0)
        assert gen._overview_future is not None
        await gen._overview_future
        await asyncio.sleep(0)
        second = captured[-1]
    finally:
        gen.close()

    assert first.header.source_stream_id != second.header.source_stream_id == source_stream_id


async def test_bounded_lease_retention_under_zoomed_frame_flood(monkeypatch: pytest.MonkeyPatch) -> None:
    orig_from_source = PreviewFrame.from_source
    viewport_started = threading.Event()
    unblock_viewport = threading.Event()
    viewport_renders: list[int] = []

    def slow_from_source(source: np.ndarray, **kwargs: Any) -> PreviewFrame:
        if kwargs.get("layer") is PreviewLayer.VIEWPORT:
            viewport_renders.append(kwargs["frame_idx"])
            viewport_started.set()
            assert unblock_viewport.wait(timeout=5.0)
        return orig_from_source(source, **kwargs)

    monkeypatch.setattr(PreviewFrame, "from_source", slow_from_source)

    lock = threading.Lock()
    active_leases = 0
    max_active_leases = 0
    released: list[int] = []
    captured: list[PreviewFrame] = []
    raw = _frame(w=120, h=90)

    def make_lease(idx: int) -> FrameLease:
        nonlocal active_leases, max_active_leases
        with lock:
            active_leases += 1
            max_active_leases = max(max_active_leases, active_leases)

        def on_release() -> None:
            nonlocal active_leases
            with lock:
                active_leases -= 1
                released.append(idx)

        return FrameLease(raw, on_release)

    gen = _gen(
        sink=captured.append,
        target_width=64,
        viewport=PreviewViewport(x=0.2, y=0.2, w=0.4, h=0.4),
    )
    try:
        for i in range(50):
            gen.submit_frame(make_lease(i), i)

        assert viewport_started.wait(timeout=5.0)
        with lock:
            assert max_active_leases <= 4
            assert active_leases <= 4
            # Intermediate superseded leases are released immediately before the slow render finishes.
            assert len(released) >= 46

        unblock_viewport.set()
        assert gen._viewport_task is not None
        await gen._viewport_task
        if gen._overview_future is not None:
            await gen._overview_future
            await asyncio.sleep(0)
    finally:
        unblock_viewport.set()
        gen.close()

    with lock:
        assert active_leases == 0
        assert len(released) == 50
        assert sorted(released) == list(range(50))

    assert viewport_renders == [0, 49]
    viewports = [frame for frame in captured if frame.header.layer is PreviewLayer.VIEWPORT]
    assert [frame.header.frame_idx for frame in viewports] == [0, 49]
    assert viewports[-1].header.frame_idx == 49


async def test_cancel_pending_immediately_releases_waiting_viewport_lease(monkeypatch: pytest.MonkeyPatch) -> None:
    orig_from_source = PreviewFrame.from_source
    viewport_started = threading.Event()
    unblock_viewport = threading.Event()

    def slow_from_source(source: np.ndarray, **kwargs: Any) -> PreviewFrame:
        if kwargs.get("layer") is PreviewLayer.VIEWPORT:
            viewport_started.set()
            assert unblock_viewport.wait(timeout=5.0)
        return orig_from_source(source, **kwargs)

    monkeypatch.setattr(PreviewFrame, "from_source", slow_from_source)

    released: list[int] = []
    lock = threading.Lock()
    raw = _frame(w=120, h=90)

    def make_lease(idx: int) -> FrameLease:
        def on_release() -> None:
            with lock:
                released.append(idx)

        return FrameLease(raw, on_release)

    captured: list[PreviewFrame] = []
    gen = _gen(
        sink=captured.append,
        target_width=64,
        viewport=PreviewViewport(x=0.2, y=0.2, w=0.4, h=0.4),
    )
    try:
        gen.submit_frame(make_lease(0), 0)
        assert viewport_started.wait(timeout=5.0)
        if gen._overview_future is not None:
            await gen._overview_future
        gen.submit_frame(make_lease(1), 1, retain_source=False)
        if gen._overview_future is not None:
            await gen._overview_future
        with lock:
            assert 1 not in released

        gen.cancel_pending()
        # Pending waiting request (frame 1) must be released immediately on cancel_pending,
        # even while frame 0's executor render is still blocked.
        with lock:
            assert 1 in released

        unblock_viewport.set()
    finally:
        unblock_viewport.set()
        gen.close()

    assert all(frame.header.layer is not PreviewLayer.VIEWPORT for frame in captured)


async def test_acquisition_zero_lease_and_throttling_path(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: list[PreviewFrame] = []
    released: list[int] = []
    raw = _frame(w=64, h=48)

    gen = _gen(
        sink=captured.append,
        target_width=32,
        viewport=PreviewViewport(x=0.2, y=0.2, w=0.4, h=0.4),
    )
    try:
        assert gen.can_accept_overview(min_interval_s=0.25) is True

        gen.submit_frame(
            FrameLease(raw, lambda: released.append(0)),
            0,
            include_viewport=False,
            retain_source=False,
        )
        assert gen.can_accept_overview(min_interval_s=0.25) is False
        assert gen._current_source is None
        assert gen._viewport_task is None
        assert gen._overview_future is not None

        await gen._overview_future
        await asyncio.sleep(0)

        # Source lease was not retained after overview completed, even without gen.close().
        assert released == [0]
        assert gen._current_source is None
        assert gen.can_accept_overview(min_interval_s=0.25) is False

        # reset_stream() resets throttling so the first frame of a new stream is eligible immediately.
        gen.reset_stream()
        assert gen.can_accept_overview(min_interval_s=0.25) is True

        # Also verify busy-drop path with retain_source=False releases lease synchronously.
        orig_from_source = PreviewFrame.from_source
        overview_started = threading.Event()
        unblock_overview = threading.Event()

        def slow_overview(source: np.ndarray, **kwargs: Any) -> PreviewFrame:
            if kwargs.get("layer") is PreviewLayer.OVERVIEW:
                overview_started.set()
                assert unblock_overview.wait(timeout=5.0)
            return orig_from_source(source, **kwargs)

        monkeypatch.setattr(PreviewFrame, "from_source", slow_overview)
        try:
            gen.submit_frame(
                FrameLease(raw, lambda: released.append(1)),
                1,
                include_viewport=False,
                retain_source=False,
            )
            assert overview_started.wait(timeout=5.0)
            gen.submit_frame(
                FrameLease(raw, lambda: released.append(2)),
                2,
                include_viewport=False,
                retain_source=False,
            )
            # Frame 2 was dropped by busy overview gate and released immediately.
            assert 2 in released
        finally:
            unblock_overview.set()
        if gen._overview_future is not None:
            await gen._overview_future
            await asyncio.sleep(0)
        assert 1 in released
    finally:
        gen.close()

    assert len(captured) >= 1
    assert all(frame.header.layer is PreviewLayer.OVERVIEW for frame in captured)


async def test_rapid_set_viewport_coalesces_to_latest_viewport(monkeypatch: pytest.MonkeyPatch) -> None:
    orig_from_source = PreviewFrame.from_source
    viewport_started = threading.Event()
    unblock_viewport = threading.Event()
    rendered_viewports: list[PreviewViewport] = []

    def slow_from_source(source: np.ndarray, **kwargs: Any) -> PreviewFrame:
        if kwargs.get("layer") is PreviewLayer.VIEWPORT:
            rendered_viewports.append(kwargs["viewport"])
            viewport_started.set()
            assert unblock_viewport.wait(timeout=5.0)
        return orig_from_source(source, **kwargs)

    monkeypatch.setattr(PreviewFrame, "from_source", slow_from_source)

    captured: list[PreviewFrame] = []
    released: list[str] = []
    raw = _frame(w=200, h=160)
    gen = _gen(sink=captured.append, target_width=64)
    try:
        gen.submit_frame(FrameLease(raw, lambda: released.append("preview")), 0)
        if gen._overview_future is not None:
            await gen._overview_future

        last_task: asyncio.Task[None] | None = None
        for step in range(20):
            vp = PreviewViewport(x=0.01 * step, y=0.01 * step, w=0.4, h=0.4)
            last_task = gen.set_viewport(vp, regenerate=True)

        assert viewport_started.wait(timeout=5.0)
        unblock_viewport.set()
        assert last_task is not None
        await last_task

        # Only the initial render and the final coalesced viewport should have executed.
        assert len(rendered_viewports) == 2
        viewports = [frame for frame in captured if frame.header.layer is PreviewLayer.VIEWPORT]
        assert len(viewports) == 1

        # Transitioning to retain_source=False releases the previously retained preview source.
        gen.reset_stream()
        gen.submit_frame(FrameLease(raw, lambda: released.append("retained")), 1, retain_source=True)
        if gen._overview_future is not None:
            await gen._overview_future
        if gen._viewport_task is not None:
            await gen._viewport_task
        assert "retained" not in released

        gen.reset_stream()
        # Re-seed _current_source then call submit_frame with retain_source=False directly:
        gen.submit_frame(
            FrameLease(raw, lambda: released.append("prev")),
            2,
            include_viewport=False,
            retain_source=True,
        )
        if gen._overview_future is not None:
            await gen._overview_future
        assert "prev" not in released

        gen.submit_frame(
            FrameLease(raw, lambda: released.append("acq")),
            3,
            include_viewport=False,
            retain_source=False,
        )
        assert "prev" in released
        if gen._overview_future is not None:
            await gen._overview_future
        assert "acq" in released
    finally:
        unblock_viewport.set()
        gen.close()


async def test_continuous_zoomed_stream_does_not_starve_sink(monkeypatch: pytest.MonkeyPatch) -> None:
    orig_from_source = PreviewFrame.from_source
    started: dict[int, threading.Event] = {}
    gates: dict[int, threading.Event] = {}
    call_count = 0
    call_lock = threading.Lock()

    def gated_from_source(source: np.ndarray, **kwargs: Any) -> PreviewFrame:
        nonlocal call_count
        if kwargs.get("layer") is PreviewLayer.VIEWPORT:
            with call_lock:
                seq = call_count
                call_count += 1
                start_ev = started.setdefault(seq, threading.Event())
                gate_ev = gates.setdefault(seq, threading.Event())
            start_ev.set()
            assert gate_ev.wait(timeout=5.0)
        return orig_from_source(source, **kwargs)

    monkeypatch.setattr(PreviewFrame, "from_source", gated_from_source)

    for seq in range(3):
        started[seq] = threading.Event()
        gates[seq] = threading.Event()

    vp1 = PreviewViewport(x=0.2, y=0.2, w=0.4, h=0.4)
    vp2 = PreviewViewport(x=0.3, y=0.3, w=0.3, h=0.3)
    captured: list[PreviewFrame] = []
    raw = _frame(w=160, h=120)
    gen = _gen(sink=captured.append, target_width=64, viewport=vp1)
    try:
        # Render 0 starts on frame 0; frame 1 arrives while frame 0 is still rendering at the same viewport.
        gen.submit_frame(raw, 0, include_viewport=True)
        assert started[0].wait(timeout=5.0)
        gen.submit_frame(raw, 1, include_viewport=True)

        # Unblock render 0 while keeping render 1 blocked: frame 0 must still emit rather than starving!
        gates[0].set()
        assert await asyncio.to_thread(started[1].wait, 5.0)
        emitted_viewports = [f for f in captured if f.header.layer is PreviewLayer.VIEWPORT]
        assert [f.header.frame_idx for f in emitted_viewports] == [0]

        # Now change viewport to vp2 while render 1 (at obsolete vp1) is still in flight.
        task = gen.set_viewport(vp2, regenerate=True)
        gates[1].set()
        gates[2].set()
        assert task is not None
        await task

        # Obsolete vp1 render for frame 1 is skipped; render 2 (at vp2 for frame 1) emits.
        emitted_viewports = [f for f in captured if f.header.layer is PreviewLayer.VIEWPORT]
        assert len(emitted_viewports) == 2
        assert emitted_viewports[0].header.frame_idx == 0
        assert emitted_viewports[1].header.frame_idx == 1
        assert emitted_viewports[0].header.source_rect_px != emitted_viewports[1].header.source_rect_px
    finally:
        for ev in gates.values():
            ev.set()
        gen.close()
