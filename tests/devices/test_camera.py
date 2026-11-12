import asyncio
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path, PurePosixPath
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest
from ome_zarr_writer import OMEZarrWriter, ScaleLevel, WriterConfig, WriterSettings
from ome_zarr_writer.sizing import ring_shm_bytes
from ome_zarr_writer.writer import BatchMetrics
from pydantic import ValidationError
from rigup.node import LocalAdapter
from vxl_records import StorageSpec
from vxlib.vector import IVec2D, Vec2D

from vxl.devices.camera import Camera, CameraController, CameraHandle, CameraMode, SensorROI, WriterCloseError
from vxl.devices.camera.base import (
    ACQUISITION_PREVIEW_INTERVAL_S,
    IntRange,
    PixelFormat,
    ROIGrid,
    StreamInfo,
    TriggerMode,
    TriggerPolarity,
)
from vxl.system import System


@pytest.fixture(autouse=True)
def _isolated_ram_registry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VOXEL_RAM_REGISTRY", str(tmp_path / "ram_consumers.json"))


class _StubCamera(Camera):
    def __init__(self, uid: str = "test_cam") -> None:
        super().__init__(uid=uid)
        self._roi = SensorROI(x=0, y=0, w=64, h=64)
        self._pixel_format: PixelFormat = "MONO12"
        self._binning = 1
        self._exposure_ms = 5.0
        self._fps = 100.0
        self.pixel_format_reads = 0
        self.pixel_format_threads: list[int] = []
        self.stop_calls = 0
        self.free_calls = 0
        self.free_error: Exception | None = None
        self._grabbed = 0

    @property
    def sensor_size_px(self) -> IVec2D:
        return IVec2D(y=64, x=64)

    @property
    def pixel_size_um(self) -> Vec2D:
        return Vec2D(y=1.0, x=1.0)

    @property
    def pixel_format(self) -> PixelFormat:
        self.pixel_format_reads += 1
        self.pixel_format_threads.append(threading.get_ident())
        return self._pixel_format

    @pixel_format.setter
    def pixel_format(self, pixel_format: str) -> None:
        self._pixel_format = cast("PixelFormat", pixel_format)

    @property
    def binning(self) -> int:
        return self._binning

    @binning.setter
    def binning(self, binning: int) -> None:
        self._binning = binning

    @property
    def exposure_time_ms(self) -> float:
        return self._exposure_ms

    @exposure_time_ms.setter
    def exposure_time_ms(self, exposure_time_ms: float) -> None:
        self._exposure_ms = exposure_time_ms

    @property
    def frame_rate_hz(self) -> float:
        return self._fps

    @frame_rate_hz.setter
    def frame_rate_hz(self, value: float) -> None:
        self._fps = value

    def _get_roi(self) -> SensorROI:
        return self._roi

    def _set_roi(self, roi: SensorROI) -> None:
        self._roi = roi

    @property
    def roi_grid(self) -> ROIGrid:
        return ROIGrid(h=IntRange(min=16, max=64, step=16), v=IntRange(min=16, max=64, step=16))

    @property
    def stream_info(self) -> StreamInfo | None:
        return None

    def _configure_trigger_mode(self, mode: TriggerMode) -> None:
        self.trigger_mode = mode

    def _configure_trigger_polarity(self, polarity: TriggerPolarity) -> None:
        self.trigger_polarity = polarity

    def _start(self, frame_count: int | None = None) -> None:
        del frame_count

    def _allocate_buffer(self) -> None:
        return

    def _free_buffer(self) -> None:
        self.free_calls += 1
        if self.free_error is not None:
            raise self.free_error

    def grab_frame(self, *, out: np.ndarray) -> None:
        self._grab_frame(out)

    def _grab_frame(self, out: np.ndarray) -> None:
        self._grabbed += 1
        out.fill(self._grabbed)

    def stop(self) -> None:
        self.stop_calls += 1


def test_sensor_roi_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        SensorROI.model_validate({"x": 0, "y": 0, "w": 128, "h": 128, "width": 128})


def test_writer_telemetry_excludes_pending_batches_and_bounds_history() -> None:

    batches = BatchMetrics.create_many(12000, 1)
    controller = cast("CameraController", SimpleNamespace(_writer=SimpleNamespace(batches=batches)))
    getter = CameraController.writer_metrics.fget
    assert getter is not None
    assert getter(controller) == []
    for batch in batches[:100]:
        batch.collecting.begin()
    assert [batch.batch_idx for batch in getter(controller)] == list(range(36, 100))
    assert len(batches) > 12000  # live filtering must not truncate the persisted history


async def test_collect_batch_throttles_overview_without_frame_leases_or_event_loop_sdk_reads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    camera = _StubCamera("acq_cam")
    controller = CameraController(camera)
    slot_buffer = np.zeros((64, 64), dtype=np.uint16)

    @contextmanager
    def new_frame() -> Iterator[np.ndarray]:
        yield slot_buffer

    writer = SimpleNamespace(
        new_frame=new_frame,
        latest_frame=MagicMock(side_effect=AssertionError("latest_frame must not be called during acquisition")),
    )
    decisions = iter([True, False, False, True, False])
    can_accept_calls: list[float] = []
    submitted: list[tuple[np.ndarray, int, dict[str, object]]] = []

    def can_accept_overview(*, min_interval_s: float = 0.0) -> bool:
        can_accept_calls.append(min_interval_s)
        return next(decisions)

    def submit_frame(frame: np.ndarray, idx: int, **kwargs: object) -> None:
        submitted.append((frame, idx, kwargs))

    monkeypatch.setattr(controller._previewer, "can_accept_overview", can_accept_overview, raising=False)
    monkeypatch.setattr(controller._previewer, "submit_frame", submit_frame)

    loop_thread = threading.get_ident()
    try:
        await controller._collect_batch(5, cast("OMEZarrWriter", writer))
        assert writer.latest_frame.call_count == 0
        assert can_accept_calls == [ACQUISITION_PREVIEW_INTERVAL_S] * 5
        assert camera.pixel_format_reads == 1
        assert camera.pixel_format_threads[0] != loop_thread
        assert camera.stop_calls == 1
        assert await controller.captured_frames() == 5

        assert [idx for _, idx, _ in submitted] == [0, 3]
        for expected_val, (arr, _idx, kwargs) in zip((1, 4), submitted, strict=True):
            assert kwargs == {"valid_bits": 12, "include_viewport": False, "retain_source": False}
            assert not np.shares_memory(arr, slot_buffer)
            assert int(arr[0, 0]) == expected_val
    finally:
        await controller.close()


async def test_close_stack_writer_failure_resets_hardware_and_raises_writer_close_error() -> None:
    camera = _StubCamera("writer_err_cam")
    controller = CameraController(camera)
    handle = CameraHandle(LocalAdapter(controller))
    controller._mode = CameraMode.ACQUISITION
    controller._writer = cast(
        "OMEZarrWriter",
        SimpleNamespace(end_stack=MagicMock(side_effect=OSError("disk full"))),
    )

    try:
        await handle.close_stack()
        assert controller._task is not None
        await asyncio.gather(controller._task, return_exceptions=True)

        assert camera.stop_calls == 1
        assert camera.free_calls == 1
        assert controller.mode is CameraMode.IDLE

        with pytest.raises(WriterCloseError, match="Writer close failed: disk full"):
            await controller.capture_state()
        with pytest.raises(WriterCloseError, match="Writer close failed: disk full"):
            await handle.capture_state()
    finally:
        controller._writer = None
        await handle.close()


async def test_camera_handle_capture_state_maps_serialized_writer_close_error() -> None:
    writer_err_handle = cast(
        "CameraHandle",
        SimpleNamespace(call=AsyncMock(side_effect=RuntimeError("Writer close failed: s3 upload failed"))),
    )
    with pytest.raises(WriterCloseError, match="Writer close failed: s3 upload failed"):
        await CameraHandle.capture_state(writer_err_handle)

    hw_err_handle = cast(
        "CameraHandle",
        SimpleNamespace(call=AsyncMock(side_effect=RuntimeError("Capture task failed: camera timeout"))),
    )
    with pytest.raises(RuntimeError, match="camera timeout") as exc_info:
        await CameraHandle.capture_state(hw_err_handle)
    assert not isinstance(exc_info.value, WriterCloseError)


@pytest.mark.parametrize("writer_also_fails", [False, True])
async def test_close_stack_free_buffer_failure_raises_non_writer_runtime_error(writer_also_fails: bool) -> None:
    camera = _StubCamera("hw_err_cam")
    camera.free_error = RuntimeError("DMA buffer pinned")
    controller = CameraController(camera)
    handle = CameraHandle(LocalAdapter(controller))
    controller._mode = CameraMode.ACQUISITION
    end_stack = MagicMock(side_effect=OSError("disk full") if writer_also_fails else None)
    controller._writer = cast("OMEZarrWriter", SimpleNamespace(end_stack=end_stack))

    try:
        await handle.close_stack()
        assert controller._task is not None
        await asyncio.gather(controller._task, return_exceptions=True)

        assert camera.stop_calls == 1
        assert camera.free_calls == 1
        assert controller.mode is CameraMode.ACQUISITION

        with pytest.raises(RuntimeError, match="DMA buffer pinned") as ctrl_exc:
            await controller.capture_state()
        assert not isinstance(ctrl_exc.value, WriterCloseError)

        with pytest.raises(RuntimeError, match="DMA buffer pinned") as handle_exc:
            await handle.capture_state()
        assert not isinstance(handle_exc.value, WriterCloseError)
    finally:
        camera.free_error = None
        controller._mode = CameraMode.IDLE
        controller._writer = None
        await handle.close()


async def test_camera_controller_records_committed_ring_bytes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    committed_calls: list[tuple[str, int]] = []
    monkeypatch.setattr(
        System.Ram,
        "record_committed",
        classmethod(lambda _cls, uid, nbytes: committed_calls.append((uid, nbytes))),
        raising=False,
    )
    monkeypatch.setattr(System.Ram, "available_bytes", classmethod(lambda _cls: 64 * 1024**3))
    camera = _StubCamera("ram_cam")
    controller = CameraController(camera, system=System(store=tmp_path, scratch=tmp_path / "scratch"))
    captured_cfgs: list[WriterConfig] = []

    class _FakeWriter:
        def __init__(self) -> None:
            self.target: Path | None = None
            self._allocated = False

        def begin_stack(self, cfg: WriterConfig, _dest: object, *, sizer: object) -> None:
            captured_cfgs.append(cfg)
            if not self._allocated:
                assert callable(sizer)
                sizer(cfg)
                self._allocated = True
            self.target = tmp_path / "stack.ome.zarr"

        def end_stack(self) -> None:
            self.target = None

        def close(self) -> None:
            self.target = None

    controller._writer = cast("OMEZarrWriter", _FakeWriter())
    try:
        await controller.open_stack(
            num_frames=4,
            z_step=1.0,
            magnification=1.0,
            storage=StorageSpec(path=PurePosixPath("run")),
            subpath=PurePosixPath("vol/ch"),
            settings=WriterSettings(max_level=ScaleLevel.L0),
        )
        cfg = captured_cfgs[0]
        expected_shm = ring_shm_bytes(controller._ring_slots, cfg.batch_shape, cfg.max_level, cfg.dtype)
        assert expected_shm > 0
        assert committed_calls == [("ram_cam", 0), ("ram_cam", expected_shm)]

        await controller.close_stack()
        assert controller._task is not None
        await controller._task
        await controller.release_writer()
        assert controller._ring_slots == 0
        assert committed_calls[-1] == ("ram_cam", 0)
    finally:
        await controller.close()
