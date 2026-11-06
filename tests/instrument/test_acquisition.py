import asyncio
from pathlib import Path, PurePosixPath
from typing import Literal

import pytest
from vxl_records import (
    AcquisitionStatus,
    DatasetStatus,
    LocalLocation,
    LocationRole,
    LocationStatus,
    SQLiteRecords,
    StorageSpec,
    VolumeStatus,
)

from vxl.devices.camera import CaptureState
from vxl.instrument import (
    AcquisitionRequest,
    AcquisitionTask,
    Activity,
    CaptureActivity,
    IdleActivity,
    Instrument,
    InstrumentBusyError,
    Point2D,
    XYDefinition,
    XYMode,
    ZRange,
)
from vxl.instrument.config import ProfilePatch


class _Writer:
    def __init__(self, root: Path, *, close_error: bool = False) -> None:
        self.uid = "camera_1"
        self._root = root
        self._close_error = close_error
        self._closing = False
        self.released = False

    async def check_writable(self, _storage: StorageSpec) -> None:
        return

    async def open_stack(self, *, storage: StorageSpec, subpath: PurePosixPath, **_kwargs: object) -> LocalLocation:
        target = self._root / storage.path.as_posix() / f"{subpath.as_posix()}.ome.zarr"
        return LocalLocation(
            role=LocationRole.DESTINATION,
            status=LocationStatus.WRITING,
            host="camera-1",
            path=str(target),
        )

    async def begin_batch(self, _num_frames: int) -> None:
        return

    async def capture_state(self) -> CaptureState:
        return CaptureState.CLOSED if self._closing else CaptureState.DONE

    async def close_stack(self) -> None:
        if self._close_error:
            raise RuntimeError("writer close failed")
        self._closing = True

    async def release_writer(self) -> None:
        self.released = True


@pytest.fixture
async def writer(opened_instrument: Instrument, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> _Writer:
    instrument = opened_instrument
    writer = _Writer(tmp_path / "data")
    camera = instrument._hal.cameras["camera_1"]
    for name in ("check_writable", "open_stack", "begin_batch", "capture_state", "close_stack", "release_writer"):
        monkeypatch.setattr(camera, name, getattr(writer, name))
    await instrument.add_task(
        AcquisitionTask(
            id="task",
            xy=XYDefinition(mode=XYMode.EXPLICIT_POINTS, points=[Point2D(x=0, y=0)]),
            profiles=["single_gfp"],
            z=ZRange(start=0, end=0),
        )
    )
    return writer


async def test_acquisition_manifest_tracks_completed_dataset(
    opened_instrument: Instrument, records: SQLiteRecords, writer: _Writer, tmp_path: Path
) -> None:
    instrument = opened_instrument
    catalog = records.acquisitions
    storage = StorageSpec(path=PurePosixPath("run"))
    activities: list[Activity] = []
    instrument.activity.subscribe(activities.append)
    started = await instrument.start_acquisition(AcquisitionRequest(storage=storage, operator="operator"))
    await asyncio.wait_for(instrument.wait_acquisition(), timeout=5)
    captures = [activity for activity in activities if isinstance(activity, CaptureActivity)]
    states = [capture.acquisition for capture in captures if capture.acquisition is not None]

    persisted = await catalog.get(started.id)
    dataset = persisted.volumes[0].datasets["gfp"]
    assert started.status is AcquisitionStatus.RUNNING
    assert started.volumes[0].task == "task"
    assert started.volumes[0].profile == "single_gfp"
    assert captures[0] == CaptureActivity()
    assert states[0].frames_captured == 0
    assert any(capture.channels == ("gfp",) for capture in captures)
    assert isinstance(activities[-1], IdleActivity)
    assert any(state.manifest_revision > started.revision and state.frames_captured == 1 for state in states)
    assert persisted.status is AcquisitionStatus.COMPLETED
    assert persisted.volumes[0].status is VolumeStatus.COMPLETED
    assert dataset.status is DatasetStatus.COMPLETED
    location = dataset.locations[0]
    assert isinstance(location, LocalLocation)
    assert location.status is LocationStatus.AVAILABLE
    assert location.path == str(tmp_path / "data/run/volumes/000001/single_gfp/gfp.ome.zarr")
    assert persisted.state_snapshot == instrument.state.value.model_dump(mode="json")
    assert (tmp_path / "data/run/manifest.json").is_file()
    assert not (tmp_path / "data/run/record.json").exists()
    assert not writer.released
    assert isinstance(instrument.activity.value, IdleActivity)


async def test_writer_close_failure_marks_dataset_and_acquisition_failed(
    opened_instrument: Instrument, records: SQLiteRecords, writer: _Writer
) -> None:
    instrument = opened_instrument
    catalog = records.acquisitions
    storage = StorageSpec(path=PurePosixPath("run"))
    writer._close_error = True

    started = await instrument.start_acquisition(AcquisitionRequest(storage=storage, operator="operator"))
    await asyncio.wait_for(instrument.wait_acquisition(), timeout=5)

    persisted = await catalog.get(started.id)
    dataset = persisted.volumes[0].datasets["gfp"]
    assert persisted.status is AcquisitionStatus.FAILED
    assert persisted.failure is not None
    assert persisted.volumes[0].status is VolumeStatus.FAILED
    assert dataset.status is DatasetStatus.PARTIAL
    assert dataset.locations[0].status is LocationStatus.FAILED
    assert not writer.released
    assert isinstance(instrument.activity.value, CaptureActivity)
    with pytest.raises(InstrumentBusyError, match="already in progress"):
        await instrument.start_acquisition(AcquisitionRequest(storage=storage))


async def test_profile_preparation_failure_still_stops_daq_and_finalizes_volume(
    opened_instrument: Instrument,
    records: SQLiteRecords,
    writer: _Writer,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    instrument = opened_instrument
    stopped = []

    async def fail():
        raise RuntimeError("profile preparation failed")

    async def stop():
        stopped.append(True)

    monkeypatch.setattr(instrument, "_apply_settings", fail)
    monkeypatch.setattr(instrument._hal.signal_generators["daq"], "stop", stop)
    started = await instrument.start_acquisition(AcquisitionRequest(storage=StorageSpec(path=PurePosixPath("run"))))
    await asyncio.wait_for(instrument.wait_acquisition(), timeout=5)
    manifest = await records.acquisitions.get(started.id)
    assert manifest.status is AcquisitionStatus.FAILED
    assert manifest.volumes[0].status is VolumeStatus.FAILED
    assert stopped
    assert not writer.released
    laser = instrument._hal.lasers["laser_488"]
    assert not (await laser.props.get("is_output_enabled")).ok["is_output_enabled"].value


@pytest.mark.parametrize(
    "selected", [{}, {"laser_561": "software"}, {"laser_488": "software", "laser_561": "software"}]
)
async def test_capture_uses_profile_lasers_independently_of_camera(
    opened_instrument: Instrument,
    writer: _Writer,
    monkeypatch: pytest.MonkeyPatch,
    records: SQLiteRecords,
    selected: dict[str, Literal["software", "digital", "analog"]],
) -> None:
    instrument = opened_instrument
    await instrument.update_profile(ProfilePatch(lasers=selected))
    observed = []

    async def start():
        enabled = set()
        for uid, laser in instrument._hal.lasers.items():
            if await laser.props.get_value("is_output_enabled"):
                enabled.add(uid)
                assert await laser.props.get_value("modulation_mode") == "software"
        observed.append(enabled)

    monkeypatch.setattr(instrument._hal.signal_generators["daq"], "start", start)
    manifest = await instrument.start_acquisition(
        AcquisitionRequest(storage=StorageSpec(path=PurePosixPath("independent-lasers")))
    )
    await asyncio.wait_for(instrument.wait_acquisition(), timeout=5)
    assert (await records.acquisitions.get(manifest.id)).status is AcquisitionStatus.COMPLETED
    assert observed == [set(selected)]
    assert not writer.released
    for laser in instrument._hal.lasers.values():
        assert not await laser.props.get_value("is_output_enabled")


async def test_acquisition_skips_unchanged_camera_reconfiguration(
    opened_instrument: Instrument,
    writer: _Writer,
    records: SQLiteRecords,
) -> None:
    instrument = opened_instrument
    camera = instrument.active_channels["gfp"].camera
    await instrument.set_device_properties("camera_1", {"exposure_time_ms": 15.0})
    await camera.props.refresh()
    await camera.roi.get()
    await camera.sensor_size_px.get()

    set_calls: list[dict[str, object]] = []
    orig_set = camera.props.set

    async def record_set(**props: object):
        set_calls.append(props)
        return await orig_set(**props)

    camera.props.set = record_set  # type: ignore[method-assign]
    manifest = await instrument.start_acquisition(
        AcquisitionRequest(storage=StorageSpec(path=PurePosixPath("no-camera-reconfig")))
    )
    await asyncio.wait_for(instrument.wait_acquisition(), timeout=5)
    assert (await records.acquisitions.get(manifest.id)).status is AcquisitionStatus.COMPLETED
    assert not writer.released
    assert set_calls == []
