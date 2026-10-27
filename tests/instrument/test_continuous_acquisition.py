import asyncio
import json
from pathlib import Path, PurePosixPath
from typing import Any

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
from vxl.devices.daq.clocked import EdgeDivision
from vxl.instrument import (
    AcquisitionRequest,
    AcquisitionTask,
    Instrument,
    InstrumentConfig,
    OperationRejectedError,
    Point2D,
    StartupError,
    XYDefinition,
    XYMode,
    ZRange,
)

STORAGE = StorageSpec(path=PurePosixPath("run"))


@pytest.fixture
def scan_options(request: pytest.FixtureRequest) -> dict[str, Any]:
    return getattr(request, "param", {})


@pytest.fixture
def instrument_config(instrument_config: InstrumentConfig, scan_options: dict[str, Any]) -> InstrumentConfig:
    """simulated-local with z scanned continuously: 20 um encoder edges, planes 80 um apart at 1 mm/s."""
    options = scan_options
    config = instrument_config.model_dump(mode="json")
    devices = config["hal"]["devices"]
    devices["daq"]["init"]["edge_source"] = "pfi1"
    devices["z_axis"]["init"]["scan_edge_spacing"] = 20.0
    devices["scan_recorder"] = {
        "target": "vxl.devices.daq.SimulatedPositionRecorder",
        "init": {
            "markers": options.get("markers", {"camera_1": "pfi4"}),
            "missing_frames": options.get("missing_frames"),
        },
    }
    config["hal"]["continuous_scan"] = {"generator": "daq", "recorder": "scan_recorder"}
    for profile in config["default"]["imaging"]["profiles"].values():
        profile["z_step"] = options.get("z_step", 80.0)
        profile["sync"]["daq"].update(duration="65 ms", rest_time="15 ms")
    return InstrumentConfig.model_validate(config)


class _Writer:
    def __init__(self, root: Path) -> None:
        self._root = root
        self._closing = False
        self.batches: list[int] = []

    async def check_writable(self, _storage: StorageSpec) -> None:
        return

    async def open_stack(self, *, storage: StorageSpec, subpath: PurePosixPath, **_kwargs: object) -> LocalLocation:
        target = self._root / storage.path.as_posix() / f"{subpath.as_posix()}.ome.zarr"
        return LocalLocation(role=LocationRole.DESTINATION, status=LocationStatus.WRITING, host="h", path=str(target))

    async def begin_batch(self, num_frames: int) -> None:
        self.batches.append(num_frames)

    async def capture_state(self) -> CaptureState:
        return CaptureState.CLOSED if self._closing else CaptureState.DONE

    async def close_stack(self) -> None:
        self._closing = True

    async def release_writer(self) -> None:
        return


@pytest.fixture
async def writer(opened_instrument: Instrument, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> _Writer:
    writer = _Writer(tmp_path / "data")
    camera = opened_instrument._hal.cameras["camera_1"]
    for name in ("check_writable", "open_stack", "begin_batch", "capture_state", "close_stack", "release_writer"):
        monkeypatch.setattr(camera, name, getattr(writer, name))
    return writer


async def add_task(instrument: Instrument, *, start: float = 1000.0, end: float = 1080.0) -> None:
    await instrument.add_task(
        AcquisitionTask(
            id="task",
            xy=XYDefinition(mode=XYMode.EXPLICIT_POINTS, points=[Point2D(x=0, y=0)]),
            profiles=["single_gfp"],
            z=ZRange(start=start, end=end),
        )
    )


def record_paced_starts(instrument: Instrument, monkeypatch: pytest.MonkeyPatch) -> list[EdgeDivision]:
    generator = instrument._hal.signal_generators["daq"]
    original = generator.start_paced
    divisions: list[EdgeDivision] = []

    async def start_paced(division: EdgeDivision) -> dict[str, float | int | str]:
        divisions.append(division)
        return await original(division)

    monkeypatch.setattr(generator, "start_paced", start_paced)
    return divisions


async def test_continuous_acquisition_captures_one_paced_pass_and_records_every_frame_position(
    opened_instrument: Instrument,
    records: SQLiteRecords,
    writer: _Writer,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    instrument = opened_instrument
    divisions = record_paced_starts(instrument, monkeypatch)
    await add_task(instrument)

    started = await instrument.start_acquisition(AcquisitionRequest(storage=STORAGE))
    await asyncio.wait_for(instrument.wait_acquisition(), timeout=10)

    manifest = await records.acquisitions.get(started.id)
    assert manifest.status is AcquisitionStatus.COMPLETED
    assert manifest.volumes[0].datasets["gfp"].status is DatasetStatus.COMPLETED
    assert writer.batches == [2], "the whole pass streams as one batch, never re-arming mid-motion"
    assert divisions == [EdgeDivision(cycles=2, divider=4, initial_delay=4)]
    scan = json.loads((tmp_path / "data/run/volumes/000001/single_gfp/scan.json").read_text())
    assert scan["plan"]["start"] == pytest.approx(940.0), "three edges of lead-in before the first plane"
    assert scan["plan"]["end"] == pytest.approx(1260.0)
    positions = [frame["position"] for frame in scan["recorded"]["frames"]["camera_1"]]
    assert positions == pytest.approx([1010.0, 1090.0])
    z_axis = instrument._hal.stage.scanning_axis
    assert await z_axis.props.get_value("scan_mode") == "IDLE"
    assert await z_axis.get_position() == pytest.approx(1260.0)


@pytest.mark.parametrize("scan_options", [{"z_step": 70.0}], indirect=True)
async def test_start_acquisition_refuses_spacing_that_is_not_whole_encoder_edges_before_moving(
    opened_instrument: Instrument, records: SQLiteRecords, writer: _Writer
) -> None:
    instrument = opened_instrument
    await add_task(instrument, end=1070.0)
    before = await instrument._hal.stage.scanning_axis.get_position()

    with pytest.raises(OperationRejectedError, match=r"profile 'single_gfp': z_step 70 is not a whole number.*use 80"):
        await instrument.start_acquisition(AcquisitionRequest(storage=STORAGE))

    assert await records.acquisitions.list_manifests() == []
    assert await instrument._hal.stage.scanning_axis.get_position() == before
    assert writer.batches == []


@pytest.mark.parametrize("scan_options", [{"missing_frames": {"camera_1": [1]}}], indirect=True)
async def test_continuous_acquisition_with_a_missing_exposure_fails_and_restores_the_scan_axis(
    opened_instrument: Instrument, records: SQLiteRecords, writer: _Writer
) -> None:
    instrument = opened_instrument
    await add_task(instrument)

    started = await instrument.start_acquisition(AcquisitionRequest(storage=STORAGE))
    await asyncio.wait_for(instrument.wait_acquisition(), timeout=10)

    manifest = await records.acquisitions.get(started.id)
    assert manifest.status is AcquisitionStatus.FAILED
    assert manifest.volumes[0].status is VolumeStatus.FAILED
    assert manifest.volumes[0].datasets["gfp"].status is DatasetStatus.PARTIAL
    assert await instrument._hal.stage.scanning_axis.props.get_value("scan_mode") == "IDLE"
    assert await instrument._hal.recorders["scan_recorder"].get_state() == "idle"
    assert await instrument._hal.signal_generators["daq"].get_state() != "running"
    laser = instrument._hal.lasers["laser_488"]
    assert not (await laser.props.get("is_output_enabled")).ok["is_output_enabled"].value
    assert writer.batches == [2]


async def test_preview_on_a_continuous_scan_instrument_runs_the_periodic_cycle(
    opened_instrument: Instrument, monkeypatch: pytest.MonkeyPatch
) -> None:
    instrument = opened_instrument
    divisions = record_paced_starts(instrument, monkeypatch)

    await instrument.start_preview()
    try:
        assert await instrument._hal.signal_generators["daq"].get_state() == "running"
    finally:
        await instrument.stop_preview()

    assert divisions == [], "preview never waits for encoder edges"


@pytest.mark.parametrize("scan_options", [{"markers": {"camera_2": "pfi0"}}], indirect=True)
async def test_open_refuses_a_continuous_scan_whose_recorder_does_not_mark_every_camera(
    instrument: Instrument,
) -> None:
    with pytest.raises(StartupError, match="images camera_1 without a recorder marker"):
        await instrument.open()


async def test_stopping_a_continuous_acquisition_mid_pass_halts_and_restores_the_scan_axis(
    opened_instrument: Instrument, records: SQLiteRecords, writer: _Writer
) -> None:
    instrument = opened_instrument
    await add_task(instrument, end=5000.0)  # a 4.3 mm pass at 1 mm/s
    z_axis = instrument._hal.stage.scanning_axis
    started = await instrument.start_acquisition(AcquisitionRequest(storage=STORAGE))

    async def scanning() -> None:
        while True:
            if await z_axis.props.get_value("scan_mode") == "CONTINUOUS" and await z_axis.is_moving():
                return
            await asyncio.sleep(0.02)

    await asyncio.wait_for(scanning(), timeout=10)
    await instrument.stop_acquisition()

    manifest = await records.acquisitions.get(started.id)
    assert manifest.status is AcquisitionStatus.CANCELLED
    assert manifest.volumes[0].datasets["gfp"].status is DatasetStatus.PARTIAL
    assert await z_axis.props.get_value("scan_mode") == "IDLE"
    assert not await z_axis.is_moving()
    assert await z_axis.get_position() < 5000.0, "the pass halts instead of running to its end"
    assert writer.batches == [51]
