import asyncio
import sys
import time
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import numpy as np
import pytest
from vxlib.vector import IVec2D, Vec2D

from rigup import DeviceConfig, Rig, RigConfig
from vxl.devices.camera.base import SensorROI, TriggerMode, TriggerPolarity
from vxl.devices.camera.handle import CameraHandle
from vxl.preview import PreviewFrame

pytest.importorskip("fcntl", reason="PVCAM interface claims use POSIX flock")

from vxl_drivers.cameras import teledyne

SERIAL = "KX00000001"
INTERNAL = teledyne.EXT_TRIG_INTERNAL
EDGE = teledyne.EXT_TRIG_EDGE_RISING
KINETIX_EXPOSURE_MODES = {
    "Internal Trigger": INTERNAL,
    "Trigger first": 2048,
    "Edge Trigger": EDGE,
    "Level Trigger": 2560,
}
KINETIX_EXPOSE_OUT_MODES = {
    "First Row": 0,
    "All Rows": 1,
    "Any Row": 2,
    "Rolling Shutter": 3,
    "Line Output": 4,
}


def kinetix_table() -> dict[str, dict]:
    """PyVCAM ``port_speed_gain_table`` for the Kinetix ports, deliberately out of index order."""

    def port(value: int, bit_depth: int, gains: dict[str, int]) -> dict:
        speed: dict = {"speed_index": 0, "pixel_time": 10, "gain_range": sorted(gains.values())}
        for gain, index in gains.items():
            speed[gain] = {"gain_index": index, "bit_depth": bit_depth}
        return {"port_value": value, "Speed_0": speed}

    return {
        "Dynamic Range": port(2, 16, {"High": 2, "Standard": 1}),
        "Sensitivity": port(0, 12, {"Standard": 1}),
        "Speed": port(1, 8, {"Standard": 1}),
        "Sub-Electron": port(3, 16, {"Standard": 1}),
    }


class FakeDevice:
    """Stateful stand-in for PyvcamDevice with PVCAM's live-acquisition rules."""

    def __init__(
        self,
        serial: str = SERIAL,
        name: str = "pvcamPCIE_0",
        sensor: tuple[int, int] = (64, 48),
        pixel_size: tuple[float, float] | None = (6.5, 6.5),
        table: dict[str, dict] | None = None,
        binnings: tuple[int, ...] = (1, 2, 4),
        exposure_range: tuple[int, int] = (10, 10_000_000),
        exposure_modes: dict[str, int] | None = None,
    ) -> None:
        self.name = name
        self.identity = teledyne.PvcamIdentity(
            serial=serial,
            product="Kinetix",
            chip="KINETIX",
            firmware="1.5",
            pvcam_version="3.10.3",
        )
        self.sensor_size_px = sensor
        self.pixel_size_um = pixel_size
        self.port_speed_gain_table = table or kinetix_table()
        self.binning_options = binnings
        self.exposure_range_us = exposure_range
        self.exposure_modes = KINETIX_EXPOSURE_MODES if exposure_modes is None else exposure_modes
        self.expose_out_modes = dict(KINETIX_EXPOSE_OUT_MODES)
        self._bit_depths = {
            (mode.port, mode.speed, gain.index): gain.bit_depth
            for mode in teledyne.readout_modes(self.port_speed_gain_table).values()
            for gain in mode.gains
        }

        self._readout = (0, 0, 1)
        self.geometry = (0, 0, sensor[0], sensor[1], 1)
        self.exposure_us = 0
        self.exposure_mode = INTERNAL
        self._expose_out_mode = 0
        self.live = False
        self.live_settings: list[dict[str, object]] = []
        self.buffer_counts: list[int] = []
        self.finish_calls = 0
        self.close_calls = 0
        self.delivered = 0
        self.missed_frames = 0
        self.fail_poll_with: Exception | None = None
        self.fill_value = 17
        self._ring_slot: np.ndarray | None = None

    def readout(self) -> tuple[int, int, int]:
        return self._readout

    def set_readout(self, port: int, speed: int, gain: int) -> None:
        if self.live:
            raise RuntimeError("PL_ERR_ACQUISITION_IN_PROGRESS")
        if (port, speed, gain) not in self._bit_depths:
            raise ValueError(f"invalid readout {(port, speed, gain)}")
        self._readout = (port, speed, gain)

    def bit_depth_host(self) -> int:
        return self._bit_depths[self._readout]

    def set_geometry(self, x: int, y: int, w: int, h: int, binning: int) -> None:
        self.geometry = (x, y, w, h, binning)

    def set_exposure_us(self, exposure_us: int) -> None:
        minimum, maximum = self.exposure_range_us
        if not minimum <= exposure_us <= maximum:
            raise ValueError(f"exposure {exposure_us} outside {minimum}..{maximum}")
        self.exposure_us = exposure_us

    def set_exposure_mode(self, mode: int) -> None:
        if self.live:
            raise RuntimeError("PL_ERR_ACQUISITION_IN_PROGRESS")
        if mode not in self.exposure_modes.values():
            raise ValueError(f"invalid exposure mode {mode}")
        self.exposure_mode = mode

    def expose_out_mode(self) -> int:
        return self._expose_out_mode

    def set_expose_out_mode(self, mode: int) -> None:
        if mode not in self.expose_out_modes.values():
            raise ValueError(f"invalid expose-out mode {mode}")
        self._expose_out_mode = mode

    def temperature_c(self) -> float | None:
        return -19.9

    def start_live(self, buffer_frame_count: int) -> None:
        if self.live:
            raise RuntimeError("already live")
        self.live = True
        self.delivered = 0
        self.buffer_counts.append(buffer_frame_count)
        self.live_settings.append(
            {
                "readout": self._readout,
                "geometry": self.geometry,
                "exposure_us": self.exposure_us,
                "exposure_mode": self.exposure_mode,
                "expose_out_mode": self._expose_out_mode,
            }
        )

    def poll_frame(self, timeout_ms: int) -> tuple[np.ndarray, int, float]:
        assert timeout_ms > 0
        if not self.live:
            raise RuntimeError("Acquisition not active.")
        # A real camera blocks between frames. Yielding here prevents the preview loop from
        # monopolizing rigup's worker executor and makes the integration test deterministic.
        time.sleep(0.001)
        if self.fail_poll_with is not None:
            error, self.fail_poll_with = self.fail_poll_with, None
            raise error
        self.delivered += 1 + self.missed_frames
        self.missed_frames = 0
        _x, _y, w, h, b = self.geometry
        shape = (h // b, w // b)
        dtype = np.dtype(np.uint8 if self.bit_depth_host() == 8 else np.uint16)
        # Like PyVCAM with copyData=False: a view of memory the camera reuses for later frames.
        slot = self._ring_slot
        if slot is None or slot.shape != shape or slot.dtype != dtype:
            slot = self._ring_slot = np.empty(shape, dtype=dtype)
        slot.fill(self.fill_value)
        return slot, self.delivered, 42.0

    def finish(self) -> None:
        self.live = False
        self.finish_calls += 1

    def close(self) -> None:
        self.live = False
        self.close_calls += 1


def install_devices(
    monkeypatch: pytest.MonkeyPatch,
    *devices: FakeDevice,
    open_errors: dict[str, Exception] | None = None,
) -> None:
    by_name = {device.name: device for device in devices}
    names = list(by_name) + list(open_errors or {})
    monkeypatch.setattr(teledyne, "list_pvcam_camera_names", lambda: names)

    def open_device(name: str) -> FakeDevice:
        if open_errors and name in open_errors:
            raise open_errors[name]
        return by_name[name]

    monkeypatch.setattr(teledyne, "open_pvcam_device", open_device)


def install_pyvcam(monkeypatch: pytest.MonkeyPatch, claim_dir: Path, serials: dict[str, str]) -> SimpleNamespace:
    """Fake the PyVCAM package so the real ``open_pvcam_device`` and ``PyvcamDevice`` run.

    Heads report ``serials`` by interface name. The returned state records native opens and
    closes, names whose native open (after acquiring the handle) or close should fail, and the
    exposure resolutions that opened heads report.
    """
    monkeypatch.setattr(teledyne, "PVCAM_CLAIM_DIR", claim_dir)
    monkeypatch.setattr(teledyne, "pvcam_runtime", lambda: SimpleNamespace(get_pvcam_version=lambda: "3.10.3"))
    state = SimpleNamespace(
        opened=[],
        closed=[],
        failing_open=set(),
        failing_close=set(),
        exp_resolutions={"One Microsecond": 1},
    )

    class Camera:
        def __init__(self, name: str) -> None:
            self.name = name
            self.is_open = False
            self.serial_no = serials[name]
            self.chip_name = "KINETIX"
            self.cam_fw = "1.5"
            self.sensor_size = (64, 48)
            self.port_speed_gain_table: dict[str, object] = {}
            self.binnings = [(1, 1)]
            self.exp_modes: dict[str, int] = {}
            self.exp_out_modes: dict[str, int] = {}
            self.exp_resolutions = dict(state.exp_resolutions)
            self.exp_res = 0

        def open(self) -> None:
            state.opened.append(self.name)
            self.is_open = True
            if self.name in state.failing_open:
                raise RuntimeError("capability read failed")

        def check_param(self, _param_id: int) -> bool:
            return False

        def get_param(self, _param_id: int, attribute: int) -> int:
            # Kinetix reports a zero exposure minimum.
            return 0 if attribute == constants.ATTR_MIN else 10_000

        def close(self) -> None:
            if self.name in state.failing_close:
                raise RuntimeError("native close failed")
            state.closed.append(self.name)
            self.is_open = False

    camera_module = ModuleType("pyvcam.camera")
    vars(camera_module).update(Camera=Camera)
    constants = ModuleType("pyvcam.constants")
    vars(constants).update(
        PARAM_PRODUCT_NAME=1,
        PARAM_PIX_SER_SIZE=2,
        PARAM_PIX_PAR_SIZE=3,
        PARAM_EXPOSURE_TIME=4,
        ATTR_MIN=1,
        ATTR_MAX=2,
        EXP_RES_ONE_MILLISEC=0,
        EXP_RES_ONE_MICROSEC=1,
    )
    monkeypatch.setitem(sys.modules, "pyvcam.camera", camera_module)
    monkeypatch.setitem(sys.modules, "pyvcam.constants", constants)
    return state


def test_open_pvcam_device_claimed_head_stays_unopened_until_its_holder_closes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    pyvcam = install_pyvcam(monkeypatch, tmp_path, {"pvcamPCIE_0": SERIAL})

    holder = teledyne.open_pvcam_device("pvcamPCIE_0")
    with pytest.raises(teledyne.PvcamClaimedError, match="pvcamPCIE_0"):
        teledyne.open_pvcam_device("pvcamPCIE_0")
    assert pyvcam.opened == ["pvcamPCIE_0"], "a claimed head must never get a second native handle"

    pyvcam.failing_close.add("pvcamPCIE_0")
    with pytest.raises(RuntimeError, match="native close failed"):
        holder.close()
    with pytest.raises(teledyne.PvcamClaimedError):
        teledyne.open_pvcam_device("pvcamPCIE_0")

    pyvcam.failing_close.clear()
    holder.close()
    reopened = teledyne.open_pvcam_device("pvcamPCIE_0")
    assert reopened.identity.serial == SERIAL
    reopened.close()


def test_open_pvcam_device_needs_us_or_ms_exposure_resolution_and_raises_zero_minimum_to_one_unit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    pyvcam = install_pyvcam(monkeypatch, tmp_path, {"pvcamPCIE_0": SERIAL})
    pyvcam.exp_resolutions = {"One Second": 2}

    with pytest.raises(teledyne.KinetixDiscoveryError, match="exposure resolution"):
        teledyne.open_pvcam_device("pvcamPCIE_0")

    for resolutions, scale_us in (({"One Microsecond": 1}, 1), ({"One Millisecond": 0}, 1000)):
        pyvcam.exp_resolutions = resolutions
        device = teledyne.open_pvcam_device("pvcamPCIE_0")
        assert device.exposure_range_us == (scale_us, 10_000 * scale_us)
        device.close()
    assert pyvcam.opened == ["pvcamPCIE_0"] * 3, "the failed open released its claim"


def test_open_pvcam_device_closes_handle_when_pyvcam_open_partially_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    pyvcam = install_pyvcam(monkeypatch, tmp_path, {"pvcamPCIE_0": SERIAL})
    pyvcam.failing_open.add("pvcamPCIE_0")

    with pytest.raises(RuntimeError, match="capability read failed"):
        teledyne.open_pvcam_device("pvcamPCIE_0")

    assert pyvcam.closed == ["pvcamPCIE_0"]
    # The failed open released its claim, so a retry reaches PyVCAM instead of PvcamClaimedError.
    with pytest.raises(RuntimeError, match="capability read failed"):
        teledyne.open_pvcam_device("pvcamPCIE_0")


@pytest.fixture
def camera(monkeypatch: pytest.MonkeyPatch) -> tuple[teledyne.Kinetix, FakeDevice]:
    device = FakeDevice()
    install_devices(monkeypatch, device)
    return teledyne.Kinetix(uid="camera_1", serial=SERIAL), device


def test_readout_modes_orders_pyvcam_table_and_labels_multi_speed_ports() -> None:
    modes = teledyne.readout_modes(kinetix_table())

    assert list(modes) == ["Sensitivity", "Speed", "Dynamic Range", "Sub-Electron"]
    assert modes["Dynamic Range"].gains == (
        teledyne.GainInfo(1, "Standard", 16),
        teledyne.GainInfo(2, "High", 16),
    )
    assert (modes["Speed"].port, modes["Speed"].speed) == (1, 0)

    gain = {"g": {"gain_index": 1, "bit_depth": 16}}
    port = {
        "port_value": 0,
        "100 MHz": {"speed_index": 1, **gain},
        "200 MHz": {"speed_index": 0, **gain},
    }
    assert list(teledyne.readout_modes({"Port": port})) == ["Port @ 200 MHz", "Port @ 100 MHz"]


def test_resolve_trigger_modes_prefers_extended_values_and_falls_back_to_legacy() -> None:
    assert teledyne.resolve_trigger_modes(KINETIX_EXPOSURE_MODES) == {
        TriggerMode.OFF: INTERNAL,
        TriggerMode.ON: EDGE,
    }
    legacy = teledyne.resolve_trigger_modes({"Timed": teledyne.TIMED_MODE, "Strobed": teledyne.STROBED_MODE})
    assert legacy == {TriggerMode.OFF: teledyne.TIMED_MODE, TriggerMode.ON: teledyne.STROBED_MODE}
    with pytest.raises(teledyne.KinetixDiscoveryError, match="internal and edge"):
        teledyne.resolve_trigger_modes({"Internal Trigger": INTERNAL})


def test_roi_grid_step_covers_every_binning_factor(monkeypatch: pytest.MonkeyPatch) -> None:
    install_devices(monkeypatch, FakeDevice(binnings=(1, 4)))

    camera = teledyne.Kinetix(uid="camera_1", serial=SERIAL, roi_step_px=6)

    assert (camera.roi_grid.h.step, camera.roi_grid.v.step) == (12, 12)


def test_buffer_frame_count_honors_budget_and_minimum() -> None:
    assert teledyne.buffer_frame_count(3200 * 3200 * 2, 2400) == 117
    assert teledyne.buffer_frame_count(3200 * 3200 * 2, 1) == teledyne.MIN_BUFFER_FRAMES


def test_ensure_pvcam_environment_fills_missing_variables_only(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "drivers" / "user-mode").mkdir(parents=True)
    (tmp_path / "pvcam.version").write_text("3.10.3.6\n")
    monkeypatch.delenv("PVCAM_UMD_PATH", raising=False)
    monkeypatch.setenv("PVCAM_VERSION", "keep-me")

    teledyne.ensure_pvcam_environment(tmp_path)

    assert Path(teledyne.os.environ["PVCAM_UMD_PATH"]) == tmp_path / "drivers" / "user-mode"
    assert teledyne.os.environ["PVCAM_VERSION"] == "keep-me"


def test_discover_camera_serial_mismatch_lists_candidates_and_closes_them(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = FakeDevice(serial="KX00000002", name="pvcamPCIE_0")
    second = FakeDevice(serial="OTHER", name="pvcamPCIE_1")
    install_devices(monkeypatch, first, second)

    with pytest.raises(teledyne.KinetixDiscoveryError, match=r"KX00000002.*OTHER"):
        teledyne.discover_camera(SERIAL)

    assert first.close_calls == 1
    assert second.close_calls == 1


def test_discover_camera_open_failure_reports_pvcam_cause_and_unpowered_head(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # PyVCAM raises a generic "Failed to open camera." chained from the PVCAM error.
    error = RuntimeError("Failed to open camera.")
    error.__cause__ = RuntimeError("Driver device failed to write data (PL_ERR_DDI_DEVICE_WRITE_FAILED)")
    install_devices(monkeypatch, open_errors={"pvcamPCIE_0": error})

    with pytest.raises(
        teledyne.KinetixDiscoveryError,
        match=r"Failed to open camera: Driver device failed.*DEVICE_WRITE_FAILED.*camera power",
    ):
        teledyne.discover_camera(SERIAL)


def test_discover_camera_skips_claimed_heads_and_names_them_when_serial_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    other = FakeDevice(serial="KX00000002", name="pvcamPCIE_1")
    claimed = teledyne.PvcamClaimedError("PVCAM interface pvcamPCIE_0 is in use")
    install_devices(monkeypatch, other, open_errors={"pvcamPCIE_0": claimed})
    # The claimed head enumerates first, so finding the free head proves discovery moved past it.
    monkeypatch.setattr(teledyne, "list_pvcam_camera_names", lambda: ["pvcamPCIE_0", "pvcamPCIE_1"])

    assert teledyne.discover_camera("KX00000002") is other

    with pytest.raises(teledyne.KinetixDiscoveryError) as raised:
        teledyne.discover_camera(SERIAL)
    message = str(raised.value)
    assert "in use by another Kinetix or process: pvcamPCIE_0" in message
    assert "camera power" not in message, "a held head is not a power or cabling fault"


def test_discover_camera_without_cameras_mentions_device_node(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_devices(monkeypatch)

    with pytest.raises(teledyne.KinetixDiscoveryError, match="pvcamPCIE"):
        teledyne.discover_camera(SERIAL)


def test_constructor_reads_identity_geometry_and_applies_defaults(
    camera: tuple[teledyne.Kinetix, FakeDevice],
) -> None:
    device_camera, device = camera

    assert device_camera.serial == SERIAL
    assert "Kinetix serial KX00000001" in device_camera.identity
    assert device_camera.sensor_size_px == IVec2D(x=64, y=48)
    assert device_camera.pixel_size_um == Vec2D(x=6.5, y=6.5)
    assert device_camera.sensor_mode == "Sensitivity"
    assert device_camera.gain == "Standard"
    assert device_camera.pixel_format == "MONO12"
    assert device_camera.binning == 1
    assert device_camera.roi == SensorROI(x=0, y=0, w=64, h=48)
    assert device_camera.exposure_time_ms == 10.0
    assert device.exposure_us == 10_000
    assert device.geometry == (0, 0, 64, 48, 1)
    assert device.exposure_mode == INTERNAL
    assert device_camera.stream_info is None
    assert device_camera.sensor_temperature_c == -19.9


def test_constructor_pixel_size_override_beats_camera_report(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    device = FakeDevice(pixel_size=None)
    install_devices(monkeypatch, device)

    # Configuration uses Voxel's (y, x) order; the device reports (x, y).
    for configured_value in ("4.0,3.0", [4.0, 3.0]):
        configured = teledyne.Kinetix(uid="camera_1", serial=SERIAL, pixel_size_um=configured_value)
        assert configured.pixel_size_um == Vec2D(x=3.0, y=4.0), configured_value

    device.pixel_size_um = (3.0, 4.0)
    reported = teledyne.Kinetix(uid="camera_1", serial=SERIAL)
    assert reported.pixel_size_um == Vec2D(x=3.0, y=4.0)


ODD_BIT_DEPTH_TABLE = {"Odd": {"port_value": 0, "Odd": {"speed_index": 0, "g": {"gain_index": 1, "bit_depth": 24}}}}


@pytest.mark.parametrize(
    ("device_kwargs", "error", "match"),
    [
        ({"pixel_size": None}, ValueError, "configure pixel_size_um"),
        ({"table": ODD_BIT_DEPTH_TABLE}, ValueError, "bit depth 24"),
        ({"binnings": (2, 4)}, teledyne.KinetixDiscoveryError, "required 1x binning"),
    ],
)
def test_constructor_unusable_camera_raises_and_closes_device(
    monkeypatch: pytest.MonkeyPatch,
    device_kwargs: dict[str, Any],
    error: type[Exception],
    match: str,
) -> None:
    device = FakeDevice(**device_kwargs)
    install_devices(monkeypatch, device)

    with pytest.raises(error, match=match):
        teledyne.Kinetix(uid="camera_1", serial=SERIAL)

    assert device.close_calls == 1


@pytest.mark.parametrize(
    "kwargs",
    [
        {"roi_step_px": 0},
        {"buffer_size_mb": float("inf")},
        {"frame_timeout_s": 0},
        {"serial": "  "},
    ],
)
def test_constructor_invalid_options_rejected_before_enumeration(monkeypatch: pytest.MonkeyPatch, kwargs: dict) -> None:
    monkeypatch.setattr(teledyne, "list_pvcam_camera_names", lambda: pytest.fail("enumerated"))

    with pytest.raises(ValueError, match="must"):
        teledyne.Kinetix(uid="camera_1", **{"serial": SERIAL, **kwargs})


def test_exposure_time_ms_converts_to_microseconds_and_clamps_to_readout_range(
    camera: tuple[teledyne.Kinetix, FakeDevice],
) -> None:
    device_camera, device = camera

    device_camera.exposure_time_ms = 12.3456
    assert device.exposure_us == 12_346
    assert device_camera.exposure_time_ms == pytest.approx(12.346)

    device_camera.exposure_time_ms = 1e9
    assert device.exposure_us == 10_000_000

    device_camera.exposure_time_ms = 0.0
    assert device.exposure_us == 10

    device.exposure_range_us = (20_000, 10_000_000)
    device_camera.sensor_mode = "Speed"
    assert device.exposure_us == 20_000, "a new readout mode can narrow the range"
    assert device_camera.exposure_time_ms == 20.0


def test_sensor_mode_and_gain_drive_pixel_format(
    camera: tuple[teledyne.Kinetix, FakeDevice],
) -> None:
    device_camera, device = camera

    device_camera.sensor_mode = "Speed"
    assert device.readout() == (1, 0, 1)
    assert device_camera.pixel_format == "MONO8"
    assert device_camera.pixel_type.itemsize == 1

    device_camera.sensor_mode = "Dynamic Range"
    device_camera.gain = "High"
    assert device.readout() == (2, 0, 2)
    assert device_camera.gain == "High"
    assert device_camera.pixel_format == "MONO16"
    gain: Any = device_camera.gain  # the enumerated value, not the narrowed literal
    assert list(gain.options) == ["Standard", "High"]

    device_camera.sensor_mode = "Sensitivity"
    assert device_camera.gain == "Standard", "gain falls back to the first entry when the new mode lacks it"

    device_camera.pixel_format = "MONO16"
    assert device_camera.pixel_format == "MONO12", "bit depth follows the sensor mode only"
    assert device.readout() == (0, 0, 1)


def test_update_roi_snaps_to_grid_that_covers_binning(
    camera: tuple[teledyne.Kinetix, FakeDevice],
) -> None:
    device_camera, device = camera
    grid = device_camera.roi_grid
    assert (grid.h.step, grid.v.step, grid.h.min) == (8, 8, 8)

    applied = device_camera.update_roi(SensorROI(x=3, y=5, w=30, h=20))

    assert applied == SensorROI(x=0, y=8, w=32, h=24)
    assert device.geometry == (0, 8, 32, 24, 1)

    with pytest.raises(ValueError, match="not in allowed options"):
        device_camera.binning = 8
    assert device.geometry[4] == 1

    device_camera.binning = 4
    assert device_camera.frame_size_px == IVec2D(x=8, y=6)
    assert device.geometry == (0, 8, 32, 24, 4)


def test_expose_out_mode_uses_camera_names(camera: tuple[teledyne.Kinetix, FakeDevice]) -> None:
    device_camera, device = camera

    assert device_camera.expose_out_mode == "First Row"
    device_camera.expose_out_mode = "All Rows"

    assert device.expose_out_mode() == 1
    assert device_camera.expose_out_mode == "All Rows"


def test_frame_rate_uses_exposure_estimate_then_measured_rate_and_setter_is_inert(
    camera: tuple[teledyne.Kinetix, FakeDevice],
) -> None:
    device_camera, _device = camera

    # The numeric descriptor snaps the reported rate to its 0.01 Hz grid.
    device_camera.exposure_time_ms = 7.5
    assert device_camera.frame_rate_hz == pytest.approx(1000 / 7.5, abs=0.01)

    device_camera.frame_rate_hz = 3.0
    assert device_camera.frame_rate_hz == pytest.approx(1000 / 7.5, abs=0.01)

    device_camera.start(frame_count=None, trigger_mode=TriggerMode.OFF)
    assert device_camera.frame_rate_hz == pytest.approx(1000 / 7.5, abs=0.01)
    device_camera.grab_frame_owned()
    assert device_camera.frame_rate_hz == 42.0, "measured rate while streaming"


def test_start_grab_stop_produces_owned_frames_with_internal_trigger(
    camera: tuple[teledyne.Kinetix, FakeDevice],
) -> None:
    device_camera, device = camera

    device_camera.start(frame_count=3, trigger_mode=TriggerMode.OFF)
    frame = device_camera.grab_frame_owned()

    assert device.live
    assert device.exposure_mode == INTERNAL
    assert device.buffer_counts == [teledyne.buffer_frame_count(64 * 48 * 2, teledyne.DEFAULT_BUFFER_SIZE_MB)]
    assert frame.shape == (48, 64)
    assert frame.dtype == np.uint16
    assert frame.flags.owndata
    assert np.all(frame == 17)
    info = device_camera.stream_info
    assert info is not None
    assert (info.frame_index, info.dropped_frames, info.frame_rate_fps) == (1, 0, 42.0)
    assert info.input_buffer_size == device.buffer_counts[0]

    device.fill_value = 99
    assert np.all(device_camera.grab_frame_owned() == 99)
    assert np.all(frame == 17), "a delivered frame must not alias the reused ring slot"

    device_camera.stop()
    assert not device.live
    assert device_camera.stream_info is None

    device_camera.close()
    assert device.close_calls == 1


def test_trigger_on_selects_edge_trigger_and_falling_polarity_is_rejected(
    camera: tuple[teledyne.Kinetix, FakeDevice],
) -> None:
    device_camera, device = camera

    device_camera.start(frame_count=None, trigger_mode=TriggerMode.ON, trigger_polarity=TriggerPolarity.RISING_EDGE)
    assert device.exposure_mode == EDGE
    device_camera.stop()

    with pytest.raises(ValueError, match="rising-edge only"):
        device_camera.start(
            frame_count=None,
            trigger_mode=TriggerMode.ON,
            trigger_polarity=TriggerPolarity.FALLING_EDGE,
        )
    assert not device.live


def test_settings_changed_while_streaming_apply_before_next_frame_or_at_restart(
    camera: tuple[teledyne.Kinetix, FakeDevice],
) -> None:
    device_camera, device = camera
    device_camera.start(frame_count=None, trigger_mode=TriggerMode.OFF)
    device_camera.grab_frame_owned()

    device_camera.exposure_time_ms = 2.0
    device_camera.binning = 2
    device_camera.sensor_mode = "Speed"
    assert device.live, "the setters never stop the acquisition themselves"
    assert device.exposure_us == 10_000, "nothing reaches PVCAM until the grab thread restarts"

    frame = device_camera.grab_frame_owned()

    assert device.finish_calls == 1
    assert len(device.live_settings) == 2
    assert device.live_settings[-1]["exposure_us"] == 2_000
    assert device.live_settings[-1]["geometry"] == (0, 0, 64, 48, 2)
    assert device.live_settings[-1]["readout"] == (1, 0, 1)
    assert frame.shape == (24, 32)
    assert frame.dtype == np.uint8
    info = device_camera.stream_info
    assert info is not None
    assert info.frame_index == 1, "counters restart with the acquisition"

    device_camera.exposure_time_ms = 3.0
    device_camera.stop()
    device_camera.start(frame_count=None, trigger_mode=TriggerMode.OFF)
    assert device.live_settings[-1]["exposure_us"] == 3_000, "a pending change applies at start"


def test_grab_frame_setting_change_after_validation_delivers_running_geometry_then_restarts(
    camera: tuple[teledyne.Kinetix, FakeDevice],
) -> None:
    device_camera, device = camera
    device_camera.start(frame_count=None, trigger_mode=TriggerMode.OFF)

    class BinningChangesDuringValidation(np.ndarray):
        # Camera.grab_frame reads ``flags`` after checking shape and dtype but before the
        # driver takes its lock, which is exactly where a concurrent setter can land.
        @property
        def flags(self):
            if device_camera.binning == 1:
                device_camera.binning = 2
            return super().flags

    out = np.zeros((48, 64), dtype=np.uint16).view(BinningChangesDuringValidation)
    device_camera.grab_frame(out=out)

    assert np.all(out == 17)
    assert device.finish_calls == 0, "the running ring still matched the validated destination"
    frame = device_camera.grab_frame_owned()
    assert frame.shape == (24, 32)
    assert device.finish_calls == 1


def test_grab_frame_delivery_counter_gap_raises_ring_overrun_and_keeps_streaming(
    camera: tuple[teledyne.Kinetix, FakeDevice],
) -> None:
    device_camera, device = camera
    device_camera.start(frame_count=None, trigger_mode=TriggerMode.OFF)

    device_camera.grab_frame_owned()
    device.missed_frames = 3
    with pytest.raises(teledyne.KinetixAcquisitionError, match="ring overrun"):
        device_camera.grab_frame_owned()

    info = device_camera.stream_info
    assert info is not None
    assert (info.frame_index, info.dropped_frames) == (1, 4), "the suspect frame counts as dropped"
    device_camera.grab_frame_owned()
    info = device_camera.stream_info
    assert info is not None
    assert (info.frame_index, info.dropped_frames) == (2, 4)


def test_grab_frame_timeout_raises_acquisition_error_and_keeps_streaming(
    camera: tuple[teledyne.Kinetix, FakeDevice],
) -> None:
    device_camera, device = camera
    device_camera.start(frame_count=None, trigger_mode=TriggerMode.ON)
    device.fail_poll_with = RuntimeError("Frame timeout. Verify the timeout exceeds the exposure time.")

    with pytest.raises(teledyne.KinetixAcquisitionError, match="Frame timeout"):
        device_camera.grab_frame_owned()

    assert device.live
    assert device_camera.stream_info is not None
    device_camera.grab_frame_owned()
    device_camera.stop()
    assert device.finish_calls == 1


def test_grab_frame_before_start_and_double_start_are_rejected(
    camera: tuple[teledyne.Kinetix, FakeDevice],
) -> None:
    device_camera, _device = camera

    with pytest.raises(teledyne.KinetixAcquisitionError, match="not acquiring"):
        device_camera.grab_frame_owned()

    device_camera.start(frame_count=None, trigger_mode=TriggerMode.OFF)
    with pytest.raises(teledyne.KinetixAcquisitionError, match="already acquiring"):
        device_camera.start(frame_count=None, trigger_mode=TriggerMode.OFF)


def test_close_stops_acquisition_retries_after_native_close_failure_and_is_idempotent(
    camera: tuple[teledyne.Kinetix, FakeDevice], monkeypatch: pytest.MonkeyPatch
) -> None:
    device_camera, device = camera
    device_camera.start(frame_count=None, trigger_mode=TriggerMode.OFF)
    native_close = device.close
    attempts = 0

    def flaky_close() -> None:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("native close failed")
        native_close()

    monkeypatch.setattr(device, "close", flaky_close)

    with pytest.raises(RuntimeError, match="native close failed"):
        device_camera.close()
    assert not device.live
    device_camera.close()
    device_camera.close()

    assert attempts == 2
    assert device.finish_calls == 1
    assert device.close_calls == 1


def test_frame_timeout_scales_with_long_exposures(
    camera: tuple[teledyne.Kinetix, FakeDevice],
) -> None:
    device_camera, _device = camera

    assert device_camera._frame_timeout_ms() == 10_000
    device_camera.exposure_time_ms = 8_000.0
    assert device_camera._frame_timeout_ms() == 17_000


@pytest.mark.parametrize(
    ("preview_config", "expected_exposure_mode"),
    [
        ({}, EDGE),
        ({"preview_trigger_mode": "off"}, INTERNAL),
        ({"preview_trigger_mode": False}, INTERNAL),  # YAML 1.1 reads unquoted off as False
    ],
)
def test_driver_preview_trigger_configuration_publishes_decodable_frames(
    monkeypatch: pytest.MonkeyPatch,
    preview_config: dict[str, str | bool],
    expected_exposure_mode: int,
) -> None:
    device = FakeDevice()
    install_devices(monkeypatch, device)
    config = RigConfig(
        devices={
            "camera_1": DeviceConfig(
                target="vxl_drivers.cameras.teledyne.Kinetix",
                init={"serial": SERIAL, **preview_config},
            ),
        }
    )

    async def run() -> None:
        rig = Rig(config, name="kinetix-test")
        await rig.open()
        try:
            assert not rig.build_errors, f"device build failed: {rig.build_errors}"
            handle = CameraHandle.wrap(rig.devices["camera_1"])
            assert await handle.sensor_size_px.get() == IVec2D(x=64, y=48)
            assert await handle.props.get_value("pixel_format") == "MONO12"
            assert await handle.props.get_value("sensor_mode") == "Sensitivity"
            packets: list[bytes] = []

            async def receive(packet: bytes) -> None:
                packets.append(packet)

            unsubscribe = handle.subscribe("preview", receive)
            try:
                assert await handle.start_preview() == "preview"
                assert device.exposure_mode == expected_exposure_mode
                for _ in range(200):
                    if packets:
                        break
                    await asyncio.sleep(0.01)
                else:
                    pytest.fail("preview published no overview frame")

                frame = PreviewFrame.from_packed(packets[-1])
                assert frame.header.camera_id == "camera_1"
                assert frame.decode().shape == (48, 64)
                await handle.stop_preview()
                assert not device.live
            finally:
                unsubscribe()
        finally:
            await rig.close()

    # Sync wrapper on purpose: this project declares no pytest-asyncio configuration.
    asyncio.run(asyncio.wait_for(run(), timeout=5.0))
    assert device.close_calls == 1


def test_controller_preview_recovers_from_a_stale_acquisition(
    camera: tuple[teledyne.Kinetix, FakeDevice],
) -> None:
    """A failed preview loop leaves the device live; the next start_preview must still succeed."""
    device_camera, device = camera
    controller = teledyne.KinetixCameraController(device_camera)

    async def run() -> None:
        device_camera.start(frame_count=None)  # what a preview loop error leaves behind
        assert device.live
        assert await controller.start_preview() == "preview"
        await controller.stop_preview()
        assert not device.live

    asyncio.run(asyncio.wait_for(run(), timeout=5.0))
