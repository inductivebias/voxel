"""Teledyne Photometrics Kinetix camera driver on PVCAM through the PyVCAM wrapper."""

import fcntl
import math
import os
import re
import tempfile
import threading
from collections.abc import Mapping
from contextlib import suppress
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from types import ModuleType
from typing import Any, cast

import numpy as np
from vxl_drivers.utils import thread_safe_singleton
from vxlib.vector import IVec2D, Vec2D

from rigup import describe, enumerated, numeric, numeric_int
from vxl.devices.camera.base import (
    PIXEL_FMT_TO_DTYPE,
    Camera,
    CameraController,
    CameraMode,
    IntRange,
    PixelFormat,
    ROIGrid,
    SensorROI,
    StreamInfo,
    TriggerMode,
    TriggerPolarity,
)

PVCAM_ROOT = Path("/opt/pvcam")
DEFAULT_EXPOSURE_US = 10_000
# Twelve full 3200 x 3200 uint16 Kinetix frames fit in this budget. Larger buffers are
# configurable, but should not make the first acquisition allocate multiple gigabytes.
DEFAULT_BUFFER_SIZE_MB = 256
DEFAULT_FRAME_TIMEOUT_S = 10.0
# PVCAM does not publish an ROI alignment step. Eight pixels keeps every ROI divisible by every
# Voxel binning factor; the real Kinetix constraint must be confirmed on hardware.
DEFAULT_ROI_STEP_PX = 8
MIN_BUFFER_FRAMES = 2
# Interface claims must be visible to every process and user on the host: /run/lock is shared and
# unaffected by systemd PrivateTmp. The temp dir is only a fallback where it does not exist (macOS).
PVCAM_CLAIM_DIR = Path("/run/lock") if Path("/run/lock").is_dir() else Path(tempfile.gettempdir())

# PVCAM exposure modes (pvcam.h ``PL_EXPOSURE_MODES``): the legacy values and the extended
# trigger values that current cameras report through PARAM_EXPOSURE_MODE.
TIMED_MODE = 0
STROBED_MODE = 1
EXT_TRIG_INTERNAL = (7 + 0) << 8
EXT_TRIG_EDGE_RISING = (7 + 2) << 8


def _coerce_trigger_mode(value: TriggerMode | str | bool | None) -> TriggerMode | None:
    if value is None:
        return None
    if isinstance(value, bool):  # YAML 1.1 reads an unquoted off/on as a boolean
        return TriggerMode.ON if value else TriggerMode.OFF
    return TriggerMode(value)


class KinetixDiscoveryError(ConnectionError):
    """The configured camera could not be enumerated, opened, or matched by serial number."""


class PvcamClaimedError(KinetixDiscoveryError):
    """Another Kinetix or process holds the PVCAM interface, so it was left unopened."""


class KinetixAcquisitionError(RuntimeError):
    """PVCAM did not deliver a frame or reported a readout failure."""


class KinetixCameraController(CameraController):
    """Apply an optional device-specific preview trigger mode."""

    @describe(label="Start Preview")
    async def start_preview(
        self,
        trigger_mode: TriggerMode = TriggerMode.ON,
        trigger_polarity: TriggerPolarity = TriggerPolarity.RISING_EDGE,
    ) -> str:
        camera = cast("Kinetix", self.device)
        if self.mode is CameraMode.IDLE:
            # Voxel's preview loop drops to IDLE on a frame error without stopping the device,
            # which would reject every later start as "already acquiring". stop() is a no-op
            # unless that stale acquisition is still live.
            await self._run_sync(camera.stop)
        return await super().start_preview(
            camera.preview_trigger_mode or trigger_mode,
            trigger_polarity,
        )


@dataclass(frozen=True)
class PvcamIdentity:
    serial: str
    product: str
    chip: str
    firmware: str
    pvcam_version: str


@dataclass(frozen=True)
class GainInfo:
    index: int
    name: str
    bit_depth: int


@dataclass(frozen=True)
class ReadoutMode:
    """One selectable sensor mode: a PVCAM readout port at one speed-table entry."""

    label: str
    port: int
    speed: int
    gains: tuple[GainInfo, ...]

    def gain_named(self, name: str) -> GainInfo:
        for gain in self.gains:
            if gain.name == name:
                return gain
        # Reachable when a concurrent sensor-mode change replaces the gain options mid-set.
        raise ValueError(f"gain {name!r} is not available in sensor mode {self.label!r}; options: {self.gain_names}")

    def gain_indexed(self, index: int) -> GainInfo | None:
        return next((gain for gain in self.gains if gain.index == index), None)

    @property
    def gain_names(self) -> list[str]:
        return [gain.name for gain in self.gains]


def pixel_format_for_bit_depth(bit_depth: int) -> PixelFormat:
    """Map a PVCAM host bit depth to the Voxel pixel format that carries it."""
    pixel_format = f"MONO{bit_depth}"
    if pixel_format not in PIXEL_FMT_TO_DTYPE:
        raise ValueError(f"unsupported PVCAM host bit depth {bit_depth}")
    return cast("PixelFormat", pixel_format)


def readout_modes(table: Mapping[str, Any]) -> dict[str, ReadoutMode]:
    """Flatten PyVCAM's ``port_speed_gain_table`` into sensor modes ordered by port and speed.

    A port with one speed keeps the port name (Kinetix: ``Sensitivity``, ``Speed``,
    ``Dynamic Range``, ``Sub-Electron``); ports with several speeds get ``"<port> @ <speed>"``.
    """
    modes: dict[str, ReadoutMode] = {}
    for port_name, port in sorted(table.items(), key=lambda item: item[1]["port_value"]):
        speeds = sorted(
            ((name, speed) for name, speed in port.items() if name != "port_value"),
            key=lambda item: item[1]["speed_index"],
        )
        for speed_name, speed in speeds:
            label = port_name if len(speeds) == 1 else f"{port_name} @ {speed_name}"
            # Gains are the dict-valued entries; the others are speed attributes (pixel_time, ...).
            gains = sorted(
                (
                    GainInfo(int(gain["gain_index"]), str(name), int(gain["bit_depth"]))
                    for name, gain in speed.items()
                    if isinstance(gain, dict)
                ),
                key=lambda gain: gain.index,
            )
            if label in modes or not gains:
                raise ValueError(f"PVCAM speed table has a duplicate or gain-less entry for {label!r}")
            modes[label] = ReadoutMode(label, int(port["port_value"]), int(speed["speed_index"]), tuple(gains))
    if not modes:
        raise ValueError("PVCAM speed table reported no readout ports")
    return modes


def resolve_trigger_modes(exposure_modes: Mapping[str, int]) -> dict[TriggerMode, int]:
    """Choose the PVCAM exposure mode for Voxel's internal (OFF) and per-frame edge (ON) triggering."""
    values = set(exposure_modes.values())
    internal = next((mode for mode in (EXT_TRIG_INTERNAL, TIMED_MODE) if mode in values), None)
    external = next((mode for mode in (EXT_TRIG_EDGE_RISING, STROBED_MODE) if mode in values), None)
    if internal is None or external is None:
        raise KinetixDiscoveryError(
            f"camera exposure modes {dict(exposure_modes)} do not include internal and edge triggering"
        )
    return {TriggerMode.OFF: internal, TriggerMode.ON: external}


def buffer_frame_count(frame_bytes: int, buffer_size_mb: float) -> int:
    """Number of ring-buffer frames that fit the configured budget, never below two."""
    return max(MIN_BUFFER_FRAMES, int(buffer_size_mb * 1_000_000 // frame_bytes))


def ensure_pvcam_environment(root: Path = PVCAM_ROOT) -> None:
    """Provide PVCAM's runtime variables when the process did not source ``/etc/profile.d/pvcam.sh``.

    libpvcam refuses to initialize (``PL_ERR_INSTALLATION_CORRUPTED``) without ``PVCAM_UMD_PATH``
    and ``PVCAM_VERSION``; services and non-login shells do not inherit the profile script.
    Existing values are never overwritten.
    """
    user_mode_drivers = root / "drivers" / "user-mode"
    if "PVCAM_UMD_PATH" not in os.environ and user_mode_drivers.is_dir():
        os.environ["PVCAM_UMD_PATH"] = str(user_mode_drivers)
    version_file = root / "pvcam.version"
    if "PVCAM_VERSION" not in os.environ and version_file.is_file():
        version = version_file.read_text(encoding="ascii").strip()
        if version:
            os.environ["PVCAM_VERSION"] = version


@thread_safe_singleton
def pvcam_runtime() -> ModuleType:
    """Initialize PVCAM once per process; it stays initialized for the process lifetime."""
    ensure_pvcam_environment()
    try:
        # The native extension is absent on macOS; importing the driver must remain hardware-free.
        pvc = import_module("pyvcam.pvc")
    except ImportError as error:
        raise KinetixDiscoveryError(
            "PyVCAM is not importable; it is installed only on Linux/Windows and needs the PVCAM runtime"
        ) from error
    pvc.init_pvcam()
    return pvc


def list_pvcam_camera_names() -> list[str]:
    """Names PVCAM assigns to attached interfaces, e.g. ``pvcamPCIE_0``."""
    pvc = pvcam_runtime()
    return [pvc.get_cam_name(index) for index in range(pvc.get_cam_total())]


def _open_claim_file(path: Path) -> int:
    try:
        # Plain open first: Linux's default fs.protected_regular refuses O_CREAT on a file that
        # another user already created in a sticky directory such as /run/lock.
        return os.open(path, os.O_RDONLY)
    except FileNotFoundError:
        return os.open(path, os.O_RDONLY | os.O_CREAT, 0o644)


# Discovery has to open a head to learn its serial. Opening a head that another Kinetix is
# streaming, then closing that second handle, is not documented as safe in PVCAM, so every opener
# (a second Kinetix, another rigup node, a diagnostic tool) must skip a claimed head instead.
class PvcamClaim:
    """Exclusive host-wide advisory lock on one PVCAM interface; the kernel drops it on exit."""

    def __init__(self, name: str) -> None:
        path = PVCAM_CLAIM_DIR / f"voxel-pvcam-{re.sub(r'[^A-Za-z0-9_.-]', '_', name)}.lock"
        try:
            fd = _open_claim_file(path)
        except OSError as error:
            raise KinetixDiscoveryError(f"cannot open PVCAM claim file {path}: {error}") from error
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            os.close(fd)
            raise PvcamClaimedError(f"PVCAM interface {name} is in use by another Kinetix or process") from error
        except BaseException:
            os.close(fd)
            raise
        self._fd: int | None = fd

    def release(self) -> None:
        """Drop the claim; repeated calls do nothing."""
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None


def open_pvcam_device(name: str) -> "PyvcamDevice":
    """Open one PVCAM camera by name and close partially initialized handles on failure.

    Raises:
        PvcamClaimedError: another Kinetix or process holds ``name``; the camera was not touched.
    """
    pvc = pvcam_runtime()
    claim = PvcamClaim(name)
    try:
        # Imported dynamically, like pyvcam.pvc, so type checks pass where PyVCAM is absent (macOS).
        camera = import_module("pyvcam.camera").Camera(name)
        try:
            camera.open()
            return PyvcamDevice(camera, pvcam_version=str(pvc.get_pvcam_version()), claim=claim)
        except Exception:
            # PyVCAM can acquire the native handle before a later capability read fails.
            # Treat a failed open as partially open and make cleanup best-effort so the
            # original PVCAM error remains visible.
            if getattr(camera, "is_open", False):
                with suppress(Exception):
                    camera.close()
            raise
    except BaseException:
        claim.release()
        raise


class PyvcamDevice:
    """A PVCAM camera opened through PyVCAM, with SDK calls serialized except the frame wait.

    Sizes are (serial/x, parallel/y); ``pixel_size_um`` is None when the camera does not report it.
    """

    def __init__(self, camera: Any, pvcam_version: str, claim: PvcamClaim) -> None:
        const = import_module("pyvcam.constants")  # dynamic, as in open_pvcam_device
        self._const = const
        self._camera = camera
        self._claim = claim
        self._lock = threading.RLock()
        self.name = str(camera.name)
        self.identity = PvcamIdentity(
            serial=str(camera.serial_no).strip(),
            product=self._read_optional_str(const.PARAM_PRODUCT_NAME),
            chip=str(camera.chip_name),
            firmware=str(camera.cam_fw),
            pvcam_version=pvcam_version,
        )
        serial_size, parallel_size = camera.sensor_size
        self.sensor_size_px = (int(serial_size), int(parallel_size))
        self.pixel_size_um = self._read_pixel_size_um()
        self._exposure_scale_us = self._select_exposure_resolution()
        self.port_speed_gain_table: Mapping[str, Any] = camera.port_speed_gain_table
        binnings = camera.binnings
        self.binning_options: tuple[int, ...] = (
            tuple(sorted({int(bin_x) for bin_x, bin_y in binnings if bin_x == bin_y})) if binnings else (1, 2, 4, 8)
        )
        self.exposure_modes: dict[str, int] = {str(k): int(v) for k, v in dict(camera.exp_modes).items()}
        self.expose_out_modes: dict[str, int] = {str(k): int(v) for k, v in dict(camera.exp_out_modes).items()}

    @property
    def exposure_range_us(self) -> tuple[int, int]:
        with self._lock:
            minimum = self._camera.get_param(self._const.PARAM_EXPOSURE_TIME, self._const.ATTR_MIN)
            maximum = self._camera.get_param(self._const.PARAM_EXPOSURE_TIME, self._const.ATTR_MAX)
        # Kinetix reports zero as the PVCAM lower bound, but exp_time uses positive integer
        # resolution units. Match set_exposure_us by exposing one unit as the writable minimum.
        minimum_units = max(1, int(minimum))
        return minimum_units * self._exposure_scale_us, int(maximum) * self._exposure_scale_us

    def readout(self) -> tuple[int, int, int]:
        """Current (port value, speed index, gain index)."""
        with self._lock:
            return int(self._camera.readout_port), int(self._camera.speed), int(self._camera.gain)

    def set_readout(self, port: int, speed: int, gain: int) -> None:
        # PyVCAM documents this exact order; legacy cameras can end up inconsistent otherwise.
        with self._lock:
            self._camera.readout_port = port
            self._camera.speed = speed
            self._camera.gain = gain

    def bit_depth_host(self) -> int:
        with self._lock:
            return int(self._camera.bit_depth_host)

    def set_geometry(self, x: int, y: int, w: int, h: int, binning: int) -> None:
        # set_roi replaces the single region and keeps PyVCAM's current binning, which the
        # binning assignment then updates on that region; both only take effect at start_live.
        with self._lock:
            self._camera.set_roi(x, y, w, h)
            self._camera.binning = binning

    def set_exposure_us(self, exposure_us: int) -> None:
        with self._lock:
            self._camera.exp_time = max(1, round(exposure_us / self._exposure_scale_us))

    def set_exposure_mode(self, mode: int) -> None:
        with self._lock:
            self._camera.exp_mode = mode

    def expose_out_mode(self) -> int:
        if not self.expose_out_modes:
            return 0
        with self._lock:
            return int(self._camera.exp_out_mode)

    def set_expose_out_mode(self, mode: int) -> None:
        with self._lock:
            self._camera.exp_out_mode = mode

    def temperature_c(self) -> float | None:
        with self._lock:
            if not self._camera.check_param(self._const.PARAM_TEMP):
                return None
            try:
                return float(self._camera.temp)
            except Exception:
                return None

    def start_live(self, buffer_frame_count: int) -> None:
        with self._lock:
            self._camera.start_live(buffer_frame_count=buffer_frame_count, reset_frame_counter=True)

    def poll_frame(self, timeout_ms: int) -> tuple[np.ndarray, int, float]:
        """Pop the oldest frame as (pixels, frames delivered since start, frames per second)."""
        # Deliberately outside the lock: the wait releases the GIL and may last a full frame
        # timeout, during which settings reads must keep working. copyData=False returns a view
        # of PVCAM's ring, valid only until the camera wraps around to its slot; the caller
        # copies it once into its own destination.
        frame, fps, count = self._camera.poll_frame(timeout_ms=timeout_ms, oldestFrame=True, copyData=False)
        return frame["pixel_data"], int(count), float(fps)

    def finish(self) -> None:
        with self._lock:
            self._camera.finish()

    def close(self) -> None:
        with self._lock:
            if self._camera.is_open:
                self._camera.close()
            # Released only after the native close succeeds, so a failed close that the caller
            # retries never lets another process open the head while this handle may hold it.
            self._claim.release()

    def _read_optional_str(self, param_id: int) -> str:
        try:
            if self._camera.check_param(param_id):
                return str(self._camera.get_param(param_id))
        except Exception:
            return ""
        return ""

    def _read_pixel_size_um(self) -> tuple[float, float] | None:
        const = self._const
        try:
            reported = self._camera.check_param(const.PARAM_PIX_SER_SIZE) and self._camera.check_param(
                const.PARAM_PIX_PAR_SIZE
            )
            if not reported:
                return None
            x_nm = int(self._camera.get_param(const.PARAM_PIX_SER_SIZE))
            y_nm = int(self._camera.get_param(const.PARAM_PIX_PAR_SIZE))
        except Exception:
            return None
        if x_nm <= 0 or y_nm <= 0:
            return None
        return x_nm / 1000, y_nm / 1000

    def _select_exposure_resolution(self) -> int:
        """Prefer microsecond exposure units; return the scale from PVCAM units to microseconds."""
        const = self._const
        resolutions = set(dict(self._camera.exp_resolutions).values())
        if const.EXP_RES_ONE_MICROSEC in resolutions:
            self._camera.exp_res = const.EXP_RES_ONE_MICROSEC
            return 1
        if const.EXP_RES_ONE_MILLISEC in resolutions:
            self._camera.exp_res = const.EXP_RES_ONE_MILLISEC
            return 1000
        raise KinetixDiscoveryError(
            f"camera reports neither microsecond nor millisecond exposure resolution: {resolutions}"
        )


def describe_error(error: BaseException) -> str:
    """Flatten an exception and its causes; PyVCAM hides the PVCAM error code in ``__cause__``."""
    parts: list[str] = []
    seen: set[int] = set()
    current: BaseException | None = error
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        text = str(current).strip().rstrip(".")
        if text and text not in parts:
            parts.append(text)
        current = current.__cause__ or current.__context__
    return ": ".join(parts) or type(error).__name__


def discover_camera(serial: str) -> PyvcamDevice:
    """Open the PVCAM camera whose head serial number matches ``serial``, closing every other one.

    Heads that another Kinetix or process holds are skipped without being opened.
    """
    wanted = serial.strip()
    if not wanted:
        raise ValueError("Kinetix serial must not be empty")

    names = list_pvcam_camera_names()
    if not names:
        raise KinetixDiscoveryError(
            "PVCAM enumerated no cameras; check that the pvcam_pcie module created /dev/pvcamPCIE_* "
            "and that the process can read it"
        )

    found: list[str] = []
    failures: list[str] = []
    in_use: list[str] = []
    for name in names:
        try:
            device = open_pvcam_device(name)
        except PvcamClaimedError:
            in_use.append(name)
            continue
        except Exception as error:
            failures.append(f"{name}: {describe_error(error)}")
            continue
        found_serial = device.identity.serial
        if found_serial == wanted:
            return device
        device.close()
        found.append(f"{name}: serial={found_serial!r}")

    busy = f"; in use by another Kinetix or process: {', '.join(in_use)}" if in_use else ""
    if failures and not found:
        raise KinetixDiscoveryError(
            f"PVCAM could not open any camera for serial {wanted!r} ({'; '.join(failures)}{busy}). "
            "A device write timeout means the interface card sees no responding camera head: "
            "check camera power and the SFF-8644 cable."
        )
    enumerated = "; ".join(found + failures) or "none opened"
    raise KinetixDiscoveryError(f"no PVCAM camera has serial {wanted!r}; enumerated: {enumerated}{busy}")


class Kinetix(Camera):
    """Teledyne Photometrics Kinetix selected by head serial number through PVCAM.

    Settings changed while streaming restart the live acquisition before the next frame, so
    the delivered frame always matches the geometry and pixel format its destination was
    validated against.

    Args:
        pixel_size_um: Pixel pitch override in Voxel's (y, x) order, e.g. ``"6.5,6.5"``; when
            omitted, the camera-reported pitch is used.
    """

    __CONTROLLER_TYPE__ = KinetixCameraController

    def __init__(
        self,
        uid: str,
        serial: str,
        pixel_size_um: Vec2D | tuple[float, float] | list[float] | str | None = None,
        roi_step_px: int = DEFAULT_ROI_STEP_PX,
        buffer_size_mb: float = DEFAULT_BUFFER_SIZE_MB,
        frame_timeout_s: float = DEFAULT_FRAME_TIMEOUT_S,
        preview_trigger_mode: TriggerMode | str | bool | None = None,
    ) -> None:
        if roi_step_px < 1:
            raise ValueError(f"roi_step_px must be at least 1, got {roi_step_px}")
        if not math.isfinite(buffer_size_mb) or buffer_size_mb <= 0:
            raise ValueError(f"buffer_size_mb must be finite and positive, got {buffer_size_mb}")
        if not math.isfinite(frame_timeout_s) or frame_timeout_s <= 0:
            raise ValueError(f"frame_timeout_s must be finite and positive, got {frame_timeout_s}")
        super().__init__(uid=uid)
        self._buffer_size_mb = buffer_size_mb
        self._frame_timeout_s = frame_timeout_s
        self._preview_trigger_mode = _coerce_trigger_mode(preview_trigger_mode)
        self._lock = threading.RLock()

        device = discover_camera(serial)
        try:
            self._adopt(device, pixel_size_um, roi_step_px)
        except Exception:
            device.close()
            raise

    @property
    def preview_trigger_mode(self) -> TriggerMode | None:
        """Configured trigger mode for previews, or None to honor the caller's request."""
        return self._preview_trigger_mode

    def _adopt(
        self,
        device: PyvcamDevice,
        pixel_size_um: Vec2D | tuple[float, float] | list[float] | str | None,
        roi_step_px: int,
    ) -> None:
        identity = device.identity
        width, height = device.sensor_size_px
        if width <= 0 or height <= 0:
            raise KinetixDiscoveryError(f"camera {identity.serial!r} reports an empty sensor {width}x{height}")

        if pixel_size_um is not None:
            pixel_size = pixel_size_um if isinstance(pixel_size_um, Vec2D) else Vec2D.parse(pixel_size_um)
        elif device.pixel_size_um is not None:
            pixel_size = Vec2D(x=device.pixel_size_um[0], y=device.pixel_size_um[1])
        else:
            raise ValueError(f"camera {identity.serial!r} does not report its pixel size; configure pixel_size_um")
        if pixel_size.x <= 0 or pixel_size.y <= 0:
            raise ValueError(f"pixel_size_um must be positive, got {pixel_size}")

        modes = readout_modes(device.port_speed_gain_table)
        for mode in modes.values():
            for gain in mode.gains:
                pixel_format_for_bit_depth(gain.bit_depth)
        port, speed, gain_index = device.readout()
        current = next((mode for mode in modes.values() if (mode.port, mode.speed) == (port, speed)), None)
        if current is None:
            raise KinetixDiscoveryError(
                f"camera {identity.serial!r} is at port {port} speed {speed}, which its speed table does not list"
            )
        if current.gain_indexed(gain_index) is None:
            gain_index = current.gains[0].index

        binning_options = device.binning_options
        if 1 not in binning_options:
            raise KinetixDiscoveryError(
                f"camera {identity.serial!r} does not report required 1x binning; options: {binning_options}"
            )

        self._device = device
        self._identity = identity
        self._sensor_size = IVec2D(x=width, y=height)
        self._pixel_size = pixel_size
        self._modes = modes
        self._binning_options = binning_options
        # Keep every ROI divisible by every binning factor.
        self._roi_step = math.lcm(roi_step_px, *binning_options)
        self._trigger_modes = resolve_trigger_modes(device.exposure_modes)
        self._expose_out_modes: dict[str, int] = dict(device.expose_out_modes)

        # Desired state; pushed to PVCAM by _apply_settings, immediately or at the next restart.
        # _apply_settings also reads the exposure range and clamps the exposure into it.
        self._mode_label = current.label
        self._gain_index = gain_index
        self._binning = 1
        self._roi = SensorROI(w=width, h=height)
        self._exposure_us = DEFAULT_EXPOSURE_US
        self._exposure_mode = self._trigger_modes[TriggerMode.OFF]
        self._expose_out_mode = device.expose_out_mode()
        self._running = False
        self._restart_needed = False
        self._closed = False
        self._buffer_frames = 0
        self._live_layout: tuple[tuple[int, ...], np.dtype] | None = None
        self._frames_grabbed = 0
        self._last_delivered = 0
        self._dropped_frames = 0
        self._measured_fps = 0.0

        self._apply_settings()
        self.log.info(
            "Initialized %s serial %s on %s: chip %s, firmware %s, PVCAM %s, %dx%d px, modes %s",
            identity.product or "PVCAM camera",
            identity.serial,
            device.name,
            identity.chip,
            identity.firmware,
            identity.pvcam_version,
            width,
            height,
            list(modes),
        )

    @property
    @describe(label="Serial Number")
    def serial(self) -> str:
        return self._identity.serial

    @property
    @describe(label="Identity")
    def identity(self) -> str:
        identity = self._identity
        return (
            f"{identity.product or 'PVCAM camera'} serial {identity.serial}, chip {identity.chip}, "
            f"firmware {identity.firmware}, PVCAM {identity.pvcam_version}"
        )

    @property
    @describe(label="Sensor Size", units="px")
    def sensor_size_px(self) -> IVec2D:
        return self._sensor_size

    @property
    @describe(label="Pixel Size", units="µm")
    def pixel_size_um(self) -> Vec2D:
        return self._pixel_size

    def _current_mode(self) -> ReadoutMode:
        return self._modes[self._mode_label]

    def _current_gain(self) -> GainInfo:
        # Every writer of _gain_index picks an index from the current mode's gains.
        return next(gain for gain in self._current_mode().gains if gain.index == self._gain_index)

    def _current_pixel_format(self) -> PixelFormat:
        return pixel_format_for_bit_depth(self._current_gain().bit_depth)

    @enumerated(options=lambda self: list(self._modes))
    @describe(label="Sensor Mode", desc="PVCAM readout port and speed; determines bit depth.", stream=True)
    def sensor_mode(self) -> str:
        return self._mode_label

    @sensor_mode.setter
    def sensor_mode(self, value: str) -> None:
        with self._lock:
            mode = self._modes[value]
            self._mode_label = mode.label
            if mode.gain_indexed(self._gain_index) is None:
                self._gain_index = mode.gains[0].index
            self._commit()

    @enumerated(options=lambda self: self._current_mode().gain_names)
    @describe(label="Gain", desc="Gain entry of the current sensor mode.", stream=True)
    def gain(self) -> str:
        return self._current_gain().name

    @gain.setter
    def gain(self, value: str) -> None:
        with self._lock:
            self._gain_index = self._current_mode().gain_named(value).index
            self._commit()

    @enumerated(options=lambda self: [self._current_pixel_format()])
    @describe(label="Pixel Format", stream=True)
    def pixel_format(self) -> PixelFormat:
        return self._current_pixel_format()

    @pixel_format.setter
    def pixel_format(self, pixel_format: str) -> None:
        # Bit depth is a consequence of sensor_mode and gain; the enumerated descriptor only
        # forwards the single current option, so nothing changes here.
        del pixel_format

    @enumerated(options=lambda self: list(self._expose_out_modes))
    @describe(label="Expose Out Mode", desc="Behavior of the camera's Expose Out signal.")
    def expose_out_mode(self) -> str:
        modes = self._expose_out_modes.items()
        return next((name for name, value in modes if value == self._expose_out_mode), "")

    @expose_out_mode.setter
    def expose_out_mode(self, value: str) -> None:
        with self._lock:
            # rigup skips option checks when the camera reports no expose-out modes.
            if value not in self._expose_out_modes:
                raise ValueError(f"unknown expose-out mode {value!r}; options: {list(self._expose_out_modes)}")
            self._expose_out_mode = self._expose_out_modes[value]
            self._commit()

    @numeric_int(options=lambda self: list(self._binning_options))
    @describe(label="Binning", stream=True)
    def binning(self) -> int:
        return self._binning

    @binning.setter
    def binning(self, binning: int) -> None:
        with self._lock:
            self._binning = binning
            self._commit()

    @numeric(
        minimum=lambda self: self._exposure_range_us[0] / 1000,
        maximum=lambda self: self._exposure_range_us[1] / 1000,
        step=0.001,
    )
    @describe(label="Exposure Time", units="ms", stream=True)
    def exposure_time_ms(self) -> float:
        return self._exposure_us / 1000

    @exposure_time_ms.setter
    def exposure_time_ms(self, exposure_time_ms: float) -> None:
        with self._lock:
            minimum_us, maximum_us = self._exposure_range_us
            self._exposure_us = min(max(round(exposure_time_ms * 1000), minimum_us), maximum_us)
            self._commit()

    @numeric(minimum=0.0, step=0.01)
    @describe(label="Frame Rate", units="Hz", stream=True)
    def frame_rate_hz(self) -> float:
        """Measured rate while streaming, otherwise the internal-trigger estimate."""
        with self._lock:
            if self._running and self._measured_fps > 0:
                return self._measured_fps
            return 1_000_000 / self._exposure_us

    @frame_rate_hz.setter
    def frame_rate_hz(self, value: float) -> None:
        del value
        self.log.warning("Kinetix frame rate follows exposure time, readout mode, and external triggers; not set")

    @property
    @describe(label="Sensor Temperature", units="°C")
    def sensor_temperature_c(self) -> float | None:
        return self._device.temperature_c()

    def _get_roi(self) -> SensorROI:
        return self._roi

    def _set_roi(self, roi: SensorROI) -> None:
        with self._lock:
            self._roi = roi
            self._commit()

    @property
    @describe(label="ROI Grid", stream=True)
    def roi_grid(self) -> ROIGrid:
        step = self._roi_step
        return ROIGrid(
            h=IntRange(min=step, max=self._sensor_size.x, step=step),
            v=IntRange(min=step, max=self._sensor_size.y, step=step),
        )

    @property
    @describe(label="Stream Info", stream=True)
    def stream_info(self) -> StreamInfo | None:
        with self._lock:
            if not self._running:
                return None
            return StreamInfo(
                frame_index=self._frames_grabbed,
                input_buffer_size=self._buffer_frames,
                # PyVCAM reports delivery count but does not expose its pending queue depth.
                output_buffer_size=0,
                dropped_frames=self._dropped_frames,
                frame_rate_fps=self._measured_fps,
                data_rate_mbs=self._measured_fps * self.frame_size_mb,
            )

    def _commit(self) -> None:
        """Apply the desired state now, or before the next frame while streaming."""
        if self._running:
            self._restart_needed = True
            self.log.debug("settings change queued; live acquisition restarts before the next frame")
        else:
            self._apply_settings()

    def _apply_settings(self) -> None:
        mode = self._current_mode()
        self._device.set_readout(mode.port, mode.speed, self._gain_index)
        self._device.set_geometry(self._roi.x, self._roi.y, self._roi.w, self._roi.h, self._binning)
        # The exposure limits can follow the readout mode, so clamp against the refreshed range.
        minimum_us, maximum_us = self._device.exposure_range_us
        self._exposure_range_us = (minimum_us, maximum_us)
        self._exposure_us = min(max(self._exposure_us, minimum_us), maximum_us)
        self._device.set_exposure_us(self._exposure_us)
        self._device.set_exposure_mode(self._exposure_mode)
        if self._expose_out_modes:
            self._device.set_expose_out_mode(self._expose_out_mode)
        self._restart_needed = False

        delivered_bit_depth = self._device.bit_depth_host()
        if delivered_bit_depth != self._current_gain().bit_depth:
            raise RuntimeError(
                f"PVCAM reports {delivered_bit_depth}-bit output but the speed table lists "
                f"{self._current_gain().bit_depth}-bit for {self._mode_label!r}/{self._current_gain().name!r}"
            )

    def _configure_trigger_mode(self, mode: TriggerMode) -> None:
        with self._lock:
            self._exposure_mode = self._trigger_modes[mode]
            self._commit()

    def _configure_trigger_polarity(self, polarity: TriggerPolarity) -> None:
        if polarity is not TriggerPolarity.RISING_EDGE:
            raise ValueError("Kinetix external trigger input is rising-edge only; falling-edge polarity is unsupported")

    def _frame_layout(self) -> tuple[tuple[int, ...], np.dtype]:
        """Frame (shape, dtype) that the desired settings produce."""
        size = self.frame_size_px
        return (size.y, size.x), np.dtype(self.pixel_type.dtype)

    def _allocate_buffer(self) -> None:
        """PyVCAM allocates and frees the ring itself; _start_live sizes it from the settings."""

    def _start(self, frame_count: int | None = None) -> None:
        del frame_count  # the ring is circular; the controller stops after the frames it wants
        with self._lock:
            if self._running:
                raise KinetixAcquisitionError(f"Kinetix {self._identity.serial} is already acquiring")
            self._start_live()

    def _start_live(self) -> None:
        if self._restart_needed:
            self._apply_settings()
        self._live_layout = self._frame_layout()
        shape, dtype = self._live_layout
        self._buffer_frames = buffer_frame_count(math.prod(shape) * dtype.itemsize, self._buffer_size_mb)
        self._device.start_live(self._buffer_frames)
        self._running = True
        self._frames_grabbed = 0
        self._last_delivered = 0
        self._dropped_frames = 0
        self._measured_fps = 0.0
        self.log.debug(
            "live acquisition started: %s, exposure %d µs, ring %d frames",
            self.frame_size_px,
            self._exposure_us,
            self._buffer_frames,
        )

    def _frame_timeout_ms(self) -> int:
        return int(max(self._frame_timeout_s * 1000, 2 * self._exposure_us / 1000 + 1000))

    def _grab_frame(self, out: np.ndarray) -> None:
        with self._lock:
            if not self._running:
                raise KinetixAcquisitionError(f"Kinetix {self._identity.serial} is not acquiring; call start() first")
            layout = (out.shape, out.dtype)
            # grab_frame sized and validated `out` before this lock, so a concurrent setting change
            # can leave it fitting the running acquisition but not the pending one. Restart only
            # for a destination that fits the new settings; otherwise deliver one more frame from
            # the running ring, and the caller's next grab_frame validates against the new size.
            if self._restart_needed and layout == self._frame_layout():
                self._device.finish()
                self._running = False
                self._start_live()
            if layout != self._live_layout:
                raise KinetixAcquisitionError(
                    f"frame destination {out.shape} {out.dtype} matches neither the running "
                    f"acquisition {self._live_layout} nor the pending settings"
                )
            timeout_ms = self._frame_timeout_ms()

        try:
            pixels, delivered, fps = self._device.poll_frame(timeout_ms)
        except Exception as error:
            raise KinetixAcquisitionError(
                f"Kinetix {self._identity.serial} delivered no frame within {timeout_ms} ms: {error}"
            ) from error
        if pixels.shape != out.shape or pixels.dtype != out.dtype:
            raise KinetixAcquisitionError(
                f"PVCAM delivered a {pixels.shape} {pixels.dtype} frame; expected {out.shape} {out.dtype}"
            )
        # PyVCAM discards queued frames only once its queue, as deep as the ring, is full. A gap
        # in the delivery counter therefore means this reader fell a whole ring behind, and the
        # popped frame occupies the slot PVCAM writes next, so copying it could mix exposures.
        # A reader that stays less than a ring behind never has its slot overwritten mid-copy.
        overrun = delivered - self._last_delivered > 1
        self._last_delivered = delivered
        if overrun:
            with self._lock:
                self._dropped_frames = delivered - self._frames_grabbed
                self._measured_fps = fps
            raise KinetixAcquisitionError(
                f"Kinetix {self._identity.serial} ring overrun: the reader fell "
                f"{self._buffer_frames} frames behind, so frame {delivered} may be overwritten"
            )
        np.copyto(out, pixels, casting="no")

        with self._lock:
            self._frames_grabbed += 1
            self._dropped_frames = max(0, delivered - self._frames_grabbed)
            self._measured_fps = fps

    def stop(self) -> None:
        with self._lock:
            if not self._running:
                return
            try:
                self._device.finish()
            finally:
                self._running = False

    def close(self) -> None:
        """Stop acquisition, drop the ring, and release the PVCAM handle; safe to repeat."""
        with self._lock:
            if self._closed:
                return
            try:
                self.stop()
            finally:
                self.free_buffer()
                self._device.close()
                self._closed = True
