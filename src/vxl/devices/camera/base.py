import asyncio
import json
import logging
import shutil
import time
from abc import abstractmethod
from contextlib import suppress
from enum import StrEnum
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np
from ome_zarr_writer import (
    BinningMode,
    FrameLease,
    OMEZarrWriter,
    PreFilterType,
    UIVec3D,
    UVec3D,
    WriterConfig,
    WriterSettings,
    filter_and_bin_2d,
    filter_and_bin_volume,
)
from ome_zarr_writer.sizing import per_slot_bytes, slots_for_budget
from ome_zarr_writer.writer import BatchMetrics
from pydantic import BaseModel, ConfigDict
from vxl_records import DatasetLocation, StorageSpec
from vxlib.dtype import Dtype
from vxlib.schema import FrozenModel
from vxlib.vector import IVec2D, Vec2D, Vec3D

from rigup import Device, DeviceController, describe, enumerated, numeric, numeric_int
from vxl._utils.tiff import write_tiff_2d
from vxl.devices.base import DeviceType
from vxl.preview import PreviewFrame, PreviewGenerator, PreviewLayer, PreviewViewport, ValidBits
from vxl.system import System, remote_store_fingerprint

if TYPE_CHECKING:
    from vxl.instrument.snr import CameraSnrAnalysis

log = logging.getLogger(__name__)


class StorageStatus(FrozenModel):
    """A node's write-access result for a `Storage` root (preflight). ``free_bytes`` is -1 when
    capacity is unknown (e.g. an S3 root)."""

    host: str
    root: str
    free_bytes: int


class IntRange(BaseModel):
    """Integer range with min, max, and step."""

    min: int
    max: int
    step: int = 1


class ROIGrid(BaseModel):
    """Hardware constraints for sensor ROI, one range per axis.

    Each axis describes valid sizes (width/height) with min, max, and step.
    Offset constraints are derived: min=0, max=axis.max - snapped_size, same step.
    """

    h: IntRange  # horizontal: valid widths
    v: IntRange  # vertical: valid heights


class SensorROI(BaseModel):
    """Region of interest on the camera sensor, in physical sensor pixels (pre-binning).

    All coordinates are in the unbinned sensor pixel space. FOV depends only on
    (x, y, w, h) and the camera's pixel_size_um — binning is independent.
    """

    model_config = ConfigDict(extra="forbid")

    x: int = 0
    y: int = 0
    w: int
    h: int

    def snap(self, grid: ROIGrid) -> "SensorROI":
        """Return a new SensorROI clamped and aligned to the given grid.

        Snaps size first, then derives offset bounds from the snapped size.
        """

        def _snap(value: int, r: IntRange) -> int:
            value = max(r.min, min(value, r.max))
            remainder = (value - r.min) % r.step
            if remainder != 0:
                value = value - remainder if remainder < r.step / 2 else value + (r.step - remainder)
            return min(value, r.max)

        w = _snap(self.w, grid.h)
        h = _snap(self.h, grid.v)
        x = _snap(self.x, IntRange(min=0, max=grid.h.max - w, step=grid.h.step))
        y = _snap(self.y, IntRange(min=0, max=grid.v.max - h, step=grid.v.step))
        return SensorROI(x=x, y=y, w=w, h=h)


class TriggerMode(StrEnum):
    OFF = "off"
    ON = "on"


class TriggerPolarity(StrEnum):
    RISING_EDGE = "rising"
    FALLING_EDGE = "falling"


class StreamInfo(FrozenModel):
    frame_index: int
    input_buffer_size: int
    output_buffer_size: int
    dropped_frames: int
    frame_rate_fps: float
    data_rate_mbs: float
    payload_mbs: float | None = None


PixelFormat = Literal["MONO8", "MONO10", "MONO12", "MONO14", "MONO16"]

PIXEL_FMT_TO_DTYPE: dict[PixelFormat, Dtype] = {
    "MONO8": Dtype.UINT8,
    "MONO10": Dtype.UINT16,
    "MONO12": Dtype.UINT16,
    "MONO14": Dtype.UINT16,
    "MONO16": Dtype.UINT16,
}

PIXEL_FMT_TO_VALID_BITS: dict[PixelFormat, ValidBits] = {
    "MONO8": 8,
    "MONO10": 10,
    "MONO12": 12,
    "MONO14": 14,
    "MONO16": 16,
}

BINNING_OPTIONS = [1, 2, 4, 8]
GPU_BINNING_OPTIONS = [1, 2, 3, 4, 6, 8]
GPU_BIN_MODE_OPTIONS = [mode.value for mode in BinningMode]
GPU_FILTER_OPTIONS = [flt.value for flt in PreFilterType]
GPU_FILTER_SIZE_OPTIONS = [3, 5]


class SnapshotFrameInfo(FrozenModel):
    """Metadata returned by a camera controller after capturing a single-frame snapshot."""

    camera_id: str
    width: int
    height: int
    dtype: str
    min_adu: float
    max_adu: float
    mean_adu: float
    std_adu: float
    hardware_binning: int = 1
    gpu_bin_y: int = 1
    gpu_bin_x: int = 1
    gpu_pre_filter: str = "none"
    gpu_filter_size: int = 3
    saved_path: str | None = None


class CaptureState(StrEnum):
    IDLE = "IDLE"
    COLLECTING = "COLLECTING"
    DONE = "DONE"
    CLOSING = "CLOSING"
    CLOSED = "CLOSED"


class CameraMode(StrEnum):
    IDLE = "IDLE"
    PREVIEW = "PREVIEW"
    ACQUISITION = "ACQUISITION"


class CameraController(DeviceController["Camera"]):
    def __init__(self, device: "Camera", stream_interval: float = 0.5, *, system: System | None = None):
        super().__init__(device, stream_interval=stream_interval)
        self._system = system if system is not None else System()
        self._mode = CameraMode.IDLE
        self._preview_task: asyncio.Task | None = None
        self._frame_idx = 0
        self._previewer = PreviewGenerator(sink=self._on_preview_frame, uid=device.uid)
        self._writer: OMEZarrWriter | None = None
        self._task: asyncio.Task[None] | None = None
        self._task_kind: Literal["batch", "close"] | None = None
        self._publish_tasks: dict[PreviewLayer, asyncio.Task[None]] = {}
        self._diag_buffers: dict[str, np.ndarray] = {}
        # Preview-publish counters (best-effort publish is skip-if-busy); reported per stack.
        System.Ram.reserve(self.device.uid, weight=1.0)

    async def close(self) -> None:
        await self.stop_preview()
        await self.close_stack()
        if self._task is not None:
            await asyncio.shield(self._task)
        self._previewer.close()
        for task in self._publish_tasks.values():
            task.cancel()
        await asyncio.gather(*self._publish_tasks.values(), return_exceptions=True)
        self._publish_tasks.clear()
        self._diag_buffers.clear()
        await self.release_writer()
        System.Ram.release(self.device.uid)
        await super().close()

    @property
    @describe(label="RAM Budget", units="bytes", stream=True)
    def ram_budget_bytes(self) -> int:
        """Current RAM share for this camera from the host RAM budget."""
        return System.Ram.reserved(self.device.uid)

    @property
    @describe(label="Camera Mode", stream=True)
    def mode(self) -> CameraMode:
        return self._mode

    @property
    @describe(label="Writer Metrics", stream=True)
    def writer_metrics(self) -> list[BatchMetrics]:
        """Recent started batches for live status; full history is saved in metrics.json."""
        if self._writer is None:
            return []
        recent: list[BatchMetrics] = []
        for batch in reversed(self._writer.batches):
            if batch.collecting.started is not None:
                recent.append(batch)
                if len(recent) == 64:
                    break
        return list(reversed(recent))

    def _on_preview_frame(self, frame: PreviewFrame) -> None:
        task = self._publish_tasks.get(frame.header.layer)
        if task is not None and not task.done():
            self._previewer.health.record_publish_drop()  # Gate 2 drop: prior publish still in flight
            return
        self._previewer.health.record_publish()
        self._publish_tasks[frame.header.layer] = asyncio.create_task(self._publish_preview(frame))

    async def _publish_preview(self, frame: PreviewFrame) -> None:
        topic = "preview" if frame.header.layer is PreviewLayer.OVERVIEW else "preview_viewport"
        with suppress(RuntimeError):
            await self.publish(topic, frame.pack())

    @describe(label="Update Preview Viewport")
    async def update_preview_viewport(self, viewport: PreviewViewport):
        task = self._previewer.set_viewport(
            viewport,
            regenerate=self._mode is CameraMode.IDLE,
        )
        await task if task is not None else None

    @describe(label="Reset Preview Stream")
    async def reset_preview_stream(self) -> str:
        """Discard pending preview work and return the new source-stream identity."""
        return self._previewer.reset_stream()

    @describe(label="Start Preview")
    async def start_preview(
        self,
        trigger_mode: TriggerMode = TriggerMode.ON,
        trigger_polarity: TriggerPolarity = TriggerPolarity.RISING_EDGE,
    ) -> str:
        """Start preview mode. Returns topic name where frames will be published."""
        if self._mode != CameraMode.IDLE:
            raise RuntimeError(f"Cannot start preview: camera in {self._mode} mode")

        # start() lazily allocates the buffer, configures the trigger, and begins (frame_count=None → infinite).
        await self._run_sync(
            lambda: self.device.start(frame_count=None, trigger_mode=trigger_mode, trigger_polarity=trigger_polarity)
        )

        self._mode = CameraMode.PREVIEW
        self._frame_idx = 0
        self._previewer.reset_stream()
        self._preview_task = asyncio.create_task(self._preview_loop())

        return "preview"

    async def _preview_loop(self):
        try:
            while self._mode == CameraMode.PREVIEW:
                frame = await self._run_sync(self.device.grab_frame_owned)
                if self._mode != CameraMode.PREVIEW:
                    break
                if self.device.has_gpu_binning:
                    frame = await self._run_sync(self.device.apply_gpu_binning_2d, frame)
                self._previewer.submit_frame(
                    frame,
                    idx=self._frame_idx,
                    valid_bits=PIXEL_FMT_TO_VALID_BITS[cast("PixelFormat", str(self.device.pixel_format))],
                )
                self._frame_idx += 1
        except asyncio.CancelledError:
            pass
        except Exception:
            self.log.exception("Preview loop error")
            self._mode = CameraMode.IDLE

    @describe(label="Stop Preview")
    async def stop_preview(self):
        if self._mode != CameraMode.PREVIEW:
            return

        self._mode = CameraMode.IDLE
        if self._preview_task:
            await self._preview_task
            self._preview_task = None
        self._previewer.cancel_pending()
        await self._run_sync(self.device.stop)
        await self._run_sync(self.device.free_buffer)

    async def _emit_preview_sync(self, frame: np.ndarray) -> None:
        """Flush pending preview publishes, render ``frame`` synchronously, and wait for publish."""
        valid_bits = PIXEL_FMT_TO_VALID_BITS[cast("PixelFormat", str(self.device.pixel_format))]
        if self._publish_tasks:
            await asyncio.gather(*self._publish_tasks.values(), return_exceptions=True)
        await self._previewer.submit_frame_and_wait(frame, self._frame_idx, valid_bits=valid_bits)
        self._frame_idx += 1
        if self._publish_tasks:
            await asyncio.gather(*self._publish_tasks.values(), return_exceptions=True)

    @describe(label="Arm Capture")
    async def arm_capture(
        self,
        num_frames: int = 1,
        trigger_mode: TriggerMode = TriggerMode.ON,
        trigger_polarity: TriggerPolarity = TriggerPolarity.RISING_EDGE,
    ) -> None:
        """Arm the camera for ``num_frames`` hardware-triggered frames (snapshot / SNR burst)."""
        if self._mode != CameraMode.IDLE:
            raise RuntimeError(f"Cannot arm capture: camera in {self._mode} mode")
        await self._run_sync(
            lambda: self.device.start(
                frame_count=num_frames,
                trigger_mode=trigger_mode,
                trigger_polarity=trigger_polarity,
            )
        )

    @describe(label="Capture Raw Frames")
    async def capture_frames_raw(self, num_frames: int = 1, *, slot_key: str = "snapshot") -> int:
        """Grab ``num_frames`` armed frames into RAM and immediately stop the camera."""
        frames: list[np.ndarray] = []
        try:
            for _ in range(num_frames):
                frames.append(await self._run_sync(self.device.grab_frame_owned))
        finally:
            with suppress(Exception):
                await self._run_sync(self.device.stop)
            with suppress(Exception):
                await self._run_sync(self.device.free_buffer)
        self._diag_buffers[slot_key] = np.stack(frames, axis=0)
        return len(frames)

    @describe(label="Process Snapshot")
    async def process_snapshot(
        self,
        *,
        slot_key: str = "snapshot",
        save_path: str | None = None,
        pixel_size_um: tuple[float, float] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> SnapshotFrameInfo:
        """Apply GPU pre-filter + binning to a buffered raw snapshot, save TIFF, and emit preview."""
        stack = self._diag_buffers.pop(slot_key, None)
        if stack is None or stack.shape[0] < 1:
            raise RuntimeError(f"No raw snapshot frame in slot {slot_key!r} on camera {self.device.uid}")
        frame = stack[-1]
        if self.device.has_gpu_binning:
            frame = await self._run_sync(self.device.apply_gpu_binning_2d, frame)

        bin_y, bin_x = int(self.device.gpu_bin_y), int(self.device.gpu_bin_x)
        eff_px = (float(pixel_size_um[0]) * bin_y, float(pixel_size_um[1]) * bin_x) if pixel_size_um else None
        saved_str: str | None = None
        if save_path is not None:
            saved_str = str(
                await self._run_sync(
                    lambda: write_tiff_2d(save_path, frame, pixel_size_um=eff_px, metadata=metadata)
                )
            )
        await self._emit_preview_sync(frame)
        return SnapshotFrameInfo(
            camera_id=self.device.uid,
            width=int(frame.shape[1]),
            height=int(frame.shape[0]),
            dtype=str(frame.dtype),
            min_adu=float(np.min(frame)),
            max_adu=float(np.max(frame)),
            mean_adu=round(float(np.mean(frame)), 2),
            std_adu=round(float(np.std(frame)), 2),
            hardware_binning=int(self.device.binning),
            gpu_bin_y=bin_y,
            gpu_bin_x=bin_x,
            gpu_pre_filter=str(self.device.gpu_pre_filter),
            gpu_filter_size=int(self.device.gpu_filter_size),
            saved_path=saved_str,
        )

    @describe(label="Analyze SNR Buffer")
    async def analyze_snr_buffer(
        self,
        *,
        output_dir: str,
        channel_id: str,
        profile_id: str,
        pixel_size_um: tuple[float, float],
        exposure_ms: float,
    ) -> "CameraSnrAnalysis":
        """Apply GPU binning to buffered raw signal + dark stacks and compute SNR diagnostics."""
        from vxl.instrument.snr import compute_camera_snr_diagnostics  # noqa: PLC0415

        sig_stack = self._diag_buffers.pop("signal", None)
        dark_stack = self._diag_buffers.pop("dark", None)
        if sig_stack is None:
            raise RuntimeError(f"No signal frames buffered on camera {self.device.uid}")

        if self.device.has_gpu_binning:
            sig_stack = await self._run_sync(self.device.apply_gpu_binning_3d, sig_stack)
            if dark_stack is not None:
                dark_stack = await self._run_sync(self.device.apply_gpu_binning_3d, dark_stack)

        bin_y, bin_x = int(self.device.gpu_bin_y), int(self.device.gpu_bin_x)
        eff_px_um = (float(pixel_size_um[0]) * bin_y, float(pixel_size_um[1]) * bin_x)
        valid_bits = int(PIXEL_FMT_TO_VALID_BITS[cast("PixelFormat", str(self.device.pixel_format))])
        await self._emit_preview_sync(sig_stack[-1])
        return await self._run_sync(
            lambda: compute_camera_snr_diagnostics(
                sig_stack,
                dark_frames=dark_stack,
                camera_id=self.device.uid,
                channel_id=channel_id,
                profile_id=profile_id,
                valid_bits=valid_bits,
                exposure_time_ms=exposure_ms,
                pixel_size_um=eff_px_um,
                hardware_binning=int(self.device.binning),
                gpu_bin_y=bin_y,
                gpu_bin_x=bin_x,
                gpu_pre_filter=str(self.device.gpu_pre_filter),
                gpu_filter_size=int(self.device.gpu_filter_size),
                output_dir=output_dir,
            )
        )

    @describe(label="Check Writable")
    async def check_writable(self, storage: StorageSpec) -> StorageStatus:
        """Resolve ``storage`` on this node and prove write access with a round-trip marker write.

        The acquisition preflight (before any task): raises with a clear reason if the destination
        is unreachable or not writable, so the run aborts before any stage motion or capture.
        """

        def _check() -> StorageStatus:
            dest = self._system.resolve_storage(storage)
            root = dest.target  # client-bound path for S3, plain path for local
            root.mkdir(parents=True, exist_ok=True)
            marker = root / f".voxel_write_check-{self.device.uid}"
            marker.write_text(self.device.uid)  # raises on missing creds / no permission / unreachable
            marker.unlink(missing_ok=True)
            free = shutil.disk_usage(root).free if isinstance(root, Path) else -1
            return StorageStatus(host=self._system.hostname(), root=str(root), free_bytes=free)

        return await self._run_sync(_check)

    @describe(label="Remote Stores")
    def remote_stores(self) -> dict[str, str]:
        """Return configured object-store names and non-secret connection fingerprints for this camera's host."""
        return {name: remote_store_fingerprint(remote.connection) for name, remote in self._system.remotes.items()}

    @describe(label="Writer Throughput")
    def writer_throughput_gb_s(self) -> tuple[float, float | None] | None:
        """Current live writer throughput ``(raw_gb_s, compressed_gb_s)``, or ``None``."""
        return self._writer.live_throughput_gb_s() if self._writer is not None else None

    @describe(label="Open Stack")
    async def open_stack(
        self,
        *,
        num_frames: int,
        z_step: float,
        magnification: float,
        storage: StorageSpec,
        subpath: PurePosixPath,
        settings: WriterSettings,
        voxel_size_yx: tuple[float, float] | None = None,
        origin: Vec3D | None = None,
        shear_xz: float = 0.0,
        shear_yz: float = 0.0,
        channel_label: str = "channel 0",
        channel_color: str = "FFFFFF",
        darkfield: dict[str, Any] | None = None,
    ) -> DatasetLocation:
        """Prepare the camera + writer for a stack and return its durable dataset location.

        ``storage`` is the run's logical destination and ``subpath`` the dataset's relative location
        under it (e.g. ``task/profile/channel``); the node resolves it against its :class:`System` and
        the writer names it ``<subpath>.ome.zarr``. ``num_frames``/``z_step`` are the volume's
        geometry; ``magnification`` converts the sensor-space ``effective_pixel_size_um`` into the
        sample-space lateral voxel size. ``settings`` are the broadcast output-format knobs. The
        returned location describes the destination while it is being written; OME-Zarr metadata is
        authoritative for dataset geometry. The trigger is configured per-batch in ``begin_batch``.
        """
        if self._mode != CameraMode.IDLE:  # a stack already open (or previewing) → mode is not IDLE
            raise RuntimeError(f"Cannot open stack: camera in {self._mode} mode")

        self._mode = CameraMode.ACQUISITION
        self._frame_idx = 0
        self._previewer.reset_stream()
        self._task = None
        self._task_kind = None

        # Allocate buffers eagerly (after ROI is set), so every batch's start() is fast.
        _t = time.perf_counter()
        await self._run_sync(self.device.allocate_buffer)
        log.info("allocate_buffer for %s took %.1fs", self.device.uid, time.perf_counter() - _t)

        # Resolve the logical StorageSpec to a concrete omezarr Storage, then build the writer config:
        # the broadcast `settings` (knobs) + this camera's geometry (frame size, voxel size from
        # z_step + magnification) + dtype from the device's pixel format. Where/how to write lives on
        # the Storage; the config is purely the dataset spec.
        dest = self._system.resolve_storage(storage, subpath)
        frame = self.device.frame_size_px
        out_frame = self.device.output_frame_size_px
        eff_px = self.device.effective_pixel_size_um(magnification)
        vy, vx = voxel_size_yx if voxel_size_yx is not None else (eff_px.y, eff_px.x)
        bin_y, bin_x = int(self.device.gpu_bin_y), int(self.device.gpu_bin_x)
        cfg = WriterConfig(
            **settings.model_dump(),
            volume_shape=UIVec3D(z=num_frames, y=out_frame.y, x=out_frame.x),
            voxel_size=UVec3D(z=z_step, y=vy * bin_y, x=vx * bin_x),
            dtype=self.device.pixel_type,
            origin=origin,
            shear_xz=shear_xz,
            shear_yz=shear_yz,
            channel_label=channel_label,
            channel_color=channel_color,
            bin_y=bin_y,
            bin_x=bin_x,
            bin_mode=BinningMode(str(self.device.gpu_bin_mode)),
            pre_filter=PreFilterType(str(self.device.gpu_pre_filter)),
            filter_size=int(self.device.gpu_filter_size),
            raw_frame_y=frame.y,
            raw_frame_x=frame.x,
            darkfield=darkfield,
        )

        # The coordinator persists across volumes so its ring is reused. Ring sizing happens inside
        # begin_stack via `_size_ring`, and only when it actually (re)allocates — reusing a ring
        # allocates nothing, so there is no RAM budget to check.
        if self._writer is None:
            self._writer = OMEZarrWriter()
        _t = time.perf_counter()
        await self._run_sync(self._writer.begin_stack, cfg, dest, sizer=self._size_ring)
        log.info(
            "OMEZarrWriter begin_stack for %s took %.1fs",
            self.device.uid,
            time.perf_counter() - _t,
        )

        log.info(
            "Stack init for %s → %s: batch_z=%d",
            self.device.uid,
            self._writer.target,
            cfg.batch_z,
        )
        target = self._writer.target
        if target is None:
            raise RuntimeError("writer did not expose a dataset target")
        return self._system.describe_dataset_location(storage, target)

    @describe(label="Update Dataset Darkfield Metadata")
    async def update_dataset_darkfield(
        self,
        *,
        storage: StorageSpec,
        subpath: PurePosixPath,
        darkfield: dict[str, Any],
    ) -> None:
        """Update ``attributes.darkfield`` in the root ``zarr.json`` of ``<subpath>.ome.zarr``."""

        def _update() -> None:
            target = self._system.resolve_storage(storage, subpath).target
            if not target.name.endswith(".ome.zarr"):
                target = target.with_name(f"{target.name}.ome.zarr")
            zarr_json = target / "zarr.json"
            if not zarr_json.exists():
                return
            raw = json.loads(zarr_json.read_text(encoding="utf-8"))
            raw.setdefault("attributes", {})["darkfield"] = darkfield
            zarr_json.write_text(f"{json.dumps(raw, indent=2)}\n", encoding="utf-8")

        await self._run_sync(_update)

    def _size_ring(self, config: WriterConfig) -> int:
        budget = self.ram_budget_bytes
        available = System.Ram.available_bytes()
        from ome_zarr_writer.pyramid_gpu import is_gpu_available  # noqa: PLC0415

        effective_gpu = config.use_gpu and is_gpu_available()
        kwargs = {
            "use_gpu": effective_gpu,
            "shard_shape": config.shard_shape if effective_gpu else None,
            "raw_batch_shape": config.raw_batch_shape,
        }
        slots = slots_for_budget(
            config.batch_shape,
            config.max_level,
            config.dtype,
            budget_bytes=budget,
            available_bytes=available,
            label=self.device.uid,
            **kwargs,
        )
        log.info(
            "Sizing ring for %s: %d slots (%.2f GB/slot peak, %.2f GB share, %.2f GB free, gpu=%s)",
            self.device.uid,
            slots,
            per_slot_bytes(config.batch_shape, config.max_level, config.dtype, **kwargs) / 1e9,
            budget / 1e9,
            available / 1e9,
            effective_gpu,
        )
        return slots

    @describe(label="Close Stack")
    async def close_stack(self) -> None:
        """Stop collection and drain the writer in the background; poll until ``CLOSED``.

        Repeated calls join the same cleanup. In-flight frame writes settle before their buffers
        are released, including when capture failed or was cancelled.
        """
        if (
            self._task_kind == "close"
            and self._task is not None
            and not self._task.cancelled()
            and (not self._task.done() or self._task.exception() is None)
        ):
            return
        collecting = self._task
        if collecting is not None and not collecting.done():
            collecting.cancel()
        self._task = asyncio.create_task(self._close_stack(collecting))
        self._task_kind = "close"

    async def _close_stack(self, collecting: asyncio.Task[None] | None) -> None:
        """Flush and close the writer, then reset the camera to a reusable IDLE state.

        The camera is always reset even if ``writer.end_stack()`` raises. A failed writer ring remains
        owned but poisoned until preview releases its readers; the next stack or :meth:`release_writer`
        then replaces it safely.
        """
        if collecting is not None:
            # Capture errors are reported through capture_state; cleanup must still run.
            await asyncio.gather(collecting, return_exceptions=True)
        if self._mode is CameraMode.ACQUISITION:
            await self._run_sync(self.device.stop)
        self._previewer.cancel_pending()
        try:
            if self._writer is not None:
                await self._run_sync(self._writer.end_stack)  # drain the dataset; keep the ring for reuse
        finally:
            await self._run_sync(self.device.free_buffer)
            self._mode = CameraMode.IDLE
        log.info("Stack closed for %s", self.device.uid)
        # Preview health for the stack: how many frames got an overview vs were dropped (best-effort),
        # overview gen-time, and publish drops. overview_busy_drops climbing means gen can't hold 6fps.
        pv = self._previewer.health.snapshot()
        log.info(
            "Preview health %s: frames=%d overview_gen=%d overview_busy_drops=%d "
            "gen_ms(avg=%.1f max=%.1f) | publish_sent=%d publish_busy_drops=%d",
            self.device.uid,
            pv.frames,
            pv.overviews_generated,
            pv.overview_busy_drops,
            pv.generation_ms_average,
            pv.generation_ms_max,
            pv.publish_sent,
            pv.publish_busy_drops,
        )

    @describe(label="Release Writer")
    async def release_writer(self) -> None:
        """Free the reusable writer ring (its worker processes + shared memory) while keeping the
        controller alive for preview / the next acquisition, which re-allocates the ring cold.

        The ring is otherwise held from the first ``open_stack`` until ``close``; releasing it here
        (e.g. once an acquisition run ends) avoids holding ~hundreds of GB resident while idle. No-op
        if a capture task is still active — freeing a ring mid-flush would corrupt in-flight writes.
        """
        if self._task is not None and not self._task.done():
            raise RuntimeError("Cannot release writer while a capture task is in progress.")
        if self._writer is not None:
            self._previewer.cancel_pending()
            self._previewer.clear_frame()
            await self._run_sync(self._writer.close)
            self._writer = None
            log.info("Writer ring released for %s", self.device.uid)

    @property
    @describe(label="Ready For Batch", stream=True)
    def ready_for_batch(self) -> bool:
        """Whether the writer has at least one free slot to accept the next batch.

        True when no stack is initialized, or when at least one ring-buffer slot
        is IDLE.
        """
        return self._writer is None or self._writer.ready_for_batch

    @describe(label="Begin Batch")
    async def begin_batch(
        self,
        num_frames: int,
        trigger_mode: TriggerMode = TriggerMode.ON,
        trigger_polarity: TriggerPolarity = TriggerPolarity.RISING_EDGE,
    ) -> None:
        """Arm the trigger for ``num_frames`` frames and start grabbing them in the background.

        Waits for a free writer slot, then arms the camera before returning — so the caller can start
        the AO/triggers as soon as this resolves. Poll :meth:`capture_state` for completion; frame and
        pipeline progress stream via ``writer_metrics``. Must call ``open_stack`` first.
        """
        if self._writer is None:
            raise RuntimeError("No stack open. Call open_stack() first.")
        if self._mode != CameraMode.ACQUISITION:
            raise RuntimeError(f"Cannot capture batch: camera in {self._mode} mode")
        if self._task is not None and not self._task.done():
            raise RuntimeError("Capture task already in progress.")
        writer = self._writer
        await self._run_sync(writer.wait_ready)
        await self._run_sync(
            lambda: self.device.start(
                frame_count=num_frames, trigger_mode=trigger_mode, trigger_polarity=trigger_polarity
            )
        )
        self._task = asyncio.create_task(self._collect_batch(num_frames, writer))
        self._task_kind = "batch"

    async def _collect_batch(self, num_frames: int, writer: OMEZarrWriter) -> None:
        """Grab ``num_frames`` frames into the writer, then stop the camera."""
        try:
            for _ in range(num_frames):
                frame = await self._capture_into_writer(writer)
                try:
                    self._previewer.submit_frame(
                        frame,
                        self._frame_idx,
                        valid_bits=PIXEL_FMT_TO_VALID_BITS[cast("PixelFormat", str(self.device.pixel_format))],
                    )
                except Exception:
                    frame.close()
                    raise
                self._frame_idx += 1
        finally:
            await self._run_sync(self.device.stop)

    async def _capture_into_writer(self, writer: OMEZarrWriter) -> FrameLease:
        """Fill and commit one writer frame as one serialized, cancellation-safe device transaction."""

        def _grab() -> FrameLease:
            with writer.new_frame() as frame:
                self.device.grab_frame(out=frame)
            return writer.latest_frame()

        task = asyncio.create_task(self._run_sync(_grab))
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            # run_in_executor work cannot be stopped once running. Wait for the transaction to settle so
            # its destination is not reused mid-write, then release a committed frame nobody will consume.
            with suppress(Exception):
                frame = await task
                frame.close()
            raise

    @describe(label="Captured Frames")
    async def captured_frames(self) -> int:
        """Frames collected in the current stack; reads memory without querying camera hardware."""
        return self._frame_idx

    @describe(label="Capture State")
    async def capture_state(self) -> CaptureState:
        """Lifecycle of the most recent :meth:`begin_batch`/:meth:`close_stack` task. Non-mutating: the
        terminal state persists until the next task starts. Raises if the task failed."""
        task = self._task
        if task is None:
            return CaptureState.IDLE
        if not task.done():
            return CaptureState.COLLECTING if self._task_kind == "batch" else CaptureState.CLOSING
        if task.cancelled():
            raise RuntimeError("Capture task was cancelled before cleanup")
        if (exc := task.exception()) is not None:
            raise RuntimeError(f"Capture task failed: {exc}")
        return CaptureState.DONE if self._task_kind == "batch" else CaptureState.CLOSED


class Camera(Device):
    __DEVICE_TYPE__ = DeviceType.CAMERA
    __CONTROLLER_TYPE__ = CameraController

    trigger_mode: TriggerMode = TriggerMode.OFF
    trigger_polarity: TriggerPolarity = TriggerPolarity.RISING_EDGE

    _buffer_allocated = False
    _gpu_bin_y: int = 1
    _gpu_bin_x: int = 1
    _gpu_bin_mode: str = "mean"
    _gpu_pre_filter: str = "none"
    _gpu_filter_size: int = 3

    @numeric_int(options=GPU_BINNING_OPTIONS)
    @describe(label="GPU Bin Y", desc="Software/GPU rectangular binning factor along Y (rows).", stream=True)
    def gpu_bin_y(self) -> int:
        return self._gpu_bin_y

    @gpu_bin_y.setter
    def gpu_bin_y(self, value: int) -> None:
        self._gpu_bin_y = int(value)

    @numeric_int(options=GPU_BINNING_OPTIONS)
    @describe(label="GPU Bin X", desc="Software/GPU rectangular binning factor along X (columns).", stream=True)
    def gpu_bin_x(self) -> int:
        return self._gpu_bin_x

    @gpu_bin_x.setter
    def gpu_bin_x(self, value: int) -> None:
        self._gpu_bin_x = int(value)

    @enumerated(options=GPU_BIN_MODE_OPTIONS)
    @describe(label="GPU Bin Mode", desc="Reduction mode for GPU rectangular (Y x X) binning.", stream=True)
    def gpu_bin_mode(self) -> str:
        return self._gpu_bin_mode

    @gpu_bin_mode.setter
    def gpu_bin_mode(self, value: str) -> None:
        self._gpu_bin_mode = str(value).lower()

    @enumerated(options=GPU_FILTER_OPTIONS)
    @describe(label="GPU Pre-Filter", desc="2D spatial filter applied prior to GPU rectangular binning.", stream=True)
    def gpu_pre_filter(self) -> str:
        return self._gpu_pre_filter

    @gpu_pre_filter.setter
    def gpu_pre_filter(self, value: str) -> None:
        self._gpu_pre_filter = str(value).lower()

    @numeric_int(options=GPU_FILTER_SIZE_OPTIONS)
    @describe(label="GPU Filter Size", desc="Kernel size (3 or 5) for GPU pre-filter.", stream=True)
    def gpu_filter_size(self) -> int:
        return self._gpu_filter_size

    @gpu_filter_size.setter
    def gpu_filter_size(self, value: int) -> None:
        self._gpu_filter_size = int(value)

    @property
    @describe(label="Output Frame Size", units="px", stream=True)
    def output_frame_size_px(self) -> IVec2D:
        """Frame dimensions after GPU rectangular (gpu_bin_y x gpu_bin_x) binning."""
        raw = self.frame_size_px
        return IVec2D(y=max(1, raw.y // self._gpu_bin_y), x=max(1, raw.x // self._gpu_bin_x))

    @property
    def has_gpu_binning(self) -> bool:
        """Whether GPU rectangular binning or spatial pre-filtering is active on this camera."""
        return self._gpu_bin_y > 1 or self._gpu_bin_x > 1 or self._gpu_pre_filter != "none"

    def apply_gpu_binning_2d(self, frame: np.ndarray) -> np.ndarray:
        """Apply this camera's configured 2D pre-filter + rectangular (gpu_bin_y x gpu_bin_x) binning."""
        if not self.has_gpu_binning:
            return frame
        return filter_and_bin_2d(
            frame,
            bin_y=self._gpu_bin_y,
            bin_x=self._gpu_bin_x,
            pre_filter=PreFilterType(self._gpu_pre_filter),
            filter_size=self._gpu_filter_size,
            bin_mode=BinningMode(self._gpu_bin_mode),
        )

    def apply_gpu_binning_3d(self, stack: np.ndarray) -> np.ndarray:
        """Apply this camera's configured 2D pre-filter + rectangular binning to a 3D stack."""
        if not self.has_gpu_binning:
            return stack
        return filter_and_bin_volume(
            stack,
            bin_y=self._gpu_bin_y,
            bin_x=self._gpu_bin_x,
            pre_filter=PreFilterType(self._gpu_pre_filter),
            filter_size=self._gpu_filter_size,
            bin_mode=BinningMode(self._gpu_bin_mode),
        )

    @property
    @abstractmethod
    @describe(label="Sensor Size", units="px")
    def sensor_size_px(self) -> IVec2D:
        """Get the size of the camera sensor in pixels."""

    @property
    @abstractmethod
    @describe(label="Pixel Size", units="µm", desc="The size of the camera pixel in microns.")
    def pixel_size_um(self) -> Vec2D:
        """Get the size of the camera pixel in microns."""

    @enumerated(options=list(PIXEL_FMT_TO_DTYPE.keys()))
    @abstractmethod
    @describe(label="Pixel Format", stream=True)
    def pixel_format(self) -> PixelFormat:
        """Get the pixel format of the camera."""

    @pixel_format.setter
    @abstractmethod
    def pixel_format(self, pixel_format: str) -> None:
        """Set the pixel format of the camera."""

    @property
    @describe(label="Pixel Type", stream=True)
    def pixel_type(self) -> Dtype:
        """Get the pixel type of the camera."""
        return PIXEL_FMT_TO_DTYPE[cast("PixelFormat", str(self.pixel_format))]

    @numeric_int(options=BINNING_OPTIONS)
    @abstractmethod
    @describe(label="Binning", stream=True)
    def binning(self) -> int:
        """Get the binning mode of the camera. Integer value, e.g. 2 is 2x2 binning."""

    @binning.setter
    @abstractmethod
    def binning(self, binning: int) -> None:
        """Set the binning mode of the camera. Integer value, e.g. 2 is 2x2 binning."""

    @numeric()
    @abstractmethod
    @describe(label="Exposure Time", units="ms", stream=True)
    def exposure_time_ms(self) -> float:
        """Get the exposure time of the camera in ms."""

    @exposure_time_ms.setter
    @abstractmethod
    def exposure_time_ms(self, exposure_time_ms: float) -> None:
        """Set the exposure time of the camera in ms."""

    @numeric()
    @abstractmethod
    @describe(label="Frame Rate", units="Hz", stream=True)
    def frame_rate_hz(self) -> float:
        """Get the frame rate of the camera in Hz."""

    @frame_rate_hz.setter
    @abstractmethod
    def frame_rate_hz(self, value: float) -> None:
        """Set the frame rate of the camera in Hz."""

    # ==================== Sensor ROI ====================

    @abstractmethod
    def _get_roi(self) -> SensorROI:
        """Read the current ROI from hardware in sensor pixel coordinates."""

    @abstractmethod
    def _set_roi(self, roi: SensorROI) -> None:
        """Apply ROI to hardware. Values are pre-snapped by update_roi."""

    @property
    @abstractmethod
    @describe(label="ROI Grid", stream=True)
    def roi_grid(self) -> ROIGrid:
        """Hardware constraints for the sensor ROI.

        Returns the valid size ranges (min/max/step) for each axis.
        These are dynamic — they may change with binning mode.
        """

    @property
    @describe(label="Sensor ROI", stream=True)
    def roi(self) -> SensorROI:
        """Current sensor ROI in physical sensor pixels (pre-binning)."""
        return self._get_roi()

    @describe(label="Update ROI")
    def update_roi(self, roi: SensorROI, *, snap: bool = True) -> SensorROI:
        """Set sensor ROI. Returns the actual applied ROI.

        Args:
            roi: Desired ROI in physical sensor pixels (pre-binning).
            snap: If True (default), clamp and align to hardware grid.
                  If False, raise ValueError if ROI doesn't conform.
        """
        grid = self.roi_grid
        snapped = roi.snap(grid)
        if not snap and roi != snapped:
            raise ValueError(f"ROI {roi} does not conform to grid {grid}; nearest valid ROI is {snapped}")
        self._set_roi(snapped)
        return self._get_roi()

    @describe(label="Reset ROI")
    def reset_roi(self) -> SensorROI:
        """Reset the sensor ROI to the full sensor."""
        return self.update_roi(SensorROI(w=self.sensor_size_px.x, h=self.sensor_size_px.y))

    @property
    @describe(label="Frame Size", units="px", stream=True)
    def frame_size_px(self) -> IVec2D:
        """Get the image size in pixels (post-binning frame coordinates)."""
        roi = self.roi
        b = int(self.binning)
        return IVec2D(y=roi.h // b, x=roi.w // b)

    @property
    @describe(label="Frame Size", units="MB", stream=True)
    def frame_size_mb(self) -> float:
        """Get the size of the camera image in MB."""
        return (self.frame_size_px.x * self.frame_size_px.y * self.pixel_type.itemsize) / 1_000_000

    @property
    @describe(label="Frame Area", units="µm", stream=True)
    def frame_area_um(self) -> Vec2D:
        """Get the physical frame size in micrometers."""
        roi = self.roi
        return Vec2D(
            x=roi.w * self.pixel_size_um.x,
            y=roi.h * self.pixel_size_um.y,
        )

    @property
    @abstractmethod
    @describe(label="Stream Info", stream=True)
    def stream_info(self) -> StreamInfo | None:
        """Return a dictionary of the acquisition state or None if not acquiring.

        - Frame Index - frame number of the acquisition
        - Input Buffer Size - number of free frames in buffer
        - Output Buffer Size - number of frames to grab from buffer
        - Dropped Frames - number of dropped frames
        - Data Rate [MB/s] - data rate of acquisition
        - Frame Rate [fps] - frames per second of acquisition
        """

    def effective_pixel_size_um(self, magnification: float = 1.0) -> Vec2D:
        """Physical size of each output pixel in µm, accounting for binning."""
        b = int(self.binning)
        return Vec2D(x=self.pixel_size_um.x * b, y=self.pixel_size_um.y * b) / magnification

    @abstractmethod
    def _configure_trigger_mode(self, mode: TriggerMode) -> None:
        """Configure the trigger mode of the camera."""

    @abstractmethod
    def _configure_trigger_polarity(self, polarity: TriggerPolarity) -> None:
        """Configure the trigger polarity of the camera."""

    @abstractmethod
    def _start(self, frame_count: int | None = None) -> None:
        """Begin acquiring ``frame_count`` frames (None = until stopped). Driver primitive for start()."""

    @abstractmethod
    def _allocate_buffer(self) -> None:
        """Allocate capture buffers. Driver primitive for allocate_buffer()."""

    def _free_buffer(self) -> None:
        """Release capture buffers. Override if the driver needs explicit cleanup."""

    def allocate_buffer(self) -> None:
        """Allocate capture buffers (releasing any existing first)."""
        self.free_buffer()
        self._allocate_buffer()
        self._buffer_allocated = True

    def free_buffer(self) -> None:
        """Release capture buffers."""
        self._free_buffer()
        self._buffer_allocated = False

    def start(
        self,
        frame_count: int | None = None,
        trigger_mode: TriggerMode = TriggerMode.ON,
        trigger_polarity: TriggerPolarity = TriggerPolarity.RISING_EDGE,
    ) -> None:
        """Start the camera to acquire a certain number of frames.

        If frame number is not specified, acquires infinitely until stopped.
        Initializes the camera buffer.

        Arguments:
            frame_count: The number of frames to acquire. If None, acquires indefinitely until stopped.
            trigger_mode: The trigger mode to use. Defaults to TriggerMode.ON.
            trigger_polarity: The trigger polarity to use. Defaults to None.
        """
        if not self._buffer_allocated:
            self.allocate_buffer()
        self.trigger_mode = trigger_mode
        self.trigger_polarity = trigger_polarity
        self._configure_trigger_mode(self.trigger_mode)
        self._configure_trigger_polarity(self.trigger_polarity)
        self._start(frame_count)

    def grab_frame(self, *, out: np.ndarray) -> None:
        """Populate a validated caller-owned frame destination."""
        expected_shape = (self.frame_size_px.y, self.frame_size_px.x)
        expected_dtype = np.dtype(self.pixel_type.dtype)
        if out.shape != expected_shape:
            raise ValueError(f"Frame destination shape {out.shape} does not match {expected_shape}")
        if out.dtype != expected_dtype:
            raise TypeError(f"Frame destination dtype {out.dtype} does not match {expected_dtype}")
        if not out.flags.c_contiguous:
            raise ValueError("Frame destination must be C-contiguous")
        if not out.flags.writeable:
            raise ValueError("Frame destination must be writable")
        if not out.dtype.isnative:
            raise ValueError("Frame destination must use native byte order")
        self._grab_frame(out)

    def grab_frame_owned(self) -> np.ndarray:
        """Allocate, populate, and return one independently owned frame."""
        size = self.frame_size_px
        frame = np.empty((size.y, size.x), dtype=self.pixel_type.dtype)
        self.grab_frame(out=frame)
        return frame

    @abstractmethod
    def _grab_frame(self, out: np.ndarray) -> None:
        """Driver primitive that fills a validated destination completely or raises."""

    @abstractmethod
    def stop(self) -> None:
        """Stop the camera."""
