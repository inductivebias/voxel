"""Camera device handle with typed methods."""

from pathlib import PurePosixPath
from typing import TYPE_CHECKING, Any, cast

from ome_zarr_writer import WriterSettings
from pydantic import TypeAdapter
from rigup.device.handle import Adapter, DeviceProperty
from vxl_records import DatasetLocation, StorageSpec
from vxlib.coalescer import Coalescer
from vxlib.vector import IVec2D, Vec2D, Vec3D

from rigup import DeviceHandle
from vxl.devices.camera.base import (
    Camera,
    CaptureState,
    SensorROI,
    SnapshotFrameInfo,
    StorageStatus,
    TriggerMode,
    TriggerPolarity,
)
from vxl.preview import PreviewViewport

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from vxl.instrument.snr import CameraSnrAnalysis

_DATASET_LOCATION_ADAPTER = TypeAdapter(DatasetLocation)
_REMOTE_STORES_ADAPTER = TypeAdapter(dict[str, str])


class CameraHandle(DeviceHandle[Camera]):
    """Camera handle with typed methods for preview operations.

    Works with both local and remote cameras - the transport handles
    the communication details.

    Viewport updates are coalesced: ``camera.preview_viewport.update(v)`` collapses rapid
    updates to the latest with one RPC in flight.
    """

    def __init__(self, adapter: Adapter[Camera]) -> None:
        super().__init__(adapter)
        self.preview_viewport: Coalescer[PreviewViewport] = Coalescer(drain=self._set_preview_viewport)
        self.roi: DeviceProperty[SensorROI] = self.props.property("roi", SensorROI.model_validate)
        self.frame_area_um: DeviceProperty[Vec2D] = self.props.property("frame_area_um", self._parse_vec2d)
        self.pixel_size_um: DeviceProperty[Vec2D] = self.props.property("pixel_size_um", self._parse_vec2d)
        self.sensor_size_px: DeviceProperty[IVec2D] = self.props.property("sensor_size_px", self._parse_ivec2d)
        self.ready_for_batch: DeviceProperty[bool] = self.props.property("ready_for_batch", bool)

    async def start_preview(
        self,
        trigger_mode: TriggerMode = TriggerMode.ON,
        trigger_polarity: TriggerPolarity = TriggerPolarity.RISING_EDGE,
    ) -> str:
        """Start camera preview mode.

        Args:
            trigger_mode: Trigger mode (default: TriggerMode.ON)
            trigger_polarity: Trigger polarity (default: TriggerPolarity.RISING_EDGE)

        Returns:
            Topic name where preview frames will be published.
        """
        result = await self.call("start_preview", trigger_mode, trigger_polarity)
        if isinstance(result, str):
            return result
        if isinstance(result, dict):
            raise RuntimeError(f"{self.uid} failed to start preview: {result.get('msg', result)}")
        raise RuntimeError(f"{self.uid} failed to start preview: returned unexpected type {type(result)}: {result}")

    async def stop_preview(self) -> None:
        """Stop camera preview mode."""
        await self.call("stop_preview")

    async def close_preview_updates(self) -> None:
        """Stop and await the handle-owned preview update workers."""
        await self.preview_viewport.close()

    async def reset_preview_stream(self) -> str:
        """End the current preview source stream and return its replacement identity."""
        result = await self.call("reset_preview_stream")
        if not isinstance(result, str):
            raise RuntimeError(f"{self.uid} returned an invalid preview source-stream identity")
        return result

    async def _set_preview_viewport(self, viewport: PreviewViewport) -> None:
        """Apply a preview viewport to the camera — drain for ``preview_viewport``."""
        await self.call("update_preview_viewport", viewport)

    async def check_writable(self, storage: StorageSpec) -> StorageStatus:
        """Preflight: prove this camera's node can write ``storage``. Raises if not. Returns the
        node's storage status (host, resolved root, free bytes)."""
        result = await self.call("check_writable", storage=storage)
        return StorageStatus.model_validate(result)

    async def remote_stores(self) -> dict[str, str]:
        """Return this camera host's configured object stores as name-to-fingerprint mappings."""
        return _REMOTE_STORES_ADAPTER.validate_python(await self.call("remote_stores"))

    async def open_stack(
        self,
        *,
        storage: StorageSpec,
        subpath: PurePosixPath,
        num_frames: int,
        z_step: float,
        magnification: float,
        settings: WriterSettings,
        voxel_size_yx: tuple[float, float] | None = None,
        origin: Vec3D | None = None,
        shear_xz: float = 0.0,
        shear_yz: float = 0.0,
        channel_label: str = "channel 0",
        channel_color: str = "FFFFFF",
        darkfield: dict[str, Any] | None = None,
    ) -> DatasetLocation:
        """Prepare camera and writer for a stack acquisition; return its durable dataset location.

        ``storage`` is the run's logical destination and ``subpath`` the dataset's relative location
        under it (the node resolves it against machine-local storage); ``settings`` are the broadcast
        output-format knobs. ``magnification`` converts the sensor-space pixel size into the
        sample-space lateral voxel size unless overridden by ``voxel_size_yx``. OME-Zarr metadata
        remains authoritative for dataset geometry.
        """
        result = await self.call(
            "open_stack",
            storage=storage,
            subpath=subpath,
            num_frames=num_frames,
            z_step=z_step,
            magnification=magnification,
            settings=settings,
            voxel_size_yx=voxel_size_yx,
            origin=origin,
            shear_xz=shear_xz,
            shear_yz=shear_yz,
            channel_label=channel_label,
            channel_color=channel_color,
            darkfield=darkfield,
        )
        return _DATASET_LOCATION_ADAPTER.validate_python(result)

    async def update_dataset_darkfield(
        self,
        *,
        storage: StorageSpec,
        subpath: PurePosixPath,
        darkfield: dict[str, Any],
    ) -> None:
        """Update ``attributes.darkfield`` in the root ``zarr.json`` of ``<subpath>.ome.zarr``."""
        await self.call("update_dataset_darkfield", storage=storage, subpath=subpath, darkfield=darkfield)

    async def close_stack(self) -> None:
        """Start draining and closing the writer in the background. Poll capture_state until CLOSED."""
        await self.call("close_stack")

    async def release_writer(self) -> None:
        """Free the writer ring (workers + shared memory); the next open_stack re-allocates it cold."""
        await self.call("release_writer")

    async def begin_batch(self, num_frames: int) -> None:
        """Arm and start grabbing num_frames frames in the background. Poll capture_state for completion."""
        await self.call("begin_batch", num_frames)

    async def captured_frames(self) -> int:
        """Number of frames collected in the current stack."""
        return int(await self.call("captured_frames"))

    async def writer_throughput_gb_s(self) -> tuple[float, float | None] | None:
        """Live writer throughput ``(raw_gb_s, compressed_gb_s)``, or ``None``."""
        res = await self.call("writer_throughput_gb_s")
        if isinstance(res, (list, tuple)) and len(res) == 2 and res[0] is not None:
            return (float(res[0]), float(res[1]) if res[1] is not None else None)
        return None

    async def capture_state(self) -> CaptureState:
        """Lifecycle of the most recent begin_batch/close_stack. Raises if the task failed."""
        result = await self.call("capture_state")
        return CaptureState(result)

    async def update_roi(self, roi: SensorROI) -> SensorROI:
        """Set sensor ROI. Returns the actual applied ROI (may differ due to alignment)."""
        result = await self.call("update_roi", roi)
        return SensorROI.model_validate(result)

    async def reset_roi(self) -> SensorROI:
        """Reset the sensor ROI to the full sensor. Returns the applied ROI."""
        result = await self.call("reset_roi")
        return SensorROI.model_validate(result)

    async def arm_capture(
        self,
        num_frames: int = 1,
        trigger_mode: TriggerMode = TriggerMode.ON,
        trigger_polarity: TriggerPolarity = TriggerPolarity.RISING_EDGE,
    ) -> None:
        """Arm the camera for ``num_frames`` hardware-triggered frames."""
        await self.call("arm_capture", num_frames, trigger_mode, trigger_polarity)

    async def capture_frames_raw(self, num_frames: int = 1, *, slot_key: str = "snapshot") -> int:
        """Grab ``num_frames`` armed frames into RAM and immediately stop the camera."""
        return int(await self.call("capture_frames_raw", num_frames, slot_key=slot_key))

    async def process_snapshot(
        self,
        *,
        slot_key: str = "snapshot",
        save_path: str | None = None,
        pixel_size_um: tuple[float, float] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> SnapshotFrameInfo:
        """Process a buffered raw snapshot frame, optionally save as TIFF, and emit preview."""
        return SnapshotFrameInfo.model_validate(
            await self.call(
                "process_snapshot",
                slot_key=slot_key,
                save_path=save_path,
                pixel_size_um=pixel_size_um,
                metadata=metadata,
            )
        )

    async def analyze_snr_buffer(
        self,
        *,
        output_dir: str,
        channel_id: str,
        profile_id: str,
        pixel_size_um: tuple[float, float],
        exposure_ms: float,
    ) -> "CameraSnrAnalysis":
        """Analyze buffered signal + dark frames and return stationary SNR + photobleaching metrics."""
        from vxl.instrument.snr import CameraSnrAnalysis  # noqa: PLC0415

        return CameraSnrAnalysis.model_validate(
            await self.call(
                "analyze_snr_buffer",
                output_dir=output_dir,
                channel_id=channel_id,
                profile_id=profile_id,
                pixel_size_um=pixel_size_um,
                exposure_ms=exposure_ms,
            )
        )

    @staticmethod
    def _parse_vec2d(value: object) -> Vec2D:
        """Parse a Vec2D property, handling its "y,x" string serialization over ZMQ."""
        if isinstance(value, Vec2D):
            return value
        if isinstance(value, str):
            return Vec2D.from_str(value)
        if isinstance(value, (list, tuple)):
            seq = cast("Sequence[int | float]", value)
            return Vec2D(y=seq[0], x=seq[1])
        mapping = cast("Mapping[str, int | float]", value)
        return Vec2D(y=mapping["y"], x=mapping["x"])

    @staticmethod
    def _parse_ivec2d(value: object) -> IVec2D:
        """Parse an IVec2D property, handling its "y,x" string serialization over ZMQ."""
        if isinstance(value, IVec2D):
            return value
        if isinstance(value, str):
            return IVec2D.from_str(value)
        if isinstance(value, (list, tuple)):
            seq = cast("Sequence[int]", value)
            return IVec2D(y=seq[0], x=seq[1])
        mapping = cast("Mapping[str, int]", value)
        return IVec2D(y=mapping["y"], x=mapping["x"])
