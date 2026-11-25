"""How much RAM and VRAM a ring slot really costs, and the bounds on ring depth."""

from vxlib.dtype import Dtype
from vxlib.vector import UIVec3D

from .dataset import ScaleLevel

MIN_SLOTS = 2
MAX_SLOTS = 4
_F32_ITEMSIZE = 4
_GPU_ENCODE_BATCH_VRAM = 2 * 256 * 1024 * 1024  # d_sub + d_comp mini-batch cap in array/gpu.py


class RingSizingError(RuntimeError):
    """A ring could not be sized. Raised by a ``sizer`` passed to :meth:`OMEZarrWriter.begin_stack`."""


class RingBudgetExceededError(RingSizingError):
    """The geometry cannot fit :data:`MIN_SLOTS` within the caller's memory budget."""


class RingMemoryUnavailableError(RingSizingError):
    """The budget allows the ring, but the machine cannot currently supply the memory."""


def _voxels(shape: UIVec3D) -> int:
    return shape.z * shape.y * shape.x


def _pos_voxels(shape: UIVec3D) -> int:
    return max(1, shape.z) * max(1, shape.y) * max(1, shape.x)


def _shm_bytes(batch_shape: UIVec3D, max_level: ScaleLevel, dtype: Dtype, raw_batch_shape: UIVec3D | None) -> int:
    shm = sum(_voxels(level.scale(batch_shape)) * dtype.itemsize for level in max_level.levels)
    if raw_batch_shape is not None and raw_batch_shape != batch_shape:
        shm += _voxels(raw_batch_shape) * dtype.itemsize
    return shm


def per_slot_bytes(
    batch_shape: UIVec3D,
    max_level: ScaleLevel,
    dtype: Dtype,
    *,
    use_gpu: bool = False,
    shard_shape: UIVec3D | None = None,
    raw_batch_shape: UIVec3D | None = None,
) -> int:
    """Peak host RAM one ring slot needs, in bytes."""
    shm = _shm_bytes(batch_shape, max_level, dtype, raw_batch_shape)
    worker_pyramid = 0
    worker_cast = 0
    if not use_gpu:
        for level in max_level.levels:
            if level != ScaleLevel.L0:
                voxels = _voxels(level.scale(batch_shape))
                worker_pyramid += voxels * _F32_ITEMSIZE
                worker_cast = max(worker_cast, voxels * dtype.itemsize)
    pinned = 2 * _voxels(shard_shape) * dtype.itemsize if use_gpu and shard_shape is not None else 0
    return shm + worker_pyramid + worker_cast + pinned


def gpu_workspace_bytes(
    batch_shape: UIVec3D,
    max_level: ScaleLevel,
    dtype: Dtype,
    *,
    raw_batch_shape: UIVec3D | None = None,
    shard_shape: UIVec3D | None = None,
) -> int:
    """Device VRAM allocated by one ``GpuPyramidWorkspace`` plus optional ``GPUArrayWriter`` buffers."""
    vram = _pos_voxels(raw_batch_shape) * dtype.itemsize if (raw_batch_shape and raw_batch_shape != batch_shape) else 0
    for level in max_level.levels:
        voxels = _pos_voxels(level.scale(batch_shape))
        vram += voxels * (dtype.itemsize + (_F32_ITEMSIZE if level != ScaleLevel.L0 else 0))
    if max_level != ScaleLevel.L0:
        vram += max(1, batch_shape.z) * max(1, batch_shape.y // 2) * max(1, batch_shape.x // 2) * _F32_ITEMSIZE
    if shard_shape is not None:
        shard_bytes = _voxels(shard_shape) * dtype.itemsize
        vram += shard_bytes + min(2 * shard_bytes, _GPU_ENCODE_BATCH_VRAM)
    return vram


def vram_fits_batch(
    batch_shape: UIVec3D,
    max_level: ScaleLevel,
    dtype: Dtype,
    *,
    free_vram_bytes: int,
    raw_batch_shape: UIVec3D | None = None,
    shard_shape: UIVec3D | None = None,
    max_vram_fraction: float = 0.85,
) -> bool:
    """Return True if a single GPU worker's transient VRAM arena fits within ``max_vram_fraction * free_vram_bytes``."""
    if free_vram_bytes <= 0:
        return False
    needed = gpu_workspace_bytes(
        batch_shape, max_level, dtype, raw_batch_shape=raw_batch_shape, shard_shape=shard_shape
    )
    return needed <= int(free_vram_bytes * max_vram_fraction)


def ring_shm_bytes(
    slots: int,
    batch_shape: UIVec3D,
    max_level: ScaleLevel,
    dtype: Dtype,
    *,
    raw_batch_shape: UIVec3D | None = None,
) -> int:
    """Resident shared memory a ring of ``slots`` holds between volumes."""
    return slots * _shm_bytes(batch_shape, max_level, dtype, raw_batch_shape)


def slots_for_budget(
    batch_shape: UIVec3D,
    max_level: ScaleLevel,
    dtype: Dtype,
    *,
    budget_bytes: int,
    available_bytes: int | None = None,
    label: str = "",
    use_gpu: bool = False,
    shard_shape: UIVec3D | None = None,
    raw_batch_shape: UIVec3D | None = None,
) -> int:
    """Deepest ring this geometry can fill within ``budget_bytes``, clamped to ``[MIN_SLOTS, MAX_SLOTS]``."""
    prefix = f"{label}: " if label else ""
    slot_bytes = per_slot_bytes(
        batch_shape, max_level, dtype, use_gpu=use_gpu, shard_shape=shard_shape, raw_batch_shape=raw_batch_shape
    )
    max_slots = budget_bytes // slot_bytes if slot_bytes else 0
    if max_slots < MIN_SLOTS:
        raise RingBudgetExceededError(
            f"{prefix}cannot fit {MIN_SLOTS} batch slots ({slot_bytes:,} B/slot, {budget_bytes:,} B budget). "
            f"Reduce batch_z_shards, shard_z_chunks, max_level, target_shard_gb, or the frame size; "
            f"or raise the budget."
        )
    slots = min(max_slots, MAX_SLOTS)
    needed = slots * slot_bytes
    if available_bytes is not None and available_bytes < needed:
        raise RingMemoryUnavailableError(
            f"{prefix}{slots} batch slots need {needed:,} B but only {available_bytes:,} B is free. "
            f"Another process is using this machine's memory."
        )
    return slots
