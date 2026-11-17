# Smart downsampling in Voxel: handoff

For Malte, from Josh. This is how to add filtering, per-axis binning and noise-bounded compression to
Voxel's GPU writer on this branch (`josh-gpu-preprocessing`). The branch starts from Steve's
`mobi-gpu-omezarr` at `e60ac02a`.

## Goal

Store each volume about **20× smaller than raw 16-bit**, with the loss held below the image's own
noise. The steps, in order:

1. **Median**: removes speckle noise.
2. **Low pass**: smooths away detail finer than a cut-off.
3. **High pass**: removes slow background.
4. **Binning**: combines blocks of voxels into one.
5. **Compression**: noise-bounded rounding, then a lossless codec Neuroglancer can read.

Every step takes **separate settings for z, y and x**. Our voxels are anisotropic: the scan step is
not the camera pixel size. That is the main reason not to use camera binning or the existing 2×2×2
pyramid.

## Reference implementation (CPU, already tuned on real data)

HappyTweaker (HT) runs the same pipeline on the CPU as **Squish**. Its code is the specification for
this work. It is in the `flourish` repo on branch `josh-happy-tweaker-tunnel` (PR
inductivebias/flourish#3201), under `src/flourish/neurodata/happy-tweaker/src/happy_tweaker/`:

| File | What to take from it |
|---|---|
| `squish.py` | The order of the steps, the per-axis settings, the cut-off → σ formula, and the halo |
| `binning.py` | Per-axis binning: reductions, edge handling, bin-centre positions |
| `compress.py` | `noise_sigma`, `lossy` (linear and sqrt domains), codec settings, size and error reports |
| `filters.py` | `_box_median`: a fast median with mirrored edges |
| `test_squish.py`, `test_compress.py` | Reference behaviours to reproduce on the GPU |

HT runs on colfax-2 (`ssh -L 8091:127.0.0.1:8091 josh@colfax-2`, then open http://localhost:8091).
Its Squish tab previews any setting on real data and reports size, bits per voxel and error. Pick
parameters there, then make the GPU output match HT's on the same input.

### The algorithm, exactly

Filters work on float32, one channel at a time, in (z, y, x) voxel order.

1. **Median** over a `(mz, my, mx)` box of voxels, with mirrored edges. Every box size is at least
   1, and at least one is above 1.
2. **Low pass**: a Gaussian blur with one σ per axis. A cut-off `c` (a structure size, in µm or
   voxels; 0 leaves that axis alone) becomes
   `σ = sqrt(ln 2 / 2) / π × c` voxels, at which the filter passes half the signal. The kernel
   reaches 3σ (`truncate=3`), with mirrored edges.
3. **High pass**: `x − GaussianBlur(x, σ_high) + highpass_add`. `σ_high` comes from its own
   per-axis cut-offs by the same formula. `highpass_add` (default 100 grey levels) keeps the
   background above 0 so it isn't clipped.
4. **Binning** by `(fz, fy, fx)` using `mean` (rounded), `median`, `sum` (clipped to the top value),
   `max` or `nearest`. Bins start at the volume's first voxel, so a batch and the whole volume bin
   the same way. A bin cut short at the far edge combines only the voxels it has. The binned voxel
   size is `voxel × f`, and the binned voxel sits at the centre of its block.
5. **Noise-bounded lossy rounding** of the binned image:
   - Noise σ = `1.4826 × median(|d − median(d)|) / √2`, where `d` is the difference between
     neighbouring voxels along the finest axis.
   - `linear` domain: round to steps of `step_noise × σ`.
   - `sqrt` domain (recommended): `y = 2√(x − offset + 3/8)`, where `offset` is the 0.1st
     percentile. Measure σ in `y`, round `y` to steps of `step_noise × σ_y`, then invert:
     `x = (y/2)² − 3/8 + offset`. Rounding in this domain is coarse where the signal is bright and
     noisy and fine where it is dim.
   - The stored values stay in the original grey-level units.
   - A step of 1σ adds about 0.29σ of RMS error, well below the noise.
6. **Codec**: Blosc (zstd or LZ4, with bit shuffle) or zstd. Neuroglancer reads these. JPEG 2000 and
   the other image codecs are out of scope here, because Neuroglancer can't read them.

The halo each batch needs in voxels, per axis (`squish.py`, `Squish.halo`) is:
`median//2 + ceil(3σ_low) + ceil(3σ_high)`.

On a real strip, HT's Squish with median 3×1×3, a low pass of 2×1×2 voxels, binning 4×1×4 and lossy
1σ (sqrt), stored as Blosc LZ4, went from 74.7 GB to 2.16 GB. That is 34.5× in total: 16× from
binning and 2.2× from compression.

## What already exists on this branch (Steve's GPU writer)

```
camera → BatchSlot ring (shared memory, batch_z planes) → worker process
  GPU: keep_bits / dark_floor → 2×2×2 pyramid (mean|max|min|gaussian) → nvCOMP shard encode → Zarr v3 shards
```

| File | Role |
|---|---|
| `omezarr/src/ome_zarr_writer/slot.py` | `_worker_process_and_write_gpu` (around line 334), the per-batch GPU pipeline. **This is where the new steps go.** |
| `omezarr/src/ome_zarr_writer/pyramid_gpu.py` | `GpuPyramidWorkspace.compute_from_host`, the kernels, and `preprocess_volume` (keep_bits and dark_floor) |
| `omezarr/src/ome_zarr_writer/array/gpu.py` | `GPUArrayWriter`: nvCOMP Blosc LZ4/zstd, zstd and gzip shard encoding |
| `omezarr/src/ome_zarr_writer/writer.py` | `WriterSettings` (user settings), `WriterConfig`, `batch_z`, shard sizing |
| `omezarr/src/ome_zarr_writer/dataset.py` | `ScaleLevel` (fixed `factor = 2**level` on every axis) and the multiscale metadata (`multiscale`, around line 732) |
| `omezarr/src/ome_zarr_writer/sizing.py` | RAM and GPU memory sizing for the ring |
| `src/vxl/devices/camera/base.py` | Builds `WriterConfig` from settings plus camera geometry (`voxel_size` from z step and magnification) |
| `web/station/src/lib/core/model/types.ts`, `web/station/src/routes/stations/[stationId]/instruments/[instrumentId]/+page.svelte` | The Format panel, which already exposes `keep_bits`, `dark_floor` and so on |
| `omezarr/tests/test_gpu_pipeline.py` | GPU tests. All 58 pass on colfax-2's RTX 5090. |

## Gaps

1. **No filters.** Nothing does a median, low pass or high pass during acquisition.
2. **The pyramid is isotropic.** `ScaleLevel.factor` halves z, y and x together, so anisotropic data
   stays anisotropic at every level. There is no way to bin y and x by 4 and leave z alone.
3. **Batches don't overlap.** Each batch of `batch_z` planes is processed on its own. The Gaussian
   pyramid clamps at batch edges (`_cuda_gauss_z_cast`), which leaves small seams. Our filters reach
   much further than that, so they need a z halo from the neighbouring batches.
4. **Rounding isn't tied to the noise.** `keep_bits` rounds to a power of two in grey levels, the same
   step at every brightness. Noise-bounded rounding sets the step from the measured noise, and in
   the sqrt domain the step follows brightness.
5. **GPU memory sizing doesn't catch large frames.** A simulated run with the `simulated-local`
   camera (10640 × 14192, batch_z 64) failed with `CUDA_ERROR_OUT_OF_MEMORY` on the 32 GB RTX 5090.
   The ring sizing logged `gpu=True` but didn't stop the run. MOBI's Kinetix frames (3200 × 3200) are
   about 15× smaller, so this may not affect us, but the sizing should refuse a batch that won't fit.
6. **A GPU setting that can't be seen.** `WriterSettings.use_gpu` defaults to the environment variable
   `VOXEL_WRITER_GPU` when the instrument is *created*, then stays saved in the instrument's
   `config.yaml` and `state.json`. The UI doesn't show it. An instrument created without the
   variable stays on the CPU even after the variable is set.
7. **To check:** for Blosc, `_parse_inner_codecs` (`array/gpu.py`) sets the byte-shuffle header flag
   (`0x01`) whenever shuffle isn't `noshuffle`, even when the metadata asks for `bitshuffle` (`0x04`).
   Decoders follow the header, so the data still reads correctly, but it may compress less than bit
   shuffle would. I read this from the code and haven't tested it.

## Proposed design

### Settings (all explicit, none hidden)

Add a `squish` block to `WriterSettings`. It is broadcast to every camera like the other settings,
and the Format panel shows every field:

```python
class SquishSettings(BaseModel):
    median: tuple[int, int, int] | None = None              # (z, y, x) box in voxels
    lowpass: tuple[float, float, float] | None = None       # (z, y, x) cut-offs; 0 = leave that axis alone
    lowpass_units: Literal["um", "voxels"] = "um"
    highpass: tuple[float, float, float] | None = None
    highpass_units: Literal["um", "voxels"] = "um"
    highpass_add: float = 100.0
    binning: tuple[int, int, int] = (1, 1, 1)                # (z, y, x)
    binning_kind: Literal["mean", "median", "sum", "max", "nearest"] = "mean"
    lossy_step_noise: float | None = None                    # in noise σ; None = lossless
    lossy_domain: Literal["linear", "sqrt"] = "sqrt"
    noise_sigma: float | None = None                         # fixed σ (see below); None = measure
```

Record the settings in the dataset's Zarr attributes, as HT does in its run records. The data then
says how it was made.

### Where it runs

Do the work in `_worker_process_and_write_gpu`, on the GPU, before `compute_from_host`:

```
L0 batch (+ z halo) → median → low pass → high pass → crop halo → bin (fz, fy, fx) → lossy round
  → this becomes the stored L0 → pyramid levels from it → nvCOMP encode
```

- **The binned image is the stored L0.** Its shape is `ceil(n / f)` per axis and its voxel size is
  `voxel × f`. Update `WriterConfig.volume_shape` and `voxel_size` to match, along with the ring and
  shard sizing.
  - The bin-centre translation is `voxel × (f − 1) / 2`. `dataset.multiscale` already uses that form
    for levels; make it per axis.
  - Check how the shear and deskew affine (`shear_xz`, `shear_yz`) combines with a per-axis scale.
- **Per-axis pyramid levels.** Replace `ScaleLevel.factor` (one int) with a factor per level and axis,
  e.g. `[(1,2,2), (1,2,2), (2,2,2), …]`. Halve an axis only while its voxel is at most half the
  coarsest axis, so levels move toward isotropic. The kernels in `pyramid_gpu.py` and the
  `chunk_shape`/`shard_shape` rules (minimum chunk edge 64) must handle a factor of 1 on an axis.
- **The z halo between batches.** Every batch needs `halo_z` planes from before and after it.
  - Option A: keep the last `halo_z` planes of the previous batch on the GPU, and hold back each
    batch's last `halo_z` planes until the next batch arrives. This adds one batch of delay; the
    final batch flushes with mirrored edges.
  - Option B: have the ring slots overlap by `2 × halo_z` planes.
  - With either option, `batch_z` must be a multiple of the z bin factor, so bins never cross a
    batch boundary.
  - y and x need no halo: each batch holds whole frames.
  - Test: the batched output must equal the whole volume filtered at once, as HT's
    `test_squish.py` checks for regions.
- **Noise σ must be the same for the whole volume.** HT measures σ and the sqrt offset separately
  for each region. If Voxel does that per batch, the rounding step changes between batches and shows
  as banding. Instead:
  - fix `noise_sigma` and the offset from a calibration (a dark stack, or Kinetix gain and read
    noise); or
  - measure them on the first batch and record them in the attributes.
  The setting must say which.
- **Kernels.**
  - Median: a small fixed-size box, mz×my×mx ≤ 27 for 3×3×3, works as one thread per voxel with a
    sorting network.
  - Gaussians: separable, one 1-D pass per axis (cupy's `cupyx.scipy.ndimage.gaussian_filter` and
    `median_filter` could be a quick first version, if adding cupy is acceptable).
  - Binning: a per-axis reduce.
  - Lossy: an element-wise kernel. The σ estimate needs a median of absolute differences, which is
    simplest done on a subsample.
- **GPU memory.** A Kinetix batch of 64 × 3200 × 3200 in float32 is 2.6 GB per working copy. The
  median, low pass and high pass need about three copies plus the halo. Two cameras means two workers
  on one GPU. Add all this to `sizing.py` so it refuses before allocating.

### UI

Add a "Squish" section to the Format panel with the same layout as HT: expandable steps from top to
bottom in the order they run, each with an on/off switch and z/y/x fields. Show the expected size in
bits per voxel and as a percentage of raw. HT's `filter_tab.py` (`_axes_inputs`, `_show_squish`) is a
working model.

## Tests to write

- GPU output equals HT's CPU `Squish` on the same block, within float tolerance, for every step on
  its own and together. This is the main correctness test.
- A volume written in batches equals the whole volume filtered at once (the halo).
- Per-axis binning: shapes, edge bins, and `voxel_size`/translation in the metadata.
- Lossy: the RMS error is about `0.29 × step_noise × σ`, and stored values are on the step grid.
- GPU memory sizing refuses a batch that won't fit, instead of failing with out-of-memory.

## Test setup on colfax-2 (RTX 5090, 32 GB, CUDA 13)

- **Code:** `~/voxel` (josh's account), copied from Josh's laptop checkout of this branch.
- **Start it:**
  ```bash
  ssh josh@colfax-2 'tmux kill-session -t voxel; tmux new-session -d -s voxel "cd ~/voxel && VOXEL_WRITER_GPU=1 uv run vxl serve 2>&1 | tee ~/voxel-serve.log"'
  ```
  Then tunnel to it: `ssh -N -L 8001:127.0.0.1:8000 josh@colfax-2` and open http://localhost:8001.
- **Station and instrument:** station `colfax-2-gpu`, instrument `simulated-local`. `use_gpu` is set
  to true in `~/.voxel/instruments/simulated-local.voxel/{config.yaml,state.json}`; the originals
  are saved as `*.bak-gpu`.
- **Frame size:** set the Sensor ROI to 3200 × 3200 to match a Kinetix before timing anything. The
  default simulated sensor runs out of GPU memory (gap 5).
- **CPU baseline:** 21 frames, 7.2 GB raw. The writer spent collect 4.1 s, process 0.6 s and
  flush 2.1 s, and 223 MB ended up on disk (the simulated frames are mostly empty).

## Open questions

- Which GPU is in the acquisition computer? Timings from a 5090 only carry over if it's similar.
- What are MOBI's voxel sizes: the scan step (0.5 µm in the X-scan example) and the camera pixel at
  the sample? These set the default per-axis binning and the pyramid factors.
- Should squishing happen in raw camera axes, before deskew, or after? It's simplest before, with
  the shear kept in the metadata. Check that Neuroglancer still places the binned data correctly.
- Should the raw L0 also be kept, e.g. for a calibration run or a subset of tiles?
