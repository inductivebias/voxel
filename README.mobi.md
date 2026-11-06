# Mobi: hardware integration and continuous scanning for Voxel

`mobi` is the maintained integration branch of
[inductivebias/voxel](https://github.com/inductivebias/voxel), based on
[AllenNeuralDynamics/voxel](https://github.com/AllenNeuralDynamics/voxel).
It brings additional microscope hardware into Voxel, adds encoder-paced continuous acquisition,
and strengthens cleanup and remote control. See the [Voxel README](README.md) for the application,
installation, and simulated microscope.

## Why these changes

Microscope timing and resource ownership have consequences beyond software state:

- A stage controller moves whole native steps. Rounding a requested distance can change the actual
  plane spacing while metadata still describes the requested spacing. The Zaber TTL driver rejects
  rounding errors greater than 0.1% and steps below native resolution.
- Encoder edges describe distance traveled. Dividing those edges can pace exposures during one
  continuous pass, avoiding a separate move-and-settle operation for every plane. The planner
  checks whole-edge spacing, trigger re-arm time, speed, acceleration, and stopping distance before
  starting. This is a timing strategy, not a measured throughput claim.
- Cancelling an asynchronous task does not stop a hardware call already running in a thread.
  Construction and capture work must finish before their devices or buffers can be released.
- Two cameras can have different image dimensions and preview settings. Each preview keeps its
  own rendering state and appears alongside the other camera.
- A web control request can change physical device state. The server defaults to loopback;
  direct remote access requires authentication and validates browser origins and host headers.

## What `mobi` adds

| Area | Changes |
| --- | --- |
| Device support | Thorlabs FW212C filter wheel; Zaber shared controller, axes, TTL stepping and continuous scans; Teledyne Photometrics Kinetix cameras; Coherent OBIS and Omicron LuxX laser sources |
| Diagnostics | Passive Zaber identity/status probe and staged PVCAM diagnostics; opt-in device tests |
| Preview | Independent side-by-side camera previews and physical stage-axis labels |
| Continuous acquisition | External-edge division for signal generation, leased NI counter/DI resources, continuous-pass planning and execution, encoder/exposure recording, and `scan.json` output |
| Ownership and cleanup | Shared device construction, retained ownership after failed cleanup, cancellation-safe thread completion, reverse dependency shutdown, and propagated local/remote close failures |
| Bounded operations | Serial read/drain/write limits, capture and storage deadlines, bounded uploads, and AO shutdown even when counter shutdown fails |
| Configuration | Validated instrument names, staged installation, unique replacement backups, and rollback after failed promotion |
| Web access | Loopback binding, host/origin validation, and authenticated HTTP/WebSocket sessions for direct remote serving |
| Verification | Physical tests excluded from ordinary checks; optional Qt, benchmark, and vendor SDK type checks separated from the core environment |

Drivers live in [`drivers/src/vxl_drivers`](drivers/src/vxl_drivers/), with their dependencies and
test entry points in the [driver catalog](drivers/README.md). Continuous acquisition is optional;
existing stepwise acquisition remains available. Its planner is in
[`src/vxl/instrument/scanning.py`](src/vxl/instrument/scanning.py).

## X-Scan tasks

With `hal.continuous_scan` configured, **+ X-Scan** sits beside Position and Area in the
Run page. Select the physical axis, a positive scan window (end exclusive), and sampling in
micrometres. A compact list under each profile shows the nominal frame rate, stage speed, frame
count, scan duration, runway, and total travel. New tasks use one second of runway per side.
The hardware configuration's `axis` identifies the axis
whose encoder is wired to the DAQ; it defaults to `z` for existing continuous-volume setups.
There is no need to remap the physical stage axes.

X-Scan tasks default to automatic camera timing (`scan.camera_rate_fraction: 0.95`). After
applying each profile, the controller queries each camera's acquisition rate limit, selects the
slowest, and applies the fraction and configured stage-speed tolerance. Speed is sampling times
that safe rate, further limited by the DAQ waveform duration, re-arm margin, and one sample tick.
Changing the DAQ rest time alone does not set automatic scan speed. Set `camera_rate_fraction`
to `null` to retain manual DAQ-period timing. Legacy continuous Z volumes keep manual timing.

The Kinetix driver prepares a two-frame PVCAM continuous buffer without starting exposure, then
reads the resulting readout and available clearing/trigger delays. First Row uses overlapping
exposure/readout timing; other expose-out modes use the conservative sum. Timing is cached until
camera settings change and unavailable during live acquisition. A missing or invalid limit prevents
automatic scanning. These SDK limits do not guarantee storage throughput. Task timing estimates
require the selected profile to be applied without divergent camera settings; actual timing is
queried again before motion. The saved `scan.json` records the executed speed.

Runway distance is speed times runway time, rounded outward to encoder edges in automatic mode.
For example, a manual 12.5 ms period with 0.5 µm sampling yields 40 µm/s. A 34–40 mm window has 12,000 samples and, with one second
of runway per side, travels from 33.960 to 40.040 mm. Runway time specifies distance at cruising
speed, not the exact time spent accelerating.

The other axes must already be stationary within 1 µm of the saved hold positions. Use their
crosshair buttons to adopt the current positions. X-Scan tasks never issue motion commands to
those held axes. Profiles must have camera triggers and no stage-step waveforms. Preflight
checks the configured encoder axis, spacing, runway, travel limits, exposure time, and DAQ
re-arm budget. Camera readout throughput still needs to be verified on the actual cameras.

Acquisition arms the recorder and all profile cameras before encoder-paced motion. Selected
lasers use the profile's independent illumination modes. Cancellation halts the scan axis,
inhibits laser output, and restores scan settings. The manifest's `scan` and each volume's
`scan.json` retain the physical axis and scan window. OME-Zarr's first spatial dimension remains
the image-stack depth, with the sample spacing; it does not denote physical stage Z for an X/Y
sweep. Interpret it with the manifest and recorded exposure positions.

The encoder has an unknown initial phase: nominal sample locations are not exact absolute
position comparisons. `scan.json` records measured exposure locations and validates their
spacing and counts. Existing Z-volume continuous configurations retain their prior behavior;
Position/Area tasks on an X/Y continuous-scan configuration retain stepwise Z behavior.

## Status and verification

The former `x-scan` and `steve-harden-voxel` work is consolidated into `mobi`, preserving the
individual commits for review and upstream proposals. Hardware-free tests cover planning,
acquisition, shutdown, driver protocol boundaries, storage, and web access. The integrated
continuous-scan path still needs supervised commissioning on physical hardware; merging it does
not establish optical timing, motion clearance, or device compatibility on another instrument.

With the development environment and required extras installed:

```bash
uv run ruff check .
uv run ruff format --check .
uv run basedpyright
uv run pytest -m "not slow and not hardware"
```

Run slow integration tests separately. Physical tests under `drivers/tests/hardware/` are opt-in
and require device selection through environment variables; motion, emission, and acquisition
checks require the corresponding operator opt-in. Optional SDK and GUI modules use
`pyrightconfig.optional.json` in environments containing their dependencies.

## Maintaining the fork

Keep generic drivers and application changes here. Instrument identities, wiring, local network
configuration, commissioning records, and acquired data belong in a private rig repository.
The public fork must remain usable without those files.

`main` tracks upstream; `mobi` carries focused integration commits. Prefer upstreaming each driver
or independent fix, then remove the equivalent fork change when upstream provides it. Keep trial
branches temporary and fold validated code into `mobi`; record physical commissioning status
separately from branch membership. Avoid a parallel patch-overlay build or a second implementation
of acquisition when the normal Voxel path can serve the same workflow.


## Optional position window for encoder-paced scans

A Zaber scanner can reserve a controller digital output with `scan_window_output` and three
`scan_window_triggers` (entry, upper exit, lower exit). The NI counter trigger must also name
its corresponding TTL-level PFI input as `window_source`. Both sides must be configured together;
the existing encoder A/B signals remain continuous for position recording. Dedicated trigger IDs
and the output must not be shared with another axis's stepping/status configuration.

The gate opens once when measured position reaches the scan start, closes at the exclusive scan
end, and closes if position reverses below the start. It cannot reopen during that pass. Initialization,
normal finish, cancellation, failure cleanup, and device close disable the gate. This is controller
trigger logic, not a process-independent safety interlock or a direction-aware position comparator.
Reversals within the window can still affect encoder-A edge division and are checked by the recorder.

NI 6738 pause gating requires a continuous counter task. Windowed acquisitions therefore validate
exact AO cycle and exposure counts after the pass rather than relying on a finite counter pulse limit.
The divider starts two encoder edges after window entry (the supported minimum), introducing an
encoder-phase offset relative to the boundary; recorded exposure positions remain authoritative.
The window and committed gate input are retained in scan metadata. A physical gate/polarity and
position-timing test is required before enabling this optional mode on an instrument.
