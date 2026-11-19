# Plan: Xbox controller jogging for the Zaber stages

Status: **implemented on this branch; awaiting commissioning on Mobi.** Branch: `xbox-controller-integration`
(from `mobi`). The decisions below settle the open questions; where the implementation differs from the original
proposal, the decisions win.

## Decisions (review of 2026-11-19)

| Question | Decision | Implemented as |
| --- | --- | --- |
| Where the controller connects | Station workstation, USB | Browser Gamepad API on the workstation (`localhost` is a secure context, so no HTTPS needed) |
| Axes | Match Voxel | Sticks drive Voxel's logical x/y/z, in the directions the stage view draws them (stick up = +Y, right stick up = +Z), following `StageOrientation` signs; no separate invert settings |
| Enabling | One press of a bumper arms it; obvious in the GUI; always off during acquisition | **Left bumper toggles** jogging. A **Jog ON/OFF** button next to the stage position boxes turns green with a ring, and the stage controls get a green outline, while armed. Starting an acquisition halts any jog and switches it off; it cannot be re-armed until the capture ends |
| Speed | Small deflection slow, full deflection fastest, cap 2 mm/s | Dead zone 0.12, then `0.25·m + 0.75·m³` response; full stick = **2 mm/s**, enforced as a hard server-side cap (`MAX_JOG_SPEED_UM_S`) |
| Jog during live preview | Yes | Allowed whenever no capture is running |

### Controls

| Control | Action |
| --- | --- |
| Left bumper (press once) | Toggle jogging on/off |
| Left stick | X/Y, speed proportional to deflection |
| Right stick, vertical | Z |
| B | Halt all axes |
| Jog button in the stage footer | Same toggle as the left bumper; shows ON/OFF/locked, a pulsing dot while moving, and the controller status in its tooltip |

### Where the code lives

- `drivers/src/vxl_drivers/axes/zaber.py` — `ZaberLinearAxis.move_vel` (Zaber `move velocity`, stopped by the controller's soft limits).
- `src/vxl/devices/axes/continuous/base.py` — `ContinuousAxis.move_vel` hook and the `jog` controller command (refused while TTL stepping or scanning); `simulated.py` implements it.
- `src/vxl/instrument/jog.py` — `StageJog`: enable switch, speed scaling and cap, 300 ms watchdog, single-client ownership, lockout.
- `src/vxl/instrument/core.py` — wiring, `set_jog_enabled`, `jog_stage`, lockout on acquisition start and close.
- `src/vxl/web/realtime.py` — `stage.jog` messages on the state WebSocket; disconnect stops and disarms. `router.py` — `POST …/instrument/stage/jog {enabled}`.
- `web/station/src/lib/core/gamepad/` — stick mapping (with tests) and the Gamepad API reader; `shell/JogToggle.svelte` — the button.

The sections below are the original proposal, kept for reference.

## Goal

An operator holds an Xbox controller and drives the stage directly:

| Stick | Motion |
| --- | --- |
| Left stick, horizontal | Stage **X** |
| Left stick, vertical | Stage **Y** |
| Right stick, vertical | Stage **Z** (height) |

Stick deflection sets **speed**, not position: a small push creeps, a full push moves at the
configured jog speed, and releasing the stick stops the axis. This works the same with the
simulated microscope and with the real Zaber axes on Mobi.

## What exists today (on `mobi`)

- **Driver.** `drivers/src/vxl_drivers/axes/zaber.py` — `ZaberLinearAxis(ContinuousAxis)` wraps one
  `zaber_motion` axis behind the shared `ZaberHub` connection and lock. It offers `move_abs`,
  `move_rel`, `go_home`, `halt`, `is_moving`, speed/acceleration/S-curve settings, and the TTL
  stepper and continuous scanner. It has **no velocity (jog) move**. The controller itself
  enforces `limit.min`/`limit.max` on every move.
- **Device interface.** `src/vxl/devices/axes/continuous/base.py` — abstract `ContinuousAxis`
  (`move_abs`, `move_rel`, `go_home`, `halt`, `await_movement`) and `ContinuousAxisController`,
  which tracks `scan_mode` (`IDLE` / `TTL_STEPPING` / `CONTINUOUS`). `SimulatedContinuousAxis`
  in `src/vxl/devices/axes/simulated.py` stands in when hardware is offline.
- **HAL.** `src/vxl/hal/core.py` — `Stage(x, y, z)` of `ContinuousAxisHandle`s; logical axes are
  named in `StageAxes` (`src/vxl/hal/topology.py`) with optional operator labels.
- **Instrument.** `src/vxl/instrument/core.py` — `Instrument.move_stage(x, y, z)` moves
  absolute positions under the instrument lock and refuses during a capture
  (`_ensure_not_busy`).
- **Web.** The station state WebSocket (`src/vxl/web/realtime.py`) already accepts binary
  (msgpack) client→server control messages — today only `PreviewViewportUpdate`. The Svelte
  stage model (`web/station/src/lib/core/model/stage.svelte.ts`, `device.svelte.ts`) moves
  axes with `move_abs` and halts with `halt`; `shell/StageAxes.svelte` shows positions and a
  halt button.
- No gamepad or host-joystick input exists anywhere in the application. (The ASI Tiger
  `joystick.py` is the ASI controller's own hardware joystick and is unrelated.)

## Recommended approach

Read the controller **in the browser** with the standard Gamepad API, and send a small
**velocity (jog) command** to the server over the existing state WebSocket. The server turns it
into Zaber velocity moves, with a dead-man watchdog that halts the stage whenever commands stop.

```
Xbox controller ──USB/BT──► operator's browser (Gamepad API, ~30 Hz poll)
   └─ dead zone + response curve + speed scaling
        └─ {"jog": [vx, vy, vz], "seq": n}  ── state WebSocket ──►  Instrument.jog_stage()
                                                                     ├─ refuse if capturing/scanning
                                                                     ├─ clamp to jog speed limits
                                                                     ├─ axis.move_vel(v) / halt()
                                                                     └─ watchdog: no update in 300 ms → halt
```

Why the browser rather than a Python input library on the server:

- The controller plugs into the computer the operator is sitting at. With the web UI that may
  not be the station host; the browser is always where the operator is.
- No new native dependencies (SDL/pygame, `inputs`, XInput) and no OS-specific device access
  on the station host. Xbox controllers report the Gamepad API's **standard mapping** in
  Chrome, Edge, and Firefox on Windows, macOS, and Linux.
- Jog commands ride the WebSocket and authentication the UI already uses, and the jog stops
  when the operator closes the tab or the connection drops.

Alternative (not recommended unless needed): a `vxl gamepad` process on the station host that
reads the controller with SDL and calls the same `Instrument.jog_stage` through the REST API.
It would only make sense for a headless, browser-free setup; the server side below is shared
by both, so it can be added later without rework.

Browser caveat: Chrome exposes the Gamepad API only in a **secure context**. `http://localhost`
qualifies; a remote station must be served over HTTPS (or accessed through an SSH tunnel to
localhost) for the controller to be visible.

## Implementation steps

### 1. Driver: velocity moves (`vxl_drivers.axes.zaber`)

- Add `ContinuousAxis.move_vel(velocity: float) -> None` to the base class, with a default that
  raises `NotImplementedError`, plus a `jog_available` capability flag on the controller so the
  UI can tell which axes support it.
- `ZaberLinearAxis.move_vel(velocity)`: validate finite; `0` → `self._axis.stop(wait_until_idle=False)`;
  otherwise `self._axis.move_velocity(velocity, self._velocity_unit)` under `hub.lock`, through
  the existing `_start` error handling. Velocity moves run until stopped or until the controller's
  soft limit, where the controller stops them itself.
- Refuse `move_vel` while the axis is TTL-stepping or continuously scanning (the controller
  already tracks `scan_mode`).
- `SimulatedContinuousAxis.move_vel`: integrate position at the commanded velocity, clamped to
  its limits, so the whole feature can be exercised without hardware.
- Expose `move_vel` on `ContinuousAxisHandle`.

### 2. Instrument: `jog_stage` with a dead-man watchdog (`vxl.instrument`)

- New `Instrument.jog_stage(x: float, y: float, z: float, *, seq: int, client: str)` taking
  normalized velocities in `[-1, 1]`.
- Refuse (and halt any jog) while a capture is running (`_ensure_not_busy`) or while any stage
  axis has `scan_mode != IDLE`. Jogging **is** allowed during live preview — that is the main
  use, finding the sample.
- Scale by per-axis jog speed limits (new `jog` settings, below), then call `move_vel` only on
  axes whose commanded velocity changed by more than a small tolerance, so a steady stick does
  not flood the serial/TCP link.
- **Watchdog:** each update re-arms a ~300 ms timer; on expiry every jogging axis is halted.
  Also halt on WebSocket disconnect, on an all-zero command, when an acquisition starts, and on
  instrument close.
- **Single controller lease:** the first client to send a non-zero jog owns jogging until it
  sends zero for a short period or disconnects; jogs from another browser are rejected, so two
  open tabs cannot fight over the stage.
- Ignore out-of-order messages using `seq`.
- Keep the call cheap: it must not wait behind long operations on the instrument lock (e.g.
  profile application). If the lock is busy, jogs are dropped rather than queued, so no stale
  velocity is applied later.

### 3. Configuration

Add an optional `jog` section to the instrument/station configuration (validated with the
existing pydantic models):

```yaml
jog:
  max_speed_um_s: {x: 2000, y: 2000, z: 500}   # full-stick speed per logical axis
  fine_speed_fraction: 0.1                     # speed while the fine-control button is held
  invert: {x: false, y: true, z: false}        # match stick direction to what the operator sees
  dead_zone: 0.15
  watchdog_ms: 300
```

Defaults are deliberately slow. Z gets a lower default because it moves the sample toward the
objective. `invert` exists because stick-up should move the image the way the operator expects,
and Mobi's physical axis labels differ from the logical letters.

### 4. Transport (`vxl.web.realtime`)

- Extend the state WebSocket's control-message handling with a `StageJogUpdate`
  (`session_id`, `velocity: [x, y, z]`, `seq`) next to `PreviewViewportUpdate`, routed to
  `instrument.jog_stage`.
- On WebSocket disconnect, halt that client's jog immediately (don't wait for the watchdog).
- Report jog state (`active`, owning client, last refusal reason) in the station feed so the UI
  can show it.

### 5. Web UI (`web/station`)

- New `lib/core/gamepad.svelte.ts`: listens for `gamepadconnected`/`gamepaddisconnected`, polls
  `navigator.getGamepads()` on `requestAnimationFrame`, and sends at most ~30 updates/s, plus a
  keep-alive while any stick is deflected.
- Mapping (standard gamepad layout):
  - `axes[0]` left stick X → stage X; `axes[1]` left stick Y → stage Y
  - `axes[3]` right stick Y → stage Z; right stick X unused
  - **B** → halt all axes (same as the existing halt button)
  - **Right bumper (hold)** → fine speed (`fine_speed_fraction`)
  - **Left bumper (hold)** → enable / dead-man switch (optional — see open questions)
- Signal conditioning: radial dead zone for the left stick so diagonals stay smooth, axial dead
  zone for Z, and a cubic response curve for fine control near center.
- Safety in the browser: send zero and stop sending when the controller disconnects, the tab
  becomes hidden (`visibilitychange`), or the window loses focus.
- An on/off toggle and status indicator in `shell/StageAxes.svelte` ("Controller: connected /
  jogging / blocked — acquisition running"). Off by default each session so a controller left on
  the desk cannot move the stage unexpectedly.
- A short help popover showing the button map.

### 6. Tests

- Driver: `drivers/tests/test_zaber.py` — `move_vel` issues `move_velocity` / `stop` in native
  units, rejects non-finite input, and refuses during TTL stepping / continuous scanning
  (mocked `zaber_motion` axis, as the existing tests do).
- Instrument: with the simulated axes — jogging moves the axes; the watchdog halts after
  silence; a capture start halts and blocks jogs; the lease rejects a second client;
  disconnect halts.
- Web: `node --test` unit tests for the pure mapping (dead zone, curve, inversion, fine mode).
- Hardware (opt-in, `drivers/tests/hardware/`): short, low-speed velocity move and stop on one
  axis, gated by the existing operator motion opt-in.

### 7. Hardware commissioning on Mobi

1. Verify soft limits (`limit.min`/`limit.max`) on all three axes, especially Z clearance to
   the objective, before enabling jogging.
2. Start with very low `max_speed_um_s`; confirm each stick direction moves the expected
   physical axis and direction; set `invert` as needed.
3. Pull the USB cable / close the tab mid-jog and confirm the stage stops within the watchdog
   time.
4. Start an acquisition while jogging and confirm the jog halts and is refused.
5. Raise speeds to comfortable values and record them in the private rig configuration.

## Estimated scope

| Area | Size |
| --- | --- |
| Driver + simulated axis + handle | small (~100 lines + tests) |
| Instrument jog, watchdog, lease, config | medium (~250 lines + tests) |
| WebSocket message + feed state | small |
| Svelte gamepad module + indicator | medium (~250 lines) |

Delivered as separate commits in that order, so the server side can be tested with the
simulated microscope before any UI work.

## Open questions for review

1. **Which machine will the controller plug into** — the station PC running the browser, or a
   separate laptop? If remote, is HTTPS (or an SSH tunnel) acceptable for the Gamepad API?
2. **Logical vs physical axes:** should the sticks drive Voxel's *logical* x/y/z (recommended,
   with `invert` per axis), or the axes by their physical Mobi labels?
3. **Dead-man button:** require holding the left bumper to move (safer), or move on stick
   deflection alone (simpler)?
4. **Jog speeds:** starting full-stick speeds for X/Y and Z on Mobi.
5. **Jogging during live preview:** allowed (recommended), or only when idle?
