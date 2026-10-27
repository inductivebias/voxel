import math
import time
from threading import Event, Thread
from typing import Any

from vxl.devices.axes.continuous import (
    ContinuousAxis,
    ContinuousScanInfo,
    ContinuousScanner,
    StepMode,
    TriggerMode,
    TTLStepper,
    TTLStepperConfig,
)
from vxl.devices.axes.discrete import DiscreteAxis
from vxl.devices.axes.discrete.base import DiscreteAxisState

# Simulated moves end exactly on target; this only absorbs float noise in axis units.
_SCAN_POSITION_TOLERANCE = 1e-6


class SimulatedTTLStepper(TTLStepper):
    """Simulated TTL stepper for testing step-and-shoot operations."""

    def __init__(self, axis: "SimulatedContinuousAxis"):
        self._axis = axis
        self._buffer: list[float] = []
        self._step_mode: StepMode = StepMode.ABSOLUTE
        self._trigger_mode: TriggerMode = TriggerMode.TTL
        self._pulse_after_move: bool = True

    def configure(self, cfg: TTLStepperConfig | Any) -> None:
        """Configure the simulated step-and-shoot operation."""
        if isinstance(cfg, TTLStepperConfig):
            self._step_mode = cfg.step_mode
            self._trigger_mode = cfg.trigger_mode
            self._pulse_after_move = cfg.pulse_after_move
        self._axis.log.info(f"Configured TTL stepper: mode={self._step_mode.value}, trigger={self._trigger_mode.value}")

    def queue_absolute_move(self, position: float) -> None:
        """Queue an absolute move to the simulated ring buffer."""
        self._buffer.append(position)
        self._axis.log.debug(f"Queued absolute move to {position} (buffer size: {len(self._buffer)})")

    def queue_relative_move(self, delta: float) -> None:
        """Queue a relative move to the simulated ring buffer."""
        self._buffer.append(delta)
        self._axis.log.debug(f"Queued relative move by {delta} (buffer size: {len(self._buffer)})")

    def reset(self) -> None:
        """Reset the step-and-shoot configuration and clear the buffer."""
        self._buffer.clear()
        self._axis.log.info("TTL stepper reset - buffer cleared")


class SimulatedContinuousScanner(ContinuousScanner):
    """Simulated continuous scan: one ordinary simulated move at the requested speed."""

    def __init__(self, axis: "SimulatedContinuousAxis", info: ContinuousScanInfo) -> None:
        self._axis = axis
        self._info = info
        self._previous_speed: float | None = None  # Set while a scan is prepared.

    @property
    def info(self) -> ContinuousScanInfo:
        return self._info

    def _state(self) -> dict[str, Any]:
        position = self._axis.position
        return {"position": position, "origin": position, "moving": self._axis.is_moving}

    def _require_prepared(self) -> None:
        if self._previous_speed is None:
            raise RuntimeError("Continuous scan not prepared")

    def _restore(self) -> None:
        if self._previous_speed is not None:
            self._axis.speed = self._previous_speed
            self._previous_speed = None

    def prepare(self, *, start: float, speed: float) -> dict[str, Any]:
        if self._previous_speed is not None:
            raise RuntimeError("A continuous scan is already prepared; finish or reset it first")
        if not 0 < speed <= self._info.max_speed:
            raise ValueError(f"Scan speed {speed} is outside (0, {self._info.max_speed}] {self._axis.units}/s")
        if self._axis.is_moving:
            raise RuntimeError("Axis must be idle to prepare a continuous scan")
        position = self._axis.position
        if not math.isclose(position, start, abs_tol=_SCAN_POSITION_TOLERANCE):
            raise RuntimeError(f"Axis is at {position} {self._axis.units}, not the scan start {start}")
        self._previous_speed = self._axis.speed
        self._axis.speed = speed
        self._axis.log.info(f"Prepared continuous scan from {start} at {speed} {self._axis.units}/s")
        return self._state()

    def start(self, target: float) -> None:
        self._require_prepared()
        lower, upper = self._axis.lower_limit, self._axis.upper_limit
        if not lower <= target <= upper:
            raise ValueError(f"Scan target {target} {self._axis.units} is outside limits [{lower}, {upper}]")
        self._axis.move_abs(target)

    def is_scanning(self) -> bool:
        self._require_prepared()
        return self._axis.is_moving

    def finish(self, target: float) -> dict[str, Any]:
        self._require_prepared()
        if self._axis.is_moving:
            raise RuntimeError("Axis is still moving")
        state = self._state()
        if not math.isclose(state["position"], target, abs_tol=_SCAN_POSITION_TOLERANCE):
            raise RuntimeError(f"Axis stopped at {state['position']} {self._axis.units}, not the scan target {target}")
        self._restore()
        self._axis.log.info(f"Finished continuous scan at {target} {self._axis.units}")
        return state

    def reset(self) -> None:
        if self._previous_speed is None:
            return
        if self._axis.is_moving:
            self._axis.halt()
        self._restore()
        self._axis.log.info("Continuous scan reset")


class SimulatedContinuousAxis(ContinuousAxis):
    def __init__(
        self,
        uid: str,
        units: str = "um",
        lower_limit: float = 0.0,
        upper_limit: float = 25000.0,
        speed: float = 1000.0,
        acceleration: float = 100000.0,
        has_ttl_stepper: bool = True,
        scan_edge_spacing: float | None = None,
        scan_acceleration: float = 20000.0,
        scan_max_speed: float = 5000.0,
    ):
        super().__init__(uid=uid, units=units)
        self._position = 0.0
        self._lower_limit = lower_limit
        self._upper_limit = upper_limit
        self._speed = speed
        self._acceleration = acceleration
        self._backlash = 0.0
        self._home = 0.0
        self._is_moving = False
        self._movement_thread: Thread | None = None
        self._stop_event = Event()
        # TTL stepper capability - optionally enabled
        self._ttl_stepper: SimulatedTTLStepper | None = None
        if has_ttl_stepper:
            self._ttl_stepper = SimulatedTTLStepper(self)
        # Continuous scan capability - enabled by giving the exported edge spacing
        self._continuous_scanner: SimulatedContinuousScanner | None = None
        if scan_edge_spacing is not None:
            info = ContinuousScanInfo(
                edge_spacing=scan_edge_spacing, acceleration=scan_acceleration, max_speed=scan_max_speed
            )
            self._continuous_scanner = SimulatedContinuousScanner(self, info)

    # Unit specification _____________________________________________________________________________________________

    # Motion commands ________________________________________________________________________________________________

    def move_abs(self, position: float, *, wait: bool = False, timeout_s: float | None = None) -> None:
        """Move to an absolute position."""
        if position < self._lower_limit or position > self._upper_limit:
            self.log.warning(
                "Position %s %s is outside limits [%s, %s] %s",
                position,
                self._units,
                self._lower_limit,
                self._upper_limit,
                self._units,
            )

        # Stop any ongoing movement
        if self._is_moving:
            self.halt()

        # Simulate movement time based on distance and speed
        distance = abs(position - self._position)
        movement_time = distance / self._speed if self._speed > 0 else 0

        if movement_time > 0:
            self._is_moving = True
            start_position = self._position
            target_position = position
            self._stop_event.clear()

            def simulate_movement():
                # Update position incrementally for smooth animation
                update_rate_hz = 30  # Update position 30 times per second
                update_interval = 1.0 / update_rate_hz
                steps = max(1, int(movement_time / update_interval))

                for i in range(steps):
                    if self._stop_event.is_set():
                        break

                    # Linear interpolation
                    progress = (i + 1) / steps
                    self._position = start_position + (target_position - start_position) * progress
                    time.sleep(update_interval)

                if not self._stop_event.is_set():
                    self._position = target_position
                self._is_moving = False
                self.log.debug(f"Reached position {target_position} {self._units}")

            self._movement_thread = Thread(target=simulate_movement, daemon=True)
            self._movement_thread.start()

            if wait:
                self.await_movement(timeout_s)
        else:
            self._position = position

    def move_rel(self, delta: float, *, wait: bool = False, timeout_s: float | None = None) -> None:
        """Move a relative distance."""
        target_pos = self._position + delta
        self.move_abs(target_pos, wait=wait, timeout_s=timeout_s)

    def go_home(self, *, wait: bool = False, timeout_s: float | None = None) -> None:
        """Move to the home position."""
        self.move_abs(self._home, wait=wait, timeout_s=timeout_s)

    def halt(self) -> None:
        """Emergency stop - halt all motion immediately."""
        self._stop_event.set()
        if self._movement_thread and self._movement_thread.is_alive():
            self._movement_thread.join(timeout=0.1)
        self._is_moving = False
        self.log.info("Motion halted")

    def await_movement(self, timeout_s: float | None = None) -> None:
        """Wait until the axis stops moving."""
        if self._movement_thread and self._movement_thread.is_alive():
            self._movement_thread.join(timeout=timeout_s)

    # State properties _______________________________________________________________________________________________

    def _get_position(self) -> float:
        """Get the current position."""
        return self._position

    @property
    def is_moving(self) -> bool:
        """Check if the axis is moving."""
        return self._is_moving

    # Configuration and calibration __________________________________________________________________________________

    def set_zero_here(self) -> None:
        """Set current position as zero."""
        offset = self._position
        self._position = 0.0
        self._lower_limit -= offset
        self._upper_limit -= offset
        self._home -= offset
        self.log.info(f"Zeroed at position (offset: {offset} {self._units})")

    def set_logical_position(self, position: float) -> None:
        """Set the logical position without moving (for calibration)."""
        offset = self._position - position
        self._position = position
        self._lower_limit -= offset
        self._upper_limit -= offset
        self._home -= offset
        self.log.info(f"Logical position set to {position} {self._units}")

    @property
    def upper_limit(self) -> float:
        """Get the upper position limit."""
        return self._upper_limit

    @upper_limit.setter
    def upper_limit(self, value: float) -> None:
        """Set the upper position limit."""
        self._upper_limit = value
        self.log.debug("upper limit set to %s %s", value, self._units)

    @property
    def lower_limit(self) -> float:
        """Get the lower position limit."""
        return self._lower_limit

    @lower_limit.setter
    def lower_limit(self, value: float) -> None:
        """Set the lower position limit."""
        self._lower_limit = value
        self.log.debug("lower limit set to %s %s", value, self._units)

    # Kinematic parameters ___________________________________________________________________________________________

    @property
    def speed(self) -> float:
        """Get the speed."""
        return self._speed

    @speed.setter
    def speed(self, value: float) -> None:
        """Set the speed."""
        self._speed = value
        self.log.debug("speed set to %s %s/s", value, self._units)

    @property
    def acceleration(self) -> float:
        """Get the acceleration."""
        return self._acceleration

    @acceleration.setter
    def acceleration(self, value: float) -> None:
        """Set the acceleration."""
        self._acceleration = value
        self.log.debug("acceleration set to %s %s/s²", value, self._units)

    @property
    def backlash(self) -> float:
        """Get backlash."""
        return self._backlash

    @backlash.setter
    def backlash(self, value: float) -> None:
        """Set backlash."""
        self._backlash = value
        self.log.debug("backlash set to %s %s", value, self._units)

    @property
    def home(self) -> float:
        """Get the home position."""
        return self._home

    @home.setter
    def home(self, position: float) -> None:
        """Set the home position."""
        self._home = position
        self.log.debug("home position set to %s %s", position, self._units)

    # Capabilities ___________________________________________________________________________________________________

    def get_ttl_stepper(self) -> TTLStepper | None:
        """Return this axis's TTL stepper, or None if constructed with ``has_ttl_stepper=False``.

        Defaults to having one — mirroring the real ASI ``TigerLinearAxis``, where every linear axis
        carries a stepper — so a simulated scanning axis supports step-and-shoot acquisition by default.
        """
        return self._ttl_stepper

    def get_continuous_scanner(self) -> ContinuousScanner | None:
        """Return this axis's continuous scanner, or None unless constructed with ``scan_edge_spacing``."""
        return self._continuous_scanner


class SimulatedDiscreteAxis(DiscreteAxis):
    def __init__(
        self,
        uid: str,
        slots: dict[int | str, str | None],  # e.g. {0: "GFP", 1: "RFP", 2: "Cy5"}
        slot_count: int | None = None,
        start_pos: int = 0,
        settle_seconds: float = 0.05,
    ) -> None:
        super().__init__(uid=uid, slots=slots, slot_count=slot_count)

        if not (0 <= start_pos < self.slot_count):
            raise ValueError("start_pos out of range")

        self._settle = float(settle_seconds)
        if not math.isfinite(self._settle) or self._settle < 0:
            raise ValueError("settle_seconds must be finite and non-negative")
        self._state = DiscreteAxisState(position=start_pos, target=None, is_moving=False)
        self._settle_deadline: float | None = None
        self.log.debug(
            "SimulatedDiscreteAxis %s: Initialized at position %s with %s slots",
            uid,
            start_pos,
            self.slot_count,
        )

    # --------- state ----------
    @property
    def state(self) -> DiscreteAxisState:
        if self._settle_deadline is not None and time.monotonic() >= self._settle_deadline:
            self._state = DiscreteAxisState(position=self._state.target, target=self._state.target, is_moving=False)
            self._settle_deadline = None
        return self._state

    # --------- commands ----------
    def move(self, slot: int, *, wait: bool = False, timeout: float | None = None) -> None:
        if not (0 <= slot < self.slot_count):
            msg = f"Invalid slot {slot}; valid range is 0..{self.slot_count - 1}"
            raise ValueError(msg)

        if self.state.position == slot or self._settle == 0:
            self._state = DiscreteAxisState(position=slot, target=slot, is_moving=False)
            self._settle_deadline = None
        else:
            self._state = DiscreteAxisState(position=None, target=slot, is_moving=True)
            self._settle_deadline = time.monotonic() + self._settle
        if wait:
            self.await_movement(timeout)

    def home(self, *, wait: bool = False, timeout: float | None = None) -> None:
        self.log.debug("SimulatedDiscreteAxis %s: home() called - moving to slot 0", self.uid)
        self.move(0, wait=wait, timeout=timeout)

    def halt(self) -> None:
        """Halt motion, retaining the target and any resolved position."""
        state = self.state
        self._settle_deadline = None
        self._state = DiscreteAxisState(position=state.position, target=state.target, is_moving=False)

    def await_movement(self, timeout: float | None = None) -> None:
        """Wait for idle using a monotonic timeout in seconds."""
        deadline = time.monotonic() + timeout if timeout is not None else None
        while self.state.is_moving:
            now = time.monotonic()
            if deadline is not None and now >= deadline:
                raise TimeoutError("Movement did not complete within timeout")
            time.sleep(min(0.01, deadline - now) if deadline is not None else 0.01)
