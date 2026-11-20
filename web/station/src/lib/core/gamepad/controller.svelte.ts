/**
 * Reads an Xbox-style controller with the browser Gamepad API and streams stage jog updates.
 *
 * - Left bumper toggles jogging on/off locally in this browser.
 * - Left stick moves X/Y, right stick (vertical) moves Z; deflection sets speed. Releasing the sticks stops.
 * - B halts the stage.
 *
 * Updates go out at ~20 Hz while any stick is deflected (well inside the server's 300 ms dead-man watchdog) and stop
 * immediately when the sticks return to rest, the page is hidden or loses focus, the controller disconnects, jogging
 * is disarmed, or a capture starts; if a stop is lost, the server watchdog halts the stage.
 */
import type { Instrument } from '$lib/core/model';
import { reportFailure } from '@voxel/kit/failures';

import { BUTTON, isRest, jogVector, type JogVector } from './mapping';

const SEND_INTERVAL_MS = 50;
const ZERO: JogVector = [0, 0, 0];

export class GamepadJog {
  /** Gamepads are only exposed in a secure context such as localhost or HTTPS. */
  static readonly supported = 'getGamepads' in navigator;

  /** Whether jogging is armed in this browser; sticks move the stage only while true. */
  armed = $state(false);
  /** The connected controller's name, or null when none is plugged in. */
  name = $state<string | null>(null);
  /** The sticks are deflected and this page is driving the stage. */
  driving = $state(false);

  readonly #instrument: () => Instrument;
  readonly #listeners = new AbortController();
  #frame = 0;
  #sentAt = 0;
  #held = new Set<number>();

  constructor(instrument: () => Instrument) {
    this.#instrument = instrument;
  }

  start(): void {
    if (!GamepadJog.supported) return;
    const stop = () => this.#stop();
    const { signal } = this.#listeners;
    window.addEventListener('blur', stop, { signal });
    window.addEventListener('gamepaddisconnected', stop, { signal });
    document.addEventListener('visibilitychange', stop, { signal });
    const loop = (now: number) => {
      this.#frame = requestAnimationFrame(loop);
      this.#tick(now);
    };
    this.#frame = requestAnimationFrame(loop);
  }

  dispose(): void {
    this.disarm();
    cancelAnimationFrame(this.#frame);
    this.#listeners.abort();
  }

  toggle(): void {
    if (this.armed) {
      this.disarm();
    } else {
      const instrument = this.#instrument();
      if (instrument.stage.jogAvailable && !instrument.busy) this.armed = true;
    }
  }

  disarm(): void {
    this.armed = false;
    this.#stop();
  }

  #tick(now: number): void {
    const pads = navigator.getGamepads();
    const pad = pads.find((p) => p?.connected && p.mapping === 'standard') ?? pads.find((p) => p?.connected) ?? null;
    this.name = pad?.id ?? null;
    if (!pad) return this.#stop();

    const instrument = this.#instrument();
    if (this.#pressed(pad, BUTTON.leftBumper)) this.toggle();
    if (this.#pressed(pad, BUTTON.b)) {
      this.#stop();
      reportFailure(instrument.stage.halt());
    }

    const velocity = jogVector(pad.axes, instrument.stage.orientation);
    if (!this.armed || instrument.busy || document.hidden || !document.hasFocus() || isRest(velocity)) {
      return this.#stop();
    }
    if (!this.driving || now - this.#sentAt >= SEND_INTERVAL_MS) this.#send(velocity, now);
  }

  /** True only on the frame a button goes down. */
  #pressed(pad: Gamepad, index: number): boolean {
    const down = pad.buttons[index]?.pressed ?? false;
    const was = this.#held.has(index);
    if (down) this.#held.add(index);
    else this.#held.delete(index);
    return down && !was;
  }

  #send(velocity: JogVector, now: number): void {
    // If the socket is down the server watchdog has halted, so start over from rest when it returns.
    const sent = this.#instrument().sendJog(velocity);
    this.#sentAt = now;
    this.driving = sent && !isRest(velocity);
  }

  /** Send one stop if this page was driving. */
  #stop(): void {
    if (!this.driving) return;
    this.#send(ZERO, 0);
  }
}
