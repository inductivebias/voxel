/**
 * Reads an Xbox-style controller with the browser Gamepad API and streams stage jog updates.
 *
 * - Left bumper toggles jogging on/off (the server keeps the switch, so every open window shows it).
 * - Left stick moves X/Y, right stick (vertical) moves Z; deflection sets speed. Releasing the sticks stops.
 * - B halts the stage.
 *
 * Updates go out at most every SEND_INTERVAL_MS and repeat every KEEPALIVE_MS while a stick is held, inside the
 * server's dead-man watchdog. The page sends a stop when it is hidden, loses focus, the controller disconnects,
 * jogging is switched off, or a capture starts; if a stop is lost, the server watchdog halts the stage.
 */
import type { Instrument } from '$lib/core/model';
import { reportFailure } from '@voxel/kit/failures';

import { BUTTON, isRest, jogVector, type JogVector } from './mapping';

const SEND_INTERVAL_MS = 33;
const KEEPALIVE_MS = 100;
const RESEND_DELTA = 0.01;
const ZERO: JogVector = [0, 0, 0];

export class GamepadJog {
  /** Gamepads are only exposed in a secure context such as localhost or HTTPS. */
  static readonly supported = 'getGamepads' in navigator;

  /** The connected controller's name, or null when none is plugged in. */
  name = $state<string | null>(null);
  /** The sticks are deflected and this page is driving the stage. */
  driving = $state(false);

  readonly #instrument: () => Instrument;
  readonly #listeners = new AbortController();
  #frame = 0;
  #seq = 0;
  #sent: JogVector = ZERO;
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
      this.#tick(now);
      this.#frame = requestAnimationFrame(loop);
    };
    this.#frame = requestAnimationFrame(loop);
  }

  dispose(): void {
    this.#stop();
    cancelAnimationFrame(this.#frame);
    this.#listeners.abort();
  }

  #tick(now: number): void {
    const pad = navigator.getGamepads().find((p) => p?.connected) ?? null;
    this.name = pad?.id ?? null;
    if (!pad) return this.#stop();

    const instrument = this.#instrument();
    const { jog } = instrument;
    if (this.#pressed(pad, BUTTON.leftBumper) && jog.available) reportFailure(instrument.setJogEnabled(!jog.enabled));
    if (this.#pressed(pad, BUTTON.b)) {
      this.#stop();
      reportFailure(instrument.stage.halt());
    }

    const velocity = jogVector(pad.axes, instrument.stage.orientation);
    if (!jog.enabled || instrument.busy || document.hidden || !document.hasFocus() || isRest(velocity)) {
      return this.#stop();
    }
    const changed = velocity.some((v, i) => Math.abs(v - this.#sent[i]) > RESEND_DELTA);
    const elapsed = now - this.#sentAt;
    if ((changed && elapsed >= SEND_INTERVAL_MS) || elapsed >= KEEPALIVE_MS) this.#send(velocity, now);
  }

  /** True only on the frame a button goes down. */
  #pressed(pad: Gamepad, index: number): boolean {
    const down = pad.buttons[index].pressed;
    const was = this.#held.has(index);
    if (down) this.#held.add(index);
    else this.#held.delete(index);
    return down && !was;
  }

  #send(velocity: JogVector, now: number): void {
    // If the socket is down the server watchdog has halted, so start over from rest when it returns.
    const sent = this.#instrument().sendJog(++this.#seq, velocity);
    this.#sent = sent ? velocity : ZERO;
    this.#sentAt = now;
    this.driving = sent && !isRest(velocity);
  }

  /** Send one stop if this page was driving. */
  #stop(): void {
    if (isRest(this.#sent)) return;
    this.#send(ZERO, 0);
  }
}
