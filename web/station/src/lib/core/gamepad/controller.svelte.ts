import type { Instrument } from '$lib/core/model';
/**
 * Reads an Xbox-style controller with the browser Gamepad API and streams stage jog updates.
 *
 * - Left bumper toggles jogging on/off (the server keeps the switch, so every open window shows it).
 * - Left stick moves X/Y, right stick (vertical) moves Z; deflection sets speed. Releasing the sticks stops.
 * - B halts the stage.
 *
 * Updates go out at most every SEND_INTERVAL_MS and are repeated every KEEPALIVE_MS while a stick is held, inside the
 * server's dead-man watchdog. The page stops sending (and sends a stop) when it is hidden, loses focus, the controller
 * disconnects, jogging is switched off, or a capture starts; if a stop is lost, the server watchdog halts the stage.
 */
import { reportFailure } from '@voxel/kit/failures';

import { BUTTON, isRest, jogVector, type JogVector } from './mapping';

const SEND_INTERVAL_MS = 33;
const KEEPALIVE_MS = 100;
const RESEND_DELTA = 0.01;
const ZERO: JogVector = [0, 0, 0];

export class GamepadJog {
  /** A controller is plugged in and visible to this page. */
  connected = $state(false);
  /** The controller's name as the browser reports it. */
  name = $state<string | null>(null);
  /** The sticks are deflected and this page is driving the stage. */
  driving = $state(false);

  readonly #getInstrument: () => Instrument | undefined;
  #frame = 0;
  #seq = 0;
  #sent: JogVector = ZERO;
  #sentAt = 0;
  #pressed = new Set<number>();
  #disposers: (() => void)[] = [];

  constructor(getInstrument: () => Instrument | undefined) {
    this.#getInstrument = getInstrument;
  }

  /** Whether this browser exposes the Gamepad API (it requires a secure context such as localhost or HTTPS). */
  static get supported(): boolean {
    return typeof navigator !== 'undefined' && typeof navigator.getGamepads === 'function';
  }

  start(): void {
    if (!GamepadJog.supported || this.#frame) return;
    const stop = () => this.#stopDriving();
    const onVisibility = () => document.hidden && stop();
    window.addEventListener('blur', stop);
    document.addEventListener('visibilitychange', onVisibility);
    window.addEventListener('gamepaddisconnected', stop);
    this.#disposers.push(
      () => window.removeEventListener('blur', stop),
      () => document.removeEventListener('visibilitychange', onVisibility),
      () => window.removeEventListener('gamepaddisconnected', stop)
    );
    const loop = (now: number) => {
      this.#tick(now);
      this.#frame = requestAnimationFrame(loop);
    };
    this.#frame = requestAnimationFrame(loop);
  }

  dispose(): void {
    this.#stopDriving();
    if (this.#frame) cancelAnimationFrame(this.#frame);
    this.#frame = 0;
    for (const dispose of this.#disposers.splice(0)) dispose();
  }

  #tick(now: number): void {
    const pad = navigator.getGamepads().find((candidate) => candidate?.connected) ?? null;
    this.connected = pad !== null;
    this.name = pad?.id ?? null;
    const instrument = this.#getInstrument();
    if (!pad || !instrument) {
      this.#pressed.clear();
      this.#stopDriving();
      return;
    }

    if (this.#pressedNow(pad, BUTTON.leftBumper)) {
      const jog = instrument.jog;
      if (jog.available) reportFailure(instrument.setJogEnabled(!jog.enabled));
    }
    if (this.#pressedNow(pad, BUTTON.b)) {
      this.#stopDriving();
      reportFailure(instrument.stage.halt());
    }

    const jog = instrument.jog;
    if (!jog.enabled || instrument.busy || document.hidden || !document.hasFocus()) {
      this.#stopDriving();
      return;
    }

    const velocity = jogVector(pad.axes, instrument.stage.orientation);
    if (isRest(velocity)) {
      this.#stopDriving();
      return;
    }
    const changed = velocity.some((v, i) => Math.abs(v - this.#sent[i]) > RESEND_DELTA);
    const elapsed = now - this.#sentAt;
    if ((changed && elapsed >= SEND_INTERVAL_MS) || elapsed >= KEEPALIVE_MS) this.#send(instrument, velocity, now);
  }

  /** True only on the frame a button goes down. */
  #pressedNow(pad: Gamepad, index: number): boolean {
    const down = pad.buttons[index]?.pressed ?? false;
    const was = this.#pressed.has(index);
    if (down) this.#pressed.add(index);
    else this.#pressed.delete(index);
    return down && !was;
  }

  #send(instrument: Instrument, velocity: JogVector, now: number): void {
    if (instrument.sendJog(++this.#seq, velocity)) {
      this.#sent = velocity;
      this.#sentAt = now;
      this.driving = !isRest(velocity);
    } else {
      this.#sent = ZERO; // Socket down: the server watchdog has halted, so start over when it returns.
      this.driving = false;
    }
  }

  /** Send one stop if this page was driving. */
  #stopDriving(): void {
    if (isRest(this.#sent)) return;
    const instrument = this.#getInstrument();
    if (instrument) instrument.sendJog(++this.#seq, ZERO);
    this.#sent = ZERO;
    this.#sentAt = 0;
    this.driving = false;
  }
}
