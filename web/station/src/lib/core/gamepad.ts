/**
 * Xbox-style gamepad reading and stick-to-stage mapping (browser standard layout).
 * Left stick drives X/Y, right stick vertical drives Z, left bumper toggles arming, B halts.
 */
import type { StageOrientation } from './model/types';

export type JogVector = [x: number, y: number, z: number];

const DEAD_ZONE = 0.12;
const LINEAR_SHARE = 0.25;
const SEND_INTERVAL_MS = 50;
const BUTTON_B = 1;
const BUTTON_LB = 4;

function curve(mag: number): number {
  const m = Math.min(1, Math.max(0, (mag - DEAD_ZONE) / (1 - DEAD_ZONE)));
  return LINEAR_SHARE * m + (1 - LINEAR_SHARE) * m ** 3;
}

/** Map raw standard-layout axes to logical stage velocities in [-1, 1] matching StageOrientation. */
export function jogVector(axes: readonly number[], signs: StageOrientation): JogVector {
  const lx = axes[0] ?? 0;
  const ly = axes[1] ?? 0;
  const mag = Math.hypot(lx, ly);
  const scale = mag > DEAD_ZONE ? curve(mag) / mag : 0;
  const rz = axes[3] ?? 0;
  return [lx * scale * signs.x, -ly * scale * signs.y, -Math.sign(rz) * curve(Math.abs(rz)) * signs.z];
}

export interface GamepadJogOptions {
  isArmed: () => boolean;
  orientation: () => StageOrientation;
  send: (velocity: JogVector) => boolean;
  onToggle: () => void;
  onHalt: () => void;
  onState: (name: string | null, driving: boolean) => void;
}

/** Poll a connected gamepad on requestAnimationFrame; returns a cleanup function for $effect. */
export function startGamepadJog(opts: GamepadJogOptions): () => void {
  if (typeof navigator === 'undefined' || !('getGamepads' in navigator)) return () => {};

  const held = new Set<number>();
  let frame = 0;
  let sentAt = 0;
  let driving = false;
  let lastName: string | null = null;

  const sync = (name: string | null, isDriving: boolean) => {
    if (name !== lastName || isDriving !== driving) {
      lastName = name;
      driving = isDriving;
      opts.onState(name, isDriving);
    }
  };

  const stop = () => {
    if (!driving) return;
    opts.send([0, 0, 0]);
    sentAt = 0;
    sync(lastName, false);
  };

  const pressed = (pad: Gamepad, idx: number) => {
    const down = pad.buttons[idx]?.pressed ?? false;
    const was = held.has(idx);
    if (down) held.add(idx);
    else held.delete(idx);
    return down && !was;
  };

  const loop = (now: number) => {
    frame = requestAnimationFrame(loop);
    const pads = navigator.getGamepads();
    const pad = pads.find((p) => p?.connected && p.mapping === 'standard') ?? pads.find((p) => p?.connected) ?? null;
    if (!pad) {
      stop();
      sync(null, false);
      return;
    }
    sync(pad.id, driving);
    if (pressed(pad, BUTTON_LB)) opts.onToggle();
    if (pressed(pad, BUTTON_B)) {
      stop();
      opts.onHalt();
    }
    const v = jogVector(pad.axes, opts.orientation());
    if (!opts.isArmed() || document.hidden || !document.hasFocus() || v.every((c) => c === 0)) {
      stop();
      return;
    }
    if (!driving || now - sentAt >= SEND_INTERVAL_MS) {
      const ok = opts.send(v);
      sentAt = now;
      sync(pad.id, ok);
    }
  };

  const listeners = new AbortController();
  const { signal } = listeners;
  window.addEventListener('blur', stop, { signal });
  window.addEventListener('gamepaddisconnected', stop, { signal });
  document.addEventListener('visibilitychange', stop, { signal });
  frame = requestAnimationFrame(loop);

  return () => {
    stop();
    cancelAnimationFrame(frame);
    listeners.abort();
  };
}
