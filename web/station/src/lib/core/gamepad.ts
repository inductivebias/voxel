/**
 * Xbox-style gamepad reading and stick-to-stage mapping (browser standard layout).
 * Left stick drives physical X/Y, right stick vertical drives Z, LB toggles arming, RB cycles speed, B halts.
 */
import type { StageAxisLabels, StageOrientation } from './model/types';

export type JogVector = [x: number, y: number, z: number];
export type JogSpeed = 'slow' | 'med' | 'fast';

export const JOG_SPEEDS: readonly { id: JogSpeed; label: string; scale: number }[] = [
  { id: 'slow', label: 'Slow', scale: 1 / 3 },
  { id: 'med', label: 'Med', scale: 2 / 3 },
  { id: 'fast', label: 'Fast', scale: 1 }
];

const DEAD_ZONE = 0.12;
const LINEAR_SHARE = 0.25;
const SEND_INTERVAL_MS = 50;
const BUTTON_B = 1;
const BUTTON_LB = 4;
const BUTTON_RB = 5;
const LOGICAL_AXES = ['x', 'y', 'z'] as const;
type LogicalIdx = 0 | 1 | 2;

function curve(mag: number): number {
  const m = Math.min(1, Math.max(0, (mag - DEAD_ZONE) / (1 - DEAD_ZONE)));
  return LINEAR_SHARE * m + (1 - LINEAR_SHARE) * m ** 3;
}

/** Map physical stick X, Y, Z to logical [x, y, z] slots when stage labels permute X/Y/Z. */
function resolveStickAxes(labels?: StageAxisLabels): readonly [LogicalIdx, LogicalIdx, LogicalIdx] {
  if (!labels) return [0, 1, 2];
  const byPhysical: Partial<Record<'x' | 'y' | 'z', LogicalIdx>> = {};
  for (let i = 0; i < 3; i++) {
    const name = (labels[LOGICAL_AXES[i]] ?? LOGICAL_AXES[i]).trim().toLowerCase();
    if (name === 'x' || name === 'y' || name === 'z') byPhysical[name] = i as LogicalIdx;
  }
  if (byPhysical.x !== undefined && byPhysical.y !== undefined && byPhysical.z !== undefined) {
    return [byPhysical.x, byPhysical.y, byPhysical.z];
  }
  return [0, 1, 2];
}

/** Map raw standard-layout axes to logical stage velocities in [-1, 1] matching physical X/Y/Z labels. */
export function jogVector(
  axes: readonly number[],
  signs: StageOrientation,
  speedScale = 1,
  labels?: StageAxisLabels
): JogVector {
  const lx = axes[0] ?? 0;
  const ly = axes[1] ?? 0;
  const mag = Math.hypot(lx, ly);
  const scale = mag > DEAD_ZONE ? (curve(mag) / mag) * speedScale : 0;
  const rz = axes[3] ?? 0;
  const stick: readonly [number, number, number] = [
    lx * scale,
    -ly * scale,
    -Math.sign(rz) * curve(Math.abs(rz)) * speedScale
  ];
  const [ix, iy, iz] = resolveStickAxes(labels);
  const out: JogVector = [0, 0, 0];
  out[ix] = stick[0] * signs[LOGICAL_AXES[ix]];
  out[iy] = stick[1] * signs[LOGICAL_AXES[iy]];
  out[iz] = stick[2] * signs[LOGICAL_AXES[iz]];
  return out;
}

export interface GamepadJogOptions {
  isArmed: () => boolean;
  orientation: () => StageOrientation;
  labels: () => StageAxisLabels;
  speedScale: () => number;
  send: (velocity: JogVector) => boolean;
  onToggle: () => void;
  onCycleSpeed: () => void;
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
    if (pressed(pad, BUTTON_RB)) opts.onCycleSpeed();
    if (pressed(pad, BUTTON_B)) {
      stop();
      opts.onHalt();
    }
    const v = jogVector(pad.axes, opts.orientation(), opts.speedScale(), opts.labels());
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
