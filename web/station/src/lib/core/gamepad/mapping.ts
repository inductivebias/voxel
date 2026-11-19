/**
 * Pure gamepad → stage-jog mapping for an Xbox-style controller in the browser's "standard" layout.
 *
 * Left stick drives logical stage X/Y, the right stick's vertical drives Z. Stick deflection sets speed: after a dead
 * zone, a response curve keeps small pushes slow for fine positioning while a full push reaches full jog speed. The
 * result is a fraction of full speed per axis in [-1, 1]; the server scales and caps it.
 */

/** Standard-mapping indices (https://w3c.github.io/gamepad/#remapping). */
export const STICK = { leftX: 0, leftY: 1, rightX: 2, rightY: 3 } as const;
export const BUTTON = { a: 0, b: 1, x: 2, y: 3, leftBumper: 4, rightBumper: 5 } as const;

/** Deflections below this are treated as a resting stick (Xbox sticks rarely rest at exactly 0). */
export const DEAD_ZONE = 0.12;
/** Share of the output that grows linearly; the rest grows with the cube, so small deflections stay slow. */
export const LINEAR_SHARE = 0.25;

export type JogVector = [x: number, y: number, z: number];

/** Display signs of the logical axes (`StageOrientation`): +1 when increasing the coordinate moves right/up on screen. */
export interface AxisSigns {
  x: 1 | -1;
  y: 1 | -1;
  z: 1 | -1;
}

/** Ease a dead-zone-rescaled magnitude in [0, 1]: slow near rest, 1 at full deflection. */
export function curve(magnitude: number): number {
  const m = Math.min(1, Math.max(0, magnitude));
  return LINEAR_SHARE * m + (1 - LINEAR_SHARE) * m ** 3;
}

/** Remove the dead zone from a magnitude and rescale what remains to [0, 1]. */
export function rescale(magnitude: number, deadZone = DEAD_ZONE): number {
  if (magnitude <= deadZone) return 0;
  return Math.min(1, (magnitude - deadZone) / (1 - deadZone));
}

/** One axis: dead zone, then curve, keeping the sign. */
export function shapeAxis(value: number, deadZone = DEAD_ZONE): number {
  if (!Number.isFinite(value)) return 0;
  return Math.sign(value) * curve(rescale(Math.abs(value), deadZone));
}

/**
 * A 2-D stick with a radial dead zone, so diagonal pushes stay smooth and the dead zone does not snap motion onto
 * the axes. Returns the shaped [x, y] in the stick's own frame.
 */
export function shapeStick(x: number, y: number, deadZone = DEAD_ZONE): [number, number] {
  if (!Number.isFinite(x) || !Number.isFinite(y)) return [0, 0];
  const magnitude = Math.hypot(x, y);
  if (magnitude <= deadZone) return [0, 0];
  const scale = curve(rescale(Math.min(1, magnitude), deadZone)) / magnitude;
  return [x * scale, y * scale];
}

/**
 * Map raw standard-layout axes to a jog vector in logical stage coordinates.
 *
 * Browser sticks report +x right and +y down. Stick right moves the stage the way +X is drawn, stick up moves it the
 * way +Y is drawn (Voxel draws +Y upward), and right stick up drives +Z, each honouring the stage orientation signs.
 */
export function jogVector(axes: readonly number[], signs: AxisSigns, deadZone = DEAD_ZONE): JogVector {
  const [sx, sy] = shapeStick(axes[STICK.leftX] ?? 0, axes[STICK.leftY] ?? 0, deadZone);
  const sz = shapeAxis(axes[STICK.rightY] ?? 0, deadZone);
  return [clean(sx * signs.x), clean(-sy * signs.y), clean(-sz * signs.z)];
}

/** Whether every component is zero (sticks at rest). */
export function isRest(v: JogVector): boolean {
  return v[0] === 0 && v[1] === 0 && v[2] === 0;
}

/** Avoid sending -0 and float dust. */
function clean(value: number): number {
  return Math.abs(value) < 1e-4 ? 0 : Math.round(value * 1e4) / 1e4;
}
