/**
 * Pure gamepad → stage-jog mapping for an Xbox-style controller in the browser's "standard" layout.
 *
 * Left stick drives logical stage X/Y, the right stick's vertical drives Z. After a dead zone, a response curve keeps
 * small pushes slow while a full push reaches full jog speed. The result is a fraction of full speed per axis in
 * [-1, 1]; the server scales and caps it.
 */

/** Standard-mapping indices (https://w3c.github.io/gamepad/#remapping). */
export const STICK = { leftX: 0, leftY: 1, rightY: 3 } as const;
export const BUTTON = { b: 1, leftBumper: 4 } as const;

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

/** Map a stick magnitude to speed: 0 inside the dead zone, then a curve from slow to 1 at full deflection. */
export function curve(magnitude: number): number {
  const m = Math.min(1, Math.max(0, (magnitude - DEAD_ZONE) / (1 - DEAD_ZONE)));
  return LINEAR_SHARE * m + (1 - LINEAR_SHARE) * m ** 3;
}

/** One axis, keeping the sign. */
export function shapeAxis(value: number): number {
  return Math.sign(value) * curve(Math.abs(value));
}

/** A 2-D stick with a radial dead zone, so diagonals stay smooth and motion does not snap onto the axes. */
export function shapeStick(x: number, y: number): [number, number] {
  const magnitude = Math.hypot(x, y);
  if (magnitude <= DEAD_ZONE) return [0, 0];
  const scale = curve(magnitude) / magnitude;
  return [x * scale, y * scale];
}

/**
 * Map raw standard-layout axes to a jog vector in logical stage coordinates.
 *
 * Browser sticks report +x right and +y down. Stick right moves the stage the way +X is drawn, stick up the way +Y is
 * drawn (Voxel draws +Y upward), and right stick up drives +Z, each honouring the stage orientation signs.
 */
export function jogVector(axes: readonly number[], signs: AxisSigns): JogVector {
  const [x, y] = shapeStick(axes[STICK.leftX], axes[STICK.leftY]);
  const z = shapeAxis(axes[STICK.rightY]);
  return [x * signs.x, -y * signs.y, -z * signs.z];
}

/** Whether every component is zero (sticks at rest). */
export function isRest(v: JogVector): boolean {
  return v.every((c) => c === 0);
}
