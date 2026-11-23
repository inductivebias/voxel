import assert from 'node:assert/strict';
import { test } from 'node:test';

import { JOG_SPEEDS, jogVector } from './gamepad.ts';

const SIGNS = { x: 1, y: 1, z: 1 } as const;
const jog = (
  axes: number[],
  signs: { x: 1 | -1; y: 1 | -1; z: 1 | -1 } = SIGNS,
  speedScale = 1,
  labels?: { x?: string; y?: string; z?: string }
) => jogVector(axes, signs, speedScale, labels).map((c) => c + 0);

test('dead zone, missing axes, and unused right-X produce zero', () => {
  assert.deepEqual(jog([0.05, -0.08, 1, 0.1]), [0, 0, 0]);
  assert.deepEqual(jog([]), [0, 0, 0]);
});

test('directions match Voxel stage orientation and respect signs', () => {
  assert.deepEqual(jog([1, 0, 0, 0]), [1, 0, 0]);
  assert.deepEqual(jog([0, -1, 0, 0]), [0, 1, 0]);
  assert.deepEqual(jog([0, 0, 0, -1]), [0, 0, 1]);
  assert.deepEqual(jog([1, -1, 0, -1], { x: -1, y: 1, z: -1 }).map(Math.sign), [-1, 1, -1]);
});

test('remapped stage labels route left stick to physical X/Y and right stick to physical Z', () => {
  const xScanLabels = { x: 'Y', y: 'Z', z: 'X' };
  // Left stick right (+X) -> logical z
  assert.deepEqual(jog([1, 0, 0, 0], SIGNS, 1, xScanLabels), [0, 0, 1]);
  // Left stick up (+Y) -> logical x
  assert.deepEqual(jog([0, -1, 0, 0], SIGNS, 1, xScanLabels), [1, 0, 0]);
  // Right stick up (+Z) -> logical y
  assert.deepEqual(jog([0, 0, 0, -1], SIGNS, 1, xScanLabels), [0, 1, 0]);
});

test('speed presets scale Slow/Med/Fast cleanly with Fast at 3x Slow', () => {
  const [slow, med, fast] = JOG_SPEEDS;
  assert.equal(slow.id, 'slow');
  assert.equal(med.id, 'med');
  assert.equal(fast.id, 'fast');
  assert.ok(Math.abs(fast.scale / slow.scale - 3) < 1e-12);
  assert.ok(Math.abs(med.scale / slow.scale - 2) < 1e-12);
  assert.deepEqual(jog([1, 0, 0, 0], SIGNS, slow.scale), [1 / 3, 0, 0]);
  assert.deepEqual(jog([0, 0, 0, -1], SIGNS, med.scale), [0, 0, 2 / 3]);
});

test('response curve is slow near center, monotonic, and bounded on diagonals', () => {
  const [small] = jog([0.22, 0, 0, 0]);
  const [mid] = jog([0.5, 0, 0, 0]);
  assert.ok(small > 0 && small < 0.05 && mid < 0.3);
  const [dx, dy] = jog([1, 1, 0, 0]);
  assert.ok(Math.hypot(dx, dy) <= 1 + 1e-12);
});
