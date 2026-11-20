import assert from 'node:assert/strict';
import { test } from 'node:test';

import { jogVector } from './gamepad.ts';

const SIGNS = { x: 1, y: 1, z: 1 } as const;
const jog = (axes: number[], signs: { x: 1 | -1; y: 1 | -1; z: 1 | -1 } = SIGNS) =>
  jogVector(axes, signs).map((c) => c + 0);

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

test('response curve is slow near center, monotonic, and bounded on diagonals', () => {
  const [small] = jog([0.22, 0, 0, 0]);
  const [mid] = jog([0.5, 0, 0, 0]);
  assert.ok(small > 0 && small < 0.05 && mid < 0.3);
  const [dx, dy] = jog([1, 1, 0, 0]);
  assert.ok(Math.hypot(dx, dy) <= 1 + 1e-12);
});
