import assert from 'node:assert/strict';
import { test } from 'node:test';

import { type AxisSigns, curve, DEAD_ZONE, isRest, jogVector, shapeAxis, shapeStick } from './mapping.ts';

const SIGNS = { x: 1, y: 1, z: 1 } as const;
/** `+ 0` turns -0 into 0 so strict deepEqual compares values. */
const jog = (axes: number[], signs: AxisSigns = SIGNS) => jogVector(axes, signs).map((c) => c + 0);

test('a resting or barely touched stick does not move the stage', () => {
  assert.deepEqual(jog([0.05, -0.08, 0, 0.1]), [0, 0, 0]);
  assert.ok(isRest(jogVector([0, 0, 0, 0], SIGNS)));
});

test('a small push is slow and a full push reaches full speed', () => {
  assert.ok(shapeAxis(DEAD_ZONE + 0.1) < 0.05);
  assert.ok(shapeAxis(0.5) < 0.3);
  assert.equal(shapeAxis(1), 1);
  assert.equal(shapeAxis(-1), -1);
  assert.equal(curve(2), 1);
  assert.equal(curve(DEAD_ZONE), 0);
});

test('speed rises monotonically with deflection', () => {
  let previous = 0;
  for (let i = 1; i <= 100; i++) {
    const value = shapeAxis(i / 100);
    assert.ok(value >= previous);
    previous = value;
  }
});

test('left stick right/up drives +X/+Y and right stick up drives +Z, as Voxel draws them', () => {
  // Browser sticks report +y downward, so "up" is a negative reading.
  assert.deepEqual(jog([1, 0, 0, 0]), [1, 0, 0]);
  assert.deepEqual(jog([0, -1, 0, 0]), [0, 1, 0]);
  assert.deepEqual(jog([0, 0, 0, -1]), [0, 0, 1]);
  assert.deepEqual(jog([0, 0, 1, 0]), [0, 0, 0]); // Right stick sideways is unused.
});

test('stage orientation signs flip the matching axis', () => {
  const raw = [0.7, -0.4, 0, -0.9];
  const normal = jog(raw);
  assert.deepEqual(jog(raw, { x: -1, y: 1, z: -1 }), [-normal[0], normal[1], -normal[2]]);
});

test('diagonals keep their direction and never exceed full speed', () => {
  const [x, y] = shapeStick(1, 1);
  assert.ok(Math.abs(x - y) < 1e-12);
  assert.ok(Math.hypot(x, y) <= 1 + 1e-12);
});
