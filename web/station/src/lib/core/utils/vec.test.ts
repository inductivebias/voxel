import assert from 'node:assert/strict';
import { test } from 'node:test';

import { fittedDrawBounds } from './vec.ts';

test('fittedDrawBounds centers pillarboxed and letterboxed viewports', () => {
  assert.deepEqual(fittedDrawBounds(2, { w: 1, h: 1 }, 1), { x: 0.25, y: 0, w: 0.5, h: 1 });
  assert.deepEqual(fittedDrawBounds(0.5, { w: 1, h: 1 }, 1), { x: 0, y: 0.25, w: 1, h: 0.5 });
  assert.deepEqual(fittedDrawBounds(2, { w: 0.5, h: 0.25 }, 1), { x: 0, y: 0, w: 1, h: 1 });
});

test('fittedDrawBounds falls back to full canvas on non-positive aspects', () => {
  assert.deepEqual(fittedDrawBounds(0, { w: 1, h: 1 }, 1), { x: 0, y: 0, w: 1, h: 1 });
  assert.deepEqual(fittedDrawBounds(1, { w: 1, h: 1 }, 0), { x: 0, y: 0, w: 1, h: 1 });
});
