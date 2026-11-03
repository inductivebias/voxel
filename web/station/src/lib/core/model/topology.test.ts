import assert from 'node:assert/strict';
import { test } from 'node:test';

import { buildDeviceUsageIndex, qualifiedStageAxisLabel, stageAxisLabel } from './topology.ts';
import type { HALConfig, StageAxisLabels } from './types';

/** Each of the 3 stage axes, x then y then z, as `format` names it. */
function names(format: typeof stageAxisLabel, labels: StageAxisLabels | undefined): string[] {
  return (['x', 'y', 'z'] as const).map((axis) => format(labels, axis));
}

test('an unlabeled stage axis shows its upper-case letter', () => {
  for (const labels of [undefined, {}]) {
    assert.deepEqual(names(stageAxisLabel, labels), ['X', 'Y', 'Z']);
    assert.deepEqual(names(qualifiedStageAxisLabel, labels), ['X', 'Y', 'Z']);
  }
});

test('a labeled stage axis shows its label, qualified by the logical axis only when they differ', () => {
  const labels = { x: 'Z', y: 'y', z: 'X' };
  assert.deepEqual(names(stageAxisLabel, labels), ['Z', 'y', 'X']);
  assert.deepEqual(names(qualifiedStageAxisLabel, labels), ['Z (x)', 'y', 'X (z)']);
});

test('the usage index names the 3 stage axes by label and does not read the labels as an axis', () => {
  const hal: HALConfig = {
    devices: {},
    nodes: {},
    stage: { x: 'a', y: 'b', z: 'c', labels: { x: 'Z', z: 'X' } },
    detection: {},
    illumination: {},
    optical_routing: {}
  };
  assert.deepEqual(Object.fromEntries(buildDeviceUsageIndex(hal)), {
    a: ['Stage Z axis'],
    b: ['Stage Y axis'],
    c: ['Stage X axis']
  });
});
