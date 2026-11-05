import { displayName } from '@voxel/kit/format';

import type { Check } from '../types';

export const lasersInProfile: Check = function* ({ instrument }) {
  for (const laser of instrument.lasers.values()) {
    if (laser.outputEnabled?.value !== true) continue;
    if (Object.hasOwn(instrument.activeProfile?.lasers ?? {}, laser.id)) continue;
    yield {
      code: 'laser.not_in_profile',
      level: 'warning',
      title: `${displayName(laser.id)} is on, but not in the active profile`,
      detail: 'This laser is not selected in the profile.',
      subject: { device: laser.id },
      actions: [{ label: 'Turn off', run: () => laser.disableOutput() }]
    };
  }
};
