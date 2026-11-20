<script lang="ts">
  import type { Instrument } from '$lib/core/model';
  import { prefs } from '$lib/core/prefs';
  import { fromMicrometers, getSpatialUnit, toMicrometers } from '$lib/core/spatial-units';
  import { Button, SpinBox } from '@voxel/kit';
  import { cn } from '@voxel/kit/cn';
  import { reportFailure } from '@voxel/kit/failures';

  import JogToggle from './JogToggle.svelte';

  interface Props {
    instrument: Instrument;
  }

  let { instrument }: Props = $props();
  const stage = $derived(instrument.stage);
  const jogging = $derived(instrument.jog.enabled && !instrument.busy);

  const unit = $derived(getSpatialUnit(prefs.spatialUnit.get()));
  // The step of each stage axis box, in µm, whatever unit shows.
  const AXIS_SPINS = [
    ['x', 10],
    ['y', 10],
    ['z', 10]
  ] as const;
</script>

<div
  class={cn(
    'grid min-w-0 grid-cols-[repeat(3,minmax(0,1fr))_auto_auto] gap-2 rounded-md',
    '@min-[64rem]/viewer-pane:grid-cols-[repeat(3,10rem)_auto_auto]',
    jogging && 'outline-2 outline-offset-4 outline-success'
  )}
  role="group"
  aria-label="Stage movement controls"
>
  {#each AXIS_SPINS as [axis, step] (axis)}
    <SpinBox
      value={fromMicrometers(stage.position(axis), unit.value)}
      onChange={(value) => reportFailure(stage.axis(axis)?.move(toMicrometers(value, unit.value)))}
      min={fromMicrometers(stage.axis(axis)?.lowerLimit?.value ?? 0, unit.value)}
      max={fromMicrometers(stage.axis(axis)?.upperLimit?.value ?? 1, unit.value)}
      step={fromMicrometers(step, unit.value)}
      decimals={unit.decimals}
      prefix={stage.label(axis)}
      suffix={unit.label}
      class={cn('w-full min-w-0', stage.moving(axis) && 'animate-pulse text-fg-muted')}
    />
  {/each}
  <Button
    variant={stage.anyMoving ? 'danger' : 'secondary'}
    class="w-full min-w-14 disabled:opacity-100"
    onclick={() => reportFailure(stage.halt())}
    disabled={!stage.anyMoving}
    title="Halt stage motion"
  >
    Halt
  </Button>
  <JogToggle {instrument} />
</div>
