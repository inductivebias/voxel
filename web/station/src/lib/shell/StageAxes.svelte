<script lang="ts">
  import { startGamepadJog } from '$lib/core/gamepad';
  import type { Instrument } from '$lib/core/model';
  import { prefs } from '$lib/core/prefs';
  import { fromMicrometers, getSpatialUnit, toMicrometers } from '$lib/core/spatial-units';
  import { Button, SpinBox, toast } from '@voxel/kit';
  import { cn } from '@voxel/kit/cn';
  import { reportFailure } from '@voxel/kit/failures';
  import { Gamepad } from '@voxel/kit/icons';

  interface Props {
    instrument: Instrument;
  }

  let { instrument }: Props = $props();
  const stage = $derived(instrument.stage);
  const unit = $derived(getSpatialUnit(prefs.spatialUnit.get()));
  const AXIS_SPINS = [
    ['x', 10],
    ['y', 10],
    ['z', 10]
  ] as const;

  let armed = $state(false);
  let driving = $state(false);
  let padName = $state<string | null>(null);
  const jogAvailable = $derived(stage.jogAvailable);
  const jogOn = $derived(armed && !instrument.busy);
  const toggleJog = () => {
    if (armed) armed = false;
    else if (jogAvailable && !instrument.busy) armed = true;
  };

  $effect(() => {
    if (instrument.busy && armed) {
      armed = false;
      toast.warning('Jogging disabled: an acquisition started');
    } else if (!jogAvailable) {
      armed = false;
    }
  });

  $effect(() =>
    startGamepadJog({
      isArmed: () => jogOn,
      orientation: () => stage.orientation,
      send: (v) => instrument.sendJog(v),
      onToggle: toggleJog,
      onHalt: () => reportFailure(stage.halt()),
      onState: (name, isDriving) => {
        padName = name;
        driving = isDriving;
      }
    })
  );

  const jogLabel = $derived(instrument.busy ? 'Jog locked' : jogOn ? 'Jog ON' : 'Jog OFF');
  const jogTitle = $derived(
    !jogAvailable
      ? 'This stage does not support controller jogging'
      : instrument.busy
        ? 'Controller jogging is disabled while an acquisition runs'
        : `${jogOn ? 'Jogging ON — left bumper or click to turn off; B halts.' : 'Jogging OFF — left bumper or click to turn on.'}\n${padName ? `Controller: ${padName}` : 'No controller detected (USB on localhost/HTTPS)'}`
  );
</script>

<div
  class={cn(
    'grid min-w-0 grid-cols-[repeat(3,minmax(0,1fr))_auto_auto] gap-2 rounded-md',
    '@min-[64rem]/viewer-pane:grid-cols-[repeat(3,10rem)_auto_auto]',
    jogOn && 'outline-2 outline-offset-4 outline-success'
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
  <Button
    variant={jogOn ? 'success' : 'secondary'}
    class={cn(
      'w-full min-w-24 gap-1.5 whitespace-nowrap disabled:opacity-60',
      jogOn && 'ring-2 ring-success/60 ring-offset-1 ring-offset-shell'
    )}
    aria-pressed={jogOn}
    disabled={!jogAvailable || instrument.busy}
    title={jogTitle}
    onclick={toggleJog}
  >
    <Gamepad class={cn('size-4 shrink-0', !padName && 'opacity-50')} aria-hidden="true" />
    {jogLabel}
    <span
      class={cn(
        'size-2 shrink-0 rounded-full',
        driving ? 'animate-pulse bg-on-success' : jogOn ? 'bg-on-success/50' : 'bg-fg-muted/40'
      )}
      aria-hidden="true"
    ></span>
  </Button>
</div>
