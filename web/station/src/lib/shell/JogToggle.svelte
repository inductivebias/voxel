<!--
  Arms or disarms Xbox-controller stage jogging in this browser and shows its state at a glance.
  The left bumper on the controller toggles the same switch. Jogging is locked while a capture runs.
-->
<script lang="ts">
  import { GamepadJog } from '$lib/core/gamepad/controller.svelte';
  import type { Instrument } from '$lib/core/model';
  import { Button, toast } from '@voxel/kit';
  import { cn } from '@voxel/kit/cn';
  import { Gamepad } from '@voxel/kit/icons';

  interface Props {
    instrument: Instrument;
    armed?: boolean;
  }

  let { instrument, armed = $bindable(false) }: Props = $props();

  const gamepad = new GamepadJog(() => instrument);
  $effect(() => {
    gamepad.start();
    return () => gamepad.dispose();
  });

  const available = $derived(instrument.stage.jogAvailable);
  const locked = $derived(instrument.busy);
  const on = $derived(gamepad.armed && !locked);
  const moving = $derived(gamepad.driving);
  const connected = $derived(gamepad.name !== null);

  $effect(() => {
    armed = on;
  });

  // Disarm and notify the operator when an acquisition starts; disarm quietly if jogging becomes unavailable.
  $effect(() => {
    if (locked && gamepad.armed) {
      gamepad.disarm();
      toast.warning('Jogging disabled: an acquisition started');
    } else if (!available && gamepad.armed) {
      gamepad.disarm();
    }
  });

  const label = $derived(locked ? 'Jog locked' : on ? 'Jog ON' : 'Jog OFF');
  const title = $derived.by(() => {
    if (!available) return 'This stage does not support controller jogging';
    if (locked) return 'Controller jogging is disabled while an acquisition runs';
    const pad = !GamepadJog.supported
      ? 'This browser blocks controllers here: open the station on localhost or over HTTPS'
      : connected
        ? `Controller: ${gamepad.name}`
        : 'No controller detected: plug it in by USB and press any button';
    const state = on
      ? 'Jogging is ON — sticks move the stage. Left bumper or click to turn off; B halts.'
      : 'Jogging is OFF. Press the left bumper or click to turn on.';
    return `${state}\n${pad}`;
  });
</script>

<Button
  variant={on ? 'success' : 'secondary'}
  class={cn(
    'w-full min-w-24 gap-1.5 whitespace-nowrap disabled:opacity-60',
    on && 'ring-2 ring-success/60 ring-offset-1 ring-offset-shell'
  )}
  aria-pressed={on}
  disabled={!available || locked}
  {title}
  onclick={() => gamepad.toggle()}
>
  <Gamepad class={cn('size-4 shrink-0', !connected && 'opacity-50')} aria-hidden="true" />
  {label}
  <span
    class={cn(
      'size-2 shrink-0 rounded-full',
      moving ? 'animate-pulse bg-on-success' : on ? 'bg-on-success/50' : 'bg-fg-muted/40'
    )}
    aria-hidden="true"
  ></span>
  <span class="sr-only">{moving ? ', moving' : ''}{connected ? '' : ', no controller connected'}</span>
</Button>
