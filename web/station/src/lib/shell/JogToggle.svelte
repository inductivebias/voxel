<!--
  Arms or disarms Xbox-controller stage jogging and shows its state at a glance.
  The left bumper on the controller toggles the same switch. Jogging is locked while a capture runs.
-->
<script lang="ts">
  import { GamepadJog } from '$lib/core/gamepad/controller.svelte';
  import type { Instrument } from '$lib/core/model';
  import { Button, toast } from '@voxel/kit';
  import { cn } from '@voxel/kit/cn';
  import { reportFailure } from '@voxel/kit/failures';
  import { Gamepad } from '@voxel/kit/icons';

  interface Props {
    instrument: Instrument;
  }

  let { instrument }: Props = $props();

  const gamepad = new GamepadJog(() => instrument);
  $effect(() => {
    gamepad.start();
    return () => gamepad.dispose();
  });

  const jog = $derived(instrument.jog);
  const locked = $derived(instrument.busy);
  const moving = $derived(jog.active || gamepad.driving);
  const speed = $derived(`${(jog.max_speed_um_s / 1000).toLocaleString(undefined, { maximumFractionDigits: 2 })} mm/s`);

  // Tell the operator when the server switched jogging off (capture started, client lost, axis error).
  let lastMessage: string | null = null;
  $effect(() => {
    const message = jog.message;
    if (message && message !== lastMessage) toast.warning(message);
    lastMessage = message;
  });

  const label = $derived(locked ? 'Jog locked' : jog.enabled ? 'Jog ON' : 'Jog OFF');
  const title = $derived.by(() => {
    if (!jog.available) return 'This stage does not support controller jogging';
    if (locked) return 'Controller jogging is disabled while an acquisition runs';
    const pad = !GamepadJog.supported
      ? 'This browser blocks controllers here: open the station on localhost or over HTTPS'
      : gamepad.connected
        ? `Controller: ${gamepad.name}`
        : 'No controller detected: plug it in by USB and press any button';
    const state = jog.enabled
      ? `Jogging is ON — sticks move the stage (up to ${speed}). Left bumper or click to turn off; B halts.`
      : 'Jogging is OFF. Press the left bumper or click to turn on.';
    return `${state}\n${pad}`;
  });
</script>

<Button
  variant={jog.enabled && !locked ? 'success' : 'secondary'}
  class={cn(
    'w-full min-w-24 gap-1.5 whitespace-nowrap disabled:opacity-60',
    jog.enabled && !locked && 'ring-2 ring-success/60 ring-offset-1 ring-offset-shell'
  )}
  aria-pressed={jog.enabled}
  disabled={!jog.available || locked}
  {title}
  onclick={() => reportFailure(instrument.setJogEnabled(!jog.enabled))}
>
  <Gamepad class={cn('size-4 shrink-0', !gamepad.connected && 'opacity-50')} aria-hidden="true" />
  {label}
  <span
    class={cn(
      'size-2 shrink-0 rounded-full',
      moving ? 'animate-pulse bg-on-success' : jog.enabled && !locked ? 'bg-on-success/50' : 'bg-fg-muted/40'
    )}
    aria-hidden="true"
  ></span>
  <span class="sr-only">{moving ? ', moving' : ''}{gamepad.connected ? '' : ', no controller connected'}</span>
</Button>
