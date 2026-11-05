<script lang="ts">
  import { getOpenInstrument } from '$lib/context.svelte';
  import {
    deviceLocked,
    EnumeratedModel,
    type LaserHandle,
    type ModulationMode,
    NumericModel,
    type Prop
  } from '$lib/core/model';
  import LaserPowerPlot from '$lib/features/devices/LaserPowerPlot.svelte';
  import { deviceIdentity } from '$lib/features/devices/snippets.svelte';
  import LinkToggle from '$lib/features/imaging/LinkToggle.svelte';
  import PropertyDivergenceDot from '$lib/features/imaging/PropertyDivergenceDot.svelte';
  import { deviceHref } from '$lib/navigation';
  import PageHeader from '$lib/shell/PageHeader.svelte';
  import { NumberBox, RadioBar, Section, Select, Slider, Switch } from '@voxel/kit';
  import { reportFailure } from '@voxel/kit/failures';
  import { Loader } from '@voxel/kit/icons';

  type Setting = {
    name: string;
    deviceId: string;
    prop: Prop | undefined;
    locked: boolean;
    measured?: Prop;
    note?: string;
  };

  const MODE_LABELS: Record<ModulationMode, string> = {
    software: 'CW (software)',
    digital: 'Digital',
    analog: 'Analog'
  };
  const COMMAND_LABELS: Record<string, string> = {
    source_enable: 'Enabling source',
    source_disable: 'Disabling source',
    output_enable: 'Enabling output',
    output_disable: 'Disabling output',
    set_modulation_mode: 'Changing modulation mode'
  };

  const instrument = $derived.by(getOpenInstrument());

  function selectLaser(laser: LaserHandle, on: boolean): void {
    const lasers = { ...instrument.activeProfile?.lasers };
    if (on) lasers[laser.id] = 'software';
    else delete lasers[laser.id];
    reportFailure(instrument.updateProfile({ lasers }));
  }

  function saveMode(laser: LaserHandle, mode: ModulationMode): void {
    reportFailure(instrument.updateProfile({ lasers: { ...instrument.activeProfile?.lasers, [laser.id]: mode } }));
  }

  async function toggleOutput(laser: LaserHandle, on: boolean): Promise<void> {
    if (on) {
      await laser.enableSource(true);
      await laser.enableOutput(true);
    } else {
      await laser.disableOutput();
    }
  }

  function reading(value: unknown): string {
    return typeof value === 'number' && Number.isFinite(value) ? Number(value.toPrecision(5)).toString() : '—';
  }

  /** A measured value with 1 decimal, so the reading keeps its width while the power changes. */
  function measuredText(value: unknown): string {
    return typeof value === 'number' && Number.isFinite(value) ? value.toFixed(1) : '—';
  }

  /** A slider needs both limits. Without them, the native input falls back to 0 to 100. */
  function hasRange({ min, max }: NumericModel): boolean {
    return min != null && max != null && Number.isFinite(min) && Number.isFinite(max) && max > min;
  }

  function modesOf(laser: LaserHandle): ModulationMode[] {
    const model = laser.modulationMode?.model;
    return model instanceof EnumeratedModel
      ? model.options.filter(
          (mode): mode is ModulationMode => typeof mode === 'string' && Object.hasOwn(MODE_LABELS, mode)
        )
      : [];
  }

  function commandLabel(laser: LaserHandle): string {
    const command = [...laser.pendingCommands][0];
    return (command && COMMAND_LABELS[command]) || 'Updating laser';
  }
</script>

{#snippet setting({ name, deviceId, prop, locked, measured, note }: Setting)}
  {@const fullName = prop?.info.label ?? name}
  {@const units = prop?.info.units ?? ''}
  {@const editable = !locked && prop?.access === 'rw'}
  <div class="col-span-full grid h-ui-sm grid-cols-subgrid items-stretch" role="group" aria-label={name} title={note}>
    <span class="flex items-center justify-between gap-1 pr-1 text-sm whitespace-nowrap text-fg-muted">
      <span class="flex min-w-0 items-center gap-1">
        {name}
        {#if prop}<PropertyDivergenceDot {instrument} {deviceId} property={prop.info.name} />{/if}
      </span>
      {#if prop}<LinkToggle {instrument} {deviceId} property={prop.info.name} />{/if}
    </span>
    <div class="min-w-0">
      {#if prop?.model instanceof NumericModel && hasRange(prop.model)}
        <Slider variant="bar" model={prop.model} disabled={!editable} aria-label={fullName} />
      {:else if prop?.model instanceof EnumeratedModel && prop.model.options.length > 4}
        <Select
          model={prop.model as EnumeratedModel<number>}
          formatLabel={(value) => `${reading(value)} ${units}`}
          disabled={!editable}
          aria-label={fullName}
        />
      {:else if prop?.model instanceof EnumeratedModel}
        <RadioBar
          model={prop.model as EnumeratedModel<number>}
          formatLabel={reading}
          disabled={!editable}
          aria-label={fullName}
        />
      {/if}
    </div>
    {#if prop?.model instanceof NumericModel}
      <NumberBox
        model={prop.model}
        suffix={units || undefined}
        disabled={!editable}
        title={note ? undefined : prop.info.desc}
        aria-label={fullName}
      />
    {:else}
      <span class="flex items-center justify-end gap-1 font-mono text-sm tabular-nums">
        <span class="text-fg">{reading(prop?.value)}</span><span class="text-fg-muted">{units}</span>
      </span>
    {/if}
    {#if measured}
      <span
        class="flex items-center justify-end gap-1 px-1 font-mono text-sm text-fg-muted tabular-nums"
        title={measured.info.desc ?? measured.info.label}
        ><span>{measuredText(measured.value)}</span><span>{measured.info.units}</span></span
      >
    {/if}
  </div>
{/snippet}

<PageHeader items={[{ label: 'Lasers' }]}>
  {#snippet trailing()}
    <span class="text-sm text-fg-faint">{instrument.lasers.size}</span>
  {/snippet}
</PageHeader>
<div class="page-body">
  <div class="grid grid-cols-1 gap-3">
    {#each instrument.lasers.values() as laser (laser.id)}
      {@const selected = Object.hasOwn(instrument.activeProfile?.lasers ?? {}, laser.id)}
      {@const enabled = laser.outputEnabled?.value === true}
      {@const unavailable = deviceLocked(laser, instrument)}
      {@const commandRunning = laser.pendingCommands.size > 0}
      {@const modes = modesOf(laser)}
      <Section variant="raised">
        {#snippet heading()}
          <span class="inline-flex max-w-full items-center gap-2">
            <a
              href={deviceHref({ stationId: instrument.stationId, instrumentId: instrument.id }, laser.id)}
              class="flex items-center gap-2 rounded-sm hover:underline focus-visible:outline-2 focus-visible:outline-focus"
              title="Open laser device"
            >
              {@render deviceIdentity(laser.wavelength?.value ? `${laser.wavelength.value} nm` : laser.id)}
            </a>
            <span
              class="inline-flex shrink-0 items-center"
              title="Manually enable or inhibit laser output now, using its current modulation mode. In CW mode, enabling output turns on the beam immediately. This does not change scan selection."
            >
              <Switch
                value={enabled}
                disabled={unavailable || commandRunning}
                onChange={(on) => reportFailure(toggleOutput(laser, on))}
                aria-label="Manual laser output"
              />
            </span>
          </span>
        {/snippet}
        {#snippet trailing()}
          <div class="flex items-center gap-1">
            <span
              class="flex size-3 shrink-0 items-center justify-center"
              title={commandRunning ? commandLabel(laser) : undefined}
            >
              {#if commandRunning}
                <span
                  role="status"
                  aria-label={commandLabel(laser)}
                  class="animate-in delay-250 fill-mode-backwards fade-in"
                  ><Loader class="size-3 animate-spin text-fg-muted" /></span
                >
              {/if}
            </span>
            <RadioBar
              value={instrument.activeProfile?.lasers[laser.id] ?? 'software'}
              options={modes}
              formatLabel={(mode) => MODE_LABELS[mode]}
              onChange={(mode) => saveMode(laser, mode)}
              disabled={unavailable || !selected || instrument.activity.kind !== 'idle'}
              size="xs"
              aria-label="Modulation mode"
              title="Profile modulation mode; applied when preview or acquisition starts"
            />
            <span
              class="ml-2 inline-flex items-center gap-1"
              title="Include this laser in the active profile's preview and acquisition. Voxel enables its output when preview or capture starts and inhibits it when stopped. Selecting this does not turn the laser on now."
            >
              <span class="text-sm">scan</span>
              <Switch
                value={selected}
                onChange={(on) => selectLaser(laser, on)}
                disabled={instrument.activity.kind !== 'idle' || instrument.edits.pending}
                aria-label="Use laser in preview and scan"
              />
            </span>
          </div>
        {/snippet}
        <div class="flex min-w-0 flex-col gap-2">
          <LaserPowerPlot {laser} />
          <div
            class="grid grid-cols-[minmax(--spacing(20),max-content)_minmax(0,1fr)_auto_--spacing(22)] gap-x-1 gap-y-2"
          >
            {@render setting({
              name: 'Source',
              deviceId: laser.id,
              prop: laser.sourcePowerSetpoint,
              locked: unavailable,
              measured: laser.sourcePower
            })}
            {@render setting({
              name: 'Output',
              deviceId: laser.id,
              prop: laser.outputLevel,
              locked: unavailable || !laser.outputLevelActive,
              measured: laser.outputPower,
              note: laser.outputLevelActive ? undefined : 'External analog modulation controls intensity'
            })}
          </div>
        </div>
      </Section>
    {:else}
      <p class="text-sm text-fg-muted">No lasers available.</p>
    {/each}
  </div>
</div>
