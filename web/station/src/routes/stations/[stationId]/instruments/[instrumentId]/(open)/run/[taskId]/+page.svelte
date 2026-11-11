<script lang="ts">
  import { goto } from '$app/navigation';
  import { resolve } from '$app/paths';
  import { page } from '$app/state';
  import { getPageInstrument } from '$lib/context.svelte';
  import {
    convexHull,
    type EditContext,
    type IterationOrder,
    type LineScan,
    type Point2D,
    pointBounds,
    resizePointsToBounds,
    type TaskPatch,
    type TileOrder,
    type XYMode
  } from '$lib/core/model';
  import { prefs } from '$lib/core/prefs';
  import { getSpatialUnit } from '$lib/core/spatial-units';
  import FieldRow from '$lib/features/run/FieldRow.svelte';
  import StageNumericField from '$lib/features/run/StageNumericField.svelte';
  import PageHeader from '$lib/shell/PageHeader.svelte';
  import {
    Button,
    buttonVariants,
    Dialog,
    DropdownMenu,
    NumberBox,
    NumberInput,
    Section,
    Select,
    SpinBox
  } from '@voxel/kit';
  import { reportFailure } from '@voxel/kit/failures';
  import { displayName } from '@voxel/kit/format';
  import { Crosshair, GripVertical, Link, Plus, Trash, Unlink, X } from '@voxel/kit/icons';
  import * as Sortable from '@voxel/kit/sortable';

  const selected = getPageInstrument();
  const routeParams = $derived(selected.params);
  const instrument = $derived(selected.open);
  const taskId = $derived(page.params.taskId ?? '');
  const taskIndex = $derived(instrument?.plan.findIndex((candidate) => candidate.id === taskId) ?? -1);
  const task = $derived(taskIndex < 0 ? undefined : instrument?.plan[taskIndex]);
  const volumeRows = $derived(
    (instrument?.plannedVolumes ?? []).flatMap((volume, index) => (volume.task === taskId ? [{ volume, index }] : []))
  );
  const positionCount = $derived(new Set(volumeRows.map(({ volume }) => `${volume.x}\0${volume.y}`)).size);
  const layoutPoints = $derived(task?.xy.points ?? []);
  const sortablePoints = $derived(layoutPoints.map((point) => ({ key: `${point.x}\0${point.y}`, point })));
  const bounds = $derived(layoutPoints.length ? pointBounds(layoutPoints) : null);
  const currentPosition = $derived.by(() => {
    const x = instrument?.stage.x.position?.value;
    const y = instrument?.stage.y.position?.value;
    return x == null || y == null || !Number.isFinite(x) || !Number.isFinite(y) ? null : { x, y };
  });
  const canAddPoint = $derived(
    !!currentPosition &&
      !!task &&
      !layoutPoints.some((point) => point.x === currentPosition.x && point.y === currentPosition.y)
  );
  const canUseBoundingBox = $derived(
    new Set(layoutPoints.map(({ x }) => x)).size >= 2 && new Set(layoutPoints.map(({ y }) => y)).size >= 2
  );
  const canUseConvexHull = $derived(layoutPoints.length >= 3 && convexHull(layoutPoints).length >= 3);
  const XY_MODE_LABELS: Record<XYMode, string> = {
    explicit_points: 'Explicit points',
    bounding_box: 'Bounding box',
    convex_hull: 'Convex hull'
  };
  const xyModes = $derived<XYMode[]>([
    'explicit_points',
    ...(canUseBoundingBox ? ['bounding_box' as const] : []),
    ...(canUseConvexHull ? ['convex_hull' as const] : [])
  ]);
  const availableProfiles = $derived(
    Object.keys(instrument?.imaging.profiles ?? {}).filter((profileId) => !task?.profiles.includes(profileId))
  );
  const unit = $derived(getSpatialUnit(prefs.spatialUnit.get()));
  const coordinateFormat = new Intl.NumberFormat(undefined, {
    minimumSignificantDigits: 5,
    maximumSignificantDigits: 5,
    useGrouping: false
  });
  const planHref = $derived(resolve('/stations/[stationId]/instruments/[instrumentId]', routeParams));
  const disabled = $derived(!instrument || instrument.busy);

  const TRAVERSAL_LABELS: Record<TileOrder, string> = {
    sweep_row: 'Sweep rows',
    sweep_column: 'Sweep columns',
    snake_row: 'Snake rows',
    snake_column: 'Snake columns',
    nearest_neighbor: 'Nearest neighbor',
    optimized: 'Optimized',
    custom: 'As defined'
  };
  const ITERATION_LABELS: Record<IterationOrder, string> = {
    position_major: 'Position first',
    profile_major: 'Profile first'
  };
  const TRAVERSALS = Object.keys(TRAVERSAL_LABELS) as TileOrder[];
  const ITERATIONS = Object.keys(ITERATION_LABELS) as IterationOrder[];

  let deleteDialogOpen = $state(false);
  let overlapLink = $state<{ task: string; linked: boolean } | null>(null);
  const overlapLinked = $derived.by(() => {
    if (!task) return false;
    return overlapLink?.task === task.id ? overlapLink.linked : task.xy.overlap.x === task.xy.overlap.y;
  });

  function patchTask(patch: TaskPatch, context: EditContext = {}): void {
    if (!instrument || !task || disabled) return;
    reportFailure(instrument.updateTask(task.id, patch, context));
  }

  function updateScan(patch: Partial<LineScan>, context: EditContext = {}): void {
    if (task?.scan) patchTask({ scan: { ...task.scan, ...patch } }, context);
  }

  function updateRows(patch: Partial<NonNullable<LineScan['y_stack']>>, context: EditContext = {}): void {
    if (task?.scan?.y_stack) updateScan({ y_stack: { ...task.scan.y_stack, ...patch } }, context);
  }

  const sweepCount = $derived(
    task?.scan?.y_stack
      ? Math.floor((task.scan.y_stack.end - task.scan.y_stack.start) / task.scan.y_stack.step + 1e-9) + 1
      : 1
  );

  function updateHold(axis: 'x' | 'y' | 'z', value: number, context: EditContext): void {
    if (!task) return;
    if (axis === 'z') patchTask({ z: { type: 'fixed', start: value, end: value } }, context);
    else patchTask({ xy: { ...task.xy, points: [{ ...task.xy.points[0], [axis]: value }] } }, context);
  }

  function updateLayoutPoints(points: Point2D[], context: EditContext = {}): void {
    if (!task) return;
    patchTask({ xy: { ...task.xy, points } }, context);
  }

  function addPoint(): void {
    if (currentPosition && canAddPoint) updateLayoutPoints([...layoutPoints, currentPosition]);
  }

  function reorderLayoutPoints(items: { key: string; point: Point2D }[]): void {
    updateLayoutPoints(items.map(({ point }) => point));
  }

  function replaceLayoutPoint(index: number, point: Point2D, context?: EditContext): void {
    updateLayoutPoints(
      layoutPoints.map((currentPoint, pointIndex) => (pointIndex === index ? point : currentPoint)),
      context
    );
  }

  function updateLayoutPoint(index: number, axis: keyof Point2D, value: number, context: EditContext): void {
    const point = layoutPoints[index];
    if (!point || Object.is(point[axis], value)) return;
    replaceLayoutPoint(index, { ...point, [axis]: value }, context);
  }

  function canUseCurrentPoint(index: number): boolean {
    if (!currentPosition) return false;
    const point = layoutPoints[index];
    if (point?.x === currentPosition.x && point.y === currentPosition.y) return false;
    return !layoutPoints.some(
      (candidate, pointIndex) =>
        pointIndex !== index && candidate.x === currentPosition.x && candidate.y === currentPosition.y
    );
  }

  function useCurrentPoint(index: number): void {
    if (currentPosition && canUseCurrentPoint(index)) replaceLayoutPoint(index, { ...currentPosition });
  }

  function removeLayoutPoint(index: number): void {
    const minimum = task?.xy.mode === 'explicit_points' ? 1 : 3;
    if (layoutPoints.length <= minimum) return;
    updateLayoutPoints(layoutPoints.filter((_, pointIndex) => pointIndex !== index));
  }

  function setPointLayout(mode: XYMode): void {
    if (!task || disabled) return;
    if (mode === 'bounding_box' && !canUseBoundingBox) return;
    if (mode === 'convex_hull' && !canUseConvexHull) return;
    patchTask({ xy: { ...task.xy, mode } });
  }

  function updateBound(axis: keyof Point2D, edge: 'min' | 'max', value: number, context: EditContext): void {
    if (!task || !bounds || task.xy.mode !== 'bounding_box') return;
    const opposite =
      edge === 'min' ? (axis === 'x' ? bounds.maxX : bounds.maxY) : axis === 'x' ? bounds.minX : bounds.minY;
    if ((edge === 'min' && value >= opposite) || (edge === 'max' && value <= opposite)) return;
    const next = { ...bounds, [`${edge}${axis.toUpperCase()}`]: value };
    patchTask({ xy: { ...task.xy, points: resizePointsToBounds(task.xy.points, next) } }, context);
  }

  function updateOverlap(axis: keyof Point2D, value: number, context: EditContext): void {
    if (!task) return;
    const overlap = overlapLinked ? { x: value, y: value } : { ...task.xy.overlap, [axis]: value };
    patchTask({ xy: { ...task.xy, overlap } }, context);
  }

  function toggleOverlapLink(): void {
    if (!task) return;
    const linked = !overlapLinked;
    overlapLink = { task: task.id, linked };
    if (linked && task.xy.overlap.x !== task.xy.overlap.y) {
      patchTask({ xy: { ...task.xy, overlap: { x: task.xy.overlap.x, y: task.xy.overlap.x } } });
    }
  }

  function updateZ(field: 'start' | 'end', value: number, context: EditContext): void {
    if (!task) return;
    patchTask({ z: { ...task.z, [field]: value } }, context);
  }

  function toggleProfile(profileId: string, checked: boolean): void {
    if (!task) return;
    const profiles = checked
      ? [...new Set([...task.profiles, profileId])]
      : task.profiles.filter((candidate) => candidate !== profileId);
    if (profiles.length) patchTask({ profiles });
  }

  function reorderProfiles(profiles: string[]): void {
    patchTask({ profiles });
  }

  function scanPeriod(profileId: string): number | null {
    if (!instrument || !task?.scan) return null;
    const profile = instrument.imaging.profiles[profileId];
    const wiring = instrument.hal.continuous_scan;
    if (!profile || !wiring) return null;
    const sync = profile.sync[wiring.generator];
    if (!sync) return null;
    if (task.scan.target_frame_rate_hz != null) {
      return Math.max(
        1 / task.scan.target_frame_rate_hz,
        (sync.duration + wiring.rearm_margin_s + 1 / sync.sample_rate) * (1 + wiring.speed_tolerance)
      );
    }
    if (task.scan.camera_rate_fraction === null) return sync.duration + sync.rest_time;
    if (profileId !== instrument.activeProfileId) return null;
    const cameraIds = [...new Set(profile.channels.map((id) => instrument.imaging.channels[id].detection))];
    const limits = cameraIds.map((id) => {
      const divergence = instrument.divergence.get(id);
      if (divergence?.roiChanged || divergence?.changed.size) return null;
      return instrument.devices.get(id)?.getProp('acquisition_rate_limit_hz')?.value;
    });
    if (!limits.length || limits.some((v) => typeof v !== 'number' || !Number.isFinite(v) || v <= 0)) return null;
    const tolerance = 1 + wiring.speed_tolerance;
    const rate = (Math.min(...(limits as number[])) * (task.scan.camera_rate_fraction ?? 0.95)) / tolerance;
    return Math.max(1 / rate, (sync.duration + wiring.rearm_margin_s + 1 / sync.sample_rate) * tolerance);
  }

  function profileLabel(profileId: string): string {
    return instrument?.imaging.profiles[profileId]?.label || displayName(profileId);
  }

  function formatCoordinate(value: number): string {
    return Number.isFinite(value) ? coordinateFormat.format(value / unit.scale) : '—';
  }

  async function deleteTask(): Promise<void> {
    if (!instrument || !task || disabled) return;
    const request = instrument.removeTasks([task.id]);
    reportFailure(request);
    try {
      await request;
    } catch {
      return;
    }
    deleteDialogOpen = false;
    await goto(planHref);
  }
</script>

{#snippet coordinate(value: number, label: string, onchange: (value: number, context: EditContext) => void)}
  <div class="flex min-w-0 items-center">
    <NumberInput
      {value}
      onChange={onchange}
      step={unit.step * unit.scale}
      bigStep={unit.bigStep * unit.scale}
      displayScale={unit.scale}
      decimals={unit.decimals}
      align="right"
      {disabled}
      aria-label={label}
      class="min-w-0 flex-1 basis-0 px-1 leading-none"
    />
    <span class="pointer-events-none shrink-0 pr-1.5 pl-0.5 font-mono text-fg-muted">{unit.label}</span>
  </div>
{/snippet}

{#snippet positionChip(axis: string, value: string)}
  <span class="rounded-sm bg-element-bg px-1.5 py-0.5 whitespace-nowrap text-fg-muted">
    <span class="mr-1 text-fg-faint">{axis}</span>{value}
  </span>
{/snippet}

<PageHeader
  items={[
    { label: 'Run', href: planHref },
    { label: taskIndex >= 0 ? `Task ${taskIndex + 1}` : 'Task', title: taskId }
  ]}
>
  {#snippet trailing()}
    <Button
      variant="outline"
      size="icon-sm"
      class="text-fg-muted"
      disabled={!task || disabled}
      aria-label="Delete task"
      title="Delete task"
      onclick={() => (deleteDialogOpen = true)}
    >
      <Trash class="size-3.5" />
    </Button>
  {/snippet}
</PageHeader>
{#if instrument && task}
  {@const stage = instrument.stage}
  <div class="flex page-body flex-col gap-4 pb-0!">
    {#if task.scan}
      <Section heading={task.scan.y_stack ? 'XY-Scan' : 'X-Scan'} variant="bordered">
        <div class="flex flex-col gap-3">
          <FieldRow label="Axis">
            <Select
              value={task.scan.axis}
              options={task.scan.y_stack ? (['x'] as const) : (['x', 'y', 'z'] as const)}
              formatLabel={(axis) => axis.toUpperCase()}
              onChange={(axis) => updateScan({ axis })}
              {disabled}
            />
          </FieldRow>
          <FieldRow label="Window">
            <StageNumericField
              stage={instrument.stage}
              axis={task.scan.axis}
              value={task.scan.start}
              prefix="Start"
              {unit}
              onChange={(value, edit) => updateScan({ start: value }, edit)}
              {disabled}
            />
            <StageNumericField
              stage={instrument.stage}
              axis={task.scan.axis}
              value={task.scan.end}
              prefix="End"
              {unit}
              onChange={(value, edit) => updateScan({ end: value }, edit)}
              {disabled}
            />
          </FieldRow>
          <FieldRow label="Sampling">
            <NumberBox
              value={task.scan.spacing}
              min={0.001}
              step={0.001}
              bigStep={0.1}
              suffix="µm"
              aria-label="Sampling distance in micrometres"
              onEditStart={() => instrument.edits.hold()}
              onChange={(spacing, edit) => updateScan({ spacing }, edit)}
              {disabled}
            />
          </FieldRow>
          <FieldRow label="Targeted framerate">
            <NumberBox
              value={task.scan.target_frame_rate_hz ??
                (scanPeriod(task.profiles[0]) ? 1 / scanPeriod(task.profiles[0])! : 100)}
              min={0.001}
              step={0.001}
              bigStep={1}
              suffix="Hz"
              aria-label="Targeted framerate"
              onEditStart={() => instrument.edits.hold()}
              onChange={(target_frame_rate_hz, edit) =>
                updateScan({ target_frame_rate_hz, camera_rate_fraction: null }, edit)}
              {disabled}
            />
          </FieldRow>
          {#if task.scan.y_stack}
            <FieldRow label="Y range">
              <StageNumericField
                stage={instrument.stage}
                axis="y"
                value={task.scan.y_stack.start}
                prefix="Start"
                {unit}
                onChange={(start, edit) => updateRows({ start }, edit)}
                {disabled}
              />
              <StageNumericField
                stage={instrument.stage}
                axis="y"
                value={task.scan.y_stack.end}
                prefix="End"
                {unit}
                onChange={(end, edit) => updateRows({ end }, edit)}
                {disabled}
              />
            </FieldRow>
            <FieldRow label="Y step">
              <NumberBox
                value={task.scan.y_stack.step}
                min={0.001}
                step={0.001}
                bigStep={10}
                suffix="µm"
                aria-label="Y step between X sweeps"
                onEditStart={() => instrument.edits.hold()}
                onChange={(step, edit) => updateRows({ step }, edit)}
                {disabled}
              />
            </FieldRow>
          {/if}
          {#each ['x', 'y', 'z'] as const as axis}
            {#if axis !== task.scan.axis && !(axis === 'y' && task.scan.y_stack)}
              <FieldRow label={`Hold ${axis.toUpperCase()}`}>
                <StageNumericField
                  stage={instrument.stage}
                  {axis}
                  value={axis === 'z' ? task.z.start : task.xy.points[0][axis]}
                  {unit}
                  onChange={(value, edit) => updateHold(axis, value, edit)}
                  {disabled}
                />
              </FieldRow>
            {/if}
          {/each}
          <p class="text-sm text-fg-muted">
            {#if task.scan.y_stack}
              Each X end is exclusive. Y moves between sweeps, up to the inclusive Y end in whole steps. Z must already
              be at its hold position. Every sweep runs in positive X and is saved as a separate volume.
            {:else}
              The end position is exclusive. Held axes must already be at these positions; this task does not move them.
            {/if}
          </p>
          {#each task.profiles as profileId}
            {@const sync = Object.values(instrument.imaging.profiles[profileId]?.sync ?? {})[0]}
            {@const period = scanPeriod(profileId)}
            {#if sync && period !== null}
              {@const speed = task.scan.spacing / period}
              <div class="text-sm text-fg-muted">
                <p class="font-medium text-fg">{profileLabel(profileId)}</p>
                <ul class="mt-1 list-disc space-y-0.5 pl-5 tabular-nums">
                  <li>
                    Frame rate: {(1 / period).toFixed(2)} Hz ({task.scan.target_frame_rate_hz != null
                      ? 'targeted framerate'
                      : task.scan.camera_rate_fraction === null
                        ? 'manual DAQ timing'
                        : 'automatic camera timing'})
                  </li>
                  <li>{task.scan.axis.toUpperCase()} stage speed: {speed.toFixed(3)} µm/s</li>
                  <li>
                    Frames per camera: {Math.round(
                      (task.scan.end - task.scan.start) / task.scan.spacing
                    ).toLocaleString()}
                  </li>
                  <li>
                    Scan duration{task.scan.y_stack ? ' per sweep' : ''}: {(
                      (task.scan.end - task.scan.start) /
                      speed
                    ).toFixed(1)} s
                  </li>
                  {#if task.scan.y_stack}
                    <li>Y sweeps: {sweepCount.toLocaleString()} at {task.scan.y_stack.step} µm steps</li>
                    <li>
                      Total frames per camera: {(
                        Math.round((task.scan.end - task.scan.start) / task.scan.spacing) * sweepCount
                      ).toLocaleString()}
                    </li>
                    <li>
                      Total sweep time: {(((task.scan.end - task.scan.start) / speed) * sweepCount).toFixed(1)} s, plus return
                      travel and Y positioning
                    </li>
                  {/if}
                  <li>
                    Runway per side: {task.scan.runway_s} s ({formatCoordinate(speed * task.scan.runway_s)}
                    {unit.label})
                  </li>
                  <li>
                    Total travel: {formatCoordinate(task.scan.start - speed * task.scan.runway_s)}–{formatCoordinate(
                      task.scan.end + speed * task.scan.runway_s
                    )}
                    {unit.label}
                  </li>
                </ul>
                <p class="mt-2">
                  {#if task.scan.target_frame_rate_hz != null}
                    Stage speed is sampling × frame rate. DAQ re-arm limits may reduce the effective rate. Camera
                    exposure timing is checked at scan start; the target is not automatically reduced to the SDK rate.
                  {:else}
                    Automatic timing uses the slower camera with a safety margin and respects DAQ re-arm time. Timing is
                    checked again after applying the profile at scan start.
                  {/if}
                </p>
              </div>
            {:else}
              <p class="text-sm text-fg-muted">
                {profileLabel(profileId)}: automatic timing will be calculated after the profile is applied and camera
                timing is available. Exposure, sensor mode, ROI and binning determine the rate.
              </p>
            {/if}
          {/each}
        </div>
      </Section>
    {:else}
      <Section heading="XY coverage" variant="bordered">
        {#snippet trailing()}
          <Select
            class="w-42"
            value={task.xy.mode}
            options={xyModes}
            formatLabel={(mode) => XY_MODE_LABELS[mode]}
            {disabled}
            onChange={setPointLayout}
          />
        {/snippet}

        <div class="flex flex-col gap-3">
          {#if task.xy.mode === 'bounding_box' && bounds}
            <FieldRow label="Minimum">
              <StageNumericField
                stage={instrument.stage}
                axis="x"
                value={bounds.minX}
                max={bounds.maxX}
                prefix={stage.qualifiedLabel('x')}
                {unit}
                onEditStart={() => instrument.edits.hold()}
                onChange={(value, edit) => updateBound('x', 'min', value, edit)}
                {disabled}
              />
              <StageNumericField
                stage={instrument.stage}
                axis="y"
                value={bounds.minY}
                max={bounds.maxY}
                prefix={stage.qualifiedLabel('y')}
                {unit}
                onEditStart={() => instrument.edits.hold()}
                onChange={(value, edit) => updateBound('y', 'min', value, edit)}
                {disabled}
              />
            </FieldRow>
            <FieldRow label="Maximum">
              <StageNumericField
                stage={instrument.stage}
                axis="x"
                value={bounds.maxX}
                min={bounds.minX}
                prefix={stage.qualifiedLabel('x')}
                {unit}
                onEditStart={() => instrument.edits.hold()}
                onChange={(value, edit) => updateBound('x', 'max', value, edit)}
                {disabled}
              />
              <StageNumericField
                stage={instrument.stage}
                axis="y"
                value={bounds.maxY}
                min={bounds.minY}
                prefix={stage.qualifiedLabel('y')}
                {unit}
                onEditStart={() => instrument.edits.hold()}
                onChange={(value, edit) => updateBound('y', 'max', value, edit)}
                {disabled}
              />
            </FieldRow>
          {:else}
            <div class="mt-3 flex flex-col gap-2">
              <Sortable.Root
                items={sortablePoints}
                key={(item) => item.key}
                onReorder={reorderLayoutPoints}
                layout="vertical"
                flipDuration={120}
                containerClass="max-h-68 min-h-0 shrink"
                class="space-y-2"
              >
                {#snippet item(item: { key: string; point: Point2D }, index: number)}
                  <Sortable.Item
                    {item}
                    class="grid h-ui-sm grid-cols-[var(--spacing-ui-sm)_minmax(0,16rem)_minmax(0,1fr)_var(--spacing-ui-sm)_var(--spacing-ui-sm)] overflow-hidden rounded border border-control-line bg-element-bg transition-colors focus-within:border-focus hover:bg-element-hover"
                  >
                    {#if task.xy.mode === 'explicit_points'}
                      <Sortable.Handle
                        {item}
                        {disabled}
                        class="flex size-ui-sm shrink-0 cursor-grab touch-none items-center justify-start text-fg-muted active:cursor-grabbing"
                      >
                        <GripVertical class="mx-0.5 size-3.5" />
                      </Sortable.Handle>
                    {:else}
                      <span aria-hidden="true"></span>
                    {/if}

                    <div class="grid min-w-0 cursor-auto grid-cols-[minmax(0,1fr)_auto_minmax(0,1fr)] items-center">
                      {@render coordinate(
                        item.point.x,
                        `Point ${index + 1} ${stage.qualifiedLabel('x')}`,
                        (value, context) => updateLayoutPoint(index, 'x', value, context)
                      )}
                      <span class="px-0.5 text-fg-muted select-none">,</span>
                      {@render coordinate(
                        item.point.y,
                        `Point ${index + 1} ${stage.qualifiedLabel('y')}`,
                        (value, context) => updateLayoutPoint(index, 'y', value, context)
                      )}
                    </div>

                    <span aria-hidden="true"></span>

                    <Button
                      variant="ghost"
                      size="icon-sm"
                      class="h-full rounded-none border-y-0 border-r-0 border-l border-control-line text-fg-muted focus-visible:ring-offset-0 focus-visible:ring-inset"
                      disabled={disabled || !canUseCurrentPoint(index)}
                      aria-label={`Use current stage position for point ${index + 1}`}
                      title="Use current stage position"
                      onclick={() => useCurrentPoint(index)}
                    >
                      <Crosshair class="size-3.5" />
                    </Button>
                    <Button
                      variant="ghost"
                      size="icon-sm"
                      class="h-full rounded-none border-y-0 border-r-0 border-l border-control-line text-fg-muted hover:text-danger-fg focus-visible:ring-offset-0 focus-visible:ring-inset"
                      disabled={disabled || layoutPoints.length <= (task.xy.mode === 'explicit_points' ? 1 : 3)}
                      aria-label={`Delete point ${index + 1}`}
                      title={`Delete point ${index + 1}`}
                      onclick={() => removeLayoutPoint(index)}
                    >
                      <Trash class="size-3.5" />
                    </Button>
                  </Sortable.Item>
                {/snippet}
              </Sortable.Root>

              <Button variant="dashed" disabled={disabled || !canAddPoint} onclick={addPoint}>
                <Plus class="size-3.5" />
                Add current point
              </Button>
            </div>
          {/if}

          {#if task.xy.mode !== 'explicit_points'}
            <FieldRow label="Overlap" fieldsClass="*:basis-20">
              {#snippet trailing()}
                <button
                  type="button"
                  class={overlapLinked
                    ? 'cursor-pointer text-fg transition-colors'
                    : 'cursor-pointer text-fg-faint transition-colors hover:text-fg'}
                  {disabled}
                  aria-label={overlapLinked ? 'Unlink X and Y overlap' : 'Link X and Y overlap'}
                  title={overlapLinked ? 'X and Y overlap are linked' : 'Link X and Y overlap'}
                  onclick={toggleOverlapLink}
                >
                  {#if overlapLinked}<Link class="size-3" />{:else}<Unlink class="size-3" />{/if}
                </button>
              {/snippet}
              <SpinBox
                value={task.xy.overlap.x}
                min={0}
                max={0.99}
                step={0.01}
                bigStep={0.1}
                displayScale={0.01}
                decimals={0}
                prefix="X"
                suffix="%"
                onEditStart={() => instrument.edits.hold()}
                onChange={(value, edit) => updateOverlap('x', value, edit)}
                {disabled}
              />
              <SpinBox
                value={task.xy.overlap.y}
                min={0}
                max={0.99}
                step={0.01}
                bigStep={0.1}
                displayScale={0.01}
                decimals={0}
                prefix="Y"
                suffix="%"
                onEditStart={() => instrument.edits.hold()}
                onChange={(value, edit) => updateOverlap('y', value, edit)}
                {disabled}
              />
            </FieldRow>
          {/if}
        </div>
      </Section>

      <Section heading="Z coverage" variant="bordered">
        <FieldRow label="Range">
          <StageNumericField
            stage={instrument.stage}
            axis="z"
            value={task.z.start}
            max={task.z.end}
            prefix="Start"
            {unit}
            onEditStart={() => instrument.edits.hold()}
            onChange={(value, edit) => updateZ('start', value, edit)}
            {disabled}
          />
          <StageNumericField
            stage={instrument.stage}
            axis="z"
            value={task.z.end}
            min={task.z.start}
            prefix="End"
            {unit}
            onEditStart={() => instrument.edits.hold()}
            onChange={(value, edit) => updateZ('end', value, edit)}
            {disabled}
          />
        </FieldRow>
      </Section>
    {/if}

    <Section heading="Profiles" variant="bordered">
      {#snippet trailing()}
        <DropdownMenu.Root>
          <DropdownMenu.Trigger
            class={buttonVariants({
              variant: 'ghost',
              size: 'icon-sm',
              class: 'text-fg-muted'
            })}
            disabled={disabled || availableProfiles.length === 0}
            aria-label="Add profile"
            title="Add profile"
          >
            <Plus class="size-3.5" />
          </DropdownMenu.Trigger>
          <DropdownMenu.Content align="start" class="font-normal">
            {#each availableProfiles as profileId (profileId)}
              <DropdownMenu.Item onclick={() => toggleProfile(profileId, true)}>
                {profileLabel(profileId)}
              </DropdownMenu.Item>
            {/each}
          </DropdownMenu.Content>
        </DropdownMenu.Root>
      {/snippet}
      <Sortable.Root
        items={task.profiles}
        key={(profileId) => profileId}
        onReorder={reorderProfiles}
        layout="flow"
        class="flex min-h-5 flex-wrap items-center gap-1.5"
      >
        {#snippet item(profileId: string)}
          <Sortable.Item
            item={profileId}
            class="flex h-5 max-w-full min-w-0 items-center rounded-sm border border-line-faint bg-element-hover text-base"
          >
            <Sortable.Handle
              item={profileId}
              {disabled}
              class="flex h-full w-4 shrink-0 cursor-grab touch-none items-center justify-center text-fg-faint active:cursor-grabbing"
            >
              <GripVertical class="size-3" />
            </Sortable.Handle>
            <span class="min-w-0 truncate pr-1.5 pl-0.5 text-fg">{profileLabel(profileId)}</span>
            <button
              type="button"
              class="flex h-full shrink-0 items-center rounded-r px-0.5 text-fg-faint transition-colors hover:text-danger-fg disabled:pointer-events-none disabled:opacity-30"
              disabled={disabled || task.profiles.length === 1}
              aria-label={`Remove ${profileLabel(profileId)}`}
              title={`Remove ${profileLabel(profileId)}`}
              onclick={() => toggleProfile(profileId, false)}
            >
              <X class="size-3.5" />
            </button>
          </Sortable.Item>
        {/snippet}
      </Sortable.Root>
    </Section>

    {#if !task.scan}
      <Section heading="Ordering" variant="bordered">
        <div class="flex flex-col gap-3">
          <FieldRow label="Mode">
            <Select
              class="w-full"
              value={task.iteration}
              options={ITERATIONS}
              formatLabel={(order) => ITERATION_LABELS[order]}
              {disabled}
              onChange={(iteration) => patchTask({ iteration })}
            />
          </FieldRow>
          <FieldRow label="Path">
            <Select
              class="w-full"
              value={task.traversal}
              options={TRAVERSALS}
              formatLabel={(order) => TRAVERSAL_LABELS[order]}
              {disabled}
              onChange={(traversal) => patchTask({ traversal })}
            />
          </FieldRow>
        </div>
      </Section>
    {/if}

    <section
      class="flex min-h-40 flex-1 flex-col overflow-hidden rounded-t-lg border-x border-t border-line-muted bg-tint"
    >
      <header class="flex shrink-0 items-center justify-between gap-3 px-2.5">
        <h2 class="my-2.5 text-base text-fg">Volumes</h2>
        <span
          class="flex h-4 shrink-0 overflow-hidden rounded-sm border border-line-muted text-sm leading-none tabular-nums"
          aria-label={`${positionCount} positions, ${volumeRows.length} volumes`}
          title={`${positionCount} positions · ${volumeRows.length} volumes`}
        >
          <span class="flex min-w-7 items-center justify-center bg-element-bg px-1.5 text-fg-muted">
            {positionCount}
          </span>
          <span
            class="flex min-w-7 items-center justify-center border-l border-line-faint bg-element-hover px-1.5 text-fg"
          >
            {volumeRows.length}
          </span>
        </span>
      </header>
      {#if volumeRows.length}
        <div class="min-h-0 flex-1 divide-y divide-line-muted overflow-y-auto border-t border-line-faint pb-2">
          {#each volumeRows as { volume, index } (`${volume.profile}:${volume.x}:${volume.y}:${index}`)}
            <div
              class="flex min-w-0 items-center gap-1.5 py-1.5 pr-2.5 pl-2 hover:bg-tint-hover"
              aria-label={`Volume ${index + 1}, ${profileLabel(volume.profile)}, ${stage.qualifiedLabel('x')} ${formatCoordinate(volume.x)} ${unit.label}, ${stage.qualifiedLabel('y')} ${formatCoordinate(volume.y)} ${unit.label}, ${stage.qualifiedLabel('z')} ${formatCoordinate(volume.z_start)} to ${formatCoordinate(volume.z_end)} ${unit.label}`}
            >
              <div class="flex shrink-0 items-center gap-1 text-sm tabular-nums">
                {#if volume.scan}
                  {@render positionChip(
                    `${volume.scan.axis.toUpperCase()} scan`,
                    `${formatCoordinate(volume.scan.start)} – ${formatCoordinate(volume.scan.end)}`
                  )}
                {:else}
                  {@render positionChip(stage.qualifiedLabel('x'), formatCoordinate(volume.x))}
                  {@render positionChip(stage.qualifiedLabel('y'), formatCoordinate(volume.y))}
                  {@render positionChip(
                    stage.qualifiedLabel('z'),
                    `${formatCoordinate(volume.z_start)} – ${formatCoordinate(volume.z_end)}`
                  )}
                {/if}
              </div>
              <span
                class="min-w-20 truncate rounded-sm bg-tint px-1.5 py-0.5 whitespace-nowrap text-fg-muted"
                title={profileLabel(volume.profile)}
              >
                {profileLabel(volume.profile)}
              </span>
              <span class="ml-auto w-8 shrink-0 text-right text-sm text-fg-muted tabular-nums">
                {index + 1}
              </span>
            </div>
          {/each}
        </div>
      {:else}
        <p class="min-h-0 flex-1 overflow-y-auto border-t border-line-faint px-2.5 py-4 text-base text-fg-muted">
          No volumes have been resolved for this task.
        </p>
      {/if}
    </section>
  </div>
{:else if !instrument}
  <p class="page-body text-base text-fg-muted">Open this instrument to edit the task.</p>
{:else}
  <p class="page-body text-base text-fg-muted">This task is no longer in the plan.</p>
{/if}

<Dialog.Root bind:open={deleteDialogOpen}>
  <Dialog.Content size="md">
    <Dialog.Header>
      <Dialog.Title>Delete task</Dialog.Title>
      <Dialog.Description>Delete Task {taskIndex + 1}? This can be restored with Undo.</Dialog.Description>
    </Dialog.Header>
    <Dialog.Footer>
      <Button
        size="lg"
        variant="outline"
        disabled={instrument?.edits.pending}
        onclick={() => (deleteDialogOpen = false)}
      >
        Cancel
      </Button>
      <Button
        size="lg"
        variant="danger"
        loading={instrument?.edits.pending}
        disabled={!task || disabled}
        onclick={deleteTask}
      >
        Delete
      </Button>
    </Dialog.Footer>
  </Dialog.Content>
</Dialog.Root>
