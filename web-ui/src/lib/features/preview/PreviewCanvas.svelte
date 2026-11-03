<script lang="ts">
  import { watch } from 'runed';
  import { onMount } from 'svelte';
  import { SvelteMap } from 'svelte/reactivity';

  import { prefs } from '$lib/core/prefs';
  import { clampTopLeft, displayName, pref } from '$lib/core/utils';
  import { RadioBar } from '$lib/kit';

  import PreviewChannels from './PreviewChannels.svelte';
  import PreviewNavigationControls from './PreviewNavigationControls.svelte';
  import { type PreviewChannel, type PreviewSession, wheelZoomFactor } from './session.svelte';

  interface Props {
    previewer: PreviewSession;
    /** Field of view as a `[width, height]` µm tuple (`instrument.fov`), or null when unavailable. */
    fov: [number, number] | null;
  }

  let { previewer, fov }: Props = $props();

  const cameraLayout = pref<'separate' | 'overlay'>('preview:camera-layout', 'separate');
  const crosshairVisible = prefs.viewer.crosshairVisible;
  const crosshairIndependent = prefs.viewer.crosshairIndependent;
  const DEFAULT_CROSSHAIR_POS = { normX: 0.5, normY: 0.5 } as const;
  let sharedCrosshairPos = $state.raw<{ normX: number; normY: number }>(DEFAULT_CROSSHAIR_POS);
  const crosshairPositions = new SvelteMap<string, { normX: number; normY: number }>();
  let activeCrosshairCamera = $state<string | 'all'>('all');

  const visibleChannels = $derived(previewer.channels.filter((channel) => channel.name && channel.visible));
  function cameraKey(channel: PreviewChannel): string {
    return channel.config?.detection ?? channel.name ?? String(channel.idx);
  }
  const cameraOptions = $derived.by(() => {
    const options: { id: string; label: string }[] = [];
    for (const channel of visibleChannels) {
      const id = cameraKey(channel);
      if (options.some((opt) => opt.id === id)) continue;
      options.push({ id, label: displayName(id) });
    }
    return options;
  });
  // Channels sharing one sensor remain composited; independent cameras get individual canvases.
  const hasMultipleCameras = $derived(cameraOptions.length > 1);
  const showSeparate = $derived(hasMultipleCameras && cameraLayout.get() === 'separate');

  function getCrosshairPos(cameraId?: string): { normX: number; normY: number } {
    if (!crosshairIndependent.get() || !cameraId) return sharedCrosshairPos;
    return crosshairPositions.get(cameraId) ?? DEFAULT_CROSSHAIR_POS;
  }

  function setCrosshairPos(pos: { normX: number; normY: number }, cameraId?: string | 'all') {
    if (!crosshairIndependent.get() || !cameraId || cameraId === 'all') {
      sharedCrosshairPos = pos;
      for (const cam of cameraOptions) crosshairPositions.set(cam.id, pos);
      return;
    }
    crosshairPositions.set(cameraId, pos);
  }

  function centerCrosshair() {
    setCrosshairPos(DEFAULT_CROSSHAIR_POS, activeCrosshairCamera);
  }

  let canvasEl = $state<HTMLCanvasElement | null>(null);
  const separateCanvases = new SvelteMap<number, HTMLCanvasElement>();
  let isRendering = false;
  let needsRedraw = false;
  let drawPending = false;
  let animFrameId: number | null = null;

  // Live image size, measured from the active canvas each frame (see syncSize).
  let viewW = $state(0);
  let viewH = $state(0);

  watch(
    () => previewer.redrawGeneration,
    () => {
      needsRedraw = true;
    }
  );

  // Measure the container from the DOM and keep the backing store + model aspect in sync. A ResizeObserver
  // can latch a 0 size when this canvas remounts inside the mode-switch fade; clientWidth never does.
  function syncSize() {
    const canvases = showSeparate ? [...separateCanvases.values()] : canvasEl ? [canvasEl] : [];
    if (!canvases.length) return;
    const dpr = devicePixelRatio;
    for (const canvas of canvases) {
      const w = canvas.clientWidth;
      const h = canvas.clientHeight;
      if (w <= 0 || h <= 0) continue;
      const width = Math.round(w * dpr);
      const height = Math.round(h * dpr);
      if (canvas.width !== width || canvas.height !== height) {
        canvas.width = width;
        canvas.height = height;
        needsRedraw = true;
      }
    }
    const w = canvases[0].clientWidth;
    const h = canvases[0].clientHeight;
    if (w !== viewW || h !== viewH) {
      viewW = w;
      viewH = h;
      if (w > 0 && h > 0) previewer.setDisplayAspect(w / h);
    }
  }

  // Pan/zoom on the canvas element itself, so overlay siblings (controls, navigator) never trip it.
  function setupPanZoom(el: HTMLCanvasElement, cameraId?: string): () => void {
    let isPanning = false;
    let isDraggingCrosshair = false;
    let dragTargetCamera: string | 'all' | undefined = cameraId;
    let panStartX = 0;
    let panStartY = 0;
    let startViewport = { ...previewer.viewport };

    const normalizedPoint = (e: MouseEvent) => {
      const rect = el.getBoundingClientRect();
      if (rect.width <= 0 || rect.height <= 0) return null;
      return {
        normX: Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width)),
        normY: Math.max(0, Math.min(1, (e.clientY - rect.top) / rect.height))
      };
    };

    const resolveOverlayTarget = (pt: { normX: number; normY: number }): string | 'all' | undefined => {
      if (cameraId) return cameraId;
      if (!crosshairIndependent.get() || cameraOptions.length <= 1) return cameraOptions[0]?.id ?? 'all';
      if (activeCrosshairCamera !== 'all') return activeCrosshairCamera;
      // In overlay mode with "All" selected and per-camera enabled, pick the nearest camera crosshair.
      let nearestId = cameraOptions[0].id;
      let nearestDist = Infinity;
      for (const cam of cameraOptions) {
        const pos = getCrosshairPos(cam.id);
        const dist = Math.hypot(pt.normX - pos.normX, pt.normY - pos.normY);
        if (dist < nearestDist) {
          nearestDist = dist;
          nearestId = cam.id;
        }
      }
      return nearestId;
    };

    const updateCrosshair = (e: MouseEvent) => {
      const pt = normalizedPoint(e);
      if (!pt) return;
      setCrosshairPos(pt, dragTargetCamera ?? activeCrosshairCamera);
    };

    const pointerDown = (e: PointerEvent) => {
      if (e.button !== 0) return;
      el.setPointerCapture(e.pointerId);
      if (crosshairVisible.get()) {
        const pt = normalizedPoint(e);
        dragTargetCamera = pt ? resolveOverlayTarget(pt) : cameraId;
        if (cameraId && crosshairIndependent.get()) {
          activeCrosshairCamera = cameraId;
        }
        isDraggingCrosshair = true;
        if (pt) setCrosshairPos(pt, dragTargetCamera);
        return;
      }
      isPanning = true;
      panStartX = e.clientX;
      panStartY = e.clientY;
      startViewport = { ...previewer.viewport };
    };

    const pointerMove = (e: PointerEvent) => {
      if (isDraggingCrosshair) {
        updateCrosshair(e);
        return;
      }
      if (!isPanning) return;
      const rect = el.getBoundingClientRect();
      const dx = ((e.clientX - panStartX) / rect.width) * previewer.viewport.w;
      const dy = ((e.clientY - panStartY) / rect.height) * previewer.viewport.h;
      const newX = clampTopLeft(startViewport.x - dx, previewer.viewport.w);
      const newY = clampTopLeft(startViewport.y - dy, previewer.viewport.h);
      previewer.setViewport({ x: newX, y: newY, w: previewer.viewport.w, h: previewer.viewport.h });
    };

    const pointerUp = (e: PointerEvent) => {
      if (e.button !== 0) return;
      if (el.hasPointerCapture(e.pointerId)) el.releasePointerCapture(e.pointerId);
      isPanning = false;
      isDraggingCrosshair = false;
    };

    const wheel = (e: WheelEvent) => {
      e.preventDefault();
      const rect = el.getBoundingClientRect();
      const vp = previewer.viewport;
      // Keep the sensor point under the cursor fixed on screen.
      const mouseX = (e.clientX - rect.left) / rect.width;
      const mouseY = (e.clientY - rect.top) / rect.height;
      previewer.zoomBy(wheelZoomFactor(e), vp.x + mouseX * vp.w, vp.y + mouseY * vp.h, mouseX, mouseY);
    };

    el.addEventListener('pointerdown', pointerDown, { passive: true });
    el.addEventListener('pointermove', pointerMove, { passive: true });
    el.addEventListener('pointerup', pointerUp, { passive: true });
    el.addEventListener('pointercancel', pointerUp, { passive: true });
    el.addEventListener('wheel', wheel, { passive: false });

    return () => {
      el.removeEventListener('pointerdown', pointerDown);
      el.removeEventListener('pointermove', pointerMove);
      el.removeEventListener('pointerup', pointerUp);
      el.removeEventListener('pointercancel', pointerUp);
      el.removeEventListener('wheel', wheel);
    };
  }

  function panZoomAction(el: HTMLCanvasElement) {
    const teardown = setupPanZoom(el);
    return { destroy: teardown };
  }

  function registerChannelCanvas(el: HTMLCanvasElement, channel: PreviewChannel) {
    separateCanvases.set(channel.idx, el);
    needsRedraw = true;
    const teardown = setupPanZoom(el, cameraKey(channel));
    return {
      destroy() {
        separateCanvases.delete(channel.idx);
        teardown();
      }
    };
  }

  function frameLoop() {
    if (!isRendering) return;
    syncSize();
    // draw() picks the overlay canvas or the side-by-side canvases; canvasEl is null side by side.
    if (needsRedraw && !drawPending) void draw();
    animFrameId = requestAnimationFrame(frameLoop);
  }

  async function draw() {
    drawPending = true;
    try {
      do {
        needsRedraw = false;
        if (showSeparate) {
          for (const channel of visibleChannels) {
            const canvas = separateCanvases.get(channel.idx);
            if (canvas && channel.name) await previewer.renderChannel(canvas, channel.name);
          }
        } else if (canvasEl) {
          await previewer.render(canvasEl);
        }
      } while (needsRedraw && isRendering);
    } catch (error) {
      console.error('[preview] WebGPU render failed:', error);
    } finally {
      drawPending = false;
    }
  }

  onMount(() => {
    syncSize(); // seed size + aspect before the first frame
    isRendering = true;
    frameLoop();

    return () => {
      isRendering = false;
      if (animFrameId !== null) cancelAnimationFrame(animFrameId);
    };
  });

  function sampleAtCrosshair(pos: { normX: number; normY: number }, channelName?: string, cameraId?: string) {
    void previewer.redrawGeneration;
    const aspect = viewW > 0 && viewH > 0 ? viewW / viewH : 1;
    const samples = previewer.samplePixel(pos.normX, pos.normY, aspect, channelName);
    if (!channelName && cameraId && crosshairIndependent.get() && hasMultipleCameras) {
      return samples.filter((s) =>
        visibleChannels.some((ch) => cameraKey(ch) === cameraId && ch.name === s.channelName)
      );
    }
    return samples;
  }

  // ── Scale bar ──
  // Pick a "nice" round bar length that fits ~15-25% of the canvas width.
  const NICE_STEPS = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000];

  const scaleBar = $derived.by(() => {
    const [fovW, fovH] = fov ?? [0, 0];
    if (!Number.isFinite(fovW) || !Number.isFinite(fovH) || fovW <= 0 || fovH <= 0) return null;

    const cw = viewW;
    const ch = viewH;
    if (cw <= 0 || ch <= 0) return null;

    const vp = previewer.viewport;
    // Use the same frame-or-FOV geometry as the navigation controls, including before streaming.
    const vpAspect = (vp.w / vp.h) * previewer.boundingBoxAspect;
    const canvasAspect = cw / ch;
    const drawW = canvasAspect > vpAspect ? ch * vpAspect : cw;

    const umPerPx = (vp.w * fovW) / drawW;
    if (!Number.isFinite(umPerPx) || umPerPx <= 0) return null;

    const targetUm = umPerPx * cw * 0.2;
    const barUm = NICE_STEPS.findLast((s) => s <= targetUm) ?? NICE_STEPS[0];
    const barPx = barUm / umPerPx;

    const label = barUm >= 1000 ? `${barUm / 1000} mm` : `${barUm} µm`;
    return { barPx, label };
  });
</script>

{#snippet scaleBarBadge(bar: { barPx: number; label: string } | null)}
  <span class="font-mono text-fg">{bar?.label ?? '--'}</span>
  <div class="h-2 rounded-full bg-fg" style:width="{bar?.barPx ?? 0}px"></div>
{/snippet}

{#snippet crosshairOverlay(opts?: { channelName?: string; cameraId?: string; cameraLabel?: string })}
  {#if crosshairVisible.get()}
    {@const pos = getCrosshairPos(opts?.cameraId)}
    {@const samples = sampleAtCrosshair(pos, opts?.channelName, opts?.cameraId)}
    <div class="pointer-events-none absolute inset-x-0 z-10 h-px bg-fg/70 halo" style:top="{pos.normY * 100}%"></div>
    <div class="pointer-events-none absolute inset-y-0 z-10 w-px bg-fg/70 halo" style:left="{pos.normX * 100}%"></div>
    <div
      class="pointer-events-none absolute z-10 flex flex-col gap-0.5 frame-overlay px-2 py-1 font-mono text-xs text-fg tabular-nums {pos.normX >
      0.7
        ? '-translate-x-2'
        : 'translate-x-2'} {pos.normY > 0.8 ? '-translate-y-2' : 'translate-y-2'}"
      style:left={pos.normX > 0.7 ? undefined : `${pos.normX * 100}%`}
      style:right={pos.normX > 0.7 ? `${(1 - pos.normX) * 100}%` : undefined}
      style:top={pos.normY > 0.8 ? undefined : `${pos.normY * 100}%`}
      style:bottom={pos.normY > 0.8 ? `${(1 - pos.normY) * 100}%` : undefined}
    >
      {#if opts?.cameraLabel}
        <span class="text-2xs font-sans text-fg-muted">{opts.cameraLabel}</span>
      {/if}
      {#if samples.length > 0}
        <span class="text-fg-muted">({samples[0].sensorX}, {samples[0].sensorY})</span>
        {#if samples.length === 1}
          <span>I = {samples[0].value}</span>
        {:else}
          {#each samples as sample (sample.channelName)}
            <div class="flex items-center gap-1.5">
              {#if sample.color}
                <span class="size-2 rounded-full" style:background-color={sample.color}></span>
              {/if}
              <span class="text-fg-muted">{sample.label}</span>
              <span class="ml-auto">{sample.value}</span>
            </div>
          {/each}
        {/if}
      {:else}
        <span class="text-fg-muted">--</span>
      {/if}
    </div>
  {/if}
{/snippet}

<div class="relative size-full">
  {#if showSeparate}
    <div class="grid size-full grid-cols-2 gap-2 p-2">
      {#each visibleChannels as channel (channel.idx)}
        <div class="relative min-h-0 min-w-0 overflow-hidden rounded-md bg-black">
          <canvas
            use:registerChannelCanvas={channel}
            class="size-full {crosshairVisible.get() ? 'cursor-crosshair' : ''}"
          ></canvas>
          {@render crosshairOverlay({ channelName: channel.name, cameraId: cameraKey(channel) })}
          <div class="pointer-events-none absolute top-2 left-2 rounded bg-canvas/80 px-2 py-1 text-sm text-fg">
            {displayName(channel.config?.detection ?? channel.name ?? '')} · {channel.label}
          </div>
        </div>
      {/each}
    </div>
  {:else}
    <canvas bind:this={canvasEl} use:panZoomAction class="size-full {crosshairVisible.get() ? 'cursor-crosshair' : ''}"
    ></canvas>
    {#if hasMultipleCameras && crosshairIndependent.get()}
      {#each cameraOptions as cam (cam.id)}
        {@render crosshairOverlay({ cameraId: cam.id, cameraLabel: cam.label })}
      {/each}
    {:else}
      {@render crosshairOverlay({ cameraId: cameraOptions[0]?.id })}
    {/if}
  {/if}

  {#if hasMultipleCameras}
    <div class="pointer-events-auto absolute top-3 left-1/2 z-20 -translate-x-1/2 frame-floating">
      <RadioBar
        value={showSeparate ? 'separate' : 'overlay'}
        options={['separate', 'overlay'] as const}
        formatLabel={(layout) => (layout === 'separate' ? 'Side by side' : 'Overlay')}
        onChange={(layout) => cameraLayout.set(layout)}
        size="sm"
        aria-label="Camera layout"
      />
    </div>
  {/if}

  {#if previewer.error}
    <div
      class="absolute inset-x-4 top-pane-header z-10 rounded border border-danger-muted bg-overlay px-3 py-2 text-danger-fg"
    >
      {previewer.error}
    </div>
  {/if}

  <!-- Center the ui-sm toolbar, including its p-1 padding and panel border; the minimap expands below it. -->
  <!-- top-[calc((var(--pane-header-height)-var(--ui-sm))/2-var(--spacing)-1px)] -->
  <div class="pointer-events-none absolute top-2 left-3 z-10 w-58">
    <PreviewNavigationControls
      {previewer}
      cameras={cameraOptions}
      {activeCrosshairCamera}
      onSelectCrosshairCamera={(id) => (activeCrosshairCamera = id)}
      onCenterCrosshair={centerCrosshair}
    />
  </div>

  <div class="pointer-events-none absolute bottom-3 left-3 z-10 w-58">
    <PreviewChannels {previewer} />
  </div>

  <!-- Isolated scale bar. -->
  <div class="pointer-events-none absolute right-4 bottom-4 z-10 flex h-6 flex-col items-end justify-end gap-0.5 halo">
    {#if scaleBar}
      {@render scaleBarBadge(scaleBar)}
    {:else}
      {@render scaleBarBadge(null)}
    {/if}
  </div>
</div>
