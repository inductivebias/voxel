<script lang="ts">
  import { watch } from 'runed';
  import { onMount } from 'svelte';
  import { SvelteMap } from 'svelte/reactivity';

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
  const visibleChannels = $derived(previewer.channels.filter((channel) => channel.name && channel.visible));
  // Channels sharing one sensor remain composited; independent cameras get individual canvases.
  const hasMultipleCameras = $derived(
    new Set(visibleChannels.map((channel) => channel.config?.detection).filter(Boolean)).size > 1
  );
  const showSeparate = $derived(hasMultipleCameras && cameraLayout.get() === 'separate');

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
  function setupPanZoom(el: HTMLCanvasElement): () => void {
    let isPanning = false;
    let panStartX = 0;
    let panStartY = 0;
    let startViewport = { ...previewer.viewport };

    const pointerDown = (e: PointerEvent) => {
      if (e.button !== 0) return;
      el.setPointerCapture(e.pointerId);
      isPanning = true;
      panStartX = e.clientX;
      panStartY = e.clientY;
      startViewport = { ...previewer.viewport };
    };

    const pointerMove = (e: PointerEvent) => {
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
      el.releasePointerCapture(e.pointerId);
      isPanning = false;
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
    el.addEventListener('wheel', wheel, { passive: false });

    return () => {
      el.removeEventListener('pointerdown', pointerDown);
      el.removeEventListener('pointermove', pointerMove);
      el.removeEventListener('pointerup', pointerUp);
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
    const teardown = setupPanZoom(el);
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
    if (needsRedraw && !drawPending && canvasEl) void draw();
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

<div class="relative size-full">
  {#if showSeparate}
    <div class="grid size-full grid-cols-2 gap-2 p-2">
      {#each visibleChannels as channel (channel.idx)}
        <div class="relative min-h-0 min-w-0 overflow-hidden rounded-md bg-black">
          <canvas use:registerChannelCanvas={channel} class="size-full"></canvas>
          <div class="pointer-events-none absolute top-2 left-2 rounded bg-canvas/80 px-2 py-1 text-sm text-fg">
            {displayName(channel.config?.detection ?? channel.name ?? '')} · {channel.label}
          </div>
        </div>
      {/each}
    </div>
  {:else}
    <canvas bind:this={canvasEl} use:panZoomAction class="size-full"></canvas>
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
    <PreviewNavigationControls {previewer} />
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
