import {
  mergeRollingAvatarQueues,
  planRealtimePlayback,
  REALTIME_PLAYBACK_POLICY,
  type AvatarQueue,
  type AvatarQueueStep
} from "@signsaarthi/avatar-engine";
import { runtimeMessageSchema, type Settings } from "@signsaarthi/shared";
import {
  readYouTubeCaptionSnapshot,
  type YouTubeCaptionSnapshot
} from "../adapters/youtubeCaptionAdapter";
import { loadMotionClipMap } from "../lib/motionLibraryLoader";
import type { RuntimeMessage } from "../lib/runtime";
import type { MotionClip } from "./motionAvatarRenderer";
import { createMotionQueuePlayer, type MotionQueuePlayer } from "./motionQueuePlayer";

const ROOT_ID = "signsaarthi-overlay-root";

type CaptionSnapshotRequest = { type: "GET_PAGE_CAPTION_SNAPSHOT" };
type CaptionSnapshotResponse =
  { ok: true; snapshot: YouTubeCaptionSnapshot } | { ok: false; error: string };

let currentQueue: AvatarQueue | undefined;
let activeStepIndex = 0;
let playbackSteps: AvatarQueueStep[] = [];
const settledStepIds = new Set<string>();
let queueFinished = false;
let errorMessage: string | undefined;
let minimized = false;
let overlayWidth = 360;
let overlayDisabled = false;
let activeAvatarName: Settings["avatarName"] = "Ananya";
let activeMotionPlayer: MotionQueuePlayer | undefined;
let activeMotionClipMap: Map<string, MotionClip> | undefined;
let motionPlayerGeneration = 0;
let activeSessionId: string | undefined;
let overlayPaused = false;
let position = clampPosition({ x: window.innerWidth - overlayWidth - 24, y: 96 });
const overlayWidthBySize: Record<Settings["avatarSize"], number> = {
  small: 300,
  medium: 360,
  large: 420
};
window.addEventListener("resize", () => {
  position = clampPosition(position);
  updateOverlayLayout();
});

chrome.runtime.sendMessage({ type: "CONTENT_READY" } satisfies RuntimeMessage, () => {
  void chrome.runtime.lastError;
});

chrome.runtime.onMessage.addListener((message: unknown, _sender, sendResponse) => {
  if (isCaptionSnapshotRequest(message)) {
    respondWithCaptionSnapshot(sendResponse);
    return;
  }

  const parsedMessage = runtimeMessageSchema.safeParse(message);
  if (!parsedMessage.success) {
    sendResponse({ ok: false, error: "Invalid SignSaarthi runtime message." });
    return;
  }
  const runtimeMessage = parsedMessage.data;

  if (
    activeSessionId === undefined &&
    runtimeMessage.type !== "SESSION_STOPPED" &&
    "sessionId" in runtimeMessage &&
    runtimeMessage.sessionId
  ) {
    // A full YouTube navigation creates a new content-script instance while the side panel stays active.
    activeSessionId = runtimeMessage.sessionId;
  }

  if (runtimeMessage.type === "SESSION_STARTED") {
    if (activeSessionId !== runtimeMessage.sessionId) {
      stopPlayback();
      currentQueue = undefined;
      activeStepIndex = 0;
      playbackSteps = [];
      settledStepIds.clear();
      queueFinished = false;
    }
    activeSessionId = runtimeMessage.sessionId;
    overlayPaused = false;
    errorMessage = undefined;
    renderOverlay();
  }
  if (runtimeMessage.type === "SESSION_STOPPED") {
    if (
      runtimeMessage.sessionId &&
      activeSessionId &&
      runtimeMessage.sessionId !== activeSessionId
    ) {
      sendResponse({ ok: true });
      return;
    }
    stopPlayback();
    activeSessionId = undefined;
    overlayPaused = false;
    currentQueue = undefined;
    playbackSteps = [];
    settledStepIds.clear();
    errorMessage = undefined;
    removeOverlay();
  }
  if (runtimeMessage.type === "AVATAR_QUEUE_UPDATED") {
    if (!isMessageForActiveSession(runtimeMessage.sessionId)) {
      return;
    }
    errorMessage = undefined;
    startQueue(runtimeMessage.queue);
  }
  if (runtimeMessage.type === "AVATAR_PLAYBACK_UPDATED") {
    if (!isMessageForActiveSession(runtimeMessage.sessionId)) {
      return;
    }
    updatePlayback(runtimeMessage.action, runtimeMessage.speed);
  }
  if (runtimeMessage.type === "ERROR_STATE") {
    if (!isMessageForActiveSession(runtimeMessage.sessionId)) {
      return;
    }
    stopPlayback();
    currentQueue = undefined;
    playbackSteps = [];
    settledStepIds.clear();
    errorMessage = runtimeMessage.message;
    renderOverlay();
  }
  if (runtimeMessage.type === "OVERLAY_SETTINGS_UPDATED") {
    if (!isMessageForActiveSession(runtimeMessage.sessionId)) {
      return;
    }
    applySettings(runtimeMessage.settings);
  }
  sendResponse({ ok: true });
});

document.documentElement.dataset["signsaarthiContentScript"] = "ready";

const overlayHostGuard = new MutationObserver(() => {
  if ((activeSessionId || currentQueue) && !overlayDisabled && !document.getElementById(ROOT_ID)) {
    renderOverlay();
  }
});
overlayHostGuard.observe(document.documentElement, { childList: true });

function isMessageForActiveSession(sessionId: string | undefined): boolean {
  return sessionId === undefined || sessionId === activeSessionId;
}

function isCaptionSnapshotRequest(message: unknown): message is CaptionSnapshotRequest {
  return Boolean(
    message &&
    typeof message === "object" &&
    "type" in message &&
    (message as { type?: unknown }).type === "GET_PAGE_CAPTION_SNAPSHOT"
  );
}

function respondWithCaptionSnapshot(
  sendResponse: (response: CaptionSnapshotResponse) => void
): void {
  try {
    sendResponse({ ok: true, snapshot: readYouTubeCaptionSnapshot() });
  } catch (caught) {
    sendResponse({
      ok: false,
      error: caught instanceof Error ? caught.message : "Unable to read page captions."
    });
  }
}

function startQueue(queue: AvatarQueue): void {
  addQueueMotionClips(queue);
  const shadow = document.getElementById(ROOT_ID)?.shadowRoot;
  if (currentQueue && queue.speed !== currentQueue.speed) {
    const durationScale = currentQueue.speed / queue.speed;
    const rescale = (step: AvatarQueueStep): AvatarQueueStep => ({
      ...step,
      durationMs: Math.max(1, Math.round(step.durationMs * durationScale))
    });
    currentQueue = {
      ...currentQueue,
      speed: queue.speed,
      steps: currentQueue.steps.map(rescale)
    };
    playbackSteps = playbackSteps.map(rescale);
    activeMotionPlayer?.setSpeed(queue.speed);
  }
  currentQueue = mergeRollingAvatarQueues(currentQueue, queue);

  if (!activeMotionPlayer || !shadow) {
    prepareNextPlaybackWindow();
    renderOverlay();
    return;
  }

  if (queueFinished || playbackSteps.length === 0) {
    loadNextPlaybackWindow();
  } else {
    appendToActivePlaybackWindow();
  }
  if (!overlayPaused && playbackSteps.length) {
    activeMotionPlayer.play();
  }
  updateOverlayStep(shadow);
}

function prepareNextPlaybackWindow(): boolean {
  if (!currentQueue) {
    playbackSteps = [];
    activeStepIndex = 0;
    queueFinished = true;
    return false;
  }

  const candidates = currentQueue.steps.filter(
    (step) => !settledStepIds.has(getStableStepId(step))
  );
  const plan = planRealtimePlayback(candidates, {
    baseSpeed: currentQueue.speed,
    historyStepCount: currentQueue.steps.length
  });
  playbackSteps = plan.playbackSteps;
  activeStepIndex = 0;
  queueFinished = playbackSteps.length === 0;
  refreshBacklogMetadata();
  return playbackSteps.length > 0;
}

function loadNextPlaybackWindow(): void {
  const hasPendingStep = currentQueue?.steps.some(
    (step) => !settledStepIds.has(getStableStepId(step))
  );
  if (!activeMotionPlayer || !hasPendingStep || !prepareNextPlaybackWindow()) {
    return;
  }
  activeMotionPlayer.load(playbackSteps, getEffectivePlaybackSpeed());
  if (!overlayPaused) {
    activeMotionPlayer.play();
  }
}

function appendToActivePlaybackWindow(): void {
  if (!currentQueue || !activeMotionPlayer) {
    return;
  }

  const queuedIds = new Set(playbackSteps.map(getStableStepId));
  const candidates = currentQueue.steps.filter((step) => {
    const stepId = getStableStepId(step);
    return !settledStepIds.has(stepId) && !queuedIds.has(stepId);
  });
  if (!candidates.length) {
    refreshBacklogMetadata();
    activeMotionPlayer.setSpeed(getEffectivePlaybackSpeed());
    return;
  }

  const remainingSteps = playbackSteps.slice(activeStepIndex);
  const remainingDurationMs = remainingSteps.reduce(
    (total, step) => total + Math.max(0, step.durationMs),
    0
  );
  const availableStepCount = Math.max(
    0,
    REALTIME_PLAYBACK_POLICY.maxPendingSteps - remainingSteps.length
  );
  const availableDurationMs = Math.max(
    0,
    REALTIME_PLAYBACK_POLICY.maxPendingMs - remainingDurationMs
  );
  if (availableStepCount > 0 && availableDurationMs > 0) {
    const plan = planRealtimePlayback(candidates, {
      baseSpeed: currentQueue.speed,
      maxPendingSteps: availableStepCount,
      maxPendingMs: availableDurationMs,
      historyStepCount: currentQueue.steps.length
    });
    if (plan.playbackSteps.length) {
      playbackSteps.push(...plan.playbackSteps);
      activeMotionPlayer.enqueue(plan.playbackSteps);
    }
  }
  refreshBacklogMetadata();
  activeMotionPlayer.setSpeed(getEffectivePlaybackSpeed());
}

function refreshBacklogMetadata(): void {
  if (!currentQueue) {
    return;
  }
  const pendingSteps = currentQueue.steps.filter(
    (step) => !settledStepIds.has(getStableStepId(step))
  );
  const pendingPlan = planRealtimePlayback(pendingSteps, {
    baseSpeed: currentQueue.speed,
    historyStepCount: currentQueue.steps.length
  });
  currentQueue = {
    ...currentQueue,
    backlog: {
      ...pendingPlan.backlog,
      historyStepCount: currentQueue.steps.length
    }
  };
}

function getEffectivePlaybackSpeed(): number {
  return currentQueue?.backlog?.catchUpSpeed ?? currentQueue?.speed ?? 1;
}

function getStableStepId(step: AvatarQueueStep): string {
  return step.sourceSegmentId
    ? `${step.sourceSegmentId}:${step.id}`
    : `${step.id}:${step.glossItemId}`;
}

function replayQueue(): void {
  if (!currentQueue) {
    return;
  }
  updatePlayback("replay");
}

function updatePlayback(
  action: "pause" | "resume" | "replay" | "set_speed",
  requestedSpeed?: number
): void {
  if (action === "pause") {
    overlayPaused = true;
    activeMotionPlayer?.pause();
  }
  if (action === "resume") {
    overlayPaused = false;
    if (!queueFinished) {
      activeMotionPlayer?.play();
    }
  }
  if (action === "replay" && currentQueue) {
    const shouldRemainPaused = overlayPaused;
    settledStepIds.clear();
    prepareNextPlaybackWindow();
    activeMotionPlayer?.load(playbackSteps, getEffectivePlaybackSpeed());
    if (!shouldRemainPaused && playbackSteps.length) {
      activeMotionPlayer?.play();
    }
  }
  if (action === "set_speed" && currentQueue && requestedSpeed !== undefined) {
    const speed = Math.min(2, Math.max(0.5, requestedSpeed));
    const durationScale = currentQueue.speed / speed;
    const rescale = (step: AvatarQueueStep): AvatarQueueStep => ({
      ...step,
      durationMs: Math.max(1, Math.round(step.durationMs * durationScale))
    });
    currentQueue = {
      ...currentQueue,
      speed,
      steps: currentQueue.steps.map(rescale)
    };
    playbackSteps = playbackSteps.map(rescale);
    refreshBacklogMetadata();
    activeMotionPlayer?.setSpeed(getEffectivePlaybackSpeed());
  }

  const shadow = document.getElementById(ROOT_ID)?.shadowRoot;
  if (shadow) {
    updateOverlayStep(shadow);
  }
}

function stopPlayback(): void {
  activeMotionPlayer?.pause();
  activeStepIndex = 0;
  queueFinished = false;
}

function renderOverlay(): void {
  if (overlayDisabled) {
    removeOverlay();
    return;
  }
  position = clampPosition(position);
  let host = document.getElementById(ROOT_ID);
  if (!host) {
    host = document.createElement("div");
    host.id = ROOT_ID;
    document.documentElement.appendChild(host);
  }

  const shadow = host.shadowRoot ?? host.attachShadow({ mode: "open" });
  const focusedAction =
    shadow.activeElement instanceof HTMLElement
      ? shadow.activeElement.dataset["action"]
      : undefined;
  destroyMotionRenderer();
  shadow.innerHTML = overlayMarkup();
  bindOverlay(shadow);
  fitOverlayIntoViewport(shadow);
  if (focusedAction) {
    shadow.querySelector<HTMLButtonElement>(`[data-action="${focusedAction}"]`)?.focus();
  }
}

function removeOverlay(): void {
  destroyMotionRenderer();
  document.getElementById(ROOT_ID)?.remove();
}

function destroyMotionRenderer(): void {
  motionPlayerGeneration += 1;
  activeMotionPlayer?.destroy();
  activeMotionPlayer = undefined;
  activeMotionClipMap = undefined;
}

function overlayMarkup(): string {
  const activeStep = getActiveStep();
  const stepDurationMs = activeStep ? getStepDurationMs(activeStep) : 1200;
  const fallbackText = errorMessage ?? getQueueNotice();
  const progressText = getProgressText();
  const statusText = getPlaybackStatusText();
  const body = `<div class="avatar-stage" style="--step-duration: ${stepDurationMs}ms">
        <div class="avatar-figure" data-step-kind="${activeStep?.kind ?? "idle"}">
          <div class="avatar-backdrop"></div>
          <div class="motion-avatar-root" data-avatar-name="${escapeHtml(activeAvatarName)}"></div>
        </div>
        <div class="step-row">
          <span class="step-badge">${escapeHtml(getStepBadgeText(activeStep))}</span>
          <span class="step-progress">${escapeHtml(progressText)}</span>
        </div>
        <div class="gloss-strip" aria-live="polite">${escapeHtml(errorMessage ? "ERROR" : (activeStep?.label ?? "READY"))}</div>
        <div class="fallback">${escapeHtml(fallbackText)}</div>
      </div>`;

  return `
    <style>
      :host { all: initial; }
      .overlay {
        position: fixed;
        left: ${position.x}px;
        top: ${position.y}px;
        width: ${minimized ? 220 : overlayWidth}px;
        max-width: calc(100vw - 32px);
        max-height: calc(100vh - 32px);
        display: flex;
        flex-direction: column;
        z-index: 2147483647;
        font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        color: #f5f7ff;
        background: #0a101b;
        border: 1px solid rgba(139, 124, 255, 0.72);
        border-radius: 8px;
        box-shadow: 0 24px 64px rgba(0, 0, 0, 0.5);
        overflow: hidden;
      }
      .overlay.is-minimized { width: 220px; }
      .overlay.is-minimized .avatar-stage { display: none; }
      .toolbar {
        min-height: 46px;
        flex: 0 0 46px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 0 12px;
        border-bottom: 1px solid rgba(145, 163, 201, 0.18);
        background: rgba(16, 23, 37, 0.98);
        cursor: move;
        user-select: none;
        font-size: 14px;
        font-weight: 700;
      }
      .toolbar-copy {
        min-width: 0;
        display: grid;
        gap: 2px;
      }
      .toolbar-title {
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
      }
      .toolbar-platform {
        display: inline-flex;
        align-items: center;
        gap: 5px;
        color: #9da9c1;
        font-size: 10px;
        font-weight: 600;
      }
      .toolbar-platform::before {
        content: "";
        width: 6px;
        height: 6px;
        border-radius: 50%;
        background: #43d69d;
        box-shadow: 0 0 8px rgba(67, 214, 157, 0.72);
      }
      .actions { display: flex; gap: 4px; }
      button {
        width: 30px;
        height: 30px;
        border: 0;
        border-radius: 6px;
        background: transparent;
        color: #c8d0e5;
        cursor: pointer;
        font-size: 15px;
      }
      button:hover,
      button:focus-visible {
        background: rgba(139, 124, 255, 0.16);
        color: #ffffff;
        outline: 2px solid #8b7cff;
        outline-offset: 2px;
      }
      .avatar-stage {
        --step-duration: 1200ms;
        min-height: min(318px, calc(100vh - 74px));
        max-height: calc(100vh - 74px);
        overflow-y: auto;
        scrollbar-width: thin;
        background: linear-gradient(145deg, #111b36 0%, #0b1121 58%, #111829 100%);
        display: grid;
        grid-template-rows: 1fr auto auto auto;
        place-items: center;
      }
      .avatar-figure {
        position: relative;
        width: 100%;
        height: 218px;
        display: grid;
        place-items: end center;
        overflow: hidden;
        isolation: isolate;
      }
      .avatar-backdrop {
        position: absolute;
        inset: 18px 32px 0;
        z-index: -1;
        border: 1px solid rgba(139, 124, 255, 0.16);
        border-bottom: 0;
        border-radius: 110px 110px 0 0;
        background: linear-gradient(180deg, rgba(91, 88, 232, 0.18), rgba(91, 88, 232, 0.02));
      }
      .motion-avatar-root {
        width: min(90%, 276px);
        height: 218px;
        filter: drop-shadow(0 18px 24px rgba(0, 0, 0, 0.48));
      }
      .step-row {
        width: calc(100% - 24px);
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 10px;
        margin: 0 12px 8px;
        font-size: 11px;
      }
      .step-badge {
        min-width: 0;
        padding: 5px 8px;
        border-radius: 999px;
        border: 1px solid rgba(139, 124, 255, 0.22);
        background: rgba(139, 124, 255, 0.12);
        color: #d9d5ff;
        font-weight: 700;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
      }
      .step-progress {
        color: #9da9c1;
        flex: 0 0 auto;
      }
      .gloss-strip {
        width: calc(100% - 24px);
        margin: 0 12px 8px;
        padding: 9px 12px;
        border: 1px solid rgba(145, 163, 201, 0.18);
        background: rgba(5, 9, 17, 0.82);
        color: #ffffff;
        border-radius: 4px;
        font-size: 13px;
        letter-spacing: 0;
        text-align: center;
      }
      .fallback {
        width: calc(100% - 24px);
        padding: 0 12px 12px;
        color: #9da9c1;
        font-size: 12px;
        line-height: 1.45;
        overflow-wrap: anywhere;
      }
    </style>
    <section class="overlay${minimized ? " is-minimized" : ""}" role="region" aria-label="ISL Avatar">
      <div class="toolbar">
        <div class="toolbar-copy">
          <span class="toolbar-title">${escapeHtml(activeAvatarName)} · ${statusText}</span>
          <span class="toolbar-platform">YouTube</span>
        </div>
        <div class="actions">
          <button type="button" data-action="replay" title="Replay" aria-label="Replay avatar queue"${currentQueue ? "" : " disabled"}>↻</button>
          <button type="button" data-action="minimize" title="Minimize" aria-label="${minimized ? "Expand avatar overlay" : "Minimize avatar overlay"}" aria-pressed="${minimized}">${minimized ? "+" : "-"}</button>
          <button type="button" data-action="resize" title="Resize" aria-label="Resize avatar overlay">□</button>
        </div>
      </div>
      ${body}
    </section>
  `;
}

function bindOverlay(shadow: ShadowRoot): void {
  const overlay = shadow.querySelector<HTMLElement>(".overlay");
  const toolbar = shadow.querySelector<HTMLElement>(".toolbar");
  const minimize = shadow.querySelector<HTMLButtonElement>('[data-action="minimize"]');
  const resize = shadow.querySelector<HTMLButtonElement>('[data-action="resize"]');
  const replay = shadow.querySelector<HTMLButtonElement>('[data-action="replay"]');
  const motionRoot = shadow.querySelector<HTMLElement>(".motion-avatar-root");
  if (!overlay || !toolbar) {
    return;
  }
  const overlayElement = overlay;
  const toolbarElement = toolbar;

  if (motionRoot) {
    void attachMotionPlayer(motionRoot, shadow);
  }

  minimize?.addEventListener("click", (event) => {
    event.stopPropagation();
    minimized = !minimized;
    position = clampPosition(position);
    updateOverlayLayout();
  });
  resize?.addEventListener("click", (event) => {
    event.stopPropagation();
    overlayWidth = overlayWidth > 300 ? 300 : 360;
    position = clampPosition(position);
    updateOverlayLayout();
  });
  replay?.addEventListener("click", (event) => {
    event.stopPropagation();
    replayQueue();
  });

  toolbar.addEventListener("pointerdown", (event) => {
    if ((event.target as Element | null)?.closest("button")) {
      return;
    }
    const startX = event.clientX;
    const startY = event.clientY;
    const startPosition = { ...position };
    toolbarElement.setPointerCapture(event.pointerId);

    function move(moveEvent: PointerEvent): void {
      position = clampPosition({
        x: startPosition.x + moveEvent.clientX - startX,
        y: startPosition.y + moveEvent.clientY - startY
      });
      overlayElement.style.left = `${position.x}px`;
      overlayElement.style.top = `${position.y}px`;
    }

    function up(upEvent: PointerEvent): void {
      if (toolbarElement.hasPointerCapture(upEvent.pointerId)) {
        toolbarElement.releasePointerCapture(upEvent.pointerId);
      }
      toolbarElement.removeEventListener("pointermove", move);
      toolbarElement.removeEventListener("pointerup", up);
      toolbarElement.removeEventListener("pointercancel", up);
    }

    toolbarElement.addEventListener("pointermove", move);
    toolbarElement.addEventListener("pointerup", up);
    toolbarElement.addEventListener("pointercancel", up);
  });
}

async function attachMotionPlayer(motionRoot: HTMLElement, shadow: ShadowRoot): Promise<void> {
  const generation = motionPlayerGeneration;
  try {
    const motionClips = new Map(await loadMotionClipMap());
    for (const clip of currentQueue?.motionClips ?? []) {
      motionClips.set(clip.id, clip as MotionClip);
    }
    if (
      generation !== motionPlayerGeneration ||
      !motionRoot.isConnected ||
      shadow.host !== document.getElementById(ROOT_ID) ||
      shadow.querySelector(".motion-avatar-root") !== motionRoot
    ) {
      return;
    }
    if (errorMessage?.startsWith("Avatar motion library could not be loaded")) {
      errorMessage = undefined;
    }
    activeMotionClipMap = motionClips;
    activeMotionPlayer = createMotionQueuePlayer(motionRoot, motionClips, {
      avatarName: activeAvatarName,
      onStepChange(index) {
        for (let settledIndex = activeStepIndex; settledIndex < index; settledIndex += 1) {
          const settledStep = playbackSteps[settledIndex];
          if (settledStep) {
            settledStepIds.add(getStableStepId(settledStep));
          }
        }
        activeStepIndex = index;
        queueFinished = false;
        refreshBacklogMetadata();
        updateOverlayStep(shadow);
      },
      onStateChange(state) {
        queueFinished = state === "complete";
        if (queueFinished) {
          for (
            let settledIndex = activeStepIndex;
            settledIndex < playbackSteps.length;
            settledIndex += 1
          ) {
            settledStepIds.add(getStableStepId(playbackSteps[settledIndex]!));
          }
          refreshBacklogMetadata();
          loadNextPlaybackWindow();
        }
        updateOverlayStep(shadow);
      }
    });
    if (currentQueue) {
      if (!playbackSteps.length) {
        prepareNextPlaybackWindow();
      }
      activeMotionPlayer.load(playbackSteps, getEffectivePlaybackSpeed(), activeStepIndex);
      if (!queueFinished && !overlayPaused) {
        activeMotionPlayer.play();
      }
    }
  } catch {
    if (generation !== motionPlayerGeneration || !motionRoot.isConnected) {
      return;
    }
    errorMessage =
      "Avatar motion library could not be loaded. Word actions remain available in the side panel.";
    updateOverlayStep(shadow);
  }
}

function updateOverlayLayout(): void {
  const shadow = document.getElementById(ROOT_ID)?.shadowRoot;
  const overlay = shadow?.querySelector<HTMLElement>(".overlay");
  if (!shadow || !overlay) {
    return;
  }
  overlay.classList.toggle("is-minimized", minimized);
  overlay.style.left = `${position.x}px`;
  overlay.style.top = `${position.y}px`;
  overlay.style.width = `${minimized ? 220 : overlayWidth}px`;
  const minimizeButton = shadow.querySelector<HTMLButtonElement>('[data-action="minimize"]');
  if (minimizeButton) {
    minimizeButton.textContent = minimized ? "+" : "-";
    minimizeButton.title = minimized ? "Expand" : "Minimize";
    minimizeButton.setAttribute(
      "aria-label",
      minimized ? "Expand avatar overlay" : "Minimize avatar overlay"
    );
    minimizeButton.setAttribute("aria-pressed", String(minimized));
  }
  fitOverlayIntoViewport(shadow);
}

function updateOverlayStep(shadow: ShadowRoot): void {
  const activeStep = getActiveStep();
  const stage = shadow.querySelector<HTMLElement>(".avatar-stage");
  const figure = shadow.querySelector<HTMLElement>(".avatar-figure");
  const badge = shadow.querySelector<HTMLElement>(".step-badge");
  const progress = shadow.querySelector<HTMLElement>(".step-progress");
  const gloss = shadow.querySelector<HTMLElement>(".gloss-strip");
  const fallback = shadow.querySelector<HTMLElement>(".fallback");
  const toolbarStatus = shadow.querySelector<HTMLElement>(".toolbar-title");
  const replayButton = shadow.querySelector<HTMLButtonElement>('[data-action="replay"]');

  stage?.style.setProperty(
    "--step-duration",
    `${activeStep ? getStepDurationMs(activeStep) : 1200}ms`
  );
  if (figure) {
    figure.dataset.stepKind = activeStep?.kind ?? "idle";
  }
  if (badge) {
    badge.textContent = getStepBadgeText(activeStep);
  }
  if (progress) {
    progress.textContent = getProgressText();
  }
  if (gloss) {
    gloss.textContent = errorMessage ? "ERROR" : (activeStep?.label ?? "READY");
  }
  if (fallback) {
    fallback.textContent = errorMessage ?? getQueueNotice();
  }
  if (toolbarStatus) {
    toolbarStatus.textContent = `${activeAvatarName} · ${getPlaybackStatusText()}`;
  }
  if (replayButton) {
    replayButton.disabled = !currentQueue;
  }
  fitOverlayIntoViewport(shadow);
}

function applySettings(settings: Settings): void {
  const avatarChanged = activeAvatarName !== settings.avatarName;
  overlayDisabled =
    settings.avatarPosition === "side_panel" || settings.outputMode === "captions_only";
  overlayWidth = overlayWidthBySize[settings.avatarSize];
  activeAvatarName = settings.avatarName;
  if (!activeSessionId && !currentQueue) {
    removeOverlay();
    return;
  }
  if (overlayDisabled) {
    stopPlayback();
    removeOverlay();
    return;
  }
  if (settings.avatarPosition === "bottom_left") {
    position = clampPosition({ x: 24, y: window.innerHeight - 360 });
  }
  if (settings.avatarPosition === "bottom_right") {
    position = clampPosition({
      x: window.innerWidth - overlayWidth - getSidePanelReservePx(),
      y: window.innerHeight - 392
    });
  }
  if (settings.avatarPosition === "top_right") {
    position = clampPosition({
      x: window.innerWidth - overlayWidth - getSidePanelReservePx(),
      y: 96
    });
  }
  if (avatarChanged || !document.getElementById(ROOT_ID)) {
    renderOverlay();
    return;
  }
  updateOverlayLayout();
  const shadow = document.getElementById(ROOT_ID)?.shadowRoot;
  if (shadow) {
    updateOverlayStep(shadow);
  }
}

function getSidePanelReservePx(): number {
  return Math.min(520, Math.max(360, Math.round(window.innerWidth * 0.29)));
}

function getActiveStep(): AvatarQueueStep | undefined {
  return playbackSteps[activeStepIndex];
}

function getStepDurationMs(step: AvatarQueueStep): number {
  if (Number.isFinite(step.durationMs) && step.durationMs >= 0) {
    return step.durationMs;
  }
  const speed = currentQueue?.speed && currentQueue.speed > 0 ? currentQueue.speed : 1;
  return Math.round(1100 / speed);
}

function getStepBadgeText(step: AvatarQueueStep | undefined): string {
  if (!step) {
    return "No step loaded";
  }
  if (step.kind === "sign") {
    if (step.motionSource === "expert_reviewed_library") {
      return "Expert-reviewed ISL motion";
    }
    if (step.motionSource === "include_dataset") {
      return "INCLUDE dataset motion";
    }
    if (step.motionSource === "isign_research") {
      return "Local iSign research motion";
    }
    return "Motion unavailable";
  }
  if (step.kind === "fingerspell") {
    return step.fingerspellingSource === "islrtc_official"
      ? "Official ISLRTC fingerspelling"
      : "Fingerspelling fallback";
  }
  if (step.kind === "caption") {
    return "Caption fallback";
  }
  return "Text fallback";
}

function addQueueMotionClips(queue: AvatarQueue): void {
  if (!activeMotionClipMap) {
    return;
  }
  for (const clip of queue.motionClips ?? []) {
    activeMotionClipMap.set(clip.id, clip as MotionClip);
  }
}

function getQueueNotice(): string {
  if (!currentQueue) {
    return "Waiting for an avatar queue from the side panel.";
  }
  if (currentQueue.steps.length === 0) {
    return "No avatar steps available for this segment.";
  }
  const backlogNotice = currentQueue.backlog?.deferredStepCount
    ? `${currentQueue.backlog.deferredStepCount} actions pending for playback.`
    : currentQueue.backlog?.status === "catching_up"
      ? `Catching up at ${formatSpeed(currentQueue.backlog.catchUpSpeed)}.`
      : undefined;
  const outputNotice = currentQueue.motionReviewNotice
    ? currentQueue.honestFallbackText
      ? `${currentQueue.motionReviewNotice} ${currentQueue.honestFallbackText}`
      : currentQueue.motionReviewNotice
    : (currentQueue.honestFallbackText ??
      (currentQueue.steps.some((step) => step.kind !== "sign")
        ? "Some words are shown as text or fingerspelling."
        : "Real keypoint motion is queued."));
  return backlogNotice ? `${backlogNotice} ${outputNotice}` : outputNotice;
}

function getProgressText(): string {
  if (!currentQueue || !playbackSteps.length) {
    return "0/0";
  }
  const playbackStepCount = playbackSteps.length;
  const totalStepCount = currentQueue.steps.length;
  const activeStep = playbackSteps[Math.min(activeStepIndex, playbackStepCount - 1)];
  if (!activeStep) {
    return "0/0";
  }
  const completedStepCount = Math.min(totalStepCount, settledStepIds.size);
  const globalStepIndex = queueFinished
    ? totalStepCount
    : Math.min(totalStepCount, completedStepCount + 1);
  const durationMs = getStepDurationMs(activeStep);
  const backlog = currentQueue.backlog;
  const backlogText = backlog?.deferredStepCount
    ? ` · ${backlog.deferredStepCount} pending`
    : backlog?.status === "catching_up"
      ? " · catch-up"
      : "";
  return `${globalStepIndex}/${totalStepCount} · ${formatDuration(durationMs)} · ${formatSpeed(
    getEffectivePlaybackSpeed()
  )}${backlogText}`;
}

function getPlaybackStatusText(): string {
  if (!currentQueue) {
    if (errorMessage) {
      return "Error";
    }
    return "Ready";
  }
  if (queueFinished) {
    return "Complete";
  }
  if (overlayPaused) {
    return "Paused";
  }
  if (currentQueue.backlog?.status === "backlog") {
    return "Backlog";
  }
  if (currentQueue.backlog?.status === "catching_up") {
    return "Catching up";
  }
  return currentQueue.replayOfQueueId ? "Replay" : "Playing";
}

function formatDuration(durationMs: number): string {
  if (durationMs < 1000) {
    return `${durationMs}ms`;
  }
  return `${(durationMs / 1000).toFixed(1)}s`;
}

function formatSpeed(speed: number): string {
  return `${speed.toFixed(2).replace(/\.?0+$/, "")}x`;
}

function fitOverlayIntoViewport(shadow: ShadowRoot): void {
  const overlay = shadow.querySelector<HTMLElement>(".overlay");
  if (!overlay) {
    return;
  }
  position = clampPosition(position, overlay.getBoundingClientRect().height);
  overlay.style.left = `${position.x}px`;
  overlay.style.top = `${position.y}px`;
}

function clampPosition(
  nextPosition: { x: number; y: number },
  measuredHeight?: number
): { x: number; y: number } {
  const width = minimized ? 220 : overlayWidth;
  const fallbackHeight = minimized ? 44 : 480;
  const requestedHeight =
    measuredHeight !== undefined && Number.isFinite(measuredHeight) && measuredHeight > 0
      ? measuredHeight
      : fallbackHeight;
  const height = Math.min(requestedHeight, Math.max(44, window.innerHeight - 32));
  const maxX = Math.max(16, window.innerWidth - width - 16);
  const maxY = Math.max(16, window.innerHeight - height - 16);
  return {
    x: Math.min(Math.max(16, nextPosition.x), maxX),
    y: Math.min(Math.max(16, nextPosition.y), maxY)
  };
}

function escapeHtml(value: string): string {
  return value
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}
