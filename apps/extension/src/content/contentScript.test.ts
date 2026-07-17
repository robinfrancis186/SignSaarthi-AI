import type { AvatarQueue, AvatarQueueStep } from "@signsaarthi/shared";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { MotionQueuePlayerOptions } from "./motionQueuePlayer";

const motionPlayer = vi.hoisted(() => ({
  load: vi.fn(),
  enqueue: vi.fn(),
  play: vi.fn(),
  pause: vi.fn(),
  replay: vi.fn(),
  setSpeed: vi.fn(),
  destroy: vi.fn(),
  getStepIndex: vi.fn(() => 0),
  getState: vi.fn(() => "playing" as const)
}));

const createMotionQueuePlayer = vi.hoisted(() => vi.fn(() => motionPlayer));
const loadMotionClipMap = vi.hoisted(() => vi.fn(async () => new Map()));

vi.mock("./motionQueuePlayer", () => ({ createMotionQueuePlayer }));
vi.mock("../lib/motionLibraryLoader", () => ({ loadMotionClipMap }));

type RuntimeListener = (
  message: unknown,
  sender: chrome.runtime.MessageSender,
  sendResponse: (response?: unknown) => void
) => void;

describe("content-script queue identity and realtime policy", () => {
  let runtimeListener: RuntimeListener;

  beforeEach(async () => {
    vi.resetModules();
    vi.clearAllMocks();
    document.documentElement.innerHTML = "<head></head><body></body>";
    const listeners: RuntimeListener[] = [];
    vi.stubGlobal("chrome", {
      runtime: {
        lastError: undefined,
        getURL: vi.fn((path: string) => `chrome-extension://test/${path}`),
        sendMessage: vi.fn((_message: unknown, callback?: () => void) => callback?.()),
        onMessage: {
          addListener: vi.fn((listener: RuntimeListener) => listeners.push(listener))
        }
      }
    });

    await import("./contentScript");
    runtimeListener = listeners[0]!;
    send({ type: "SESSION_STARTED", sessionId: "session_1" });
    await flushMotionLoad();
  });

  afterEach(() => {
    document.documentElement.innerHTML = "<head></head><body></body>";
    vi.unstubAllGlobals();
  });

  it("enqueues only true stable-ID additions while preserving repeated labels", async () => {
    const first = queue("interp_first", [
      step("segment_roll", "step_go_1", "gloss_go_1"),
      step("segment_roll", "step_go_2", "gloss_go_2")
    ]);
    const rolling = queue("interp_extended", [
      step("segment_roll", "step_go_2", "gloss_go_2"),
      step("segment_roll", "step_go_3", "gloss_go_3")
    ]);
    const nextSegment = queue("interp_next", [
      step("segment_next", "step_go_1", "gloss_next_go_1")
    ]);

    send({ type: "AVATAR_QUEUE_UPDATED", queue: first, sessionId: "session_1" });
    expect(motionPlayer.load).toHaveBeenCalledWith(first.steps, 1);

    send({ type: "AVATAR_QUEUE_UPDATED", queue: rolling, sessionId: "session_1" });
    send({ type: "AVATAR_QUEUE_UPDATED", queue: nextSegment, sessionId: "session_1" });

    expect(
      motionPlayer.enqueue.mock.calls.map(([steps]) =>
        steps.map((item: AvatarQueueStep) => item.id)
      )
    ).toEqual([["step_go_3"], ["step_go_1"]]);
  });

  it("loads a bounded earliest playback window while retaining backlog status", async () => {
    const steps = Array.from({ length: 20 }, (_, index) => ({
      ...step("segment_backlog", `step_${index}`, `gloss_${index}`),
      durationMs: 1_500
    }));

    send({
      type: "AVATAR_QUEUE_UPDATED",
      queue: queue("interp_backlog", steps),
      sessionId: "session_1"
    });

    const loadedSteps = motionPlayer.load.mock.calls.at(-1)?.[0] as AvatarQueueStep[];
    const playbackSpeed = motionPlayer.load.mock.calls.at(-1)?.[1];
    expect(loadedSteps).toHaveLength(8);
    expect(playbackSpeed).toBe(1);
    expect(loadedSteps.map((item) => item.id)).toEqual(steps.slice(0, 8).map((item) => item.id));
    const shadow = document.getElementById("signsaarthi-overlay-root")?.shadowRoot;
    expect(shadow?.querySelector(".toolbar-platform")?.textContent).toBe("YouTube");
    expect(shadow?.textContent).toContain("12 actions pending for playback.");
    expect(shadow?.querySelector(".step-progress")?.textContent).toMatch(/^1\/20/);
  });

  it("plays every backlogged action once in order and includes it in replay", async () => {
    const steps = [
      {
        ...step("segment_backlog", "step_today", "gloss_today"),
        kind: "sign" as const,
        label: "TODAY",
        motionClipId: "include-today",
        motionSource: "include_dataset" as const,
        expertReviewed: false,
        durationMs: 1_500
      },
      ...Array.from({ length: 19 }, (_, index) => ({
        ...step("segment_backlog", `step_later_${index}`, `gloss_later_${index}`),
        label: `LATER ${index + 1}`,
        durationMs: 1_500
      }))
    ];
    const backlogQueue = queue("interp_backlog_order", steps);
    const expectedActionIds = steps.map((item) => item.planningActionId);
    const firstLoadCallIndex = motionPlayer.load.mock.calls.length;

    send({
      type: "AVATAR_QUEUE_UPDATED",
      queue: backlogQueue,
      sessionId: "session_1"
    });

    const firstPass = completeLoadedWindows(firstLoadCallIndex);
    const firstPassActionIds = firstPass.steps.map((item) => item.planningActionId);
    expect(firstPass.steps[0]).toMatchObject({ label: "TODAY", motionClipId: "include-today" });
    expect(firstPassActionIds).toEqual(expectedActionIds);
    expect(new Set(firstPassActionIds).size).toBe(expectedActionIds.length);
    expect(
      document
        .getElementById("signsaarthi-overlay-root")
        ?.shadowRoot?.querySelector(".step-progress")?.textContent
    ).toMatch(/^20\/20/);

    const settledLoadCount = motionPlayer.load.mock.calls.length;
    send({
      type: "AVATAR_QUEUE_UPDATED",
      queue: backlogQueue,
      sessionId: "session_1"
    });
    expect(motionPlayer.load).toHaveBeenCalledTimes(settledLoadCount);

    send({
      type: "AVATAR_PLAYBACK_UPDATED",
      action: "replay",
      sessionId: "session_1"
    });
    const replayPass = completeLoadedWindows(firstPass.nextLoadCallIndex);
    expect(replayPass.steps.map((item) => item.planningActionId)).toEqual(expectedActionIds);
    expect(replayPass.steps[0]).toMatchObject({ label: "TODAY", motionClipId: "include-today" });
  });

  function send(message: unknown): void {
    runtimeListener(message, {}, vi.fn());
  }

  function completeLoadedWindows(startLoadCallIndex = 0): {
    steps: AvatarQueueStep[];
    nextLoadCallIndex: number;
  } {
    const options = getMotionPlayerOptions();
    const completedSteps: AvatarQueueStep[] = [];
    let loadCallIndex = startLoadCallIndex;

    while (loadCallIndex < motionPlayer.load.mock.calls.length) {
      const loadedSteps = motionPlayer.load.mock.calls[loadCallIndex]?.[0] as AvatarQueueStep[];
      for (const [index, loadedStep] of loadedSteps.entries()) {
        options.onStepChange?.(index, loadedStep);
        completedSteps.push(loadedStep);
      }
      loadCallIndex += 1;
      options.onStateChange?.("complete");
      if (loadCallIndex - startLoadCallIndex > 10) {
        throw new Error("Avatar playback did not settle within ten bounded windows.");
      }
    }

    return { steps: completedSteps, nextLoadCallIndex: loadCallIndex };
  }
});

function getMotionPlayerOptions(): MotionQueuePlayerOptions {
  const calls = createMotionQueuePlayer.mock.calls as unknown as Array<
    [HTMLElement, ReadonlyMap<string, unknown>, MotionQueuePlayerOptions]
  >;
  const options = calls.at(-1)?.[2];
  if (!options) {
    throw new Error("Motion player was not created.");
  }
  return options;
}

async function flushMotionLoad(): Promise<void> {
  for (let index = 0; index < 4; index += 1) {
    await Promise.resolve();
  }
}

function step(sourceSegmentId: string, id: string, glossItemId: string): AvatarQueueStep {
  return {
    id,
    glossItemId,
    kind: "fingerspell",
    label: "GO",
    fingerspellingSource: "islrtc_official",
    durationMs: 1_000,
    confidence: "low",
    sourceSegmentId,
    planningActionId: id.replace("step", "action")
  };
}

function queue(interpretationId: string, steps: AvatarQueueStep[]): AvatarQueue {
  return {
    id: `queue_${interpretationId}`,
    interpretationId,
    speed: 1,
    steps,
    createdAt: "2026-07-15T00:00:00.000Z"
  };
}
