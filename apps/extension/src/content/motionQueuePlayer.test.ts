import type { AvatarQueueStep } from "@signsaarthi/shared";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { MotionClip, MotionFrame } from "./motionAvatarRenderer";
import { createMotionQueuePlayer } from "./motionQueuePlayer";

const steps: AvatarQueueStep[] = [
  {
    id: "step_1",
    glossItemId: "gloss_1",
    kind: "sign",
    label: "TODAY",
    motionClipId: "include-today",
    motionSource: "include_dataset",
    expertReviewed: false,
    durationMs: 100,
    confidence: "high"
  },
  {
    id: "step_2",
    glossItemId: "gloss_2",
    kind: "sign",
    label: "COMPUTER",
    motionClipId: "include-computer",
    motionSource: "include_dataset",
    expertReviewed: false,
    durationMs: 120,
    confidence: "high"
  }
];

const clips = new Map<string, MotionClip>([
  ["include-today", createClip("include-today", 0)],
  ["include-computer", createClip("include-computer", 0.18)]
]);

describe("motion queue player", () => {
  beforeEach(() => {
    vi.useFakeTimers();
    document.body.innerHTML = "";
  });

  afterEach(() => {
    vi.clearAllTimers();
    vi.useRealTimers();
    document.body.innerHTML = "";
  });

  it("advances a multi-sign queue on one stable SVG and blends between clips", () => {
    const host = document.createElement("div");
    document.body.append(host);
    const onStepChange = vi.fn();
    const player = createMotionQueuePlayer(host, clips, { onStepChange });

    player.load(steps, 1);
    const svg = getSvg(host);
    player.play();
    vi.advanceTimersByTime(100);

    expect(player.getStepIndex()).toBe(1);
    expect(host.querySelector("svg[data-motion-avatar]")).toBe(svg);
    expect(svg.dataset.clipId).toBe("include-computer");
    expect(svg.dataset.transitionFrames).toBe("5");
    expect(onStepChange.mock.calls.map(([index]) => index)).toEqual([0, 1]);

    vi.advanceTimersByTime(320);
    expect(player.getState()).toBe("complete");
    player.destroy();
  });

  it("enqueues steps without replacing or restarting the active step", () => {
    const host = document.createElement("div");
    document.body.append(host);
    const onStepChange = vi.fn();
    const player = createMotionQueuePlayer(host, clips, { onStepChange });

    player.load(steps.slice(0, 1), 1);
    player.play();
    vi.advanceTimersByTime(40);
    const svg = getSvg(host);
    const frameIndex = svg.dataset.frameIndex;

    player.enqueue(steps.slice(1));

    expect(player.getStepIndex()).toBe(0);
    expect(player.getState()).toBe("playing");
    expect(host.querySelector("svg[data-motion-avatar]")).toBe(svg);
    expect(svg.dataset.clipId).toBe("include-today");
    expect(svg.dataset.frameIndex).toBe(frameIndex);
    expect(onStepChange.mock.calls.map(([index]) => index)).toEqual([0]);

    vi.advanceTimersByTime(60);
    expect(player.getStepIndex()).toBe(1);
    expect(onStepChange.mock.calls.map(([index]) => index)).toEqual([0, 1]);
    player.destroy();
  });

  it("retimes the remaining active step when speed changes without resetting playback", () => {
    const host = document.createElement("div");
    document.body.append(host);
    const player = createMotionQueuePlayer(host, clips);

    player.load(steps, 1);
    player.play();
    vi.advanceTimersByTime(40);
    const svg = getSvg(host);
    const frameIndex = svg.dataset.frameIndex;

    player.setSpeed(2);

    expect(player.getStepIndex()).toBe(0);
    expect(player.getState()).toBe("playing");
    expect(host.querySelector("svg[data-motion-avatar]")).toBe(svg);
    expect(svg.dataset.clipId).toBe("include-today");
    expect(svg.dataset.frameIndex).toBe(frameIndex);
    expect(svg.dataset.state).toBe("playing");

    vi.advanceTimersByTime(30);
    expect(player.getStepIndex()).toBe(1);
    vi.advanceTimersByTime(200);
    expect(player.getState()).toBe("complete");
    player.destroy();
  });

  it("pauses with the remaining step time and resumes without restarting", () => {
    const host = document.createElement("div");
    document.body.append(host);
    const player = createMotionQueuePlayer(host, clips);

    player.load(steps, 1);
    player.play();
    vi.advanceTimersByTime(40);
    player.pause();
    const pausedFrame = getSvg(host).dataset.frameIndex;

    vi.advanceTimersByTime(1_000);
    expect(player.getStepIndex()).toBe(0);
    expect(getSvg(host).dataset.frameIndex).toBe(pausedFrame);

    player.play();
    vi.advanceTimersByTime(100);
    expect(player.getStepIndex()).toBe(1);

    player.replay();
    expect(player.getStepIndex()).toBe(0);
    expect(getSvg(host).dataset.clipId).toBe("include-today");
    player.destroy();
  });

  it("rewinds while paused and remains paused until explicitly played", () => {
    const host = document.createElement("div");
    document.body.append(host);
    const player = createMotionQueuePlayer(host, clips);

    player.load(steps, 1);
    player.play();
    vi.advanceTimersByTime(110);
    player.pause();
    expect(player.getStepIndex()).toBe(1);

    const svg = getSvg(host);
    player.replay(false);

    expect(player.getStepIndex()).toBe(0);
    expect(player.getState()).toBe("paused");
    expect(host.querySelector("svg[data-motion-avatar]")).toBe(svg);
    expect(svg.dataset.clipId).toBe("include-today");
    expect(svg.dataset.frameIndex).toBe("0");
    expect(svg.dataset.state).toBe("paused");

    vi.advanceTimersByTime(500);
    expect(player.getStepIndex()).toBe(0);
    expect(player.getState()).toBe("paused");

    player.play();
    vi.advanceTimersByTime(70);
    expect(player.getStepIndex()).toBe(0);
    vi.advanceTimersByTime(50);
    expect(player.getStepIndex()).toBe(1);
    player.destroy();
  });

  it("uses the selected identity theme and stays idle for fallback-only steps", () => {
    const host = document.createElement("div");
    document.body.append(host);
    const player = createMotionQueuePlayer(host, clips, { avatarName: "Meera" });

    player.load([
      {
        id: "fallback",
        glossItemId: "gloss_fallback",
        kind: "caption",
        label: "UNKNOWN",
        durationMs: 80,
        confidence: "low"
      }
    ]);
    player.play();

    const svg = getSvg(host);
    expect(svg.dataset.avatarName).toBe("Meera");
    expect(svg.dataset.state).toBe("idle");
    vi.advanceTimersByTime(80);
    expect(player.getState()).toBe("complete");
    player.destroy();
  });

  it("plays a dynamic official ISLRTC compound clip for an unsupported word", () => {
    const host = document.createElement("div");
    document.body.append(host);
    const player = createMotionQueuePlayer(host, new Map());

    player.load([
      {
        id: "fallback-ai",
        glossItemId: "gloss_ai",
        kind: "fingerspell",
        label: "AI",
        fingerspellingSource: "islrtc_official",
        durationMs: 1_300,
        confidence: "low"
      }
    ]);
    player.play();

    const svg = getSvg(host);
    expect(svg.dataset.clipId).toMatch(/^islrtc-fingerspell-ai-/);
    expect(svg.dataset.state).toBe("playing");
    vi.advanceTimersByTime(300);
    expect(Number(svg.dataset.frameIndex)).toBeGreaterThan(0);
    player.destroy();
  });

  it("does not silently fingerspell a sign step whose advertised clip is missing", () => {
    const host = document.createElement("div");
    document.body.append(host);
    const player = createMotionQueuePlayer(host, new Map());

    player.load([
      {
        id: "missing-sign",
        glossItemId: "gloss_missing",
        kind: "sign",
        label: "MISSING",
        motionClipId: "include-missing",
        motionSource: "include_dataset",
        expertReviewed: false,
        durationMs: 500,
        confidence: "low"
      }
    ]);
    player.play();

    const svg = getSvg(host);
    expect(svg.dataset.clipId).toBe("motion-unavailable");
    expect(svg.dataset.state).toBe("idle");
    player.destroy();
  });
});

function createClip(id: string, offset: number): MotionClip {
  return {
    id,
    label: id,
    fps: 25,
    frames: [createFrame(offset), createFrame(offset + 0.08), createFrame(offset + 0.12)]
  };
}

function createFrame(offset: number): MotionFrame {
  return {
    pose: [
      { x: 0.5, y: 0.2 },
      { x: 0.5, y: 0.2 },
      { x: 0.47, y: 0.19 },
      { x: 0.5, y: 0.2 },
      { x: 0.5, y: 0.2 },
      { x: 0.53, y: 0.19 },
      { x: 0.5, y: 0.2 },
      { x: 0.44, y: 0.21 },
      { x: 0.56, y: 0.21 },
      { x: 0.5, y: 0.25 },
      { x: 0.5, y: 0.25 },
      { x: 0.35, y: 0.35 },
      { x: 0.65, y: 0.35 },
      { x: 0.29, y: 0.5 },
      { x: 0.71 - offset, y: 0.5 },
      { x: 0.32, y: 0.64 },
      { x: 0.68 - offset, y: 0.62 }
    ],
    leftHand: createHand(0.32, 0.64),
    rightHand: createHand(0.68 - offset, 0.62)
  };
}

function createHand(wristX: number, wristY: number): Array<{ x: number; y: number }> {
  return Array.from({ length: 21 }, (_, index) => ({
    x: wristX + (index % 4) * 0.008,
    y: wristY - Math.floor(index / 4) * 0.018
  }));
}

function getSvg(host: HTMLElement): SVGSVGElement {
  const svg = host.querySelector<SVGSVGElement>("svg[data-motion-avatar]");
  expect(svg).toBeTruthy();
  return svg as SVGSVGElement;
}
