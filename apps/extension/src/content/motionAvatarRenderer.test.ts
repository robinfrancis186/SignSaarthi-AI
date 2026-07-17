import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  createMotionAvatarRenderer,
  type MotionClip,
  type MotionFrame,
  type MotionPoint
} from "./motionAvatarRenderer";

let animationFrames: Map<number, FrameRequestCallback>;
let nextAnimationFrameId: number;

describe("motion avatar renderer", () => {
  beforeEach(() => {
    animationFrames = new Map();
    nextAnimationFrameId = 1;
    document.body.innerHTML = "";
    vi.stubGlobal(
      "requestAnimationFrame",
      vi.fn((callback: FrameRequestCallback) => {
        const id = nextAnimationFrameId;
        nextAnimationFrameId += 1;
        animationFrames.set(id, callback);
        return id;
      })
    );
    vi.stubGlobal(
      "cancelAnimationFrame",
      vi.fn((id: number) => {
        animationFrames.delete(id);
      })
    );
  });

  afterEach(() => {
    vi.unstubAllGlobals();
    document.body.innerHTML = "";
  });

  it("creates an accessible, stable SVG with a neutral upper-body pose", () => {
    const container = document.createElement("div");
    document.body.append(container);

    const controller = createMotionAvatarRenderer(container);
    const svg = getSvg(container);

    expect(svg.getAttribute("role")).toBe("img");
    expect(svg.getAttribute("viewBox")).toBe("0 0 320 360");
    expect(svg.getAttribute("width")).toBe("320");
    expect(svg.getAttribute("height")).toBe("360");
    expect(svg.getAttribute("aria-labelledby")?.split(" ")).toHaveLength(2);
    expect(svg.querySelector("title")?.textContent).toContain("at rest");
    expect(svg.querySelector('[data-avatar-part="head"]')).toBeTruthy();
    expect(svg.querySelector('[data-avatar-part="torso"]')).toBeTruthy();
    expect(svg.querySelector('[data-avatar-part="left-upper-arm"]')).toBeTruthy();
    expect(svg.querySelectorAll("[data-bone]")).toHaveLength(42);
    expect(
      svg.querySelector('[data-hand="left"][data-bone="0-1"]')?.getAttribute("visibility")
    ).toBe("visible");
    expect(controller.isPlaying()).toBe(false);
    expect(controller.getFrameIndex()).toBe(0);

    controller.destroy();
  });

  it("applies the selected avatar identity palette to the shared motion rig", () => {
    const container = document.createElement("div");
    document.body.append(container);

    const controller = createMotionAvatarRenderer(container, { avatarName: "Arjun" });
    const svg = getSvg(container);
    const firstGradientStops = [...svg.querySelectorAll("linearGradient")][0]?.querySelectorAll("stop");

    expect(svg.dataset.avatarName).toBe("Arjun");
    expect(firstGradientStops?.[0]?.getAttribute("stop-color")).toBe("#e8b28d");
    expect(firstGradientStops?.[2]?.getAttribute("stop-color")).toBe("#78443d");

    controller.destroy();
  });

  it("interpolates real pose and hand landmarks between clip frames", () => {
    const container = document.createElement("div");
    document.body.append(container);
    const onFrame = vi.fn();
    const controller = createMotionAvatarRenderer(container, { onFrame });
    const clip = createClip([createFrame(0), createFrame(0.12)]);

    controller.loadClip(clip);
    const svg = getSvg(container);
    const fingertip = getElement<SVGCircleElement>(svg, '[data-motion-landmark="right-hand-8"]');
    const torso = getElement<SVGPathElement>(svg, '[data-avatar-part="torso"]');
    const startTipX = numberAttribute(fingertip, "cx");
    const startTorsoPath = torso.getAttribute("d");

    controller.play();
    advanceAnimationFrame(0);
    advanceAnimationFrame(50);

    const interpolatedTipX = numberAttribute(fingertip, "cx");
    expect(interpolatedTipX).not.toBe(startTipX);
    expect(interpolatedTipX).toBeLessThan(startTipX);
    expect(torso.getAttribute("d")).not.toBe(startTorsoPath);
    expect(svg.dataset.frameProgress).toBe("0.500");
    expect(controller.getFrameIndex()).toBe(0);

    advanceAnimationFrame(100);
    expect(controller.getFrameIndex()).toBe(1);
    expect(controller.isPlaying()).toBe(false);
    expect(onFrame.mock.calls.map(([index]) => index)).toEqual([0, 1]);

    controller.destroy();
  });

  it("supports pause, replay, playback speed, and scoped destruction", () => {
    const container = document.createElement("div");
    const existingChild = document.createElement("span");
    existingChild.textContent = "keep me";
    container.append(existingChild);
    document.body.append(container);
    const onFrame = vi.fn();
    const controller = createMotionAvatarRenderer(container, { onFrame });

    controller.loadClip(createClip([createFrame(0), createFrame(0.08), createFrame(0.16)]));
    const svg = getSvg(container);
    const fingertip = getElement<SVGCircleElement>(svg, '[data-motion-landmark="right-hand-8"]');
    const firstTipX = numberAttribute(fingertip, "cx");

    controller.play();
    advanceAnimationFrame(0);
    advanceAnimationFrame(50);
    const movingTipX = numberAttribute(fingertip, "cx");
    expect(movingTipX).not.toBe(firstTipX);

    controller.pause();
    expect(controller.isPlaying()).toBe(false);
    advanceAnimationFrame(1_000);
    expect(numberAttribute(fingertip, "cx")).toBe(movingTipX);

    controller.replay();
    expect(controller.isPlaying()).toBe(true);
    expect(controller.getFrameIndex()).toBe(0);
    expect(numberAttribute(fingertip, "cx")).toBe(firstTipX);

    controller.setSpeed(2);
    advanceAnimationFrame(1_000);
    advanceAnimationFrame(1_050);
    expect(controller.getFrameIndex()).toBe(1);
    expect(onFrame.mock.calls.map(([index]) => index)).toContain(1);

    controller.destroy();
    expect(controller.isPlaying()).toBe(false);
    expect(container.querySelector("svg[data-motion-avatar]")).toBeNull();
    expect(container.contains(existingChild)).toBe(true);
    expect(() => {
      controller.play();
      controller.pause();
      controller.replay();
      controller.setSpeed(1.5);
      controller.loadClip(createClip([createFrame(0)]));
      controller.destroy();
    }).not.toThrow();
  });

  it("blends from the displayed pose into the next queued sign without a hard reset", () => {
    const container = document.createElement("div");
    document.body.append(container);
    const onComplete = vi.fn();
    const onFrame = vi.fn();
    const controller = createMotionAvatarRenderer(container, { onComplete, onFrame });

    controller.loadClip(createClip([createFrame(0), createFrame(0.12)]));
    controller.play();
    advanceAnimationFrame(0);
    advanceAnimationFrame(100);

    const svg = getSvg(container);
    const fingertip = getElement<SVGCircleElement>(svg, '[data-motion-landmark="right-hand-8"]');
    const previousTipX = numberAttribute(fingertip, "cx");
    const nextClip = { ...createClip([createFrame(0.36), createFrame(0.48)]), id: "next-motion" };

    controller.loadClip(nextClip, { transitionFrames: 4 });

    expect(svg.dataset.clipId).toBe("next-motion");
    expect(svg.dataset.transitionFrames).toBe("4");
    expect(numberAttribute(fingertip, "cx")).toBe(previousTipX);

    controller.play();
    advanceAnimationFrame(200);
    advanceAnimationFrame(250);
    expect(numberAttribute(fingertip, "cx")).not.toBe(previousTipX);

    advanceAnimationFrame(300);
    advanceAnimationFrame(400);
    advanceAnimationFrame(500);
    advanceAnimationFrame(600);
    advanceAnimationFrame(700);
    expect(svg.dataset.state).toBe("complete");
    expect(svg.dataset.frameIndex).toBe("5");
    expect(onFrame).toHaveBeenLastCalledWith(5);
    expect(onComplete).toHaveBeenCalledTimes(2);

    controller.destroy();
  });

  it("renders sparse and invalid landmark data without malformed SVG geometry", () => {
    const container = document.createElement("div");
    document.body.append(container);
    const controller = createMotionAvatarRenderer(container);
    const sparsePose = new Array<MotionPoint>(33);
    sparsePose[11] = { x: 0.36, y: 0.35 };
    sparsePose[12] = { x: Number.NaN, y: 0.35 };
    const sparseHand = new Array<MotionPoint>(21);
    sparseHand[0] = { x: 0.68, y: 0.62, visibility: 0.9 };
    sparseHand[8] = { x: 0.58, y: 0.38, visibility: 0.9 };

    expect(() => {
      controller.loadClip({
        id: "sparse",
        label: "Sparse motion",
        fps: 0,
        frames: [
          { pose: sparsePose, leftHand: [], rightHand: sparseHand },
          { pose: [], leftHand: new Array<MotionPoint>(21), rightHand: [] }
        ]
      });
      controller.play();
      advanceAnimationFrame(0);
      advanceAnimationFrame(1_000);
    }).not.toThrow();

    const svg = getSvg(container);
    expect(svg.outerHTML).not.toMatch(/NaN|Infinity|undefined/);
    expect(svg.querySelector('[data-avatar-part="torso"]')?.getAttribute("d")).toMatch(/^M /);

    controller.loadClip({ id: "empty", label: "Empty", fps: 30, frames: [] });
    controller.play();
    expect(controller.isPlaying()).toBe(false);
    expect(svg.dataset.state).toBe("idle");

    controller.destroy();
  });

  it("renders source-observed brow and mouth motion from the compact face subset", () => {
    const container = document.createElement("div");
    document.body.append(container);
    const controller = createMotionAvatarRenderer(container);
    const first = createFrame(0);
    const second = createFrame(0.1);
    first.face = createFace(0);
    second.face = createFace(0.025);

    controller.loadClip(createClip([first, second]));
    const svg = getSvg(container);
    const brow = getElement<SVGPathElement>(svg, '[data-avatar-part="left-brow"]');
    const mouth = getElement<SVGPathElement>(svg, '[data-avatar-part="mouth"]');
    const initialBrow = brow.getAttribute("d");
    const initialMouth = mouth.getAttribute("d");

    controller.play();
    advanceAnimationFrame(0);
    advanceAnimationFrame(100);

    expect(brow.getAttribute("d")).not.toBe(initialBrow);
    expect(mouth.getAttribute("d")).not.toBe(initialMouth);
    controller.destroy();
  });
});

function createClip(frames: MotionFrame[]): MotionClip {
  return {
    id: "hello-motion",
    label: "HELLO",
    fps: 10,
    frames
  };
}

function createFrame(movement: number): MotionFrame {
  const pose = new Array<MotionPoint>(33);
  pose[0] = { x: 0.5 - movement * 0.08, y: 0.2, visibility: 0.99 };
  pose[2] = { x: 0.475 - movement * 0.08, y: 0.19, visibility: 0.99 };
  pose[5] = { x: 0.525 - movement * 0.08, y: 0.19, visibility: 0.99 };
  pose[7] = { x: 0.44 - movement * 0.08, y: 0.21, visibility: 0.99 };
  pose[8] = { x: 0.56 - movement * 0.08, y: 0.21, visibility: 0.99 };
  pose[11] = { x: 0.35, y: 0.35, visibility: 0.99 };
  pose[12] = { x: 0.65 - movement * 0.18, y: 0.35 - movement * 0.08, visibility: 0.99 };
  pose[13] = { x: 0.29, y: 0.5, visibility: 0.99 };
  pose[14] = { x: 0.71 - movement * 0.48, y: 0.5 - movement * 0.3, visibility: 0.99 };
  pose[15] = { x: 0.32, y: 0.64, visibility: 0.99 };
  pose[16] = { x: 0.68 - movement, y: 0.62 - movement * 0.45, visibility: 0.99 };
  pose[23] = { x: 0.4, y: 0.79, visibility: 0.99 };
  pose[24] = { x: 0.6 - movement * 0.08, y: 0.79, visibility: 0.99 };

  return {
    pose,
    leftHand: createHand(0.32, 0.64, -1),
    rightHand: createHand(0.68 - movement, 0.62 - movement * 0.45, 1)
  };
}

function createHand(wristX: number, wristY: number, direction: -1 | 1): MotionPoint[] {
  const point = (x: number, y: number): MotionPoint => ({
    x: wristX + x * direction,
    y: wristY + y,
    visibility: 0.96
  });

  return [
    point(0, 0),
    point(-0.018, -0.015),
    point(-0.034, -0.03),
    point(-0.047, -0.05),
    point(-0.06, -0.07),
    point(0.002, -0.035),
    point(0.005, -0.075),
    point(0.008, -0.11),
    point(0.011, -0.145),
    point(0.018, -0.038),
    point(0.022, -0.083),
    point(0.025, -0.123),
    point(0.028, -0.16),
    point(0.033, -0.034),
    point(0.04, -0.075),
    point(0.046, -0.111),
    point(0.052, -0.14),
    point(0.046, -0.026),
    point(0.057, -0.058),
    point(0.064, -0.085),
    point(0.071, -0.109)
  ];
}

function createFace(expression: number): MotionPoint[] {
  return [
    { x: 0.465, y: 0.175 + expression },
    { x: 0.473, y: 0.17 + expression },
    { x: 0.482, y: 0.168 + expression },
    { x: 0.49, y: 0.17 + expression },
    { x: 0.497, y: 0.175 + expression },
    { x: 0.503, y: 0.175 + expression },
    { x: 0.51, y: 0.17 + expression },
    { x: 0.518, y: 0.168 + expression },
    { x: 0.527, y: 0.17 + expression },
    { x: 0.535, y: 0.175 + expression },
    { x: 0.478, y: 0.186 },
    { x: 0.478, y: 0.194 },
    { x: 0.522, y: 0.186 },
    { x: 0.522, y: 0.194 },
    { x: 0.48 - expression, y: 0.235 },
    { x: 0.52 + expression, y: 0.235 },
    { x: 0.5, y: 0.23 - expression },
    { x: 0.5, y: 0.24 + expression }
  ];
}

function advanceAnimationFrame(timestamp: number): void {
  const queuedFrames = [...animationFrames.values()];
  animationFrames.clear();
  for (const callback of queuedFrames) {
    callback(timestamp);
  }
}

function getSvg(container: HTMLElement): SVGSVGElement {
  const svg = container.querySelector<SVGSVGElement>("svg[data-motion-avatar]");
  expect(svg).toBeTruthy();
  return svg as SVGSVGElement;
}

function getElement<T extends Element>(root: ParentNode, selector: string): T {
  const element = root.querySelector<T>(selector);
  expect(element).toBeTruthy();
  return element as T;
}

function numberAttribute(element: Element, name: string): number {
  const value = Number(element.getAttribute(name));
  expect(Number.isFinite(value)).toBe(true);
  return value;
}
