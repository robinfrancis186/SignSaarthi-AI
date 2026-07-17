import type { AvatarQueueStep } from "@signsaarthi/shared";
import {
  createMotionAvatarRenderer,
  type MotionAvatarController,
  type MotionAvatarIdentity,
  type MotionClip
} from "./motionAvatarRenderer";
import { createFingerspellingMotion } from "./fingerspellingMotion";

export type MotionQueuePlayerState = "idle" | "playing" | "paused" | "complete";

export type MotionQueuePlayerOptions = {
  avatarName?: MotionAvatarIdentity;
  transitionFrames?: number;
  onStepChange?: (index: number, step: AvatarQueueStep) => void;
  onStateChange?: (state: MotionQueuePlayerState) => void;
};

export interface MotionQueuePlayer {
  load(steps: readonly AvatarQueueStep[], speed?: number, startIndex?: number): void;
  enqueue(steps: readonly AvatarQueueStep[]): void;
  play(): void;
  pause(): void;
  replay(autoplay?: boolean): void;
  setSpeed(speed: number): void;
  destroy(): void;
  getStepIndex(): number;
  getState(): MotionQueuePlayerState;
}

const EMPTY_CLIP: MotionClip = {
  id: "motion-unavailable",
  label: "Motion unavailable",
  fps: 30,
  frames: []
};

export function createMotionQueuePlayer(
  container: HTMLElement,
  motionClips: ReadonlyMap<string, MotionClip>,
  options: MotionQueuePlayerOptions = {}
): MotionQueuePlayer {
  let currentStepUsesMotion = false;
  const renderer: MotionAvatarController = createMotionAvatarRenderer(container, {
    ...(options.avatarName ? { avatarName: options.avatarName } : {}),
    onComplete() {
      if (currentStepUsesMotion) {
        advanceStep();
      }
    }
  });
  const transitionFrames = Math.max(0, Math.min(12, Math.round(options.transitionFrames ?? 5)));

  let steps: AvatarQueueStep[] = [];
  let stepDurationsMs: number[] = [];
  let stepIndex = 0;
  let speed = 1;
  let state: MotionQueuePlayerState = "idle";
  let remainingMs = 0;
  let stepStartedAt = 0;
  let timer: number | undefined;
  let destroyed = false;

  function updateState(nextState: MotionQueuePlayerState): void {
    if (state === nextState) {
      return;
    }
    state = nextState;
    options.onStateChange?.(state);
  }

  function clearTimer(): void {
    if (timer === undefined) {
      return;
    }
    window.clearTimeout(timer);
    timer = undefined;
  }

  function currentStep(): AvatarQueueStep | undefined {
    return steps[stepIndex];
  }

  function loadCurrentStep(blendFromPrevious: boolean): void {
    const step = currentStep();
    if (!step) {
      currentStepUsesMotion = false;
      renderer.loadClip(EMPTY_CLIP);
      remainingMs = 0;
      return;
    }

    const directClip = step.motionClipId ? motionClips.get(step.motionClipId) : undefined;
    const fingerspellingClip =
      !directClip &&
      (step.fingerspellingSource === "islrtc_official" ||
        step.kind === "fingerspell")
        ? createFingerspellingMotion(step.label)
        : undefined;
    const clip = directClip ?? fingerspellingClip;
    currentStepUsesMotion = Boolean(clip && clip.frames.length > 1);
    renderer.loadClip(clip ?? { ...EMPTY_CLIP, label: step.label }, {
      transitionFrames: clip && blendFromPrevious ? transitionFrames : 0
    });
    renderer.setSpeed(speed);
    remainingMs = stepDurationsMs[stepIndex] ?? Math.max(0, step.durationMs);
    options.onStepChange?.(stepIndex, step);
  }

  function scheduleAdvance(): void {
    clearTimer();
    if (state !== "playing" || currentStepUsesMotion) {
      return;
    }
    if (remainingMs <= 0) {
      advanceStep();
      return;
    }
    stepStartedAt = Date.now();
    timer = window.setTimeout(advanceStep, remainingMs);
  }

  function advanceStep(): void {
    timer = undefined;
    if (destroyed || state !== "playing") {
      return;
    }
    if (stepIndex >= steps.length - 1) {
      renderer.pause();
      currentStepUsesMotion = false;
      remainingMs = 0;
      updateState("complete");
      return;
    }

    stepIndex += 1;
    loadCurrentStep(true);
    renderer.play();
    scheduleAdvance();
  }

  function startPlayback(): void {
    if (destroyed || steps.length === 0 || state === "playing" || state === "complete") {
      return;
    }
    renderer.play();
    updateState("playing");
    scheduleAdvance();
  }

  return {
    load(nextSteps, nextSpeed = 1, startIndex = 0) {
      if (destroyed) {
        return;
      }
      clearTimer();
      renderer.pause();
      steps = [...nextSteps];
      stepDurationsMs = steps.map((step) => Math.max(0, step.durationMs));
      speed = normalizeSpeed(nextSpeed);
      stepIndex = steps.length ? Math.min(Math.max(0, Math.round(startIndex)), steps.length - 1) : 0;
      loadCurrentStep(false);
      updateState(steps.length ? "paused" : "idle");
    },

    enqueue(nextSteps) {
      if (destroyed || nextSteps.length === 0) {
        return;
      }

      const hadSteps = steps.length > 0;
      steps.push(...nextSteps);
      stepDurationsMs.push(...nextSteps.map((step) => Math.max(0, step.durationMs)));

      if (!hadSteps) {
        stepIndex = 0;
        loadCurrentStep(false);
        updateState("paused");
      } else if (state === "complete") {
        updateState("paused");
      }
    },

    play() {
      startPlayback();
    },

    pause() {
      if (destroyed) {
        return;
      }
      if (state === "playing") {
        if (currentStepUsesMotion) {
          renderer.pause();
          updateState("paused");
          return;
        }
        remainingMs = Math.max(0, remainingMs - Math.max(0, Date.now() - stepStartedAt));
      }
      clearTimer();
      renderer.pause();
      if (steps.length && state !== "complete") {
        updateState("paused");
      }
    },

    replay(autoplay = true) {
      if (destroyed || steps.length === 0) {
        return;
      }
      clearTimer();
      stepIndex = 0;
      loadCurrentStep(false);
      updateState("paused");
      if (autoplay) {
        startPlayback();
      }
    },

    setSpeed(nextSpeed) {
      if (destroyed) {
        return;
      }

      const normalizedSpeed = normalizeSpeed(nextSpeed);
      if (normalizedSpeed === speed) {
        renderer.setSpeed(normalizedSpeed);
        return;
      }

      const wasPlaying = state === "playing";
      if (wasPlaying && !currentStepUsesMotion) {
        remainingMs = Math.max(0, remainingMs - Math.max(0, Date.now() - stepStartedAt));
        clearTimer();
      }

      const durationScale = speed / normalizedSpeed;
      remainingMs = Math.max(0, remainingMs * durationScale);
      stepDurationsMs = stepDurationsMs.map((durationMs) => Math.max(0, durationMs * durationScale));
      speed = normalizedSpeed;
      renderer.setSpeed(speed);
      if (wasPlaying) {
        scheduleAdvance();
      }
    },

    destroy() {
      if (destroyed) {
        return;
      }
      destroyed = true;
      clearTimer();
      renderer.destroy();
      steps = [];
      stepDurationsMs = [];
      remainingMs = 0;
      currentStepUsesMotion = false;
      state = "idle";
    },

    getStepIndex() {
      return stepIndex;
    },

    getState() {
      return state;
    }
  };
}

function normalizeSpeed(value: number): number {
  if (!Number.isFinite(value)) {
    return 1;
  }
  return Math.min(2, Math.max(0.5, value));
}
