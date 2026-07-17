import type {
  AvatarQueue,
  AvatarQueueStep,
  AvatarStepKind,
  ISLGlossItem,
  PlanningActionDecision,
  RealtimePlaybackBacklog
} from "@signsaarthi/shared";
import { findMotionClip, findMotionClipById } from "./motionCatalog.js";

export type { AvatarQueue, AvatarQueueStep, AvatarStepKind };
export {
  findMotionClip,
  findMotionClipById,
  motionCatalog,
  normalizeMotionLabel
} from "./motionCatalog.js";

export type CreateAvatarQueueInput = {
  interpretationId: string;
  speed: number;
  glossSequence: ISLGlossItem[];
  now?: () => string;
};

export const REALTIME_PLAYBACK_POLICY = {
  maxPendingSteps: 12,
  maxPendingMs: 12_000,
  catchUpThresholdMs: 6_000,
  maxAutomaticSpeedMultiplier: 1
} as const;

export type RealtimePlaybackPlan = {
  playbackSteps: AvatarQueueStep[];
  backlog: RealtimePlaybackBacklog;
};

export type RealtimePlaybackPolicyOptions = {
  baseSpeed?: number;
  historyStepCount?: number;
  maxPendingSteps?: number;
  maxPendingMs?: number;
  catchUpThresholdMs?: number;
  maxSpeed?: number;
};

export function createAvatarQueue(input: CreateAvatarQueueInput): AvatarQueue {
  const safeSpeed = Math.min(2, Math.max(0.5, input.speed));
  const steps = input.glossSequence.map<AvatarQueueStep>((item) => {
    const catalogMotion =
      findMotionClipById(item.signAssetId) ?? findMotionClip(item.token, item.gloss);
    const motion = catalogMotion?.playable ? catalogMotion : undefined;
    const kind = getStepKind(item, Boolean(motion));
    const sourceSegmentId =
      item.sourceSegmentId ?? `segment_${stableHash(`interpretation:${input.interpretationId}`)}`;
    const planningActionId = item.planningActionId ?? item.id;
    const plannedAction = item.plannedAction ?? inferPlannedAction(item);
    const resolvedAction = {
      kind,
      label: item.gloss
    } satisfies PlanningActionDecision;
    const durationMs = Math.round(
      (kind === "sign" && motion
        ? motion.durationMs
        : kind === "fingerspell"
          ? estimateFingerspellingDurationMs(item.gloss)
          : kind === "sign"
            ? 1400
            : 1500) / safeSpeed
    );
    const base = {
      id: `step_${stableHash(`${sourceSegmentId}:${planningActionId}`)}`,
      glossItemId: item.id,
      kind,
      label: item.gloss,
      durationMs,
      confidence: item.confidence,
      sourceSegmentId,
      planningActionId,
      plannedAction,
      resolvedAction,
      ...(item.sourceTokenIds ? { sourceTokenIds: item.sourceTokenIds } : {}),
      ...(item.sourceSpan ? { sourceSpan: item.sourceSpan } : {}),
      ...(item.clauseId ? { clauseId: item.clauseId } : {}),
      ...(item.provenance ? { provenance: item.provenance } : {}),
      ...(item.reviewStatus ? { reviewStatus: item.reviewStatus } : {})
    };

    if (motion) {
      return {
        ...base,
        ...(item.signAssetId ? { signAssetId: item.signAssetId } : {}),
        motionClipId: motion.id,
        motionSource: motion.source,
        expertReviewed: motion.expertReviewed
      };
    }
    const fallback = item.signAssetId ? { ...base, signAssetId: item.signAssetId } : base;
    return kind === "fingerspell"
      ? { ...fallback, fingerspellingSource: "islrtc_official" as const }
      : fallback;
  });
  const hasFallback = steps.some((step) => step.kind !== "sign");
  const hasDatasetMotion = steps.some((step) => step.motionSource === "include_dataset");
  const createdAt = input.now?.() ?? new Date().toISOString();
  const backlog = planRealtimePlayback(steps, {
    baseSpeed: safeSpeed,
    historyStepCount: steps.length
  }).backlog;

  const queue: AvatarQueue = {
    id: `queue_${stableHash(
      `${input.interpretationId}:${safeSpeed}:${steps.map((step) => step.id).join("|")}`
    )}`,
    interpretationId: input.interpretationId,
    speed: safeSpeed,
    steps,
    backlog,
    createdAt
  };
  if (hasFallback) {
    queue.honestFallbackText =
      "Unsupported words use official ISLRTC A-Z fingerspelling; automated avatar transfer is not expert-reviewed.";
  }
  if (hasDatasetMotion) {
    queue.motionReviewNotice =
      "Real INCLUDE keypoint motion; dataset-derived isolated signs are not expert-certified ISL.";
  }
  return queue;
}

export function planRealtimePlayback(
  steps: readonly AvatarQueueStep[],
  options: RealtimePlaybackPolicyOptions = {}
): RealtimePlaybackPlan {
  const baseSpeed = clampSpeed(options.baseSpeed ?? 1);
  const defaultMaxSpeed = clampSpeed(
    baseSpeed * REALTIME_PLAYBACK_POLICY.maxAutomaticSpeedMultiplier
  );
  const maxSpeed = Math.max(
    baseSpeed,
    clampSpeed(options.maxSpeed ?? defaultMaxSpeed)
  );
  const maxPendingSteps = Math.max(
    1,
    Math.round(options.maxPendingSteps ?? REALTIME_PLAYBACK_POLICY.maxPendingSteps)
  );
  const maxPendingMs = Math.max(
    1,
    Math.round(options.maxPendingMs ?? REALTIME_PLAYBACK_POLICY.maxPendingMs)
  );
  const catchUpThresholdMs = Math.max(
    1,
    Math.round(options.catchUpThresholdMs ?? REALTIME_PLAYBACK_POLICY.catchUpThresholdMs)
  );
  const estimatedPendingMs = sumStepDurations(steps);
  const playbackSteps: AvatarQueueStep[] = [];
  let playbackDurationMs = 0;

  for (const step of steps) {
    const stepDurationMs = Math.max(0, Math.round(step.durationMs));
    const wouldExceedCount = playbackSteps.length >= maxPendingSteps;
    const wouldExceedDuration =
      playbackSteps.length > 0 && playbackDurationMs + stepDurationMs > maxPendingMs;
    if (wouldExceedCount || wouldExceedDuration) {
      break;
    }
    playbackSteps.push(step);
    playbackDurationMs += stepDurationMs;
  }

  const deferredStepCount = Math.max(0, steps.length - playbackSteps.length);
  const catchUpBasisMs = deferredStepCount > 0 ? estimatedPendingMs : playbackDurationMs;
  const catchUpSpeed =
    catchUpBasisMs > catchUpThresholdMs
      ? Math.min(maxSpeed, Math.max(baseSpeed, roundSpeed(catchUpBasisMs / catchUpThresholdMs)))
      : baseSpeed;
  const status =
    deferredStepCount > 0 ? "backlog" : catchUpSpeed > baseSpeed ? "catching_up" : "realtime";

  return {
    playbackSteps,
    backlog: {
      policy: "bounded_realtime",
      status,
      historyStepCount: Math.max(
        steps.length,
        Math.round(options.historyStepCount ?? steps.length)
      ),
      pendingStepCount: steps.length,
      playbackStepCount: playbackSteps.length,
      deferredStepCount,
      estimatedPendingMs,
      playbackDurationMs,
      maxPendingSteps,
      maxPendingMs,
      catchUpSpeed
    }
  };
}

export function getRollingAvatarQueueOverlap(
  currentQueue: AvatarQueue,
  incomingQueue: AvatarQueue
): number {
  const maxOverlap = Math.min(currentQueue.steps.length, incomingQueue.steps.length);
  for (let overlap = maxOverlap; overlap > 0; overlap -= 1) {
    const currentOffset = currentQueue.steps.length - overlap;
    const matches = incomingQueue.steps
      .slice(0, overlap)
      .every((incomingStep, index) =>
        isSameStableStep(
          currentQueue.steps[currentOffset + index]!,
          incomingStep,
          currentQueue.interpretationId,
          incomingQueue.interpretationId
        )
      );
    if (matches) {
      return overlap;
    }
  }
  return 0;
}

export function mergeRollingAvatarQueues(
  currentQueue: AvatarQueue | undefined,
  incomingQueue: AvatarQueue
): AvatarQueue {
  if (!currentQueue) {
    return incomingQueue;
  }

  const overlap = getRollingAvatarQueueOverlap(currentQueue, incomingQueue);
  const steps = [...currentQueue.steps, ...incomingQueue.steps.slice(overlap)];
  const motionClips = mergeMotionClips(currentQueue.motionClips, incomingQueue.motionClips);
  return {
    ...incomingQueue,
    steps,
    ...(motionClips.length ? { motionClips } : {}),
    backlog: planRealtimePlayback(steps, {
      baseSpeed: incomingQueue.speed,
      historyStepCount: steps.length
    }).backlog
  };
}

function mergeMotionClips(
  current: AvatarQueue["motionClips"],
  incoming: AvatarQueue["motionClips"]
): NonNullable<AvatarQueue["motionClips"]> {
  const byId = new Map<string, NonNullable<AvatarQueue["motionClips"]>[number]>();
  for (const clip of [...(current ?? []), ...(incoming ?? [])]) {
    byId.set(clip.id, clip);
  }
  return [...byId.values()];
}

export function getFingerspellingLetters(value: string): string[] {
  return value.normalize("NFKD").replace(/\p{M}/gu, "").toUpperCase().match(/[A-Z]/g) ?? [];
}

export function estimateFingerspellingDurationMs(value: string): number {
  const letterCount = getFingerspellingLetters(value).length;
  if (letterCount === 0) {
    return 1500;
  }
  const frameCount = letterCount * 12 + Math.max(0, letterCount - 1) * 3;
  return Math.round(((frameCount - 1) / 20) * 1000);
}

export function replayLastSegment(
  queue: AvatarQueue,
  now: () => string = () => new Date().toISOString()
): AvatarQueue {
  const createdAt = now();
  return {
    ...queue,
    id: `queue_replay_${stableHash(`${queue.id}:${createdAt}`)}`,
    replayOfQueueId: queue.id,
    createdAt
  };
}

function getStepKind(item: ISLGlossItem, hasMotionClip: boolean): AvatarStepKind {
  if (hasMotionClip) {
    return "sign";
  }
  if (item.fallback === "fingerspell") {
    return "fingerspell";
  }
  if (item.fallback === "caption") {
    return "caption";
  }
  return item.fallback === "unknown" ? "unknown" : "caption";
}

function inferPlannedAction(item: ISLGlossItem): PlanningActionDecision {
  if (item.signAssetId) {
    return { kind: "sign_candidate", label: item.gloss };
  }
  if (item.fallback === "fingerspell") {
    return { kind: "fingerspell", label: item.gloss };
  }
  if (item.fallback === "unknown") {
    return { kind: "unknown", label: item.gloss };
  }
  return { kind: "caption", label: item.gloss };
}

function isSameStableStep(
  currentStep: AvatarQueueStep,
  incomingStep: AvatarQueueStep,
  currentInterpretationId: string,
  incomingInterpretationId: string
): boolean {
  if (currentStep.sourceSegmentId || incomingStep.sourceSegmentId) {
    return (
      Boolean(currentStep.sourceSegmentId) &&
      currentStep.sourceSegmentId === incomingStep.sourceSegmentId &&
      currentStep.id === incomingStep.id
    );
  }
  return currentInterpretationId === incomingInterpretationId && currentStep.id === incomingStep.id;
}

function sumStepDurations(steps: readonly AvatarQueueStep[]): number {
  return steps.reduce(
    (total, step) =>
      total + Math.max(0, Math.round(Number.isFinite(step.durationMs) ? step.durationMs : 0)),
    0
  );
}

function clampSpeed(value: number): number {
  return Math.min(2, Math.max(0.5, Number.isFinite(value) ? value : 1));
}

function roundSpeed(value: number): number {
  return Math.round(value * 100) / 100;
}

function stableHash(value: string): string {
  let hash = 0;
  for (let index = 0; index < value.length; index += 1) {
    hash = (hash * 31 + value.charCodeAt(index)) >>> 0;
  }
  return hash.toString(36);
}
