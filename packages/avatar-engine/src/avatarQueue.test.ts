import { describe, expect, it } from "vitest";
import {
  createAvatarQueue,
  estimateFingerspellingDurationMs,
  findMotionClip,
  getRollingAvatarQueueOverlap,
  mergeRollingAvatarQueues,
  motionCatalog,
  planRealtimePlayback,
  replayLastSegment
} from "./index";

describe("avatar engine", () => {
  it("creates ordered official-source fingerspelling steps for unsupported words", () => {
    const queue = createAvatarQueue({
      interpretationId: "interp_1",
      speed: 1.25,
      glossSequence: [
        {
          id: "gloss_1",
          token: "alphanova",
          gloss: "ALPHANOVA",
          signAssetId: "sign_alphanova_001",
          isVerified: false,
          fallback: "fingerspell",
          confidence: "low"
        },
        {
          id: "gloss_2",
          token: "Mumbai",
          gloss: "MUMBAI",
          isVerified: false,
          fallback: "fingerspell",
          confidence: "low"
        }
      ]
    });

    expect(queue.steps.map((step) => [step.kind, step.label])).toEqual([
      ["fingerspell", "ALPHANOVA"],
      ["fingerspell", "MUMBAI"]
    ]);
    expect(queue.steps[0]).toMatchObject({
      fingerspellingSource: "islrtc_official",
      durationMs: Math.round(estimateFingerspellingDurationMs("ALPHANOVA") / 1.25)
    });
    expect(queue.steps[0]?.motionClipId).toBeUndefined();
    expect(queue.motionReviewNotice).toBeUndefined();
    expect(queue.honestFallbackText).toContain("official ISLRTC A-Z fingerspelling");
  });

  it("uses timestamp-derived clip timing for every bundled INCLUDE motion", () => {
    expect(motionCatalog).toHaveLength(262);
    for (const motion of motionCatalog) {
      const reconstructedDurationMs = ((motion.frameCount - 1) / motion.fps) * 1000;
      expect(Math.abs(reconstructedDurationMs - motion.durationMs), motion.id).toBeLessThan(0.5);
      expect(motion.playable).toBe(true);
    }
  });

  it("mixes real INCLUDE motion with official fingerspelling without dropping words", () => {
    const queue = createAvatarQueue({
      interpretationId: "interp_mixed_motion",
      speed: 1,
      glossSequence: [
        {
          id: "gloss_computer",
          token: "computer",
          gloss: "COMPUTER",
          isVerified: false,
          fallback: "fingerspell",
          confidence: "medium"
        },
        {
          id: "gloss_alphanova",
          token: "alphanova",
          gloss: "ALPHANOVA",
          isVerified: false,
          fallback: "fingerspell",
          confidence: "low"
        }
      ]
    });

    expect(queue.steps).toEqual([
      expect.objectContaining({
        glossItemId: "gloss_computer",
        kind: "sign",
        motionClipId: "include-computer",
        motionSource: "include_dataset",
        expertReviewed: false
      }),
      expect.objectContaining({
        glossItemId: "gloss_alphanova",
        kind: "fingerspell",
        fingerspellingSource: "islrtc_official"
      })
    ]);
    expect(queue.motionReviewNotice).toContain("not expert-certified ISL");
    expect(queue.honestFallbackText).toContain("official ISLRTC A-Z fingerspelling");
  });

  it("does not collapse punctuation and attach Bed motion metadata to B.Ed", () => {
    expect(findMotionClip("Bed")?.id).toBe("include-bed");
    expect(findMotionClip("B.Ed")).toBeUndefined();
  });

  it("does not call an approved label a sign when no motion clip exists", () => {
    const queue = createAvatarQueue({
      interpretationId: "interp_static_label",
      speed: 1,
      glossSequence: [
        {
          id: "gloss_ai",
          token: "artificial intelligence",
          gloss: "AI",
          signAssetId: "sign_ai_001",
          isVerified: true,
          fallback: "none",
          confidence: "high"
        }
      ]
    });

    expect(queue.steps[0]).toMatchObject({ kind: "caption", label: "AI" });
    expect(queue.steps[0]?.motionClipId).toBeUndefined();
    expect(queue.honestFallbackText).toContain("official ISLRTC A-Z fingerspelling");
  });

  it("preserves every ordered word action without dropping fallback steps", () => {
    const labels = ["ONE", "TWO", "THREE", "FOUR", "FIVE", "SIX", "SEVEN", "EIGHT"];
    const queue = createAvatarQueue({
      interpretationId: "interp_all_words",
      speed: 1,
      glossSequence: labels.map((label, index) => ({
        id: `gloss_${index + 1}`,
        token: label.toLocaleLowerCase(),
        gloss: label,
        isVerified: false,
        fallback: "unknown" as const,
        confidence: "low" as const
      }))
    });

    expect(queue.steps.map((step) => step.label)).toEqual(labels);
    expect(queue.steps.map((step) => step.glossItemId)).toEqual(
      labels.map((_, index) => `gloss_${index + 1}`)
    );
  });

  it("creates deterministic segment-specific step identities", () => {
    const makeQueue = (sourceSegmentId: string) =>
      createAvatarQueue({
        interpretationId: "interp_stable",
        speed: 1,
        now: () => "2026-07-15T00:00:00.000Z",
        glossSequence: [
          {
            id: "gloss_go",
            token: "go",
            gloss: "GO",
            isVerified: false,
            fallback: "fingerspell",
            confidence: "low",
            sourceSegmentId,
            planningActionId: "action_go"
          }
        ]
      });

    const first = makeQueue("segment_1");
    const repeated = makeQueue("segment_1");
    const nextSegment = makeQueue("segment_2");

    expect(repeated.steps[0]?.id).toBe(first.steps[0]?.id);
    expect(repeated.id).toBe(first.id);
    expect(nextSegment.steps[0]?.id).not.toBe(first.steps[0]?.id);
  });

  it("dedupes only a true stable-ID rolling overlap and preserves repeated labels", () => {
    const first = createAvatarQueue({
      interpretationId: "interp_first",
      speed: 1,
      glossSequence: [
        plannedFallback("segment_roll", "action_go_1", "gloss_go_1", "go"),
        plannedFallback("segment_roll", "action_go_2", "gloss_go_2", "go")
      ]
    });
    const rolling = createAvatarQueue({
      interpretationId: "interp_extended",
      speed: 1,
      glossSequence: [
        plannedFallback("segment_roll", "action_go_2", "gloss_go_2", "go"),
        plannedFallback("segment_roll", "action_go_3", "gloss_go_3", "go")
      ]
    });
    const nextSegment = createAvatarQueue({
      interpretationId: "interp_next",
      speed: 1,
      glossSequence: [
        plannedFallback("segment_next", "action_go_1", "gloss_go_1", "go"),
        plannedFallback("segment_next", "action_go_2", "gloss_go_2", "go")
      ]
    });

    expect(getRollingAvatarQueueOverlap(first, rolling)).toBe(1);
    const mergedRolling = mergeRollingAvatarQueues(first, rolling);
    expect(mergedRolling.steps.map((step) => step.label)).toEqual(["GO", "GO", "GO"]);
    expect(getRollingAvatarQueueOverlap(mergedRolling, nextSegment)).toBe(0);
    expect(mergeRollingAvatarQueues(mergedRolling, nextSegment).steps).toHaveLength(5);
  });

  it("bounds the realtime playback window without silently accelerating signs", () => {
    const queue = createAvatarQueue({
      interpretationId: "interp_backlog",
      speed: 1,
      glossSequence: Array.from({ length: 20 }, (_, index) =>
        plannedFallback(
          "segment_backlog",
          `action_${index}`,
          `gloss_${index}`,
          index === 0 ? "today" : `word${index}`
        )
      )
    });
    const plan = planRealtimePlayback(queue.steps, {
      maxPendingSteps: 5,
      maxPendingMs: 4_000,
      catchUpThresholdMs: 2_000,
      historyStepCount: queue.steps.length
    });

    expect(queue.steps).toHaveLength(20);
    expect(queue.steps[0]).toMatchObject({ label: "TODAY", motionClipId: "include-today" });
    expect(plan.playbackSteps.length).toBeLessThanOrEqual(5);
    expect(plan.playbackSteps.map((step) => step.id)).toEqual(
      queue.steps.slice(0, plan.playbackSteps.length).map((step) => step.id)
    );
    expect(plan.backlog).toMatchObject({
      status: "backlog",
      historyStepCount: 20,
      deferredStepCount: 20 - plan.playbackSteps.length,
      catchUpSpeed: 1
    });
  });

  it("replays the most recent avatar queue without mutating the original", () => {
    const queue = createAvatarQueue({
      interpretationId: "interp_1",
      speed: 1,
      glossSequence: []
    });

    const replay = replayLastSegment(queue);

    expect(replay.id).not.toBe(queue.id);
    expect(replay.replayOfQueueId).toBe(queue.id);
  });
});

function plannedFallback(
  sourceSegmentId: string,
  planningActionId: string,
  id: string,
  token: string
) {
  return {
    id,
    token,
    gloss: token.toLocaleUpperCase(),
    isVerified: false,
    fallback: "unknown" as const,
    confidence: "low" as const,
    sourceSegmentId,
    planningActionId
  };
}
