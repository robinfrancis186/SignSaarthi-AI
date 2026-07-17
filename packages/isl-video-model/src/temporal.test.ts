import { describe, expect, it } from "vitest";
import {
  INCLUDE_FEATURE_COUNT,
  INCLUDE_FRAME_HEIGHT,
  INCLUDE_FRAME_WIDTH,
  INCLUDE_HAND_LANDMARK_COUNT,
  INCLUDE_POSE_INDICES,
  INCLUDE_POSE_LANDMARK_COUNT,
  INCLUDE_POSE_LANDMARKS,
  INCLUDE_TIME_STEPS,
  assessTemporalInputQuality,
  classifyTemporalLogits,
  preprocessTemporalSequence,
  topKSoftmax,
  type TemporalInputFrame,
  type TemporalKeypointFrame,
  type TemporalLandmark
} from "./temporal";

type PartValues = Readonly<Record<number, TemporalLandmark | null>>;

type FrameValues = {
  readonly pose?: PartValues;
  readonly leftHand?: PartValues;
  readonly rightHand?: PartValues;
};

describe("INCLUDE temporal preprocessing", () => {
  it("produces the exact 64 by 134 pixel-scaled feature layout", () => {
    const tensor = preprocessTemporalSequence([
      makeFrame({
        pose: {
          0: landmark(0.5, 0.25),
          24: landmark(0.1, 0.2)
        },
        leftHand: { 0: landmark(0.25, 0.5) },
        rightHand: { 20: landmark(0.75, 1) }
      })
    ]);

    expect(tensor).toHaveLength(INCLUDE_TIME_STEPS);
    expect(tensor.every((frame) => frame.length === INCLUDE_FEATURE_COUNT)).toBe(true);
    expect(tensor[0]?.slice(0, 2)).toEqual([960, 270]);
    expect(tensor[0]?.slice(48, 50)).toEqual([192, 216]);
    expect(tensor[0]?.slice(50, 52)).toEqual([480, 540]);
    expect(tensor[0]?.slice(132, 134)).toEqual([1440, 1080]);
    expect(tensor[INCLUDE_TIME_STEPS - 1]?.every((coordinate) => coordinate === 0)).toBe(true);

    expect(INCLUDE_POSE_LANDMARKS).toHaveLength(INCLUDE_POSE_LANDMARK_COUNT);
    expect(INCLUDE_POSE_LANDMARKS[15]).toEqual({ index: 15, name: "left_wrist" });
    expect(INCLUDE_POSE_INDICES).toEqual(Array.from({ length: 25 }, (_, index) => index));
  });

  it("linearly interpolates gaps and carries the nearest endpoint", () => {
    const frames = Array.from({ length: INCLUDE_TIME_STEPS }, () => makeFrame());
    frames[10] = makeFrame({ pose: { 0: landmark(0.2, 0.3) } });
    frames[12] = makeFrame({ pose: { 0: landmark(0.4, 0.7) } });

    const tensor = preprocessTemporalSequence(frames);

    expect(tensor[0]?.slice(0, 2)).toEqual([0.2 * INCLUDE_FRAME_WIDTH, 0.3 * INCLUDE_FRAME_HEIGHT]);
    expect(tensor[11]?.[0]).toBeCloseTo(0.3 * INCLUDE_FRAME_WIDTH);
    expect(tensor[11]?.[1]).toBeCloseTo(0.5 * INCLUDE_FRAME_HEIGHT);
    expect(tensor[INCLUDE_TIME_STEPS - 1]?.slice(0, 2)).toEqual([
      0.4 * INCLUDE_FRAME_WIDTH,
      0.7 * INCLUDE_FRAME_HEIGHT
    ]);
  });

  it("pads short input and resamples long input while preserving observed endpoints", () => {
    const shortTensor = preprocessTemporalSequence([
      makeFrame({ pose: { 0: landmark(0.1, 0.2) } }),
      makeFrame({ pose: { 0: landmark(0.9, 0.8) } })
    ]);

    expect(shortTensor[0]?.slice(0, 2)).toEqual([192, 216]);
    expect(shortTensor[1]?.slice(0, 2)).toEqual([1728, 864]);
    expect(shortTensor[2]?.every((coordinate) => coordinate === 0)).toBe(true);

    const longTensor = preprocessTemporalSequence(
      Array.from({ length: 65 }, (_, index) =>
        makeFrame({ pose: { 0: landmark(index / 64, index / 128) } })
      )
    );

    expect(longTensor[0]?.slice(0, 2)).toEqual([0, 0]);
    expect(longTensor[INCLUDE_TIME_STEPS - 1]?.slice(0, 2)).toEqual([1920, 540]);
  });

  it("zeros only entirely absent tracks and carries a lone observed track", () => {
    const tensor = preprocessTemporalSequence([
      makeFrame(),
      makeFrame({ rightHand: { 0: landmark(0.25, 0.75) } }),
      makeFrame()
    ]);
    const leftHandStart = INCLUDE_POSE_LANDMARK_COUNT * 2;
    const rightHandStart = leftHandStart + INCLUDE_HAND_LANDMARK_COUNT * 2;

    expect(
      tensor.every((frame) =>
        frame.slice(leftHandStart, rightHandStart).every((coordinate) => coordinate === 0)
      )
    ).toBe(true);
    expect(tensor[0]?.slice(rightHandStart, rightHandStart + 2)).toEqual([480, 810]);
    expect(tensor[2]?.slice(rightHandStart, rightHandStart + 2)).toEqual([480, 810]);
    expect(tensor[3]?.slice(rightHandStart, rightHandStart + 2)).toEqual([0, 0]);
  });

  it("is deterministic and does not mutate input frames", () => {
    const frames = [
      makeFrame({ pose: { 0: landmark(0.12, 0.34) } }),
      makeFrame(),
      makeFrame({ pose: { 0: landmark(0.56, 0.78) } })
    ];
    const snapshot = structuredClone(frames);

    expect(preprocessTemporalSequence(frames)).toEqual(preprocessTemporalSequence(frames));
    expect(frames).toEqual(snapshot);
  });

  it("returns a correctly shaped all-zero tensor for an empty sequence", () => {
    const tensor = preprocessTemporalSequence([]);

    expect(tensor).toHaveLength(INCLUDE_TIME_STEPS);
    expect(tensor.every((frame) => frame.length === INCLUDE_FEATURE_COUNT)).toBe(true);
    expect(tensor.every((frame) => frame.every((coordinate) => coordinate === 0))).toBe(true);
  });
});

describe("temporal logit probabilities", () => {
  it("returns stable top-k probabilities in descending order", () => {
    const ranked = topKSoftmax([0, 2, 1], 2);
    const denominator = Math.exp(0) + Math.exp(2) + Math.exp(1);

    expect(ranked.map((score) => score.index)).toEqual([1, 2]);
    expect(ranked[0]?.probability).toBeCloseTo(Math.exp(2) / denominator);
    expect(ranked[1]?.probability).toBeCloseTo(Math.exp(1) / denominator);
    expect((ranked[0]?.probability ?? 0) > (ranked[1]?.probability ?? 0)).toBe(true);
    expect(topKSoftmax([5, 5, 1], 2).map((score) => score.index)).toEqual([0, 1]);
  });

  it("uses caption fallback instead of accepting a low-confidence class", () => {
    const uncertain = classifyTemporalLogits([0, 0], ["HELLO", "THANK_YOU"], {
      lowConfidenceThreshold: 0.75,
      topK: 2
    });
    const confident = classifyTemporalLogits([8, 0], ["HELLO", "THANK_YOU"], {
      lowConfidenceThreshold: 0.75
    });

    expect(uncertain.predictions).toEqual([]);
    expect(uncertain.prediction).toBeNull();
    expect(uncertain.lowConfidence).toBe(true);
    expect(uncertain.fallback).toBe("caption");
    expect(confident.prediction?.label).toBe("HELLO");
    expect(confident.lowConfidence).toBe(false);
    expect(confident.fallback).toBe("none");
  });
});

describe("temporal input quality gate", () => {
  it("accepts a moving normalized window with sufficient supported coverage", () => {
    const quality = assessTemporalInputQuality(makeQualityFrames());

    expect(quality.accepted).toBe(true);
    expect(quality.reasons).toEqual([]);
    expect(quality.supportedLandmarkCoverage).toBeGreaterThanOrEqual(0.1);
    expect(quality.coveredFrameRatio).toBe(1);
  });

  it.each([
    ["insufficient frames", makeQualityFrames().slice(0, 4), "insufficient_frames"],
    [
      "insufficient coverage",
      makeQualityFrames().map((frame) => ({ ...frame, landmarks: frame.landmarks.slice(0, 1) })),
      "insufficient_landmark_coverage"
    ],
    [
      "unsupported-only parts",
      makeQualityFrames().map((frame) => ({
        ...frame,
        landmarks: frame.landmarks.map((point) => ({ ...point, part: "face" }))
      })),
      "unsupported_landmark_parts"
    ],
    [
      "non-finite coordinates",
      replaceQualityPoint(makeQualityFrames(), { x: Number.NaN }),
      "non_finite_coordinates"
    ],
    [
      "pixel-scaled coordinates",
      replaceQualityPoint(makeQualityFrames(), { x: 960, y: 540 }),
      "implausible_coordinate_scale"
    ],
    [
      "non-monotonic timestamps",
      makeQualityFrames().map((frame, index) => ({
        ...frame,
        timestampMs: index === 4 ? 80 : index * 40
      })),
      "non_monotonic_timestamps"
    ],
    ["a frozen window", makeQualityFrames({ frozen: true }), "motionless_window"]
  ] as const)("rejects %s", (_name, frames, expectedReason) => {
    const quality = assessTemporalInputQuality(frames);

    expect(quality.accepted).toBe(false);
    expect(quality.reasons).toContain(expectedReason);
  });
});

function landmark(x: number, y: number): TemporalLandmark {
  return { x, y, visibility: 0.99 };
}

function makeFrame(values: FrameValues = {}): TemporalKeypointFrame {
  return {
    pose: makePart(INCLUDE_POSE_LANDMARK_COUNT, values.pose),
    left_hand: makePart(INCLUDE_HAND_LANDMARK_COUNT, values.leftHand),
    right_hand: makePart(INCLUDE_HAND_LANDMARK_COUNT, values.rightHand)
  };
}

function makePart(count: number, values: PartValues = {}): Array<TemporalLandmark | null> {
  const part = Array.from({ length: count }, (): TemporalLandmark | null => null);
  for (const [rawIndex, value] of Object.entries(values)) {
    part[Number(rawIndex)] = value;
  }
  return part;
}

function makeQualityFrames(options: { frozen?: boolean } = {}): TemporalInputFrame[] {
  return Array.from({ length: 12 }, (_, frameIndex) => ({
    timestampMs: frameIndex * 40,
    landmarks: Array.from({ length: 8 }, (_, landmarkIndex) => ({
      part: landmarkIndex < 4 ? "pose" : "right_hand",
      index: landmarkIndex < 4 ? landmarkIndex : landmarkIndex - 4,
      x: 0.3 + landmarkIndex * 0.02 + (options.frozen ? 0 : frameIndex * 0.002),
      y: 0.4 + landmarkIndex * 0.01 + (options.frozen ? 0 : frameIndex * 0.001),
      visibility: 0.99
    }))
  }));
}

function replaceQualityPoint(
  frames: TemporalInputFrame[],
  replacement: Partial<TemporalInputFrame["landmarks"][number]>
): TemporalInputFrame[] {
  return frames.map((frame, frameIndex) => ({
    ...frame,
    landmarks: frame.landmarks.map((landmark, landmarkIndex) =>
      frameIndex === 0 && landmarkIndex === 0 ? { ...landmark, ...replacement } : landmark
    )
  }));
}
