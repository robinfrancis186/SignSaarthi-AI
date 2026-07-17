export const INCLUDE_TIME_STEPS = 64;
export const INCLUDE_FEATURE_COUNT = 134;
export const INCLUDE_FRAME_WIDTH = 1920;
export const INCLUDE_FRAME_HEIGHT = 1080;
export const INCLUDE_POSE_LANDMARK_COUNT = 25;
export const INCLUDE_HAND_LANDMARK_COUNT = 21;
export const INCLUDE_INPUT_SHAPE = [INCLUDE_TIME_STEPS, INCLUDE_FEATURE_COUNT] as const;

/**
 * MediaPipe Pose indices used by the INCLUDE upper-body keypoint pipeline.
 * The feature tensor uses these entries in order, followed by left-hand 0..20
 * and right-hand 0..20, with each landmark represented as adjacent x/y values.
 */
export const INCLUDE_POSE_LANDMARKS = [
  { index: 0, name: "nose" },
  { index: 1, name: "left_eye_inner" },
  { index: 2, name: "left_eye" },
  { index: 3, name: "left_eye_outer" },
  { index: 4, name: "right_eye_inner" },
  { index: 5, name: "right_eye" },
  { index: 6, name: "right_eye_outer" },
  { index: 7, name: "left_ear" },
  { index: 8, name: "right_ear" },
  { index: 9, name: "mouth_left" },
  { index: 10, name: "mouth_right" },
  { index: 11, name: "left_shoulder" },
  { index: 12, name: "right_shoulder" },
  { index: 13, name: "left_elbow" },
  { index: 14, name: "right_elbow" },
  { index: 15, name: "left_wrist" },
  { index: 16, name: "right_wrist" },
  { index: 17, name: "left_pinky" },
  { index: 18, name: "right_pinky" },
  { index: 19, name: "left_index" },
  { index: 20, name: "right_index" },
  { index: 21, name: "left_thumb" },
  { index: 22, name: "right_thumb" },
  { index: 23, name: "left_hip" },
  { index: 24, name: "right_hip" }
] as const;

export const INCLUDE_POSE_INDICES: readonly number[] = Object.freeze(
  INCLUDE_POSE_LANDMARKS.map((landmark) => landmark.index)
);

export const DEFAULT_TEMPORAL_TOP_K = 3;
export const DEFAULT_LOW_CONFIDENCE_THRESHOLD = 0.5;
export const MIN_TEMPORAL_FRAME_COUNT = 8;
export const MIN_TEMPORAL_LANDMARK_COVERAGE = 0.1;
export const MIN_TEMPORAL_COVERED_FRAME_RATIO = 0.75;
export const MIN_TEMPORAL_LANDMARKS_PER_FRAME = 4;
export const MIN_TEMPORAL_MOTION_RANGE = 0.005;
export const MIN_TEMPORAL_NORMALIZED_COORDINATE = -0.25;
export const MAX_TEMPORAL_NORMALIZED_COORDINATE = 1.25;

export type TemporalLandmark = {
  readonly x: number | null;
  readonly y: number | null;
  readonly visibility?: number | null;
};

export type TemporalKeypointFrame = {
  readonly pose: readonly (TemporalLandmark | null)[];
  readonly left_hand: readonly (TemporalLandmark | null)[];
  readonly right_hand: readonly (TemporalLandmark | null)[];
};

export type TemporalInputLandmark = {
  readonly part: string;
  readonly index: number;
  readonly x: number;
  readonly y: number;
  readonly z?: number | undefined;
  readonly visibility?: number | undefined;
};

export type TemporalInputFrame = {
  readonly timestampMs?: number | undefined;
  readonly landmarks: readonly TemporalInputLandmark[];
};

export type TemporalInputQualityRejection =
  | "insufficient_frames"
  | "insufficient_landmark_coverage"
  | "unsupported_landmark_parts"
  | "non_finite_coordinates"
  | "implausible_coordinate_scale"
  | "non_monotonic_timestamps"
  | "motionless_window";

export type TemporalInputQuality = {
  readonly accepted: boolean;
  readonly reasons: TemporalInputQualityRejection[];
  readonly frameCount: number;
  readonly supportedLandmarkCoverage: number;
  readonly coveredFrameRatio: number;
  readonly motionRange: number;
};

export type SoftmaxScore = {
  readonly index: number;
  readonly probability: number;
};

export type TemporalClassPrediction = SoftmaxScore & {
  readonly label: string;
};

export type TemporalClassificationOptions = {
  readonly topK?: number;
  readonly lowConfidenceThreshold?: number;
};

export type TemporalClassification = {
  readonly predictions: TemporalClassPrediction[];
  readonly prediction: TemporalClassPrediction | null;
  readonly lowConfidence: boolean;
  readonly fallback: "none" | "caption";
};

type KeypointPart = keyof TemporalKeypointFrame;

const FEATURE_PARTS: ReadonlyArray<{ readonly part: KeypointPart; readonly count: number }> = [
  { part: "pose", count: INCLUDE_POSE_LANDMARK_COUNT },
  { part: "left_hand", count: INCLUDE_HAND_LANDMARK_COUNT },
  { part: "right_hand", count: INCLUDE_HAND_LANDMARK_COUNT }
];

const SUPPORTED_PART_COUNTS: Readonly<Record<KeypointPart, number>> = {
  pose: INCLUDE_POSE_LANDMARK_COUNT,
  left_hand: INCLUDE_HAND_LANDMARK_COUNT,
  right_hand: INCLUDE_HAND_LANDMARK_COUNT
};

const SUPPORTED_LANDMARKS_PER_FRAME = FEATURE_PARTS.reduce((total, part) => total + part.count, 0);

/**
 * Rejects malformed or out-of-distribution normalized keypoint windows before
 * they reach temporal preprocessing or ONNX inference.
 */
export function assessTemporalInputQuality(
  frames: readonly TemporalInputFrame[]
): TemporalInputQuality {
  const reasons = new Set<TemporalInputQualityRejection>();
  const tracks = new Map<string, { minX: number; maxX: number; minY: number; maxY: number }>();
  let inputLandmarkCount = 0;
  let structurallySupportedLandmarkCount = 0;
  let supportedLandmarkCount = 0;
  let coveredFrameCount = 0;

  if (frames.length < MIN_TEMPORAL_FRAME_COUNT) {
    reasons.add("insufficient_frames");
  }

  for (const frame of frames) {
    const supportedInFrame = new Map<string, TemporalInputLandmark>();
    inputLandmarkCount += frame.landmarks.length;

    for (const landmark of frame.landmarks) {
      const key = getSupportedLandmarkKey(landmark);
      if (key !== null) {
        structurallySupportedLandmarkCount += 1;
      }
      if (
        !Number.isFinite(landmark.x) ||
        !Number.isFinite(landmark.y) ||
        (landmark.z !== undefined && !Number.isFinite(landmark.z)) ||
        (landmark.visibility !== undefined && !Number.isFinite(landmark.visibility))
      ) {
        reasons.add("non_finite_coordinates");
        continue;
      }
      if (
        landmark.x < MIN_TEMPORAL_NORMALIZED_COORDINATE ||
        landmark.x > MAX_TEMPORAL_NORMALIZED_COORDINATE ||
        landmark.y < MIN_TEMPORAL_NORMALIZED_COORDINATE ||
        landmark.y > MAX_TEMPORAL_NORMALIZED_COORDINATE
      ) {
        reasons.add("implausible_coordinate_scale");
        continue;
      }

      if (key !== null) {
        supportedInFrame.set(key, landmark);
      }
    }

    supportedLandmarkCount += supportedInFrame.size;
    if (supportedInFrame.size >= MIN_TEMPORAL_LANDMARKS_PER_FRAME) {
      coveredFrameCount += 1;
    }

    for (const [key, landmark] of supportedInFrame) {
      const track = tracks.get(key);
      if (track) {
        track.minX = Math.min(track.minX, landmark.x);
        track.maxX = Math.max(track.maxX, landmark.x);
        track.minY = Math.min(track.minY, landmark.y);
        track.maxY = Math.max(track.maxY, landmark.y);
      } else {
        tracks.set(key, {
          minX: landmark.x,
          maxX: landmark.x,
          minY: landmark.y,
          maxY: landmark.y
        });
      }
    }
  }

  const supportedLandmarkCoverage = frames.length
    ? supportedLandmarkCount / (frames.length * SUPPORTED_LANDMARKS_PER_FRAME)
    : 0;
  const coveredFrameRatio = frames.length ? coveredFrameCount / frames.length : 0;
  if (
    supportedLandmarkCoverage < MIN_TEMPORAL_LANDMARK_COVERAGE ||
    coveredFrameRatio < MIN_TEMPORAL_COVERED_FRAME_RATIO
  ) {
    reasons.add("insufficient_landmark_coverage");
  }
  if (inputLandmarkCount > 0 && structurallySupportedLandmarkCount === 0) {
    reasons.add("unsupported_landmark_parts");
  }
  if (!hasMonotonicTimestamps(frames)) {
    reasons.add("non_monotonic_timestamps");
  }

  const motionRange = Math.max(
    0,
    ...[...tracks.values()].map((track) =>
      Math.hypot(track.maxX - track.minX, track.maxY - track.minY)
    )
  );
  if (supportedLandmarkCount > 0 && motionRange < MIN_TEMPORAL_MOTION_RANGE) {
    reasons.add("motionless_window");
  }

  return {
    accepted: reasons.size === 0,
    reasons: [...reasons],
    frameCount: frames.length,
    supportedLandmarkCoverage,
    coveredFrameRatio,
    motionRange
  };
}

/**
 * Builds the INCLUDE transformer tensor in this exact feature order:
 * pose[0..24] x/y, left_hand[0..20] x/y, right_hand[0..20] x/y.
 * Missing x and y tracks are interpolated independently before resampling.
 */
export function preprocessTemporalSequence(frames: readonly TemporalKeypointFrame[]): number[][] {
  if (frames.length === 0) {
    return makeZeroTensor();
  }

  const flattenedFrames = frames.map(flattenFrame);
  const featureTracks = Array.from({ length: INCLUDE_FEATURE_COUNT }, (_, featureIndex) => {
    const normalizedTrack = flattenedFrames.map((frame) => frame[featureIndex] ?? null);
    const pixelScale = featureIndex % 2 === 0 ? INCLUDE_FRAME_WIDTH : INCLUDE_FRAME_HEIGHT;
    return interpolateTrack(normalizedTrack).map((value) => value * pixelScale);
  });

  if (frames.length > INCLUDE_TIME_STEPS) {
    return resampleTracks(featureTracks, frames.length);
  }

  const observedFrames = Array.from({ length: frames.length }, (_, frameIndex) =>
    featureTracks.map((track) => track[frameIndex] ?? 0)
  );
  return [
    ...observedFrames,
    ...Array.from({ length: INCLUDE_TIME_STEPS - observedFrames.length }, () =>
      Array.from({ length: INCLUDE_FEATURE_COUNT }, () => 0)
    )
  ];
}

/** Returns stable full-softmax probabilities for the highest-ranked logits. */
export function topKSoftmax(
  logits: readonly number[],
  topK = DEFAULT_TEMPORAL_TOP_K
): SoftmaxScore[] {
  if (!Number.isInteger(topK) || topK <= 0) {
    throw new RangeError("topK must be a positive integer.");
  }
  if (logits.length === 0) {
    return [];
  }

  let maximum = Number.NEGATIVE_INFINITY;
  for (const [index, logit] of logits.entries()) {
    if (!Number.isFinite(logit)) {
      throw new RangeError(`Logit at index ${index} must be finite.`);
    }
    maximum = Math.max(maximum, logit);
  }

  const exponentials = logits.map((logit) => Math.exp(logit - maximum));
  const denominator = exponentials.reduce((total, value) => total + value, 0);

  return exponentials
    .map((value, index) => ({ index, probability: value / denominator }))
    .sort((left, right) => right.probability - left.probability || left.index - right.index)
    .slice(0, Math.min(topK, logits.length));
}

/** Ranks model logits and withholds every class label when confidence is low. */
export function classifyTemporalLogits(
  logits: readonly number[],
  labels: readonly string[],
  options: TemporalClassificationOptions = {}
): TemporalClassification {
  if (logits.length !== labels.length) {
    throw new RangeError("Logits and labels must have the same length.");
  }

  const threshold = options.lowConfidenceThreshold ?? DEFAULT_LOW_CONFIDENCE_THRESHOLD;
  if (!Number.isFinite(threshold) || threshold < 0 || threshold > 1) {
    throw new RangeError("lowConfidenceThreshold must be between 0 and 1.");
  }

  const predictions = topKSoftmax(logits, options.topK ?? DEFAULT_TEMPORAL_TOP_K).map((score) => {
    const label = labels[score.index];
    if (label === undefined) {
      throw new RangeError(`Missing label for logit index ${score.index}.`);
    }
    return { ...score, label };
  });
  const candidate = predictions[0] ?? null;
  const lowConfidence = candidate === null || candidate.probability < threshold;

  return {
    predictions: lowConfidence ? [] : predictions,
    prediction: lowConfidence ? null : candidate,
    lowConfidence,
    fallback: lowConfidence ? "caption" : "none"
  };
}

function flattenFrame(frame: TemporalKeypointFrame): Array<number | null> {
  const features: Array<number | null> = [];

  for (const { part, count } of FEATURE_PARTS) {
    const landmarks = frame[part];
    for (let index = 0; index < count; index += 1) {
      const landmark = landmarks[index] ?? null;
      features.push(finiteOrNull(landmark?.x), finiteOrNull(landmark?.y));
    }
  }

  return features;
}

function getSupportedLandmarkKey(landmark: TemporalInputLandmark): string | null {
  if (!Object.hasOwn(SUPPORTED_PART_COUNTS, landmark.part)) {
    return null;
  }
  const part = landmark.part as KeypointPart;
  const count = SUPPORTED_PART_COUNTS[part];
  if (!Number.isInteger(landmark.index) || landmark.index < 0 || landmark.index >= count) {
    return null;
  }
  return `${part}:${landmark.index}`;
}

function hasMonotonicTimestamps(frames: readonly TemporalInputFrame[]): boolean {
  const hasTimestamps = frames.some((frame) => frame.timestampMs !== undefined);
  if (!hasTimestamps) {
    return true;
  }

  let previous = Number.NEGATIVE_INFINITY;
  for (const frame of frames) {
    const timestamp = frame.timestampMs;
    if (timestamp === undefined || !Number.isFinite(timestamp) || timestamp <= previous) {
      return false;
    }
    previous = timestamp;
  }
  return true;
}

function finiteOrNull(value: number | null | undefined): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function interpolateTrack(values: readonly (number | null)[]): number[] {
  const interpolated = Array.from({ length: values.length }, () => 0);
  const firstKnownIndex = values.findIndex((value) => value !== null);
  if (firstKnownIndex === -1) {
    return interpolated;
  }

  const firstKnownValue = values[firstKnownIndex];
  if (firstKnownValue === null || firstKnownValue === undefined) {
    return interpolated;
  }

  interpolated.fill(firstKnownValue, 0, firstKnownIndex + 1);
  let previousKnownIndex = firstKnownIndex;
  let previousKnownValue = firstKnownValue;

  for (let index = firstKnownIndex + 1; index < values.length; index += 1) {
    const value = values[index];
    if (value === null || value === undefined) {
      continue;
    }

    const span = index - previousKnownIndex;
    for (let fillIndex = previousKnownIndex + 1; fillIndex < index; fillIndex += 1) {
      const ratio = (fillIndex - previousKnownIndex) / span;
      interpolated[fillIndex] = previousKnownValue + (value - previousKnownValue) * ratio;
    }
    interpolated[index] = value;
    previousKnownIndex = index;
    previousKnownValue = value;
  }

  interpolated.fill(previousKnownValue, previousKnownIndex + 1);
  return interpolated;
}

function resampleTracks(featureTracks: readonly number[][], sourceLength: number): number[][] {
  const lastSourceIndex = sourceLength - 1;
  const lastTargetIndex = INCLUDE_TIME_STEPS - 1;

  return Array.from({ length: INCLUDE_TIME_STEPS }, (_, targetIndex) => {
    if (targetIndex === 0) {
      return featureTracks.map((track) => track[0] ?? 0);
    }
    if (targetIndex === lastTargetIndex) {
      return featureTracks.map((track) => track[lastSourceIndex] ?? 0);
    }

    const sourcePosition = (targetIndex * lastSourceIndex) / lastTargetIndex;
    const leftIndex = Math.floor(sourcePosition);
    const rightIndex = Math.ceil(sourcePosition);
    const ratio = sourcePosition - leftIndex;

    return featureTracks.map((track) => {
      const left = track[leftIndex] ?? 0;
      const right = track[rightIndex] ?? left;
      return left + (right - left) * ratio;
    });
  });
}

function makeZeroTensor(): number[][] {
  return Array.from({ length: INCLUDE_TIME_STEPS }, () =>
    Array.from({ length: INCLUDE_FEATURE_COUNT }, () => 0)
  );
}
