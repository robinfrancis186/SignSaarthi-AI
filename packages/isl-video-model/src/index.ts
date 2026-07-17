import {
  keypointFrameSchema,
  keypointSequenceSchema,
  videoInferenceResponseSchema,
  videoModelArtifactSchema,
  type KeypointFrame,
  type KeypointLandmark,
  type KeypointSequence,
  type ModelMetadata,
  type SignDatasetId,
  type VideoInferenceResponse,
  type VideoModelArtifact
} from "@signsaarthi/shared";

export * from "./temporal.js";

export { videoModelArtifactSchema } from "@signsaarthi/shared";

export type VideoKeypointModel = VideoModelArtifact;

export type TrainVideoKeypointModelOptions = {
  now?: () => string;
  version?: string;
  displayName?: string;
};

export type VideoKeypointEvaluation = {
  accuracy: number;
  correctCount: number;
  sampleCount: number;
  classCount: number;
};

const MODEL_VERSION = "0.1.0";
const EPSILON = 0.000001;
const PART_ORDER: Record<KeypointLandmark["part"], number> = {
  pose: 0,
  left_hand: 1,
  right_hand: 2,
  face: 3,
  unknown: 4
};

export const defaultKeypointSequences: KeypointSequence[] = keypointSequenceSchema.array().parse([
  makeSequence("include_today_train_001", "73. Today", "TODAY", "train", [
    makeFrame(0, [
      ["left_hand", 0, 0.44, 0.52],
      ["left_hand", 1, 0.42, 0.62],
      ["right_hand", 0, 0.58, 0.25],
      ["right_hand", 1, 0.61, 0.35]
    ]),
    makeFrame(80, [
      ["left_hand", 0, 0.44, 0.53],
      ["left_hand", 1, 0.42, 0.63],
      ["right_hand", 0, 0.57, 0.31],
      ["right_hand", 1, 0.6, 0.42]
    ]),
    makeFrame(160, [
      ["left_hand", 0, 0.45, 0.53],
      ["left_hand", 1, 0.43, 0.63],
      ["right_hand", 0, 0.57, 0.38],
      ["right_hand", 1, 0.6, 0.48]
    ])
  ]),
  makeSequence("include_today_val_001", "73. Today", "TODAY", "val", [
    makeFrame(0, [
      ["left_hand", 0, 0.43, 0.52],
      ["left_hand", 1, 0.42, 0.61],
      ["right_hand", 0, 0.59, 0.26],
      ["right_hand", 1, 0.62, 0.36]
    ]),
    makeFrame(80, [
      ["left_hand", 0, 0.44, 0.53],
      ["left_hand", 1, 0.42, 0.62],
      ["right_hand", 0, 0.58, 0.32],
      ["right_hand", 1, 0.61, 0.42]
    ]),
    makeFrame(160, [
      ["left_hand", 0, 0.45, 0.53],
      ["left_hand", 1, 0.43, 0.62],
      ["right_hand", 0, 0.58, 0.39],
      ["right_hand", 1, 0.61, 0.48]
    ])
  ]),
  makeSequence("include_peace_train_001", "12. Peace", "PEACE", "train", [
    makeFrame(0, [
      ["left_hand", 0, 0.38, 0.43],
      ["left_hand", 1, 0.48, 0.49],
      ["right_hand", 0, 0.62, 0.43],
      ["right_hand", 1, 0.52, 0.49]
    ]),
    makeFrame(80, [
      ["left_hand", 0, 0.4, 0.45],
      ["left_hand", 1, 0.5, 0.5],
      ["right_hand", 0, 0.6, 0.45],
      ["right_hand", 1, 0.5, 0.5]
    ]),
    makeFrame(160, [
      ["left_hand", 0, 0.42, 0.47],
      ["left_hand", 1, 0.51, 0.51],
      ["right_hand", 0, 0.58, 0.47],
      ["right_hand", 1, 0.49, 0.51]
    ])
  ]),
  makeSequence("include_peace_val_001", "12. Peace", "PEACE", "val", [
    makeFrame(0, [
      ["left_hand", 0, 0.39, 0.44],
      ["left_hand", 1, 0.48, 0.5],
      ["right_hand", 0, 0.61, 0.44],
      ["right_hand", 1, 0.52, 0.5]
    ]),
    makeFrame(80, [
      ["left_hand", 0, 0.41, 0.45],
      ["left_hand", 1, 0.5, 0.5],
      ["right_hand", 0, 0.59, 0.45],
      ["right_hand", 1, 0.5, 0.5]
    ]),
    makeFrame(160, [
      ["left_hand", 0, 0.43, 0.47],
      ["left_hand", 1, 0.51, 0.51],
      ["right_hand", 0, 0.57, 0.47],
      ["right_hand", 1, 0.49, 0.51]
    ])
  ])
]);

export function normalizeSignLabel(label: string): string {
  return label
    .replace(/\.[a-z0-9]+$/i, "")
    .replace(/^\s*\d+[\s.)_-]*/u, "")
    .replace(/[_-]+/gu, " ")
    .replace(/[^\p{Letter}\p{Number}\s]/gu, " ")
    .replace(/\s+/gu, " ")
    .trim()
    .toLowerCase();
}

export function collectLandmarkKeys(sequences: KeypointSequence[]): string[] {
  const keys = new Set<string>();
  for (const sequence of keypointSequenceSchema.array().parse(sequences)) {
    for (const frame of sequence.frames) {
      for (const landmark of frame.landmarks) {
        keys.add(getLandmarkKey(landmark));
      }
    }
  }

  return [...keys].sort(compareLandmarkKeys);
}

export function vectorizeKeypointSequence(
  sequence: KeypointSequence,
  landmarkKeys: string[]
): number[] {
  const parsed = keypointSequenceSchema.parse(sequence);
  return vectorizeFrames(parsed.frames, landmarkKeys);
}

export function trainVideoKeypointModel(
  sequences: KeypointSequence[],
  options: TrainVideoKeypointModelOptions = {}
): VideoKeypointModel {
  const parsedSequences = keypointSequenceSchema.array().min(1).parse(sequences);
  const trainingSequences = parsedSequences.filter((sequence) => sequence.split === "train");
  const prototypeSource = trainingSequences.length ? trainingSequences : parsedSequences;
  const landmarkKeys = collectLandmarkKeys(prototypeSource);
  const groups = new Map<string, Array<{ sequence: KeypointSequence; feature: number[] }>>();

  for (const sequence of prototypeSource) {
    const normalizedLabel = normalizeSignLabel(sequence.label);
    if (!/\p{Letter}/u.test(normalizedLabel)) {
      throw new RangeError(
        `Sign label "${sequence.label}" has no semantic text after class-number normalization.`
      );
    }
    const current = groups.get(normalizedLabel) ?? [];
    current.push({ sequence, feature: vectorizeKeypointSequence(sequence, landmarkKeys) });
    groups.set(normalizedLabel, current);
  }

  const trainedAt = options.now?.() ?? new Date().toISOString();
  const prototypes = [...groups.entries()]
    .map(([normalizedLabel, rows]) => {
      const first = rows[0];
      if (!first) {
        throw new Error(`No rows available for label ${normalizedLabel}`);
      }
      return {
        label: first.sequence.label,
        normalizedLabel,
        centroid: meanVector(rows.map((row) => row.feature)),
        sampleCount: rows.length
      };
    })
    .sort((left, right) => left.normalizedLabel.localeCompare(right.normalizedLabel));

  const datasetIds = Array.from(new Set(parsedSequences.map((sequence) => sequence.datasetId)));
  const labelMap = prototypes.map((prototype) => {
    const source = prototypeSource.find(
      (sequence) => normalizeSignLabel(sequence.label) === prototype.normalizedLabel
    );
    return {
      label: prototype.label,
      normalizedLabel: prototype.normalizedLabel,
      gloss: source?.gloss ?? prototype.normalizedLabel.toUpperCase()
    };
  });
  const metadata: ModelMetadata = {
    id: `isl-keypoint-${trainedAt.slice(0, 10)}`,
    version: options.version ?? MODEL_VERSION,
    status: "ready",
    engine: "keypoint_centroid",
    trainedAt,
    trainingDataset: {
      primaryDataset: datasetIds.length === 1 ? (datasetIds[0] ?? "curated_mvp") : "mixed",
      displayName: options.displayName ?? getDatasetDisplayName(datasetIds),
      recordCount: parsedSequences.length,
      classCount: prototypes.length,
      citationUrls: getCitationUrls(datasetIds)
    },
    notes: getTrainingNotes(datasetIds)
  };

  const baseModel = videoModelArtifactSchema.parse({
    schemaVersion: 1,
    metadata,
    labelMap,
    landmarkKeys,
    featureSize: landmarkKeys.length * 3,
    prototypes,
    metrics: {
      sampleCount: parsedSequences.length,
      classCount: prototypes.length
    }
  });
  const metrics = {
    sampleCount: parsedSequences.length,
    classCount: prototypes.length,
    ...accuracyBySplit(parsedSequences, baseModel)
  };
  const evaluationNote =
    metrics.testAccuracy === undefined
      ? "No held-out test split was available; this model remains experimental."
      : `Held-out test accuracy is ${(metrics.testAccuracy * 100).toFixed(2)}%; this model remains experimental.`;

  return videoModelArtifactSchema.parse({
    ...baseModel,
    metadata: {
      ...baseModel.metadata,
      notes: [...baseModel.metadata.notes, evaluationNote]
    },
    metrics
  });
}

export function inferVideoKeypoints(
  frames: KeypointFrame[],
  model: VideoKeypointModel,
  topK = 3
): VideoInferenceResponse {
  keypointFrameSchema.array().min(1).parse(frames);
  const parsedModel = videoModelArtifactSchema.parse(model);
  if (!Number.isInteger(topK) || topK < 1 || topK > 5) {
    throw new RangeError("topK must be an integer between 1 and 5.");
  }

  return videoInferenceResponseSchema.parse({
    prediction: null,
    predictions: [],
    model: parsedModel.metadata,
    accepted: false,
    fallback: "caption",
    rawVideoStored: false,
    notes: [
      "The centroid artifact is retained for offline evaluation and is not accepted as user-facing recognition.",
      "No class label is returned from rejected centroid inference; use the caption fallback.",
      "Inference used normalized keypoints only; raw video is not accepted or stored by this endpoint."
    ]
  });
}

export function predictVideoKeypoints(
  frames: KeypointFrame[],
  model: VideoKeypointModel,
  topK = 3
): VideoInferenceResponse["predictions"] {
  const parsedFrames = keypointFrameSchema.array().min(1).parse(frames);
  const parsedModel = videoModelArtifactSchema.parse(model);
  const feature = vectorizeFrames(parsedFrames, parsedModel.landmarkKeys);
  const rows = parsedModel.prototypes
    .map((prototype) => {
      const label = parsedModel.labelMap.find(
        (item) => item.normalizedLabel === prototype.normalizedLabel
      );
      return {
        label: prototype.label,
        normalizedLabel: prototype.normalizedLabel,
        gloss: label?.gloss ?? prototype.normalizedLabel.toUpperCase(),
        distance: euclideanDistance(feature, prototype.centroid)
      };
    })
    .sort((left, right) => left.distance - right.distance);
  const worstDistance = rows.at(-1)?.distance ?? 0;

  return rows.slice(0, Math.max(1, Math.min(5, topK))).map((row) => ({
    ...row,
    confidence: confidenceFromDistance(row.distance, worstDistance)
  }));
}

export function evaluateVideoKeypointModel(
  sequences: KeypointSequence[],
  model: VideoKeypointModel
): VideoKeypointEvaluation {
  const parsedSequences = keypointSequenceSchema.array().parse(sequences);
  const labels = new Set(parsedSequences.map((sequence) => normalizeSignLabel(sequence.label)));
  let correctCount = 0;

  for (const sequence of parsedSequences) {
    const prediction = predictVideoKeypoints(sequence.frames, model, 1)[0];
    if (prediction?.normalizedLabel === normalizeSignLabel(sequence.label)) {
      correctCount += 1;
    }
  }

  return {
    accuracy: parsedSequences.length ? correctCount / parsedSequences.length : 0,
    correctCount,
    sampleCount: parsedSequences.length,
    classCount: labels.size
  };
}

function vectorizeFrames(frames: KeypointFrame[], landmarkKeys: string[]): number[] {
  const featureSize = landmarkKeys.length * 3;
  const totals = Array.from({ length: featureSize }, () => 0);

  for (const frame of frames) {
    const frameVector = vectorizeFrame(frame, landmarkKeys);
    for (let index = 0; index < featureSize; index += 1) {
      totals[index] = (totals[index] ?? 0) + (frameVector[index] ?? 0);
    }
  }

  return totals.map((value) => value / frames.length);
}

function vectorizeFrame(frame: KeypointFrame, landmarkKeys: string[]): number[] {
  const landmarks = new Map(
    frame.landmarks.map((landmark) => [getLandmarkKey(landmark), landmark])
  );
  const present = [...landmarks.values()];
  const centerX = mean(present.map((landmark) => landmark.x));
  const centerY = mean(present.map((landmark) => landmark.y));
  const centerZ = mean(present.map((landmark) => landmark.z ?? 0));
  const minX = Math.min(...present.map((landmark) => landmark.x));
  const maxX = Math.max(...present.map((landmark) => landmark.x));
  const minY = Math.min(...present.map((landmark) => landmark.y));
  const maxY = Math.max(...present.map((landmark) => landmark.y));
  const scale = Math.max(maxX - minX, maxY - minY, EPSILON);

  return landmarkKeys.flatMap((key) => {
    const landmark = landmarks.get(key);
    if (!landmark) {
      return [0, 0, 0];
    }
    return [
      roundFeature((landmark.x - centerX) / scale),
      roundFeature((landmark.y - centerY) / scale),
      roundFeature(((landmark.z ?? 0) - centerZ) / scale)
    ];
  });
}

function makeSequence(
  sampleId: string,
  label: string,
  gloss: string,
  split: KeypointSequence["split"],
  frames: KeypointFrame[]
): KeypointSequence {
  return keypointSequenceSchema.parse({
    sampleId,
    label,
    gloss,
    datasetId: "include",
    split,
    signerHash: "fixture_signer",
    fps: 25,
    durationMs: 240,
    extractorVersion: "mediapipe-hands-blazepose-fixture",
    missingKeypointRatio: 0,
    license: "INCLUDE dataset terms; synthetic keypoint fixture for local tests",
    frames
  });
}

function makeFrame(
  timestampMs: number,
  landmarks: Array<[KeypointLandmark["part"], number, number, number]>
): KeypointFrame {
  return keypointFrameSchema.parse({
    timestampMs,
    landmarks: landmarks.map(([part, index, x, y]) => ({ part, index, x, y, visibility: 0.98 }))
  });
}

function getLandmarkKey(landmark: Pick<KeypointLandmark, "part" | "index">): string {
  return `${landmark.part}:${landmark.index}`;
}

function compareLandmarkKeys(left: string, right: string): number {
  const [leftPart = "unknown", leftIndex = "0"] = left.split(":");
  const [rightPart = "unknown", rightIndex = "0"] = right.split(":");
  const partDifference =
    (PART_ORDER[leftPart as KeypointLandmark["part"]] ?? PART_ORDER.unknown) -
    (PART_ORDER[rightPart as KeypointLandmark["part"]] ?? PART_ORDER.unknown);
  return partDifference || Number(leftIndex) - Number(rightIndex);
}

function meanVector(vectors: number[][]): number[] {
  const first = vectors[0];
  if (!first) {
    return [];
  }
  const totals = Array.from({ length: first.length }, () => 0);
  for (const vector of vectors) {
    for (let index = 0; index < first.length; index += 1) {
      totals[index] = (totals[index] ?? 0) + (vector[index] ?? 0);
    }
  }
  return totals.map((value) => roundFeature(value / vectors.length));
}

function mean(values: number[]): number {
  return values.length ? values.reduce((sum, value) => sum + value, 0) / values.length : 0;
}

function euclideanDistance(left: number[], right: number[]): number {
  const size = Math.max(left.length, right.length);
  let total = 0;
  for (let index = 0; index < size; index += 1) {
    total += ((left[index] ?? 0) - (right[index] ?? 0)) ** 2;
  }
  return Math.sqrt(total);
}

function confidenceFromDistance(distance: number, worstDistance: number): number {
  if (worstDistance <= EPSILON) {
    return roundConfidence(1 / (1 + distance));
  }
  return roundConfidence(Math.max(0.01, Math.min(0.99, 1 - distance / (worstDistance + EPSILON))));
}

function accuracyBySplit(
  sequences: KeypointSequence[],
  model: VideoKeypointModel
): {
  trainAccuracy?: number;
  valAccuracy?: number;
  testAccuracy?: number;
} {
  const entries = [
    ["trainAccuracy", sequences.filter((sequence) => sequence.split === "train")],
    ["valAccuracy", sequences.filter((sequence) => sequence.split === "val")],
    ["testAccuracy", sequences.filter((sequence) => sequence.split === "test")]
  ] as const;
  const metrics: { trainAccuracy?: number; valAccuracy?: number; testAccuracy?: number } = {};

  for (const [key, splitSequences] of entries) {
    if (splitSequences.length) {
      metrics[key] = roundConfidence(evaluateVideoKeypointModel(splitSequences, model).accuracy);
    }
  }

  return metrics;
}

function getDatasetDisplayName(datasetIds: SignDatasetId[]): string {
  if (datasetIds.includes("include") && datasetIds.length === 1) {
    return "INCLUDE keypoint model";
  }
  return `${datasetIds.map((datasetId) => datasetId.toUpperCase()).join(" + ")} keypoint model`;
}

function getTrainingNotes(datasetIds: SignDatasetId[]): string[] {
  const notes = ["Local keypoint prototype classifier for MVP inference and integration testing."];
  if (datasetIds.includes("include")) {
    notes.push("This artifact is trained from MediaPipe-extracted real INCLUDE keypoints.");
  }
  if (datasetIds.includes("isign")) {
    notes.push(
      "This artifact includes iSign keypoints from an accepted Hugging Face dataset license."
    );
  }
  if (!datasetIds.includes("include") && !datasetIds.includes("isign")) {
    notes.push(
      "Replace fixture keypoints with MediaPipe-extracted INCLUDE/iSign keypoints before production evaluation."
    );
  } else if (!datasetIds.includes("isign")) {
    notes.push(
      "iSign media is not part of this artifact; the large gated pose/video archives were not downloaded or redistributed."
    );
  }
  return notes;
}

function getCitationUrls(datasetIds: SignDatasetId[]): string[] {
  const urls = new Set<string>();
  for (const datasetId of datasetIds) {
    if (datasetId === "include") {
      urls.add("https://github.com/AI4Bharat/INCLUDE");
      urls.add("https://huggingface.co/datasets/ai4bharat/INCLUDE");
    }
    if (datasetId === "isign") {
      urls.add("https://huggingface.co/datasets/Exploration-Lab/iSign");
    }
    if (datasetId === "isltranslate") {
      urls.add("https://github.com/Exploration-Lab/ISLTranslate");
    }
  }
  urls.add("https://www.w3.org/WAI/media/av/sign-languages/");
  return [...urls];
}

function roundFeature(value: number): number {
  return Math.round(value * 1_000_000) / 1_000_000;
}

function roundConfidence(value: number): number {
  return Math.round(value * 10_000) / 10_000;
}
