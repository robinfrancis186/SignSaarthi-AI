import {
  temporalModelArtifactSchema,
  type KeypointFrame,
  type TemporalModelArtifact
} from "@signsaarthi/shared";
import { describe, expect, it, vi } from "vitest";
import { createTemporalModelRuntime } from "./temporalModel";

describe("temporal model runtime safety", () => {
  it("withholds every class label from a low-confidence result", async () => {
    const runLogits = vi.fn(async () => [0, 0]);
    const runtime = createTemporalModelRuntime(makeArtifact(), "/unused/model.onnx", {
      runLogits
    });

    const response = await runtime.infer(makeValidFrames(), 2);

    expect(runLogits).toHaveBeenCalledOnce();
    expect(response).toMatchObject({
      prediction: null,
      predictions: [],
      accepted: false,
      fallback: "caption"
    });
  });

  it("publishes a class only after it passes the calibrated threshold", async () => {
    const runtime = createTemporalModelRuntime(makeArtifact(), "/unused/model.onnx", {
      runLogits: async () => [8, 0]
    });

    const response = await runtime.infer(makeValidFrames(), 2);

    expect(response).toMatchObject({
      prediction: {
        normalizedLabel: "hello",
        gloss: "HELLO"
      },
      accepted: true,
      fallback: "none"
    });
    expect(response.predictions).toHaveLength(2);
  });

  it.each([
    ["insufficient_frames", makeValidFrames().slice(0, 4)],
    [
      "insufficient_landmark_coverage",
      makeValidFrames().map((frame) => ({ ...frame, landmarks: frame.landmarks.slice(0, 1) }))
    ],
    [
      "unsupported_landmark_parts",
      makeValidFrames().map((frame) => ({
        ...frame,
        landmarks: frame.landmarks.map((landmark) => ({ ...landmark, part: "face" as const }))
      }))
    ],
    ["non_finite_coordinates", replacePoint(makeValidFrames(), { x: Number.NaN })],
    ["implausible_coordinate_scale", replacePoint(makeValidFrames(), { x: 640, y: 360 })],
    [
      "non_monotonic_timestamps",
      makeValidFrames().map((frame, index) => ({
        ...frame,
        timestampMs: index === 5 ? 100 : index * 40
      }))
    ],
    ["motionless_window", makeValidFrames({ frozen: true })]
  ] as const)("returns caption fallback for %s before running ONNX", async (reason, frames) => {
    const runLogits = vi.fn(async () => [8, 0]);
    const runtime = createTemporalModelRuntime(makeArtifact(), "/unused/model.onnx", {
      runLogits
    });

    const response = await runtime.infer(frames, 2);

    expect(runLogits).not.toHaveBeenCalled();
    expect(response).toMatchObject({
      prediction: null,
      predictions: [],
      accepted: false,
      fallback: "caption"
    });
    expect(response.notes.join(" ")).toContain(reason);
  });
});

function makeArtifact(): TemporalModelArtifact {
  const splitMetrics = {
    accuracy: 0.8,
    macro_f1: 0.8,
    top3_accuracy: 1,
    sample_count: 2
  };
  return temporalModelArtifactSchema.parse({
    schemaVersion: 1,
    metadata: {
      id: "temporal-test",
      version: "1.0.0",
      status: "ready",
      engine: "keypoint_transformer",
      trainingDataset: {
        primaryDataset: "include",
        displayName: "INCLUDE test fixture",
        recordCount: 4,
        classCount: 2,
        citationUrls: ["https://huggingface.co/datasets/ai4bharat/INCLUDE"]
      }
    },
    modelFile: "model.onnx",
    modelBytes: 128,
    labels: [
      {
        index: 0,
        label: "1. Hello",
        normalizedLabel: "hello",
        gloss: "1",
        trainSamples: 1,
        valSamples: 1,
        testSamples: 1,
        motionClipId: "include_hello"
      },
      {
        index: 1,
        label: "2. Thank you",
        normalizedLabel: "thank you",
        gloss: "THANK YOU",
        trainSamples: 1,
        valSamples: 1,
        testSamples: 1,
        motionClipId: "include_thank_you"
      }
    ],
    preprocessing: {
      sequenceLength: 64,
      inputSize: 134,
      poseLandmarks: 25,
      handLandmarksPerHand: 21,
      coordinates: ["x", "y"],
      frameWidth: 1920,
      frameHeight: 1080
    },
    metrics: {
      train: splitMetrics,
      val: splitMetrics,
      test: splitMetrics,
      quantizedTest: splitMetrics
    },
    confidencePolicy: {
      deployedThreshold: 0.75,
      fallback: "caption",
      coverage: 0.5,
      acceptedAccuracy: 0.9,
      acceptedCount: 1,
      fallbackCount: 1
    },
    rawVideoStored: false,
    rawAudioStored: false
  });
}

function makeValidFrames(options: { frozen?: boolean } = {}): KeypointFrame[] {
  return Array.from({ length: 12 }, (_, frameIndex) => ({
    timestampMs: frameIndex * 40,
    landmarks: Array.from({ length: 8 }, (_, landmarkIndex) => ({
      part: landmarkIndex < 4 ? ("pose" as const) : ("right_hand" as const),
      index: landmarkIndex < 4 ? landmarkIndex : landmarkIndex - 4,
      x: 0.3 + landmarkIndex * 0.02 + (options.frozen ? 0 : frameIndex * 0.002),
      y: 0.4 + landmarkIndex * 0.01 + (options.frozen ? 0 : frameIndex * 0.001),
      visibility: 0.99
    }))
  }));
}

function replacePoint(
  frames: KeypointFrame[],
  replacement: Partial<KeypointFrame["landmarks"][number]>
): KeypointFrame[] {
  return frames.map((frame, frameIndex) => ({
    ...frame,
    landmarks: frame.landmarks.map((landmark, landmarkIndex) =>
      frameIndex === 0 && landmarkIndex === 0 ? { ...landmark, ...replacement } : landmark
    )
  }));
}
