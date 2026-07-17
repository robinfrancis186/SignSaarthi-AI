import { describe, expect, it } from "vitest";
import {
  defaultKeypointSequences,
  evaluateVideoKeypointModel,
  inferVideoKeypoints,
  normalizeSignLabel,
  trainVideoKeypointModel,
  vectorizeKeypointSequence
} from "./index";

describe("ISL video keypoint model", () => {
  it("normalizes INCLUDE labels into stable gloss labels", () => {
    expect(normalizeSignLabel("73. Today")).toBe("today");
    expect(normalizeSignLabel("12_Peace.mp4")).toBe("peace");
  });

  it("trains a deterministic keypoint classifier and predicts the nearest sign", () => {
    const model = trainVideoKeypointModel(defaultKeypointSequences, {
      now: () => "2026-07-07T00:00:00.000Z"
    });
    const todaySample = defaultKeypointSequences.find(
      (sequence) => sequence.label === "73. Today" && sequence.split === "val"
    );

    expect(todaySample).toBeDefined();
    expect(model.metadata.trainingDataset.primaryDataset).toBe("include");
    expect(model.featureSize).toBe(model.landmarkKeys.length * 3);
    expect(model.metrics?.trainAccuracy).toBe(1);

    const response = inferVideoKeypoints(todaySample?.frames ?? [], model, 2);
    expect(response.rawVideoStored).toBe(false);
    expect(response.prediction).toBeNull();
    expect(response.accepted).toBe(false);
    expect(response.fallback).toBe("caption");
    expect(response.predictions).toEqual([]);
  });

  it("repairs numeric-only gloss artifacts and rejects numeric-only class labels", () => {
    const sequencesWithNumericGloss = defaultKeypointSequences.map((sequence) =>
      sequence.label === "73. Today" ? { ...sequence, gloss: "73" } : sequence
    );
    const model = trainVideoKeypointModel(sequencesWithNumericGloss, {
      now: () => "2026-07-07T00:00:00.000Z"
    });
    const todaySample = sequencesWithNumericGloss.find(
      (sequence) => sequence.label === "73. Today" && sequence.split === "val"
    );
    inferVideoKeypoints(todaySample?.frames ?? [], model, 1);

    expect(model.labelMap.find((label) => label.normalizedLabel === "today")?.gloss).toBe("TODAY");
    expect(() =>
      trainVideoKeypointModel([
        {
          ...defaultKeypointSequences[0]!,
          sampleId: "numeric_only",
          label: "73",
          gloss: "73"
        }
      ])
    ).toThrow(/no semantic text/);
  });

  it("evaluates held-out fixture keypoints and exposes fixed-size vectors", () => {
    const model = trainVideoKeypointModel(defaultKeypointSequences, {
      now: () => "2026-07-07T00:00:00.000Z"
    });
    const validationSamples = defaultKeypointSequences.filter(
      (sequence) => sequence.split === "val"
    );
    const evaluation = evaluateVideoKeypointModel(validationSamples, model);
    const vector = vectorizeKeypointSequence(defaultKeypointSequences[0]!, model.landmarkKeys);

    expect(evaluation.sampleCount).toBe(2);
    expect(evaluation.accuracy).toBe(1);
    expect(vector).toHaveLength(model.featureSize);
  });
});
