import {
  assessTemporalInputQuality,
  classifyTemporalLogits,
  preprocessTemporalSequence,
  type TemporalKeypointFrame,
  type TemporalLandmark
} from "@signsaarthi/isl-video-model";
import {
  temporalModelArtifactSchema,
  videoInferenceResponseSchema,
  type KeypointFrame,
  type TemporalModelArtifact,
  type VideoInferenceResponse
} from "@signsaarthi/shared";
import * as ort from "onnxruntime-node";
import { existsSync, readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

export type TemporalModelRuntime = {
  artifact: TemporalModelArtifact;
  infer(frames: KeypointFrame[], topK: number): Promise<VideoInferenceResponse>;
};

export type TemporalLogitsRunner = (
  values: Float32Array,
  dimensions: readonly [number, number, number]
) => Promise<readonly number[]>;

export type TemporalModelRuntimeOptions = {
  runLogits?: TemporalLogitsRunner;
};

export function loadTemporalModelRuntime(): TemporalModelRuntime | undefined {
  const configuredPath = process.env["SIGNSAARTHI_TEMPORAL_MODEL_METADATA_PATH"];
  const metadataPaths = [
    configuredPath ? resolve(process.cwd(), configuredPath) : undefined,
    resolve(process.cwd(), "data/models/isl-temporal-model.json"),
    fileURLToPath(new URL("../../../data/models/isl-temporal-model.json", import.meta.url))
  ].filter((candidate): candidate is string => Boolean(candidate));

  for (const metadataPath of metadataPaths) {
    if (!existsSync(metadataPath)) {
      continue;
    }
    const artifact = temporalModelArtifactSchema.parse(
      JSON.parse(readFileSync(metadataPath, "utf8"))
    );
    const configuredModelPath = process.env["SIGNSAARTHI_TEMPORAL_MODEL_PATH"];
    const modelPath = configuredModelPath
      ? resolve(process.cwd(), configuredModelPath)
      : resolve(dirname(metadataPath), artifact.modelFile);
    if (!existsSync(modelPath)) {
      continue;
    }
    return createTemporalModelRuntime(artifact, modelPath);
  }

  return undefined;
}

export function createTemporalModelRuntime(
  artifact: TemporalModelArtifact,
  modelPath: string,
  options: TemporalModelRuntimeOptions = {}
): TemporalModelRuntime {
  let sessionPromise: Promise<ort.InferenceSession> | undefined;
  const getSession = (): Promise<ort.InferenceSession> => {
    sessionPromise ??= ort.InferenceSession.create(modelPath, {
      executionProviders: ["cpu"]
    });
    return sessionPromise;
  };
  const runLogits: TemporalLogitsRunner =
    options.runLogits ??
    (async (values, dimensions) => {
      const session = await getSession();
      const result = await session.run({
        keypoints: new ort.Tensor("float32", values, [...dimensions])
      });
      const logitsTensor = result["logits"];
      if (!logitsTensor) {
        throw new Error("Temporal model did not return a logits tensor.");
      }
      return Array.from(logitsTensor.data as Float32Array);
    });

  return {
    artifact,
    async infer(frames, topK) {
      const confidenceThreshold = artifact.confidencePolicy.deployedThreshold;
      const inputQuality = assessTemporalInputQuality(frames);
      if (!inputQuality.accepted) {
        return videoInferenceResponseSchema.parse({
          prediction: null,
          predictions: [],
          model: artifact.metadata,
          accepted: false,
          fallback: "caption",
          confidenceThreshold,
          rawVideoStored: false,
          notes: [
            `Temporal inference was skipped by the input-quality gate: ${inputQuality.reasons.join(", ")}.`,
            `Observed ${inputQuality.frameCount} frames with ${(inputQuality.supportedLandmarkCoverage * 100).toFixed(1)}% supported-landmark coverage.`,
            "Use the caption fallback; no ONNX class was evaluated for this window.",
            "Raw video and audio are not accepted or stored by this endpoint."
          ]
        });
      }

      const tensorValues = preprocessTemporalSequence(frames.map(toTemporalFrame));
      const flatValues = Float32Array.from(tensorValues.flat());
      const logits = await runLogits(flatValues, [
        1,
        artifact.preprocessing.sequenceLength,
        artifact.preprocessing.inputSize
      ]);
      const labels = artifact.labels.map((label) => label.label);
      const classification = classifyTemporalLogits(logits, labels, {
        topK,
        lowConfidenceThreshold: confidenceThreshold
      });
      const predictions = classification.predictions.map((prediction) => {
        const label = artifact.labels[prediction.index];
        if (!label) {
          throw new Error(`Temporal model label ${prediction.index} is missing.`);
        }
        return {
          label: label.label,
          normalizedLabel: label.normalizedLabel,
          gloss: label.gloss,
          confidence: prediction.probability,
          distance: 1 - prediction.probability
        };
      });
      if (!classification.lowConfidence && predictions.length === 0) {
        throw new Error("Temporal model returned no class predictions.");
      }
      const prediction = classification.prediction === null ? null : (predictions[0] ?? null);

      return videoInferenceResponseSchema.parse({
        prediction,
        predictions,
        model: artifact.metadata,
        accepted: prediction !== null,
        fallback: classification.fallback,
        confidenceThreshold,
        rawVideoStored: false,
        notes: [
          "Inference used a 64-frame temporal transformer over pose and hand keypoints only.",
          classification.lowConfidence
            ? `Confidence is below ${confidenceThreshold.toFixed(2)}; use the caption fallback instead of presenting the class as certain.`
            : `Confidence passed the calibrated ${confidenceThreshold.toFixed(2)} selective threshold.`,
          ...(classification.lowConfidence
            ? ["No class labels are returned for a rejected recognition window."]
            : []),
          "Raw video and audio are not accepted or stored by this endpoint."
        ]
      });
    }
  };
}

function toTemporalFrame(frame: KeypointFrame): TemporalKeypointFrame {
  const pose: Array<TemporalLandmark | null> = Array.from({ length: 25 }, () => null);
  const leftHand: Array<TemporalLandmark | null> = Array.from({ length: 21 }, () => null);
  const rightHand: Array<TemporalLandmark | null> = Array.from({ length: 21 }, () => null);

  for (const landmark of frame.landmarks) {
    const target =
      landmark.part === "pose"
        ? pose
        : landmark.part === "left_hand"
          ? leftHand
          : landmark.part === "right_hand"
            ? rightHand
            : undefined;
    if (!target || landmark.index >= target.length) {
      continue;
    }
    target[landmark.index] = {
      x: landmark.x,
      y: landmark.y,
      ...(landmark.visibility === undefined ? {} : { visibility: landmark.visibility })
    };
  }

  return { pose, left_hand: leftHand, right_hand: rightHand };
}
