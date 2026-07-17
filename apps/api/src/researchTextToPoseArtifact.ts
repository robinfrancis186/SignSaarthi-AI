import type { ModelMetadata } from "@signsaarthi/shared";
import { createHash } from "node:crypto";
import { existsSync, readFileSync, statSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { z } from "zod";

const MAX_METADATA_BYTES = 2 * 1024 * 1024;
const MAX_MODEL_BYTES = 128 * 1024 * 1024;
const MAX_COMPANION_BYTES = 16 * 1024 * 1024;
const sha256Schema = z.string().regex(/^[a-f0-9]{64}$/);

const metricSchema = z
  .object({
    normalized_mpjpe: z.number().finite().nonnegative(),
    velocity_mpjpe: z.number().finite().nonnegative(),
    normalized_dtw: z.number().finite().nonnegative(),
    dtw_sample_count: z.number().int().positive(),
    predicted_hand_motion_energy: z.number().finite().nonnegative(),
    target_hand_motion_energy: z.number().finite().nonnegative(),
    hand_motion_energy_ratio: z.number().finite().nonnegative(),
    non_static_sample_rate: z.number().finite().min(0).max(1),
    sample_count: z.number().int().positive()
  })
  .strict();

const artifactSchema = z.object({
  schemaVersion: z.literal(1),
  model: z.object({
    id: z.string().min(1),
    engine: z.literal("text_conditioned_temporal_transformer"),
    onnxFile: z.string().regex(/^[a-zA-Z0-9._-]+$/),
    onnxSha256: sha256Schema,
    vocabularyFile: z.string().regex(/^[a-zA-Z0-9._-]+$/),
    vocabularySha256: sha256Schema,
    normalizationFile: z.string().regex(/^[a-zA-Z0-9._-]+$/),
    normalizationSha256: sha256Schema,
    requiresAtLeastOneToken: z.literal(true),
    frameCount: z.literal(64),
    pointCount: z.literal(93),
    maxTokens: z.number().int().positive(),
    vocabularySize: z.number().int().positive(),
    hiddenSize: z.number().int().positive(),
    layers: z.number().int().positive(),
    heads: z.number().int().positive(),
    onnxValidation: z.object({
      passed: z.literal(true),
      batchSize: z.number().int().positive(),
      outputShape: z.tuple([z.number().int().positive(), z.literal(64), z.literal(93), z.literal(2)]),
      maximumAbsoluteError: z.number().finite().nonnegative(),
      provider: z.literal("CPUExecutionProvider")
    })
  }),
  trainingDataset: z.object({
    source: z.literal("Exploration-Lab/iSign"),
    license: z.string().min(1),
    recordCount: z.number().int().positive(),
    splits: z.object({
      train: z.number().int().positive(),
      val: z.number().int().positive(),
      test: z.number().int().positive()
    }),
    leakageAudit: z.object({
      passed: z.literal(true)
    }).passthrough(),
    rawArchiveStored: z.literal(false)
  }),
  metrics: z.object({
    validation: metricSchema,
    test: metricSchema,
    meanBaselineValidation: metricSchema,
    meanBaselineTest: metricSchema
  }),
  qualityGate: z.object({
    passed: z.boolean(),
    runtimeEligible: z.literal(false),
    runtimeBlocker: z.string().min(1)
  }).passthrough(),
  expertReview: z.object({
    status: z.literal("pending"),
    requiredForRuntime: z.literal(true),
    statement: z.string().min(1)
  })
});

export type ResearchTextToPoseArtifact = {
  metadata: ModelMetadata;
  metrics: z.infer<typeof artifactSchema>["metrics"] & {
    qualityGate: z.infer<typeof artifactSchema>["qualityGate"];
    expertReview: z.infer<typeof artifactSchema>["expertReview"];
    onnxValidation: z.infer<typeof artifactSchema>["model"]["onnxValidation"];
  };
};

export function loadResearchTextToPoseArtifact(
  configuredPath = process.env["SIGNSAARTHI_ISIGN_TEXT_TO_POSE_METADATA_PATH"]
): ResearchTextToPoseArtifact | undefined {
  const candidatePaths = configuredPath
    ? [resolve(process.cwd(), configuredPath)]
    : [
        resolve(process.cwd(), "data/private/isign-research/text-to-pose/metadata.json"),
        fileURLToPath(
          new URL("../../../data/private/isign-research/text-to-pose/metadata.json", import.meta.url)
        )
      ];
  const metadataPath = candidatePaths.find(existsSync);
  if (!metadataPath) {
    return undefined;
  }

  try {
    if (statSync(metadataPath).size > MAX_METADATA_BYTES) {
      return undefined;
    }
    const payload = artifactSchema.parse(JSON.parse(readFileSync(metadataPath, "utf8")));
    const modelPath = resolve(dirname(metadataPath), payload.model.onnxFile);
    if (!existsSync(modelPath) || statSync(modelPath).size > MAX_MODEL_BYTES) {
      return undefined;
    }
    const digest = createHash("sha256").update(readFileSync(modelPath)).digest("hex");
    if (digest !== payload.model.onnxSha256) {
      return undefined;
    }
    for (const [filename, expectedDigest] of [
      [payload.model.vocabularyFile, payload.model.vocabularySha256],
      [payload.model.normalizationFile, payload.model.normalizationSha256]
    ] as const) {
      const companionPath = resolve(dirname(metadataPath), filename);
      if (!existsSync(companionPath) || statSync(companionPath).size > MAX_COMPANION_BYTES) {
        return undefined;
      }
      const companionDigest = createHash("sha256")
        .update(readFileSync(companionPath))
        .digest("hex");
      if (companionDigest !== expectedDigest) {
        return undefined;
      }
    }

    return {
      metadata: {
        id: payload.model.id,
        version: "1.0.0",
        status: "unavailable",
        engine: "text_to_pose_research",
        trainingDataset: {
          primaryDataset: "isign",
          displayName: "Private gated iSign text-to-pose research corpus",
          recordCount: payload.trainingDataset.recordCount,
          classCount: payload.model.vocabularySize,
          citationUrls: [
            "https://huggingface.co/datasets/Exploration-Lab/iSign",
            "https://aclanthology.org/2024.findings-acl.643/"
          ]
        },
        notes: [
          `Dynamic geometry preview gate ${payload.qualityGate.passed ? "passed" : "failed"}.`,
          payload.qualityGate.runtimeBlocker,
          "The artifact is local-only, noncommercial research output and is not used for live signing."
        ]
      },
      metrics: {
        ...payload.metrics,
        qualityGate: payload.qualityGate,
        expertReview: payload.expertReview,
        onnxValidation: payload.model.onnxValidation
      }
    };
  } catch {
    return undefined;
  }
}
