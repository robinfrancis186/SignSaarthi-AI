import { createHash } from "node:crypto";
import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import { loadResearchTextToPoseArtifact } from "./researchTextToPoseArtifact";

const temporaryDirectories: string[] = [];

afterEach(() => {
  for (const directory of temporaryDirectories.splice(0)) {
    rmSync(directory, { recursive: true, force: true });
  }
});

describe("private iSign text-to-pose artifact", () => {
  it("loads only a hash-matched, runtime-blocked research artifact", () => {
    const directory = mkdtempSync(join(tmpdir(), "signsaarthi-text-to-pose-"));
    temporaryDirectories.push(directory);
    const model = Buffer.from("bounded local research model");
    const vocabulary = Buffer.from('{"<pad>":0,"<unk>":1}');
    const normalization = Buffer.from("bounded normalization arrays");
    writeFileSync(join(directory, "model.onnx"), model);
    writeFileSync(join(directory, "vocabulary.json"), vocabulary);
    writeFileSync(join(directory, "normalization.npz"), normalization);
    const metadataPath = join(directory, "metadata.json");
    writeFileSync(
      metadataPath,
      JSON.stringify(
        createFixture(
          createHash("sha256").update(model).digest("hex"),
          createHash("sha256").update(vocabulary).digest("hex"),
          createHash("sha256").update(normalization).digest("hex")
        )
      )
    );

    const artifact = loadResearchTextToPoseArtifact(metadataPath);

    expect(artifact?.metadata).toMatchObject({
      status: "unavailable",
      engine: "text_to_pose_research",
      trainingDataset: { recordCount: 10_000, classCount: 1_200 }
    });
    expect(artifact?.metrics.qualityGate.runtimeEligible).toBe(false);
  });

  it("rejects a model whose bytes do not match the recorded digest", () => {
    const directory = mkdtempSync(join(tmpdir(), "signsaarthi-text-to-pose-"));
    temporaryDirectories.push(directory);
    writeFileSync(join(directory, "model.onnx"), "changed bytes");
    writeFileSync(join(directory, "vocabulary.json"), "{}");
    writeFileSync(join(directory, "normalization.npz"), "normalization");
    const metadataPath = join(directory, "metadata.json");
    writeFileSync(
      metadataPath,
      JSON.stringify(createFixture("0".repeat(64), "0".repeat(64), "0".repeat(64)))
    );

    expect(loadResearchTextToPoseArtifact(metadataPath)).toBeUndefined();
  });
});

function createFixture(
  onnxSha256: string,
  vocabularySha256: string,
  normalizationSha256: string
): object {
  const metric = {
    normalized_mpjpe: 0.4,
    velocity_mpjpe: 0.2,
    normalized_dtw: 0.3,
    dtw_sample_count: 128,
    predicted_hand_motion_energy: 0.2,
    target_hand_motion_energy: 0.25,
    hand_motion_energy_ratio: 0.8,
    non_static_sample_rate: 0.95,
    sample_count: 1_000
  };
  return {
    schemaVersion: 1,
    model: {
      id: "signsaarthi-isign-text-to-pose-research-v1",
      engine: "text_conditioned_temporal_transformer",
      onnxFile: "model.onnx",
      onnxSha256,
      vocabularyFile: "vocabulary.json",
      vocabularySha256,
      normalizationFile: "normalization.npz",
      normalizationSha256,
      requiresAtLeastOneToken: true,
      frameCount: 64,
      pointCount: 93,
      maxTokens: 20,
      vocabularySize: 1_200,
      hiddenSize: 128,
      layers: 2,
      heads: 4,
      onnxValidation: {
        passed: true,
        batchSize: 2,
        outputShape: [2, 64, 93, 2],
        maximumAbsoluteError: 0.000001,
        provider: "CPUExecutionProvider"
      }
    },
    trainingDataset: {
      source: "Exploration-Lab/iSign",
      license: "gated research terms",
      recordCount: 10_000,
      splits: { train: 8_000, val: 1_000, test: 1_000 },
      leakageAudit: { passed: true },
      rawArchiveStored: false
    },
    metrics: {
      validation: metric,
      test: metric,
      meanBaselineValidation: { ...metric, normalized_mpjpe: 0.8 },
      meanBaselineTest: { ...metric, normalized_mpjpe: 0.8 }
    },
    qualityGate: {
      passed: true,
      runtimeEligible: false,
      runtimeBlocker: "Deaf/ISL expert sentence-level validation has not been completed."
    },
    expertReview: {
      status: "pending",
      requiredForRuntime: true,
      statement: "No automated metric can certify linguistic ISL correctness."
    }
  };
}
