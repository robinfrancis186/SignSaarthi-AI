import type { MeaningChunker } from "@signsaarthi/isl-engine";
import type { ModelMetadata } from "@signsaarthi/shared";
import { existsSync, readFileSync, statSync } from "node:fs";
import { resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { z } from "zod";

const MAX_ARTIFACT_BYTES = 2 * 1024 * 1024;
const MAX_CHUNK_WORDS = 12;
const MIN_CHUNK_WORDS = 6;

const boundaryFeatureSchema = z.object({
  boundaryAfterProbability: z.number().min(0).max(1),
  support: z.number().int().positive(),
  videoSupport: z.number().int().positive()
});

const plannerArtifactSchema = z.object({
  schemaVersion: z.literal(1),
  artifactId: z.literal("isign-sentence-planner.private"),
  visibility: z.literal("private"),
  claims: z.object({
    isIslGrammarModel: z.literal(false),
    isSignLanguageTranslationModel: z.literal(false),
    isText2PoseModel: z.literal(false),
    isPoseOrMotionModel: z.literal(false)
  }).passthrough(),
  distribution: z.object({
    publicHuggingFaceUpload: z.literal(false),
    containsSourceMedia: z.literal(false),
    containsPoseOrLandmarkData: z.literal(false)
  }).passthrough(),
  privacy: z.object({
    rawSentencesIncluded: z.literal(false),
    sourceIdentifiersIncluded: z.literal(false),
    videoIdentifiersIncluded: z.literal(false)
  }).passthrough(),
  training: z.object({
    retainedUniqueCaptionRows: z.number().int().positive(),
    leakageChecks: z.object({
      videoGroupOverlapAcrossSplits: z.literal(0),
      exactNormalizedCaptionOverlapAcrossRetainedSplits: z.literal(0),
      status: z.literal("passed")
    })
  }).passthrough(),
  evaluation: z.object({
    heldOut: z.object({
      validation: z.object({
        lexicalTokenCoverage: z.number().min(0).max(1),
        exportedBoundaryFeatureCoverage: z.number().min(0).max(1)
      }).passthrough(),
      test: z.object({
        lexicalTokenCoverage: z.number().min(0).max(1),
        exportedBoundaryFeatureCoverage: z.number().min(0).max(1)
      }).passthrough()
    })
  }).passthrough(),
  chunkBoundaryStatistics: z.object({
    phrases: z.array(
      boundaryFeatureSchema.extend({ tokens: z.array(z.string().min(1)).min(2).max(4) })
    ).max(1024),
    tokens: z.array(boundaryFeatureSchema.extend({ token: z.string().min(1) })).max(1024)
  }).passthrough()
});

type BoundaryFeature = {
  probability: number;
  videoSupport: number;
};

export type ResearchSentencePlanner = {
  metadata: ModelMetadata;
  splitMeaningChunks: MeaningChunker;
};

export function loadResearchSentencePlanner(
  configuredPath = process.env["SIGNSAARTHI_ISIGN_SENTENCE_PLANNER_PATH"]
): ResearchSentencePlanner | undefined {
  const candidatePaths = configuredPath
    ? [resolve(process.cwd(), configuredPath)]
    : [
        resolve(process.cwd(), "data/models/isl-sentence-planner.private.json"),
        fileURLToPath(
          new URL("../../../data/models/isl-sentence-planner.private.json", import.meta.url)
        )
      ];
  const artifactPath = candidatePaths.find(existsSync);
  if (!artifactPath) {
    return undefined;
  }

  try {
    if (statSync(artifactPath).size > MAX_ARTIFACT_BYTES) {
      return undefined;
    }
    const artifact = plannerArtifactSchema.parse(JSON.parse(readFileSync(artifactPath, "utf8")));
    const phrases = new Map<string, BoundaryFeature>(
      artifact.chunkBoundaryStatistics.phrases.map((feature) => [
        feature.tokens.join(" "),
        { probability: feature.boundaryAfterProbability, videoSupport: feature.videoSupport }
      ])
    );
    const tokens = new Map<string, BoundaryFeature>(
      artifact.chunkBoundaryStatistics.tokens.map((feature) => [
        feature.token,
        { probability: feature.boundaryAfterProbability, videoSupport: feature.videoSupport }
      ])
    );
    const heldOut = artifact.evaluation.heldOut;
    return {
      metadata: {
        id: artifact.artifactId,
        version: "1.0.0",
        status: "ready",
        engine: "caption_boundary_planner",
        trainingDataset: {
          primaryDataset: "isign",
          displayName: "Private iSign caption boundary planner",
          recordCount: artifact.training.retainedUniqueCaptionRows,
          classCount: phrases.size + tokens.size,
          citationUrls: [
            "https://huggingface.co/datasets/Exploration-Lab/iSign",
            "https://aclanthology.org/2024.findings-acl.643/"
          ]
        },
        notes: [
          `Held-out token coverage is ${(heldOut.test.lexicalTokenCoverage * 100).toFixed(1)}%.`,
          `Held-out boundary-feature coverage is ${(heldOut.test.exportedBoundaryFeatureCoverage * 100).toFixed(1)}%.`,
          "Used only to choose caption chunk boundaries; it is not ISL grammar or translation."
        ]
      },
      splitMeaningChunks: createMeaningChunker(phrases, tokens)
    };
  } catch {
    return undefined;
  }
}

function createMeaningChunker(
  phrases: ReadonlyMap<string, BoundaryFeature>,
  tokens: ReadonlyMap<string, BoundaryFeature>
): MeaningChunker {
  return (cleanedText) => {
    const words = cleanedText.split(/\s+/).filter(Boolean);
    const chunks: string[] = [];
    let start = 0;
    while (words.length - start > MAX_CHUNK_WORDS) {
      const maximumEnd = Math.min(words.length, start + MAX_CHUNK_WORDS);
      let selectedEnd = maximumEnd;
      let selectedScore = 0;
      for (let end = start + MIN_CHUNK_WORDS; end <= maximumEnd; end += 1) {
        const score = boundaryScore(words, end, phrases, tokens);
        if (score > selectedScore || (score === selectedScore && end > selectedEnd)) {
          selectedEnd = end;
          selectedScore = score;
        }
      }
      chunks.push(words.slice(start, selectedEnd).join(" "));
      start = selectedEnd;
    }
    chunks.push(words.slice(start).join(" "));
    return chunks;
  };
}

function boundaryScore(
  words: readonly string[],
  end: number,
  phrases: ReadonlyMap<string, BoundaryFeature>,
  tokens: ReadonlyMap<string, BoundaryFeature>
): number {
  const normalized = words.map(normalizeToken);
  let feature = tokens.get(normalized[end - 1] ?? "");
  for (let length = 2; length <= 4 && end - length >= 0; length += 1) {
    const phrase = phrases.get(normalized.slice(end - length, end).join(" "));
    if (phrase && (!feature || phrase.probability > feature.probability)) {
      feature = phrase;
    }
  }
  if (!feature || feature.probability < 0.65 || feature.videoSupport < 20) {
    return 0;
  }
  return feature.probability + Math.min(0.1, feature.videoSupport / 10_000);
}

function normalizeToken(value: string): string {
  return value.normalize("NFKC").replace(/[’]/gu, "'").toLocaleLowerCase().replace(/[^\p{L}\p{N}'-]/gu, "");
}
