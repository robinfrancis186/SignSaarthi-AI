import { describe, expect, it } from "vitest";
import type { GlossaryEntry } from "@signsaarthi/shared";
import {
  computeLexicalFeatures,
  createLexicalFallbackResolver,
  defaultISLDatasetManifest,
  inferModelGlossaryEntries,
  islDatasetIdSchema,
  lexicalMatcherArtifactSchema,
  lexicalMatcherFeatureNames,
  mergeGlossaryEntries,
  trainISLLexiconModel,
  type ISLDatasetManifest,
  type LexicalMatcherArtifact
} from "./index";

const manifest: ISLDatasetManifest = {
  schemaVersion: 1,
  generatedAt: "2026-07-07T00:00:00.000Z",
  datasets: [
    {
      id: "include",
      name: "INCLUDE - Indian Lexicon Sign Language Dataset",
      task: "isolated_sign_recognition",
      sourceUrl: "https://github.com/AI4Bharat/INCLUDE",
      citation:
        "Sridhar et al., INCLUDE: A Large Scale Dataset for Indian Sign Language Recognition, ACM MM 2020",
      license: "dataset terms; code MIT"
    },
    {
      id: "isltranslate",
      name: "ISLTranslate",
      task: "continuous_sign_translation",
      sourceUrl: "https://github.com/Exploration-Lab/ISLTranslate",
      citation:
        "Joshi et al., ISLTranslate: Dataset for Translating Indian Sign Language, ACL Findings 2023",
      license: "research dataset terms"
    }
  ],
  records: [
    {
      term: "constitution",
      gloss: "CONSTITUTION",
      aliases: ["constitutional"],
      category: "government",
      signAssetId: "include_constitution_001",
      datasetIds: ["include", "isltranslate"],
      frequency: 42,
      reviewStatus: "approved"
    },
    {
      term: "justice",
      gloss: "JUSTICE",
      aliases: [],
      category: "government",
      signAssetId: "include_justice_001",
      datasetIds: ["include"],
      frequency: 18,
      reviewStatus: "approved"
    }
  ]
};

describe("ISL model training", () => {
  it("keeps the iSign ID parseable only for legacy manifest compatibility", () => {
    expect(islDatasetIdSchema.parse("isign")).toBe("isign");
    expect(JSON.stringify(defaultISLDatasetManifest).toLowerCase()).not.toContain("isign");
  });

  it("rejects legacy iSign rows as lexicon model provenance", () => {
    const legacyManifest: ISLDatasetManifest = {
      schemaVersion: 1,
      generatedAt: "2026-07-07T00:00:00.000Z",
      datasets: [
        {
          id: "isign",
          name: "Legacy gated source",
          task: "benchmark_translation",
          sourceUrl: "https://example.com/legacy-source",
          citation: "Legacy compatibility fixture",
          license: "not distributed"
        }
      ],
      records: [
        {
          term: "legacy term",
          gloss: "LEGACY TERM",
          aliases: [],
          category: "general",
          datasetIds: ["isign"],
          frequency: 1,
          reviewStatus: "pending_review"
        }
      ]
    };

    expect(() => trainISLLexiconModel(legacyManifest)).toThrow(
      "The iSign dataset ID is retained only for legacy manifest compatibility and is not supported as model provenance."
    );
  });

  it("trains a ranked model while downgrading approvals with unresolved asset IDs", () => {
    const model = trainISLLexiconModel(manifest, {
      now: () => "2026-07-07T00:00:00.000Z"
    });

    expect(model.metadata.status).toBe("ready");
    expect(model.metadata.trainingDataset.primaryDataset).toBe("include");
    expect(model.metadata.trainingDataset.recordCount).toBe(2);
    expect(model.metadata.notes).toContain(
      "The release ships generated INCLUDE playback motions separately and does not ship a trained sign-recognition model."
    );
    expect(JSON.stringify(model).toLowerCase()).not.toContain("isign");
    expect(model.glossaryEntries[0]).toMatchObject({
      term: "constitution",
      islGloss: "CONSTITUTION",
      signAssetId: "include_constitution_001",
      reviewStatus: "pending_review",
      confidence: "medium"
    });
    expect(model.glossaryEntries.every((entry) => entry.reviewStatus !== "approved")).toBe(true);
    expect(model.glossaryEntries.every((entry) => entry.confidence !== "high")).toBe(true);
  });

  it("keeps approval and high confidence only for a resolved expert-reviewed playable asset", () => {
    const model = trainISLLexiconModel(manifest, {
      resolveSignAsset: (assetId) => {
        if (assetId === "include_constitution_001") {
          return { expertReviewed: true, playable: true };
        }
        if (assetId === "include_justice_001") {
          return { expertReviewed: false, playable: true };
        }
        return undefined;
      }
    });

    expect(model.glossaryEntries.find((entry) => entry.term === "constitution")).toMatchObject({
      reviewStatus: "approved",
      confidence: "high"
    });
    expect(model.glossaryEntries.find((entry) => entry.term === "justice")).toMatchObject({
      reviewStatus: "pending_review",
      confidence: "medium"
    });
  });

  it("does not pre-approve built-in records that only carry asset identifiers", () => {
    expect(
      defaultISLDatasetManifest.records.every((record) => record.reviewStatus === "pending_review")
    ).toBe(true);
  });

  it("keeps built-in record provenance within the declared dataset references", () => {
    const declaredIds = new Set(defaultISLDatasetManifest.datasets.map((dataset) => dataset.id));

    expect(
      defaultISLDatasetManifest.records.every((entry) =>
        entry.datasetIds.every((datasetId) => declaredIds.has(datasetId))
      )
    ).toBe(true);
  });

  it("returns model-backed glossary entries for matching transcript text", () => {
    const model = trainISLLexiconModel(manifest);

    const entries = inferModelGlossaryEntries("The Constitution promises justice.", model);

    expect(entries.map((entry) => entry.term)).toEqual(["constitution", "justice"]);
    expect(entries.every((entry) => entry.source === "external_reference")).toBe(true);
  });

  it("keeps primary provenance while merging curated fallback aliases", () => {
    const primary = lexicalEntry("help", "HELP");
    const fallback = { ...lexicalEntry("Help", "HELP"), aliases: ["helps"] };

    const [merged] = mergeGlossaryEntries([primary], [fallback]);

    expect(merged).toMatchObject({
      id: primary.id,
      source: primary.source,
      term: primary.term,
      aliases: ["helps"]
    });
  });
});

describe("lexical fallback matcher", () => {
  it("validates the fixed feature order and rejects unknown artifact fields", () => {
    const artifact = lexicalMatcherArtifact();

    expect(lexicalMatcherArtifactSchema.parse(artifact).featureNames).toEqual(
      lexicalMatcherFeatureNames
    );
    expect(lexicalMatcherArtifactSchema.safeParse({ ...artifact, unexpected: true }).success).toBe(
      false
    );
    expect(
      lexicalMatcherArtifactSchema.safeParse({
        ...artifact,
        featureNames: [
          "editSimilarity",
          "exact",
          "trigramJaccard",
          "prefixRatio",
          "lengthRatio",
          "tokenJaccard"
        ]
      }).success
    ).toBe(false);
    expect(
      lexicalMatcherArtifactSchema.safeParse({
        ...artifact,
        metadata: { ...artifact.metadata, status: "unavailable" }
      }).success
    ).toBe(false);
  });

  it("computes NFKC-normalized Python-compatible lexical features", () => {
    expect(computeLexicalFeatures("computre", "computer")).toEqual([0, 0.75, 0.5, 0.75, 1, 0]);
    expect(computeLexicalFeatures("Ｃafé", "café")).toEqual([1, 1, 1, 1, 1, 1]);
    expect(computeLexicalFeatures("ＣＡＦÉ—Test", "café test")).toEqual([1, 1, 1, 1, 1, 1]);
    expect(computeLexicalFeatures("Straße!", "STRASSE")).toEqual([1, 1, 1, 1, 1, 1]);
  });

  it("resolves a high-confidence transposition and rejects uncertain text", () => {
    const computer = lexicalEntry("computer", "COMPUTER");
    const resolver = createLexicalFallbackResolver(
      [computer, lexicalEntry("commuter", "COMMUTER"), lexicalEntry("computer science", "CS")],
      lexicalMatcherArtifact()
    );

    expect(resolver("computre")).toBe(computer);
    expect(resolver("unrelated")).toBeUndefined();
    expect(resolver("computer")).toBeUndefined();
    expect(resolver("cat")).toBeUndefined();
    expect(resolver("कंप्यूटर")).toBeUndefined();
    expect(
      createLexicalFallbackResolver([computer, lexicalEntry("commuter", "COMMUTER")], {
        ...lexicalMatcherArtifact(),
        marginThreshold: 1
      })("computre")
    ).toBeUndefined();
    expect(
      createLexicalFallbackResolver([computer], unavailableLexicalMatcherArtifact())("computre")
    ).toBeUndefined();
  });

  it("resolves conservative exact inflections before learned typo scoring", () => {
    const help = lexicalEntry("help", "HELP");
    const student = lexicalEntry("student", "STUDENT");
    const resolver = createLexicalFallbackResolver(
      [help, lexicalEntry("hello", "HELLO"), student, lexicalEntry("new", "NEW")],
      { ...lexicalMatcherArtifact(), threshold: 0.999999 }
    );

    expect(resolver("helps")).toBe(help);
    expect(resolver("students")).toBe(student);
    expect(resolver("news")).toBeUndefined();
  });
});

function unavailableLexicalMatcherArtifact(): LexicalMatcherArtifact {
  const artifact = lexicalMatcherArtifact();
  return {
    ...artifact,
    metadata: { ...artifact.metadata, status: "unavailable" },
    metrics: {
      ...artifact.metrics,
      acceptance: {
        ...artifact.metrics.acceptance,
        ready: false,
        checks: {
          ...artifact.metrics.acceptance.checks,
          "testTop1AtLeast0.95": false
        },
        failedChecks: ["testTop1AtLeast0.95"]
      }
    }
  };
}

function lexicalMatcherArtifact(): LexicalMatcherArtifact {
  const binary = {
    precision: 1,
    recall: 1,
    f1: 1,
    truePositive: 2,
    falsePositive: 0,
    trueNegative: 2,
    falseNegative: 0,
    support: 4,
    predictedPositive: 2
  };
  const grouped = { top1: 1, mrr: 1, groupCount: 2, correctTop1: 2 };
  const evaluation = { binary, grouped };
  const splitSummary = {
    rowCount: 4,
    groupCount: 2,
    positiveRowCount: 2,
    negativeRowCount: 2
  };
  const sha256 = "0".repeat(64);

  return {
    schemaVersion: 1,
    metadata: {
      id: "isl-text-matcher-fixture",
      version: "test",
      status: "ready",
      engine: "lexicon_pair_matcher",
      trainingDataset: {
        primaryDataset: "fixture",
        displayName: "Lexical matcher fixture",
        recordCount: 4,
        classCount: 2,
        citationUrls: ["https://example.com/fixture"]
      },
      notes: ["Test-only coefficients; not a production model."]
    },
    featureNames: [...lexicalMatcherFeatureNames],
    means: [0, 0, 0, 0, 0, 0],
    scales: [1, 1, 1, 1, 1, 1],
    coefficients: [0, 30, 10, 10, 0, 0],
    intercept: -30,
    threshold: 0.99,
    marginThreshold: 0.1,
    minimumQueryLength: 4,
    metrics: {
      thresholdSelection: {
        split: "val",
        objective: "fixture threshold selection",
        precisionConstraint: 0.98,
        precisionConstraintFeasible: true,
        candidateThresholdCount: 2,
        selectedMetrics: binary
      },
      marginSelection: {
        split: "val",
        minimumMargin: 0.01,
        objective: "fixture margin selection",
        precisionConstraint: 0.98,
        precisionConstraintFeasible: true,
        acceptedGroupCount: 2,
        acceptedPrecision: 1,
        coverage: 1
      },
      val: evaluation,
      test: evaluation,
      baseline: {
        name: "exact_normalized_match",
        definition: "Fixture exact-match baseline.",
        val: evaluation,
        test: evaluation
      },
      acceptance: {
        ready: true,
        requirements: {
          valPrecision: 0.98,
          testPrecision: 0.98,
          testTop1: 0.95,
          integrityStatus: "passed",
          baselineImprovement: "strictly higher test F1 or grouped top1"
        },
        checks: {
          "valPrecisionAtLeast0.98": true,
          "testPrecisionAtLeast0.98": true,
          "testTop1AtLeast0.95": true,
          splitGroupIntegrityPassed: true,
          beatsExactBaselineOnTestF1OrTop1: true
        },
        failedChecks: [],
        baselineComparison: {
          trainedTestF1: 1,
          baselineTestF1: 0.5,
          trainedTestTop1: 1,
          baselineTestTop1: 0.5,
          beatsBaselineOnF1: true,
          beatsBaselineOnTop1: true
        }
      }
    },
    constraints: {
      singleTokenOnly: true,
      preserveSourceText: true,
      noGrammarClaims: true,
      noMotionSynthesis: true
    },
    dataHashes: {
      aggregateSha256: sha256,
      files: ["train", "val", "test"].map((split) => ({
        path: `data/matcher_${split}.jsonl`,
        sha256,
        byteLength: 1,
        rowCount: 4
      })) as LexicalMatcherArtifact["dataHashes"]["files"]
    },
    trainingConfig: {
      optimizer: "deterministic full-batch gradient descent",
      iterations: 500,
      learningRate: 0.2,
      l2: 0.01,
      randomness: "none",
      deterministicOrder: "fixture order",
      normalization: "Unicode NFKC casefold fixture",
      featureStandardization: "fixture means and scales",
      classWeighting: {
        strategy: "balanced fixture",
        activationImbalanceRatio: 1.25,
        negative: 1,
        positive: 1
      },
      collapsedTrainingFeatureVectors: 4,
      finalWeightedLogLoss: 0.1,
      thresholdSelection: "fixture validation threshold",
      marginSelection: "fixture validation margin",
      trainedAtPolicy: "fixed fixture timestamp",
      trainerPath: "scripts/train_isl_text_matcher.py",
      trainerSha256: sha256
    },
    integrity: {
      status: "passed",
      groupOverlapAcrossSplits: 0,
      normalizedQueryOverlapAcrossSplits: 0,
      duplicateCandidateIdsWithinGroups: 0,
      inconsistentCandidateMetadata: 0,
      splits: {
        train: splitSummary,
        val: splitSummary,
        test: splitSummary
      },
      uniqueCandidateCount: 2
    }
  };
}

function lexicalEntry(term: string, islGloss: string): GlossaryEntry {
  return {
    id: `fixture_${term.replace(/\s+/gu, "_")}`,
    term,
    aliases: [],
    language: "en",
    category: "technology",
    islGloss,
    confidence: "medium",
    source: "internal",
    reviewStatus: "pending_review",
    createdAt: "2026-07-16T00:00:00.000Z",
    updatedAt: "2026-07-16T00:00:00.000Z"
  };
}
