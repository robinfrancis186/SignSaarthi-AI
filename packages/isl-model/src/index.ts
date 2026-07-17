import {
  glossaryEntrySchema,
  modelMetadataSchema,
  type GlossaryEntry,
  type ModelMetadata
} from "@signsaarthi/shared";
import { z } from "zod";

export const islDatasetIdSchema = z.enum(["include", "isltranslate", "curated_mvp", "isign"]);
export const islDatasetTaskSchema = z.enum([
  "isolated_sign_recognition",
  "continuous_sign_translation",
  "benchmark_translation",
  "curated_seed"
]);

export const islDatasetReferenceSchema = z.object({
  id: islDatasetIdSchema,
  name: z.string(),
  task: islDatasetTaskSchema,
  sourceUrl: z.string().url(),
  citation: z.string(),
  license: z.string()
});

export const islDatasetRecordSchema = z.object({
  term: z.string().min(1),
  gloss: z.string().min(1),
  aliases: z.array(z.string()).default([]),
  category: glossaryEntrySchema.shape.category,
  signAssetId: z.string().optional(),
  datasetIds: z.array(islDatasetIdSchema).min(1),
  frequency: z.number().int().positive().default(1),
  reviewStatus: glossaryEntrySchema.shape.reviewStatus.default("pending_review")
});

export const islDatasetManifestSchema = z.object({
  schemaVersion: z.literal(1),
  generatedAt: z.string(),
  datasets: z.array(islDatasetReferenceSchema).min(1),
  records: z.array(islDatasetRecordSchema).min(1)
});

export const islLexiconModelSchema = z.object({
  schemaVersion: z.literal(1),
  metadata: modelMetadataSchema,
  glossaryEntries: z.array(glossaryEntrySchema),
  tokenWeights: z.record(z.number().nonnegative())
});

export const lexicalMatcherFeatureNames = [
  "exact",
  "editSimilarity",
  "trigramJaccard",
  "prefixRatio",
  "lengthRatio",
  "tokenJaccard"
] as const;

const lexicalFeatureVectorSchema = z.tuple([
  z.number().finite(),
  z.number().finite(),
  z.number().finite(),
  z.number().finite(),
  z.number().finite(),
  z.number().finite()
]);

const lexicalScaleVectorSchema = z.tuple([
  z.number().finite().positive(),
  z.number().finite().positive(),
  z.number().finite().positive(),
  z.number().finite().positive(),
  z.number().finite().positive(),
  z.number().finite().positive()
]);

const probabilitySchema = z.number().finite().min(0).max(1);
const countSchema = z.number().int().nonnegative();
const sha256Schema = z.string().regex(/^[a-f0-9]{64}$/u);

const lexicalBinaryMetricsSchema = z
  .object({
    precision: probabilitySchema,
    recall: probabilitySchema,
    f1: probabilitySchema,
    truePositive: countSchema,
    falsePositive: countSchema,
    trueNegative: countSchema,
    falseNegative: countSchema,
    support: countSchema,
    predictedPositive: countSchema
  })
  .strict();

const lexicalGroupedMetricsSchema = z
  .object({
    top1: probabilitySchema,
    mrr: probabilitySchema,
    groupCount: countSchema,
    correctTop1: countSchema
  })
  .strict();

const lexicalEvaluationSchema = z
  .object({
    binary: lexicalBinaryMetricsSchema,
    grouped: lexicalGroupedMetricsSchema
  })
  .strict();

const lexicalAcceptanceCheckSchema = z.enum([
  "valPrecisionAtLeast0.98",
  "testPrecisionAtLeast0.98",
  "testTop1AtLeast0.95",
  "splitGroupIntegrityPassed",
  "beatsExactBaselineOnTestF1OrTop1"
]);

const lexicalAcceptanceChecksSchema = z
  .object({
    "valPrecisionAtLeast0.98": z.boolean(),
    "testPrecisionAtLeast0.98": z.boolean(),
    "testTop1AtLeast0.95": z.boolean(),
    splitGroupIntegrityPassed: z.boolean(),
    beatsExactBaselineOnTestF1OrTop1: z.boolean()
  })
  .strict();

const lexicalSplitSummarySchema = z
  .object({
    rowCount: countSchema,
    groupCount: countSchema,
    positiveRowCount: countSchema,
    negativeRowCount: countSchema
  })
  .strict();

export const lexicalMatcherArtifactSchema = z
  .object({
    schemaVersion: z.literal(1),
    metadata: modelMetadataSchema
      .extend({
        status: z.enum(["ready", "unavailable"]),
        engine: z.literal("lexicon_pair_matcher"),
        trainingDataset: modelMetadataSchema.shape.trainingDataset.strict()
      })
      .strict(),
    featureNames: z.tuple([
      z.literal("exact"),
      z.literal("editSimilarity"),
      z.literal("trigramJaccard"),
      z.literal("prefixRatio"),
      z.literal("lengthRatio"),
      z.literal("tokenJaccard")
    ]),
    means: lexicalFeatureVectorSchema,
    scales: lexicalScaleVectorSchema,
    coefficients: lexicalFeatureVectorSchema,
    intercept: z.number().finite(),
    threshold: probabilitySchema,
    marginThreshold: probabilitySchema,
    minimumQueryLength: z.literal(4),
    metrics: z
      .object({
        thresholdSelection: z
          .object({
            split: z.literal("val"),
            objective: z.string().min(1),
            precisionConstraint: probabilitySchema,
            precisionConstraintFeasible: z.boolean(),
            candidateThresholdCount: z.number().int().positive(),
            selectedMetrics: lexicalBinaryMetricsSchema
          })
          .strict(),
        marginSelection: z
          .object({
            split: z.literal("val"),
            minimumMargin: probabilitySchema,
            objective: z.string().min(1),
            precisionConstraint: probabilitySchema,
            precisionConstraintFeasible: z.boolean(),
            acceptedGroupCount: countSchema,
            acceptedPrecision: probabilitySchema,
            coverage: probabilitySchema
          })
          .strict(),
        val: lexicalEvaluationSchema,
        test: lexicalEvaluationSchema,
        baseline: z
          .object({
            name: z.literal("exact_normalized_match"),
            definition: z.string().min(1),
            val: lexicalEvaluationSchema,
            test: lexicalEvaluationSchema
          })
          .strict(),
        acceptance: z
          .object({
            ready: z.boolean(),
            requirements: z
              .object({
                valPrecision: probabilitySchema,
                testPrecision: probabilitySchema,
                testTop1: probabilitySchema,
                integrityStatus: z.literal("passed"),
                baselineImprovement: z.string().min(1)
              })
              .strict(),
            checks: lexicalAcceptanceChecksSchema,
            failedChecks: z.array(lexicalAcceptanceCheckSchema),
            baselineComparison: z
              .object({
                trainedTestF1: probabilitySchema,
                baselineTestF1: probabilitySchema,
                trainedTestTop1: probabilitySchema,
                baselineTestTop1: probabilitySchema,
                beatsBaselineOnF1: z.boolean(),
                beatsBaselineOnTop1: z.boolean()
              })
              .strict()
          })
          .strict()
      })
      .strict(),
    constraints: z
      .object({
        singleTokenOnly: z.literal(true),
        preserveSourceText: z.literal(true),
        noGrammarClaims: z.literal(true),
        noMotionSynthesis: z.literal(true)
      })
      .strict(),
    dataHashes: z
      .object({
        aggregateSha256: sha256Schema,
        files: z
          .array(
            z
              .object({
                path: z.string().min(1),
                sha256: sha256Schema,
                byteLength: countSchema,
                rowCount: countSchema
              })
              .strict()
          )
          .length(3)
      })
      .strict(),
    trainingConfig: z
      .object({
        optimizer: z.string().min(1),
        iterations: z.number().int().positive(),
        learningRate: z.number().finite().positive(),
        l2: z.number().finite().nonnegative(),
        randomness: z.literal("none"),
        deterministicOrder: z.string().min(1),
        normalization: z.string().min(1),
        featureStandardization: z.string().min(1),
        classWeighting: z
          .object({
            strategy: z.string().min(1),
            activationImbalanceRatio: z.number().finite().positive(),
            negative: z.number().finite().positive(),
            positive: z.number().finite().positive()
          })
          .strict(),
        collapsedTrainingFeatureVectors: countSchema,
        finalWeightedLogLoss: z.number().finite().nonnegative(),
        thresholdSelection: z.string().min(1),
        marginSelection: z.string().min(1),
        trainedAtPolicy: z.string().min(1),
        trainerPath: z.literal("scripts/train_isl_text_matcher.py"),
        trainerSha256: sha256Schema
      })
      .strict(),
    integrity: z
      .object({
        status: z.literal("passed"),
        groupOverlapAcrossSplits: z.literal(0),
        normalizedQueryOverlapAcrossSplits: z.literal(0),
        duplicateCandidateIdsWithinGroups: z.literal(0),
        inconsistentCandidateMetadata: z.literal(0),
        splits: z
          .object({
            train: lexicalSplitSummarySchema,
            val: lexicalSplitSummarySchema,
            test: lexicalSplitSummarySchema
          })
          .strict(),
        uniqueCandidateCount: countSchema
      })
      .strict()
  })
  .strict()
  .superRefine((artifact, context) => {
    const expectedStatus = artifact.metrics.acceptance.ready ? "ready" : "unavailable";
    if (artifact.metadata.status !== expectedStatus) {
      context.addIssue({
        code: z.ZodIssueCode.custom,
        path: ["metadata", "status"],
        message: "Matcher metadata status must reflect the acceptance result."
      });
    }

    const failedChecks = new Set(artifact.metrics.acceptance.failedChecks);
    const checksReady = Object.values(artifact.metrics.acceptance.checks).every(Boolean);
    if (artifact.metrics.acceptance.ready !== checksReady) {
      context.addIssue({
        code: z.ZodIssueCode.custom,
        path: ["metrics", "acceptance", "ready"],
        message: "Matcher readiness must reflect every acceptance check."
      });
    }
    for (const [name, passed] of Object.entries(artifact.metrics.acceptance.checks)) {
      if (failedChecks.has(name as z.infer<typeof lexicalAcceptanceCheckSchema>) === passed) {
        context.addIssue({
          code: z.ZodIssueCode.custom,
          path: ["metrics", "acceptance", "failedChecks"],
          message: "Matcher failedChecks must match the acceptance checks."
        });
        break;
      }
    }
  });

export type ISLDatasetManifest = z.infer<typeof islDatasetManifestSchema>;
export type ISLLexiconModel = z.infer<typeof islLexiconModelSchema>;
export type LexicalMatcherArtifact = z.infer<typeof lexicalMatcherArtifactSchema>;
export type LexicalFeatureVector = z.infer<typeof lexicalFeatureVectorSchema>;
export type LexicalFallbackResolver = (token: string) => GlossaryEntry | undefined;

type PreparedLexicalValue = {
  text: string;
  characters: string[];
  trigrams: Set<string>;
  tokens: Set<string>;
};

export type SignAssetResolution = {
  expertReviewed: boolean;
  playable: boolean;
};

export type SignAssetResolver = (signAssetId: string) => SignAssetResolution | undefined;

type TrainOptions = {
  now?: () => string;
  resolveSignAsset?: SignAssetResolver;
};

const MODEL_VERSION = "0.1.0";

export const defaultISLDatasetManifest: ISLDatasetManifest = {
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
    record("today", "TODAY", "general", "include_today_001", ["now"], ["include"], 31),
    record(
      "artificial intelligence",
      "AI",
      "technology",
      "include_ai_001",
      ["ai"],
      ["isltranslate"],
      26
    ),
    record(
      "education",
      "EDUCATION",
      "education",
      "include_education_001",
      ["learn", "learning"],
      ["include"],
      40
    ),
    record(
      "equal opportunity",
      "EQUAL OPPORTUNITY",
      "general",
      "include_equal_opportunity_001",
      ["equal opportunities", "equal"],
      ["isltranslate"],
      19
    ),
    record("meeting", "MEETING", "meeting", "include_meeting_001", [], ["include"], 16),
    record("help", "HELP", "general", "include_help_001", ["helps"], ["include"], 34),
    record("internet", "INTERNET", "technology", "include_internet_001", [], ["include"], 15),
    record(
      "barriers",
      "BARRIER REDUCE",
      "general",
      "include_barrier_001",
      ["barrier", "break barriers"],
      ["isltranslate"],
      10
    ),
    record(
      "constitution",
      "CONSTITUTION",
      "government",
      "include_constitution_001",
      ["constitutional"],
      ["include", "isltranslate"],
      42
    ),
    record("preamble", "PREAMBLE", "government", "include_preamble_001", [], ["include"], 22),
    record(
      "fundamental rights",
      "FUNDAMENTAL RIGHTS",
      "government",
      "include_fundamental_rights_001",
      ["right", "rights"],
      ["isltranslate"],
      30
    ),
    record("justice", "JUSTICE", "government", "include_justice_001", [], ["include"], 18),
    record("liberty", "LIBERTY", "government", "include_liberty_001", [], ["include"], 17),
    record("fraternity", "FRATERNITY", "government", "include_fraternity_001", [], ["include"], 11)
  ]
};

export function trainISLLexiconModel(
  manifest: ISLDatasetManifest,
  options: TrainOptions = {}
): ISLLexiconModel {
  const parsedManifest = islDatasetManifestSchema.parse(manifest);
  assertSupportedModelProvenance(parsedManifest);
  const trainedAt = options.now?.() ?? new Date().toISOString();
  const glossaryEntries = parsedManifest.records
    .slice()
    .sort((left, right) => right.frequency - left.frequency || left.term.localeCompare(right.term))
    .map((entry): GlossaryEntry => {
      const signAsset = entry.signAssetId;
      const hasApprovedSignAsset =
        entry.reviewStatus === "approved" &&
        resolvesToExpertReviewedPlayableAsset(signAsset, options.resolveSignAsset);
      const reviewStatus =
        entry.reviewStatus === "approved" && !hasApprovedSignAsset
          ? "pending_review"
          : entry.reviewStatus;
      return {
        id: `model_${slug(entry.term)}`,
        term: entry.term,
        aliases: entry.aliases,
        language: "en",
        category: entry.category,
        islGloss: entry.gloss,
        ...(signAsset ? { signAssetId: signAsset } : {}),
        confidence: hasApprovedSignAsset ? "high" : reviewStatus === "rejected" ? "low" : "medium",
        source: "external_reference",
        reviewStatus,
        regionalVariant: "Indian Sign Language",
        createdAt: trainedAt,
        updatedAt: trainedAt
      };
    });

  const tokenWeights = Object.fromEntries(
    parsedManifest.records.map((entry) => [entry.term, entry.frequency])
  );
  const citationUrls = Array.from(
    new Set(parsedManifest.datasets.map((dataset) => dataset.sourceUrl))
  );
  const [primaryDataset] = parsedManifest.datasets;
  if (!primaryDataset) {
    throw new Error("An ISL dataset manifest must declare at least one dataset.");
  }
  const datasetNames = parsedManifest.datasets.map((dataset) => dataset.name.split(" - ")[0]);
  const metadata: ModelMetadata = {
    id: `isl-lexicon-${trainedAt.slice(0, 10)}`,
    version: MODEL_VERSION,
    status: "ready",
    engine: "lexicon_ranker",
    trainedAt,
    trainingDataset: {
      primaryDataset: primaryDataset.id,
      displayName: `${datasetNames.join(" + ")} reference lexicon`,
      recordCount: parsedManifest.records.length,
      classCount: new Set(parsedManifest.records.map((entry) => entry.term)).size,
      citationUrls
    },
    notes: [
      "Deterministic lexicon/ranking model compiled from the supplied ISL dataset manifest for text-to-gloss lookup.",
      "Glossary provenance is limited to the dataset references declared in this manifest.",
      "The release ships generated INCLUDE playback motions separately and does not ship a trained sign-recognition model."
    ]
  };

  return islLexiconModelSchema.parse({
    schemaVersion: 1,
    metadata,
    glossaryEntries,
    tokenWeights
  });
}

export function inferModelGlossaryEntries(text: string, model: ISLLexiconModel): GlossaryEntry[] {
  const parsedModel = islLexiconModelSchema.parse(model);
  const lower = ` ${text.toLowerCase()} `;
  const matches = parsedModel.glossaryEntries.filter((entry) => {
    const candidates = [entry.term, ...entry.aliases].sort(
      (left, right) => right.length - left.length
    );
    return candidates.some((candidate) =>
      new RegExp(`\\b${escapeRegExp(candidate.toLowerCase())}\\b`, "i").test(lower)
    );
  });

  return matches.sort((left, right) => firstIndex(text, left) - firstIndex(text, right));
}

export function mergeGlossaryEntries(
  primary: GlossaryEntry[],
  fallback: GlossaryEntry[]
): GlossaryEntry[] {
  const merged = new Map<string, GlossaryEntry>();
  for (const entry of [...primary, ...fallback]) {
    const key = entry.term.toLowerCase();
    const existing = merged.get(key);
    if (!existing) {
      merged.set(key, { ...entry, aliases: [...entry.aliases] });
      continue;
    }
    const aliases = [...new Set([...existing.aliases, ...entry.aliases])].filter(
      (alias) => alias.toLowerCase() !== key
    );
    merged.set(key, { ...existing, aliases });
  }
  return [...merged.values()];
}

export function computeLexicalFeatures(query: string, candidate: string): LexicalFeatureVector {
  return computeNormalizedLexicalFeatures(
    normalizeLexicalText(query),
    normalizeLexicalText(candidate)
  );
}

export function createLexicalFallbackResolver(
  entries: GlossaryEntry[],
  artifact: LexicalMatcherArtifact
): LexicalFallbackResolver {
  const parsedArtifact = lexicalMatcherArtifactSchema.parse(artifact);
  if (parsedArtifact.metadata.status !== "ready") {
    return () => undefined;
  }
  const seenTerms = new Set<string>();
  const candidates = entries.flatMap((entry, index) => {
    const normalizedTerm = normalizeLexicalText(entry.term);
    if (!isSingleLatinToken(normalizedTerm) || seenTerms.has(normalizedTerm)) {
      return [];
    }
    seenTerms.add(normalizedTerm);
    return [{ entry, index, preparedTerm: prepareLexicalValue(normalizedTerm) }];
  });
  const entriesByNormalizedTerm = new Map(
    candidates.map((candidate) => [candidate.preparedTerm.text, candidate.entry])
  );

  return (token) => {
    const normalizedQuery = normalizeLexicalText(token);
    if (
      !isSingleLatinToken(normalizedQuery) ||
      Array.from(normalizedQuery).length < parsedArtifact.minimumQueryLength ||
      seenTerms.has(normalizedQuery)
    ) {
      return undefined;
    }

    const inflectedEntry = resolveConservativeInflection(normalizedQuery, entriesByNormalizedTerm);
    if (inflectedEntry) {
      return inflectedEntry;
    }

    const preparedQuery = prepareLexicalValue(normalizedQuery);
    const ranked = candidates
      .map((candidate) => ({
        ...candidate,
        probability: scoreLexicalPair(preparedQuery, candidate.preparedTerm, parsedArtifact)
      }))
      .sort((left, right) => right.probability - left.probability || left.index - right.index);
    const best = ranked[0];
    if (!best || best.probability < parsedArtifact.threshold) {
      return undefined;
    }

    const runnerUpProbability = ranked[1]?.probability ?? 0;
    if (best.probability - runnerUpProbability < parsedArtifact.marginThreshold) {
      return undefined;
    }
    return best.entry;
  };
}

function resolveConservativeInflection(
  normalizedQuery: string,
  entriesByNormalizedTerm: ReadonlyMap<string, GlossaryEntry>
): GlossaryEntry | undefined {
  if (
    normalizedQuery.length < 5 ||
    !normalizedQuery.endsWith("s") ||
    /(?:ss|us|is|ws|ics)$/u.test(normalizedQuery)
  ) {
    return undefined;
  }
  return entriesByNormalizedTerm.get(normalizedQuery.slice(0, -1));
}

function scoreLexicalPair(
  query: PreparedLexicalValue,
  candidate: PreparedLexicalValue,
  artifact: LexicalMatcherArtifact
): number {
  const features = computePreparedLexicalFeatures(query, candidate);
  const logit = features.reduce((sum, feature, index) => {
    const mean = artifact.means[index]!;
    const scale = artifact.scales[index]!;
    const coefficient = artifact.coefficients[index]!;
    return sum + ((feature - mean) / scale) * coefficient;
  }, artifact.intercept);
  return sigmoid(logit);
}

function computeNormalizedLexicalFeatures(
  normalizedQuery: string,
  normalizedCandidate: string
): LexicalFeatureVector {
  return computePreparedLexicalFeatures(
    prepareLexicalValue(normalizedQuery),
    prepareLexicalValue(normalizedCandidate)
  );
}

function prepareLexicalValue(text: string): PreparedLexicalValue {
  const characters = Array.from(text);
  return {
    text,
    characters,
    trigrams: trigrams(characters),
    tokens: tokenSet(text)
  };
}

function computePreparedLexicalFeatures(
  query: PreparedLexicalValue,
  candidate: PreparedLexicalValue
): LexicalFeatureVector {
  const queryCharacters = query.characters;
  const candidateCharacters = candidate.characters;
  const longestLength = Math.max(queryCharacters.length, candidateCharacters.length);
  let sharedPrefixLength = 0;
  while (
    sharedPrefixLength < queryCharacters.length &&
    sharedPrefixLength < candidateCharacters.length &&
    queryCharacters[sharedPrefixLength] === candidateCharacters[sharedPrefixLength]
  ) {
    sharedPrefixLength += 1;
  }

  return [
    query.text === candidate.text ? 1 : 0,
    longestLength === 0
      ? 1
      : 1 - levenshteinDistance(queryCharacters, candidateCharacters) / longestLength,
    jaccard(query.trigrams, candidate.trigrams),
    longestLength === 0 ? 1 : sharedPrefixLength / longestLength,
    longestLength === 0
      ? 1
      : Math.min(queryCharacters.length, candidateCharacters.length) / longestLength,
    jaccard(query.tokens, candidate.tokens)
  ];
}

function normalizeLexicalText(value: string): string {
  return caseFoldLatin(value.normalize("NFKC"))
    .replace(/[^\p{L}\p{M}\p{N}]/gu, " ")
    .trim()
    .replace(/\s+/gu, " ");
}

function isSingleLatinToken(value: string): boolean {
  return /^[\p{Script=Latin}\p{M}\p{N}]+$/u.test(value) && /\p{Script=Latin}/u.test(value);
}

function caseFoldLatin(value: string): string {
  return value
    .toLowerCase()
    .replace(/\u00df/gu, "ss")
    .replace(/\u01f0/gu, "j\u030c")
    .replace(/\u1e96/gu, "h\u0331")
    .replace(/\u1e97/gu, "t\u0308")
    .replace(/\u1e98/gu, "w\u030a")
    .replace(/\u1e99/gu, "y\u030a")
    .replace(/\u03c2/gu, "\u03c3");
}

function trigrams(characters: string[]): Set<string> {
  if (characters.length === 0) {
    return new Set();
  }
  if (characters.length < 3) {
    return new Set([characters.join("")]);
  }

  const grams = new Set<string>();
  for (let index = 0; index <= characters.length - 3; index += 1) {
    grams.add(characters.slice(index, index + 3).join(""));
  }
  return grams;
}

function tokenSet(value: string): Set<string> {
  return new Set(value ? value.split(" ") : []);
}

function jaccard(left: Set<string>, right: Set<string>): number {
  const union = new Set([...left, ...right]);
  if (union.size === 0) {
    return 1;
  }
  let intersectionSize = 0;
  for (const value of left) {
    if (right.has(value)) {
      intersectionSize += 1;
    }
  }
  return intersectionSize / union.size;
}

function levenshteinDistance(left: string[], right: string[]): number {
  if (left.length < right.length) {
    return levenshteinDistance(right, left);
  }

  let previous = Array.from({ length: right.length + 1 }, (_, index) => index);
  for (let leftIndex = 1; leftIndex <= left.length; leftIndex += 1) {
    const current = [leftIndex];
    for (let rightIndex = 1; rightIndex <= right.length; rightIndex += 1) {
      current.push(
        Math.min(
          current[rightIndex - 1]! + 1,
          previous[rightIndex]! + 1,
          previous[rightIndex - 1]! + (left[leftIndex - 1] === right[rightIndex - 1] ? 0 : 1)
        )
      );
    }
    previous = current;
  }
  return previous[right.length]!;
}

function sigmoid(value: number): number {
  if (value >= 0) {
    return 1 / (1 + Math.exp(-value));
  }
  const exponential = Math.exp(value);
  return exponential / (1 + exponential);
}

function record(
  term: string,
  gloss: string,
  category: GlossaryEntry["category"],
  signAssetId: string,
  aliases: string[],
  datasetIds: Array<"include" | "isltranslate" | "curated_mvp">,
  frequency: number
): ISLDatasetManifest["records"][number] {
  return {
    term,
    gloss,
    aliases,
    category,
    signAssetId,
    datasetIds,
    frequency,
    reviewStatus: "pending_review"
  };
}

function resolvesToExpertReviewedPlayableAsset(
  signAssetId: string | undefined,
  resolveSignAsset: SignAssetResolver | undefined
): boolean {
  if (!signAssetId || !resolveSignAsset) {
    return false;
  }

  const asset = resolveSignAsset(signAssetId);
  return asset?.expertReviewed === true && asset.playable === true;
}

function assertSupportedModelProvenance(manifest: ISLDatasetManifest): void {
  const hasLegacyISignProvenance =
    manifest.datasets.some((dataset) => dataset.id === "isign") ||
    manifest.records.some((entry) => entry.datasetIds.includes("isign"));

  if (hasLegacyISignProvenance) {
    throw new Error(
      "The iSign dataset ID is retained only for legacy manifest compatibility and is not supported as model provenance."
    );
  }
}

function firstIndex(text: string, entry: GlossaryEntry): number {
  const lower = text.toLowerCase();
  const indexes = [entry.term, ...entry.aliases]
    .map((candidate) => lower.indexOf(candidate.toLowerCase()))
    .filter((index) => index >= 0);
  return indexes.length ? Math.min(...indexes) : Number.MAX_SAFE_INTEGER;
}

function slug(value: string): string {
  return value
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "_")
    .replace(/^_+|_+$/g, "");
}

function escapeRegExp(value: string): string {
  return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}
