import { z } from "zod";

type InferenceLabel = {
  label: string;
  normalizedLabel: string;
  gloss: string;
};

function repairNumericOnlyGloss<T extends InferenceLabel>(value: T): T {
  const gloss = value.gloss.trim();
  if (/\p{Letter}/u.test(gloss) || !/\p{Number}/u.test(gloss)) {
    return { ...value, gloss };
  }

  const labelGloss = value.label
    .replace(/\.[a-z0-9]+$/i, "")
    .replace(/^\s*\d+[\s.)_-]*/u, "")
    .replace(/[_-]+/gu, " ")
    .replace(/[^\p{Letter}\p{Number}\s]/gu, " ")
    .replace(/\s+/gu, " ")
    .trim();
  const normalizedGloss = value.normalizedLabel
    .replace(/[_-]+/gu, " ")
    .replace(/\s+/gu, " ")
    .trim();
  const repaired = /\p{Letter}/u.test(labelGloss) ? labelGloss : normalizedGloss;

  return { ...value, gloss: repaired.toUpperCase() };
}

function hasSemanticGloss(value: InferenceLabel): boolean {
  return /\p{Letter}/u.test(value.gloss);
}

export const transcriptSourceSchema = z.enum(["youtube_captions", "meeting_captions", "mock"]);

export const confidenceLevelSchema = z.enum(["high", "medium", "low"]);
export const languageHintSchema = z.enum(["en", "hi", "hinglish", "unknown"]);
export const fallbackSchema = z.enum(["none", "caption", "fingerspell", "unknown"]);
export const simplificationLevelSchema = z.enum(["light", "standard", "strong"]);

export const sourceCharacterSpanSchema = z
  .object({
    start: z.number().int().nonnegative().max(Number.MAX_SAFE_INTEGER),
    end: z.number().int().nonnegative().max(Number.MAX_SAFE_INTEGER)
  })
  .strict()
  .refine((span) => span.end > span.start, {
    path: ["end"],
    message: "Source character span end must be greater than start."
  });

export const planningFeatureSchema = z
  .object({
    isQuestion: z.boolean(),
    hasNegation: z.boolean(),
    questionCueTokenIds: z.array(z.string()),
    negationCueTokenIds: z.array(z.string())
  })
  .strict();

export const planningReviewStatusSchema = z.enum([
  "unreviewed",
  "draft",
  "pending_review",
  "approved",
  "rejected"
]);

export const planningProvenanceSchema = z
  .object({
    source: z.enum(["source_text", "controlled_glossary", "deterministic_fallback"]),
    glossaryEntryId: z.string().optional()
  })
  .strict()
  .superRefine((provenance, context) => {
    if (provenance.source === "controlled_glossary" && !provenance.glossaryEntryId) {
      context.addIssue({
        code: z.ZodIssueCode.custom,
        path: ["glossaryEntryId"],
        message: "Controlled-glossary provenance requires a glossaryEntryId."
      });
    }
    if (provenance.source !== "controlled_glossary" && provenance.glossaryEntryId) {
      context.addIssue({
        code: z.ZodIssueCode.custom,
        path: ["glossaryEntryId"],
        message: "Only controlled-glossary provenance may include a glossaryEntryId."
      });
    }
  });

export const planningActionKindSchema = z.enum([
  "sign_candidate",
  "sign",
  "fingerspell",
  "caption",
  "unknown"
]);

export const planningActionDecisionSchema = z
  .object({
    kind: planningActionKindSchema,
    label: z.string().min(1)
  })
  .strict();

export const planningSourceTokenSchema = z
  .object({
    id: z.string(),
    text: z.string().min(1),
    normalizedText: z.string().min(1),
    sourceSpan: sourceCharacterSpanSchema,
    clauseId: z.string()
  })
  .strict();

export const planningClauseSchema = z
  .object({
    id: z.string(),
    sourceSpan: sourceCharacterSpanSchema,
    sourceTokenIds: z.array(z.string()).min(1),
    features: planningFeatureSchema
  })
  .strict();

export const planningActionSchema = z
  .object({
    id: z.string(),
    sourceText: z.string().min(1),
    sourceSpan: sourceCharacterSpanSchema,
    sourceTokenIds: z.array(z.string()).min(1),
    clauseId: z.string(),
    plannedAction: planningActionDecisionSchema,
    resolvedAction: planningActionDecisionSchema,
    provenance: planningProvenanceSchema,
    reviewStatus: planningReviewStatusSchema
  })
  .strict();

const expertGateSchema = z
  .object({
    applied: z.literal(false),
    status: z.literal("expert_gated"),
    reviewStatus: z.literal("pending_review")
  })
  .strict();

export const interpretationPlanningIRSchema = z
  .object({
    id: z.string(),
    sourceSegmentId: z.string(),
    sourceText: z.string().min(1),
    sourceTokens: z.array(planningSourceTokenSchema),
    clauses: z.array(planningClauseSchema),
    actions: z.array(planningActionSchema),
    features: planningFeatureSchema,
    grammar: z
      .object({
        sourceOrderPreserved: z.literal(true),
        reordering: expertGateSchema,
        nonManualSignals: expertGateSchema
      })
      .strict(),
    reviewStatus: z.literal("pending_review")
  })
  .strict()
  .superRefine((planning, context) => {
    const tokenIds = planning.sourceTokens.map((token) => token.id);
    const actionTokenIds = planning.actions.flatMap((action) => action.sourceTokenIds);
    const clauseTokenIds = planning.clauses.flatMap((clause) => clause.sourceTokenIds);
    if (new Set(tokenIds).size !== tokenIds.length) {
      context.addIssue({
        code: z.ZodIssueCode.custom,
        path: ["sourceTokens"],
        message: "Planning source token IDs must be unique."
      });
    }
    if (actionTokenIds.join("\u0000") !== tokenIds.join("\u0000")) {
      context.addIssue({
        code: z.ZodIssueCode.custom,
        path: ["actions"],
        message:
          "Planning actions must preserve every source token exactly once and in source order."
      });
    }
    if (clauseTokenIds.join("\u0000") !== tokenIds.join("\u0000")) {
      context.addIssue({
        code: z.ZodIssueCode.custom,
        path: ["clauses"],
        message:
          "Planning clauses must preserve every source token exactly once and in source order."
      });
    }

    const clauseIds = new Set(planning.clauses.map((clause) => clause.id));
    for (const [index, token] of planning.sourceTokens.entries()) {
      if (planning.sourceText.slice(token.sourceSpan.start, token.sourceSpan.end) !== token.text) {
        context.addIssue({
          code: z.ZodIssueCode.custom,
          path: ["sourceTokens", index, "sourceSpan"],
          message: "Source token spans must resolve to the original source text."
        });
      }
      if (!clauseIds.has(token.clauseId)) {
        context.addIssue({
          code: z.ZodIssueCode.custom,
          path: ["sourceTokens", index, "clauseId"],
          message: "Source tokens must reference a planning clause."
        });
      }
    }
    for (const [index, action] of planning.actions.entries()) {
      if (
        planning.sourceText.slice(action.sourceSpan.start, action.sourceSpan.end) !==
        action.sourceText
      ) {
        context.addIssue({
          code: z.ZodIssueCode.custom,
          path: ["actions", index, "sourceSpan"],
          message: "Planning action spans must resolve to the original source text."
        });
      }
      if (!clauseIds.has(action.clauseId)) {
        context.addIssue({
          code: z.ZodIssueCode.custom,
          path: ["actions", index, "clauseId"],
          message: "Planning actions must reference a planning clause."
        });
      }
    }
  });

export const sourceSpanSchema = sourceCharacterSpanSchema;
export const planningIRSchema = interpretationPlanningIRSchema;

export const apiRequestLimits = {
  identifierLength: 128,
  pageTitleLength: 512,
  urlLength: 2048,
  transcriptLength: 8_000,
  glossaryQueryLength: 256,
  feedbackTextLength: 8_000,
  feedbackGlossItems: 512,
  feedbackGlossTextLength: 256,
  keypointFrames: 240,
  keypointLandmarksPerFrame: 128,
  sessionDurationMs: 7 * 24 * 60 * 60 * 1000
} as const;

const apiIdentifierSchema = z.string().min(1).max(apiRequestLimits.identifierLength);

export const outputModeSchema = z.enum([
  "captions_only",
  "captions_avatar",
  "avatar_only",
  "summary_avatar"
]);

export const transcriptSegmentSchema = z.object({
  id: z.string(),
  sessionId: z.string(),
  source: transcriptSourceSchema,
  rawText: z.string().min(1),
  cleanedText: z.string(),
  languageHint: languageHintSchema.optional(),
  startMs: z.number().int().nonnegative().optional(),
  endMs: z.number().int().nonnegative().optional(),
  confidence: z.number().min(0).max(1),
  createdAt: z.string()
});

export const simplifiedChunkSchema = z.object({
  id: z.string(),
  transcriptSegmentId: z.string(),
  text: z.string(),
  meaningUnits: z.array(z.string()),
  confidence: confidenceLevelSchema
});

export const islGlossItemSchema = z.object({
  id: z.string(),
  token: z.string(),
  gloss: z.string(),
  signAssetId: z.string().optional(),
  isVerified: z.boolean(),
  fallback: fallbackSchema,
  confidence: confidenceLevelSchema,
  sourceSegmentId: z.string().optional(),
  planningActionId: z.string().optional(),
  sourceTokenIds: z.array(z.string()).min(1).optional(),
  sourceSpan: sourceCharacterSpanSchema.optional(),
  clauseId: z.string().optional(),
  plannedAction: planningActionDecisionSchema.optional(),
  resolvedAction: planningActionDecisionSchema.optional(),
  provenance: planningProvenanceSchema.optional(),
  reviewStatus: planningReviewStatusSchema.optional()
});

export const interpretationSchema = z.object({
  id: z.string(),
  sessionId: z.string(),
  chunkId: z.string(),
  originalText: z.string(),
  cleanedText: z.string(),
  simplifiedText: z.string(),
  keyPoints: z.array(z.string()),
  detectedTerms: z.array(z.string()),
  glossSequence: z.array(islGlossItemSchema),
  planning: interpretationPlanningIRSchema.optional(),
  confidence: confidenceLevelSchema,
  warnings: z.array(z.string()),
  createdAt: z.string()
});

export const glossaryEntrySchema = z.object({
  id: z.string(),
  term: z.string(),
  aliases: z.array(z.string()).default([]),
  language: z.enum(["en", "hi", "hinglish"]),
  category: z.enum(["education", "technology", "meeting", "government", "general"]),
  islGloss: z.string(),
  signAssetId: z.string().optional(),
  exampleSentence: z.string().optional(),
  confidence: confidenceLevelSchema,
  source: z.enum(["internal", "user_suggested", "expert_reviewed", "external_reference"]),
  reviewStatus: z.enum(["draft", "pending_review", "approved", "rejected"]),
  regionalVariant: z.string().optional(),
  createdAt: z.string(),
  updatedAt: z.string()
});

export const feedbackReasonSchema = z.enum([
  "wrong_sign",
  "incorrect_translation",
  "missing_sign",
  "poor_avatar_motion",
  "regional_variation",
  "other"
]);

export const timestampSchema = z
  .object({
    startMs: z.number().int().nonnegative().max(Number.MAX_SAFE_INTEGER),
    endMs: z.number().int().nonnegative().max(Number.MAX_SAFE_INTEGER)
  })
  .strict()
  .refine((timestamp) => timestamp.endMs >= timestamp.startMs, {
    path: ["endMs"],
    message: "Timestamp endMs must be greater than or equal to startMs."
  });

export const feedbackReportSchema = z.object({
  id: z.string(),
  sessionId: z.string(),
  interpretationId: z.string(),
  reason: feedbackReasonSchema,
  timestamp: timestampSchema,
  transcriptText: z.string().max(apiRequestLimits.feedbackTextLength).optional(),
  glossSequence: z.array(islGlossItemSchema).max(apiRequestLimits.feedbackGlossItems).optional(),
  userComment: z.string().max(1000).optional(),
  suggestedCorrection: z.string().max(1000).optional(),
  createdAt: z.string()
});

export const interpretRequestSchema = z
  .object({
    sessionId: apiIdentifierSchema,
    source: transcriptSourceSchema,
    rawText: z.string().min(1).max(apiRequestLimits.transcriptLength),
    languageHint: languageHintSchema.optional(),
    simplificationLevel: simplificationLevelSchema.optional(),
    avatarSpeed: z.number().min(0.5).max(2).optional(),
    timestamp: timestampSchema.optional()
  })
  .strict();

export const sessionStartRequestSchema = z
  .object({
    source: transcriptSourceSchema,
    pageTitle: z.string().max(apiRequestLimits.pageTitleLength).optional(),
    url: z.string().max(apiRequestLimits.urlLength).url().optional()
  })
  .strict();

export const sessionEndRequestSchema = z
  .object({
    sessionId: apiIdentifierSchema,
    durationMs: z.number().int().nonnegative().max(apiRequestLimits.sessionDurationMs).optional()
  })
  .strict();

const feedbackGlossItemRequestSchema = islGlossItemSchema
  .omit({
    sourceSegmentId: true,
    planningActionId: true,
    sourceTokenIds: true,
    sourceSpan: true,
    clauseId: true,
    plannedAction: true,
    resolvedAction: true,
    provenance: true,
    reviewStatus: true
  })
  .extend({
    id: apiIdentifierSchema,
    token: z.string().min(1).max(apiRequestLimits.feedbackGlossTextLength),
    gloss: z.string().min(1).max(apiRequestLimits.feedbackGlossTextLength),
    signAssetId: apiIdentifierSchema.optional()
  })
  .strict();

export const feedbackSubmitRequestSchema = feedbackReportSchema
  .omit({ id: true, createdAt: true })
  .extend({
    sessionId: apiIdentifierSchema,
    interpretationId: apiIdentifierSchema,
    transcriptText: z.string().max(apiRequestLimits.feedbackTextLength).optional(),
    glossSequence: z
      .array(feedbackGlossItemRequestSchema)
      .max(apiRequestLimits.feedbackGlossItems)
      .optional()
  })
  .strict();

export const glossarySearchQuerySchema = z
  .object({
    q: z.string().max(apiRequestLimits.glossaryQueryLength).optional().default("")
  })
  .strict();

export const modelRouteParamsSchema = z
  .object({
    id: apiIdentifierSchema
  })
  .strict();

export const privacyPurgeRequestSchema = z.discriminatedUnion("scope", [
  z
    .object({
      scope: z.literal("session"),
      sessionId: apiIdentifierSchema,
      confirmation: z.literal("purge_session")
    })
    .strict(),
  z
    .object({
      scope: z.literal("all"),
      confirmation: z.literal("purge_all_local_data")
    })
    .strict()
]);

const privacyDeletionCountsSchema = z
  .object({
    sessions: z.number().int().nonnegative(),
    interpretations: z.number().int().nonnegative(),
    avatarQueues: z.number().int().nonnegative(),
    feedback: z.number().int().nonnegative(),
    rateLimitEntries: z.number().int().nonnegative()
  })
  .strict();

export const privacyPurgeResponseSchema = z.discriminatedUnion("scope", [
  z
    .object({
      scope: z.literal("session"),
      sessionId: apiIdentifierSchema,
      status: z.literal("purged"),
      deleted: privacyDeletionCountsSchema,
      rawAudioStored: z.literal(false),
      rawVideoStored: z.literal(false)
    })
    .strict(),
  z
    .object({
      scope: z.literal("all"),
      status: z.literal("purged"),
      deleted: privacyDeletionCountsSchema,
      rawAudioStored: z.literal(false),
      rawVideoStored: z.literal(false)
    })
    .strict()
]);

export const settingsSchema = z.object({
  outputMode: outputModeSchema.default("captions_avatar"),
  avatarName: z.enum(["Ananya", "Arjun", "Meera", "Kabir"]).default("Ananya"),
  avatarPosition: z
    .enum(["bottom_right", "bottom_left", "top_right", "side_panel"])
    .default("bottom_left"),
  avatarSize: z.enum(["small", "medium", "large"]).default("medium"),
  avatarSpeed: z.number().min(0.5).max(2).default(0.75),
  captionFontSize: z.number().min(14).max(28).default(18),
  captionLanguage: z.enum(["auto", "en", "hi", "hinglish"]).default("auto"),
  simplificationLevel: simplificationLevelSchema.default("standard"),
  showGlossDebug: z.boolean().default(false),
  saveHistory: z.boolean().default(false),
  feedbackSharing: z.boolean().default(true),
  privacyMode: z.boolean().default(true)
});

export const avatarStepKindSchema = z.enum(["sign", "caption", "fingerspell", "unknown"]);

export const motionPointSchema = z
  .object({
    x: z.number().finite(),
    y: z.number().finite(),
    visibility: z.number().finite().min(0).max(1).optional()
  })
  .strict();

export const motionFrameSchema = z
  .object({
    pose: z.array(motionPointSchema).max(33),
    leftHand: z.array(motionPointSchema).max(21),
    rightHand: z.array(motionPointSchema).max(21),
    face: z.array(motionPointSchema).max(18).optional()
  })
  .strict();

export const motionClipSchema = z
  .object({
    id: z.string().min(1),
    label: z.string().min(1),
    fps: z.number().finite().min(1).max(120),
    frames: z.array(motionFrameSchema).min(2).max(64)
  })
  .strict();

export const realtimePlaybackStatusSchema = z.enum(["realtime", "catching_up", "backlog"]);

export const realtimePlaybackBacklogSchema = z
  .object({
    policy: z.literal("bounded_realtime"),
    status: realtimePlaybackStatusSchema,
    historyStepCount: z.number().int().nonnegative(),
    pendingStepCount: z.number().int().nonnegative(),
    playbackStepCount: z.number().int().nonnegative(),
    deferredStepCount: z.number().int().nonnegative(),
    estimatedPendingMs: z.number().int().nonnegative(),
    playbackDurationMs: z.number().int().nonnegative(),
    maxPendingSteps: z.number().int().positive(),
    maxPendingMs: z.number().int().positive(),
    catchUpSpeed: z.number().min(0.5).max(2)
  })
  .strict();

export const avatarQueueStepSchema = z.object({
  id: z.string(),
  glossItemId: z.string(),
  kind: avatarStepKindSchema,
  label: z.string(),
  signAssetId: z.string().optional(),
  motionClipId: z.string().optional(),
  motionSource: z
    .enum(["include_dataset", "isign_research", "expert_reviewed_library"])
    .optional(),
  fingerspellingSource: z.literal("islrtc_official").optional(),
  expertReviewed: z.boolean().optional(),
  durationMs: z.number().int().nonnegative(),
  confidence: confidenceLevelSchema,
  sourceSegmentId: z.string().optional(),
  planningActionId: z.string().optional(),
  sourceTokenIds: z.array(z.string()).min(1).optional(),
  sourceSpan: sourceCharacterSpanSchema.optional(),
  clauseId: z.string().optional(),
  plannedAction: planningActionDecisionSchema.optional(),
  resolvedAction: planningActionDecisionSchema.optional(),
  provenance: planningProvenanceSchema.optional(),
  reviewStatus: planningReviewStatusSchema.optional()
});

export const avatarQueueSchema = z.object({
  id: z.string(),
  interpretationId: z.string(),
  speed: z.number().min(0.5).max(2),
  steps: z.array(avatarQueueStepSchema),
  motionClips: z.array(motionClipSchema).max(1024).optional(),
  backlog: realtimePlaybackBacklogSchema.optional(),
  honestFallbackText: z.string().optional(),
  motionReviewNotice: z.string().optional(),
  replayOfQueueId: z.string().optional(),
  createdAt: z.string()
});

export const modelStatusSchema = z.enum(["ready", "training", "unavailable"]);
export const modelEngineSchema = z.enum([
  "lexicon_ranker",
  "lexicon_pair_matcher",
  "motion_library",
  "caption_boundary_planner",
  "text_to_pose_research",
  "external_video_model",
  "keypoint_centroid",
  "keypoint_transformer"
]);

export const modelMetadataSchema = z.object({
  id: z.string(),
  version: z.string(),
  status: modelStatusSchema,
  engine: modelEngineSchema,
  trainedAt: z.string().optional(),
  trainingDataset: z.object({
    primaryDataset: z.string(),
    displayName: z.string(),
    recordCount: z.number().int().nonnegative(),
    classCount: z.number().int().nonnegative(),
    citationUrls: z.array(z.string().url())
  }),
  notes: z.array(z.string()).default([])
});

export const datasetSplitSchema = z.enum(["train", "val", "test"]);
export const signDatasetIdSchema = z.enum(["include", "isltranslate", "isign", "curated_mvp"]);

export const keypointPartSchema = z.enum(["pose", "left_hand", "right_hand", "face", "unknown"]);

export const keypointLandmarkSchema = z.object({
  part: keypointPartSchema.default("unknown"),
  index: z.number().int().nonnegative(),
  x: z.number().finite(),
  y: z.number().finite(),
  z: z.number().finite().optional(),
  visibility: z.number().finite().min(0).max(1).optional()
});

export const keypointFrameSchema = z.object({
  timestampMs: z.number().finite().nonnegative().optional(),
  landmarks: z.array(keypointLandmarkSchema).min(1)
});

export const keypointSequenceSchema = z.object({
  sampleId: z.string().min(1),
  label: z.string().min(1),
  gloss: z.string().min(1).optional(),
  datasetId: signDatasetIdSchema,
  split: datasetSplitSchema,
  signerHash: z.string().optional(),
  fps: z.number().positive().optional(),
  durationMs: z.number().nonnegative().optional(),
  extractorVersion: z.string().min(1).optional(),
  missingKeypointRatio: z.number().min(0).max(1).default(0),
  checksum: z.string().optional(),
  license: z.string().optional(),
  frames: z.array(keypointFrameSchema).min(1)
});

export const signAssetSchema = z.object({
  id: z.string().min(1),
  label: z.string().min(1),
  gloss: z.string().min(1),
  datasetId: signDatasetIdSchema,
  format: z.enum(["keypoints", "video", "animation_clip", "placeholder"]),
  uri: z.string().min(1),
  durationMs: z.number().int().nonnegative().optional(),
  checksum: z.string().optional(),
  license: z.string().optional()
});

export const motionCatalogArtifactSchema = z.object({
  schemaVersion: z.literal(1),
  clipCount: z.number().int().nonnegative(),
  clips: z.array(
    z.object({
      id: z.string().min(1),
      label: z.string().min(1),
      normalizedLabel: z.string().min(1),
      datasetId: signDatasetIdSchema,
      sourceSampleId: z.string().min(1),
      frameCount: z.number().int().positive(),
      fps: z.number().positive().optional(),
      durationMs: z.number().nonnegative().optional(),
      timingSource: z.string().min(1).optional(),
      expertReviewed: z.boolean()
    })
  )
});

export const videoModelPrototypeSchema = z.object({
  label: z.string().min(1),
  normalizedLabel: z.string().min(1),
  centroid: z.array(z.number()).min(1),
  sampleCount: z.number().int().positive()
});

export const videoModelArtifactSchema = z.object({
  schemaVersion: z.literal(1),
  metadata: modelMetadataSchema,
  labelMap: z
    .array(
      z
        .object({
          label: z.string().min(1),
          normalizedLabel: z.string().min(1),
          gloss: z.string().min(1)
        })
        .transform(repairNumericOnlyGloss)
        .refine(hasSemanticGloss, {
          path: ["gloss"],
          message: "Gloss must contain semantic text after class-number repair."
        })
    )
    .min(1),
  landmarkKeys: z.array(z.string().min(1)).min(1),
  featureSize: z.number().int().positive(),
  prototypes: z.array(videoModelPrototypeSchema).min(1),
  metrics: z
    .object({
      trainAccuracy: z.number().min(0).max(1).optional(),
      valAccuracy: z.number().min(0).max(1).optional(),
      testAccuracy: z.number().min(0).max(1).optional(),
      sampleCount: z.number().int().nonnegative(),
      classCount: z.number().int().nonnegative()
    })
    .optional()
});

export const temporalModelLabelSchema = z
  .object({
    index: z.number().int().nonnegative(),
    label: z.string().min(1),
    normalizedLabel: z.string().min(1),
    gloss: z.string().min(1),
    trainSamples: z.number().int().nonnegative(),
    valSamples: z.number().int().nonnegative(),
    testSamples: z.number().int().nonnegative(),
    motionClipId: z.string().min(1)
  })
  .transform(repairNumericOnlyGloss)
  .refine(hasSemanticGloss, {
    path: ["gloss"],
    message: "Gloss must contain semantic text after class-number repair."
  });

const temporalSplitMetricsSchema = z.object({
  accuracy: z.number().min(0).max(1),
  macro_f1: z.number().min(0).max(1),
  top3_accuracy: z.number().min(0).max(1),
  sample_count: z.number().int().nonnegative()
});

const temporalSelectiveMetricsSchema = z.object({
  threshold: z.number().min(0).max(1),
  coverage: z.number().min(0).max(1),
  acceptedAccuracy: z.number().min(0).max(1),
  acceptedCount: z.number().int().nonnegative(),
  fallbackCount: z.number().int().nonnegative()
});

export const temporalModelArtifactSchema = z
  .object({
    schemaVersion: z.literal(1),
    metadata: modelMetadataSchema,
    modelFile: z.string().min(1),
    modelBytes: z.number().int().positive(),
    labels: z.array(temporalModelLabelSchema).min(1),
    preprocessing: z
      .object({
        sequenceLength: z.number().int().positive(),
        inputSize: z.number().int().positive(),
        poseLandmarks: z.number().int().positive(),
        handLandmarksPerHand: z.number().int().positive(),
        coordinates: z.array(z.string()).min(1),
        frameWidth: z.number().positive(),
        frameHeight: z.number().positive()
      })
      .passthrough(),
    metrics: z.object({
      train: temporalSplitMetricsSchema,
      val: temporalSplitMetricsSchema,
      test: temporalSplitMetricsSchema,
      quantizedTest: temporalSplitMetricsSchema
    }),
    confidencePolicy: z
      .object({
        deployedThreshold: z.number().min(0).max(1),
        fallback: z.literal("caption"),
        coverage: z.number().min(0).max(1),
        acceptedAccuracy: z.number().min(0).max(1),
        acceptedCount: z.number().int().nonnegative(),
        fallbackCount: z.number().int().nonnegative(),
        calibration: z
          .object({
            split: z.literal("val"),
            sampleCount: z.number().int().positive(),
            targetAcceptedAccuracy: z.number().min(0).max(1),
            minimumCoverage: z.number().min(0).max(1),
            minimumAcceptedCount: z.number().int().positive(),
            selectionRule: z.string().min(1),
            candidateThresholds: z.array(z.number().min(0).max(1)).min(1),
            selectedValidationMetrics: temporalSelectiveMetricsSchema,
            targetMet: z.boolean(),
            testDataUsed: z.literal(false)
          })
          .optional(),
        thresholdFrozenBeforeTestEvaluation: z.literal(true).optional(),
        testEvaluation: z
          .object({
            split: z.literal("test"),
            role: z.literal("final untouched evaluation"),
            testDataUsedForPolicySelection: z.literal(false),
            selectiveMetrics: temporalSelectiveMetricsSchema
          })
          .optional()
      })
      .passthrough(),
    rawVideoStored: z.literal(false),
    rawAudioStored: z.literal(false)
  })
  .passthrough();

const videoInferenceLandmarkSchema = keypointLandmarkSchema.strict();

const videoInferenceFrameSchema = keypointFrameSchema
  .extend({
    landmarks: z
      .array(videoInferenceLandmarkSchema)
      .min(1)
      .max(apiRequestLimits.keypointLandmarksPerFrame)
  })
  .strict();

export const videoInferenceRequestSchema = z
  .object({
    frames: z.array(videoInferenceFrameSchema).min(1).max(apiRequestLimits.keypointFrames),
    topK: z.number().int().min(1).max(5).default(3)
  })
  .strict();

export const videoInferencePredictionSchema = z
  .object({
    label: z.string().min(1),
    normalizedLabel: z.string().min(1),
    gloss: z.string().min(1),
    confidence: z.number().min(0).max(1),
    distance: z.number().nonnegative()
  })
  .transform(repairNumericOnlyGloss)
  .refine(hasSemanticGloss, {
    path: ["gloss"],
    message: "Gloss must contain semantic text after class-number repair."
  });

export const videoInferenceResponseSchema = z
  .object({
    error: z.literal("recognition_model_unavailable").optional(),
    prediction: videoInferencePredictionSchema.nullable(),
    predictions: z.array(videoInferencePredictionSchema),
    model: modelMetadataSchema.nullable(),
    accepted: z.boolean(),
    fallback: z.enum(["none", "caption"]),
    confidenceThreshold: z.number().min(0).max(1).optional(),
    rawVideoStored: z.literal(false),
    notes: z.array(z.string()).default([])
  })
  .superRefine((response, context) => {
    if (response.accepted) {
      if (response.prediction === null) {
        context.addIssue({
          code: z.ZodIssueCode.custom,
          path: ["prediction"],
          message: "Accepted inference responses require a prediction."
        });
      }
      if (response.predictions.length === 0) {
        context.addIssue({
          code: z.ZodIssueCode.custom,
          path: ["predictions"],
          message: "Accepted inference responses require ranked predictions."
        });
      }
      if (response.model === null) {
        context.addIssue({
          code: z.ZodIssueCode.custom,
          path: ["model"],
          message: "Accepted inference responses require a loaded model."
        });
      }
      if (response.error !== undefined) {
        context.addIssue({
          code: z.ZodIssueCode.custom,
          path: ["error"],
          message: "Accepted inference responses cannot include an unavailable-model error."
        });
      }
      if (response.fallback !== "none") {
        context.addIssue({
          code: z.ZodIssueCode.custom,
          path: ["fallback"],
          message: "Accepted inference responses cannot use a fallback."
        });
      }
      return;
    }

    if (response.prediction !== null) {
      context.addIssue({
        code: z.ZodIssueCode.custom,
        path: ["prediction"],
        message: "Rejected inference responses must withhold the user-facing prediction."
      });
    }
    if (response.predictions.length !== 0) {
      context.addIssue({
        code: z.ZodIssueCode.custom,
        path: ["predictions"],
        message: "Rejected inference responses must withhold ranked class predictions."
      });
    }
    if (response.fallback !== "caption") {
      context.addIssue({
        code: z.ZodIssueCode.custom,
        path: ["fallback"],
        message: "Rejected inference responses must use the caption fallback."
      });
    }
    if (response.model === null && response.error !== "recognition_model_unavailable") {
      context.addIssue({
        code: z.ZodIssueCode.custom,
        path: ["error"],
        message: "Inference responses without a model must report recognition_model_unavailable."
      });
    }
    if (response.error === "recognition_model_unavailable" && response.model !== null) {
      context.addIssue({
        code: z.ZodIssueCode.custom,
        path: ["model"],
        message: "Unavailable recognition responses cannot advertise a loaded model."
      });
    }
  });

export const trainingJobSchema = z.object({
  id: z.string().min(1),
  type: z.enum([
    "dataset_validation",
    "keypoint_extraction",
    "video_model_training",
    "video_model_evaluation"
  ]),
  status: z.enum(["queued", "running", "succeeded", "failed"]),
  datasetId: signDatasetIdSchema.optional(),
  inputUri: z.string().optional(),
  outputUri: z.string().optional(),
  createdAt: z.string(),
  updatedAt: z.string(),
  message: z.string().optional()
});

export const interpretationResponseSchema = interpretationSchema.extend({
  avatarQueue: avatarQueueSchema,
  model: modelMetadataSchema.optional(),
  modelBackedGlossCount: z.number().int().nonnegative().default(0),
  motionBackedGlossCount: z.number().int().nonnegative().default(0)
});

export const sessionStartResponseSchema = z.object({
  sessionId: z.string(),
  status: z.literal("active"),
  rawAudioStored: z.literal(false)
});

export const sessionEndResponseSchema = z.object({
  sessionId: z.string().optional(),
  status: z.literal("ended"),
  rawAudioStored: z.literal(false).optional()
});

export const glossarySearchResponseSchema = z.object({
  results: z.array(glossaryEntrySchema)
});

export const modelRegistryResponseSchema = z.object({
  models: z.array(modelMetadataSchema),
  rawVideoStored: z.literal(false)
});

export const feedbackSubmitResponseSchema = z.object({
  feedbackId: z.string(),
  status: z.literal("saved"),
  rawAudioStored: z.literal(false)
});

export const extensionStatusSchema = z.enum(["inactive", "ready", "active", "processing", "error"]);

export const runtimePublicMessageTypeSchema = z.enum([
  "SESSION_STARTED",
  "SESSION_STOPPED",
  "TRANSCRIPT_SEGMENT",
  "INTERPRETATION_READY",
  "AVATAR_QUEUE_UPDATED",
  "AVATAR_PLAYBACK_UPDATED",
  "OVERLAY_SETTINGS_UPDATED",
  "ERROR_STATE"
]);

export const runtimeInternalMessageTypeSchema = z.enum(["CONTENT_READY"]);

export const sessionStartedRuntimeMessageSchema = z.object({
  type: z.literal("SESSION_STARTED"),
  sessionId: z.string(),
  targetTabId: z.number().int().nonnegative().optional()
});

export const sessionStoppedRuntimeMessageSchema = z.object({
  type: z.literal("SESSION_STOPPED"),
  sessionId: z.string().optional(),
  targetTabId: z.number().int().nonnegative().optional()
});

export const transcriptSegmentRuntimeMessageSchema = z.object({
  type: z.literal("TRANSCRIPT_SEGMENT"),
  segment: transcriptSegmentSchema,
  targetTabId: z.number().int().nonnegative().optional()
});

export const interpretationReadyRuntimeMessageSchema = z.object({
  type: z.literal("INTERPRETATION_READY"),
  interpretation: interpretationSchema,
  targetTabId: z.number().int().nonnegative().optional()
});

export const avatarQueueUpdatedRuntimeMessageSchema = z.object({
  type: z.literal("AVATAR_QUEUE_UPDATED"),
  queue: avatarQueueSchema,
  sessionId: z.string().optional(),
  targetTabId: z.number().int().nonnegative().optional()
});

export const avatarPlaybackUpdatedRuntimeMessageSchema = z
  .object({
    type: z.literal("AVATAR_PLAYBACK_UPDATED"),
    action: z.enum(["pause", "resume", "replay", "set_speed"]),
    speed: z.number().min(0.5).max(2).optional(),
    sessionId: z.string().optional(),
    targetTabId: z.number().int().nonnegative().optional()
  })
  .superRefine((message, context) => {
    if (message.action === "set_speed" && message.speed === undefined) {
      context.addIssue({
        code: z.ZodIssueCode.custom,
        path: ["speed"],
        message: "set_speed playback commands require a speed."
      });
    }
    if (message.action !== "set_speed" && message.speed !== undefined) {
      context.addIssue({
        code: z.ZodIssueCode.custom,
        path: ["speed"],
        message: "Only set_speed playback commands may include a speed."
      });
    }
  });

export const overlaySettingsUpdatedRuntimeMessageSchema = z.object({
  type: z.literal("OVERLAY_SETTINGS_UPDATED"),
  settings: settingsSchema,
  sessionId: z.string().optional(),
  targetTabId: z.number().int().nonnegative().optional()
});

export const errorStateRuntimeMessageSchema = z.object({
  type: z.literal("ERROR_STATE"),
  message: z.string(),
  sessionId: z.string().optional(),
  targetTabId: z.number().int().nonnegative().optional()
});

export const contentReadyRuntimeMessageSchema = z.object({
  type: z.literal("CONTENT_READY")
});

export const runtimePublicMessageSchema = z.union([
  sessionStartedRuntimeMessageSchema,
  sessionStoppedRuntimeMessageSchema,
  transcriptSegmentRuntimeMessageSchema,
  interpretationReadyRuntimeMessageSchema,
  avatarQueueUpdatedRuntimeMessageSchema,
  avatarPlaybackUpdatedRuntimeMessageSchema,
  overlaySettingsUpdatedRuntimeMessageSchema,
  errorStateRuntimeMessageSchema
]);

export const runtimeMessageSchema = z.union([
  contentReadyRuntimeMessageSchema,
  sessionStartedRuntimeMessageSchema,
  sessionStoppedRuntimeMessageSchema,
  transcriptSegmentRuntimeMessageSchema,
  interpretationReadyRuntimeMessageSchema,
  avatarQueueUpdatedRuntimeMessageSchema,
  avatarPlaybackUpdatedRuntimeMessageSchema,
  overlaySettingsUpdatedRuntimeMessageSchema,
  errorStateRuntimeMessageSchema
]);

export type TranscriptSource = z.infer<typeof transcriptSourceSchema>;
export type ConfidenceLevel = z.infer<typeof confidenceLevelSchema>;
export type LanguageHint = z.infer<typeof languageHintSchema>;
export type FallbackMode = z.infer<typeof fallbackSchema>;
export type SimplificationLevel = z.infer<typeof simplificationLevelSchema>;
export type SourceCharacterSpan = z.infer<typeof sourceCharacterSpanSchema>;
export type PlanningFeature = z.infer<typeof planningFeatureSchema>;
export type PlanningReviewStatus = z.infer<typeof planningReviewStatusSchema>;
export type PlanningProvenance = z.infer<typeof planningProvenanceSchema>;
export type PlanningActionKind = z.infer<typeof planningActionKindSchema>;
export type PlanningActionDecision = z.infer<typeof planningActionDecisionSchema>;
export type PlanningSourceToken = z.infer<typeof planningSourceTokenSchema>;
export type PlanningClause = z.infer<typeof planningClauseSchema>;
export type PlanningAction = z.infer<typeof planningActionSchema>;
export type InterpretationPlanningIR = z.infer<typeof interpretationPlanningIRSchema>;
export type PlanningIR = InterpretationPlanningIR;
export type OutputMode = z.infer<typeof outputModeSchema>;
export type TranscriptSegment = z.infer<typeof transcriptSegmentSchema>;
export type SimplifiedChunk = z.infer<typeof simplifiedChunkSchema>;
export type ISLGlossItem = z.infer<typeof islGlossItemSchema>;
export type ISLInterpretation = z.infer<typeof interpretationSchema>;
export type ISLInterpretationResponse = z.infer<typeof interpretationResponseSchema>;
export type ModelMetadata = z.infer<typeof modelMetadataSchema>;
export type DatasetSplit = z.infer<typeof datasetSplitSchema>;
export type SignDatasetId = z.infer<typeof signDatasetIdSchema>;
export type KeypointLandmark = z.infer<typeof keypointLandmarkSchema>;
export type KeypointFrame = z.infer<typeof keypointFrameSchema>;
export type KeypointSequence = z.infer<typeof keypointSequenceSchema>;
export type SignAsset = z.infer<typeof signAssetSchema>;
export type MotionCatalogArtifact = z.infer<typeof motionCatalogArtifactSchema>;
export type VideoModelArtifact = z.infer<typeof videoModelArtifactSchema>;
export type TemporalModelArtifact = z.infer<typeof temporalModelArtifactSchema>;
export type VideoInferenceRequest = z.infer<typeof videoInferenceRequestSchema>;
export type VideoInferenceResponse = z.infer<typeof videoInferenceResponseSchema>;
export type TrainingJob = z.infer<typeof trainingJobSchema>;
export type GlossaryEntry = z.infer<typeof glossaryEntrySchema>;
export type FeedbackReport = z.infer<typeof feedbackReportSchema>;
export type InterpretRequest = z.infer<typeof interpretRequestSchema>;
export type PrivacyPurgeRequest = z.infer<typeof privacyPurgeRequestSchema>;
export type PrivacyPurgeResponse = z.infer<typeof privacyPurgeResponseSchema>;
export type Settings = z.infer<typeof settingsSchema>;
export type AvatarStepKind = z.infer<typeof avatarStepKindSchema>;
export type RealtimePlaybackStatus = z.infer<typeof realtimePlaybackStatusSchema>;
export type RealtimePlaybackBacklog = z.infer<typeof realtimePlaybackBacklogSchema>;
export type AvatarQueueStep = z.infer<typeof avatarQueueStepSchema>;
export type MotionClip = z.infer<typeof motionClipSchema>;
export type MotionFrame = z.infer<typeof motionFrameSchema>;
export type MotionPoint = z.infer<typeof motionPointSchema>;
export type AvatarQueue = z.infer<typeof avatarQueueSchema>;
export type ExtensionStatus = z.infer<typeof extensionStatusSchema>;
export type RuntimePublicMessage = z.infer<typeof runtimePublicMessageSchema>;
export type RuntimeInternalMessage = z.infer<typeof contentReadyRuntimeMessageSchema>;
export type RuntimeMessage = z.infer<typeof runtimeMessageSchema>;
