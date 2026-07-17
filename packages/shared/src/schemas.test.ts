import { describe, expect, it } from "vitest";
import {
  apiRequestLimits,
  feedbackReportSchema,
  feedbackSubmitRequestSchema,
  feedbackSubmitResponseSchema,
  glossarySearchQuerySchema,
  glossarySearchResponseSchema,
  interpretRequestSchema,
  interpretationPlanningIRSchema,
  interpretationSchema,
  interpretationResponseSchema,
  keypointSequenceSchema,
  modelEngineSchema,
  modelMetadataSchema,
  modelRegistryResponseSchema,
  modelRouteParamsSchema,
  privacyPurgeRequestSchema,
  privacyPurgeResponseSchema,
  realtimePlaybackBacklogSchema,
  runtimeMessageSchema,
  sessionEndRequestSchema,
  sessionStartRequestSchema,
  settingsSchema,
  sessionStartResponseSchema,
  transcriptSourceSchema,
  trainingJobSchema,
  videoInferenceRequestSchema,
  videoInferenceResponseSchema,
  videoModelArtifactSchema
} from "./schemas";

describe("shared schemas", () => {
  it("validates a lossless pre-expert planning IR with source spans and detected features", () => {
    const planning = interpretationPlanningIRSchema.parse({
      id: "plan_1",
      sourceSegmentId: "segment_1000_4000",
      sourceText: "Are we not ready?",
      sourceTokens: [
        token("token_are", "Are", 0, 3),
        token("token_we", "we", 4, 6),
        token("token_not", "not", 7, 10),
        token("token_ready", "ready", 11, 16)
      ],
      clauses: [
        {
          id: "clause_1",
          sourceSpan: { start: 0, end: 17 },
          sourceTokenIds: ["token_are", "token_we", "token_not", "token_ready"],
          features: {
            isQuestion: true,
            hasNegation: true,
            questionCueTokenIds: ["token_are"],
            negationCueTokenIds: ["token_not"]
          }
        }
      ],
      actions: [
        action("action_are", "Are", 0, 3, "token_are", "fingerspell"),
        action("action_we", "we", 4, 6, "token_we", "fingerspell"),
        action("action_not", "not", 7, 10, "token_not", "fingerspell"),
        action("action_ready", "ready", 11, 16, "token_ready", "fingerspell")
      ],
      features: {
        isQuestion: true,
        hasNegation: true,
        questionCueTokenIds: ["token_are"],
        negationCueTokenIds: ["token_not"]
      },
      grammar: {
        sourceOrderPreserved: true,
        reordering: { applied: false, status: "expert_gated", reviewStatus: "pending_review" },
        nonManualSignals: { applied: false, status: "expert_gated", reviewStatus: "pending_review" }
      },
      reviewStatus: "pending_review"
    });

    expect(planning.sourceTokens.map((sourceToken) => sourceToken.text)).toEqual([
      "Are",
      "we",
      "not",
      "ready"
    ]);
    expect(planning.features).toMatchObject({ isQuestion: true, hasNegation: true });
    expect(planning.grammar.reordering.applied).toBe(false);
  });

  it("rejects planning IR actions that omit a source token", () => {
    const parsed = interpretationPlanningIRSchema.safeParse({
      id: "plan_incomplete",
      sourceSegmentId: "segment_1",
      sourceText: "hello world",
      sourceTokens: [token("token_hello", "hello", 0, 5), token("token_world", "world", 6, 11)],
      clauses: [
        {
          id: "clause_1",
          sourceSpan: { start: 0, end: 11 },
          sourceTokenIds: ["token_hello", "token_world"],
          features: emptyFeatures()
        }
      ],
      actions: [action("action_hello", "hello", 0, 5, "token_hello", "fingerspell")],
      features: emptyFeatures(),
      grammar: {
        sourceOrderPreserved: true,
        reordering: { applied: false, status: "expert_gated", reviewStatus: "pending_review" },
        nonManualSignals: { applied: false, status: "expert_gated", reviewStatus: "pending_review" }
      },
      reviewStatus: "pending_review"
    });

    expect(parsed.success).toBe(false);
  });

  it("validates concise bounded realtime backlog metadata", () => {
    const backlog = realtimePlaybackBacklogSchema.parse({
      policy: "bounded_realtime",
      status: "backlog",
      historyStepCount: 20,
      pendingStepCount: 12,
      playbackStepCount: 8,
      deferredStepCount: 4,
      estimatedPendingMs: 18_000,
      playbackDurationMs: 11_500,
      maxPendingSteps: 12,
      maxPendingMs: 12_000,
      catchUpSpeed: 2
    });

    expect(backlog.historyStepCount).toBe(20);
    expect(backlog.deferredStepCount).toBe(4);
  });

  it("parses a valid interpretation response with verified and fallback gloss items", () => {
    const parsed = interpretationSchema.parse({
      id: "interp_1",
      sessionId: "session_1",
      chunkId: "chunk_1",
      originalText: "Artificial intelligence can help education.",
      cleanedText: "Artificial intelligence can help education.",
      simplifiedText: "AI can help education.",
      keyPoints: ["AI can help education"],
      detectedTerms: ["artificial intelligence", "education"],
      glossSequence: [
        {
          id: "gloss_1",
          token: "artificial intelligence",
          gloss: "AI",
          signAssetId: "sign_ai_001",
          isVerified: true,
          fallback: "none",
          confidence: "high"
        },
        {
          id: "gloss_2",
          token: "unmapped term",
          gloss: "UNMAPPED TERM",
          isVerified: false,
          fallback: "fingerspell",
          confidence: "low"
        }
      ],
      confidence: "medium",
      warnings: ["Some terms use fallback output."],
      createdAt: "2026-07-07T00:00:00.000Z"
    });

    expect(parsed.glossSequence[1]?.fallback).toBe("fingerspell");
  });

  it("parses an interpretation response with an avatar queue", () => {
    const interpretation = interpretationSchema.parse({
      id: "interp_1",
      sessionId: "session_1",
      chunkId: "chunk_1",
      originalText: "Today we learn.",
      cleanedText: "Today we learn.",
      simplifiedText: "Today topic: learning.",
      keyPoints: ["Today topic: learning"],
      detectedTerms: ["today", "learn"],
      glossSequence: [
        {
          id: "gloss_1",
          token: "today",
          gloss: "TODAY",
          signAssetId: "sign_today_001",
          isVerified: true,
          fallback: "none",
          confidence: "high"
        }
      ],
      confidence: "high",
      warnings: [],
      createdAt: "2026-07-07T00:00:00.000Z"
    });

    expect(
      runtimeMessageSchema.parse({
        type: "AVATAR_QUEUE_UPDATED",
        queue: {
          id: "queue_1",
          interpretationId: interpretation.id,
          speed: 1,
          steps: [
            {
              id: "step_1",
              glossItemId: "gloss_1",
              kind: "sign",
              label: "TODAY",
              signAssetId: "sign_today_001",
              durationMs: 1100,
              confidence: "high"
            }
          ],
          createdAt: "2026-07-07T00:00:00.000Z"
        }
      }).type
    ).toBe("AVATAR_QUEUE_UPDATED");
  });

  it("validates model metadata and interpretation responses with model provenance", () => {
    const metadata = modelMetadataSchema.parse({
      id: "isl-lexicon-2026-07-07",
      version: "0.1.0",
      status: "ready",
      engine: "lexicon_ranker",
      trainedAt: "2026-07-07T00:00:00.000Z",
      trainingDataset: {
        primaryDataset: "include",
        displayName: "INCLUDE + ISLTranslate seed",
        recordCount: 12,
        classCount: 12,
        citationUrls: ["https://github.com/AI4Bharat/INCLUDE"]
      }
    });

    const response = interpretationResponseSchema.parse({
      id: "interp_1",
      sessionId: "session_1",
      chunkId: "chunk_1",
      originalText: "The Constitution promises justice.",
      cleanedText: "The Constitution promises justice.",
      simplifiedText: "The Constitution promises justice.",
      keyPoints: ["The Constitution promises justice"],
      detectedTerms: ["constitution", "justice"],
      glossSequence: [
        {
          id: "gloss_1",
          token: "constitution",
          gloss: "CONSTITUTION",
          signAssetId: "include_constitution_001",
          isVerified: true,
          fallback: "none",
          confidence: "high"
        }
      ],
      confidence: "high",
      warnings: [],
      createdAt: "2026-07-07T00:00:00.000Z",
      avatarQueue: {
        id: "queue_1",
        interpretationId: "interp_1",
        speed: 1,
        steps: [
          {
            id: "step_1",
            glossItemId: "gloss_1",
            kind: "sign",
            label: "CONSTITUTION",
            signAssetId: "include_constitution_001",
            durationMs: 1100,
            confidence: "high"
          }
        ],
        createdAt: "2026-07-07T00:00:00.000Z"
      },
      model: metadata,
      modelBackedGlossCount: 1
    });

    expect(response.model?.trainingDataset.primaryDataset).toBe("include");
    expect(response.modelBackedGlossCount).toBe(1);
  });

  it("accepts the lexical pair matcher model engine", () => {
    expect(modelEngineSchema.parse("lexicon_pair_matcher")).toBe("lexicon_pair_matcher");
  });

  it("validates video/keypoint model contracts without raw video storage", () => {
    const sequence = keypointSequenceSchema.parse({
      sampleId: "include_today_001",
      label: "73. Today",
      gloss: "TODAY",
      datasetId: "include",
      split: "train",
      signerHash: "signer_hash_1",
      fps: 25,
      durationMs: 1240,
      extractorVersion: "mediapipe-hands-blazepose-0.1",
      missingKeypointRatio: 0,
      checksum: "sha256-fixture",
      license: "INCLUDE dataset terms",
      frames: [
        {
          timestampMs: 0,
          landmarks: [
            { part: "right_hand", index: 0, x: 0.55, y: 0.26, visibility: 0.98 },
            { part: "right_hand", index: 1, x: 0.59, y: 0.36, visibility: 0.96 }
          ]
        }
      ]
    });

    const model = videoModelArtifactSchema.parse({
      schemaVersion: 1,
      metadata: {
        id: "isl-keypoint-2026-07-07",
        version: "0.1.0",
        status: "ready",
        engine: "keypoint_centroid",
        trainedAt: "2026-07-07T00:00:00.000Z",
        trainingDataset: {
          primaryDataset: "include",
          displayName: "INCLUDE keypoint fixture",
          recordCount: 1,
          classCount: 1,
          citationUrls: ["https://huggingface.co/datasets/ai4bharat/INCLUDE"]
        }
      },
      labelMap: [{ label: sequence.label, normalizedLabel: "today", gloss: "TODAY" }],
      landmarkKeys: ["right_hand:0", "right_hand:1"],
      featureSize: 6,
      prototypes: [
        {
          label: sequence.label,
          normalizedLabel: "today",
          centroid: [0, 0, 0, 1, 1, 0],
          sampleCount: 1
        }
      ],
      metrics: { sampleCount: 1, classCount: 1, trainAccuracy: 1 }
    });

    const request = videoInferenceRequestSchema.parse({ frames: sequence.frames, topK: 1 });
    const response = videoInferenceResponseSchema.parse({
      prediction: {
        label: "73. Today",
        normalizedLabel: "today",
        gloss: "TODAY",
        confidence: 0.99,
        distance: 0
      },
      predictions: [
        {
          label: "73. Today",
          normalizedLabel: "today",
          gloss: "TODAY",
          confidence: 0.99,
          distance: 0
        }
      ],
      model: model.metadata,
      accepted: true,
      fallback: "none",
      rawVideoStored: false
    });

    expect(request.frames).toHaveLength(1);
    expect(response.rawVideoStored).toBe(false);
  });

  it("withholds every class prediction from rejected inference responses", () => {
    const model = modelMetadataSchema.parse({
      id: "isl-keypoint-2026-07-13",
      version: "0.1.0",
      status: "ready",
      engine: "keypoint_centroid",
      trainingDataset: {
        primaryDataset: "include",
        displayName: "INCLUDE keypoints",
        recordCount: 1649,
        classCount: 101,
        citationUrls: ["https://huggingface.co/datasets/ai4bharat/INCLUDE"]
      }
    });
    const response = videoInferenceResponseSchema.parse({
      prediction: null,
      predictions: [],
      model,
      accepted: false,
      fallback: "caption",
      rawVideoStored: false
    });

    expect(response.prediction).toBeNull();
    expect(response.predictions).toEqual([]);
    expect(() =>
      videoInferenceResponseSchema.parse({
        ...response,
        prediction: {
          label: "83. Afternoon",
          normalizedLabel: "afternoon",
          gloss: "AFTERNOON",
          confidence: 0.31,
          distance: 0.69
        }
      })
    ).toThrow(/withhold the user-facing prediction/);
    expect(() =>
      videoInferenceResponseSchema.parse({
        ...response,
        predictions: [
          {
            label: "83. Afternoon",
            normalizedLabel: "afternoon",
            gloss: "AFTERNOON",
            confidence: 0.31,
            distance: 0.69
          }
        ]
      })
    ).toThrow(/withhold ranked class predictions/);

    const unavailable = videoInferenceResponseSchema.parse({
      error: "recognition_model_unavailable",
      prediction: null,
      predictions: [],
      model: null,
      accepted: false,
      fallback: "caption",
      rawVideoStored: false
    });
    expect(unavailable.model).toBeNull();
  });

  it("rejects non-finite keypoint coordinates", () => {
    expect(() =>
      videoInferenceRequestSchema.parse({
        frames: [
          {
            timestampMs: 0,
            landmarks: [{ part: "pose", index: 0, x: Number.NaN, y: 0.5 }]
          }
        ]
      })
    ).toThrow();
  });

  it("validates local training job metadata", () => {
    const job = trainingJobSchema.parse({
      id: "job_1",
      type: "video_model_training",
      status: "queued",
      datasetId: "include",
      inputUri: "data/isl/keypoints.sample.json",
      outputUri: "data/models/isl-keypoint-model.json",
      createdAt: "2026-07-07T00:00:00.000Z",
      updatedAt: "2026-07-07T00:00:00.000Z"
    });

    expect(job.type).toBe("video_model_training");
  });

  it("rejects feedback without an allowed reason", () => {
    expect(() =>
      feedbackReportSchema.parse({
        id: "feedback_1",
        sessionId: "session_1",
        interpretationId: "interp_1",
        reason: "bad",
        createdAt: "2026-07-07T00:00:00.000Z"
      })
    ).toThrow();
  });

  it("requires feedback timestamp metadata", () => {
    expect(() =>
      feedbackReportSchema.parse({
        id: "feedback_1",
        sessionId: "session_1",
        interpretationId: "interp_1",
        reason: "wrong_sign",
        createdAt: "2026-07-07T00:00:00.000Z"
      })
    ).toThrow();
  });

  it("defaults settings to privacy-forward output", () => {
    const parsed = settingsSchema.parse({});

    expect(parsed.outputMode).toBe("captions_avatar");
    expect(parsed.avatarName).toBe("Ananya");
    expect(parsed.avatarPosition).toBe("bottom_left");
    expect(parsed.saveHistory).toBe(false);
    expect(parsed.privacyMode).toBe(true);
  });

  it("validates an interpret request with timestamp metadata", () => {
    const parsed = interpretRequestSchema.parse({
      sessionId: "session_1",
      source: "youtube_captions",
      rawText: "Today we learn about equal opportunity.",
      languageHint: "en",
      simplificationLevel: "strong",
      timestamp: {
        startMs: 1000,
        endMs: 4000
      }
    });

    expect(parsed.timestamp?.endMs).toBe(4000);
    expect(parsed.simplificationLevel).toBe("strong");
  });

  it("enforces route-specific text, identifier, and collection limits", () => {
    const validInterpretRequest = {
      sessionId: "session_1",
      source: "youtube_captions" as const,
      rawText: "a".repeat(apiRequestLimits.transcriptLength)
    };
    expect(interpretRequestSchema.safeParse(validInterpretRequest).success).toBe(true);
    expect(
      interpretRequestSchema.safeParse({
        ...validInterpretRequest,
        rawText: `${validInterpretRequest.rawText}a`
      }).success
    ).toBe(false);
    expect(
      interpretRequestSchema.safeParse({ ...validInterpretRequest, rawVideo: "not accepted" })
        .success
    ).toBe(false);
    expect(
      interpretRequestSchema.safeParse({
        ...validInterpretRequest,
        timestamp: { startMs: 10, endMs: 9 }
      }).success
    ).toBe(false);

    expect(
      sessionStartRequestSchema.safeParse({
        source: "mock",
        pageTitle: "p".repeat(apiRequestLimits.pageTitleLength + 1)
      }).success
    ).toBe(false);
    expect(
      sessionEndRequestSchema.safeParse({
        sessionId: "session_1",
        durationMs: apiRequestLimits.sessionDurationMs + 1
      }).success
    ).toBe(false);
    expect(
      glossarySearchQuerySchema.safeParse({
        q: "q".repeat(apiRequestLimits.glossaryQueryLength + 1)
      }).success
    ).toBe(false);
    expect(
      modelRouteParamsSchema.safeParse({
        id: "m".repeat(apiRequestLimits.identifierLength + 1)
      }).success
    ).toBe(false);

    const feedbackItem = {
      id: "gloss_1",
      token: "today",
      gloss: "TODAY",
      isVerified: true,
      fallback: "none" as const,
      confidence: "high" as const
    };
    const feedbackRequest = {
      sessionId: "session_1",
      interpretationId: "interp_1",
      reason: "wrong_sign" as const,
      timestamp: { startMs: 0, endMs: 1000 }
    };
    expect(
      feedbackSubmitRequestSchema.safeParse({
        ...feedbackRequest,
        transcriptText: "t".repeat(apiRequestLimits.feedbackTextLength + 1)
      }).success
    ).toBe(false);
    expect(
      feedbackSubmitRequestSchema.safeParse({
        ...feedbackRequest,
        glossSequence: Array.from(
          { length: apiRequestLimits.feedbackGlossItems + 1 },
          (_, index) => ({ ...feedbackItem, id: `gloss_${index}` })
        )
      }).success
    ).toBe(false);
    expect(
      feedbackSubmitRequestSchema.safeParse({
        ...feedbackRequest,
        glossSequence: [
          {
            ...feedbackItem,
            sourceSegmentId: "segment_not_allowed_in_feedback"
          }
        ]
      }).success
    ).toBe(false);

    const frame = {
      timestampMs: 0,
      landmarks: [{ part: "pose" as const, index: 0, x: 0.5, y: 0.5 }]
    };
    expect(
      videoInferenceRequestSchema.safeParse({
        frames: Array.from({ length: apiRequestLimits.keypointFrames + 1 }, () => frame)
      }).success
    ).toBe(false);
    expect(
      videoInferenceRequestSchema.safeParse({
        frames: [
          {
            ...frame,
            landmarks: Array.from(
              { length: apiRequestLimits.keypointLandmarksPerFrame + 1 },
              (_, index) => ({ part: "pose" as const, index, x: 0.5, y: 0.5 })
            )
          }
        ]
      }).success
    ).toBe(false);
  });

  it("requires an exact privacy purge contract and preserves raw-media guarantees", () => {
    expect(
      privacyPurgeRequestSchema.parse({
        scope: "session",
        sessionId: "session_1",
        confirmation: "purge_session"
      }).scope
    ).toBe("session");
    expect(
      privacyPurgeRequestSchema.parse({
        scope: "all",
        confirmation: "purge_all_local_data"
      }).scope
    ).toBe("all");
    expect(
      privacyPurgeRequestSchema.safeParse({
        scope: "all",
        confirmation: "purge_all",
        sessionId: "session_1"
      }).success
    ).toBe(false);

    const response = privacyPurgeResponseSchema.parse({
      scope: "session",
      sessionId: "session_1",
      status: "purged",
      deleted: {
        sessions: 1,
        interpretations: 1,
        avatarQueues: 1,
        feedback: 1,
        rateLimitEntries: 0
      },
      rawAudioStored: false,
      rawVideoStored: false
    });
    expect(response.rawAudioStored).toBe(false);
    expect(response.rawVideoStored).toBe(false);
  });

  it("validates extension API response envelopes", () => {
    const session = sessionStartResponseSchema.parse({
      sessionId: "session_1",
      status: "active",
      rawAudioStored: false
    });
    const models = modelRegistryResponseSchema.parse({
      models: [
        {
          id: "isl-lexicon-1",
          version: "0.1.0",
          status: "ready",
          engine: "lexicon_ranker",
          trainingDataset: {
            primaryDataset: "include",
            displayName: "INCLUDE + iSign text",
            recordCount: 600,
            classCount: 600,
            citationUrls: ["https://github.com/AI4Bharat/INCLUDE"]
          }
        }
      ],
      rawVideoStored: false
    });
    const glossary = glossarySearchResponseSchema.parse({ results: [] });
    const feedback = feedbackSubmitResponseSchema.parse({
      feedbackId: "feedback_1",
      status: "saved",
      rawAudioStored: false
    });

    expect(session.rawAudioStored).toBe(false);
    expect(models.models).toHaveLength(1);
    expect(glossary.results).toEqual([]);
    expect(feedback.status).toBe("saved");
  });

  it("rejects deferred tab audio as a transcript source", () => {
    expect(transcriptSourceSchema.safeParse("tab_audio").success).toBe(false);
    expect(() =>
      interpretRequestSchema.parse({
        sessionId: "session_1",
        source: "tab_audio",
        rawText: "Audio capture is not part of the MVP."
      })
    ).toThrow();
  });

  it("validates public extension runtime messages", () => {
    const segment = {
      id: "segment_1",
      sessionId: "session_1",
      source: "youtube_captions",
      rawText: "Artificial intelligence can help education.",
      cleanedText: "Artificial intelligence can help education.",
      languageHint: "en",
      startMs: 1000,
      endMs: 4000,
      confidence: 0.92,
      createdAt: "2026-07-07T00:00:00.000Z"
    };
    const interpretation = interpretationSchema.parse({
      id: "interp_1",
      sessionId: "session_1",
      chunkId: "chunk_1",
      originalText: "Artificial intelligence can help education.",
      cleanedText: "Artificial intelligence can help education.",
      simplifiedText: "AI can help education.",
      keyPoints: ["AI can help education"],
      detectedTerms: ["artificial intelligence", "education"],
      glossSequence: [
        {
          id: "gloss_1",
          token: "artificial intelligence",
          gloss: "AI",
          signAssetId: "sign_ai_001",
          isVerified: true,
          fallback: "none",
          confidence: "high"
        }
      ],
      confidence: "high",
      warnings: [],
      createdAt: "2026-07-07T00:00:00.000Z"
    });
    const queue = {
      id: "queue_1",
      interpretationId: interpretation.id,
      speed: 1,
      steps: [
        {
          id: "step_1",
          glossItemId: "gloss_1",
          kind: "sign",
          label: "AI",
          signAssetId: "sign_ai_001",
          durationMs: 1100,
          confidence: "high"
        }
      ],
      createdAt: "2026-07-07T00:00:00.000Z"
    };

    const messages = [
      { type: "SESSION_STARTED", sessionId: "session_1" },
      { type: "SESSION_STOPPED", sessionId: "session_1" },
      { type: "TRANSCRIPT_SEGMENT", segment },
      { type: "INTERPRETATION_READY", interpretation },
      { type: "AVATAR_QUEUE_UPDATED", queue },
      { type: "AVATAR_PLAYBACK_UPDATED", action: "set_speed", speed: 1.5 },
      { type: "OVERLAY_SETTINGS_UPDATED", settings: settingsSchema.parse({}) },
      { type: "ERROR_STATE", message: "Unable to process captions." }
    ];

    expect(messages.map((message) => runtimeMessageSchema.parse(message).type)).toEqual([
      "SESSION_STARTED",
      "SESSION_STOPPED",
      "TRANSCRIPT_SEGMENT",
      "INTERPRETATION_READY",
      "AVATAR_QUEUE_UPDATED",
      "AVATAR_PLAYBACK_UPDATED",
      "OVERLAY_SETTINGS_UPDATED",
      "ERROR_STATE"
    ]);
  });

  it("requires speed only for set-speed playback commands", () => {
    expect(
      runtimeMessageSchema.safeParse({ type: "AVATAR_PLAYBACK_UPDATED", action: "set_speed" })
        .success
    ).toBe(false);
    expect(
      runtimeMessageSchema.safeParse({ type: "AVATAR_PLAYBACK_UPDATED", action: "pause", speed: 1 })
        .success
    ).toBe(false);
    expect(
      runtimeMessageSchema.safeParse({ type: "AVATAR_PLAYBACK_UPDATED", action: "replay" }).success
    ).toBe(true);
  });

  it("validates the internal content-ready runtime message", () => {
    expect(runtimeMessageSchema.parse({ type: "CONTENT_READY" }).type).toBe("CONTENT_READY");
  });
});

function token(id: string, text: string, start: number, end: number) {
  return {
    id,
    text,
    normalizedText: text.toLocaleLowerCase(),
    sourceSpan: { start, end },
    clauseId: "clause_1"
  };
}

function action(
  id: string,
  sourceText: string,
  start: number,
  end: number,
  sourceTokenId: string,
  kind: "fingerspell"
) {
  return {
    id,
    sourceText,
    sourceSpan: { start, end },
    sourceTokenIds: [sourceTokenId],
    clauseId: "clause_1",
    plannedAction: { kind, label: sourceText.toLocaleUpperCase() },
    resolvedAction: { kind, label: sourceText.toLocaleUpperCase() },
    provenance: { source: "deterministic_fallback" },
    reviewStatus: "unreviewed"
  };
}

function emptyFeatures() {
  return {
    isQuestion: false,
    hasNegation: false,
    questionCueTokenIds: [],
    negationCueTokenIds: []
  };
}
