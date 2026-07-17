import cors from "@fastify/cors";
import { createAvatarQueue } from "@signsaarthi/avatar-engine";
import {
  interpretTranscript,
  seedGlossary,
  type InterpretInput
} from "@signsaarthi/isl-engine";
import {
  createLexicalFallbackResolver,
  defaultISLDatasetManifest,
  islLexiconModelSchema,
  lexicalMatcherArtifactSchema,
  mergeGlossaryEntries,
  trainISLLexiconModel,
  type ISLLexiconModel,
  type LexicalMatcherArtifact
} from "@signsaarthi/isl-model";
import {
  inferVideoKeypoints,
  videoModelArtifactSchema,
  type VideoKeypointModel
} from "@signsaarthi/isl-video-model";
import {
  feedbackSubmitRequestSchema,
  glossarySearchQuerySchema,
  interpretRequestSchema,
  modelRouteParamsSchema,
  privacyPurgeRequestSchema,
  privacyPurgeResponseSchema,
  type ModelMetadata,
  sessionEndRequestSchema,
  sessionStartRequestSchema,
  videoInferenceRequestSchema,
  videoInferenceResponseSchema
} from "@signsaarthi/shared";
import Fastify from "fastify";
import type { FastifyInstance } from "fastify";
import { existsSync, readFileSync } from "node:fs";
import { resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { ZodError } from "zod";
import { createMemoryStore, type SessionRecord } from "./store/memoryStore.js";
import { loadTemporalModelRuntime, type TemporalModelRuntime } from "./temporalModel.js";
import { loadResearchMotionLibrary } from "./researchMotionLibrary.js";
import { loadResearchSentencePlanner } from "./researchSentencePlanner.js";
import { loadResearchTextToPoseArtifact } from "./researchTextToPoseArtifact.js";

const DEFAULT_EXTENSION_ORIGIN = "chrome-extension://kgjeaadkknfadmlipfjaekalpgbcnela";
const CLIENT_HEADER = "extension-v1";
const RATE_LIMIT_WINDOW_MS = 60_000;
const RATE_LIMIT_MAX_REQUESTS = 180;
const RATE_LIMIT_MAX_TRACKED_CLIENTS = 4_096;

type RateLimitOptions = {
  windowMs?: number;
  maxRequests?: number;
  maxTrackedClients?: number;
  now?: () => number;
};

export type ServerOptions = {
  requireClientHeader?: boolean;
  allowedExtensionOrigins?: string[];
  allowedDevPreviewOrigins?: string[];
  recognitionModelLoading?: boolean;
  researchMotionLoading?: boolean;
  researchSentencePlannerLoading?: boolean;
  researchTextToPoseLoading?: boolean;
  rateLimit?: RateLimitOptions;
};

export function buildServer(options: ServerOptions = {}): FastifyInstance {
  const app = Fastify({ logger: false, bodyLimit: 1_000_000 });
  const store = createMemoryStore();
  const islModel = loadISLModel();
  const textMatcher = loadTextMatcher();
  const researchMotionLoading = options.researchMotionLoading ?? process.env.NODE_ENV !== "test";
  const researchMotionLibrary = researchMotionLoading ? loadResearchMotionLibrary() : undefined;
  const researchSentencePlannerLoading =
    options.researchSentencePlannerLoading ?? process.env.NODE_ENV !== "test";
  const researchSentencePlanner = researchSentencePlannerLoading
    ? loadResearchSentencePlanner()
    : undefined;
  const researchTextToPoseLoading =
    options.researchTextToPoseLoading ?? process.env.NODE_ENV !== "test";
  const researchTextToPoseArtifact = researchTextToPoseLoading
    ? loadResearchTextToPoseArtifact()
    : undefined;
  const recognitionModelLoading = options.recognitionModelLoading ?? true;
  const videoModel = recognitionModelLoading ? loadVideoModel() : undefined;
  const temporalModel = recognitionModelLoading ? loadTemporalModelRuntime() : undefined;
  const readyTemporalModel =
    temporalModel?.artifact.metadata.status === "ready" ? temporalModel : undefined;
  const readyVideoModel = videoModel?.metadata.status === "ready" ? videoModel : undefined;
  const activeKeypointModel = readyTemporalModel?.artifact.metadata ?? readyVideoModel?.metadata;
  const reportedKeypointModel =
    activeKeypointModel ?? temporalModel?.artifact.metadata ?? videoModel?.metadata;
  const loadedModels = uniqueModelMetadata([
    islModel.metadata,
    ...(textMatcher ? [textMatcher.metadata] : []),
    ...(researchMotionLibrary ? [researchMotionLibrary.metadata] : []),
    ...(researchSentencePlanner ? [researchSentencePlanner.metadata] : []),
    ...(researchTextToPoseArtifact ? [researchTextToPoseArtifact.metadata] : []),
    ...(temporalModel ? [temporalModel.artifact.metadata] : []),
    ...(videoModel ? [videoModel.metadata] : [])
  ]);
  const mergedGlossary = mergeGlossaryEntries(islModel.glossaryEntries, seedGlossary);
  const resolveUnmatchedToken =
    textMatcher?.metadata.status === "ready"
      ? createLexicalFallbackResolver(mergedGlossary, textMatcher)
      : undefined;
  const modelTerms = new Set(
    islModel.glossaryEntries.flatMap((entry) => [entry.term, ...entry.aliases]).map(normalizeTerm)
  );
  const lexiconModelReady = islModel.metadata.status === "ready";
  const recognitionModelReady = activeKeypointModel !== undefined;
  const allowedExtensionOrigins = configuredOrigins(
    options.allowedExtensionOrigins,
    "SIGNSAARTHI_EXTENSION_ORIGINS",
    [DEFAULT_EXTENSION_ORIGIN],
    isExtensionOrigin
  );
  const allowedDevPreviewOrigins = configuredOrigins(
    options.allowedDevPreviewOrigins,
    "SIGNSAARTHI_DEV_PREVIEW_ORIGINS",
    [],
    isDevPreviewOrigin
  );
  const rateLimitWindowMs = positiveInteger(
    options.rateLimit?.windowMs ?? RATE_LIMIT_WINDOW_MS,
    "rate-limit window"
  );
  const rateLimitMaxRequests = positiveInteger(
    options.rateLimit?.maxRequests ?? RATE_LIMIT_MAX_REQUESTS,
    "rate-limit request count"
  );
  const rateLimitMaxTrackedClients = positiveInteger(
    options.rateLimit?.maxTrackedClients ?? RATE_LIMIT_MAX_TRACKED_CLIENTS,
    "rate-limit tracked-client count"
  );
  const rateLimitNow = options.rateLimit?.now ?? Date.now;
  const requestWindows = new Map<string, { count: number; startedAt: number }>();

  void app.register(cors, {
    origin(origin, callback) {
      const allowed =
        !origin || allowedExtensionOrigins.has(origin) || allowedDevPreviewOrigins.has(origin);
      callback(null, allowed);
    }
  });

  app.addHook("onRequest", async (request, reply) => {
    const path = request.url.split("?", 1)[0] ?? request.url;
    if (request.method === "OPTIONS" || !path.startsWith("/api/")) {
      return;
    }
    const authenticationRequired =
      options.requireClientHeader === true || path === "/api/privacy/purge";
    if (!authenticationRequired) {
      return;
    }
    if (request.headers["x-signsaarthi-client"] !== CLIENT_HEADER) {
      await reply.status(401).send({ error: "unauthorized_client" });
      return reply;
    }

    const now = rateLimitNow();
    pruneExpiredRequestWindows(requestWindows, now, rateLimitWindowMs);
    const currentWindow = requestWindows.get(request.ip);
    if (!currentWindow) {
      if (requestWindows.size >= rateLimitMaxTrackedClients) {
        reply.header("retry-after", Math.ceil(rateLimitWindowMs / 1000));
        await reply.status(429).send({ error: "rate_limit_exceeded" });
        return reply;
      }
      requestWindows.set(request.ip, { count: 1, startedAt: now });
      return;
    }

    if (currentWindow.count >= rateLimitMaxRequests) {
      const retryAfterMs = Math.max(1, rateLimitWindowMs - (now - currentWindow.startedAt));
      reply.header("retry-after", Math.ceil(retryAfterMs / 1000));
      await reply.status(429).send({ error: "rate_limit_exceeded" });
      return reply;
    }
    currentWindow.count += 1;
  });

  app.setErrorHandler((error, request, reply) => {
    if (error instanceof ZodError) {
      void reply.status(400).send({ error: "validation_error", issues: error.issues });
      return;
    }
    const statusCode = getErrorStatusCode(error);
    if (statusCode && statusCode >= 400 && statusCode < 500) {
      const code = statusCode === 413 ? "payload_too_large" : "invalid_request";
      void reply.status(statusCode).send({ error: code });
      return;
    }
    request.log.error({ err: error }, "Unhandled SignSaarthi API error");
    void reply.status(500).send({ error: "internal_error" });
  });

  app.get("/health", async (_request, reply) => {
    return reply.send({
      status: "ok",
      modelsReady: lexiconModelReady,
      lexiconModelReady,
      recognitionModelReady,
      researchMotionReady: researchMotionLibrary !== undefined,
      researchMotionClipCount: researchMotionLibrary?.clipCount ?? 0,
      captionBoundaryPlannerReady: researchSentencePlanner !== undefined,
      textToPoseResearchArtifactReady: researchTextToPoseArtifact !== undefined,
      textToPoseRuntimeReady: false,
      rawAudioStored: false,
      rawVideoStored: false,
      runtimeId: process.env.SIGNSAARTHI_RUNTIME_ID ?? "development"
    });
  });

  app.post("/api/session/start", async (request, reply) => {
    const input = sessionStartRequestSchema.parse(request.body);
    const sessionId = store.nextId("session");
    const sessionRecord: SessionRecord = {
      id: sessionId,
      source: input.source,
      status: "active",
      startedAt: new Date().toISOString(),
      rawAudioStored: false
    };
    if (input.pageTitle) {
      sessionRecord.pageTitle = input.pageTitle;
    }
    if (input.url) {
      sessionRecord.url = input.url;
    }
    store.sessions.set(sessionId, sessionRecord);

    return reply.send({ sessionId, status: "active", rawAudioStored: false });
  });

  app.post("/api/session/end", async (request, reply) => {
    const input = sessionEndRequestSchema.parse(request.body);
    const session = store.sessions.get(input.sessionId);
    if (!session) {
      return reply.status(404).send({ error: "session_not_found" });
    }

    const updated = {
      ...session,
      status: "ended" as const,
      endedAt: new Date().toISOString()
    };
    if (input.durationMs !== undefined) {
      updated.durationMs = input.durationMs;
    }
    store.sessions.set(input.sessionId, updated);

    return reply.send({ sessionId: input.sessionId, status: "ended", rawAudioStored: false });
  });

  app.get("/api/model/status", async (_request, reply) => {
    return reply.send({ model: islModel.metadata });
  });

  app.get("/api/keypoint-model/status", async (_request, reply) => {
    return reply.send({
      status: reportedKeypointModel?.status ?? "unavailable",
      model: reportedKeypointModel ?? null,
      metrics: reportedKeypointModel
        ? getRecognitionMetrics(reportedKeypointModel.id, temporalModel, videoModel)
        : null,
      recognitionModelReady,
      centroidFallbackReady: readyVideoModel !== undefined
    });
  });

  app.get("/api/models", async (_request, reply) => {
    return reply.send({
      models: loadedModels,
      rawVideoStored: false
    });
  });

  app.get("/api/models/:id/status", async (request, reply) => {
    const { id: modelId } = modelRouteParamsSchema.parse(request.params);
    const metadata = findModelMetadata(modelId, ...loadedModels);
    if (!metadata) {
      return reply.status(404).send({ error: "model_not_found" });
    }

    return reply.send({ model: metadata });
  });

  app.get("/api/models/:id/metrics", async (request, reply) => {
    const { id: modelId } = modelRouteParamsSchema.parse(request.params);
    const metadata = findModelMetadata(modelId, ...loadedModels);
    if (!metadata) {
      return reply.status(404).send({ error: "model_not_found" });
    }

    return reply.send({
      model: metadata,
      metrics:
        metadata.engine === "keypoint_transformer" || metadata.engine === "keypoint_centroid"
          ? getRecognitionMetrics(metadata.id, temporalModel, videoModel)
          : metadata.engine === "lexicon_pair_matcher"
            ? textMatcher?.metrics ?? null
            : metadata.engine === "motion_library"
              ? {
                  clipCount: researchMotionLibrary?.clipCount ?? 0,
                  expertReviewed: false,
                  continuousTranslationReady: false
                }
              : metadata.engine === "text_to_pose_research"
                ? researchTextToPoseArtifact?.metrics ?? null
                : metadata.engine === "caption_boundary_planner"
                  ? {
                      recordCount: metadata.trainingDataset.recordCount,
                      boundaryFeatureCount: metadata.trainingDataset.classCount,
                      isIslGrammarModel: false
                    }
                : {
                    recordCount: islModel.metadata.trainingDataset.recordCount,
                    classCount: islModel.metadata.trainingDataset.classCount
                  }
    });
  });

  app.post("/api/keypoints/infer", async (request, reply) => {
    const input = videoInferenceRequestSchema.parse(request.body);
    if (!readyTemporalModel && !readyVideoModel) {
      const response = videoInferenceResponseSchema.parse({
        error: "recognition_model_unavailable",
        prediction: null,
        predictions: [],
        model: null,
        accepted: false,
        fallback: "caption",
        rawVideoStored: false,
        notes: [
          "No persisted recognition model artifact is available.",
          "Use the caption fallback; no recognition class was evaluated.",
          "Raw video and audio are not accepted or stored by this endpoint."
        ]
      });
      return reply.status(503).send(response);
    }

    const response = readyTemporalModel
      ? await readyTemporalModel.infer(input.frames, input.topK)
      : inferVideoKeypoints(input.frames, readyVideoModel!, input.topK);

    return reply.send(response);
  });

  app.post("/api/interpret", async (request, reply) => {
    const input = interpretRequestSchema.parse(request.body);
    const session = store.sessions.get(input.sessionId);
    if (!session) {
      return reply.status(404).send({ error: "session_not_found" });
    }
    if (session.status !== "active") {
      return reply.status(409).send({ error: "session_not_active" });
    }

    const interpretInput: InterpretInput = {
      sessionId: input.sessionId,
      source: input.source,
      rawText: input.rawText,
      transcriptConfidence: 0.9,
      glossaryEntries: mergedGlossary,
      ...(researchSentencePlanner
        ? { splitMeaningChunks: researchSentencePlanner.splitMeaningChunks }
        : {}),
      ...(resolveUnmatchedToken ? { resolveUnmatchedToken } : {})
    };
    if (input.simplificationLevel) {
      Object.assign(interpretInput, { simplificationLevel: input.simplificationLevel });
    }
    if (input.timestamp) {
      Object.assign(interpretInput, {
        startMs: input.timestamp.startMs,
        endMs: input.timestamp.endMs
      });
    }
    const interpretation = interpretTranscript(interpretInput);
    const baseAvatarQueue = createAvatarQueue({
      interpretationId: interpretation.id,
      speed: input.avatarSpeed ?? 0.75,
      glossSequence: interpretation.glossSequence
    });
    const avatarQueue = researchMotionLibrary
      ? researchMotionLibrary.enrichQueue(baseAvatarQueue, interpretation.glossSequence)
      : baseAvatarQueue;
    const modelBackedGlossCount = interpretation.glossSequence.filter((item) =>
      modelTerms.has(normalizeTerm(item.token))
    ).length;
    const motionBackedGlossCount = avatarQueue.steps.filter((step) =>
      Boolean(step.motionClipId || step.fingerspellingSource)
    ).length;
    store.interpretations.set(interpretation.id, interpretation);
    store.avatarQueues.set(avatarQueue.id, avatarQueue);

    return reply.send({
      ...interpretation,
      avatarQueue,
      model: islModel.metadata,
      modelBackedGlossCount,
      motionBackedGlossCount
    });
  });

  app.get("/api/glossary/search", async (request, reply) => {
    const { q: query } = glossarySearchQuerySchema.parse(request.query);
    const normalized = query.trim().toLowerCase();
    const results = mergedGlossary.filter((entry) => {
      const candidates = [entry.term, ...entry.aliases].map((item) => item.toLowerCase());
      return (
        normalized.length === 0 || candidates.some((candidate) => candidate.includes(normalized))
      );
    });

    return reply.send({ results: results.slice(0, 50) });
  });

  app.post("/api/feedback", async (request, reply) => {
    const input = feedbackSubmitRequestSchema.parse(request.body);
    if (!store.sessions.has(input.sessionId)) {
      return reply.status(404).send({ error: "session_not_found" });
    }
    const interpretation = store.interpretations.get(input.interpretationId);
    if (!interpretation) {
      return reply.status(404).send({ error: "interpretation_not_found" });
    }
    if (interpretation.sessionId !== input.sessionId) {
      return reply.status(409).send({ error: "interpretation_session_mismatch" });
    }
    const feedbackId = store.nextId("feedback");
    const report = {
      id: feedbackId,
      ...input,
      createdAt: new Date().toISOString()
    };
    store.feedback.set(feedbackId, report);

    return reply.send({ feedbackId, status: "saved", rawAudioStored: false });
  });

  app.post("/api/privacy/purge", async (request, reply) => {
    const input = privacyPurgeRequestSchema.parse(request.body);
    if (input.scope === "session") {
      const deleted = store.purgeSession(input.sessionId);
      const response = privacyPurgeResponseSchema.parse({
        scope: "session",
        sessionId: input.sessionId,
        status: "purged",
        deleted: { ...deleted, rateLimitEntries: 0 },
        rawAudioStored: false,
        rawVideoStored: false
      });
      return reply.send(response);
    }

    const rateLimitEntries = requestWindows.size;
    const deleted = store.purgeAll();
    requestWindows.clear();
    const response = privacyPurgeResponseSchema.parse({
      scope: "all",
      status: "purged",
      deleted: { ...deleted, rateLimitEntries },
      rawAudioStored: false,
      rawVideoStored: false
    });
    return reply.send(response);
  });

  return app;
}

function loadVideoModel(): VideoKeypointModel | undefined {
  const configuredPath = process.env["SIGNSAARTHI_VIDEO_MODEL_PATH"];
  const candidatePaths = [
    configuredPath ? resolve(process.cwd(), configuredPath) : undefined,
    resolve(process.cwd(), "data/models/isl-keypoint-model.json"),
    fileURLToPath(new URL("../../../data/models/isl-keypoint-model.json", import.meta.url))
  ].filter((candidate): candidate is string => Boolean(candidate));

  for (const modelPath of candidatePaths) {
    if (existsSync(modelPath)) {
      return videoModelArtifactSchema.parse(JSON.parse(readFileSync(modelPath, "utf8")));
    }
  }

  return undefined;
}

function loadTextMatcher(): LexicalMatcherArtifact | undefined {
  const configuredPath = process.env["SIGNSAARTHI_TEXT_MATCHER_PATH"];
  const candidatePaths = configuredPath
    ? [resolve(process.cwd(), configuredPath)]
    : [
        resolve(process.cwd(), "data/models/isl-text-matcher.json"),
        fileURLToPath(new URL("../../../data/models/isl-text-matcher.json", import.meta.url))
      ];

  for (const matcherPath of candidatePaths) {
    if (!existsSync(matcherPath)) {
      continue;
    }
    try {
      return lexicalMatcherArtifactSchema.parse(JSON.parse(readFileSync(matcherPath, "utf8")));
    } catch {
      return undefined;
    }
  }
  return undefined;
}

function uniqueModelMetadata(models: ModelMetadata[]): ModelMetadata[] {
  return models.filter(
    (model, index) => models.findIndex((candidate) => candidate.id === model.id) === index
  );
}

function getRecognitionMetrics(
  modelId: string,
  temporalModel: TemporalModelRuntime | undefined,
  videoModel: VideoKeypointModel | undefined
): unknown {
  if (temporalModel?.artifact.metadata.id === modelId) {
    return temporalModel.artifact.metrics;
  }
  if (videoModel?.metadata.id === modelId) {
    return videoModel.metrics ?? null;
  }
  return null;
}

function normalizeTerm(value: string): string {
  return value.trim().toLowerCase();
}

function findModelMetadata(modelId: string, ...models: ModelMetadata[]): ModelMetadata | undefined {
  return models.find((model) => model.id === modelId);
}

function configuredOrigins(
  explicitValues: string[] | undefined,
  environmentName: string,
  fallback: string[],
  isAllowedOrigin: (value: string) => boolean
): Set<string> {
  const environmentValue = process.env[environmentName];
  const values =
    explicitValues ?? (environmentValue === undefined ? fallback : environmentValue.split(","));
  const origins = values.map((value) => value.trim()).filter(Boolean);
  const invalidOrigin = origins.find((origin) => !isAllowedOrigin(origin));
  if (invalidOrigin) {
    throw new Error(`${environmentName} contains an invalid origin: ${invalidOrigin}`);
  }
  return new Set(origins);
}

function isExtensionOrigin(value: string): boolean {
  return /^chrome-extension:\/\/[a-p]{32}$/.test(value);
}

function isDevPreviewOrigin(value: string): boolean {
  try {
    const url = new URL(value);
    return (
      url.origin === value &&
      url.protocol === "http:" &&
      (url.hostname === "127.0.0.1" || url.hostname === "localhost")
    );
  } catch {
    return false;
  }
}

function positiveInteger(value: number, label: string): number {
  if (!Number.isSafeInteger(value) || value <= 0) {
    throw new Error(`${label} must be a positive safe integer.`);
  }
  return value;
}

function getErrorStatusCode(error: unknown): number | undefined {
  if (!error || typeof error !== "object" || !("statusCode" in error)) {
    return undefined;
  }
  const statusCode = error.statusCode;
  return typeof statusCode === "number" && Number.isInteger(statusCode) ? statusCode : undefined;
}

function pruneExpiredRequestWindows(
  requestWindows: Map<string, { count: number; startedAt: number }>,
  now: number,
  windowMs: number
): void {
  for (const [client, window] of requestWindows) {
    if (now < window.startedAt || now - window.startedAt >= windowMs) {
      requestWindows.delete(client);
    }
  }
}

function loadISLModel(): ISLLexiconModel {
  const configuredPath = process.env["SIGNSAARTHI_MODEL_PATH"];
  const candidatePaths = [
    configuredPath ? resolve(process.cwd(), configuredPath) : undefined,
    resolve(process.cwd(), "data/models/isl-lexicon-model.json"),
    fileURLToPath(new URL("../../../data/models/isl-lexicon-model.json", import.meta.url))
  ].filter((candidate): candidate is string => Boolean(candidate));

  for (const modelPath of candidatePaths) {
    if (existsSync(modelPath)) {
      return islLexiconModelSchema.parse(JSON.parse(readFileSync(modelPath, "utf8")));
    }
  }

  return trainISLLexiconModel(defaultISLDatasetManifest, {
    now: () => "2026-07-07T00:00:00.000Z"
  });
}
