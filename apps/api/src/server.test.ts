import { lexicalMatcherFeatureNames, type LexicalMatcherArtifact } from "@signsaarthi/isl-model";
import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import { buildServer } from "./server";

const EXTENSION_CLIENT_HEADERS = { "x-signsaarthi-client": "extension-v1" } as const;

describe("SignSaarthi API", () => {
  it("restricts production CORS and allows only explicitly configured dev previews", async () => {
    const app = buildServerWithoutRecognition();

    const health = await app.inject({
      method: "GET",
      url: "/health",
      headers: { origin: "chrome-extension://kgjeaadkknfadmlipfjaekalpgbcnela" }
    });
    const denied = await app.inject({
      method: "GET",
      url: "/health",
      headers: { origin: "https://example.com" }
    });
    const preview = await app.inject({
      method: "GET",
      url: "/health",
      headers: { origin: "http://127.0.0.1:5174" }
    });
    const arbitraryExtension = await app.inject({
      method: "GET",
      url: "/health",
      headers: { origin: "chrome-extension://aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa" }
    });

    expect(health.statusCode).toBe(200);
    expect(
      health.json<{
        status: string;
        modelsReady: boolean;
        lexiconModelReady: boolean;
        recognitionModelReady: boolean;
        rawAudioStored: boolean;
        rawVideoStored: boolean;
      }>()
    ).toMatchObject({
      status: "ok",
      modelsReady: true,
      lexiconModelReady: true,
      recognitionModelReady: false,
      rawAudioStored: false,
      rawVideoStored: false
    });
    expect(health.headers["access-control-allow-origin"]).toBe(
      "chrome-extension://kgjeaadkknfadmlipfjaekalpgbcnela"
    );
    expect(denied.headers["access-control-allow-origin"]).toBeUndefined();
    expect(preview.headers["access-control-allow-origin"]).toBeUndefined();
    expect(arbitraryExtension.headers["access-control-allow-origin"]).toBeUndefined();
    await app.close();

    const devApp = buildServer({
      recognitionModelLoading: false,
      allowedDevPreviewOrigins: ["http://127.0.0.1:5174"]
    });
    const allowedPreview = await devApp.inject({
      method: "GET",
      url: "/health",
      headers: { origin: "http://127.0.0.1:5174" }
    });
    const otherPreviewPort = await devApp.inject({
      method: "GET",
      url: "/health",
      headers: { origin: "http://127.0.0.1:5173" }
    });
    const preflight = await devApp.inject({
      method: "OPTIONS",
      url: "/api/session/start",
      headers: {
        origin: "http://127.0.0.1:5174",
        "access-control-request-method": "POST",
        "access-control-request-headers": "content-type,x-signsaarthi-client"
      }
    });

    expect(allowedPreview.headers["access-control-allow-origin"]).toBe("http://127.0.0.1:5174");
    expect(otherPreviewPort.headers["access-control-allow-origin"]).toBeUndefined();
    expect(preflight.statusCode).toBe(204);
    expect(preflight.headers["access-control-allow-origin"]).toBe("http://127.0.0.1:5174");
    await devApp.close();
  });

  it("requires the extension client header on production API routes", async () => {
    const app = buildServer({ requireClientHeader: true });

    const unauthorized = await app.inject({
      method: "POST",
      url: "/api/session/start",
      payload: { source: "mock" }
    });
    const authorized = await app.inject({
      method: "POST",
      url: "/api/session/start",
      headers: EXTENSION_CLIENT_HEADERS,
      payload: { source: "mock" }
    });

    expect(unauthorized.statusCode).toBe(401);
    expect(authorized.statusCode).toBe(200);
    await app.close();
  });

  it("sanitizes unexpected server errors", async () => {
    const app = buildServerWithoutRecognition();
    app.get("/__test__/internal-error", async () => {
      throw new Error("sensitive model path: /private/model.onnx");
    });

    const response = await app.inject({ method: "GET", url: "/__test__/internal-error" });

    expect(response.statusCode).toBe(500);
    expect(response.json()).toEqual({ error: "internal_error" });
    expect(response.payload).not.toContain("sensitive model path");
    await app.close();
  });

  it("bounds and expires production rate-limit identity state", async () => {
    let now = 10_000;
    const app = buildServer({
      requireClientHeader: true,
      recognitionModelLoading: false,
      rateLimit: {
        windowMs: 1_000,
        maxRequests: 2,
        maxTrackedClients: 2,
        now: () => now
      }
    });
    const requestAs = (remoteAddress: string) =>
      app.inject({
        method: "GET",
        url: "/api/models",
        headers: EXTENSION_CLIENT_HEADERS,
        remoteAddress
      });

    expect((await requestAs("10.0.0.1")).statusCode).toBe(200);
    expect((await requestAs("10.0.0.1")).statusCode).toBe(200);
    const limited = await requestAs("10.0.0.1");
    expect(limited.statusCode).toBe(429);
    expect(limited.json()).toEqual({ error: "rate_limit_exceeded" });
    expect((await requestAs("10.0.0.2")).statusCode).toBe(200);

    const atCapacity = await requestAs("10.0.0.3");
    expect(atCapacity.statusCode).toBe(429);
    expect(atCapacity.json()).toEqual({ error: "rate_limit_exceeded" });

    now += 1_000;
    expect((await requestAs("10.0.0.3")).statusCode).toBe(200);
    await app.close();
  });

  it("requires authentication and purges only data linked to one session", async () => {
    const app = buildServerWithoutRecognition();
    const first = await createStoredSession(app, "Today we learn.");
    const second = await createStoredSession(app, "Tomorrow we learn.");

    const unauthorized = await app.inject({
      method: "POST",
      url: "/api/privacy/purge",
      payload: {
        scope: "session",
        sessionId: first.sessionId,
        confirmation: "purge_session"
      }
    });
    const malformed = await app.inject({
      method: "POST",
      url: "/api/privacy/purge",
      headers: EXTENSION_CLIENT_HEADERS,
      payload: {
        scope: "session",
        sessionId: first.sessionId,
        confirmation: "purge_all_local_data"
      }
    });
    const purge = await app.inject({
      method: "POST",
      url: "/api/privacy/purge",
      headers: EXTENSION_CLIENT_HEADERS,
      payload: {
        scope: "session",
        sessionId: first.sessionId,
        confirmation: "purge_session"
      }
    });

    expect(unauthorized.statusCode).toBe(401);
    expect(malformed.statusCode).toBe(400);
    expect(purge.statusCode).toBe(200);
    expect(purge.json()).toMatchObject({
      scope: "session",
      sessionId: first.sessionId,
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

    const deletedSession = await app.inject({
      method: "POST",
      url: "/api/session/end",
      payload: { sessionId: first.sessionId }
    });
    const retainedInterpretation = await app.inject({
      method: "POST",
      url: "/api/feedback",
      payload: {
        sessionId: second.sessionId,
        interpretationId: second.interpretationId,
        reason: "wrong_sign",
        timestamp: { startMs: 0, endMs: 1000 }
      }
    });

    expect(deletedSession.statusCode).toBe(404);
    expect(retainedInterpretation.statusCode).toBe(200);
    await app.close();
  });

  it("purges all mutable local API data without reusing stale identifiers", async () => {
    const app = buildServer({
      recognitionModelLoading: false,
      rateLimit: { maxRequests: 1 }
    });
    const stored = await createStoredSession(app, "Today we use a computer.");

    const purge = await app.inject({
      method: "POST",
      url: "/api/privacy/purge",
      headers: EXTENSION_CLIENT_HEADERS,
      payload: { scope: "all", confirmation: "purge_all_local_data" }
    });
    const body = purge.json<{
      deleted: {
        sessions: number;
        interpretations: number;
        avatarQueues: number;
        feedback: number;
        rateLimitEntries: number;
      };
      rawAudioStored: boolean;
      rawVideoStored: boolean;
    }>();

    expect(purge.statusCode).toBe(200);
    expect(body.deleted).toEqual({
      sessions: 1,
      interpretations: 1,
      avatarQueues: 1,
      feedback: 1,
      rateLimitEntries: 1
    });
    expect(body.rawAudioStored).toBe(false);
    expect(body.rawVideoStored).toBe(false);

    const deletedSession = await app.inject({
      method: "POST",
      url: "/api/session/end",
      payload: { sessionId: stored.sessionId }
    });
    const restarted = await app.inject({
      method: "POST",
      url: "/api/session/start",
      payload: { source: "mock" }
    });
    expect(deletedSession.statusCode).toBe(404);
    expect(restarted.json<{ sessionId: string }>().sessionId).not.toBe(stored.sessionId);

    const secondPurge = await app.inject({
      method: "POST",
      url: "/api/privacy/purge",
      headers: EXTENSION_CLIENT_HEADERS,
      payload: { scope: "all", confirmation: "purge_all_local_data" }
    });
    expect(secondPurge.statusCode).toBe(200);
    await app.close();
  });

  it("starts and ends a privacy-forward session", async () => {
    const app = buildServer();

    const start = await app.inject({
      method: "POST",
      url: "/api/session/start",
      payload: {
        source: "mock",
        pageTitle: "Indian Constitution - Part 1",
        url: "https://www.youtube.com/watch?v=demo"
      }
    });
    const session = start.json<{ sessionId: string; status: string; rawAudioStored: boolean }>();

    const end = await app.inject({
      method: "POST",
      url: "/api/session/end",
      payload: { sessionId: session.sessionId, durationMs: 42000 }
    });

    expect(start.statusCode).toBe(200);
    expect(session.rawAudioStored).toBe(false);
    expect(end.json<{ status: string }>().status).toBe("ended");
    await app.close();
  });

  it("interprets transcript text into simplified text, gloss, confidence, and avatar-safe warnings", async () => {
    const app = buildServer();
    const session = await app.inject({
      method: "POST",
      url: "/api/session/start",
      payload: { source: "mock" }
    });
    const sessionId = session.json<{ sessionId: string }>().sessionId;

    const response = await app.inject({
      method: "POST",
      url: "/api/interpret",
      payload: {
        sessionId,
        source: "mock",
        rawText: "Artificial intelligence can help education in Mumbai.",
        languageHint: "en",
        timestamp: { startMs: 0, endMs: 4000 }
      }
    });

    const body = response.json<{
      simplifiedText: string;
      confidence: string;
      glossSequence: Array<{ token: string; fallback: string }>;
      avatarQueue: { steps: Array<{ label: string; kind: string }> };
      warnings: string[];
    }>();

    expect(response.statusCode).toBe(200);
    expect(body.simplifiedText).toContain("AI can help education");
    expect(body.confidence).toBe("low");
    expect(body.glossSequence.some((item) => item.fallback === "fingerspell")).toBe(true);
    expect(body.avatarQueue.steps.length).toBeGreaterThan(0);
    expect(body.warnings).toContain("Some terms use fallback output.");
    await app.close();
  });

  it("applies the requested transcript simplification level", async () => {
    const app = buildServer();
    const session = await app.inject({
      method: "POST",
      url: "/api/session/start",
      payload: { source: "mock" }
    });
    const sessionId = session.json<{ sessionId: string }>().sessionId;
    const rawText = "The project will support students with accessible education in the classroom.";

    const light = await app.inject({
      method: "POST",
      url: "/api/interpret",
      payload: { sessionId, source: "mock", rawText, simplificationLevel: "light" }
    });
    const strong = await app.inject({
      method: "POST",
      url: "/api/interpret",
      payload: { sessionId, source: "mock", rawText, simplificationLevel: "strong" }
    });

    expect(light.json<{ simplifiedText: string }>().simplifiedText).toBe(rawText);
    expect(strong.json<{ simplifiedText: string }>().simplifiedText).not.toBe(rawText);
    expect(strong.json<{ simplifiedText: string }>().simplifiedText.length).toBeLessThan(
      rawText.length
    );
    await app.close();
  });

  it("counts loaded lexicon vocabulary even when motion falls back to fingerspelling", async () => {
    const app = buildServer();
    const session = await app.inject({
      method: "POST",
      url: "/api/session/start",
      payload: { source: "mock" }
    });
    const sessionId = session.json<{ sessionId: string }>().sessionId;

    const response = await app.inject({
      method: "POST",
      url: "/api/interpret",
      payload: {
        sessionId,
        source: "mock",
        rawText: "constitution justice.",
        languageHint: "en",
        timestamp: { startMs: 0, endMs: 5000 }
      }
    });

    const body = response.json<{
      model: { status: string; trainingDataset: { primaryDataset: string; recordCount: number } };
      modelBackedGlossCount: number;
      glossSequence: Array<{ token: string; fallback: string; signAssetId?: string }>;
    }>();

    expect(response.statusCode).toBe(200);
    expect(body.model.status).toBe("ready");
    expect(body.model.trainingDataset.primaryDataset).toBe("islrtc");
    expect(body.model.trainingDataset.recordCount).toBe(12_434);
    expect(body.modelBackedGlossCount).toBe(2);
    expect(body.glossSequence).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          token: "constitution",
          fallback: "fingerspell"
        }),
        expect.objectContaining({
          token: "justice",
          fallback: "fingerspell"
        })
      ])
    );
    await app.close();
  });

  it("reports trained model status and dataset provenance", async () => {
    const app = buildServer();

    const response = await app.inject({ method: "GET", url: "/api/model/status" });
    const body = response.json<{
      model: {
        status: string;
        trainingDataset: {
          primaryDataset: string;
          displayName: string;
          recordCount: number;
          classCount: number;
          citationUrls: string[];
        };
      };
    }>();

    expect(response.statusCode).toBe(200);
    expect(body.model.status).toBe("ready");
    expect(body.model.trainingDataset).toMatchObject({
      primaryDataset: "islrtc",
      recordCount: 12_434,
      classCount: 12_434
    });
    expect(body.model.trainingDataset.citationUrls).toEqual(
      expect.arrayContaining([
        "https://divyangjan.depwd.gov.in/islrtc/",
        "https://huggingface.co/datasets/ai4bharat/INCLUDE"
      ])
    );
    await app.close();
  });

  it("lists only genuinely loaded models with privacy state", async () => {
    const app = buildServerWithoutRecognition();

    const response = await app.inject({ method: "GET", url: "/api/models" });
    const body = response.json<{ models: Array<{ engine: string }>; rawVideoStored: boolean }>();

    expect(response.statusCode).toBe(200);
    expect(body.rawVideoStored).toBe(false);
    expect(body.models).toEqual(
      expect.arrayContaining([expect.objectContaining({ engine: "lexicon_ranker" })])
    );
    expect(
      body.models.every(
        (model) => model.engine === "lexicon_ranker" || model.engine === "lexicon_pair_matcher"
      )
    ).toBe(true);
    await app.close();
  });

  it("loads a valid text matcher, registers it, and preserves the typo source token", async () => {
    await withTextMatcherArtifact(lexicalMatcherArtifact(), async () => {
      const app = buildServerWithoutRecognition();
      const models = await app.inject({ method: "GET", url: "/api/models" });
      const session = await app.inject({
        method: "POST",
        url: "/api/session/start",
        payload: { source: "mock" }
      });
      const sessionId = session.json<{ sessionId: string }>().sessionId;
      const interpretation = await app.inject({
        method: "POST",
        url: "/api/interpret",
        payload: { sessionId, source: "mock", rawText: "computre" }
      });
      const body = interpretation.json<{
        glossSequence: Array<{
          token: string;
          gloss: string;
          isVerified: boolean;
          reviewStatus: string;
          provenance: { source: string; glossaryEntryId?: string };
        }>;
      }>();

      expect(models.json<{ models: Array<{ id: string; engine: string }> }>().models).toEqual(
        expect.arrayContaining([
          expect.objectContaining({
            id: "isl-text-matcher-fixture",
            engine: "lexicon_pair_matcher"
          })
        ])
      );
      expect(interpretation.statusCode).toBe(200);
      expect(body.glossSequence[0]).toMatchObject({
        token: "computre",
        gloss: "COMPUTER",
        isVerified: false,
        reviewStatus: "pending_review",
        provenance: { source: "controlled_glossary" }
      });
      await app.close();
    });
  });

  it("prefers curated aliases and exact inflections over a learned near-neighbor", async () => {
    const app = buildServerWithoutRecognition();
    const session = await app.inject({
      method: "POST",
      url: "/api/session/start",
      payload: { source: "mock" }
    });
    const response = await app.inject({
      method: "POST",
      url: "/api/interpret",
      payload: {
        sessionId: session.json<{ sessionId: string }>().sessionId,
        source: "mock",
        rawText: "helps students"
      }
    });

    expect(response.statusCode).toBe(200);
    expect(
      response
        .json<{ glossSequence: Array<{ token: string; gloss: string }> }>()
        .glossSequence.map(({ token, gloss }) => ({ token, gloss }))
    ).toEqual([
      { token: "helps", gloss: "HELP" },
      { token: "students", gloss: "STUDENT" }
    ]);
    await app.close();
  });

  it("ignores an invalid text matcher without breaking caption fallbacks", async () => {
    await withTextMatcherArtifact({ schemaVersion: 1, invalid: true }, async () => {
      const app = buildServerWithoutRecognition();
      const models = await app.inject({ method: "GET", url: "/api/models" });
      const session = await app.inject({
        method: "POST",
        url: "/api/session/start",
        payload: { source: "mock" }
      });
      const sessionId = session.json<{ sessionId: string }>().sessionId;
      const interpretation = await app.inject({
        method: "POST",
        url: "/api/interpret",
        payload: { sessionId, source: "mock", rawText: "computre" }
      });
      const body = interpretation.json<{
        glossSequence: Array<{
          token: string;
          gloss: string;
          fallback: string;
          provenance: { source: string };
        }>;
      }>();

      expect(
        models
          .json<{ models: Array<{ engine: string }> }>()
          .models.some((model) => model.engine === "lexicon_pair_matcher")
      ).toBe(false);
      expect(interpretation.statusCode).toBe(200);
      expect(body.glossSequence[0]).toMatchObject({
        token: "computre",
        gloss: "COMPUTRE",
        fallback: "fingerspell",
        provenance: { source: "deterministic_fallback" }
      });
      await app.close();
    });
  });

  it("reports recognition unavailable without exposing class names", async () => {
    const app = buildServerWithoutRecognition();

    const status = await app.inject({ method: "GET", url: "/api/keypoint-model/status" });
    const inference = await app.inject({
      method: "POST",
      url: "/api/keypoints/infer",
      payload: {
        frames: [
          {
            timestampMs: 0,
            landmarks: [{ part: "pose", index: 0, x: 0.5, y: 0.5 }]
          }
        ],
        topK: 2
      }
    });
    const body = inference.json<{
      error: string;
      prediction: null;
      predictions: unknown[];
      rawVideoStored: boolean;
      accepted: boolean;
      fallback: string;
      model: null;
    }>();

    expect(status.statusCode).toBe(200);
    expect(status.json()).toMatchObject({
      status: "unavailable",
      model: null,
      metrics: null,
      recognitionModelReady: false,
      centroidFallbackReady: false
    });
    expect(inference.statusCode).toBe(503);
    expect(body.error).toBe("recognition_model_unavailable");
    expect(body.rawVideoStored).toBe(false);
    expect(body.model).toBeNull();
    expect(body.prediction).toBeNull();
    expect(body.predictions).toEqual([]);
    expect(body.accepted).toBe(false);
    expect(body.fallback).toBe("caption");
    await app.close();
  });

  it("rejects malformed keypoint inference payloads", async () => {
    const app = buildServerWithoutRecognition();

    const response = await app.inject({
      method: "POST",
      url: "/api/keypoints/infer",
      payload: { frames: [] }
    });

    expect(response.statusCode).toBe(400);
    expect(response.json<{ error: string }>().error).toBe("validation_error");
    await app.close();
  });

  it("rejects raw media request bodies without changing privacy guarantees", async () => {
    const app = buildServerWithoutRecognition();

    const response = await app.inject({
      method: "POST",
      url: "/api/keypoints/infer",
      headers: { "content-type": "video/mp4" },
      payload: Buffer.from([0, 0, 0, 24])
    });
    const health = await app.inject({ method: "GET", url: "/health" });

    expect(response.statusCode).toBe(415);
    expect(response.json()).toEqual({ error: "invalid_request" });
    expect(health.json()).toMatchObject({ rawAudioStored: false, rawVideoStored: false });
    await app.close();
  });

  it("falls back honestly when the trained model has no matching sign", async () => {
    const app = buildServer();
    const session = await app.inject({
      method: "POST",
      url: "/api/session/start",
      payload: { source: "mock" }
    });
    const sessionId = session.json<{ sessionId: string }>().sessionId;

    const response = await app.inject({
      method: "POST",
      url: "/api/interpret",
      payload: {
        sessionId,
        source: "mock",
        rawText: "Mumbai skyline updates.",
        timestamp: { startMs: 0, endMs: 2000 }
      }
    });
    const body = response.json<{
      modelBackedGlossCount: number;
      glossSequence: Array<{ token: string; fallback: string }>;
      warnings: string[];
    }>();

    expect(response.statusCode).toBe(200);
    expect(body.modelBackedGlossCount).toBe(0);
    expect(body.glossSequence.some((item) => item.fallback !== "none")).toBe(true);
    expect(body.warnings).toContain("Some terms use fallback output.");
    await app.close();
  });

  it("rejects interpretation after a session has ended", async () => {
    const app = buildServer();
    const session = await app.inject({
      method: "POST",
      url: "/api/session/start",
      payload: { source: "mock" }
    });
    const sessionId = session.json<{ sessionId: string }>().sessionId;
    await app.inject({ method: "POST", url: "/api/session/end", payload: { sessionId } });

    const response = await app.inject({
      method: "POST",
      url: "/api/interpret",
      payload: {
        sessionId,
        source: "mock",
        rawText: "Today we learn.",
        timestamp: { startMs: 0, endMs: 1000 }
      }
    });

    expect(response.statusCode).toBe(409);
    expect(response.json<{ error: string }>().error).toBe("session_not_active");
    await app.close();
  });

  it("searches glossary entries by phrase", async () => {
    const app = buildServer();

    const response = await app.inject({
      method: "GET",
      url: "/api/glossary/search?q=artificial%20intelligence"
    });

    expect(response.statusCode).toBe(200);
    expect(response.json<{ results: Array<{ term: string }> }>().results[0]?.term).toBe(
      "Artificial Intelligence"
    );
    await app.close();
  });

  it("searches loaded model glossary entries beyond the seed glossary", async () => {
    const app = buildServer();

    const response = await app.inject({
      method: "GET",
      url: "/api/glossary/search?q=constitution"
    });

    expect(response.statusCode).toBe(200);
    expect(response.json<{ results: Array<{ term: string; source: string }> }>().results).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          term: "Constitution",
          source: "external_reference"
        })
      ])
    );
    await app.close();
  });

  it("saves wrong-sign feedback without raw audio", async () => {
    const app = buildServer();
    const session = await app.inject({
      method: "POST",
      url: "/api/session/start",
      payload: { source: "mock" }
    });
    const sessionId = session.json<{ sessionId: string }>().sessionId;
    const interpretation = await app.inject({
      method: "POST",
      url: "/api/interpret",
      payload: {
        sessionId,
        source: "mock",
        rawText: "Artificial intelligence can help education in Mumbai.",
        timestamp: { startMs: 0, endMs: 4000 }
      }
    });
    const interpretationId = interpretation.json<{ id: string }>().id;
    const response = await app.inject({
      method: "POST",
      url: "/api/feedback",
      payload: {
        sessionId,
        interpretationId,
        reason: "wrong_sign",
        timestamp: { startMs: 0, endMs: 4000 },
        userComment: "The sign looked wrong for this context.",
        suggestedCorrection: "Use classroom variant."
      }
    });

    const body = response.json<{ feedbackId: string; rawAudioStored: boolean }>();

    expect(response.statusCode).toBe(200);
    expect(body.feedbackId).toMatch(/^feedback_/);
    expect(body.rawAudioStored).toBe(false);
    await app.close();
  });

  it("rejects feedback for unknown session or interpretation ids", async () => {
    const app = buildServer();
    const missingSession = await app.inject({
      method: "POST",
      url: "/api/feedback",
      payload: {
        sessionId: "missing",
        interpretationId: "missing",
        reason: "wrong_sign",
        timestamp: { startMs: 0, endMs: 1000 }
      }
    });
    const session = await app.inject({
      method: "POST",
      url: "/api/session/start",
      payload: { source: "mock" }
    });
    const sessionId = session.json<{ sessionId: string }>().sessionId;
    const missingInterpretation = await app.inject({
      method: "POST",
      url: "/api/feedback",
      payload: {
        sessionId,
        interpretationId: "missing",
        reason: "wrong_sign",
        timestamp: { startMs: 0, endMs: 1000 }
      }
    });

    expect(missingSession.statusCode).toBe(404);
    expect(missingInterpretation.statusCode).toBe(404);
    await app.close();
  });

  it("rejects feedback that pairs an interpretation with a different session", async () => {
    const app = buildServer();
    const first = await app.inject({
      method: "POST",
      url: "/api/session/start",
      payload: { source: "mock" }
    });
    const second = await app.inject({
      method: "POST",
      url: "/api/session/start",
      payload: { source: "mock" }
    });
    const firstSessionId = first.json<{ sessionId: string }>().sessionId;
    const secondSessionId = second.json<{ sessionId: string }>().sessionId;
    const interpretation = await app.inject({
      method: "POST",
      url: "/api/interpret",
      payload: { sessionId: firstSessionId, source: "mock", rawText: "Today we learn." }
    });

    const response = await app.inject({
      method: "POST",
      url: "/api/feedback",
      payload: {
        sessionId: secondSessionId,
        interpretationId: interpretation.json<{ id: string }>().id,
        reason: "wrong_sign",
        timestamp: { startMs: 0, endMs: 1000 }
      }
    });

    expect(response.statusCode).toBe(409);
    expect(response.json<{ error: string }>().error).toBe("interpretation_session_mismatch");
    await app.close();
  });
});

function buildServerWithoutRecognition() {
  return buildServer({
    recognitionModelLoading: false
  });
}

async function withTextMatcherArtifact(artifact: unknown, run: () => Promise<void>): Promise<void> {
  const directory = mkdtempSync(join(tmpdir(), "signsaarthi-text-matcher-"));
  const artifactPath = join(directory, "isl-text-matcher.json");
  const previousPath = process.env["SIGNSAARTHI_TEXT_MATCHER_PATH"];
  writeFileSync(artifactPath, JSON.stringify(artifact), "utf8");
  process.env["SIGNSAARTHI_TEXT_MATCHER_PATH"] = artifactPath;
  try {
    await run();
  } finally {
    if (previousPath === undefined) {
      delete process.env["SIGNSAARTHI_TEXT_MATCHER_PATH"];
    } else {
      process.env["SIGNSAARTHI_TEXT_MATCHER_PATH"] = previousPath;
    }
    rmSync(directory, { recursive: true, force: true });
  }
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
    marginThreshold: 0.02,
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

async function createStoredSession(
  app: ReturnType<typeof buildServer>,
  rawText: string
): Promise<{ sessionId: string; interpretationId: string }> {
  const session = await app.inject({
    method: "POST",
    url: "/api/session/start",
    payload: { source: "mock" }
  });
  const sessionId = session.json<{ sessionId: string }>().sessionId;
  const interpretation = await app.inject({
    method: "POST",
    url: "/api/interpret",
    payload: { sessionId, source: "mock", rawText }
  });
  const interpretationId = interpretation.json<{ id: string }>().id;
  const feedback = await app.inject({
    method: "POST",
    url: "/api/feedback",
    payload: {
      sessionId,
      interpretationId,
      reason: "wrong_sign",
      timestamp: { startMs: 0, endMs: 1000 }
    }
  });
  if (
    session.statusCode !== 200 ||
    interpretation.statusCode !== 200 ||
    feedback.statusCode !== 200
  ) {
    throw new Error("Unable to prepare an in-memory API session for the purge test.");
  }
  return { sessionId, interpretationId };
}
