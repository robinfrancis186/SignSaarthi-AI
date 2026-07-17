import {
  feedbackSubmitResponseSchema,
  glossarySearchResponseSchema,
  interpretationResponseSchema,
  modelRegistryResponseSchema,
  privacyPurgeRequestSchema,
  privacyPurgeResponseSchema,
  sessionEndResponseSchema,
  sessionStartResponseSchema,
  type GlossaryEntry,
  type ISLInterpretationResponse,
  type ModelMetadata,
  type PrivacyPurgeRequest,
  type PrivacyPurgeResponse,
  type SimplificationLevel,
  type TranscriptSource
} from "@signsaarthi/shared";

const API_BASE_URL = (
  import.meta.env.VITE_SIGNSAARTHI_API_BASE_URL ?? "http://127.0.0.1:8787"
).replace(/\/$/, "");
const API_TIMEOUT_MS = 8000;
const CLIENT_HEADER = "extension-v1";

type Parser<T> = { parse: (value: unknown) => T };

export type StartSessionInput = {
  source: TranscriptSource;
  pageTitle?: string;
  url?: string;
};

export type SubmitFeedbackInput = {
  sessionId: string;
  interpretationId: string;
  reason:
    | "wrong_sign"
    | "incorrect_translation"
    | "missing_sign"
    | "poor_avatar_motion"
    | "regional_variation"
    | "other";
  transcriptText?: string;
  glossSequence?: ISLInterpretationResponse["glossSequence"];
  timestamp: { startMs: number; endMs: number };
  userComment?: string;
  suggestedCorrection?: string;
};

export async function startSession(
  input: StartSessionInput
): Promise<{ sessionId: string; rawAudioStored: false }> {
  return postJson("/api/session/start", input, sessionStartResponseSchema);
}

export async function endSession(
  sessionId: string,
  durationMs?: number
): Promise<{ status: "ended" }> {
  return postJson("/api/session/end", { sessionId, durationMs }, sessionEndResponseSchema);
}

export async function purgePrivacyData(input: PrivacyPurgeRequest): Promise<PrivacyPurgeResponse> {
  const request = privacyPurgeRequestSchema.parse(input);
  return postJson("/api/privacy/purge", request, privacyPurgeResponseSchema);
}

export async function interpretSegment(input: {
  sessionId: string;
  source: TranscriptSource;
  rawText: string;
  languageHint?: "en" | "hi" | "hinglish" | "unknown";
  simplificationLevel?: SimplificationLevel;
  avatarSpeed?: number;
  timestamp?: { startMs: number; endMs: number };
}): Promise<ISLInterpretationResponse> {
  return postJson("/api/interpret", input, interpretationResponseSchema);
}

export async function submitFeedback(
  input: SubmitFeedbackInput
): Promise<{ feedbackId: string; rawAudioStored: false }> {
  return postJson("/api/feedback", input, feedbackSubmitResponseSchema);
}

export async function searchGlossary(query: string): Promise<{ results: GlossaryEntry[] }> {
  return getJson(
    `/api/glossary/search?q=${encodeURIComponent(query)}`,
    glossarySearchResponseSchema
  );
}

export async function getModelRegistry(): Promise<{
  models: ModelMetadata[];
  rawVideoStored: false;
}> {
  return getJson("/api/models", modelRegistryResponseSchema);
}

async function postJson<T>(path: string, body: unknown, parser: Parser<T>): Promise<T> {
  return requestJson(path, parser, {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify(body)
  });
}

async function getJson<T>(path: string, parser: Parser<T>): Promise<T> {
  return requestJson(path, parser);
}

async function requestJson<T>(path: string, parser: Parser<T>, init?: RequestInit): Promise<T> {
  const controller = new AbortController();
  const timeout = globalThis.setTimeout(() => controller.abort(), API_TIMEOUT_MS);

  try {
    const headers = new Headers(init?.headers);
    headers.set("X-SignSaarthi-Client", CLIENT_HEADER);
    const response = await fetch(`${API_BASE_URL}${path}`, {
      ...init,
      headers,
      signal: controller.signal
    });

    if (!response.ok) {
      throw new Error(`Local API request failed (${response.status}).`);
    }

    return parser.parse(await response.json());
  } catch (error) {
    if (error instanceof DOMException && error.name === "AbortError") {
      throw new Error("Local SignSaarthi API did not respond in time.");
    }
    if (error instanceof TypeError) {
      throw new Error("Local SignSaarthi API is unavailable.");
    }
    throw error;
  } finally {
    globalThis.clearTimeout(timeout);
  }
}
