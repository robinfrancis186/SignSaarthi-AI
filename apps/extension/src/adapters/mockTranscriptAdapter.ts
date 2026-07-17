import type { TranscriptSegment, TranscriptSource } from "@signsaarthi/shared";

const MOCK_TEXT = "Artificial intelligence can help education in Mumbai.";

type CreateMockTranscriptSegmentOptions = {
  rawText?: string;
  source?: TranscriptSource;
};

export function createMockTranscriptSegment(
  sessionId: string,
  options: CreateMockTranscriptSegmentOptions = {}
): TranscriptSegment {
  const rawText = options.rawText ?? MOCK_TEXT;
  return {
    id: `segment_${Date.now()}`,
    sessionId,
    source: options.source ?? "mock",
    rawText,
    cleanedText: rawText,
    languageHint: "en",
    startMs: 0,
    endMs: 4000,
    confidence: 0.9,
    createdAt: new Date().toISOString()
  };
}
