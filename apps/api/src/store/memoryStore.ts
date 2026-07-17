import type {
  AvatarQueue,
  FeedbackReport,
  ISLInterpretation,
  TranscriptSource
} from "@signsaarthi/shared";

export type SessionRecord = {
  id: string;
  source: TranscriptSource;
  status: "active" | "ended";
  pageTitle?: string;
  url?: string;
  startedAt: string;
  endedAt?: string;
  durationMs?: number;
  rawAudioStored: false;
};

export type MemoryStoreDeletionCounts = {
  sessions: number;
  interpretations: number;
  avatarQueues: number;
  feedback: number;
};

export type MemoryStore = {
  sessions: Map<string, SessionRecord>;
  interpretations: Map<string, ISLInterpretation>;
  avatarQueues: Map<string, AvatarQueue>;
  feedback: Map<string, FeedbackReport>;
  nextId(prefix: string): string;
  purgeSession(sessionId: string): MemoryStoreDeletionCounts;
  purgeAll(): MemoryStoreDeletionCounts;
};

export function createMemoryStore(): MemoryStore {
  const sessions = new Map<string, SessionRecord>();
  const interpretations = new Map<string, ISLInterpretation>();
  const avatarQueues = new Map<string, AvatarQueue>();
  const feedback = new Map<string, FeedbackReport>();
  let counter = 0;

  return {
    sessions,
    interpretations,
    avatarQueues,
    feedback,
    nextId(prefix: string) {
      counter += 1;
      return `${prefix}_${counter.toString().padStart(4, "0")}`;
    },
    purgeSession(sessionId: string) {
      const interpretationIds = new Set(
        [...interpretations.values()]
          .filter((interpretation) => interpretation.sessionId === sessionId)
          .map((interpretation) => interpretation.id)
      );

      return {
        sessions: Number(sessions.delete(sessionId)),
        interpretations: deleteMatching(
          interpretations,
          (interpretation) => interpretation.sessionId === sessionId
        ),
        avatarQueues: deleteMatching(avatarQueues, (queue) =>
          interpretationIds.has(queue.interpretationId)
        ),
        feedback: deleteMatching(
          feedback,
          (report) =>
            report.sessionId === sessionId || interpretationIds.has(report.interpretationId)
        )
      };
    },
    purgeAll() {
      const deleted = {
        sessions: sessions.size,
        interpretations: interpretations.size,
        avatarQueues: avatarQueues.size,
        feedback: feedback.size
      };
      sessions.clear();
      interpretations.clear();
      avatarQueues.clear();
      feedback.clear();
      return deleted;
    }
  };
}

function deleteMatching<K, V>(map: Map<K, V>, predicate: (value: V) => boolean): number {
  let deleted = 0;
  for (const [key, value] of map) {
    if (predicate(value) && map.delete(key)) {
      deleted += 1;
    }
  }
  return deleted;
}
