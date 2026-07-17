import type {
  ConfidenceLevel,
  GlossaryEntry,
  ISLGlossItem,
  ISLInterpretation,
  InterpretationPlanningIR,
  PlanningActionDecision,
  PlanningClause,
  PlanningFeature,
  PlanningProvenance,
  PlanningReviewStatus,
  PlanningSourceToken,
  SourceCharacterSpan,
  SimplificationLevel,
  TranscriptSource
} from "@signsaarthi/shared";

type Now = () => string;

export type SignAssetResolution = {
  expertReviewed: boolean;
  playable: boolean;
};

export type SignAssetResolver = (signAssetId: string) => SignAssetResolution | undefined;
export type UnmatchedTokenResolver = (token: string) => GlossaryEntry | undefined;
export type MeaningChunker = (cleanedText: string) => string[];

export type GenerateGlossOptions = {
  resolveSignAsset?: SignAssetResolver;
  resolveUnmatchedToken?: UnmatchedTokenResolver;
  sourceSegmentId?: string;
};

type PlannedGlossItem = ISLGlossItem & {
  sourceSegmentId: string;
  planningActionId: string;
  sourceTokenIds: string[];
  sourceSpan: SourceCharacterSpan;
  clauseId: string;
  plannedAction: PlanningActionDecision;
  resolvedAction: PlanningActionDecision;
  provenance: PlanningProvenance;
  reviewStatus: PlanningReviewStatus;
};

type SourcePlanningContext = {
  sourceSegmentId: string;
  sourceTokens: PlanningSourceToken[];
  clauses: PlanningClause[];
  features: PlanningFeature;
};

export type InterpretInput = {
  sessionId: string;
  source: TranscriptSource;
  rawText: string;
  transcriptConfidence?: number;
  startMs?: number;
  endMs?: number;
  glossaryEntries?: GlossaryEntry[];
  simplificationLevel?: SimplificationLevel;
  resolveSignAsset?: SignAssetResolver;
  resolveUnmatchedToken?: UnmatchedTokenResolver;
  splitMeaningChunks?: MeaningChunker;
  now?: Now;
};

type ConfidenceInput = {
  transcriptConfidence: number;
  glossaryMatches: number;
  unknownTerms: number;
  verifiedSigns: number;
  totalGlossItems: number;
};

const CREATED_AT = "2026-07-07T00:00:00.000Z";
const FILLERS = new Set(["um", "uh", "like"]);
const STOP_WORDS = new Set([
  "a",
  "an",
  "and",
  "are",
  "be",
  "can",
  "for",
  "how",
  "in",
  "is",
  "it",
  "of",
  "or",
  "students",
  "support",
  "supports",
  "the",
  "to",
  "use",
  "using",
  "we",
  "will",
  "with"
]);
const MAX_CHUNK_WORDS = 12;
const QUESTION_CUES = new Set([
  "can",
  "could",
  "did",
  "do",
  "does",
  "how",
  "is",
  "may",
  "what",
  "when",
  "where",
  "which",
  "who",
  "why",
  "will",
  "would"
]);
const NEGATION_CUES = new Set([
  "aren't",
  "cannot",
  "can't",
  "didn't",
  "doesn't",
  "don't",
  "isn't",
  "neither",
  "never",
  "no",
  "nobody",
  "none",
  "nor",
  "not",
  "nothing",
  "wasn't",
  "won't",
  "wouldn't"
]);

export const seedGlossary: GlossaryEntry[] = [
  glossary("today", "TODAY", "general", "sign_today_001", ["now"]),
  glossary("artificial intelligence", "AI", "technology", "sign_ai_001", ["ai"]),
  glossary("education", "EDUCATION", "education", "sign_education_001", ["learn", "learning"]),
  glossary("accessibility", "ACCESSIBILITY", "general", "sign_accessibility_001"),
  glossary("inclusion", "INCLUSION", "general", "sign_inclusion_001"),
  glossary("equal opportunity", "EQUAL OPPORTUNITY", "general", "sign_equal_opportunity_001", [
    "equal opportunities",
    "equal"
  ]),
  glossary("meeting", "MEETING", "meeting", "sign_meeting_001"),
  glossary("help", "HELP", "general", "sign_help_001", ["helps"]),
  glossary("learn", "LEARN", "education", "sign_learn_001", ["learning"]),
  glossary("internet", "INTERNET", "technology", "sign_internet_001"),
  glossary("barriers", "BARRIER REDUCE", "general", "sign_barrier_001", [
    "barrier",
    "break barriers"
  ])
];

function glossary(
  term: string,
  islGloss: string,
  category: GlossaryEntry["category"],
  signAssetId: string,
  aliases: string[] = []
): GlossaryEntry {
  return {
    id: `term_${term.replace(/\s+/g, "_")}`,
    term,
    aliases,
    language: "en",
    category,
    islGloss,
    signAssetId,
    exampleSentence: `Example use of ${term}.`,
    confidence: "medium",
    source: "internal",
    reviewStatus: "pending_review",
    createdAt: CREATED_AT,
    updatedAt: CREATED_AT
  };
}

export function cleanTranscript(rawText: string): string {
  const words = rawText.trim().replace(/\s+/g, " ").split(" ").filter(Boolean);
  const cleaned: string[] = [];

  for (const word of words) {
    const normalized = word.toLowerCase().replace(/[^\p{L}\p{N}]/gu, "");
    const previous = cleaned
      .at(-1)
      ?.toLowerCase()
      .replace(/[^\p{L}\p{N}]/gu, "");
    if (FILLERS.has(normalized)) {
      continue;
    }
    if (previous === normalized) {
      continue;
    }
    cleaned.push(word);
  }

  const sentence = cleaned.join(" ");
  if (!sentence) {
    return "";
  }

  const capitalized = sentence.charAt(0).toUpperCase() + sentence.slice(1);
  return /[.!?]$/.test(capitalized) ? capitalized : `${capitalized}.`;
}

export function splitIntoMeaningChunks(cleanedText: string): string[] {
  const sentenceChunks = cleanedText
    .split(/(?<=[.!?])\s+/)
    .map((chunk) => chunk.trim())
    .filter(Boolean);

  if (sentenceChunks.length > 0) {
    return sentenceChunks.flatMap(splitLongChunk);
  }

  const words = cleanedText.split(/\s+/).filter(Boolean);
  const chunks: string[] = [];
  for (let index = 0; index < words.length; index += MAX_CHUNK_WORDS) {
    chunks.push(words.slice(index, index + MAX_CHUNK_WORDS).join(" "));
  }
  return chunks;
}

function getMeaningChunks(cleanedText: string, chunker?: MeaningChunker): string[] {
  if (!chunker) {
    return splitIntoMeaningChunks(cleanedText);
  }
  try {
    const chunks = chunker(cleanedText).map((chunk) => chunk.trim()).filter(Boolean);
    const original = cleanedText.replace(/\s+/g, " ").trim();
    const reconstructed = chunks.join(" ").replace(/\s+/g, " ").trim();
    if (chunks.length && reconstructed === original) {
      return chunks;
    }
  } catch {
    // A private research planner must fail closed to deterministic chunking.
  }
  return splitIntoMeaningChunks(cleanedText);
}

export function simplifyChunks(
  chunks: string[],
  level: SimplificationLevel = "standard"
): Array<{ id: string; text: string; meaningUnits: string[]; confidence: ConfidenceLevel }> {
  return chunks.map((chunk, index) => {
    const lower = chunk.toLowerCase();
    const meaningUnits: string[] = [];

    if (
      level !== "light" &&
      lower.includes("artificial intelligence") &&
      lower.includes("barrier")
    ) {
      meaningUnits.push("Today topic: Artificial Intelligence", "AI can reduce barriers");
    }
    if (
      level !== "light" &&
      (lower.includes("equal opportunities") ||
        lower.includes("equal opportunity") ||
        (lower.includes("education") && lower.includes("equal")))
    ) {
      meaningUnits.push("AI can help education become more equal");
    }
    if (
      level !== "light" &&
      meaningUnits.length === 0 &&
      lower.includes("artificial intelligence") &&
      lower.includes("education")
    ) {
      meaningUnits.push("AI can help education");
    }
    if (
      level !== "light" &&
      meaningUnits.length === 0 &&
      lower.includes("today") &&
      lower.includes("learn")
    ) {
      meaningUnits.push("Today topic: learning");
    }

    const text = meaningUnits.length
      ? `${meaningUnits.join(". ")}.`
      : level === "strong"
        ? stronglySimplifyChunk(chunk)
        : chunk;
    const parsedMeaningUnits = meaningUnits.length
      ? meaningUnits
      : text
          .split(/(?<=[.!?])\s+/)
          .map((unit) => unit.replace(/[.!?]$/, "").trim())
          .filter(Boolean);

    return {
      id: `chunk_${index + 1}`,
      text,
      meaningUnits: parsedMeaningUnits,
      confidence: "high"
    };
  });
}

export function detectKeywords(text: string, glossaryEntries: GlossaryEntry[]): GlossaryEntry[] {
  return detectKeywordsWithIndex(text, createKeywordIndex(glossaryEntries));
}

function detectKeywordsWithIndex(text: string, keywordIndex: KeywordIndex): GlossaryEntry[] {
  const uniqueEntries = new Map<string, GlossaryEntry>();
  for (const match of selectKeywordMatches(text, keywordIndex)) {
    uniqueEntries.set(match.entry.id, match.entry);
  }
  return [...uniqueEntries.values()];
}

export function generateISLGloss(
  chunks: string[],
  glossaryEntries: GlossaryEntry[],
  options: GenerateGlossOptions = {}
): ISLGlossItem[] {
  return generateISLGlossWithIndex(chunks, createKeywordIndex(glossaryEntries), options);
}

function generateISLGlossWithIndex(
  chunks: string[],
  keywordIndex: KeywordIndex,
  options: GenerateGlossOptions,
  suppliedPlanningContext?: SourcePlanningContext
): PlannedGlossItem[] {
  const text = chunks.join(" ");
  const sourceSegmentId = options.sourceSegmentId ?? `segment_${stableHash(`standalone:${text}`)}`;
  const planningContext =
    suppliedPlanningContext ?? createSourcePlanningContext(text, sourceSegmentId);
  const matches = selectKeywordMatches(text, keywordIndex);
  const matchedItems = matches.map((match) => ({
    start: match.start,
    end: match.end,
    entry: match.entry,
    token: match.matchedText,
    resolvedUnmatchedToken: false
  }));
  const unmatchedItems = tokenizeSourceWords(text)
    .filter((token) => !matches.some((match) => token.start < match.end && match.start < token.end))
    .map((token) => {
      const entry = options.resolveUnmatchedToken?.(token.token);
      return {
        ...token,
        entry,
        resolvedUnmatchedToken: entry !== undefined
      };
    });
  const orderedItems = [...matchedItems, ...unmatchedItems].sort(
    (left, right) => left.start - right.start || left.end - right.end
  );

  return orderedItems.map<PlannedGlossItem>((sourceItem) => {
    const { entry, token } = sourceItem;
    const sourceSpan = { start: sourceItem.start, end: sourceItem.end };
    const sourceTokens = planningContext.sourceTokens.filter(
      (sourceToken) =>
        sourceToken.sourceSpan.start < sourceSpan.end &&
        sourceSpan.start < sourceToken.sourceSpan.end
    );
    const clauseId = sourceTokens[0]?.clauseId ?? planningContext.clauses[0]?.id;
    if (!sourceTokens.length || !clauseId) {
      throw new Error("Every planned gloss action must resolve to source tokens and a clause.");
    }
    const planningActionId = `action_${stableHash(
      `${sourceSegmentId}:${sourceSpan.start}:${sourceSpan.end}:${token}`
    )}`;
    const glossItemId = `gloss_${stableHash(planningActionId)}`;

    if (!entry) {
      const fallback = isLatinFingerspellable(token) ? "fingerspell" : "unknown";
      const action = {
        kind: fallback,
        label: token.toLocaleUpperCase()
      } satisfies PlanningActionDecision;
      return {
        id: glossItemId,
        token,
        gloss: action.label,
        isVerified: false,
        fallback,
        confidence: "low",
        sourceSegmentId,
        planningActionId,
        sourceTokenIds: sourceTokens.map((sourceToken) => sourceToken.id),
        sourceSpan,
        clauseId,
        plannedAction: action,
        resolvedAction: action,
        provenance: { source: "deterministic_fallback" },
        reviewStatus: "unreviewed"
      };
    }

    const hasVerifiedSignAsset =
      entry.reviewStatus === "approved" &&
      resolvesToExpertReviewedPlayableAsset(entry.signAssetId, options.resolveSignAsset);
    const fallback = hasVerifiedSignAsset
      ? "none"
      : isLatinFingerspellable(token)
        ? "fingerspell"
        : "caption";
    const item: PlannedGlossItem = {
      id: glossItemId,
      token,
      gloss: entry.islGloss,
      isVerified: hasVerifiedSignAsset,
      fallback,
      confidence: hasVerifiedSignAsset
        ? entry.confidence
        : capUnverifiedConfidence(entry.confidence),
      sourceSegmentId,
      planningActionId,
      sourceTokenIds: sourceTokens.map((sourceToken) => sourceToken.id),
      sourceSpan,
      clauseId,
      plannedAction: { kind: "sign_candidate", label: entry.islGloss },
      resolvedAction: {
        kind: hasVerifiedSignAsset ? "sign" : fallback === "none" ? "caption" : fallback,
        label: entry.islGloss
      },
      provenance: { source: "controlled_glossary", glossaryEntryId: entry.id },
      reviewStatus: sourceItem.resolvedUnmatchedToken ? "pending_review" : entry.reviewStatus
    };
    if (entry.signAssetId) {
      item.signAssetId = entry.signAssetId;
    }
    return item;
  });
}

function createSourcePlanningContext(
  sourceText: string,
  sourceSegmentId: string
): SourcePlanningContext {
  const sourceWords = tokenizeSourceWords(sourceText);
  const clauseRanges = createClauseRanges(sourceText, sourceWords);
  const clauseIds = clauseRanges.map(
    (range) => `clause_${stableHash(`${sourceSegmentId}:${range.start}:${range.end}`)}`
  );
  const sourceTokens = sourceWords.map<PlanningSourceToken>((word) => {
    const clauseIndex = clauseRanges.findIndex(
      (range) => range.start <= word.start && word.end <= range.end
    );
    const clauseId = clauseIds[Math.max(0, clauseIndex)];
    if (!clauseId) {
      throw new Error("Every source token must resolve to a planning clause.");
    }
    const normalizedText = normalizePlanningToken(word.token);
    return {
      id: `token_${stableHash(`${sourceSegmentId}:${word.start}:${word.end}:${normalizedText}`)}`,
      text: word.token,
      normalizedText,
      sourceSpan: { start: word.start, end: word.end },
      clauseId
    };
  });
  const clauses = clauseRanges.map<PlanningClause>((range, index) => {
    const clauseId = clauseIds[index]!;
    const clauseTokens = sourceTokens.filter((token) => token.clauseId === clauseId);
    return {
      id: clauseId,
      sourceSpan: range,
      sourceTokenIds: clauseTokens.map((token) => token.id),
      features: detectPlanningFeatures(sourceText.slice(range.start, range.end), clauseTokens)
    };
  });

  return {
    sourceSegmentId,
    sourceTokens,
    clauses,
    features: combinePlanningFeatures(clauses.map((clause) => clause.features))
  };
}

function createPlanningIR(
  sourceText: string,
  context: SourcePlanningContext,
  glossSequence: PlannedGlossItem[]
): InterpretationPlanningIR {
  const expertGate = {
    applied: false as const,
    status: "expert_gated" as const,
    reviewStatus: "pending_review" as const
  };

  return {
    id: `plan_${stableHash(
      `${context.sourceSegmentId}:${glossSequence.map((item) => item.planningActionId).join("|")}`
    )}`,
    sourceSegmentId: context.sourceSegmentId,
    sourceText,
    sourceTokens: context.sourceTokens,
    clauses: context.clauses,
    actions: glossSequence.map((item) => ({
      id: item.planningActionId,
      sourceText: sourceText.slice(item.sourceSpan.start, item.sourceSpan.end),
      sourceSpan: item.sourceSpan,
      sourceTokenIds: item.sourceTokenIds,
      clauseId: item.clauseId,
      plannedAction: item.plannedAction,
      resolvedAction: item.resolvedAction,
      provenance: item.provenance,
      reviewStatus: item.reviewStatus
    })),
    features: context.features,
    grammar: {
      sourceOrderPreserved: true,
      reordering: expertGate,
      nonManualSignals: expertGate
    },
    reviewStatus: "pending_review"
  };
}

function createClauseRanges(sourceText: string, sourceWords: SourceWord[]): SourceCharacterSpan[] {
  if (!sourceWords.length) {
    return [];
  }

  const ranges: SourceCharacterSpan[] = [];
  let firstWordIndex = 0;
  for (let index = 0; index < sourceWords.length; index += 1) {
    const word = sourceWords[index]!;
    const nextWord = sourceWords[index + 1];
    const boundaryEnd = nextWord?.start ?? sourceText.length;
    const separator = sourceText.slice(word.end, boundaryEnd);
    const isBoundary = /[.!?;]/u.test(separator) || !nextWord;
    if (!isBoundary) {
      continue;
    }

    const firstWord = sourceWords[firstWordIndex]!;
    const rawEnd = boundaryEnd;
    const trailingWhitespace =
      sourceText.slice(firstWord.start, rawEnd).match(/\s+$/u)?.[0].length ?? 0;
    ranges.push({ start: firstWord.start, end: rawEnd - trailingWhitespace });
    firstWordIndex = index + 1;
  }
  return ranges;
}

function detectPlanningFeatures(
  clauseText: string,
  clauseTokens: PlanningSourceToken[]
): PlanningFeature {
  const firstToken = clauseTokens[0];
  const questionCueTokenIds =
    firstToken && QUESTION_CUES.has(firstToken.normalizedText) ? [firstToken.id] : [];
  const negationCueTokenIds = clauseTokens
    .filter((token) => NEGATION_CUES.has(token.normalizedText))
    .map((token) => token.id);
  return {
    isQuestion: clauseText.includes("?") || questionCueTokenIds.length > 0,
    hasNegation: negationCueTokenIds.length > 0,
    questionCueTokenIds,
    negationCueTokenIds
  };
}

function combinePlanningFeatures(features: PlanningFeature[]): PlanningFeature {
  return {
    isQuestion: features.some((feature) => feature.isQuestion),
    hasNegation: features.some((feature) => feature.hasNegation),
    questionCueTokenIds: features.flatMap((feature) => feature.questionCueTokenIds),
    negationCueTokenIds: features.flatMap((feature) => feature.negationCueTokenIds)
  };
}

function normalizePlanningToken(value: string): string {
  return value.normalize("NFKC").replace(/[’]/gu, "'").toLocaleLowerCase();
}

export function calculateConfidence(input: ConfidenceInput): ConfidenceLevel {
  if (input.totalGlossItems === 0 || input.transcriptConfidence < 0.5) {
    return "low";
  }

  const coverage = input.glossaryMatches / input.totalGlossItems;
  const verifiedCoverage = input.verifiedSigns / input.totalGlossItems;

  if (coverage >= 0.95 && verifiedCoverage >= 0.95 && input.unknownTerms === 0) {
    return "high";
  }

  if (coverage >= 0.55 && verifiedCoverage >= 0.5 && input.transcriptConfidence >= 0.7) {
    return "medium";
  }

  return "low";
}

export function interpretTranscript(input: InterpretInput): ISLInterpretation {
  const cleanedText = cleanTranscript(input.rawText);
  const chunks = getMeaningChunks(cleanedText, input.splitMeaningChunks);
  const simplified = simplifyChunks(chunks, input.simplificationLevel);
  const simplifiedText = simplified.map((chunk) => chunk.text).join(" ");
  const glossaryEntries = input.glossaryEntries ?? seedGlossary;
  const keywordIndex = createKeywordIndex(glossaryEntries);
  const sourceSegmentId = createSourceSegmentId(input);
  const planningContext = createSourcePlanningContext(input.rawText, sourceSegmentId);
  const glossOptions: GenerateGlossOptions = {
    sourceSegmentId,
    ...(input.resolveSignAsset ? { resolveSignAsset: input.resolveSignAsset } : {}),
    ...(input.resolveUnmatchedToken ? { resolveUnmatchedToken: input.resolveUnmatchedToken } : {})
  };
  const glossSequence = generateISLGlossWithIndex(
    [input.rawText],
    keywordIndex,
    glossOptions,
    planningContext
  );
  const planning = createPlanningIR(input.rawText, planningContext, glossSequence);
  const unknownTerms = glossSequence.filter((item) => item.fallback !== "none").length;
  const glossaryMatches = glossSequence.length - unknownTerms;
  const verifiedSigns = glossSequence.filter((item) => item.isVerified).length;
  const confidence = calculateConfidence({
    transcriptConfidence: input.transcriptConfidence ?? 0.9,
    glossaryMatches,
    unknownTerms,
    verifiedSigns,
    totalGlossItems: glossSequence.length
  });
  const warnings =
    unknownTerms > 0
      ? ["Some terms use fallback output."]
      : confidence === "low"
        ? ["This segment may be inaccurate. Showing captions and summary."]
        : [];

  return {
    id: createInterpretationId(input),
    sessionId: input.sessionId,
    chunkId: simplified[0]?.id ?? "chunk_1",
    originalText: input.rawText,
    cleanedText,
    simplifiedText,
    keyPoints: simplified.flatMap((chunk) => chunk.meaningUnits),
    detectedTerms: detectKeywordsWithIndex(`${cleanedText} ${simplifiedText}`, keywordIndex).map(
      (entry) => entry.term
    ),
    glossSequence,
    planning,
    confidence,
    warnings,
    createdAt: input.now?.() ?? new Date().toISOString()
  };
}

function createSourceSegmentId(input: InterpretInput): string {
  const timestampIdentity = getTimestampIdentity(input);
  const identity = timestampIdentity
    ? `${input.sessionId}:${input.source}:${timestampIdentity}`
    : `${input.sessionId}:${input.source}:${input.rawText}`;
  return timestampIdentity
    ? `segment_${timestampIdentity.replace(":", "_")}_${stableHash(identity)}`
    : `segment_${stableHash(identity)}`;
}

function createInterpretationId(input: InterpretInput): string {
  const timestampIdentity = getTimestampIdentity(input);
  const identity = timestampIdentity
    ? `${input.sessionId}:${input.rawText}:${timestampIdentity}`
    : `${input.sessionId}:${input.rawText}`;
  return timestampIdentity
    ? `interp_${timestampIdentity.replace(":", "_")}_${stableHash(identity)}`
    : `interp_${stableHash(identity)}`;
}

function getTimestampIdentity(input: InterpretInput): string | undefined {
  if (input.startMs === undefined && input.endMs === undefined) {
    return undefined;
  }
  return `${input.startMs ?? "unset"}:${input.endMs ?? "unset"}`;
}

function stronglySimplifyChunk(chunk: string): string {
  const contentWords = chunk
    .replace(/[.!?]+$/, "")
    .split(/\s+/)
    .filter(Boolean)
    .filter((word) => !STOP_WORDS.has(word.toLowerCase().replace(/[^\p{L}\p{N}-]/gu, "")))
    .slice(0, 10);

  if (contentWords.length < 2) {
    return chunk;
  }

  const text = contentWords.join(" ");
  return `${text.charAt(0).toUpperCase()}${text.slice(1)}.`;
}

function splitLongChunk(chunk: string): string[] {
  const words = chunk.split(/\s+/).filter(Boolean);
  if (words.length <= MAX_CHUNK_WORDS) {
    return [chunk];
  }

  const terminal = chunk.match(/[.!?]$/)?.[0] ?? "";
  const strippedWords = words.map((word, index) => {
    if (index === words.length - 1 && terminal) {
      return word.replace(/[.!?]$/, "");
    }
    return word;
  });
  const chunks: string[] = [];
  for (let index = 0; index < strippedWords.length; index += MAX_CHUNK_WORDS) {
    const next = strippedWords.slice(index, index + MAX_CHUNK_WORDS).join(" ");
    chunks.push(
      index + MAX_CHUNK_WORDS >= strippedWords.length && terminal ? `${next}${terminal}` : next
    );
  }
  return chunks;
}

type KeywordMatch = {
  entry: GlossaryEntry;
  entryIndex: number;
  isCanonicalTerm: boolean;
  start: number;
  end: number;
  matchedText: string;
};

type SourceWord = {
  token: string;
  start: number;
  end: number;
};

type IndexedKeywordCandidate = {
  entry: GlossaryEntry;
  entryIndex: number;
  isCanonicalTerm: boolean;
  patternSource: string;
  pattern: RegExp | undefined;
};

type KeywordIndex = {
  candidatesByFirstPart: Map<string, IndexedKeywordCandidate[]>;
  unkeyedCandidates: IndexedKeywordCandidate[];
};

// Candidate patterns stay lazy so matching only compiles and scans buckets present in the text.
const keywordPatternCache = new Map<string, RegExp>();
const keywordIndexCache = new WeakMap<GlossaryEntry[], KeywordIndex>();

function selectKeywordMatches(text: string, index: KeywordIndex): KeywordMatch[] {
  const relevantCandidates = [...index.unkeyedCandidates];
  const relevantFirstParts = new Set(
    [...text.matchAll(/[\p{L}\p{N}]+/gu)].map((match) => keywordIndexKey(match[0]))
  );

  for (const firstPart of relevantFirstParts) {
    const candidates = index.candidatesByFirstPart.get(firstPart);
    if (candidates) {
      relevantCandidates.push(...candidates);
    }
  }

  const candidates = relevantCandidates.flatMap((candidate) => {
    const pattern = candidate.pattern ?? compileKeywordPattern(candidate.patternSource);
    candidate.pattern = pattern;

    return [...text.matchAll(pattern)].map((match): KeywordMatch => ({
      entry: candidate.entry,
      entryIndex: candidate.entryIndex,
      isCanonicalTerm: candidate.isCanonicalTerm,
      start: match.index,
      end: match.index + match[0].length,
      matchedText: match[0]
    }));
  });
  candidates.sort(
    (left, right) =>
      right.end - right.start - (left.end - left.start) ||
      left.start - right.start ||
      Number(right.isCanonicalTerm) - Number(left.isCanonicalTerm) ||
      left.entryIndex - right.entryIndex
  );

  const selected: KeywordMatch[] = [];
  for (const candidate of candidates) {
    const overlaps = selected.some(
      (match) => candidate.start < match.end && match.start < candidate.end
    );
    if (!overlaps) {
      selected.push(candidate);
    }
  }

  return selected.sort(
    (left, right) => left.start - right.start || left.entryIndex - right.entryIndex
  );
}

function createKeywordIndex(glossaryEntries: GlossaryEntry[]): KeywordIndex {
  const cached = keywordIndexCache.get(glossaryEntries);
  if (cached) {
    return cached;
  }
  const index: KeywordIndex = {
    candidatesByFirstPart: new Map(),
    unkeyedCandidates: []
  };

  glossaryEntries.forEach((entry, entryIndex) => {
    for (const [candidateIndex, candidate] of [entry.term, ...entry.aliases].entries()) {
      const patternSource = phrasePatternSource(candidate);
      if (!patternSource) {
        continue;
      }

      const indexedCandidate: IndexedKeywordCandidate = {
        entry,
        entryIndex,
        isCanonicalTerm: candidateIndex === 0,
        patternSource,
        pattern: undefined
      };
      const firstPart = candidate.match(/[\p{L}\p{N}]+/u)?.[0];
      if (!firstPart) {
        index.unkeyedCandidates.push(indexedCandidate);
        continue;
      }

      const key = keywordIndexKey(firstPart);
      const bucket = index.candidatesByFirstPart.get(key);
      if (bucket) {
        bucket.push(indexedCandidate);
      } else {
        index.candidatesByFirstPart.set(key, [indexedCandidate]);
      }
    }
  });

  keywordIndexCache.set(glossaryEntries, index);
  return index;
}

function tokenizeSourceWords(text: string): SourceWord[] {
  const pattern = /[\p{L}\p{M}\p{N}]+(?:['’-][\p{L}\p{M}\p{N}]+)*/gu;
  return [...text.matchAll(pattern)].map((match) => ({
    token: match[0],
    start: match.index,
    end: match.index + match[0].length
  }));
}

function isLatinFingerspellable(token: string): boolean {
  const letters = token
    .normalize("NFKD")
    .replace(/\p{M}/gu, "")
    .replace(/[^A-Za-z]/g, "");
  return letters.length > 0;
}

function phrasePatternSource(candidate: string): string | undefined {
  const words = candidate.trim().split(/\s+/u).filter(Boolean);
  if (words.length === 0) {
    return undefined;
  }

  return words.map(escapeRegExp).join("\\s+");
}

function compileKeywordPattern(patternSource: string): RegExp {
  const cached = keywordPatternCache.get(patternSource);
  if (cached) {
    return cached;
  }

  const pattern = new RegExp(
    `(?<![\\p{L}\\p{N}'’-])${patternSource}(?![\\p{L}\\p{N}'’-])`,
    "giu"
  );
  keywordPatternCache.set(patternSource, pattern);
  return pattern;
}

function keywordIndexKey(value: string): string {
  return value.normalize("NFKC").toUpperCase();
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

function capUnverifiedConfidence(confidence: ConfidenceLevel): ConfidenceLevel {
  return confidence === "low" ? "low" : "medium";
}

function escapeRegExp(value: string): string {
  return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

function stableHash(value: string): string {
  let hash = 0;
  for (let index = 0; index < value.length; index += 1) {
    hash = (hash * 31 + value.charCodeAt(index)) >>> 0;
  }
  return hash.toString(36);
}
