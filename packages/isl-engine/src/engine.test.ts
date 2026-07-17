import { interpretationSchema, type GlossaryEntry } from "@signsaarthi/shared";
import { describe, expect, it, vi } from "vitest";
import {
  calculateConfidence,
  cleanTranscript,
  detectKeywords,
  generateISLGloss,
  interpretTranscript,
  seedGlossary,
  simplifyChunks,
  splitIntoMeaningChunks
} from "./index";

describe("ISL engine", () => {
  it("cleans repeated filler words while preserving technical terms", () => {
    expect(cleanTranscript("um today today we will learn AI and artificial intelligence")).toBe(
      "Today we will learn AI and artificial intelligence."
    );
  });

  it("splits long transcript into short meaning chunks", () => {
    const chunks = splitIntoMeaningChunks(
      "Today we will explore artificial intelligence. It can break barriers and create equal opportunities in education."
    );

    expect(chunks).toEqual([
      "Today we will explore artificial intelligence.",
      "It can break barriers and create equal opportunities in education."
    ]);
  });

  it("uses a local chunk planner only when it preserves every cleaned word", () => {
    const preserved = interpretTranscript({
      sessionId: "session_chunker",
      source: "mock",
      rawText: "First idea and second idea.",
      splitMeaningChunks: () => ["First idea", "and second idea."]
    });
    const rejected = interpretTranscript({
      sessionId: "session_chunker_rejected",
      source: "mock",
      rawText: "First idea and second idea.",
      splitMeaningChunks: () => ["First idea."]
    });

    expect(preserved.simplifiedText).toBe("First idea and second idea.");
    expect(rejected.simplifiedText).toBe("First idea and second idea.");
    expect(rejected.glossSequence.map((item) => item.token).join(" ")).toContain("second");
  });

  it("splits a long unpunctuated transcript after cleanup adds punctuation", () => {
    const chunks = splitIntoMeaningChunks(
      "Today artificial intelligence accessibility education meeting internet inclusion equal opportunity barriers help learn students support captions glossary avatar replay."
    );

    expect(chunks.length).toBeGreaterThan(1);
    expect(chunks.every((chunk) => chunk.split(/\s+/).length <= 12)).toBe(true);
  });

  it("simplifies transcript chunks into signing-friendly meaning units", () => {
    const simplified = simplifyChunks([
      "Today we will explore how artificial intelligence can break barriers.",
      "It can create equal opportunities in education."
    ]);

    expect(simplified.map((chunk) => chunk.text)).toEqual([
      "Today topic: Artificial Intelligence. AI can reduce barriers.",
      "AI can help education become more equal."
    ]);
  });

  it("keeps both barrier and equal-opportunity meaning when both appear in one sentence", () => {
    const [simplified] = simplifyChunks([
      "Today artificial intelligence can break barriers and create equal opportunities in education."
    ]);

    expect(simplified?.meaningUnits).toContain("AI can reduce barriers");
    expect(simplified?.meaningUnits).toContain("AI can help education become more equal");
  });

  it("supports light and strong simplification levels deterministically", () => {
    const chunk = "The project will support students with accessible education in the classroom.";

    expect(simplifyChunks([chunk], "light")[0]?.text).toBe(chunk);
    expect(simplifyChunks([chunk], "strong")[0]?.text).toBe(
      "Project accessible education classroom."
    );
  });

  it("detects controlled glossary phrases before individual words", () => {
    const terms = detectKeywords(
      "Artificial intelligence supports equal opportunity in education.",
      seedGlossary
    );

    expect(terms.map((term) => term.term)).toEqual([
      "artificial intelligence",
      "equal opportunity",
      "education"
    ]);
  });

  it("prefers a canonical term over another entry's identical alias", () => {
    const [item] = generateISLGloss(["Learn."], seedGlossary);

    expect(item).toMatchObject({ token: "Learn", gloss: "LEARN" });
  });

  it("selects longest matches by span without blocking shared words elsewhere", () => {
    const entries = [
      glossaryEntry("good", "GOOD"),
      glossaryEntry("good evening", "GOOD EVENING"),
      glossaryEntry("good night", "GOOD NIGHT")
    ];

    const terms = detectKeywords("Good evening and good night.", entries);

    expect(terms.map((term) => term.islGloss)).toEqual(["GOOD EVENING", "GOOD NIGHT"]);
  });

  it("keeps repeated occurrences in the signing sequence while deduplicating detected terms", () => {
    const entries = [glossaryEntry("thank you", "THANK YOU")];

    expect(detectKeywords("Thank you, thank you.", entries)).toHaveLength(1);
    expect(generateISLGloss(["Thank you, thank you."], entries).map((item) => item.gloss)).toEqual([
      "THANK YOU",
      "THANK YOU"
    ]);
  });

  it("emits a lossless source plan with spans, clause IDs, and detected question and negation cues", () => {
    const interpretation = interpretTranscript({
      sessionId: "session_plan",
      source: "youtube_captions",
      rawText: "Can we not learn learn?",
      startMs: 1_000,
      endMs: 5_000
    });

    expect(
      interpretation.planning?.sourceTokens.map((token) => [token.text, token.sourceSpan])
    ).toEqual([
      ["Can", { start: 0, end: 3 }],
      ["we", { start: 4, end: 6 }],
      ["not", { start: 7, end: 10 }],
      ["learn", { start: 11, end: 16 }],
      ["learn", { start: 17, end: 22 }]
    ]);
    expect(interpretation.planning?.features).toMatchObject({
      isQuestion: true,
      hasNegation: true
    });
    expect(interpretation.planning?.actions.flatMap((action) => action.sourceTokenIds)).toEqual(
      interpretation.planning?.sourceTokens.map((token) => token.id)
    );
    expect(interpretation.planning?.grammar).toMatchObject({
      sourceOrderPreserved: true,
      reordering: { applied: false, status: "expert_gated" },
      nonManualSignals: { applied: false, status: "expert_gated" }
    });
    expect(new Set(interpretation.glossSequence.map((item) => item.id)).size).toBe(
      interpretation.glossSequence.length
    );
    expect(interpretationSchema.parse(interpretation).planning?.sourceSegmentId).toBe(
      interpretation.planning?.sourceSegmentId
    );
  });

  it("includes supplied timestamps in interpretation and segment identity", () => {
    const baseInput = {
      sessionId: "session_timestamps",
      source: "youtube_captions" as const,
      rawText: "Today we learn.",
      endMs: 4_000
    };
    const first = interpretTranscript({ ...baseInput, startMs: 0 });
    const repeated = interpretTranscript({ ...baseInput, startMs: 0 });
    const later = interpretTranscript({ ...baseInput, startMs: 1_500, endMs: 5_500 });

    expect(repeated.id).toBe(first.id);
    expect(first.id).toContain("0_4000");
    expect(repeated.planning?.sourceSegmentId).toBe(first.planning?.sourceSegmentId);
    expect(later.id).not.toBe(first.id);
    expect(later.planning?.sourceSegmentId).not.toBe(first.planning?.sourceSegmentId);
  });

  it("keeps rolling-prefix occurrence IDs stable only within the same timestamped source segment", () => {
    const baseInput = {
      sessionId: "session_rolling",
      source: "youtube_captions" as const,
      startMs: 2_000,
      endMs: 6_000
    };
    const first = interpretTranscript({ ...baseInput, rawText: "Today we" });
    const extended = interpretTranscript({ ...baseInput, rawText: "Today we learn" });
    const nextSegment = interpretTranscript({
      ...baseInput,
      rawText: "Today we",
      startMs: 6_000,
      endMs: 10_000
    });

    expect(extended.glossSequence.slice(0, 2).map((item) => item.id)).toEqual(
      first.glossSequence.map((item) => item.id)
    );
    expect(nextSegment.glossSequence.map((item) => item.id)).not.toEqual(
      first.glossSequence.map((item) => item.id)
    );
  });

  it("keeps punctuation-sensitive terms distinct while preserving repetition and order", () => {
    const entries = [
      glossaryEntry("B.Ed", "B.ED"),
      glossaryEntry("B.Ed course", "B.ED COURSE"),
      glossaryEntry("Bed", "BED")
    ];

    const gloss = generateISLGloss(["B.Ed course and Bed, then B.Ed."], entries);

    expect(gloss.map((item) => [item.token, item.gloss])).toEqual([
      ["B.Ed course", "B.ED COURSE"],
      ["and", "AND"],
      ["Bed", "BED"],
      ["then", "THEN"],
      ["B.Ed", "B.ED"]
    ]);
    expect(
      detectKeywords("B.Ed course and Bed, then B.Ed.", entries).map((entry) => entry.term)
    ).toEqual(["B.Ed course", "Bed", "B.Ed"]);
  });

  it("compiles and scans only relevant buckets in a 12,380-entry lexicon", () => {
    const entries = Array.from({ length: 12_379 }, (_, index) =>
      glossaryEntry(`unused${index}`, `UNUSED ${index}`)
    );
    const needle = glossaryEntry("needle", "NEEDLE");
    entries.push(needle);

    const OriginalRegExp = globalThis.RegExp;
    const compiledPatterns: string[] = [];
    const RegExpSpy = function (pattern: string | RegExp, flags?: string): RegExp {
      compiledPatterns.push(String(pattern));
      return new OriginalRegExp(pattern, flags);
    };
    vi.stubGlobal("RegExp", RegExpSpy);
    const matchAllSpy = vi.spyOn(String.prototype, "matchAll");

    try {
      expect(detectKeywords("Needle, needle.", entries)).toEqual([needle]);
      expect(detectKeywords("Needle, needle.", entries)).toEqual([needle]);
      expect(compiledPatterns).toHaveLength(1);
      expect(matchAllSpy).toHaveBeenCalledTimes(4);
    } finally {
      matchAllSpy.mockRestore();
      vi.unstubAllGlobals();
    }
  });

  it("keeps every source word in order without duplicating simplified output", () => {
    const entries = [
      glossaryEntry("today", "TODAY"),
      glossaryEntry("we", "WE"),
      glossaryEntry("computer", "COMPUTER"),
      glossaryEntry("thank you", "THANK YOU")
    ];

    const interpretation = interpretTranscript({
      sessionId: "session_queue",
      source: "mock",
      rawText: "Today we use a computer. Thank you.",
      glossaryEntries: entries
    });

    expect(interpretation.glossSequence.map((item) => item.gloss)).toEqual([
      "TODAY",
      "WE",
      "USE",
      "A",
      "COMPUTER",
      "THANK YOU"
    ]);
  });

  it("uses a lexical resolver only for unmatched tokens without changing source identity", () => {
    const computer = glossaryEntry("computer", "COMPUTER");
    const input = {
      sessionId: "session_lexical",
      source: "mock" as const,
      rawText: "computre",
      glossaryEntries: [computer],
      now: () => "2026-07-16T00:00:00.000Z"
    };
    const fallback = interpretTranscript(input);
    const resolved = interpretTranscript({
      ...input,
      resolveUnmatchedToken: (token: string) => (token === "computre" ? computer : undefined)
    });

    expect(resolved.glossSequence[0]).toMatchObject({
      token: "computre",
      gloss: "COMPUTER",
      isVerified: false,
      fallback: "fingerspell",
      provenance: { source: "controlled_glossary", glossaryEntryId: computer.id },
      reviewStatus: "pending_review"
    });
    expect(resolved.planning?.sourceTokens[0]).toMatchObject({
      text: "computre",
      sourceSpan: { start: 0, end: 8 }
    });
    expect(resolved.glossSequence[0]?.planningActionId).toBe(
      fallback.glossSequence[0]?.planningActionId
    );
    expect(resolved.planning?.actions[0]?.id).toBe(fallback.planning?.actions[0]?.id);
    expect(resolved.planning?.actions[0]?.sourceSpan).toEqual(
      fallback.planning?.actions[0]?.sourceSpan
    );
  });

  it("does not call the lexical resolver for exact controlled-glossary matches", () => {
    const computer = glossaryEntry("computer", "COMPUTER");
    let resolverCalls = 0;
    const input = {
      sessionId: "session_exact",
      source: "mock" as const,
      rawText: "computer",
      glossaryEntries: [computer],
      now: () => "2026-07-16T00:00:00.000Z"
    };
    const baseline = interpretTranscript(input);
    const withResolver = interpretTranscript({
      ...input,
      resolveUnmatchedToken: () => {
        resolverCalls += 1;
        return computer;
      }
    });

    expect(resolverCalls).toBe(0);
    expect(withResolver).toEqual(baseline);
  });

  it("keeps matcher hits pending and preserves the expert asset verification gate", () => {
    const computer = glossaryEntry("computer", "COMPUTER", {
      signAssetId: "asset_computer_001",
      confidence: "high",
      source: "expert_reviewed",
      reviewStatus: "approved"
    });
    const input = {
      sessionId: "session_matcher_gate",
      source: "mock" as const,
      rawText: "computre",
      glossaryEntries: [computer],
      resolveUnmatchedToken: () => computer
    };
    const withoutAssetGate = interpretTranscript(input);
    const withAssetGate = interpretTranscript({
      ...input,
      resolveSignAsset: (assetId: string) =>
        assetId === "asset_computer_001" ? { expertReviewed: true, playable: true } : undefined
    });

    expect(withoutAssetGate.glossSequence[0]).toMatchObject({
      token: "computre",
      isVerified: false,
      fallback: "fingerspell",
      reviewStatus: "pending_review"
    });
    expect(withAssetGate.glossSequence[0]).toMatchObject({
      token: "computre",
      isVerified: true,
      fallback: "none",
      reviewStatus: "pending_review"
    });
  });

  it("keeps filler and repeated raw words in actions while cleaning only the summary text", () => {
    const interpretation = interpretTranscript({
      sessionId: "session_raw_words",
      source: "mock",
      rawText: "um today today we learn"
    });

    expect(interpretation.cleanedText).toBe("Today we learn.");
    expect(interpretation.glossSequence.map((item) => item.token)).toEqual([
      "um",
      "today",
      "today",
      "we",
      "learn"
    ]);
  });

  it("keeps the complete original word sequence when simplified text differs", () => {
    const interpretation = interpretTranscript({
      sessionId: "session_named_fallback",
      source: "mock",
      rawText: "Artificial intelligence can help education in Mumbai.",
      glossaryEntries: seedGlossary
    });

    expect(interpretation.glossSequence.map((item) => item.token)).toEqual([
      "Artificial intelligence",
      "can",
      "help",
      "education",
      "in",
      "Mumbai"
    ]);
  });

  it("preserves apostrophized source words instead of matching a shorter glossary prefix", () => {
    const interpretation = interpretTranscript({
      sessionId: "session_apostrophe",
      source: "mock",
      rawText: "All right, so today's lecture.",
      glossaryEntries: [glossaryEntry("today", "TODAY")]
    });

    expect(interpretation.glossSequence.map((item) => item.token)).toEqual([
      "All",
      "right",
      "so",
      "today's",
      "lecture"
    ]);
  });

  it("uses official-source fingerspelling for every unresolved Latin word", () => {
    const gloss = generateISLGloss(["AI can help education and Mumbai students."], seedGlossary);

    expect(gloss.map((item) => [item.token, item.fallback])).toEqual([
      ["AI", "fingerspell"],
      ["can", "fingerspell"],
      ["help", "fingerspell"],
      ["education", "fingerspell"],
      ["and", "fingerspell"],
      ["Mumbai", "fingerspell"],
      ["students", "fingerspell"]
    ]);
    expect(seedGlossary.every((entry) => entry.reviewStatus === "pending_review")).toBe(true);
  });

  it("fingerspells ordinary unresolved words regardless of capitalization", () => {
    const gloss = generateISLGloss(["Quantum systems involve Mumbai."], []);

    expect(gloss.map((item) => [item.token, item.fallback])).toEqual([
      ["Quantum", "fingerspell"],
      ["systems", "fingerspell"],
      ["involve", "fingerspell"],
      ["Mumbai", "fingerspell"]
    ]);
  });

  it("emits a visible fallback action for every unmatched word without a length cap", () => {
    const gloss = generateISLGloss(
      ["One two three four five six seven eight nine ten and a half."],
      []
    );

    expect(gloss.map((item) => item.token)).toEqual([
      "One",
      "two",
      "three",
      "four",
      "five",
      "six",
      "seven",
      "eight",
      "nine",
      "ten",
      "and",
      "a",
      "half"
    ]);
    expect(gloss.every((item) => item.fallback !== "none")).toBe(true);
  });

  it("requires an approved expert-reviewed playable asset for verification and high confidence", () => {
    const entry = glossaryEntry("verified concept", "VERIFIED CONCEPT", {
      signAssetId: "asset_verified_001",
      confidence: "high",
      source: "expert_reviewed",
      reviewStatus: "approved"
    });
    const unresolved = interpretTranscript({
      sessionId: "session_unresolved",
      source: "mock",
      rawText: "verified concept",
      glossaryEntries: [entry]
    });
    const resolved = interpretTranscript({
      sessionId: "session_resolved",
      source: "mock",
      rawText: "verified concept",
      glossaryEntries: [entry],
      resolveSignAsset: (assetId) =>
        assetId === "asset_verified_001" ? { expertReviewed: true, playable: true } : undefined
    });

    expect(unresolved.glossSequence[0]).toMatchObject({
      isVerified: false,
      fallback: "fingerspell",
      confidence: "medium"
    });
    expect(unresolved.confidence).toBe("low");
    expect(resolved.glossSequence[0]).toMatchObject({
      isVerified: true,
      fallback: "none",
      confidence: "high"
    });
    expect(resolved.confidence).toBe("high");
  });

  it("uses fingerspelling for text-only dataset vocabulary without a reviewed sign asset", () => {
    const gloss = generateISLGloss(
      ["We reflect together."],
      [
        {
          id: "model_reflect",
          term: "reflect",
          aliases: [],
          language: "en",
          category: "general",
          islGloss: "REFLECT",
          confidence: "medium",
          source: "external_reference",
          reviewStatus: "pending_review",
          createdAt: "2026-07-07T00:00:00.000Z",
          updatedAt: "2026-07-07T00:00:00.000Z"
        }
      ]
    );

    expect(gloss.find((item) => item.token === "reflect")).toMatchObject({
      token: "reflect",
      isVerified: false,
      fallback: "fingerspell",
      confidence: "medium"
    });
  });

  it("emits fingerspelling fallbacks for lower-case unmapped concepts and lowers confidence", () => {
    const interpretation = interpretTranscript({
      sessionId: "session_1",
      source: "mock",
      rawText: "today quantum entanglement rocks",
      now: () => "2026-07-07T00:00:00.000Z"
    });

    expect(interpretation.glossSequence.map((item) => [item.token, item.fallback])).toEqual(
      expect.arrayContaining([
        ["today", "fingerspell"],
        ["quantum", "fingerspell"],
        ["entanglement", "fingerspell"],
        ["rocks", "fingerspell"]
      ])
    );
    expect(interpretation.confidence).toBe("low");
    expect(interpretation.warnings).toContain("Some terms use fallback output.");
  });

  it("scores confidence by glossary coverage and verified sign availability", () => {
    const confidence = calculateConfidence({
      transcriptConfidence: 0.9,
      glossaryMatches: 3,
      unknownTerms: 1,
      verifiedSigns: 3,
      totalGlossItems: 4
    });

    expect(confidence).toBe("medium");
  });

  it("returns a full interpretation with warnings for fallback output", () => {
    const interpretation = interpretTranscript({
      sessionId: "session_1",
      source: "mock",
      rawText: "Artificial intelligence can help education in Mumbai.",
      now: () => "2026-07-07T00:00:00.000Z"
    });

    expect(interpretation.simplifiedText).toContain("AI can help education");
    expect(interpretation.confidence).toBe("low");
    expect(interpretation.warnings).toContain("Some terms use fallback output.");
  });
});

function glossaryEntry(
  term: string,
  islGloss: string,
  overrides: Partial<GlossaryEntry> = {}
): GlossaryEntry {
  return {
    id: `test_${term.replace(/\s+/g, "_")}`,
    term,
    aliases: [],
    language: "en",
    category: "general",
    islGloss,
    confidence: "medium",
    source: "internal",
    reviewStatus: "pending_review",
    createdAt: "2026-07-07T00:00:00.000Z",
    updatedAt: "2026-07-07T00:00:00.000Z",
    ...overrides
  };
}
