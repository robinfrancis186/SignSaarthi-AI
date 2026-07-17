import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { islDatasetManifestSchema, type ISLDatasetManifest } from "../packages/isl-model/src/index";
import type { GlossaryEntry } from "@signsaarthi/shared";

type CsvRow = Record<string, string>;

type WordStats = {
  word: string;
  descriptions: string[];
  examples: string[];
  translationCount: number;
};

const baseManifestPath = resolve(process.cwd(), process.argv[2] ?? "data/isl/isl-dataset-manifest.json");
const isignRawDir = resolve(process.cwd(), process.argv[3] ?? "data/raw/isign");
const outputPath = resolve(process.cwd(), process.argv[4] ?? baseManifestPath);

const baseManifest = islDatasetManifestSchema.parse(JSON.parse(readFileSync(baseManifestPath, "utf8")));
const descriptions = parseCsv(readFileSync(resolve(isignRawDir, "word-description-dataset_v1.1.csv"), "utf8"));
const examples = parseCsv(readFileSync(resolve(isignRawDir, "word-presence-dataset_v1.1.csv"), "utf8"));
const translations = parseCsv(readFileSync(resolve(isignRawDir, "iSign_v1.1.csv"), "utf8"));

const stats = new Map<string, WordStats>();
for (const row of descriptions) {
  const word = normalizeTerm(row["word"]);
  if (!word) {
    continue;
  }
  getStats(stats, word).descriptions.push(row["sentence"] ?? "");
}

for (const row of examples) {
  const word = normalizeTerm(row["word"]);
  if (!word) {
    continue;
  }
  getStats(stats, word).examples.push(row["sentence"] ?? "");
}

const translationTexts = translations.map((row) => row["text"] ?? "").filter(Boolean);
for (const entry of stats.values()) {
  entry.translationCount = countPhraseMatches(translationTexts, entry.word);
}

const existingByTerm = new Map(baseManifest.records.map((record) => [normalizeTerm(record.term), record]));
const generatedRecords: ISLDatasetManifest["records"] = [];
for (const entry of [...stats.values()].sort((left, right) => left.word.localeCompare(right.word))) {
  const existing = existingByTerm.get(entry.word);
  const frequency = Math.max(1, entry.examples.length + entry.descriptions.length + entry.translationCount);
  if (existing) {
    existing.frequency = Math.max(existing.frequency, frequency);
    if (!existing.datasetIds.includes("isign")) {
      existing.datasetIds.push("isign");
    }
    if (existing.datasetIds.every((datasetId) => datasetId === "isign")) {
      existing.reviewStatus = "pending_review";
    }
    existing.aliases = unique(existing.aliases.map(normalizeTerm).filter(Boolean));
    continue;
  }

  generatedRecords.push({
    term: entry.word,
    gloss: toGloss(entry.word),
    aliases: aliasesFor(entry.word),
    category: inferCategory(entry.word, [...entry.descriptions, ...entry.examples]),
    datasetIds: ["isign"],
    frequency,
    reviewStatus: "pending_review"
  });
}

const nextManifest = islDatasetManifestSchema.parse({
  ...baseManifest,
  generatedAt: new Date().toISOString(),
  records: [...baseManifest.records, ...generatedRecords].sort(
    (left, right) => right.frequency - left.frequency || left.term.localeCompare(right.term)
  )
});

mkdirSync(dirname(outputPath), { recursive: true });
writeFileSync(outputPath, `${JSON.stringify(nextManifest, null, 2)}\n`);

const isignRecordCount = nextManifest.records.filter((record) => record.datasetIds.includes("isign")).length;
console.log(
  `Built iSign text manifest with ${nextManifest.records.length} record(s), ${isignRecordCount} iSign-backed term(s), ` +
    `${translations.length} translation row(s), and ${stats.size} explicit word task term(s): ${outputPath}`
);

function getStats(statsMap: Map<string, WordStats>, word: string): WordStats {
  const existing = statsMap.get(word);
  if (existing) {
    return existing;
  }
  const created: WordStats = { word, descriptions: [], examples: [], translationCount: 0 };
  statsMap.set(word, created);
  return created;
}

function parseCsv(input: string): CsvRow[] {
  const rows: string[][] = [];
  let row: string[] = [];
  let field = "";
  let quoted = false;

  for (let index = 0; index < input.length; index += 1) {
    const character = input[index];
    const next = input[index + 1];
    if (quoted) {
      if (character === "\"" && next === "\"") {
        field += "\"";
        index += 1;
      } else if (character === "\"") {
        quoted = false;
      } else {
        field += character;
      }
    } else if (character === "\"") {
      quoted = true;
    } else if (character === ",") {
      row.push(field);
      field = "";
    } else if (character === "\n") {
      row.push(field);
      rows.push(row);
      row = [];
      field = "";
    } else if (character !== "\r") {
      field += character;
    }
  }

  if (field || row.length) {
    row.push(field);
    rows.push(row);
  }

  const [headers, ...body] = rows.filter((candidate) => candidate.some((value) => value.trim()));
  if (!headers) {
    return [];
  }

  return body.map((values) =>
    Object.fromEntries(headers.map((header, index) => [header, values[index] ?? ""]))
  );
}

function normalizeTerm(value: string | undefined): string {
  return (value ?? "").trim().toLowerCase().replace(/\s+/g, " ");
}

function toGloss(term: string): string {
  return term.replace(/[^a-z0-9]+/g, " ").trim().toUpperCase();
}

function aliasesFor(term: string): string[] {
  if (!term.includes(" ")) {
    return [];
  }
  return unique([term.replace(/\s+/g, "-"), term.replace(/\s+/g, "_")]);
}

function countPhraseMatches(texts: string[], phrase: string): number {
  const pattern = new RegExp(`\\b${escapeRegExp(phrase)}\\b`, "gi");
  let count = 0;
  for (const text of texts) {
    count += text.match(pattern)?.length ?? 0;
  }
  return count;
}

function inferCategory(term: string, context: string[]): GlossaryEntry["category"] {
  const haystack = `${term} ${context.join(" ")}`.toLowerCase();
  if (/\b(computer|internet|software|algorithm|data|digital|technology|machine|network|screen)\b/.test(haystack)) {
    return "technology";
  }
  if (/\b(learn|student|school|education|teach|class|knowledge|study|concept)\b/.test(haystack)) {
    return "education";
  }
  if (/\b(meet|meeting|agenda|team|discuss|conference)\b/.test(haystack)) {
    return "meeting";
  }
  if (/\b(government|law|rights|justice|constitution|public|minister|court|state)\b/.test(haystack)) {
    return "government";
  }
  return "general";
}

function unique(values: string[]): string[] {
  return [...new Set(values)];
}

function escapeRegExp(value: string): string {
  return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}
