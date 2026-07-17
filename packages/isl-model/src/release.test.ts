import { existsSync, readFileSync, readdirSync } from "node:fs";
import { resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";
import { z } from "zod";
import { islLexiconModelSchema, lexicalMatcherArtifactSchema } from "./index";

const RELEASE_LEXICON_ENTRY_COUNT = 12_434;
const RELEASE_INCLUDE_LABEL_COUNT = 263;
const RELEASE_INCLUDE_MOTION_COUNT = 262;
const QUARANTINED_INCLUDE_LABEL = "second number";
const workspaceRoot = fileURLToPath(new URL("../../../", import.meta.url));

const includeTrainRowSchema = z.object({
  label: z.string().min(1)
});

const motionClipSchema = z.object({
  id: z.string().min(1),
  normalizedLabel: z.string().min(1),
  datasetId: z.literal("include"),
  expertReviewed: z.literal(false),
  sourceVideoPath: z.string().min(1)
});

const motionCatalogSchema = z.object({
  schemaVersion: z.literal(1),
  clipCount: z.number().int().nonnegative(),
  timingAudit: z.object({
    clipCount: z.number().int().nonnegative(),
    status: z.literal("passed")
  }),
  clips: z.array(
    motionClipSchema.extend({
      frameCount: z.number().int().min(2)
    })
  )
});

const playbackLibrarySchema = z.object({
  schemaVersion: z.literal(1),
  source: z.string().includes("INCLUDE"),
  expertReviewStatus: z.literal("not_expert_certified"),
  clipCount: z.number().int().nonnegative(),
  timingAudit: z.object({
    clipCount: z.number().int().nonnegative(),
    status: z.literal("passed")
  }),
  clips: z.array(
    motionClipSchema.extend({
      frames: z.array(z.unknown()).min(2)
    })
  )
});

describe("release ML artifacts", () => {
  it("ships the deterministic 12,434-entry lexicon", () => {
    const model = islLexiconModelSchema.parse(readJson("data/models/isl-lexicon-model.json"));

    expect(model.metadata.engine).toBe("lexicon_ranker");
    expect(model.metadata.trainingDataset.recordCount).toBe(RELEASE_LEXICON_ENTRY_COUNT);
    expect(model.metadata.trainingDataset.classCount).toBe(RELEASE_LEXICON_ENTRY_COUNT);
    expect(model.glossaryEntries).toHaveLength(RELEASE_LEXICON_ENTRY_COUNT);
    expect(Object.keys(model.tokenWeights)).toHaveLength(RELEASE_LEXICON_ENTRY_COUNT);
    expect(new Set(model.glossaryEntries.map((entry) => entry.id)).size).toBe(
      RELEASE_LEXICON_ENTRY_COUNT
    );
    expect(model.glossaryEntries.every((entry) => entry.signAssetId === undefined)).toBe(true);
  });

  it("ships only a qualified source-preserving lexical matcher", () => {
    const matcher = lexicalMatcherArtifactSchema.parse(
      readJson("data/models/isl-text-matcher.json")
    );

    expect(matcher.metadata.status).toBe("ready");
    expect(matcher.metrics.acceptance.ready).toBe(true);
    expect(Object.values(matcher.metrics.acceptance.checks).every(Boolean)).toBe(true);
    expect(matcher.constraints).toEqual({
      singleTokenOnly: true,
      preserveSourceText: true,
      noGrammarClaims: true,
      noMotionSynthesis: true
    });
  });

  it("ships every source-consistent INCLUDE motion and quarantines the conflicting label", () => {
    const catalog = motionCatalogSchema.parse(readJson("data/models/isl-motion-catalog.json"));
    const library = playbackLibrarySchema.parse(
      readJson("apps/extension/src/assets/motion/isl-motion-library.json")
    );
    const catalogIds = catalog.clips.map((clip) => clip.id);
    const libraryIds = library.clips.map((clip) => clip.id);
    const normalizedTrainLabels = new Set(
      readJsonLines("data/isl/include-metadata/train.jsonl").map((row) =>
        normalizeIncludeLabel(includeTrainRowSchema.parse(row).label)
      )
    );

    expect(normalizedTrainLabels.size).toBe(RELEASE_INCLUDE_LABEL_COUNT);
    expect(catalog.clipCount).toBe(RELEASE_INCLUDE_MOTION_COUNT);
    expect(catalog.timingAudit.clipCount).toBe(RELEASE_INCLUDE_MOTION_COUNT);
    expect(catalog.clips).toHaveLength(RELEASE_INCLUDE_MOTION_COUNT);
    expect(new Set(catalogIds).size).toBe(RELEASE_INCLUDE_MOTION_COUNT);
    expect(new Set(catalog.clips.map((clip) => clip.normalizedLabel)).size).toBe(
      RELEASE_INCLUDE_MOTION_COUNT
    );
    const playableLabels = new Set(catalog.clips.map((clip) => clip.normalizedLabel));
    expect([...normalizedTrainLabels].filter((label) => !playableLabels.has(label))).toEqual([
      QUARANTINED_INCLUDE_LABEL
    ]);
    for (const clip of catalog.clips) {
      const sourceDirectory = clip.sourceVideoPath.split("/").at(-2);
      expect(sourceDirectory, clip.id).toBeDefined();
      expect(normalizeIncludeLabel(sourceDirectory!), clip.id).toBe(clip.normalizedLabel);
    }
    expect(library.clipCount).toBe(RELEASE_INCLUDE_MOTION_COUNT);
    expect(library.timingAudit.clipCount).toBe(RELEASE_INCLUDE_MOTION_COUNT);
    expect(library.clips).toHaveLength(RELEASE_INCLUDE_MOTION_COUNT);
    expect(libraryIds).toEqual(catalogIds);
  });

  it("does not claim or ship a trained recognition model", () => {
    expect(existsSync(workspacePath("data/models/isl-keypoint-model.json"))).toBe(false);
    expect(existsSync(workspacePath("data/models/isl-temporal-model.json"))).toBe(false);
    expect(readdirSync(workspacePath("data/models")).some((name) => name.endsWith(".onnx"))).toBe(
      false
    );
  });
});

function readJson(relativePath: string): unknown {
  return JSON.parse(readFileSync(workspacePath(relativePath), "utf8"));
}

function readJsonLines(relativePath: string): unknown[] {
  return readFileSync(workspacePath(relativePath), "utf8")
    .split(/\r?\n/u)
    .filter((line) => line.trim().length > 0)
    .map((line): unknown => JSON.parse(line));
}

function normalizeIncludeLabel(value: string): string {
  return value
    .replace(/^\s*\d+\s*[.)-]\s*/u, "")
    .replaceAll("&", " and ")
    .replace(/[^a-zA-Z0-9]+/gu, " ")
    .toLowerCase()
    .trim()
    .replace(/\s+/gu, " ");
}

function workspacePath(relativePath: string): string {
  return resolve(workspaceRoot, relativePath);
}
