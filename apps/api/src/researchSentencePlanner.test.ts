import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import { loadResearchSentencePlanner } from "./researchSentencePlanner";

const temporaryDirectories: string[] = [];

afterEach(() => {
  for (const directory of temporaryDirectories.splice(0)) {
    rmSync(directory, { recursive: true, force: true });
  }
});

describe("private iSign caption boundary planner", () => {
  it("loads a privacy-safe artifact and preserves all caption words", () => {
    const path = writeFixture();
    const planner = loadResearchSentencePlanner(path);
    const text = "one two three four five six natural break nine ten eleven twelve thirteen fourteen";

    const chunks = planner?.splitMeaningChunks(text) ?? [];

    expect(planner?.metadata).toMatchObject({
      status: "ready",
      engine: "caption_boundary_planner",
      trainingDataset: { recordCount: 119_960 }
    });
    expect(chunks.join(" ")).toBe(text);
    expect(chunks.every((chunk) => chunk.split(/\s+/).length <= 12)).toBe(true);
    expect(chunks[0]).toBe("one two three four five six natural break");
  });

  it("rejects an artifact that claims to model ISL grammar", () => {
    const path = writeFixture({ isIslGrammarModel: true });
    expect(loadResearchSentencePlanner(path)).toBeUndefined();
  });
});

function writeFixture(claimOverrides: Record<string, boolean> = {}): string {
  const directory = mkdtempSync(join(tmpdir(), "signsaarthi-caption-planner-"));
  temporaryDirectories.push(directory);
  const path = join(directory, "planner.json");
  writeFileSync(
    path,
    JSON.stringify({
      schemaVersion: 1,
      artifactId: "isign-sentence-planner.private",
      visibility: "private",
      claims: {
        isIslGrammarModel: false,
        isSignLanguageTranslationModel: false,
        isText2PoseModel: false,
        isPoseOrMotionModel: false,
        ...claimOverrides
      },
      distribution: {
        publicHuggingFaceUpload: false,
        containsSourceMedia: false,
        containsPoseOrLandmarkData: false
      },
      privacy: {
        rawSentencesIncluded: false,
        sourceIdentifiersIncluded: false,
        videoIdentifiersIncluded: false
      },
      training: {
        retainedUniqueCaptionRows: 119_960,
        leakageChecks: {
          videoGroupOverlapAcrossSplits: 0,
          exactNormalizedCaptionOverlapAcrossRetainedSplits: 0,
          status: "passed"
        }
      },
      evaluation: {
        heldOut: {
          validation: { lexicalTokenCoverage: 0.88, exportedBoundaryFeatureCoverage: 0.33 },
          test: { lexicalTokenCoverage: 0.887, exportedBoundaryFeatureCoverage: 0.332 }
        }
      },
      chunkBoundaryStatistics: {
        phrases: [
          {
            tokens: ["natural", "break"],
            boundaryAfterProbability: 0.95,
            support: 100,
            videoSupport: 80
          }
        ],
        tokens: []
      }
    })
  );
  return path;
}
