import type { AvatarQueue, ISLGlossItem } from "@signsaarthi/shared";
import { gzipSync } from "node:zlib";
import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import { loadResearchMotionLibrary } from "./researchMotionLibrary.js";

const originalPath = process.env.SIGNSAARTHI_ISIGN_MOTION_LIBRARY_PATH;
const temporaryDirectories: string[] = [];

afterEach(() => {
  if (originalPath === undefined) {
    delete process.env.SIGNSAARTHI_ISIGN_MOTION_LIBRARY_PATH;
  } else {
    process.env.SIGNSAARTHI_ISIGN_MOTION_LIBRARY_PATH = originalPath;
  }
  for (const directory of temporaryDirectories.splice(0)) {
    rmSync(directory, { recursive: true, force: true });
  }
});

describe("local iSign research motion library", () => {
  it("loads a gated clip and enriches only an unresolved word action", () => {
    const path = writeFixture();
    process.env.SIGNSAARTHI_ISIGN_MOTION_LIBRARY_PATH = path;
    const library = loadResearchMotionLibrary();
    expect(library?.clipCount).toBe(1);

    const glossItem = createGlossItem();
    const queue = createQueue(glossItem);
    const enriched = library?.enrichQueue(queue, [glossItem]);

    expect(enriched?.steps[0]).toMatchObject({
      kind: "sign",
      motionClipId: "isign-research-test",
      motionSource: "isign_research",
      expertReviewed: false
    });
    expect(enriched?.motionClips?.[0]?.frames).toHaveLength(32);
    expect(enriched?.motionReviewNotice).toContain("noncommercial research data");
  });

  it("does not replace a bundled motion clip", () => {
    const path = writeFixture();
    process.env.SIGNSAARTHI_ISIGN_MOTION_LIBRARY_PATH = path;
    const library = loadResearchMotionLibrary();
    const glossItem = createGlossItem();
    const queue = createQueue(glossItem);
    queue.steps[0] = {
      ...queue.steps[0]!,
      kind: "sign",
      motionClipId: "include-existing",
      motionSource: "include_dataset"
    };

    const enriched = library?.enrichQueue(queue, [glossItem]);
    expect(enriched?.steps[0]?.motionClipId).toBe("include-existing");
    expect(enriched?.motionClips).toBeUndefined();
  });

  it("rejects a clip whose frames are static despite claimed quality metadata", () => {
    const path = writeFixture(true);
    process.env.SIGNSAARTHI_ISIGN_MOTION_LIBRARY_PATH = path;

    expect(loadResearchMotionLibrary()).toBeUndefined();
  });
});

function writeFixture(staticFrames = false): string {
  const directory = mkdtempSync(join(tmpdir(), "signsaarthi-isign-motion-"));
  temporaryDirectories.push(directory);
  const path = join(directory, "library.json.gz");
  const frames = Array.from({ length: 32 }, (_, frameIndex) => {
    const pose = Array.from({ length: 33 }, () => ({ x: 0.5, y: 0.5, visibility: 1 }));
    pose[11] = { x: 0.35, y: 0.4, visibility: 1 };
    pose[12] = { x: 0.65, y: 0.4, visibility: 1 };
    const handX = 0.3 + (staticFrames ? 0 : frameIndex * 0.008);
    return {
      pose,
      leftHand: Array.from({ length: 21 }, () => ({ x: handX, y: 0.5, visibility: 1 })),
      rightHand: Array.from({ length: 21 }, () => ({ x: 1 - handX, y: 0.5, visibility: 1 })),
      face: Array.from({ length: 18 }, () => ({ x: 0.5, y: 0.25, visibility: 1 }))
    };
  });
  const payload = {
    schemaVersion: 1,
    source: {
      dataset: "Exploration-Lab/iSign",
      revision: "e4ee6c5f0d9dfcbc74205e3f1388ce94da26c298",
      license: "research only",
      gated: true,
      redistributionAllowed: false,
      rawArchiveStored: false,
      archiveDirectorySha256: "a".repeat(64),
      partFingerprints: ["a", "b", "c", "d"].map((suffix) => ({
        name: `iSign-poses_v1.1_part_a${suffix}`,
        size: 1,
        etag: "fixture"
      }))
    },
    usage: "Local test fixture",
    expertReviewStatus: "pending",
    clipCount: 1,
    candidateCount: 1,
    failureCount: 0,
    failuresByType: {},
    bytesTransferred: 1,
    clips: [
      {
        id: "isign-research-test",
        label: "Researchword",
        normalizedLabel: "researchword",
        fps: 20,
        durationMs: 2000,
        expertReviewed: false,
        sourceUidHash: "a".repeat(24),
        quality: {
          score: 0.9,
          poseVisibility: 0.9,
          handVisibility: 0.9,
          handMotionRange: 0.2,
          handMotionEnergy: 0.02,
          dynamicFrameRatio: 1
        },
        frames
      }
    ]
  };
  writeFileSync(path, gzipSync(JSON.stringify(payload)));
  return path;
}

function createGlossItem(): ISLGlossItem {
  return {
    id: "gloss-1",
    token: "Researchword",
    gloss: "RESEARCHWORD",
    fallback: "fingerspell",
    isVerified: false,
    confidence: "medium"
  };
}

function createQueue(item: ISLGlossItem): AvatarQueue {
  return {
    id: "queue-1",
    interpretationId: "interpretation-1",
    speed: 1,
    steps: [
      {
        id: "step-1",
        glossItemId: item.id,
        kind: "fingerspell",
        label: item.gloss,
        durationMs: 1500,
        confidence: item.confidence,
        fingerspellingSource: "islrtc_official"
      }
    ],
    honestFallbackText: "Unsupported words use fingerspelling.",
    createdAt: "2026-07-17T00:00:00.000Z"
  };
}
