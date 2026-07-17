import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";
import type { MotionClip } from "./motionAvatarRenderer";

type Point = { x: number; y: number };
type PlaybackClip = MotionClip & {
  datasetId: string;
  expertReviewed: boolean;
  sourceSampleId: string;
};
type MotionLibrary = { clipCount: number; clips: PlaybackClip[] };
type MotionCatalog = {
  clipCount: number;
  clips: Array<{ id: string; playable?: boolean }>;
};
const MOTION_LIBRARY_TEST_TIMEOUT_MS = 15_000;

const motionLibrary = readJson<MotionLibrary>("../assets/motion/isl-motion-library.json");
const motionCatalog = readJson<MotionCatalog>(
  "../../../../packages/avatar-engine/src/motionCatalog.generated.json"
);

describe("bundled ISL motion library", () => {
  it(
    "reports only native INCLUDE clips that are actually bundled and validates every frame",
    () => {
      const clips = motionLibrary.clips;

      expect(clips).toHaveLength(262);
      expect(motionLibrary.clipCount).toBe(clips.length);
      expect(motionCatalog.clipCount).toBe(clips.length);
      expect(motionCatalog.clips.map((clip) => clip.id)).toEqual(clips.map((clip) => clip.id));
      expect(
        motionCatalog.clips.every((clip) => !("playable" in clip) || clip.playable === true)
      ).toBe(true);
      expect(new Set(clips.map((clip) => clip.id)).size).toBe(clips.length);
      expect(new Set(clips.map((clip) => clip.sourceSampleId)).size).toBe(clips.length);

      for (const clip of clips) {
        expect(clip.datasetId).toBe("include");
        expect(clip.expertReviewed).toBe(false);
        expect(clip.fps).toBeGreaterThanOrEqual(5);
        expect(clip.fps).toBeLessThanOrEqual(60);
        expect(clip.frames.length).toBeGreaterThanOrEqual(16);

        let invalidPoint: Point | undefined;
        for (const frame of clip.frames) {
          expect(frame.pose).toHaveLength(25);
          expect(frame.leftHand).toHaveLength(21);
          expect(frame.rightHand).toHaveLength(21);
          for (const point of [...frame.pose, ...frame.leftHand, ...frame.rightHand] as Point[]) {
            if (
              !Number.isFinite(point.x) ||
              !Number.isFinite(point.y) ||
              point.x < -0.5 ||
              point.x > 1.5 ||
              point.y < -0.5 ||
              point.y > 1.5
            ) {
              invalidPoint = point;
            }
          }
        }
        expect(invalidPoint, `${clip.id} contains an invalid coordinate`).toBeUndefined();

        const trackedPoints = clip.frames.flatMap((frame) => [
          frame.pose[0],
          frame.leftHand[8],
          frame.rightHand[8]
        ]) as Point[];
        const xRange =
          Math.max(...trackedPoints.map((point) => point.x)) -
          Math.min(...trackedPoints.map((point) => point.x));
        const yRange =
          Math.max(...trackedPoints.map((point) => point.y)) -
          Math.min(...trackedPoints.map((point) => point.y));
        expect(Math.max(xRange, yRange), `${clip.id} must not be a static pose`).toBeGreaterThan(
          0.005
        );
      }
    },
    MOTION_LIBRARY_TEST_TIMEOUT_MS
  );
});

function readJson<T>(relativePath: string): T {
  return JSON.parse(
    readFileSync(fileURLToPath(new URL(relativePath, import.meta.url)), "utf8")
  ) as T;
}
