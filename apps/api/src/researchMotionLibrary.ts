import {
  motionClipSchema,
  type AvatarQueue,
  type ISLGlossItem,
  type ModelMetadata,
  type MotionClip
} from "@signsaarthi/shared";
import { existsSync, readFileSync } from "node:fs";
import { resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { gunzipSync } from "node:zlib";
import { z } from "zod";

const MAX_COMPRESSED_BYTES = 64 * 1024 * 1024;
const MAX_DECOMPRESSED_BYTES = 256 * 1024 * 1024;
const ISIGN_REVISION = "e4ee6c5f0d9dfcbc74205e3f1388ce94da26c298";
const ISIGN_PARTS = [
  "iSign-poses_v1.1_part_aa",
  "iSign-poses_v1.1_part_ab",
  "iSign-poses_v1.1_part_ac",
  "iSign-poses_v1.1_part_ad"
] as const;

const researchClipSchema = motionClipSchema.extend({
  normalizedLabel: z.string().min(1),
  durationMs: z.number().finite().min(1600).max(3200),
  expertReviewed: z.literal(false),
  sourceUidHash: z.string().regex(/^[a-f0-9]{24}$/),
  quality: z
    .object({
      score: z.number().finite().min(0).max(1),
      poseVisibility: z.number().finite().min(0).max(1),
      handVisibility: z.number().finite().min(0).max(1),
      handMotionRange: z.number().finite().nonnegative(),
      handMotionEnergy: z.number().finite().min(0.01),
      dynamicFrameRatio: z.number().finite().min(0.2).max(1)
    })
    .strict()
});

const researchLibrarySchema = z.object({
  schemaVersion: z.literal(1),
  source: z.object({
    dataset: z.literal("Exploration-Lab/iSign"),
    revision: z.literal(ISIGN_REVISION),
    license: z.string().min(1),
    gated: z.literal(true),
    redistributionAllowed: z.literal(false),
    rawArchiveStored: z.literal(false),
    archiveDirectorySha256: z.string().regex(/^[a-f0-9]{64}$/),
    partFingerprints: z.array(
      z.object({
        name: z.string().min(1),
        size: z.number().int().positive(),
        etag: z.string()
      })
    ).length(4)
  }),
  usage: z.string().min(1),
  expertReviewStatus: z.literal("pending"),
  clipCount: z.number().int().nonnegative(),
  candidateCount: z.number().int().nonnegative(),
  failureCount: z.number().int().nonnegative(),
  failuresByType: z.record(z.number().int().nonnegative()),
  bytesTransferred: z.number().int().nonnegative(),
  clips: z.array(researchClipSchema)
});

type ResearchClip = z.infer<typeof researchClipSchema>;

export type ResearchMotionLibrary = {
  metadata: ModelMetadata;
  clipCount: number;
  enrichQueue(queue: AvatarQueue, glossSequence: readonly ISLGlossItem[]): AvatarQueue;
};

export function loadResearchMotionLibrary(): ResearchMotionLibrary | undefined {
  const configuredPath = process.env["SIGNSAARTHI_ISIGN_MOTION_LIBRARY_PATH"];
  const candidatePaths = configuredPath
    ? [resolve(process.cwd(), configuredPath)]
    : [
        resolve(process.cwd(), "data/private/isign-research/word-motion-library.json.gz"),
        fileURLToPath(
          new URL("../../../data/private/isign-research/word-motion-library.json.gz", import.meta.url)
        )
      ];
  const libraryPath = candidatePaths.find(existsSync);
  if (!libraryPath) {
    return undefined;
  }

  try {
    const compressed = readFileSync(libraryPath);
    if (compressed.byteLength > MAX_COMPRESSED_BYTES) {
      return undefined;
    }
    const decoded = gunzipSync(compressed, { maxOutputLength: MAX_DECOMPRESSED_BYTES });
    const payload = researchLibrarySchema.parse(JSON.parse(decoded.toString("utf8")));
    if (payload.clipCount !== payload.clips.length) {
      return undefined;
    }
    if (
      payload.source.partFingerprints.map((part) => part.name).join("\0") !==
        ISIGN_PARTS.join("\0") ||
      payload.clips.some((clip) => !hasDynamicMotion(clip))
    ) {
      return undefined;
    }
    return createResearchMotionLibrary(payload);
  } catch {
    return undefined;
  }
}

function createResearchMotionLibrary(
  payload: z.infer<typeof researchLibrarySchema>
): ResearchMotionLibrary {
  const byLabel = new Map(payload.clips.map((clip) => [clip.normalizedLabel, clip]));
  const metadata: ModelMetadata = {
    id: "signsaarthi-isign-research-motion-v1",
    version: "1.0.0",
    status: "ready",
    engine: "motion_library",
    trainingDataset: {
      primaryDataset: "isign",
      displayName: "Local gated iSign word motion",
      recordCount: payload.candidateCount,
      classCount: payload.clipCount,
      citationUrls: [
        "https://huggingface.co/datasets/Exploration-Lab/iSign",
        "https://aclanthology.org/2024.findings-acl.643/"
      ]
    },
    notes: [
      `${payload.clipCount} isolated-word clips are available only in this local research runtime.`,
      "Gated CC BY-NC-SA 4.0 noncommercial and no-redistribution terms apply.",
      "Automated pose transfer is not expert-reviewed and does not establish continuous ISL grammar."
    ]
  };

  return {
    metadata,
    clipCount: payload.clipCount,
    enrichQueue(queue, glossSequence) {
      const glossById = new Map(glossSequence.map((item) => [item.id, item]));
      const usedClips = new Map<string, MotionClip>();
      const steps = queue.steps.map((step) => {
        if (step.motionClipId) {
          return step;
        }
        const item = glossById.get(step.glossItemId);
        if (!item) {
          return step;
        }
        const clip = findResearchClip(byLabel, item.token, item.gloss);
        if (!clip) {
          return step;
        }
        usedClips.set(clip.id, toRuntimeClip(clip));
        return {
          ...step,
          kind: "sign" as const,
          signAssetId: clip.id,
          motionClipId: clip.id,
          motionSource: "isign_research" as const,
          expertReviewed: false,
          durationMs: Math.max(1, Math.round(clip.durationMs / queue.speed)),
          resolvedAction: { kind: "sign" as const, label: item.gloss }
        };
      });
      if (!usedClips.size) {
        return queue;
      }
      const hasFallback = steps.some((step) => step.kind !== "sign");
      return {
        ...queue,
        steps,
        motionClips: [...usedClips.values()],
        ...(hasFallback && queue.honestFallbackText
          ? { honestFallbackText: queue.honestFallbackText }
          : {}),
        motionReviewNotice:
          "Local gated iSign isolated-word motion is active; it is noncommercial research data, automated, and not expert-reviewed."
      };
    }
  };
}

function findResearchClip(
  byLabel: ReadonlyMap<string, ResearchClip>,
  ...candidates: string[]
): ResearchClip | undefined {
  for (const candidate of candidates) {
    const clip = byLabel.get(normalizeLabel(candidate));
    if (clip) {
      return clip;
    }
  }
  return undefined;
}

function normalizeLabel(value: string): string {
  return value
    .normalize("NFKD")
    .replace(/\p{M}/gu, "")
    .replace(/[^a-zA-Z0-9]+/g, " ")
    .trim()
    .toLowerCase();
}

function hasDynamicMotion(clip: ResearchClip): boolean {
  if (clip.frames.length !== 32) {
    return false;
  }
  const frameEnergy: number[] = [];
  let velocitySum = 0;
  let velocityCount = 0;
  const shoulderWidths = clip.frames
    .map((frame) => {
      if (
        frame.pose.length !== 33 ||
        frame.leftHand.length !== 21 ||
        frame.rightHand.length !== 21 ||
        (frame.face?.length ?? 0) !== 18
      ) {
        return 0;
      }
      const left = frame.pose[11]!;
      const right = frame.pose[12]!;
      return Math.hypot(left.x - right.x, left.y - right.y);
    })
    .filter((width) => Number.isFinite(width) && width > 0.0001)
    .sort((first, second) => first - second);
  const shoulderScale = shoulderWidths[Math.floor(shoulderWidths.length / 2)] ?? 0;
  if (shoulderScale <= 0) {
    return false;
  }
  for (let frameIndex = 1; frameIndex < clip.frames.length; frameIndex += 1) {
    const previous = clip.frames[frameIndex - 1]!;
    const current = clip.frames[frameIndex]!;
    const previousHands = [...previous.leftHand, ...previous.rightHand];
    const currentHands = [...current.leftHand, ...current.rightHand];
    let currentSum = 0;
    let currentCount = 0;
    for (let pointIndex = 0; pointIndex < currentHands.length; pointIndex += 1) {
      const left = previousHands[pointIndex]!;
      const right = currentHands[pointIndex]!;
      if (Math.min(left.visibility ?? 1, right.visibility ?? 1) < 0.08) {
        continue;
      }
      const velocity = Math.hypot(right.x - left.x, right.y - left.y) / shoulderScale;
      currentSum += velocity;
      currentCount += 1;
      velocitySum += velocity;
      velocityCount += 1;
    }
    frameEnergy.push(currentSum / Math.max(1, currentCount));
  }
  const energy = velocitySum / Math.max(1, velocityCount);
  const dynamicRatio = frameEnergy.filter((value) => value >= 0.002).length / frameEnergy.length;
  return energy >= 0.01 && dynamicRatio >= 0.2;
}

function toRuntimeClip(clip: ResearchClip): MotionClip {
  return {
    id: clip.id,
    label: clip.label,
    fps: clip.fps,
    frames: clip.frames
  };
}
