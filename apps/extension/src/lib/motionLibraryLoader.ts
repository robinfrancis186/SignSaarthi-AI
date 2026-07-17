import motionLibraryAssetPath from "../assets/motion/isl-motion-library.json?url";
import type { MotionClip } from "../content/motionAvatarRenderer";

type MotionLibraryPayload = {
  clipCount: number;
  clips: MotionClip[];
};

let motionClipMapPromise: Promise<ReadonlyMap<string, MotionClip>> | undefined;

export function loadMotionClipMap(): Promise<ReadonlyMap<string, MotionClip>> {
  motionClipMapPromise ??= fetchMotionClipMap().catch((error: unknown) => {
    motionClipMapPromise = undefined;
    throw error;
  });
  return motionClipMapPromise;
}

export function resetMotionClipMapForTests(): void {
  motionClipMapPromise = undefined;
}

async function fetchMotionClipMap(): Promise<ReadonlyMap<string, MotionClip>> {
  const response = await fetch(chrome.runtime.getURL(motionLibraryAssetPath));
  if (!response.ok) {
    throw new Error(`Motion library request failed (${response.status}).`);
  }

  const payload = validateMotionLibrary(await response.json());
  return new Map(payload.clips.map((clip) => [clip.id, clip]));
}

function validateMotionLibrary(value: unknown): MotionLibraryPayload {
  if (!value || typeof value !== "object") {
    throw new Error("Motion library payload is not an object.");
  }
  const payload = value as { clipCount?: unknown; clips?: unknown };
  if (!Number.isInteger(payload.clipCount) || !Array.isArray(payload.clips)) {
    throw new Error("Motion library payload is missing its clip count or clips.");
  }
  if (payload.clipCount !== payload.clips.length) {
    throw new Error("Motion library clip count does not match its payload.");
  }

  const seenIds = new Set<string>();
  for (const [index, valueClip] of payload.clips.entries()) {
    if (!valueClip || typeof valueClip !== "object") {
      throw new Error(`Motion library clip ${index} is invalid.`);
    }
    const clip = valueClip as Partial<MotionClip>;
    if (
      typeof clip.id !== "string" ||
      !clip.id ||
      seenIds.has(clip.id) ||
      typeof clip.label !== "string" ||
      !clip.label ||
      !Number.isFinite(clip.fps) ||
      !Array.isArray(clip.frames) ||
      clip.frames.length !== 32
    ) {
      throw new Error(`Motion library clip ${index} failed runtime validation.`);
    }
    seenIds.add(clip.id);
  }

  return payload as MotionLibraryPayload;
}
