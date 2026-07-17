import motionCatalogData from "./motionCatalog.generated.json" with { type: "json" };

export type MotionCatalogEntry = {
  id: string;
  normalizedLabel: string;
  source: "include_dataset";
  expertReviewed: false;
  frameCount: number;
  fps: number;
  durationMs: number;
  playable: boolean;
};

export const motionCatalog: readonly MotionCatalogEntry[] = motionCatalogData.clips.map((clip) => ({
  id: clip.id,
  normalizedLabel: clip.normalizedLabel,
  source: "include_dataset",
  expertReviewed: false,
  frameCount: clip.frameCount,
  fps: clip.fps,
  durationMs: clip.durationMs,
  // Generated catalogs contain only clips whose frame payload passed extraction validation.
  playable: !("playable" in clip) || clip.playable === true
}));

const motionById = new Map(motionCatalog.map((entry) => [entry.id, entry]));
const motionCandidatesByLabel = new Map<string, MotionCatalogEntry[]>();
for (const entry of motionCatalog) {
  const label = motionCatalogData.clips.find((clip) => clip.id === entry.id)?.label ?? entry.normalizedLabel;
  const key = normalizeMotionLabel(label);
  motionCandidatesByLabel.set(key, [...(motionCandidatesByLabel.get(key) ?? []), entry]);
}

export function normalizeMotionLabel(value: string): string {
  return value
    .normalize("NFKD")
    .replace(/\p{M}/gu, "")
    .replace(/[^a-zA-Z0-9]+/g, " ")
    .trim()
    .toLowerCase();
}

export function findMotionClip(...candidates: string[]): MotionCatalogEntry | undefined {
  for (const candidate of candidates) {
    const entries = motionCandidatesByLabel.get(normalizeMotionLabel(candidate));
    if (entries?.length === 1) {
      return entries[0];
    }
  }
  return undefined;
}

export function findMotionClipById(id: string | undefined): MotionCatalogEntry | undefined {
  return id ? motionById.get(id) : undefined;
}
