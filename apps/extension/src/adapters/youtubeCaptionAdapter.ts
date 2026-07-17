export type YouTubeCaptionSnapshot = {
  title: string;
  url: string;
  captionText?: string;
  mediaTimeMs?: number;
  targetTabId?: number;
  supportStatus: "supported" | "captions_unavailable" | "unsupported";
};

type CaptionWord = {
  start: number;
  value: string;
};

const CAPTION_WORD_PATTERN = /[\p{L}\p{M}\p{N}]+(?:['’-][\p{L}\p{M}\p{N}]+)*/gu;
const MIN_SLIDING_OVERLAP_WORDS = 2;

export function readYouTubeCaptionSnapshot(
  documentRef: Document = document
): YouTubeCaptionSnapshot {
  const title = documentRef
    .querySelector<HTMLHeadingElement>("h1 yt-formatted-string")
    ?.textContent?.trim();
  const captionText =
    readVisibleCaptionText(documentRef, ".ytp-caption-segment") ||
    readVisibleCaptionText(documentRef, ".caption-window");
  const pageLocation = documentRef.defaultView?.location ?? location;
  const mediaTimeMs = Math.max(
    0,
    Math.round((documentRef.querySelector<HTMLVideoElement>("video")?.currentTime ?? 0) * 1000)
  );

  const snapshot: YouTubeCaptionSnapshot = {
    title: title || documentRef.title || "YouTube video",
    url: pageLocation.href,
    mediaTimeMs,
    supportStatus: pageLocation.hostname.includes("youtube.com")
      ? captionText
        ? "supported"
        : "captions_unavailable"
      : "unsupported"
  };
  if (captionText) {
    snapshot.captionText = captionText;
  }
  return snapshot;
}

export function getRollingCaptionDelta(previousText: string, nextText: string): string {
  const previous = normalizeCaptionText(previousText);
  const next = normalizeCaptionText(nextText);
  if (!next || previous === next) {
    return "";
  }
  if (!previous) {
    return next;
  }

  const previousWords = getCaptionWords(previous);
  const nextWords = getCaptionWords(next);
  if (!previousWords.length || !nextWords.length) {
    return next;
  }

  if (isWordPrefix(nextWords, previousWords)) {
    return "";
  }
  if (isWordPrefix(previousWords, nextWords)) {
    return next.slice(nextWords[previousWords.length]!.start).trim();
  }

  const overlapLength = getSlidingOverlapLength(previousWords, nextWords);
  if (overlapLength === 0) {
    return next;
  }
  if (overlapLength >= nextWords.length) {
    return "";
  }
  return next.slice(nextWords[overlapLength]!.start).trim();
}

function readVisibleCaptionText(documentRef: Document, selector: string): string {
  return Array.from(documentRef.querySelectorAll<HTMLElement>(selector))
    .filter(isVisible)
    .map((element) => element.textContent?.trim())
    .filter((value): value is string => Boolean(value))
    .join(" ")
    .replace(/\s+/g, " ")
    .trim();
}

function normalizeCaptionText(value: string): string {
  return value.replace(/\s+/g, " ").trim();
}

function getCaptionWords(value: string): CaptionWord[] {
  return Array.from(value.matchAll(CAPTION_WORD_PATTERN), (match) => ({
    start: match.index,
    value: match[0].normalize("NFKC").toLocaleLowerCase()
  }));
}

function isWordPrefix(prefix: CaptionWord[], words: CaptionWord[]): boolean {
  return (
    prefix.length <= words.length &&
    prefix.every((word, index) => word.value === words[index]?.value)
  );
}

function getSlidingOverlapLength(previousWords: CaptionWord[], nextWords: CaptionWord[]): number {
  const maximumLength = Math.min(previousWords.length, nextWords.length);
  for (let length = maximumLength; length >= MIN_SLIDING_OVERLAP_WORDS; length -= 1) {
    const previousStart = previousWords.length - length;
    const matches = nextWords
      .slice(0, length)
      .every((word, index) => word.value === previousWords[previousStart + index]?.value);
    if (matches) {
      return length;
    }
  }
  return 0;
}

function isVisible(element: HTMLElement): boolean {
  const view = element.ownerDocument.defaultView;

  for (let current: HTMLElement | null = element; current; current = current.parentElement) {
    if (current.hidden || current.getAttribute("aria-hidden") === "true") {
      return false;
    }

    const style = view?.getComputedStyle(current) ?? current.style;
    if (
      style.display === "none" ||
      style.visibility === "hidden" ||
      style.visibility === "collapse" ||
      style.opacity === "0"
    ) {
      return false;
    }
  }

  return true;
}
