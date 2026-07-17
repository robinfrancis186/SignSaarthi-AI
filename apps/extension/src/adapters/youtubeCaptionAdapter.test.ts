import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { getRollingCaptionDelta, readYouTubeCaptionSnapshot } from "./youtubeCaptionAdapter";

describe("readYouTubeCaptionSnapshot", () => {
  beforeEach(() => {
    document.documentElement.innerHTML = "<head><title>Video page</title></head><body></body>";
  });

  afterEach(() => {
    document.body.innerHTML = "";
  });

  it("uses visible caption segments without duplicating their caption-window text", () => {
    document.body.innerHTML = `
      <div class="caption-window">
        <span class="ytp-caption-segment">Welcome to</span>
        <span class="ytp-caption-segment">SignSaarthi.</span>
      </div>
    `;

    expect(readYouTubeCaptionSnapshot().captionText).toBe("Welcome to SignSaarthi.");
  });

  it("ignores caption segments hidden directly or by an ancestor", () => {
    document.body.innerHTML = `
      <div class="caption-window" style="display: none">
        <span class="ytp-caption-segment">Old display-hidden caption.</span>
      </div>
      <div class="caption-window" aria-hidden="true">
        <span class="ytp-caption-segment">Old aria-hidden caption.</span>
      </div>
      <div class="caption-window" style="opacity: 0">
        <span class="ytp-caption-segment">Old transparent caption.</span>
      </div>
      <div class="caption-window">
        <span class="ytp-caption-segment">Current visible caption.</span>
      </div>
    `;

    expect(readYouTubeCaptionSnapshot().captionText).toBe("Current visible caption.");
  });

  it("falls back to visible caption-window text when segment nodes are unavailable", () => {
    document.body.innerHTML = `
      <div class="caption-window" hidden>Stale legacy caption.</div>
      <div class="caption-window">Legacy caption without segment markup.</div>
    `;

    expect(readYouTubeCaptionSnapshot().captionText).toBe("Legacy caption without segment markup.");
  });

  it("falls back to a caption window when the only segment text is hidden", () => {
    document.body.innerHTML = `
      <span class="ytp-caption-segment" style="visibility: hidden">Stale caption.</span>
      <div class="caption-window">Fallback caption.</div>
    `;

    expect(readYouTubeCaptionSnapshot().captionText).toBe("Fallback caption.");
  });
});

describe("getRollingCaptionDelta", () => {
  it("returns only newly appended source words from a growing caption snapshot", () => {
    expect(getRollingCaptionDelta("Welcome to", "Welcome to SignSaarthi")).toBe("SignSaarthi");
  });

  it("keeps repeated filler words that were newly appended", () => {
    expect(getRollingCaptionDelta("um um", "um um um, welcome")).toBe("um, welcome");
  });

  it("removes a sliding caption prefix already present at the previous suffix", () => {
    expect(
      getRollingCaptionDelta(
        "Accessible lessons use sign language",
        "use sign language for every student"
      )
    ).toBe("for every student");
  });

  it("preserves a repeated word that is genuinely new after a sliding overlap", () => {
    expect(getRollingCaptionDelta("Students learn learn", "learn learn learn together")).toBe(
      "learn together"
    );
  });

  it("does not resubmit a sliding caption made entirely of old suffix words", () => {
    expect(getRollingCaptionDelta("Today we use sign language", "use sign language")).toBe("");
  });

  it("does not drop words from a genuinely different caption", () => {
    expect(getRollingCaptionDelta("Thank you", "You are welcome")).toBe("You are welcome");
  });

  it("ignores whitespace-only repeats and contractions of the same rolling prefix", () => {
    expect(getRollingCaptionDelta("Welcome   to", " Welcome to ")).toBe("");
    expect(getRollingCaptionDelta("Welcome to SignSaarthi", "Welcome to")).toBe("");
  });
});
