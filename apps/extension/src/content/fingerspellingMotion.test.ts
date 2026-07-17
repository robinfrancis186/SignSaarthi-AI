import { describe, expect, it } from "vitest";
import alphabetLibrary from "../assets/motion/islrtc-alphabet-library.json";
import { alphabetMotionClips, createFingerspellingMotion } from "./fingerspellingMotion";

describe("official ISLRTC fingerspelling motion", () => {
  it("contains exactly one dynamic clip for every A-Z letter", () => {
    expect([...alphabetMotionClips.keys()]).toEqual([..."ABCDEFGHIJKLMNOPQRSTUVWXYZ"]);
    expect(alphabetLibrary.clips).toHaveLength(26);
    for (const clip of alphabetLibrary.clips) {
      expect(clip.frames).toHaveLength(12);
      const coordinates = clip.frames.flatMap((frame) =>
        [...frame.pose, ...frame.leftHand, ...frame.rightHand].flatMap((point) => [point.x, point.y])
      );
      expect(coordinates.every(Number.isFinite)).toBe(true);
      expect(Math.max(...coordinates) - Math.min(...coordinates)).toBeGreaterThan(0.005);
    }
  });

  it("composes every letter with three transition frames", () => {
    const clip = createFingerspellingMotion("AI access");

    expect(clip).toBeDefined();
    expect(clip?.frames).toHaveLength(8 * 12 + 7 * 3);
    expect(clip?.fps).toBe(20);
    expect(clip?.id).toMatch(/^islrtc-fingerspell-aiaccess-/);
  });

  it("does not invent letter motion for unsupported scripts", () => {
    expect(createFingerspellingMotion("नमस्ते")).toBeUndefined();
  });
});
