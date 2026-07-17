import { afterEach, describe, expect, it, vi } from "vitest";
import { loadMotionClipMap, resetMotionClipMapForTests } from "./motionLibraryLoader";

const validClip = {
  id: "include-test",
  label: "TEST",
  fps: 25,
  frames: Array.from({ length: 32 }, () => ({ pose: [], leftHand: [], rightHand: [] }))
};

afterEach(() => {
  resetMotionClipMapForTests();
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
});

describe("motion library loader", () => {
  it("loads, validates, and caches the emitted extension asset", async () => {
    const getUrl = vi.fn(() => "chrome-extension://id/motion.json");
    vi.stubGlobal("chrome", { runtime: { getURL: getUrl } });
    const fetchMock = vi.fn().mockResolvedValue(
      Response.json({ clipCount: 1, clips: [validClip] })
    );
    vi.stubGlobal("fetch", fetchMock);

    const first = await loadMotionClipMap();
    const second = await loadMotionClipMap();

    expect(first).toBe(second);
    expect(first.get("include-test")).toMatchObject({ id: "include-test", fps: 25 });
    expect(getUrl).toHaveBeenCalledTimes(1);
    expect(fetchMock).toHaveBeenCalledTimes(1);
  });

  it("rejects malformed libraries and permits a later retry", async () => {
    vi.stubGlobal("chrome", {
      runtime: { getURL: vi.fn(() => "chrome-extension://id/motion.json") }
    });
    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce(Response.json({ clipCount: 2, clips: [validClip] }))
      .mockResolvedValueOnce(Response.json({ clipCount: 1, clips: [validClip] }));
    vi.stubGlobal("fetch", fetchMock);

    await expect(loadMotionClipMap()).rejects.toThrow(/clip count/i);
    expect((await loadMotionClipMap()).size).toBe(1);
    expect(fetchMock).toHaveBeenCalledTimes(2);
  });
});
