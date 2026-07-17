import { afterEach, describe, expect, it, vi } from "vitest";
import {
  requestPageCaptionSnapshot,
  requestPageCaptionSnapshotAttempt,
  sendRuntimeMessage
} from "./runtime";

describe("extension runtime helpers", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("reports a negative service-worker acknowledgement as a delivery failure", async () => {
    vi.stubGlobal("chrome", {
      runtime: {
        lastError: undefined,
        sendMessage: vi.fn((_message: unknown, callback: (response: unknown) => void) => {
          callback({ ok: false, error: "Receiving end does not exist." });
        })
      }
    });

    await expect(
      sendRuntimeMessage({ type: "SESSION_STARTED", sessionId: "session_1", targetTabId: 42 })
    ).resolves.toBe(false);
  });

  it("classifies a missing receiver as transient so caption polling can recover", async () => {
    vi.stubGlobal("chrome", {
      runtime: {
        lastError: undefined,
        sendMessage: vi.fn((_message: unknown, callback: (response: unknown) => void) => {
          callback({ ok: false, error: "Receiving end does not exist." });
        })
      }
    });

    await expect(requestPageCaptionSnapshotAttempt(42)).resolves.toEqual({
      ok: false,
      reason: "transient",
      error: "Receiving end does not exist."
    });
  });

  it("keeps an explicit unavailable-page response distinct from transport failure", async () => {
    vi.stubGlobal("chrome", {
      runtime: {
        lastError: undefined,
        sendMessage: vi.fn((_message: unknown, callback: (response: unknown) => void) => {
          callback({ ok: false, error: "Open a YouTube tab to read visible captions." });
        })
      }
    });

    await expect(requestPageCaptionSnapshotAttempt()).resolves.toEqual({
      ok: false,
      reason: "unavailable",
      error: "Open a YouTube tab to read visible captions."
    });
  });

  it("passes the bound tab through caption snapshot requests and responses", async () => {
    const sendMessage = vi.fn(
      (message: { type: string; targetTabId?: number }, callback: (response: unknown) => void) => {
        callback({
          ok: true,
          targetTabId: message.targetTabId,
          snapshot: {
            title: "Lecture",
            url: "https://www.youtube.com/watch?v=demo",
            captionText: "Visible caption",
            supportStatus: "supported"
          }
        });
      }
    );
    vi.stubGlobal("chrome", { runtime: { lastError: undefined, sendMessage } });

    await expect(requestPageCaptionSnapshot(73)).resolves.toMatchObject({
      captionText: "Visible caption",
      targetTabId: 73
    });
    expect(sendMessage).toHaveBeenCalledWith(
      { type: "GET_PAGE_CAPTION_SNAPSHOT", targetTabId: 73 },
      expect.any(Function)
    );
  });

  it("validates and forwards playback speed commands", async () => {
    const sendMessage = vi.fn((_message: unknown, callback: (response: unknown) => void) =>
      callback({ ok: true })
    );
    vi.stubGlobal("chrome", { runtime: { lastError: undefined, sendMessage } });

    await expect(
      sendRuntimeMessage({
        type: "AVATAR_PLAYBACK_UPDATED",
        action: "set_speed",
        speed: 1.5,
        sessionId: "session_1",
        targetTabId: 42
      })
    ).resolves.toBe(true);
    expect(sendMessage).toHaveBeenCalledWith(
      expect.objectContaining({ action: "set_speed", speed: 1.5, targetTabId: 42 }),
      expect.any(Function)
    );
  });
});
