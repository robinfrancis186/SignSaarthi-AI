import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

type MessageListener = (
  message: unknown,
  sender: chrome.runtime.MessageSender,
  sendResponse: (response?: unknown) => void
) => boolean | void;

type TabMessageCallback = (response?: unknown) => void;

const NO_RECEIVER_ERROR = "Could not establish connection. Receiving end does not exist.";
const SIDE_PANEL_URL = "chrome-extension://extension-id/src/sidepanel/index.html";
const YOUTUBE_URL = "https://www.youtube.com/watch?v=demo";
const UNSUPPORTED_RECOVERY_TARGET_ERROR =
  "Content-script recovery is available only on supported YouTube tabs.";

let listener: MessageListener;
let runtimeLastError: string | undefined;
let queryMock: ReturnType<typeof vi.fn>;
let getTabMock: ReturnType<typeof vi.fn>;
let sendMessageMock: ReturnType<typeof vi.fn>;
let executeScriptMock: ReturnType<typeof vi.fn>;
let setOptionsMock: ReturnType<typeof vi.fn>;

describe("background service worker", () => {
  beforeEach(async () => {
    vi.useFakeTimers();
    vi.resetModules();
    runtimeLastError = undefined;
    queryMock = vi.fn((_query, callback: (tabs: chrome.tabs.Tab[]) => void) =>
      callback([{ id: 42, active: true, url: YOUTUBE_URL } as chrome.tabs.Tab])
    );
    getTabMock = vi.fn((tabId: number, callback: (tab: chrome.tabs.Tab) => void) =>
      callback({ id: tabId, url: YOUTUBE_URL } as chrome.tabs.Tab)
    );
    sendMessageMock = vi.fn(
      (_tabId: number, message: { type?: string }, callback?: TabMessageCallback) => {
        completeTabMessage(
          callback,
          message.type === "GET_PAGE_CAPTION_SNAPSHOT"
            ? {
                ok: true,
                snapshot: {
                  title: "Lecture",
                  url: YOUTUBE_URL,
                  captionText: "Visible caption",
                  mediaTimeMs: 1200,
                  supportStatus: "supported"
                }
              }
            : { ok: true }
        );
      }
    );
    executeScriptMock = vi.fn(() => Promise.resolve([]));
    setOptionsMock = vi.fn(() => Promise.resolve());

    vi.stubGlobal("chrome", {
      runtime: {
        id: "extension-id",
        getURL: vi.fn((path: string) => `chrome-extension://extension-id/${path}`),
        getManifest: vi.fn(() => ({
          content_scripts: [
            {
              matches: ["https://www.youtube.com/*"],
              js: ["src/content/contentScript.ts"]
            }
          ]
        })),
        get lastError() {
          return runtimeLastError === undefined ? undefined : { message: runtimeLastError };
        },
        onInstalled: { addListener: vi.fn() },
        onMessage: {
          addListener: vi.fn((nextListener: MessageListener) => {
            listener = nextListener;
          })
        }
      },
      sidePanel: {
        setPanelBehavior: vi.fn(() => Promise.resolve()),
        setOptions: setOptionsMock
      },
      scripting: {
        executeScript: executeScriptMock
      },
      tabs: {
        get: getTabMock,
        query: queryMock,
        sendMessage: sendMessageMock
      }
    });

    await import("./serviceWorker");
  });

  afterEach(() => {
    vi.useRealTimers();
    vi.unstubAllGlobals();
  });

  it("enables a tab-specific panel when the YouTube content script is ready", () => {
    const respond = vi.fn();

    listener({ type: "CONTENT_READY" }, { tab: { id: 42 } as chrome.tabs.Tab }, respond);

    expect(setOptionsMock).toHaveBeenCalledWith({
      tabId: 42,
      path: "src/sidepanel/index.html",
      enabled: true
    });
    expect(respond).toHaveBeenCalledWith({ ok: true });
  });

  it("routes panel messages only to the active YouTube tab in the current window", async () => {
    listener({ type: "SESSION_STARTED", sessionId: "session_1" }, { url: SIDE_PANEL_URL }, vi.fn());
    await settleServiceWorker();

    expect(queryMock).toHaveBeenCalledWith(
      { active: true, currentWindow: true },
      expect.any(Function)
    );
    expect(sendMessageMock).toHaveBeenCalledWith(
      42,
      { type: "SESSION_STARTED", sessionId: "session_1" },
      expect.any(Function)
    );
  });

  it("keeps session traffic on its explicit originating tab", async () => {
    const respond = vi.fn();

    listener(
      { type: "SESSION_STARTED", sessionId: "session_1", targetTabId: 99 },
      { url: SIDE_PANEL_URL },
      respond
    );
    await settleServiceWorker();

    expect(queryMock).not.toHaveBeenCalled();
    expect(sendMessageMock).toHaveBeenCalledWith(
      99,
      { type: "SESSION_STARTED", sessionId: "session_1", targetTabId: 99 },
      expect.any(Function)
    );
    expect(respond).toHaveBeenCalledWith({ ok: true });
  });

  it("returns a caption snapshot only to the side panel and rejects malformed messages", async () => {
    const captionResponse = vi.fn();
    const keepChannelOpen = listener(
      { type: "GET_PAGE_CAPTION_SNAPSHOT" },
      { url: SIDE_PANEL_URL },
      captionResponse
    );
    const invalidResponse = vi.fn();
    listener({ type: "UNKNOWN" }, {}, invalidResponse);
    const untrustedResponse = vi.fn();
    const untrustedChannel = listener(
      { type: "GET_PAGE_CAPTION_SNAPSHOT" },
      { tab: { id: 42 } as chrome.tabs.Tab, url: YOUTUBE_URL },
      untrustedResponse
    );
    await settleServiceWorker();

    expect(keepChannelOpen).toBe(true);
    expect(captionResponse).toHaveBeenCalledWith(
      expect.objectContaining({
        ok: true,
        targetTabId: 42,
        snapshot: expect.objectContaining({ captionText: "Visible caption" })
      })
    );
    expect(invalidResponse).toHaveBeenCalledWith({
      ok: false,
      error: "Invalid SignSaarthi runtime message."
    });
    expect(untrustedChannel).toBe(false);
    expect(untrustedResponse).toHaveBeenCalledWith({
      ok: false,
      error: "Caption snapshots are available only to the SignSaarthi side panel."
    });
  });

  it.each([
    ["Google Meet", "https://meet.google.com/abc-defg-hij"],
    ["Zoom", "https://app.zoom.us/wc/123456789/join"],
    ["Other tab", "https://example.com/accessibility"]
  ])("reports the active %s tab without attempting caption capture", async (title, url) => {
    queryMock.mockImplementation((_query, callback: (tabs: chrome.tabs.Tab[]) => void) =>
      callback([{ id: 77, active: true, title, url } as chrome.tabs.Tab])
    );
    const respond = vi.fn();

    listener({ type: "GET_PAGE_CAPTION_SNAPSHOT" }, { url: SIDE_PANEL_URL }, respond);
    await settleServiceWorker();

    expect(queryMock).toHaveBeenCalledWith(
      { active: true, currentWindow: true },
      expect.any(Function)
    );
    expect(sendMessageMock).not.toHaveBeenCalled();
    expect(executeScriptMock).not.toHaveBeenCalled();
    expect(respond).toHaveBeenCalledWith({
      ok: true,
      targetTabId: 77,
      snapshot: { title, url, supportStatus: "unsupported" }
    });
  });

  it("rejects public runtime commands from a content-script sender", () => {
    const respond = vi.fn();

    const keepChannelOpen = listener(
      { type: "SESSION_STARTED", sessionId: "session_1" },
      { tab: { id: 42 } as chrome.tabs.Tab, url: YOUTUBE_URL },
      respond
    );

    expect(keepChannelOpen).toBe(false);
    expect(sendMessageMock).not.toHaveBeenCalled();
    expect(respond).toHaveBeenCalledWith({
      ok: false,
      error: "Runtime commands are accepted only from SignSaarthi extension pages."
    });
  });

  it("recovers a no-receiver delivery by injecting once and retrying", async () => {
    sendMessageMock.mockImplementationOnce(failTabMessage(NO_RECEIVER_ERROR));
    const respond = vi.fn();

    expect(
      listener(
        { type: "SESSION_STARTED", sessionId: "session_1", targetTabId: 42 },
        { url: SIDE_PANEL_URL },
        respond
      )
    ).toBe(true);
    await settleServiceWorker();

    expect(executeScriptMock).toHaveBeenCalledTimes(1);
    expect(executeScriptMock).toHaveBeenCalledWith({
      target: { tabId: 42 },
      files: ["src/content/contentScript.ts"]
    });
    expect(sendMessageMock).toHaveBeenCalledTimes(2);
    expect(respond).toHaveBeenCalledWith({ ok: true });
  });

  it("recovers caption delivery through the same bounded receiver path", async () => {
    sendMessageMock.mockImplementationOnce(failTabMessage(NO_RECEIVER_ERROR));
    const respond = vi.fn();

    listener(
      { type: "GET_PAGE_CAPTION_SNAPSHOT", targetTabId: 42 },
      { url: SIDE_PANEL_URL },
      respond
    );
    await settleServiceWorker();

    expect(executeScriptMock).toHaveBeenCalledTimes(1);
    expect(respond).toHaveBeenCalledWith(
      expect.objectContaining({
        ok: true,
        targetTabId: 42,
        snapshot: expect.objectContaining({ captionText: "Visible caption" })
      })
    );
  });

  it("rejects generic delivery errors without inspecting or injecting into the tab", async () => {
    sendMessageMock.mockImplementationOnce(
      failTabMessage("The message port closed before a response was received.")
    );
    const respond = vi.fn();

    listener(
      { type: "SESSION_STARTED", sessionId: "session_1", targetTabId: 42 },
      { url: SIDE_PANEL_URL },
      respond
    );
    await settleServiceWorker();

    expect(getTabMock).not.toHaveBeenCalled();
    expect(executeScriptMock).not.toHaveBeenCalled();
    expect(sendMessageMock).toHaveBeenCalledTimes(1);
    expect(respond).toHaveBeenCalledWith({
      ok: false,
      error: "The message port closed before a response was received."
    });
  });

  it("stops retrying when a recovered delivery fails with a generic error", async () => {
    sendMessageMock
      .mockImplementationOnce(failTabMessage(NO_RECEIVER_ERROR))
      .mockImplementationOnce(failTabMessage("The tab was closed."));
    const respond = vi.fn();

    listener(
      { type: "SESSION_STARTED", sessionId: "session_1", targetTabId: 42 },
      { url: SIDE_PANEL_URL },
      respond
    );
    await settleServiceWorker();

    expect(executeScriptMock).toHaveBeenCalledTimes(1);
    expect(sendMessageMock).toHaveBeenCalledTimes(2);
    expect(respond).toHaveBeenCalledWith({ ok: false, error: "The tab was closed." });
  });

  it("rejects an unsupported target URL before content-script injection", async () => {
    sendMessageMock.mockImplementationOnce(failTabMessage(NO_RECEIVER_ERROR));
    getTabMock.mockImplementation((tabId: number, callback: (tab: chrome.tabs.Tab) => void) =>
      callback({
        id: tabId,
        url: "https://www.youtube.com.example.test/watch?v=demo"
      } as chrome.tabs.Tab)
    );
    const respond = vi.fn();

    listener(
      { type: "SESSION_STARTED", sessionId: "session_1", targetTabId: 42 },
      { url: SIDE_PANEL_URL },
      respond
    );
    await settleServiceWorker();

    expect(executeScriptMock).not.toHaveBeenCalled();
    expect(sendMessageMock).toHaveBeenCalledTimes(1);
    expect(respond).toHaveBeenCalledWith({
      ok: false,
      error: UNSUPPORTED_RECOVERY_TARGET_ERROR
    });
  });

  it("stops recovery when the tab navigates away before a retry", async () => {
    sendMessageMock.mockImplementationOnce(failTabMessage(NO_RECEIVER_ERROR));
    getTabMock
      .mockImplementationOnce((tabId: number, callback: (tab: chrome.tabs.Tab) => void) =>
        callback({ id: tabId, url: YOUTUBE_URL } as chrome.tabs.Tab)
      )
      .mockImplementationOnce((tabId: number, callback: (tab: chrome.tabs.Tab) => void) =>
        callback({
          id: tabId,
          url: YOUTUBE_URL,
          pendingUrl: "https://example.com/next"
        } as chrome.tabs.Tab)
      );
    const respond = vi.fn();

    listener(
      { type: "SESSION_STARTED", sessionId: "session_1", targetTabId: 42 },
      { url: SIDE_PANEL_URL },
      respond
    );
    await settleServiceWorker();

    expect(executeScriptMock).toHaveBeenCalledTimes(1);
    expect(sendMessageMock).toHaveBeenCalledTimes(1);
    expect(respond).toHaveBeenCalledWith({
      ok: false,
      error: UNSUPPORTED_RECOVERY_TARGET_ERROR
    });
  });

  it("exhausts a fixed retry budget without reinjecting", async () => {
    sendMessageMock.mockImplementation(failTabMessage(NO_RECEIVER_ERROR));
    const respond = vi.fn();

    listener(
      { type: "SESSION_STARTED", sessionId: "session_1", targetTabId: 42 },
      { url: SIDE_PANEL_URL },
      respond
    );
    await settleServiceWorker();

    expect(executeScriptMock).toHaveBeenCalledTimes(1);
    expect(sendMessageMock).toHaveBeenCalledTimes(11);
    expect(respond).toHaveBeenCalledTimes(1);
    expect(respond).toHaveBeenCalledWith({ ok: false, error: NO_RECEIVER_ERROR });
  });

  it("coalesces concurrent recovery for the same tab into one injection", async () => {
    sendMessageMock
      .mockImplementationOnce(failTabMessage(NO_RECEIVER_ERROR))
      .mockImplementationOnce(failTabMessage(NO_RECEIVER_ERROR));
    let resolveInjection: (() => void) | undefined;
    executeScriptMock.mockImplementationOnce(
      () =>
        new Promise<unknown[]>((resolve) => {
          resolveInjection = () => resolve([]);
        })
    );
    const firstResponse = vi.fn();
    const secondResponse = vi.fn();

    listener(
      { type: "SESSION_STARTED", sessionId: "session_1", targetTabId: 42 },
      { url: SIDE_PANEL_URL },
      firstResponse
    );
    listener(
      { type: "SESSION_STOPPED", sessionId: "session_1", targetTabId: 42 },
      { url: SIDE_PANEL_URL },
      secondResponse
    );
    await flushMicrotasks();

    expect(executeScriptMock).toHaveBeenCalledTimes(1);
    resolveInjection?.();
    await settleServiceWorker();

    expect(executeScriptMock).toHaveBeenCalledTimes(1);
    expect(sendMessageMock).toHaveBeenCalledTimes(4);
    expect(firstResponse).toHaveBeenCalledWith({ ok: true });
    expect(secondResponse).toHaveBeenCalledWith({ ok: true });
  });

  it("does not fall back to a stale YouTube tab when another tab is active", async () => {
    listener({ type: "CONTENT_READY" }, { tab: { id: 42 } as chrome.tabs.Tab }, vi.fn());
    queryMock.mockImplementation((_query, callback: (tabs: chrome.tabs.Tab[]) => void) =>
      callback([
        {
          id: 77,
          active: true,
          title: "Google Meet",
          url: "https://meet.google.com/abc-defg-hij"
        } as chrome.tabs.Tab
      ])
    );
    sendMessageMock.mockClear();

    const respond = vi.fn();
    listener({ type: "SESSION_STARTED", sessionId: "session_1" }, { url: SIDE_PANEL_URL }, respond);
    await settleServiceWorker();

    expect(sendMessageMock).not.toHaveBeenCalled();
    expect(executeScriptMock).not.toHaveBeenCalled();
    expect(respond).toHaveBeenCalledWith({
      ok: false,
      error: "The active tab is not a supported YouTube page."
    });
  });
});

function completeTabMessage(
  callback: TabMessageCallback | undefined,
  response: unknown,
  error?: string
): void {
  runtimeLastError = error;
  callback?.(response);
  runtimeLastError = undefined;
}

function failTabMessage(error: string) {
  return (_tabId: number, _message: unknown, callback?: TabMessageCallback): void => {
    completeTabMessage(callback, undefined, error);
  };
}

async function flushMicrotasks(): Promise<void> {
  await Promise.resolve();
  await Promise.resolve();
  await Promise.resolve();
}

async function settleServiceWorker(): Promise<void> {
  await flushMicrotasks();
  await vi.runAllTimersAsync();
  await flushMicrotasks();
}
