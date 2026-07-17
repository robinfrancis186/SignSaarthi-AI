import { runtimePublicMessageSchema, type RuntimeMessage } from "@signsaarthi/shared";
import type { CaptionSnapshotRequest, CaptionSnapshotResponse } from "../lib/runtime";

const CONTENT_SCRIPT_RECOVERY_ATTEMPTS = 10;
const CONTENT_SCRIPT_RECOVERY_DELAY_MS = 50;
const SIDE_PANEL_PATH = "src/sidepanel/index.html";
const YOUTUBE_CONTENT_SCRIPT_MATCH = "https://www.youtube.com/*";
const YOUTUBE_ORIGIN = "https://www.youtube.com";
const UNSUPPORTED_RECOVERY_TARGET_ERROR =
  "Content-script recovery is available only on supported YouTube tabs.";

type DeliveryAttempt<T> = { delivered: true; value: T } | { delivered: false; error: string };

type RecoveryCheck = { ok: true } | { ok: false; error: string };
type RuntimeDeliveryResponse = { ok: boolean; error?: string };

const contentScriptRecoveryByTab = new Map<number, Promise<void>>();

chrome.runtime.onInstalled.addListener(() => {
  void chrome.sidePanel.setPanelBehavior({ openPanelOnActionClick: false });
});

chrome.runtime.onMessage.addListener((message: unknown, sender, sendResponse) => {
  if (isContentReadyMessage(message) && sender.tab?.id !== undefined) {
    void chrome.sidePanel.setOptions({
      tabId: sender.tab.id,
      path: SIDE_PANEL_PATH,
      enabled: true
    });
    sendResponse({ ok: true });
    return;
  }

  if (isCaptionSnapshotRequest(message)) {
    if (sender.url !== chrome.runtime.getURL(SIDE_PANEL_PATH)) {
      sendResponse({
        ok: false,
        error: "Caption snapshots are available only to the SignSaarthi side panel."
      });
      return false;
    }
    requestCaptionSnapshot(message.targetTabId, sendResponse);
    return true;
  }

  const parsedMessage = runtimePublicMessageSchema.safeParse(message);
  if (!parsedMessage.success) {
    sendResponse({ ok: false, error: "Invalid SignSaarthi runtime message." });
    return false;
  }

  const fromExtensionPage =
    sender.url?.startsWith(`chrome-extension://${chrome.runtime.id}/`) ?? false;
  if (!fromExtensionPage) {
    sendResponse({
      ok: false,
      error: "Runtime commands are accepted only from SignSaarthi extension pages."
    });
    return false;
  }

  const runtimeMessage = parsedMessage.data;
  const targetTabId = "targetTabId" in runtimeMessage ? runtimeMessage.targetTabId : undefined;
  if (targetTabId !== undefined) {
    sendToTab(targetTabId, runtimeMessage, sendResponse);
    return true;
  }

  broadcastToActiveYouTubeTab(runtimeMessage, sendResponse);
  return true;
});

function broadcastToActiveYouTubeTab(
  message: RuntimeMessage,
  sendResponse: (response: { ok: boolean; error?: string }) => void
): void {
  chrome.tabs.query(
    { active: true, currentWindow: true },
    (tabs) => {
      const activeTab = tabs[0];
      if (!activeTab || activeTab.id === undefined) {
        sendResponse({ ok: false, error: "Open a YouTube tab to start the avatar overlay." });
        return;
      }
      const tabId = activeTab.id;
      if (!isSupportedYouTubeUrl(getCurrentTabUrl(activeTab))) {
        sendResponse({
          ok: false,
          error: "The active tab is not a supported YouTube page."
        });
        return;
      }
      sendToTab(tabId, message, sendResponse);
    }
  );
}

function sendToTab(
  tabId: number,
  message: RuntimeMessage,
  sendResponse: (response: { ok: boolean; error?: string }) => void
): void {
  void deliverToTab(tabId, message)
    .then((delivery) => {
      if (delivery.delivered) {
        return delivery.value;
      }
      return recoverContentScriptAndRetry(tabId, message, delivery.error);
    })
    .then(sendResponse);
}

async function recoverContentScriptAndRetry(
  tabId: number,
  message: RuntimeMessage,
  initialError: string
): Promise<RuntimeDeliveryResponse> {
  return recoverDelivery<RuntimeDeliveryResponse>(
    tabId,
    initialError,
    () => deliverToTab(tabId, message),
    (error) => ({ ok: false, error })
  );
}

async function injectDeclaredContentScript(tabId: number): Promise<RecoveryCheck> {
  const target = await validateRecoveryTarget(tabId);
  if (!target.ok) {
    return target;
  }

  const files = [
    ...new Set(
      chrome.runtime
        .getManifest()
        .content_scripts?.filter((script) => script.matches?.includes(YOUTUBE_CONTENT_SCRIPT_MATCH))
        .flatMap((script) => script.js ?? []) ?? []
    )
  ];
  if (!files.length || !chrome.scripting?.executeScript) {
    return { ok: false, error: "No declared YouTube content script is available for recovery." };
  }

  try {
    await chrome.scripting.executeScript({ target: { tabId }, files });
    return { ok: true };
  } catch (caught) {
    return {
      ok: false,
      error:
        caught instanceof Error ? caught.message : "Unable to inject the YouTube content script."
    };
  }
}

function deliverToTab(
  tabId: number,
  message: RuntimeMessage
): Promise<DeliveryAttempt<RuntimeDeliveryResponse>> {
  return new Promise((resolve) => {
    chrome.tabs.sendMessage(tabId, message, () => {
      const error = chrome.runtime.lastError?.message;
      resolve(error ? { delivered: false, error } : { delivered: true, value: { ok: true } });
    });
  });
}

function requestCaptionSnapshot(
  targetTabId: number | undefined,
  sendResponse: (response: CaptionSnapshotResponse) => void
): void {
  const respondFromTab = (tabId: number): void => {
    void requestCaptionFromTab(tabId)
      .then((delivery) => {
        if (delivery.delivered) {
          return delivery.value;
        }
        return recoverCaptionSnapshot(tabId, delivery.error);
      })
      .then(sendResponse);
  };

  if (targetTabId !== undefined) {
    respondFromTab(targetTabId);
    return;
  }

  chrome.tabs.query(
    { active: true, currentWindow: true },
    (activeTabs) => {
      const activeTab = activeTabs[0];
      if (!activeTab || activeTab.id === undefined) {
        sendResponse({ ok: false, error: "Open a browser tab to inspect its caption support." });
        return;
      }
      const tabId = activeTab.id;
      if (!isSupportedYouTubeUrl(getCurrentTabUrl(activeTab))) {
        sendResponse(createUnsupportedCaptionSnapshot(activeTab, tabId));
        return;
      }
      respondFromTab(tabId);
    }
  );
}

async function recoverCaptionSnapshot(
  tabId: number,
  initialError: string
): Promise<CaptionSnapshotResponse> {
  return recoverDelivery(
    tabId,
    initialError,
    () => requestCaptionFromTab(tabId),
    (error) => ({ ok: false, error })
  );
}

function requestCaptionFromTab(tabId: number): Promise<DeliveryAttempt<CaptionSnapshotResponse>> {
  return new Promise((resolve) => {
    chrome.tabs.sendMessage(
      tabId,
      { type: "GET_PAGE_CAPTION_SNAPSHOT" } satisfies CaptionSnapshotRequest,
      (response?: CaptionSnapshotResponse) => {
        const error = chrome.runtime.lastError?.message;
        if (error || !response) {
          resolve({ delivered: false, error: error ?? "No caption response from page." });
          return;
        }
        resolve({
          delivered: true,
          value: response.ok ? { ...response, targetTabId: tabId } : response
        });
      }
    );
  });
}

async function recoverDelivery<T>(
  tabId: number,
  initialError: string,
  deliver: () => Promise<DeliveryAttempt<T>>,
  failure: (error: string) => T
): Promise<T> {
  if (!isNoReceiverError(initialError)) {
    return failure(initialError);
  }

  const activeRecovery = contentScriptRecoveryByTab.get(tabId);
  if (activeRecovery) {
    await activeRecovery;
    const target = await validateRecoveryTarget(tabId);
    if (!target.ok) {
      return failure(target.error);
    }
    const delivery = await deliver();
    return delivery.delivered ? delivery.value : failure(delivery.error);
  }

  let finishRecovery: (() => void) | undefined;
  const recoveryCompleted = new Promise<void>((resolve) => {
    finishRecovery = resolve;
  });
  contentScriptRecoveryByTab.set(tabId, recoveryCompleted);

  try {
    const injection = await injectDeclaredContentScript(tabId);
    if (!injection.ok) {
      return failure(injection.error);
    }

    let lastError = initialError;
    for (let attempt = 0; attempt < CONTENT_SCRIPT_RECOVERY_ATTEMPTS; attempt += 1) {
      await delay(CONTENT_SCRIPT_RECOVERY_DELAY_MS);
      const target = await validateRecoveryTarget(tabId);
      if (!target.ok) {
        return failure(target.error);
      }

      const delivery = await deliver();
      if (delivery.delivered) {
        return delivery.value;
      }
      lastError = delivery.error;
      if (!isNoReceiverError(lastError)) {
        return failure(lastError);
      }
    }

    return failure(lastError);
  } finally {
    if (contentScriptRecoveryByTab.get(tabId) === recoveryCompleted) {
      contentScriptRecoveryByTab.delete(tabId);
    }
    finishRecovery?.();
  }
}

function validateRecoveryTarget(tabId: number): Promise<RecoveryCheck> {
  return new Promise((resolve) => {
    try {
      chrome.tabs.get(tabId, (tab) => {
        const error = chrome.runtime.lastError?.message;
        if (error) {
          resolve({ ok: false, error });
          return;
        }

        if (
          !tab?.url ||
          !isSupportedYouTubeUrl(tab.url) ||
          (tab.pendingUrl !== undefined && !isSupportedYouTubeUrl(tab.pendingUrl))
        ) {
          resolve({ ok: false, error: UNSUPPORTED_RECOVERY_TARGET_ERROR });
          return;
        }
        resolve({ ok: true });
      });
    } catch (caught) {
      resolve({
        ok: false,
        error:
          caught instanceof Error ? caught.message : "Unable to inspect the recovery target tab."
      });
    }
  });
}

function isSupportedYouTubeUrl(url: string): boolean {
  try {
    return new URL(url).origin === YOUTUBE_ORIGIN;
  } catch {
    return false;
  }
}

function getCurrentTabUrl(tab: chrome.tabs.Tab): string {
  return tab.pendingUrl ?? tab.url ?? "about:blank";
}

function createUnsupportedCaptionSnapshot(
  tab: chrome.tabs.Tab,
  tabId: number
): CaptionSnapshotResponse {
  const url = getCurrentTabUrl(tab);
  return {
    ok: true,
    targetTabId: tabId,
    snapshot: {
      title: tab.title?.trim() || "Active browser tab",
      url,
      supportStatus: "unsupported"
    }
  };
}

function isNoReceiverError(error: string): boolean {
  return /^(?:Could not establish connection\.\s*)?Receiving end does not exist\.?$/i.test(
    error.trim()
  );
}

function delay(durationMs: number): Promise<void> {
  return new Promise((resolve) => globalThis.setTimeout(resolve, durationMs));
}

function isContentReadyMessage(message: unknown): message is { type: "CONTENT_READY" } {
  return (
    typeof message === "object" &&
    message !== null &&
    (message as { type?: unknown }).type === "CONTENT_READY"
  );
}

function isCaptionSnapshotRequest(message: unknown): message is CaptionSnapshotRequest {
  if (typeof message !== "object" || message === null) {
    return false;
  }
  const candidate = message as { type?: unknown; targetTabId?: unknown };
  return (
    candidate.type === "GET_PAGE_CAPTION_SNAPSHOT" &&
    (candidate.targetTabId === undefined ||
      (typeof candidate.targetTabId === "number" &&
        Number.isInteger(candidate.targetTabId) &&
        candidate.targetTabId >= 0))
  );
}
