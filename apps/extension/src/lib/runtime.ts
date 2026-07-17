import { runtimeMessageSchema, type RuntimeMessage } from "@signsaarthi/shared";
import type { YouTubeCaptionSnapshot } from "../adapters/youtubeCaptionAdapter";

export type { ExtensionStatus, RuntimeMessage } from "@signsaarthi/shared";

export type CaptionSnapshotRequest = { type: "GET_PAGE_CAPTION_SNAPSHOT"; targetTabId?: number };
export type PageCaptionSnapshot = YouTubeCaptionSnapshot & { targetTabId?: number };
export type CaptionSnapshotResponse =
  | { ok: true; snapshot: YouTubeCaptionSnapshot; targetTabId?: number }
  | { ok: false; error: string };
export type CaptionSnapshotAttempt =
  | { ok: true; snapshot: PageCaptionSnapshot }
  | { ok: false; reason: "transient" | "unavailable"; error: string };

export async function sendRuntimeMessage(message: RuntimeMessage): Promise<boolean> {
  const validatedMessage = runtimeMessageSchema.parse(message);

  if (!globalThis.chrome?.runtime?.sendMessage) {
    return true;
  }

  return new Promise<boolean>((resolve) => {
    globalThis.chrome.runtime.sendMessage(validatedMessage, (response?: { ok?: boolean }) => {
      const lastError = globalThis.chrome?.runtime?.lastError;
      if (lastError) {
        resolve(false);
        return;
      }
      resolve(response?.ok !== false);
    });
  });
}

export async function requestPageCaptionSnapshotAttempt(
  targetTabId?: number
): Promise<CaptionSnapshotAttempt> {
  if (!globalThis.chrome?.runtime?.sendMessage) {
    return {
      ok: false,
      reason: "unavailable",
      error: "Chrome runtime messaging is unavailable."
    };
  }

  return new Promise<CaptionSnapshotAttempt>((resolve) => {
    let settled = false;
    const timeout = globalThis.setTimeout(() => {
      if (!settled) {
        settled = true;
        resolve({
          ok: false,
          reason: "transient",
          error: "Caption snapshot request timed out."
        });
      }
    }, 1500);

    globalThis.chrome.runtime.sendMessage(
      {
        type: "GET_PAGE_CAPTION_SNAPSHOT",
        ...(targetTabId === undefined ? {} : { targetTabId })
      } satisfies CaptionSnapshotRequest,
      (response?: CaptionSnapshotResponse) => {
        if (settled) {
          return;
        }
        settled = true;
        globalThis.clearTimeout(timeout);
        const lastError = globalThis.chrome?.runtime?.lastError;
        if (lastError) {
          resolve({
            ok: false,
            reason: "transient",
            error: lastError.message ?? "Caption snapshot message failed."
          });
          return;
        }
        if (!response) {
          resolve({ ok: false, reason: "transient", error: "No caption snapshot response." });
          return;
        }
        if (!response.ok) {
          resolve({
            ok: false,
            reason: isTransientRuntimeError(response.error) ? "transient" : "unavailable",
            error: response.error
          });
          return;
        }
        resolve({
          ok: true,
          snapshot: {
            ...response.snapshot,
            ...(response.targetTabId === undefined ? {} : { targetTabId: response.targetTabId })
          }
        });
      }
    );
  });
}

export async function requestPageCaptionSnapshot(
  targetTabId?: number
): Promise<PageCaptionSnapshot | undefined> {
  const attempt = await requestPageCaptionSnapshotAttempt(targetTabId);
  return attempt.ok ? attempt.snapshot : undefined;
}

function isTransientRuntimeError(error: string): boolean {
  return /receiving end does not exist|could not establish connection|message port closed|no caption response|extension context invalidated/i.test(
    error
  );
}
