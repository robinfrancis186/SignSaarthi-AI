import { settingsSchema, type Settings } from "@signsaarthi/shared";

const SETTINGS_KEY = "signsaarthi.settings";
const HISTORY_KEY = "signsaarthi.sessionHistory";
const MAX_HISTORY_ITEMS = 8;

export type SessionHistoryItem = {
  id: string;
  sourceLabel: string;
  pageTitle?: string;
  summary: string;
  confidence: string;
  createdAt: string;
};

export async function loadSettings(): Promise<Settings> {
  if (!globalThis.chrome?.storage?.local) {
    return parseSettings(readLocalStorageValue(SETTINGS_KEY));
  }

  const value = await chromeStorageGet(SETTINGS_KEY);
  return parseSettings(value);
}

export async function saveSettings(settings: Settings): Promise<void> {
  if (!globalThis.chrome?.storage?.local) {
    writeLocalStorageValue(SETTINGS_KEY, settings);
    return;
  }
  await chromeStorageSet({ [SETTINGS_KEY]: settings });
}

export async function clearLocalData(): Promise<void> {
  if (!globalThis.chrome?.storage?.local) {
    removeLocalStorageValue(SETTINGS_KEY);
    removeLocalStorageValue(HISTORY_KEY);
    return;
  }
  await new Promise<void>((resolve, reject) => {
    globalThis.chrome.storage.local.clear(() => {
      const lastError = globalThis.chrome.runtime?.lastError;
      if (lastError) {
        reject(new Error(lastError.message));
        return;
      }
      resolve();
    });
  });
}

export async function loadSessionHistory(): Promise<SessionHistoryItem[]> {
  if (!globalThis.chrome?.storage?.local) {
    return parseSessionHistory(readLocalStorageValue(HISTORY_KEY));
  }

  const value = await chromeStorageGet(HISTORY_KEY);
  return parseSessionHistory(value);
}

export async function appendSessionHistory(item: SessionHistoryItem): Promise<SessionHistoryItem[]> {
  const current = await loadSessionHistory();
  const nextHistory = [item, ...current.filter((entry) => entry.id !== item.id)].slice(0, MAX_HISTORY_ITEMS);

  if (!globalThis.chrome?.storage?.local) {
    writeLocalStorageValue(HISTORY_KEY, nextHistory);
    return nextHistory;
  }

  await chromeStorageSet({ [HISTORY_KEY]: nextHistory });
  return nextHistory;
}

function parseSettings(value: unknown): Settings {
  const parsed = settingsSchema.safeParse(value ?? {});
  return parsed.success ? parsed.data : settingsSchema.parse({});
}

function parseSessionHistory(value: unknown): SessionHistoryItem[] {
  if (!Array.isArray(value)) {
    return [];
  }

  return value.filter(isSessionHistoryItem).slice(0, MAX_HISTORY_ITEMS);
}

function readLocalStorageValue(key: string): unknown {
  try {
    const value = globalThis.localStorage?.getItem(key);
    return value === null || value === undefined ? undefined : (JSON.parse(value) as unknown);
  } catch {
    return undefined;
  }
}

function writeLocalStorageValue(key: string, value: unknown): void {
  try {
    globalThis.localStorage?.setItem(key, JSON.stringify(value));
  } catch {
    // Preview mode must remain usable when storage access is blocked.
  }
}

function removeLocalStorageValue(key: string): void {
  try {
    globalThis.localStorage?.removeItem(key);
  } catch {
    // Preview mode must remain usable when storage access is blocked.
  }
}

function chromeStorageGet(key: string): Promise<unknown> {
  return new Promise((resolve, reject) => {
    globalThis.chrome.storage.local.get([key], (result) => {
      const lastError = globalThis.chrome.runtime?.lastError;
      if (lastError) {
        reject(new Error(lastError.message));
        return;
      }
      resolve(result[key]);
    });
  });
}

function chromeStorageSet(value: Record<string, unknown>): Promise<void> {
  return new Promise((resolve, reject) => {
    globalThis.chrome.storage.local.set(value, () => {
      const lastError = globalThis.chrome.runtime?.lastError;
      if (lastError) {
        reject(new Error(lastError.message));
        return;
      }
      resolve();
    });
  });
}

function isSessionHistoryItem(value: unknown): value is SessionHistoryItem {
  if (!value || typeof value !== "object") {
    return false;
  }
  const item = value as Record<string, unknown>;
  return (
    typeof item.id === "string" &&
    typeof item.sourceLabel === "string" &&
    (item.pageTitle === undefined || typeof item.pageTitle === "string") &&
    typeof item.summary === "string" &&
    typeof item.confidence === "string" &&
    typeof item.createdAt === "string"
  );
}
