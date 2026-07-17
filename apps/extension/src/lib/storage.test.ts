import { settingsSchema } from "@signsaarthi/shared";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  appendSessionHistory,
  clearLocalData,
  loadSessionHistory,
  loadSettings,
  saveSettings,
  type SessionHistoryItem
} from "./storage";

const SETTINGS_KEY = "signsaarthi.settings";
const HISTORY_KEY = "signsaarthi.sessionHistory";

function makeHistoryItem(id: string, summary = `Summary ${id}`): SessionHistoryItem {
  return {
    id,
    sourceLabel: "YouTube captions",
    pageTitle: "Accessible lecture",
    summary,
    confidence: "medium",
    createdAt: "2026-07-13T00:00:00.000Z"
  };
}

describe("storage", () => {
  beforeEach(() => {
    vi.unstubAllGlobals();
    globalThis.localStorage.clear();
    vi.stubGlobal("chrome", undefined);
  });

  afterEach(() => {
    vi.unstubAllGlobals();
    globalThis.localStorage.clear();
    vi.restoreAllMocks();
  });

  it("persists schema-normalized settings in localStorage when Chrome storage is absent", async () => {
    const settings = settingsSchema.parse({
      avatarName: "Meera",
      avatarSpeed: 1.5,
      saveHistory: true
    });

    await saveSettings(settings);

    expect(JSON.parse(globalThis.localStorage.getItem(SETTINGS_KEY) ?? "null")).toEqual(settings);
    await expect(loadSettings()).resolves.toEqual(settings);
  });

  it("recovers defaults from malformed JSON or invalid legacy settings", async () => {
    const defaults = settingsSchema.parse({});

    globalThis.localStorage.setItem(SETTINGS_KEY, "{not-json");
    await expect(loadSettings()).resolves.toEqual(defaults);

    globalThis.localStorage.setItem(SETTINGS_KEY, JSON.stringify({ avatarSpeed: "fast" }));
    await expect(loadSettings()).resolves.toEqual(defaults);

    globalThis.localStorage.setItem(SETTINGS_KEY, JSON.stringify({ avatarSpeed: 1.5 }));
    await expect(loadSettings()).resolves.toEqual(
      expect.objectContaining({ avatarSpeed: 1.5, avatarName: "Ananya" })
    );
  });

  it("filters malformed local history and caps deduplicated appends", async () => {
    const storedHistory = [
      makeHistoryItem("existing", "Old summary"),
      { ...makeHistoryItem("bad-page-title"), pageTitle: 42 },
      { summary: "Missing required fields" }
    ];
    globalThis.localStorage.setItem(HISTORY_KEY, JSON.stringify(storedHistory));

    await expect(loadSessionHistory()).resolves.toEqual([storedHistory[0]]);

    for (let index = 0; index < 9; index += 1) {
      await appendSessionHistory(makeHistoryItem(`session-${index}`));
    }
    const updated = await appendSessionHistory(makeHistoryItem("session-8", "Updated summary"));

    expect(updated).toHaveLength(8);
    expect(updated[0]).toMatchObject({ id: "session-8", summary: "Updated summary" });
    expect(updated.filter((item) => item.id === "session-8")).toHaveLength(1);
    await expect(loadSessionHistory()).resolves.toEqual(updated);
  });

  it("clears only SignSaarthi fallback keys", async () => {
    globalThis.localStorage.setItem(SETTINGS_KEY, JSON.stringify(settingsSchema.parse({})));
    globalThis.localStorage.setItem(HISTORY_KEY, JSON.stringify([makeHistoryItem("one")]));
    globalThis.localStorage.setItem("unrelated.preview.state", "keep");

    await clearLocalData();

    expect(globalThis.localStorage.getItem(SETTINGS_KEY)).toBeNull();
    expect(globalThis.localStorage.getItem(HISTORY_KEY)).toBeNull();
    expect(globalThis.localStorage.getItem("unrelated.preview.state")).toBe("keep");
  });

  it("keeps preview mode usable when localStorage access throws", async () => {
    const storageError = new DOMException("Blocked", "SecurityError");
    vi.stubGlobal("localStorage", {
      getItem: vi.fn(() => {
        throw storageError;
      }),
      setItem: vi.fn(() => {
        throw storageError;
      }),
      removeItem: vi.fn(() => {
        throw storageError;
      })
    });

    await expect(loadSettings()).resolves.toEqual(settingsSchema.parse({}));
    await expect(loadSessionHistory()).resolves.toEqual([]);
    await expect(saveSettings(settingsSchema.parse({}))).resolves.toBeUndefined();
    await expect(appendSessionHistory(makeHistoryItem("blocked"))).resolves.toEqual([
      makeHistoryItem("blocked")
    ]);
    await expect(clearLocalData()).resolves.toBeUndefined();
  });

  it("continues to prefer Chrome storage when it is available", async () => {
    const chromeSettings = settingsSchema.parse({ avatarName: "Kabir", avatarSpeed: 1.25 });
    const nextSettings = settingsSchema.parse({ avatarName: "Arjun" });
    const storageState: Record<string, unknown> = {
      [SETTINGS_KEY]: chromeSettings,
      [HISTORY_KEY]: [makeHistoryItem("chrome-history")]
    };
    const get = vi.fn((keys: string[], callback: (result: Record<string, unknown>) => void) => {
      callback(Object.fromEntries(keys.map((key) => [key, storageState[key]])));
    });
    const set = vi.fn((value: Record<string, unknown>, callback: () => void) => {
      Object.assign(storageState, value);
      callback();
    });
    const clear = vi.fn((callback: () => void) => callback());
    vi.stubGlobal("chrome", {
      runtime: {},
      storage: { local: { get, set, clear } }
    });
    globalThis.localStorage.setItem(
      SETTINGS_KEY,
      JSON.stringify(settingsSchema.parse({ avatarName: "Meera" }))
    );

    await expect(loadSettings()).resolves.toEqual(chromeSettings);
    await saveSettings(nextSettings);
    await expect(appendSessionHistory(makeHistoryItem("new-chrome-history"))).resolves.toEqual([
      makeHistoryItem("new-chrome-history"),
      makeHistoryItem("chrome-history")
    ]);
    await clearLocalData();

    expect(set).toHaveBeenCalledWith({ [SETTINGS_KEY]: nextSettings }, expect.any(Function));
    expect(set).toHaveBeenCalledWith(
      {
        [HISTORY_KEY]: [makeHistoryItem("new-chrome-history"), makeHistoryItem("chrome-history")]
      },
      expect.any(Function)
    );
    expect(clear).toHaveBeenCalledTimes(1);
    expect(JSON.parse(globalThis.localStorage.getItem(SETTINGS_KEY) ?? "null")).toMatchObject({
      avatarName: "Meera"
    });
  });

  it("recovers default settings from malformed Chrome storage values", async () => {
    const get = vi.fn((keys: string[], callback: (result: Record<string, unknown>) => void) => {
      callback(Object.fromEntries(keys.map((key) => [key, { avatarSpeed: "fast" }])));
    });
    vi.stubGlobal("chrome", {
      runtime: {},
      storage: {
        local: {
          get,
          set: vi.fn((_value: Record<string, unknown>, callback: () => void) => callback()),
          clear: vi.fn((callback: () => void) => callback())
        }
      }
    });

    await expect(loadSettings()).resolves.toEqual(settingsSchema.parse({}));
  });
});
