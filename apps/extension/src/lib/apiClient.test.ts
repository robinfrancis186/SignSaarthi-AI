import { afterEach, describe, expect, it, vi } from "vitest";
import { purgePrivacyData, startSession } from "./apiClient";

const deletedCounts = {
  sessions: 1,
  interpretations: 2,
  avatarQueues: 3,
  feedback: 4,
  rateLimitEntries: 5
};

describe("local API client", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("marks requests as coming from the SignSaarthi extension client", async () => {
    const fetchMock = vi.fn<typeof fetch>();
    fetchMock.mockResolvedValue(
      Response.json({ sessionId: "session_1", status: "active", rawAudioStored: false })
    );
    vi.stubGlobal("fetch", fetchMock);

    await expect(startSession({ source: "mock" })).resolves.toMatchObject({
      sessionId: "session_1"
    });

    const request = fetchMock.mock.calls[0];
    const init = request?.[1];
    expect(init).toBeDefined();
    if (!init) {
      throw new Error("Expected fetch request options.");
    }
    expect(new Headers(init.headers).get("X-SignSaarthi-Client")).toBe("extension-v1");
    const requestUrl = new URL(String(request?.[0]));
    const configuredOrigin = new URL(
      import.meta.env.VITE_SIGNSAARTHI_API_BASE_URL ?? "http://127.0.0.1:8787"
    ).origin;
    expect(requestUrl.origin).toBe(configuredOrigin);
    expect(requestUrl.pathname).toBe("/api/session/start");
  });

  it.each([
    {
      label: "session",
      requestBody: {
        scope: "session",
        sessionId: "session_1",
        confirmation: "purge_session"
      } as const,
      responseBody: {
        scope: "session",
        sessionId: "session_1",
        status: "purged",
        deleted: deletedCounts,
        rawAudioStored: false,
        rawVideoStored: false
      } as const
    },
    {
      label: "all-data",
      requestBody: {
        scope: "all",
        confirmation: "purge_all_local_data"
      } as const,
      responseBody: {
        scope: "all",
        status: "purged",
        deleted: deletedCounts,
        rawAudioStored: false,
        rawVideoStored: false
      } as const
    }
  ])("posts the exact $label privacy purge contract", async ({ requestBody, responseBody }) => {
    const fetchMock = vi.fn<typeof fetch>();
    fetchMock.mockResolvedValue(Response.json(responseBody));
    vi.stubGlobal("fetch", fetchMock);

    await expect(purgePrivacyData(requestBody)).resolves.toEqual(responseBody);

    const request = fetchMock.mock.calls[0];
    const init = request?.[1];
    expect(init).toBeDefined();
    if (!init) {
      throw new Error("Expected fetch request options.");
    }
    expect(init.method).toBe("POST");
    expect(new Headers(init.headers).get("Content-Type")).toBe("application/json");
    expect(new Headers(init.headers).get("X-SignSaarthi-Client")).toBe("extension-v1");
    expect(JSON.parse(String(init.body))).toEqual(requestBody);
    expect(new URL(String(request?.[0])).pathname).toBe("/api/privacy/purge");
  });

  it("rejects a privacy purge response that violates the shared response schema", async () => {
    const fetchMock = vi.fn<typeof fetch>();
    fetchMock.mockResolvedValue(
      Response.json({
        scope: "all",
        status: "purged",
        deleted: deletedCounts,
        rawAudioStored: false,
        rawVideoStored: true
      })
    );
    vi.stubGlobal("fetch", fetchMock);

    await expect(
      purgePrivacyData({ scope: "all", confirmation: "purge_all_local_data" })
    ).rejects.toMatchObject({ name: "ZodError" });
  });

  it("reports privacy purge server errors", async () => {
    const fetchMock = vi.fn<typeof fetch>();
    fetchMock.mockResolvedValue(Response.json({ error: "purge_failed" }, { status: 500 }));
    vi.stubGlobal("fetch", fetchMock);

    await expect(
      purgePrivacyData({ scope: "all", confirmation: "purge_all_local_data" })
    ).rejects.toThrow("Local API request failed (500).");
  });

  it("reports an offline local API while purging privacy data", async () => {
    const fetchMock = vi.fn<typeof fetch>();
    fetchMock.mockRejectedValue(new TypeError("Failed to fetch"));
    vi.stubGlobal("fetch", fetchMock);

    await expect(
      purgePrivacyData({
        scope: "session",
        sessionId: "session_1",
        confirmation: "purge_session"
      })
    ).rejects.toThrow("Local SignSaarthi API is unavailable.");
  });
});
