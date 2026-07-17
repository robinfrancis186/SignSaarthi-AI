import { act, cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { resetMotionClipMapForTests } from "../lib/motionLibraryLoader";
import {
  LIVE_CAPTION_LOSS_POLL_COUNT,
  LIVE_CAPTION_POLL_INTERVAL_MS,
  SidePanelApp
} from "./SidePanelApp";

const baseInterpretResponse = {
  id: "interp_1",
  sessionId: "session_1",
  chunkId: "chunk_1",
  originalText: "Artificial intelligence can help education in Mumbai.",
  cleanedText: "Artificial intelligence can help education in Mumbai.",
  simplifiedText: "AI can help education.",
  keyPoints: ["AI can help education"],
  detectedTerms: ["artificial intelligence", "education"],
  glossSequence: [
    {
      id: "gloss_1",
      token: "computer",
      gloss: "COMPUTER",
      isVerified: false,
      fallback: "fingerspell",
      confidence: "medium",
      sourceSegmentId: "segment_rolling",
      planningActionId: "action_computer"
    },
    {
      id: "gloss_2",
      token: "Mumbai",
      gloss: "MUMBAI",
      isVerified: false,
      fallback: "fingerspell",
      confidence: "low",
      sourceSegmentId: "segment_rolling",
      planningActionId: "action_mumbai"
    }
  ],
  confidence: "medium",
  warnings: ["Some terms use fallback output."],
  createdAt: "2026-07-07T00:00:00.000Z",
  avatarQueue: {
    id: "queue_api_1",
    interpretationId: "interp_1",
    speed: 0.75,
    steps: [
      {
        id: "step_1",
        glossItemId: "gloss_1",
        kind: "sign",
        label: "COMPUTER",
        motionClipId: "include-computer",
        motionSource: "include_dataset",
        expertReviewed: false,
        durationMs: 1100,
        confidence: "high",
        sourceSegmentId: "segment_rolling",
        planningActionId: "action_computer"
      },
      {
        id: "step_2",
        glossItemId: "gloss_2",
        kind: "fingerspell",
        label: "MUMBAI",
        fingerspellingSource: "islrtc_official",
        durationMs: 1500,
        confidence: "low",
        sourceSegmentId: "segment_rolling",
        planningActionId: "action_mumbai"
      }
    ],
    honestFallbackText:
      "Unsupported words use official ISLRTC A-Z fingerspelling; automated avatar transfer is not expert-reviewed.",
    motionReviewNotice:
      "Real INCLUDE keypoint motion; dataset-derived isolated signs are not expert-certified ISL.",
    createdAt: "2026-07-07T00:00:00.000Z"
  },
  model: {
    id: "isl-lexicon-2026-07-07",
    version: "0.1.0",
    status: "ready",
    engine: "lexicon_ranker",
    trainedAt: "2026-07-07T00:00:00.000Z",
    trainingDataset: {
      primaryDataset: "islrtc",
      displayName: "ISLRTC official index + INCLUDE motion metadata",
      recordCount: 12434,
      classCount: 12434,
      citationUrls: [
        "https://divyangjan.depwd.gov.in/islrtc/",
        "https://huggingface.co/datasets/ai4bharat/INCLUDE"
      ]
    },
    notes: ["Vocabulary provenance does not claim an expert-reviewed avatar sign."]
  },
  modelBackedGlossCount: 1,
  motionBackedGlossCount: 2
} as const;

type StorageState = Record<string, unknown>;
type SnapshotConfig = {
  captionTexts?: string[];
  captionSnapshots?: Array<{
    captionText?: string;
    supportStatus?: "supported" | "captions_unavailable" | "unsupported";
    error?: string;
  }>;
  rejectMessageType?: string;
  storageState?: StorageState;
  targetTabId?: number;
  url?: string;
};
type ApiFailureMode = "server" | "offline";
type FetchMockOptions = {
  failPrivacyPurge?: boolean;
  failSessionStart?: boolean;
  feedbackFailure?: ApiFailureMode;
  glossaryFailure?: ApiFailureMode;
  models?: unknown[];
  motionClipIds?: string[];
  privacyPurgeFailure?: "offline";
};

function makeRuntimeMotionClip(index: number) {
  const makeFrame = (offset: number) => ({
    pose: Array.from({ length: 25 }, (_, pointIndex) => ({
      x: 0.42 + (pointIndex % 5) * 0.035 + offset,
      y: 0.14 + Math.floor(pointIndex / 5) * 0.08
    })),
    leftHand: Array.from({ length: 21 }, (_, pointIndex) => ({
      x: 0.25 + (pointIndex % 4) * 0.018 + offset,
      y: 0.44 + Math.floor(pointIndex / 4) * 0.025
    })),
    rightHand: Array.from({ length: 21 }, (_, pointIndex) => ({
      x: 0.64 + (pointIndex % 4) * 0.018 - offset,
      y: 0.44 + Math.floor(pointIndex / 4) * 0.025
    }))
  });
  return {
    id: `runtime-backlog-${index}`,
    label: `WORD${index}`,
    fps: 60,
    frames: [makeFrame(0), makeFrame(0.03)]
  };
}

function makeInterpretResponse(rawText: string, requestNumber: number) {
  const id = `interp_${requestNumber}`;
  const isLiveUpdate = rawText === "Second caption.";
  if (rawText === "Backlog caption.") {
    const glossSequence = Array.from({ length: 24 }, (_, index) => ({
      id: `gloss_backlog_${index}`,
      token: `word${index}`,
      gloss: `WORD${index}`,
      isVerified: false,
      fallback: "unknown" as const,
      confidence: "low" as const,
      sourceSegmentId: "segment_backlog",
      planningActionId: `action_backlog_${index}`
    }));
    return {
      ...baseInterpretResponse,
      id,
      originalText: rawText,
      cleanedText: rawText,
      simplifiedText: rawText,
      glossSequence,
      avatarQueue: {
        ...baseInterpretResponse.avatarQueue,
        id: `queue_api_${requestNumber}`,
        interpretationId: id,
        steps: glossSequence.map((item, index) => ({
          id: `step_backlog_${index}`,
          glossItemId: item.id,
          kind: "sign" as const,
          label: item.gloss,
          motionClipId: `runtime-backlog-${index}`,
          motionSource: "isign_research" as const,
          expertReviewed: false,
          durationMs: 1_500,
          confidence: "low" as const,
          sourceSegmentId: item.sourceSegmentId,
          planningActionId: item.planningActionId
        })),
        motionClips: glossSequence.map((_, index) => makeRuntimeMotionClip(index))
      }
    };
  }
  return {
    ...baseInterpretResponse,
    id,
    chunkId: `chunk_${requestNumber}`,
    originalText: rawText,
    cleanedText: rawText === "Um, um, welcome welcome." ? "Welcome." : rawText,
    simplifiedText: isLiveUpdate ? "Second caption meaning." : baseInterpretResponse.simplifiedText,
    glossSequence: isLiveUpdate
      ? [
          baseInterpretResponse.glossSequence[1],
          {
            id: "gloss_3",
            token: "thank",
            gloss: "THANK",
            isVerified: false,
            fallback: "fingerspell" as const,
            confidence: "low" as const,
            sourceSegmentId: "segment_rolling",
            planningActionId: "action_thank"
          }
        ]
      : baseInterpretResponse.glossSequence,
    avatarQueue: {
      ...baseInterpretResponse.avatarQueue,
      id: `queue_api_${requestNumber}`,
      interpretationId: id,
      steps: isLiveUpdate
        ? [
            baseInterpretResponse.avatarQueue.steps[1],
            {
              id: "step_3",
              glossItemId: "gloss_3",
              kind: "sign" as const,
              label: "THANK",
              motionClipId: "include-thankyou",
              motionSource: "include_dataset" as const,
              expertReviewed: false,
              durationMs: 800,
              confidence: "high" as const,
              sourceSegmentId: "segment_rolling",
              planningActionId: "action_thank"
            }
          ]
        : baseInterpretResponse.avatarQueue.steps
    }
  };
}

function installFetchMock(options: FetchMockOptions = {}) {
  let interpretationRequestCount = 0;
  const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
    const url = String(input);
    if (url.startsWith("chrome-extension://")) {
      const motionClipIds = options.motionClipIds ?? ["include-computer", "include-thankyou"];
      return Response.json({
        clipCount: motionClipIds.length,
        clips: motionClipIds.map((id) => ({
          id,
          label: id.toUpperCase(),
          fps: 25,
          frames: Array.from({ length: 32 }, () => ({
            pose: [],
            leftHand: [],
            rightHand: []
          }))
        }))
      });
    }
    if (url.endsWith("/api/session/start")) {
      if (options.failSessionStart) {
        return Response.json({ error: "offline" }, { status: 503 });
      }
      return Response.json({
        sessionId: "session_1",
        status: "active",
        rawAudioStored: false
      });
    }
    if (url.endsWith("/api/models")) {
      return Response.json({
        models: options.models ?? [baseInterpretResponse.model],
        rawVideoStored: false
      });
    }
    if (url.endsWith("/api/privacy/purge")) {
      if (options.privacyPurgeFailure === "offline") {
        throw new TypeError("Failed to fetch");
      }
      if (options.failPrivacyPurge) {
        return Response.json({ error: "offline" }, { status: 503 });
      }
      return Response.json({
        scope: "all",
        status: "purged",
        deleted: {
          sessions: 1,
          interpretations: 1,
          avatarQueues: 1,
          feedback: 1,
          rateLimitEntries: 1
        },
        rawAudioStored: false,
        rawVideoStored: false
      });
    }
    if (url.endsWith("/api/interpret")) {
      interpretationRequestCount += 1;
      const requestBody = JSON.parse(String(init?.body)) as { rawText: string };
      return Response.json(makeInterpretResponse(requestBody.rawText, interpretationRequestCount));
    }
    if (url.endsWith("/api/feedback")) {
      if (options.feedbackFailure === "offline") {
        throw new TypeError("Failed to fetch");
      }
      if (options.feedbackFailure === "server") {
        return Response.json({ error: "feedback_failed" }, { status: 500 });
      }
      return Response.json({
        feedbackId: "feedback_1",
        status: "saved",
        rawAudioStored: false
      });
    }
    if (url.includes("/api/glossary/search")) {
      if (options.glossaryFailure === "offline") {
        throw new TypeError("Failed to fetch");
      }
      if (options.glossaryFailure === "server") {
        return Response.json({ error: "glossary_failed" }, { status: 500 });
      }
      return Response.json({
        results: [
          {
            id: "glossary_1",
            term: "education",
            aliases: ["learning"],
            language: "en",
            category: "education",
            islGloss: "EDUCATION",
            signAssetId: "sign_education_001",
            confidence: "high",
            source: "expert_reviewed",
            reviewStatus: "approved",
            createdAt: "2026-07-07T00:00:00.000Z",
            updatedAt: "2026-07-07T00:00:00.000Z"
          }
        ]
      });
    }
    if (url.endsWith("/api/session/end")) {
      const requestBody = JSON.parse(String(init?.body)) as { sessionId: string };
      return Response.json({
        sessionId: requestBody.sessionId,
        status: "ended",
        rawAudioStored: false
      });
    }
    return Response.json({}, { status: 404 });
  });
  vi.stubGlobal("fetch", fetchMock);
  return fetchMock;
}

function installChromeMock({
  captionTexts = [],
  captionSnapshots,
  rejectMessageType,
  storageState = {},
  targetTabId = 42,
  url = "https://www.youtube.com/watch?v=demo"
}: SnapshotConfig = {}) {
  let captionIndex = 0;
  const runtimeSendMessage = vi.fn(
    (message: { type: string }, callback?: (response?: unknown) => void) => {
      if (message.type === "GET_PAGE_CAPTION_SNAPSHOT") {
        const snapshot =
          captionSnapshots?.[Math.min(captionIndex, Math.max(0, captionSnapshots.length - 1))];
        const captionText = snapshot
          ? snapshot.captionText
          : captionTexts[Math.min(captionIndex, Math.max(0, captionTexts.length - 1))];
        captionIndex += 1;
        if (snapshot?.error) {
          callback?.({ ok: false, error: snapshot.error });
          return;
        }
        if (snapshot || captionText !== undefined) {
          callback?.({
            ok: true,
            targetTabId,
            snapshot: {
              title: "Lecture video",
              url,
              ...(captionText === undefined ? {} : { captionText }),
              supportStatus: snapshot?.supportStatus ?? "supported"
            }
          });
        } else {
          callback?.({ ok: false, error: "No visible captions." });
        }
        return;
      }
      callback?.({ ok: message.type !== rejectMessageType });
    }
  );
  const setMock = vi.fn((value: StorageState, callback?: () => void) => {
    Object.assign(storageState, value);
    callback?.();
  });
  const clearMock = vi.fn((callback?: () => void) => {
    for (const key of Object.keys(storageState)) {
      delete storageState[key];
    }
    callback?.();
  });

  vi.stubGlobal("chrome", {
    runtime: {
      getURL: vi.fn((path: string) => `chrome-extension://test/${path}`),
      sendMessage: runtimeSendMessage
    },
    storage: {
      local: {
        get: vi.fn((keys: string[], callback: (value: StorageState) => void) => {
          callback(Object.fromEntries(keys.map((key) => [key, storageState[key]])));
        }),
        set: setMock,
        clear: clearMock
      }
    }
  });

  return { clearMock, runtimeSendMessage, setMock, storageState };
}

function getRequests(fetchMock: ReturnType<typeof installFetchMock>, suffix: string) {
  return fetchMock.mock.calls.filter(([input]) => String(input).endsWith(suffix));
}

async function flushAsyncWork(iterations = 16): Promise<void> {
  await act(async () => {
    for (let index = 0; index < iterations; index += 1) {
      await Promise.resolve();
    }
  });
}

describe("SidePanelApp", () => {
  beforeEach(() => {
    document.body.innerHTML = "";
    document.title = "";
    vi.useRealTimers();
    resetMotionClipMapForTests();
    installFetchMock();
    installChromeMock();
  });

  afterEach(() => {
    cleanup();
    resetMotionClipMapForTests();
    vi.useRealTimers();
    vi.unstubAllGlobals();
    vi.restoreAllMocks();
  });

  it("switches between Interpreter, Captions, Summary, and Glossary tabs", async () => {
    render(<SidePanelApp />);
    await flushAsyncWork();

    const interpreterTab = screen.getByRole("tab", { name: "Interpreter" });
    expect(interpreterTab).toHaveAttribute("aria-selected", "true");
    expect(interpreterTab).toHaveAttribute("tabindex", "0");
    expect(screen.getByRole("heading", { name: "Interpreter avatar" })).toBeInTheDocument();

    interpreterTab.focus();
    fireEvent.keyDown(interpreterTab, { key: "ArrowRight" });
    expect(screen.getByRole("tab", { name: "Captions" })).toHaveAttribute("aria-selected", "true");
    expect(screen.getByRole("tab", { name: "Interpreter" })).toHaveAttribute("tabindex", "-1");

    expect(screen.getByRole("tab", { name: "Captions" })).toHaveAttribute("aria-selected", "true");
    expect(screen.getByRole("heading", { name: "Live captions" })).toBeInTheDocument();

    fireEvent.click(screen.getByRole("tab", { name: "Summary" }));
    expect(screen.getByRole("heading", { name: "Simplified meaning" })).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: /Key points/ })).toBeInTheDocument();

    fireEvent.click(screen.getByRole("tab", { name: "Glossary" }));
    expect(screen.getByRole("heading", { name: /Detected terms/ })).toBeInTheDocument();
    expect(screen.queryByRole("heading", { name: "Interpreter avatar" })).not.toBeInTheDocument();
  });

  it("shows the local API connection and active YouTube platform", async () => {
    installChromeMock({ captionTexts: ["Visible caption from YouTube."] });
    render(<SidePanelApp />);
    await flushAsyncWork();

    const statusBar = screen.getByRole("region", { name: /connection and platform status/i });
    expect(within(statusBar).getByText("Connected")).toBeInTheDocument();
    expect(within(statusBar).getByText("None")).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: /start interpretation/i }));
    await waitFor(() => expect(within(statusBar).getByText("YouTube")).toBeInTheDocument());
    expect(within(statusBar).getByText("Live captions")).toBeInTheDocument();
  });

  it("identifies Google Meet honestly as detection-only", async () => {
    installChromeMock({
      captionSnapshots: [{ supportStatus: "unsupported" }],
      url: "https://meet.google.com/abc-defg-hij"
    });
    render(<SidePanelApp />);

    fireEvent.click(screen.getByRole("button", { name: /start interpretation/i }));

    const statusBar = screen.getByRole("region", { name: /connection and platform status/i });
    await waitFor(() => expect(within(statusBar).getByText("Google Meet")).toBeInTheDocument());
    expect(within(statusBar).getByText("Detection only")).toBeInTheDocument();
    expect(
      within(statusBar).getByText("Google Meet").closest(".connection-bar__copy")
    ).toHaveTextContent("PlatformGoogle MeetDetection only");
    expect(
      screen.getByText(/live caption capture is not enabled for this MVP/i)
    ).toBeInTheDocument();
  });

  it("navigates Home, Settings, History, and About while keeping session controls visible", async () => {
    render(<SidePanelApp />);

    fireEvent.click(screen.getByRole("button", { name: "Settings" }));
    expect(screen.getByRole("heading", { name: "Settings" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /start interpretation/i })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Settings" })).toHaveAttribute(
      "aria-current",
      "page"
    );

    fireEvent.click(screen.getByRole("button", { name: "History" }));
    expect(screen.getByRole("heading", { name: "History" })).toBeInTheDocument();
    expect(screen.getByRole("checkbox", { name: /save local history/i })).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "About" }));
    expect(screen.getByRole("heading", { name: "About" })).toBeInTheDocument();
    expect(
      await screen.findByText("ISLRTC official index + INCLUDE motion metadata")
    ).toBeInTheDocument();
    expect(screen.getByText(/Lexicon ranker · v0\.1\.0/i)).toBeInTheDocument();
    expect(screen.queryByText(/Temporal ISL classifier/i)).not.toBeInTheDocument();
    expect(screen.getByText(/not a certified interpreter replacement/i)).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "Home" }));
    fireEvent.click(screen.getByRole("button", { name: /open settings/i }));
    expect(screen.getByRole("heading", { name: "Settings" })).toBeInTheDocument();
  });

  it("keeps unavailable research models visibly unavailable", async () => {
    const unavailableResearchModel = {
      ...baseInterpretResponse.model,
      id: "isign-text-to-pose-research",
      status: "unavailable",
      engine: "text_to_pose_research",
      trainingDataset: {
        ...baseInterpretResponse.model.trainingDataset,
        primaryDataset: "isign",
        displayName: "Private iSign text-to-pose research artifact"
      },
      notes: ["Disabled for live signing until expert review is complete."]
    };
    installFetchMock({ models: [baseInterpretResponse.model, unavailableResearchModel] });
    render(<SidePanelApp />);

    fireEvent.click(screen.getByRole("button", { name: "About" }));
    const modelName = await screen.findByText("Private iSign text-to-pose research artifact");
    const modelItem = modelName.closest("li");

    expect(modelItem).not.toBeNull();
    if (!modelItem) {
      throw new Error("Unavailable research model item was not rendered.");
    }
    expect(within(modelItem).getByText("Unavailable")).toHaveClass("model-state--unavailable");
    expect(within(modelItem).queryByText(/^Experimental$/)).not.toBeInTheDocument();
    expect(screen.getByText("1 ready · 1 unavailable")).toBeInTheDocument();
  });

  it("makes all four avatar choices stateful and keeps the shared motion rig explicit", async () => {
    render(<SidePanelApp />);
    await flushAsyncWork();

    expect(screen.getAllByRole("radio")).toHaveLength(4);
    expect(screen.getByRole("radio", { name: /Ananya avatar option/i })).toHaveAttribute(
      "aria-checked",
      "true"
    );
    expect(screen.getByRole("radio", { name: /Ananya avatar option/i })).toHaveAttribute(
      "tabindex",
      "0"
    );
    expect(screen.getByRole("radio", { name: /Arjun avatar option/i })).toHaveAttribute(
      "tabindex",
      "-1"
    );

    const ananyaRadio = screen.getByRole("radio", { name: /Ananya avatar option/i });
    ananyaRadio.focus();
    fireEvent.keyDown(ananyaRadio, { key: "ArrowRight" });

    expect(screen.getByRole("radio", { name: /Arjun avatar option/i })).toHaveAttribute(
      "aria-checked",
      "true"
    );
    expect(screen.getByRole("radio", { name: /Ananya avatar option/i })).toHaveAttribute(
      "aria-checked",
      "false"
    );
    expect(screen.getByRole("radio", { name: /Arjun avatar option/i })).toHaveAttribute(
      "tabindex",
      "0"
    );
    expect(screen.getByRole("img", { name: /SignSaarthi motion avatar at rest/i })).toHaveAttribute(
      "data-avatar-name",
      "Arjun"
    );
    expect(
      screen.getByText(
        /Arjun theme on dataset-derived isolated-sign motion; not expert-certified ISL/i
      )
    ).toBeInTheDocument();
  });

  it("starts interpretation, keeps confidence and signing fallbacks, and sends simplification level", async () => {
    const fetchMock = vi.mocked(fetch);
    render(<SidePanelApp />);

    fireEvent.click(screen.getByRole("button", { name: /start interpretation/i }));

    await waitFor(() => {
      expect(getRequests(fetchMock, "/api/interpret")).toHaveLength(1);
    });
    expect(screen.getByText("Medium")).toBeInTheDocument();
    expect(screen.getByText("Some terms use fallback output.")).toBeInTheDocument();
    expect(
      screen.getByText(/Glossary: ISLRTC official index \+ INCLUDE motion metadata · 12434 terms/i)
    ).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /stop interpretation/i })).toBeInTheDocument();
    const wordActions = screen.getByRole("region", { name: /all word actions/i });
    expect(within(wordActions).getAllByRole("listitem")).toHaveLength(2);
    expect(within(wordActions).getByText("MUMBAI")).toBeInTheDocument();
    expect(within(wordActions).getByText(/2 ordered · 2 motion · 1 lexicon/i)).toBeInTheDocument();
    expect(
      await within(wordActions).findByText("INCLUDE motion · review pending")
    ).toBeInTheDocument();
    expect(within(wordActions).getByText("Official ISLRTC fingerspelling")).toBeInTheDocument();

    const interpretRequest = getRequests(fetchMock, "/api/interpret")[0];
    expect(JSON.parse(String(interpretRequest?.[1]?.body))).toMatchObject({
      source: "mock",
      simplificationLevel: "standard"
    });
    expect(globalThis.chrome.runtime.sendMessage).toHaveBeenCalledWith(
      expect.objectContaining({
        type: "AVATAR_QUEUE_UPDATED",
        queue: expect.objectContaining({ id: "queue_api_1" })
      }),
      expect.any(Function)
    );

    fireEvent.click(screen.getByRole("tab", { name: "Captions" }));
    expect(screen.getByText("Today we use a computer. Thank you.")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("tab", { name: "Summary" }));
    expect(screen.getByText("AI can help education.")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("tab", { name: "Glossary" }));
    expect(screen.getByRole("button", { name: "education" })).toBeInTheDocument();
  });

  it("labels an advertised but missing motion clip as a static fallback", async () => {
    installFetchMock({ motionClipIds: [] });
    render(<SidePanelApp />);

    fireEvent.click(screen.getByRole("button", { name: /start interpretation/i }));

    const wordActions = await screen.findByRole("region", { name: /all word actions/i });
    expect(
      await within(wordActions).findByText("Static fallback · motion unavailable")
    ).toBeInTheDocument();
    expect(
      within(wordActions).queryByText("INCLUDE motion · review pending")
    ).not.toBeInTheDocument();
    expect(within(wordActions).getByText(/2 ordered · 1 motion · 1 lexicon/i)).toBeInTheDocument();
    expect(document.querySelector(".stage-caption__status--fallback")).toHaveTextContent(
      "Static fallback · motion unavailable"
    );
  });

  it("bounds realtime preview playback while retaining every visible Word Action", async () => {
    installChromeMock({ captionTexts: ["Backlog caption."] });
    render(<SidePanelApp />);

    fireEvent.click(screen.getByRole("button", { name: /start interpretation/i }));
    await waitFor(() =>
      expect(screen.getByRole("button", { name: /stop interpretation/i })).toBeInTheDocument()
    );

    const wordActions = screen.getByRole("region", { name: /all word actions/i });
    expect(wordActions).toHaveAttribute("data-backlog-status", "backlog");
    expect(within(wordActions).getAllByRole("listitem")).toHaveLength(24);
    expect(within(wordActions).getByText(/retained in Word actions/i)).toBeInTheDocument();
    expect(document.querySelector(".sidepanel-motion-avatar")).toHaveAttribute(
      "data-motion-step-count",
      "8"
    );
  });

  it("drains every action through the avatar when side-panel-only mode has 20+ actions", async () => {
    vi.useFakeTimers();
    installChromeMock({
      captionTexts: ["Backlog caption."],
      storageState: {
        "signsaarthi.settings": {
          avatarPosition: "side_panel"
        }
      }
    });
    render(<SidePanelApp />);
    await flushAsyncWork(32);

    fireEvent.click(screen.getByRole("button", { name: /start interpretation/i }));
    await flushAsyncWork(32);

    const wordActions = screen.getByRole("region", { name: /all word actions/i });
    expect(wordActions).toHaveAttribute("data-backlog-status", "side_panel_queue");
    expect(within(wordActions).getAllByRole("listitem")).toHaveLength(24);
    expect(within(wordActions).getByText(/all 24 actions queued/i)).toBeInTheDocument();
    expect(document.querySelector(".sidepanel-motion-avatar")).toHaveAttribute(
      "data-motion-step-count",
      "24"
    );
    expect(document.querySelector(".stage-caption strong")).toHaveTextContent("WORD0");
    expect(screen.getByRole("button", { name: /replay last segment/i })).toBeEnabled();
    expect(document.querySelector("svg[data-motion-avatar]")).toHaveAttribute(
      "data-clip-id",
      "runtime-backlog-0"
    );
    expect(document.querySelector("svg[data-motion-avatar]")).toHaveAttribute(
      "data-state",
      "playing"
    );

    await act(async () => {
      await vi.advanceTimersByTimeAsync(5_000);
    });

    expect(document.querySelector(".stage-caption strong")).toHaveTextContent("WORD23");
  });

  it("routes playback controls to the session tab without restarting the avatar queue", async () => {
    const { runtimeSendMessage } = installChromeMock({
      captionTexts: ["First caption."],
      targetTabId: 73
    });
    render(<SidePanelApp />);
    await waitFor(() =>
      expect(
        within(screen.getByRole("region", { name: /connection and platform status/i })).getByText(
          "Connected"
        )
      ).toBeInTheDocument()
    );

    const confidenceProgress = screen.getByRole("progressbar", {
      name: "Interpretation confidence"
    });
    expect(confidenceProgress).toHaveAttribute("aria-valuemin", "0");
    expect(confidenceProgress).toHaveAttribute("aria-valuemax", "100");
    expect(confidenceProgress).toHaveAttribute("aria-valuenow", "0");
    expect(confidenceProgress).toHaveAttribute("aria-valuetext", "Not started");
    expect(screen.getByRole("button", { name: /^pause$/i })).toBeDisabled();

    fireEvent.click(screen.getByRole("button", { name: /start interpretation/i }));
    await waitFor(() =>
      expect(screen.getByRole("button", { name: /stop interpretation/i })).toBeInTheDocument()
    );
    expect(confidenceProgress).toHaveAttribute("aria-valuenow", "68");
    expect(confidenceProgress).toHaveAttribute("aria-valuetext", "Medium confidence, 68%");

    fireEvent.click(screen.getByRole("button", { name: /^pause$/i }));
    await waitFor(() =>
      expect(runtimeSendMessage).toHaveBeenCalledWith(
        expect.objectContaining({
          type: "AVATAR_PLAYBACK_UPDATED",
          action: "pause",
          sessionId: "session_1",
          targetTabId: 73
        }),
        expect.any(Function)
      )
    );
    fireEvent.click(screen.getByRole("button", { name: /replay last segment/i }));
    fireEvent.click(screen.getByRole("button", { name: /adjust signing pace/i }));

    expect(runtimeSendMessage).toHaveBeenCalledWith(
      expect.objectContaining({
        type: "SESSION_STARTED",
        sessionId: "session_1",
        targetTabId: 73
      }),
      expect.any(Function)
    );
    expect(runtimeSendMessage).toHaveBeenCalledWith(
      expect.objectContaining({
        type: "AVATAR_PLAYBACK_UPDATED",
        action: "pause",
        sessionId: "session_1",
        targetTabId: 73
      }),
      expect.any(Function)
    );
    expect(runtimeSendMessage).toHaveBeenCalledWith(
      expect.objectContaining({
        type: "AVATAR_PLAYBACK_UPDATED",
        action: "replay",
        targetTabId: 73
      }),
      expect.any(Function)
    );
    expect(screen.getByRole("heading", { name: "Settings" })).toBeInTheDocument();
    expect(screen.getByRole("slider", { name: /signing pace/i })).toHaveAttribute(
      "aria-valuetext",
      "Slow, 0.75x"
    );
  });

  it("keeps the session usable and reports when overlay delivery is rejected", async () => {
    installChromeMock({
      captionTexts: ["First caption."],
      rejectMessageType: "AVATAR_QUEUE_UPDATED"
    });
    render(<SidePanelApp />);

    fireEvent.click(screen.getByRole("button", { name: /start interpretation/i }));

    expect(await screen.findByRole("alert")).toHaveTextContent(
      /YouTube avatar overlay could not be reached/i
    );
    expect(screen.getByRole("button", { name: /stop interpretation/i })).toBeInTheDocument();
  });

  it("does not let a delayed interpretation response restore stale overlay settings", async () => {
    const baseFetch = vi.mocked(fetch);
    let resolveInterpretation: (() => void) | undefined;
    const deferredFetch = vi.fn((input: RequestInfo | URL, init?: RequestInit) => {
      if (String(input).endsWith("/api/interpret")) {
        return new Promise<Response>((resolve) => {
          const body = JSON.parse(String(init?.body)) as { rawText: string };
          resolveInterpretation = () =>
            resolve(Response.json(makeInterpretResponse(body.rawText, 1)));
        });
      }
      return baseFetch(input, init);
    });
    vi.stubGlobal("fetch", deferredFetch);
    const { runtimeSendMessage } = installChromeMock();
    render(<SidePanelApp />);

    fireEvent.click(screen.getByRole("button", { name: /start interpretation/i }));
    await waitFor(() => expect(resolveInterpretation).toBeTypeOf("function"));
    fireEvent.click(screen.getByRole("button", { name: /open settings/i }));
    fireEvent.change(screen.getByLabelText(/output mode/i), { target: { value: "captions_only" } });

    await act(async () => resolveInterpretation?.());
    await waitFor(() =>
      expect(screen.getByRole("button", { name: /stop interpretation/i })).toBeInTheDocument()
    );

    const queueMessages = runtimeSendMessage.mock.calls.filter(
      ([message]) => message.type === "AVATAR_QUEUE_UPDATED"
    );
    expect(queueMessages).toHaveLength(0);
    const settingsMessages = runtimeSendMessage.mock.calls
      .filter(([message]) => message.type === "OVERLAY_SETTINGS_UPDATED")
      .map(([message]) => message);
    expect(settingsMessages.at(-1)).toMatchObject({ settings: { outputMode: "captions_only" } });
  });

  it("submits validated wrong-sign feedback and preserves accessible dialog behavior", async () => {
    const fetchMock = vi.mocked(fetch);
    render(<SidePanelApp />);

    fireEvent.click(screen.getByRole("button", { name: /start interpretation/i }));
    await waitFor(() => expect(getRequests(fetchMock, "/api/interpret")).toHaveLength(1));

    const reportButton = screen.getByRole("button", { name: /report wrong sign/i });
    reportButton.focus();
    fireEvent.click(reportButton);
    expect(screen.getByRole("dialog", { name: /report wrong sign/i })).toBeInTheDocument();
    await waitFor(() => expect(screen.getByLabelText(/reason/i)).toHaveFocus());
    fireEvent.keyDown(document, { key: "Escape" });
    expect(screen.queryByRole("dialog", { name: /report wrong sign/i })).not.toBeInTheDocument();
    expect(reportButton).toHaveFocus();

    fireEvent.click(reportButton);
    fireEvent.change(screen.getByLabelText(/reason/i), { target: { value: "missing_sign" } });
    fireEvent.change(screen.getByLabelText(/comment/i), {
      target: { value: "The Mumbai fallback needs review." }
    });
    fireEvent.click(screen.getByRole("button", { name: /submit feedback/i }));

    await screen.findByText("Feedback saved for reviewer follow-up.");
    const feedbackRequest = getRequests(fetchMock, "/api/feedback")[0];
    expect(JSON.parse(String(feedbackRequest?.[1]?.body))).toMatchObject({
      reason: "missing_sign",
      userComment: "The Mumbai fallback needs review."
    });
  });

  it("shows a service error rather than offline when feedback receives an HTTP failure", async () => {
    const fetchMock = installFetchMock({ feedbackFailure: "server" });
    render(<SidePanelApp />);

    fireEvent.click(screen.getByRole("button", { name: /start interpretation/i }));
    await waitFor(() => expect(getRequests(fetchMock, "/api/interpret")).toHaveLength(1));
    fireEvent.click(screen.getByRole("button", { name: /report wrong sign/i }));
    fireEvent.change(screen.getByLabelText(/reason/i), { target: { value: "missing_sign" } });
    fireEvent.click(screen.getByRole("button", { name: /submit feedback/i }));

    expect(
      await screen.findByText("Feedback was not saved. Local API request failed (500).")
    ).toBeInTheDocument();
    const statusBar = screen.getByRole("region", { name: /connection and platform status/i });
    expect(within(statusBar).getByText("Service error")).toBeInTheDocument();
    expect(within(statusBar).queryByText("Offline")).not.toBeInTheDocument();
  });

  it("persists settings, applies output gates, pauses captions, and searches the glossary", async () => {
    const { setMock } = installChromeMock();
    const fetchMock = vi.mocked(fetch);
    render(<SidePanelApp />);

    fireEvent.click(screen.getByRole("button", { name: /open settings/i }));
    fireEvent.change(screen.getByLabelText(/output mode/i), { target: { value: "captions_only" } });
    fireEvent.change(screen.getByLabelText(/avatar position/i), {
      target: { value: "side_panel" }
    });
    fireEvent.change(screen.getByLabelText(/avatar size/i), { target: { value: "large" } });
    fireEvent.change(screen.getByLabelText(/caption language/i), { target: { value: "hi" } });
    fireEvent.change(screen.getByLabelText(/simplification/i), { target: { value: "strong" } });
    fireEvent.click(screen.getByRole("checkbox", { name: /feedback sharing/i }));
    fireEvent.click(screen.getByRole("checkbox", { name: /gloss debug/i }));

    fireEvent.click(screen.getByRole("button", { name: "Home" }));
    fireEvent.click(screen.getByRole("tab", { name: "Captions" }));
    fireEvent.click(screen.getByRole("button", { name: /increase caption text size/i }));
    fireEvent.click(screen.getByRole("button", { name: /start interpretation/i }));

    await waitFor(() => expect(getRequests(fetchMock, "/api/interpret")).toHaveLength(1));
    fireEvent.click(screen.getByRole("button", { name: /pause captions/i }));
    expect(screen.getByText("Captions paused.")).toBeInTheDocument();
    expect(screen.getByText("Hindi")).toBeInTheDocument();
    expect(
      JSON.parse(String(getRequests(fetchMock, "/api/interpret")[0]?.[1]?.body))
    ).toMatchObject({
      simplificationLevel: "strong"
    });

    fireEvent.click(screen.getByRole("tab", { name: "Interpreter" }));
    expect(screen.queryByText(/Signing plan/i)).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: /replay last segment/i })).toBeDisabled();
    expect(screen.getByRole("button", { name: /report wrong sign/i })).toBeDisabled();

    fireEvent.click(screen.getByRole("tab", { name: "Glossary" }));
    fireEvent.click(screen.getByRole("button", { name: "education" }));
    expect(await screen.findByText("EDUCATION")).toBeInTheDocument();
    expect(setMock).toHaveBeenCalledWith(
      expect.objectContaining({ "signsaarthi.settings": expect.any(Object) }),
      expect.any(Function)
    );
  });

  it("marks the API offline only when a glossary request cannot connect", async () => {
    const fetchMock = installFetchMock({ glossaryFailure: "offline" });
    render(<SidePanelApp />);

    fireEvent.click(screen.getByRole("button", { name: /start interpretation/i }));
    await waitFor(() => expect(getRequests(fetchMock, "/api/interpret")).toHaveLength(1));
    fireEvent.click(screen.getByRole("tab", { name: "Glossary" }));
    fireEvent.click(screen.getByRole("button", { name: "education" }));

    expect(
      await screen.findByText("Glossary unavailable while the local API is offline.")
    ).toBeInTheDocument();
    const statusBar = screen.getByRole("region", { name: /connection and platform status/i });
    expect(within(statusBar).getByText("Offline")).toBeInTheDocument();
  });

  it("persists summary-only local history when enabled", async () => {
    const storageState: StorageState = {
      "signsaarthi.settings": {
        avatarSpeed: 0.75,
        saveHistory: true
      }
    };
    const { setMock } = installChromeMock({ storageState });
    render(<SidePanelApp />);

    fireEvent.click(screen.getByRole("button", { name: "History" }));
    await waitFor(() =>
      expect(screen.getByRole("checkbox", { name: /save local history/i })).toBeChecked()
    );
    fireEvent.click(screen.getByRole("button", { name: /start interpretation/i }));

    await waitFor(() => {
      expect(setMock).toHaveBeenCalledWith(
        expect.objectContaining({
          "signsaarthi.sessionHistory": expect.arrayContaining([
            expect.objectContaining({
              summary: "AI can help education.",
              confidence: "medium"
            })
          ])
        }),
        expect.any(Function)
      );
    });
  });

  it("deletes local data, resets in-memory settings, and does not immediately recreate history", async () => {
    const storageState: StorageState = {
      "signsaarthi.settings": {
        saveHistory: true,
        avatarSpeed: 1.5
      }
    };
    const { clearMock, runtimeSendMessage, setMock } = installChromeMock({ storageState });
    const fetchMock = vi.mocked(fetch);
    render(<SidePanelApp />);

    fireEvent.click(screen.getByRole("button", { name: "History" }));
    await waitFor(() =>
      expect(screen.getByRole("checkbox", { name: /save local history/i })).toBeChecked()
    );
    fireEvent.click(screen.getByRole("button", { name: /delete local data/i }));

    await screen.findByText("Local extension and API data deleted.");
    expect(clearMock).toHaveBeenCalledTimes(1);
    const purgeRequest = getRequests(fetchMock, "/api/privacy/purge")[0];
    expect(JSON.parse(String(purgeRequest?.[1]?.body))).toEqual({
      scope: "all",
      confirmation: "purge_all_local_data"
    });
    expect(screen.getByRole("checkbox", { name: /save local history/i })).not.toBeChecked();
    expect(runtimeSendMessage).toHaveBeenCalledWith(
      expect.objectContaining({
        type: "OVERLAY_SETTINGS_UPDATED",
        settings: expect.objectContaining({ saveHistory: false, avatarSpeed: 0.75 })
      }),
      expect.any(Function)
    );

    setMock.mockClear();
    fireEvent.click(screen.getByRole("button", { name: /start interpretation/i }));
    await waitFor(() => expect(getRequests(fetchMock, "/api/interpret")).toHaveLength(1));
    expect(
      setMock.mock.calls.some(([value]) =>
        Object.hasOwn(value as object, "signsaarthi.sessionHistory")
      )
    ).toBe(false);
  });

  it("keeps extension deletion successful and reports a purge service error", async () => {
    installFetchMock({ failPrivacyPurge: true });
    const { clearMock } = installChromeMock();
    render(<SidePanelApp />);

    fireEvent.click(screen.getByRole("button", { name: "History" }));
    fireEvent.click(screen.getByRole("button", { name: /delete local data/i }));

    await screen.findByText(
      "Extension data deleted. API data was not deleted. Local API request failed (503)."
    );
    expect(clearMock).toHaveBeenCalledTimes(1);
    const statusBar = screen.getByRole("region", { name: /connection and platform status/i });
    expect(within(statusBar).getByText("Service error")).toBeInTheDocument();
  });

  it("marks the API offline when privacy purge cannot establish a connection", async () => {
    installFetchMock({ privacyPurgeFailure: "offline" });
    const { clearMock } = installChromeMock();
    render(<SidePanelApp />);

    fireEvent.click(screen.getByRole("button", { name: "History" }));
    fireEvent.click(screen.getByRole("button", { name: /delete local data/i }));

    await screen.findByText("Extension data deleted. Local API data could not be reached.");
    expect(clearMock).toHaveBeenCalledTimes(1);
    const statusBar = screen.getByRole("region", { name: /connection and platform status/i });
    expect(within(statusBar).getByText("Offline")).toBeInTheDocument();
  });

  it("shows captured captions locally before a session API service error", async () => {
    const fetchMock = installFetchMock({ failSessionStart: true });
    installChromeMock({ captionTexts: ["Visible caption from YouTube."] });
    render(<SidePanelApp />);

    fireEvent.click(screen.getByRole("tab", { name: "Captions" }));
    fireEvent.click(screen.getByRole("button", { name: /start interpretation/i }));

    expect(await screen.findByText("Visible caption from YouTube.")).toBeInTheDocument();
    expect(await screen.findByText("Local API request failed (503).")).toBeInTheDocument();
    expect(screen.getByText("YouTube captions")).toBeInTheDocument();
    expect(
      within(screen.getByRole("region", { name: /connection and platform status/i })).getByText(
        "Service error"
      )
    ).toBeInTheDocument();
    expect(getRequests(fetchMock, "/api/interpret")).toHaveLength(0);
    expect(
      JSON.parse(String(getRequests(fetchMock, "/api/session/start")[0]?.[1]?.body))
    ).toMatchObject({
      source: "youtube_captions"
    });
  });

  it("shows raw live caption text even when interpretation cleaning removes filler", async () => {
    const fetchMock = installFetchMock();
    installChromeMock({ captionTexts: ["Um, um, welcome welcome."] });
    render(<SidePanelApp />);

    fireEvent.click(screen.getByRole("tab", { name: "Captions" }));
    fireEvent.click(screen.getByRole("button", { name: /start interpretation/i }));

    await waitFor(() => expect(getRequests(fetchMock, "/api/interpret")).toHaveLength(1));
    expect(screen.getByText("Um, um, welcome welcome.")).toBeInTheDocument();
    expect(screen.queryByText("Welcome.")).not.toBeInTheDocument();
    expect(
      JSON.parse(String(getRequests(fetchMock, "/api/interpret")[0]?.[1]?.body))
    ).toMatchObject({ rawText: "Um, um, welcome welcome." });
  });

  it("plans only appended words from growing rolling captions while displaying the full raw snapshot", async () => {
    vi.useFakeTimers();
    const fetchMock = installFetchMock();
    installChromeMock({ captionTexts: ["Welcome to", "Welcome to SignSaarthi"] });
    render(<SidePanelApp />);

    fireEvent.click(screen.getByRole("button", { name: /start interpretation/i }));
    await flushAsyncWork();
    await act(async () => {
      await vi.advanceTimersByTimeAsync(LIVE_CAPTION_POLL_INTERVAL_MS);
    });

    const interpretBodies = getRequests(fetchMock, "/api/interpret").map(([, init]) =>
      JSON.parse(String(init?.body))
    );
    expect(interpretBodies).toEqual([
      expect.objectContaining({ rawText: "Welcome to" }),
      expect.objectContaining({ rawText: "SignSaarthi" })
    ]);
    fireEvent.click(screen.getByRole("tab", { name: "Captions" }));
    expect(screen.getByText("Welcome to SignSaarthi")).toBeInTheDocument();
  });

  it("recovers polling and binds the tab after a transient initial messaging failure", async () => {
    vi.useFakeTimers();
    const fetchMock = installFetchMock();
    const { runtimeSendMessage } = installChromeMock({
      captionSnapshots: [
        { error: "Receiving end does not exist." },
        { captionText: "Recovered caption.", supportStatus: "supported" }
      ],
      targetTabId: 91
    });
    render(<SidePanelApp />);

    fireEvent.click(screen.getByRole("button", { name: /start interpretation/i }));
    await flushAsyncWork();
    expect(getRequests(fetchMock, "/api/interpret")).toHaveLength(1);
    expect(screen.getByText(/keeps checking this video/i)).toBeInTheDocument();

    await act(async () => {
      await vi.advanceTimersByTimeAsync(LIVE_CAPTION_POLL_INTERVAL_MS);
    });

    const interpretBodies = getRequests(fetchMock, "/api/interpret").map(([, init]) =>
      JSON.parse(String(init?.body))
    );
    expect(interpretBodies).toEqual([
      expect.objectContaining({ source: "mock", rawText: "Today we use a computer. Thank you." }),
      expect.objectContaining({ source: "youtube_captions", rawText: "Recovered caption." })
    ]);
    expect(runtimeSendMessage).toHaveBeenCalledWith(
      expect.objectContaining({ type: "AVATAR_QUEUE_UPDATED", targetTabId: 91 }),
      expect.any(Function)
    );
    fireEvent.click(screen.getByRole("tab", { name: "Captions" }));
    expect(screen.getByText("Recovered caption.")).toBeInTheDocument();
    expect(screen.getByText("YouTube captions")).toBeInTheDocument();
  });

  it("clears and downgrades stale live captions after sustained caption loss", async () => {
    vi.useFakeTimers();
    const fetchMock = installFetchMock();
    installChromeMock({
      captionSnapshots: [
        { captionText: "Current live words.", supportStatus: "supported" },
        ...Array.from({ length: LIVE_CAPTION_LOSS_POLL_COUNT }, () => ({
          supportStatus: "captions_unavailable" as const
        }))
      ]
    });
    render(<SidePanelApp />);

    fireEvent.click(screen.getByRole("tab", { name: "Captions" }));
    fireEvent.click(screen.getByRole("button", { name: /start interpretation/i }));
    await flushAsyncWork();
    expect(screen.getByText("Current live words.")).toBeInTheDocument();

    for (let poll = 0; poll < LIVE_CAPTION_LOSS_POLL_COUNT; poll += 1) {
      await act(async () => {
        await vi.advanceTimersByTimeAsync(LIVE_CAPTION_POLL_INTERVAL_MS);
      });
    }

    expect(getRequests(fetchMock, "/api/interpret")).toHaveLength(1);
    expect(screen.queryByText("Current live words.")).not.toBeInTheDocument();
    expect(screen.getAllByText("Captions unavailable")).not.toHaveLength(0);
    expect(screen.getByText(/Live captions are no longer visible/i)).toBeInTheDocument();
    expect(screen.getByText("Start interpretation to read visible captions.")).toBeInTheDocument();
  });

  it("clears transcript, interpretation, word history, and pause state across stop and restart", async () => {
    vi.useFakeTimers();
    const fetchMock = installFetchMock();
    installChromeMock({ captionTexts: ["First caption.", "Second caption."] });
    render(<SidePanelApp />);

    fireEvent.click(screen.getByRole("button", { name: /start interpretation/i }));
    await flushAsyncWork();
    fireEvent.click(screen.getByRole("tab", { name: "Captions" }));
    expect(screen.getByText("First caption.")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("tab", { name: "Interpreter" }));
    fireEvent.click(screen.getByRole("button", { name: /^pause$/i }));
    await flushAsyncWork();
    expect(screen.getByRole("button", { name: /^resume$/i })).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: /stop interpretation/i }));
    await flushAsyncWork();

    fireEvent.click(screen.getByRole("tab", { name: "Captions" }));
    expect(screen.queryByText("First caption.")).not.toBeInTheDocument();
    expect(screen.getByText("Start interpretation to read visible captions.")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("tab", { name: "Interpreter" }));
    expect(screen.getByRole("progressbar", { name: "Interpretation confidence" })).toHaveAttribute(
      "aria-valuenow",
      "0"
    );
    expect(screen.queryByRole("region", { name: /all word actions/i })).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: /^pause$/i })).toBeDisabled();

    fireEvent.click(screen.getByRole("button", { name: /start interpretation/i }));
    await flushAsyncWork();
    expect(getRequests(fetchMock, "/api/interpret")).toHaveLength(2);
    fireEvent.click(screen.getByRole("tab", { name: "Captions" }));
    expect(screen.getByText("Second caption.")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /pause captions/i })).toBeInTheDocument();
  });

  it("polls live captions, dedupes unchanged text, applies changed results, and stops cleanly", async () => {
    vi.useFakeTimers();
    const fetchMock = installFetchMock();
    const { runtimeSendMessage } = installChromeMock({
      captionTexts: ["First caption.", "  First   caption.  ", "Second caption.", "Second caption."]
    });
    render(<SidePanelApp />);

    fireEvent.click(screen.getByRole("button", { name: /start interpretation/i }));
    await flushAsyncWork();
    expect(getRequests(fetchMock, "/api/interpret")).toHaveLength(1);

    await act(async () => {
      await vi.advanceTimersByTimeAsync(LIVE_CAPTION_POLL_INTERVAL_MS);
    });
    expect(getRequests(fetchMock, "/api/interpret")).toHaveLength(1);

    await act(async () => {
      await vi.advanceTimersByTimeAsync(LIVE_CAPTION_POLL_INTERVAL_MS);
    });
    expect(getRequests(fetchMock, "/api/interpret")).toHaveLength(2);
    const interpretBodies = getRequests(fetchMock, "/api/interpret").map(([, init]) =>
      JSON.parse(String(init?.body))
    );
    expect(interpretBodies).toEqual([
      expect.objectContaining({ rawText: "First caption.", simplificationLevel: "standard" }),
      expect.objectContaining({ rawText: "Second caption.", simplificationLevel: "standard" })
    ]);

    expect(document.querySelector(".sidepanel-motion-avatar")).toHaveAttribute(
      "data-motion-step-count",
      "3"
    );
    expect(
      within(screen.getByRole("region", { name: /all word actions/i })).getAllByRole("listitem")
    ).toHaveLength(3);
    fireEvent.click(screen.getByRole("tab", { name: "Captions" }));
    expect(screen.getByText("Second caption.")).toBeInTheDocument();
    expect(runtimeSendMessage).toHaveBeenCalledWith(
      expect.objectContaining({
        type: "AVATAR_QUEUE_UPDATED",
        queue: expect.objectContaining({ id: "queue_api_2" })
      }),
      expect.any(Function)
    );
    const captionRequestsBeforeStop = runtimeSendMessage.mock.calls.filter(
      ([message]) => message.type === "GET_PAGE_CAPTION_SNAPSHOT"
    ).length;
    fireEvent.click(screen.getByRole("button", { name: /stop interpretation/i }));
    await flushAsyncWork();
    const endRequest = getRequests(fetchMock, "/api/session/end").at(-1);
    expect(JSON.parse(String(endRequest?.[1]?.body))).toMatchObject({
      sessionId: "session_1",
      durationMs: expect.any(Number)
    });

    await act(async () => {
      await vi.advanceTimersByTimeAsync(LIVE_CAPTION_POLL_INTERVAL_MS * 2);
    });
    expect(
      runtimeSendMessage.mock.calls.filter(
        ([message]) => message.type === "GET_PAGE_CAPTION_SNAPSHOT"
      )
    ).toHaveLength(captionRequestsBeforeStop);
    expect(getRequests(fetchMock, "/api/interpret")).toHaveLength(2);
  });

  it("keeps polling a YouTube tab when captions are unavailable at startup", async () => {
    vi.useFakeTimers();
    const fetchMock = installFetchMock();
    installChromeMock({
      captionSnapshots: [
        { supportStatus: "captions_unavailable" },
        { captionText: "Late caption.", supportStatus: "supported" }
      ],
      targetTabId: 86
    });
    render(<SidePanelApp />);

    fireEvent.click(screen.getByRole("button", { name: /start interpretation/i }));
    await flushAsyncWork();
    expect(getRequests(fetchMock, "/api/interpret")).toHaveLength(1);
    expect(screen.getByText(/keeps checking this video/i)).toBeInTheDocument();

    await act(async () => {
      await vi.advanceTimersByTimeAsync(LIVE_CAPTION_POLL_INTERVAL_MS);
    });

    expect(getRequests(fetchMock, "/api/interpret")).toHaveLength(2);
    expect(
      JSON.parse(String(getRequests(fetchMock, "/api/interpret")[1]?.[1]?.body))
    ).toMatchObject({
      rawText: "Late caption.",
      source: "youtube_captions"
    });
    expect(screen.getByText("YouTube captions")).toBeInTheDocument();
  });

  it("keeps the demo fallback one-shot outside a supported YouTube tab", async () => {
    const fetchMock = vi.mocked(fetch);
    const { runtimeSendMessage } = installChromeMock();
    render(<SidePanelApp />);

    fireEvent.click(screen.getByRole("button", { name: /start interpretation/i }));
    await waitFor(() => expect(getRequests(fetchMock, "/api/interpret")).toHaveLength(1));

    expect(
      runtimeSendMessage.mock.calls.filter(
        ([message]) => message.type === "GET_PAGE_CAPTION_SNAPSHOT"
      )
    ).toHaveLength(1);
    expect(
      JSON.parse(String(getRequests(fetchMock, "/api/interpret")[0]?.[1]?.body))
    ).toMatchObject({
      source: "mock",
      rawText: "Today we use a computer. Thank you."
    });
  });
});
