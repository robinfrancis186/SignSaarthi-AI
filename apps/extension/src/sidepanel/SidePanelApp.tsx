import { mergeRollingAvatarQueues, planRealtimePlayback } from "@signsaarthi/avatar-engine";
import { settingsSchema } from "@signsaarthi/shared";
import type {
  FeedbackReport,
  GlossaryEntry,
  ISLGlossItem,
  ISLInterpretation,
  ISLInterpretationResponse,
  ModelMetadata,
  OutputMode,
  Settings,
  TranscriptSegment,
  TranscriptSource
} from "@signsaarthi/shared";
import {
  AlertTriangle,
  BookOpenText,
  Captions,
  CircleHelp,
  Flag,
  Gauge,
  History as HistoryIcon,
  Home,
  Laptop,
  LoaderCircle,
  Pause,
  Play,
  RefreshCw,
  RotateCcw,
  Settings2,
  ShieldCheck,
  Sparkles,
  Trash2,
  UserRound,
  Video,
  Wifi,
  WifiOff,
  Youtube
} from "lucide-react";
import type { LucideIcon } from "lucide-react";
import type { ReactElement } from "react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { createMockTranscriptSegment } from "../adapters/mockTranscriptAdapter";
import {
  getRollingCaptionDelta,
  readYouTubeCaptionSnapshot
} from "../adapters/youtubeCaptionAdapter";
import ananyaAvatarUrl from "../assets/avatars/avatar-ananya.png";
import arjunAvatarUrl from "../assets/avatars/avatar-arjun.png";
import kabirAvatarUrl from "../assets/avatars/avatar-kabir.png";
import meeraAvatarUrl from "../assets/avatars/avatar-meera.png";
import {
  endSession,
  getModelRegistry,
  interpretSegment,
  purgePrivacyData,
  searchGlossary,
  startSession,
  submitFeedback
} from "../lib/apiClient";
import {
  requestPageCaptionSnapshotAttempt,
  sendRuntimeMessage,
  type RuntimeMessage
} from "../lib/runtime";
import {
  appendSessionHistory,
  clearLocalData,
  loadSessionHistory,
  loadSettings,
  saveSettings,
  type SessionHistoryItem
} from "../lib/storage";
import { loadMotionClipMap } from "../lib/motionLibraryLoader";
import { BrandMark } from "./components/BrandMark";
import { FeedbackDialog } from "./components/FeedbackDialog";
import { MotionAvatarPreview } from "./components/MotionAvatarPreview";

type SessionState = "ready" | "active" | "processing" | "error";
type DashboardSection = "interpreter" | "captions" | "summary" | "glossary";
type PrimaryView = "home" | "settings" | "history" | "about";
type AvatarName = "Ananya" | "Arjun" | "Meera" | "Kabir";
type ConnectionStatus = "checking" | "connected" | "degraded" | "offline";
type MotionLibraryState = "loading" | "ready" | "error";
type ActivePlatform = {
  kind: "none" | "youtube" | "meet" | "zoom" | "unsupported" | "demo";
  label: string;
  detail: string;
};
type InterpretRequest = Parameters<typeof interpretSegment>[0] & {
  simplificationLevel: Settings["simplificationLevel"];
};
type WordActionRecord = {
  interpretationId: string;
  item: ISLGlossItem;
};

const DEFAULT_SETTINGS = settingsSchema.parse({});
const DEMO_TRANSCRIPT = "Today we use a computer. Thank you.";
const OVERLAY_DELIVERY_ERROR = "The YouTube avatar overlay could not be reached.";
const NO_ACTIVE_PLATFORM: ActivePlatform = {
  kind: "none",
  label: "None",
  detail: "Session not started"
};

export const LIVE_CAPTION_POLL_INTERVAL_MS = 1500;
export const LIVE_CAPTION_LOSS_POLL_COUNT = 3;

const avatarOptions: Array<{
  name: AvatarName;
  thumbnailUrl: string;
  stageNote: string;
}> = [
  {
    name: "Ananya",
    thumbnailUrl: ananyaAvatarUrl,
    stageNote: "Ananya theme on dataset-derived isolated-sign motion; not expert-certified ISL."
  },
  {
    name: "Arjun",
    thumbnailUrl: arjunAvatarUrl,
    stageNote: "Arjun theme on dataset-derived isolated-sign motion; not expert-certified ISL."
  },
  {
    name: "Meera",
    thumbnailUrl: meeraAvatarUrl,
    stageNote: "Meera theme on dataset-derived isolated-sign motion; not expert-certified ISL."
  },
  {
    name: "Kabir",
    thumbnailUrl: kabirAvatarUrl,
    stageNote: "Kabir theme on dataset-derived isolated-sign motion; not expert-certified ISL."
  }
];

const dashboardSections: Array<{ id: DashboardSection; label: string; icon: LucideIcon }> = [
  { id: "interpreter", label: "Interpreter", icon: Sparkles },
  { id: "captions", label: "Captions", icon: Captions },
  { id: "summary", label: "Summary", icon: BookOpenText },
  { id: "glossary", label: "Glossary", icon: CircleHelp }
];

const primaryViews: Array<{ id: PrimaryView; label: string; icon: LucideIcon }> = [
  { id: "home", label: "Home", icon: Home },
  { id: "settings", label: "Settings", icon: Settings2 },
  { id: "history", label: "History", icon: HistoryIcon },
  { id: "about", label: "About", icon: CircleHelp }
];

export function SidePanelApp(): ReactElement {
  const [status, setStatus] = useState<SessionState>("ready");
  const [sessionId, setSessionId] = useState<string>();
  const [segment, setSegment] = useState<TranscriptSegment>();
  const [liveCaptionText, setLiveCaptionText] = useState<string>();
  const [interpretation, setInterpretation] = useState<ISLInterpretation>();
  const [avatarQueue, setAvatarQueue] = useState<ISLInterpretationResponse["avatarQueue"]>();
  const [wordActionHistory, setWordActionHistory] = useState<WordActionRecord[]>([]);
  const [modelStatus, setModelStatus] = useState<ISLInterpretationResponse["model"]>();
  const [modelRegistry, setModelRegistry] = useState<ModelMetadata[]>([]);
  const [modelRegistryStatus, setModelRegistryStatus] = useState("Checking local models");
  const [bundledMotionClipIds, setBundledMotionClipIds] = useState<ReadonlySet<string>>(
    () => new Set()
  );
  const [motionLibraryState, setMotionLibraryState] = useState<MotionLibraryState>("loading");
  const [connectionStatus, setConnectionStatus] = useState<ConnectionStatus>("checking");
  const [activePlatform, setActivePlatform] = useState<ActivePlatform>(NO_ACTIVE_PLATFORM);
  const [modelBackedGlossCount, setModelBackedGlossCount] = useState(0);
  const [previewReplayKey, setPreviewReplayKey] = useState(0);
  const [previewStepIndex, setPreviewStepIndex] = useState(0);
  const [settings, setSettings] = useState<Settings>();
  const [error, setError] = useState<string>();
  const [sourceNotice, setSourceNotice] = useState<string>();
  const [feedbackOpen, setFeedbackOpen] = useState(false);
  const [feedbackStatus, setFeedbackStatus] = useState<string>();
  const [isSubmittingFeedback, setIsSubmittingFeedback] = useState(false);
  const [sourceLabel, setSourceLabel] = useState("Demo transcript");
  const [sessionHistory, setSessionHistory] = useState<SessionHistoryItem[]>([]);
  const [isCaptionPaused, setIsCaptionPaused] = useState(false);
  const [glossaryEntries, setGlossaryEntries] = useState<GlossaryEntry[]>([]);
  const [glossaryStatus, setGlossaryStatus] = useState(
    "Select a detected term to review glossary details."
  );
  const [activeView, setActiveView] = useState<PrimaryView>("home");
  const [activeDashboardSection, setActiveDashboardSection] =
    useState<DashboardSection>("interpreter");
  const [selectedAvatarName, setSelectedAvatarName] = useState<AvatarName>("Ananya");

  const settingsRef = useRef<Settings | undefined>(undefined);
  const mountedRef = useRef(true);
  const sessionIdRef = useRef<string | undefined>(undefined);
  const sessionTargetTabIdRef = useRef<number | undefined>(undefined);
  const sessionStartedAtRef = useRef<number | undefined>(undefined);
  const sessionLifecycleRef = useRef(0);
  const captionPollingGenerationRef = useRef(0);
  const connectionStatusVersionRef = useRef(0);
  const captionPollingTimerRef = useRef<ReturnType<typeof globalThis.setTimeout> | undefined>(
    undefined
  );
  const lastCaptionTextRef = useRef("");
  const missingCaptionPollCountRef = useRef(0);
  const hasSeenLiveCaptionRef = useRef(false);
  const historyWriteRef = useRef<Promise<void>>(Promise.resolve());

  const resolvedSettings = settings ?? DEFAULT_SETTINGS;
  const selectedAvatar =
    avatarOptions.find((avatar) => avatar.name === selectedAvatarName) ?? avatarOptions[0]!;
  const confidencePercent = getConfidencePercent(interpretation);
  const isRunning = status === "active" || status === "processing";
  const showCaptions =
    resolvedSettings.outputMode === "captions_avatar" ||
    resolvedSettings.outputMode === "captions_only";
  const showSummary = resolvedSettings.outputMode !== "avatar_only";
  const showMeaningTools = resolvedSettings.outputMode !== "avatar_only";
  const showSigningPlan = resolvedSettings.outputMode !== "captions_only";
  const showAvatarOverlay = shouldShowAvatarOverlay(resolvedSettings);
  const sidePanelOnlyPlayback = shouldUseCompleteSidePanelPlayback(resolvedSettings);
  const realtimePlayback = useMemo(
    () =>
      avatarQueue
        ? planRealtimePlayback(avatarQueue.steps, {
            baseSpeed: avatarQueue.speed,
            historyStepCount: avatarQueue.steps.length
          })
        : undefined,
    [avatarQueue]
  );
  const previewSteps =
    resolvedSettings.outputMode === "captions_only"
      ? []
      : sidePanelOnlyPlayback
        ? (avatarQueue?.steps ?? [])
        : (realtimePlayback?.playbackSteps ?? []);
  const canReplayPreview = previewSteps.length > 0;
  const previewStep = previewSteps[previewStepIndex];
  const availableMotionClipIds = useMemo(() => {
    const clipIds = new Set(bundledMotionClipIds);
    for (const clip of avatarQueue?.motionClips ?? []) {
      clipIds.add(clip.id);
    }
    return clipIds as ReadonlySet<string>;
  }, [avatarQueue?.motionClips, bundledMotionClipIds]);
  const previewStepStatusLabel = getAvatarStepStatusLabel(
    previewStep,
    availableMotionClipIds,
    motionLibraryState
  );
  const previewStepHasStaticFallback = isMotionClipUnavailable(
    previewStep,
    availableMotionClipIds,
    motionLibraryState
  );
  const displayedMotionBackedGlossCount =
    avatarQueue?.steps.filter((step) =>
      isAvatarStepDynamicallyPlayable(step, availableMotionClipIds)
    ).length ?? 0;
  const sourceWordCount = countWordActionSourceTokens(wordActionHistory);
  const readyModelCount = modelRegistry.filter((model) => model.status === "ready").length;
  const trainingModelCount = modelRegistry.filter((model) => model.status === "training").length;
  const unavailableModelCount = modelRegistry.filter(
    (model) => model.status === "unavailable"
  ).length;
  const registryDetails = modelRegistry.length
    ? [
        readyModelCount ? `${readyModelCount} ready` : undefined,
        trainingModelCount ? `${trainingModelCount} training` : undefined,
        unavailableModelCount ? `${unavailableModelCount} unavailable` : undefined
      ]
        .filter((label): label is string => Boolean(label))
        .join(" · ") || `${modelRegistry.length} reported`
    : modelRegistryStatus;
  const connectionLabel = getConnectionLabel(connectionStatus);
  const ConnectionIcon =
    connectionStatus === "connected"
      ? Wifi
      : connectionStatus === "offline"
        ? WifiOff
        : connectionStatus === "degraded"
          ? AlertTriangle
          : LoaderCircle;
  const PlatformIcon =
    activePlatform.kind === "youtube"
      ? Youtube
      : activePlatform.kind === "meet" || activePlatform.kind === "zoom"
        ? Video
        : Laptop;
  const closeFeedback = useCallback(() => setFeedbackOpen(false), []);
  const updateConnectionStatus = useCallback((nextStatus: ConnectionStatus): void => {
    connectionStatusVersionRef.current += 1;
    setConnectionStatus(nextStatus);
  }, []);

  const refreshLocalConnection = useCallback(async (): Promise<void> => {
    updateConnectionStatus("checking");
    const refreshVersion = connectionStatusVersionRef.current;
    try {
      const registry = await getModelRegistry();
      if (!mountedRef.current) {
        return;
      }
      setModelRegistry(registry.models);
      setModelRegistryStatus(
        registry.models.length
          ? `${registry.models.length} local models available`
          : "No local models found"
      );
      if (connectionStatusVersionRef.current === refreshVersion) {
        updateConnectionStatus("connected");
      }
    } catch (caught) {
      if (mountedRef.current) {
        const nextConnectionStatus = getConnectionStatusForError(caught);
        setModelRegistryStatus(
          nextConnectionStatus === "offline"
            ? "Models offline"
            : `Model registry unavailable. ${getOperationErrorDetail(
                caught,
                "The local API returned an invalid response."
              )}`
        );
        if (connectionStatusVersionRef.current === refreshVersion) {
          updateConnectionStatus(nextConnectionStatus);
        }
      }
    }
  }, [updateConnectionStatus]);

  useEffect(() => {
    let cancelled = false;
    setMotionLibraryState("loading");
    void loadMotionClipMap().then(
      (clips) => {
        if (!cancelled) {
          setBundledMotionClipIds(new Set(clips.keys()));
          setMotionLibraryState("ready");
        }
      },
      () => {
        if (!cancelled) {
          setBundledMotionClipIds(new Set());
          setMotionLibraryState("error");
        }
      }
    );
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    setPreviewStepIndex((current) =>
      previewSteps.length ? Math.min(current, previewSteps.length - 1) : 0
    );
  }, [previewSteps.length]);

  useEffect(() => {
    mountedRef.current = true;
    let cancelled = false;

    void loadSettings().then((loadedSettings) => {
      if (cancelled || settingsRef.current) {
        return;
      }
      settingsRef.current = loadedSettings;
      setSelectedAvatarName(loadedSettings.avatarName);
      setSettings(loadedSettings);
    });
    void loadSessionHistory().then((history) => {
      if (!cancelled) {
        setSessionHistory(history);
      }
    });
    void refreshLocalConnection();

    return () => {
      const closingSessionId = sessionIdRef.current;
      const closingTargetTabId = sessionTargetTabIdRef.current;
      const closingSessionStartedAt = sessionStartedAtRef.current;
      cancelled = true;
      mountedRef.current = false;
      sessionLifecycleRef.current += 1;
      sessionIdRef.current = undefined;
      sessionTargetTabIdRef.current = undefined;
      sessionStartedAtRef.current = undefined;
      captionPollingGenerationRef.current += 1;
      if (captionPollingTimerRef.current !== undefined) {
        globalThis.clearTimeout(captionPollingTimerRef.current);
        captionPollingTimerRef.current = undefined;
      }
      if (closingSessionId) {
        void sendRuntimeMessage({
          type: "SESSION_STOPPED",
          sessionId: closingSessionId,
          ...(closingTargetTabId === undefined ? {} : { targetTabId: closingTargetTabId })
        });
        void endSession(
          closingSessionId,
          closingSessionStartedAt ? Math.max(0, Date.now() - closingSessionStartedAt) : 0
        ).catch(() => undefined);
      }
    };
  }, [refreshLocalConnection]);

  function getCurrentSettings(): Settings {
    return settingsRef.current ?? resolvedSettings;
  }

  function getRuntimeTarget(): { targetTabId?: number } {
    const targetTabId = sessionTargetTabIdRef.current;
    return targetTabId === undefined ? {} : { targetTabId };
  }

  function getRuntimeSessionRouting(): { sessionId?: string; targetTabId?: number } {
    const activeSessionId = sessionIdRef.current;
    return {
      ...(activeSessionId ? { sessionId: activeSessionId } : {}),
      ...getRuntimeTarget()
    };
  }

  function isSessionCurrent(activeSessionId: string, pollingGeneration?: number): boolean {
    return (
      mountedRef.current &&
      sessionIdRef.current === activeSessionId &&
      (pollingGeneration === undefined || pollingGeneration === captionPollingGenerationRef.current)
    );
  }

  async function sendOverlayMessage(message: RuntimeMessage): Promise<boolean> {
    const delivered = await sendRuntimeMessage(message);
    if (!delivered && mountedRef.current) {
      setError(OVERLAY_DELIVERY_ERROR);
    }
    return delivered;
  }

  function stopCaptionPolling(): void {
    captionPollingGenerationRef.current += 1;
    lastCaptionTextRef.current = "";
    missingCaptionPollCountRef.current = 0;
    hasSeenLiveCaptionRef.current = false;
    if (captionPollingTimerRef.current !== undefined) {
      globalThis.clearTimeout(captionPollingTimerRef.current);
      captionPollingTimerRef.current = undefined;
    }
  }

  function resetSessionOutput(): void {
    setSegment(undefined);
    setLiveCaptionText(undefined);
    setInterpretation(undefined);
    setAvatarQueue(undefined);
    setWordActionHistory([]);
    setModelStatus(undefined);
    setModelBackedGlossCount(0);
    setPreviewStepIndex(0);
    setIsCaptionPaused(false);
    setGlossaryEntries([]);
    setGlossaryStatus("Select a detected term to review glossary details.");
    setFeedbackOpen(false);
  }

  async function interpretTranscript(input: {
    activeSessionId: string;
    source: TranscriptSource;
    rawText: string;
    pageTitle?: string;
    sourceName: string;
    startMs?: number;
    saveToHistory?: boolean;
    pollingGeneration?: number;
    replaceRollingOutput?: boolean;
  }): Promise<boolean> {
    if (!isSessionCurrent(input.activeSessionId, input.pollingGeneration)) {
      return false;
    }

    const activeSettings = getCurrentSettings();
    const baseSegment = createMockTranscriptSegment(input.activeSessionId, {
      rawText: input.rawText,
      source: input.source
    });
    const startMs = input.startMs ?? 0;
    const nextSegment: TranscriptSegment = {
      ...baseSegment,
      startMs,
      endMs: startMs + 4000
    };
    setSegment(nextSegment);
    await sendRuntimeMessage({
      type: "TRANSCRIPT_SEGMENT",
      segment: nextSegment,
      ...getRuntimeTarget()
    });

    const interpretationRequest: InterpretRequest = {
      sessionId: input.activeSessionId,
      source: input.source,
      rawText: nextSegment.rawText,
      languageHint: getLanguageHint(activeSettings.captionLanguage),
      avatarSpeed: activeSettings.avatarSpeed,
      simplificationLevel: activeSettings.simplificationLevel,
      timestamp: { startMs: nextSegment.startMs ?? 0, endMs: nextSegment.endMs ?? 4000 }
    };
    const nextInterpretation = await interpretSegment(interpretationRequest);
    updateConnectionStatus("connected");

    if (!isSessionCurrent(input.activeSessionId, input.pollingGeneration)) {
      return false;
    }

    const currentSettings = getCurrentSettings();
    const nextQueue =
      nextInterpretation.avatarQueue.speed === currentSettings.avatarSpeed
        ? nextInterpretation.avatarQueue
        : rescaleAvatarQueue(nextInterpretation.avatarQueue, currentSettings.avatarSpeed);
    setInterpretation(nextInterpretation);
    if (input.pollingGeneration === undefined || input.replaceRollingOutput) {
      setAvatarQueue(nextQueue);
      setWordActionHistory(
        nextInterpretation.glossSequence.map((item) => ({
          interpretationId: nextInterpretation.id,
          item
        }))
      );
      setPreviewStepIndex(0);
    } else {
      setAvatarQueue((currentQueue) => mergeRollingAvatarQueues(currentQueue, nextQueue));
      setWordActionHistory((current) =>
        mergeWordActionHistory(current, nextInterpretation.id, nextInterpretation.glossSequence)
      );
    }
    setModelStatus(nextInterpretation.model);
    setModelBackedGlossCount(nextInterpretation.modelBackedGlossCount ?? 0);
    await sendRuntimeMessage({
      type: "INTERPRETATION_READY",
      interpretation: nextInterpretation,
      ...getRuntimeTarget()
    });
    if (input.replaceRollingOutput) {
      await sendOverlayMessage({
        type: "SESSION_STOPPED",
        sessionId: input.activeSessionId,
        ...getRuntimeTarget()
      });
      await sendOverlayMessage({
        type: "SESSION_STARTED",
        sessionId: input.activeSessionId,
        ...getRuntimeTarget()
      });
    }
    await sendOverlayMessage({
      type: "OVERLAY_SETTINGS_UPDATED",
      settings: currentSettings,
      ...getRuntimeSessionRouting()
    });
    if (shouldShowAvatarOverlay(currentSettings)) {
      await sendOverlayMessage({
        type: "AVATAR_QUEUE_UPDATED",
        queue: nextQueue,
        ...getRuntimeSessionRouting()
      });
    }

    if (getCurrentSettings().saveHistory && input.saveToHistory !== false) {
      const historyWrite = appendSessionHistory({
        id: nextInterpretation.id,
        sourceLabel: input.sourceName,
        ...(input.pageTitle ? { pageTitle: input.pageTitle } : {}),
        summary: nextInterpretation.simplifiedText,
        confidence: nextInterpretation.confidence,
        createdAt: nextInterpretation.createdAt
      });
      historyWriteRef.current = historyWrite.then(
        () => undefined,
        () => undefined
      );
      const nextHistory = await historyWrite;
      if (mountedRef.current) {
        setSessionHistory(nextHistory);
      }
    }

    return true;
  }

  function startCaptionPolling(
    activeSessionId: string,
    initialCaptionText: string,
    initialPageTitle: string
  ): void {
    stopCaptionPolling();
    const pollingGeneration = captionPollingGenerationRef.current;
    lastCaptionTextRef.current = normalizeCaptionText(initialCaptionText);
    hasSeenLiveCaptionRef.current = Boolean(lastCaptionTextRef.current);

    const scheduleNextPoll = (): void => {
      if (!isSessionCurrent(activeSessionId, pollingGeneration)) {
        return;
      }
      captionPollingTimerRef.current = globalThis.setTimeout(() => {
        void pollCaptions();
      }, LIVE_CAPTION_POLL_INTERVAL_MS);
    };

    const pollCaptions = async (): Promise<void> => {
      captionPollingTimerRef.current = undefined;
      if (!isSessionCurrent(activeSessionId, pollingGeneration)) {
        return;
      }

      try {
        const snapshotAttempt = await requestPageCaptionSnapshotAttempt(
          sessionTargetTabIdRef.current
        );
        if (!isSessionCurrent(activeSessionId, pollingGeneration)) {
          return;
        }
        const snapshot = snapshotAttempt.ok ? snapshotAttempt.snapshot : undefined;
        if (snapshot?.targetTabId !== undefined) {
          sessionTargetTabIdRef.current = snapshot.targetTabId;
        }
        if (snapshot) {
          setActivePlatform(getActivePlatform(snapshot.url, snapshot.supportStatus));
        }
        const nextCaptionText = normalizeCaptionText(snapshot?.captionText);
        if (!nextCaptionText) {
          missingCaptionPollCountRef.current += 1;
          if (
            hasSeenLiveCaptionRef.current &&
            missingCaptionPollCountRef.current >= LIVE_CAPTION_LOSS_POLL_COUNT
          ) {
            setLiveCaptionText(undefined);
            setSegment(undefined);
            setSourceLabel("Captions unavailable");
            setSourceNotice(
              "Live captions are no longer visible. SignSaarthi is still checking this video."
            );
          }
          return;
        }

        missingCaptionPollCountRef.current = 0;
        hasSeenLiveCaptionRef.current = true;
        setLiveCaptionText(nextCaptionText);
        setSourceLabel("YouTube captions");
        setSourceNotice(undefined);

        const previousCaptionText = lastCaptionTextRef.current;
        const captionDelta = getRollingCaptionDelta(previousCaptionText, nextCaptionText);
        if (captionDelta) {
          const wasApplied = await interpretTranscript({
            activeSessionId,
            source: "youtube_captions",
            rawText: captionDelta,
            pageTitle: snapshot?.title || initialPageTitle,
            sourceName: "YouTube captions",
            ...(snapshot?.mediaTimeMs !== undefined ? { startMs: snapshot.mediaTimeMs } : {}),
            saveToHistory: false,
            pollingGeneration,
            replaceRollingOutput: !previousCaptionText
          });
          if (wasApplied) {
            lastCaptionTextRef.current = nextCaptionText;
          }
        } else {
          lastCaptionTextRef.current = nextCaptionText;
        }
      } catch (caught) {
        if (isSessionCurrent(activeSessionId, pollingGeneration)) {
          updateConnectionStatus(getConnectionStatusForError(caught));
          setSourceNotice(
            "Live caption update could not be interpreted. Retrying while the session stays active."
          );
        }
      } finally {
        scheduleNextPoll();
      }
    };

    scheduleNextPoll();
  }

  async function handleStart(): Promise<void> {
    const operationStartedAt = Date.now();
    const lifecycle = sessionLifecycleRef.current + 1;
    sessionLifecycleRef.current = lifecycle;
    stopCaptionPolling();
    setStatus("processing");
    updateConnectionStatus("checking");
    setError(undefined);
    setFeedbackStatus(undefined);
    setSourceNotice(undefined);
    resetSessionOutput();

    try {
      const snapshotAttempt = await requestPageCaptionSnapshotAttempt();
      const snapshot = snapshotAttempt.ok ? snapshotAttempt.snapshot : readYouTubeCaptionSnapshot();
      if (!mountedRef.current || lifecycle !== sessionLifecycleRef.current) {
        return;
      }
      const initialCaptionText = normalizeCaptionText(snapshot.captionText);
      const detectedPlatform: ActivePlatform = snapshotAttempt.ok
        ? getActivePlatform(snapshot.url, snapshot.supportStatus)
        : snapshotAttempt.reason === "transient"
          ? { kind: "youtube", label: "YouTube", detail: "Connecting to captions" }
          : { kind: "demo", label: "Local demo", detail: "Demo transcript" };
      setActivePlatform(detectedPlatform);
      const transcriptSource: TranscriptSource = initialCaptionText ? "youtube_captions" : "mock";
      const shouldPollCaptions =
        snapshot.supportStatus !== "unsupported" ||
        (!snapshotAttempt.ok && snapshotAttempt.reason === "transient");
      const nextSourceLabel =
        transcriptSource === "mock" && shouldPollCaptions
          ? "Demo transcript"
          : getSourceLabel(transcriptSource, snapshot.supportStatus);
      const capturedSegmentStartMs = snapshot.mediaTimeMs ?? 0;
      const capturedSegment: TranscriptSegment = {
        ...createMockTranscriptSegment(`pending_${operationStartedAt}`, {
          rawText: initialCaptionText || DEMO_TRANSCRIPT,
          source: transcriptSource
        }),
        startMs: capturedSegmentStartMs,
        endMs: capturedSegmentStartMs + 4000
      };
      setSegment(capturedSegment);
      setLiveCaptionText(initialCaptionText || undefined);
      setSourceLabel(nextSourceLabel);
      const started = await startSession({
        source: transcriptSource,
        pageTitle: snapshot.title,
        url: snapshot.url
      });
      updateConnectionStatus("connected");
      if (!mountedRef.current || lifecycle !== sessionLifecycleRef.current) {
        await endSession(started.sessionId, Math.max(0, Date.now() - operationStartedAt)).catch(
          () => undefined
        );
        return;
      }

      sessionIdRef.current = started.sessionId;
      sessionTargetTabIdRef.current = snapshot.targetTabId;
      sessionStartedAtRef.current = Date.now();
      setSessionId(started.sessionId);
      await sendOverlayMessage({
        type: "SESSION_STARTED",
        sessionId: started.sessionId,
        ...getRuntimeTarget()
      });
      await sendOverlayMessage({
        type: "OVERLAY_SETTINGS_UPDATED",
        settings: getCurrentSettings(),
        ...getRuntimeSessionRouting()
      });
      setSourceNotice(
        transcriptSource === "mock" && shouldPollCaptions
          ? "Captions are not visible yet. Using the demo transcript while SignSaarthi keeps checking this video."
          : transcriptSource === "mock" &&
              (detectedPlatform.kind === "meet" || detectedPlatform.kind === "zoom")
            ? `${detectedPlatform.label} detected. Live caption capture is not enabled for this MVP, so the demo transcript is active.`
            : undefined
      );

      const wasApplied = await interpretTranscript({
        activeSessionId: started.sessionId,
        source: transcriptSource,
        rawText: initialCaptionText || DEMO_TRANSCRIPT,
        pageTitle: snapshot.title,
        sourceName: nextSourceLabel,
        ...(snapshot.mediaTimeMs !== undefined ? { startMs: snapshot.mediaTimeMs } : {})
      });
      if (!wasApplied || lifecycle !== sessionLifecycleRef.current) {
        return;
      }

      setStatus("active");
      if (shouldPollCaptions) {
        startCaptionPolling(started.sessionId, initialCaptionText, snapshot.title);
      }
    } catch (caught) {
      if (!mountedRef.current || lifecycle !== sessionLifecycleRef.current) {
        return;
      }
      const failedSessionId = sessionIdRef.current;
      const failedTargetTabId = sessionTargetTabIdRef.current;
      const failedSessionStartedAt = sessionStartedAtRef.current;
      stopCaptionPolling();
      sessionIdRef.current = undefined;
      sessionTargetTabIdRef.current = undefined;
      sessionStartedAtRef.current = undefined;
      setSessionId(undefined);
      if (failedSessionId) {
        await endSession(
          failedSessionId,
          failedSessionStartedAt ? Math.max(0, Date.now() - failedSessionStartedAt) : 0
        ).catch(() => undefined);
      }
      const message =
        caught instanceof Error
          ? caught.message
          : "AI service unavailable. Captions and local glossary still work.";
      setError(message);
      setStatus("error");
      updateConnectionStatus(getConnectionStatusForError(caught));
      void sendRuntimeMessage({
        type: "ERROR_STATE",
        message,
        ...(failedSessionId ? { sessionId: failedSessionId } : {}),
        ...(failedTargetTabId === undefined ? {} : { targetTabId: failedTargetTabId })
      });
    }
  }

  async function handleStop(): Promise<void> {
    const stoppedSessionId = sessionIdRef.current ?? sessionId;
    const stoppedTargetTabId = sessionTargetTabIdRef.current;
    const stoppedSessionStartedAt = sessionStartedAtRef.current;
    sessionLifecycleRef.current += 1;
    stopCaptionPolling();
    sessionIdRef.current = undefined;
    sessionTargetTabIdRef.current = undefined;
    sessionStartedAtRef.current = undefined;
    setStatus("ready");
    setActivePlatform(NO_ACTIVE_PLATFORM);
    setSessionId(undefined);
    setSourceLabel("Demo transcript");
    setSourceNotice(undefined);
    setError(undefined);
    setFeedbackStatus(undefined);
    resetSessionOutput();

    if (stoppedSessionId) {
      const elapsedDuration = stoppedSessionStartedAt
        ? Math.max(0, Date.now() - stoppedSessionStartedAt)
        : 0;
      await sendOverlayMessage({
        type: "SESSION_STOPPED",
        sessionId: stoppedSessionId,
        ...(stoppedTargetTabId === undefined ? {} : { targetTabId: stoppedTargetTabId })
      });
      await endSession(stoppedSessionId, elapsedDuration).catch(() => undefined);
    }
  }

  async function handleFeedback(input: {
    reason: FeedbackReport["reason"];
    comment: string;
  }): Promise<void> {
    if (!sessionId || !interpretation) {
      return;
    }

    setIsSubmittingFeedback(true);
    try {
      await submitFeedback({
        sessionId,
        interpretationId: interpretation.id,
        reason: input.reason,
        transcriptText: interpretation.cleanedText,
        glossSequence: interpretation.glossSequence.map((item) => ({
          id: item.id,
          token: item.token,
          gloss: item.gloss,
          ...(item.signAssetId ? { signAssetId: item.signAssetId } : {}),
          isVerified: item.isVerified,
          fallback: item.fallback,
          confidence: item.confidence
        })),
        timestamp: { startMs: segment?.startMs ?? 0, endMs: segment?.endMs ?? 0 },
        userComment: input.comment
      });
      setFeedbackOpen(false);
      setFeedbackStatus("Feedback saved for reviewer follow-up.");
      updateConnectionStatus("connected");
    } catch (caught) {
      const nextConnectionStatus = getConnectionStatusForError(caught);
      updateConnectionStatus(nextConnectionStatus);
      setFeedbackStatus(
        nextConnectionStatus === "offline"
          ? "Feedback could not be saved while the local API is offline."
          : `Feedback was not saved. ${getOperationErrorDetail(
              caught,
              "The local API returned an invalid response."
            )}`
      );
    } finally {
      setIsSubmittingFeedback(false);
    }
  }

  async function handleAvatarSpeedChange(value: number): Promise<void> {
    const nextSettings = { ...getCurrentSettings(), avatarSpeed: value };
    await persistSettings(nextSettings);
    await sendOverlayMessage({
      type: "OVERLAY_SETTINGS_UPDATED",
      settings: nextSettings,
      ...getRuntimeSessionRouting()
    });
    if (avatarQueue) {
      const nextQueue = rescaleAvatarQueue(avatarQueue, value);
      setAvatarQueue(nextQueue);
      if (shouldShowAvatarOverlay(nextSettings)) {
        await sendOverlayMessage({
          type: "AVATAR_PLAYBACK_UPDATED",
          action: "set_speed",
          speed: value,
          ...getRuntimeSessionRouting()
        });
      }
    }
  }

  async function handleAvatarSizeChange(value: Settings["avatarSize"]): Promise<void> {
    const nextSettings = { ...getCurrentSettings(), avatarSize: value };
    await persistSettings(nextSettings);
    await sendOverlayMessage({
      type: "OVERLAY_SETTINGS_UPDATED",
      settings: nextSettings,
      ...getRuntimeSessionRouting()
    });
  }

  async function handleClearLocalData(): Promise<void> {
    const defaultSettings = settingsSchema.parse({});
    settingsRef.current = defaultSettings;
    setSelectedAvatarName(defaultSettings.avatarName);
    setSettings(defaultSettings);
    await sendOverlayMessage({
      type: "OVERLAY_SETTINGS_UPDATED",
      settings: defaultSettings,
      ...getRuntimeSessionRouting()
    });
    await historyWriteRef.current;
    let purgeError: unknown;
    await Promise.all([
      clearLocalData(),
      purgePrivacyData({ scope: "all", confirmation: "purge_all_local_data" }).catch((caught) => {
        purgeError = caught;
      })
    ]);
    setSessionHistory([]);
    if (purgeError === undefined) {
      updateConnectionStatus("connected");
      setFeedbackStatus("Local extension and API data deleted.");
      return;
    }

    const nextConnectionStatus = getConnectionStatusForError(purgeError);
    updateConnectionStatus(nextConnectionStatus);
    setFeedbackStatus(
      nextConnectionStatus === "offline"
        ? "Extension data deleted. Local API data could not be reached."
        : `Extension data deleted. API data was not deleted. ${getOperationErrorDetail(
            purgeError,
            "The local API returned an invalid response."
          )}`
    );
  }

  async function handleSaveHistoryChange(value: boolean): Promise<void> {
    await persistSettings({ ...getCurrentSettings(), saveHistory: value });
  }

  async function handleOutputModeChange(value: OutputMode): Promise<void> {
    const nextSettings = { ...getCurrentSettings(), outputMode: value };
    await persistSettings(nextSettings);
    await sendOverlayMessage({
      type: "OVERLAY_SETTINGS_UPDATED",
      settings: nextSettings,
      ...getRuntimeSessionRouting()
    });
    if (avatarQueue && shouldShowAvatarOverlay(nextSettings)) {
      await sendOverlayMessage({
        type: "AVATAR_QUEUE_UPDATED",
        queue: avatarQueue,
        ...getRuntimeSessionRouting()
      });
    }
  }

  async function handleAvatarSelection(value: AvatarName): Promise<void> {
    setSelectedAvatarName(value);
    const nextSettings = { ...getCurrentSettings(), avatarName: value };
    await persistSettings(nextSettings);
    await sendOverlayMessage({
      type: "OVERLAY_SETTINGS_UPDATED",
      settings: nextSettings,
      ...getRuntimeSessionRouting()
    });
  }

  async function handleCaptionFontSizeChange(delta: number): Promise<void> {
    const currentSettings = getCurrentSettings();
    const nextSize = Math.min(28, Math.max(14, currentSettings.captionFontSize + delta));
    await persistSettings({ ...currentSettings, captionFontSize: nextSize });
  }

  async function handleAvatarPositionChange(value: Settings["avatarPosition"]): Promise<void> {
    const nextSettings = { ...getCurrentSettings(), avatarPosition: value };
    await persistSettings(nextSettings);
    await sendOverlayMessage({
      type: "OVERLAY_SETTINGS_UPDATED",
      settings: nextSettings,
      ...getRuntimeSessionRouting()
    });
    if (avatarQueue && shouldShowAvatarOverlay(nextSettings)) {
      await sendOverlayMessage({
        type: "AVATAR_QUEUE_UPDATED",
        queue: avatarQueue,
        ...getRuntimeSessionRouting()
      });
    }
  }

  async function handleCaptionLanguageChange(value: Settings["captionLanguage"]): Promise<void> {
    await persistSettings({ ...getCurrentSettings(), captionLanguage: value });
  }

  async function handleSimplificationLevelChange(
    value: Settings["simplificationLevel"]
  ): Promise<void> {
    await persistSettings({ ...getCurrentSettings(), simplificationLevel: value });
  }

  async function handleBooleanSettingChange(
    key: "feedbackSharing" | "privacyMode" | "showGlossDebug",
    value: boolean
  ): Promise<void> {
    await persistSettings({ ...getCurrentSettings(), [key]: value });
  }

  async function handleGlossarySearch(term: string): Promise<void> {
    setGlossaryStatus(`Searching glossary for ${term}...`);
    try {
      const response = await searchGlossary(term);
      setGlossaryEntries(response.results);
      setGlossaryStatus(
        response.results.length ? `Glossary matches for ${term}` : `No glossary match for ${term}.`
      );
      updateConnectionStatus("connected");
    } catch (caught) {
      const nextConnectionStatus = getConnectionStatusForError(caught);
      updateConnectionStatus(nextConnectionStatus);
      setGlossaryEntries([]);
      setGlossaryStatus(
        nextConnectionStatus === "offline"
          ? "Glossary unavailable while the local API is offline."
          : `Glossary search failed. ${getOperationErrorDetail(
              caught,
              "The local API returned an invalid response."
            )}`
      );
    }
  }

  async function persistSettings(nextSettings: Settings): Promise<void> {
    settingsRef.current = nextSettings;
    setSettings(nextSettings);
    await saveSettings(nextSettings);
  }

  function replayAvatarQueue(): void {
    setPreviewReplayKey((current) => current + 1);
    if (avatarQueue && showAvatarOverlay) {
      void sendOverlayMessage({
        type: "AVATAR_PLAYBACK_UPDATED",
        action: "replay",
        ...getRuntimeSessionRouting()
      });
    }
  }

  async function togglePlaybackPause(): Promise<void> {
    const nextPaused = !isCaptionPaused;
    if (!sessionIdRef.current || !showAvatarOverlay) {
      setIsCaptionPaused(nextPaused);
      return;
    }

    const delivered = await sendOverlayMessage({
      type: "AVATAR_PLAYBACK_UPDATED",
      action: nextPaused ? "pause" : "resume",
      ...getRuntimeSessionRouting()
    });
    if (delivered && mountedRef.current && sessionIdRef.current) {
      setIsCaptionPaused(nextPaused);
    }
  }

  return (
    <main className="panel-shell">
      <header className="panel-header">
        <div className="brand">
          <BrandMark />
          <div>
            <h1>
              SignSaarthi <span>AI</span>
            </h1>
            <p className={`status-dot status-dot--${status}`}>
              {status === "ready" ? "Ready" : titleCase(status)}
            </p>
          </div>
        </div>
        <button
          className="icon-button"
          type="button"
          aria-label="Open settings"
          onClick={() => setActiveView("settings")}
        >
          <Settings2 size={18} />
        </button>
      </header>

      <section
        className="connection-bar"
        aria-label="Connection and platform status"
        aria-live="polite"
      >
        <div className={`connection-bar__item connection-bar__item--${connectionStatus}`}>
          <ConnectionIcon
            className={connectionStatus === "checking" ? "connection-bar__spinner" : undefined}
            size={16}
            aria-hidden="true"
          />
          <span className="connection-bar__copy">
            <small>Local API</small>
            <strong>{connectionLabel}</strong>
          </span>
          <button
            className="connection-bar__retry"
            type="button"
            aria-label="Refresh local API connection"
            title="Refresh connection"
            onClick={() => void refreshLocalConnection()}
          >
            <RefreshCw size={13} />
          </button>
        </div>
        <div className={`connection-bar__item connection-bar__item--${activePlatform.kind}`}>
          <PlatformIcon size={16} aria-hidden="true" />
          <span className="connection-bar__copy">
            <small>Platform</small>
            <strong>{activePlatform.label}</strong>
            <em>{activePlatform.detail}</em>
          </span>
        </div>
      </section>

      <section className="session-toolbar" aria-label="Session controls">
        <button
          className="primary-button primary-button--wide"
          type="button"
          aria-busy={status === "processing"}
          onClick={() => void (isRunning ? handleStop() : handleStart())}
        >
          {isRunning ? <Pause size={18} /> : <Play size={18} />}
          {isRunning ? "Stop Interpretation" : "Start Interpretation"}
        </button>
        <div className="session-context">
          <div className="session-context__item" title={sourceLabel}>
            <BookOpenText size={15} />
            <span>
              <small>Input</small>
              <strong>{sourceLabel}</strong>
            </span>
          </div>
          <button
            className="session-context__item session-context__button"
            type="button"
            onClick={() => setActiveView("settings")}
          >
            <Settings2 size={15} />
            <span>
              <small>Output</small>
              <strong>{getOutputModeLabel(resolvedSettings.outputMode)}</strong>
            </span>
          </button>
        </div>
      </section>

      {sourceNotice ? (
        <p className="warning" role="status">
          <AlertTriangle size={15} /> {sourceNotice}
        </p>
      ) : null}
      {error ? (
        <p className="warning" role="alert">
          <AlertTriangle size={15} /> {error}
        </p>
      ) : null}
      {feedbackStatus ? (
        <p className="success" role="status">
          {feedbackStatus}
        </p>
      ) : null}

      {activeView === "home" ? (
        <section className="app-view" aria-labelledby="home-view-title">
          <h2 className="sr-only" id="home-view-title">
            Home
          </h2>
          <section className="interpreter-dashboard" aria-label="Interpreter dashboard">
            <div className="dashboard-tabs" role="tablist" aria-label="Dashboard sections">
              {dashboardSections.map(({ id, label, icon: Icon }) => {
                const isActive = activeDashboardSection === id;
                return (
                  <button
                    type="button"
                    role="tab"
                    id={`dashboard-tab-${id}`}
                    aria-controls="dashboard-panel"
                    aria-selected={isActive}
                    tabIndex={isActive ? 0 : -1}
                    className={isActive ? "dashboard-tab dashboard-tab--active" : "dashboard-tab"}
                    key={id}
                    onClick={() => setActiveDashboardSection(id)}
                    onKeyDown={(event) => {
                      const currentIndex = dashboardSections.findIndex(
                        (section) => section.id === id
                      );
                      const direction =
                        event.key === "ArrowRight" ? 1 : event.key === "ArrowLeft" ? -1 : 0;
                      if (!direction) {
                        return;
                      }
                      event.preventDefault();
                      const nextSection =
                        dashboardSections[
                          (currentIndex + direction + dashboardSections.length) %
                            dashboardSections.length
                        ];
                      if (nextSection) {
                        setActiveDashboardSection(nextSection.id);
                        globalThis.requestAnimationFrame(() => {
                          document.getElementById(`dashboard-tab-${nextSection.id}`)?.focus();
                        });
                      }
                    }}
                  >
                    <Icon size={14} />
                    <span>{label}</span>
                  </button>
                );
              })}
            </div>

            <div
              className="dashboard-panel"
              role="tabpanel"
              tabIndex={0}
              id="dashboard-panel"
              aria-labelledby={`dashboard-tab-${activeDashboardSection}`}
            >
              {activeDashboardSection === "interpreter" ? (
                <>
                  <div className="mode-grid" aria-label="Output mode shortcuts">
                    <button
                      type="button"
                      aria-pressed={resolvedSettings.outputMode === "captions_avatar"}
                      className={
                        resolvedSettings.outputMode === "captions_avatar"
                          ? "mode-card mode-card--active"
                          : "mode-card"
                      }
                      onClick={() => void handleOutputModeChange("captions_avatar")}
                    >
                      <Sparkles size={17} />
                      <strong>Auto</strong>
                      <span>Captions + ISL</span>
                    </button>
                    <button
                      type="button"
                      aria-pressed={resolvedSettings.outputMode === "captions_only"}
                      className={
                        resolvedSettings.outputMode === "captions_only"
                          ? "mode-card mode-card--active"
                          : "mode-card"
                      }
                      onClick={() => void handleOutputModeChange("captions_only")}
                    >
                      <Captions size={17} />
                      <strong>Captions</strong>
                      <span>Text first</span>
                    </button>
                    <button
                      type="button"
                      aria-pressed={resolvedSettings.outputMode === "avatar_only"}
                      className={
                        resolvedSettings.outputMode === "avatar_only"
                          ? "mode-card mode-card--active"
                          : "mode-card"
                      }
                      onClick={() => void handleOutputModeChange("avatar_only")}
                    >
                      <UserRound size={17} />
                      <strong>ISL only</strong>
                      <span>Avatar focus</span>
                    </button>
                  </div>

                  <div className="section-title-row">
                    <h3>Interpreter avatar</h3>
                    <span>{selectedAvatar.name}</span>
                  </div>
                  <div className="avatar-picker" role="radiogroup" aria-label="Avatar selection">
                    {avatarOptions.map((avatar) => {
                      const isSelected = avatar.name === selectedAvatar.name;
                      return (
                        <button
                          type="button"
                          role="radio"
                          aria-checked={isSelected}
                          tabIndex={isSelected ? 0 : -1}
                          aria-label={`${avatar.name} avatar option`}
                          className={isSelected ? "avatar-card avatar-card--active" : "avatar-card"}
                          key={avatar.name}
                          onClick={() => void handleAvatarSelection(avatar.name)}
                          onKeyDown={(event) => {
                            const currentIndex = avatarOptions.findIndex(
                              (option) => option.name === avatar.name
                            );
                            const nextIndex =
                              event.key === "Home"
                                ? 0
                                : event.key === "End"
                                  ? avatarOptions.length - 1
                                  : event.key === "ArrowRight" || event.key === "ArrowDown"
                                    ? (currentIndex + 1) % avatarOptions.length
                                    : event.key === "ArrowLeft" || event.key === "ArrowUp"
                                      ? (currentIndex - 1 + avatarOptions.length) %
                                        avatarOptions.length
                                      : undefined;
                            if (nextIndex === undefined) {
                              return;
                            }
                            event.preventDefault();
                            const nextAvatar = avatarOptions[nextIndex];
                            if (nextAvatar) {
                              void handleAvatarSelection(nextAvatar.name);
                              globalThis.requestAnimationFrame(() => {
                                document
                                  .querySelector<HTMLButtonElement>(
                                    `[aria-label="${nextAvatar.name} avatar option"]`
                                  )
                                  ?.focus();
                              });
                            }
                          }}
                        >
                          <img src={avatar.thumbnailUrl} alt="" aria-hidden="true" />
                          <span>{avatar.name}</span>
                        </button>
                      );
                    })}
                  </div>

                  <section
                    className="stage-card"
                    aria-label={`${selectedAvatar.name} interpreter preview`}
                  >
                    <div className="stage-visual">
                      <div className="stage-orbit" aria-hidden="true" />
                      <MotionAvatarPreview
                        avatarName={selectedAvatar.name}
                        steps={previewSteps}
                        runtimeMotionClips={avatarQueue?.motionClips}
                        paused={isCaptionPaused}
                        replayKey={previewReplayKey}
                        speed={
                          realtimePlayback?.backlog.catchUpSpeed ?? resolvedSettings.avatarSpeed
                        }
                        onStepChange={setPreviewStepIndex}
                      />
                    </div>
                    <div className="stage-caption">
                      <strong>{previewStep?.label ?? "READY"}</strong>
                      <span
                        className={
                          previewStepHasStaticFallback
                            ? "stage-caption__status stage-caption__status--fallback"
                            : "stage-caption__status"
                        }
                      >
                        {previewStepStatusLabel}
                      </span>
                    </div>
                    <p className="stage-note">{selectedAvatar.stageNote}</p>
                    <div className="stage-controls" aria-label="Playback preview controls">
                      <button
                        type="button"
                        disabled={!canReplayPreview}
                        onClick={() => void togglePlaybackPause()}
                      >
                        {isCaptionPaused ? <Play size={14} /> : <Pause size={14} />}
                        {isCaptionPaused ? "Resume" : "Pause"}
                      </button>
                      <button
                        type="button"
                        disabled={!canReplayPreview}
                        onClick={replayAvatarQueue}
                      >
                        <RotateCcw size={14} /> Rewind
                      </button>
                      <button
                        type="button"
                        aria-label={`Adjust signing pace, ${getSigningPaceLabel(resolvedSettings.avatarSpeed)} ${formatSpeed(resolvedSettings.avatarSpeed)}`}
                        title="Adjust signing pace"
                        onClick={() => setActiveView("settings")}
                      >
                        <Gauge size={14} />
                        {formatSpeed(resolvedSettings.avatarSpeed)}
                      </button>
                    </div>
                  </section>

                  <section
                    className="data-section confidence-card"
                    aria-label="Interpretation confidence"
                  >
                    <div className="card-title-row">
                      <h3>Confidence</h3>
                      <strong>
                        {interpretation ? titleCase(interpretation.confidence) : "Not started"}
                      </strong>
                    </div>
                    <div
                      className="confidence-track"
                      role="progressbar"
                      aria-label="Interpretation confidence"
                      aria-valuemin={0}
                      aria-valuemax={100}
                      aria-valuenow={confidencePercent}
                      aria-valuetext={
                        interpretation
                          ? `${titleCase(interpretation.confidence)} confidence, ${confidencePercent}%`
                          : "Not started"
                      }
                    >
                      <div className="confidence-fill" style={{ width: `${confidencePercent}%` }} />
                    </div>
                    <div className="confidence-labels">
                      <span>0%</span>
                      <span>50%</span>
                      <span>100%</span>
                    </div>
                  </section>

                  {interpretation && showSigningPlan ? (
                    <section
                      className="data-section signing-plan-card"
                      aria-label="All word actions"
                      data-backlog-status={
                        sidePanelOnlyPlayback
                          ? "side_panel_queue"
                          : (realtimePlayback?.backlog.status ?? "realtime")
                      }
                    >
                      <div className="card-title-row">
                        <h3>
                          Word actions <span className="count-pill">{sourceWordCount}</span>
                        </h3>
                        <span>
                          {wordActionHistory.length} ordered · {displayedMotionBackedGlossCount}{" "}
                          motion · {modelBackedGlossCount} lexicon
                        </span>
                      </div>
                      {realtimePlayback ? (
                        <p className="model-status" aria-live="polite">
                          {formatRealtimeBacklog(realtimePlayback.backlog, sidePanelOnlyPlayback)}
                        </p>
                      ) : null}
                      <ol className="signing-plan">
                        {wordActionHistory.map(({ interpretationId, item }, index) => {
                          const step = findAvatarStepForWordAction(avatarQueue?.steps, item);
                          const hasStaticFallback = isMotionClipUnavailable(
                            step,
                            availableMotionClipIds,
                            motionLibraryState
                          );
                          return (
                            <li
                              key={getWordActionIdentity(interpretationId, item)}
                              data-word-action-index={index + 1}
                            >
                              <strong>{item.gloss}</strong>
                              <span>{item.token}</span>
                              <small
                                className={`plan-badge plan-badge--${
                                  hasStaticFallback ? "static" : item.fallback
                                }`}
                              >
                                {getWordActionLabel(
                                  item.fallback,
                                  item.isVerified,
                                  step,
                                  availableMotionClipIds,
                                  motionLibraryState
                                )}
                              </small>
                              {resolvedSettings.showGlossDebug && item.signAssetId ? (
                                <code>{item.signAssetId}</code>
                              ) : null}
                            </li>
                          );
                        })}
                      </ol>
                      {modelStatus ? (
                        <p className="model-status">
                          Glossary: {modelStatus.trainingDataset.displayName} ·{" "}
                          {modelStatus.trainingDataset.classCount} terms
                        </p>
                      ) : null}
                    </section>
                  ) : null}

                  {interpretation?.warnings.map((warning) => (
                    <p className="warning warning--compact" key={warning}>
                      <AlertTriangle size={15} /> {warning}
                    </p>
                  ))}

                  <div className="session-actions">
                    <button
                      className="report-button"
                      type="button"
                      onClick={() => setFeedbackOpen(true)}
                      disabled={!interpretation || !sessionId || !resolvedSettings.feedbackSharing}
                    >
                      <Flag size={16} />
                      Report wrong sign
                    </button>
                    <button
                      className="replay-button"
                      type="button"
                      disabled={!canReplayPreview}
                      onClick={replayAvatarQueue}
                    >
                      <RotateCcw size={16} />
                      Replay last segment
                    </button>
                  </div>
                  {!resolvedSettings.feedbackSharing ? (
                    <p className="fallback-text">Feedback sharing is off.</p>
                  ) : null}
                  {interpretation && !showAvatarOverlay ? (
                    <p className="fallback-text">
                      {sidePanelOnlyPlayback
                        ? "All avatar actions are playing in the side panel; the page overlay is hidden."
                        : "Avatar output is hidden by output settings."}
                    </p>
                  ) : null}
                </>
              ) : null}

              {activeDashboardSection === "captions" ? (
                showCaptions ? (
                  <section className="tab-content-section" aria-labelledby="captions-heading">
                    <div className="card-title-row">
                      <h3 id="captions-heading">
                        <Captions size={16} /> Live captions
                      </h3>
                      <span>{getCaptionLanguageLabel(resolvedSettings.captionLanguage)}</span>
                    </div>
                    <p
                      className="caption-box"
                      aria-live="polite"
                      style={{ fontSize: `${resolvedSettings.captionFontSize}px` }}
                    >
                      {isCaptionPaused
                        ? "Captions paused."
                        : (liveCaptionText ??
                          segment?.rawText ??
                          "Start interpretation to read visible captions.")}
                    </p>
                    <div className="caption-controls">
                      <button
                        type="button"
                        aria-label="Decrease caption text size"
                        onClick={() => void handleCaptionFontSizeChange(-2)}
                      >
                        A-
                      </button>
                      <span>{resolvedSettings.captionFontSize}px</span>
                      <button
                        type="button"
                        aria-label="Increase caption text size"
                        onClick={() => void handleCaptionFontSizeChange(2)}
                      >
                        A+
                      </button>
                      <button type="button" onClick={() => void togglePlaybackPause()}>
                        {isCaptionPaused ? <Play size={14} /> : <Pause size={14} />}
                        {isCaptionPaused ? "Resume captions" : "Pause captions"}
                      </button>
                    </div>
                  </section>
                ) : (
                  <div className="output-gate">
                    <Captions size={20} />
                    <strong>Captions are hidden</strong>
                    <p>
                      Avatar-only mode keeps text output off. Choose Auto or Captions to show it.
                    </p>
                  </div>
                )
              ) : null}

              {activeDashboardSection === "summary" ? (
                showSummary ? (
                  <section className="tab-content-section" aria-labelledby="summary-heading">
                    <div className="card-title-row">
                      <h3 id="summary-heading">Simplified meaning</h3>
                      <span>{titleCase(resolvedSettings.simplificationLevel)}</span>
                    </div>
                    <p className="summary-box">
                      {interpretation?.simplifiedText ||
                        "Start interpretation to generate meaning chunks."}
                    </p>
                    <div className="data-section data-section--flush">
                      <div className="card-title-row">
                        <h3>
                          Key points{" "}
                          <span className="count-pill">
                            {interpretation?.keyPoints.length ?? 0}
                          </span>
                        </h3>
                      </div>
                      <ul className="key-points">
                        {(
                          interpretation?.keyPoints ?? [
                            "Start interpretation to extract key points."
                          ]
                        ).map((point) => (
                          <li key={point}>{point}</li>
                        ))}
                      </ul>
                    </div>
                  </section>
                ) : (
                  <div className="output-gate">
                    <BookOpenText size={20} />
                    <strong>Summary is hidden</strong>
                    <p>
                      Avatar-only mode keeps meaning text off. Change the output mode to review it.
                    </p>
                  </div>
                )
              ) : null}

              {activeDashboardSection === "glossary" ? (
                showMeaningTools ? (
                  <section className="tab-content-section" aria-labelledby="glossary-heading">
                    <div className="card-title-row">
                      <h3 id="glossary-heading">
                        Detected terms{" "}
                        <span className="count-pill">
                          {interpretation?.detectedTerms.length ?? 0}
                        </span>
                      </h3>
                    </div>
                    <div className="term-list">
                      {(interpretation?.detectedTerms ?? []).map((term) => (
                        <button
                          type="button"
                          key={term}
                          onClick={() => void handleGlossarySearch(term)}
                        >
                          {term}
                        </button>
                      ))}
                    </div>
                    {!interpretation ? (
                      <p className="fallback-text">
                        Start interpretation to detect glossary terms.
                      </p>
                    ) : null}
                    <div className="glossary-panel" aria-live="polite">
                      <p>{glossaryStatus}</p>
                      {glossaryEntries.length ? (
                        <ul>
                          {glossaryEntries.slice(0, 3).map((entry) => (
                            <li key={entry.id}>
                              <strong>{entry.term}</strong>
                              <span>{entry.islGloss}</span>
                              <small>
                                {entry.reviewStatus === "approved"
                                  ? "Verified sign"
                                  : "Needs review"}
                              </small>
                            </li>
                          ))}
                        </ul>
                      ) : null}
                    </div>
                  </section>
                ) : (
                  <div className="output-gate">
                    <CircleHelp size={20} />
                    <strong>Glossary tools are hidden</strong>
                    <p>
                      Avatar-only mode keeps supporting text off. Choose another output mode to
                      review terms.
                    </p>
                  </div>
                )
              ) : null}
            </div>
          </section>
        </section>
      ) : null}

      {activeView === "settings" ? (
        <section className="app-view" aria-labelledby="settings-view-title">
          <div className="view-heading">
            <div>
              <h2 id="settings-view-title">Settings</h2>
              <p>Control text, avatar, language, and privacy preferences.</p>
            </div>
          </div>
          <section className="view-surface settings-panel" aria-label="Settings">
            <div className="settings-section">
              <h3>Interpretation</h3>
              <div className="settings-grid">
                <label className="field">
                  Output mode
                  <select
                    value={resolvedSettings.outputMode}
                    onChange={(event) =>
                      void handleOutputModeChange(event.target.value as OutputMode)
                    }
                  >
                    <option value="captions_avatar">Captions + ISL avatar</option>
                    <option value="captions_only">Captions only</option>
                    <option value="avatar_only">Avatar only</option>
                    <option value="summary_avatar">Summary + avatar</option>
                  </select>
                </label>
                <label className="field">
                  Caption language
                  <select
                    value={resolvedSettings.captionLanguage}
                    onChange={(event) =>
                      void handleCaptionLanguageChange(
                        event.target.value as Settings["captionLanguage"]
                      )
                    }
                  >
                    <option value="auto">Auto</option>
                    <option value="en">English</option>
                    <option value="hi">Hindi</option>
                    <option value="hinglish">Hinglish</option>
                  </select>
                </label>
                <label className="field">
                  Simplification
                  <select
                    value={resolvedSettings.simplificationLevel}
                    onChange={(event) =>
                      void handleSimplificationLevelChange(
                        event.target.value as Settings["simplificationLevel"]
                      )
                    }
                  >
                    <option value="light">Light</option>
                    <option value="standard">Standard</option>
                    <option value="strong">Strong</option>
                  </select>
                </label>
              </div>
            </div>

            <div className="settings-section">
              <h3>Avatar</h3>
              <div className="settings-grid">
                <label className="field">
                  Avatar position
                  <select
                    value={resolvedSettings.avatarPosition}
                    onChange={(event) =>
                      void handleAvatarPositionChange(
                        event.target.value as Settings["avatarPosition"]
                      )
                    }
                  >
                    <option value="bottom_right">Bottom right</option>
                    <option value="bottom_left">Bottom left</option>
                    <option value="top_right">Top right</option>
                    <option value="side_panel">Side panel only</option>
                  </select>
                </label>
                <label className="field">
                  Avatar size
                  <select
                    value={resolvedSettings.avatarSize}
                    onChange={(event) =>
                      void handleAvatarSizeChange(event.target.value as Settings["avatarSize"])
                    }
                  >
                    <option value="small">Small</option>
                    <option value="medium">Medium</option>
                    <option value="large">Large</option>
                  </select>
                </label>
                <label className="field range-field">
                  <span>
                    Signing pace{" "}
                    <strong>
                      {getSigningPaceLabel(resolvedSettings.avatarSpeed)} ·{" "}
                      {formatSpeed(resolvedSettings.avatarSpeed)}
                    </strong>
                  </span>
                  <input
                    type="range"
                    aria-label="Signing pace"
                    aria-valuetext={`${getSigningPaceLabel(resolvedSettings.avatarSpeed)}, ${formatSpeed(resolvedSettings.avatarSpeed)}`}
                    min="0.5"
                    max="2"
                    step="0.25"
                    value={resolvedSettings.avatarSpeed}
                    onChange={(event) => void handleAvatarSpeedChange(Number(event.target.value))}
                  />
                  <span className="range-scale" aria-hidden="true">
                    <span>Slower</span>
                    <span>Faster</span>
                  </span>
                </label>
              </div>
            </div>

            <div className="settings-section">
              <h3>Privacy and diagnostics</h3>
              <div className="toggle-list">
                <label className="checkbox-field settings-checkbox">
                  <span>
                    <strong>Privacy mode</strong>
                    <small>Use the most restrictive local handling available.</small>
                  </span>
                  <input
                    type="checkbox"
                    checked={resolvedSettings.privacyMode}
                    onChange={(event) =>
                      void handleBooleanSettingChange("privacyMode", event.target.checked)
                    }
                  />
                </label>
                <label className="checkbox-field settings-checkbox">
                  <span>
                    <strong>Feedback sharing</strong>
                    <small>Allow wrong-sign reports to be submitted for review.</small>
                  </span>
                  <input
                    type="checkbox"
                    checked={resolvedSettings.feedbackSharing}
                    onChange={(event) =>
                      void handleBooleanSettingChange("feedbackSharing", event.target.checked)
                    }
                  />
                </label>
                <label className="checkbox-field settings-checkbox">
                  <span>
                    <strong>Gloss debug</strong>
                    <small>Show sign asset identifiers in the signing plan.</small>
                  </span>
                  <input
                    type="checkbox"
                    checked={resolvedSettings.showGlossDebug}
                    onChange={(event) =>
                      void handleBooleanSettingChange("showGlossDebug", event.target.checked)
                    }
                  />
                </label>
              </div>
            </div>
          </section>
        </section>
      ) : null}

      {activeView === "history" ? (
        <section className="app-view" aria-labelledby="history-view-title">
          <div className="view-heading">
            <div>
              <h2 id="history-view-title">History</h2>
              <p>Summary-only session records stored in this browser.</p>
            </div>
            <span className="count-pill">{sessionHistory.length}</span>
          </div>
          <section className="view-surface history-card" aria-label="Local session history">
            <div className="history-controls">
              <label className="checkbox-field history-toggle">
                <span>
                  <strong>Save local history</strong>
                  <small>{resolvedSettings.saveHistory ? "Stored locally" : "Off"}</small>
                </span>
                <input
                  type="checkbox"
                  checked={resolvedSettings.saveHistory}
                  onChange={(event) => void handleSaveHistoryChange(event.target.checked)}
                />
              </label>
              <button
                type="button"
                className="secondary-button danger-button"
                onClick={handleClearLocalData}
              >
                <Trash2 size={15} /> Delete local data
              </button>
            </div>
            {sessionHistory.length ? (
              <ul className="history-list">
                {sessionHistory.map((item) => (
                  <li key={item.id}>
                    <div>
                      <strong>{item.sourceLabel}</strong>
                      <small>{titleCase(item.confidence)} confidence</small>
                    </div>
                    <span>{item.summary}</span>
                  </li>
                ))}
              </ul>
            ) : (
              <div className="empty-history">
                <HistoryIcon size={22} />
                <strong>No saved sessions</strong>
                <p>Enable local history to keep summary-only records.</p>
              </div>
            )}
          </section>
        </section>
      ) : null}

      {activeView === "about" ? (
        <section className="app-view" aria-labelledby="about-view-title">
          <div className="view-heading">
            <div>
              <h2 id="about-view-title">About</h2>
              <p>Local model readiness, privacy, and responsible-use details.</p>
            </div>
          </div>
          <section className="view-surface about-panel">
            <div className="about-section">
              <div className="card-title-row">
                <h3>Local models</h3>
                <span className="model-readiness">{registryDetails}</span>
              </div>
              {modelRegistry.length ? (
                <ul className="model-list">
                  {modelRegistry.map((model) => (
                    <li key={model.id}>
                      <div className="model-list__summary">
                        <strong>{model.trainingDataset.displayName}</strong>
                        <span>
                          {getModelEngineLabel(model.engine)}
                          {" · "}v{model.version}
                        </span>
                      </div>
                      <small className={getModelStateClassName(model)}>
                        {getModelStateLabel(model)}
                      </small>
                      {model.notes.length ? (
                        <ul className="model-notes">
                          {model.notes.map((note) => (
                            <li key={note}>{note}</li>
                          ))}
                        </ul>
                      ) : null}
                      {model.engine === "keypoint_transformer" ? (
                        <p className="model-disclaimer">
                          Local 64-frame pose-and-hand classifier. Low-confidence recognition falls
                          back to captions; dataset-derived avatar motion is not expert-certified
                          ISL. Live YouTube interpretation uses deterministic transcript-to-gloss
                          and motion-clip lookup; this classifier is exposed only through the local
                          keypoint inference API.
                        </p>
                      ) : model.engine === "keypoint_centroid" ? (
                        <p className="model-disclaimer">
                          Research prototype. It is not used for avatar motion.
                        </p>
                      ) : model.engine === "motion_library" ? (
                        <p className="model-disclaimer">
                          Local gated iSign isolated-word motion. It is available only for
                          noncommercial research, is not redistributed in the extension, and has not
                          been expert-reviewed. It improves word motion coverage but does not
                          establish continuous ISL grammar.
                        </p>
                      ) : model.engine === "text_to_pose_research" ? (
                        <p className="model-disclaimer">
                          Private text-to-pose research artifact with held-out geometric metrics and
                          ONNX parity validation. It remains disabled for live signing until
                          sentence-level Deaf/ISL expert review is completed.
                        </p>
                      ) : model.engine === "caption_boundary_planner" ? (
                        <p className="model-disclaimer">
                          Private text-only caption statistics can choose boundaries in long caption
                          chunks. The planner cannot remove or reorder source words and is not an
                          ISL grammar model.
                        </p>
                      ) : model.engine === "lexicon_pair_matcher" ? (
                        <p className="model-disclaimer">
                          Conservatively resolves high-confidence single-word caption typos while
                          preserving the original source text. It does not translate ISL grammar or
                          synthesize avatar motion.
                        </p>
                      ) : null}
                    </li>
                  ))}
                </ul>
              ) : (
                <p className="fallback-text">{modelRegistryStatus}</p>
              )}
            </div>

            <div className="about-section privacy-note">
              <ShieldCheck size={20} />
              <div>
                <h3>Privacy by design</h3>
                <p>
                  Raw audio and video are not stored. When history is enabled, only generated
                  summaries and basic session metadata are saved in Chrome local storage.
                </p>
              </div>
            </div>

            <div className="about-section">
              <h3>Responsible use</h3>
              <p className="safety-note">
                SignSaarthi AI is AI-assisted, may make mistakes, and is not a certified interpreter
                replacement. Use a qualified human interpreter for legal, medical, emergency, or
                high-stakes situations.
              </p>
            </div>
          </section>
        </section>
      ) : null}

      <nav className="bottom-nav" aria-label="Primary navigation">
        {primaryViews.map(({ id, label, icon: Icon }) => {
          const isActive = activeView === id;
          return (
            <button
              type="button"
              className={
                isActive ? "bottom-nav__item bottom-nav__item--active" : "bottom-nav__item"
              }
              aria-current={isActive ? "page" : undefined}
              key={id}
              onClick={() => setActiveView(id)}
            >
              <Icon size={18} />
              <span>{label}</span>
            </button>
          );
        })}
      </nav>

      <FeedbackDialog
        isOpen={feedbackOpen}
        isSubmitting={isSubmittingFeedback}
        onClose={closeFeedback}
        onSubmit={handleFeedback}
      />
    </main>
  );
}

function normalizeCaptionText(value?: string): string {
  return value?.replace(/\s+/g, " ").trim() ?? "";
}

function formatSpeed(speed: number): string {
  return `${speed.toFixed(2).replace(/\.?0+$/, "")}x`;
}

function getSigningPaceLabel(speed: number): string {
  if (speed <= 0.5) {
    return "Very slow";
  }
  if (speed <= 0.75) {
    return "Slow";
  }
  if (speed <= 1) {
    return "Comfortable";
  }
  if (speed <= 1.25) {
    return "Brisk";
  }
  return "Fast";
}

function getConnectionLabel(status: ConnectionStatus): string {
  if (status === "connected") {
    return "Connected";
  }
  if (status === "offline") {
    return "Offline";
  }
  if (status === "degraded") {
    return "Service error";
  }
  return "Checking";
}

function getActivePlatform(
  url: string,
  supportStatus: "supported" | "captions_unavailable" | "unsupported"
): ActivePlatform {
  try {
    const hostname = new URL(url).hostname;
    if (hostname === "youtube.com" || hostname.endsWith(".youtube.com")) {
      return {
        kind: "youtube",
        label: "YouTube",
        detail:
          supportStatus === "supported"
            ? "Live captions"
            : supportStatus === "captions_unavailable"
              ? "Captions unavailable"
              : "Unsupported page"
      };
    }
    if (hostname === "meet.google.com") {
      return { kind: "meet", label: "Google Meet", detail: "Detection only" };
    }
    if (hostname === "zoom.us" || hostname.endsWith(".zoom.us")) {
      return { kind: "zoom", label: "Zoom", detail: "Detection only" };
    }
    return { kind: "unsupported", label: "Other tab", detail: hostname };
  } catch {
    // Invalid page URLs are treated as local demo input.
  }
  return { kind: "demo", label: "Local demo", detail: "Demo transcript" };
}

function isLocalApiConnectionFailure(error: unknown): boolean {
  return (
    error instanceof Error &&
    /local signsaarthi api (?:is unavailable|did not respond in time)|failed to fetch|network(?:error| request failed)|load failed/i.test(
      error.message
    )
  );
}

function getConnectionStatusForError(error: unknown): "degraded" | "offline" {
  return isLocalApiConnectionFailure(error) ? "offline" : "degraded";
}

function getOperationErrorDetail(error: unknown, fallback: string): string {
  return error instanceof Error && error.message.trim() ? error.message : fallback;
}

function countDisplayWords(value: string): number {
  return value.match(/[\p{L}\p{M}\p{N}]+(?:['’-][\p{L}\p{M}\p{N}]+)*/gu)?.length ?? 0;
}

function countWordActionSourceTokens(history: readonly WordActionRecord[]): number {
  return history.reduce(
    (count, record) =>
      count + (record.item.sourceTokenIds?.length ?? countDisplayWords(record.item.token)),
    0
  );
}

function rescaleAvatarQueue(
  queue: ISLInterpretationResponse["avatarQueue"],
  requestedSpeed: number
): ISLInterpretationResponse["avatarQueue"] {
  const speed = Math.min(2, Math.max(0.5, requestedSpeed));
  const durationScale = queue.speed / speed;
  const steps = queue.steps.map((step) => ({
    ...step,
    durationMs: Math.max(1, Math.round(step.durationMs * durationScale))
  }));
  return {
    ...queue,
    id: `${queue.id}_speed_${String(speed).replace(".", "_")}`,
    speed,
    steps,
    backlog: planRealtimePlayback(steps, {
      baseSpeed: speed,
      historyStepCount: steps.length
    }).backlog,
    createdAt: new Date().toISOString()
  };
}

function mergeWordActionHistory(
  current: readonly WordActionRecord[],
  interpretationId: string,
  incomingItems: readonly ISLGlossItem[]
): WordActionRecord[] {
  const incoming = incomingItems.map((item) => ({ interpretationId, item }));
  const maxOverlap = Math.min(current.length, incoming.length);
  for (let overlap = maxOverlap; overlap > 0; overlap -= 1) {
    const currentOffset = current.length - overlap;
    const matches = incoming
      .slice(0, overlap)
      .every(
        (record, index) =>
          getWordActionIdentity(record.interpretationId, record.item) ===
          getWordActionIdentity(
            current[currentOffset + index]!.interpretationId,
            current[currentOffset + index]!.item
          )
      );
    if (matches) {
      return [...current, ...incoming.slice(overlap)];
    }
  }
  return [...current, ...incoming];
}

function getWordActionIdentity(interpretationId: string, item: ISLGlossItem): string {
  return item.sourceSegmentId && item.planningActionId
    ? `${item.sourceSegmentId}:${item.planningActionId}`
    : `${interpretationId}:${item.id}`;
}

function findAvatarStepForWordAction(
  steps: readonly ISLInterpretationResponse["avatarQueue"]["steps"][number][] | undefined,
  item: ISLGlossItem
): ISLInterpretationResponse["avatarQueue"]["steps"][number] | undefined {
  if (!steps) {
    return undefined;
  }
  if (item.sourceSegmentId && item.planningActionId) {
    return steps.find(
      (step) =>
        step.sourceSegmentId === item.sourceSegmentId &&
        step.planningActionId === item.planningActionId
    );
  }
  return steps.find((step) => step.glossItemId === item.id);
}

function formatRealtimeBacklog(
  backlog: NonNullable<ISLInterpretationResponse["avatarQueue"]["backlog"]>,
  sidePanelOnlyPlayback = false
): string {
  if (sidePanelOnlyPlayback) {
    return `Side panel playback · all ${backlog.historyStepCount} actions queued`;
  }
  if (backlog.status === "backlog") {
    return `Backlog · ${backlog.playbackStepCount} playing · ${backlog.deferredStepCount} retained in Word actions`;
  }
  if (backlog.status === "catching_up") {
    return `Catching up at ${backlog.catchUpSpeed.toFixed(2).replace(/\.?0+$/, "")}x · ${backlog.pendingStepCount} queued`;
  }
  return `Realtime · ${backlog.pendingStepCount} queued`;
}

function getConfidencePercent(interpretation?: ISLInterpretation): number {
  if (!interpretation) {
    return 0;
  }
  return interpretation.confidence === "high"
    ? 92
    : interpretation.confidence === "medium"
      ? 68
      : 32;
}

function titleCase(value: string): string {
  return value.charAt(0).toUpperCase() + value.slice(1);
}

function getModelEngineLabel(engine: ModelMetadata["engine"]): string {
  if (engine === "keypoint_transformer") {
    return "Temporal ISL classifier";
  }
  if (engine === "keypoint_centroid") {
    return "Research centroid fallback";
  }
  if (engine === "lexicon_pair_matcher") {
    return "Caption typo matcher";
  }
  if (engine === "motion_library") {
    return "Local iSign motion library";
  }
  if (engine === "text_to_pose_research") {
    return "iSign text-to-pose research";
  }
  if (engine === "caption_boundary_planner") {
    return "iSign caption boundary planner";
  }
  return "Lexicon ranker";
}

function isExperimentalModelEngine(engine: ModelMetadata["engine"]): boolean {
  return (
    engine === "keypoint_centroid" ||
    engine === "text_to_pose_research" ||
    engine === "caption_boundary_planner"
  );
}

function getModelStateClassName(model: ModelMetadata): string {
  const classes = ["model-state", `model-state--${model.status}`];
  if (model.status === "ready" && isExperimentalModelEngine(model.engine)) {
    classes.push("model-state--experimental");
  }
  return classes.join(" ");
}

function getModelStateLabel(model: ModelMetadata): string {
  if (model.status !== "ready") {
    return titleCase(model.status);
  }
  return isExperimentalModelEngine(model.engine) ? "Ready · Experimental" : "Ready";
}

function getSourceLabel(source: TranscriptSource, supportStatus: string): string {
  if (source === "youtube_captions") {
    return "YouTube captions";
  }
  if (supportStatus === "unsupported") {
    return "Mock transcript";
  }
  return "Demo transcript";
}

function getOutputModeLabel(mode: OutputMode): string {
  const labels: Record<OutputMode, string> = {
    captions_only: "Captions only",
    captions_avatar: "Captions + ISL avatar",
    avatar_only: "Avatar only",
    summary_avatar: "Summary + avatar"
  };
  return labels[mode];
}

function getCaptionLanguageLabel(language: Settings["captionLanguage"]): string {
  const labels: Record<Settings["captionLanguage"], string> = {
    auto: "Auto captions",
    en: "English",
    hi: "Hindi",
    hinglish: "Hinglish"
  };
  return labels[language];
}

function getLanguageHint(
  language: Settings["captionLanguage"]
): "en" | "hi" | "hinglish" | "unknown" {
  return language === "auto" ? "unknown" : language;
}

function shouldShowAvatarOverlay(settings: Settings): boolean {
  return settings.outputMode !== "captions_only" && settings.avatarPosition !== "side_panel";
}

function shouldUseCompleteSidePanelPlayback(settings: Settings): boolean {
  return settings.outputMode !== "captions_only" && settings.avatarPosition === "side_panel";
}

function getFallbackLabel(
  fallback: ISLInterpretation["glossSequence"][number]["fallback"],
  isVerified: boolean
): string {
  if (fallback === "none" && isVerified) {
    return "Verified";
  }
  if (fallback === "fingerspell") {
    return "Fingerspelling";
  }
  if (fallback === "caption") {
    return "Caption fallback";
  }
  return "Low confidence";
}

type AvatarStep = ISLInterpretationResponse["avatarQueue"]["steps"][number];

function isAvatarStepDynamicallyPlayable(
  step: AvatarStep,
  availableMotionClipIds: ReadonlySet<string>
): boolean {
  const motionClipId = step.motionClipId;
  return (
    (motionClipId !== undefined && availableMotionClipIds.has(motionClipId)) ||
    step.kind === "fingerspell" ||
    Boolean(step.fingerspellingSource)
  );
}

function isMotionClipUnavailable(
  step: AvatarStep | undefined,
  availableMotionClipIds: ReadonlySet<string>,
  motionLibraryState: MotionLibraryState
): boolean {
  return Boolean(
    step?.motionClipId &&
    motionLibraryState !== "loading" &&
    !availableMotionClipIds.has(step.motionClipId)
  );
}

function getMotionClipStatusLabel(
  step: AvatarStep,
  availableMotionClipIds: ReadonlySet<string>,
  motionLibraryState: MotionLibraryState
): string | undefined {
  if (!step.motionClipId) {
    return undefined;
  }
  if (!availableMotionClipIds.has(step.motionClipId)) {
    return motionLibraryState === "loading"
      ? "Checking motion clip"
      : "Static fallback · motion unavailable";
  }
  if (step.expertReviewed) {
    return "Expert-reviewed ISL motion";
  }
  return step.motionSource === "isign_research"
    ? "Local iSign research motion · review pending"
    : "INCLUDE motion · review pending";
}

function getAvatarStepStatusLabel(
  step: AvatarStep | undefined,
  availableMotionClipIds: ReadonlySet<string>,
  motionLibraryState: MotionLibraryState
): string {
  if (!step) {
    return "Waiting for an action";
  }
  return (
    getMotionClipStatusLabel(step, availableMotionClipIds, motionLibraryState) ??
    getAvatarStepKindLabel(step)
  );
}

function getAvatarStepKindLabel(step: AvatarStep): string {
  if (step.kind === "fingerspell") {
    return step.fingerspellingSource === "islrtc_official"
      ? "Official ISLRTC fingerspelling · avatar transfer review pending"
      : "Fingerspelling fallback";
  }
  if (step.kind === "caption") {
    return "Caption fallback";
  }
  if (step.kind === "unknown") {
    return "Text fallback";
  }
  return "Static fallback · motion unavailable";
}

function getWordActionLabel(
  fallback: ISLInterpretation["glossSequence"][number]["fallback"],
  isVerified: boolean,
  step: AvatarStep | undefined,
  availableMotionClipIds: ReadonlySet<string>,
  motionLibraryState: MotionLibraryState
): string {
  if (step) {
    const motionLabel = getMotionClipStatusLabel(step, availableMotionClipIds, motionLibraryState);
    if (motionLabel) {
      return motionLabel;
    }
  }
  if (step?.fingerspellingSource === "islrtc_official") {
    return "Official ISLRTC fingerspelling";
  }
  return getFallbackLabel(fallback, isVerified);
}
