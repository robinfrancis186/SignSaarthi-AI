import { buildServer } from "../apps/api/src/server";
import { chromium, type BrowserContext, type Page, type Worker } from "playwright";
import { mkdir, mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { fetchJson } from "./lib/script-utils";

const API_URL = "http://127.0.0.1:8787";
const FIXTURE_YOUTUBE_URL = "https://www.youtube.com/watch?v=signsaarthi-smoke";
const LIVE_YOUTUBE_URL =
  process.env["SIGNSAARTHI_YOUTUBE_URL"] ??
  "https://www.youtube.com/watch?v=kyQ0CRkYhy4&cc_load_policy=1&cc_lang_pref=en";
const FIXTURE_CAPTION_TEXT = "Artificial intelligence can help education in Mumbai.";
const outputDirectory = resolve(process.cwd(), "output/playwright");
const extensionDirectory = resolve(process.cwd(), "apps/extension/dist");
const headed = process.env["SIGNSAARTHI_E2E_HEADED"] === "true";
const liveYouTube = process.env["SIGNSAARTHI_REAL_YOUTUBE"] === "true";
const youtubeSmokeUrl = liveYouTube ? LIVE_YOUTUBE_URL : FIXTURE_YOUTUBE_URL;
const screenshotPrefix = liveYouTube ? "signsaarthi-youtube-live" : "signsaarthi-youtube";

type ChromeWorkerGlobal = typeof globalThis & {
  chrome: {
    runtime: { lastError?: { message?: string } };
    tabs: {
      query(
        queryInfo: Record<string, unknown>,
        callback: (tabs: Array<{ id?: number; url?: string }>) => void
      ): void;
      update(tabId: number, updateProperties: { active: boolean }, callback: () => void): void;
    };
  };
};

type OverlaySnapshot = {
  avatarClipId: string;
  avatarFrameIndex: number;
  avatarHeight: number;
  avatarState: string;
  avatarWidth: number;
  bottom: number;
  gloss: string;
  left: number;
  progress: string;
  right: number;
  top: number;
  viewportHeight: number;
  viewportWidth: number;
  width: number;
};

type CaptionPreparation = {
  availability: "visible";
  captionText: string;
  declaredTrackCount: number;
  sourceLabel: "YouTube captions";
};

let context: BrowserContext | undefined;
let ownedApi: ReturnType<typeof buildServer> | undefined;
const extensionErrors: string[] = [];
const profileDirectory = await mkdtemp(join(tmpdir(), "signsaarthi-extension-smoke-"));

try {
  ownedApi = await ensureApi();
  await mkdir(outputDirectory, { recursive: true });

  context = await chromium.launchPersistentContext(profileDirectory, {
    channel: "chromium",
    headless: !headed,
    viewport: { width: 1440, height: 900 },
    args: [
      "--autoplay-policy=no-user-gesture-required",
      "--disable-blink-features=AutomationControlled",
      "--mute-audio",
      `--disable-extensions-except=${extensionDirectory}`,
      `--load-extension=${extensionDirectory}`
    ]
  });

  const serviceWorker = await getExtensionServiceWorker(context);
  serviceWorker.on("console", (message) => {
    if (message.type() === "error") {
      extensionErrors.push(`service worker: ${message.text()}`);
    }
  });
  const extensionId = new URL(serviceWorker.url()).host;
  assert(extensionId.length > 0, "The production extension did not expose an extension id.");

  const youtubePage = context.pages()[0] ?? (await context.newPage());
  youtubePage.on("console", (message) => {
    if (
      message.type() === "error" &&
      /chrome-extension:|signsaarthi/i.test(`${message.location().url} ${message.text()}`)
    ) {
      extensionErrors.push(`content script: ${message.text()}`);
    }
  });
  if (!liveYouTube) {
    await youtubePage.route("https://www.youtube.com/**", async (route) => {
      if (route.request().resourceType() !== "document") {
        await route.abort();
        return;
      }

      await route.fulfill({
        status: 200,
        contentType: "text/html",
        body: youtubeFixture()
      });
    });
  }
  await youtubePage.goto(youtubeSmokeUrl, { waitUntil: "domcontentloaded", timeout: 45_000 });
  await youtubePage.waitForFunction(
    () => document.documentElement.dataset["signsaarthiContentScript"] === "ready",
    undefined,
    { timeout: 15_000 }
  );
  const captionPreparation: CaptionPreparation = liveYouTube
    ? await prepareLiveYouTubeCaption(youtubePage)
    : {
        availability: "visible",
        captionText: FIXTURE_CAPTION_TEXT,
        declaredTrackCount: 1,
        sourceLabel: "YouTube captions"
      };
  const captionText = captionPreparation.captionText;

  const activeYoutubeUrl = youtubePage.url();
  const youtubeTabId = await findTabId(serviceWorker, activeYoutubeUrl);
  const sidePanelPage = await context.newPage();
  sidePanelPage.on("console", (message) => {
    if (message.type() === "error") {
      extensionErrors.push(`side panel: ${message.text()}`);
    }
  });
  sidePanelPage.on("pageerror", (error) => {
    extensionErrors.push(`side panel: ${error.message}`);
  });
  await sidePanelPage.setViewportSize({ width: 420, height: 900 });
  await sidePanelPage.goto(`chrome-extension://${extensionId}/src/sidepanel/index.html`, {
    waitUntil: "domcontentloaded"
  });
  await sidePanelPage.getByRole("button", { name: "Start Interpretation" }).waitFor();

  await activateTab(serviceWorker, youtubeTabId);
  await clickButtonWithoutActivating(sidePanelPage, "Start Interpretation");
  await sidePanelPage
    .getByRole("button", { name: "Stop Interpretation" })
    .waitFor({ timeout: 15_000 });
  await sidePanelPage
    .getByText(captionPreparation.sourceLabel, { exact: true })
    .waitFor({ timeout: 15_000 });
  await sidePanelPage.locator("[data-word-action-index]").first().waitFor({ timeout: 15_000 });
  const wordActionTokens = await sidePanelPage
    .locator("[data-word-action-index] > span")
    .allTextContents();
  const representedWords = wordActionTokens.flatMap(
    (token) => token.toLocaleLowerCase().match(/[a-z]+/g) ?? []
  );
  const expectedWords = captionText.toLocaleLowerCase().match(/[a-z]+/g) ?? [];
  assert(
    JSON.stringify(representedWords) === JSON.stringify(expectedWords),
    `Word actions lost or reordered source words: ${JSON.stringify(wordActionTokens)}.`
  );
  const expectedStepCount = wordActionTokens.length;

  await youtubePage.waitForFunction(
    () => {
      const shadow = document.getElementById("signsaarthi-overlay-root")?.shadowRoot;
      const avatar = shadow?.querySelector<SVGSVGElement>("svg[data-motion-avatar]");
      const progress = shadow?.querySelector(".step-progress")?.textContent ?? "";
      return Boolean(
        avatar &&
        avatar.getBoundingClientRect().width > 0 &&
        Number(avatar.dataset.frameIndex ?? 0) > 0 &&
        /^\d+\/\d+/.test(progress)
      );
    },
    undefined,
    { timeout: 15_000 }
  );

  const overlay = await readOverlay(youtubePage);
  assert(overlay.avatarWidth > 0 && overlay.avatarHeight > 0, "Dynamic avatar SVG did not render.");
  assert(overlay.avatarFrameIndex > 0, "Dynamic avatar did not advance beyond its first frame.");
  assert(
    overlay.avatarClipId.startsWith("include-") ||
      overlay.avatarClipId.startsWith("islrtc-fingerspell-"),
    `Unexpected avatar motion source: ${overlay.avatarClipId}`
  );
  assert(
    readTotalActionCount(overlay.progress) === expectedStepCount,
    `Expected ${expectedStepCount} represented avatar actions in global queue progress, received ${overlay.progress}.`
  );
  const totalActionCount = readTotalActionCount(overlay.progress);
  assert(
    overlay.left <= 40,
    `Expected the default bottom-left placement, received x=${overlay.left}.`
  );
  assert(
    overlay.left >= 0 &&
      overlay.top >= 0 &&
      overlay.right <= overlay.viewportWidth + 0.5 &&
      overlay.bottom <= overlay.viewportHeight + 0.5,
    `Avatar overlay escaped the viewport: ${JSON.stringify({
      left: overlay.left,
      top: overlay.top,
      right: overlay.right,
      bottom: overlay.bottom,
      viewportWidth: overlay.viewportWidth,
      viewportHeight: overlay.viewportHeight
    })}.`
  );

  await clickButtonWithoutActivating(sidePanelPage, "Pause");
  const pausedFrame = await readAvatarFrame(youtubePage);
  await youtubePage.waitForTimeout(240);
  assert(
    (await readAvatarFrame(youtubePage)) === pausedFrame,
    "Pause did not freeze the dynamic avatar frame."
  );
  await clickButtonWithoutActivating(sidePanelPage, "Resume");
  await youtubePage.waitForFunction((frame) => {
    const avatar = document
      .getElementById("signsaarthi-overlay-root")
      ?.shadowRoot?.querySelector<SVGSVGElement>("svg[data-motion-avatar]");
    return Number(avatar?.dataset.frameIndex ?? 0) !== frame;
  }, pausedFrame);

  await youtubePage.screenshot({
    path: join(outputDirectory, `${screenshotPrefix}-overlay.png`),
    fullPage: !liveYouTube
  });
  await sidePanelPage.screenshot({
    path: join(outputDirectory, `${screenshotPrefix}-sidepanel.png`)
  });

  if (!liveYouTube) {
    await youtubePage.waitForFunction(
      (stepCount) => {
        const shadow = document.getElementById("signsaarthi-overlay-root")?.shadowRoot;
        const progress = shadow?.querySelector(".step-progress")?.textContent?.trim() ?? "";
        const status = shadow?.querySelector(".toolbar-title")?.textContent?.trim() ?? "";
        return progress.startsWith(`${stepCount}/${stepCount}`) && status.endsWith("Complete");
      },
      expectedStepCount,
      { timeout: 45_000 }
    );
  }

  await clickShadowButton(youtubePage, "minimize");
  const minimizedWidth = await readOverlayWidth(youtubePage);
  assert(
    minimizedWidth <= 222,
    `Minimize control did not collapse the overlay; width=${minimizedWidth}.`
  );
  await clickShadowButton(youtubePage, "minimize");
  await clickShadowButton(youtubePage, "replay");
  await youtubePage.waitForFunction(
    (stepCount) =>
      document
        .getElementById("signsaarthi-overlay-root")
        ?.shadowRoot?.querySelector(".step-progress")
        ?.textContent?.startsWith(`1/${stepCount}`),
    totalActionCount
  );

  let liveUpdateWordActionCount: number | undefined;
  if (liveYouTube && captionPreparation.availability === "visible") {
    await resumeLiveYouTube(youtubePage);
    await sidePanelPage.waitForFunction(
      (initialCount) => document.querySelectorAll("[data-word-action-index]").length > initialCount,
      expectedStepCount,
      { timeout: 30_000 }
    );
    liveUpdateWordActionCount = await sidePanelPage.locator("[data-word-action-index]").count();
    await pauseLiveYouTube(youtubePage);
  }

  await sidePanelPage.getByRole("button", { name: "Report wrong sign" }).click();
  await sidePanelPage
    .getByRole("textbox", { name: "Comment" })
    .fill("Automated production extension smoke test.");
  await sidePanelPage.getByRole("button", { name: "Submit feedback" }).click();
  await sidePanelPage.getByText("Feedback saved for reviewer follow-up.").waitFor();

  await activateTab(serviceWorker, youtubeTabId);
  await clickButtonWithoutActivating(sidePanelPage, "Stop Interpretation");
  await sidePanelPage.getByRole("button", { name: "Start Interpretation" }).waitFor();
  await youtubePage.waitForFunction(() => !document.getElementById("signsaarthi-overlay-root"));

  const health = await fetchJson<HealthResponse>(`${API_URL}/health`);
  assert(
    health.modelsReady && health.lexiconModelReady,
    "API lexicon was not ready during the browser smoke test."
  );
  assert(
    !health.recognitionModelReady,
    "Smoke test unexpectedly found a deployed recognition model."
  );
  assert(
    !health.rawAudioStored && !health.rawVideoStored,
    "Privacy invariant failed: raw media storage was reported."
  );
  assert(
    extensionErrors.length === 0,
    `Extension console errors were reported: ${extensionErrors.join(" | ")}`
  );

  console.log(
    JSON.stringify(
      {
        status: "passed",
        extensionId,
        mode: liveYouTube ? "live-youtube" : "deterministic-fixture",
        url: activeYoutubeUrl,
        source: captionPreparation.sourceLabel,
        captionAvailability: captionPreparation.availability,
        declaredCaptionTrackCount: captionPreparation.declaredTrackCount,
        captionText,
        liveUpdateWordActionCount,
        queue: overlay.progress,
        firstObservedGloss: overlay.gloss,
        avatar: {
          clipId: overlay.avatarClipId,
          frameIndex: overlay.avatarFrameIndex,
          state: overlay.avatarState,
          size: `${Math.round(overlay.avatarWidth)}x${Math.round(overlay.avatarHeight)}`
        },
        screenshots: [
          join(outputDirectory, `${screenshotPrefix}-overlay.png`),
          join(outputDirectory, `${screenshotPrefix}-sidepanel.png`)
        ]
      },
      null,
      2
    )
  );
} finally {
  await context?.close();
  await ownedApi?.close();
  await rm(profileDirectory, { recursive: true, force: true });
}

async function prepareLiveYouTubeCaption(page: Page): Promise<CaptionPreparation> {
  await dismissYouTubeConsent(page);
  await page.locator("video").waitFor({ state: "attached", timeout: 45_000 });
  await resumeLiveYouTube(page);
  await waitForYouTubeAd(page);
  await resumeLiveYouTube(page);
  const declaredTrackCount = await readDeclaredCaptionTrackCount(page);

  const captionsButton = page.locator(".ytp-subtitles-button");
  assert((await captionsButton.count()) === 1, "YouTube caption control was not available.");
  const captionsAvailable = await page
    .waitForFunction(
      (trackCount) => {
        const button = document.querySelector<HTMLButtonElement>(".ytp-subtitles-button");
        const accessibleState = `${button?.getAttribute("aria-label") ?? ""} ${
          button?.getAttribute("title") ?? ""
        }`;
        return Boolean(
          button &&
          !button.disabled &&
          (trackCount > 0 ||
            button.getAttribute("aria-pressed") === "true" ||
            !/unavailable/i.test(accessibleState))
        );
      },
      declaredTrackCount,
      { timeout: 15_000 }
    )
    .then(() => true)
    .catch(() => false);

  if (!captionsAvailable) {
    throw new Error(
      `Live YouTube smoke requires an enabled caption track; declared tracks: ${declaredTrackCount}.`
    );
  }

  if ((await captionsButton.getAttribute("aria-pressed")) !== "true") {
    await captionsButton.click();
  }

  const visibleCaptionAppeared = await page
    .waitForFunction(
      () =>
        Array.from(document.querySelectorAll<HTMLElement>(".ytp-caption-segment"))
          .filter((element) => {
            const style = getComputedStyle(element);
            return (
              style.display !== "none" &&
              style.visibility !== "hidden" &&
              style.opacity !== "0" &&
              element.getBoundingClientRect().width > 0
            );
          })
          .some((element) => Boolean(element.textContent?.trim())),
      undefined,
      { timeout: 30_000 }
    )
    .then(() => true)
    .catch(() => false);
  await pauseLiveYouTube(page);
  if (!visibleCaptionAppeared) {
    const diagnostic = await page.evaluate(() => {
      const video = document.querySelector<HTMLVideoElement>("video");
      const player = document.querySelector(".html5-video-player");
      return {
        adShowing: player?.classList.contains("ad-showing") ?? false,
        captionNodes: Array.from(document.querySelectorAll<HTMLElement>(".ytp-caption-segment")).map(
          (element) => ({
            text: element.textContent?.trim() ?? "",
            width: Math.round(element.getBoundingClientRect().width)
          })
        ),
        currentTime: video?.currentTime ?? null,
        duration: Number.isFinite(video?.duration) ? video?.duration : null,
        paused: video?.paused ?? null,
        subtitlesLabel:
          document.querySelector(".ytp-subtitles-button")?.getAttribute("aria-label") ?? null
      };
    });
    const diagnosticScreenshot = join(
      outputDirectory,
      `${screenshotPrefix}-caption-timeout.png`
    );
    await page.screenshot({ path: diagnosticScreenshot });
    throw new Error(
      `Live YouTube smoke did not observe visible caption text within 30 seconds; declared tracks: ${declaredTrackCount}; state: ${JSON.stringify(diagnostic)}; screenshot: ${diagnosticScreenshot}.`
    );
  }
  const captionText = await page.evaluate(() =>
    Array.from(document.querySelectorAll<HTMLElement>(".ytp-caption-segment"))
      .filter((element) => {
        const style = getComputedStyle(element);
        return (
          style.display !== "none" &&
          style.visibility !== "hidden" &&
          style.opacity !== "0" &&
          element.getBoundingClientRect().width > 0
        );
      })
      .map((element) => element.textContent?.trim())
      .filter((value): value is string => Boolean(value))
      .join(" ")
      .replace(/\s+/g, " ")
      .trim()
  );
  assert(captionText.length > 0, "YouTube did not expose a visible caption for interpretation.");
  return {
    availability: "visible",
    captionText,
    declaredTrackCount,
    sourceLabel: "YouTube captions"
  };
}

async function readDeclaredCaptionTrackCount(page: Page): Promise<number> {
  return page.evaluate(() => {
    const pageGlobal = globalThis as typeof globalThis & {
      ytInitialPlayerResponse?: {
        captions?: {
          playerCaptionsTracklistRenderer?: { captionTracks?: unknown[] };
        };
      };
    };
    return (
      pageGlobal.ytInitialPlayerResponse?.captions?.playerCaptionsTracklistRenderer?.captionTracks
        ?.length ?? 0
    );
  });
}

async function dismissYouTubeConsent(page: Page): Promise<void> {
  const rejectButton = page.getByRole("button", { name: "Reject all" });
  if ((await rejectButton.count()) === 1 && (await rejectButton.isVisible())) {
    await rejectButton.click();
  }
}

async function waitForYouTubeAd(page: Page): Promise<void> {
  const deadline = Date.now() + 60_000;
  let noAdSince = Date.now();
  let sawAd = false;
  while (Date.now() < deadline) {
    const skipButton = page.locator(
      ".ytp-ad-skip-button-modern, .ytp-ad-skip-button, .ytp-skip-ad-button"
    );
    const skipButtonCount = await skipButton.count();
    for (let index = 0; index < skipButtonCount; index += 1) {
      const candidate = skipButton.nth(index);
      if (await candidate.isVisible()) {
        await candidate.click();
        break;
      }
    }

    const adShowing = await page
      .locator(".html5-video-player")
      .evaluate((player) => player.classList.contains("ad-showing"));
    if (adShowing) {
      sawAd = true;
      noAdSince = Date.now();
    } else if (Date.now() - noAdSince >= (sawAd ? 3_000 : 10_000)) {
      return;
    }
    await page.waitForTimeout(250);
  }
  throw new Error("YouTube advertisement did not finish within 60 seconds.");
}

async function pauseLiveYouTube(page: Page): Promise<void> {
  await page.locator("video").evaluate((video) => {
    (video as HTMLVideoElement).pause();
  });
  await page.waitForFunction(() => document.querySelector<HTMLVideoElement>("video")?.paused);
}

async function resumeLiveYouTube(page: Page): Promise<void> {
  await page.locator("video").evaluate(async (video) => {
    const media = video as HTMLVideoElement;
    media.muted = true;
    await media.play();
  });
  await page.waitForFunction(
    () => document.querySelector<HTMLVideoElement>("video")?.paused === false
  );
}

async function ensureApi(): Promise<ReturnType<typeof buildServer> | undefined> {
  let existingHealth: HealthResponse | undefined;
  try {
    existingHealth = await fetchJson<HealthResponse>(`${API_URL}/health`);
  } catch {
    // Start an isolated local API only when the configured port is free.
  }

  if (existingHealth) {
    assert(
      existingHealth.status === "ok" &&
        existingHealth.modelsReady &&
        existingHealth.lexiconModelReady &&
        !existingHealth.recognitionModelReady &&
        !existingHealth.rawAudioStored &&
        !existingHealth.rawVideoStored,
      "Port 8787 is occupied by an incompatible SignSaarthi API. Stop it and rerun the smoke test."
    );
    return undefined;
  }

  const app = buildServer();
  await app.listen({ host: "127.0.0.1", port: 8787 });
  return app;
}

type HealthResponse = {
  status: string;
  modelsReady: boolean;
  lexiconModelReady: boolean;
  recognitionModelReady: boolean;
  rawAudioStored: boolean;
  rawVideoStored: boolean;
};

async function getExtensionServiceWorker(browserContext: BrowserContext): Promise<Worker> {
  return (
    browserContext.serviceWorkers()[0] ??
    browserContext.waitForEvent("serviceworker", { timeout: 15_000 })
  );
}

async function findTabId(serviceWorker: Worker, expectedUrl: string): Promise<number> {
  const tabId = await serviceWorker.evaluate((url) => {
    const chromeApi = (globalThis as unknown as ChromeWorkerGlobal).chrome;
    return new Promise<number | undefined>((resolvePromise) => {
      chromeApi.tabs.query({}, (tabs) => {
        resolvePromise(tabs.find((tab) => tab.url === url)?.id);
      });
    });
  }, expectedUrl);

  assert(tabId !== undefined, `Could not find the YouTube smoke tab for ${expectedUrl}.`);
  return tabId;
}

async function activateTab(serviceWorker: Worker, tabId: number): Promise<void> {
  await serviceWorker.evaluate((id) => {
    const chromeApi = (globalThis as unknown as ChromeWorkerGlobal).chrome;
    return new Promise<void>((resolvePromise, rejectPromise) => {
      chromeApi.tabs.update(id, { active: true }, () => {
        const error = chromeApi.runtime.lastError?.message;
        if (error) {
          rejectPromise(new Error(error));
          return;
        }
        resolvePromise();
      });
    });
  }, tabId);
}

async function clickButtonWithoutActivating(page: Page, buttonName: string): Promise<void> {
  await page.evaluate((name) => {
    const button = [...document.querySelectorAll<HTMLButtonElement>("button")].find(
      (candidate) => candidate.textContent?.trim() === name
    );
    if (!button) {
      throw new Error(`Button not found: ${name}`);
    }
    button.click();
  }, buttonName);
}

async function clickShadowButton(page: Page, action: string): Promise<void> {
  await page.evaluate((buttonAction) => {
    const button = document
      .getElementById("signsaarthi-overlay-root")
      ?.shadowRoot?.querySelector<HTMLButtonElement>(`[data-action="${buttonAction}"]`);
    if (!button) {
      throw new Error(`Overlay action not found: ${buttonAction}`);
    }
    button.click();
  }, action);
}

async function readOverlay(page: Page): Promise<OverlaySnapshot> {
  return page.evaluate(() => {
    const shadow = document.getElementById("signsaarthi-overlay-root")?.shadowRoot;
    const overlayElement = shadow?.querySelector<HTMLElement>(".overlay");
    const avatar = shadow?.querySelector<SVGSVGElement>("svg[data-motion-avatar]");
    if (!shadow || !overlayElement || !avatar) {
      throw new Error("Avatar overlay is incomplete.");
    }
    const bounds = overlayElement.getBoundingClientRect();
    const avatarBounds = avatar.getBoundingClientRect();
    return {
      avatarClipId: avatar.dataset.clipId ?? "",
      avatarFrameIndex: Number(avatar.dataset.frameIndex ?? 0),
      avatarHeight: avatarBounds.height,
      avatarState: avatar.dataset.state ?? "",
      avatarWidth: avatarBounds.width,
      bottom: bounds.bottom,
      gloss: shadow.querySelector(".gloss-strip")?.textContent?.trim() ?? "",
      left: bounds.left,
      progress: shadow.querySelector(".step-progress")?.textContent?.trim() ?? "",
      right: bounds.right,
      top: bounds.top,
      viewportHeight: window.innerHeight,
      viewportWidth: window.innerWidth,
      width: bounds.width
    };
  });
}

async function readOverlayWidth(page: Page): Promise<number> {
  return page.evaluate(() => {
    const overlayElement = document
      .getElementById("signsaarthi-overlay-root")
      ?.shadowRoot?.querySelector<HTMLElement>(".overlay");
    if (!overlayElement) {
      throw new Error("Avatar overlay is missing.");
    }
    return overlayElement.getBoundingClientRect().width;
  });
}

async function readAvatarFrame(page: Page): Promise<number> {
  return page.evaluate(() => {
    const avatar = document
      .getElementById("signsaarthi-overlay-root")
      ?.shadowRoot?.querySelector<SVGSVGElement>("svg[data-motion-avatar]");
    if (!avatar) {
      throw new Error("Dynamic avatar is missing.");
    }
    return Number(avatar.dataset.frameIndex ?? 0);
  });
}

function assert(condition: unknown, message: string): asserts condition {
  if (!condition) {
    throw new Error(message);
  }
}

function readTotalActionCount(progress: string): number {
  const match = progress.match(/^\d+\/(\d+)/);
  return Number(match?.[1] ?? 0);
}

function youtubeFixture(): string {
  return `<!doctype html>
    <html lang="en">
      <head>
        <meta charset="utf-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <title>SignSaarthi YouTube smoke video</title>
        <style>
          body { margin: 0; min-height: 100vh; color: #fff; background: #0f0f0f; font: 16px system-ui, sans-serif; }
          header { height: 64px; display: flex; align-items: center; padding: 0 28px; border-bottom: 1px solid #303030; }
          main { max-width: 1080px; margin: 28px auto; padding: 0 24px; }
          .video { aspect-ratio: 16 / 9; display: grid; place-items: center; background: #050914; border: 1px solid #263047; }
          .ytp-caption-segment { max-width: 760px; padding: 10px 16px; background: rgba(0, 0, 0, 0.76); font-size: 22px; }
        </style>
      </head>
      <body>
        <header><strong>YouTube · deterministic extension smoke page</strong></header>
        <main>
          <section class="video" aria-label="Video player">
            <div class="ytp-caption-segment">${FIXTURE_CAPTION_TEXT}</div>
          </section>
          <h1>SignSaarthi accessibility verification</h1>
        </main>
      </body>
    </html>`;
}
