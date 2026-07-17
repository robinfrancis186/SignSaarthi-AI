import { Captions, ExternalLink, ShieldCheck } from "lucide-react";
import type { ReactElement } from "react";
import { useEffect, useState } from "react";
import { BrandMark } from "../sidepanel/components/BrandMark";

const DEMO_MESSAGE = "Open a YouTube video for captions, or use the built-in demo transcript.";
const NO_ACTIVE_TAB_MESSAGE = "Open a YouTube video in this window, then reopen SignSaarthi AI.";
const TAB_CHECK_FAILED_MESSAGE =
  "Chrome could not check the active tab. Reopen the popup or use the built-in demo transcript.";

export function PopupApp(): ReactElement {
  const [message, setMessage] = useState("Ready for YouTube or demo transcript mode.");
  const [tabStatus, setTabStatus] = useState("Checking tab");

  useEffect(() => {
    const tabs = globalThis.chrome?.tabs;
    if (!tabs?.query) {
      setTabStatus("Demo mode");
      setMessage(DEMO_MESSAGE);
      return;
    }

    let active = true;
    async function checkActiveTab(): Promise<void> {
      try {
        const [tab] = await tabs.query({ active: true, currentWindow: true });
        if (!active) {
          return;
        }
        if (!tab) {
          setTabStatus("No active tab");
          setMessage(NO_ACTIVE_TAB_MESSAGE);
          return;
        }

        const isYouTube = tab.url?.startsWith("https://www.youtube.com/") ?? false;
        setTabStatus(isYouTube ? "YouTube ready" : "Demo mode");
        setMessage(
          isYouTube ? "Visible YouTube captions will be used when available." : DEMO_MESSAGE
        );
      } catch {
        if (active) {
          setTabStatus("Tab check failed");
          setMessage(TAB_CHECK_FAILED_MESSAGE);
        }
      }
    }

    void checkActiveTab();
    return () => {
      active = false;
    };
  }, []);

  async function openPanel(): Promise<void> {
    const tabs = globalThis.chrome?.tabs;
    const sidePanel = globalThis.chrome?.sidePanel;
    if (!tabs?.query || !sidePanel?.open) {
      setMessage("Open SignSaarthi AI in Chrome, then use Chrome's side panel menu.");
      return;
    }

    let tab: chrome.tabs.Tab | undefined;
    try {
      [tab] = await tabs.query({ active: true, currentWindow: true });
    } catch {
      setTabStatus("Tab check failed");
      setMessage(TAB_CHECK_FAILED_MESSAGE);
      return;
    }

    if (typeof tab?.id !== "number") {
      setTabStatus("No active tab");
      setMessage(NO_ACTIVE_TAB_MESSAGE);
      return;
    }

    try {
      await sidePanel.open({ tabId: tab.id });
    } catch {
      setMessage("Open the side panel from Chrome's side panel menu.");
    }
  }

  return (
    <main className="popup-shell">
      <header className="popup-header">
        <BrandMark />
        <div>
          <h1>SignSaarthi AI</h1>
          <p>{tabStatus}</p>
        </div>
      </header>
      <button className="popup-primary" type="button" onClick={() => void openPanel()}>
        <ExternalLink size={17} />
        Open Side Panel
      </button>
      <div className="popup-status">
        <p><Captions size={15} /> Captions and mock transcript first.</p>
        <p><ShieldCheck size={15} /> No hidden capture. No raw audio storage.</p>
      </div>
      <p className="popup-message">{message}</p>
    </main>
  );
}
