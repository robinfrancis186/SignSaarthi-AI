import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { PopupApp } from "./PopupApp";

describe("PopupApp", () => {
  beforeEach(() => {
    vi.unstubAllGlobals();
  });

  afterEach(() => {
    cleanup();
    vi.unstubAllGlobals();
  });

  it("detects a YouTube tab and opens its side panel", async () => {
    const open = vi.fn(() => Promise.resolve());
    vi.stubGlobal("chrome", {
      tabs: {
        query: vi.fn(() =>
          Promise.resolve([{ id: 42, url: "https://www.youtube.com/watch?v=demo" }])
        )
      },
      sidePanel: { open }
    });

    render(<PopupApp />);

    expect(await screen.findByText("YouTube ready")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: /open side panel/i }));
    await waitFor(() => {
      expect(open).toHaveBeenCalledWith({ tabId: 42 });
    });
  });

  it("labels unsupported pages as demo mode", async () => {
    vi.stubGlobal("chrome", {
      tabs: {
        query: vi.fn(() => Promise.resolve([{ id: 7, url: "https://example.com" }]))
      },
      sidePanel: { open: vi.fn(() => Promise.resolve()) }
    });

    render(<PopupApp />);

    expect(await screen.findByText("Demo mode")).toBeInTheDocument();
    expect(screen.getByText(/open a youtube video/i)).toBeInTheDocument();
  });

  it("exits the loading state with recovery guidance when tab lookup fails", async () => {
    vi.stubGlobal("chrome", {
      tabs: {
        query: vi.fn(() => Promise.reject(new Error("Tabs permission unavailable")))
      },
      sidePanel: { open: vi.fn(() => Promise.resolve()) }
    });

    render(<PopupApp />);

    expect(await screen.findByText("Tab check failed")).toBeInTheDocument();
    expect(screen.queryByText("Checking tab")).not.toBeInTheDocument();
    expect(screen.getByText(/reopen the popup or use the built-in demo transcript/i)).toBeInTheDocument();
  });

  it("reports an empty active-tab result and does not attempt to open the panel", async () => {
    const query = vi.fn(() => Promise.resolve([]));
    const open = vi.fn(() => Promise.resolve());
    vi.stubGlobal("chrome", {
      tabs: { query },
      sidePanel: { open }
    });

    render(<PopupApp />);

    expect(await screen.findByText("No active tab")).toBeInTheDocument();
    expect(screen.queryByText("Checking tab")).not.toBeInTheDocument();
    expect(screen.getByText(/open a youtube video in this window/i)).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: /open side panel/i }));
    await waitFor(() => expect(query).toHaveBeenCalledTimes(2));
    expect(open).not.toHaveBeenCalled();
  });
});
