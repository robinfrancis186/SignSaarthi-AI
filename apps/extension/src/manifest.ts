import type { ManifestV3Export } from "@crxjs/vite-plugin";

const apiBaseUrl = process.env["VITE_SIGNSAARTHI_API_BASE_URL"] ?? "http://127.0.0.1:8787";
const apiHostPermission = `${new URL(apiBaseUrl).origin}/*`;

const manifest: ManifestV3Export = {
  manifest_version: 3,
  name: "SignSaarthi AI",
  description: "AI-assisted Indian Sign Language accessibility companion for online video.",
  version: "0.1.0",
  key: "MIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8AMIIBCgKCAQEA2u27yfLDTQKXIoZf0+MaKcMZJI2O+H36wSEcMyAjFvF+2zdQ9Msmyag3xY2c3vdK4+QCBFLJ7dYEzy9Jz2PXppInrnWDJEXOT1qrsK/WvX4zGHQxjCC0Q3unl9kgJT6eFXRgks/fis4nTNmvQzqiaMbZ3pziJplRgbUvd8w4GZZ8b0/pGql3B0s3LB97S0pCx4w/Ci/4WcVSnigsddxehF+2f3YejpaN9ujZ1xjfqhzOHis6KEqTLY5hOw61E13gGk8TV9NCKWp6QDJkiZ5cWBEqEzoPnwuw5/oU5XTtO+31FTvP/+ru51Ab8p5b8MZIz8Z6YLRjc3iPbUeBhQDYEwIDAQAB",
  action: {
    default_title: "Open SignSaarthi AI",
    default_popup: "src/popup/index.html"
  },
  side_panel: {
    default_path: "src/sidepanel/index.html"
  },
  background: {
    service_worker: "src/background/serviceWorker.ts",
    type: "module"
  },
  content_scripts: [
    {
      matches: ["https://www.youtube.com/*"],
      js: ["src/content/contentScript.ts"],
      run_at: "document_idle"
    }
  ],
  host_permissions: ["https://www.youtube.com/*", apiHostPermission],
  permissions: ["sidePanel", "storage", "activeTab", "scripting"],
  minimum_chrome_version: "116",
  content_security_policy: {
    extension_pages: "script-src 'self'; object-src 'self'"
  }
};

export default manifest;
