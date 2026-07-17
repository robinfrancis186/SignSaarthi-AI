import { crx } from "@crxjs/vite-plugin";
import react from "@vitejs/plugin-react";
import { fileURLToPath, URL } from "node:url";
import { defineConfig } from "vite";
import { configDefaults } from "vitest/config";
import manifest from "./src/manifest";

export default defineConfig({
  plugins: [react(), crx({ manifest })],
  resolve: {
    alias: {
      "@signsaarthi/shared": fileURLToPath(
        new URL("../../packages/shared/src/index.ts", import.meta.url)
      ),
      "@signsaarthi/avatar-engine": fileURLToPath(
        new URL("../../packages/avatar-engine/src/index.ts", import.meta.url)
      )
    }
  },
  test: {
    exclude: [...configDefaults.exclude, "**/._*"],
    environment: "jsdom",
    setupFiles: ["src/test/setup.ts"],
    globals: true
  }
});
