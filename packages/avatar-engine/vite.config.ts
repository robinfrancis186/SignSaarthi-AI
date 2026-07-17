import { fileURLToPath, URL } from "node:url";
import { configDefaults, defineConfig } from "vitest/config";

export default defineConfig({
  resolve: {
    alias: {
      "@signsaarthi/shared": fileURLToPath(new URL("../shared/src/index.ts", import.meta.url))
    }
  },
  test: {
    exclude: [...configDefaults.exclude, "**/._*"],
    environment: "node"
  }
});
