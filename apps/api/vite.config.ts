import { fileURLToPath, URL } from "node:url";
import { configDefaults, defineConfig } from "vitest/config";

export default defineConfig({
  resolve: {
    alias: {
      "@signsaarthi/shared": fileURLToPath(
        new URL("../../packages/shared/src/index.ts", import.meta.url)
      ),
      "@signsaarthi/avatar-engine": fileURLToPath(
        new URL("../../packages/avatar-engine/src/index.ts", import.meta.url)
      ),
      "@signsaarthi/isl-engine": fileURLToPath(
        new URL("../../packages/isl-engine/src/index.ts", import.meta.url)
      ),
      "@signsaarthi/isl-model": fileURLToPath(
        new URL("../../packages/isl-model/src/index.ts", import.meta.url)
      ),
      "@signsaarthi/isl-video-model": fileURLToPath(
        new URL("../../packages/isl-video-model/src/index.ts", import.meta.url)
      )
    }
  },
  test: {
    exclude: [...configDefaults.exclude, "**/._*"],
    environment: "node"
  }
});
