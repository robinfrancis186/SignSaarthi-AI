import { readFileSync, renameSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";
import { videoModelArtifactSchema } from "../packages/shared/src/index";

const artifactPath = resolve(
  process.cwd(),
  process.argv[2] ?? "data/models/isl-keypoint-model.json"
);
const temporaryPath = `${artifactPath}.tmp`;
const repaired = videoModelArtifactSchema.parse(
  JSON.parse(readFileSync(artifactPath, "utf8"))
);

writeFileSync(temporaryPath, `${JSON.stringify(repaired, null, 2)}\n`);
renameSync(temporaryPath, artifactPath);

console.log(`Repaired semantic gloss metadata in ${artifactPath}`);
