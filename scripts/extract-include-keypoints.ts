import { existsSync, mkdirSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { spawnSync } from "node:child_process";

const metadataPath = resolve(process.cwd(), process.env["SIGNSAARTHI_INCLUDE_METADATA"] ?? "data/isl/include-metadata.json");
const zipArchive = process.env["SIGNSAARTHI_INCLUDE_ZIP_ARCHIVE"]
  ? resolve(process.cwd(), process.env["SIGNSAARTHI_INCLUDE_ZIP_ARCHIVE"])
  : undefined;
const videoRoot = zipArchive
  ? undefined
  : resolve(process.cwd(), process.env["SIGNSAARTHI_INCLUDE_VIDEO_ROOT"] ?? "data/raw/include");
const outputPath = resolve(process.cwd(), process.env["SIGNSAARTHI_KEYPOINTS_OUTPUT"] ?? "data/isl/keypoints.include.json");
const pythonBin = process.env["SIGNSAARTHI_PYTHON_BIN"] ?? "python3";
const splits = (process.env["SIGNSAARTHI_KEYPOINT_SPLITS"] ?? "")
  .split(",")
  .map((split) => split.trim())
  .filter(Boolean);
const limit = process.env["SIGNSAARTHI_KEYPOINT_LIMIT"];

if (!existsSync(metadataPath)) {
  throw new Error(`Missing INCLUDE metadata at ${metadataPath}. Run pnpm dataset:include:all-metadata first.`);
}

if (zipArchive && !existsSync(zipArchive)) {
  throw new Error(`Missing INCLUDE zip archive at ${zipArchive}. Download the Zenodo archive first.`);
}

if (videoRoot && !existsSync(videoRoot)) {
  throw new Error(
    `Missing INCLUDE extracted video root at ${videoRoot}. Download/unzip Zenodo archives outside git, then set SIGNSAARTHI_INCLUDE_VIDEO_ROOT.`
  );
}

mkdirSync(dirname(outputPath), { recursive: true });

const args = [
  "scripts/extract_video_keypoints.py",
  "--metadata",
  metadataPath,
  "--output",
  outputPath
];
if (zipArchive) {
  args.push("--zip-archive", zipArchive);
} else if (videoRoot) {
  args.push("--video-root", videoRoot);
}
for (const split of splits) {
  args.push("--split", split);
}
if (limit) {
  args.push("--limit", limit);
}

const result = spawnSync(pythonBin, args, { stdio: "inherit" });
if (result.status !== 0) {
  throw new Error(`MediaPipe keypoint extraction failed with exit code ${result.status ?? "unknown"}.`);
}
