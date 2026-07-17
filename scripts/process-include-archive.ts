import { createReadStream, existsSync, mkdirSync, readFileSync, unlinkSync } from "node:fs";
import { resolve } from "node:path";
import { spawnSync } from "node:child_process";
import { createHash } from "node:crypto";
import { bytesToGiB, getFreeBytes, readBooleanEnv } from "./lib/script-utils";

type ZenodoManifest = {
  files: Array<{
    key: string;
    size: number;
    checksum?: string;
    url: string;
  }>;
};

const manifestPath = resolve(
  process.cwd(),
  process.env["SIGNSAARTHI_ZENODO_MANIFEST"] ?? "data/isl/include-zenodo-files.json"
);
const pattern = process.argv[2] ?? process.env["SIGNSAARTHI_INCLUDE_ARCHIVE_PATTERN"];
const rawDataDir = resolve(
  process.cwd(),
  process.env["SIGNSAARTHI_RAW_DATA_DIR"] ?? "data/raw/include"
);
const combinedPath = resolve(
  process.cwd(),
  process.env["SIGNSAARTHI_COMBINED_KEYPOINTS"] ?? "data/isl/keypoints.include-combined.json"
);
const pythonBin = process.env["SIGNSAARTHI_PYTHON_BIN"] ?? ".venv-mediapipe/bin/python";
const deleteArchiveAfterKeypoints = readBooleanEnv("SIGNSAARTHI_DELETE_ARCHIVE_AFTER_KEYPOINTS");

if (!pattern) {
  throw new Error(
    "Provide an archive pattern as an argument or SIGNSAARTHI_INCLUDE_ARCHIVE_PATTERN."
  );
}

const manifest = JSON.parse(readFileSync(manifestPath, "utf8")) as ZenodoManifest;
const file =
  manifest.files.find((entry) => entry.key === pattern) ??
  manifest.files.find((entry) => entry.key.includes(pattern));
if (!file) {
  throw new Error(`No INCLUDE Zenodo archive matched ${pattern} in ${manifestPath}`);
}
if (!file.key.endsWith(".zip")) {
  throw new Error(`${file.key} is not a zip archive.`);
}

mkdirSync(rawDataDir, { recursive: true });
const archivePath = resolve(rawDataDir, file.key);
const slug = file.key
  .replace(/\.zip$/i, "")
  .replace(/[^a-z0-9]+/giu, "-")
  .replace(/^-|-$/gu, "")
  .toLowerCase();
const archiveMetadataPath = resolve(process.cwd(), `data/isl/include-${slug}-auto-metadata.json`);
const keypointsPath = resolve(process.cwd(), `data/isl/keypoints.include-${slug}.json`);

if (!existsSync(archivePath)) {
  const freeBytes = getFreeBytes(rawDataDir);
  if (file.size > freeBytes * 0.9) {
    throw new Error(
      `Not enough free space for ${file.key}: need ${bytesToGiB(file.size)} GiB, free ${bytesToGiB(freeBytes)} GiB.`
    );
  }
  run("curl", ["-L", "--continue-at", "-", "--fail", "--output", archivePath, file.url]);
}

await verifyChecksum(archivePath, file.checksum);
run("pnpm", ["dataset:include:archive-metadata", archivePath, archiveMetadataPath]);
run("pnpm", ["keypoints:include:extract"], {
  SIGNSAARTHI_PYTHON_BIN: pythonBin,
  SIGNSAARTHI_INCLUDE_METADATA: archiveMetadataPath,
  SIGNSAARTHI_INCLUDE_ZIP_ARCHIVE: archivePath,
  SIGNSAARTHI_KEYPOINTS_OUTPUT: keypointsPath
});
run("pnpm", ["keypoints:combine", combinedPath, combinedPath, keypointsPath]);
run("pnpm", ["model:video:train", combinedPath, "data/models/isl-keypoint-model.json"]);
run("pnpm", [
  "model:video:evaluate",
  combinedPath,
  "data/models/isl-keypoint-model.json",
  "data/models/isl-keypoint-evaluation.json"
]);

if (deleteArchiveAfterKeypoints) {
  unlinkSync(archivePath);
  console.log(`Deleted processed archive ${archivePath}`);
}

console.log(`Processed INCLUDE archive ${file.key}; keypoints: ${keypointsPath}`);

function run(command: string, args: string[], env: Record<string, string> = {}): void {
  const result = spawnSync(command, args, {
    cwd: process.cwd(),
    stdio: "inherit",
    env: { ...process.env, ...env }
  });
  if (result.status !== 0) {
    throw new Error(
      `${command} ${args.join(" ")} failed with exit code ${result.status ?? "unknown"}`
    );
  }
}

async function verifyChecksum(filePath: string, checksum?: string): Promise<void> {
  if (!checksum) {
    console.warn(
      `No checksum available for ${filePath}; continuing without archive hash verification.`
    );
    return;
  }

  const [algorithm, expectedHash] = checksum.split(":");
  if (!algorithm || !expectedHash) {
    throw new Error(`Unsupported checksum format: ${checksum}`);
  }
  if (algorithm !== "md5") {
    throw new Error(`Unsupported checksum algorithm: ${algorithm}`);
  }

  const actualHash = await hashFile(filePath, algorithm);
  if (actualHash !== expectedHash.toLowerCase()) {
    unlinkSync(filePath);
    throw new Error(
      `Checksum mismatch for ${filePath}: expected ${expectedHash}, got ${actualHash}. Deleted corrupt archive.`
    );
  }
  console.log(`Verified ${algorithm} checksum for ${filePath}`);
}

async function hashFile(filePath: string, algorithm: string): Promise<string> {
  const hash = createHash(algorithm);
  await new Promise<void>((resolvePromise, reject) => {
    const stream = createReadStream(filePath);
    stream.on("data", (chunk) => hash.update(chunk));
    stream.on("error", reject);
    stream.on("end", resolvePromise);
  });
  return hash.digest("hex");
}
