import { closeSync, existsSync, mkdirSync, openSync, statSync, writeFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { spawnSync } from "node:child_process";
import { bytesToGiB, fetchJson, getFreeBytes, readBooleanEnv, readNumberEnv } from "./lib/script-utils";

type ZenodoRecord = {
  id: number;
  title?: string;
  files: Array<{
    key: string;
    size: number;
    links: {
      self: string;
    };
    checksum?: string;
  }>;
};

const rawDataDir = resolve(process.cwd(), process.env["SIGNSAARTHI_RAW_DATA_DIR"] ?? "data/raw/include");
const manifestPath = resolve(process.cwd(), process.argv[2] ?? "data/isl/include-zenodo-files.json");
const dryRun = readBooleanEnv("SIGNSAARTHI_DRY_RUN", true);
const filePattern = process.env["SIGNSAARTHI_DOWNLOAD_PATTERN"];
const downloadLimit = readNumberEnv("SIGNSAARTHI_DOWNLOAD_LIMIT", Number.POSITIVE_INFINITY);
mkdirSync(dirname(rawDataDir), { recursive: true });
const freeBytes = getFreeBytes(dirname(rawDataDir));

const record = await fetchJson<ZenodoRecord>("https://zenodo.org/api/records/4010759");
const files = record.files
  .filter((file) => !filePattern || file.key.toLowerCase().includes(filePattern.toLowerCase()))
  .slice(0, downloadLimit);
const totalBytes = files.reduce((sum, file) => sum + file.size, 0);

mkdirSync(dirname(manifestPath), { recursive: true });
writeFileSync(
  manifestPath,
  `${JSON.stringify(
    {
      schemaVersion: 1,
      generatedAt: new Date().toISOString(),
      source: {
        recordId: record.id,
        title: record.title ?? "INCLUDE: A Large Scale Dataset for Indian Sign Language Recognition",
        url: "https://zenodo.org/records/4010759",
        apiUrl: "https://zenodo.org/api/records/4010759"
      },
      dryRun,
      rawDataDir,
      selectedFileCount: files.length,
      selectedBytes: totalBytes,
      freeBytes,
      files: files.map((file) => ({
        key: file.key,
        size: file.size,
        sizeGiB: bytesToGiB(file.size),
        checksum: file.checksum,
        url: file.links.self
      }))
    },
    null,
    2
  )}\n`
);

console.log(
  `INCLUDE Zenodo plan: ${files.length} file(s), ${bytesToGiB(totalBytes)} GiB selected, ${bytesToGiB(
    freeBytes
  )} GiB free. Manifest: ${manifestPath}`
);

if (dryRun) {
  console.log("Dry run only. Set SIGNSAARTHI_DRY_RUN=false to download archives.");
  process.exit(0);
}

if (totalBytes > freeBytes * 0.9) {
  throw new Error(
    `Not enough free disk for selected INCLUDE archives: need ${bytesToGiB(totalBytes)} GiB, free ${bytesToGiB(
      freeBytes
    )} GiB. Use SIGNSAARTHI_RAW_DATA_DIR on a larger volume or narrow SIGNSAARTHI_DOWNLOAD_PATTERN.`
  );
}

mkdirSync(rawDataDir, { recursive: true });
for (const file of files) {
  const outputPath = resolve(rawDataDir, file.key);
  if (existsSync(outputPath) && statSync(outputPath).size === file.size) {
    console.log(`Already downloaded ${file.key}`);
    continue;
  }

  const logPath = `${outputPath}.download.log`;
  const logFd = openSync(logPath, "a");
  console.log(`Downloading ${file.key} -> ${outputPath}`);
  const result = spawnSync("curl", ["-L", "--continue-at", "-", "--fail", "--output", outputPath, file.links.self], {
    stdio: ["ignore", logFd, logFd]
  });
  closeSync(logFd);
  if (result.status !== 0) {
    throw new Error(`curl failed for ${file.key}; see ${logPath}`);
  }
}
