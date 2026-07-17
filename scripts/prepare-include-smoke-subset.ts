import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { spawnSync } from "node:child_process";
import { readBooleanEnv } from "./lib/script-utils";

type IncludeMetadata = {
  source: unknown;
  records: Array<{
    split: string;
    parentLabel: string;
    label: string;
    videoPath: string;
  }>;
};

const metadataPath = resolve(process.cwd(), process.env["SIGNSAARTHI_INCLUDE_METADATA"] ?? "data/isl/include-metadata.json");
const archivePath = resolve(process.cwd(), process.env["SIGNSAARTHI_INCLUDE_ARCHIVE"] ?? "data/raw/include/Electronics_2of2.zip");
const outputMetadataPath = resolve(
  process.cwd(),
  process.env["SIGNSAARTHI_INCLUDE_SUBSET_METADATA"] ?? "data/isl/include-electronics-subset-metadata.json"
);
const extractDir = resolve(process.cwd(), process.env["SIGNSAARTHI_INCLUDE_EXTRACT_DIR"] ?? "data/raw/include/extracted");
const parentLabel = process.env["SIGNSAARTHI_INCLUDE_PARENT_LABEL"] ?? "Electronics";
const labels = (process.env["SIGNSAARTHI_INCLUDE_LABELS"] ?? "56. Laptop,57. Screen,58. Camera,59. Television,60. Radio")
  .split(",")
  .map((label) => label.trim())
  .filter(Boolean);
const perSplit = Number(process.env["SIGNSAARTHI_INCLUDE_PER_SPLIT"] ?? "5");
const perLabelPerSplit = Number(process.env["SIGNSAARTHI_INCLUDE_PER_LABEL_PER_SPLIT"] ?? "0");
const skipExtract = readBooleanEnv("SIGNSAARTHI_SKIP_EXTRACT");

const metadata = JSON.parse(readFileSync(metadataPath, "utf8")) as IncludeMetadata;
const labelSet = new Set(labels);
const records =
  perLabelPerSplit > 0
    ? ["train", "val", "test"].flatMap((split) =>
        labels.flatMap((label) =>
          metadata.records
            .filter((record) => record.parentLabel === parentLabel && record.label === label && record.split === split)
            .slice(0, perLabelPerSplit)
        )
      )
    : ["train", "val", "test"].flatMap((split) =>
        metadata.records
          .filter((record) => record.parentLabel === parentLabel && labelSet.has(record.label) && record.split === split)
          .slice(0, perSplit)
      );

if (records.length === 0) {
  throw new Error(`No INCLUDE records matched ${parentLabel} / ${labels.join(", ")} in ${metadataPath}`);
}

mkdirSync(dirname(outputMetadataPath), { recursive: true });
writeFileSync(
  outputMetadataPath,
  `${JSON.stringify(
    {
      schemaVersion: 1,
      generatedAt: new Date().toISOString(),
      source: {
        original: metadata.source,
        subset: "real_video_smoke",
        archivePath,
        parentLabel,
        labels,
        perSplit,
        perLabelPerSplit
      },
      records
    },
    null,
    2
  )}\n`
);

if (!skipExtract) {
  mkdirSync(extractDir, { recursive: true });
  const result = spawnSync("unzip", ["-n", archivePath, "-d", extractDir, ...records.map((record) => record.videoPath)], {
    stdio: "inherit"
  });
  if (result.status !== 0) {
    throw new Error(`Unable to extract INCLUDE smoke subset from ${archivePath}`);
  }
}

console.log(
  `Prepared ${records.length} INCLUDE smoke records in ${outputMetadataPath}${skipExtract ? " without extracting videos" : ""}`
);
