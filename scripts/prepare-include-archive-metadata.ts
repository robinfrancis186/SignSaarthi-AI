import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { spawnSync } from "node:child_process";

type IncludeRecord = {
  split: string;
  parentLabel: string;
  label: string;
  videoPath: string;
};

type IncludeMetadata = {
  source: unknown;
  records: IncludeRecord[];
};

const metadataPath = resolve(process.cwd(), process.env["SIGNSAARTHI_INCLUDE_METADATA"] ?? "data/isl/include-metadata.json");
const archivePath = resolve(process.cwd(), process.argv[2] ?? process.env["SIGNSAARTHI_INCLUDE_ARCHIVE"] ?? "");
const outputPath = resolve(
  process.cwd(),
  process.argv[3] ??
    process.env["SIGNSAARTHI_INCLUDE_ARCHIVE_METADATA"] ??
    archivePath.replace(/\.zip$/i, ".metadata.json").replace("/raw/", "/isl/")
);

if (!archivePath) {
  throw new Error("Provide an INCLUDE archive path as an argument or SIGNSAARTHI_INCLUDE_ARCHIVE.");
}

const metadata = JSON.parse(readFileSync(metadataPath, "utf8")) as IncludeMetadata;
const archiveList = spawnSync("unzip", ["-Z1", archivePath], { encoding: "utf8" });
if (archiveList.status !== 0) {
  throw new Error(`Unable to list archive ${archivePath}: ${archiveList.stderr}`);
}

const archiveNames = archiveList.stdout
  .split("\n")
  .map((name) => name.trim())
  .filter((name) => /\.(mov|mp4)$/iu.test(name));
const recordsByPath = new Map<string, IncludeRecord>();
const duplicateRows = new Map<string, IncludeRecord[]>();
for (const record of metadata.records) {
  const existing = recordsByPath.get(record.videoPath);
  if (existing) {
    duplicateRows.set(record.videoPath, [...(duplicateRows.get(record.videoPath) ?? [existing]), record]);
  } else {
    recordsByPath.set(record.videoPath, record);
  }
}
const records = archiveNames.flatMap((name) => {
  const record = recordsByPath.get(name);
  return record ? [record] : [];
});
const missingFromMetadata = archiveNames.filter((name) => !recordsByPath.has(name));
const duplicateArchivePathCount = archiveNames.filter((name) => duplicateRows.has(name)).length;

mkdirSync(dirname(outputPath), { recursive: true });
writeFileSync(
  outputPath,
  `${JSON.stringify(
    {
      schemaVersion: 1,
      generatedAt: new Date().toISOString(),
      source: {
        original: metadata.source,
        archivePath,
        archiveVideoCount: archiveNames.length,
        matchedRecordCount: records.length,
        missingFromMetadataCount: missingFromMetadata.length,
        duplicateArchivePathCount
      },
      records
    },
    null,
    2
  )}\n`
);

console.log(
  `Matched ${records.length}/${archiveNames.length} INCLUDE archive videos to metadata in ${outputPath}${
    missingFromMetadata.length ? `; ${missingFromMetadata.length} archive video(s) missing from metadata` : ""
  }${duplicateArchivePathCount ? `; ${duplicateArchivePathCount} duplicate metadata path(s) deduped` : ""}`
);
