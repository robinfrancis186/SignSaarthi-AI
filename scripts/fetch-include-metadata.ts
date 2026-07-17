import { mkdirSync, writeFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fetchJson, readBooleanEnv, readListEnv } from "./lib/script-utils";

type DatasetViewerRow = {
  row_idx?: number;
  row: {
    parent_label?: string;
    label?: string;
    video_path?: string;
    include_50?: boolean;
  };
};

type DatasetViewerResponse = {
  rows?: DatasetViewerRow[];
  num_rows_total?: number;
};

const requestedSplits = readListEnv(
  "SIGNSAARTHI_INCLUDE_SPLITS",
  [process.env["SIGNSAARTHI_INCLUDE_SPLIT"] ?? "train"]
);
const limitRaw = process.env["SIGNSAARTHI_INCLUDE_LIMIT"] ?? "25";
const pageSize = Math.min(Number(process.env["SIGNSAARTHI_INCLUDE_PAGE_SIZE"] ?? "100"), 100);
const include50Only = readBooleanEnv("SIGNSAARTHI_INCLUDE_INCLUDE50_ONLY");
const outputPath = resolve(
  process.cwd(),
  process.argv[2] ?? (limitRaw === "all" ? "data/isl/include-metadata.json" : `data/isl/include-metadata.sample.json`)
);

if (!Number.isFinite(pageSize) || pageSize < 1) {
  throw new Error("SIGNSAARTHI_INCLUDE_PAGE_SIZE must be between 1 and 100.");
}

const records = [];
const splits = [];

for (const split of requestedSplits) {
  const firstPage = await fetchPage(split, 0, pageSize);
  const totalRows = firstPage.num_rows_total ?? firstPage.rows?.length ?? 0;
  const splitLimit = limitRaw === "all" ? totalRows : Math.min(Number(limitRaw), totalRows);

  if (!Number.isFinite(splitLimit) || splitLimit < 1) {
    throw new Error("SIGNSAARTHI_INCLUDE_LIMIT must be a positive number or all.");
  }

  const pages = [firstPage];
  for (let offset = pageSize; offset < splitLimit; offset += pageSize) {
    pages.push(await fetchPage(split, offset, Math.min(pageSize, splitLimit - offset)));
  }

  const splitRecords = pages
    .flatMap((payload) => payload.rows ?? [])
    .slice(0, splitLimit)
    .map((entry) => ({
      datasetId: "include",
      split,
      rowIndex: entry.row_idx,
      parentLabel: entry.row.parent_label ?? "",
      label: entry.row.label ?? "",
      videoPath: entry.row.video_path ?? "",
      include50: Boolean(entry.row.include_50)
    }))
    .filter((record) => !include50Only || record.include50);

  records.push(...splitRecords);
  splits.push({
    split,
    totalRows,
    fetchedRows: splitRecords.length,
    include50Only
  });
}

const metadata = {
  schemaVersion: 1,
  generatedAt: new Date().toISOString(),
  source: {
    dataset: "ai4bharat/INCLUDE",
    config: "default",
    datasetViewerUrl: "https://datasets-server.huggingface.co/rows",
    license: "cc-by-4.0",
    citations: [
      "https://huggingface.co/datasets/ai4bharat/INCLUDE",
      "https://zenodo.org/records/4010759",
      "https://github.com/AI4Bharat/INCLUDE"
    ],
    splits
  },
  records
};

mkdirSync(dirname(outputPath), { recursive: true });
writeFileSync(outputPath, `${JSON.stringify(metadata, null, 2)}\n`);
console.log(
  `Fetched ${records.length} INCLUDE metadata rows for ${requestedSplits.join(", ")} into ${outputPath}`
);

async function fetchPage(split: string, offset: number, length: number): Promise<DatasetViewerResponse> {
  const url = new URL("https://datasets-server.huggingface.co/rows");
  url.searchParams.set("dataset", "ai4bharat/INCLUDE");
  url.searchParams.set("config", "default");
  url.searchParams.set("split", split);
  url.searchParams.set("offset", String(offset));
  url.searchParams.set("length", String(length));
  return fetchJson<DatasetViewerResponse>(url);
}
