import { createWriteStream, readFileSync, renameSync, rmSync, mkdirSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { once } from "node:events";
import { finished } from "node:stream/promises";
import { keypointSequenceSchema, type KeypointSequence } from "../packages/shared/src/index";

const [outputArg, ...inputArgs] = process.argv.slice(2);
if (!outputArg || inputArgs.length < 1) {
  throw new Error(
    "Usage: tsx scripts/combine-keypoint-datasets.ts <output.json> <input-a.json> [input-b.json ...input-n.json]"
  );
}

const outputPath = resolve(process.cwd(), outputArg);
const merged = new Map<string, KeypointSequence>();

for (const inputArg of inputArgs) {
  const inputPath = resolve(process.cwd(), inputArg);
  const sequences = keypointSequenceSchema.array().parse(JSON.parse(readFileSync(inputPath, "utf8")));
  for (const sequence of sequences) {
    const key = `${sequence.datasetId}:${sequence.split}:${sequence.checksum ?? sequence.sampleId}`;
    if (!merged.has(key)) {
      merged.set(key, sequence);
    }
  }
}

const records = [...merged.values()].sort((left, right) => {
  const splitOrder = { train: 0, val: 1, test: 2 };
  return (
    splitOrder[left.split] - splitOrder[right.split] ||
    left.label.localeCompare(right.label) ||
    left.sampleId.localeCompare(right.sampleId)
  );
});

mkdirSync(dirname(outputPath), { recursive: true });
const tmpOutputPath = `${outputPath}.tmp-${process.pid}`;
const output = createWriteStream(tmpOutputPath, { encoding: "utf8" });

async function writeChunk(chunk: string): Promise<void> {
  if (!output.write(chunk)) {
    await once(output, "drain");
  }
}

try {
  await writeChunk("[");
  for (let index = 0; index < records.length; index += 1) {
    const record = keypointSequenceSchema.parse(records[index]);
    await writeChunk(`${index === 0 ? "\n" : ",\n"}${JSON.stringify(record)}`);
  }
  output.end("\n]\n");
  await finished(output);
  renameSync(tmpOutputPath, outputPath);
} catch (error) {
  output.destroy();
  rmSync(tmpOutputPath, { force: true });
  throw error;
}

console.log(`Combined ${records.length} unique keypoint sequence(s) into ${outputPath}`);
