import { existsSync, readFileSync, writeFileSync, mkdirSync } from "node:fs";
import { dirname, relative, resolve } from "node:path";
import {
  evaluateVideoKeypointModel,
  videoModelArtifactSchema
} from "../packages/isl-video-model/src/index";
import { keypointSequenceSchema } from "../packages/shared/src/index";

const keypointsPath = resolve(
  process.cwd(),
  process.argv[2] ??
    firstExistingPath([
      "data/isl/keypoints.include.json",
      "data/isl/keypoints.include-combined.json",
      "data/isl/keypoints.include-electronics-combined.json",
      "data/isl/keypoints.include-electronics2of2.json",
      "data/isl/keypoints.include-electronics-smoke.json"
    ])
);
const modelPath = resolve(process.cwd(), process.argv[3] ?? "data/models/isl-keypoint-model.json");
const outputPath = resolve(process.cwd(), process.argv[4] ?? "data/models/isl-keypoint-evaluation.json");

if (!existsSync(keypointsPath)) {
  throw new Error(`Real keypoint evaluation input is required; no dataset exists at ${keypointsPath}.`);
}
if (!existsSync(modelPath)) {
  throw new Error(`A trained keypoint model is required for evaluation; no artifact exists at ${modelPath}.`);
}
const sequences = keypointSequenceSchema.array().min(1).parse(JSON.parse(readFileSync(keypointsPath, "utf8")));
const existingModel = videoModelArtifactSchema.safeParse(JSON.parse(readFileSync(modelPath, "utf8")));
if (!existingModel.success) {
  throw new Error(`The keypoint model at ${modelPath} is invalid; evaluation will not train a replacement implicitly.`);
}
const model = existingModel.data;

const bySplit = Object.fromEntries(
  (["train", "val", "test"] as const).map((split) => {
    const splitSequences = sequences.filter((sequence) => sequence.split === split);
    return [split, splitSequences.length ? evaluateVideoKeypointModel(splitSequences, model) : null];
  })
);
const all = evaluateVideoKeypointModel(sequences, model);
const report = {
  schemaVersion: 1,
  generatedAt: new Date().toISOString(),
  keypointsPath: relative(process.cwd(), keypointsPath),
  modelPath: relative(process.cwd(), modelPath),
  model: model.metadata,
  all,
  bySplit
};

mkdirSync(dirname(outputPath), { recursive: true });
writeFileSync(outputPath, `${JSON.stringify(report, null, 2)}\n`);
console.log(
  `Evaluated ${sequences.length} keypoint sample(s) against ${model.metadata.id}; accuracy ${all.accuracy.toFixed(
    4
  )}. Report: ${outputPath}`
);

function firstExistingPath(paths: string[]): string {
  return paths.find((path) => existsSync(resolve(process.cwd(), path))) ?? "data/isl/keypoints.include-combined.json";
}
