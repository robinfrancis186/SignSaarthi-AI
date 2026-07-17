import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, relative, resolve } from "node:path";
import { trainVideoKeypointModel } from "../packages/isl-video-model/src/index";
import { keypointSequenceSchema } from "../packages/shared/src/index";

const inputPath = resolve(
  process.cwd(),
  process.argv[2] ??
    process.env["SIGNSAARTHI_KEYPOINTS_PATH"] ??
    firstExistingPath([
      "data/isl/keypoints.include.json",
      "data/isl/keypoints.include-combined.json",
      "data/isl/keypoints.include-electronics-combined.json",
      "data/isl/keypoints.include-electronics2of2.json",
      "data/isl/keypoints.include-electronics-smoke.json"
    ])
);
const outputPath = resolve(process.cwd(), process.argv[3] ?? "data/models/isl-keypoint-model.json");
if (!existsSync(inputPath)) {
  throw new Error(`Real keypoint input is required; no dataset exists at ${inputPath}.`);
}
const isFixture = inputPath.endsWith("keypoints.sample.json");
if (isFixture && process.env["SIGNSAARTHI_ALLOW_FIXTURE_MODEL"] !== "true") {
  throw new Error(
    "Refusing to train a release artifact from keypoints.sample.json. Set SIGNSAARTHI_ALLOW_FIXTURE_MODEL=true and provide a non-release output path for an explicit fixture run."
  );
}
if (isFixture && !process.argv[3]) {
  throw new Error("Fixture training requires an explicit output path so it cannot overwrite the release model.");
}
const inputSequences = keypointSequenceSchema.array().min(1).parse(JSON.parse(readFileSync(inputPath, "utf8")));
const model = trainVideoKeypointModel(inputSequences, {
  displayName: isFixture ? "Fixture keypoint model for local integration" : "Real extracted ISL keypoint model"
});
const artifact = {
  ...model,
  metadata: {
    ...model.metadata,
    notes: [
      ...model.metadata.notes,
      isFixture
        ? "This artifact was trained from the bundled synthetic/sample fixture, not real ISL video data."
        : `This artifact was trained from extracted keypoints at ${relative(process.cwd(), inputPath)}.`
    ]
  }
};

mkdirSync(dirname(outputPath), { recursive: true });
writeFileSync(outputPath, `${JSON.stringify(artifact, null, 2)}\n`);

console.log(
  `Trained ${artifact.metadata.id} from ${artifact.metrics?.sampleCount ?? inputSequences.length} keypoint samples into ${outputPath}${
    isFixture ? " (fixture artifact)" : ""
  }`
);

function firstExistingPath(paths: string[]): string {
  return paths.find((path) => existsSync(resolve(process.cwd(), path))) ?? "data/isl/keypoints.include-combined.json";
}
