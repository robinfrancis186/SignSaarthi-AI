import { spawnSync } from "node:child_process";
import { existsSync, mkdirSync, readFileSync, rmSync } from "node:fs";
import { resolve } from "node:path";

type MotionCatalog = {
  clipCount: number;
  clips: Array<{
    id: string;
    normalizedLabel: string;
    sourceVideoPath?: string;
    expertReviewed: boolean;
    playable?: boolean;
  }>;
};

type MotionLibrary = {
  clipCount: number;
  clips: Array<{ id: string; frames: unknown[] }>;
};

const workspaceRoot = resolve(import.meta.dirname, "..");
const verificationTempDirectory = resolve(workspaceRoot, "output/.tmp");
rmSync(verificationTempDirectory, { force: true, recursive: true });
mkdirSync(verificationTempDirectory, { recursive: true });
const expectedMotionClipCount = 262;
const pythonExecutable =
  process.env["SIGNSAARTHI_PYTHON"] ??
  (existsSync(resolve(workspaceRoot, ".venv-mediapipe/bin/python"))
    ? ".venv-mediapipe/bin/python"
    : "python3");
const trainingPythonExecutable =
  process.env["SIGNSAARTHI_TRAINING_PYTHON"] ??
  (existsSync(resolve(workspaceRoot, ".venv-isign-training/bin/python"))
    ? ".venv-isign-training/bin/python"
    : "python3");

const phases = [
  {
    name: "source verification",
    commands: [
      ["pnpm", ["typecheck"]],
      ["pnpm", ["scripts:typecheck"]],
      ["pnpm", ["lint"]],
      ["pnpm", ["test"]],
      ["node", ["--import", "tsx", "--test", "scripts/tests/package-local-release.test.ts"]],
      [
        pythonExecutable,
        ["-m", "unittest", "discover", "-s", "scripts/tests", "-p", "test_*.py", "-v"]
      ],
      [trainingPythonExecutable, ["-c", "import torch, onnxruntime"]],
      [
        trainingPythonExecutable,
        ["-m", "unittest", "scripts.tests.test_train_isign_text_to_pose", "-v"]
      ]
    ] as const
  },
  {
    name: "production verification",
    commands: [
      ["pnpm", ["build"]],
      ["pnpm", ["dataset:unified:validate"]],
      ["pnpm", ["dataset:training:validate"]],
      ["pnpm", ["model:text-matcher:train"]],
      ["pnpm", ["smoke:extension"]],
      ["pnpm", ["package:release"]],
      [
        "pnpm",
        [
          "exec",
          "tsx",
          "--tsconfig",
          "tsconfig.scripts.json",
          "scripts/package-local-release.ts",
          "--verify-bundle",
          "output/signsaarthi-local-release"
        ]
      ]
    ] as const
  }
];

const verificationPasses = 2;

for (let pass = 1; pass <= verificationPasses; pass += 1) {
  console.log(`\n######## verification pass ${pass}/${verificationPasses} ########`);
  for (const phase of phases) {
    console.log(`\n== ${phase.name} ==`);
    for (const [command, args] of phase.commands) {
      run(command, [...args]);
    }
    assertMotionArtifacts();
  }
}

console.log(
  `\nSignSaarthi verification loop passed ${verificationPasses} complete passes with production artifacts.`
);

function run(command: string, args: string[]): void {
  console.log(`\n$ ${command} ${args.join(" ")}`);
  const env = { ...process.env };
  if (command === "pnpm" && args[0] === "test") {
    env["TMPDIR"] = verificationTempDirectory;
    env["TMP"] = verificationTempDirectory;
    env["TEMP"] = verificationTempDirectory;
  }
  const result = spawnSync(command, args, {
    cwd: workspaceRoot,
    env,
    stdio: "inherit"
  });
  if (result.error) {
    throw result.error;
  }
  if (result.status !== 0) {
    throw new Error(
      `${command} ${args.join(" ")} failed with exit code ${result.status ?? "unknown"}.`
    );
  }
}

function assertMotionArtifacts(): void {
  const catalog = readJson<MotionCatalog>(
    "packages/avatar-engine/src/motionCatalog.generated.json"
  );
  const library = readJson<MotionLibrary>(
    "apps/extension/src/assets/motion/isl-motion-library.json"
  );
  if (
    catalog.clipCount !== expectedMotionClipCount ||
    library.clipCount !== expectedMotionClipCount
  ) {
    throw new Error(
      `Expected ${expectedMotionClipCount} validated INCLUDE motions; catalog=${catalog.clipCount}, library=${library.clipCount}.`
    );
  }
  const catalogIds = catalog.clips.map((clip) => clip.id);
  const libraryIds = library.clips.map((clip) => clip.id);
  if (JSON.stringify(catalogIds) !== JSON.stringify(libraryIds)) {
    throw new Error("Motion catalog and frame library IDs are not identical and ordered.");
  }
  if (catalog.clips.some((clip) => clip.expertReviewed || clip.playable === false)) {
    throw new Error("Generated INCLUDE catalog contains an invalid review/playability state.");
  }
  for (const clip of catalog.clips) {
    const sourceDirectory = clip.sourceVideoPath?.split("/").at(-2);
    if (!sourceDirectory || normalizeIncludeLabel(sourceDirectory) !== clip.normalizedLabel) {
      throw new Error(`Motion ${clip.id} does not match its source directory label.`);
    }
  }
  if (library.clips.some((clip) => clip.frames.length !== 32)) {
    throw new Error("Every generated INCLUDE motion must contain exactly 32 renderer frames.");
  }
}

function normalizeIncludeLabel(value: string): string {
  return value
    .replace(/^\s*\d+\s*[.)-]\s*/u, "")
    .replaceAll("&", " and ")
    .replace(/[^a-zA-Z0-9]+/gu, " ")
    .toLowerCase()
    .trim()
    .replace(/\s+/gu, " ");
}

function readJson<T>(relativePath: string): T {
  return JSON.parse(readFileSync(resolve(workspaceRoot, relativePath), "utf8")) as T;
}
