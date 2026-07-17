import {
  islLexiconModelSchema,
  lexicalMatcherArtifactSchema,
  type ISLLexiconModel,
  type LexicalMatcherArtifact
} from "@signsaarthi/isl-model";
import {
  motionCatalogArtifactSchema,
  temporalModelArtifactSchema,
  videoModelArtifactSchema,
  type MotionCatalogArtifact,
  type TemporalModelArtifact,
  type VideoModelArtifact
} from "@signsaarthi/shared";
import { createHash } from "node:crypto";
import { spawnSync } from "node:child_process";
import {
  chmodSync,
  copyFileSync,
  existsSync,
  mkdirSync,
  mkdtempSync,
  readFileSync,
  readdirSync,
  realpathSync,
  renameSync,
  rmSync,
  writeFileSync
} from "node:fs";
import { basename, dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { isDeepStrictEqual } from "node:util";
import {
  MINIMUM_NODE_VERSION,
  RELEASE_MANIFEST_SCHEMA_VERSION,
  REPRODUCIBLE_ZIP_TIMESTAMP,
  assertSupportedNode,
  createReleaseAttestation,
  createReleaseTreeManifest,
  digestFile,
  digestSourceFiles,
  digestTree,
  filterAllowlistedSourcePaths,
  replaceBundledWorkspaceDependencies,
  releaseSourceSelectionPolicy,
  resolveReleaseMode,
  sha256,
  snapshotFilesystemSource,
  verifyReleaseArchive,
  verifyReleaseBundle,
  type ReleaseSourceState,
  writeDeterministicZip
} from "./lib/release-utils.js";

const workspaceRoot = fileURLToPath(new URL("../", import.meta.url));
const extensionDirectory = resolve(workspaceRoot, "apps/extension/dist");
const extensionManifestPath = resolve(extensionDirectory, "manifest.json");
const apiEntrypointPath = resolve(workspaceRoot, "apps/api/dist/main.js");
const lockfilePath = resolve(workspaceRoot, "pnpm-lock.yaml");
const packageJsonPath = resolve(workspaceRoot, "package.json");
const modelDirectory = resolve(workspaceRoot, "data/models");
const lexiconModelPath = resolve(modelDirectory, "isl-lexicon-model.json");
const lexicalMatcherModelPath = resolve(modelDirectory, "isl-text-matcher.json");
const centroidModelPath = resolve(modelDirectory, "isl-keypoint-model.json");
const temporalMetadataPath = resolve(workspaceRoot, "data/models/isl-temporal-model.json");
const motionCatalogPath = resolve(modelDirectory, "isl-motion-catalog.json");
const avatarMotionCatalogPath = resolve(
  workspaceRoot,
  "packages/avatar-engine/src/motionCatalog.generated.json"
);
const builtAvatarMotionCatalogPath = resolve(
  workspaceRoot,
  "packages/avatar-engine/dist/motionCatalog.generated.json"
);
const motionLibraryPath = resolve(
  workspaceRoot,
  "apps/extension/src/assets/motion/isl-motion-library.json"
);
const alphabetLibraryPath = resolve(
  workspaceRoot,
  "apps/extension/src/assets/motion/islrtc-alphabet-library.json"
);
const outputDirectory = resolve(workspaceRoot, "output");
const finalBundleDirectory = resolve(outputDirectory, "signsaarthi-local-release");
const releaseManifestName = "signsaarthi-release-manifest.json";
const releaseArchivePrefix = "signsaarthi-local-release";
const extensionZipRelativePath = "extension/signsaarthi-extension.zip";
const jsonModuleLoaderRelativePath = "json-module-loader.mjs";
const startScriptRelativePath = "start-api.mjs";
const expectedExtensionId = "kgjeaadkknfadmlipfjaekalpgbcnela";
const expectedMotionClipCount = 262;
const expectedAlphabetClipCount = 26;
const maximumGitOutputBytes = 128 * 1024 * 1024;
const buildInputDirectories = [
  extensionDirectory,
  resolve(workspaceRoot, "apps/api/dist"),
  resolve(workspaceRoot, "packages/avatar-engine/dist"),
  resolve(workspaceRoot, "packages/isl-engine/dist"),
  resolve(workspaceRoot, "packages/isl-model/dist"),
  resolve(workspaceRoot, "packages/isl-video-model/dist"),
  resolve(workspaceRoot, "packages/shared/dist")
];
const bundledWorkspacePackageDirectories = [
  resolve(workspaceRoot, "packages/avatar-engine"),
  resolve(workspaceRoot, "packages/isl-engine"),
  resolve(workspaceRoot, "packages/isl-model"),
  resolve(workspaceRoot, "packages/isl-video-model"),
  resolve(workspaceRoot, "packages/shared")
];

assertSupportedNode();

if (process.argv[2] === "--verify-bundle") {
  const bundlePath = resolve(workspaceRoot, process.argv[3] ?? "output/signsaarthi-local-release");
  console.log(
    JSON.stringify(
      {
        status: "verified",
        bundle: bundlePath,
        ...verifyReleaseBundle(bundlePath, releaseManifestName)
      },
      null,
      2
    )
  );
  process.exit(0);
}

if (process.argv[2] === "--verify-archive") {
  const archivePath = process.argv[3];
  const checksumPath = process.argv[4];
  if (!archivePath || !checksumPath) {
    throw new Error("--verify-archive requires archive and checksum paths.");
  }
  console.log(
    JSON.stringify(
      {
        status: "verified",
        archivePath: resolve(workspaceRoot, archivePath),
        ...verifyReleaseArchive(
          resolve(workspaceRoot, archivePath),
          resolve(workspaceRoot, checksumPath),
          releaseManifestName
        )
      },
      null,
      2
    )
  );
  process.exit(0);
}

if (process.argv.length > 2) {
  throw new Error(`Unknown release packaging argument: ${process.argv[2]}.`);
}

const releaseMode = resolveReleaseMode();

const rootPackage = readJson<{
  name: string;
  version: string;
  packageManager: string;
}>(packageJsonPath);
const packageManagerVersion = rootPackage.packageManager.match(/^pnpm@(.+)$/)?.[1];
if (!packageManagerVersion) {
  throw new Error(
    `package.json must pin pnpm in packageManager; received ${rootPackage.packageManager}.`
  );
}

const pnpmVersion = commandVersion("pnpm");
if (pnpmVersion !== packageManagerVersion) {
  throw new Error(
    `Release packaging requires ${rootPackage.packageManager}; current pnpm is ${pnpmVersion}. Run corepack install.`
  );
}
const corepackVersion = commandVersion("corepack");

const extensionManifest = readJson<{
  host_permissions: string[];
  key: string;
  manifest_version: number;
  name: string;
  version: string;
}>(extensionManifestPath);
const extensionId = extensionIdFromPublicKey(extensionManifest.key);
if (extensionId !== expectedExtensionId) {
  throw new Error(
    `Extension public key resolves to ${extensionId}, expected stable ID ${expectedExtensionId}.`
  );
}
const extensionPublicKeySha256 = sha256(Buffer.from(extensionManifest.key, "base64"));
const apiBaseUrl = normalizeApiBaseUrl(
  process.env["VITE_SIGNSAARTHI_API_BASE_URL"] ?? "http://127.0.0.1:8787"
);
const apiPort = Number(new URL(apiBaseUrl).port);
const expectedApiHostPermission = `${apiBaseUrl}/*`;
if (!extensionManifest.host_permissions.includes(expectedApiHostPermission)) {
  throw new Error(
    `Built extension is missing host permission ${expectedApiHostPermission}. Rebuild with VITE_SIGNSAARTHI_API_BASE_URL=${apiBaseUrl}.`
  );
}
const requiredArtifacts = loadRequiredReleaseArtifacts();
const recognitionArtifacts = loadOptionalRecognitionArtifacts(
  new Set(requiredArtifacts.motionCatalog.clips.map((clip) => clip.id))
);
const artifactInputs = [
  ...requiredArtifacts.inputs,
  ...recognitionArtifacts.flatMap((artifact) => artifact.inputs)
];

for (const requiredPath of [
  extensionManifestPath,
  apiEntrypointPath,
  lockfilePath,
  packageJsonPath,
  ...artifactInputs.map((input) => input.path)
]) {
  if (!existsSync(requiredPath)) {
    throw new Error(`Release input is missing: ${requiredPath}. Run pnpm build first.`);
  }
}

for (const directory of buildInputDirectories) {
  removeFilesystemMetadataFiles(directory);
}

const sourceState = readSourceState();
const releaseAttestation = createReleaseAttestation(releaseMode, sourceState);
const buildInputState = readBuildInputState();
mkdirSync(outputDirectory, { recursive: true });
const stagingDirectory = mkdtempSync(join(outputDirectory, ".signsaarthi-local-release-"));
const archiveStagingDirectory = mkdtempSync(
  join(outputDirectory, ".signsaarthi-local-release-archive-")
);
let published = false;

try {
  const extensionZipPath = resolve(stagingDirectory, extensionZipRelativePath);
  const extensionDigest = writeDeterministicZip(extensionDirectory, extensionZipPath);

  const apiDirectory = resolve(stagingDirectory, "api");
  deployApi(apiDirectory);
  sanitizeApiDeploymentManifest(apiDirectory);
  pruneApiDeployment(apiDirectory);
  removeFilesystemMetadataFiles(apiDirectory);
  const apiDigest = digestTree(apiDirectory);

  const packagedFiles = new Map<string, PackagedFile>();
  for (const input of artifactInputs) {
    if (packagedFiles.has(input.relativePath)) {
      throw new Error(`Release artifact path is duplicated: ${input.relativePath}.`);
    }
    packagedFiles.set(input.relativePath, copyReleaseFile(stagingDirectory, input));
  }
  const packagedFile = (input: ReleaseFileInput): PackagedFile => {
    const file = packagedFiles.get(input.relativePath);
    if (!file) {
      throw new Error(`Release artifact was not staged: ${input.relativePath}.`);
    }
    return file;
  };
  const recognitionModels = recognitionArtifacts.map((artifact) => ({
    kind: artifact.kind,
    metadata: artifact.metadata,
    files: artifact.inputs.map(packagedFile),
    ...(artifact.metrics === undefined ? {} : { metrics: artifact.metrics })
  }));
  const recognitionModelReady = recognitionArtifacts.some(
    (artifact) => artifact.metadata.status === "ready"
  );

  const provenanceDirectory = resolve(stagingDirectory, "provenance");
  mkdirSync(provenanceDirectory, { recursive: true });
  const bundledLockfilePath = resolve(provenanceDirectory, "pnpm-lock.yaml");
  copyFileSync(lockfilePath, bundledLockfilePath);
  const lockfileDigest = digestFile(bundledLockfilePath);

  const releaseRuntimeId = `release-${releaseAttestation.status}-${sourceState.tree.sha256.slice(0, 16)}-${extensionDigest.sha256.slice(0, 16)}`;
  const jsonModuleLoaderPath = resolve(stagingDirectory, jsonModuleLoaderRelativePath);
  writeFileSync(jsonModuleLoaderPath, jsonModuleLoader(), "utf8");

  const startScriptPath = resolve(stagingDirectory, startScriptRelativePath);
  writeFileSync(startScriptPath, startScript(releaseRuntimeId, apiPort), {
    encoding: "utf8",
    mode: 0o755
  });

  const bundleReadmePath = resolve(stagingDirectory, "README.md");
  writeFileSync(
    bundleReadmePath,
    bundleReadme({
      arch: process.arch,
      apiBaseUrl,
      extensionId,
      minimumNodeVersion: MINIMUM_NODE_VERSION,
      motionClipCount: requiredArtifacts.motionLibrary.clipCount,
      platform: process.platform,
      releaseAttestation,
      recognitionModelReady,
      releaseRuntimeId
    }),
    "utf8"
  );

  removeFilesystemMetadataFiles(stagingDirectory);

  const releaseManifest = {
    schemaVersion: RELEASE_MANIFEST_SCHEMA_VERSION,
    product: extensionManifest.name,
    version: extensionManifest.version,
    lexiconModelReady: true,
    lexicalMatcherReady: true,
    recognitionModelReady,
    motionClipCount: requiredArtifacts.motionLibrary.clipCount,
    officialAlphabetFallback: true,
    reproducibility: {
      deterministicZip: true,
      entryOrder: "UTF-16 code unit ascending",
      timestamp: REPRODUCIBLE_ZIP_TIMESTAMP,
      completeArchive: true,
      contentAddressedArchive: true
    },
    artifacts: {
      extension: {
        path: extensionZipRelativePath,
        id: extensionId,
        manifestVersion: extensionManifest.manifest_version,
        ...extensionDigest
      },
      localApi: {
        directory: "api",
        entrypoint: "api/dist/main.js",
        startScript: startScriptRelativePath,
        startCommand: `node ./${startScriptRelativePath}`,
        startup: {
          script: {
            path: startScriptRelativePath,
            ...digestFile(startScriptPath)
          },
          jsonModuleLoader: {
            path: jsonModuleLoaderRelativePath,
            ...digestFile(jsonModuleLoaderPath)
          }
        },
        baseUrl: apiBaseUrl,
        healthUrl: `${apiBaseUrl}/health`,
        port: apiPort,
        runtimeId: releaseRuntimeId,
        target: `${process.platform}-${process.arch}`,
        ...apiDigest
      },
      lexiconModel: packagedFile(requiredArtifacts.files.lexicon),
      lexicalMatcherModel: packagedFile(requiredArtifacts.files.lexicalMatcher),
      motion: {
        catalog: packagedFile(requiredArtifacts.files.motionCatalog),
        library: packagedFile(requiredArtifacts.files.motionLibrary),
        officialAlphabetFallback: packagedFile(requiredArtifacts.files.alphabetLibrary)
      },
      recognitionModels,
      instructions: {
        path: "README.md",
        ...digestFile(bundleReadmePath)
      }
    },
    motion: {
      clipCount: requiredArtifacts.motionLibrary.clipCount,
      source: requiredArtifacts.motionLibrary.source,
      expertReviewStatus: requiredArtifacts.motionLibrary.expertReviewStatus,
      alphabetFallback: {
        enabled: true,
        officialSource: true,
        clipCount: requiredArtifacts.alphabetLibrary.clips.length,
        source: requiredArtifacts.alphabetLibrary.source,
        sourceUrl: requiredArtifacts.alphabetLibrary.sourceUrl,
        usageTermsUrl: requiredArtifacts.alphabetLibrary.usageTermsUrl,
        attribution: requiredArtifacts.alphabetLibrary.attribution,
        expertReviewStatus: requiredArtifacts.alphabetLibrary.expertReviewStatus
      }
    },
    privacy: {
      rawAudioStored: false,
      rawVideoStored: false
    },
    provenance: {
      attestation: releaseAttestation,
      lockfile: {
        path: "provenance/pnpm-lock.yaml",
        ...lockfileDigest
      },
      source: sourceState,
      extensionIdentity: {
        id: extensionId,
        publicKeySha256: extensionPublicKeySha256
      },
      runtime: {
        node: process.versions.node,
        minimumNode: MINIMUM_NODE_VERSION,
        nodeModulesAbi: process.versions.modules,
        napi: process.versions.napi,
        corepack: corepackVersion,
        packageManager: rootPackage.packageManager,
        pnpm: pnpmVersion,
        platform: process.platform,
        arch: process.arch
      }
    },
    integrity: {
      schemaVersion: 1,
      manifest: {
        path: releaseManifestName,
        sealedBy: "content-addressed archive SHA-256"
      },
      payload: createReleaseTreeManifest(stagingDirectory, [releaseManifestName])
    }
  };

  const releaseManifestPath = resolve(stagingDirectory, releaseManifestName);
  writeFileSync(releaseManifestPath, `${JSON.stringify(releaseManifest, null, 2)}\n`, "utf8");
  removeFilesystemMetadataFiles(stagingDirectory);
  const bundleVerification = verifyReleaseBundle(stagingDirectory, releaseManifestName);

  const temporaryArchivePath = resolve(archiveStagingDirectory, "release.zip");
  const archiveDigest = writeDeterministicZip(stagingDirectory, temporaryArchivePath);
  const archiveTrustLabel =
    releaseAttestation.status === "attested"
      ? `git-${releaseAttestation.gitHead!.slice(0, 12)}`
      : "unattested";
  const archiveName = `${releaseArchivePrefix}-v${rootPackage.version}-${process.platform}-${process.arch}-${archiveTrustLabel}-${archiveDigest.sha256}.zip`;
  const stagedArchivePath = resolve(archiveStagingDirectory, archiveName);
  renameSync(temporaryArchivePath, stagedArchivePath);
  const checksumName = `${archiveName}.sha256`;
  const stagedChecksumPath = resolve(archiveStagingDirectory, checksumName);
  writeFileSync(stagedChecksumPath, `${archiveDigest.sha256}  ${archiveName}\n`, "utf8");
  removeFilesystemMetadataFiles(archiveStagingDirectory);
  verifyReleaseArchive(stagedArchivePath, stagedChecksumPath, releaseManifestName);

  assertPackagingInputsStable(sourceState, buildInputState);
  const finalArchivePath = resolve(outputDirectory, archiveName);
  const finalChecksumPath = resolve(outputDirectory, checksumName);
  publishImmutableFile(stagedArchivePath, finalArchivePath);
  publishImmutableFile(stagedChecksumPath, finalChecksumPath);
  publishBundle(stagingDirectory, finalBundleDirectory);
  published = true;
  removeFilesystemMetadataFiles(outputDirectory);
  verifyReleaseBundle(finalBundleDirectory, releaseManifestName);
  verifyReleaseArchive(finalArchivePath, finalChecksumPath, releaseManifestName);

  rmSync(resolve(outputDirectory, "signsaarthi-extension.zip"), { force: true });
  rmSync(resolve(outputDirectory, "signsaarthi-release-manifest.json"), { force: true });

  console.log(
    JSON.stringify(
      {
        status: "packaged",
        bundle: finalBundleDirectory,
        releaseManifest: resolve(finalBundleDirectory, releaseManifestName),
        extensionZip: resolve(finalBundleDirectory, extensionZipRelativePath),
        extensionSha256: extensionDigest.sha256,
        apiTreeSha256: apiDigest.sha256,
        archive: finalArchivePath,
        archiveSha256: archiveDigest.sha256,
        checksum: finalChecksumPath,
        manifestEntries: bundleVerification.entries,
        attestation: releaseAttestation.status,
        releaseMode,
        sourceState: sourceState.state,
        sourceTreeSha256: sourceState.tree.sha256,
        runtimeTarget: `${process.platform}-${process.arch}`
      },
      null,
      2
    )
  );
} finally {
  if (!published) {
    rmSync(stagingDirectory, { force: true, recursive: true });
  }
  rmSync(archiveStagingDirectory, { force: true, recursive: true });
  removeAppleDoubleSibling(stagingDirectory);
  removeAppleDoubleSibling(archiveStagingDirectory);
}

type MotionPoint = {
  x: number;
  y: number;
};

type MotionFrame = {
  pose: MotionPoint[];
  leftHand: MotionPoint[];
  rightHand: MotionPoint[];
};

type MotionLibraryClip = {
  id: string;
  label: string;
  normalizedLabel: string;
  fps: number;
  sourceSampleId: string;
  datasetId: "include";
  expertReviewed: false;
  frames: MotionFrame[];
};

type MotionLibrary = {
  schemaVersion: 1;
  source: string;
  expertReviewStatus: string;
  clipCount: number;
  clips: MotionLibraryClip[];
};

type AlphabetLibrary = {
  schemaVersion: 1;
  source: string;
  sourceUrl: string;
  usageTermsUrl: string;
  attribution: string;
  expertReviewStatus: string;
  derivation: {
    rawVideoStored: false;
    rawAudioStored: false;
  };
  clips: Array<{
    id: string;
    label: string;
    datasetId: "islrtc";
    expertReviewed: false;
    frames: MotionFrame[];
  }>;
};

type ReleaseFileInput = {
  path: string;
  relativePath: string;
  role: string;
};

type PackagedFile = {
  path: string;
  role: string;
  bytes: number;
  sha256: string;
};

type RequiredReleaseArtifacts = {
  lexicon: ISLLexiconModel;
  lexicalMatcher: LexicalMatcherArtifact;
  motionCatalog: MotionCatalogArtifact;
  motionLibrary: MotionLibrary;
  alphabetLibrary: AlphabetLibrary;
  files: {
    lexicon: ReleaseFileInput;
    lexicalMatcher: ReleaseFileInput;
    motionCatalog: ReleaseFileInput;
    motionLibrary: ReleaseFileInput;
    alphabetLibrary: ReleaseFileInput;
  };
  inputs: ReleaseFileInput[];
};

type RecognitionReleaseArtifact = {
  kind: "centroid" | "temporal";
  metadata: VideoModelArtifact["metadata"] | TemporalModelArtifact["metadata"];
  metrics?: VideoModelArtifact["metrics"] | TemporalModelArtifact["metrics"];
  inputs: ReleaseFileInput[];
};

function loadRequiredReleaseArtifacts(): RequiredReleaseArtifacts {
  const lexicon = islLexiconModelSchema.parse(readJson<unknown>(lexiconModelPath));
  assertLexiconArtifact(lexicon);
  const lexicalMatcher = lexicalMatcherArtifactSchema.parse(
    readJson<unknown>(lexicalMatcherModelPath)
  );
  if (lexicalMatcher.metadata.status !== "ready" || !lexicalMatcher.metrics.acceptance.ready) {
    throw new Error("Lexical matcher must pass every readiness gate before packaging.");
  }

  const rawMotionCatalog = readJson<unknown>(motionCatalogPath);
  const motionCatalog = motionCatalogArtifactSchema.parse(rawMotionCatalog);
  const avatarMotionCatalog = motionCatalogArtifactSchema.parse(
    readJson<unknown>(avatarMotionCatalogPath)
  );
  const builtAvatarMotionCatalog = motionCatalogArtifactSchema.parse(
    readJson<unknown>(builtAvatarMotionCatalogPath)
  );
  const motionLibrary = parseMotionLibrary(readJson<unknown>(motionLibraryPath));
  const alphabetLibrary = parseAlphabetLibrary(readJson<unknown>(alphabetLibraryPath));

  assertMotionArtifacts(
    rawMotionCatalog,
    motionCatalog,
    avatarMotionCatalog,
    builtAvatarMotionCatalog,
    motionLibrary
  );
  assertMatchingFiles(
    motionCatalogPath,
    avatarMotionCatalogPath,
    "model and source avatar motion catalogs"
  );
  assertMatchingJsonFiles(
    motionCatalogPath,
    builtAvatarMotionCatalogPath,
    "source and built avatar motion catalogs"
  );
  assertAlphabetLibrary(alphabetLibrary);
  assertAlphabetEmbeddedInExtension(alphabetLibrary);

  const files = {
    lexicon: {
      path: lexiconModelPath,
      relativePath: "data/models/isl-lexicon-model.json",
      role: "required lexicon metadata and weights"
    },
    lexicalMatcher: {
      path: lexicalMatcherModelPath,
      relativePath: "data/models/isl-text-matcher.json",
      role: "required evaluated caption typo matcher"
    },
    motionCatalog: {
      path: motionCatalogPath,
      relativePath: "data/models/isl-motion-catalog.json",
      role: "required INCLUDE motion catalog"
    },
    motionLibrary: {
      path: motionLibraryPath,
      relativePath: "data/motion/isl-motion-library.json",
      role: "required INCLUDE renderer motion library"
    },
    alphabetLibrary: {
      path: alphabetLibraryPath,
      relativePath: "data/motion/islrtc-alphabet-library.json",
      role: "required official-source alphabet fallback"
    }
  } satisfies RequiredReleaseArtifacts["files"];

  return {
    lexicon,
    lexicalMatcher,
    motionCatalog,
    motionLibrary,
    alphabetLibrary,
    files,
    inputs: [
      files.lexicon,
      files.lexicalMatcher,
      files.motionCatalog,
      files.motionLibrary,
      files.alphabetLibrary
    ]
  };
}

function loadOptionalRecognitionArtifacts(
  motionClipIds: ReadonlySet<string>
): RecognitionReleaseArtifact[] {
  const artifacts: RecognitionReleaseArtifact[] = [];

  if (existsSync(centroidModelPath)) {
    const centroid = videoModelArtifactSchema.parse(readJson<unknown>(centroidModelPath));
    assertCentroidArtifact(centroid);
    const artifact: RecognitionReleaseArtifact = {
      kind: "centroid",
      metadata: centroid.metadata,
      inputs: [
        {
          path: centroidModelPath,
          relativePath: "data/models/isl-keypoint-model.json",
          role: "optional validated centroid recognition model"
        }
      ]
    };
    if (centroid.metrics !== undefined) {
      artifact.metrics = centroid.metrics;
    }
    artifacts.push(artifact);
  }

  if (existsSync(temporalMetadataPath)) {
    const temporal = temporalModelArtifactSchema.parse(readJson<unknown>(temporalMetadataPath));
    const runtimePath = validateTemporalArtifact(temporal, motionClipIds);
    artifacts.push({
      kind: "temporal",
      metadata: temporal.metadata,
      metrics: temporal.metrics,
      inputs: [
        {
          path: temporalMetadataPath,
          relativePath: "data/models/isl-temporal-model.json",
          role: "optional validated temporal recognition metadata"
        },
        {
          path: runtimePath,
          relativePath: `data/models/${temporal.modelFile}`,
          role: "optional validated temporal ONNX runtime model"
        }
      ]
    });
  }

  return artifacts;
}

function assertLexiconArtifact(lexicon: ISLLexiconModel): void {
  if (lexicon.metadata.status !== "ready" || lexicon.metadata.engine !== "lexicon_ranker") {
    throw new Error(
      `Lexicon model must be a ready lexicon_ranker; received ${lexicon.metadata.status}/${lexicon.metadata.engine}.`
    );
  }
  const entryCount = lexicon.glossaryEntries.length;
  const weightCount = Object.keys(lexicon.tokenWeights).length;
  if (entryCount === 0 || entryCount !== lexicon.metadata.trainingDataset.recordCount) {
    throw new Error(
      `Lexicon entry count ${entryCount} does not match its ${lexicon.metadata.trainingDataset.recordCount} training records.`
    );
  }
  if (weightCount !== lexicon.metadata.trainingDataset.classCount) {
    throw new Error(
      `Lexicon token-weight count ${weightCount} does not match its ${lexicon.metadata.trainingDataset.classCount} classes.`
    );
  }
}

function parseMotionLibrary(value: unknown): MotionLibrary {
  const library = requireRecord(value, "motion library");
  if (library["schemaVersion"] !== 1) {
    throw new Error("Motion library schemaVersion must be 1.");
  }
  const clips = requireArray(library["clips"], "motion library clips").map(
    (clipValue, index): MotionLibraryClip => {
      const clip = requireRecord(clipValue, `motion library clip ${index}`);
      const fps = requireFiniteNumber(clip["fps"], `motion library clip ${index} fps`);
      if (fps < 5 || fps > 60) {
        throw new Error(`Motion library clip ${index} fps must be between 5 and 60.`);
      }
      if (clip["datasetId"] !== "include" || clip["expertReviewed"] !== false) {
        throw new Error(`Motion library clip ${index} has invalid provenance/review state.`);
      }
      return {
        id: requireString(clip["id"], `motion library clip ${index} id`),
        label: requireString(clip["label"], `motion library clip ${index} label`),
        normalizedLabel: requireString(
          clip["normalizedLabel"],
          `motion library clip ${index} normalizedLabel`
        ),
        fps,
        sourceSampleId: requireString(
          clip["sourceSampleId"],
          `motion library clip ${index} sourceSampleId`
        ),
        datasetId: "include",
        expertReviewed: false,
        frames: parseMotionFrames(clip["frames"], `motion library clip ${index}`, 16)
      };
    }
  );

  return {
    schemaVersion: 1,
    source: requireString(library["source"], "motion library source"),
    expertReviewStatus: requireString(
      library["expertReviewStatus"],
      "motion library expertReviewStatus"
    ),
    clipCount: requireNonnegativeInteger(library["clipCount"], "motion library clipCount"),
    clips
  };
}

function parseAlphabetLibrary(value: unknown): AlphabetLibrary {
  const library = requireRecord(value, "alphabet library");
  if (library["schemaVersion"] !== 1) {
    throw new Error("Alphabet library schemaVersion must be 1.");
  }
  const derivation = requireRecord(library["derivation"], "alphabet library derivation");
  if (derivation["rawVideoStored"] !== false || derivation["rawAudioStored"] !== false) {
    throw new Error("Alphabet fallback must report raw audio/video storage as false.");
  }
  const clips = requireArray(library["clips"], "alphabet library clips").map(
    (clipValue, index): AlphabetLibrary["clips"][number] => {
      const clip = requireRecord(clipValue, `alphabet clip ${index}`);
      const label = requireString(clip["label"], `alphabet clip ${index} label`);
      if (
        label.length !== 1 ||
        clip["datasetId"] !== "islrtc" ||
        clip["expertReviewed"] !== false
      ) {
        throw new Error(`Alphabet clip ${index} has invalid label/provenance/review state.`);
      }
      return {
        id: requireString(clip["id"], `alphabet clip ${index} id`),
        label,
        datasetId: "islrtc",
        expertReviewed: false,
        frames: parseMotionFrames(clip["frames"], `alphabet clip ${index}`, 1, true)
      };
    }
  );

  return {
    schemaVersion: 1,
    source: requireString(library["source"], "alphabet library source"),
    sourceUrl: requireUrl(library["sourceUrl"], "alphabet library sourceUrl"),
    usageTermsUrl: requireUrl(library["usageTermsUrl"], "alphabet library usageTermsUrl"),
    attribution: requireString(library["attribution"], "alphabet library attribution"),
    expertReviewStatus: requireString(
      library["expertReviewStatus"],
      "alphabet library expertReviewStatus"
    ),
    derivation: {
      rawVideoStored: false,
      rawAudioStored: false
    },
    clips
  };
}

function parseMotionFrames(
  value: unknown,
  label: string,
  minimumFrames: number,
  allowInactiveHands = false
): MotionFrame[] {
  const frames = requireArray(value, `${label} frames`);
  if (frames.length < minimumFrames) {
    throw new Error(`${label} must contain at least ${minimumFrames} frames.`);
  }
  return frames.map((frameValue, frameIndex): MotionFrame => {
    const frame = requireRecord(frameValue, `${label} frame ${frameIndex}`);
    const leftHand = parseMotionPoints(
      frame["leftHand"],
      `${label} frame ${frameIndex} leftHand`,
      21,
      allowInactiveHands
    );
    const rightHand = parseMotionPoints(
      frame["rightHand"],
      `${label} frame ${frameIndex} rightHand`,
      21,
      allowInactiveHands
    );
    if (allowInactiveHands && leftHand.length === 0 && rightHand.length === 0) {
      throw new Error(`${label} frame ${frameIndex} must contain at least one tracked hand.`);
    }
    return {
      pose: parseMotionPoints(frame["pose"], `${label} frame ${frameIndex} pose`, 25),
      leftHand,
      rightHand
    };
  });
}

function parseMotionPoints(
  value: unknown,
  label: string,
  expectedLength: number,
  allowEmpty = false
): MotionPoint[] {
  const points = requireArray(value, label);
  if (points.length !== expectedLength && !(allowEmpty && points.length === 0)) {
    throw new Error(`${label} must contain ${expectedLength} points.`);
  }
  return points.map((pointValue, pointIndex): MotionPoint => {
    const point = requireRecord(pointValue, `${label} point ${pointIndex}`);
    const x = requireFiniteNumber(point["x"], `${label} point ${pointIndex} x`);
    const y = requireFiniteNumber(point["y"], `${label} point ${pointIndex} y`);
    if (x < -0.5 || x > 1.5 || y < -0.5 || y > 1.5) {
      throw new Error(`${label} point ${pointIndex} is outside the supported coordinate range.`);
    }
    return { x, y };
  });
}

function requireRecord(value: unknown, label: string): Record<string, unknown> {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    throw new Error(`${label} must be an object.`);
  }
  return value as Record<string, unknown>;
}

function requireArray(value: unknown, label: string): unknown[] {
  if (!Array.isArray(value)) {
    throw new Error(`${label} must be an array.`);
  }
  return value;
}

function requireString(value: unknown, label: string): string {
  if (typeof value !== "string" || value.length === 0) {
    throw new Error(`${label} must be a non-empty string.`);
  }
  return value;
}

function requireFiniteNumber(value: unknown, label: string): number {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    throw new Error(`${label} must be a finite number.`);
  }
  return value;
}

function requireNonnegativeInteger(value: unknown, label: string): number {
  const parsed = requireFiniteNumber(value, label);
  if (!Number.isInteger(parsed) || parsed < 0) {
    throw new Error(`${label} must be a nonnegative integer.`);
  }
  return parsed;
}

function requireUrl(value: unknown, label: string): string {
  const parsed = requireString(value, label);
  try {
    new URL(parsed);
  } catch {
    throw new Error(`${label} must be a valid URL.`);
  }
  return parsed;
}

function assertMotionArtifacts(
  rawMotionCatalog: unknown,
  motionCatalog: MotionCatalogArtifact,
  avatarMotionCatalog: MotionCatalogArtifact,
  builtAvatarMotionCatalog: MotionCatalogArtifact,
  motionLibrary: MotionLibrary
): void {
  const catalogs = [motionCatalog, avatarMotionCatalog, builtAvatarMotionCatalog];
  for (const catalog of catalogs) {
    if (
      catalog.clipCount !== expectedMotionClipCount ||
      catalog.clips.length !== expectedMotionClipCount
    ) {
      throw new Error(
        `Release requires ${expectedMotionClipCount} validated motion clips; catalog reports ${catalog.clipCount}/${catalog.clips.length}.`
      );
    }
  }
  if (
    motionLibrary.clipCount !== expectedMotionClipCount ||
    motionLibrary.clips.length !== expectedMotionClipCount
  ) {
    throw new Error(
      `Release requires ${expectedMotionClipCount} renderer motions; library reports ${motionLibrary.clipCount}/${motionLibrary.clips.length}.`
    );
  }
  if (!motionLibrary.source.includes("INCLUDE")) {
    throw new Error(
      `Motion library must retain INCLUDE provenance; received ${motionLibrary.source}.`
    );
  }

  const catalogIds = motionCatalog.clips.map((clip) => clip.id);
  const libraryIds = motionLibrary.clips.map((clip) => clip.id);
  if (new Set(catalogIds).size !== expectedMotionClipCount) {
    throw new Error("Motion catalog clip IDs must be unique.");
  }
  if (JSON.stringify(catalogIds) !== JSON.stringify(libraryIds)) {
    throw new Error("Motion catalog and renderer library IDs must be identical and ordered.");
  }
  const sourceSampleIds = motionLibrary.clips.map((clip) => clip.sourceSampleId);
  if (new Set(sourceSampleIds).size !== expectedMotionClipCount) {
    throw new Error("Motion library source sample IDs must be unique.");
  }

  for (const [index, clip] of motionLibrary.clips.entries()) {
    const catalogClip = motionCatalog.clips[index];
    if (!catalogClip || catalogClip.frameCount !== clip.frames.length) {
      throw new Error(`Motion frame count does not match the catalog for ${clip.id}.`);
    }
    assertMotionIsNotStatic(clip.id, clip.frames);
  }

  const rawClips = requireArray(
    requireRecord(rawMotionCatalog, "motion catalog")["clips"],
    "motion catalog clips"
  );
  if (rawClips.length !== expectedMotionClipCount) {
    throw new Error("Raw motion catalog count changed during validation.");
  }
  for (const [index, value] of rawClips.entries()) {
    const clip = requireRecord(value, `motion catalog clip ${index}`);
    if (clip["expertReviewed"] !== false || ("playable" in clip && clip["playable"] !== true)) {
      throw new Error(`Motion catalog clip ${index} has an invalid review/playability state.`);
    }
    const normalizedLabel = requireString(
      clip["normalizedLabel"],
      `motion catalog clip ${index} normalizedLabel`
    );
    const sourceVideoPath = requireString(
      clip["sourceVideoPath"],
      `motion catalog clip ${index} sourceVideoPath`
    );
    const sourceDirectory = sourceVideoPath.split("/").at(-2);
    if (!sourceDirectory || normalizeIncludeLabel(sourceDirectory) !== normalizedLabel) {
      throw new Error(`Motion catalog clip ${index} label does not match its source directory.`);
    }
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

function assertMotionIsNotStatic(
  id: string,
  frames: MotionLibrary["clips"][number]["frames"]
): void {
  const trackedPoints = frames.flatMap((frame) => [
    frame.pose[0]!,
    frame.leftHand[8]!,
    frame.rightHand[8]!
  ]);
  const xValues = trackedPoints.map((point) => point.x);
  const yValues = trackedPoints.map((point) => point.y);
  const motionRange = Math.max(
    Math.max(...xValues) - Math.min(...xValues),
    Math.max(...yValues) - Math.min(...yValues)
  );
  if (motionRange <= 0.005) {
    throw new Error(`Motion clip ${id} is effectively static.`);
  }
}

function assertAlphabetLibrary(alphabetLibrary: AlphabetLibrary): void {
  if (
    alphabetLibrary.clips.length !== expectedAlphabetClipCount ||
    !alphabetLibrary.source.startsWith("ISLRTC official") ||
    !alphabetLibrary.attribution.includes("ISLRTC")
  ) {
    throw new Error("Alphabet fallback must contain the 26 official-source ISLRTC letter clips.");
  }
  const expectedLabels = Array.from({ length: expectedAlphabetClipCount }, (_, index) =>
    String.fromCharCode("A".charCodeAt(0) + index)
  );
  const labels = alphabetLibrary.clips.map((clip) => clip.label);
  const ids = alphabetLibrary.clips.map((clip) => clip.id);
  if (
    JSON.stringify(labels) !== JSON.stringify(expectedLabels) ||
    JSON.stringify(ids) !==
      JSON.stringify(expectedLabels.map((label) => `islrtc-alphabet-${label.toLowerCase()}`))
  ) {
    throw new Error("Alphabet fallback clips must be ordered A through Z with stable IDs.");
  }
}

function assertAlphabetEmbeddedInExtension(alphabetLibrary: AlphabetLibrary): void {
  const assetsDirectory = resolve(extensionDirectory, "assets");
  const javascriptBundle = readdirSync(assetsDirectory)
    .filter((name) => name.endsWith(".js"))
    .map((name) => readFileSync(resolve(assetsDirectory, name), "utf8"))
    .join("\n");
  const missingClipIds = alphabetLibrary.clips
    .map((clip) => clip.id)
    .filter((id) => !javascriptBundle.includes(id));
  if (missingClipIds.length > 0) {
    throw new Error(
      `Built extension is missing alphabet clips: ${missingClipIds.join(", ")}. Run pnpm build first.`
    );
  }
}

function assertCentroidArtifact(centroid: VideoModelArtifact): void {
  if (centroid.metadata.status !== "ready" || centroid.metadata.engine !== "keypoint_centroid") {
    throw new Error(
      `Centroid recognition artifact must be ready/keypoint_centroid; received ${centroid.metadata.status}/${centroid.metadata.engine}.`
    );
  }
  if (
    centroid.prototypes.length !== centroid.labelMap.length ||
    centroid.prototypes.length !== centroid.metadata.trainingDataset.classCount
  ) {
    throw new Error(
      "Centroid recognition labels, prototypes, and metadata class count do not match."
    );
  }
  if (centroid.prototypes.some((prototype) => prototype.centroid.length !== centroid.featureSize)) {
    throw new Error("Centroid recognition prototype feature sizes do not match the artifact.");
  }
  if (
    centroid.metrics &&
    (centroid.metrics.classCount !== centroid.prototypes.length ||
      centroid.metrics.sampleCount !== centroid.metadata.trainingDataset.recordCount)
  ) {
    throw new Error("Centroid recognition metrics do not match the validated artifact metadata.");
  }
}

function validateTemporalArtifact(
  temporal: TemporalModelArtifact,
  motionClipIds: ReadonlySet<string>
): string {
  if (temporal.metadata.status !== "ready" || temporal.metadata.engine !== "keypoint_transformer") {
    throw new Error(
      `Temporal recognition artifact must be ready/keypoint_transformer; received ${temporal.metadata.status}/${temporal.metadata.engine}.`
    );
  }
  if (
    basename(temporal.modelFile) !== temporal.modelFile ||
    !temporal.modelFile.endsWith(".onnx")
  ) {
    throw new Error(`Temporal modelFile must be an ONNX filename; received ${temporal.modelFile}.`);
  }
  if (
    temporal.labels.length !== temporal.metadata.trainingDataset.classCount ||
    temporal.labels.some((label, index) => label.index !== index)
  ) {
    throw new Error("Temporal labels must be contiguous and match the metadata class count.");
  }
  const unknownMotionId = temporal.labels.find((label) => !motionClipIds.has(label.motionClipId));
  if (unknownMotionId) {
    throw new Error(
      `Temporal label ${unknownMotionId.label} references missing motion ${unknownMotionId.motionClipId}.`
    );
  }

  const runtimePath = resolve(modelDirectory, temporal.modelFile);
  if (!existsSync(runtimePath)) {
    throw new Error(`Temporal metadata exists but its ONNX model is missing: ${runtimePath}.`);
  }
  const runtimeDigest = digestFile(runtimePath);
  if (temporal.modelBytes !== runtimeDigest.bytes) {
    throw new Error(
      `Temporal metadata reports ${temporal.modelBytes} bytes, but ${temporal.modelFile} has ${runtimeDigest.bytes}.`
    );
  }
  return runtimePath;
}

function assertMatchingFiles(leftPath: string, rightPath: string, label: string): void {
  if (digestFile(leftPath).sha256 !== digestFile(rightPath).sha256) {
    throw new Error(
      `Release ${label} are not identical. Run pnpm build after generation finishes.`
    );
  }
}

function assertMatchingJsonFiles(leftPath: string, rightPath: string, label: string): void {
  if (!isDeepStrictEqual(readJson<unknown>(leftPath), readJson<unknown>(rightPath))) {
    throw new Error(
      `Release ${label} are not structurally identical. Run pnpm build after generation finishes.`
    );
  }
}

function copyReleaseFile(stagingDirectory: string, input: ReleaseFileInput): PackagedFile {
  const destination = resolve(stagingDirectory, ...input.relativePath.split("/"));
  mkdirSync(dirname(destination), { recursive: true });
  copyFileSync(input.path, destination);
  return {
    path: input.relativePath,
    role: input.role,
    ...digestFile(destination)
  };
}

function deployApi(destination: string): void {
  const executable = process.platform === "win32" ? "pnpm.cmd" : "pnpm";
  const result = spawnSync(
    executable,
    [
      "--config.inject-workspace-packages=true",
      "--filter",
      "@signsaarthi/api",
      "--prod",
      "--offline",
      "deploy",
      destination
    ],
    {
      cwd: workspaceRoot,
      env: { ...process.env, CI: "true" },
      stdio: "inherit"
    }
  );

  if (result.error) {
    throw new Error(`API deployment failed: ${result.error.message}`);
  }
  if (result.status !== 0) {
    throw new Error(`API deployment failed with status ${result.status ?? "unknown"}.`);
  }
}

function sanitizeApiDeploymentManifest(apiDirectory: string): void {
  const packageVersions = Object.fromEntries(
    bundledWorkspacePackageDirectories.map((directory) => {
      const packageManifest = readJson<{ name: string; version: string }>(
        resolve(directory, "package.json")
      );
      return [packageManifest.name, packageManifest.version];
    })
  );
  const manifestPath = resolve(apiDirectory, "package.json");
  const manifest = readJson<Record<string, unknown> & { dependencies?: Record<string, string> }>(
    manifestPath
  );
  const sanitizedManifest = replaceBundledWorkspaceDependencies(manifest, packageVersions);
  const serializedManifest = `${JSON.stringify(sanitizedManifest, null, 2)}\n`;

  if (/file:\/\/\/|file:[A-Za-z]:[\\/]/u.test(serializedManifest)) {
    throw new Error("Deployed API package manifest contains an absolute local file reference.");
  }
  writeFileSync(manifestPath, serializedManifest, "utf8");
}

function pruneApiDeployment(apiDirectory: string): void {
  for (const path of [
    "src",
    "pnpm-lock.yaml",
    "tsconfig.json",
    "tsconfig.tsbuildinfo",
    "vite.config.ts",
    "node_modules/.modules.yaml",
    "node_modules/.pnpm-workspace-state-v1.json",
    "node_modules/.pnpm/lock.yaml",
    "node_modules/.pnpm/node_modules/@signsaarthi/api"
  ]) {
    rmSync(resolve(apiDirectory, path), { force: true, recursive: true });
  }
  removeNamedDirectories(apiDirectory, ".bin");
  removeBuildMetadata(resolve(apiDirectory, "dist"));

  const virtualStore = resolve(apiDirectory, "node_modules/.pnpm");
  for (const entry of readdirSync(virtualStore, { withFileTypes: true })) {
    if (!entry.isDirectory() || !entry.name.startsWith("@signsaarthi+")) {
      continue;
    }
    const packageScope = resolve(virtualStore, entry.name, "node_modules/@signsaarthi");
    if (!existsSync(packageScope)) {
      continue;
    }
    for (const packageEntry of readdirSync(packageScope, { withFileTypes: true })) {
      if (!packageEntry.isDirectory()) {
        continue;
      }
      const packageRoot = resolve(packageScope, packageEntry.name);
      rmSync(resolve(packageRoot, "src"), { force: true, recursive: true });
      rmSync(resolve(packageRoot, "tsconfig.json"), { force: true });
      rmSync(resolve(packageRoot, "tsconfig.tsbuildinfo"), { force: true });
      rmSync(resolve(packageRoot, "vite.config.ts"), { force: true });
      removeBuildMetadata(resolve(packageRoot, "dist"));
    }
  }

  const onnxRuntimeDirectory = realpathSync(resolve(apiDirectory, "node_modules/onnxruntime-node"));
  const nativeRuntimeRoot = resolve(onnxRuntimeDirectory, "bin/napi-v6");
  const targetRuntimeDirectory = resolve(nativeRuntimeRoot, process.platform, process.arch);
  if (!existsSync(targetRuntimeDirectory)) {
    throw new Error(
      `onnxruntime-node does not contain a native runtime for ${process.platform}-${process.arch}.`
    );
  }
  for (const platformEntry of readdirSync(nativeRuntimeRoot, { withFileTypes: true })) {
    const platformPath = resolve(nativeRuntimeRoot, platformEntry.name);
    if (platformEntry.name !== process.platform) {
      rmSync(platformPath, { force: true, recursive: true });
      continue;
    }
    for (const architectureEntry of readdirSync(platformPath, { withFileTypes: true })) {
      if (architectureEntry.name !== process.arch) {
        rmSync(resolve(platformPath, architectureEntry.name), { force: true, recursive: true });
      }
    }
  }
}

function removeNamedDirectories(directory: string, name: string): void {
  if (!existsSync(directory)) {
    return;
  }
  for (const entry of readdirSync(directory, { withFileTypes: true })) {
    const path = resolve(directory, entry.name);
    if (entry.isDirectory() && entry.name === name) {
      rmSync(path, { force: true, recursive: true });
    } else if (entry.isDirectory()) {
      removeNamedDirectories(path, name);
    }
  }
}

function removeFilesystemMetadataFiles(directory: string): void {
  if (!existsSync(directory)) {
    return;
  }
  for (const entry of readdirSync(directory, { withFileTypes: true })) {
    const path = resolve(directory, entry.name);
    if (entry.isDirectory()) {
      removeFilesystemMetadataFiles(path);
    } else if (entry.name === ".DS_Store" || entry.name.startsWith("._")) {
      rmSync(path, { force: true });
    }
  }
}

function removeAppleDoubleSibling(path: string): void {
  rmSync(resolve(dirname(path), `._${basename(path)}`), { force: true });
}

function removeBuildMetadata(directory: string): void {
  if (!existsSync(directory)) {
    return;
  }
  for (const entry of readdirSync(directory, { withFileTypes: true })) {
    const path = resolve(directory, entry.name);
    if (entry.isDirectory()) {
      removeBuildMetadata(path);
    } else if (entry.name.endsWith(".d.ts") || entry.name.endsWith(".test.js")) {
      rmSync(path, { force: true });
    }
  }
}

function readSourceState(): ReleaseSourceState {
  if (!isGitWorkTree()) {
    const snapshot = snapshotFilesystemSource(workspaceRoot);
    return {
      method: "filesystem",
      state: "filesystem",
      head: null,
      dirty: null,
      changes: null,
      trackedFiles: null,
      untrackedFiles: null,
      allowlistedFiles: snapshot.paths,
      excludedGitFiles: null,
      statusSha256: null,
      pathSetSha256: snapshot.pathSetSha256,
      allowlist: snapshot.allowlist,
      exclusions: snapshot.exclusions,
      tree: snapshot.tree
    };
  }

  const headResult = spawnSync("git", ["rev-parse", "--verify", "HEAD"], {
    cwd: workspaceRoot,
    encoding: "utf8"
  });
  const head = headResult.status === 0 ? headResult.stdout.trim() : null;
  const status = gitOutput(["status", "--porcelain=v1", "--untracked-files=all"]);
  const trackedFiles = nullSeparated(gitOutput(["ls-files", "--cached", "-z"]));
  const untrackedFiles = nullSeparated(
    gitOutput(["ls-files", "--others", "--exclude-standard", "-z"])
  );
  const allGitFiles = [...new Set([...trackedFiles, ...untrackedFiles])];
  const sourceFiles = filterAllowlistedSourcePaths(allGitFiles);
  const tree = digestSourceFiles(workspaceRoot, sourceFiles);
  const changes = status.split("\n").filter(Boolean).length;
  const dirty = changes > 0;

  return {
    method: "git",
    state: head ? (dirty ? "uncommitted" : "clean") : dirty ? "no-head-uncommitted" : "no-head",
    head,
    dirty,
    changes,
    trackedFiles: trackedFiles.length,
    untrackedFiles: untrackedFiles.length,
    allowlistedFiles: sourceFiles.length,
    excludedGitFiles: allGitFiles.length - sourceFiles.length,
    statusSha256: sha256(status),
    pathSetSha256: sha256(sourceFiles.join("\0")),
    ...releaseSourceSelectionPolicy(),
    tree
  };
}

function isGitWorkTree(): boolean {
  const result = spawnSync("git", ["rev-parse", "--is-inside-work-tree"], {
    cwd: workspaceRoot,
    encoding: "utf8"
  });
  return result.status === 0 && result.stdout.trim() === "true";
}

function readBuildInputState(): string {
  const hash = createHash("sha256");
  hash.update("signsaarthi-build-inputs-v1\0");
  for (const directory of buildInputDirectories) {
    if (!existsSync(directory)) {
      throw new Error(`Release build input is missing: ${directory}. Run pnpm build first.`);
    }
    const digest = digestTree(directory);
    hash.update(`${relativeBuildPath(directory)}\0${digest.sha256}\0`);
  }
  return hash.digest("hex");
}

function assertPackagingInputsStable(
  initialSourceState: ReleaseSourceState,
  initialBuildInputState: string
): void {
  const finalSourceState = readSourceState();
  const finalBuildInputState = readBuildInputState();
  if (
    finalSourceState.method !== initialSourceState.method ||
    finalSourceState.tree.sha256 !== initialSourceState.tree.sha256 ||
    finalSourceState.statusSha256 !== initialSourceState.statusSha256 ||
    finalSourceState.pathSetSha256 !== initialSourceState.pathSetSha256 ||
    finalSourceState.head !== initialSourceState.head ||
    finalBuildInputState !== initialBuildInputState
  ) {
    throw new Error(
      "Source or build outputs changed while release staging was in progress. No bundle was published; retry after concurrent edits/builds finish."
    );
  }
}

function relativeBuildPath(path: string): string {
  const prefix = `${workspaceRoot.replace(/[\\/]$/, "")}/`;
  return path.startsWith(prefix) ? path.slice(prefix.length) : path;
}

function publishBundle(stagingDirectory: string, destination: string): void {
  const backupDirectory = `${destination}.previous-${process.pid}`;
  rmSync(backupDirectory, { force: true, recursive: true });
  const hadPreviousBundle = existsSync(destination);
  if (hadPreviousBundle) {
    renameSync(destination, backupDirectory);
  }

  try {
    renameSync(stagingDirectory, destination);
  } catch (error) {
    if (hadPreviousBundle && existsSync(backupDirectory) && !existsSync(destination)) {
      renameSync(backupDirectory, destination);
    }
    throw error;
  }

  rmSync(backupDirectory, { force: true, recursive: true });
}

function publishImmutableFile(stagedPath: string, destination: string): void {
  if (existsSync(destination)) {
    if (digestFile(stagedPath).sha256 !== digestFile(destination).sha256) {
      throw new Error(
        `Content-addressed release path already exists with different bytes: ${destination}.`
      );
    }
    rmSync(stagedPath, { force: true });
  } else {
    renameSync(stagedPath, destination);
  }
  chmodSync(destination, 0o444);
}

function commandVersion(command: "corepack" | "pnpm"): string {
  const executable = process.platform === "win32" ? `${command}.cmd` : command;
  const result = spawnSync(executable, ["--version"], {
    cwd: workspaceRoot,
    encoding: "utf8"
  });
  if (result.error || result.status !== 0) {
    throw new Error(
      `Could not run ${command} --version. Install Node >= ${MINIMUM_NODE_VERSION} and run corepack enable.`
    );
  }
  return result.stdout.trim();
}

function gitOutput(args: string[]): string {
  const result = spawnSync("git", args, {
    cwd: workspaceRoot,
    encoding: "utf8",
    maxBuffer: maximumGitOutputBytes
  });
  if (result.error || result.status !== 0) {
    throw new Error(
      `git ${args.join(" ")} failed: ${result.stderr.trim() || result.error?.message}`
    );
  }
  return result.stdout;
}

function nullSeparated(output: string): string[] {
  return output.split("\0").filter(Boolean);
}

function extensionIdFromPublicKey(encodedKey: string): string {
  const publicKey = Buffer.from(encodedKey, "base64");
  if (publicKey.byteLength === 0) {
    throw new Error("The production extension manifest is missing its fixed public key.");
  }
  const idBytes = createHash("sha256").update(publicKey).digest().subarray(0, 16);
  return [...idBytes]
    .flatMap((byte) => [byte >> 4, byte & 0x0f])
    .map((nibble) => String.fromCharCode("a".charCodeAt(0) + nibble))
    .join("");
}

function normalizeApiBaseUrl(value: string): string {
  const url = new URL(value);
  if (
    url.protocol !== "http:" ||
    url.hostname !== "127.0.0.1" ||
    !url.port ||
    url.username ||
    url.password ||
    (url.pathname !== "/" && url.pathname !== "") ||
    url.search ||
    url.hash
  ) {
    throw new Error(
      `VITE_SIGNSAARTHI_API_BASE_URL must be an explicit 127.0.0.1 HTTP origin with a port; received ${value}.`
    );
  }
  return url.origin;
}

function readJson<T>(path: string): T {
  if (!existsSync(path)) {
    throw new Error(`Release input is missing: ${path}. Run pnpm build first.`);
  }
  return JSON.parse(readFileSync(path, "utf8")) as T;
}

function startScript(releaseRuntimeId: string, apiPort: number): string {
  return `import { register } from "node:module";
import { dirname } from "node:path";
import { fileURLToPath } from "node:url";

const minimum = [22, 13, 0];
const current = process.versions.node.split(".").map(Number);
const supported = minimum.every((part, index) => {
  const prefixMatches = minimum.slice(0, index).every((value, prefixIndex) => current[prefixIndex] === value);
  return !prefixMatches || (current[index] ?? 0) >= part;
});
if (!supported) {
  throw new Error("SignSaarthi requires Node ${MINIMUM_NODE_VERSION} or newer.");
}

const bundleDirectory = dirname(fileURLToPath(import.meta.url));
process.chdir(bundleDirectory);
process.env.API_PORT ??= ${JSON.stringify(String(apiPort))};
process.env.SIGNSAARTHI_RUNTIME_ID ??= ${JSON.stringify(releaseRuntimeId)};
register(${JSON.stringify(`./${jsonModuleLoaderRelativePath}`)}, import.meta.url);
await import("./api/dist/main.js");
`;
}

function jsonModuleLoader(): string {
  return `export async function load(url, context, nextLoad) {
  if (url.endsWith(".json")) {
    return nextLoad(url, {
      ...context,
      importAttributes: { ...context.importAttributes, type: "json" }
    });
  }
  return nextLoad(url, context);
}
`;
}

function bundleReadme(input: {
  apiBaseUrl: string;
  arch: string;
  extensionId: string;
  minimumNodeVersion: string;
  motionClipCount: number;
  platform: string;
  releaseAttestation: ReturnType<typeof createReleaseAttestation>;
  recognitionModelReady: boolean;
  releaseRuntimeId: string;
}): string {
  const provenanceNotice =
    input.releaseAttestation.status === "attested"
      ? `ATTESTED PRODUCTION RELEASE: clean Git HEAD ${input.releaseAttestation.gitHead}.`
      : `UNATTESTED DEVELOPMENT RELEASE: ${input.releaseAttestation.reason}`;
  return `# SignSaarthi AI local release

> ${provenanceNotice}

This directory is a self-contained local runtime bundle for ${input.platform}-${input.arch}. The Chrome extension ZIP is portable, and the API dependency tree contains the native runtime dependencies for this target.

The ready release content is the validated lexicon, ${input.motionClipCount} INCLUDE motion clips, and the official-source ISLRTC A-Z fingerspelling fallback. Recognition model ready: ${input.recognitionModelReady ? "yes; validated recognition artifacts are included" : "no; no recognition artifacts or recognition metrics are included"}. Raw audio and video storage are disabled.

## Requirements

- Node.js ${input.minimumNodeVersion} or newer
- Chrome 116 or newer

No Python environment, training corpus, workspace checkout, package install, or model retraining is required.

## Start the API

From this directory, run:

\`\`\`bash
node ./start-api.mjs
\`\`\`

The API listens on ${input.apiBaseUrl}. Its expected health runtime ID is \`${input.releaseRuntimeId}\`.

## Load the extension

1. Extract \`extension/signsaarthi-extension.zip\` into a new directory.
2. Open \`chrome://extensions\`.
3. Enable Developer mode, select Load unpacked, and choose the extracted directory.
4. Keep the local API running while using the extension.

Chrome must show extension ID \`${input.extensionId}\`. The API accepts extension traffic only from that fixed identity and expects the \`X-SignSaarthi-Client: extension-v1\` request header.

The complete payload inventory, checksums, artifact provenance, allowlisted source state, and runtime versions are recorded in \`${releaseManifestName}\`. Every path in that manifest is relative to the manifest itself. The complete release archive is content-addressed by SHA-256 and accompanied by a matching \`.sha256\` sidecar.
`;
}
