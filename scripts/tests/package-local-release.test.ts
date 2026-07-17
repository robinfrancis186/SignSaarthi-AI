import assert from "node:assert/strict";
import {
  chmodSync,
  mkdirSync,
  mkdtempSync,
  readFileSync,
  renameSync,
  rmSync,
  symlinkSync,
  writeFileSync
} from "node:fs";
import { tmpdir } from "node:os";
import { basename, join } from "node:path";
import { afterEach, test } from "node:test";
import {
  RELEASE_INTEGRITY_SCHEMA_VERSION,
  RELEASE_MANIFEST_SCHEMA_VERSION,
  createReleaseAttestation,
  createReleaseTreeManifest,
  digestFile,
  digestSourceFiles,
  digestTree,
  filterAllowlistedSourcePaths,
  replaceBundledWorkspaceDependencies,
  resolveReleaseMode,
  snapshotFilesystemSource,
  verifyReleaseArchive,
  verifyReleaseBundle,
  type ReleaseSourceState,
  writeDeterministicZip
} from "../lib/release-utils.js";

const releaseManifestName = "signsaarthi-release-manifest.json";
const temporaryDirectories: string[] = [];

afterEach(() => {
  for (const directory of temporaryDirectories.splice(0)) {
    rmSync(directory, { force: true, recursive: true });
  }
});

void test("filesystem source snapshots use an explicit allowlist and exclude private/local roots", () => {
  const root = makeTemporaryDirectory();
  writeFixture(root, "apps/extension/src/index.ts", "export const ready = true;\n");
  writeFixture(root, "data/models/model.json", "{}\n");
  writeFixture(root, "data/isl/catalog.json", "{}\n");
  writeFixture(root, ".env.example", "API_PORT=8787\n");

  for (const path of [
    "data/raw/isign/private.json",
    "private/credentials.json",
    "notes/internal.md",
    "data/models/isl-sentence-planner.private.json",
    "candidates/result.json",
    "node_modules/package/index.js",
    "packages/example/dist/index.js",
    "output/release.zip",
    "tmp/scratch.txt",
    "scripts/__pycache__/cache.pyc",
    ".venv-release/secret.txt",
    ".github/workflows/unrelated.yml"
  ]) {
    writeFixture(root, path, "not part of release source\n");
  }
  for (const path of [
    ".env",
    ".env.production",
    "apps/extension/local.pem",
    "apps/extension/settings.local",
    "data/models/._model.json",
    "scripts/.DS_Store"
  ]) {
    writeFixture(root, path, "excluded\n");
  }

  const first = snapshotFilesystemSource(root);
  const second = snapshotFilesystemSource(root);
  assert.deepEqual(second, first);
  assert.equal(first.method, "filesystem");
  assert.equal(first.paths, 4);
  assert.equal(first.tree.files, 4);
  assert.equal(first.tree.missingFiles, 0);
  assert.deepEqual(first.allowlist.directoryRoots, [
    "apps",
    "data/isl",
    "data/models",
    "docs",
    "packages",
    "scripts"
  ]);
  assert.deepEqual(first.exclusions.pathPrefixes, ["data/raw", "private"]);

  writeFixture(root, "data/raw/isign/private.json", "excluded mutation\n");
  writeFixture(root, "private/credentials.json", "excluded mutation\n");
  writeFixture(root, "notes/internal.md", "excluded mutation\n");
  writeFixture(root, "data/models/isl-sentence-planner.private.json", "excluded mutation\n");
  writeFixture(root, ".env", "SECRET=changed\n");
  assert.deepEqual(snapshotFilesystemSource(root), first);

  writeFixture(root, "apps/extension/src/index.ts", "export const ready = false;\n");
  assert.notEqual(snapshotFilesystemSource(root).tree.sha256, first.tree.sha256);

  assert.deepEqual(
    filterAllowlistedSourcePaths([
      "private/credentials.json",
      "apps/extension/src/index.ts",
      "data/raw/isign/private.json",
      "data/models/isl-sentence-planner.private.json",
      "apps/extension/src/index.ts",
      ".env"
    ]),
    ["apps/extension/src/index.ts"]
  );
});

void test("bundled workspace dependency manifests do not retain local file references", () => {
  const input = {
    name: "@signsaarthi/api",
    dependencies: {
      "@signsaarthi/shared": "@signsaarthi/shared@file:///Users/example/project/packages/shared",
      fastify: "5.10.0"
    }
  };

  const sanitized = replaceBundledWorkspaceDependencies(input, {
    "@signsaarthi/shared": "0.1.0"
  });
  assert.deepEqual(sanitized.dependencies, {
    "@signsaarthi/shared": "0.1.0",
    fastify: "5.10.0"
  });
  assert.match(input.dependencies["@signsaarthi/shared"], /^@signsaarthi\/shared@file:/);
  assert.throws(
    () => replaceBundledWorkspaceDependencies(input, {}),
    /has no verified package version/
  );
});

void test("source digests sort and deduplicate paths while recording missing files", () => {
  const root = makeTemporaryDirectory();
  writeFixture(root, "b.ts", "b\n");
  writeFixture(root, "a.ts", "a\n");
  chmodSync(join(root, "b.ts"), 0o755);

  const first = digestSourceFiles(root, ["b.ts", "missing.ts", "a.ts", "b.ts"]);
  const second = digestSourceFiles(root, ["a.ts", "b.ts", "missing.ts"]);
  assert.deepEqual(second, first);
  assert.equal(first.files, 2);
  assert.equal(first.missingFiles, 1);
  assert.throws(() => digestSourceFiles(root, ["../escape.ts"]), /escapes the workspace/);
});

void test("production releases require clean Git provenance while development stays unattested", () => {
  const root = makeTemporaryDirectory();
  writeFixture(root, "package.json", "{}\n");
  const filesystemSource = filesystemSourceState(root);

  assert.deepEqual(resolveReleaseMode({}), "development");
  assert.deepEqual(resolveReleaseMode({ CI: "true" }), "production");
  assert.deepEqual(
    resolveReleaseMode({ CI: "true", SIGNSAARTHI_RELEASE_MODE: "development" }),
    "development"
  );
  assert.throws(
    () => resolveReleaseMode({ SIGNSAARTHI_RELEASE_MODE: "preview" }),
    /must be development or production/
  );

  const development = createReleaseAttestation("development", filesystemSource);
  assert.equal(development.status, "unattested");
  assert.equal(development.basis, "development-filesystem");
  assert.throws(
    () => createReleaseAttestation("production", filesystemSource),
    /requires clean Git provenance/
  );

  const cleanGitSource: ReleaseSourceState = {
    ...filesystemSource,
    method: "git",
    state: "clean",
    head: "a".repeat(40),
    dirty: false,
    changes: 0,
    trackedFiles: 1,
    untrackedFiles: 0,
    excludedGitFiles: 0,
    statusSha256: "b".repeat(64)
  };
  const production = createReleaseAttestation("production", cleanGitSource);
  assert.equal(production.status, "attested");
  assert.equal(production.gitHead, cleanGitSource.head);

  assert.throws(
    () =>
      createReleaseAttestation("production", {
        ...cleanGitSource,
        dirty: true,
        changes: 1,
        state: "uncommitted"
      }),
    /requires clean Git provenance/
  );
});

void test("full release manifests detect modified, missing, extra, and forbidden payload files", () => {
  const root = makeTemporaryDirectory();
  const bundle = join(root, "bundle");
  writeReleaseFixture(bundle);

  const verified = verifyReleaseBundle(bundle, releaseManifestName);
  assert.equal(verified.entries, 3);
  assert.equal(verified.payload.files, 2);
  assert.equal(verified.payload.symlinks, 1);

  writeFixture(bundle, "README.md", "tampered\n");
  assert.throws(
    () => verifyReleaseBundle(bundle, releaseManifestName),
    /failed integrity verification: README\.md/
  );
  writeFixture(bundle, "README.md", "release instructions\n");
  assert.doesNotThrow(() => verifyReleaseBundle(bundle, releaseManifestName));

  writeFixture(bundle, "unexpected.txt", "not in manifest\n");
  assert.throws(
    () => verifyReleaseBundle(bundle, releaseManifestName),
    /unmanifested entry unexpected\.txt/
  );
  rmSync(join(bundle, "unexpected.txt"));

  rmSync(join(bundle, "api/payload.txt"));
  assert.throws(
    () => verifyReleaseBundle(bundle, releaseManifestName),
    /missing manifest entry api\/payload\.txt/
  );

  writeFixture(bundle, "api/payload.txt", "runtime payload\n");
  const withoutFinderMetadata = createReleaseTreeManifest(bundle, [releaseManifestName]);
  writeFixture(bundle, ".DS_Store", "finder metadata\n");
  assert.deepEqual(createReleaseTreeManifest(bundle, [releaseManifestName]), withoutFinderMetadata);
  writeFixture(bundle, "._README.md", "appledouble metadata\n");
  assert.throws(
    () => createReleaseTreeManifest(bundle, [releaseManifestName]),
    /forbidden AppleDouble metadata/
  );
  rmSync(join(bundle, "._README.md"));
  writeFixture(bundle, "private/secret.txt", "private release data\n");
  assert.throws(
    () => createReleaseTreeManifest(bundle, [releaseManifestName]),
    /forbidden private\/raw path/
  );
});

void test("content-addressed archives preserve symlinks and reject checksum or payload tampering", () => {
  const root = makeTemporaryDirectory();
  const bundle = join(root, "bundle");
  writeReleaseFixture(bundle);

  const temporaryArchive = join(root, "temporary.zip");
  const digest = writeDeterministicZip(bundle, temporaryArchive);
  const archive = join(root, `signsaarthi-local-release-${digest.sha256}.zip`);
  renameSync(temporaryArchive, archive);
  const checksum = `${archive}.sha256`;
  writeFileSync(checksum, `${digest.sha256}  ${basename(archive)}\n`, "utf8");

  const verified = verifyReleaseArchive(archive, checksum, releaseManifestName);
  assert.equal(verified.archive.sha256, digest.sha256);
  assert.equal(verified.payload.symlinks, 1);

  const originalArchive = readFileSync(archive);
  const tamperedArchive = Buffer.from(originalArchive);
  tamperedArchive[Math.floor(tamperedArchive.byteLength / 2)]! ^= 0xff;
  writeFileSync(archive, tamperedArchive);
  assert.throws(
    () => verifyReleaseArchive(archive, checksum, releaseManifestName),
    /archive checksum mismatch/
  );
  writeFileSync(archive, originalArchive);

  writeFixture(bundle, "README.md", "tampered after manifest generation\n");
  const rebuiltTemporaryArchive = join(root, "rebuilt.zip");
  const rebuiltDigest = writeDeterministicZip(bundle, rebuiltTemporaryArchive);
  const rebuiltArchive = join(root, `signsaarthi-local-release-${rebuiltDigest.sha256}.zip`);
  renameSync(rebuiltTemporaryArchive, rebuiltArchive);
  const rebuiltChecksum = `${rebuiltArchive}.sha256`;
  writeFileSync(rebuiltChecksum, `${rebuiltDigest.sha256}  ${basename(rebuiltArchive)}\n`, "utf8");
  assert.throws(
    () => verifyReleaseArchive(rebuiltArchive, rebuiltChecksum, releaseManifestName),
    /failed integrity verification: README\.md/
  );
});

function writeReleaseFixture(bundle: string): void {
  writeFixture(bundle, "README.md", "release instructions\n");
  writeFixture(bundle, "api/payload.txt", "runtime payload\n");
  symlinkSync("payload.txt", join(bundle, "api/payload-link.txt"));
  const source = filesystemSourceState(bundle);
  const manifest = {
    schemaVersion: RELEASE_MANIFEST_SCHEMA_VERSION,
    product: "Test release",
    artifacts: {
      localApi: {
        directory: "api",
        ...digestTree(join(bundle, "api"))
      },
      instructions: {
        path: "README.md",
        ...digestFile(join(bundle, "README.md"))
      }
    },
    provenance: {
      attestation: createReleaseAttestation("development", source),
      source
    },
    integrity: {
      schemaVersion: RELEASE_INTEGRITY_SCHEMA_VERSION,
      manifest: {
        path: releaseManifestName,
        sealedBy: "content-addressed archive SHA-256"
      },
      payload: createReleaseTreeManifest(bundle, [releaseManifestName])
    }
  };
  writeFileSync(
    join(bundle, releaseManifestName),
    `${JSON.stringify(manifest, null, 2)}\n`,
    "utf8"
  );
}

function filesystemSourceState(root: string): ReleaseSourceState {
  const snapshot = snapshotFilesystemSource(root);
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

function makeTemporaryDirectory(): string {
  const directory = mkdtempSync(join(tmpdir(), "signsaarthi-release-test-"));
  temporaryDirectories.push(directory);
  return directory;
}

function writeFixture(root: string, relativePath: string, contents: string): void {
  const path = join(root, ...relativePath.split("/"));
  mkdirSync(join(path, ".."), { recursive: true });
  writeFileSync(path, contents, "utf8");
}
