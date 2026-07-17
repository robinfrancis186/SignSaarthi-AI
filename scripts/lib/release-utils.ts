import { createHash } from "node:crypto";
import {
  chmodSync,
  existsSync,
  lstatSync,
  mkdirSync,
  mkdtempSync,
  readFileSync,
  readdirSync,
  readlinkSync,
  rmSync,
  symlinkSync,
  writeFileSync
} from "node:fs";
import { tmpdir } from "node:os";
import { basename, dirname, isAbsolute, join, posix, relative, resolve, sep } from "node:path";
import { isDeepStrictEqual } from "node:util";
import { strFromU8, strToU8, unzipSync, zipSync, type Zippable } from "fflate";

export const MINIMUM_NODE_VERSION = "22.13.0";
export const REPRODUCIBLE_ZIP_TIMESTAMP = "1980-01-01 00:00:00";
export const RELEASE_MANIFEST_SCHEMA_VERSION = 4;
export const RELEASE_INTEGRITY_SCHEMA_VERSION = 1;

const fixedZipMtime = new Date(1980, 0, 1, 0, 0, 0);
const sha256Pattern = /^[a-f0-9]{64}$/;
const gitHeadPattern = /^[a-f0-9]{40,64}$/;

export type ArtifactDigest = {
  bytes: number;
  sha256: string;
};

export type TreeDigest = ArtifactDigest & {
  files: number;
  symlinks: number;
};

export type SourceTreeDigest = TreeDigest & {
  missingFiles: number;
};

export const FILESYSTEM_SOURCE_ALLOWED_DIRECTORIES = [
  "apps",
  "data/isl",
  "data/models",
  "docs",
  "packages",
  "scripts"
] as const;

export const FILESYSTEM_SOURCE_ALLOWED_FILES = [
  ".env.example",
  ".github/workflows/release.yml",
  ".gitignore",
  ".node-version",
  ".prettierrc",
  "LICENSE",
  "NOTICE",
  "README.md",
  "eslint.config.js",
  "package.json",
  "pnpm-lock.yaml",
  "pnpm-workspace.yaml",
  "tsconfig.base.json",
  "tsconfig.json",
  "tsconfig.scripts.json"
] as const;

export const FILESYSTEM_SOURCE_EXCLUDED_PATH_PREFIXES = ["data/raw", "private"] as const;

export const FILESYSTEM_SOURCE_EXCLUDED_DIRECTORIES = [
  ".cache",
  ".git",
  ".hg",
  ".svn",
  ".venv",
  ".vite",
  "__pycache__",
  "coverage",
  "dist",
  "node_modules",
  "output",
  "secrets",
  "temp",
  "tmp"
] as const;

export const FILESYSTEM_SOURCE_EXCLUDED_FILE_PATTERNS = [
  "._*",
  ".DS_Store",
  ".env",
  ".env.* (except .env.example)",
  "*.local",
  "*.private.*",
  "*.pyc",
  "*.tsbuildinfo",
  "*.key",
  "*.p12",
  "*.pem",
  "*.pfx"
] as const;

export type SourceSelectionPolicy = {
  allowlist: {
    directoryRoots: readonly string[];
    filePaths: readonly string[];
  };
  exclusions: {
    pathPrefixes: readonly string[];
    directoryNames: readonly string[];
    filePatterns: readonly string[];
  };
};

export type FilesystemSourceSnapshot = SourceSelectionPolicy & {
  method: "filesystem";
  pathSetSha256: string;
  paths: number;
  tree: SourceTreeDigest;
};

export type ReleaseSourceState = SourceSelectionPolicy & {
  method: "filesystem" | "git";
  state: "clean" | "filesystem" | "no-head" | "no-head-uncommitted" | "uncommitted";
  head: string | null;
  dirty: boolean | null;
  changes: number | null;
  trackedFiles: number | null;
  untrackedFiles: number | null;
  allowlistedFiles: number;
  excludedGitFiles: number | null;
  statusSha256: string | null;
  pathSetSha256: string;
  tree: SourceTreeDigest;
};

export type ReleaseMode = "development" | "production";

export type PackageManifestLike = {
  dependencies?: Record<string, string>;
  [key: string]: unknown;
};

export function replaceBundledWorkspaceDependencies(
  manifest: PackageManifestLike,
  packageVersions: Readonly<Record<string, string>>
): PackageManifestLike {
  const dependencies = manifest.dependencies;
  if (!dependencies) {
    return { ...manifest };
  }

  const rewrittenDependencies = { ...dependencies };
  for (const [name, reference] of Object.entries(rewrittenDependencies)) {
    if (!name.startsWith("@signsaarthi/")) {
      continue;
    }

    const version = packageVersions[name];
    if (!version) {
      throw new Error(`Bundled workspace dependency ${name} has no verified package version.`);
    }
    const isExpectedReference =
      reference === version ||
      reference.startsWith("workspace:") ||
      reference.startsWith("file:") ||
      reference.startsWith(`${name}@file:`);
    if (!isExpectedReference) {
      throw new Error(
        `Bundled workspace dependency ${name} has unexpected reference ${reference}.`
      );
    }
    rewrittenDependencies[name] = version;
  }

  return {
    ...manifest,
    dependencies: rewrittenDependencies
  };
}

export type ReleaseAttestation = {
  status: "attested" | "unattested";
  releaseMode: ReleaseMode;
  basis: "clean-git-head" | "development-filesystem" | "development-git-worktree";
  gitHead: string | null;
  reason: string;
};

export type ReleaseTreeManifestFile = {
  path: string;
  kind: "file";
  mode: "100644" | "100755";
  bytes: number;
  sha256: string;
};

export type ReleaseTreeManifestSymlink = {
  path: string;
  kind: "symlink";
  target: string;
};

export type ReleaseTreeManifestEntry = ReleaseTreeManifestFile | ReleaseTreeManifestSymlink;

export type ReleaseTreeManifest = {
  schemaVersion: 1;
  algorithm: "sha256";
  entryCount: number;
  tree: TreeDigest;
  entries: ReleaseTreeManifestEntry[];
};

export type ReleaseBundleVerification = {
  manifest: ArtifactDigest;
  payload: TreeDigest;
  entries: number;
};

export type ReleaseArchiveVerification = ReleaseBundleVerification & {
  archive: ArtifactDigest;
  checksumPath: string;
};

export function assertSupportedNode(): void {
  if (compareVersions(process.versions.node, MINIMUM_NODE_VERSION) < 0) {
    throw new Error(
      `Node ${MINIMUM_NODE_VERSION} or newer is required; current runtime is ${process.versions.node}.`
    );
  }
}

export function resolveReleaseMode(environment: NodeJS.ProcessEnv = process.env): ReleaseMode {
  const explicitMode = environment["SIGNSAARTHI_RELEASE_MODE"];
  if (explicitMode !== undefined) {
    if (explicitMode !== "development" && explicitMode !== "production") {
      throw new Error(
        `SIGNSAARTHI_RELEASE_MODE must be development or production; received ${explicitMode}.`
      );
    }
    return explicitMode;
  }
  return /^(?:1|true)$/i.test(environment["CI"] ?? "") ? "production" : "development";
}

export function createReleaseAttestation(
  releaseMode: ReleaseMode,
  source: ReleaseSourceState
): ReleaseAttestation {
  const cleanGitSource =
    source.method === "git" &&
    source.state === "clean" &&
    source.dirty === false &&
    source.changes === 0 &&
    source.untrackedFiles === 0 &&
    source.trackedFiles !== null &&
    source.trackedFiles > 0 &&
    source.tree.missingFiles === 0 &&
    source.head !== null &&
    gitHeadPattern.test(source.head);

  if (releaseMode === "production") {
    if (!cleanGitSource) {
      throw new Error(
        "Production release requires clean Git provenance: a valid HEAD, zero tracked or untracked changes, and no missing allowlisted source files."
      );
    }
    return {
      status: "attested",
      releaseMode,
      basis: "clean-git-head",
      gitHead: source.head,
      reason: `Built from clean Git HEAD ${source.head}.`
    };
  }

  if (source.method === "filesystem") {
    return {
      status: "unattested",
      releaseMode,
      basis: "development-filesystem",
      gitHead: null,
      reason:
        "No Git worktree was available; provenance is an allowlisted filesystem snapshot only."
    };
  }

  return {
    status: "unattested",
    releaseMode,
    basis: "development-git-worktree",
    gitHead: source.head,
    reason:
      "Development mode does not claim production provenance, even when Git metadata is present."
  };
}

export function digestFile(path: string): ArtifactDigest {
  const contents = readFileSync(path);
  return {
    bytes: contents.byteLength,
    sha256: sha256(contents)
  };
}

export function digestTree(root: string): TreeDigest {
  return digestTreeEntries(
    root,
    listTreeEntries(root).filter((entry) => !isIgnoredReleaseMetadata(entry.path))
  );
}

export function digestSourceFiles(root: string, paths: readonly string[]): SourceTreeDigest {
  const hash = createHash("sha256");
  let bytes = 0;
  let files = 0;
  let missingFiles = 0;
  let symlinks = 0;
  const sortedPaths = [...new Set(paths)].sort(comparePaths);
  hash.update("signsaarthi-source-tree-v1\0");

  for (const path of sortedPaths) {
    assertRelativeSourcePath(path);
    const absolutePath = resolve(root, ...path.split("/"));
    if (!lstatExists(absolutePath)) {
      hash.update(`missing\0${path}\0`);
      missingFiles += 1;
      continue;
    }
    const stat = lstatSync(absolutePath);
    if (stat.isSymbolicLink()) {
      const target = readlinkSync(absolutePath);
      hash.update(`symlink\0${path}\0${target}\0`);
      symlinks += 1;
      continue;
    }
    if (!stat.isFile()) {
      throw new Error(`Source state contains an unsupported entry: ${absolutePath}`);
    }
    const contents = readFileSync(absolutePath);
    const mode = (stat.mode & 0o111) !== 0 ? "100755" : "100644";
    hash.update(`file\0${path}\0${mode}\0${contents.byteLength}\0`);
    hash.update(contents);
    hash.update("\0");
    bytes += contents.byteLength;
    files += 1;
  }

  return {
    bytes,
    files,
    missingFiles,
    sha256: hash.digest("hex"),
    symlinks
  };
}

export function snapshotFilesystemSource(root: string): FilesystemSourceSnapshot {
  const paths = listFilesystemSourcePaths(root);
  return {
    method: "filesystem",
    pathSetSha256: sha256(paths.join("\0")),
    paths: paths.length,
    ...sourceSelectionPolicy(),
    tree: digestSourceFiles(root, paths)
  };
}

export function releaseSourceSelectionPolicy(): SourceSelectionPolicy {
  return sourceSelectionPolicy();
}

export function filterAllowlistedSourcePaths(paths: readonly string[]): string[] {
  const selected = new Set<string>();
  for (const path of paths) {
    assertRelativeSourcePath(path);
    if (isAllowlistedSourcePath(path)) {
      selected.add(path);
    }
  }
  return [...selected].sort(comparePaths);
}

export function isAllowlistedSourcePath(path: string): boolean {
  assertRelativeSourcePath(path);
  if (isExplicitlyExcludedSourcePath(path)) {
    return false;
  }
  if (
    FILESYSTEM_SOURCE_ALLOWED_FILES.includes(
      path as (typeof FILESYSTEM_SOURCE_ALLOWED_FILES)[number]
    )
  ) {
    return true;
  }
  return FILESYSTEM_SOURCE_ALLOWED_DIRECTORIES.some(
    (directory) => path.startsWith(`${directory}/`) && !isExcludedSourcePath(path)
  );
}

export function createReleaseTreeManifest(
  root: string,
  excludedPaths: readonly string[] = []
): ReleaseTreeManifest {
  const excluded = new Set(excludedPaths);
  for (const path of excluded) {
    assertRelativeSourcePath(path);
  }
  const treeEntries = listTreeEntries(root).filter(
    (entry) => !excluded.has(entry.path) && !isIgnoredReleaseMetadata(entry.path)
  );
  const entries = treeEntries.map<ReleaseTreeManifestEntry>((entry) => {
    assertReleaseEntryAllowed(entry.path);
    const absolutePath = resolve(root, ...entry.path.split("/"));
    if (entry.kind === "symlink") {
      return { path: entry.path, kind: "symlink", target: readlinkSync(absolutePath) };
    }
    return {
      path: entry.path,
      kind: "file",
      mode: entry.executable ? "100755" : "100644",
      ...digestFile(absolutePath)
    };
  });
  return {
    schemaVersion: RELEASE_INTEGRITY_SCHEMA_VERSION,
    algorithm: "sha256",
    entryCount: entries.length,
    tree: digestTreeEntries(root, treeEntries),
    entries
  };
}

export function verifyReleaseTreeManifest(
  root: string,
  value: unknown,
  excludedPaths: readonly string[] = []
): ReleaseTreeManifest {
  const expected = parseReleaseTreeManifest(value);
  const actual = createReleaseTreeManifest(root, excludedPaths);
  const expectedByPath = new Map(expected.entries.map((entry) => [entry.path, entry]));
  const actualByPath = new Map(actual.entries.map((entry) => [entry.path, entry]));

  for (const path of expectedByPath.keys()) {
    if (!actualByPath.has(path)) {
      throw new Error(`Release payload is missing manifest entry ${path}.`);
    }
  }
  for (const path of actualByPath.keys()) {
    if (!expectedByPath.has(path)) {
      throw new Error(`Release payload contains an unmanifested entry ${path}.`);
    }
  }
  for (const [path, expectedEntry] of expectedByPath) {
    if (!isDeepStrictEqual(actualByPath.get(path), expectedEntry)) {
      throw new Error(`Release payload entry failed integrity verification: ${path}.`);
    }
  }
  if (!isDeepStrictEqual(actual.tree, expected.tree)) {
    throw new Error("Release payload tree digest does not match the full manifest.");
  }
  return expected;
}

export function verifyReleaseBundle(
  bundleDirectory: string,
  manifestName: string
): ReleaseBundleVerification {
  assertRelativeSourcePath(manifestName);
  const root = resolve(bundleDirectory);
  const manifestPath = resolve(root, ...manifestName.split("/"));
  assertPathInside(root, manifestPath, `release manifest ${manifestName}`);
  if (!existsSync(manifestPath) || !lstatSync(manifestPath).isFile()) {
    throw new Error(`Release manifest is missing or is not a regular file: ${manifestPath}.`);
  }

  const manifest = requireRecord(
    JSON.parse(readFileSync(manifestPath, "utf8")) as unknown,
    "release manifest"
  );
  if (manifest["schemaVersion"] !== RELEASE_MANIFEST_SCHEMA_VERSION) {
    throw new Error(
      `Release manifest schema must be ${RELEASE_MANIFEST_SCHEMA_VERSION}; received ${String(manifest["schemaVersion"])}.`
    );
  }
  const integrity = requireRecord(manifest["integrity"], "release manifest integrity");
  if (integrity["schemaVersion"] !== RELEASE_INTEGRITY_SCHEMA_VERSION) {
    throw new Error("Release manifest integrity schema is unsupported.");
  }
  const manifestSeal = requireRecord(integrity["manifest"], "release manifest seal");
  if (
    manifestSeal["path"] !== manifestName ||
    manifestSeal["sealedBy"] !== "content-addressed archive SHA-256"
  ) {
    throw new Error("Release manifest does not declare its external archive seal.");
  }

  const payload = verifyReleaseTreeManifest(root, integrity["payload"], [manifestName]);
  verifyManifestDigestReferences(root, manifest, payload);
  verifyLocalApiTree(root, manifest);
  verifyManifestProvenance(manifest);

  return {
    manifest: digestFile(manifestPath),
    payload: payload.tree,
    entries: payload.entryCount
  };
}

export function writeDeterministicZip(sourceDirectory: string, outputPath: string): ArtifactDigest {
  const archive: Zippable = {};
  const entries = listTreeEntries(sourceDirectory).filter(
    (entry) => !isIgnoredReleaseMetadata(entry.path)
  );

  for (const entry of entries) {
    assertReleaseEntryAllowed(entry.path);
    const absolutePath = resolve(sourceDirectory, ...entry.path.split("/"));
    if (entry.kind === "symlink") {
      const target = readlinkSync(absolutePath);
      assertSafeSymlinkTarget(sourceDirectory, entry.path, target);
      archive[entry.path] = [
        strToU8(target),
        {
          attrs: 0o120777 * 0x10000,
          level: 0,
          mtime: fixedZipMtime,
          os: 3
        }
      ];
      continue;
    }
    const mode = entry.executable ? 0o100755 : 0o100644;
    archive[entry.path] = [
      new Uint8Array(readFileSync(absolutePath)),
      {
        attrs: mode * 0x10000,
        level: 9,
        mtime: fixedZipMtime,
        os: 3
      }
    ];
  }

  mkdirSync(dirname(outputPath), { recursive: true });
  const bytes = zipSync(archive, {
    level: 9,
    mtime: fixedZipMtime,
    os: 3
  });
  writeFileSync(outputPath, bytes);
  return { bytes: bytes.byteLength, sha256: sha256(bytes) };
}

export function extractZipSafely(zipPath: string, destinationDirectory: string): number {
  const zipBytes = new Uint8Array(readFileSync(zipPath));
  const metadata = readZipEntryMetadata(zipBytes);
  const extracted = unzipSync(zipBytes);
  const extractedNames = Object.keys(extracted).sort(comparePaths);
  const metadataNames = metadata.map((entry) => entry.name).sort(comparePaths);
  if (!isDeepStrictEqual(extractedNames, metadataNames)) {
    throw new Error("ZIP entry metadata does not match the extracted entry set.");
  }

  const symlinkNames = new Set(
    metadata.filter((entry) => entry.kind === "symlink").map((entry) => entry.name)
  );
  for (const entry of metadata) {
    assertSafeArchivePath(entry.name);
    assertNoSymlinkAncestor(entry.name, symlinkNames);
    if (entry.kind === "symlink") {
      const target = strFromU8(extracted[entry.name]!);
      assertSafeExtractedSymlinkTarget(entry.name, target, metadataNames);
    }
  }

  mkdirSync(destinationDirectory, { recursive: true });
  for (const entry of metadata.filter((candidate) => candidate.kind === "file")) {
    const outputPath = resolve(destinationDirectory, ...entry.name.split("/"));
    assertPathInside(destinationDirectory, outputPath, `ZIP entry ${entry.name}`);
    mkdirSync(dirname(outputPath), { recursive: true });
    writeFileSync(outputPath, extracted[entry.name]!, { mode: entry.mode & 0o777 });
    chmodSync(outputPath, entry.mode & 0o777);
  }
  for (const entry of metadata.filter((candidate) => candidate.kind === "symlink")) {
    const outputPath = resolve(destinationDirectory, ...entry.name.split("/"));
    assertPathInside(destinationDirectory, outputPath, `ZIP entry ${entry.name}`);
    mkdirSync(dirname(outputPath), { recursive: true });
    symlinkSync(strFromU8(extracted[entry.name]!), outputPath);
  }

  return metadata.length;
}

export function verifyReleaseArchive(
  archivePath: string,
  checksumPath: string,
  manifestName: string
): ReleaseArchiveVerification {
  const checksum = readFileSync(checksumPath, "utf8");
  const match = /^([a-f0-9]{64}) {2}([^\r\n]+)\r?\n?$/.exec(checksum);
  if (!match) {
    throw new Error(`Release checksum has an invalid SHA-256 sidecar format: ${checksumPath}.`);
  }
  const expectedSha256 = match[1]!;
  const expectedFilename = match[2]!;
  if (expectedFilename !== basename(archivePath)) {
    throw new Error(
      `Release checksum names ${expectedFilename}, expected ${basename(archivePath)}.`
    );
  }
  const archive = digestFile(archivePath);
  if (archive.sha256 !== expectedSha256) {
    throw new Error(`Release archive checksum mismatch for ${archivePath}.`);
  }
  if (!basename(archivePath).includes(archive.sha256)) {
    throw new Error("Release archive filename is not content-addressed by its complete SHA-256.");
  }

  const extractionDirectory = mkdtempSync(join(tmpdir(), "signsaarthi-release-verify-"));
  try {
    extractZipSafely(archivePath, extractionDirectory);
    return {
      archive,
      checksumPath,
      ...verifyReleaseBundle(extractionDirectory, manifestName)
    };
  } finally {
    rmSync(extractionDirectory, { force: true, recursive: true });
  }
}

export function resolveManifestPath(manifestPath: string, relativePath: string): string {
  if (!relativePath || isAbsolute(relativePath) || relativePath.includes("\\")) {
    throw new Error(`Release manifest path must be relative: ${relativePath || "<empty>"}.`);
  }
  const normalized = posix.normalize(relativePath);
  if (normalized !== relativePath || normalized === ".." || normalized.startsWith("../")) {
    throw new Error(`Release manifest path escapes or is not normalized: ${relativePath}.`);
  }

  const manifestDirectory = dirname(manifestPath);
  const resolvedPath = resolve(manifestDirectory, ...normalized.split("/"));
  assertPathInside(manifestDirectory, resolvedPath, `manifest path ${relativePath}`);
  return resolvedPath;
}

export function sha256(contents: string | Uint8Array): string {
  return createHash("sha256").update(contents).digest("hex");
}

type TreeEntry = {
  executable: boolean;
  kind: "file" | "symlink";
  path: string;
};

type ZipEntryMetadata = {
  name: string;
  kind: "file" | "symlink";
  mode: number;
};

function sourceSelectionPolicy(): SourceSelectionPolicy {
  return {
    allowlist: {
      directoryRoots: FILESYSTEM_SOURCE_ALLOWED_DIRECTORIES,
      filePaths: FILESYSTEM_SOURCE_ALLOWED_FILES
    },
    exclusions: {
      pathPrefixes: FILESYSTEM_SOURCE_EXCLUDED_PATH_PREFIXES,
      directoryNames: FILESYSTEM_SOURCE_EXCLUDED_DIRECTORIES,
      filePatterns: FILESYSTEM_SOURCE_EXCLUDED_FILE_PATTERNS
    }
  };
}

function listFilesystemSourcePaths(root: string): string[] {
  const paths: string[] = [];

  for (const path of FILESYSTEM_SOURCE_ALLOWED_FILES) {
    const absolutePath = resolve(root, ...path.split("/"));
    if (!lstatExists(absolutePath)) {
      continue;
    }
    const stat = lstatSync(absolutePath);
    if (!stat.isFile() && !stat.isSymbolicLink()) {
      throw new Error(`Allowlisted source file is not a file: ${absolutePath}.`);
    }
    if (!isExcludedSourcePath(path)) {
      paths.push(path);
    }
  }

  for (const directoryRoot of FILESYSTEM_SOURCE_ALLOWED_DIRECTORIES) {
    const absoluteRoot = resolve(root, ...directoryRoot.split("/"));
    if (!lstatExists(absoluteRoot)) {
      continue;
    }
    if (!lstatSync(absoluteRoot).isDirectory()) {
      throw new Error(`Allowlisted source root is not a directory: ${absoluteRoot}.`);
    }
    visitAllowedSourceDirectory(absoluteRoot, directoryRoot, paths);
  }

  return [...new Set(paths)].sort(comparePaths);
}

function visitAllowedSourceDirectory(directory: string, prefix: string, paths: string[]): void {
  const children = readdirSync(directory, { withFileTypes: true }).sort((left, right) =>
    comparePaths(left.name, right.name)
  );
  for (const child of children) {
    const path = `${prefix}/${child.name}`;
    const absolutePath = resolve(directory, child.name);
    const stat = lstatSync(absolutePath);
    if (stat.isDirectory()) {
      if (!isExcludedSourceDirectory(child.name) && !isExplicitlyExcludedSourcePath(path)) {
        visitAllowedSourceDirectory(absolutePath, path, paths);
      }
    } else if (stat.isFile() || stat.isSymbolicLink()) {
      if (!isExcludedSourcePath(path)) {
        paths.push(path);
      }
    } else {
      throw new Error(`Source tree contains an unsupported entry: ${absolutePath}`);
    }
  }
}

function isExplicitlyExcludedSourcePath(path: string): boolean {
  return FILESYSTEM_SOURCE_EXCLUDED_PATH_PREFIXES.some(
    (prefix) => path === prefix || path.startsWith(`${prefix}/`)
  );
}

function isExcludedSourcePath(path: string): boolean {
  if (isExplicitlyExcludedSourcePath(path)) {
    return true;
  }
  const components = path.split("/");
  return (
    components.slice(0, -1).some(isExcludedSourceDirectory) ||
    isExcludedSourceFile(components.at(-1) ?? "")
  );
}

function isExcludedSourceDirectory(name: string): boolean {
  const normalized = name.toLowerCase();
  return (
    FILESYSTEM_SOURCE_EXCLUDED_DIRECTORIES.includes(
      normalized as (typeof FILESYSTEM_SOURCE_EXCLUDED_DIRECTORIES)[number]
    ) || normalized.startsWith(".venv-")
  );
}

function isExcludedSourceFile(name: string): boolean {
  const normalized = name.toLowerCase();
  return (
    normalized.startsWith("._") ||
    normalized === ".ds_store" ||
    normalized === ".env" ||
    (normalized.startsWith(".env.") && normalized !== ".env.example") ||
    normalized.endsWith(".local") ||
    normalized.includes(".private.") ||
    normalized.endsWith(".pyc") ||
    normalized.endsWith(".tsbuildinfo") ||
    normalized.endsWith(".key") ||
    normalized.endsWith(".p12") ||
    normalized.endsWith(".pem") ||
    normalized.endsWith(".pfx")
  );
}

function listTreeEntries(root: string): TreeEntry[] {
  const entries: TreeEntry[] = [];

  function visit(directory: string, prefix: string): void {
    const children = readdirSync(directory, { withFileTypes: true }).sort((left, right) =>
      comparePaths(left.name, right.name)
    );

    for (const child of children) {
      const path = prefix ? `${prefix}/${child.name}` : child.name;
      const absolutePath = resolve(directory, child.name);
      const stat = lstatSync(absolutePath);
      if (stat.isDirectory()) {
        visit(absolutePath, path);
      } else if (stat.isFile()) {
        entries.push({ executable: (stat.mode & 0o111) !== 0, kind: "file", path });
      } else if (stat.isSymbolicLink()) {
        entries.push({ executable: false, kind: "symlink", path });
      } else {
        throw new Error(`Release tree contains an unsupported entry: ${absolutePath}`);
      }
    }
  }

  visit(root, "");
  return entries.sort((left, right) => comparePaths(left.path, right.path));
}

function digestTreeEntries(root: string, entries: readonly TreeEntry[]): TreeDigest {
  const hash = createHash("sha256");
  let bytes = 0;
  let files = 0;
  let symlinks = 0;
  hash.update("signsaarthi-tree-v1\0");

  for (const entry of entries) {
    const absolutePath = resolve(root, ...entry.path.split("/"));
    if (entry.kind === "symlink") {
      const target = readlinkSync(absolutePath);
      hash.update(`symlink\0${entry.path}\0${target}\0`);
      symlinks += 1;
      continue;
    }
    const contents = readFileSync(absolutePath);
    const mode = entry.executable ? "100755" : "100644";
    hash.update(`file\0${entry.path}\0${mode}\0${contents.byteLength}\0`);
    hash.update(contents);
    hash.update("\0");
    bytes += contents.byteLength;
    files += 1;
  }

  return { bytes, files, sha256: hash.digest("hex"), symlinks };
}

function parseReleaseTreeManifest(value: unknown): ReleaseTreeManifest {
  const record = requireRecord(value, "release payload manifest");
  if (
    record["schemaVersion"] !== RELEASE_INTEGRITY_SCHEMA_VERSION ||
    record["algorithm"] !== "sha256"
  ) {
    throw new Error("Release payload manifest schema or digest algorithm is unsupported.");
  }
  const entriesValue = record["entries"];
  if (!Array.isArray(entriesValue)) {
    throw new Error("Release payload manifest entries must be an array.");
  }
  const entries = entriesValue.map((entry, index) => parseReleaseTreeEntry(entry, index));
  const sortedPaths = entries.map((entry) => entry.path).sort(comparePaths);
  if (
    new Set(sortedPaths).size !== sortedPaths.length ||
    !isDeepStrictEqual(
      entries.map((entry) => entry.path),
      sortedPaths
    )
  ) {
    throw new Error("Release payload manifest paths must be unique and sorted.");
  }
  const entryCount = requireNonnegativeInteger(record["entryCount"], "payload entryCount");
  if (entryCount !== entries.length) {
    throw new Error("Release payload manifest entryCount does not match its entries.");
  }
  return {
    schemaVersion: RELEASE_INTEGRITY_SCHEMA_VERSION,
    algorithm: "sha256",
    entryCount,
    tree: parseTreeDigest(record["tree"], "payload tree"),
    entries
  };
}

function parseReleaseTreeEntry(value: unknown, index: number): ReleaseTreeManifestEntry {
  const record = requireRecord(value, `release payload entry ${index}`);
  const path = requireString(record["path"], `release payload entry ${index} path`);
  assertRelativeSourcePath(path);
  assertReleaseEntryAllowed(path);
  if (record["kind"] === "symlink") {
    return {
      path,
      kind: "symlink",
      target: requireString(record["target"], `release payload entry ${path} target`)
    };
  }
  if (record["kind"] !== "file") {
    throw new Error(`Release payload entry ${path} has an unsupported kind.`);
  }
  const mode = record["mode"];
  if (mode !== "100644" && mode !== "100755") {
    throw new Error(`Release payload entry ${path} has an unsupported mode.`);
  }
  return {
    path,
    kind: "file",
    mode,
    bytes: requireNonnegativeInteger(record["bytes"], `release payload entry ${path} bytes`),
    sha256: requireSha256(record["sha256"], `release payload entry ${path} sha256`)
  };
}

function verifyManifestDigestReferences(
  bundleRoot: string,
  manifest: Record<string, unknown>,
  payload: ReleaseTreeManifest
): void {
  const entries = new Map(payload.entries.map((entry) => [entry.path, entry]));

  function visit(value: unknown, label: string): void {
    if (Array.isArray(value)) {
      value.forEach((item, index) => visit(item, `${label}[${index}]`));
      return;
    }
    if (value === null || typeof value !== "object") {
      return;
    }
    const record = value as Record<string, unknown>;
    const hasBytes = Object.hasOwn(record, "bytes");
    const hasSha256 = Object.hasOwn(record, "sha256");
    if (typeof record["path"] === "string" && (hasBytes || hasSha256)) {
      if (!hasBytes || !hasSha256) {
        throw new Error(`${label} must provide both bytes and sha256 for its path.`);
      }
      const path = record["path"];
      resolveManifestPath(resolve(bundleRoot, "manifest.json"), path);
      const entry = entries.get(path);
      if (!entry || entry.kind !== "file") {
        throw new Error(`${label} references an unmanifested release file: ${path}.`);
      }
      if (
        requireNonnegativeInteger(record["bytes"], `${label} bytes`) !== entry.bytes ||
        requireSha256(record["sha256"], `${label} sha256`) !== entry.sha256
      ) {
        throw new Error(`${label} digest does not match the full payload manifest: ${path}.`);
      }
    }
    for (const [key, child] of Object.entries(record)) {
      visit(child, `${label}.${key}`);
    }
  }

  visit(manifest, "release manifest");
}

function verifyLocalApiTree(bundleRoot: string, manifest: Record<string, unknown>): void {
  const artifacts = requireRecord(manifest["artifacts"], "release artifacts");
  const localApi = requireRecord(artifacts["localApi"], "local API artifact");
  const directory = requireString(localApi["directory"], "local API directory");
  const apiDirectory = resolveManifestPath(resolve(bundleRoot, "manifest.json"), directory);
  if (!existsSync(apiDirectory) || !lstatSync(apiDirectory).isDirectory()) {
    throw new Error(`Local API artifact directory is missing: ${directory}.`);
  }
  const actual = digestTree(apiDirectory);
  const expected: TreeDigest = {
    bytes: requireNonnegativeInteger(localApi["bytes"], "local API bytes"),
    files: requireNonnegativeInteger(localApi["files"], "local API files"),
    sha256: requireSha256(localApi["sha256"], "local API sha256"),
    symlinks: requireNonnegativeInteger(localApi["symlinks"], "local API symlinks")
  };
  if (!isDeepStrictEqual(actual, expected)) {
    throw new Error("Local API aggregate tree digest does not match the release manifest.");
  }
}

function verifyManifestProvenance(manifest: Record<string, unknown>): void {
  const provenance = requireRecord(manifest["provenance"], "release provenance");
  const attestation = requireRecord(provenance["attestation"], "release attestation");
  const source = requireRecord(provenance["source"], "release source provenance");
  const releaseMode = attestation["releaseMode"];
  if (releaseMode === "production") {
    if (
      attestation["status"] !== "attested" ||
      attestation["basis"] !== "clean-git-head" ||
      source["method"] !== "git" ||
      source["state"] !== "clean" ||
      source["dirty"] !== false ||
      source["changes"] !== 0 ||
      source["untrackedFiles"] !== 0 ||
      typeof source["head"] !== "string" ||
      !gitHeadPattern.test(source["head"])
    ) {
      throw new Error("Production release manifest does not contain clean Git attestation.");
    }
    if (attestation["gitHead"] !== source["head"]) {
      throw new Error("Production release attestation Git HEAD does not match source provenance.");
    }
    return;
  }
  if (
    releaseMode !== "development" ||
    attestation["status"] !== "unattested" ||
    (attestation["basis"] !== "development-filesystem" &&
      attestation["basis"] !== "development-git-worktree")
  ) {
    throw new Error("Development release manifest must be explicitly marked unattested.");
  }
}

function parseTreeDigest(value: unknown, label: string): TreeDigest {
  const record = requireRecord(value, label);
  return {
    bytes: requireNonnegativeInteger(record["bytes"], `${label} bytes`),
    files: requireNonnegativeInteger(record["files"], `${label} files`),
    sha256: requireSha256(record["sha256"], `${label} sha256`),
    symlinks: requireNonnegativeInteger(record["symlinks"], `${label} symlinks`)
  };
}

function assertReleaseEntryAllowed(path: string): void {
  assertRelativeSourcePath(path);
  const normalized = path.toLowerCase();
  if (
    normalized === "private" ||
    normalized.startsWith("private/") ||
    normalized === "data/raw" ||
    normalized.startsWith("data/raw/")
  ) {
    throw new Error(`Release payload contains a forbidden private/raw path: ${path}.`);
  }
  const components = normalized.split("/");
  const filename = components.at(-1) ?? "";
  if (filename.startsWith("._")) {
    throw new Error(`Release payload contains forbidden AppleDouble metadata: ${path}.`);
  }
  if (
    components.includes(".ds_store") ||
    components.some((component) => component === ".venv" || component.startsWith(".venv-")) ||
    filename === ".env" ||
    (filename.startsWith(".env.") && filename !== ".env.example") ||
    filename.endsWith(".local") ||
    filename.includes(".private.")
  ) {
    throw new Error(`Release payload contains a forbidden local/private file: ${path}.`);
  }
}

function isIgnoredReleaseMetadata(path: string): boolean {
  return path.split("/").some((component) => component.toLowerCase() === ".ds_store");
}

function assertSafeSymlinkTarget(root: string, entryPath: string, target: string): void {
  if (!target || isAbsolute(target) || target.includes("\\")) {
    throw new Error(`Release symlink ${entryPath} has an unsafe target: ${target || "<empty>"}.`);
  }
  const linkPath = resolve(root, ...entryPath.split("/"));
  const resolvedTarget = resolve(dirname(linkPath), target);
  assertPathInside(root, resolvedTarget, `release symlink ${entryPath}`);
  if (!lstatExists(resolvedTarget)) {
    throw new Error(`Release symlink ${entryPath} targets a missing path: ${target}.`);
  }
}

function readZipEntryMetadata(bytes: Uint8Array): ZipEntryMetadata[] {
  const buffer = Buffer.from(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  let endOffset = -1;
  const minimumOffset = Math.max(0, buffer.byteLength - 65_557);
  for (let offset = buffer.byteLength - 22; offset >= minimumOffset; offset -= 1) {
    if (buffer.readUInt32LE(offset) === 0x06054b50) {
      endOffset = offset;
      break;
    }
  }
  if (endOffset < 0) {
    throw new Error("ZIP end-of-central-directory record is missing.");
  }
  const entryCount = buffer.readUInt16LE(endOffset + 10);
  const centralSize = buffer.readUInt32LE(endOffset + 12);
  const centralOffset = buffer.readUInt32LE(endOffset + 16);
  if (entryCount === 0xffff || centralSize === 0xffffffff || centralOffset === 0xffffffff) {
    throw new Error("ZIP64 release archives are not supported.");
  }
  if (centralOffset + centralSize > endOffset) {
    throw new Error("ZIP central directory extends outside the archive.");
  }

  const entries: ZipEntryMetadata[] = [];
  const seen = new Set<string>();
  let offset = centralOffset;
  for (let index = 0; index < entryCount; index += 1) {
    if (offset + 46 > buffer.byteLength || buffer.readUInt32LE(offset) !== 0x02014b50) {
      throw new Error("ZIP central directory entry is malformed.");
    }
    const madeBy = buffer.readUInt16LE(offset + 4);
    const flags = buffer.readUInt16LE(offset + 8);
    const filenameLength = buffer.readUInt16LE(offset + 28);
    const extraLength = buffer.readUInt16LE(offset + 30);
    const commentLength = buffer.readUInt16LE(offset + 32);
    const attributes = buffer.readUInt32LE(offset + 38);
    if ((flags & 0x0001) !== 0) {
      throw new Error("Encrypted ZIP entries are not supported.");
    }
    const nameStart = offset + 46;
    const nameEnd = nameStart + filenameLength;
    if (nameEnd > buffer.byteLength) {
      throw new Error("ZIP entry filename extends outside the archive.");
    }
    const name = strFromU8(buffer.subarray(nameStart, nameEnd));
    assertSafeArchivePath(name);
    if (seen.has(name)) {
      throw new Error(`ZIP contains a duplicate entry: ${name}.`);
    }
    seen.add(name);
    const origin = madeBy >>> 8;
    const unixMode = origin === 3 ? attributes >>> 16 : 0o100644;
    entries.push({
      name,
      kind: (unixMode & 0o170000) === 0o120000 ? "symlink" : "file",
      mode: unixMode || 0o100644
    });
    offset = nameEnd + extraLength + commentLength;
  }
  if (offset !== centralOffset + centralSize) {
    throw new Error("ZIP central directory size does not match its entries.");
  }
  return entries.sort((left, right) => comparePaths(left.name, right.name));
}

function assertNoSymlinkAncestor(path: string, symlinkPaths: ReadonlySet<string>): void {
  const components = path.split("/");
  for (let index = 1; index < components.length; index += 1) {
    if (symlinkPaths.has(components.slice(0, index).join("/"))) {
      throw new Error(`ZIP entry ${path} traverses an archived symlink.`);
    }
  }
}

function assertSafeExtractedSymlinkTarget(
  entryPath: string,
  target: string,
  archivePaths: readonly string[]
): void {
  if (!target || posix.isAbsolute(target) || target.includes("\\")) {
    throw new Error(`ZIP symlink ${entryPath} has an unsafe target: ${target || "<empty>"}.`);
  }
  const resolvedTarget = posix.normalize(posix.join(posix.dirname(entryPath), target));
  if (resolvedTarget === ".." || resolvedTarget.startsWith("../")) {
    throw new Error(`ZIP symlink ${entryPath} escapes the archive: ${target}.`);
  }
  if (
    !archivePaths.some(
      (path) => path === resolvedTarget || path.startsWith(`${resolvedTarget.replace(/\/$/, "")}/`)
    )
  ) {
    throw new Error(`ZIP symlink ${entryPath} targets an unarchived path: ${target}.`);
  }
}

function assertRelativeSourcePath(path: string): void {
  if (!path || isAbsolute(path) || path.includes("\\")) {
    throw new Error(`Source path must be relative and use forward slashes: ${path || "<empty>"}.`);
  }
  const normalized = posix.normalize(path);
  if (normalized !== path || normalized === ".." || normalized.startsWith("../")) {
    throw new Error(`Source path escapes the workspace: ${path}.`);
  }
}

function lstatExists(path: string): boolean {
  try {
    lstatSync(path);
    return true;
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code === "ENOENT") {
      return false;
    }
    throw error;
  }
}

function comparePaths(left: string, right: string): number {
  return left < right ? -1 : left > right ? 1 : 0;
}

function assertSafeArchivePath(path: string): void {
  if (!path || path.endsWith("/") || path.includes("\\") || isAbsolute(path)) {
    throw new Error(`ZIP contains an invalid file path: ${path || "<empty>"}.`);
  }
  const normalized = posix.normalize(path);
  if (normalized !== path || normalized === ".." || normalized.startsWith("../")) {
    throw new Error(`ZIP contains an unsafe file path: ${path}.`);
  }
}

function assertPathInside(root: string, candidate: string, label: string): void {
  const pathFromRoot = relative(resolve(root), resolve(candidate));
  if (pathFromRoot === ".." || pathFromRoot.startsWith(`..${sep}`) || isAbsolute(pathFromRoot)) {
    throw new Error(`${label} escapes ${root}.`);
  }
}

function requireRecord(value: unknown, label: string): Record<string, unknown> {
  if (value === null || typeof value !== "object" || Array.isArray(value)) {
    throw new Error(`${label} must be an object.`);
  }
  return value as Record<string, unknown>;
}

function requireString(value: unknown, label: string): string {
  if (typeof value !== "string" || value.length === 0) {
    throw new Error(`${label} must be a non-empty string.`);
  }
  return value;
}

function requireNonnegativeInteger(value: unknown, label: string): number {
  if (typeof value !== "number" || !Number.isInteger(value) || value < 0) {
    throw new Error(`${label} must be a nonnegative integer.`);
  }
  return value;
}

function requireSha256(value: unknown, label: string): string {
  if (typeof value !== "string" || !sha256Pattern.test(value)) {
    throw new Error(`${label} must be a lowercase SHA-256 digest.`);
  }
  return value;
}

function compareVersions(left: string, right: string): number {
  const leftParts = left.split(".").map(Number);
  const rightParts = right.split(".").map(Number);
  for (let index = 0; index < Math.max(leftParts.length, rightParts.length); index += 1) {
    const difference = (leftParts[index] ?? 0) - (rightParts[index] ?? 0);
    if (difference !== 0) {
      return difference;
    }
  }
  return 0;
}
