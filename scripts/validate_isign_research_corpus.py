#!/usr/bin/env python3
"""Validate the local gated iSign sentence-pose corpus before training."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import math
import sqlite3
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from build_isign_research_corpus import (
    MIN_DYNAMIC_FRAME_RATIO,
    MIN_HAND_MOTION_ENERGY,
    SENTENCE_FRAMES,
    TARGET_FRAMES,
    grouped_split,
    load_word_candidates,
    stable_hash,
    temporal_motion_metrics,
    tokenize,
    video_group_id,
)
from isign_remote_archive import ISIGN_DATASET_REVISION, ISIGN_POSE_PARTS


POINT_COUNT = 93
MAX_SOURCE_BYTES = 20 * 1024**3


class ValidationError(RuntimeError):
    """Raised when the private corpus does not satisfy the training contract."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_pose_blob(blob: bytes) -> dict[str, float]:
    with np.load(io.BytesIO(blob)) as payload:
        if set(payload.files) != {"xy", "confidence"}:
            raise ValidationError("A pose blob contains unexpected arrays.")
        xy = np.asarray(payload["xy"])
        confidence = np.asarray(payload["confidence"])
    if xy.shape != (SENTENCE_FRAMES, POINT_COUNT, 2):
        raise ValidationError(f"Unexpected pose shape {xy.shape}.")
    if confidence.shape != (SENTENCE_FRAMES, POINT_COUNT):
        raise ValidationError(f"Unexpected confidence shape {confidence.shape}.")
    if not np.isfinite(xy).all() or not np.isfinite(confidence).all():
        raise ValidationError("A pose blob contains non-finite values.")
    if float(confidence.min()) < 0 or float(confidence.max()) > 255:
        raise ValidationError("A pose blob contains out-of-range confidence values.")
    temporal = temporal_motion_metrics(xy.astype(np.float32), confidence.astype(np.float32) / 255)
    if (
        temporal["handMotionEnergy"] < MIN_HAND_MOTION_ENERGY
        or temporal["dynamicFrameRatio"] < MIN_DYNAMIC_FRAME_RATIO
    ):
        raise ValidationError("A pose blob is effectively static after normalization.")
    return temporal


def validate_word_library(path: Path, raw_root: Path, minimum_clips: int) -> dict[str, Any]:
    with gzip.open(path, "rt", encoding="utf-8") as source:
        payload = json.load(source)
    clips = payload.get("clips")
    if not isinstance(clips, list) or payload.get("clipCount") != len(clips):
        raise ValidationError("The private word-motion clip count is inconsistent.")
    if len(clips) < minimum_clips:
        raise ValidationError(
            f"Word-motion library has {len(clips)} clips; at least {minimum_clips} are required."
        )
    source_metadata = payload.get("source", {})
    directory_sha256 = str(source_metadata.get("archiveDirectorySha256", ""))
    if (
        source_metadata.get("dataset") != "Exploration-Lab/iSign"
        or source_metadata.get("revision") != ISIGN_DATASET_REVISION
        or source_metadata.get("gated") is not True
        or source_metadata.get("redistributionAllowed") is not False
        or source_metadata.get("rawArchiveStored") is not False
        or len(directory_sha256) != 64
        or any(character not in "0123456789abcdef" for character in directory_sha256)
    ):
        raise ValidationError("The private word-motion source boundary is invalid.")

    candidates = load_word_candidates(raw_root)
    candidate_labels = {row.normalized_label for row in candidates}
    candidate_hashes: dict[str, set[str]] = {}
    for candidate in candidates:
        candidate_hashes.setdefault(candidate.normalized_label, set()).add(stable_hash(candidate.uid))
    clip_labels: set[str] = set()
    frame_sources: dict[str, str] = {}
    source_alias_count = 0
    minimum_motion_energy = math.inf
    minimum_dynamic_ratio = math.inf
    for clip in clips:
        label = str(clip.get("normalizedLabel", ""))
        if not label or label in clip_labels or label not in candidate_labels:
            raise ValidationError("The word-motion library contains an invalid or duplicate label.")
        clip_labels.add(label)
        if clip.get("expertReviewed") is not False:
            raise ValidationError("Automated iSign motion must never be marked expert-reviewed.")
        if not 1600 <= float(clip.get("durationMs", 0)) <= 3200:
            raise ValidationError("A word-motion clip violates the bounded playback duration.")
        if not 5 <= float(clip.get("fps", 0)) <= 60:
            raise ValidationError("A word-motion clip has an invalid playback frame rate.")
        source_hash = str(clip.get("sourceUidHash", ""))
        if len(source_hash) != 24 or any(character not in "0123456789abcdef" for character in source_hash):
            raise ValidationError("A word-motion clip does not use a bounded source UID hash.")
        if source_hash not in candidate_hashes[label]:
            raise ValidationError("A word-motion source UID does not belong to its annotated label.")
        frames = clip.get("frames")
        if not isinstance(frames, list) or len(frames) != TARGET_FRAMES:
            raise ValidationError("A word-motion clip does not contain exactly 32 frames.")
        for frame in frames:
            for part, expected_points in (("pose", 33), ("leftHand", 21), ("rightHand", 21), ("face", 18)):
                points = frame.get(part)
                if not isinstance(points, list) or len(points) != expected_points:
                    raise ValidationError(f"A word-motion frame has an invalid {part} point count.")
                for point in points:
                    values = (point.get("x"), point.get("y"), point.get("visibility", 1))
                    if not all(isinstance(value, (int, float)) and math.isfinite(float(value)) for value in values):
                        raise ValidationError("A word-motion point contains a non-finite coordinate.")
        frame_digest = hashlib.sha256(
            json.dumps(frames, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        prior_source_hash = frame_sources.get(frame_digest)
        if prior_source_hash is not None and prior_source_hash != source_hash:
            raise ValidationError(
                "The word-motion library contains duplicate frames from different source UIDs."
            )
        if prior_source_hash == source_hash:
            source_alias_count += 1
        frame_sources[frame_digest] = source_hash
        xy = np.asarray(
            [
                [(point["x"], point["y"]) for part in ("pose", "leftHand", "rightHand", "face") for point in frame[part]]
                for frame in frames
            ],
            dtype=np.float32,
        )
        confidence = np.asarray(
            [
                [point.get("visibility", 1) for part in ("pose", "leftHand", "rightHand", "face") for point in frame[part]]
                for frame in frames
            ],
            dtype=np.float32,
        )
        temporal = temporal_motion_metrics(xy, confidence)
        if (
            temporal["handMotionEnergy"] < MIN_HAND_MOTION_ENERGY
            or temporal["dynamicFrameRatio"] < MIN_DYNAMIC_FRAME_RATIO
        ):
            raise ValidationError("A word-motion clip is effectively static.")
        minimum_motion_energy = min(minimum_motion_energy, temporal["handMotionEnergy"])
        minimum_dynamic_ratio = min(minimum_dynamic_ratio, temporal["dynamicFrameRatio"])

    missing_labels = sorted(candidate_labels - clip_labels)
    if missing_labels:
        raise ValidationError(
            f"Word-motion extraction did not cover {len(missing_labels)} annotated labels."
        )
    return {
        "status": "valid",
        "clipCount": len(clips),
        "annotatedLabelCount": len(candidate_labels),
        "framesPerClip": TARGET_FRAMES,
        "minimumDurationMs": min(float(clip["durationMs"]) for clip in clips),
        "maximumDurationMs": max(float(clip["durationMs"]) for clip in clips),
        "minimumHandMotionEnergy": minimum_motion_energy,
        "minimumDynamicFrameRatio": minimum_dynamic_ratio,
        "sameSourceAliasCount": source_alias_count,
        "fileSha256": sha256_file(path),
        "expertReviewed": False,
        "rawArchiveStored": False,
        "redistributionAllowed": False,
    }


def validate_corpus(database: Path, plan: Path, minimum_rows: int) -> dict[str, Any]:
    planned_rows = [json.loads(line) for line in plan.read_text(encoding="utf-8").splitlines() if line.strip()]
    planned_by_hash: dict[str, dict[str, Any]] = {}
    for row in planned_rows:
        uid = str(row.get("uid", ""))
        text = str(row.get("text", ""))
        normalized_text = " ".join(tokenize(text))
        video_id = str(row.get("videoId", ""))
        split = str(row.get("split", ""))
        if (
            not uid
            or video_id != video_group_id(uid)
            or split != grouped_split(video_id)
            or row.get("normalizedText") != normalized_text
            or row.get("wordCount") != len(tokenize(text))
        ):
            raise ValidationError("A sentence plan row violates the deterministic planning contract.")
        uid_hash = stable_hash(uid)
        if uid_hash in planned_by_hash:
            raise ValidationError("The sentence plan contains duplicate source UIDs.")
        planned_by_hash[uid_hash] = {
            "videoHash": stable_hash(video_id),
            "split": split,
            "text": text,
            "wordCount": int(row["wordCount"]),
        }
    if len(planned_by_hash) != len(planned_rows):
        raise ValidationError("The sentence plan contains duplicate source UIDs.")

    connection = sqlite3.connect(database)
    try:
        connection.execute("PRAGMA busy_timeout = 5000")
        checkpoint = connection.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
        if checkpoint is None or int(checkpoint[0]) != 0:
            raise ValidationError("SQLite WAL checkpoint could not acquire an exclusive checkpoint.")
        integrity = str(connection.execute("PRAGMA integrity_check").fetchone()[0])
        if integrity != "ok":
            raise ValidationError(f"SQLite integrity check failed: {integrity}")
        columns = {
            str(row[1]) for row in connection.execute("PRAGMA table_info(samples)").fetchall()
        }
        if "uid" in columns or "source_uid" in columns:
            raise ValidationError("Raw gated source UIDs must not be stored in the corpus.")
        rows = connection.execute(
            """
            SELECT uid_hash, video_hash, split, text, word_count, quality_score,
                   compressed_source_bytes, pose_npz
            FROM samples ORDER BY uid_hash
            """
        ).fetchall()
        if len(rows) < minimum_rows:
            raise ValidationError(
                f"Corpus has {len(rows)} accepted rows; at least {minimum_rows} are required."
            )
        extra_hashes = {str(row[0]) for row in rows} - set(planned_by_hash)
        if extra_hashes:
            raise ValidationError("The corpus contains rows outside the current deterministic plan.")

        split_groups: dict[str, set[str]] = {"train": set(), "val": set(), "test": set()}
        split_counts: Counter[str] = Counter()
        compressed_source_bytes = 0
        quality_minimum = math.inf
        motion_energy_minimum = math.inf
        dynamic_ratio_minimum = math.inf
        for uid_hash, video_hash, split, text, word_count, quality, source_bytes, blob in rows:
            if split not in split_groups:
                raise ValidationError(f"Unexpected split {split!r}.")
            if not str(uid_hash) or not str(video_hash):
                raise ValidationError("A corpus row is missing a hashed identifier.")
            expected = planned_by_hash[str(uid_hash)]
            if (
                str(video_hash) != expected["videoHash"]
                or str(split) != expected["split"]
                or str(text) != expected["text"]
                or int(word_count) != expected["wordCount"]
            ):
                raise ValidationError("A corpus row does not match its deterministic plan entry.")
            tokens = tokenize(str(text))
            if len(tokens) != int(word_count) or not 5 <= int(word_count) <= 15:
                raise ValidationError("A corpus row violates the 5-15 word text policy.")
            if not math.isfinite(float(quality)) or not 0 <= float(quality) <= 1:
                raise ValidationError("A corpus row has an invalid quality score.")
            if int(source_bytes) <= 0:
                raise ValidationError("A corpus row has an invalid source byte count.")
            temporal = validate_pose_blob(bytes(blob))
            split_groups[str(split)].add(str(video_hash))
            split_counts[str(split)] += 1
            compressed_source_bytes += int(source_bytes)
            quality_minimum = min(quality_minimum, float(quality))
            motion_energy_minimum = min(motion_energy_minimum, temporal["handMotionEnergy"])
            dynamic_ratio_minimum = min(dynamic_ratio_minimum, temporal["dynamicFrameRatio"])

        overlaps = {
            "trainVal": len(split_groups["train"] & split_groups["val"]),
            "trainTest": len(split_groups["train"] & split_groups["test"]),
            "valTest": len(split_groups["val"] & split_groups["test"]),
        }
        if any(overlaps.values()):
            raise ValidationError("Video-group leakage exists between train and held-out splits.")
        if any(split_counts[split] < minimum_rows * ratio * 0.95 for split, ratio in (("train", 0.8), ("val", 0.1), ("test", 0.1))):
            raise ValidationError("One or more corpus splits are materially below the planned ratio.")
        if compressed_source_bytes > MAX_SOURCE_BYTES:
            raise ValidationError("Accepted source members exceed the 20 GiB compact-corpus contract.")

        metadata_row = connection.execute(
            "SELECT value FROM metadata WHERE key = 'corpus'"
        ).fetchone()
        if metadata_row is None:
            raise ValidationError("Corpus extraction metadata is missing.")
        metadata = json.loads(str(metadata_row[0]))
        directory_sha256 = str(metadata.get("archiveDirectorySha256", ""))
        part_fingerprints = metadata.get("partFingerprints")
        if (
            metadata.get("rawArchiveStored") is not False
            or metadata.get("revision") != ISIGN_DATASET_REVISION
            or not isinstance(part_fingerprints, list)
            or [part.get("name") for part in part_fingerprints] != list(ISIGN_POSE_PARTS)
            or len(directory_sha256) != 64
            or any(character not in "0123456789abcdef" for character in directory_sha256)
        ):
            raise ValidationError("Corpus metadata does not assert that the raw archive was not stored.")
        if metadata.get("targetFrames") != SENTENCE_FRAMES or metadata.get("pointCount") != POINT_COUNT:
            raise ValidationError("Corpus metadata does not match the expected pose tensor contract.")
        if metadata.get("acceptedCompressedSourceBytes") != compressed_source_bytes:
            raise ValidationError("Corpus metadata byte accounting does not match accepted rows.")
    finally:
        connection.close()

    wal_path = Path(f"{database}-wal")
    if wal_path.exists() and wal_path.stat().st_size:
        raise ValidationError("SQLite WAL still contains data after the validation checkpoint.")
    local_archive_parts = sorted(
        str(path)
        for root in (Path("data/raw"), Path("data/private"))
        if root.exists()
        for name in ISIGN_POSE_PARTS
        for path in root.rglob(name)
    )
    if local_archive_parts:
        raise ValidationError("A full iSign archive part is stored locally inside the project data roots.")
    before_hash = database.stat()
    database_sha256 = sha256_file(database)
    after_hash = database.stat()
    if (before_hash.st_size, before_hash.st_mtime_ns) != (
        after_hash.st_size,
        after_hash.st_mtime_ns,
    ):
        raise ValidationError("Corpus database changed while its validation hash was computed.")

    return {
        "schemaVersion": 2,
        "status": "valid",
        "database": str(database),
        "databaseSha256": database_sha256,
        "planSha256": sha256_file(plan),
        "sourceRevision": ISIGN_DATASET_REVISION,
        "archiveDirectorySha256": directory_sha256,
        "plannedRows": len(planned_rows),
        "acceptedRows": len(rows),
        "missingOrRejectedRows": len(planned_rows) - len(rows),
        "splits": dict(sorted(split_counts.items())),
        "uniqueVideoGroups": {split: len(values) for split, values in split_groups.items()},
        "splitOverlaps": overlaps,
        "acceptedCompressedSourceBytes": compressed_source_bytes,
        "qualityMinimum": quality_minimum,
        "minimumHandMotionEnergy": motion_energy_minimum,
        "minimumDynamicFrameRatio": dynamic_ratio_minimum,
        "rawArchiveStored": False,
        "fullArchiveDownloaded": False,
        "localArchiveParts": local_archive_parts,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--database",
        type=Path,
        default=Path("data/private/isign-research/sentence-pose-corpus.private.sqlite"),
    )
    parser.add_argument(
        "--plan",
        type=Path,
        default=Path("data/private/isign-research/sentence-plan.private.jsonl"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/private/isign-research/validation-report.private.json"),
    )
    parser.add_argument("--minimum-rows", type=int, default=10_000)
    parser.add_argument(
        "--word-library",
        type=Path,
        default=Path("data/private/isign-research/word-motion-library.json.gz"),
    )
    parser.add_argument("--minimum-word-motions", type=int, default=585)
    parser.add_argument("--raw-root", type=Path, default=Path("data/raw/isign"))
    parser.add_argument("--accept-research-only-terms", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.accept_research_only_terms:
        raise SystemExit("Pass --accept-research-only-terms after accepting the gated iSign terms.")
    report = validate_corpus(args.database, args.plan, args.minimum_rows)
    report["wordMotionLibrary"] = validate_word_library(
        args.word_library, args.raw_root, args.minimum_word_motions
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
