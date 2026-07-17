#!/usr/bin/env python3
"""Build bounded, private iSign motion artifacts without storing source archives."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import math
import re
import sqlite3
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from pose_format import Pose

from isign_remote_archive import (
    ArchiveAccessError,
    ISIGN_DATASET_REVISION,
    ISignRemotePoseArchive,
)


SEED = 186
TARGET_FRAMES = 32
SENTENCE_FRAMES = 64
MAX_WORD_MOTION_SECONDS = 3.2
ISIGN_LICENSE = "CC BY-NC-SA 4.0; gated research-only terms also apply"
FACE_POINT_INDICES = (70, 63, 105, 66, 107, 336, 296, 334, 293, 300, 159, 145, 386, 374, 61, 291, 13, 14)
TOKEN_PATTERN = re.compile(r"[\w]+(?:['’-][\w]+)*", re.UNICODE)
DEFAULT_PRIVATE_ROOT = Path("data/private/isign-research")
PRIVATE_DATA_ROOT = Path("data/private")
MIN_HAND_MOTION_ENERGY = 0.01
MIN_DYNAMIC_FRAME_RATIO = 0.20


class CorpusBuildError(RuntimeError):
    """Raised when a private corpus invariant is violated."""


@dataclass(frozen=True)
class CaptionRecord:
    uid: str
    text: str
    normalized_text: str
    tokens: tuple[str, ...]
    video_id: str
    split: str


@dataclass(frozen=True)
class WordCandidate:
    normalized_label: str
    label: str
    uid: str


def normalize_text(value: str) -> str:
    return " ".join(tokenize(value)).lower()


def normalize_label(value: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", value.casefold()))


def tokenize(value: str) -> tuple[str, ...]:
    return tuple(match.group(0).casefold() for match in TOKEN_PATTERN.finditer(value))


def video_group_id(uid: str) -> str:
    value = uid.strip()
    sentence_match = re.fullmatch(r"(.+)-(\d+)", value)
    if sentence_match:
        return sentence_match.group(1)
    isolated_match = re.fullmatch(r"(.+)_([a-zA-Z]\d*)", value)
    if isolated_match:
        return isolated_match.group(1)
    raise CorpusBuildError("iSign UID does not match a documented sentence or isolated-sign form.")


def grouped_split(video_id: str) -> str:
    bucket = int(hashlib.sha256(f"{SEED}:{video_id}".encode()).hexdigest()[:8], 16) % 100
    return "train" if bucket < 80 else "val" if bucket < 90 else "test"


def stable_hash(value: str, length: int = 24) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:length]


def load_caption_records(csv_path: Path) -> list[CaptionRecord]:
    records: list[CaptionRecord] = []
    seen_text: set[str] = set()
    with csv_path.open(newline="", encoding="utf-8") as source:
        for row in csv.DictReader(source):
            uid = str(row.get("uid", "")).strip()
            text = " ".join(str(row.get("text", "")).split())
            tokens = tokenize(text)
            normalized = " ".join(tokens)
            if not uid or not normalized or normalized in seen_text:
                continue
            seen_text.add(normalized)
            video_id = video_group_id(uid)
            records.append(
                CaptionRecord(
                    uid=uid,
                    text=text,
                    normalized_text=normalized,
                    tokens=tokens,
                    video_id=video_id,
                    split=grouped_split(video_id),
                )
            )
    return records


def balanced_sentence_subset(records: Sequence[CaptionRecord], target_count: int) -> list[CaptionRecord]:
    eligible = [record for record in records if 5 <= len(record.tokens) <= 15]
    if target_count <= 0 or target_count > len(eligible):
        raise CorpusBuildError(
            f"Requested {target_count} sentences but {len(eligible)} satisfy the 5-15 word policy."
        )
    split_targets = {
        "train": round(target_count * 0.8),
        "val": round(target_count * 0.1),
    }
    split_targets["test"] = target_count - split_targets["train"] - split_targets["val"]
    selected: list[CaptionRecord] = []
    for split in ("train", "val", "test"):
        candidates = [record for record in eligible if record.split == split]
        token_counts = Counter(token for record in candidates for token in set(record.tokens))

        def priority(record: CaptionRecord) -> tuple[float, str]:
            rare = sum(token_counts[token] < 5 for token in set(record.tokens))
            mid = sum(5 <= token_counts[token] <= 50 for token in set(record.tokens))
            diversity = len(set(record.tokens)) / len(record.tokens)
            score = rare * 4 + mid * 2 + diversity
            return (-score, stable_hash(f"{SEED}:{record.uid}", 64))

        candidates.sort(key=priority)
        quota = split_targets[split]
        if len(candidates) < quota:
            raise CorpusBuildError(f"The {split} group contains only {len(candidates)} eligible rows.")
        selected.extend(candidates[:quota])
    selected.sort(key=lambda record: (record.split, stable_hash(record.uid, 64)))
    if len({record.video_id for record in selected if record.split == "train"} & {record.video_id for record in selected if record.split != "train"}):
        raise CorpusBuildError("Video groups leaked across train and held-out splits.")
    return selected


def load_word_candidates(raw_root: Path) -> list[WordCandidate]:
    rows: list[dict[str, str]] = []
    for filename in ("word-presence-dataset_v1.1.csv", "word-description-dataset_v1.1.csv"):
        with (raw_root / filename).open(newline="", encoding="utf-8") as source:
            rows.extend(csv.DictReader(source))
    candidates: dict[tuple[str, str], WordCandidate] = {}
    for row in rows:
        uid = str(row.get("word_id", "")).strip()
        label = " ".join(str(row.get("word", "")).split())
        normalized = normalize_label(label)
        if uid and normalized:
            candidates[(normalized, uid)] = WordCandidate(normalized, label, uid)
    return sorted(candidates.values(), key=lambda row: (row.normalized_label, row.uid))


def component_slices(pose: Pose) -> dict[str, slice]:
    slices: dict[str, slice] = {}
    offset = 0
    for component in pose.header.components:
        end = offset + len(component.points)
        slices[str(component.name)] = slice(offset, end)
        offset = end
    return slices


def interpolate_track(values: np.ndarray, confidence: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=np.float32).copy()
    confidence = np.asarray(confidence, dtype=np.float32).copy()
    timeline = np.arange(values.shape[0], dtype=np.float32)
    for point_index in range(values.shape[1]):
        valid = np.isfinite(values[:, point_index]).all(axis=1) & (confidence[:, point_index] > 0.02)
        if not valid.any():
            values[:, point_index] = 0
            confidence[:, point_index] = 0
            continue
        for coordinate in range(2):
            values[:, point_index, coordinate] = np.interp(
                timeline, timeline[valid], values[valid, point_index, coordinate]
            )
        confidence[:, point_index] = np.interp(
            timeline, timeline[valid], confidence[valid, point_index]
        )
    return values, np.clip(confidence, 0, 1)


def resample_track(values: np.ndarray, confidence: np.ndarray, frame_count: int) -> tuple[np.ndarray, np.ndarray]:
    if values.shape[0] < 2:
        raise CorpusBuildError("Pose sequence has fewer than two frames.")
    source_timeline = np.linspace(0, 1, values.shape[0], dtype=np.float32)
    target_timeline = np.linspace(0, 1, frame_count, dtype=np.float32)
    output = np.empty((frame_count, values.shape[1], 2), dtype=np.float32)
    output_confidence = np.empty((frame_count, values.shape[1]), dtype=np.float32)
    for point_index in range(values.shape[1]):
        for coordinate in range(2):
            output[:, point_index, coordinate] = np.interp(
                target_timeline, source_timeline, values[:, point_index, coordinate]
            )
        output_confidence[:, point_index] = np.interp(
            target_timeline, source_timeline, confidence[:, point_index]
        )
    return output, np.clip(output_confidence, 0, 1)


def trim_motion_window(
    values: np.ndarray,
    confidence: np.ndarray,
    fps: float,
    max_duration_seconds: float = MAX_WORD_MOTION_SECONDS,
) -> tuple[np.ndarray, np.ndarray]:
    """Keep the densest upper-body/hand motion window and discard idle context."""

    if values.shape[0] < 3:
        return values, confidence
    selected = np.concatenate([np.arange(11, 17), np.arange(33, 75)])
    selected_values = values[:, selected]
    selected_confidence = confidence[:, selected]
    shoulder_width = np.linalg.norm(values[:, 11] - values[:, 12], axis=1)
    valid_width = shoulder_width[np.isfinite(shoulder_width) & (shoulder_width > 1e-3)]
    scale = float(np.median(valid_width)) if valid_width.size else 1.0
    displacement = np.linalg.norm(np.diff(selected_values, axis=0), axis=-1) / max(scale, 1e-3)
    visibility = np.minimum(selected_confidence[1:], selected_confidence[:-1])
    weighted = displacement * (visibility >= 0.1)
    denominator = np.maximum(1, (visibility >= 0.1).sum(axis=1))
    energy = np.concatenate([[0.0], weighted.sum(axis=1) / denominator])
    kernel_size = min(7, max(1, values.shape[0] // 8))
    kernel = np.ones(kernel_size, dtype=np.float32) / kernel_size
    smoothed = np.convolve(energy, kernel, mode="same")

    max_frames = max(16, int(round(max_duration_seconds * max(1.0, fps))))
    if values.shape[0] > max_frames:
        scores = np.convolve(smoothed, np.ones(max_frames, dtype=np.float32), mode="valid")
        start = int(np.argmax(scores))
        end = start + max_frames
        return values[start:end], confidence[start:end]

    positive = smoothed[smoothed > 0]
    if not positive.size:
        return values, confidence
    threshold = max(1e-4, float(np.percentile(positive, 60)) * 0.25)
    active = np.flatnonzero(smoothed >= threshold)
    if not active.size:
        return values, confidence
    padding = max(2, int(round(fps * 0.2)))
    start = max(0, int(active[0]) - padding)
    end = min(values.shape[0], int(active[-1]) + padding + 1)
    minimum_frames = min(values.shape[0], max(16, int(round(fps * 0.8))))
    if end - start < minimum_frames:
        center = (start + end) // 2
        start = max(0, center - minimum_frames // 2)
        end = min(values.shape[0], start + minimum_frames)
        start = max(0, end - minimum_frames)
    return values[start:end], confidence[start:end]


def extract_components(raw_pose: bytes, target_frames: int) -> dict[str, Any]:
    pose = Pose.read(raw_pose)
    data = np.asarray(pose.body.data, dtype=np.float32)
    confidence = np.asarray(pose.body.confidence, dtype=np.float32)
    if data.ndim != 4 or confidence.ndim != 3 or data.shape[1] != 1:
        raise CorpusBuildError("Unexpected iSign pose tensor shape.")
    data = data[:, 0]
    confidence = confidence[:, 0]
    slices = component_slices(pose)
    required = ("POSE_LANDMARKS", "LEFT_HAND_LANDMARKS", "RIGHT_HAND_LANDMARKS", "FACE_LANDMARKS")
    if any(name not in slices for name in required):
        raise CorpusBuildError("iSign pose is missing a required MediaPipe component.")
    selected_indices = np.concatenate(
        [
            np.arange(slices["POSE_LANDMARKS"].start, slices["POSE_LANDMARKS"].stop),
            np.arange(slices["LEFT_HAND_LANDMARKS"].start, slices["LEFT_HAND_LANDMARKS"].stop),
            np.arange(slices["RIGHT_HAND_LANDMARKS"].start, slices["RIGHT_HAND_LANDMARKS"].stop),
            np.asarray([slices["FACE_LANDMARKS"].start + index for index in FACE_POINT_INDICES]),
        ]
    )
    xy, visibility = interpolate_track(data[:, selected_indices, :2], confidence[:, selected_indices])
    source_fps = float(pose.body.fps or 25)
    if target_frames == TARGET_FRAMES:
        xy, visibility = trim_motion_window(xy, visibility, source_fps)
    quality = motion_quality(xy, visibility)
    source_window_frames = int(xy.shape[0])
    xy, visibility = resample_track(xy, visibility, target_frames)
    measured_duration_ms = ((source_window_frames - 1) / max(1.0, source_fps)) * 1000
    duration_ms = (
        min(MAX_WORD_MOTION_SECONDS * 1000, max(1600.0, measured_duration_ms))
        if target_frames == TARGET_FRAMES
        else measured_duration_ms
    )
    effective_fps = ((target_frames - 1) * 1000 / duration_ms) if duration_ms > 0 else source_fps
    return {
        "xy": xy,
        "visibility": visibility,
        "sourceFrames": int(data.shape[0]),
        "sourceWindowFrames": source_window_frames,
        "sourceFps": source_fps,
        "fps": min(60.0, max(5.0, effective_fps)),
        "durationMs": duration_ms,
        "quality": quality,
    }


def motion_quality(xy: np.ndarray, visibility: np.ndarray) -> dict[str, float]:
    pose_visibility = float((visibility[:, :33] >= 0.35).mean())
    left_visibility = float((visibility[:, 33:54] >= 0.25).mean())
    right_visibility = float((visibility[:, 54:75] >= 0.25).mean())
    hand_visibility = max(left_visibility, right_visibility)
    visible_hands = xy[:, 33:75][visibility[:, 33:75] >= 0.25]
    motion_range = 0.0
    if visible_hands.size:
        width = max(1.0, float(np.ptp(xy[:, :33, 0])))
        motion_range = float(max(np.ptp(visible_hands[:, 0]), np.ptp(visible_hands[:, 1])) / width)
    temporal = temporal_motion_metrics(xy, visibility)
    score = (
        pose_visibility * 0.40
        + hand_visibility * 0.40
        + min(1.0, motion_range) * 0.08
        + min(1.0, temporal["handMotionEnergy"] * 4) * 0.08
        + temporal["dynamicFrameRatio"] * 0.04
    )
    return {
        "score": round(score, 6),
        "poseVisibility": round(pose_visibility, 6),
        "handVisibility": round(hand_visibility, 6),
        "handMotionRange": round(motion_range, 6),
        **temporal,
    }


def temporal_motion_metrics(xy: np.ndarray, visibility: np.ndarray) -> dict[str, float]:
    """Measure actual temporal hand movement in shoulder-width units."""

    shoulder_width = np.linalg.norm(xy[:, 11] - xy[:, 12], axis=1)
    valid_width = shoulder_width[np.isfinite(shoulder_width) & (shoulder_width > 1e-4)]
    scale = float(np.median(valid_width)) if valid_width.size else 1.0
    hand_velocity = np.linalg.norm(np.diff(xy[:, 33:75], axis=0), axis=-1) / max(
        scale, 1e-4
    )
    visible = np.minimum(visibility[1:, 33:75], visibility[:-1, 33:75]) >= 0.08
    if not visible.any():
        return {"handMotionEnergy": 0.0, "dynamicFrameRatio": 0.0}
    frame_energy = np.divide(
        (hand_velocity * visible).sum(axis=1),
        np.maximum(1, visible.sum(axis=1)),
    )
    return {
        "handMotionEnergy": round(float(hand_velocity[visible].mean()), 6),
        "dynamicFrameRatio": round(float((frame_energy >= 0.002).mean()), 6),
    }


def validate_motion(components: dict[str, Any]) -> None:
    if components["sourceFrames"] < 16:
        raise CorpusBuildError("Pose sequence is too short for stable playback.")
    quality = components["quality"]
    if quality["poseVisibility"] < 0.45 or quality["handVisibility"] < 0.12:
        raise CorpusBuildError("Pose sequence does not contain enough visible body and hand motion.")
    if (
        quality["handMotionEnergy"] < MIN_HAND_MOTION_ENERGY
        or quality["dynamicFrameRatio"] < MIN_DYNAMIC_FRAME_RATIO
    ):
        raise CorpusBuildError("Pose sequence is effectively static after normalization.")


def frame_payload(xy: np.ndarray, visibility: np.ndarray) -> list[dict[str, Any]]:
    slices = {"pose": (0, 33), "leftHand": (33, 54), "rightHand": (54, 75), "face": (75, 93)}
    frames: list[dict[str, Any]] = []
    for frame_index in range(xy.shape[0]):
        frame: dict[str, Any] = {}
        for name, (start, end) in slices.items():
            frame[name] = [
                {
                    "x": round(float(xy[frame_index, index, 0]), 5),
                    "y": round(float(xy[frame_index, index, 1]), 5),
                    "visibility": round(float(visibility[frame_index, index]), 4),
                }
                for index in range(start, end)
            ]
        frames.append(frame)
    return frames


def normalize_for_training(xy: np.ndarray, visibility: np.ndarray) -> np.ndarray:
    output = xy.astype(np.float32, copy=True)
    shoulders = (output[:, 11] + output[:, 12]) / 2
    scale = np.linalg.norm(output[:, 0] - shoulders, axis=1)
    valid_scale = scale[np.isfinite(scale) & (scale > 1e-3)]
    fallback = float(np.median(valid_scale)) if valid_scale.size else 1.0
    scale = np.where(np.isfinite(scale) & (scale > 1e-3), scale, fallback)
    output = (output - shoulders[:, None, :]) / scale[:, None, None]
    output[visibility < 0.02] = 0
    return np.clip(output, -8, 8)


def build_word_library(args: argparse.Namespace) -> None:
    candidates = load_word_candidates(args.raw_root)
    accepted_by_label: dict[str, dict[str, Any]] = {}
    failures: Counter[str] = Counter()
    transferred_before = 0
    with ISignRemotePoseArchive(revision=args.revision) as archive:
        transferred_before = archive.bytes_transferred
        selected: list[WordCandidate] = []
        planned_bytes = 0
        for candidate in candidates:
            info = archive.pose_info(candidate.uid)
            if planned_bytes + info.compress_size > args.max_bytes:
                break
            planned_bytes += info.compress_size
            selected.append(candidate)

        def process(candidate: WordCandidate) -> tuple[WordCandidate, dict[str, Any]]:
            components = extract_components(archive.read_pose(candidate.uid), TARGET_FRAMES)
            validate_motion(components)
            clip_id = f"isign-research-{stable_hash(candidate.uid + ':' + candidate.normalized_label, 20)}"
            return candidate, {
                "id": clip_id,
                "label": candidate.label,
                "normalizedLabel": candidate.normalized_label,
                "fps": round(float(components["fps"]), 6),
                "durationMs": round(float(components["durationMs"]), 3),
                "expertReviewed": False,
                "sourceUidHash": stable_hash(candidate.uid),
                "quality": components["quality"],
                "frames": frame_payload(components["xy"], components["visibility"]),
            }

        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(process, candidate): candidate for candidate in selected}
            for index, future in enumerate(as_completed(futures), 1):
                candidate = futures[future]
                try:
                    _, clip = future.result()
                    current = accepted_by_label.get(candidate.normalized_label)
                    if current is None or clip["quality"]["score"] > current["quality"]["score"]:
                        accepted_by_label[candidate.normalized_label] = clip
                except (ArchiveAccessError, CorpusBuildError, ValueError) as exc:
                    failures[type(exc).__name__] += 1
                if index % 25 == 0 or index == len(selected):
                    print(f"Processed word poses {index}/{len(selected)}; accepted {len(accepted_by_label)}.", flush=True)

        clips = sorted(accepted_by_label.values(), key=lambda clip: (clip["normalizedLabel"], clip["id"]))
        payload = {
            "schemaVersion": 1,
            "source": {
                "dataset": "Exploration-Lab/iSign",
                "revision": args.revision,
                "license": ISIGN_LICENSE,
                "gated": True,
                "redistributionAllowed": False,
                "rawArchiveStored": False,
                "partFingerprints": archive.part_fingerprints,
                "archiveDirectorySha256": archive.archive_directory_sha256,
            },
            "usage": "Local, noncommercial research only. Do not redistribute this derived motion library.",
            "expertReviewStatus": "pending",
            "clipCount": len(clips),
            "candidateCount": len(selected),
            "failureCount": sum(failures.values()),
            "failuresByType": dict(sorted(failures.items())),
            "bytesTransferred": archive.bytes_transferred - transferred_before,
            "clips": clips,
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(args.output, "wt", encoding="utf-8", compresslevel=9) as target:
        json.dump(payload, target, separators=(",", ":"), ensure_ascii=True)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "clipCount": payload["clipCount"],
                "candidateCount": payload["candidateCount"],
                "bytesTransferred": payload["bytesTransferred"],
                "rawArchiveStored": False,
            },
            indent=2,
        )
    )


def write_sentence_plan(args: argparse.Namespace) -> None:
    records = load_caption_records(args.raw_root / "iSign_v1.1.csv")
    selected = balanced_sentence_subset(records, args.samples)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as target:
        for record in selected:
            target.write(
                json.dumps(
                    {
                        "uid": record.uid,
                        "text": record.text,
                        "normalizedText": record.normalized_text,
                        "videoId": record.video_id,
                        "split": record.split,
                        "wordCount": len(record.tokens),
                    },
                    ensure_ascii=True,
                )
                + "\n"
            )
    summary = {
        "output": str(args.output),
        "sourceRows": len(records),
        "selectedRows": len(selected),
        "splits": dict(Counter(record.split for record in selected)),
        "uniqueVideoGroups": len({record.video_id for record in selected}),
        "wordCountRange": [min(len(row.tokens) for row in selected), max(len(row.tokens) for row in selected)],
        "selection": "5-15 words, vocabulary-balanced, deterministic, video-grouped split",
    }
    print(json.dumps(summary, indent=2))


def create_corpus_database(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS samples (
          uid_hash TEXT PRIMARY KEY,
          video_hash TEXT NOT NULL,
          split TEXT NOT NULL CHECK(split IN ('train','val','test')),
          text TEXT NOT NULL,
          word_count INTEGER NOT NULL,
          source_frames INTEGER NOT NULL,
          source_fps REAL NOT NULL,
          quality_score REAL NOT NULL,
          compressed_source_bytes INTEGER NOT NULL,
          pose_npz BLOB NOT NULL
        )
        """
    )
    connection.execute("CREATE INDEX IF NOT EXISTS samples_split_idx ON samples(split)")
    connection.execute(
        "CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
    )
    return connection


def compact_pose_blob(components: dict[str, Any]) -> bytes:
    normalized = normalize_for_training(components["xy"], components["visibility"])
    buffer = io.BytesIO()
    np.savez_compressed(
        buffer,
        xy=normalized.astype(np.float16),
        confidence=np.rint(components["visibility"] * 255).astype(np.uint8),
    )
    return buffer.getvalue()


def prune_static_corpus_rows(connection: sqlite3.Connection) -> int:
    invalid_hashes: list[str] = []
    for uid_hash, blob in connection.execute("SELECT uid_hash, pose_npz FROM samples"):
        try:
            with np.load(io.BytesIO(bytes(blob))) as payload:
                xy = np.asarray(payload["xy"], dtype=np.float32)
                confidence = np.asarray(payload["confidence"], dtype=np.float32) / 255
        except (KeyError, OSError, ValueError) as exc:
            raise CorpusBuildError("Existing compact corpus contains an unreadable pose blob.") from exc
        if xy.shape != (SENTENCE_FRAMES, 93, 2) or confidence.shape != (
            SENTENCE_FRAMES,
            93,
        ):
            raise CorpusBuildError("Existing compact corpus contains an invalid pose tensor shape.")
        temporal = temporal_motion_metrics(xy, confidence)
        if (
            temporal["handMotionEnergy"] < MIN_HAND_MOTION_ENERGY
            or temporal["dynamicFrameRatio"] < MIN_DYNAMIC_FRAME_RATIO
        ):
            invalid_hashes.append(str(uid_hash))
    if invalid_hashes:
        connection.executemany(
            "DELETE FROM samples WHERE uid_hash = ?",
            ((uid_hash,) for uid_hash in invalid_hashes),
        )
        connection.commit()
    return len(invalid_hashes)


def extract_sentence_corpus(args: argparse.Namespace) -> None:
    plan_rows = [json.loads(line) for line in args.plan.read_text(encoding="utf-8").splitlines() if line.strip()]
    connection = create_corpus_database(args.output)
    pruned_static_rows = prune_static_corpus_rows(connection)
    completed = {row[0] for row in connection.execute("SELECT uid_hash FROM samples")}
    failures: Counter[str] = Counter()
    with ISignRemotePoseArchive(revision=args.revision) as archive:
        prior_metadata_row = connection.execute(
            "SELECT value FROM metadata WHERE key = 'corpus'"
        ).fetchone()
        if prior_metadata_row is not None:
            prior_metadata = json.loads(str(prior_metadata_row[0]))
            prior_fingerprints = prior_metadata.get("partFingerprints")
            if prior_fingerprints != archive.part_fingerprints:
                raise CorpusBuildError(
                    "Existing corpus archive fingerprints differ from the pinned iSign revision."
                )
            prior_revision = prior_metadata.get("revision")
            if prior_revision not in {args.revision, "main"}:
                raise CorpusBuildError(
                    "Existing corpus was created from a different immutable iSign revision."
                )
        transferred_before = archive.bytes_transferred
        requests_before = archive.range_requests
        selected: list[tuple[dict[str, Any], int]] = []
        accepted_bytes = int(
            connection.execute(
                "SELECT COALESCE(SUM(compressed_source_bytes), 0) FROM samples"
            ).fetchone()[0]
        )
        planned_bytes = accepted_bytes
        for row in plan_rows:
            uid_hash = stable_hash(str(row["uid"]))
            if uid_hash in completed:
                continue
            info = archive.pose_info(str(row["uid"]))
            if planned_bytes + info.compress_size > args.max_bytes:
                break
            selected.append((row, info.compress_size))
            planned_bytes += info.compress_size

        def process(item: tuple[dict[str, Any], int]) -> tuple[dict[str, Any], dict[str, Any], bytes, int]:
            row, compressed_size = item
            components = extract_components(archive.read_pose(str(row["uid"])), SENTENCE_FRAMES)
            validate_motion(components)
            return row, components, compact_pose_blob(components), compressed_size

        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(process, item): item[0] for item in selected}
            for index, future in enumerate(as_completed(futures), 1):
                row = futures[future]
                try:
                    row, components, blob, compressed_size = future.result()
                    connection.execute(
                        """
                        INSERT OR REPLACE INTO samples
                        (uid_hash, video_hash, split, text, word_count, source_frames, source_fps,
                         quality_score, compressed_source_bytes, pose_npz)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            stable_hash(str(row["uid"])),
                            stable_hash(str(row["videoId"])),
                            str(row["split"]),
                            str(row["text"]),
                            int(row["wordCount"]),
                            int(components["sourceFrames"]),
                            float(components["sourceFps"]),
                            float(components["quality"]["score"]),
                            compressed_size,
                            blob,
                        ),
                    )
                except (ArchiveAccessError, CorpusBuildError, ValueError) as exc:
                    failures[type(exc).__name__] += 1
                if index % 25 == 0 or index == len(selected):
                    connection.commit()
                    total = connection.execute("SELECT COUNT(*) FROM samples").fetchone()[0]
                    print(f"Processed sentence poses {index}/{len(selected)}; corpus rows {total}.", flush=True)

        metadata = {
            "schemaVersion": 1,
            "source": "Exploration-Lab/iSign",
            "revision": args.revision,
            "license": ISIGN_LICENSE,
            "rawArchiveStored": False,
            "targetFrames": SENTENCE_FRAMES,
            "pointCount": 93,
            "failuresByType": dict(sorted(failures.items())),
            "prunedStaticRows": pruned_static_rows,
            "partFingerprints": archive.part_fingerprints,
            "archiveDirectorySha256": archive.archive_directory_sha256,
            "acceptedCompressedSourceBytes": int(
                connection.execute(
                    "SELECT COALESCE(SUM(compressed_source_bytes), 0) FROM samples"
                ).fetchone()[0]
            ),
            "bytesTransferredThisRun": archive.bytes_transferred - transferred_before,
            "rangeRequestsThisRun": archive.range_requests - requests_before,
            "transferAccounting": "successful and failed HTTP response bodies for this invocation",
        }
        connection.execute(
            "INSERT OR REPLACE INTO metadata(key, value) VALUES ('corpus', ?)",
            (json.dumps(metadata, sort_keys=True),),
        )
        connection.commit()

    counts = dict(connection.execute("SELECT split, COUNT(*) FROM samples GROUP BY split"))
    total_bytes = int(connection.execute("SELECT COALESCE(SUM(compressed_source_bytes), 0) FROM samples").fetchone()[0])
    connection.close()
    print(
        json.dumps(
            {
                "output": str(args.output),
                "rows": sum(counts.values()),
                "splits": counts,
                "sourceBytesReadForRows": total_bytes,
                "rawArchiveStored": False,
                "failuresByType": dict(sorted(failures.items())),
                "prunedStaticRows": pruned_static_rows,
            },
            indent=2,
        )
    )


def parse_bytes(value: str) -> int:
    match = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*([kmgt]?i?b)?\s*", value, re.IGNORECASE)
    if not match:
        raise argparse.ArgumentTypeError("Use a byte count such as 2GB or 16GiB.")
    number = float(match.group(1))
    suffix = (match.group(2) or "b").lower()
    powers = {"b": 0, "kb": 1, "kib": 1, "mb": 2, "mib": 2, "gb": 3, "gib": 3, "tb": 4, "tib": 4}
    base = 1024 if "i" in suffix else 1000
    return int(number * (base ** powers[suffix]))


def require_terms(args: argparse.Namespace) -> None:
    if not args.accept_research_only_terms:
        raise SystemExit(
            "Pass --accept-research-only-terms after accepting the gated iSign noncommercial and no-redistribution terms."
        )


def require_private_path(path: Path, label: str) -> None:
    root = PRIVATE_DATA_ROOT.resolve()
    candidate = path.resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise CorpusBuildError(
            f"{label} must remain under {PRIVATE_DATA_ROOT} because iSign artifacts are gated and non-redistributable."
        ) from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=Path("data/raw/isign"))
    parser.add_argument("--revision", default=ISIGN_DATASET_REVISION)
    subparsers = parser.add_subparsers(dest="command", required=True)

    words = subparsers.add_parser("extract-words", help="Build a private isolated-word motion library.")
    words.add_argument("--output", type=Path, default=DEFAULT_PRIVATE_ROOT / "word-motion-library.json.gz")
    words.add_argument("--max-bytes", type=parse_bytes, default=parse_bytes("2GiB"))
    words.add_argument("--workers", type=int, default=4)
    words.add_argument("--accept-research-only-terms", action="store_true")
    words.set_defaults(handler=build_word_library, requires_terms=True)

    plan = subparsers.add_parser("plan-sentences", help="Create a deterministic private balanced subset plan.")
    plan.add_argument("--samples", type=int, default=10_000)
    plan.add_argument("--output", type=Path, default=DEFAULT_PRIVATE_ROOT / "sentence-plan.private.jsonl")
    plan.set_defaults(handler=write_sentence_plan, requires_terms=False)

    extract = subparsers.add_parser("extract-sentences", help="Range-read selected poses into compact SQLite.")
    extract.add_argument("--plan", type=Path, default=DEFAULT_PRIVATE_ROOT / "sentence-plan.private.jsonl")
    extract.add_argument("--output", type=Path, default=DEFAULT_PRIVATE_ROOT / "sentence-pose-corpus.private.sqlite")
    extract.add_argument("--max-bytes", type=parse_bytes, default=parse_bytes("16GiB"))
    extract.add_argument("--workers", type=int, default=4)
    extract.add_argument("--accept-research-only-terms", action="store_true")
    extract.set_defaults(handler=extract_sentence_corpus, requires_terms=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if getattr(args, "workers", 1) < 1 or getattr(args, "workers", 1) > 12:
        raise SystemExit("--workers must be between 1 and 12.")
    if getattr(args, "requires_terms", False):
        require_terms(args)
    try:
        if args.command == "plan-sentences":
            require_private_path(args.output, "Sentence plan")
        elif args.command == "extract-words":
            require_private_path(args.output, "Word-motion output")
        elif args.command == "extract-sentences":
            require_private_path(args.plan, "Sentence plan")
            require_private_path(args.output, "Sentence corpus")
        args.handler(args)
    except (ArchiveAccessError, CorpusBuildError, FileNotFoundError, sqlite3.Error) as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()
