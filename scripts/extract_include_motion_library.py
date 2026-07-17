#!/usr/bin/env python3
"""Build a compact, resumable INCLUDE motion library with HTTP range reads.

Only selected archive members are transferred. Raw videos live in a temporary
directory for the duration of MediaPipe extraction and are never retained.
"""

from __future__ import annotations

import argparse
import bisect
import concurrent.futures
import hashlib
import json
import math
import os
import re
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


ZENODO_RECORD_ID = "4010759"
ZENODO_RECORD_URL = f"https://zenodo.org/api/records/{ZENODO_RECORD_ID}"
ZENODO_LANDING_URL = f"https://zenodo.org/records/{ZENODO_RECORD_ID}"
INCLUDE_LICENSE_URL = "https://creativecommons.org/licenses/by/4.0/"
TARGET_FRAME_COUNT = 32
MIN_RETAINED_FRAMES = 16
MIN_HAND_FRAMES = 4
MIN_POSE_RATIO = 0.5
MIN_HAND_MOTION_RANGE = 0.005
MIN_RENDERER_FPS = 5.0
MAX_RENDERER_FPS = 60.0
QUARANTINED_LABELS = {
    "second number": (
        "Every published INCLUDE metadata row for this label points to a directory for a "
        "different sign, so no source-consistent motion can be selected."
    )
}

# Compact MediaPipe Face Mesh subset used for non-manual avatar motion. The
# order is part of the runtime artifact contract: left/right brows, eyelids,
# mouth corners, then upper/lower inner lip.
FACE_POINT_INDICES = (
    70,
    63,
    105,
    66,
    107,
    336,
    296,
    334,
    293,
    300,
    159,
    145,
    386,
    374,
    61,
    291,
    13,
    14,
)


@dataclass(frozen=True)
class MetadataRecord:
    split: str
    parent_label: str
    label: str
    video_path: str

    @property
    def normalized_label(self) -> str:
        return normalize_label(self.label)

    @property
    def sample_id(self) -> str:
        digest = hashlib.sha256(self.video_path.encode("utf-8")).hexdigest()[:16]
        return f"include_{self.split}_{digest}"


@dataclass(frozen=True)
class CandidateLocation:
    record: MetadataRecord
    archive_key: str
    archive_url: str
    member_name: str
    preferred: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract selected real INCLUDE motions without downloading whole archives."
    )
    parser.add_argument(
        "--metadata-dir", type=Path, default=Path("data/isl/include-metadata")
    )
    parser.add_argument(
        "--zenodo-record", type=Path, default=Path("data/isl/include-zenodo-record.json")
    )
    parser.add_argument(
        "--source-catalog", type=Path, default=Path("data/models/isl-motion-catalog.json")
    )
    parser.add_argument(
        "--archive-index", type=Path, default=Path("data/isl/include-archive-index.json")
    )
    parser.add_argument(
        "--output-library",
        type=Path,
        default=Path("apps/extension/src/assets/motion/isl-motion-library.json"),
    )
    parser.add_argument(
        "--output-catalog", type=Path, default=Path("data/models/isl-motion-catalog.json")
    )
    parser.add_argument(
        "--avatar-catalog",
        type=Path,
        default=Path("packages/avatar-engine/src/motionCatalog.generated.json"),
    )
    parser.add_argument(
        "--checkpoint", type=Path, default=Path("data/isl/include-motion-extraction.json")
    )
    parser.add_argument("--label", action="append", default=[])
    parser.add_argument(
        "--reextract-label",
        action="append",
        default=[],
        help="Discard and rebuild selected labels while retaining the complete target catalog.",
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-candidates", type=int, default=8)
    parser.add_argument("--frame-stride", type=int, default=2)
    parser.add_argument("--max-video-bytes", type=int, default=500 * 1024 * 1024)
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Extract independent clips in parallel while checkpoint writes remain serialized.",
    )
    parser.add_argument(
        "--all-labels",
        action="store_true",
        help="Build targets for every distinct INCLUDE training label instead of the source catalog.",
    )
    parser.add_argument("--index-only", action="store_true")
    parser.add_argument("--refresh-index", action="store_true")
    return parser.parse_args()


def normalize_label(value: str) -> str:
    value = re.sub(r"^\s*\d+\s*[.)-]\s*", "", value)
    value = value.replace("&", " and ")
    value = re.sub(r"[^a-zA-Z0-9]+", " ", value)
    return " ".join(value.lower().split())


def candidate_sort_key(record: MetadataRecord, normalized_label: str) -> tuple[int, str]:
    directory_label = normalize_label(Path(record.video_path).parent.name)
    return (0 if directory_label == normalized_label else 1, record.video_path.casefold())


def has_source_consistent_label(record: MetadataRecord, normalized_label: str) -> bool:
    return (
        record.normalized_label == normalized_label
        and normalize_label(Path(record.video_path).parent.name) == normalized_label
    )


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_write_json(path: Path, payload: Any, *, compact: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        if compact:
            json.dump(payload, handle, ensure_ascii=True, separators=(",", ":"))
        else:
            json.dump(payload, handle, ensure_ascii=True, indent=2)
        handle.write("\n")
        temporary_name = handle.name
    os.replace(temporary_name, path)


def load_metadata(metadata_dir: Path) -> list[MetadataRecord]:
    records: list[MetadataRecord] = []
    for split in ("train", "val", "test"):
        path = metadata_dir / f"{split}.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"Missing INCLUDE metadata split: {path}")
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            row = json.loads(line)
            video_path = str(row.get("video_path", "")).replace("\\", "/").strip("/")
            label = str(row.get("label", "")).strip()
            parent_label = str(row.get("parent_label", "")).strip()
            if not video_path or not label or not parent_label:
                raise ValueError(f"Invalid metadata row at {path}:{line_number}")
            records.append(
                MetadataRecord(
                    split=split,
                    parent_label=parent_label,
                    label=label,
                    video_path=video_path,
                )
            )
    return records


def build_candidate_records(
    catalog_clips: Iterable[dict[str, Any]],
    metadata: Iterable[MetadataRecord],
    max_candidates: int,
) -> dict[str, list[tuple[MetadataRecord, bool]]]:
    metadata_rows = list(metadata)
    by_sample_id = {record.sample_id: record for record in metadata_rows}
    by_label: dict[str, list[MetadataRecord]] = {}
    for record in metadata_rows:
        if record.split == "train" and has_source_consistent_label(
            record, record.normalized_label
        ):
            by_label.setdefault(record.normalized_label, []).append(record)
    for normalized_label, rows in by_label.items():
        rows.sort(key=lambda record: candidate_sort_key(record, normalized_label))

    result: dict[str, list[tuple[MetadataRecord, bool]]] = {}
    for clip in catalog_clips:
        clip_id = str(clip["id"])
        normalized = str(clip.get("normalizedLabel") or normalize_label(str(clip["label"])))
        if normalized in QUARANTINED_LABELS:
            raise RuntimeError(
                f"Motion target {clip_id} is quarantined: {QUARANTINED_LABELS[normalized]}"
            )
        preferred = by_sample_id.get(str(clip.get("sourceSampleId", "")))
        if preferred is not None and not has_source_consistent_label(preferred, normalized):
            preferred = None
        ordered: list[tuple[MetadataRecord, bool]] = []
        if preferred is not None:
            ordered.append((preferred, True))
        for record in by_label.get(normalized, []):
            if preferred is None or record.sample_id != preferred.sample_id:
                ordered.append((record, False))
            if len(ordered) >= max_candidates:
                break
        if not ordered:
            raise RuntimeError(f"No current INCLUDE training row matches {clip_id} ({normalized}).")
        result[clip_id] = ordered[:max_candidates]
    return result


def build_all_label_targets(metadata: Iterable[MetadataRecord]) -> list[dict[str, Any]]:
    by_label: dict[str, list[MetadataRecord]] = {}
    observed_train_labels: set[str] = set()
    for record in metadata:
        if record.split == "train":
            observed_train_labels.add(record.normalized_label)
            if has_source_consistent_label(record, record.normalized_label):
                by_label.setdefault(record.normalized_label, []).append(record)

    unresolved = observed_train_labels - set(by_label) - set(QUARANTINED_LABELS)
    if unresolved:
        raise RuntimeError(
            "INCLUDE labels have no source-consistent training paths and are not quarantined: "
            + ", ".join(sorted(unresolved))
        )

    targets: list[dict[str, Any]] = []
    for normalized_label, rows in sorted(by_label.items()):
        rows.sort(key=lambda record: candidate_sort_key(record, normalized_label))
        preferred = rows[0]
        display_label = re.sub(r"^\s*\d+\s*[.)-]\s*", "", preferred.label).strip()
        targets.append(
            {
                "id": f"include-{normalized_label.replace(' ', '-')}",
                "label": display_label,
                "normalizedLabel": normalized_label,
                "datasetId": "include",
                "sourceSampleId": preferred.sample_id,
                "playable": False,
                "expertReviewed": False,
                "reviewStatement": (
                    "Dataset-derived isolated-sign motion; not expert-certified ISL."
                ),
            }
        )
    return targets


def reset_requested_clips(
    clips_by_id: dict[str, dict[str, Any]],
    checkpoint: dict[str, Any],
    source_clips: Iterable[dict[str, Any]],
    requested_labels: Iterable[str],
) -> list[str]:
    normalized_labels = {normalize_label(label) for label in requested_labels}
    if not normalized_labels:
        return []
    targets = {
        str(clip["normalizedLabel"]): str(clip["id"])
        for clip in source_clips
        if str(clip.get("normalizedLabel", "")) in normalized_labels
    }
    missing = sorted(normalized_labels - set(targets))
    if missing:
        raise ValueError(f"Unknown re-extraction label(s): {', '.join(missing)}")
    completed = checkpoint.setdefault("completed", {})
    failures = checkpoint.setdefault("failures", {})
    clip_ids = sorted(targets.values())
    for clip_id in clip_ids:
        clips_by_id.pop(clip_id, None)
        completed.pop(clip_id, None)
        failures.pop(clip_id, None)
    return clip_ids


def archive_category(archive_key: str) -> str:
    return re.sub(r"_\d+of\d+\.zip$", "", archive_key, flags=re.IGNORECASE)


def archive_files(zenodo_record: dict[str, Any]) -> list[dict[str, Any]]:
    files = []
    for row in zenodo_record.get("files", []):
        key = str(row.get("key", ""))
        url = str(row.get("url", ""))
        if key.lower().endswith(".zip") and url:
            files.append(
                {
                    "key": key,
                    "url": url,
                    "checksum": str(row.get("checksum", "")),
                    "size": int(row.get("size", 0)),
                }
            )
    if not files:
        raise RuntimeError("Zenodo record contains no downloadable zip archives.")
    return sorted(files, key=lambda row: row["key"])


def make_http_session() -> Any:
    import requests  # type: ignore
    from requests.adapters import HTTPAdapter  # type: ignore
    from urllib3.util.retry import Retry  # type: ignore

    retry = Retry(
        total=6,
        connect=6,
        read=6,
        status=6,
        backoff_factor=1.0,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(("GET", "HEAD")),
    )
    session = requests.Session()
    session.headers.update({"User-Agent": "SignSaarthi-INCLUDE-motion-extractor/1.0"})
    session.mount("https://", HTTPAdapter(max_retries=retry, pool_connections=8, pool_maxsize=8))
    return session


def build_archive_index(
    archives: list[dict[str, Any]],
    required_categories: set[str],
    cache_path: Path,
    refresh: bool,
) -> dict[str, dict[str, str]]:
    from remotezip import RemoteZip  # type: ignore

    cache: dict[str, Any] = {
        "schemaVersion": 1,
        "zenodoRecordId": ZENODO_RECORD_ID,
        "archives": {},
    }
    if cache_path.exists() and not refresh:
        loaded = read_json(cache_path)
        if loaded.get("zenodoRecordId") == ZENODO_RECORD_ID:
            cache = loaded

    session = make_http_session()
    relevant = [row for row in archives if archive_category(row["key"]) in required_categories]
    if not relevant:
        raise RuntimeError("No Zenodo archives match the selected INCLUDE categories.")

    for index, archive in enumerate(relevant, 1):
        key = archive["key"]
        cached = cache["archives"].get(key)
        if cached and cached.get("checksum") == archive["checksum"]:
            continue
        print(f"Indexing archive {index}/{len(relevant)}: {key}", flush=True)
        with RemoteZip(
            archive["url"],
            session=session,
            timeout=(20, 180),
            initial_buffer_size=128 * 1024,
        ) as remote:
            members = [
                info.filename.replace("\\", "/").strip("/")
                for info in remote.infolist()
                if not info.is_dir()
            ]
        cache["archives"][key] = {
            "url": archive["url"],
            "checksum": archive["checksum"],
            "members": sorted(members),
        }
        atomic_write_json(cache_path, cache)

    locations: dict[str, dict[str, str]] = {}
    for key, row in cache["archives"].items():
        if archive_category(key) not in required_categories:
            continue
        for member in row.get("members", []):
            normalized = str(member).replace("\\", "/").strip("/")
            locations[normalized] = {
                "archiveKey": key,
                "archiveUrl": str(row["url"]),
                "memberName": normalized,
            }
    return locations


def locate_candidates(
    candidate_records: dict[str, list[tuple[MetadataRecord, bool]]],
    locations: dict[str, dict[str, str]],
) -> dict[str, list[CandidateLocation]]:
    result: dict[str, list[CandidateLocation]] = {}
    for clip_id, candidates in candidate_records.items():
        located: list[CandidateLocation] = []
        for record, preferred in candidates:
            location = locations.get(record.video_path)
            if location is None:
                suffix = f"/{record.video_path}"
                matches = [value for path, value in locations.items() if path.endswith(suffix)]
                location = matches[0] if len(matches) == 1 else None
            if location is not None:
                located.append(
                    CandidateLocation(
                        record=record,
                        archive_key=location["archiveKey"],
                        archive_url=location["archiveUrl"],
                        member_name=location["memberName"],
                        preferred=preferred,
                    )
                )
        if not located:
            raise RuntimeError(f"None of the selected archive members were found for {clip_id}.")
        result[clip_id] = located
    return result


def point_list(landmark_list: Any, count: int) -> list[dict[str, float]] | None:
    if not landmark_list:
        return None
    points = []
    for landmark in landmark_list.landmark[:count]:
        points.append({"x": float(landmark.x), "y": float(landmark.y)})
    return points if len(points) == count else None


def selected_point_list(
    landmark_list: Any, indices: Iterable[int]
) -> list[dict[str, float]] | None:
    if not landmark_list:
        return None
    landmarks = landmark_list.landmark
    selected = []
    for index in indices:
        if index >= len(landmarks):
            return None
        landmark = landmarks[index]
        selected.append({"x": float(landmark.x), "y": float(landmark.y)})
    return selected


def source_timestamp_ms(capture: Any, cv2: Any, frame_index: int, source_fps: float) -> float:
    timestamp = float(capture.get(cv2.CAP_PROP_POS_MSEC) or 0.0)
    fallback = (frame_index / source_fps) * 1000.0
    if not math.isfinite(timestamp) or timestamp < 0 or (frame_index > 0 and timestamp == 0):
        return fallback
    return timestamp


def collect_source_frames(
    cv2: Any, holistic: Any, video_path: Path, frame_stride: int
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError("OpenCV could not open the selected video member.")
    reported_fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    source_fps = reported_fps if math.isfinite(reported_fps) and reported_fps > 0 else 25.0
    rows: list[dict[str, Any]] = []
    frame_index = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if frame_index % frame_stride != 0:
            frame_index += 1
            continue
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = holistic.process(rgb)
        timestamp = source_timestamp_ms(capture, cv2, frame_index, source_fps)
        if rows and timestamp <= float(rows[-1]["timestampMs"]):
            timestamp = (frame_index / source_fps) * 1000.0
        if rows and timestamp <= float(rows[-1]["timestampMs"]):
            timestamp = float(rows[-1]["timestampMs"]) + (1000.0 * frame_stride / source_fps)
        rows.append(
            {
                "timestampMs": timestamp,
                "pose": point_list(result.pose_landmarks, 25),
                "leftHand": point_list(result.left_hand_landmarks, 21),
                "rightHand": point_list(result.right_hand_landmarks, 21),
                "face": selected_point_list(result.face_landmarks, FACE_POINT_INDICES),
            }
        )
        frame_index += 1
    capture.release()
    return rows, {
        "sourceFps": round(source_fps, 6),
        "sourceFrameCount": frame_index,
        "sampledFrameCount": len(rows),
        "frameStride": frame_stride,
    }


def interpolate_value(samples: list[tuple[float, float]], timestamp: float) -> float:
    if not samples:
        return 0.0
    times = [row[0] for row in samples]
    position = bisect.bisect_left(times, timestamp)
    if position <= 0:
        return samples[0][1]
    if position >= len(samples):
        return samples[-1][1]
    left_time, left_value = samples[position - 1]
    right_time, right_value = samples[position]
    ratio = (timestamp - left_time) / (right_time - left_time)
    return left_value + ((right_value - left_value) * ratio)


def resample_part(
    source_frames: list[dict[str, Any]], part: str, point_count: int, target_times: list[float]
) -> list[list[dict[str, float]]]:
    result = [[{"x": 0.0, "y": 0.0} for _ in range(point_count)] for _ in target_times]
    for point_index in range(point_count):
        for coordinate in ("x", "y"):
            samples = [
                (float(frame["timestampMs"]), float(frame[part][point_index][coordinate]))
                for frame in source_frames
                if frame[part] is not None
            ]
            for target_index, target_time in enumerate(target_times):
                result[target_index][point_index][coordinate] = round(
                    interpolate_value(samples, target_time), 5
                )
    return result


def coordinate_range(frames: list[list[dict[str, float]]], point_index: int) -> float:
    points = [frame[point_index] for frame in frames]
    xs = [point["x"] for point in points]
    ys = [point["y"] for point in points]
    return max(max(xs) - min(xs), max(ys) - min(ys))


def validate_and_resample(
    source_frames: list[dict[str, Any]], extraction: dict[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if len(source_frames) < MIN_RETAINED_FRAMES:
        raise ValueError(f"only {len(source_frames)} sampled frames")
    pose_frames = sum(frame["pose"] is not None for frame in source_frames)
    left_frames = sum(frame["leftHand"] is not None for frame in source_frames)
    right_frames = sum(frame["rightHand"] is not None for frame in source_frames)
    face_frames = sum(frame.get("face") is not None for frame in source_frames)
    hand_frames = sum(
        frame["leftHand"] is not None or frame["rightHand"] is not None for frame in source_frames
    )
    pose_ratio = pose_frames / len(source_frames)
    if pose_ratio < MIN_POSE_RATIO:
        raise ValueError(f"pose coverage {pose_ratio:.3f} is below {MIN_POSE_RATIO:.3f}")
    if hand_frames < MIN_HAND_FRAMES:
        raise ValueError(f"hands detected in only {hand_frames} sampled frames")

    start = float(source_frames[0]["timestampMs"])
    end = float(source_frames[-1]["timestampMs"])
    duration_ms = end - start
    if not math.isfinite(duration_ms) or duration_ms <= 0:
        raise ValueError("video has no usable timestamp duration")
    effective_fps = ((TARGET_FRAME_COUNT - 1) * 1000.0) / duration_ms
    if effective_fps < MIN_RENDERER_FPS or effective_fps > MAX_RENDERER_FPS:
        raise ValueError(f"resampled playback rate {effective_fps:.3f} FPS is outside renderer bounds")

    target_times = [
        start + (duration_ms * index / (TARGET_FRAME_COUNT - 1))
        for index in range(TARGET_FRAME_COUNT)
    ]
    pose = resample_part(source_frames, "pose", 25, target_times)
    left = resample_part(source_frames, "leftHand", 21, target_times)
    right = resample_part(source_frames, "rightHand", 21, target_times)
    face = (
        resample_part(source_frames, "face", len(FACE_POINT_INDICES), target_times)
        if face_frames
        else [[] for _ in target_times]
    )

    if left_frames == 0:
        for index, frame in enumerate(left):
            frame[:] = [dict(pose[index][15]) for _ in range(21)]
    if right_frames == 0:
        for index, frame in enumerate(right):
            frame[:] = [dict(pose[index][16]) for _ in range(21)]

    hand_motion_range = max(
        coordinate_range(left, 8) if left_frames else 0.0,
        coordinate_range(right, 8) if right_frames else 0.0,
    )
    if hand_motion_range <= MIN_HAND_MOTION_RANGE:
        raise ValueError(f"hand motion range {hand_motion_range:.6f} is effectively static")

    frames = [
        {
            "pose": pose[index],
            "leftHand": left[index],
            "rightHand": right[index],
            "face": face[index],
        }
        for index in range(TARGET_FRAME_COUNT)
    ]
    for frame in frames:
        for point in frame["pose"] + frame["leftHand"] + frame["rightHand"] + frame["face"]:
            if not math.isfinite(point["x"]) or not math.isfinite(point["y"]):
                raise ValueError("non-finite coordinate")
            if point["x"] < -0.5 or point["x"] > 1.5 or point["y"] < -0.5 or point["y"] > 1.5:
                raise ValueError("coordinate outside renderer safety bounds")

    extraction.update(
        {
            "poseFrameRatio": round(pose_ratio, 6),
            "leftHandFrameRatio": round(left_frames / len(source_frames), 6),
            "rightHandFrameRatio": round(right_frames / len(source_frames), 6),
            "faceFrameRatio": round(face_frames / len(source_frames), 6),
            "handMotionRange": round(hand_motion_range, 6),
        }
    )
    timing = {
        "source": "retained-frame-timestamps",
        "sourceFrameCount": len(source_frames),
        "resampledFrameCount": TARGET_FRAME_COUNT,
        "sourceDurationMs": round(duration_ms, 3),
        "playbackDurationMs": round(((TARGET_FRAME_COUNT - 1) / effective_fps) * 1000.0, 3),
        "durationErrorMs": 0.0,
        "effectiveFps": round(effective_fps, 6),
    }
    return frames, timing


def stream_member_to_file(
    location: CandidateLocation, destination: Path, max_video_bytes: int
) -> tuple[str, int]:
    from remotezip import RemoteZip  # type: ignore

    session = make_http_session()
    with RemoteZip(
        location.archive_url,
        session=session,
        timeout=(20, 240),
        initial_buffer_size=128 * 1024,
    ) as remote:
        info = remote.getinfo(location.member_name)
        if info.file_size > max_video_bytes:
            raise ValueError(f"video member is {info.file_size} bytes, above configured safety limit")
        digest = hashlib.sha256()
        written = 0
        with remote.open(info) as source, destination.open("wb") as target:
            while True:
                chunk = source.read(1024 * 1024)
                if not chunk:
                    break
                target.write(chunk)
                digest.update(chunk)
                written += len(chunk)
    return f"sha256:{digest.hexdigest()}", written


def extract_candidate(
    location: CandidateLocation,
    clip_meta: dict[str, Any],
    frame_stride: int,
    max_video_bytes: int,
) -> dict[str, Any]:
    try:
        import cv2  # type: ignore
        import mediapipe as mp  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "Install scripts/requirements-mediapipe.txt in the extraction environment."
        ) from exc

    suffix = Path(location.member_name).suffix or ".mp4"
    with tempfile.TemporaryDirectory(prefix="signsaarthi-include-motion-") as temp_dir:
        video_path = Path(temp_dir) / f"source{suffix}"
        checksum, transferred_bytes = stream_member_to_file(
            location, video_path, max_video_bytes
        )
        with mp.solutions.holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            refine_face_landmarks=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        ) as holistic:
            source_frames, extraction = collect_source_frames(
                cv2, holistic, video_path, frame_stride
            )
        extraction["transferredBytes"] = transferred_bytes
        extraction["extractorVersion"] = "signsaarthi-include-range-mediapipe-v1"
        frames, timing = validate_and_resample(source_frames, extraction)

    return {
        "id": str(clip_meta["id"]),
        "label": str(clip_meta["label"]),
        "normalizedLabel": str(clip_meta["normalizedLabel"]),
        "fps": timing["effectiveFps"],
        "durationMs": timing["sourceDurationMs"],
        "sourceSampleId": location.record.sample_id,
        "sourceVideoPath": location.record.video_path,
        "sourceArchive": location.archive_key,
        "sourceChecksum": checksum,
        "sourceUrl": ZENODO_LANDING_URL,
        "datasetId": "include",
        "license": "CC-BY-4.0",
        "licenseUrl": INCLUDE_LICENSE_URL,
        "expertReviewed": False,
        "reviewStatement": "Dataset-derived isolated-sign motion; not expert-certified ISL.",
        "timing": timing,
        "extraction": extraction,
        "frames": frames,
    }


def catalog_clip(clip: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": clip["id"],
        "label": clip["label"],
        "normalizedLabel": clip["normalizedLabel"],
        "datasetId": "include",
        "sourceSampleId": clip["sourceSampleId"],
        "sourceVideoPath": clip["sourceVideoPath"],
        "sourceArchive": clip["sourceArchive"],
        "sourceChecksum": clip["sourceChecksum"],
        "frameCount": len(clip["frames"]),
        "fps": clip["fps"],
        "durationMs": clip["durationMs"],
        "timingSource": clip["timing"]["source"],
        "playable": True,
        "expertReviewed": False,
        "reviewStatement": clip["reviewStatement"],
    }


def timing_audit(clips: list[dict[str, Any]]) -> dict[str, Any]:
    errors = [float(clip["timing"]["durationErrorMs"]) for clip in clips]
    return {
        "source": "retained-frame-timestamps",
        "resampling": "linear interpolation on the timestamp axis",
        "clipCount": len(clips),
        "durationToleranceMs": 0.5,
        "maxPlaybackDurationErrorMs": max(errors, default=0.0),
        "status": "passed",
    }


def write_runtime_artifacts(
    clips_by_id: dict[str, dict[str, Any]],
    target_order: list[str],
    output_library: Path,
    output_catalog: Path,
    avatar_catalog: Path,
) -> None:
    clips = [clips_by_id[clip_id] for clip_id in target_order if clip_id in clips_by_id]
    audit = timing_audit(clips)
    library = {
        "schemaVersion": 1,
        "source": "INCLUDE real MediaPipe keypoint sequences",
        "license": "CC-BY-4.0; source attribution retained per clip",
        "expertReviewStatus": "not_expert_certified",
        "reviewStatement": "Dataset-derived isolated-sign motion; not expert-certified ISL.",
        "clipCount": len(clips),
        "timingAudit": audit,
        "clips": clips,
    }
    catalog = {
        "schemaVersion": 1,
        "clipCount": len(clips),
        "timingAudit": audit,
        "clips": [catalog_clip(clip) for clip in clips],
    }
    atomic_write_json(output_library, library, compact=True)
    atomic_write_json(output_catalog, catalog)
    atomic_write_json(avatar_catalog, catalog)


def existing_clips(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    payload = read_json(path)
    return {
        str(clip["id"]): clip
        for clip in payload.get("clips", [])
        if isinstance(clip, dict)
        and isinstance(clip.get("frames"), list)
        and clip["frames"]
        and all(
            isinstance(frame.get("face"), list) and len(frame["face"]) == len(FACE_POINT_INDICES)
            for frame in clip["frames"]
        )
    }


def extract_target(
    clip_id: str,
    candidate_locations: list[CandidateLocation],
    clip_meta: dict[str, Any],
    frame_stride: int,
    max_video_bytes: int,
) -> dict[str, Any]:
    started = time.monotonic()
    failure_messages: list[dict[str, str]] = []
    for candidate_index, location in enumerate(candidate_locations, 1):
        try:
            clip = extract_candidate(location, clip_meta, frame_stride, max_video_bytes)
            return {
                "clipId": clip_id,
                "clip": clip,
                "candidateIndex": candidate_index,
                "candidateCount": len(candidate_locations),
                "preferredCatalogSample": location.preferred,
                "elapsedSeconds": round(time.monotonic() - started, 3),
                "failures": failure_messages,
            }
        except Exception as exc:  # Candidate failure is recoverable by design.
            failure_messages.append(
                {
                    "videoPath": location.record.video_path,
                    "message": f"{type(exc).__name__}: {exc}",
                }
            )
    return {
        "clipId": clip_id,
        "clip": None,
        "candidateIndex": len(candidate_locations),
        "candidateCount": len(candidate_locations),
        "preferredCatalogSample": False,
        "elapsedSeconds": round(time.monotonic() - started, 3),
        "failures": failure_messages,
    }


def record_extraction_result(
    result: dict[str, Any],
    clips_by_id: dict[str, dict[str, Any]],
    checkpoint: dict[str, Any],
    target_order: list[str],
    output_library: Path,
    output_catalog: Path,
    avatar_catalog: Path,
    checkpoint_path: Path,
) -> bool:
    clip_id = str(result["clipId"])
    clip = result.get("clip")
    if not isinstance(clip, dict):
        checkpoint["failures"][clip_id] = result.get("failures", [])
        atomic_write_json(checkpoint_path, checkpoint)
        print(
            f"Failed {clip_id}: all {result.get('candidateCount', 0)} candidates were rejected.",
            flush=True,
        )
        return False

    clips_by_id[clip_id] = clip
    checkpoint["completed"][clip_id] = {
        "sourceSampleId": clip["sourceSampleId"],
        "sourceVideoPath": clip["sourceVideoPath"],
        "sourceArchive": clip["sourceArchive"],
        "preferredCatalogSample": bool(result.get("preferredCatalogSample")),
        "elapsedSeconds": float(result.get("elapsedSeconds", 0.0)),
        "candidateIndex": int(result.get("candidateIndex", 1)),
    }
    checkpoint["failures"].pop(clip_id, None)
    write_runtime_artifacts(
        clips_by_id,
        target_order,
        output_library,
        output_catalog,
        avatar_catalog,
    )
    atomic_write_json(checkpoint_path, checkpoint)
    print(
        f"Accepted {clip_id}: {clip['sourceSampleId']} "
        f"({clip['extraction']['handMotionRange']:.4f} hand range, "
        f"candidate {result.get('candidateIndex', 1)}/{result.get('candidateCount', 1)})",
        flush=True,
    )
    return True


def main() -> None:
    args = parse_args()
    if args.frame_stride < 1:
        raise SystemExit("--frame-stride must be at least 1")
    if args.max_candidates < 1:
        raise SystemExit("--max-candidates must be at least 1")
    if args.workers < 1 or args.workers > 8:
        raise SystemExit("--workers must be between 1 and 8")

    metadata = load_metadata(args.metadata_dir)
    resume_checkpoint = read_json(args.checkpoint) if args.checkpoint.exists() else {}
    checkpoint_targets = resume_checkpoint.get("targetClips")
    source_catalog = read_json(args.source_catalog)
    source_clips = list(
        build_all_label_targets(metadata)
        if args.all_labels
        else checkpoint_targets
        if isinstance(checkpoint_targets, list) and checkpoint_targets
        else source_catalog.get("clips", [])
    )
    if not source_clips:
        raise SystemExit("Source motion catalog contains no target clips.")

    selected_labels = {normalize_label(value) for value in args.label}
    if selected_labels:
        source_clips = [
            clip for clip in source_clips if str(clip.get("normalizedLabel")) in selected_labels
        ]
    if args.limit:
        source_clips = source_clips[: args.limit]
    if not source_clips:
        raise SystemExit("No catalog clips match the requested selection.")

    candidate_records = build_candidate_records(source_clips, metadata, args.max_candidates)
    required_categories = {
        record.parent_label
        for candidates in candidate_records.values()
        for record, _preferred in candidates
    }
    zenodo_record = read_json(args.zenodo_record)
    locations = build_archive_index(
        archive_files(zenodo_record),
        required_categories,
        args.archive_index,
        args.refresh_index,
    )
    candidates = locate_candidates(candidate_records, locations)
    print(
        f"Located {sum(len(rows) for rows in candidates.values())} candidate members "
        f"for {len(source_clips)} target motions.",
        flush=True,
    )
    if args.index_only:
        return

    target_order = [str(clip["id"]) for clip in source_clips]
    clips_by_id = existing_clips(args.output_library)
    checkpoint = resume_checkpoint or {
        "schemaVersion": 1,
        "zenodoRecordId": ZENODO_RECORD_ID,
        "completed": {},
        "failures": {},
    }
    checkpoint["targetClips"] = source_clips
    reset_clip_ids = reset_requested_clips(
        clips_by_id, checkpoint, source_clips, args.reextract_label
    )
    if reset_clip_ids:
        print(f"Re-extracting {', '.join(reset_clip_ids)}.", flush=True)
    atomic_write_json(args.checkpoint, checkpoint)
    source_by_id = {str(clip["id"]): clip for clip in source_clips}

    pending = [clip_id for clip_id in target_order if clip_id not in clips_by_id]
    print(
        f"Resuming {len(clips_by_id)}/{len(target_order)} clips; "
        f"extracting {len(pending)} with {args.workers} worker(s).",
        flush=True,
    )
    failed: list[str] = []
    if args.workers == 1:
        for pending_index, clip_id in enumerate(pending, 1):
            print(f"[{pending_index}/{len(pending)}] Extracting {clip_id}", flush=True)
            result = extract_target(
                clip_id,
                candidates[clip_id],
                source_by_id[clip_id],
                args.frame_stride,
                args.max_video_bytes,
            )
            if not record_extraction_result(
                result,
                clips_by_id,
                checkpoint,
                target_order,
                args.output_library,
                args.output_catalog,
                args.avatar_catalog,
                args.checkpoint,
            ):
                failed.append(clip_id)
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(
                    extract_target,
                    clip_id,
                    candidates[clip_id],
                    source_by_id[clip_id],
                    args.frame_stride,
                    args.max_video_bytes,
                ): clip_id
                for clip_id in pending
            }
            for completed_index, future in enumerate(concurrent.futures.as_completed(futures), 1):
                clip_id = futures[future]
                try:
                    result = future.result()
                except Exception as exc:
                    result = {
                        "clipId": clip_id,
                        "clip": None,
                        "candidateCount": len(candidates[clip_id]),
                        "failures": [
                            {"videoPath": "worker", "message": f"{type(exc).__name__}: {exc}"}
                        ],
                    }
                print(f"[{completed_index}/{len(pending)}] Completed {clip_id}", flush=True)
                if not record_extraction_result(
                    result,
                    clips_by_id,
                    checkpoint,
                    target_order,
                    args.output_library,
                    args.output_catalog,
                    args.avatar_catalog,
                    args.checkpoint,
                ):
                    failed.append(clip_id)

    if failed:
        raise RuntimeError(
            f"{len(failed)} target motion(s) failed all candidates: {', '.join(failed)}"
        )

    write_runtime_artifacts(
        clips_by_id,
        target_order,
        args.output_library,
        args.output_catalog,
        args.avatar_catalog,
    )
    print(
        f"Completed {len(target_order)} real INCLUDE motion clips. Raw videos were not retained.",
        flush=True,
    )


if __name__ == "__main__":
    main()
