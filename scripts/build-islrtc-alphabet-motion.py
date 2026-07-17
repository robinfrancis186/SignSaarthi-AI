#!/usr/bin/env python3
"""Build compact A-Z avatar motion clips from the official ISLRTC alphabet video.

The source video is downloaded to a temporary directory unless --source-video is
provided. Only normalized MediaPipe pose/hand landmarks are written to the
repository; no source video or audio is retained.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


SOURCE_VIDEO_ID = "VdwKSyza5oI"
SOURCE_VIDEO_URL = f"https://youtu.be/{SOURCE_VIDEO_ID}"
SOURCE_BROCHURE_URL = (
    "https://cdnbbsr.s3waas.gov.in/"
    "s3f7e2b2b75b04175610e5a00c1e221ebb/uploads/2025/06/202506021220778202.pdf"
)
SOURCE_FAQ_URL = "https://islrtc.nic.in/faq/"
OUTPUT_FPS = 20.0
OUTPUT_FRAME_COUNT = 12
POSE_POINT_COUNT = 25
HAND_POINT_COUNT = 21


@dataclass(frozen=True)
class LetterSegment:
    letter: str
    start_seconds: float
    end_seconds: float
    variant: str = "Sign 1"


# Timings select the first published variant for letters that have alternatives.
# They were audited against the 143-second official ISLRTC A-Z video.
LETTER_SEGMENTS = (
    LetterSegment("A", 5.2, 9.8),
    LetterSegment("B", 9.8, 13.0),
    LetterSegment("C", 13.0, 16.6),
    LetterSegment("D", 20.5, 25.1),
    LetterSegment("E", 25.1, 27.6),
    LetterSegment("F", 34.8, 39.0),
    LetterSegment("G", 39.0, 43.2),
    LetterSegment("H", 43.2, 47.3),
    LetterSegment("I", 47.3, 50.1),
    LetterSegment("J", 53.8, 57.1),
    LetterSegment("K", 61.4, 65.6),
    LetterSegment("L", 65.6, 68.9),
    LetterSegment("M", 68.9, 73.1),
    LetterSegment("N", 73.1, 77.0),
    LetterSegment("O", 77.0, 81.1),
    LetterSegment("P", 81.1, 85.1),
    LetterSegment("Q", 85.1, 89.0),
    LetterSegment("R", 89.0, 92.5),
    LetterSegment("S", 92.5, 96.9),
    LetterSegment("T", 96.9, 100.9),
    LetterSegment("U", 104.7, 108.4),
    LetterSegment("V", 112.5, 115.8),
    LetterSegment("W", 119.6, 123.7),
    LetterSegment("X", 123.7, 127.8),
    LetterSegment("Y", 127.8, 131.4),
    LetterSegment("Z", 131.4, 134.8),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-video",
        type=Path,
        help="Existing local copy of the official video. Omit to download it temporarily.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("apps/extension/src/assets/motion/islrtc-alphabet-library.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        import cv2  # type: ignore
        import mediapipe as mp  # type: ignore
    except ImportError as exc:
        raise SystemExit(
            "Missing dependencies. Run this script with .venv-mediapipe/bin/python after "
            "installing scripts/requirements-mediapipe.txt and yt-dlp."
        ) from exc

    if args.source_video:
        source_video = args.source_video.resolve()
        if not source_video.exists():
            raise FileNotFoundError(source_video)
        artifact = build_artifact(cv2, mp, source_video)
    else:
        with tempfile.TemporaryDirectory(prefix="signsaarthi-islrtc-alphabet-") as temp_dir:
            source_video = Path(temp_dir) / "islrtc-alphabet.mp4"
            download_source_video(source_video)
            artifact = build_artifact(cv2, mp, source_video)

    validate_artifact(artifact)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, separators=(",", ":")) + "\n", encoding="utf-8")
    print(
        f"Wrote {len(artifact['clips'])} official-source alphabet clips to {args.output} "
        f"({args.output.stat().st_size / 1024:.1f} KiB)."
    )


def download_source_video(output_path: Path) -> None:
    command = [
        sys.executable,
        "-m",
        "yt_dlp",
        "--no-playlist",
        "-f",
        "best[ext=mp4][height<=720]/best[height<=720]",
        "-o",
        str(output_path),
        SOURCE_VIDEO_URL,
    ]
    try:
        subprocess.run(command, check=True)
    except (subprocess.CalledProcessError, ModuleNotFoundError) as exc:
        raise RuntimeError(
            "Unable to download the official alphabet source. Install yt-dlp in the "
            "MediaPipe virtual environment or pass --source-video."
        ) from exc


def build_artifact(cv2: Any, mp: Any, source_video: Path) -> dict[str, Any]:
    holistic = mp.solutions.holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        refine_face_landmarks=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    try:
        clips = [extract_letter_clip(cv2, holistic, source_video, segment) for segment in LETTER_SEGMENTS]
    finally:
        holistic.close()

    return {
        "schemaVersion": 1,
        "snapshotDate": "2026-07-14",
        "source": "ISLRTC official Alphabets A Z in ISL",
        "sourceVideoId": SOURCE_VIDEO_ID,
        "sourceUrl": SOURCE_VIDEO_URL,
        "sourceBrochureUrl": SOURCE_BROCHURE_URL,
        "usageTermsUrl": SOURCE_FAQ_URL,
        "attribution": (
            "Indian Sign Language Research and Training Centre (ISLRTC), "
            "Department of Empowerment of Persons with Disabilities, Government of India"
        ),
        "license": (
            "ISLRTC permits dictionary use for research, teaching, and technology with "
            "acknowledgement, and prohibits resale or profiteering; see the official FAQ. "
            "Raw source video is not redistributed."
        ),
        "expertReviewStatus": "official_source_avatar_transfer_not_expert_reviewed",
        "reviewStatement": (
            "Letter signs come from the official ISLRTC source. MediaPipe landmark extraction "
            "and avatar rendering are automated and require Deaf/ISL expert validation."
        ),
        "derivation": {
            "extractor": "MediaPipe Holistic 0.10.21",
            "poseLandmarks": POSE_POINT_COUNT,
            "handLandmarksPerHand": HAND_POINT_COUNT,
            "framesPerClip": OUTPUT_FRAME_COUNT,
            "fps": OUTPUT_FPS,
            "rawVideoStored": False,
            "rawAudioStored": False,
        },
        "clips": clips,
    }


def extract_letter_clip(
    cv2: Any,
    holistic: Any,
    source_video: Path,
    segment: LetterSegment,
) -> dict[str, Any]:
    capture = cv2.VideoCapture(str(source_video))
    source_fps = float(capture.get(cv2.CAP_PROP_FPS) or 25.0)
    start_frame = max(0, math.floor(segment.start_seconds * source_fps))
    end_frame = max(start_frame + 1, math.ceil(segment.end_seconds * source_fps))
    capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    timestamps: list[float] = []
    pose_rows: list[np.ndarray] = []
    left_rows: list[np.ndarray] = []
    right_rows: list[np.ndarray] = []
    frame_index = start_frame
    while frame_index < end_frame:
        ok, frame = capture.read()
        if not ok:
            break
        if (frame_index - start_frame) % 2 == 0:
            result = holistic.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            timestamps.append(frame_index / source_fps)
            pose_rows.append(landmark_array(result.pose_landmarks, POSE_POINT_COUNT))
            left_rows.append(landmark_array(result.left_hand_landmarks, HAND_POINT_COUNT))
            right_rows.append(landmark_array(result.right_hand_landmarks, HAND_POINT_COUNT))
        frame_index += 1
    capture.release()

    if len(timestamps) < OUTPUT_FRAME_COUNT:
        raise RuntimeError(f"Too few source frames for letter {segment.letter}.")

    target_times = np.linspace(timestamps[0], timestamps[-1], OUTPUT_FRAME_COUNT)
    pose = resample_landmarks(np.stack(pose_rows), np.asarray(timestamps), target_times)
    left = resample_landmarks(np.stack(left_rows), np.asarray(timestamps), target_times)
    right = resample_landmarks(np.stack(right_rows), np.asarray(timestamps), target_times)
    if not np.isfinite(pose).all() or not (np.isfinite(left).all() or np.isfinite(right).all()):
        raise RuntimeError(f"MediaPipe did not recover usable pose and hand motion for {segment.letter}.")

    frames = []
    for index in range(OUTPUT_FRAME_COUNT):
        frames.append(
            {
                "pose": point_list(pose[index]),
                "leftHand": point_list(left[index]) if np.isfinite(left[index]).all() else [],
                "rightHand": point_list(right[index]) if np.isfinite(right[index]).all() else [],
            }
        )

    duration_ms = ((OUTPUT_FRAME_COUNT - 1) / OUTPUT_FPS) * 1000.0
    return {
        "id": f"islrtc-alphabet-{segment.letter.lower()}",
        "label": segment.letter,
        "normalizedLabel": segment.letter.lower(),
        "fps": OUTPUT_FPS,
        "durationMs": round(duration_ms, 3),
        "datasetId": "islrtc",
        "motionSource": "islrtc_official",
        "expertReviewed": False,
        "officialVariant": segment.variant,
        "sourceStartMs": round(segment.start_seconds * 1000),
        "sourceEndMs": round(segment.end_seconds * 1000),
        "frames": frames,
    }


def landmark_array(landmark_list: Any, count: int) -> np.ndarray:
    values = np.full((count, 2), np.nan, dtype=np.float64)
    if not landmark_list:
        return values
    for index, landmark in enumerate(landmark_list.landmark[:count]):
        values[index] = (float(landmark.x), float(landmark.y))
    return values


def resample_landmarks(
    values: np.ndarray,
    source_times: np.ndarray,
    target_times: np.ndarray,
) -> np.ndarray:
    output = np.full((target_times.shape[0], values.shape[1], values.shape[2]), np.nan)
    for point_index in range(values.shape[1]):
        for axis in range(values.shape[2]):
            series = values[:, point_index, axis]
            valid = np.isfinite(series)
            if not np.any(valid):
                continue
            output[:, point_index, axis] = np.interp(
                target_times,
                source_times[valid],
                series[valid],
            )
    return output


def point_list(points: np.ndarray) -> list[dict[str, float]]:
    return [
        {"x": round(float(point[0]), 5), "y": round(float(point[1]), 5)}
        for point in points
    ]


def validate_artifact(artifact: dict[str, Any]) -> None:
    clips = artifact["clips"]
    labels = [clip["label"] for clip in clips]
    expected = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    if labels != expected:
        raise ValueError(f"Alphabet coverage mismatch: expected {expected}, received {labels}.")
    if len({clip["id"] for clip in clips}) != len(expected):
        raise ValueError("Alphabet clip IDs must be unique.")
    for clip in clips:
        if len(clip["frames"]) != OUTPUT_FRAME_COUNT:
            raise ValueError(f"{clip['id']} has an invalid frame count.")
        points = [
            point
            for frame in clip["frames"]
            for key in ("pose", "leftHand", "rightHand")
            for point in frame[key]
        ]
        if not points:
            raise ValueError(f"{clip['id']} has no landmarks.")
        for point in points:
            if not (math.isfinite(point["x"]) and math.isfinite(point["y"])):
                raise ValueError(f"{clip['id']} contains non-finite coordinates.")
        x_values = [point["x"] for point in points]
        y_values = [point["y"] for point in points]
        if max(max(x_values) - min(x_values), max(y_values) - min(y_values)) < 0.005:
            raise ValueError(f"{clip['id']} is effectively static.")


if __name__ == "__main__":
    main()
