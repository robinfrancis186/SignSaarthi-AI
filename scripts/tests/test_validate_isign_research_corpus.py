from __future__ import annotations

import io
import json
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


SCRIPTS = Path(__file__).resolve().parents[1]
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from build_isign_research_corpus import grouped_split, stable_hash  # noqa: E402
from isign_remote_archive import ISIGN_DATASET_REVISION, ISIGN_POSE_PARTS  # noqa: E402
from validate_isign_research_corpus import (  # noqa: E402
    ValidationError,
    validate_corpus,
    validate_pose_blob,
)


class PrivateCorpusValidationTests(unittest.TestCase):
    def test_validates_every_blob_and_rejects_split_group_leakage(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            plan = root / "plan.jsonl"
            database = root / "corpus.sqlite"
            plan_rows = make_plan_rows()
            plan.write_text(
                "".join(json.dumps(row) + "\n" for row in plan_rows), encoding="utf-8"
            )
            connection = create_fixture_database(database)
            for row in plan_rows:
                connection.execute(
                    """
                    INSERT INTO samples
                    (uid_hash, video_hash, split, text, word_count, source_frames, source_fps,
                     quality_score, compressed_source_bytes, pose_npz)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        stable_hash(row["uid"]),
                        stable_hash(row["videoId"]),
                        row["split"],
                        "one two three four five",
                        5,
                        80,
                        25.0,
                        0.8,
                        100,
                        pose_blob(),
                    ),
                )
            connection.commit()
            connection.close()

            report = validate_corpus(database, plan, minimum_rows=10)
            self.assertEqual(report["acceptedRows"], 10)
            self.assertEqual(report["splitOverlaps"], {"trainVal": 0, "trainTest": 0, "valTest": 0})

            connection = sqlite3.connect(database)
            train_group = connection.execute(
                "SELECT video_hash FROM samples WHERE split = 'train' LIMIT 1"
            ).fetchone()[0]
            connection.execute(
                "UPDATE samples SET video_hash = ? WHERE split = 'val'", (train_group,)
            )
            connection.commit()
            connection.close()
            with self.assertRaisesRegex(ValidationError, "deterministic plan entry"):
                validate_corpus(database, plan, minimum_rows=10)

    def test_rejects_a_static_pose_blob(self) -> None:
        with self.assertRaisesRegex(ValidationError, "static"):
            validate_pose_blob(static_pose_blob())


def create_fixture_database(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path)
    connection.execute(
        """
        CREATE TABLE samples (
          uid_hash TEXT PRIMARY KEY,
          video_hash TEXT NOT NULL,
          split TEXT NOT NULL,
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
    connection.execute("CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
    connection.execute(
        "INSERT INTO metadata(key, value) VALUES ('corpus', ?)",
        (
            json.dumps(
                {
                    "rawArchiveStored": False,
                    "targetFrames": 64,
                    "pointCount": 93,
                    "revision": ISIGN_DATASET_REVISION,
                    "archiveDirectorySha256": "a" * 64,
                    "acceptedCompressedSourceBytes": 1_000,
                    "partFingerprints": [
                        {"name": name, "size": 1_000, "etag": "b" * 64}
                        for name in ISIGN_POSE_PARTS
                    ],
                }
            ),
        ),
    )
    return connection


def pose_blob() -> bytes:
    target = io.BytesIO()
    xy = np.zeros((64, 93, 2), dtype=np.float16)
    xy[:, 11, 0] = -0.5
    xy[:, 12, 0] = 0.5
    xy[:, 33:75, 0] = np.linspace(0, 1.5, 64, dtype=np.float16)[:, None]
    np.savez_compressed(
        target,
        xy=xy,
        confidence=np.full((64, 93), 255, dtype=np.uint8),
    )
    return target.getvalue()


def static_pose_blob() -> bytes:
    target = io.BytesIO()
    np.savez_compressed(
        target,
        xy=np.zeros((64, 93, 2), dtype=np.float16),
        confidence=np.full((64, 93), 255, dtype=np.uint8),
    )
    return target.getvalue()


def make_plan_rows() -> list[dict[str, object]]:
    required = {"train": 8, "val": 1, "test": 1}
    selected: dict[str, list[str]] = {split: [] for split in required}
    index = 0
    while any(len(selected[split]) < count for split, count in required.items()):
        video_id = f"video-{index}"
        split = grouped_split(video_id)
        if len(selected[split]) < required[split]:
            selected[split].append(video_id)
        index += 1
    text = "one two three four five"
    return [
        {
            "uid": f"{video_id}-1",
            "videoId": video_id,
            "split": split,
            "text": text,
            "normalizedText": text,
            "wordCount": 5,
        }
        for split in ("train", "val", "test")
        for video_id in selected[split]
    ]


if __name__ == "__main__":
    unittest.main()
