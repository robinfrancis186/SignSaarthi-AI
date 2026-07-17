from __future__ import annotations

import json
import unittest
from pathlib import Path


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIRECTORY = WORKSPACE_ROOT / "data" / "models"
MOTION_CATALOG_PATH = MODEL_DIRECTORY / "isl-motion-catalog.json"
MOTION_LIBRARY_PATH = (
    WORKSPACE_ROOT
    / "apps"
    / "extension"
    / "src"
    / "assets"
    / "motion"
    / "isl-motion-library.json"
)
PUBLIC_DATA_PATH = (
    WORKSPACE_ROOT / "output" / "huggingface" / "Robin186-ISL" / "data" / "isl.jsonl"
)


class RecognitionReleaseBoundaryTests(unittest.TestCase):
    def test_release_does_not_ship_an_unverified_recognition_model(self) -> None:
        forbidden_paths = [
            MODEL_DIRECTORY / "isl-keypoint-model.json",
            MODEL_DIRECTORY / "isl-temporal-model.json",
            *MODEL_DIRECTORY.glob("*.onnx"),
        ]

        self.assertEqual(forbidden_paths, [
            MODEL_DIRECTORY / "isl-keypoint-model.json",
            MODEL_DIRECTORY / "isl-temporal-model.json",
        ])
        self.assertTrue(all(not path.exists() for path in forbidden_paths))

    def test_release_motion_artifacts_are_complete_and_synchronized(self) -> None:
        catalog = json.loads(MOTION_CATALOG_PATH.read_text(encoding="utf-8"))
        library = json.loads(MOTION_LIBRARY_PATH.read_text(encoding="utf-8"))

        self.assertEqual(catalog["clipCount"], 262)
        self.assertEqual(library["clipCount"], 262)
        self.assertEqual(catalog["timingAudit"]["status"], "passed")
        self.assertEqual(library["timingAudit"]["status"], "passed")
        self.assertEqual(
            [clip["id"] for clip in catalog["clips"]],
            [clip["id"] for clip in library["clips"]],
        )
        self.assertTrue(all(clip["playable"] is True for clip in catalog["clips"]))
        self.assertTrue(all(clip["expertReviewed"] is False for clip in catalog["clips"]))
        self.assertTrue(all(len(clip["frames"]) == 32 for clip in library["clips"]))

    def test_public_bundle_contains_metadata_without_landmark_payloads(self) -> None:
        rows = [
            json.loads(line)
            for line in PUBLIC_DATA_PATH.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

        self.assertEqual(len(rows), 12_434)
        self.assertEqual(sum(row["motion"] is not None for row in rows), 262)
        quarantined = [
            row
            for row in rows
            if row.get("include", {}).get("motionStatus")
            == "quarantined_source_label_conflict"
        ]
        self.assertEqual([row["term"] for row in quarantined], ["Second (Number)"])
        self.assertTrue(
            all(
                row["motion"] is None or row["motion"]["framePayloadIncluded"] is False
                for row in rows
            )
        )
        self.assertTrue(all("isign" not in row["datasetIds"] for row in rows))
        serialized = PUBLIC_DATA_PATH.read_text(encoding="utf-8")
        for forbidden_key in ('"frames"', '"leftHand"', '"rightHand"', '"pose"'):
            self.assertNotIn(forbidden_key, serialized)


if __name__ == "__main__":
    unittest.main()
