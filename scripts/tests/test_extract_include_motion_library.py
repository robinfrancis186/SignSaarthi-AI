import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "extract_include_motion_library.py"
SPEC = importlib.util.spec_from_file_location("extract_include_motion_library", MODULE_PATH)
assert SPEC and SPEC.loader
motion = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = motion
SPEC.loader.exec_module(motion)


class IncludeMotionExtractorTests(unittest.TestCase):
    def test_normalizes_numbered_include_labels_without_collapsing_words(self):
        self.assertEqual(motion.normalize_label("58. Camera"), "camera")
        self.assertEqual(motion.normalize_label("Means-of-Transportation"), "means of transportation")

    def test_sample_id_matches_the_canonical_video_path_digest(self):
        record = motion.MetadataRecord(
            split="train",
            parent_label="Electronics",
            label="58. Camera",
            video_path="Electronics/58. Camera/MVI_4553.MOV",
        )
        self.assertEqual(record.sample_id, "include_train_7b741cc3bc7e5c78")

    def test_candidate_selection_prefers_the_catalog_sample_then_train_fallbacks(self):
        preferred = motion.MetadataRecord("train", "Home", "1. Bed", "Home/1. Bed/a.mp4")
        fallback = motion.MetadataRecord("train", "Home", "1. Bed", "Home/1. Bed/b.mp4")
        catalog = [
            {
                "id": "include-bed",
                "label": "Bed",
                "normalizedLabel": "bed",
                "sourceSampleId": preferred.sample_id,
            }
        ]
        result = motion.build_candidate_records(catalog, [fallback, preferred], 8)
        self.assertEqual(result["include-bed"], [(preferred, True), (fallback, False)])

    def test_all_label_targets_are_unique_deterministic_and_train_backed(self):
        rows = [
            motion.MetadataRecord("test", "Home", "1. Bed", "Home/1. Bed/test.mp4"),
            motion.MetadataRecord("train", "Home", "1. Bed", "Home/1. Bed/b.mp4"),
            motion.MetadataRecord("train", "Home", "1. Bed", "Home/1. Bed/a.mp4"),
            motion.MetadataRecord("train", "Places", "2. Bus Stop", "Places/2. Bus Stop/a.mp4"),
        ]
        targets = motion.build_all_label_targets(rows)
        self.assertEqual([row["id"] for row in targets], ["include-bed", "include-bus-stop"])
        self.assertEqual(targets[0]["label"], "Bed")
        self.assertEqual(
            targets[0]["sourceSampleId"],
            motion.MetadataRecord("train", "Home", "1. Bed", "Home/1. Bed/a.mp4").sample_id,
        )

    def test_all_label_targets_prefer_a_matching_class_directory(self):
        mislabeled = motion.MetadataRecord(
            "train", "Places", "28. Store or Shop", "Electronics/53. Fan/a.mov"
        )
        matching = motion.MetadataRecord(
            "train", "Places", "28. Store or Shop", "Places/28. Store or Shop/z.mov"
        )

        target = motion.build_all_label_targets([mislabeled, matching])[0]

        self.assertEqual(target["sourceSampleId"], matching.sample_id)
        candidates = motion.build_candidate_records([target], [mislabeled, matching], 8)
        self.assertEqual(candidates[target["id"]][0], (matching, True))
        self.assertEqual(candidates[target["id"]], [(matching, True)])

    def test_stale_catalog_sample_with_a_conflicting_directory_is_not_preferred(self):
        stale = motion.MetadataRecord(
            "train", "Places", "35. Bank", "Animals/1. Dog/MVI_3060.MOV"
        )
        matching = motion.MetadataRecord(
            "train", "Places", "35. Bank", "Places/35. Bank/MVI_3421.MOV"
        )
        catalog = [
            {
                "id": "include-bank",
                "label": "Bank",
                "normalizedLabel": "bank",
                "sourceSampleId": stale.sample_id,
            }
        ]

        candidates = motion.build_candidate_records(catalog, [stale, matching], 8)

        self.assertEqual(candidates["include-bank"], [(matching, False)])

    def test_known_corrupt_label_is_quarantined_from_all_label_targets(self):
        corrupt = motion.MetadataRecord(
            "train",
            "Days_and_Time",
            "Second (Number)",
            "Adjectives/2. quiet/MVI_9453.MOV",
        )

        self.assertEqual(motion.build_all_label_targets([corrupt]), [])

    def test_reextract_reset_discards_only_requested_clips(self):
        clips = {"include-fan": {"id": "include-fan"}, "include-store": {"id": "include-store"}}
        checkpoint = {
            "completed": {"include-fan": {}, "include-store": {}},
            "failures": {"include-store": []},
        }
        source = [
            {"id": "include-fan", "normalizedLabel": "fan"},
            {"id": "include-store", "normalizedLabel": "store or shop"},
        ]

        reset = motion.reset_requested_clips(clips, checkpoint, source, ["Store or Shop"])

        self.assertEqual(reset, ["include-store"])
        self.assertIn("include-fan", clips)
        self.assertNotIn("include-store", clips)
        self.assertNotIn("include-store", checkpoint["completed"])
        self.assertNotIn("include-store", checkpoint["failures"])

    def test_resampling_preserves_motion_and_complete_renderer_shapes(self):
        pose_a = [{"x": 0.4, "y": 0.4} for _ in range(25)]
        pose_b = [{"x": 0.6, "y": 0.6} for _ in range(25)]
        left_a = [{"x": 0.3, "y": 0.3} for _ in range(21)]
        left_b = [{"x": 0.7, "y": 0.7} for _ in range(21)]
        source = []
        for index in range(20):
            ratio = index / 19
            source.append(
                {
                    "timestampMs": index * 100,
                    "pose": [
                        {
                            "x": a["x"] + ((b["x"] - a["x"]) * ratio),
                            "y": a["y"] + ((b["y"] - a["y"]) * ratio),
                        }
                        for a, b in zip(pose_a, pose_b)
                    ],
                    "leftHand": [
                        {
                            "x": a["x"] + ((b["x"] - a["x"]) * ratio),
                            "y": a["y"] + ((b["y"] - a["y"]) * ratio),
                        }
                        for a, b in zip(left_a, left_b)
                    ],
                    "rightHand": None,
                    "face": [{"x": 0.45, "y": 0.25} for _ in motion.FACE_POINT_INDICES],
                }
            )
        extraction = {}
        frames, timing = motion.validate_and_resample(source, extraction)
        self.assertEqual(len(frames), 32)
        self.assertTrue(all(len(frame["pose"]) == 25 for frame in frames))
        self.assertTrue(all(len(frame["leftHand"]) == 21 for frame in frames))
        self.assertTrue(all(len(frame["rightHand"]) == 21 for frame in frames))
        self.assertTrue(
            all(len(frame["face"]) == len(motion.FACE_POINT_INDICES) for frame in frames)
        )
        self.assertGreater(extraction["handMotionRange"], 0.1)
        self.assertGreater(timing["effectiveFps"], 5)

    def test_atomic_writer_replaces_a_complete_json_document(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            motion.atomic_write_json(path, {"clipCount": 1})
            self.assertEqual(json.loads(path.read_text()), {"clipCount": 1})

    def test_timing_audit_is_empty_safe_for_resumable_output(self):
        self.assertEqual(motion.timing_audit([])["clipCount"], 0)
        self.assertEqual(motion.timing_audit([])["maxPlaybackDurationErrorMs"], 0.0)

    def test_failed_parallel_target_records_failures_without_corrupting_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = {"completed": {}, "failures": {}}
            ok = motion.record_extraction_result(
                {
                    "clipId": "include-missing",
                    "clip": None,
                    "candidateCount": 2,
                    "failures": [{"videoPath": "a.mov", "message": "invalid"}],
                },
                {},
                checkpoint,
                ["include-missing"],
                root / "library.json",
                root / "catalog.json",
                root / "avatar.json",
                root / "checkpoint.json",
            )
            self.assertFalse(ok)
            self.assertIn("include-missing", checkpoint["failures"])
            self.assertEqual(
                json.loads((root / "checkpoint.json").read_text())["failures"],
                checkpoint["failures"],
            )


if __name__ == "__main__":
    unittest.main()
