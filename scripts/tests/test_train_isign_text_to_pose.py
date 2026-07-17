from __future__ import annotations

import sys
import json
import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np


SCRIPTS = Path(__file__).resolve().parents[1]
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from train_isign_text_to_pose import (  # noqa: E402
    build_vocabulary,
    create_model,
    encode_text,
    evaluate_arrays,
    load_validation_report,
    split_leakage_report,
    technical_gate,
    TrainingError,
)
from preview_isign_text_to_pose import (  # noqa: E402
    canonicalize_for_avatar,
    frame_payload,
    motion_energy,
)


class TextToPoseTrainingTests(unittest.TestCase):
    def test_vocabulary_is_built_only_from_supplied_training_text(self) -> None:
        vocabulary = build_vocabulary(["known known token"], minimum_frequency=1)
        encoded = encode_text("known heldout", vocabulary, max_tokens=3)
        self.assertNotEqual(encoded[0], vocabulary["<unk>"])
        self.assertEqual(encoded[1], vocabulary["<unk>"])
        self.assertEqual(encoded[2], vocabulary["<pad>"])

    def test_video_group_leakage_is_detected(self) -> None:
        rows = [
            {"split": "train", "video_hash": "same"},
            {"split": "val", "video_hash": "same"},
            {"split": "test", "video_hash": "different"},
        ]
        report = split_leakage_report(rows)
        self.assertFalse(report["passed"])
        self.assertEqual(report["overlaps"]["trainVal"], 1)

    def test_metrics_reward_exact_motion(self) -> None:
        target = np.zeros((2, 64, 93, 2), dtype=np.float32)
        confidence = np.ones((2, 64, 93), dtype=np.float32)
        exact = evaluate_arrays(target, target, confidence, dtw_limit=2)
        wrong = evaluate_arrays(np.ones_like(target), target, confidence, dtw_limit=2)
        self.assertEqual(exact.normalized_mpjpe, 0)
        self.assertGreater(wrong.normalized_mpjpe, exact.normalized_mpjpe)

    def test_technical_gate_never_marks_runtime_eligible(self) -> None:
        target = np.zeros((2, 64, 93, 2), dtype=np.float32)
        target[:, :, 33:75, 0] = np.linspace(0, 1.5, 64, dtype=np.float32)[None, :, None]
        confidence = np.ones((2, 64, 93), dtype=np.float32)
        good = evaluate_arrays(target, target, confidence, dtw_limit=2)
        baseline = evaluate_arrays(np.zeros_like(target), target, confidence, dtw_limit=2)
        gate = technical_gate(good, good, baseline, baseline, 10_000)
        self.assertTrue(gate["passed"])
        self.assertFalse(gate["runtimeEligible"])
        self.assertTrue(gate["testAudit"]["reportOnlyNotUsedForModelSelection"])

    @unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch is installed only in the training environment")
    def test_model_returns_finite_output_for_all_padding(self) -> None:
        import torch

        model = create_model(torch, vocabulary_size=8, hidden_size=32, layers=1, heads=4).eval()
        token_ids = torch.zeros((2, 20), dtype=torch.long)
        with torch.no_grad():
            output = model(token_ids, token_ids.ne(0))
        self.assertTrue(torch.isfinite(output).all())

    def test_validation_report_must_match_the_exact_database(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            database = root / "corpus.sqlite"
            database.write_bytes(b"validated database")
            plan = root / "plan.jsonl"
            plan.write_text("{}\n", encoding="utf-8")
            word_library = root / "word-motion.json.gz"
            word_library.write_bytes(b"validated word library")
            import hashlib

            report = root / "report.json"
            report.write_text(
                json.dumps(
                    {
                        "status": "valid",
                        "acceptedRows": 10_000,
                        "databaseSha256": hashlib.sha256(database.read_bytes()).hexdigest(),
                        "rawArchiveStored": False,
                        "fullArchiveDownloaded": False,
                        "planSha256": hashlib.sha256(plan.read_bytes()).hexdigest(),
                        "wordMotionLibrary": {
                            "status": "valid",
                            "fileSha256": hashlib.sha256(word_library.read_bytes()).hexdigest(),
                        },
                    }
                ),
                encoding="utf-8",
            )
            self.assertEqual(
                load_validation_report(
                    report, database, plan, word_library, 10_000
                )["acceptedRows"],
                10_000,
            )
            database.write_bytes(b"changed database")
            with self.assertRaises(TrainingError):
                load_validation_report(report, database, plan, word_library, 10_000)

    def test_preview_preserves_landmarks_and_detects_dynamic_hands(self) -> None:
        xy = np.zeros((64, 93, 2), dtype=np.float32)
        confidence = np.ones((64, 93), dtype=np.float32)
        xy[:, :, 0] = np.linspace(0.2, 0.8, 93, dtype=np.float32)
        xy[:, :, 1] = np.linspace(0.1, 0.9, 93, dtype=np.float32)
        xy[:, 33:75, 0] += np.linspace(0, 0.3, 64, dtype=np.float32)[:, None]

        canonical = canonicalize_for_avatar(xy, confidence)
        payload = frame_payload(canonical, confidence)

        self.assertGreater(motion_energy(canonical, confidence), 0)
        self.assertEqual(len(payload), 64)
        self.assertEqual(
            sum(len(payload[0][part]) for part in ("pose", "leftHand", "rightHand", "face")),
            93,
        )


if __name__ == "__main__":
    unittest.main()
