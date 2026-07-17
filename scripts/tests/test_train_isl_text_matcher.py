from __future__ import annotations

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPTS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS_DIR))

import train_isl_text_matcher as trainer  # noqa: E402


GOOD_CASES = {
    "train": [
        ("applf", "apple", ("table", "chair")),
        ("bananx", "banana", ("window", "pencil")),
        ("computr", "computer", ("village", "teacher")),
        ("educaton", "education", ("hospital", "language")),
        ("famliy", "family", ("market", "garden")),
        ("goverment", "government", ("festival", "kitchen")),
        ("helth", "health", ("school", "bridge")),
        ("importnt", "important", ("weather", "station")),
    ],
    "val": [
        ("journay", "journey", ("library", "morning")),
        ("knowlege", "knowledge", ("building", "evening")),
        ("medicin", "medicine", ("district", "holiday")),
        ("natur", "nature", ("office", "street")),
    ],
    "test": [
        ("opportnity", "opportunity", ("telephone", "mountain")),
        ("populaton", "population", ("community", "railway")),
        ("queston", "question", ("relative", "uniform")),
        ("responsble", "responsible", ("vegetable", "workshop")),
    ],
}


def make_group(
    split: str,
    index: int,
    query: str,
    positive_term: str,
    negative_terms: tuple[str, str],
) -> list[dict[str, object]]:
    group_id = f"{split}-group-{index}"
    return [
        {
            "groupId": group_id,
            "query": query,
            "candidateId": f"z-{split}-{index}-positive",
            "candidateTerm": positive_term,
            "label": 1,
            "split": split,
            "sourceType": "synthetic_lexicon",
            "hasMotion": index % 2 == 0,
        },
        {
            "groupId": group_id,
            "query": query,
            "candidateId": f"a-{split}-{index}-negative",
            "candidateTerm": negative_terms[0],
            "label": 0,
            "split": split,
            "sourceType": "synthetic_lexicon",
            "hasMotion": False,
        },
        {
            "groupId": group_id,
            "query": query,
            "candidateId": f"b-{split}-{index}-negative",
            "candidateTerm": negative_terms[1],
            "label": 0,
            "split": split,
            "sourceType": "synthetic_lexicon",
            "hasMotion": False,
        },
    ]


def good_rows() -> dict[str, list[dict[str, object]]]:
    rows: dict[str, list[dict[str, object]]] = {}
    for split, cases in GOOD_CASES.items():
        rows[split] = []
        for index, (query, positive, negatives) in enumerate(cases):
            rows[split].extend(make_group(split, index, query, positive, negatives))
    return rows


def indistinguishable_rows() -> dict[str, list[dict[str, object]]]:
    rows: dict[str, list[dict[str, object]]] = {}
    for split, count in (("train", 6), ("val", 3), ("test", 3)):
        rows[split] = []
        for index in range(count):
            query = f"same{split}{index}"
            group = make_group(split, index, query, query, (query, query))
            rows[split].extend(group)
    return rows


def write_bundle(root: Path, rows: dict[str, list[dict[str, object]]]) -> None:
    data_dir = root / "data"
    data_dir.mkdir(parents=True)
    for split in trainer.SPLITS:
        payload = "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in rows[split]
        )
        path = data_dir / trainer.INPUT_FILES[split]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(payload, encoding="utf-8")


class FeatureTests(unittest.TestCase):
    def test_unicode_normalization_and_feature_order_are_stable(self) -> None:
        self.assertEqual(trainer.normalize_text("  ＣＡＦÉ—Test  "), "café test")
        self.assertEqual(
            trainer.FEATURE_NAMES,
            (
                "exact",
                "editSimilarity",
                "trigramJaccard",
                "prefixRatio",
                "lengthRatio",
                "tokenJaccard",
            ),
        )

        exact = trainer.extract_features("ＣＡＦÉ—Test", "café test")
        near = trainer.extract_features("computr", "computer")

        self.assertEqual(exact, (1.0, 1.0, 1.0, 1.0, 1.0, 1.0))
        self.assertEqual(len(near), 6)
        self.assertGreater(near[1], 0.8)
        self.assertGreater(near[2], 0.5)
        self.assertEqual(near[5], 0.0)


class ArtifactTests(unittest.TestCase):
    def test_training_is_byte_deterministic_and_passes_all_gates(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            write_bundle(root, good_rows())

            first = trainer.build_artifact(root)
            second = trainer.build_artifact(root)
            serialized = trainer.canonical_json_bytes(first).decode("utf-8")

            self.assertEqual(
                trainer.canonical_json_bytes(first), trainer.canonical_json_bytes(second)
            )
            self.assertEqual(first["metadata"]["status"], "ready")
            self.assertEqual(first["metadata"]["engine"], "lexicon_pair_matcher")
            self.assertEqual(first["featureNames"], list(trainer.FEATURE_NAMES))
            self.assertEqual(first["minimumQueryLength"], 4)
            self.assertTrue(all(first["constraints"].values()))
            self.assertEqual(first["integrity"]["status"], "passed")
            self.assertGreaterEqual(first["metrics"]["val"]["binary"]["precision"], 0.98)
            self.assertGreaterEqual(
                first["metrics"]["test"]["binary"]["precision"], 0.98
            )
            self.assertGreaterEqual(first["metrics"]["test"]["grouped"]["top1"], 0.95)
            self.assertTrue(first["metrics"]["acceptance"]["ready"])
            self.assertTrue(
                first["metrics"]["acceptance"]["checks"][
                    "beatsExactBaselineOnTestF1OrTop1"
                ]
            )
            self.assertEqual(len(first["dataHashes"]["files"]), 3)
            self.assertNotIn("opportnity", serialized)
            self.assertNotIn("opportunity", serialized)

    def test_verify_determinism_cli_writes_the_expected_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            output = root / "isl-text-matcher.json"
            write_bundle(root, good_rows())

            stdout = io.StringIO()
            stderr = io.StringIO()
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                return_code = trainer.main(
                    [
                        "--input-dir",
                        str(root),
                        "--output",
                        str(output),
                        "--verify-determinism",
                    ]
                )

            self.assertEqual(return_code, 0, stderr.getvalue())
            self.assertIn("Determinism check: passed", stdout.getvalue())
            self.assertEqual(
                output.read_bytes(),
                trainer.canonical_json_bytes(trainer.build_artifact(root)),
            )

    def test_quality_failure_cannot_receive_ready_status(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            write_bundle(root, indistinguishable_rows())

            artifact = trainer.build_artifact(root)

            self.assertEqual(artifact["metadata"]["status"], "unavailable")
            self.assertFalse(artifact["metrics"]["acceptance"]["ready"])
            self.assertIn(
                "valPrecisionAtLeast0.98",
                artifact["metrics"]["acceptance"]["failedChecks"],
            )
            self.assertIn(
                "beatsExactBaselineOnTestF1OrTop1",
                artifact["metrics"]["acceptance"]["failedChecks"],
            )


class RejectionTests(unittest.TestCase):
    def test_cross_split_normalized_query_leakage_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            rows = good_rows()
            leaked_query = rows["train"][0]["query"]
            leaked_group = rows["val"][0]["groupId"]
            for row in rows["val"]:
                if row["groupId"] == leaked_group:
                    row["query"] = leaked_query
            write_bundle(root, rows)

            with self.assertRaisesRegex(
                trainer.MatcherTrainingError, "Normalized-query leakage"
            ):
                trainer.load_dataset(root)

    def test_cross_split_group_leakage_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            rows = good_rows()
            leaked_group = rows["train"][0]["groupId"]
            original_group = rows["val"][0]["groupId"]
            for row in rows["val"]:
                if row["groupId"] == original_group:
                    row["groupId"] = leaked_group
            write_bundle(root, rows)

            with self.assertRaisesRegex(trainer.MatcherTrainingError, "Group leakage"):
                trainer.load_dataset(root)

    def test_row_split_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            rows = good_rows()
            rows["test"][0]["split"] = "val"
            write_bundle(root, rows)

            with self.assertRaisesRegex(trainer.MatcherTrainingError, "does not match"):
                trainer.load_dataset(root)


if __name__ == "__main__":
    unittest.main()
