from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path


SCRIPTS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS_DIR))

import train_isign_sentence_planner as trainer  # noqa: E402


def video_for_split(split: str, start: int = 0) -> str:
    for index in range(start, start + 100_000):
        candidate = f"fixture-video-{index}"
        if trainer.assign_video_split(candidate) == split:
            return candidate
    raise AssertionError(f"Could not find a deterministic fixture ID for {split}.")


def fixture_record(split: str, sequence: int, text: str, start: int = 0) -> trainer.CaptionRecord:
    return trainer.make_caption_record(video_for_split(split, start), sequence, text)


def write_csv(path: Path, headers: list[str], rows: list[list[str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.writer(output, lineterminator="\n")
        writer.writerow(headers)
        writer.writerows(rows)


def write_fixture_sources(directory: Path) -> dict[str, str]:
    train_video = video_for_split("train")
    validation_video = video_for_split("validation")
    test_video = video_for_split("test")
    main_rows = [
        [f"{train_video}-1", "Shared exact caption."],
        [f"{train_video}-2", "Alpha beta begins here."],
        [f"{train_video}-3", "Alpha beta continues there."],
        [f"{train_video}-4", "Alpha beta closes now."],
        [f"{train_video}-5", "Gamma follows alpha, then beta."],
        [f"{validation_video}-1", "Shared exact caption."],
        [f"{validation_video}-2", "Alpha validation remains distinct."],
        [f"{test_video}-1", "SHARED   exact caption."],
        [f"{test_video}-2", "Beta test remains distinct."],
        ["lexical-task_w", "This task row is not a caption."],
        [f"{test_video}-3", ""],
    ]
    write_csv(directory / "iSign_v1.1.csv", ["uid", "text"], main_rows)
    write_csv(
        directory / "word-description-dataset_v1.1.csv",
        ["word_id", "word", "sentence_id", "sentence"],
        [
            ["private_w", "private", "private_d", "auxiliaryonly auxiliaryonly token"],
            ["private2_w", "private", "private2_d", "auxiliaryonly appears again"],
        ],
    )
    write_csv(
        directory / "word-presence-dataset_v1.1.csv",
        ["word_id", "word", "sentence_id", "sentence"],
        [
            ["private_w", "private", "private_e1", "auxiliaryonly is diagnostic"],
            ["private_w", "private", "private_e2", "auxiliaryonly stays outside training"],
        ],
    )
    (directory / "README.md").write_text(
        "license: cc-by-nc-sa-4.0\nprivate research fixture\n", encoding="utf-8"
    )
    return {
        "train": train_video,
        "validation": validation_video,
        "test": test_video,
    }


class SplitLeakageTests(unittest.TestCase):
    def test_split_assignment_is_deterministic_and_video_disjoint(self) -> None:
        records = [
            fixture_record("train", 1, "Train caption one."),
            fixture_record("train", 2, "Train caption two."),
            fixture_record("validation", 1, "Validation caption."),
            fixture_record("test", 1, "Test caption."),
        ]

        first = trainer.partition_corpus(records)
        second = trainer.partition_corpus(list(reversed(records)))

        self.assertEqual(first.retained, second.retained)
        video_sets = {
            split: {record.video_id for record in first.raw[split]}
            for split in trainer.SPLIT_ORDER
        }
        self.assertFalse(video_sets["train"] & video_sets["validation"])
        self.assertFalse(video_sets["train"] & video_sets["test"])
        self.assertFalse(video_sets["validation"] & video_sets["test"])
        self.assertEqual(len(first.raw["train"]), 2)

    def test_exact_normalized_sentences_cannot_cross_retained_splits(self) -> None:
        records = [
            fixture_record("train", 1, "A shared caption."),
            fixture_record("train", 2, "A distinct train caption."),
            fixture_record("validation", 1, "A  SHARED caption."),
            fixture_record("validation", 2, "A distinct validation caption."),
            fixture_record("test", 1, "A shared caption."),
            fixture_record("test", 2, "A distinct test caption."),
        ]

        partition = trainer.partition_corpus(records)
        text_sets = {
            split: {record.normalized_text for record in partition.retained[split]}
            for split in trainer.SPLIT_ORDER
        }

        self.assertFalse(text_sets["train"] & text_sets["validation"])
        self.assertFalse(text_sets["train"] & text_sets["test"])
        self.assertFalse(text_sets["validation"] & text_sets["test"])
        self.assertEqual(partition.cross_split_duplicates["validation"], 1)
        self.assertEqual(partition.cross_split_duplicates["test"], 1)


class ArtifactTests(unittest.TestCase):
    def test_repeated_training_is_byte_identical_and_private(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            input_dir = Path(temporary_directory)
            source_ids = write_fixture_sources(input_dir)

            first = trainer.build_artifact(input_dir)
            second = trainer.build_artifact(input_dir)
            first_bytes = trainer.canonical_json_bytes(first)
            second_bytes = trainer.canonical_json_bytes(second)
            serialized = first_bytes.decode("utf-8")

            self.assertEqual(first_bytes, second_bytes)
            self.assertFalse(first["privacy"]["rawSentencesIncluded"])
            self.assertFalse(first["privacy"]["sourceIdentifiersIncluded"])
            self.assertFalse(first["claims"]["isIslGrammarModel"])
            self.assertFalse(first["claims"]["isText2PoseModel"])
            self.assertFalse(first["distribution"]["publicHuggingFaceUpload"])
            self.assertNotIn("Shared exact caption.", serialized)
            self.assertNotIn("auxiliaryonly auxiliaryonly token", serialized)
            for source_id in source_ids.values():
                self.assertNotIn(source_id, serialized)
            self.assertEqual(
                first["training"]["leakageChecks"]["status"], "passed"
            )
            self.assertEqual(len(first["source"]["files"]), 4)
            self.assertTrue(
                all(len(row["sha256"]) == 64 for row in first["source"]["files"])
            )

    def test_boundary_phrases_never_reproduce_a_complete_short_caption(self) -> None:
        records = []
        for video_index in range(5):
            for sequence in range(4):
                records.append(
                    trainer.make_caption_record(
                        f"privacy-video-{video_index}",
                        sequence,
                        "Prefix privacy boundary phrase.",
                    )
                )
        token_counts = Counter(token for record in records for token in record.tokens)
        forbidden = {("privacy", "boundary", "phrase")}

        statistics = trainer.build_boundary_statistics(records, token_counts, forbidden)
        exported = {tuple(row["tokens"]) for row in statistics["phrases"]}

        self.assertNotIn(("privacy", "boundary", "phrase"), exported)

    def test_auxiliary_sentences_are_diagnostics_not_training_input(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            input_dir = Path(temporary_directory)
            write_fixture_sources(input_dir)

            artifact = trainer.build_artifact(input_dir)
            vocabulary = {row["token"] for row in artifact["model"]["unigrams"]}

            self.assertNotIn("auxiliaryonly", vocabulary)
            diagnostics = artifact["evaluation"]["auxiliaryCoverageDiagnostics"]
            self.assertEqual(diagnostics["wordDescription"]["sourceRowCount"], 2)
            self.assertEqual(diagnostics["wordPresence"]["sourceRowCount"], 2)

    def test_source_change_changes_artifact_without_exposing_source_text(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            input_dir = Path(temporary_directory)
            write_fixture_sources(input_dir)
            before = trainer.build_artifact(input_dir)

            readme = input_dir / "README.md"
            readme.write_text(readme.read_text(encoding="utf-8") + "changed\n", encoding="utf-8")
            after = trainer.build_artifact(input_dir)

            self.assertNotEqual(
                before["source"]["aggregateSha256"], after["source"]["aggregateSha256"]
            )
            self.assertNotEqual(
                trainer.canonical_json_bytes(before), trainer.canonical_json_bytes(after)
            )


if __name__ == "__main__":
    unittest.main()
