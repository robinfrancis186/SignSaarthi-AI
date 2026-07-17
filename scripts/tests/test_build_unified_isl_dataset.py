from __future__ import annotations

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPTS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS_DIR))

import build_unified_isl_dataset as builder  # noqa: E402


def index_page(letter: str, entries: list[tuple[str, str]]) -> builder.IndexPageSnapshot:
    anchors = "".join(
        f"<a href='search.php?type=list&id={entry_id}&search=ignored'><span>{term}</span></a>"
        for entry_id, term in entries
    )
    html = f"<html><body><a href='index.php'>Home</a>{anchors}</body></html>"
    return builder.parse_index_page(html.encode("utf-8"), letter)


def motion_catalog(label: str = "Hello", clip_id: str = "include-hello") -> dict[str, object]:
    return {
        "schemaVersion": 1,
        "clipCount": 1,
        "clips": [
            {
                "id": clip_id,
                "label": label,
                "normalizedLabel": builder.term_match_key(label),
                "datasetId": "include",
                "sourceSampleId": "include_train_fixture",
                "frameCount": 32,
                "fps": 16.0,
                "durationMs": 2000.0,
                "timingSource": "retained-frame-timestamps",
                "expertReviewed": False,
            }
        ],
    }


def lexicon(entries: list[dict[str, object]] | None = None) -> dict[str, object]:
    return {
        "schemaVersion": 1,
        "glossaryEntries": entries
        if entries is not None
        else [
            {
                "id": "model_hello",
                "term": "Hello",
                "islGloss": "HELLO",
                "category": "greeting",
                "confidence": "medium",
                "language": "en",
                "reviewStatus": "pending_review",
                "signAssetId": "include-hello",
            },
            {
                "id": "model_private_only",
                "term": "Private Only Term",
                "islGloss": "PRIVATE ONLY TERM",
                "category": "general",
                "confidence": "medium",
                "language": "en",
                "reviewStatus": "pending_review",
            },
        ],
        "tokenWeights": {"private only term": 999},
    }


def include_row(
    split: str,
    label: str,
    video_path: str,
    parent_label: str = "General",
) -> dict[str, object]:
    return {
        "split": split,
        "parent_label": parent_label,
        "label": label,
        "video_path": video_path,
        "include_50": False,
    }


def fixture_build() -> tuple[
    list[builder.IndexPageSnapshot], list[dict[str, object]], dict[str, object]
]:
    pages = [index_page("A", [("1", "Hello"), ("2", "hello"), ("3", "World")])]
    records, stats = builder.build_records(
        pages,
        lexicon(),
        motion_catalog(),
        expected_letters=("A",),
        min_official_entries=1,
    )
    return pages, records, stats


class StructuredIndexParserTests(unittest.TestCase):
    def test_extracts_only_dictionary_anchors_and_canonicalizes_urls(self) -> None:
        html = b"""
        <html><body>
          <a href="index.php">Home</a>
          <a class="entry" href="search.php?type=list&id=42&search=wrong">
            Air &amp; <strong>Water</strong>
          </a>
        </body></html>
        """

        page = builder.parse_index_page(html, "A")

        self.assertEqual(len(page.entries), 1)
        self.assertEqual(page.entries[0].entry_id, "42")
        self.assertEqual(page.entries[0].display_term, "Air & Water")
        self.assertEqual(
            page.entries[0].source_url,
            "https://divyangjan.depwd.gov.in/islrtc/search.php?type=list&id=42&search=Air%20%26%20Water",
        )

    def test_rejects_malformed_dictionary_ids(self) -> None:
        html = b"<a href='search.php?type=list&id=not-numeric&search=Bad'>Bad</a>"

        with self.assertRaisesRegex(builder.DatasetBuildError, "Malformed dictionary entry ID"):
            builder.parse_index_page(html, "A")

    def test_official_audit_rejects_duplicate_ids(self) -> None:
        page_a = index_page("A", [("7", "Alpha")])
        page_b = index_page("B", [("7", "Beta")])

        with self.assertRaisesRegex(builder.DatasetBuildError, "Duplicate official entry ID 7"):
            builder.audit_official_pages(
                [page_a, page_b], expected_letters=("A", "B"), min_entries=1
            )


class MergeBoundaryTests(unittest.TestCase):
    def test_accepts_readable_multiword_motion_normalization_but_rejects_mismatch(self) -> None:
        catalog = motion_catalog("Big/Large", "include-big-large")
        catalog["clips"][0]["normalizedLabel"] = "big large"

        validated = builder.validate_motion_catalog(catalog)

        self.assertEqual(validated[0]["normalizedLabel"], "big large")
        catalog["clips"][0]["normalizedLabel"] = "small"
        with self.assertRaisesRegex(builder.DatasetBuildError, "does not match its label"):
            builder.validate_motion_catalog(catalog)

    def test_consolidates_official_variants_and_excludes_unmatched_lexicon_rows(self) -> None:
        _, records, stats = fixture_build()
        by_term = {record["normalizedTerm"]: record for record in records}

        self.assertEqual(set(by_term), {"hello", "world"})
        self.assertEqual(by_term["hello"]["official"]["entryCount"], 2)
        self.assertEqual(
            [entry["entryId"] for entry in by_term["hello"]["official"]["entries"]],
            ["1", "2"],
        )
        self.assertEqual(by_term["hello"]["datasetIds"], ["islrtc", "include"])
        self.assertEqual(by_term["hello"]["motion"]["catalogClipId"], "include-hello")
        self.assertFalse(by_term["hello"]["motion"]["framePayloadIncluded"])
        self.assertEqual(by_term["hello"]["lexicon"]["motionLinkStatus"], "resolved")
        self.assertEqual(stats["official"]["duplicateTermGroupCount"], 1)
        self.assertEqual(stats["lexicon"]["excludedEntryCount"], 1)
        serialized = builder.canonical_json(records)
        self.assertNotIn("Private Only Term", serialized)
        self.assertNotIn("private only term", serialized)
        self.assertNotIn("tokenWeights", serialized)

    def test_explicit_isign_lexicon_row_cannot_enrich_an_official_term(self) -> None:
        pages = [index_page("A", [("1", "Hello")])]
        gated_lexicon = lexicon(
            [
                {
                    "id": "model_hello",
                    "term": "Hello",
                    "islGloss": "HELLO",
                    "datasetIds": ["isign"],
                }
            ]
        )

        records, stats = builder.build_records(
            pages,
            gated_lexicon,
            motion_catalog(),
            expected_letters=("A",),
            min_official_entries=1,
        )

        self.assertIsNone(records[0]["lexicon"])
        self.assertEqual(
            stats["lexicon"]["joinCounts"]["explicit_gated_source_excluded"], 1
        )

    def test_motion_only_include_term_is_admissible_but_lexicon_only_term_is_not(self) -> None:
        pages = [index_page("A", [("1", "Alpha")])]

        records, _ = builder.build_records(
            pages,
            lexicon(
                [
                    {"id": "model_motion", "term": "Motion Term", "islGloss": "MOTION TERM"},
                    {"id": "model_lexicon", "term": "Lexicon Only", "islGloss": "LEXICON ONLY"},
                ]
            ),
            motion_catalog("Motion Term", "include-motionterm"),
            expected_letters=("A",),
            min_official_entries=1,
        )
        by_term = {record["normalizedTerm"]: record for record in records}

        self.assertEqual(set(by_term), {"alpha", "motion term"})
        self.assertEqual(by_term["motion term"]["datasetIds"], ["include"])
        self.assertIsNotNone(by_term["motion term"]["lexicon"])

    def test_punctuation_cannot_fuzzily_attach_a_different_motion(self) -> None:
        pages = [index_page("A", [("1", "B.Ed")])]

        records, _ = builder.build_records(
            pages,
            lexicon([]),
            motion_catalog("Bed", "include-bed"),
            expected_letters=("A",),
            min_official_entries=1,
        )
        by_term = {record["normalizedTerm"]: record for record in records}

        self.assertEqual(set(by_term), {"b.ed", "bed"})
        self.assertIsNone(by_term["b.ed"]["motion"])
        self.assertEqual(by_term["bed"]["motion"]["catalogClipId"], "include-bed")

    def test_keeps_quarantined_include_label_as_vocabulary_without_motion(self) -> None:
        pages = [index_page("A", [("1", "Alpha")])]
        metadata_rows = [
            include_row("train", "1. Hello", "General/1. Hello/hello.mov"),
            include_row(
                "train",
                "2. Second (Number)",
                "Adjectives/2. quiet/quiet.mov",
            ),
            include_row(
                "val",
                "2. Second (Number)",
                "Adjectives/2. quiet/quiet.mov",
            ),
        ]

        records, stats = builder.build_records(
            pages,
            lexicon([]),
            motion_catalog(),
            include_metadata_rows=metadata_rows,
            expected_letters=("A",),
            min_official_entries=1,
        )
        by_term = {record["normalizedTerm"]: record for record in records}

        quarantined = by_term["second (number)"]
        self.assertEqual(quarantined["datasetIds"], ["include"])
        self.assertIsNone(quarantined["motion"])
        self.assertEqual(
            quarantined["include"]["motionStatus"],
            "quarantined_source_label_conflict",
        )
        self.assertEqual(stats["includeMetadata"]["quarantinedLabels"], ["Second (Number)"])
        self.assertEqual(
            stats["includeMetadata"]["splitVideoPathOverlapCounts"]["trainVal"], 1
        )


class OutputValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        _, self.records, _ = fixture_build()

    def test_rejects_isign_only_dataset_ids(self) -> None:
        malicious = copy.deepcopy(self.records[0])
        malicious["datasetIds"] = ["isign"]

        with self.assertRaisesRegex(builder.DatasetValidationError, "prohibited iSign"):
            builder.validate_records([malicious])

    def test_rejects_raw_isign_source_path(self) -> None:
        malicious = copy.deepcopy(self.records[0])
        malicious["source"] = "data/raw/isign/iSign_v1.1.csv"

        with self.assertRaisesRegex(builder.DatasetValidationError, "forbidden gated iSign"):
            builder.validate_records([malicious])

    def test_rejects_duplicate_record_ids(self) -> None:
        duplicate = copy.deepcopy(self.records[0])

        with self.assertRaisesRegex(builder.DatasetValidationError, "Duplicate output record ID"):
            builder.validate_records([self.records[0], duplicate])

    def test_rejects_duplicate_normalized_terms(self) -> None:
        duplicate = copy.deepcopy(self.records[0])
        duplicate["id"] = "different-id"

        with self.assertRaisesRegex(builder.DatasetValidationError, "Duplicate output normalized term"):
            builder.validate_records([self.records[0], duplicate])

    def test_rejects_duplicate_official_source_urls(self) -> None:
        malicious = copy.deepcopy(self.records[0])
        malicious["official"]["entries"].append(
            copy.deepcopy(malicious["official"]["entries"][0])
        )
        malicious["official"]["entryCount"] += 1

        with self.assertRaisesRegex(builder.DatasetValidationError, "Duplicate official source URL"):
            builder.validate_records([malicious])


class DeterministicBundleTests(unittest.TestCase):
    def test_writes_byte_identical_valid_bundles_for_fixed_inputs(self) -> None:
        pages, records, stats = fixture_build()
        metadata = builder.create_metadata(
            pages=pages,
            records=records,
            stats=stats,
            snapshot_date="2026-07-14",
            lexicon_sha256="a" * 64,
            motion_sha256="b" * 64,
        )

        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "first"
            second = Path(directory) / "second"
            first_report = builder.write_bundle(first, records, metadata)
            second_report = builder.write_bundle(second, records, metadata)

            self.assertEqual(first_report, second_report)
            for relative_path in ("README.md", "metadata.json", builder.OUTPUT_DATA_FILE):
                self.assertEqual(
                    (first / relative_path).read_bytes(),
                    (second / relative_path).read_bytes(),
                )
            self.assertEqual(builder.validate_bundle(first)["status"], "passed")
            generated_metadata = json.loads((first / "metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(generated_metadata["dataset"]["visibility"], "public")
            self.assertEqual(generated_metadata["exclusions"]["iSign"]["status"], "excluded")

    def test_create_and_overwrite_never_remove_parent_files(self) -> None:
        pages, records, stats = fixture_build()
        metadata = builder.create_metadata(
            pages=pages,
            records=records,
            stats=stats,
            snapshot_date="2026-07-14",
            lexicon_sha256="a" * 64,
            motion_sha256="b" * 64,
        )

        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            sentinel = parent / "workspace-sentinel.txt"
            sentinel.write_text("must survive", encoding="utf-8")
            output = parent / "bundle"

            builder.write_bundle(output, records, metadata)
            self.assertEqual(sentinel.read_text(encoding="utf-8"), "must survive")
            builder.write_bundle(output, records, metadata, overwrite=True)

            self.assertEqual(sentinel.read_text(encoding="utf-8"), "must survive")
            self.assertEqual(builder.validate_bundle(output)["status"], "passed")
            self.assertEqual(
                [path.name for path in parent.iterdir()],
                ["bundle", "workspace-sentinel.txt"],
            )

    def test_cleanup_guard_rejects_non_builder_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            unrelated = parent / "unrelated"
            unrelated.mkdir()

            with self.assertRaisesRegex(builder.DatasetBuildError, "Refusing to remove"):
                builder._remove_guarded_temporary_tree(unrelated, parent)

            self.assertTrue(unrelated.is_dir())


if __name__ == "__main__":
    unittest.main()
