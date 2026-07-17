from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from collections import Counter, defaultdict
from pathlib import Path


SCRIPTS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS_DIR))

import build_signsaarthi_training_dataset as builder  # noqa: E402


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(builder.canonical_json(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def train_ids(count: int) -> list[str]:
    result: list[str] = []
    index = 0
    while len(result) < count:
        candidate = f"fixture-canonical-{index:03d}"
        if builder.stable_split(candidate) == "train":
            result.append(candidate)
        index += 1
    return result


class BundleFixture:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.canonical = root / "inputs/canonical.jsonl"
        self.include_dir = root / "inputs/include"
        self.motion = root / "inputs/motion.json"
        self.paths = builder.InputPaths(
            canonical=self.canonical,
            include_dir=self.include_dir,
            motion_catalog=self.motion,
        )
        self._write_inputs()

    def _write_inputs(self) -> None:
        ids = train_ids(10)
        terms = [
            "Actor",
            "Active",
            "Adapt",
            "Adult",
            "Accept",
            "Except",
            "Access",
            "Second (Number)",
            "Cross-Country",
            "Cross Country",
        ]
        canonical_rows: list[dict[str, object]] = []
        for record_id, term in zip(ids, terms, strict=True):
            has_declared_motion = term in {"Actor", "Second (Number)"}
            motion_clip_id = (
                "include-actor" if term == "Actor" else "include-second-number"
            )
            canonical_rows.append(
                {
                    "datasetIds": ["include"] if has_declared_motion else ["islrtc"],
                    "id": record_id,
                    "language": "en",
                    "license": {
                        "licenseIds": (
                            ["include-cc-by-4.0"]
                            if has_declared_motion
                            else ["islrtc-no-profiteering-attribution"]
                        )
                    },
                    "motion": (
                        {"catalogClipId": motion_clip_id}
                        if has_declared_motion
                        else None
                    ),
                    "normalizedTerm": term.casefold(),
                    "review": {
                        "status": (
                            "motion_not_expert_reviewed"
                            if has_declared_motion
                            else "official_index_only_motion_unavailable"
                        )
                    },
                    "term": term,
                }
            )
        write_jsonl(self.canonical, canonical_rows)

        actor = {
            "parent_label": "Jobs",
            "label": "98. Actor",
            "video_path": "Jobs/98. Actor/MVI_4516.MOV",
            "include_50": False,
        }
        second_train = {
            "parent_label": "Days_and_Time",
            "label": "Second (Number)",
            "video_path": "Home/50. Tool/MVI_4453.MOV",
            "include_50": False,
        }
        second_val = {
            "parent_label": "Days_and_Time",
            "label": "Second (Number)",
            "video_path": "Animals/3. Fish/MVI_3034.MOV",
            "include_50": False,
        }
        second_test = {
            "parent_label": "Days_and_Time",
            "label": "Second (Number)",
            "video_path": "Clothes/37. Hat/MVI_5156.MOV",
            "include_50": False,
        }
        write_jsonl(self.include_dir / "train.jsonl", [actor, actor, second_train])
        write_jsonl(self.include_dir / "val.jsonl", [actor, second_val])
        write_jsonl(self.include_dir / "test.jsonl", [actor, second_test])

        motion_payload = {
            "schemaVersion": 1,
            "clipCount": 1,
            "clips": [
                {
                    "id": "include-actor",
                    "label": "Actor",
                    "normalizedLabel": "actor",
                    "datasetId": "include",
                    "sourceSampleId": builder.source_sample_id(
                        "train", str(actor["video_path"])
                    ),
                    "sourceVideoPath": actor["video_path"],
                    "sourceArchive": "Jobs.zip",
                    "sourceChecksum": "sha256:" + "a" * 64,
                    "frameCount": 32,
                    "fps": 16.0,
                    "durationMs": 2000.0,
                    "timingSource": "retained-frame-timestamps",
                    "playable": True,
                    "expertReviewed": False,
                    "reviewStatement": (
                        "Dataset-derived isolated-sign motion; not expert-certified ISL."
                    ),
                }
            ],
        }
        self.motion.parent.mkdir(parents=True, exist_ok=True)
        self.motion.write_text(
            builder.canonical_json(motion_payload, pretty=True), encoding="utf-8"
        )

    def build(self, name: str = "bundle") -> Path:
        output = self.root / name
        builder.build_bundle(
            output,
            self.paths,
            expected_counts=None,
        )
        return output


class SourcePolicyTests(unittest.TestCase):
    def test_registry_covers_every_required_source_and_status(self) -> None:
        payload = builder.source_registry_payload()
        sources = payload["sources"]

        self.assertEqual(
            {source["sourceId"] for source in sources},
            {
                "cislr",
                "data-gov-in-isl-dictionary-catalog",
                "hemg-indian-sign-language-dataset",
                "include",
                "islrtc",
                "isign",
                "isltranslate",
                "krish09bha-hindi-indian-sign-language-dataset-isl",
                "navneeth017-neo-isign-metadata-ref",
                "posestitch-isl",
                "third-party-government-reencode",
            },
        )
        self.assertEqual(
            Counter(source["status"] for source in sources),
            {"excluded": 4, "publish": 1, "reference": 5, "train": 1},
        )
        self.assertTrue(all(source["reason"] for source in sources))
        newly_audited = {
            "data-gov-in-isl-dictionary-catalog",
            "hemg-indian-sign-language-dataset",
            "krish09bha-hindi-indian-sign-language-dataset-isl",
            "navneeth017-neo-isign-metadata-ref",
        }
        self.assertTrue(
            all(
                source["contentIncluded"] is False
                for source in sources
                if source["sourceId"] in newly_audited
            )
        )
        self.assertEqual(
            builder.MOTION_QUARANTINE["reason"],
            "Every published INCLUDE metadata row for this label points to a directory "
            "for a different sign, so no source-consistent motion can be selected.",
        )


class PairGenerationTests(unittest.TestCase):
    def test_stable_split_matches_documented_hash_formula(self) -> None:
        found: dict[str, str] = {}
        index = 0
        while len(found) < 3:
            record_id = f"split-fixture-{index}"
            found.setdefault(builder.stable_split(record_id), record_id)
            index += 1

        self.assertEqual(set(found), set(builder.SPLITS))
        for split, record_id in found.items():
            bucket = int.from_bytes(
                hashlib.sha256(record_id.encode("utf-8")).digest()[:8], "big"
            ) % 100
            expected = "train" if bucket < 80 else "val" if bucket < 90 else "test"
            self.assertEqual(split, expected)

    def test_query_variants_are_collision_safe_and_capped(self) -> None:
        ambiguous = [
            {"id": "b-ed", "term": "B.Ed"},
            {"id": "bed", "term": "Bed"},
        ]
        owners = builder._query_owners(ambiguous)
        b_ed_variants = builder.query_variants(ambiguous[0], owners)

        self.assertEqual(b_ed_variants[0], ("exact", "B.Ed"))
        self.assertNotIn("punctuation", {kind for kind, _ in b_ed_variants})
        self.assertLessEqual(len(b_ed_variants), 3)

        hello = {"id": "hello", "term": "Hello"}
        hello_variants = builder.query_variants(
            hello, builder._query_owners([hello])
        )
        self.assertEqual(
            [kind for kind, _ in hello_variants], ["exact", "typo"]
        )
        self.assertEqual(
            len({builder.lexical_key(query) for _, query in hello_variants}),
            len(hello_variants),
        )

    def test_representatives_use_lowest_id_per_lexical_key(self) -> None:
        canonical = [
            {"id": "z-alias", "term": "Cross-Country"},
            {"id": "a-representative", "term": "Cross Country"},
            {"id": "middle", "term": "Different"},
        ]

        representatives, counts = builder.select_match_representatives(
            list(reversed(canonical))
        )

        self.assertEqual(
            [row["id"] for row in representatives], ["a-representative", "middle"]
        )
        self.assertEqual(
            counts,
            {
                "aliasedLexicalKeys": 1,
                "canonicalLexicalKeys": 2,
                "excludedAliasRecords": 1,
                "recordsInAliasGroups": 2,
                "representativeRecords": 2,
            },
        )

    def test_query_allocation_reserves_exact_keys_before_typos(self) -> None:
        canonical = [
            {"id": "first", "term": "Abc"},
            {"id": "second", "term": "Acb"},
        ]
        representatives, _ = builder.select_match_representatives(canonical)

        selected = builder.select_match_queries(canonical, representatives)
        normalized_groups: dict[str, str] = {}
        for record_id, queries in selected.items():
            for _, query in queries:
                normalized = builder.lexical_key(query)
                self.assertNotIn(normalized, normalized_groups)
                normalized_groups[normalized] = record_id
        self.assertEqual(selected["first"][0], ("exact", "Abc"))
        self.assertNotIn(
            "acb",
            {builder.lexical_key(query) for _, query in selected["first"]},
        )

    def test_fallback_index_matches_reference_order_without_full_sort(self) -> None:
        ids = train_ids(8)
        terms = ["Al", "A", "Aaaa", "B", "Bbb", "Ccccc", "Azz", "Ddd"]
        canonical = [
            {"id": record_id, "term": term}
            for record_id, term in zip(ids, terms, strict=True)
        ]
        index = builder.HardNegativeIndex(canonical)
        target_id = ids[0]
        target_key = builder.lexical_key(terms[0])
        expected = sorted(
            (record_id for record_id in ids if record_id != target_id),
            key=lambda candidate_id: (
                target_key[:1]
                != builder.lexical_key(
                    canonical[ids.index(candidate_id)]["term"]
                )[:1],
                abs(
                    len(target_key)
                    - len(
                        builder.lexical_key(
                            canonical[ids.index(candidate_id)]["term"]
                        )
                    )
                ),
                candidate_id,
            ),
        )

        actual = index._fallback_candidate_ids(
            target_id, "train", set(), len(expected)
        )

        self.assertEqual(actual, expected)
        self.assertEqual(
            index._fallback_candidate_ids(target_id, "train", set(), len(expected)),
            expected,
        )


class BundleTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.fixture = BundleFixture(self.root)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_build_is_byte_deterministic_and_preserves_all_rows(self) -> None:
        first = self.fixture.build("first")
        second = self.fixture.build("second")

        first_files = {
            path.relative_to(first).as_posix(): path.read_bytes()
            for path in first.rglob("*")
            if path.is_file()
        }
        second_files = {
            path.relative_to(second).as_posix(): path.read_bytes()
            for path in second.rglob("*")
            if path.is_file()
        }
        self.assertEqual(first_files, second_files)

        report = builder.validate_bundle(
            first, self.fixture.paths, expected_counts=None
        )
        self.assertEqual(report["status"], "passed")
        self.assertEqual(report["includeVocabularyLabels"], 2)
        self.assertEqual(report["motionClips"], 1)
        self.assertEqual(report["motionQuarantineRecords"], 1)
        self.assertEqual(report["matcherRepresentativeRecords"], 9)
        self.assertEqual(report["matcherExcludedAliasRecords"], 1)

        train_rows = [
            json.loads(line)
            for line in (first / builder.INCLUDE_FILES["train"]).read_text().splitlines()
        ]
        actor_rows = [row for row in train_rows if row["term"] == "Actor"]
        self.assertEqual(len(actor_rows), 2)
        self.assertEqual(len({row["id"] for row in actor_rows}), 2)
        self.assertEqual(len({row["sourceSampleId"] for row in actor_rows}), 1)

        metadata = json.loads((first / "metadata.json").read_text())
        self.assertEqual(
            metadata["sources"]["include"]["exactVideoPathOverlapCounts"],
            {"trainTest": 1, "trainVal": 1, "valTest": 1},
        )
        self.assertEqual(
            metadata["sources"]["include"]["splitRole"],
            "upstream_provenance_only_not_model_evaluation_partitions",
        )
        self.assertEqual(
            metadata["counts"]["matcherEligibility"],
            {
                "aliasedLexicalKeys": 1,
                "canonicalLexicalKeys": 9,
                "excludedAliasRecords": 1,
                "recordsInAliasGroups": 2,
                "representativeRecords": 9,
            },
        )

        canonical = [
            json.loads(line)
            for line in (first / builder.CANONICAL_FILE).read_text().splitlines()
        ]
        quarantined = [row for row in canonical if row["term"] == "Second (Number)"]
        self.assertEqual(len(quarantined), 1)
        self.assertFalse(quarantined[0]["hasMotion"])
        self.assertEqual(quarantined[0]["motionStatus"], "quarantined")
        self.assertNotIn("motionClipId", quarantined[0])

    def test_match_groups_have_one_positive_and_independent_splits(self) -> None:
        bundle = self.fixture.build()
        all_candidate_splits: defaultdict[str, set[str]] = defaultdict(set)
        all_candidate_ids: set[str] = set()
        positive_ids: set[str] = set()
        normalized_query_groups: dict[str, str] = {}
        for split in builder.SPLITS:
            groups: defaultdict[str, list[dict[str, object]]] = defaultdict(list)
            path = bundle / builder.PAIR_FILES[split]
            for line in path.read_text().splitlines():
                row = json.loads(line)
                groups[row["groupId"]].append(row)
                all_candidate_splits[row["candidateId"]].add(split)
                all_candidate_ids.add(row["candidateId"])
                self.assertEqual(builder.stable_split(row["candidateId"]), split)
            for current_group_id, rows in groups.items():
                self.assertEqual(sum(row["label"] == 1 for row in rows), 1)
                self.assertIn(sum(row["label"] == 0 for row in rows), {2, 3})
                positive_ids.add(
                    next(row["candidateId"] for row in rows if row["label"] == 1)
                )
                normalized = builder.lexical_key(rows[0]["query"])
                self.assertNotIn(normalized, normalized_query_groups)
                normalized_query_groups[normalized] = current_group_id
        self.assertTrue(all(len(splits) == 1 for splits in all_candidate_splits.values()))
        canonical = [
            json.loads(line)
            for line in (bundle / builder.CANONICAL_FILE).read_text().splitlines()
        ]
        alias_ids = sorted(
            row["id"]
            for row in canonical
            if builder.lexical_key(row["term"]) == "cross country"
        )
        self.assertEqual(len(alias_ids), 2)
        self.assertIn(alias_ids[0], positive_ids)
        self.assertNotIn(alias_ids[1], all_candidate_ids)

    def test_validator_rejects_normalized_query_reuse_across_groups(self) -> None:
        bundle = self.fixture.build()
        pair_path = bundle / builder.PAIR_FILES["train"]
        rows = [json.loads(line) for line in pair_path.read_text().splitlines()]
        group_ids = list(dict.fromkeys(row["groupId"] for row in rows))
        first_query = next(
            row["query"] for row in rows if row["groupId"] == group_ids[0]
        )
        for row in rows:
            if row["groupId"] == group_ids[1]:
                row["query"] = first_query
                row.update(builder.feature_values(first_query, row["candidateTerm"]))
        write_jsonl(pair_path, rows)

        with self.assertRaisesRegex(
            builder.TrainingDatasetValidationError, "Trainer-normalized query"
        ):
            builder.validate_bundle(bundle, self.fixture.paths, expected_counts=None)

    def test_validator_rejects_forbidden_fragment(self) -> None:
        bundle = self.fixture.build()
        readme = bundle / "README.md"
        readme.write_text(
            readme.read_text() + "\nprivate source: data/raw/isign/iSign_v1.1.csv\n",
            encoding="utf-8",
        )

        with self.assertRaisesRegex(
            builder.TrainingDatasetValidationError, "private iSign source fragment"
        ):
            builder.validate_bundle(bundle, self.fixture.paths, expected_counts=None)

    def test_validator_rejects_appledouble(self) -> None:
        bundle = self.fixture.build()
        (bundle / "._README.md").write_bytes(b"appledouble")

        with self.assertRaisesRegex(
            builder.TrainingDatasetValidationError, "AppleDouble"
        ):
            builder.validate_bundle(bundle, self.fixture.paths, expected_counts=None)

    def test_builder_scrubs_appledouble_only_from_owned_staging(self) -> None:
        staging = self.root / ".signsaarthi-training-staging-fixture"
        staging.mkdir()
        sidecar = staging / "._metadata.json"
        sidecar.write_bytes(b"appledouble")

        builder._remove_generated_appledouble(staging)

        self.assertFalse(sidecar.exists())
        with self.assertRaisesRegex(builder.TrainingDatasetError, "outside builder staging"):
            builder._remove_generated_appledouble(self.root)

    def test_validator_rejects_a_second_positive(self) -> None:
        bundle = self.fixture.build()
        pair_path = bundle / builder.PAIR_FILES["train"]
        rows = [json.loads(line) for line in pair_path.read_text().splitlines()]
        first_group = rows[0]["groupId"]
        changed = False
        for row in rows:
            if row["groupId"] == first_group and row["label"] == 0:
                row["label"] = 1
                changed = True
                break
        self.assertTrue(changed)
        rows.sort(key=builder._pair_sort_key)
        write_jsonl(pair_path, rows)

        with self.assertRaisesRegex(
            builder.TrainingDatasetValidationError, "exactly one positive"
        ):
            builder.validate_bundle(bundle, self.fixture.paths, expected_counts=None)

    def test_validator_rejects_missing_canonical_input_row(self) -> None:
        bundle = self.fixture.build()
        canonical_path = bundle / builder.CANONICAL_FILE
        rows = [json.loads(line) for line in canonical_path.read_text().splitlines()]
        write_jsonl(canonical_path, rows[:-1])

        with self.assertRaisesRegex(
            builder.TrainingDatasetValidationError, "every expected input row"
        ):
            builder.validate_bundle(bundle, self.fixture.paths, expected_counts=None)

    def test_overwrite_refuses_an_unexpected_existing_layout(self) -> None:
        bundle = self.fixture.build()
        (bundle / "unexpected.txt").write_text("do not delete", encoding="utf-8")

        with self.assertRaisesRegex(
            builder.TrainingDatasetValidationError, "unexpected.txt"
        ):
            builder.build_bundle(
                bundle,
                self.fixture.paths,
                overwrite=True,
                expected_counts=None,
            )
        self.assertEqual((bundle / "unexpected.txt").read_text(), "do not delete")


if __name__ == "__main__":
    unittest.main()
