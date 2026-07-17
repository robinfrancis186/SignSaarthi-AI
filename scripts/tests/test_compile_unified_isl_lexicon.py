from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPTS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS_DIR))

import compile_unified_isl_lexicon as compiler  # noqa: E402


class UnifiedLexiconCompilerTests(unittest.TestCase):
    def test_compiles_only_metadata_backed_terms_without_fake_assets(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle = root / "bundle"
            (bundle / "data").mkdir(parents=True)
            (bundle / "metadata.json").write_text(
                json.dumps(
                    {
                        "build": {"snapshotDate": "2026-07-14"},
                        "coverage": {"recordCount": 2},
                        "validation": {"status": "passed"},
                        "exclusions": {"iSign": {"status": "excluded"}},
                    }
                ),
                encoding="utf-8",
            )
            records = [
                {
                    "term": "Access",
                    "normalizedTerm": "access",
                    "official": {"listed": True, "entryCount": 2},
                    "motion": None,
                },
                {
                    "term": "Bed",
                    "normalizedTerm": "bed",
                    "official": {"listed": False, "entryCount": 0},
                    "motion": {"catalogClipId": "include-bed", "framePayloadIncluded": False},
                },
            ]
            (bundle / "data/isl.jsonl").write_text(
                "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
            )

            model = compiler.compile_model(bundle)
            report = compiler.validate_model(model)
            output = root / "isl-lexicon-model.json"
            sentinel = root / "sentinel.txt"
            sentinel.write_text("keep", encoding="utf-8")
            compiler.write_model(output, model)

            self.assertEqual(report, {"entryCount": 2, "classCount": 2})
            self.assertEqual(sentinel.read_text(encoding="utf-8"), "keep")
            self.assertFalse(any("signAssetId" in entry for entry in model["glossaryEntries"]))
            self.assertEqual(model["metadata"]["trainingDataset"]["primaryDataset"], "islrtc")

    def test_rejects_count_drift(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            bundle = Path(directory)
            (bundle / "data").mkdir()
            (bundle / "metadata.json").write_text(
                json.dumps(
                    {
                        "build": {"snapshotDate": "2026-07-14"},
                        "coverage": {"recordCount": 2},
                        "validation": {"status": "passed"},
                        "exclusions": {"iSign": {"status": "excluded"}},
                    }
                ),
                encoding="utf-8",
            )
            (bundle / "data/isl.jsonl").write_text(
                json.dumps(
                    {
                        "term": "Access",
                        "normalizedTerm": "access",
                        "official": {"listed": True, "entryCount": 1},
                        "motion": None,
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(compiler.LexiconCompileError, "declares 2"):
                compiler.compile_model(bundle)


if __name__ == "__main__":
    unittest.main()
