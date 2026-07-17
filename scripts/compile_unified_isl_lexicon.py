#!/usr/bin/env python3
"""Compile the validated unified ISL bundle into the local runtime lexicon model."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence


DEFAULT_BUNDLE = Path("output/huggingface/Robin186-ISL")
DEFAULT_OUTPUT = Path("data/models/isl-lexicon-model.json")
ISLRTC_URL = "https://divyangjan.depwd.gov.in/islrtc/"
INCLUDE_URL = "https://huggingface.co/datasets/ai4bharat/INCLUDE"


class LexiconCompileError(RuntimeError):
    pass


def iter_jsonl(path: Path) -> Iterator[Mapping[str, Any]]:
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise LexiconCompileError(f"Invalid JSONL at line {line_number}: {error}") from error
            if not isinstance(record, dict):
                raise LexiconCompileError(f"Record {line_number} is not an object.")
            yield record


def stable_entry_id(term: str) -> str:
    normalized = term.casefold()
    slug = re.sub(r"[^a-z0-9]+", "_", normalized).strip("_")[:48]
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:10]
    return f"model_{slug or 'term'}_{digest}"


def infer_category(term: str) -> str:
    normalized = term.casefold()
    categories = (
        ("technology", ("computer", "digital", "internet", "software", "technology", "ai")),
        ("education", ("college", "education", "learn", "school", "student", "teacher")),
        ("meeting", ("agenda", "conference", "date", "meeting", "schedule")),
        ("government", ("constitution", "court", "government", "law", "minister", "parliament")),
    )
    for category, markers in categories:
        if any(marker in normalized for marker in markers):
            return category
    return "general"


def compile_model(bundle_dir: Path) -> dict[str, Any]:
    metadata_path = bundle_dir / "metadata.json"
    data_path = bundle_dir / "data/isl.jsonl"
    if not metadata_path.is_file() or not data_path.is_file():
        raise LexiconCompileError(f"Validated bundle files are missing under {bundle_dir}.")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("validation", {}).get("status") != "passed":
        raise LexiconCompileError("The source bundle does not record passed validation.")
    if metadata.get("exclusions", {}).get("iSign", {}).get("status") != "excluded":
        raise LexiconCompileError("The source bundle does not enforce the iSign exclusion.")

    snapshot_date = metadata.get("build", {}).get("snapshotDate")
    if not isinstance(snapshot_date, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", snapshot_date):
        raise LexiconCompileError("The source bundle has an invalid snapshot date.")
    trained_at = f"{snapshot_date}T00:00:00.000Z"

    glossary_entries: list[dict[str, Any]] = []
    token_weights: dict[str, int] = {}
    seen_terms: set[str] = set()
    seen_ids: set[str] = set()
    has_include = False
    for position, record in enumerate(iter_jsonl(data_path), start=1):
        term = record.get("term")
        normalized_term = record.get("normalizedTerm")
        if not isinstance(term, str) or not term.strip():
            raise LexiconCompileError(f"Record {position} has no term.")
        if not isinstance(normalized_term, str) or term.casefold() != normalized_term:
            raise LexiconCompileError(f"Record {position} has an inconsistent normalized term.")
        if normalized_term in seen_terms:
            raise LexiconCompileError(f"Duplicate normalized term: {normalized_term}")
        seen_terms.add(normalized_term)

        entry_id = stable_entry_id(term)
        if entry_id in seen_ids:
            raise LexiconCompileError(f"Stable ID collision for term: {term}")
        seen_ids.add(entry_id)

        official = record.get("official")
        official_listed = isinstance(official, dict) and official.get("listed") is True
        motion = record.get("motion")
        has_include = has_include or motion is not None
        glossary_entries.append(
            {
                "aliases": [],
                "category": infer_category(term),
                "confidence": "medium" if official_listed else "low",
                "createdAt": trained_at,
                "id": entry_id,
                "islGloss": term.upper(),
                "language": "en",
                "regionalVariant": "Indian Sign Language",
                "reviewStatus": "pending_review",
                "source": "external_reference",
                "term": term,
                "updatedAt": trained_at,
            }
        )
        official_count = official.get("entryCount", 0) if isinstance(official, dict) else 0
        token_weights[term] = max(1, int(official_count) if isinstance(official_count, int) else 1)

    expected_count = metadata.get("coverage", {}).get("recordCount")
    if expected_count != len(glossary_entries):
        raise LexiconCompileError(
            f"Compiled {len(glossary_entries)} entries but bundle metadata declares {expected_count}."
        )

    citation_urls = [ISLRTC_URL]
    display_name = "ISLRTC official index"
    if has_include:
        citation_urls.append(INCLUDE_URL)
        display_name += " + INCLUDE motion metadata"

    return {
        "schemaVersion": 1,
        "metadata": {
            "id": f"isl-lexicon-{snapshot_date}",
            "version": "1.0.0",
            "status": "ready",
            "engine": "lexicon_ranker",
            "trainedAt": trained_at,
            "trainingDataset": {
                "primaryDataset": "islrtc",
                "displayName": display_name,
                "recordCount": len(glossary_entries),
                "classCount": len(glossary_entries),
                "citationUrls": citation_urls,
            },
            "notes": [
                "Deterministically compiled from the validated unified vocabulary snapshot.",
                "No gated iSign rows, raw video, raw audio, or signer media are included.",
                "A dictionary term does not claim a playable or expert-reviewed avatar motion.",
                "Terms without a local motion clip use official-source A-Z fingerspelling fallback.",
            ],
        },
        "glossaryEntries": glossary_entries,
        "tokenWeights": token_weights,
    }


def validate_model(model: Mapping[str, Any]) -> dict[str, int]:
    entries = model.get("glossaryEntries")
    if not isinstance(entries, list) or not entries:
        raise LexiconCompileError("Compiled model has no glossary entries.")
    ids = [entry.get("id") for entry in entries if isinstance(entry, dict)]
    terms = [entry.get("term") for entry in entries if isinstance(entry, dict)]
    if len(ids) != len(entries) or len(set(ids)) != len(ids):
        raise LexiconCompileError("Compiled model IDs are invalid or duplicated.")
    if len(terms) != len(entries) or len(set(term.casefold() for term in terms)) != len(terms):
        raise LexiconCompileError("Compiled model terms are invalid or duplicated.")
    if any("signAssetId" in entry for entry in entries):
        raise LexiconCompileError("Metadata-only compilation must not claim playable sign assets.")
    if "isign" in json.dumps(model, ensure_ascii=False).casefold():
        notes = model.get("metadata", {}).get("notes", [])
        if not any("no gated isign" in str(note).casefold() for note in notes):
            raise LexiconCompileError("Compiled model contains unexplained iSign provenance.")
    return {"entryCount": len(entries), "classCount": len(set(terms))}


def write_model(output_path: Path, model: Mapping[str, Any]) -> None:
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.", suffix=".tmp", dir=output_path.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as destination:
            json.dump(model, destination, ensure_ascii=False, indent=2)
            destination.write("\n")
        os.replace(temporary_path, output_path)
    finally:
        if temporary_path.exists():
            if temporary_path.parent != output_path.parent or not temporary_path.name.startswith(
                f".{output_path.name}."
            ):
                raise LexiconCompileError(f"Refusing to remove untrusted temporary path: {temporary_path}")
            temporary_path.unlink()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-dir", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        model = compile_model(args.bundle_dir.resolve())
        report = validate_model(model)
        write_model(args.output, model)
    except (LexiconCompileError, OSError, json.JSONDecodeError) as error:
        print(f"unified ISL lexicon error: {error}", file=os.sys.stderr)
        return 1
    print(json.dumps({**report, "output": str(args.output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
