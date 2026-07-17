#!/usr/bin/env python3
"""Build and validate the private SignSaarthi lexical-pair training bundle.

The pipeline is deterministic for fixed local inputs and performs no network
requests. It emits compact canonical, INCLUDE, and motion metadata plus
split-isolated match pairs. Raw media, landmark payloads, signer information,
iSign-derived content, credentials, and absolute local paths are excluded.
"""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import os
import re
import shutil
import sys
import tempfile
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence


SCHEMA_VERSION = 1
BUILDER_VERSION = "1.0.0"
DATASET_REPOSITORY = "Robin186/SignSaarthi-ISL-Training"
DATASET_VISIBILITY = "private"
SPLITS = ("train", "val", "test")
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CANONICAL_PATH = REPO_ROOT / "output/huggingface/Robin186-ISL/data/isl.jsonl"
DEFAULT_INCLUDE_DIR = REPO_ROOT / "data/isl/include-metadata"
DEFAULT_MOTION_PATH = REPO_ROOT / "data/models/isl-motion-catalog.json"
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT / "output/huggingface/Robin186-SignSaarthi-ISL-Training"
)

PRODUCTION_COUNTS: Mapping[str, Any] = {
    "canonical": 12_434,
    "include": {"train": 3_816, "val": 425, "test": 1_009},
    "includePathOverlaps": {"trainVal": 120, "trainTest": 277, "valTest": 27},
    "includeVocabulary": 263,
    "motion": 262,
    "motionQuarantine": 1,
    "matcherEligibility": {
        "aliasedLexicalKeys": 12,
        "canonicalLexicalKeys": 12_422,
        "excludedAliasRecords": 12,
        "recordsInAliasGroups": 24,
        "representativeRecords": 12_422,
    },
}

CANONICAL_FILE = "data/canonical.jsonl"
INCLUDE_VOCABULARY_FILE = "data/include_vocabulary.jsonl"
MOTION_FILE = "data/motion_catalog.jsonl"
MOTION_QUARANTINE_FILE = "data/motion_quarantine.jsonl"
REGISTRY_FILE = "source_registry.json"
INCLUDE_FILES = {
    split: f"data/include_metadata/{split}.jsonl" for split in SPLITS
}
PAIR_FILES = {split: f"data/match_pairs/{split}.jsonl" for split in SPLITS}
EXPECTED_FILE_PATHS = {
    "README.md",
    "metadata.json",
    REGISTRY_FILE,
    CANONICAL_FILE,
    INCLUDE_VOCABULARY_FILE,
    MOTION_FILE,
    MOTION_QUARANTINE_FILE,
    *INCLUDE_FILES.values(),
    *PAIR_FILES.values(),
}
EXPECTED_DIRECTORY_PATHS = {"data", "data/include_metadata", "data/match_pairs"}

CANONICAL_FIELDS = {
    "datasetIds",
    "hasMotion",
    "id",
    "language",
    "licenseIds",
    "normalizedTerm",
    "motionStatus",
    "reviewStatus",
    "term",
}
CANONICAL_MOTION_FIELDS = CANONICAL_FIELDS | {"motionClipId"}
INCLUDE_FIELDS = {
    "id",
    "include50",
    "label",
    "parentLabel",
    "sourceRelativePath",
    "sourceSampleId",
    "split",
    "term",
}
MOTION_FIELDS = {
    "clipId",
    "datasetId",
    "durationMs",
    "expertReviewed",
    "fps",
    "frameCount",
    "normalizedTerm",
    "playable",
    "reviewStatement",
    "sourceChecksum",
    "sourceSampleId",
    "term",
    "timingSource",
}
INCLUDE_VOCABULARY_FIELDS = {
    "datasetId",
    "hasPlayableMotion",
    "motionStatus",
    "normalizedTerm",
    "term",
    "vocabularyId",
}
INCLUDE_VOCABULARY_MOTION_FIELDS = INCLUDE_VOCABULARY_FIELDS | {"motionClipId"}
INCLUDE_VOCABULARY_QUARANTINE_FIELDS = INCLUDE_VOCABULARY_FIELDS | {
    "quarantineId"
}
MOTION_QUARANTINE_FIELDS = {
    "allSourcePathsConflict",
    "datasetId",
    "motionMetadataIncluded",
    "normalizedTerm",
    "quarantineId",
    "reason",
    "sourceRowCounts",
    "status",
    "term",
}
PAIR_FIELDS = {
    "candidateId",
    "candidateTerm",
    "charSimilarity",
    "exactNormalizedMatch",
    "groupId",
    "hasMotion",
    "label",
    "lengthDelta",
    "query",
    "sourceType",
    "split",
    "tokenJaccard",
}
SOURCE_TYPES = ("exact", "punctuation", "typo")

MOTION_QUARANTINE = {
    "datasetId": "include",
    "normalizedTerm": "second number",
    "quarantineId": "include-second-number",
    "reason": (
        "Every published INCLUDE metadata row for this label points to a directory "
        "for a different sign, so no source-consistent motion can be selected."
    ),
    "status": "quarantined",
    "term": "Second (Number)",
}

STATUS_DEFINITIONS = {
    "excluded": "No source content or derived values may enter the bundle.",
    "publish": (
        "Compact metadata is included and may be published only under the stated "
        "source terms."
    ),
    "reference": (
        "Citation and policy metadata only; no source rows or derived values are included."
    ),
    "train": (
        "Included only in this private training bundle; this registry does not approve "
        "public redistribution."
    ),
}

SOURCE_REGISTRY: tuple[Mapping[str, Any], ...] = (
    {
        "contentIncluded": False,
        "license": "AFL-3.0 declared by the source dataset card",
        "name": "CISLR",
        "reason": (
            "No CISLR snapshot is an approved input to this build; keep it citation-only "
            "until provenance and redistribution are reviewed for this exact training use."
        ),
        "sourceId": "cislr",
        "sourceUrl": "https://huggingface.co/datasets/Exploration-Lab/CISLR",
        "status": "reference",
    },
    {
        "contentIncluded": False,
        "license": "No license declared by the source dataset card",
        "name": "Hemg/Indian_sign_language_dataset",
        "reason": (
            "The image dataset declares no license, so no redistribution or training "
            "rights are assumed; no rows, images, labels, or derived values are admitted."
        ),
        "sourceId": "hemg-indian-sign-language-dataset",
        "sourceUrl": (
            "https://huggingface.co/datasets/Hemg/Indian_sign_language_dataset"
        ),
        "status": "excluded",
    },
    {
        "contentIncluded": True,
        "license": "CC BY 4.0",
        "name": "INCLUDE",
        "reason": (
            "All 5,250 source rows and 263 vocabulary labels are retained as provenance; "
            "262 validated clip records are admitted for private training, one conflicting "
            "label is quarantined, and raw media and landmark frames remain excluded."
        ),
        "sourceId": "include",
        "sourceUrl": "https://huggingface.co/datasets/ai4bharat/INCLUDE",
        "status": "train",
    },
    {
        "contentIncluded": True,
        "license": (
            "ISLRTC dictionary policy: research, teaching, and ISL technology use; "
            "no resale or profiteering; attribution required"
        ),
        "name": "ISLRTC",
        "reason": (
            "The validated compact canonical metadata snapshot may be published with "
            "acknowledgement under the official use conditions; dictionary media is not "
            "redistributed."
        ),
        "sourceId": "islrtc",
        "sourceUrl": "https://islrtc.nic.in/faq/",
        "status": "publish",
    },
    {
        "contentIncluded": False,
        "license": "CC BY-NC 4.0 declared by the source repository",
        "name": "ISLTranslate",
        "reason": (
            "No ISLTranslate snapshot is an input and its declared terms are "
            "non-commercial; retain only the citation until a separate approved use exists."
        ),
        "sourceId": "isltranslate",
        "sourceUrl": "https://github.com/Exploration-Lab/ISLTranslate",
        "status": "reference",
    },
    {
        "contentIncluded": False,
        "license": "CC BY-NC-SA 4.0 plus gated dataset terms",
        "name": "iSign",
        "reason": (
            "Gated terms restrict use to research and prohibit dataset re-upload; no source "
            "text, derived statistics, media, poses, identifiers, or private artifacts may "
            "enter this bundle."
        ),
        "sourceId": "isign",
        "sourceUrl": "https://huggingface.co/datasets/Exploration-Lab/iSign",
        "status": "excluded",
    },
    {
        "contentIncluded": False,
        "license": "Apache-2.0 tag; repository audited as empty and README-only",
        "name": "KRISH09bha/Hindi-Indian-Sign-language-dataset-ISL",
        "reason": (
            "The repository is empty and README-only at audit time, so it contributes "
            "citation metadata only; no rows or externally described media are admitted."
        ),
        "sourceId": "krish09bha-hindi-indian-sign-language-dataset-isl",
        "sourceUrl": (
            "https://huggingface.co/datasets/KRISH09bha/"
            "Hindi-Indian-Sign-language-dataset-ISL"
        ),
        "status": "reference",
    },
    {
        "contentIncluded": False,
        "license": "No independent rights; metadata declares gated iSign as its source",
        "name": "Navneeth017/neo_isign_metadata_ref",
        "reason": (
            "The repository is derived from gated iSign metadata and supplies no independent "
            "redistribution grant; no rows, statistics, text, or derived values are admitted."
        ),
        "sourceId": "navneeth017-neo-isign-metadata-ref",
        "sourceUrl": (
            "https://huggingface.co/datasets/Navneeth017/neo_isign_metadata_ref"
        ),
        "status": "excluded",
    },
    {
        "contentIncluded": False,
        "license": "CC BY-NC-ND 4.0 declared by the source dataset card",
        "name": "PoseStitch-ISL",
        "reason": (
            "The source is non-commercial and no-derivatives, and no approved snapshot is an "
            "input; keep only the citation and admit no rows or derived values."
        ),
        "sourceId": "posestitch-isl",
        "sourceUrl": "https://huggingface.co/datasets/Exploration-Lab/PoseStitch-ISL",
        "status": "reference",
    },
    {
        "contentIncluded": False,
        "license": "National Data Sharing and Accessibility Policy (NDSAP)",
        "name": "Open Government Data Indian Sign Language Dictionary catalog",
        "reason": (
            "The official data.gov.in catalog is retained as a provenance reference only; "
            "no catalog media, downloads, rows, or derived values are imported."
        ),
        "sourceId": "data-gov-in-isl-dictionary-catalog",
        "sourceUrl": "https://www.data.gov.in/catalog/indian-sign-language-dictionary",
        "status": "reference",
    },
    {
        "contentIncluded": False,
        "license": (
            "Third-party MIT claim; upstream ISLRTC/data.gov.in conditions still apply"
        ),
        "name": "Third-party government dictionary re-encode",
        "reason": (
            "A third-party license label cannot supersede the upstream government source "
            "conditions, and the re-encode contains raw human video; no bytes or metadata "
            "are admitted."
        ),
        "sourceId": "third-party-government-reencode",
        "sourceUrl": (
            "https://huggingface.co/datasets/silentone0725/"
            "Indian_Sign_Language_Data.gov_Rencoded"
        ),
        "status": "excluded",
    },
)

FORBIDDEN_TEXT_PATTERNS = (
    (
        "absolute macOS local path",
        re.compile(r"/(?:Users|Volumes)/", re.IGNORECASE),
    ),
    (
        "absolute Linux home path",
        re.compile(r"/home/[A-Za-z0-9._-]+/"),
    ),
    (
        "absolute Windows local path",
        re.compile(r"(?:^|[^A-Za-z])[A-Za-z]:\\\\"),
    ),
    (
        "private iSign source fragment",
        re.compile(
            r"data/raw/isign|iSign_v1[.]1[.]csv|word-description-dataset_v1[.]1[.]csv|"
            r"word-presence-dataset_v1[.]1[.]csv|isl-sentence-planner[.]private[.]json",
            re.IGNORECASE,
        ),
    ),
    (
        "landmark or signer payload field",
        re.compile(
            r'"(?:frames|landmarks|signer|signerHash|signerId|signerInfo)"\s*:',
            re.IGNORECASE,
        ),
    ),
    (
        "credential field",
        re.compile(
            r'"(?:apiKey|api_key|password|secret|token)"\s*:', re.IGNORECASE
        ),
    ),
    (
        "Hugging Face token",
        re.compile(r"\bhf_[A-Za-z0-9]{20,}\b"),
    ),
    (
        "authorization bearer value",
        re.compile(r"authorization\s*[:=]\s*bearer\s+[A-Za-z0-9._-]{16,}", re.IGNORECASE),
    ),
    (
        "private key",
        re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    ),
)

SEMANTIC_STOPWORDS = {
    "and",
    "for",
    "from",
    "into",
    "off",
    "sign",
    "the",
    "this",
    "that",
    "with",
}


class TrainingDatasetError(RuntimeError):
    """Raised when an input or generated training bundle is invalid."""


class TrainingDatasetValidationError(TrainingDatasetError):
    """Raised when independent bundle validation fails."""


@dataclass(frozen=True)
class InputPaths:
    canonical: Path = DEFAULT_CANONICAL_PATH
    include_dir: Path = DEFAULT_INCLUDE_DIR
    motion_catalog: Path = DEFAULT_MOTION_PATH

    def include(self, split: str) -> Path:
        return self.include_dir / f"{split}.jsonl"


def canonical_json(value: Any, *, pretty: bool = False) -> str:
    if pretty:
        return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def clean_text(value: Any, field: str) -> str:
    if not isinstance(value, str):
        raise TrainingDatasetError(f"{field} must be a string.")
    normalized = unicodedata.normalize("NFKC", value)
    normalized = normalized.replace("\u200b", "").replace("\ufeff", "")
    cleaned = " ".join(normalized.split())
    if not cleaned:
        raise TrainingDatasetError(f"{field} must not be empty.")
    if any(unicodedata.category(character) in {"Cc", "Cs"} for character in cleaned):
        raise TrainingDatasetError(f"{field} contains unsupported control characters.")
    return cleaned


def lexical_key(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return " ".join(
        "".join(character if character.isalnum() else " " for character in normalized).split()
    )


def collision_key(value: str) -> str:
    return "".join(character for character in lexical_key(value) if character.isalnum())


def normalize_include_label(value: str) -> str:
    normalized = re.sub(r"^\s*\d+\s*[.)-]\s*", "", clean_text(value, "INCLUDE label"))
    normalized = normalized.replace("&", " and ")
    normalized = re.sub(r"[^a-zA-Z0-9]+", " ", normalized)
    return " ".join(normalized.casefold().split())


def stable_split(record_id: str) -> str:
    digest = hashlib.sha256(record_id.encode("utf-8")).digest()
    bucket = int.from_bytes(digest[:8], "big") % 100
    if bucket < 80:
        return "train"
    if bucket < 90:
        return "val"
    return "test"


def source_sample_id(split: str, source_relative_path: str) -> str:
    digest = hashlib.sha256(source_relative_path.encode("utf-8")).hexdigest()[:16]
    return f"include_{split}_{digest}"


def source_registry_payload() -> dict[str, Any]:
    return {
        "schemaVersion": SCHEMA_VERSION,
        "statusDefinitions": STATUS_DEFINITIONS,
        "sources": sorted(SOURCE_REGISTRY, key=lambda source: str(source["sourceId"])),
    }


def _load_json(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise TrainingDatasetError(f"Missing input: {path}") from error
    except json.JSONDecodeError as error:
        raise TrainingDatasetError(f"Invalid JSON in {path}: {error}") from error
    if not isinstance(value, dict):
        raise TrainingDatasetError(f"Expected a JSON object in {path}.")
    return value


def _load_source_jsonl(path: Path) -> list[Mapping[str, Any]]:
    rows: list[Mapping[str, Any]] = []
    try:
        source = path.open("r", encoding="utf-8", newline="")
    except FileNotFoundError as error:
        raise TrainingDatasetError(f"Missing input: {path}") from error
    with source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                raise TrainingDatasetError(f"Blank JSONL line at {path}:{line_number}.")
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise TrainingDatasetError(
                    f"Invalid JSONL at {path}:{line_number}: {error}"
                ) from error
            if not isinstance(value, dict):
                raise TrainingDatasetError(
                    f"JSONL row at {path}:{line_number} must be an object."
                )
            rows.append(value)
    return rows


def transform_canonical_rows(path: Path) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    seen_terms: set[str] = set()
    for line_number, source in enumerate(_load_source_jsonl(path), start=1):
        record_id = clean_text(source.get("id"), f"canonical row {line_number} id")
        term = clean_text(source.get("term"), f"canonical row {line_number} term")
        normalized_term = clean_text(
            source.get("normalizedTerm"), f"canonical row {line_number} normalizedTerm"
        )
        if normalized_term != term.casefold():
            raise TrainingDatasetError(
                f"Canonical row {record_id} has an inconsistent normalizedTerm."
            )
        if record_id in seen_ids or normalized_term in seen_terms:
            raise TrainingDatasetError(f"Duplicate canonical ID or term at {record_id}.")
        seen_ids.add(record_id)
        seen_terms.add(normalized_term)

        dataset_ids = source.get("datasetIds")
        if not isinstance(dataset_ids, list) or not dataset_ids:
            raise TrainingDatasetError(f"Canonical row {record_id} has invalid datasetIds.")
        cleaned_dataset_ids = sorted(
            {clean_text(value, f"canonical row {record_id} datasetId") for value in dataset_ids}
        )
        if not set(cleaned_dataset_ids).issubset({"include", "islrtc"}):
            raise TrainingDatasetError(f"Canonical row {record_id} has a forbidden datasetId.")

        license_value = source.get("license")
        if not isinstance(license_value, dict) or not isinstance(
            license_value.get("licenseIds"), list
        ):
            raise TrainingDatasetError(f"Canonical row {record_id} has invalid license data.")
        license_ids = sorted(
            clean_text(value, f"canonical row {record_id} licenseId")
            for value in license_value["licenseIds"]
        )
        review = source.get("review")
        if not isinstance(review, dict):
            raise TrainingDatasetError(f"Canonical row {record_id} has invalid review data.")
        review_status = clean_text(
            review.get("status"), f"canonical row {record_id} review status"
        )

        motion = source.get("motion")
        has_motion = motion is not None
        if has_motion and not isinstance(motion, dict):
            raise TrainingDatasetError(f"Canonical row {record_id} has invalid motion data.")
        record: dict[str, Any] = {
            "datasetIds": cleaned_dataset_ids,
            "hasMotion": has_motion,
            "id": record_id,
            "language": clean_text(
                source.get("language"), f"canonical row {record_id} language"
            ),
            "licenseIds": license_ids,
            "motionStatus": "source_declared" if has_motion else "unavailable",
            "normalizedTerm": normalized_term,
            "reviewStatus": review_status,
            "term": term,
        }
        if has_motion:
            assert isinstance(motion, dict)
            record["motionClipId"] = clean_text(
                motion.get("catalogClipId"), f"canonical row {record_id} motion clip ID"
            )
        output.append(record)
    return sorted(output, key=lambda record: str(record["id"]))


def _safe_relative_source_path(value: Any, field: str) -> str:
    path_text = clean_text(value, field)
    if "\\" in path_text:
        raise TrainingDatasetError(f"{field} must use POSIX separators.")
    path = PurePosixPath(path_text)
    if path.is_absolute() or ".." in path.parts or path_text.startswith("~"):
        raise TrainingDatasetError(f"{field} must be a safe relative path.")
    return path.as_posix()


def transform_include_rows(path: Path, split: str) -> list[dict[str, Any]]:
    if split not in SPLITS:
        raise TrainingDatasetError(f"Unknown INCLUDE split: {split}")
    output: list[dict[str, Any]] = []
    occurrences: Counter[str] = Counter()
    for line_number, source in enumerate(_load_source_jsonl(path), start=1):
        if set(source) != {"parent_label", "label", "video_path", "include_50"}:
            raise TrainingDatasetError(
                f"INCLUDE {split} row {line_number} has unexpected fields."
            )
        parent_label = clean_text(
            source.get("parent_label"), f"INCLUDE {split} row {line_number} parent_label"
        )
        label = clean_text(source.get("label"), f"INCLUDE {split} row {line_number} label")
        source_path = _safe_relative_source_path(
            source.get("video_path"), f"INCLUDE {split} row {line_number} video_path"
        )
        include_50 = source.get("include_50")
        if type(include_50) is not bool:
            raise TrainingDatasetError(
                f"INCLUDE {split} row {line_number} include_50 must be boolean."
            )
        term = clean_text(
            re.sub(r"^\s*\d+[.]\s*", "", label),
            f"INCLUDE {split} row {line_number} term",
        )
        source_fingerprint = canonical_json(source)
        digest = hashlib.sha256(source_fingerprint.encode("utf-8")).hexdigest()[:16]
        occurrences[source_fingerprint] += 1
        row_id = f"include-{split}-{digest}-{occurrences[source_fingerprint]:02d}"
        output.append(
            {
                "id": row_id,
                "include50": include_50,
                "label": label,
                "parentLabel": parent_label,
                "sourceRelativePath": source_path,
                "sourceSampleId": source_sample_id(split, source_path),
                "split": split,
                "term": term,
            }
        )
    sorted_rows = sorted(output, key=lambda row: str(row["id"]))
    if len({str(row["id"]) for row in sorted_rows}) != len(sorted_rows):
        raise TrainingDatasetError(f"INCLUDE {split} output IDs are not unique.")
    return sorted_rows


def _require_number(value: Any, field: str, *, positive: bool = False) -> int | float:
    if type(value) not in {int, float}:
        raise TrainingDatasetError(f"{field} must be numeric.")
    if positive and value <= 0:
        raise TrainingDatasetError(f"{field} must be positive.")
    return value


def transform_motion_catalog(path: Path) -> list[dict[str, Any]]:
    payload = _load_json(path)
    clips = payload.get("clips")
    if not isinstance(clips, list) or payload.get("clipCount") != len(clips):
        raise TrainingDatasetError("Motion catalog clipCount does not match clips.")
    output: list[dict[str, Any]] = []
    for index, source in enumerate(clips, start=1):
        if not isinstance(source, dict):
            raise TrainingDatasetError(f"Motion clip {index} must be an object.")
        clip_id = clean_text(source.get("id"), f"motion clip {index} id")
        dataset_id = clean_text(source.get("datasetId"), f"motion clip {clip_id} datasetId")
        if dataset_id != "include":
            raise TrainingDatasetError(f"Motion clip {clip_id} is not INCLUDE-backed.")
        playable = source.get("playable")
        expert_reviewed = source.get("expertReviewed")
        if playable is not True or expert_reviewed is not False:
            raise TrainingDatasetError(
                f"Motion clip {clip_id} must be playable and not expert-reviewed."
            )
        source_checksum = clean_text(
            source.get("sourceChecksum"), f"motion clip {clip_id} sourceChecksum"
        )
        if not re.fullmatch(r"sha256:[0-9a-f]{64}", source_checksum):
            raise TrainingDatasetError(f"Motion clip {clip_id} has an invalid source checksum.")
        output.append(
            {
                "clipId": clip_id,
                "datasetId": dataset_id,
                "durationMs": _require_number(
                    source.get("durationMs"), f"motion clip {clip_id} durationMs", positive=True
                ),
                "expertReviewed": False,
                "fps": _require_number(
                    source.get("fps"), f"motion clip {clip_id} fps", positive=True
                ),
                "frameCount": _require_number(
                    source.get("frameCount"), f"motion clip {clip_id} frameCount", positive=True
                ),
                "normalizedTerm": clean_text(
                    source.get("normalizedLabel"),
                    f"motion clip {clip_id} normalizedLabel",
                ),
                "playable": True,
                "reviewStatement": clean_text(
                    source.get("reviewStatement"),
                    f"motion clip {clip_id} reviewStatement",
                ),
                "sourceChecksum": source_checksum,
                "sourceSampleId": clean_text(
                    source.get("sourceSampleId"),
                    f"motion clip {clip_id} sourceSampleId",
                ),
                "term": clean_text(source.get("label"), f"motion clip {clip_id} label"),
                "timingSource": clean_text(
                    source.get("timingSource"), f"motion clip {clip_id} timingSource"
                ),
            }
        )
    output.sort(key=lambda clip: str(clip["clipId"]))
    if len({str(clip["clipId"]) for clip in output}) != len(output):
        raise TrainingDatasetError("Motion catalog contains duplicate clip IDs.")
    return output


def build_motion_quarantine(
    include_rows: Mapping[str, Sequence[Mapping[str, Any]]],
) -> list[dict[str, Any]]:
    normalized_term = str(MOTION_QUARANTINE["normalizedTerm"])
    source_row_counts: dict[str, int] = {}
    for split in SPLITS:
        matching_rows = [
            row
            for row in include_rows[split]
            if normalize_include_label(str(row["label"])) == normalized_term
        ]
        source_row_counts[split] = len(matching_rows)
        for row in matching_rows:
            directory_label = PurePosixPath(str(row["sourceRelativePath"])).parent.name
            if normalize_include_label(directory_label) == normalized_term:
                raise TrainingDatasetError(
                    "The quarantined INCLUDE label now has a source-consistent path; "
                    "review the quarantine before rebuilding."
                )
    if sum(source_row_counts.values()) == 0:
        raise TrainingDatasetError("The quarantined INCLUDE label is absent from metadata.")
    return [
        {
            "allSourcePathsConflict": True,
            "datasetId": MOTION_QUARANTINE["datasetId"],
            "motionMetadataIncluded": False,
            "normalizedTerm": normalized_term,
            "quarantineId": MOTION_QUARANTINE["quarantineId"],
            "reason": MOTION_QUARANTINE["reason"],
            "sourceRowCounts": source_row_counts,
            "status": MOTION_QUARANTINE["status"],
            "term": MOTION_QUARANTINE["term"],
        }
    ]


def build_include_vocabulary(
    include_rows: Mapping[str, Sequence[Mapping[str, Any]]],
    motion_rows: Sequence[Mapping[str, Any]],
    quarantine_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    train_labels = {
        normalize_include_label(str(row["label"])) for row in include_rows["train"]
    }
    motion_by_term = {str(row["normalizedTerm"]): row for row in motion_rows}
    quarantine_by_term = {
        str(row["normalizedTerm"]): row for row in quarantine_rows
    }
    if set(motion_by_term) & set(quarantine_by_term):
        raise TrainingDatasetError("A quarantined label also has playable motion metadata.")
    if train_labels != set(motion_by_term) | set(quarantine_by_term):
        missing = sorted(train_labels - set(motion_by_term) - set(quarantine_by_term))
        unexpected = sorted((set(motion_by_term) | set(quarantine_by_term)) - train_labels)
        raise TrainingDatasetError(
            "INCLUDE vocabulary does not match playable plus quarantined labels "
            f"(missing={missing}; unexpected={unexpected})."
        )

    output: list[dict[str, Any]] = []
    for normalized_term in sorted(train_labels):
        vocabulary_id = f"include-{normalized_term.replace(' ', '-')}"
        if normalized_term in motion_by_term:
            motion = motion_by_term[normalized_term]
            output.append(
                {
                    "datasetId": "include",
                    "hasPlayableMotion": True,
                    "motionClipId": motion["clipId"],
                    "motionStatus": "playable",
                    "normalizedTerm": normalized_term,
                    "term": motion["term"],
                    "vocabularyId": vocabulary_id,
                }
            )
        else:
            quarantine = quarantine_by_term[normalized_term]
            output.append(
                {
                    "datasetId": "include",
                    "hasPlayableMotion": False,
                    "motionStatus": "quarantined",
                    "normalizedTerm": normalized_term,
                    "quarantineId": quarantine["quarantineId"],
                    "term": quarantine["term"],
                    "vocabularyId": vocabulary_id,
                }
            )
    if len({str(row["vocabularyId"]) for row in output}) != len(output):
        raise TrainingDatasetError("INCLUDE vocabulary IDs are not unique.")
    return output


def reconcile_canonical_motion(
    canonical_rows: Sequence[dict[str, Any]],
    motion_rows: Sequence[Mapping[str, Any]],
    quarantine_rows: Sequence[Mapping[str, Any]],
) -> None:
    playable_ids = {str(row["clipId"]) for row in motion_rows}
    quarantine_terms = {
        str(row["normalizedTerm"]) for row in quarantine_rows
    }
    for canonical in canonical_rows:
        declared_clip_id = canonical.get("motionClipId")
        normalized_motion_term = lexical_key(str(canonical["normalizedTerm"]))
        if declared_clip_id in playable_ids:
            canonical["hasMotion"] = True
            canonical["motionStatus"] = "playable"
        elif normalized_motion_term in quarantine_terms:
            canonical.pop("motionClipId", None)
            canonical["hasMotion"] = False
            canonical["motionStatus"] = "quarantined"
            canonical["reviewStatus"] = "motion_quarantined_source_label_conflict"
        elif declared_clip_id is not None:
            raise TrainingDatasetError(
                f"Canonical row {canonical['id']} references missing motion {declared_clip_id}."
            )
        else:
            canonical["hasMotion"] = False
            canonical["motionStatus"] = "unavailable"


def include_path_overlap_counts(
    include_rows: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, int]:
    paths = {
        split: {str(row["sourceRelativePath"]) for row in include_rows[split]}
        for split in SPLITS
    }
    return {
        "trainTest": len(paths["train"] & paths["test"]),
        "trainVal": len(paths["train"] & paths["val"]),
        "valTest": len(paths["val"] & paths["test"]),
    }


def _validate_cross_source_links(
    canonical_rows: Sequence[Mapping[str, Any]],
    include_rows: Mapping[str, Sequence[Mapping[str, Any]]],
    include_vocabulary: Sequence[Mapping[str, Any]],
    motion_rows: Sequence[Mapping[str, Any]],
    quarantine_rows: Sequence[Mapping[str, Any]],
) -> None:
    canonical_motion = {
        str(row["motionClipId"]): row for row in canonical_rows if row["hasMotion"]
    }
    motion_by_id = {str(row["clipId"]): row for row in motion_rows}
    if set(canonical_motion) != set(motion_by_id):
        raise TrainingDatasetError(
            "Canonical motion references do not exactly match the motion catalog."
        )
    include_sample_ids = {
        str(row["sourceSampleId"])
        for split in SPLITS
        for row in include_rows[split]
    }
    for clip_id, motion in motion_by_id.items():
        canonical = canonical_motion[clip_id]
        if motion["sourceSampleId"] not in include_sample_ids:
            raise TrainingDatasetError(
                f"Motion clip {clip_id} is not represented in INCLUDE metadata."
            )
        if collision_key(str(motion["normalizedTerm"])) != collision_key(
            str(canonical["normalizedTerm"])
        ):
            raise TrainingDatasetError(
                f"Motion clip {clip_id} does not match its canonical term."
            )
    vocabulary_motion_ids = {
        str(row["motionClipId"])
        for row in include_vocabulary
        if row["hasPlayableMotion"]
    }
    if vocabulary_motion_ids != set(motion_by_id):
        raise TrainingDatasetError(
            "INCLUDE vocabulary playable motion references are incomplete."
        )
    quarantined_vocabulary = [
        row for row in include_vocabulary if row["motionStatus"] == "quarantined"
    ]
    if len(quarantined_vocabulary) != len(quarantine_rows):
        raise TrainingDatasetError("INCLUDE quarantine coverage is incomplete.")
    for quarantine in quarantine_rows:
        matching_canonical = [
            row
            for row in canonical_rows
            if lexical_key(str(row["normalizedTerm"])) == quarantine["normalizedTerm"]
        ]
        if len(matching_canonical) != 1 or matching_canonical[0]["motionStatus"] != "quarantined":
            raise TrainingDatasetError(
                "The motion quarantine is not reflected in canonical metadata."
            )


def _query_owners(
    canonical_rows: Sequence[Mapping[str, Any]],
) -> Mapping[str, frozenset[str]]:
    owners: defaultdict[str, set[str]] = defaultdict(set)
    for row in canonical_rows:
        owners[collision_key(str(row["term"]))].add(str(row["id"]))
    return {key: frozenset(value) for key, value in owners.items()}


def select_match_representatives(
    canonical_rows: Sequence[Mapping[str, Any]],
) -> tuple[list[Mapping[str, Any]], dict[str, int]]:
    """Choose the lowest canonical ID for each trainer-normalized lexical key."""
    by_lexical_key: defaultdict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in canonical_rows:
        key = lexical_key(str(row["term"]))
        if not key:
            raise TrainingDatasetError(
                f"Canonical record {row['id']} has an empty trainer-normalized term."
            )
        by_lexical_key[key].append(row)

    alias_groups = [rows for rows in by_lexical_key.values() if len(rows) > 1]
    representatives = sorted(
        (
            min(rows, key=lambda row: str(row["id"]))
            for rows in by_lexical_key.values()
        ),
        key=lambda row: str(row["id"]),
    )
    stats = {
        "aliasedLexicalKeys": len(alias_groups),
        "canonicalLexicalKeys": len(by_lexical_key),
        "excludedAliasRecords": len(canonical_rows) - len(representatives),
        "recordsInAliasGroups": sum(len(rows) for rows in alias_groups),
        "representativeRecords": len(representatives),
    }
    return representatives, stats


def _safe_variant(query: str, record_id: str, owners: Mapping[str, frozenset[str]]) -> bool:
    query_owners = owners.get(collision_key(query), frozenset())
    return not query_owners or query_owners == {record_id}


def _punctuation_variant(term: str) -> str:
    stripped = " ".join(
        "".join(
            " " if unicodedata.category(character).startswith("P") else character
            for character in term
        ).split()
    )
    if stripped and stripped != term:
        return stripped
    return term + "?"


def _typo_candidates(term: str) -> Iterable[str]:
    spans = list(re.finditer(r"[^\W\d_]+", term, flags=re.UNICODE))
    spans.sort(key=lambda match: (-(match.end() - match.start()), match.start()))
    for match in spans:
        midpoint = (match.start() + match.end() - 1) / 2
        positions = sorted(
            range(match.start(), match.end() - 1),
            key=lambda index: (abs(index - midpoint), index),
        )
        for index in positions:
            if term[index].casefold() == term[index + 1].casefold():
                continue
            yield term[:index] + term[index + 1] + term[index] + term[index + 2 :]


def query_variants(
    canonical: Mapping[str, Any], owners: Mapping[str, frozenset[str]]
) -> list[tuple[str, str]]:
    record_id = str(canonical["id"])
    term = str(canonical["term"])
    variants: list[tuple[str, str]] = [("exact", term)]
    normalized_queries = {lexical_key(term)}
    punctuation = _punctuation_variant(term)
    punctuation_key = lexical_key(punctuation)
    if (
        punctuation != term
        and punctuation_key
        and punctuation_key not in normalized_queries
        and _safe_variant(punctuation, record_id, owners)
    ):
        variants.append(("punctuation", punctuation))
        normalized_queries.add(punctuation_key)
    for typo in _typo_candidates(term):
        typo_key = lexical_key(typo)
        if (
            typo != term
            and typo_key
            and typo_key not in normalized_queries
            and _safe_variant(typo, record_id, owners)
        ):
            variants.append(("typo", typo))
            break
    return variants[:3]


def select_match_queries(
    canonical_rows: Sequence[Mapping[str, Any]],
    representatives: Sequence[Mapping[str, Any]],
) -> dict[str, tuple[tuple[str, str], ...]]:
    """Allocate globally unique trainer-normalized queries to representatives."""
    owners = _query_owners(canonical_rows)
    reserved = {lexical_key(str(row["term"])) for row in representatives}
    selected: dict[str, tuple[tuple[str, str], ...]] = {}
    for canonical in sorted(representatives, key=lambda row: str(row["id"])):
        record_id = str(canonical["id"])
        queries: list[tuple[str, str]] = []
        for source_type, query in query_variants(canonical, owners):
            query_key = lexical_key(query)
            if source_type == "exact":
                queries.append((source_type, query))
            elif query_key not in reserved:
                reserved.add(query_key)
                queries.append((source_type, query))
        selected[record_id] = tuple(queries)
    return selected


def _semantic_tokens(value: str) -> frozenset[str]:
    return frozenset(
        token
        for token in lexical_key(value).split()
        if len(token) > 2 and not token.isdigit() and token not in SEMANTIC_STOPWORDS
    )


def _character_ngrams(value: str) -> frozenset[str]:
    compact = lexical_key(value).replace(" ", "_")
    if not compact:
        return frozenset()
    size = 3 if len(compact) >= 3 else len(compact)
    return frozenset(compact[index : index + size] for index in range(len(compact) - size + 1))


def feature_values(query: str, candidate_term: str) -> dict[str, Any]:
    query_key = lexical_key(query)
    candidate_key = lexical_key(candidate_term)
    query_tokens = frozenset(query_key.split())
    candidate_tokens = frozenset(candidate_key.split())
    union = query_tokens | candidate_tokens
    token_jaccard = len(query_tokens & candidate_tokens) / len(union) if union else 1.0
    return {
        "charSimilarity": round(
            SequenceMatcher(None, query_key, candidate_key, autojunk=False).ratio(), 6
        ),
        "exactNormalizedMatch": query_key == candidate_key,
        "lengthDelta": abs(len(query_key) - len(candidate_key)),
        "tokenJaccard": round(token_jaccard, 6),
    }


class HardNegativeIndex:
    """Select deterministic lexical near-neighbours within each hash split."""

    def __init__(self, canonical_rows: Sequence[Mapping[str, Any]]) -> None:
        self._records = {str(row["id"]): row for row in canonical_rows}
        self._by_split: dict[str, list[Mapping[str, Any]]] = {
            split: sorted(
                (row for row in canonical_rows if stable_split(str(row["id"])) == split),
                key=lambda row: str(row["id"]),
            )
            for split in SPLITS
        }
        self._keys = {
            record_id: lexical_key(str(row["term"]))
            for record_id, row in self._records.items()
        }
        self._tokens = {
            record_id: _semantic_tokens(str(row["term"]))
            for record_id, row in self._records.items()
        }
        self._grams = {
            record_id: _character_ngrams(str(row["term"]))
            for record_id, row in self._records.items()
        }
        self._postings: dict[str, defaultdict[str, list[str]]] = {
            split: defaultdict(list) for split in SPLITS
        }
        self._length_prefix_postings: dict[
            str, defaultdict[tuple[int, str], list[str]]
        ] = {split: defaultdict(list) for split in SPLITS}
        self._prefixes: dict[str, set[str]] = {split: set() for split in SPLITS}
        self._length_bounds: dict[str, tuple[int, int]] = {}
        for split in SPLITS:
            for row in self._by_split[split]:
                record_id = str(row["id"])
                key = self._keys[record_id]
                prefix = key[:1]
                self._length_prefix_postings[split][(len(key), prefix)].append(
                    record_id
                )
                self._prefixes[split].add(prefix)
                for gram in sorted(self._grams[record_id]):
                    self._postings[split][gram].append(record_id)
            lengths = [len(self._keys[str(row["id"])]) for row in self._by_split[split]]
            if lengths:
                self._length_bounds[split] = (min(lengths), max(lengths))
        self._cache: dict[str, tuple[Mapping[str, Any], ...]] = {}

    def _fallback_candidate_ids(
        self,
        record_id: str,
        split: str,
        excluded: set[str],
        limit: int,
    ) -> list[str]:
        """Reproduce the fallback rank through bounded posting-list merges."""
        if limit <= 0:
            return []
        target_key = self._keys[record_id]
        target_length = len(target_key)
        target_prefix = target_key[:1]
        minimum_length, maximum_length = self._length_bounds[split]
        maximum_delta = max(
            target_length - minimum_length, maximum_length - target_length
        )
        result: list[str] = []

        def add_ranked(prefixes: Sequence[str]) -> bool:
            for delta in range(maximum_delta + 1):
                lengths = [target_length - delta]
                if delta:
                    lengths.append(target_length + delta)
                postings = [
                    self._length_prefix_postings[split][(length, prefix)]
                    for length in lengths
                    for prefix in prefixes
                    if minimum_length <= length <= maximum_length
                    and self._length_prefix_postings[split].get((length, prefix))
                ]
                for candidate_id in heapq.merge(*postings):
                    if candidate_id == record_id or candidate_id in excluded:
                        continue
                    excluded.add(candidate_id)
                    result.append(candidate_id)
                    if len(result) == limit:
                        return True
            return False

        if add_ranked((target_prefix,)):
            return result
        other_prefixes = tuple(
            prefix
            for prefix in sorted(self._prefixes[split])
            if prefix != target_prefix
        )
        add_ranked(other_prefixes)
        return result

    def select(self, canonical: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
        record_id = str(canonical["id"])
        if record_id in self._cache:
            return self._cache[record_id]
        split = stable_split(record_id)
        required = min(3, len(self._by_split[split]) - 1)
        if required < 2:
            raise TrainingDatasetError(
                f"Split {split} needs at least three canonical rows for hard negatives."
            )

        overlap: Counter[str] = Counter()
        for gram in self._grams[record_id]:
            overlap.update(self._postings[split].get(gram, ()))
        overlap.pop(record_id, None)
        candidate_ids = sorted(
            overlap,
            key=lambda candidate_id: (
                -overlap[candidate_id],
                abs(len(self._keys[record_id]) - len(self._keys[candidate_id])),
                candidate_id,
            ),
        )[:128]

        if len(candidate_ids) < 64:
            candidate_ids.extend(
                self._fallback_candidate_ids(
                    record_id,
                    split,
                    set(candidate_ids),
                    128 - len(candidate_ids),
                )
            )

        def rank(candidate_id: str) -> tuple[Any, ...]:
            features = feature_values(
                str(canonical["term"]), str(self._records[candidate_id]["term"])
            )
            hardness = round(
                0.85 * features["charSimilarity"] + 0.15 * features["tokenJaccard"], 6
            )
            return (
                -hardness,
                -features["charSimilarity"],
                features["lengthDelta"],
                candidate_id,
            )

        strict = [
            candidate_id
            for candidate_id in candidate_ids
            if self._keys[candidate_id] != self._keys[record_id]
            and not (self._tokens[candidate_id] & self._tokens[record_id])
        ]
        strict.sort(key=rank)
        selected = strict[:required]
        if len(selected) < required:
            relaxed = [
                candidate_id
                for candidate_id in candidate_ids
                if candidate_id not in selected
                and self._keys[candidate_id] != self._keys[record_id]
            ]
            relaxed.sort(key=rank)
            selected.extend(relaxed[: required - len(selected)])
        if len(selected) != required:
            raise TrainingDatasetError(
                f"Could not select {required} hard negatives for {record_id}."
            )
        result = tuple(self._records[candidate_id] for candidate_id in selected)
        self._cache[record_id] = result
        return result


def group_id(record_id: str, source_type: str, query: str) -> str:
    digest = hashlib.sha256(
        f"{record_id}\0{source_type}\0{query}".encode("utf-8")
    ).hexdigest()[:24]
    return f"match-{digest}"


def _pair_sort_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (str(row["groupId"]), -int(row["label"]), str(row["candidateId"]))


def generate_match_pairs(
    canonical_rows: Sequence[Mapping[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    representatives, _ = select_match_representatives(canonical_rows)
    selected_queries = select_match_queries(canonical_rows, representatives)
    negative_index = HardNegativeIndex(representatives)
    output: dict[str, list[dict[str, Any]]] = {split: [] for split in SPLITS}
    for canonical in representatives:
        split = stable_split(str(canonical["id"]))
        negatives = negative_index.select(canonical)
        for source_type, query in selected_queries[str(canonical["id"])]:
            current_group_id = group_id(str(canonical["id"]), source_type, query)
            for label, candidate in [(1, canonical), *((0, row) for row in negatives)]:
                features = feature_values(query, str(candidate["term"]))
                output[split].append(
                    {
                        "candidateId": candidate["id"],
                        "candidateTerm": candidate["term"],
                        **features,
                        "groupId": current_group_id,
                        "hasMotion": candidate["hasMotion"],
                        "label": label,
                        "query": query,
                        "sourceType": source_type,
                        "split": split,
                    }
                )
    for split in SPLITS:
        output[split].sort(key=_pair_sort_key)
    return output


def _enforce_production_counts(
    canonical_rows: Sequence[Mapping[str, Any]],
    include_rows: Mapping[str, Sequence[Mapping[str, Any]]],
    include_vocabulary: Sequence[Mapping[str, Any]],
    motion_rows: Sequence[Mapping[str, Any]],
    quarantine_rows: Sequence[Mapping[str, Any]],
    expected_counts: Mapping[str, Any] | None,
) -> None:
    if expected_counts is None:
        return
    if len(canonical_rows) != expected_counts["canonical"]:
        raise TrainingDatasetError(
            f"Expected {expected_counts['canonical']} canonical rows; got {len(canonical_rows)}."
        )
    for split in SPLITS:
        expected = expected_counts["include"][split]
        if len(include_rows[split]) != expected:
            raise TrainingDatasetError(
                f"Expected {expected} INCLUDE {split} rows; got {len(include_rows[split])}."
            )
    if len(include_vocabulary) != expected_counts["includeVocabulary"]:
        raise TrainingDatasetError(
            f"Expected {expected_counts['includeVocabulary']} INCLUDE vocabulary labels; "
            f"got {len(include_vocabulary)}."
        )
    if len(motion_rows) != expected_counts["motion"]:
        raise TrainingDatasetError(
            f"Expected {expected_counts['motion']} motion clips; got {len(motion_rows)}."
        )
    if len(quarantine_rows) != expected_counts["motionQuarantine"]:
        raise TrainingDatasetError(
            f"Expected {expected_counts['motionQuarantine']} motion quarantine record; "
            f"got {len(quarantine_rows)}."
        )
    overlaps = include_path_overlap_counts(include_rows)
    if overlaps != expected_counts["includePathOverlaps"]:
        raise TrainingDatasetError(
            "INCLUDE exact video_path overlaps changed "
            f"(expected={expected_counts['includePathOverlaps']}; actual={overlaps})."
        )
    _, matcher_eligibility = select_match_representatives(canonical_rows)
    if matcher_eligibility != expected_counts["matcherEligibility"]:
        raise TrainingDatasetError(
            "Canonical matcher alias counts changed "
            f"(expected={expected_counts['matcherEligibility']}; "
            f"actual={matcher_eligibility})."
        )


def prepare_inputs(
    paths: InputPaths,
    expected_counts: Mapping[str, Any] | None = PRODUCTION_COUNTS,
) -> tuple[
    list[dict[str, Any]],
    dict[str, list[dict[str, Any]]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
]:
    canonical_rows = transform_canonical_rows(paths.canonical)
    include_rows = {
        split: transform_include_rows(paths.include(split), split) for split in SPLITS
    }
    motion_rows = transform_motion_catalog(paths.motion_catalog)
    quarantine_rows = build_motion_quarantine(include_rows)
    include_vocabulary = build_include_vocabulary(
        include_rows, motion_rows, quarantine_rows
    )
    reconcile_canonical_motion(canonical_rows, motion_rows, quarantine_rows)
    _enforce_production_counts(
        canonical_rows,
        include_rows,
        include_vocabulary,
        motion_rows,
        quarantine_rows,
        expected_counts,
    )
    _validate_cross_source_links(
        canonical_rows,
        include_rows,
        include_vocabulary,
        motion_rows,
        quarantine_rows,
    )
    overlaps = include_path_overlap_counts(include_rows)
    sources = {
        "canonical": {
            "logicalPath": "output/huggingface/Robin186-ISL/data/isl.jsonl",
            "rowCount": len(canonical_rows),
            "sha256": sha256_file(paths.canonical),
        },
        "include": {
            "exactVideoPathOverlapCounts": overlaps,
            "splitRole": "upstream_provenance_only_not_model_evaluation_partitions",
            "splits": {
                split: {
                    "logicalPath": f"data/isl/include-metadata/{split}.jsonl",
                    "rowCount": len(include_rows[split]),
                    "sha256": sha256_file(paths.include(split)),
                }
                for split in SPLITS
            },
            "vocabularyLabelCount": len(include_vocabulary),
        },
        "motionCatalog": {
            "clipCount": len(motion_rows),
            "logicalPath": "data/models/isl-motion-catalog.json",
            "sha256": sha256_file(paths.motion_catalog),
        },
    }
    return (
        canonical_rows,
        include_rows,
        include_vocabulary,
        motion_rows,
        quarantine_rows,
        sources,
    )


def _match_pair_counts(pair_rows: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, Any]:
    counts: dict[str, Any] = {}
    total_groups = 0
    total_rows = 0
    for split in SPLITS:
        rows = pair_rows[split]
        group_ids = {str(row["groupId"]) for row in rows}
        source_type_groups = Counter(
            (str(row["groupId"]), str(row["sourceType"])) for row in rows
        )
        source_types = Counter(source_type for _, source_type in source_type_groups)
        split_counts = {
            "groups": len(group_ids),
            "negatives": sum(row["label"] == 0 for row in rows),
            "positives": sum(row["label"] == 1 for row in rows),
            "rows": len(rows),
            "sourceTypeGroups": dict(sorted(source_types.items())),
        }
        counts[split] = split_counts
        total_groups += split_counts["groups"]
        total_rows += split_counts["rows"]
    counts["totalGroups"] = total_groups
    counts["totalRows"] = total_rows
    return counts


def _create_counts(
    canonical_rows: Sequence[Mapping[str, Any]],
    include_rows: Mapping[str, Sequence[Mapping[str, Any]]],
    include_vocabulary: Sequence[Mapping[str, Any]],
    motion_rows: Sequence[Mapping[str, Any]],
    quarantine_rows: Sequence[Mapping[str, Any]],
    pair_rows: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    canonical_by_split = Counter(stable_split(str(row["id"])) for row in canonical_rows)
    include_counts = {split: len(include_rows[split]) for split in SPLITS}
    include_counts["total"] = sum(include_counts.values())
    _, matcher_eligibility = select_match_representatives(canonical_rows)
    return {
        "canonicalBySplit": {
            split: canonical_by_split.get(split, 0) for split in SPLITS
        },
        "canonicalRecords": len(canonical_rows),
        "includeMetadataRows": include_counts,
        "includeVocabularyLabels": len(include_vocabulary),
        "matcherEligibility": matcher_eligibility,
        "matchPairs": _match_pair_counts(pair_rows),
        "motionClips": len(motion_rows),
        "motionQuarantineRecords": len(quarantine_rows),
    }


def _create_metadata(
    sources: Mapping[str, Any], counts: Mapping[str, Any]
) -> dict[str, Any]:
    registry = source_registry_payload()
    status_counts = Counter(str(source["status"]) for source in registry["sources"])
    return {
        "build": {
            "builderVersion": BUILDER_VERSION,
            "deterministicForFixedInputs": True,
            "networkAccessRequired": False,
            "schemaVersion": SCHEMA_VERSION,
            "splitAlgorithm": "sha256(record_id) first 8 bytes modulo 100; 0-79 train, 80-89 val, 90-99 test",
        },
        "counts": counts,
        "dataset": {
            "repository": DATASET_REPOSITORY,
            "visibility": DATASET_VISIBILITY,
        },
        "exclusions": {
            "absoluteLocalPathsIncluded": False,
            "credentialsIncluded": False,
            "iSignTextOrStatisticsIncluded": False,
            "landmarkFramesIncluded": False,
            "rawMediaIncluded": False,
            "signerInformationIncluded": False,
        },
        "licenses": {
            "bundle": "other",
            "include": "CC BY 4.0 with attribution",
            "islrtc": (
                "Official dictionary policy: research, teaching, and ISL technology use; "
                "no resale or profiteering; acknowledgement required"
            ),
            "privateHubRequired": True,
        },
        "sourceRegistry": {
            "file": REGISTRY_FILE,
            "sourceCount": len(registry["sources"]),
            "statusCounts": dict(sorted(status_counts.items())),
        },
        "sources": sources,
        "validation": {
            "checks": [
                "canonical_input_coverage",
                "deterministic_canonical_json",
                "forbidden_fragment_scan",
                "include_split_and_row_coverage",
                "include_vocabulary_coverage",
                "matcher_alias_representative_selection",
                "match_group_integrity",
                "motion_metadata_only_coverage",
                "motion_quarantine_coverage",
                "no_appledouble",
                "normalized_query_global_uniqueness",
                "output_allowlist",
                "pair_splits_independent_of_upstream_splits",
                "source_hashes",
                "split_isolation",
                "upstream_split_overlap_disclosed",
            ],
            "expectedFiles": sorted(EXPECTED_FILE_PATHS),
            "status": "passed",
        },
    }


def render_readme(metadata: Mapping[str, Any]) -> str:
    counts = metadata["counts"]
    matcher_eligibility = counts["matcherEligibility"]
    pair_counts = counts["matchPairs"]
    overlaps = metadata["sources"]["include"]["exactVideoPathOverlapCounts"]
    registry_rows = "\n".join(
        f"| {source['name']} | `{source['status']}` | {source['reason']} |"
        for source in source_registry_payload()["sources"]
    )
    return f"""---
license: other
language:
- en
pretty_name: SignSaarthi ISL Training
task_categories:
- text-classification
configs:
- config_name: match_pairs
  data_files:
  - split: train
    path: data/match_pairs/train.jsonl
  - split: validation
    path: data/match_pairs/val.jsonl
  - split: test
    path: data/match_pairs/test.jsonl
- config_name: canonical
  data_files:
  - split: train
    path: data/canonical.jsonl
- config_name: include_metadata
  data_files:
  - split: train
    path: data/include_metadata/train.jsonl
  - split: validation
    path: data/include_metadata/val.jsonl
  - split: test
    path: data/include_metadata/test.jsonl
- config_name: include_vocabulary
  data_files:
  - split: train
    path: data/include_vocabulary.jsonl
- config_name: motion_catalog
  data_files:
  - split: train
    path: data/motion_catalog.jsonl
- config_name: motion_quarantine
  data_files:
  - split: train
    path: data/motion_quarantine.jsonl
---

# SignSaarthi ISL Training

Deterministic private-Hub-ready metadata and lexical match-pair data for a
high-precision typo-to-canonical classifier. The builder performs no network
requests or uploads.

## Coverage

| Artifact | Count |
| --- | ---: |
| Canonical records | {counts['canonicalRecords']} |
| INCLUDE train metadata rows | {counts['includeMetadataRows']['train']} |
| INCLUDE validation metadata rows | {counts['includeMetadataRows']['val']} |
| INCLUDE test metadata rows | {counts['includeMetadataRows']['test']} |
| INCLUDE vocabulary labels | {counts['includeVocabularyLabels']} |
| Canonical trainer lexical keys | {matcher_eligibility['canonicalLexicalKeys']} |
| Aliased trainer lexical keys | {matcher_eligibility['aliasedLexicalKeys']} |
| Canonical records in alias groups | {matcher_eligibility['recordsInAliasGroups']} |
| Alias records excluded from matcher pairs | {matcher_eligibility['excludedAliasRecords']} |
| Matcher representative records | {matcher_eligibility['representativeRecords']} |
| Playable metadata-only motion clips | {counts['motionClips']} |
| Metadata-only motion quarantines | {counts['motionQuarantineRecords']} |
| Match groups | {pair_counts['totalGroups']} |
| Match candidate rows | {pair_counts['totalRows']} |

Match groups are split by SHA-256 of the positive canonical record ID using
80/10/10 buckets. Every candidate in a group belongs to the same split. Each
group contains one positive and two or three deterministic hard negatives. All
{counts['canonicalRecords']} canonical records remain in `canonical.jsonl`, but
matcher pairs use only the lowest canonical ID for each trainer-normalized
lexical key. This excludes {matcher_eligibility['excludedAliasRecords']} alias
records across {matcher_eligibility['aliasedLexicalKeys']} aliased keys from
matcher positives and candidates. Exact queries are reserved first; normalized
no-op punctuation variants and any query reused by another group are omitted.

The upstream INCLUDE split names are provenance fields only. Exact
`video_path` overlaps are train/validation `{overlaps['trainVal']}`,
train/test `{overlaps['trainTest']}`, and validation/test `{overlaps['valTest']}`.
Those upstream partitions cannot be used as leakage-safe model evaluation
splits. Match-pair splits are independent and use only the stable canonical-ID
hash rule above.

## Source Registry

`train` means private-training-only in this registry, `publish` permits only the
listed compact metadata under the stated terms, `reference` contributes citation
metadata only, and `excluded` contributes no content or derived values.

| Source | Status | Exact decision reason |
| --- | --- | --- |
{registry_rows}

## License And Safety

The bundle license is `other` because source-specific terms apply. INCLUDE is
tracked as CC BY 4.0 with attribution. ISLRTC metadata remains subject to the
official research, teaching, and ISL-technology conditions, including
acknowledgement and the no-resale/no-profiteering restriction.

No iSign text or derived statistics, raw media, landmark frame arrays, signer
information, credentials, or absolute local paths are included. Motion records
are metadata only and are not expert-certified ISL. `Second (Number)` remains
in the 263-label vocabulary but has no playable motion: every published source
path conflicts with that label, so its motion is explicitly quarantined. This
dataset must remain private unless every registry decision is reviewed again
for the intended publication context.

## Files

- `data/canonical.jsonl`: compact canonical IDs, terms, source/license labels,
  review state, and motion availability.
- `data/include_metadata/*.jsonl`: every source row, duplicate-preserving and
  kept in its original train/validation/test split, with relative media references only.
- `data/include_vocabulary.jsonl`: all 263 normalized train vocabulary labels,
  including playable or quarantined motion status.
- `data/motion_catalog.jsonl`: 262 compact playable clip records without source archive
  paths or landmark frames.
- `data/motion_quarantine.jsonl`: the explicit metadata-only quarantine and reason.
- `data/match_pairs/*.jsonl`: exact and globally trainer-distinct safe query groups.
- `source_registry.json`: hard-coded source decisions and exact reasons.
- `metadata.json`: source hashes, counts, output hashes, exclusions, and validation.
"""


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as destination:
        for row in rows:
            destination.write(canonical_json(row))
            destination.write("\n")


def _write_staging_bundle(
    staging: Path,
    canonical_rows: Sequence[Mapping[str, Any]],
    include_rows: Mapping[str, Sequence[Mapping[str, Any]]],
    include_vocabulary: Sequence[Mapping[str, Any]],
    motion_rows: Sequence[Mapping[str, Any]],
    quarantine_rows: Sequence[Mapping[str, Any]],
    pair_rows: Mapping[str, Sequence[Mapping[str, Any]]],
    metadata: dict[str, Any],
) -> None:
    _write_jsonl(staging / CANONICAL_FILE, canonical_rows)
    for split in SPLITS:
        _write_jsonl(staging / INCLUDE_FILES[split], include_rows[split])
        _write_jsonl(staging / PAIR_FILES[split], pair_rows[split])
    _write_jsonl(staging / INCLUDE_VOCABULARY_FILE, include_vocabulary)
    _write_jsonl(staging / MOTION_FILE, motion_rows)
    _write_jsonl(staging / MOTION_QUARANTINE_FILE, quarantine_rows)
    (staging / REGISTRY_FILE).write_text(
        canonical_json(source_registry_payload(), pretty=True),
        encoding="utf-8",
        newline="\n",
    )
    (staging / "README.md").write_text(
        render_readme(metadata), encoding="utf-8", newline="\n"
    )
    metadata["outputs"] = {
        relative_path: {
            "byteCount": (staging / relative_path).stat().st_size,
            "sha256": sha256_file(staging / relative_path),
        }
        for relative_path in sorted(EXPECTED_FILE_PATHS - {"metadata.json"})
    }
    (staging / "metadata.json").write_text(
        canonical_json(metadata, pretty=True), encoding="utf-8", newline="\n"
    )


def _observed_layout(bundle_dir: Path) -> set[str]:
    return {path.relative_to(bundle_dir).as_posix() for path in bundle_dir.rglob("*")}


def _remove_generated_appledouble(staging: Path) -> None:
    if not staging.name.startswith(".signsaarthi-training-staging-"):
        raise TrainingDatasetError(
            f"Refusing to scrub AppleDouble outside builder staging: {staging}"
        )
    for path in sorted(staging.rglob("._*"), reverse=True):
        if path.is_file():
            path.unlink()
        elif path.is_dir():
            path.rmdir()


def _check_layout(bundle_dir: Path) -> None:
    if not bundle_dir.is_dir():
        raise TrainingDatasetValidationError(f"Bundle directory does not exist: {bundle_dir}")
    for path in bundle_dir.rglob("*"):
        if any(part.startswith("._") for part in path.relative_to(bundle_dir).parts):
            raise TrainingDatasetValidationError("AppleDouble files are forbidden.")
        if path.is_symlink():
            raise TrainingDatasetValidationError("Symlinks are forbidden in the bundle.")
    expected = EXPECTED_FILE_PATHS | EXPECTED_DIRECTORY_PATHS
    observed = _observed_layout(bundle_dir)
    missing = sorted(expected - observed)
    unexpected = sorted(observed - expected)
    if missing or unexpected:
        details = []
        if missing:
            details.append("missing: " + ", ".join(missing))
        if unexpected:
            details.append("unexpected: " + ", ".join(unexpected))
        raise TrainingDatasetValidationError(
            "Invalid bundle layout (" + "; ".join(details) + ")."
        )


def _scan_forbidden_fragments(bundle_dir: Path) -> None:
    for relative_path in sorted(EXPECTED_FILE_PATHS):
        path = bundle_dir / relative_path
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError as error:
            raise TrainingDatasetValidationError(
                f"Bundle file is not UTF-8 text: {relative_path}"
            ) from error
        if text and not text.endswith("\n"):
            raise TrainingDatasetValidationError(
                f"Bundle text file lacks a final newline: {relative_path}"
            )
        for name, pattern in FORBIDDEN_TEXT_PATTERNS:
            if pattern.search(text):
                raise TrainingDatasetValidationError(
                    f"Forbidden {name} found in {relative_path}."
                )


def _read_output_jsonl(
    path: Path, *, sort_key: Any
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    text = path.read_text(encoding="utf-8")
    if text and not text.endswith("\n"):
        raise TrainingDatasetValidationError(f"JSONL file lacks final newline: {path.name}")
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line:
            raise TrainingDatasetValidationError(f"Blank JSONL line at {path}:{line_number}.")
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise TrainingDatasetValidationError(
                f"Invalid JSONL at {path}:{line_number}: {error}"
            ) from error
        if not isinstance(row, dict):
            raise TrainingDatasetValidationError(
                f"JSONL row at {path}:{line_number} must be an object."
            )
        if line != canonical_json(row):
            raise TrainingDatasetValidationError(
                f"Non-canonical JSON at {path}:{line_number}."
            )
        rows.append(row)
    if rows != sorted(rows, key=sort_key):
        raise TrainingDatasetValidationError(f"JSONL rows are not deterministically sorted: {path}")
    return rows


def _validate_exact_rows(
    actual: Sequence[Mapping[str, Any]],
    expected: Sequence[Mapping[str, Any]],
    name: str,
    allowed_fields: set[str] | tuple[set[str], set[str]],
) -> None:
    for row in actual:
        observed_fields = set(row)
        if isinstance(allowed_fields, tuple):
            if observed_fields not in allowed_fields:
                raise TrainingDatasetValidationError(f"{name} contains unexpected fields.")
        elif observed_fields != allowed_fields:
            raise TrainingDatasetValidationError(f"{name} contains unexpected fields.")
    if list(actual) != list(expected):
        raise TrainingDatasetValidationError(
            f"{name} does not represent every expected input row exactly once."
        )


def _validate_match_pairs(
    pair_rows: Mapping[str, Sequence[Mapping[str, Any]]],
    canonical_rows: Sequence[Mapping[str, Any]],
) -> None:
    canonical_by_id = {str(row["id"]): row for row in canonical_rows}
    representatives, _ = select_match_representatives(canonical_rows)
    representative_by_id = {str(row["id"]): row for row in representatives}
    representative_ids = set(representative_by_id)
    negative_index = HardNegativeIndex(representatives)
    selected_queries = select_match_queries(canonical_rows, representatives)
    expected_variants = {
        record_id: set(queries) for record_id, queries in selected_queries.items()
    }
    observed_variants: defaultdict[str, set[tuple[str, str]]] = defaultdict(set)
    global_group_ids: set[str] = set()
    normalized_query_groups: dict[str, tuple[str, str]] = {}
    candidate_ids_by_split: dict[str, set[str]] = {split: set() for split in SPLITS}

    for split in SPLITS:
        groups: defaultdict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for row in pair_rows[split]:
            if set(row) != PAIR_FIELDS:
                raise TrainingDatasetValidationError("Match-pair row contains unexpected fields.")
            if row["split"] != split or row["sourceType"] not in SOURCE_TYPES:
                raise TrainingDatasetValidationError("Match-pair split or sourceType is invalid.")
            if type(row["label"]) is not int or row["label"] not in {0, 1}:
                raise TrainingDatasetValidationError("Match-pair label must be 0 or 1.")
            candidate_id = str(row["candidateId"])
            candidate = canonical_by_id.get(candidate_id)
            if candidate is None:
                raise TrainingDatasetValidationError(
                    f"Unknown match-pair candidate ID: {candidate_id}"
                )
            if candidate_id not in representative_ids:
                raise TrainingDatasetValidationError(
                    f"Alias record {candidate_id} is not eligible for matcher pairs."
                )
            if stable_split(candidate_id) != split:
                raise TrainingDatasetValidationError(
                    f"Candidate {candidate_id} crosses split boundaries."
                )
            if row["candidateTerm"] != candidate["term"] or row["hasMotion"] != candidate["hasMotion"]:
                raise TrainingDatasetValidationError(
                    f"Candidate metadata mismatch for {candidate_id}."
                )
            expected_features = feature_values(str(row["query"]), str(candidate["term"]))
            for feature_name, expected_value in expected_features.items():
                if row[feature_name] != expected_value:
                    raise TrainingDatasetValidationError(
                        f"Feature {feature_name} is invalid for {candidate_id}."
                    )
            groups[str(row["groupId"])].append(row)
            candidate_ids_by_split[split].add(candidate_id)

        for current_group_id, rows in groups.items():
            if current_group_id in global_group_ids:
                raise TrainingDatasetValidationError(
                    f"Match group {current_group_id} appears in multiple splits."
                )
            global_group_ids.add(current_group_id)
            if len(rows) not in {3, 4}:
                raise TrainingDatasetValidationError(
                    f"Match group {current_group_id} must contain one positive and 2-3 negatives."
                )
            if len({str(row["candidateId"]) for row in rows}) != len(rows):
                raise TrainingDatasetValidationError(
                    f"Match group {current_group_id} repeats a candidate."
                )
            positives = [row for row in rows if row["label"] == 1]
            negatives = [row for row in rows if row["label"] == 0]
            if len(positives) != 1 or len(negatives) not in {2, 3}:
                raise TrainingDatasetValidationError(
                    f"Match group {current_group_id} must contain exactly one positive."
                )
            positive = positives[0]
            positive_id = str(positive["candidateId"])
            source_type = str(positive["sourceType"])
            query = str(positive["query"])
            if any(
                row["query"] != query
                or row["sourceType"] != source_type
                or row["split"] != split
                for row in rows
            ):
                raise TrainingDatasetValidationError(
                    f"Match group {current_group_id} has inconsistent group fields."
                )
            normalized_query = lexical_key(query)
            if not normalized_query:
                raise TrainingDatasetValidationError(
                    f"Match group {current_group_id} has an empty normalized query."
                )
            prior_group = normalized_query_groups.get(normalized_query)
            if prior_group is not None and prior_group[0] != current_group_id:
                raise TrainingDatasetValidationError(
                    "Trainer-normalized query appears in more than one match group or split "
                    f"({prior_group[0]} in {prior_group[1]}; "
                    f"{current_group_id} in {split})."
                )
            normalized_query_groups[normalized_query] = (current_group_id, split)
            if (source_type, query) not in expected_variants[positive_id]:
                raise TrainingDatasetValidationError(
                    f"Match group {current_group_id} has an unsafe or unexpected query."
                )
            if group_id(positive_id, source_type, query) != current_group_id:
                raise TrainingDatasetValidationError(
                    f"Match group {current_group_id} has a non-deterministic ID."
                )
            expected_negative_ids = {
                str(row["id"])
                for row in negative_index.select(representative_by_id[positive_id])
            }
            actual_negative_ids = {str(row["candidateId"]) for row in negatives}
            if actual_negative_ids != expected_negative_ids:
                raise TrainingDatasetValidationError(
                    f"Match group {current_group_id} has non-deterministic hard negatives."
                )
            observed_variants[positive_id].add((source_type, query))

    for left_index, left in enumerate(SPLITS):
        for right in SPLITS[left_index + 1 :]:
            if candidate_ids_by_split[left] & candidate_ids_by_split[right]:
                raise TrainingDatasetValidationError("Candidate IDs leak across pair splits.")
    for canonical_id, variants in expected_variants.items():
        if observed_variants[canonical_id] != variants:
            raise TrainingDatasetValidationError(
                f"Matcher representative {canonical_id} is not represented by every expected query."
            )


def _validate_output_hashes(bundle_dir: Path, metadata: Mapping[str, Any]) -> None:
    outputs = metadata.get("outputs")
    if not isinstance(outputs, dict):
        raise TrainingDatasetValidationError("metadata.json is missing output hashes.")
    expected_paths = EXPECTED_FILE_PATHS - {"metadata.json"}
    if set(outputs) != expected_paths:
        raise TrainingDatasetValidationError("metadata.json output paths are incomplete.")
    for relative_path in sorted(expected_paths):
        expected = outputs[relative_path]
        path = bundle_dir / relative_path
        if not isinstance(expected, dict):
            raise TrainingDatasetValidationError(
                f"Invalid output integrity metadata for {relative_path}."
            )
        if expected.get("sha256") != sha256_file(path) or expected.get(
            "byteCount"
        ) != path.stat().st_size:
            raise TrainingDatasetValidationError(
                f"Output hash or size mismatch for {relative_path}."
            )


def validate_bundle(
    bundle_dir: Path,
    paths: InputPaths = InputPaths(),
    expected_counts: Mapping[str, Any] | None = PRODUCTION_COUNTS,
) -> dict[str, Any]:
    bundle_dir = bundle_dir.resolve()
    _check_layout(bundle_dir)
    _scan_forbidden_fragments(bundle_dir)
    (
        canonical_expected,
        include_expected,
        include_vocabulary_expected,
        motion_expected,
        quarantine_expected,
        sources,
    ) = prepare_inputs(paths, expected_counts)

    canonical_actual = _read_output_jsonl(
        bundle_dir / CANONICAL_FILE, sort_key=lambda row: str(row["id"])
    )
    _validate_exact_rows(
        canonical_actual,
        canonical_expected,
        "canonical output",
        (CANONICAL_FIELDS, CANONICAL_MOTION_FIELDS),
    )
    include_actual: dict[str, list[dict[str, Any]]] = {}
    for split in SPLITS:
        include_actual[split] = _read_output_jsonl(
            bundle_dir / INCLUDE_FILES[split], sort_key=lambda row: str(row["id"])
        )
        _validate_exact_rows(
            include_actual[split],
            include_expected[split],
            f"INCLUDE {split} output",
            INCLUDE_FIELDS,
        )
    include_vocabulary_actual = _read_output_jsonl(
        bundle_dir / INCLUDE_VOCABULARY_FILE,
        sort_key=lambda row: str(row["vocabularyId"]),
    )
    _validate_exact_rows(
        include_vocabulary_actual,
        include_vocabulary_expected,
        "INCLUDE vocabulary output",
        (
            INCLUDE_VOCABULARY_FIELDS,
            INCLUDE_VOCABULARY_MOTION_FIELDS,
            INCLUDE_VOCABULARY_QUARANTINE_FIELDS,
        ),
    )
    motion_actual = _read_output_jsonl(
        bundle_dir / MOTION_FILE, sort_key=lambda row: str(row["clipId"])
    )
    _validate_exact_rows(
        motion_actual, motion_expected, "motion output", MOTION_FIELDS
    )
    quarantine_actual = _read_output_jsonl(
        bundle_dir / MOTION_QUARANTINE_FILE,
        sort_key=lambda row: str(row["quarantineId"]),
    )
    _validate_exact_rows(
        quarantine_actual,
        quarantine_expected,
        "motion quarantine output",
        MOTION_QUARANTINE_FIELDS,
    )

    pair_actual = {
        split: _read_output_jsonl(
            bundle_dir / PAIR_FILES[split], sort_key=_pair_sort_key
        )
        for split in SPLITS
    }
    _validate_match_pairs(pair_actual, canonical_actual)
    counts = _create_counts(
        canonical_actual,
        include_actual,
        include_vocabulary_actual,
        motion_actual,
        quarantine_actual,
        pair_actual,
    )

    metadata = _load_json(bundle_dir / "metadata.json")
    if metadata.get("build", {}).get("schemaVersion") != SCHEMA_VERSION:
        raise TrainingDatasetValidationError("metadata.json has the wrong schema version.")
    if metadata.get("dataset") != {
        "repository": DATASET_REPOSITORY,
        "visibility": DATASET_VISIBILITY,
    }:
        raise TrainingDatasetValidationError("metadata.json has the wrong dataset identity.")
    if metadata.get("sources") != sources:
        raise TrainingDatasetValidationError("metadata.json source hashes do not match inputs.")
    if metadata.get("counts") != counts:
        raise TrainingDatasetValidationError("metadata.json counts do not match outputs.")
    if metadata.get("validation", {}).get("status") != "passed":
        raise TrainingDatasetValidationError("metadata.json does not record passed validation.")
    if metadata.get("exclusions") != _create_metadata(sources, counts)["exclusions"]:
        raise TrainingDatasetValidationError("metadata.json exclusions are incomplete.")
    registry = _load_json(bundle_dir / REGISTRY_FILE)
    if registry != source_registry_payload():
        raise TrainingDatasetValidationError("source_registry.json differs from policy constants.")
    if (bundle_dir / "README.md").read_text(encoding="utf-8") != render_readme(metadata):
        raise TrainingDatasetValidationError("README.md is not deterministic for metadata.json.")
    _validate_output_hashes(bundle_dir, metadata)
    return {
        "canonicalRecords": counts["canonicalRecords"],
        "includeMetadataRows": counts["includeMetadataRows"]["total"],
        "includeVocabularyLabels": counts["includeVocabularyLabels"],
        "matcherExcludedAliasRecords": counts["matcherEligibility"][
            "excludedAliasRecords"
        ],
        "matcherRepresentativeRecords": counts["matcherEligibility"][
            "representativeRecords"
        ],
        "matchGroups": counts["matchPairs"]["totalGroups"],
        "matchRows": counts["matchPairs"]["totalRows"],
        "motionClips": counts["motionClips"],
        "motionQuarantineRecords": counts["motionQuarantineRecords"],
        "status": "passed",
    }


def _remove_guarded_temporary_tree(path: Path | None, parent: Path) -> None:
    if path is None or not path.exists():
        return
    resolved = path.resolve()
    allowed_parent = parent.resolve()
    prefixes = (
        ".signsaarthi-training-staging-",
        ".signsaarthi-training-backup-",
    )
    if (
        resolved.parent != allowed_parent
        or not resolved.name.startswith(prefixes)
        or not resolved.is_dir()
        or resolved == allowed_parent
    ):
        raise TrainingDatasetError(f"Refusing to remove untrusted temporary path: {resolved}")
    shutil.rmtree(resolved)


def build_bundle(
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    paths: InputPaths = InputPaths(),
    *,
    overwrite: bool = False,
    expected_counts: Mapping[str, Any] | None = PRODUCTION_COUNTS,
) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    if output_dir.exists():
        if not output_dir.is_dir():
            raise TrainingDatasetError(f"Output path is not a directory: {output_dir}")
        if not overwrite:
            raise TrainingDatasetError(
                f"Output directory already exists; pass --overwrite: {output_dir}"
            )
        _check_layout(output_dir)

    (
        canonical_rows,
        include_rows,
        include_vocabulary,
        motion_rows,
        quarantine_rows,
        sources,
    ) = prepare_inputs(paths, expected_counts)
    pair_rows = generate_match_pairs(canonical_rows)
    counts = _create_counts(
        canonical_rows,
        include_rows,
        include_vocabulary,
        motion_rows,
        quarantine_rows,
        pair_rows,
    )
    metadata = _create_metadata(sources, counts)

    staging: Path | None = Path(
        tempfile.mkdtemp(
            prefix=".signsaarthi-training-staging-", dir=output_dir.parent
        )
    )
    backup: Path | None = None
    try:
        assert staging is not None
        _write_staging_bundle(
            staging,
            canonical_rows,
            include_rows,
            include_vocabulary,
            motion_rows,
            quarantine_rows,
            pair_rows,
            metadata,
        )
        _remove_generated_appledouble(staging)
        report = validate_bundle(staging, paths, expected_counts)

        if output_dir.exists():
            backup = output_dir.parent / (
                f".signsaarthi-training-backup-{output_dir.name}-{os.getpid()}"
            )
            if backup.exists():
                raise TrainingDatasetError(f"Temporary backup already exists: {backup}")
            os.replace(output_dir, backup)
            try:
                os.replace(staging, output_dir)
                staging = None
            except BaseException:
                os.replace(backup, output_dir)
                backup = None
                raise
        else:
            os.replace(staging, output_dir)
            staging = None

        _remove_guarded_temporary_tree(backup, output_dir.parent)
        backup = None
        return report
    finally:
        _remove_guarded_temporary_tree(staging, output_dir.parent)
        _remove_guarded_temporary_tree(backup, output_dir.parent)


def _input_paths_from_args(args: argparse.Namespace) -> InputPaths:
    return InputPaths(
        canonical=args.canonical.resolve(),
        include_dir=args.include_metadata_dir.resolve(),
        motion_catalog=args.motion_catalog.resolve(),
    )


def _add_input_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--canonical", type=Path, default=DEFAULT_CANONICAL_PATH)
    parser.add_argument(
        "--include-metadata-dir", type=Path, default=DEFAULT_INCLUDE_DIR
    )
    parser.add_argument("--motion-catalog", type=Path, default=DEFAULT_MOTION_PATH)
    parser.add_argument(
        "--allow-nonproduction-counts",
        action="store_true",
        help="Allow small fixtures instead of enforcing 12,434/5,250/263/262/1 counts.",
    )


def make_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_parser = subparsers.add_parser("build", help="Build and validate the bundle.")
    _add_input_arguments(build_parser)
    build_parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    build_parser.add_argument("--overwrite", action="store_true")

    validate_parser = subparsers.add_parser(
        "validate", help="Independently validate an existing bundle."
    )
    _add_input_arguments(validate_parser)
    validate_parser.add_argument("--bundle-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = make_argument_parser().parse_args(argv)
    expected_counts = None if args.allow_nonproduction_counts else PRODUCTION_COUNTS
    paths = _input_paths_from_args(args)
    try:
        if args.command == "build":
            report = build_bundle(
                args.output_dir,
                paths,
                overwrite=args.overwrite,
                expected_counts=expected_counts,
            )
        else:
            report = validate_bundle(
                args.bundle_dir, paths, expected_counts=expected_counts
            )
    except (TrainingDatasetError, OSError) as error:
        print(f"SignSaarthi training dataset error: {error}", file=sys.stderr)
        return 1
    print(canonical_json(report, pretty=True), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
