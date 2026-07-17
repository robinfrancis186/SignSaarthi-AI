#!/usr/bin/env python3
"""Build a deterministic, low-disk unified Indian Sign Language dataset.

The builder snapshots the public ISLRTC A-Z index one page at a time, merges
only independently admissible local lexicon annotations and INCLUDE motion
catalog metadata, and writes a private-Hugging-Face-ready JSONL bundle. It does
not fetch dictionary videos, raw corpora, or gated iSign files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import string
import sys
import tempfile
import time
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date, datetime, timezone
from html.parser import HTMLParser
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Iterator, Mapping, Sequence
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, quote, urlencode, urlparse
from urllib.request import Request, urlopen


SCHEMA_VERSION = 1
BUILDER_VERSION = "1.1.0"
DATASET_REPOSITORY = "Robin186/ISL"
DATASET_VISIBILITY = "public"
LETTERS = tuple(string.ascii_uppercase)
ISLRTC_HOST = "divyangjan.depwd.gov.in"
ISLRTC_INDEX_URL = "https://divyangjan.depwd.gov.in/islrtc/listpage.php?type={letter}"
ISLRTC_DETAIL_URL = "https://divyangjan.depwd.gov.in/islrtc/search.php"
INCLUDE_SOURCE_URL = "https://huggingface.co/datasets/ai4bharat/INCLUDE"
LEXICON_ARTIFACT = "data/models/isl-lexicon-model.json"
MOTION_ARTIFACT = "data/models/isl-motion-catalog.json"
INCLUDE_METADATA_ARTIFACT = "data/isl/include-metadata"
INCLUDE_METADATA_SPLITS = ("train", "val", "test")
OUTPUT_DATA_FILE = "data/isl.jsonl"
FORBIDDEN_ISIGN_FRAGMENT = "data/raw/isign"
MAX_INDEX_PAGE_BYTES = 4 * 1024 * 1024
MAX_LOCAL_JSON_BYTES = 16 * 1024 * 1024
DEFAULT_MIN_OFFICIAL_ENTRIES = 10_000
USER_AGENT = "SignSaarthi-ISL-Dataset-Builder/1.0 (+https://huggingface.co/Robin186/ISL)"
DATASET_ID_ORDER = {"islrtc": 0, "include": 1}
EXPECTED_BUNDLE_PATHS = {
    "README.md",
    "metadata.json",
    "data",
    OUTPUT_DATA_FILE,
}
REPO_ROOT = Path(__file__).resolve().parents[1]


class DatasetBuildError(RuntimeError):
    """Raised when an input, source snapshot, or bundle is invalid."""


class DatasetValidationError(DatasetBuildError):
    """Raised when output validation fails."""


@dataclass(frozen=True)
class OfficialEntry:
    index_letter: str
    index_page_url: str
    entry_id: str
    display_term: str
    source_url: str


@dataclass(frozen=True)
class IndexPageSnapshot:
    letter: str
    index_url: str
    sha256: str
    byte_count: int
    entries: tuple[OfficialEntry, ...]


class _AnchorParser(HTMLParser):
    """Collect anchors with nested text while tolerating the source HTML."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.anchors: list[tuple[str, str]] = []
        self._active_href: str | None = None
        self._text_parts: list[str] = []

    def _finish_anchor(self) -> None:
        if self._active_href is not None:
            self.anchors.append((self._active_href, "".join(self._text_parts)))
        self._active_href = None
        self._text_parts = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.casefold() != "a":
            return
        if self._active_href is not None:
            self._finish_anchor()
        href = dict(attrs).get("href")
        self._active_href = href or ""
        self._text_parts = []

    def handle_data(self, data: str) -> None:
        if self._active_href is not None:
            self._text_parts.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag.casefold() == "a" and self._active_href is not None:
            self._finish_anchor()

    def close(self) -> None:
        super().close()
        if self._active_href is not None:
            self._finish_anchor()


def clean_text(value: Any, field: str = "text") -> str:
    if not isinstance(value, str):
        raise DatasetBuildError(f"{field} must be a string.")
    normalized = unicodedata.normalize("NFKC", value)
    normalized = normalized.replace("\u200b", "").replace("\ufeff", "")
    cleaned = " ".join(normalized.split())
    if not cleaned:
        raise DatasetBuildError(f"{field} must not be empty.")
    if any(unicodedata.category(character) in {"Cc", "Cs"} for character in cleaned):
        raise DatasetBuildError(f"{field} contains unsupported control characters.")
    return cleaned


def optional_clean_text(value: Any, field: str) -> str | None:
    if value is None or value == "":
        return None
    return clean_text(value, field)


def normalize_term(value: str) -> str:
    return clean_text(value, "term").casefold()


def term_match_key(value: str) -> str:
    normalized = normalize_term(value).replace("&", " and ")
    return "".join(character for character in normalized if character.isalnum())


def normalize_include_label(value: str) -> str:
    """Remove INCLUDE's numeric display prefix without changing the sign label."""

    return clean_text(re.sub(r"^\s*\d+\s*[.)-]\s*", "", value), "INCLUDE label")


def stable_record_id(normalized_term: str) -> str:
    digest = hashlib.sha256(normalized_term.encode("utf-8")).hexdigest()[:12]
    ascii_term = (
        unicodedata.normalize("NFKD", normalized_term)
        .encode("ascii", errors="ignore")
        .decode("ascii")
    )
    slug = re.sub(r"[^a-z0-9]+", "-", ascii_term.casefold()).strip("-")[:48].rstrip("-")
    return f"isl-{slug or 'term'}-{digest}"


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(value: Any, *, pretty: bool = False) -> str:
    if pretty:
        return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def canonical_detail_url(entry_id: str, display_term: str) -> str:
    query = urlencode(
        [("type", "list"), ("id", entry_id), ("search", display_term)],
        quote_via=quote,
        safe="",
    )
    return f"{ISLRTC_DETAIL_URL}?{query}"


def validate_official_detail_url(
    url: str,
    *,
    expected_id: str | None = None,
    expected_term: str | None = None,
) -> None:
    parsed = urlparse(url)
    if (
        parsed.scheme != "https"
        or parsed.hostname != ISLRTC_HOST
        or parsed.path != "/islrtc/search.php"
        or parsed.params
        or parsed.fragment
        or parsed.username
        or parsed.password
    ):
        raise DatasetValidationError(f"Invalid ISLRTC detail URL: {url}")
    query = parse_qs(parsed.query, keep_blank_values=True)
    if set(query) != {"type", "id", "search"}:
        raise DatasetValidationError(f"Unexpected ISLRTC detail URL query: {url}")
    if query["type"] != ["list"] or len(query["id"]) != 1 or len(query["search"]) != 1:
        raise DatasetValidationError(f"Malformed ISLRTC detail URL query: {url}")
    entry_id = query["id"][0]
    term = query["search"][0]
    if not entry_id.isdigit() or not term:
        raise DatasetValidationError(f"Malformed ISLRTC detail URL values: {url}")
    if expected_id is not None and entry_id != expected_id:
        raise DatasetValidationError(f"ISLRTC URL ID does not match entry {expected_id}: {url}")
    if expected_term is not None and clean_text(term, "URL search term") != clean_text(
        expected_term, "display term"
    ):
        raise DatasetValidationError(f"ISLRTC URL term does not match entry {entry_id}: {url}")
    if canonical_detail_url(entry_id, term) != url:
        raise DatasetValidationError(f"ISLRTC detail URL is not canonical: {url}")


def parse_index_page(raw_html: bytes, letter: str) -> IndexPageSnapshot:
    normalized_letter = clean_text(letter, "index letter").upper()
    if normalized_letter not in LETTERS:
        raise DatasetBuildError(f"Unsupported index letter: {letter}")
    index_url = ISLRTC_INDEX_URL.format(letter=normalized_letter)
    html = raw_html.decode("utf-8", errors="replace")
    parser = _AnchorParser()
    parser.feed(html)
    parser.close()

    entries: list[OfficialEntry] = []
    for href, anchor_text in parser.anchors:
        parsed = urlparse(href.strip())
        if not parsed.path.rstrip().endswith("search.php"):
            continue
        query = parse_qs(parsed.query, keep_blank_values=True)
        if [value.casefold() for value in query.get("type", [])] != ["list"]:
            continue
        entry_ids = query.get("id", [])
        if len(entry_ids) != 1 or not entry_ids[0].isdigit():
            raise DatasetBuildError(
                f"Malformed dictionary entry ID on the {normalized_letter} index page: {href}"
            )
        display_term = clean_text(anchor_text, f"ISLRTC entry {entry_ids[0]} term")
        source_url = canonical_detail_url(entry_ids[0], display_term)
        validate_official_detail_url(
            source_url,
            expected_id=entry_ids[0],
            expected_term=display_term,
        )
        entries.append(
            OfficialEntry(
                index_letter=normalized_letter,
                index_page_url=index_url,
                entry_id=entry_ids[0],
                display_term=display_term,
                source_url=source_url,
            )
        )

    if not entries:
        raise DatasetBuildError(
            f"The {normalized_letter} index page contained no structured dictionary entries."
        )
    return IndexPageSnapshot(
        letter=normalized_letter,
        index_url=index_url,
        sha256=sha256_bytes(raw_html),
        byte_count=len(raw_html),
        entries=tuple(entries),
    )


def _read_bounded(path: Path, max_bytes: int) -> bytes:
    if not path.is_file():
        raise DatasetBuildError(f"Required file does not exist: {path}")
    if path.stat().st_size > max_bytes:
        raise DatasetBuildError(f"File exceeds the {max_bytes}-byte safety limit: {path}")
    payload = path.read_bytes()
    if len(payload) > max_bytes:
        raise DatasetBuildError(f"File exceeds the {max_bytes}-byte safety limit: {path}")
    return payload


def _fetch_index_bytes(url: str, *, timeout: float, retries: int, max_bytes: int) -> bytes:
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            request = Request(
                url,
                headers={
                    "Accept": "text/html,application/xhtml+xml",
                    "User-Agent": USER_AGENT,
                },
            )
            with urlopen(request, timeout=timeout) as response:
                final_url = urlparse(response.geturl())
                if final_url.scheme != "https" or final_url.hostname != ISLRTC_HOST:
                    raise DatasetBuildError(f"ISLRTC request redirected to an unexpected host: {response.geturl()}")
                content_type = response.headers.get_content_type()
                if content_type not in {"text/html", "application/xhtml+xml"}:
                    raise DatasetBuildError(
                        f"ISLRTC returned unexpected content type {content_type!r} for {url}"
                    )
                content_length = response.headers.get("Content-Length")
                if content_length and int(content_length) > max_bytes:
                    raise DatasetBuildError(f"ISLRTC response exceeds the {max_bytes}-byte limit: {url}")
                payload = response.read(max_bytes + 1)
                if len(payload) > max_bytes:
                    raise DatasetBuildError(f"ISLRTC response exceeds the {max_bytes}-byte limit: {url}")
                return payload
        except HTTPError as error:
            last_error = error
            if error.code not in {408, 425, 429, 500, 502, 503, 504}:
                break
        except (URLError, TimeoutError, OSError) as error:
            last_error = error
        if attempt < retries:
            time.sleep(min(2 ** (attempt - 1), 4))
    raise DatasetBuildError(f"Unable to fetch {url} after {retries} attempt(s): {last_error}")


def snapshot_index_pages(
    *,
    index_html_dir: Path | None = None,
    timeout: float = 30.0,
    retries: int = 3,
    request_delay: float = 0.2,
    max_page_bytes: int = MAX_INDEX_PAGE_BYTES,
) -> list[IndexPageSnapshot]:
    if retries < 1:
        raise DatasetBuildError("retries must be at least 1.")
    if timeout <= 0 or request_delay < 0 or max_page_bytes < 1:
        raise DatasetBuildError("timeout/max-page-bytes must be positive and request-delay non-negative.")

    pages: list[IndexPageSnapshot] = []
    for position, letter in enumerate(LETTERS):
        if index_html_dir is not None:
            raw_html = _read_bounded(index_html_dir / f"{letter}.html", max_page_bytes)
        else:
            raw_html = _fetch_index_bytes(
                ISLRTC_INDEX_URL.format(letter=letter),
                timeout=timeout,
                retries=retries,
                max_bytes=max_page_bytes,
            )
        pages.append(parse_index_page(raw_html, letter))
        if index_html_dir is None and position < len(LETTERS) - 1 and request_delay:
            time.sleep(request_delay)
    return pages


def load_json_file(path: Path, *, max_bytes: int = MAX_LOCAL_JSON_BYTES) -> Any:
    payload = _read_bounded(path, max_bytes)
    try:
        return json.loads(payload)
    except json.JSONDecodeError as error:
        raise DatasetBuildError(f"Invalid JSON in {path}: {error}") from error


def _entry_sort_key(entry: OfficialEntry | Mapping[str, Any]) -> tuple[int, int | str, str]:
    entry_id = entry.entry_id if isinstance(entry, OfficialEntry) else str(entry["entryId"])
    if entry_id.isdigit():
        return (0, int(entry_id), entry_id)
    return (1, entry_id, entry_id)


def audit_official_pages(
    pages: Sequence[IndexPageSnapshot],
    *,
    expected_letters: Sequence[str] | None = None,
    min_entries: int = 1,
) -> tuple[dict[str, list[OfficialEntry]], dict[str, int]]:
    if min_entries < 1:
        raise DatasetBuildError("min_entries must be at least 1.")
    page_letters = [page.letter for page in pages]
    if len(page_letters) != len(set(page_letters)):
        raise DatasetBuildError("The source snapshot contains duplicate alphabet pages.")
    if expected_letters is not None and tuple(page_letters) != tuple(expected_letters):
        raise DatasetBuildError(
            "The source snapshot must contain exactly these ordered pages: "
            + ", ".join(expected_letters)
        )

    seen_ids: dict[str, str] = {}
    seen_urls: dict[str, str] = {}
    grouped: dict[str, list[OfficialEntry]] = defaultdict(list)
    entry_count = 0
    for page in pages:
        if not page.entries:
            raise DatasetBuildError(f"The {page.letter} index page is empty.")
        for entry in page.entries:
            entry_count += 1
            if entry.entry_id in seen_ids:
                raise DatasetBuildError(
                    f"Duplicate official entry ID {entry.entry_id}: "
                    f"{seen_ids[entry.entry_id]} and {entry.source_url}"
                )
            if entry.source_url in seen_urls:
                raise DatasetBuildError(
                    f"Duplicate official source URL {entry.source_url}: "
                    f"entry {seen_urls[entry.source_url]} and {entry.entry_id}"
                )
            validate_official_detail_url(
                entry.source_url,
                expected_id=entry.entry_id,
                expected_term=entry.display_term,
            )
            seen_ids[entry.entry_id] = entry.source_url
            seen_urls[entry.source_url] = entry.entry_id
            grouped[normalize_term(entry.display_term)].append(entry)

    if entry_count < min_entries:
        raise DatasetBuildError(
            f"The official snapshot has only {entry_count} entries; expected at least {min_entries}."
        )
    duplicate_groups = [entries for entries in grouped.values() if len(entries) > 1]
    return dict(grouped), {
        "entryCount": entry_count,
        "canonicalTermCount": len(grouped),
        "duplicateTermGroupCount": len(duplicate_groups),
        "duplicateTermExtraEntryCount": sum(len(entries) - 1 for entries in duplicate_groups),
        "duplicateIdCount": 0,
        "duplicateSourceUrlCount": 0,
    }


def _positive_number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DatasetBuildError(f"{field} must be numeric.")
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise DatasetBuildError(f"{field} must be finite and positive.")
    return number


def validate_motion_catalog(payload: Any) -> list[dict[str, Any]]:
    if not isinstance(payload, dict) or not isinstance(payload.get("clips"), list):
        raise DatasetBuildError("Motion catalog must be an object with a clips array.")
    clips = payload["clips"]
    declared_count = payload.get("clipCount")
    if declared_count is not None and declared_count != len(clips):
        raise DatasetBuildError(
            f"Motion catalog clipCount={declared_count} does not match {len(clips)} clips."
        )

    seen_ids: set[str] = set()
    seen_labels: set[str] = set()
    seen_samples: set[str] = set()
    validated: list[dict[str, Any]] = []
    for position, clip in enumerate(clips):
        if not isinstance(clip, dict):
            raise DatasetBuildError(f"Motion clip {position} must be an object.")
        clip_id = clean_text(clip.get("id"), f"motion clip {position} id")
        label = clean_text(clip.get("label"), f"motion clip {clip_id} label")
        normalized_label = clean_text(
            clip.get("normalizedLabel"), f"motion clip {clip_id} normalizedLabel"
        ).casefold()
        if term_match_key(normalized_label) != term_match_key(label):
            raise DatasetBuildError(
                f"Motion clip {clip_id} normalizedLabel does not match its label."
            )
        dataset_id = clean_text(clip.get("datasetId"), f"motion clip {clip_id} datasetId").casefold()
        if dataset_id != "include":
            raise DatasetBuildError(
                f"Motion clip {clip_id} is not INCLUDE-backed (datasetId={dataset_id!r})."
            )
        source_sample_id = clean_text(
            clip.get("sourceSampleId"), f"motion clip {clip_id} sourceSampleId"
        )
        frame_count = clip.get("frameCount")
        if isinstance(frame_count, bool) or not isinstance(frame_count, int) or frame_count <= 0:
            raise DatasetBuildError(f"Motion clip {clip_id} frameCount must be a positive integer.")
        fps = _positive_number(clip.get("fps"), f"motion clip {clip_id} fps")
        duration_ms = _positive_number(
            clip.get("durationMs"), f"motion clip {clip_id} durationMs"
        )
        timing_source = clean_text(
            clip.get("timingSource"), f"motion clip {clip_id} timingSource"
        )
        expert_reviewed = clip.get("expertReviewed")
        if not isinstance(expert_reviewed, bool):
            raise DatasetBuildError(f"Motion clip {clip_id} expertReviewed must be boolean.")
        label_key = normalize_term(label)
        if clip_id in seen_ids:
            raise DatasetBuildError(f"Duplicate motion clip ID: {clip_id}")
        if label_key in seen_labels:
            raise DatasetBuildError(f"Duplicate normalized motion label: {label}")
        if source_sample_id in seen_samples:
            raise DatasetBuildError(f"Duplicate motion source sample: {source_sample_id}")
        seen_ids.add(clip_id)
        seen_labels.add(label_key)
        seen_samples.add(source_sample_id)
        validated.append(
            {
                "catalogClipId": clip_id,
                "datasetId": "include",
                "durationMs": duration_ms,
                "expertReviewed": expert_reviewed,
                "fps": fps,
                "frameCount": frame_count,
                "framePayloadIncluded": False,
                "label": label,
                "normalizedLabel": normalized_label,
                "sourceRepresentation": "catalog_metadata",
                "sourceSampleId": source_sample_id,
                "timingSource": timing_source,
            }
        )
    return sorted(validated, key=lambda clip: str(clip["catalogClipId"]))


def _contains_forbidden_isign_source(value: Any) -> bool:
    if isinstance(value, str):
        normalized = value.replace("\\", "/").casefold()
        return FORBIDDEN_ISIGN_FRAGMENT in normalized
    if isinstance(value, list):
        return any(_contains_forbidden_isign_source(item) for item in value)
    if isinstance(value, dict):
        return any(_contains_forbidden_isign_source(item) for item in value.values())
    return False


def _is_explicit_isign_only(value: Mapping[str, Any]) -> bool:
    dataset_ids = value.get("datasetIds")
    if not isinstance(dataset_ids, list):
        return False
    normalized = [str(dataset_id).casefold() for dataset_id in dataset_ids]
    return normalized == ["isign"] or (normalized and set(normalized) == {"isign"})


def validate_lexicon(payload: Any) -> list[dict[str, Any]]:
    if not isinstance(payload, dict) or not isinstance(payload.get("glossaryEntries"), list):
        raise DatasetBuildError("Lexicon must be an object with a glossaryEntries array.")
    seen_ids: set[str] = set()
    seen_terms: set[str] = set()
    validated: list[dict[str, Any]] = []
    for position, entry in enumerate(payload["glossaryEntries"]):
        if not isinstance(entry, dict):
            raise DatasetBuildError(f"Lexicon entry {position} must be an object.")
        entry_id = clean_text(entry.get("id"), f"lexicon entry {position} id")
        term = clean_text(entry.get("term"), f"lexicon entry {entry_id} term")
        normalized = normalize_term(term)
        if entry_id in seen_ids:
            raise DatasetBuildError(f"Duplicate lexicon entry ID: {entry_id}")
        if normalized in seen_terms:
            raise DatasetBuildError(f"Duplicate normalized lexicon term: {term}")
        seen_ids.add(entry_id)
        seen_terms.add(normalized)
        validated.append(
            {
                "admissibleForEnrichment": not _contains_forbidden_isign_source(entry)
                and not _is_explicit_isign_only(entry),
                "category": optional_clean_text(entry.get("category"), f"lexicon {entry_id} category"),
                "confidence": optional_clean_text(
                    entry.get("confidence"), f"lexicon {entry_id} confidence"
                ),
                "entryId": entry_id,
                "islGloss": optional_clean_text(
                    entry.get("islGloss"), f"lexicon {entry_id} islGloss"
                ),
                "language": optional_clean_text(entry.get("language"), f"lexicon {entry_id} language"),
                "reviewStatus": optional_clean_text(
                    entry.get("reviewStatus"), f"lexicon {entry_id} reviewStatus"
                ),
                "signAssetId": optional_clean_text(
                    entry.get("signAssetId"), f"lexicon {entry_id} signAssetId"
                ),
                "term": term,
            }
        )
    return validated


def _new_accumulator(term: str, normalized: str) -> dict[str, Any]:
    return {
        "term": term,
        "normalizedTerm": normalized,
        "officialEntries": [],
        "includeMetadata": None,
        "motion": None,
        "lexicon": None,
    }


def _build_match_index(records: Mapping[str, Mapping[str, Any]]) -> dict[str, set[str]]:
    index: dict[str, set[str]] = defaultdict(set)
    for normalized, record in records.items():
        index[term_match_key(str(record["term"]))].add(normalized)
    return dict(index)


def _resolve_record(
    term: str,
    records: Mapping[str, Mapping[str, Any]],
    match_index: Mapping[str, set[str]],
    *,
    allow_relaxed: bool = False,
) -> tuple[str | None, str]:
    normalized = normalize_term(term)
    if normalized in records:
        return normalized, "exact_normalized_term"
    if not allow_relaxed:
        return None, "unmatched"
    candidates = match_index.get(term_match_key(term), set())
    if len(candidates) == 1:
        return next(iter(candidates)), "unique_relaxed_term"
    if len(candidates) > 1:
        return None, "ambiguous_relaxed_term"
    return None, "unmatched"


def validate_include_metadata_rows(
    rows: Sequence[Mapping[str, Any]] | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Validate compact INCLUDE row metadata and derive label-level provenance.

    Source paths are inspected only for integrity and split overlap. They are never
    copied into the public vocabulary bundle.
    """

    if rows is None:
        return [], {
            "inputRowCount": 0,
            "labelCount": 0,
            "pathConflictRowCount": 0,
            "quarantinedLabelCount": 0,
            "quarantinedLabels": [],
            "splitCounts": {},
            "splitVideoPathOverlapCounts": {},
        }

    labels: dict[str, dict[str, Any]] = {}
    split_counts: Counter[str] = Counter()
    split_paths: dict[str, set[str]] = {split: set() for split in INCLUDE_METADATA_SPLITS}
    path_conflict_count = 0

    for index, row in enumerate(rows, start=1):
        if not isinstance(row, Mapping):
            raise DatasetBuildError(f"INCLUDE metadata row {index} must be an object.")
        split = clean_text(row.get("split"), f"INCLUDE row {index} split").casefold()
        if split not in INCLUDE_METADATA_SPLITS:
            raise DatasetBuildError(f"INCLUDE row {index} has unsupported split {split!r}.")
        label = normalize_include_label(
            clean_text(row.get("label"), f"INCLUDE row {index} label")
        )
        parent_label = clean_text(
            row.get("parent_label"), f"INCLUDE row {index} parent_label"
        )
        video_path = clean_text(
            row.get("video_path"), f"INCLUDE row {index} video_path"
        ).replace("\\", "/")
        parsed_path = PurePosixPath(video_path)
        if parsed_path.is_absolute() or ".." in parsed_path.parts or len(parsed_path.parts) < 2:
            raise DatasetBuildError(f"INCLUDE row {index} has an unsafe video_path.")
        if not isinstance(row.get("include_50"), bool):
            raise DatasetBuildError(f"INCLUDE row {index} include_50 must be boolean.")

        normalized = normalize_term(label)
        directory_label = normalize_include_label(parsed_path.parent.name)
        path_consistent = term_match_key(directory_label) == term_match_key(label)
        if not path_consistent:
            path_conflict_count += 1

        label_state = labels.setdefault(
            normalized,
            {
                "label": label,
                "normalizedTerm": normalized,
                "rowCount": 0,
                "sourceSplits": set(),
                "consistentTrainPathCount": 0,
                "pathConflictRowCount": 0,
            },
        )
        if label_state["label"] != label:
            raise DatasetBuildError(
                f"INCLUDE normalized label {normalized!r} has conflicting display forms."
            )
        label_state["rowCount"] += 1
        label_state["sourceSplits"].add(split)
        label_state["pathConflictRowCount"] += int(not path_consistent)
        if split == "train" and path_consistent:
            label_state["consistentTrainPathCount"] += 1
        split_counts[split] += 1
        split_paths[split].add(video_path)
        del parent_label  # Validated above; not published in the vocabulary record.

    output_labels: list[dict[str, Any]] = []
    quarantined_labels: list[str] = []
    for normalized, state in sorted(labels.items()):
        source_splits = sorted(state["sourceSplits"], key=INCLUDE_METADATA_SPLITS.index)
        quarantined = "train" in source_splits and state["consistentTrainPathCount"] == 0
        if quarantined:
            quarantined_labels.append(str(state["label"]))
        output_labels.append(
            {
                "label": state["label"],
                "normalizedTerm": normalized,
                "rowCount": state["rowCount"],
                "sourceSplits": source_splits,
                "consistentTrainPathCount": state["consistentTrainPathCount"],
                "pathConflictRowCount": state["pathConflictRowCount"],
                "quarantined": quarantined,
            }
        )

    overlap_counts = {
        "trainVal": len(split_paths["train"] & split_paths["val"]),
        "trainTest": len(split_paths["train"] & split_paths["test"]),
        "valTest": len(split_paths["val"] & split_paths["test"]),
    }
    return output_labels, {
        "inputRowCount": len(rows),
        "labelCount": len(output_labels),
        "pathConflictRowCount": path_conflict_count,
        "quarantinedLabelCount": len(quarantined_labels),
        "quarantinedLabels": quarantined_labels,
        "splitCounts": dict(sorted(split_counts.items())),
        "splitVideoPathOverlapCounts": overlap_counts,
    }


def build_records(
    pages: Sequence[IndexPageSnapshot],
    lexicon_payload: Any,
    motion_payload: Any,
    *,
    include_metadata_rows: Sequence[Mapping[str, Any]] | None = None,
    expected_letters: Sequence[str] | None = None,
    min_official_entries: int = 1,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    official_groups, official_stats = audit_official_pages(
        pages,
        expected_letters=expected_letters,
        min_entries=min_official_entries,
    )
    records: dict[str, dict[str, Any]] = {}
    for normalized, entries in official_groups.items():
        sorted_entries = sorted(entries, key=_entry_sort_key)
        accumulator = _new_accumulator(sorted_entries[0].display_term, normalized)
        accumulator["officialEntries"] = sorted_entries
        records[normalized] = accumulator

    include_labels, include_metadata_stats = validate_include_metadata_rows(
        include_metadata_rows
    )
    for include_label in include_labels:
        normalized = str(include_label["normalizedTerm"])
        if normalized not in records:
            records[normalized] = _new_accumulator(str(include_label["label"]), normalized)
        records[normalized]["includeMetadata"] = include_label

    motion_clips = validate_motion_catalog(motion_payload)
    motion_join_counts: Counter[str] = Counter()
    match_index = _build_match_index(records)
    for motion in motion_clips:
        target, match_method = _resolve_record(
            str(motion["label"]), records, match_index, allow_relaxed=False
        )
        if target is None:
            normalized = normalize_term(str(motion["label"]))
            if normalized in records:
                raise DatasetBuildError(f"Unable to resolve motion label deterministically: {motion['label']}")
            records[normalized] = _new_accumulator(str(motion["label"]), normalized)
            target = normalized
            match_method = "include_motion_only"
            match_index = _build_match_index(records)
        if records[target]["motion"] is not None:
            raise DatasetBuildError(
                f"Multiple motion clips resolve to the same term: {records[target]['term']}"
            )
        motion_copy = dict(motion)
        motion_copy["joinMethod"] = match_method
        records[target]["motion"] = motion_copy
        motion_join_counts[match_method] += 1

    lexicon_entries = validate_lexicon(lexicon_payload)
    all_motion_ids = {str(clip["catalogClipId"]) for clip in motion_clips}
    lexicon_join_counts: Counter[str] = Counter()
    match_index = _build_match_index(records)
    for lexicon_entry in lexicon_entries:
        if not lexicon_entry["admissibleForEnrichment"]:
            lexicon_join_counts["explicit_gated_source_excluded"] += 1
            continue
        target, match_method = _resolve_record(
            str(lexicon_entry["term"]), records, match_index, allow_relaxed=False
        )
        if target is None:
            lexicon_join_counts[match_method] += 1
            continue
        if records[target]["lexicon"] is not None:
            raise DatasetBuildError(
                f"Multiple lexicon entries resolve to the same admitted term: {records[target]['term']}"
            )

        safe_gloss: str | None = None
        gloss = lexicon_entry["islGloss"]
        if gloss is not None:
            gloss_target, _ = _resolve_record(
                str(gloss), records, match_index, allow_relaxed=False
            )
            if gloss_target == target:
                safe_gloss = str(gloss)
                lexicon_join_counts["gloss_copied"] += 1
            else:
                lexicon_join_counts["gloss_excluded_not_independently_admitted"] += 1

        sign_asset_id = lexicon_entry["signAssetId"]
        target_motion = records[target]["motion"]
        if sign_asset_id is None:
            motion_link_status = "not_declared"
        elif target_motion is not None and sign_asset_id == target_motion["catalogClipId"]:
            motion_link_status = "resolved"
        elif sign_asset_id in all_motion_ids:
            motion_link_status = "catalog_clip_resolves_to_another_term"
        else:
            motion_link_status = "catalog_clip_missing"
        lexicon_join_counts[f"motion_link_{motion_link_status}"] += 1

        records[target]["lexicon"] = {
            "artifact": LEXICON_ARTIFACT,
            "category": lexicon_entry["category"],
            "confidence": lexicon_entry["confidence"],
            "entryId": lexicon_entry["entryId"],
            "islGloss": safe_gloss,
            "language": lexicon_entry["language"],
            "matchMethod": match_method,
            "motionLinkStatus": motion_link_status,
            "reviewStatus": lexicon_entry["reviewStatus"],
        }
        lexicon_join_counts[match_method] += 1

    output_records: list[dict[str, Any]] = []
    for normalized in sorted(records):
        accumulator = records[normalized]
        official_entries: list[OfficialEntry] = accumulator["officialEntries"]
        include_metadata = accumulator["includeMetadata"]
        motion = accumulator["motion"]
        lexicon = accumulator["lexicon"]
        has_official = bool(official_entries)
        has_include_metadata = include_metadata is not None
        has_motion = motion is not None
        dataset_ids = [dataset_id for dataset_id in ("islrtc", "include") if (
            dataset_id == "islrtc" and has_official
        ) or (dataset_id == "include" and (has_include_metadata or has_motion))]
        if not dataset_ids:
            raise DatasetBuildError(f"Term has no admissible primary provenance: {accumulator['term']}")

        official_output_entries = [
            {
                "displayTerm": entry.display_term,
                "entryId": entry.entry_id,
                "indexLetter": entry.index_letter,
                "sourceUrl": entry.source_url,
            }
            for entry in sorted(official_entries, key=_entry_sort_key)
        ]
        official_urls = [entry["sourceUrl"] for entry in official_output_entries]
        has_include = has_include_metadata or has_motion
        source_urls = sorted(set(official_urls + ([INCLUDE_SOURCE_URL] if has_include else [])))

        if has_official and has_include:
            license_status = "mixed_islrtc_usage_policy_and_include_cc_by_4_0"
            license_ids = ["islrtc-no-profiteering-attribution", "include-cc-by-4.0"]
        elif has_official:
            license_status = "islrtc-no-profiteering-attribution"
            license_ids = ["islrtc-no-profiteering-attribution"]
        else:
            license_status = "include-cc-by-4.0"
            license_ids = ["include-cc-by-4.0"]

        if has_motion and motion["expertReviewed"]:
            review_status = "motion_expert_reviewed_merge_not_reviewed"
        elif has_motion:
            review_status = "motion_not_expert_reviewed"
        elif has_include_metadata and include_metadata["quarantined"]:
            review_status = "include_metadata_only_motion_quarantined"
        elif has_include_metadata:
            review_status = "include_metadata_only_motion_unavailable"
        else:
            review_status = "official_index_only_motion_unavailable"

        if has_include_metadata:
            if has_motion:
                motion_status = "available"
            elif include_metadata["quarantined"]:
                motion_status = "quarantined_source_label_conflict"
            else:
                motion_status = "unavailable"
            include_output = {
                "label": include_metadata["label"],
                "listed": True,
                "metadataRowCount": include_metadata["rowCount"],
                "motionStatus": motion_status,
                "sourceSplits": include_metadata["sourceSplits"],
            }
            if include_metadata["quarantined"]:
                include_output["quarantineReason"] = (
                    "no source-consistent training video path for this label"
                )
        elif has_motion:
            include_output = {
                "label": motion["label"],
                "listed": True,
                "metadataRowCount": 0,
                "motionStatus": "available",
                "sourceSplits": [],
            }
        else:
            include_output = {
                "label": None,
                "listed": False,
                "metadataRowCount": 0,
                "motionStatus": "not_applicable",
                "sourceSplits": [],
            }

        output_records.append(
            {
                "datasetIds": dataset_ids,
                "id": stable_record_id(normalized),
                "include": include_output,
                "language": "en",
                "lexicon": lexicon,
                "license": {
                    "licenseIds": license_ids,
                    "redistributionReviewRequired": has_official,
                    "status": license_status,
                },
                "motion": motion,
                "normalizedTerm": normalized,
                "official": {
                    "entries": official_output_entries,
                    "entryCount": len(official_output_entries),
                    "indexPageUrls": sorted(
                        {entry.index_page_url for entry in official_entries}
                    ),
                    "listed": has_official,
                },
                "provenance": {
                    "includeMetadataSnapshot": has_include_metadata,
                    "includeMotionCatalog": has_motion,
                    "localLexiconEnrichment": lexicon is not None,
                    "officialIndexSnapshot": has_official,
                },
                "review": {
                    "mergeExpertReviewed": False,
                    "motionExpertReviewed": motion["expertReviewed"] if has_motion else None,
                    "officialVocabularySource": has_official,
                    "status": review_status,
                },
                "sourceUrls": source_urls,
                "term": accumulator["term"],
            }
        )

    validation = validate_records(output_records)
    stats = {
        "official": official_stats,
        "motion": {
            "inputClipCount": len(motion_clips),
            "joinCounts": dict(sorted(motion_join_counts.items())),
            "framePayloadCopiedCount": 0,
        },
        "includeMetadata": include_metadata_stats,
        "lexicon": {
            "inputEntryCount": len(lexicon_entries),
            "joinCounts": dict(sorted(lexicon_join_counts.items())),
            "matchedAdmittedRecordCount": validation["lexiconRecordCount"],
            "excludedEntryCount": len(lexicon_entries) - validation["lexiconRecordCount"],
            "aliasesCopiedCount": 0,
            "tokenWeightsCopiedCount": 0,
            "unmatchedTermsCopiedCount": 0,
        },
        "validation": validation,
    }
    return output_records, stats


def _walk_strings(value: Any) -> Iterator[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, list):
        for item in value:
            yield from _walk_strings(item)
    elif isinstance(value, dict):
        for item in value.values():
            yield from _walk_strings(item)


def validate_records(records: Iterable[Mapping[str, Any]]) -> dict[str, int]:
    seen_record_ids: set[str] = set()
    seen_terms: set[str] = set()
    seen_official_ids: set[str] = set()
    seen_official_urls: set[str] = set()
    seen_motion_ids: set[str] = set()
    record_count = 0
    official_record_count = 0
    official_entry_count = 0
    include_record_count = 0
    include_quarantine_count = 0
    motion_record_count = 0
    lexicon_record_count = 0

    for line_number, record in enumerate(records, start=1):
        record_count += 1
        if not isinstance(record, Mapping):
            raise DatasetValidationError(f"Record {line_number} is not an object.")
        for string_value in _walk_strings(record):
            normalized_value = string_value.replace("\\", "/").casefold()
            if FORBIDDEN_ISIGN_FRAGMENT in normalized_value:
                raise DatasetValidationError(
                    f"Record {line_number} points to forbidden gated iSign source data."
                )

        record_id = clean_text(record.get("id"), f"record {line_number} id")
        term = clean_text(record.get("term"), f"record {line_number} term")
        normalized_term = clean_text(
            record.get("normalizedTerm"), f"record {line_number} normalizedTerm"
        )
        if normalize_term(term) != normalized_term:
            raise DatasetValidationError(f"Record {record_id} has inconsistent normalizedTerm.")
        if record_id in seen_record_ids:
            raise DatasetValidationError(f"Duplicate output record ID: {record_id}")
        if normalized_term in seen_terms:
            raise DatasetValidationError(f"Duplicate output normalized term: {normalized_term}")
        if stable_record_id(normalized_term) != record_id:
            raise DatasetValidationError(f"Record {record_id} does not use the stable ID algorithm.")
        seen_record_ids.add(record_id)
        seen_terms.add(normalized_term)

        dataset_ids = record.get("datasetIds")
        if not isinstance(dataset_ids, list) or not dataset_ids:
            raise DatasetValidationError(f"Record {record_id} has no datasetIds.")
        normalized_dataset_ids = [clean_text(value, "datasetId").casefold() for value in dataset_ids]
        if "isign" in normalized_dataset_ids:
            raise DatasetValidationError(
                f"Record {record_id} contains prohibited iSign dataset provenance."
            )
        if any(dataset_id not in DATASET_ID_ORDER for dataset_id in normalized_dataset_ids):
            raise DatasetValidationError(f"Record {record_id} has an unsupported datasetId.")
        expected_order = sorted(set(normalized_dataset_ids), key=DATASET_ID_ORDER.__getitem__)
        if normalized_dataset_ids != expected_order:
            raise DatasetValidationError(f"Record {record_id} datasetIds are duplicated or unsorted.")

        official = record.get("official")
        if not isinstance(official, Mapping) or not isinstance(official.get("entries"), list):
            raise DatasetValidationError(f"Record {record_id} has invalid official provenance.")
        official_entries = official["entries"]
        listed = official.get("listed")
        if not isinstance(listed, bool) or listed != bool(official_entries):
            raise DatasetValidationError(f"Record {record_id} has inconsistent official.listed.")
        if official.get("entryCount") != len(official_entries):
            raise DatasetValidationError(f"Record {record_id} has inconsistent official.entryCount.")
        if ("islrtc" in normalized_dataset_ids) != listed:
            raise DatasetValidationError(f"Record {record_id} has inconsistent ISLRTC provenance.")
        if listed:
            official_record_count += 1
        expected_official_urls: list[str] = []
        for official_entry in official_entries:
            if not isinstance(official_entry, Mapping):
                raise DatasetValidationError(f"Record {record_id} has a malformed official entry.")
            entry_id = clean_text(official_entry.get("entryId"), "official entry ID")
            display_term = clean_text(official_entry.get("displayTerm"), "official display term")
            source_url = clean_text(official_entry.get("sourceUrl"), "official source URL")
            validate_official_detail_url(
                source_url,
                expected_id=entry_id,
                expected_term=display_term,
            )
            if normalize_term(display_term) != normalized_term:
                raise DatasetValidationError(
                    f"Official entry {entry_id} does not match record term {normalized_term}."
                )
            if source_url in seen_official_urls:
                raise DatasetValidationError(f"Duplicate official source URL in output: {source_url}")
            if entry_id in seen_official_ids:
                raise DatasetValidationError(f"Duplicate official entry ID in output: {entry_id}")
            seen_official_ids.add(entry_id)
            seen_official_urls.add(source_url)
            expected_official_urls.append(source_url)
            official_entry_count += 1
        if official_entries != sorted(official_entries, key=_entry_sort_key):
            raise DatasetValidationError(f"Record {record_id} official entries are not sorted.")

        include = record.get("include")
        if not isinstance(include, Mapping) or not isinstance(include.get("listed"), bool):
            raise DatasetValidationError(f"Record {record_id} has invalid INCLUDE provenance.")
        has_include = bool(include["listed"])
        if ("include" in normalized_dataset_ids) != has_include:
            raise DatasetValidationError(f"Record {record_id} has inconsistent INCLUDE provenance.")
        include_label = include.get("label")
        if has_include:
            if normalize_term(clean_text(include_label, "INCLUDE label")) != normalized_term:
                raise DatasetValidationError(
                    f"Record {record_id} INCLUDE label does not match the record term."
                )
            include_record_count += 1
        elif include_label is not None:
            raise DatasetValidationError(f"Record {record_id} has an unlisted INCLUDE label.")
        source_splits = include.get("sourceSplits")
        if not isinstance(source_splits, list) or any(
            split not in INCLUDE_METADATA_SPLITS for split in source_splits
        ):
            raise DatasetValidationError(f"Record {record_id} has invalid INCLUDE source splits.")
        expected_splits = sorted(set(source_splits), key=INCLUDE_METADATA_SPLITS.index)
        if source_splits != expected_splits:
            raise DatasetValidationError(f"Record {record_id} INCLUDE source splits are unsorted.")
        metadata_row_count = include.get("metadataRowCount")
        if not isinstance(metadata_row_count, int) or metadata_row_count < 0:
            raise DatasetValidationError(f"Record {record_id} has invalid INCLUDE row count.")
        motion_status = include.get("motionStatus")
        if motion_status == "quarantined_source_label_conflict":
            include_quarantine_count += 1
            if not has_include or not isinstance(include.get("quarantineReason"), str):
                raise DatasetValidationError(f"Record {record_id} has invalid quarantine metadata.")

        motion = record.get("motion")
        has_motion = motion is not None
        if has_motion and not has_include:
            raise DatasetValidationError(f"Record {record_id} has motion without INCLUDE provenance.")
        if has_motion:
            if not isinstance(motion, Mapping) or motion.get("datasetId") != "include":
                raise DatasetValidationError(f"Record {record_id} has invalid motion metadata.")
            motion_id = clean_text(motion.get("catalogClipId"), "motion catalog clip ID")
            if motion_id in seen_motion_ids:
                raise DatasetValidationError(f"Duplicate output motion clip ID: {motion_id}")
            seen_motion_ids.add(motion_id)
            if motion.get("framePayloadIncluded") is not False:
                raise DatasetValidationError(
                    f"Record {record_id} incorrectly claims an embedded motion frame payload."
                )
            motion_record_count += 1
            if motion_status != "available":
                raise DatasetValidationError(f"Record {record_id} has inconsistent motion status.")
        elif motion_status == "available":
            raise DatasetValidationError(f"Record {record_id} claims unavailable motion as playable.")

        lexicon = record.get("lexicon")
        if lexicon is not None:
            if not isinstance(lexicon, Mapping):
                raise DatasetValidationError(f"Record {record_id} has malformed lexicon enrichment.")
            prohibited_fields = {
                "aliases",
                "createdAt",
                "datasetIds",
                "source",
                "sourcePath",
                "term",
                "tokenWeight",
                "updatedAt",
            }
            if prohibited_fields.intersection(lexicon):
                raise DatasetValidationError(
                    f"Record {record_id} copies prohibited local lexicon fields."
                )
            lexicon_record_count += 1

        source_urls = record.get("sourceUrls")
        if not isinstance(source_urls, list) or any(not isinstance(url, str) for url in source_urls):
            raise DatasetValidationError(f"Record {record_id} has invalid sourceUrls.")
        if source_urls != sorted(set(source_urls)):
            raise DatasetValidationError(f"Record {record_id} sourceUrls are duplicated or unsorted.")
        expected_source_urls = sorted(
            set(expected_official_urls + ([INCLUDE_SOURCE_URL] if has_include else []))
        )
        if source_urls != expected_source_urls:
            raise DatasetValidationError(f"Record {record_id} sourceUrls do not match provenance.")
        for source_url in source_urls:
            if source_url == INCLUDE_SOURCE_URL:
                continue
            validate_official_detail_url(source_url)

    if record_count == 0:
        raise DatasetValidationError("The output dataset contains no records.")
    return {
        "duplicateOfficialIdCount": 0,
        "duplicateOfficialSourceUrlCount": 0,
        "duplicateRecordIdCount": 0,
        "duplicateTermCount": 0,
        "forbiddenIsignRecordCount": 0,
        "includeQuarantineCount": include_quarantine_count,
        "includeRecordCount": include_record_count,
        "lexiconRecordCount": lexicon_record_count,
        "motionRecordCount": motion_record_count,
        "officialEntryCount": official_entry_count,
        "officialRecordCount": official_record_count,
        "recordCount": record_count,
    }


def create_metadata(
    *,
    pages: Sequence[IndexPageSnapshot],
    records: Sequence[Mapping[str, Any]],
    stats: Mapping[str, Any],
    snapshot_date: str,
    lexicon_sha256: str,
    motion_sha256: str,
) -> dict[str, Any]:
    official_record_count = sum(bool(record["official"]["listed"]) for record in records)
    motion_record_count = sum(record["motion"] is not None for record in records)
    official_with_motion_count = sum(
        bool(record["official"]["listed"]) and record["motion"] is not None for record in records
    )
    motion_only_count = sum(
        not bool(record["official"]["listed"]) and record["motion"] is not None for record in records
    )
    review_counts = Counter(str(record["review"]["status"]) for record in records)
    return {
        "build": {
            "builderVersion": BUILDER_VERSION,
            "deterministicForFixedInputs": True,
            "schemaVersion": SCHEMA_VERSION,
            "snapshotDate": snapshot_date,
        },
        "coverage": {
            "includeMetadataLabelCount": stats["includeMetadata"]["labelCount"],
            "includeMetadataRowCount": stats["includeMetadata"]["inputRowCount"],
            "includeQuarantineCount": stats["validation"]["includeQuarantineCount"],
            "lexiconEnrichedRecordCount": stats["validation"]["lexiconRecordCount"],
            "motionOnlyRecordCount": motion_only_count,
            "motionRecordCount": motion_record_count,
            "officialEntryCount": stats["official"]["entryCount"],
            "officialRecordCount": official_record_count,
            "officialWithMotionCount": official_with_motion_count,
            "officialWithMotionPercent": round(
                official_with_motion_count / official_record_count * 100, 4
            )
            if official_record_count
            else 0.0,
            "recordCount": len(records),
            "reviewStatusCounts": dict(sorted(review_counts.items())),
        },
        "dataset": {
            "dataFile": OUTPUT_DATA_FILE,
            "format": "jsonl",
            "repository": DATASET_REPOSITORY,
            "visibility": DATASET_VISIBILITY,
        },
        "exclusions": {
            "iSign": {
                "aliasesCopied": False,
                "gatedFilesRead": False,
                "onlyIndependentlyAdmittedLocalEntriesMayEnrichRecords": True,
                "rawRowsCopied": False,
                "status": "excluded",
                "tokenWeightsCopied": False,
            }
        },
        "inputs": {
            "includeMetadata": {
                "artifact": INCLUDE_METADATA_ARTIFACT,
                **stats["includeMetadata"],
            },
            "lexicon": {
                "artifact": LEXICON_ARTIFACT,
                "entryCount": stats["lexicon"]["inputEntryCount"],
                "merge": stats["lexicon"],
                "sha256": lexicon_sha256,
            },
            "motionCatalog": {
                "artifact": MOTION_ARTIFACT,
                "clipCount": stats["motion"]["inputClipCount"],
                "merge": stats["motion"],
                "sha256": motion_sha256,
            },
        },
        "licenses": {
            "datasetCardLicense": "other",
            "sourcePolicies": [
                {
                    "datasetId": "islrtc",
                    "license": "ISLRTC dictionary usage policy",
                    "redistributionStatus": (
                        "research_teaching_and_technology_only; no resale or profiteering; "
                        "attribution required"
                    ),
                    "sourceUrl": "https://islrtc.nic.in/faq/",
                    "status": "official_faq_conditions_recorded",
                },
                {
                    "datasetId": "include",
                    "license": "CC-BY-4.0",
                    "redistributionStatus": "attribution_required",
                    "sourceUrl": INCLUDE_SOURCE_URL,
                    "status": "declared_in_local_include_metadata",
                },
            ],
        },
        "officialSnapshot": {
            "canonicalTermCount": stats["official"]["canonicalTermCount"],
            "duplicateIdCount": stats["official"]["duplicateIdCount"],
            "duplicateSourceUrlCount": stats["official"]["duplicateSourceUrlCount"],
            "duplicateTermExtraEntryCount": stats["official"]["duplicateTermExtraEntryCount"],
            "duplicateTermGroupCount": stats["official"]["duplicateTermGroupCount"],
            "entryCount": stats["official"]["entryCount"],
            "pages": [
                {
                    "byteCount": page.byte_count,
                    "entryCount": len(page.entries),
                    "indexUrl": page.index_url,
                    "letter": page.letter,
                    "sha256": page.sha256,
                }
                for page in pages
            ],
            "sourceAuthority": "Indian Sign Language Research and Training Centre",
        },
        "outputs": {},
        "validation": {
            **stats["validation"],
            "status": "passed",
        },
    }


def render_readme(metadata: Mapping[str, Any]) -> str:
    coverage = metadata["coverage"]
    snapshot = metadata["officialSnapshot"]
    snapshot_date = metadata["build"]["snapshotDate"]
    return f"""---
pretty_name: Robin186 Unified Indian Sign Language Vocabulary
license: other
language:
- en
tags:
- indian-sign-language
- accessibility
- vocabulary
configs:
- config_name: default
  data_files:
  - split: train
    path: data/isl.jsonl
---

# Robin186/ISL

Public, provenance-first vocabulary index for Indian Sign Language. The {snapshot_date}
snapshot combines the public ISLRTC A-Z dictionary index with a small local INCLUDE
metadata snapshot, a validated motion catalog, and narrowly filtered SignSaarthi
lexicon annotations. It does not
download or contain dictionary videos, raw video corpora, audio, or signer media.

## Coverage

| Measure | Count |
| --- | ---: |
| Dataset records | {coverage['recordCount']} |
| Official ISLRTC entries preserved | {coverage['officialEntryCount']} |
| Canonical official terms | {snapshot['canonicalTermCount']} |
| Official duplicate-term groups consolidated | {snapshot['duplicateTermGroupCount']} |
| INCLUDE vocabulary labels | {coverage['includeMetadataLabelCount']} |
| Records with INCLUDE motion metadata | {coverage['motionRecordCount']} |
| INCLUDE labels quarantined from motion playback | {coverage['includeQuarantineCount']} |
| Official terms with INCLUDE motion metadata | {coverage['officialWithMotionCount']} |
| Locally enriched admitted records | {coverage['lexiconEnrichedRecordCount']} |

Repeated official terms are one dataset row with every distinct official entry ID and
detail URL retained under `official.entries`; they are not silently discarded.

## Record fields

- `id`, `term`, `normalizedTerm`, `language`: stable vocabulary identity.
- `datasetIds`, `sourceUrls`, `official`, `include`: primary provenance, official
  variants, source-split presence, and honest motion availability.
- `motion`: compact INCLUDE catalog metadata. The source catalog contains no landmark
  frame arrays, so `framePayloadIncluded` is always `false` in this bundle.
- `lexicon`: safe local annotations only after the term independently matches ISLRTC
  or INCLUDE. Aliases, token weights, source fields, and timestamps are not copied.
- `license`, `review`: per-row rights and expert-review status without certification claims.

## Licensing and review

This dataset card uses `license: other` because it combines source-specific terms.
The official ISLRTC FAQ permits dictionary use for research, teaching, and technology
only when the data are not resold or used for profiteering and ISLRTC is properly
acknowledged. INCLUDE is tracked as CC-BY-4.0 based on its source metadata. This public
bundle contains metadata and source links only; it does not grant commercial rights or
permission to redistribute source media or derived landmark payloads.

The upstream INCLUDE metadata contains exact media-path overlap between its published
splits, so those split names are retained for provenance only and are not represented
as a leakage-safe evaluation benchmark. A label with no source-consistent training
path is kept as vocabulary but quarantined from motion playback.

An official dictionary listing establishes vocabulary provenance; it does not prove
that this merge or an INCLUDE motion clip received independent ISL expert review.
Motion review status is copied literally from the catalog and is currently represented
per record.

## iSign exclusion

No iSign-only record or gated CSV content is included. The blended local lexicon may
enrich a term only after that term is independently admitted by ISLRTC or an INCLUDE
motion. Unmatched local terms are counted only in `metadata.json`; their text is not
written. The validator rejects iSign dataset provenance and gated local source paths.

## Files

- `data/isl.jsonl`: deterministic records sorted by normalized term.
- `metadata.json`: input/page hashes, coverage, exclusions, licenses, and validation.
- `README.md`: this Hugging Face dataset card.

The bundle is intended for vocabulary lookup, coverage analysis, and motion
catalog joins. It is not a complete motion corpus, translation benchmark, or substitute
for review by qualified ISL users and experts.
"""


def _iter_jsonl(path: Path) -> Iterator[Mapping[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                raise DatasetValidationError(f"Blank JSONL line at {line_number} in {path}.")
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise DatasetValidationError(
                    f"Invalid JSONL at {path}:{line_number}: {error}"
                ) from error
            if not isinstance(record, dict):
                raise DatasetValidationError(f"JSONL record {line_number} is not an object.")
            yield record


def load_include_metadata_directory(
    directory: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not directory.is_dir():
        raise DatasetBuildError(f"INCLUDE metadata directory does not exist: {directory}")
    rows: list[dict[str, Any]] = []
    source_files: list[dict[str, Any]] = []
    for split in INCLUDE_METADATA_SPLITS:
        path = directory / f"{split}.jsonl"
        if not path.is_file():
            raise DatasetBuildError(f"Missing INCLUDE metadata split: {path}")
        split_rows = [dict(record) for record in _iter_jsonl(path)]
        for record in split_rows:
            record["split"] = split
            rows.append(record)
        source_files.append(
            {
                "path": f"{INCLUDE_METADATA_ARTIFACT}/{split}.jsonl",
                "rowCount": len(split_rows),
                "sha256": sha256_file(path),
                "split": split,
            }
        )
    return rows, source_files


def _check_bundle_layout(bundle_dir: Path) -> None:
    if not bundle_dir.is_dir():
        raise DatasetValidationError(f"Bundle directory does not exist: {bundle_dir}")
    observed = {
        path.relative_to(bundle_dir).as_posix()
        for path in bundle_dir.rglob("*")
    }
    unexpected = sorted(observed - EXPECTED_BUNDLE_PATHS)
    missing = sorted(EXPECTED_BUNDLE_PATHS - observed)
    if unexpected or missing:
        details = []
        if missing:
            details.append("missing: " + ", ".join(missing))
        if unexpected:
            details.append("unexpected: " + ", ".join(unexpected))
        raise DatasetValidationError("Invalid bundle layout (" + "; ".join(details) + ").")


def validate_bundle(bundle_dir: Path) -> dict[str, Any]:
    _check_bundle_layout(bundle_dir)
    metadata_path = bundle_dir / "metadata.json"
    data_path = bundle_dir / OUTPUT_DATA_FILE
    readme_path = bundle_dir / "README.md"
    metadata = load_json_file(metadata_path)
    if not isinstance(metadata, dict):
        raise DatasetValidationError("metadata.json must contain an object.")
    if metadata.get("dataset", {}).get("repository") != DATASET_REPOSITORY:
        raise DatasetValidationError("metadata.json has the wrong dataset repository.")
    if metadata.get("dataset", {}).get("visibility") != DATASET_VISIBILITY:
        raise DatasetValidationError(
            f"The generated dataset must be marked {DATASET_VISIBILITY}."
        )
    outputs = metadata.get("outputs")
    if not isinstance(outputs, dict):
        raise DatasetValidationError("metadata.json is missing output integrity data.")
    actual_sha256 = sha256_file(data_path)
    if outputs.get("dataFile") != OUTPUT_DATA_FILE or outputs.get("dataSha256") != actual_sha256:
        raise DatasetValidationError("JSONL path or SHA-256 does not match metadata.json.")

    validation = validate_records(_iter_jsonl(data_path))
    if outputs.get("recordCount") != validation["recordCount"]:
        raise DatasetValidationError("JSONL record count does not match metadata.json.")
    for key, value in validation.items():
        if metadata.get("validation", {}).get(key) != value:
            raise DatasetValidationError(f"Validation metric {key} does not match metadata.json.")
    if metadata.get("validation", {}).get("status") != "passed":
        raise DatasetValidationError("metadata.json does not record passed validation.")
    exclusion = metadata.get("exclusions", {}).get("iSign", {})
    if exclusion.get("status") != "excluded" or exclusion.get("rawRowsCopied") is not False:
        raise DatasetValidationError("metadata.json does not enforce the iSign exclusion.")

    readme = readme_path.read_text(encoding="utf-8")
    if "license: other" not in readme or "No iSign-only record" not in readme:
        raise DatasetValidationError("README.md is missing license or iSign safety disclosure.")
    return {
        **validation,
        "dataSha256": actual_sha256,
        "status": "passed",
    }


def write_bundle(
    output_dir: Path,
    records: Sequence[Mapping[str, Any]],
    metadata: Mapping[str, Any],
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    if output_dir.exists():
        if not output_dir.is_dir():
            raise DatasetBuildError(f"Output path is not a directory: {output_dir}")
        if any(output_dir.iterdir()) and not overwrite:
            raise DatasetBuildError(
                f"Output directory is not empty; pass --overwrite for a known bundle: {output_dir}"
            )
        if any(output_dir.iterdir()):
            _check_bundle_layout(output_dir)

    staging: Path | None = Path(
        tempfile.mkdtemp(prefix=".unified-isl-staging-", dir=output_dir.parent)
    )
    backup: Path | None = None
    try:
        assert staging is not None
        data_path = staging / OUTPUT_DATA_FILE
        data_path.parent.mkdir(parents=True)
        with data_path.open("w", encoding="utf-8", newline="\n") as destination:
            for record in records:
                destination.write(canonical_json(record))
                destination.write("\n")

        finalized_metadata = json.loads(canonical_json(metadata))
        finalized_metadata["outputs"] = {
            "dataByteCount": data_path.stat().st_size,
            "dataFile": OUTPUT_DATA_FILE,
            "dataSha256": sha256_file(data_path),
            "metadataFile": "metadata.json",
            "readmeFile": "README.md",
            "recordCount": len(records),
        }
        (staging / "metadata.json").write_text(
            canonical_json(finalized_metadata, pretty=True), encoding="utf-8", newline="\n"
        )
        (staging / "README.md").write_text(
            render_readme(finalized_metadata), encoding="utf-8", newline="\n"
        )
        _remove_appledouble_files(staging)
        report = validate_bundle(staging)

        if output_dir.exists() and any(output_dir.iterdir()):
            backup = output_dir.parent / (
                f".unified-isl-backup-{output_dir.name}-{os.getpid()}-{time.time_ns()}"
            )
            os.replace(output_dir, backup)
            try:
                os.replace(staging, output_dir)
                staging = None
            except BaseException:
                os.replace(backup, output_dir)
                backup = None
                raise
        else:
            if output_dir.exists():
                output_dir.rmdir()
            os.replace(staging, output_dir)
            staging = None

        _remove_appledouble_files(output_dir)

        _remove_guarded_temporary_tree(backup, output_dir.parent)
        backup = None
        return report
    finally:
        _remove_guarded_temporary_tree(staging, output_dir.parent)
        _remove_guarded_temporary_tree(backup, output_dir.parent)


def _remove_appledouble_files(root: Path) -> None:
    """Remove only ExFAT metadata sidecars from a builder-owned bundle tree."""

    for path in sorted(root.rglob("._*"), reverse=True):
        if path.is_file() or path.is_symlink():
            path.unlink()
        elif path.is_dir():
            path.rmdir()


def _remove_guarded_temporary_tree(path: Path | None, allowed_parent: Path) -> None:
    """Remove only builder-owned sibling directories, never cwd or a parent tree."""

    if path is None or not path.exists():
        return
    resolved = path.resolve()
    parent = allowed_parent.resolve()
    allowed_prefixes = (".unified-isl-staging-", ".unified-isl-backup-")
    if resolved.parent != parent or not resolved.name.startswith(allowed_prefixes):
        raise DatasetBuildError(f"Refusing to remove untrusted temporary path: {resolved}")
    if not resolved.is_dir() or resolved == parent or parent in resolved.parents[1:]:
        raise DatasetBuildError(f"Refusing unsafe temporary cleanup path: {resolved}")
    shutil.rmtree(resolved)


def parse_snapshot_date(value: str) -> str:
    try:
        parsed = date.fromisoformat(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("snapshot date must use YYYY-MM-DD") from error
    if parsed.isoformat() != value:
        raise argparse.ArgumentTypeError("snapshot date must use YYYY-MM-DD")
    return value


def build_command(args: argparse.Namespace) -> dict[str, Any]:
    lexicon_path = args.lexicon.resolve()
    motion_path = args.motion_catalog.resolve()
    include_metadata_directory = args.include_metadata_dir.resolve()
    pages = snapshot_index_pages(
        index_html_dir=args.index_html_dir.resolve() if args.index_html_dir else None,
        timeout=args.timeout,
        retries=args.retries,
        request_delay=args.request_delay,
        max_page_bytes=args.max_page_bytes,
    )
    lexicon_payload = load_json_file(lexicon_path)
    motion_payload = load_json_file(motion_path)
    include_metadata_rows, include_metadata_files = load_include_metadata_directory(
        include_metadata_directory
    )
    records, stats = build_records(
        pages,
        lexicon_payload,
        motion_payload,
        include_metadata_rows=include_metadata_rows,
        expected_letters=LETTERS,
        min_official_entries=args.min_official_entries,
    )
    stats["includeMetadata"]["sourceFiles"] = include_metadata_files
    metadata = create_metadata(
        pages=pages,
        records=records,
        stats=stats,
        snapshot_date=args.snapshot_date,
        lexicon_sha256=sha256_file(lexicon_path),
        motion_sha256=sha256_file(motion_path),
    )
    return write_bundle(args.output_dir, records, metadata, overwrite=args.overwrite)


def make_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_parser = subparsers.add_parser("build", help="Fetch/parse, merge, and write a bundle.")
    build_parser.add_argument("--output-dir", type=Path, required=True)
    build_parser.add_argument(
        "--lexicon", type=Path, default=REPO_ROOT / LEXICON_ARTIFACT
    )
    build_parser.add_argument(
        "--motion-catalog", type=Path, default=REPO_ROOT / MOTION_ARTIFACT
    )
    build_parser.add_argument(
        "--include-metadata-dir",
        type=Path,
        default=REPO_ROOT / INCLUDE_METADATA_ARTIFACT,
    )
    build_parser.add_argument(
        "--snapshot-date",
        type=parse_snapshot_date,
        default=datetime.now(timezone.utc).date().isoformat(),
        help="Date recorded in deterministic metadata (default: current UTC date).",
    )
    build_parser.add_argument(
        "--index-html-dir",
        type=Path,
        help="Offline directory containing A.html through Z.html instead of network access.",
    )
    build_parser.add_argument("--timeout", type=float, default=30.0)
    build_parser.add_argument("--retries", type=int, default=3)
    build_parser.add_argument("--request-delay", type=float, default=0.2)
    build_parser.add_argument("--max-page-bytes", type=int, default=MAX_INDEX_PAGE_BYTES)
    build_parser.add_argument(
        "--min-official-entries", type=int, default=DEFAULT_MIN_OFFICIAL_ENTRIES
    )
    build_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace only an existing bundle with the exact expected layout.",
    )
    build_parser.set_defaults(handler=build_command)

    validate_parser = subparsers.add_parser("validate", help="Independently validate a bundle.")
    validate_parser.add_argument("--bundle-dir", type=Path, required=True)
    validate_parser.set_defaults(handler=lambda args: validate_bundle(args.bundle_dir.resolve()))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = make_argument_parser()
    args = parser.parse_args(argv)
    try:
        report = args.handler(args)
    except (DatasetBuildError, OSError) as error:
        print(f"unified ISL dataset error: {error}", file=sys.stderr)
        return 1
    print(canonical_json(report, pretty=True), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
