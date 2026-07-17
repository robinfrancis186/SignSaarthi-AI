#!/usr/bin/env python3
"""Train a deterministic, private text-only planner from the iSign captions.

The exported artifact is a compact statistical English-caption model. It is
research-only and is not an ISL grammar, translation, Text2Pose, or motion
model. Source identifiers and complete source sentences are never serialized.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import tempfile
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


SCHEMA_VERSION = 1
DEFAULT_SEED = "isign-sentence-planner-v1"
SPLIT_ORDER = ("train", "validation", "test")
SPLIT_THRESHOLDS = (("train", 0.8), ("validation", 0.9), ("test", 1.0))

UNKNOWN_TOKEN = "<unk>"
END_TOKEN = "<eos>"
BEGIN_TOKEN = "<bos>"
MAX_VOCABULARY_SIZE = 4096
MIN_TOKEN_COUNT = 3
MAX_CONTEXTS = 512
MAX_TRANSITIONS_PER_CONTEXT = 8
MIN_CONTEXT_COUNT = 5
MIN_TRANSITION_COUNT = 2
UNIGRAM_ALPHA = 0.1
BACKOFF_STRENGTH = 20.0
MAX_CONTEXT_WEIGHT = 0.9

MAX_BOUNDARY_TOKENS = 256
MAX_BOUNDARY_PHRASES = 256
MAX_PHRASE_LENGTH = 4
MIN_PHRASE_BOUNDARY_COUNT = 5
MIN_PHRASE_SUPPORT = 20
MIN_PHRASE_VIDEO_SUPPORT = 5
MIN_TOKEN_BOUNDARY_SUPPORT = 20
MIN_TOKEN_VIDEO_SUPPORT = 5

SOURCE_FILES = (
    ("iSign_v1.1.csv", "caption_training_and_held_out_evaluation"),
    ("word-description-dataset_v1.1.csv", "auxiliary_coverage_diagnostic_only"),
    ("word-presence-dataset_v1.1.csv", "auxiliary_coverage_diagnostic_only"),
    ("README.md", "license_and_dataset_documentation"),
)

CAPTION_IDENTIFIER = re.compile(r"^(.+)-(\d+)$")
TOKEN_PATTERN = re.compile(r"[^\W_]+(?:['-][^\W_]+)*|[.!?,;:-]", re.UNICODE)
PUNCTUATION_KINDS = {
    ".": "strong",
    "!": "strong",
    "?": "strong",
    ",": "weak",
    ";": "weak",
    ":": "weak",
    "-": "weak",
}
BOUNDARY_PRIORITY = {"end": 0, "weak": 1, "strong": 2}
BOUNDARY_INDEX = {"strong": 0, "weak": 1, "end": 2}
BOUNDARY_NAMES = ("strong", "weak", "end")


class PlannerTrainingError(RuntimeError):
    """Raised when the source corpus cannot produce a valid private artifact."""


@dataclass(frozen=True)
class CaptionRecord:
    """In-memory caption features; identifiers and normalized text are not exported."""

    video_id: str
    sequence: int
    normalized_text: str
    tokens: tuple[str, ...]
    boundaries: tuple[tuple[int, str], ...]


@dataclass(frozen=True)
class LoadedCorpus:
    records: tuple[CaptionRecord, ...]
    counts: Mapping[str, int]


@dataclass(frozen=True)
class PartitionedCorpus:
    raw: Mapping[str, tuple[CaptionRecord, ...]]
    retained: Mapping[str, tuple[CaptionRecord, ...]]
    within_split_duplicates: Mapping[str, int]
    cross_split_duplicates: Mapping[str, int]


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True) + "\n"
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_source_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    replacements = {
        "\u2018": "'",
        "\u2019": "'",
        "\u2010": "-",
        "\u2011": "-",
        "\u2012": "-",
        "\u2013": "-",
        "\u2014": "-",
        "\u2212": "-",
    }
    for source, target in replacements.items():
        normalized = normalized.replace(source, target)
    return " ".join(normalized.split())


def analyze_text(text: str) -> tuple[str, tuple[str, ...], tuple[tuple[int, str], ...]]:
    normalized = normalize_source_text(text)
    raw_tokens = TOKEN_PATTERN.findall(normalized)
    lexical_tokens: list[str] = []
    boundary_by_index: dict[int, str] = {}

    for token in raw_tokens:
        boundary_kind = PUNCTUATION_KINDS.get(token)
        if boundary_kind is None:
            lexical_tokens.append(token)
            continue
        if not lexical_tokens:
            continue
        token_index = len(lexical_tokens) - 1
        previous = boundary_by_index.get(token_index)
        if previous is None or BOUNDARY_PRIORITY[boundary_kind] > BOUNDARY_PRIORITY[previous]:
            boundary_by_index[token_index] = boundary_kind

    if lexical_tokens:
        final_index = len(lexical_tokens) - 1
        boundary_by_index.setdefault(final_index, "end")

    deduplication_key = " ".join(raw_tokens)
    boundaries = tuple(sorted(boundary_by_index.items()))
    return deduplication_key, tuple(lexical_tokens), boundaries


def make_caption_record(video_id: str, sequence: int, text: str) -> CaptionRecord:
    normalized_text, tokens, boundaries = analyze_text(text)
    if not video_id:
        raise PlannerTrainingError("A caption record has an empty video group identifier.")
    if not normalized_text or not tokens:
        raise PlannerTrainingError("A caption record has no lexical text after normalization.")
    return CaptionRecord(video_id, sequence, normalized_text, tokens, boundaries)


def load_caption_corpus(path: Path) -> LoadedCorpus:
    counts: Counter[str] = Counter()
    records: list[CaptionRecord] = []
    seen_positions: set[tuple[str, int]] = set()

    with path.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        if reader.fieldnames is None or not {"uid", "text"}.issubset(reader.fieldnames):
            raise PlannerTrainingError(f"{path} must contain uid and text columns.")

        for row in reader:
            counts["sourceCsvRows"] += 1
            source_identifier = (row.get("uid") or "").strip()
            match = CAPTION_IDENTIFIER.fullmatch(source_identifier)
            if match is None:
                counts["excludedNonCaptionTaskRows"] += 1
                continue

            counts["eligibleCaptionRows"] += 1
            video_id = match.group(1)
            sequence = int(match.group(2))
            position = (video_id, sequence)
            if position in seen_positions:
                raise PlannerTrainingError(
                    "The source contains a duplicate video-group and sequence position."
                )
            seen_positions.add(position)

            text = row.get("text") or ""
            if not text.strip():
                counts["excludedBlankCaptionRows"] += 1
                continue
            normalized_text, tokens, boundaries = analyze_text(text)
            if not normalized_text or not tokens:
                counts["excludedNoLexicalTokenRows"] += 1
                continue
            records.append(
                CaptionRecord(video_id, sequence, normalized_text, tokens, boundaries)
            )

    records.sort(key=lambda record: (record.video_id, record.sequence))
    counts["preparedCaptionRows"] = len(records)
    counts["videoCount"] = len({record.video_id for record in records})
    if not records:
        raise PlannerTrainingError(f"No usable numeric-sequence caption rows were found in {path}.")
    return LoadedCorpus(tuple(records), dict(sorted(counts.items())))


def assign_video_split(video_id: str, seed: str = DEFAULT_SEED) -> str:
    digest = hashlib.sha256(f"{seed}\0{video_id}".encode("utf-8")).digest()
    fraction = int.from_bytes(digest[:8], "big") / float(1 << 64)
    for split, threshold in SPLIT_THRESHOLDS:
        if fraction < threshold:
            return split
    raise AssertionError("The final split threshold must include every SHA-256 value.")


def partition_corpus(
    records: Sequence[CaptionRecord], seed: str = DEFAULT_SEED
) -> PartitionedCorpus:
    raw_lists: dict[str, list[CaptionRecord]] = {split: [] for split in SPLIT_ORDER}
    for record in records:
        raw_lists[assign_video_split(record.video_id, seed)].append(record)

    for split in SPLIT_ORDER:
        raw_lists[split].sort(key=lambda record: (record.video_id, record.sequence))

    retained_lists: dict[str, list[CaptionRecord]] = {split: [] for split in SPLIT_ORDER}
    within_split_duplicates: dict[str, int] = {split: 0 for split in SPLIT_ORDER}
    cross_split_duplicates: dict[str, int] = {split: 0 for split in SPLIT_ORDER}
    previous_split_texts: set[str] = set()

    for split in SPLIT_ORDER:
        current_split_texts: set[str] = set()
        for record in raw_lists[split]:
            if record.normalized_text in previous_split_texts:
                cross_split_duplicates[split] += 1
                continue
            if record.normalized_text in current_split_texts:
                within_split_duplicates[split] += 1
                continue
            current_split_texts.add(record.normalized_text)
            retained_lists[split].append(record)
        previous_split_texts.update(current_split_texts)

    partition = PartitionedCorpus(
        raw={split: tuple(raw_lists[split]) for split in SPLIT_ORDER},
        retained={split: tuple(retained_lists[split]) for split in SPLIT_ORDER},
        within_split_duplicates=within_split_duplicates,
        cross_split_duplicates=cross_split_duplicates,
    )
    assert_partition_is_leak_free(partition)
    return partition


def assert_partition_is_leak_free(partition: PartitionedCorpus) -> None:
    video_sets = {
        split: {record.video_id for record in partition.raw[split]} for split in SPLIT_ORDER
    }
    text_sets = {
        split: {record.normalized_text for record in partition.retained[split]}
        for split in SPLIT_ORDER
    }
    for left_index, left in enumerate(SPLIT_ORDER):
        for right in SPLIT_ORDER[left_index + 1 :]:
            if video_sets[left] & video_sets[right]:
                raise PlannerTrainingError(
                    f"Video-group leakage detected between {left} and {right}."
                )
            if text_sets[left] & text_sets[right]:
                raise PlannerTrainingError(
                    f"Exact normalized-sentence leakage detected between {left} and {right}."
                )
    if any(not partition.retained[split] for split in SPLIT_ORDER):
        raise PlannerTrainingError(
            "Every split must retain at least one caption after leakage decontamination."
        )


def iter_ngrams(tokens: Sequence[str], minimum: int = 2, maximum: int = 4) -> Iterable[tuple[str, ...]]:
    for size in range(minimum, maximum + 1):
        for start in range(0, len(tokens) - size + 1):
            yield tuple(tokens[start : start + size])


def build_boundary_statistics(
    records: Sequence[CaptionRecord],
    token_counts: Counter[str],
    complete_caption_phrases: set[tuple[str, ...]] | None = None,
) -> dict[str, Any]:
    complete_caption_phrases = complete_caption_phrases or set()
    token_boundary_counts: dict[str, Counter[str]] = defaultdict(Counter)
    token_video_support: Counter[str] = Counter()
    token_last_video: dict[str, str] = {}
    candidate_boundary_counts: dict[tuple[str, ...], list[int]] = {}

    for record in records:
        for token in set(record.tokens):
            if token_last_video.get(token) != record.video_id:
                token_video_support[token] += 1
                token_last_video[token] = record.video_id
        for endpoint, boundary_kind in record.boundaries:
            token_boundary_counts[record.tokens[endpoint]][boundary_kind] += 1
            for size in range(2, MAX_PHRASE_LENGTH + 1):
                start = endpoint - size + 1
                if start < 0:
                    continue
                if start == 0 and endpoint == len(record.tokens) - 1:
                    # Never export a complete short source sentence as a phrase statistic.
                    continue
                phrase = tuple(record.tokens[start : endpoint + 1])
                if phrase in complete_caption_phrases:
                    continue
                counts = candidate_boundary_counts.setdefault(phrase, [0, 0, 0])
                counts[BOUNDARY_INDEX[boundary_kind]] += 1

    candidates = {
        phrase
        for phrase, counts in candidate_boundary_counts.items()
        if sum(counts) >= MIN_PHRASE_BOUNDARY_COUNT
    }
    phrase_support: Counter[tuple[str, ...]] = Counter()
    phrase_video_support: Counter[tuple[str, ...]] = Counter()
    phrase_last_video: dict[tuple[str, ...], str] = {}

    if candidates:
        for record in records:
            for phrase in iter_ngrams(record.tokens, 2, MAX_PHRASE_LENGTH):
                if phrase not in candidates:
                    continue
                phrase_support[phrase] += 1
                if phrase_last_video.get(phrase) != record.video_id:
                    phrase_video_support[phrase] += 1
                    phrase_last_video[phrase] = record.video_id

    token_rows: list[tuple[float, dict[str, Any]]] = []
    for token, kinds in token_boundary_counts.items():
        support = token_counts[token]
        video_support = token_video_support[token]
        boundary_count = sum(kinds.values())
        if support < MIN_TOKEN_BOUNDARY_SUPPORT or video_support < MIN_TOKEN_VIDEO_SUPPORT:
            continue
        probability = boundary_count / support
        score = boundary_count * probability * math.log1p(video_support)
        token_rows.append(
            (
                score,
                {
                    "token": token,
                    "support": support,
                    "videoSupport": video_support,
                    "boundaryCount": boundary_count,
                    "boundaryAfterProbability": round(probability, 6),
                    "boundaryKinds": {
                        kind: kinds.get(kind, 0) for kind in BOUNDARY_NAMES
                    },
                },
            )
        )
    token_rows.sort(
        key=lambda item: (-item[0], -item[1]["support"], item[1]["token"])
    )

    phrase_rows: list[tuple[float, dict[str, Any]]] = []
    for phrase in candidates:
        support = phrase_support[phrase]
        video_support = phrase_video_support[phrase]
        kind_counts = candidate_boundary_counts[phrase]
        boundary_count = sum(kind_counts)
        if support < MIN_PHRASE_SUPPORT or video_support < MIN_PHRASE_VIDEO_SUPPORT:
            continue
        probability = boundary_count / support
        score = boundary_count * probability * math.log1p(video_support)
        phrase_rows.append(
            (
                score,
                {
                    "tokens": list(phrase),
                    "support": support,
                    "videoSupport": video_support,
                    "boundaryCount": boundary_count,
                    "boundaryAfterProbability": round(probability, 6),
                    "boundaryKinds": {
                        kind: kind_counts[index]
                        for index, kind in enumerate(BOUNDARY_NAMES)
                    },
                },
            )
        )
    phrase_rows.sort(
        key=lambda item: (
            -item[0],
            -item[1]["support"],
            tuple(item[1]["tokens"]),
        )
    )

    return {
        "interpretation": (
            "Aggregate English-caption cues for candidate chunk boundaries; these statistics "
            "do not encode ISL syntax or signing order."
        ),
        "boundaryKinds": {
            "strong": ". ! ?",
            "weak": ", ; : -",
            "end": "caption end without trailing boundary punctuation",
        },
        "privacyThresholds": {
            "minimumTokenSupport": MIN_TOKEN_BOUNDARY_SUPPORT,
            "minimumTokenVideoSupport": MIN_TOKEN_VIDEO_SUPPORT,
            "minimumPhraseSupport": MIN_PHRASE_SUPPORT,
            "minimumPhraseVideoSupport": MIN_PHRASE_VIDEO_SUPPORT,
            "maximumPhraseLength": MAX_PHRASE_LENGTH,
            "completeCaptionPhrasesExcluded": True,
        },
        "tokens": [row for _, row in token_rows[:MAX_BOUNDARY_TOKENS]],
        "phrases": [row for _, row in phrase_rows[:MAX_BOUNDARY_PHRASES]],
    }


def select_vocabulary(token_counts: Counter[str]) -> tuple[str, ...]:
    candidates = [
        (token, count) for token, count in token_counts.items() if count >= MIN_TOKEN_COUNT
    ]
    candidates.sort(key=lambda item: (-item[1], item[0]))
    lexical_limit = MAX_VOCABULARY_SIZE - 2
    selected = sorted(token for token, _ in candidates[:lexical_limit])
    return tuple(selected)


def map_token(token: str, vocabulary: set[str]) -> str:
    return token if token in vocabulary else UNKNOWN_TOKEN


def build_language_model(
    records: Sequence[CaptionRecord], token_counts: Counter[str]
) -> dict[str, Any]:
    lexical_vocabulary = select_vocabulary(token_counts)
    lexical_set = set(lexical_vocabulary)
    predicted_counts: Counter[str] = Counter()
    transitions_by_context: dict[str, Counter[str]] = defaultdict(Counter)
    context_counts: Counter[str] = Counter()

    for record in records:
        mapped = [map_token(token, lexical_set) for token in record.tokens]
        predicted_counts.update(mapped)
        predicted_counts[END_TOKEN] += 1
        previous = BEGIN_TOKEN
        for next_token in (*mapped, END_TOKEN):
            transitions_by_context[previous][next_token] += 1
            context_counts[previous] += 1
            previous = next_token

    predicted_vocabulary = (UNKNOWN_TOKEN, END_TOKEN, *lexical_vocabulary)
    denominator = sum(predicted_counts.values()) + UNIGRAM_ALPHA * len(predicted_vocabulary)
    unigram_entries = [
        {
            "token": token,
            "count": predicted_counts[token],
            "probability": (predicted_counts[token] + UNIGRAM_ALPHA) / denominator,
        }
        for token in predicted_vocabulary
    ]

    context_candidates = [
        (context, count)
        for context, count in context_counts.items()
        if count >= MIN_CONTEXT_COUNT or context == BEGIN_TOKEN
    ]
    context_candidates.sort(key=lambda item: (-item[1], item[0]))
    selected_contexts = context_candidates[:MAX_CONTEXTS]
    if BEGIN_TOKEN not in {context for context, _ in selected_contexts}:
        selected_contexts[-1:] = [(BEGIN_TOKEN, context_counts[BEGIN_TOKEN])]
        selected_contexts.sort(key=lambda item: (-item[1], item[0]))

    contexts: list[dict[str, Any]] = []
    for context, context_count in selected_contexts:
        transitions = list(transitions_by_context[context].items())
        transitions.sort(key=lambda item: (-item[1], item[0]))
        retained = [
            item for item in transitions if item[1] >= MIN_TRANSITION_COUNT
        ][:MAX_TRANSITIONS_PER_CONTEXT]
        if not retained and transitions:
            retained = transitions[:1]
        retained_total = sum(count for _, count in retained)
        if retained_total == 0:
            continue
        stored_mass = retained_total / context_count
        reliability = context_count / (context_count + BACKOFF_STRENGTH)
        context_weight = min(MAX_CONTEXT_WEIGHT, reliability * stored_mass)
        contexts.append(
            {
                "context": context,
                "count": context_count,
                "contextWeight": context_weight,
                "storedTransitionMass": stored_mass,
                "transitions": [
                    {
                        "token": next_token,
                        "count": count,
                        "conditionalWeight": count / retained_total,
                    }
                    for next_token, count in retained
                ],
            }
        )

    return {
        "type": "compact_interpolated_backoff_bigram",
        "input": "normalized English-caption lexical tokens",
        "smoothing": {
            "unigramAdditiveAlpha": UNIGRAM_ALPHA,
            "contextBackoffStrength": BACKOFF_STRENGTH,
            "maximumContextWeight": MAX_CONTEXT_WEIGHT,
        },
        "limits": {
            "maximumVocabularySize": MAX_VOCABULARY_SIZE,
            "minimumTokenCount": MIN_TOKEN_COUNT,
            "maximumContexts": MAX_CONTEXTS,
            "maximumTransitionsPerContext": MAX_TRANSITIONS_PER_CONTEXT,
            "minimumTransitionCount": MIN_TRANSITION_COUNT,
        },
        "specialTokens": {
            "begin": BEGIN_TOKEN,
            "end": END_TOKEN,
            "unknown": UNKNOWN_TOKEN,
        },
        "vocabularySize": len(predicted_vocabulary),
        "unigrams": unigram_entries,
        "contexts": contexts,
    }


def language_model_indexes(
    model: Mapping[str, Any],
) -> tuple[set[str], dict[str, float], dict[str, tuple[float, dict[str, float]]]]:
    unigrams = {entry["token"]: float(entry["probability"]) for entry in model["unigrams"]}
    lexical_vocabulary = set(unigrams) - {UNKNOWN_TOKEN, END_TOKEN}
    contexts: dict[str, tuple[float, dict[str, float]]] = {}
    for row in model["contexts"]:
        contexts[row["context"]] = (
            float(row["contextWeight"]),
            {
                transition["token"]: float(transition["conditionalWeight"])
                for transition in row["transitions"]
            },
        )
    return lexical_vocabulary, unigrams, contexts


def evaluate_records(
    records: Sequence[CaptionRecord],
    model: Mapping[str, Any],
    boundary_statistics: Mapping[str, Any],
) -> dict[str, Any]:
    lexical_vocabulary, unigrams, contexts = language_model_indexes(model)
    boundary_tokens = {row["token"] for row in boundary_statistics["tokens"]}
    boundary_phrases = {
        tuple(row["tokens"]) for row in boundary_statistics["phrases"]
    }
    lexical_token_count = 0
    covered_lexical_token_count = 0
    all_tokens_covered_caption_count = 0
    predicted_token_count = 0
    negative_log_likelihood = 0.0
    boundary_count = 0
    covered_boundary_count = 0

    for record in records:
        mapped = [map_token(token, lexical_vocabulary) for token in record.tokens]
        lexical_token_count += len(record.tokens)
        known_count = sum(token in lexical_vocabulary for token in record.tokens)
        covered_lexical_token_count += known_count
        if known_count == len(record.tokens):
            all_tokens_covered_caption_count += 1

        previous = BEGIN_TOKEN
        for next_token in (*mapped, END_TOKEN):
            unigram_probability = unigrams[next_token]
            context_weight, transitions = contexts.get(previous, (0.0, {}))
            probability = (1.0 - context_weight) * unigram_probability
            probability += context_weight * transitions.get(next_token, 0.0)
            if probability <= 0.0:
                raise PlannerTrainingError("The exported model assigned zero probability.")
            negative_log_likelihood -= math.log(probability)
            predicted_token_count += 1
            previous = next_token

        for endpoint, _ in record.boundaries:
            boundary_count += 1
            token = record.tokens[endpoint]
            covered = token in boundary_tokens
            if not covered:
                for size in range(2, MAX_PHRASE_LENGTH + 1):
                    start = endpoint - size + 1
                    if start >= 0 and tuple(record.tokens[start : endpoint + 1]) in boundary_phrases:
                        covered = True
                        break
            covered_boundary_count += int(covered)

    if predicted_token_count == 0 or lexical_token_count == 0:
        raise PlannerTrainingError("Held-out evaluation requires non-empty lexical text.")
    caption_count = len(records)
    return {
        "captionCount": caption_count,
        "lexicalTokenCount": lexical_token_count,
        "predictedTokenCountIncludingEnd": predicted_token_count,
        "perplexityIncludingEnd": round(
            math.exp(negative_log_likelihood / predicted_token_count), 6
        ),
        "lexicalTokenCoverage": round(
            covered_lexical_token_count / lexical_token_count, 6
        ),
        "fullyCoveredCaptionRate": round(
            all_tokens_covered_caption_count / caption_count, 6
        ),
        "exportedBoundaryFeatureCoverage": round(
            covered_boundary_count / boundary_count, 6
        )
        if boundary_count
        else 0.0,
        "boundaryCount": boundary_count,
    }


def load_auxiliary_coverage(path: Path, vocabulary: set[str]) -> dict[str, Any]:
    row_count = 0
    nonblank_sentence_count = 0
    lexical_token_count = 0
    covered_token_count = 0
    fully_covered_count = 0

    with path.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        if reader.fieldnames is None or "sentence" not in reader.fieldnames:
            raise PlannerTrainingError(f"{path} must contain a sentence column.")
        for row in reader:
            row_count += 1
            sentence = row.get("sentence") or ""
            _, tokens, _ = analyze_text(sentence)
            if not tokens:
                continue
            nonblank_sentence_count += 1
            lexical_token_count += len(tokens)
            known_count = sum(token in vocabulary for token in tokens)
            covered_token_count += known_count
            fully_covered_count += int(known_count == len(tokens))

    return {
        "role": "coverage diagnostic only; never used to fit the model or held-out score",
        "sourceRowCount": row_count,
        "nonblankSentenceCount": nonblank_sentence_count,
        "lexicalTokenCount": lexical_token_count,
        "lexicalTokenCoverage": round(covered_token_count / lexical_token_count, 6)
        if lexical_token_count
        else 0.0,
        "fullyCoveredSentenceRate": round(fully_covered_count / nonblank_sentence_count, 6)
        if nonblank_sentence_count
        else 0.0,
    }


def source_provenance(input_dir: Path) -> dict[str, Any]:
    files: list[dict[str, Any]] = []
    aggregate = hashlib.sha256()
    for filename, role in SOURCE_FILES:
        path = input_dir / filename
        if not path.is_file():
            raise PlannerTrainingError(f"Required private iSign input is missing: {path}")
        digest = sha256_file(path)
        relative_path = f"data/raw/isign/{filename}"
        aggregate.update(relative_path.encode("utf-8"))
        aggregate.update(b"\0")
        aggregate.update(bytes.fromhex(digest))
        files.append(
            {
                "path": relative_path,
                "role": role,
                "byteLength": path.stat().st_size,
                "sha256": digest,
            }
        )
    return {"aggregateSha256": aggregate.hexdigest(), "files": files}


def split_summary(partition: PartitionedCorpus) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for split in SPLIT_ORDER:
        raw_records = partition.raw[split]
        retained_records = partition.retained[split]
        summary[split] = {
            "videoCount": len({record.video_id for record in raw_records}),
            "sourceCaptionRows": len(raw_records),
            "retainedUniqueCaptionRows": len(retained_records),
            "withinSplitExactDuplicatesExcluded": partition.within_split_duplicates[split],
            "earlierSplitExactDuplicatesExcluded": partition.cross_split_duplicates[split],
            "lexicalTokenCount": sum(len(record.tokens) for record in retained_records),
        }
    return summary


def training_configuration(seed: str) -> dict[str, Any]:
    return {
        "seed": seed,
        "splitRatios": {"train": 0.8, "validation": 0.1, "test": 0.1},
        "splitAssignment": (
            "SHA-256(seed + NUL + video group identifier), first 64 bits mapped to fixed ranges"
        ),
        "splitUnit": "source video group; all numeric sequence rows from one video stay together",
        "decontamination": (
            "exact normalized caption deduplication with train, validation, test precedence"
        ),
        "tokenization": (
            "Unicode NFKC and casefold; lexical tokens plus strong/weak punctuation boundaries"
        ),
        "trainingInput": "numeric-sequence rows from iSign_v1.1.csv only",
        "auxiliaryCsvUse": "coverage diagnostics only; no fitting and no held-out scoring",
    }


def build_artifact(
    input_dir: Path,
    seed: str = DEFAULT_SEED,
    trainer_path: Path | None = None,
) -> dict[str, Any]:
    input_dir = input_dir.resolve()
    trainer_path = (trainer_path or Path(__file__)).resolve()
    provenance = source_provenance(input_dir)
    loaded = load_caption_corpus(input_dir / "iSign_v1.1.csv")
    partition = partition_corpus(loaded.records, seed)
    training_records = partition.retained["train"]
    token_counts = Counter(token for record in training_records for token in record.tokens)
    complete_caption_phrases = {
        record.tokens
        for record in loaded.records
        if 2 <= len(record.tokens) <= MAX_PHRASE_LENGTH
    }
    boundary_statistics = build_boundary_statistics(
        training_records, token_counts, complete_caption_phrases
    )
    model = build_language_model(training_records, token_counts)
    lexical_vocabulary, _, _ = language_model_indexes(model)
    configuration = training_configuration(seed)
    configuration_sha256 = sha256_bytes(canonical_json_bytes(configuration))

    validation_metrics = evaluate_records(
        partition.retained["validation"], model, boundary_statistics
    )
    test_metrics = evaluate_records(partition.retained["test"], model, boundary_statistics)
    auxiliary_metrics = {
        "wordDescription": load_auxiliary_coverage(
            input_dir / "word-description-dataset_v1.1.csv", lexical_vocabulary
        ),
        "wordPresence": load_auxiliary_coverage(
            input_dir / "word-presence-dataset_v1.1.csv", lexical_vocabulary
        ),
    }

    partition_counts = split_summary(partition)
    retained_total = sum(
        partition_counts[split]["retainedUniqueCaptionRows"] for split in SPLIT_ORDER
    )
    artifact: dict[str, Any] = {
        "schemaVersion": SCHEMA_VERSION,
        "artifactId": "isign-sentence-planner.private",
        "visibility": "private",
        "purpose": (
            "Deterministic research-only English-caption planning statistics for candidate "
            "chunk boundaries and compact next-token plausibility."
        ),
        "claims": {
            "isTextOnly": True,
            "isIslGrammarModel": False,
            "isSignLanguageTranslationModel": False,
            "isText2PoseModel": False,
            "isPoseOrMotionModel": False,
        },
        "source": {
            **provenance,
            "declaredLicense": "CC BY-NC-SA 4.0 in the local iSign README",
            "accessBoundary": (
                "The local dataset terms describe non-commercial research use and require "
                "that the gated dataset is not shared or uploaded elsewhere."
            ),
        },
        "distribution": {
            "use": "research-only and non-commercial",
            "artifactVisibility": "private local file",
            "redistribution": "prohibited unless the source terms and permission are re-reviewed",
            "publicHuggingFaceUpload": False,
            "containsSourceMedia": False,
            "containsPoseOrLandmarkData": False,
        },
        "privacy": {
            "rawSentencesIncluded": False,
            "sourceIdentifiersIncluded": False,
            "videoIdentifiersIncluded": False,
            "aggregateTokenStatisticsIncluded": True,
            "aggregateMultiVideoPhraseStatisticsIncluded": True,
            "minimumPhraseVideoSupport": MIN_PHRASE_VIDEO_SUPPORT,
        },
        "trainer": {
            "path": "scripts/train_isign_sentence_planner.py",
            "sha256": sha256_file(trainer_path),
            "runtimeDependencies": ["Python standard library"],
            "configurationSha256": configuration_sha256,
        },
        "training": {
            **configuration,
            "sourceCorpus": loaded.counts,
            "splits": partition_counts,
            "retainedUniqueCaptionRows": retained_total,
            "leakageChecks": {
                "videoGroupOverlapAcrossSplits": 0,
                "exactNormalizedCaptionOverlapAcrossRetainedSplits": 0,
                "status": "passed",
            },
        },
        "model": model,
        "chunkBoundaryStatistics": boundary_statistics,
        "evaluation": {
            "heldOut": {
                "validation": validation_metrics,
                "test": test_metrics,
            },
            "auxiliaryCoverageDiagnostics": auxiliary_metrics,
            "metricNotes": {
                "perplexityIncludingEnd": (
                    "Natural-exponential perplexity from the exported compact model, including "
                    "one end token per retained caption. Lower is better only for this frozen corpus."
                ),
                "lexicalTokenCoverage": (
                    "Fraction of held-out lexical token occurrences present in the training vocabulary."
                ),
                "exportedBoundaryFeatureCoverage": (
                    "Fraction of held-out punctuation/end boundaries whose preceding token or suffix "
                    "phrase is present in the exported aggregate boundary tables."
                ),
            },
        },
        "limitations": [
            "This is a statistical English-caption planner, not an ISL grammar model.",
            "This is not a sign-language translation or Text2Pose model.",
            "No video, pose, landmarks, signer features, or motion supervision were used.",
            "Chunk cues reflect punctuation and frequency in the source English text, not ISL syntax.",
            "Held-out scores are corpus diagnostics and do not measure signing or translation quality.",
            "The private artifact must not be copied into public release or Hugging Face output.",
        ],
    }
    validate_artifact(artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    claims = artifact.get("claims", {})
    if claims.get("isIslGrammarModel") is not False or claims.get("isText2PoseModel") is not False:
        raise PlannerTrainingError("The artifact must explicitly reject grammar and Text2Pose claims.")
    privacy = artifact.get("privacy", {})
    if privacy.get("rawSentencesIncluded") is not False:
        raise PlannerTrainingError("The artifact privacy boundary must exclude raw sentences.")
    if privacy.get("sourceIdentifiersIncluded") is not False:
        raise PlannerTrainingError("The artifact privacy boundary must exclude source identifiers.")
    leakage = artifact.get("training", {}).get("leakageChecks", {})
    if leakage.get("status") != "passed" or any(
        leakage.get(key) != 0
        for key in (
            "videoGroupOverlapAcrossSplits",
            "exactNormalizedCaptionOverlapAcrossRetainedSplits",
        )
    ):
        raise PlannerTrainingError("The exported leakage checks did not pass.")
    if artifact.get("distribution", {}).get("publicHuggingFaceUpload") is not False:
        raise PlannerTrainingError("The private artifact cannot allow public Hugging Face upload.")


def write_artifact(path: Path, artifact: Mapping[str, Any]) -> tuple[int, str]:
    payload = canonical_json_bytes(artifact)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", prefix=f".{path.name}.", dir=path.parent, delete=False
        ) as temporary:
            temporary.write(payload)
            temporary.flush()
            os.fsync(temporary.fileno())
            temporary_name = temporary.name
        os.replace(temporary_name, path)
        temporary_name = None
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)
    return len(payload), sha256_bytes(payload)


def parse_args() -> argparse.Namespace:
    workspace_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=workspace_root / "data/raw/isign",
        help="Private iSign text directory (default: data/raw/isign).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=workspace_root / "data/models/isl-sentence-planner.private.json",
        help="Private local artifact path.",
    )
    parser.add_argument("--seed", default=DEFAULT_SEED)
    parser.add_argument(
        "--verify-determinism",
        action="store_true",
        help="Build twice in memory and fail unless both canonical payloads are byte-identical.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    artifact = build_artifact(args.input_dir, args.seed)
    if args.verify_determinism:
        repeated = build_artifact(args.input_dir, args.seed)
        if canonical_json_bytes(artifact) != canonical_json_bytes(repeated):
            raise PlannerTrainingError("Repeated training was not byte-deterministic.")
    byte_length, artifact_sha256 = write_artifact(args.output, artifact)
    splits = artifact["training"]["splits"]
    held_out = artifact["evaluation"]["heldOut"]
    print(f"Wrote private iSign sentence planner: {args.output.resolve()}")
    print(f"Artifact bytes: {byte_length}; SHA-256: {artifact_sha256}")
    print(
        "Prepared captions: "
        f"{artifact['training']['sourceCorpus']['preparedCaptionRows']}; "
        f"videos: {artifact['training']['sourceCorpus']['videoCount']}"
    )
    for split in SPLIT_ORDER:
        row = splits[split]
        print(
            f"{split}: videos={row['videoCount']}, source_rows={row['sourceCaptionRows']}, "
            f"retained_rows={row['retainedUniqueCaptionRows']}, tokens={row['lexicalTokenCount']}"
        )
    for split in ("validation", "test"):
        row = held_out[split]
        print(
            f"{split}: perplexity={row['perplexityIncludingEnd']}, "
            f"token_coverage={row['lexicalTokenCoverage']}, "
            f"boundary_coverage={row['exportedBoundaryFeatureCoverage']}"
        )
    if args.verify_determinism:
        print("Determinism check: passed (two byte-identical in-memory builds)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
