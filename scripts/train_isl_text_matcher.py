#!/usr/bin/env python3
"""Train a deterministic, standard-library-only lexical pair matcher."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import tempfile
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = 1
FEATURE_NAMES = (
    "exact",
    "editSimilarity",
    "trigramJaccard",
    "prefixRatio",
    "lengthRatio",
    "tokenJaccard",
)
SPLITS = ("train", "val", "test")
INPUT_FILES = {
    "train": "match_pairs/train.jsonl",
    "val": "match_pairs/val.jsonl",
    "test": "match_pairs/test.jsonl",
}
MINIMUM_QUERY_LENGTH = 4
TRAINED_AT = "2026-07-16T00:00:00.000Z"
PRECISION_TARGET = 0.98
TEST_TOP1_TARGET = 0.95
MINIMUM_MARGIN = 0.01


class MatcherTrainingError(RuntimeError):
    """Raised when matcher inputs or outputs violate the training contract."""


@dataclass(frozen=True)
class PairRow:
    group_id: str
    query: str
    candidate_id: str
    candidate_term: str
    label: int
    split: str
    source_type: str
    has_motion: bool
    features: tuple[float, ...]


@dataclass(frozen=True)
class OptimizerConfig:
    iterations: int = 500
    learning_rate: float = 0.2
    l2: float = 0.01
    class_weight_imbalance_ratio: float = 1.25


@dataclass(frozen=True)
class FittedModel:
    means: tuple[float, ...]
    scales: tuple[float, ...]
    coefficients: tuple[float, ...]
    intercept: float
    class_weights: Mapping[int, float]
    class_weight_strategy: str
    collapsed_training_vectors: int
    final_weighted_log_loss: float


@dataclass(frozen=True)
class LoadedDataset:
    rows: Mapping[str, tuple[PairRow, ...]]
    data_hashes: Mapping[str, Any]
    integrity: Mapping[str, Any]


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True) + "\n"
    ).encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _clean_float(value: float, digits: int = 12) -> float:
    rounded = round(value, digits)
    return 0.0 if rounded == 0 else rounded


def normalize_text(text: str) -> str:
    """Normalize matching text without transliterating or emitting replacement text."""

    normalized = unicodedata.normalize("NFKC", text).casefold()
    characters = [
        character if unicodedata.category(character)[0] in {"L", "M", "N"} else " "
        for character in normalized
    ]
    return " ".join("".join(characters).split())


def _levenshtein_distance(left: str, right: str) -> int:
    if len(left) < len(right):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for left_index, left_character in enumerate(left, 1):
        current = [left_index]
        for right_index, right_character in enumerate(right, 1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[right_index] + 1,
                    previous[right_index - 1] + (left_character != right_character),
                )
            )
        previous = current
    return previous[-1]


def _trigrams(text: str) -> set[str]:
    if len(text) < 3:
        return {text} if text else set()
    return {text[index : index + 3] for index in range(len(text) - 2)}


def _jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 1.0


def extract_features(query: str, candidate_term: str) -> tuple[float, ...]:
    normalized_query = normalize_text(query)
    normalized_candidate = normalize_text(candidate_term)
    if not normalized_query or not normalized_candidate:
        raise MatcherTrainingError("Matcher features require non-empty normalized text.")

    maximum_length = max(len(normalized_query), len(normalized_candidate))
    common_prefix = 0
    for left, right in zip(normalized_query, normalized_candidate):
        if left != right:
            break
        common_prefix += 1

    return (
        float(normalized_query == normalized_candidate),
        1.0
        - _levenshtein_distance(normalized_query, normalized_candidate) / maximum_length,
        _jaccard(_trigrams(normalized_query), _trigrams(normalized_candidate)),
        common_prefix / maximum_length,
        min(len(normalized_query), len(normalized_candidate)) / maximum_length,
        _jaccard(set(normalized_query.split()), set(normalized_candidate.split())),
    )


def _required_string(row: Mapping[str, Any], field: str, location: str) -> str:
    value = row.get(field)
    if not isinstance(value, str) or not value.strip():
        raise MatcherTrainingError(f"{location}: {field} must be a non-empty string.")
    return value.strip()


def _parse_label(value: Any, location: str) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int) and value in (0, 1):
        return value
    raise MatcherTrainingError(f"{location}: label must be 0 or 1.")


def _load_split(path: Path, expected_split: str) -> tuple[PairRow, ...]:
    try:
        lines = path.read_text(encoding="utf-8-sig").splitlines()
    except UnicodeDecodeError as error:
        raise MatcherTrainingError(f"{path} is not valid UTF-8 JSONL.") from error
    if not lines:
        raise MatcherTrainingError(f"{path} is empty.")

    rows: list[PairRow] = []
    for line_number, line in enumerate(lines, 1):
        location = f"{path}:{line_number}"
        if not line.strip():
            raise MatcherTrainingError(f"{location}: blank JSONL rows are not allowed.")
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as error:
            raise MatcherTrainingError(f"{location}: invalid JSON: {error.msg}.") from error
        if not isinstance(raw, dict):
            raise MatcherTrainingError(f"{location}: each JSONL row must be an object.")

        split = _required_string(raw, "split", location)
        if split != expected_split:
            raise MatcherTrainingError(
                f"{location}: row split {split!r} does not match {expected_split!r}."
            )
        query = _required_string(raw, "query", location)
        candidate_term = _required_string(raw, "candidateTerm", location)
        has_motion = raw.get("hasMotion")
        if type(has_motion) is not bool:
            raise MatcherTrainingError(f"{location}: hasMotion must be a boolean.")

        rows.append(
            PairRow(
                group_id=_required_string(raw, "groupId", location),
                query=normalize_text(query),
                candidate_id=_required_string(raw, "candidateId", location),
                candidate_term=normalize_text(candidate_term),
                label=_parse_label(raw.get("label"), location),
                split=split,
                source_type=_required_string(raw, "sourceType", location),
                has_motion=has_motion,
                features=extract_features(query, candidate_term),
            )
        )

    return tuple(
        sorted(
            rows,
            key=lambda row: (
                row.group_id,
                row.candidate_id,
                row.query,
                row.candidate_term,
                row.label,
            ),
        )
    )


def validate_split_group_integrity(
    rows_by_split: Mapping[str, Sequence[PairRow]],
) -> dict[str, Any]:
    group_splits: dict[str, str] = {}
    query_splits: dict[str, str] = {}
    candidate_catalog: dict[str, tuple[str, bool]] = {}
    summaries: dict[str, Any] = {}

    for split in SPLITS:
        rows = rows_by_split.get(split, ())
        if not rows:
            raise MatcherTrainingError(f"The {split} split has no rows.")
        if {row.label for row in rows} != {0, 1}:
            raise MatcherTrainingError(f"The {split} split must contain both labels.")

        groups: dict[str, list[PairRow]] = defaultdict(list)
        for row in rows:
            previous_split = group_splits.setdefault(row.group_id, split)
            if previous_split != split:
                raise MatcherTrainingError(
                    f"Group leakage detected for {row.group_id!r} between "
                    f"{previous_split} and {split}."
                )
            groups[row.group_id].append(row)

            # sourceType describes the query variant, not the candidate record.
            catalog_value = (row.candidate_term, row.has_motion)
            previous_catalog_value = candidate_catalog.setdefault(
                row.candidate_id, catalog_value
            )
            if previous_catalog_value != catalog_value:
                raise MatcherTrainingError(
                    f"Candidate {row.candidate_id!r} has inconsistent metadata."
                )

        seen_queries: dict[str, str] = {}
        positive_rows = 0
        for group_id, group_rows in sorted(groups.items()):
            queries = {row.query for row in group_rows}
            if len(queries) != 1:
                raise MatcherTrainingError(
                    f"Group {group_id!r} contains multiple normalized queries."
                )
            query = next(iter(queries))
            previous_group = seen_queries.setdefault(query, group_id)
            if previous_group != group_id:
                raise MatcherTrainingError(
                    f"Normalized query {query!r} occurs in multiple {split} groups."
                )
            previous_query_split = query_splits.setdefault(query, split)
            if previous_query_split != split:
                raise MatcherTrainingError(
                    f"Normalized-query leakage detected between {previous_query_split} "
                    f"and {split}: {query!r}."
                )

            candidate_ids = [row.candidate_id for row in group_rows]
            if len(candidate_ids) != len(set(candidate_ids)):
                raise MatcherTrainingError(
                    f"Group {group_id!r} contains a duplicate candidateId."
                )
            positives = sum(row.label for row in group_rows)
            negatives = len(group_rows) - positives
            if positives == 0 or negatives == 0:
                raise MatcherTrainingError(
                    f"Group {group_id!r} must contain at least one positive and one negative."
                )
            positive_rows += positives

        summaries[split] = {
            "rowCount": len(rows),
            "groupCount": len(groups),
            "positiveRowCount": positive_rows,
            "negativeRowCount": len(rows) - positive_rows,
        }

    return {
        "status": "passed",
        "groupOverlapAcrossSplits": 0,
        "normalizedQueryOverlapAcrossSplits": 0,
        "duplicateCandidateIdsWithinGroups": 0,
        "inconsistentCandidateMetadata": 0,
        "splits": summaries,
        "uniqueCandidateCount": len(candidate_catalog),
    }


def load_dataset(input_dir: Path) -> LoadedDataset:
    data_dir = input_dir.resolve() / "data"
    rows_by_split: dict[str, tuple[PairRow, ...]] = {}
    files: list[dict[str, Any]] = []
    aggregate = hashlib.sha256()

    for split in SPLITS:
        filename = INPUT_FILES[split]
        path = data_dir / filename
        if not path.is_file():
            raise MatcherTrainingError(f"Required matcher input is missing: {path}")
        digest = sha256_file(path)
        relative_path = f"data/{filename}"
        rows = _load_split(path, split)
        rows_by_split[split] = rows
        aggregate.update(relative_path.encode("utf-8"))
        aggregate.update(b"\0")
        aggregate.update(bytes.fromhex(digest))
        files.append(
            {
                "path": relative_path,
                "sha256": digest,
                "byteLength": path.stat().st_size,
                "rowCount": len(rows),
            }
        )

    integrity = validate_split_group_integrity(rows_by_split)
    return LoadedDataset(
        rows={split: rows_by_split[split] for split in SPLITS},
        data_hashes={"aggregateSha256": aggregate.hexdigest(), "files": files},
        integrity=integrity,
    )


def _standardization(
    rows: Sequence[PairRow],
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    count = len(rows)
    means = tuple(
        math.fsum(row.features[index] for row in rows) / count
        for index in range(len(FEATURE_NAMES))
    )
    scales = []
    for index, mean in enumerate(means):
        variance = math.fsum(
            (row.features[index] - mean) ** 2 for row in rows
        ) / count
        scale = math.sqrt(variance)
        scales.append(scale if scale > 1e-12 else 1.0)
    return means, tuple(scales)


def _sigmoid(value: float) -> float:
    if value >= 0:
        decay = math.exp(-value)
        return 1.0 / (1.0 + decay)
    growth = math.exp(value)
    return growth / (1.0 + growth)


def _weighted_log_loss(
    vectors: Sequence[tuple[tuple[float, ...], int, int]],
    coefficients: Sequence[float],
    intercept: float,
    class_weights: Mapping[int, float],
    l2: float,
) -> float:
    weighted_loss = 0.0
    total_weight = 0.0
    for values, label, count in vectors:
        score = intercept + math.fsum(
            coefficient * value for coefficient, value in zip(coefficients, values)
        )
        loss = max(score, 0.0) - score * label + math.log1p(math.exp(-abs(score)))
        weight = class_weights[label] * count
        weighted_loss += weight * loss
        total_weight += weight
    penalty = 0.5 * l2 * math.fsum(value * value for value in coefficients)
    return weighted_loss / total_weight + penalty


def fit_model(
    rows: Sequence[PairRow], config: OptimizerConfig = OptimizerConfig()
) -> FittedModel:
    if config.iterations <= 0 or config.learning_rate <= 0 or config.l2 < 0:
        raise MatcherTrainingError("Optimizer settings must be finite and non-negative.")
    if not all(
        math.isfinite(value)
        for value in (config.learning_rate, config.l2, config.class_weight_imbalance_ratio)
    ):
        raise MatcherTrainingError("Optimizer settings must be finite.")

    label_counts = Counter(row.label for row in rows)
    if set(label_counts) != {0, 1}:
        raise MatcherTrainingError("Training requires both positive and negative rows.")
    imbalance = max(label_counts.values()) / min(label_counts.values())
    if imbalance >= config.class_weight_imbalance_ratio:
        total = len(rows)
        class_weights = {
            label: total / (2.0 * label_counts[label]) for label in (0, 1)
        }
        class_weight_strategy = "balanced"
    else:
        class_weights = {0: 1.0, 1: 1.0}
        class_weight_strategy = "none"

    means, scales = _standardization(rows)
    collapsed = Counter((row.features, row.label) for row in rows)
    vectors = []
    for (features, label), count in sorted(collapsed.items()):
        standardized = tuple(
            (features[index] - means[index]) / scales[index]
            for index in range(len(FEATURE_NAMES))
        )
        vectors.append((standardized, label, count))

    weighted_positive = class_weights[1] * label_counts[1]
    total_weight = math.fsum(
        class_weights[label] * count for _, label, count in vectors
    )
    prior = min(max(weighted_positive / total_weight, 1e-9), 1.0 - 1e-9)
    coefficients = [0.0] * len(FEATURE_NAMES)
    intercept = math.log(prior / (1.0 - prior))

    for _ in range(config.iterations):
        coefficient_gradients = [0.0] * len(FEATURE_NAMES)
        intercept_gradient = 0.0
        for values, label, count in vectors:
            linear = intercept + math.fsum(
                coefficient * value
                for coefficient, value in zip(coefficients, values)
            )
            weighted_error = (
                (_sigmoid(linear) - label) * class_weights[label] * count
            )
            intercept_gradient += weighted_error
            for index, value in enumerate(values):
                coefficient_gradients[index] += weighted_error * value

        intercept -= config.learning_rate * intercept_gradient / total_weight
        for index in range(len(coefficients)):
            gradient = (
                coefficient_gradients[index] / total_weight
                + config.l2 * coefficients[index]
            )
            coefficients[index] -= config.learning_rate * gradient

    clean_means = tuple(_clean_float(value) for value in means)
    clean_scales = tuple(_clean_float(value) for value in scales)
    clean_coefficients = tuple(_clean_float(value) for value in coefficients)
    clean_intercept = _clean_float(intercept)
    final_loss = _weighted_log_loss(
        vectors,
        clean_coefficients,
        clean_intercept,
        class_weights,
        config.l2,
    )
    return FittedModel(
        means=clean_means,
        scales=clean_scales,
        coefficients=clean_coefficients,
        intercept=clean_intercept,
        class_weights={label: _clean_float(class_weights[label]) for label in (0, 1)},
        class_weight_strategy=class_weight_strategy,
        collapsed_training_vectors=len(vectors),
        final_weighted_log_loss=_clean_float(final_loss),
    )


def predict_probability(features: Sequence[float], model: FittedModel) -> float:
    standardized = (
        (features[index] - model.means[index]) / model.scales[index]
        for index in range(len(FEATURE_NAMES))
    )
    linear = model.intercept + math.fsum(
        coefficient * value
        for coefficient, value in zip(model.coefficients, standardized)
    )
    return _clean_float(_sigmoid(linear))


def _binary_metrics(
    labels: Sequence[int], scores: Sequence[float], threshold: float
) -> dict[str, Any]:
    true_positive = false_positive = true_negative = false_negative = 0
    for label, score in zip(labels, scores):
        predicted = score >= threshold
        if predicted and label:
            true_positive += 1
        elif predicted:
            false_positive += 1
        elif label:
            false_negative += 1
        else:
            true_negative += 1

    precision = (
        true_positive / (true_positive + false_positive)
        if true_positive + false_positive
        else 0.0
    )
    recall = (
        true_positive / (true_positive + false_negative)
        if true_positive + false_negative
        else 0.0
    )
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": _clean_float(precision),
        "recall": _clean_float(recall),
        "f1": _clean_float(f1),
        "truePositive": true_positive,
        "falsePositive": false_positive,
        "trueNegative": true_negative,
        "falseNegative": false_negative,
        "support": len(labels),
        "predictedPositive": true_positive + false_positive,
    }


def select_probability_threshold(
    rows: Sequence[PairRow], scores: Sequence[float]
) -> tuple[float, dict[str, Any]]:
    ranked = sorted(
        ((score, row.label) for row, score in zip(rows, scores)), reverse=True
    )
    total_positive = sum(row.label for row in rows)
    candidates: list[tuple[float, dict[str, Any]]] = []
    true_positive = false_positive = 0
    index = 0
    while index < len(ranked):
        score = ranked[index][0]
        while index < len(ranked) and ranked[index][0] == score:
            if ranked[index][1]:
                true_positive += 1
            else:
                false_positive += 1
            index += 1
        next_score = ranked[index][0] if index < len(ranked) else None
        threshold = score if next_score is None else (score + next_score) / 2.0
        false_negative = total_positive - true_positive
        true_negative = len(rows) - total_positive - false_positive
        precision = true_positive / (true_positive + false_positive)
        recall = true_positive / total_positive
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
        candidates.append(
            (
                _clean_float(threshold),
                {
                    "precision": _clean_float(precision),
                    "recall": _clean_float(recall),
                    "f1": _clean_float(f1),
                    "truePositive": true_positive,
                    "falsePositive": false_positive,
                    "trueNegative": true_negative,
                    "falseNegative": false_negative,
                    "support": len(rows),
                    "predictedPositive": true_positive + false_positive,
                },
            )
        )

    feasible = [
        candidate
        for candidate in candidates
        if candidate[1]["precision"] >= PRECISION_TARGET
        and candidate[1]["truePositive"] > 0
    ]
    pool = feasible or candidates
    threshold, metrics = max(
        pool,
        key=lambda candidate: (
            candidate[1]["f1"],
            candidate[1]["precision"],
            candidate[1]["recall"],
            candidate[0],
        ),
    )
    return threshold, {
        "split": "val",
        "objective": "maximize F1 subject to precision >= 0.98 when feasible",
        "precisionConstraint": PRECISION_TARGET,
        "precisionConstraintFeasible": bool(feasible),
        "candidateThresholdCount": len(candidates),
        "selectedMetrics": metrics,
    }


def _ranked_groups(
    rows: Sequence[PairRow], scores: Sequence[float]
) -> list[list[tuple[PairRow, float]]]:
    grouped: dict[str, list[tuple[PairRow, float]]] = defaultdict(list)
    for row, score in zip(rows, scores):
        grouped[row.group_id].append((row, score))
    return [
        sorted(
            grouped[group_id],
            key=lambda item: (-item[1], item[0].candidate_id, item[0].candidate_term),
        )
        for group_id in sorted(grouped)
    ]


def _grouped_metrics(
    rows: Sequence[PairRow], scores: Sequence[float]
) -> dict[str, Any]:
    groups = _ranked_groups(rows, scores)
    correct_top1 = 0
    reciprocal_rank = 0.0
    for ranked in groups:
        correct_top1 += ranked[0][0].label
        positive_rank = next(
            index for index, (row, _) in enumerate(ranked, 1) if row.label
        )
        reciprocal_rank += 1.0 / positive_rank
    return {
        "top1": _clean_float(correct_top1 / len(groups)),
        "mrr": _clean_float(reciprocal_rank / len(groups)),
        "groupCount": len(groups),
        "correctTop1": correct_top1,
    }


def select_margin_threshold(
    rows: Sequence[PairRow], scores: Sequence[float], probability_threshold: float
) -> tuple[float, dict[str, Any]]:
    groups = _ranked_groups(rows, scores)
    outcomes = []
    for ranked in groups:
        top_score = ranked[0][1]
        second_score = ranked[1][1]
        outcomes.append((top_score - second_score, bool(ranked[0][0].label), top_score))

    candidate_margins = sorted(
        {MINIMUM_MARGIN, *(max(MINIMUM_MARGIN, margin) for margin, _, _ in outcomes)}
    )
    candidates = []
    for margin_threshold in candidate_margins:
        accepted = [
            correct
            for margin, correct, top_score in outcomes
            if top_score >= probability_threshold and margin >= margin_threshold
        ]
        if not accepted:
            continue
        precision = sum(accepted) / len(accepted)
        candidates.append(
            (
                _clean_float(margin_threshold),
                {
                    "acceptedGroupCount": len(accepted),
                    "acceptedPrecision": _clean_float(precision),
                    "coverage": _clean_float(len(accepted) / len(groups)),
                },
            )
        )

    feasible = [
        candidate
        for candidate in candidates
        if candidate[1]["acceptedPrecision"] >= PRECISION_TARGET
    ]
    if feasible:
        margin, selected = max(
            feasible,
            key=lambda candidate: (
                candidate[1]["coverage"],
                candidate[1]["acceptedPrecision"],
                -candidate[0],
            ),
        )
    else:
        margin = _clean_float(
            min(1.0, max((outcome[0] for outcome in outcomes), default=0.0) + 1e-6)
        )
        selected = {
            "acceptedGroupCount": 0,
            "acceptedPrecision": 0.0,
            "coverage": 0.0,
        }

    return margin, {
        "split": "val",
        "minimumMargin": MINIMUM_MARGIN,
        "objective": "smallest conservative margin retaining >= 0.98 accepted precision",
        "precisionConstraint": PRECISION_TARGET,
        "precisionConstraintFeasible": bool(feasible),
        **selected,
    }


def _evaluate(
    rows: Sequence[PairRow], scores: Sequence[float], threshold: float
) -> dict[str, Any]:
    return {
        "binary": _binary_metrics([row.label for row in rows], scores, threshold),
        "grouped": _grouped_metrics(rows, scores),
    }


def _baseline_evaluation(rows: Sequence[PairRow]) -> dict[str, Any]:
    scores = [row.features[0] for row in rows]
    return _evaluate(rows, scores, 1.0)


def _acceptance(
    integrity: Mapping[str, Any],
    val_metrics: Mapping[str, Any],
    test_metrics: Mapping[str, Any],
    baseline_test: Mapping[str, Any],
) -> dict[str, Any]:
    trained_f1 = test_metrics["binary"]["f1"]
    trained_top1 = test_metrics["grouped"]["top1"]
    beats_baseline_f1 = trained_f1 > baseline_test["binary"]["f1"]
    beats_baseline_top1 = trained_top1 > baseline_test["grouped"]["top1"]
    checks = {
        "valPrecisionAtLeast0.98": val_metrics["binary"]["precision"]
        >= PRECISION_TARGET,
        "testPrecisionAtLeast0.98": test_metrics["binary"]["precision"]
        >= PRECISION_TARGET,
        "testTop1AtLeast0.95": trained_top1 >= TEST_TOP1_TARGET,
        "splitGroupIntegrityPassed": integrity.get("status") == "passed",
        "beatsExactBaselineOnTestF1OrTop1": beats_baseline_f1
        or beats_baseline_top1,
    }
    failed = [name for name, passed in checks.items() if not passed]
    return {
        "ready": not failed,
        "requirements": {
            "valPrecision": PRECISION_TARGET,
            "testPrecision": PRECISION_TARGET,
            "testTop1": TEST_TOP1_TARGET,
            "integrityStatus": "passed",
            "baselineImprovement": "strictly higher test F1 or grouped top1",
        },
        "checks": checks,
        "failedChecks": failed,
        "baselineComparison": {
            "trainedTestF1": trained_f1,
            "baselineTestF1": baseline_test["binary"]["f1"],
            "trainedTestTop1": trained_top1,
            "baselineTestTop1": baseline_test["grouped"]["top1"],
            "beatsBaselineOnF1": beats_baseline_f1,
            "beatsBaselineOnTop1": beats_baseline_top1,
        },
    }


def build_artifact(
    input_dir: Path,
    optimizer_config: OptimizerConfig = OptimizerConfig(),
    trainer_path: Path | None = None,
) -> dict[str, Any]:
    dataset = load_dataset(input_dir)
    model = fit_model(dataset.rows["train"], optimizer_config)
    scores = {
        split: [predict_probability(row.features, model) for row in dataset.rows[split]]
        for split in SPLITS
    }
    threshold, threshold_selection = select_probability_threshold(
        dataset.rows["val"], scores["val"]
    )
    margin_threshold, margin_selection = select_margin_threshold(
        dataset.rows["val"], scores["val"], threshold
    )
    val_metrics = _evaluate(dataset.rows["val"], scores["val"], threshold)
    test_metrics = _evaluate(dataset.rows["test"], scores["test"], threshold)
    baseline = {
        "name": "exact_normalized_match",
        "definition": (
            "Predict positive only for exact normalized text equality; grouped ties are "
            "resolved by candidateId then candidateTerm."
        ),
        "val": _baseline_evaluation(dataset.rows["val"]),
        "test": _baseline_evaluation(dataset.rows["test"]),
    }
    acceptance = _acceptance(
        dataset.integrity, val_metrics, test_metrics, baseline["test"]
    )
    trainer_path = (trainer_path or Path(__file__)).resolve()
    total_rows = sum(len(dataset.rows[split]) for split in SPLITS)

    notes = [
        "Deterministic text-only pair classifier; no pose, video, or motion synthesis.",
        "Source query text must be preserved; a match score is not an ISL grammar claim.",
        "The probability threshold is selected on validation and evaluated unchanged on test.",
    ]
    if not acceptance["ready"]:
        notes.append(
            "Acceptance gates failed: " + ", ".join(acceptance["failedChecks"])
        )

    artifact: dict[str, Any] = {
        "schemaVersion": SCHEMA_VERSION,
        "metadata": {
            "id": "isl-text-matcher-2026-07-16",
            "version": "1.0.0",
            "status": "ready" if acceptance["ready"] else "unavailable",
            "engine": "lexicon_pair_matcher",
            "trainedAt": TRAINED_AT,
            "trainingDataset": {
                "primaryDataset": "Robin186/SignSaarthi-ISL-Training",
                "displayName": "SignSaarthi ISL lexical matcher pairs",
                "recordCount": total_rows,
                "classCount": 2,
                "citationUrls": [
                    "https://huggingface.co/datasets/Robin186/SignSaarthi-ISL-Training"
                ],
            },
            "notes": notes,
        },
        "featureNames": list(FEATURE_NAMES),
        "means": list(model.means),
        "scales": list(model.scales),
        "coefficients": list(model.coefficients),
        "intercept": model.intercept,
        "threshold": threshold,
        "marginThreshold": margin_threshold,
        "minimumQueryLength": MINIMUM_QUERY_LENGTH,
        "metrics": {
            "thresholdSelection": threshold_selection,
            "marginSelection": margin_selection,
            "val": val_metrics,
            "test": test_metrics,
            "baseline": baseline,
            "acceptance": acceptance,
        },
        "constraints": {
            "singleTokenOnly": True,
            "preserveSourceText": True,
            "noGrammarClaims": True,
            "noMotionSynthesis": True,
        },
        "dataHashes": dataset.data_hashes,
        "trainingConfig": {
            "optimizer": "deterministic full-batch gradient descent",
            "iterations": optimizer_config.iterations,
            "learningRate": optimizer_config.learning_rate,
            "l2": optimizer_config.l2,
            "randomness": "none",
            "deterministicOrder": "split, groupId, candidateId, normalized text, label",
            "normalization": (
                "Unicode NFKC, casefold, non-letter/mark/number separators collapsed "
                "to single spaces"
            ),
            "featureStandardization": (
                "train-only population mean and standard deviation; zero-variance scale is 1"
            ),
            "classWeighting": {
                "strategy": model.class_weight_strategy,
                "activationImbalanceRatio": optimizer_config.class_weight_imbalance_ratio,
                "negative": model.class_weights[0],
                "positive": model.class_weights[1],
            },
            "collapsedTrainingFeatureVectors": model.collapsed_training_vectors,
            "finalWeightedLogLoss": model.final_weighted_log_loss,
            "thresholdSelection": (
                "validation F1 maximum subject to precision >= 0.98 when feasible"
            ),
            "marginSelection": (
                "validation grouped acceptance with >= 0.98 precision and a 0.01 floor"
            ),
            "trainedAtPolicy": "fixed dataset snapshot timestamp for deterministic bytes",
            "trainerPath": "scripts/train_isl_text_matcher.py",
            "trainerSha256": sha256_file(trainer_path),
        },
        "integrity": dataset.integrity,
    }
    validate_artifact(artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    if artifact.get("schemaVersion") != SCHEMA_VERSION:
        raise MatcherTrainingError("Matcher artifact schemaVersion must be 1.")
    if artifact.get("featureNames") != list(FEATURE_NAMES):
        raise MatcherTrainingError("Matcher artifact feature names or order changed.")
    if artifact.get("metadata", {}).get("engine") != "lexicon_pair_matcher":
        raise MatcherTrainingError("Matcher artifact engine is invalid.")
    for key in ("means", "scales", "coefficients"):
        values = artifact.get(key)
        if not isinstance(values, list) or len(values) != len(FEATURE_NAMES):
            raise MatcherTrainingError(f"Matcher artifact {key} has invalid dimensions.")
        if not all(isinstance(value, (int, float)) and math.isfinite(value) for value in values):
            raise MatcherTrainingError(f"Matcher artifact {key} contains non-finite values.")
    if any(value <= 0 for value in artifact["scales"]):
        raise MatcherTrainingError("Matcher feature scales must be positive.")
    for key in ("threshold", "marginThreshold"):
        value = artifact.get(key)
        if not isinstance(value, (int, float)) or not 0.0 <= value <= 1.0:
            raise MatcherTrainingError(f"Matcher artifact {key} must be within [0, 1].")
    if artifact.get("minimumQueryLength") != MINIMUM_QUERY_LENGTH:
        raise MatcherTrainingError("Matcher minimumQueryLength must remain 4.")
    if artifact.get("constraints") != {
        "singleTokenOnly": True,
        "preserveSourceText": True,
        "noGrammarClaims": True,
        "noMotionSynthesis": True,
    }:
        raise MatcherTrainingError("Matcher constraints are incomplete.")
    acceptance_ready = artifact.get("metrics", {}).get("acceptance", {}).get("ready")
    expected_status = "ready" if acceptance_ready else "unavailable"
    if artifact.get("metadata", {}).get("status") != expected_status:
        raise MatcherTrainingError("Matcher readiness status bypassed acceptance gates.")
    if acceptance_ready and not all(
        artifact["metrics"]["acceptance"]["checks"].values()
    ):
        raise MatcherTrainingError("A ready matcher has a failed acceptance check.")


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
    return len(payload), hashlib.sha256(payload).hexdigest()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    workspace_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=(
            workspace_root
            / "output/huggingface/Robin186-SignSaarthi-ISL-Training"
        ),
        help="Matcher bundle root containing data/match_pairs/{train,val,test}.jsonl.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=workspace_root / "data/models/isl-text-matcher.json",
        help="Output matcher artifact path.",
    )
    parser.add_argument(
        "--verify-determinism",
        action="store_true",
        help="Train twice and fail unless canonical artifact bytes are identical.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        artifact = build_artifact(args.input_dir)
        if args.verify_determinism:
            repeated = build_artifact(args.input_dir)
            if canonical_json_bytes(artifact) != canonical_json_bytes(repeated):
                raise MatcherTrainingError(
                    "Repeated matcher training was not byte-deterministic."
                )
        byte_length, artifact_hash = write_artifact(args.output, artifact)
    except MatcherTrainingError as error:
        print(f"Matcher training failed: {error}", file=sys.stderr)
        return 2

    val = artifact["metrics"]["val"]
    test = artifact["metrics"]["test"]
    print(f"Wrote ISL text matcher: {args.output.resolve()}")
    print(f"Artifact bytes: {byte_length}; SHA-256: {artifact_hash}")
    print(
        "val: "
        f"precision={val['binary']['precision']}, recall={val['binary']['recall']}, "
        f"f1={val['binary']['f1']}, top1={val['grouped']['top1']}, "
        f"mrr={val['grouped']['mrr']}"
    )
    print(
        "test: "
        f"precision={test['binary']['precision']}, recall={test['binary']['recall']}, "
        f"f1={test['binary']['f1']}, top1={test['grouped']['top1']}, "
        f"mrr={test['grouped']['mrr']}"
    )
    print(f"Model status: {artifact['metadata']['status']}")
    if args.verify_determinism:
        print("Determinism check: passed (two byte-identical in-memory builds)")
    if artifact["metadata"]["status"] != "ready":
        print(
            "Failed readiness checks: "
            + ", ".join(artifact["metrics"]["acceptance"]["failedChecks"]),
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
