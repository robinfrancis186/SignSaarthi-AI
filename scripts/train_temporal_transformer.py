#!/usr/bin/env python3
"""Fine-tune a compact temporal ISL classifier and export real motion clips.

The script streams the combined keypoint JSON with ijson. It never downloads or
loads source videos. By default it trains every label with source training data.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import random
import re
import shutil
import tempfile
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import ijson
import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn
import torch.nn.functional as F
from onnxruntime.quantization import QuantType, quantize_dynamic
from sklearn.metrics import accuracy_score, f1_score, top_k_accuracy_score
from torch.utils.data import DataLoader, Dataset
from transformers import BertConfig
from transformers.models.bert.modeling_bert import BertLayer


SEED = 186
SEQUENCE_LENGTH = 64
INPUT_SIZE = 134
CALIBRATION_TARGET_ACCEPTED_ACCURACY = 0.9
CALIBRATION_MIN_COVERAGE = 0.2
CALIBRATION_THRESHOLDS = tuple(round(index / 100, 2) for index in range(0, 96))
POSE_COUNT = 25
HAND_COUNT = 21
FRAME_WIDTH = 1920.0
FRAME_HEIGHT = 1080.0
MOTION_FRAME_COUNT = 32
MOTION_DURATION_TOLERANCE_MS = 0.5
MAX_RENDERER_FPS = 120.0
PRETRAINED_URL = (
    "https://storage.googleapis.com/ai4bharat-public-indic-data/"
    "INCLUDE/pretrained_models/include50/no_cnn/transformer_small.pth"
)

# Mutable in place so the standalone evaluator can configure the exact exported
# vocabulary from model metadata while retaining compatibility with imports.
SELECTED_LABELS: list[str] = []


@dataclass
class Metrics:
    accuracy: float
    macro_f1: float
    top3_accuracy: float
    sample_count: int


class PositionEmbedding(nn.Module):
    def __init__(self, config: BertConfig) -> None:
        super().__init__()
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.register_buffer(
            "position_ids",
            torch.arange(config.max_position_embeddings).expand((1, -1)),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        position_ids = self.position_ids[:, : values.shape[1]]
        return self.dropout(self.LayerNorm(values + self.position_embeddings(position_ids)))


class IncludeTransformer(nn.Module):
    """Architecture-compatible INCLUDE small transformer with a new class head."""

    def __init__(self, class_count: int) -> None:
        super().__init__()
        config = BertConfig(
            hidden_size=256,
            num_attention_heads=4,
            num_hidden_layers=2,
            max_position_embeddings=256,
            hidden_dropout_prob=0.1,
        )
        self.l1 = nn.Linear(INPUT_SIZE, config.hidden_size)
        self.embedding = PositionEmbedding(config)
        self.layers = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])
        self.l2 = nn.Linear(config.hidden_size, class_count)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        hidden = self.embedding(self.l1(values))
        for layer in self.layers:
            layer_output = layer(hidden)
            # Transformers 4 returned a tuple while Transformers 5 returns the
            # hidden-state tensor directly for a standalone BertLayer.
            hidden = layer_output[0] if isinstance(layer_output, tuple) else layer_output
        hidden = torch.max(hidden, dim=1).values
        hidden = F.dropout(hidden, p=0.2, training=self.training)
        return self.l2(hidden)


class SequenceDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        valid_lengths: np.ndarray,
        augment: bool,
    ) -> None:
        self.features = features
        self.labels = labels
        self.valid_lengths = valid_lengths
        self.augment = augment

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        values = self.features[index].copy()
        if self.augment:
            values = augment_sequence(values, int(self.valid_lengths[index]))
        return torch.from_numpy(values), torch.tensor(int(self.labels[index]), dtype=torch.long)


def normalize_label(label: str) -> str:
    return "".join(character for character in label if character.isalpha()).lower()


def display_label(label: str) -> str:
    cleaned = re.sub(r"^\s*\d+[\s.)_-]*", "", label).strip()
    return cleaned or label


def set_selected_labels(labels: Iterable[str]) -> list[str]:
    normalized = [normalize_label(label) for label in labels]
    if not normalized or any(not label for label in normalized):
        raise ValueError("The temporal vocabulary must contain non-empty labels.")
    if len(set(normalized)) != len(normalized):
        raise ValueError("The temporal vocabulary contains duplicate normalized labels.")
    SELECTED_LABELS[:] = normalized
    return SELECTED_LABELS


def discover_labels(path: Path) -> list[str]:
    train_counts: Counter[str] = Counter()
    observed: set[str] = set()
    with path.open("rb") as source:
        for sample in ijson.items(source, "item"):
            label = normalize_label(str(sample.get("label", "")))
            if not label:
                continue
            observed.add(label)
            if str(sample.get("split", "")) == "train":
                train_counts[label] += 1
    without_training = sorted(label for label in observed if train_counts[label] == 0)
    if without_training:
        raise RuntimeError(
            "Labels without source training examples cannot be exported: "
            + ", ".join(without_training)
        )
    if not observed:
        raise RuntimeError(f"No temporal labels were found in {path}.")
    return sorted(observed)


def resolve_selected_labels(path: Path, labels_spec: str) -> list[str]:
    available = discover_labels(path)
    if labels_spec.strip().lower() == "all":
        return set_selected_labels(available)
    requested = [part.strip() for part in labels_spec.split(",") if part.strip()]
    normalized = [normalize_label(label) for label in requested]
    missing = sorted(set(normalized) - set(available))
    if missing:
        raise RuntimeError("Requested labels are absent from the corpus: " + ", ".join(missing))
    return set_selected_labels(normalized)


def stream_selected_samples(path: Path) -> list[dict[str, Any]]:
    if not SELECTED_LABELS:
        raise RuntimeError("Configure the temporal vocabulary before streaming samples.")
    selected = set(SELECTED_LABELS)
    samples: list[dict[str, Any]] = []
    with path.open("rb") as source:
        for sample in ijson.items(source, "item"):
            if normalize_label(str(sample["label"])) in selected:
                samples.append(sample)
    return samples


def interpolate_track(track: np.ndarray) -> np.ndarray:
    output = track.astype(np.float32, copy=True)
    timeline = np.arange(output.shape[0])
    for point_index in range(output.shape[1]):
        for coordinate in range(2):
            values = output[:, point_index, coordinate]
            valid = np.isfinite(values)
            if not valid.any():
                output[:, point_index, coordinate] = 0.0
                continue
            output[:, point_index, coordinate] = np.interp(timeline, timeline[valid], values[valid])
    return output


def sequence_to_tracks(sample: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    frame_count = len(sample["frames"])
    pose = np.full((frame_count, POSE_COUNT, 2), np.nan, dtype=np.float32)
    left = np.full((frame_count, HAND_COUNT, 2), np.nan, dtype=np.float32)
    right = np.full((frame_count, HAND_COUNT, 2), np.nan, dtype=np.float32)
    targets = {"pose": pose, "left_hand": left, "right_hand": right}

    for frame_index, frame in enumerate(sample["frames"]):
        for landmark in frame["landmarks"]:
            part = str(landmark.get("part", ""))
            target = targets.get(part)
            if target is None:
                continue
            point_index = int(landmark["index"])
            if point_index >= target.shape[1]:
                continue
            target[frame_index, point_index] = (float(landmark["x"]), float(landmark["y"]))

    return interpolate_track(pose), interpolate_track(left), interpolate_track(right)


def sequence_timestamps_ms(sample: dict[str, Any]) -> np.ndarray:
    timestamps = np.asarray(
        [float(frame.get("timestampMs", math.nan)) for frame in sample["frames"]],
        dtype=np.float64,
    )
    sample_id = str(sample.get("sampleId", "unknown"))
    if not np.isfinite(timestamps).all():
        raise RuntimeError(f"Sample {sample_id} is missing finite frame timestamps.")
    if timestamps.shape[0] > 1 and np.any(np.diff(timestamps) <= 0):
        raise RuntimeError(f"Sample {sample_id} frame timestamps are not strictly increasing.")
    return timestamps


def resample_track(
    track: np.ndarray,
    target_length: int,
    source_timestamps_ms: np.ndarray | None = None,
) -> np.ndarray:
    if track.shape[0] == target_length and source_timestamps_ms is None:
        return track
    if track.shape[0] == 1:
        return np.repeat(track, target_length, axis=0)
    if source_timestamps_ms is None:
        source_positions = np.linspace(0.0, 1.0, track.shape[0])
    else:
        if source_timestamps_ms.shape != (track.shape[0],):
            raise ValueError("Timestamp count must match the source track length.")
        if not np.isfinite(source_timestamps_ms).all() or np.any(np.diff(source_timestamps_ms) <= 0):
            raise ValueError("Source timestamps must be finite and strictly increasing.")
        source_positions = source_timestamps_ms
    target_positions = np.linspace(source_positions[0], source_positions[-1], target_length)
    result = np.empty((target_length, track.shape[1], track.shape[2]), dtype=np.float32)
    for point_index in range(track.shape[1]):
        for coordinate in range(track.shape[2]):
            result[:, point_index, coordinate] = np.interp(
                target_positions, source_positions, track[:, point_index, coordinate]
            )
    return result


def sample_to_features(sample: dict[str, Any]) -> tuple[np.ndarray, int]:
    pose, left, right = sequence_to_tracks(sample)
    timestamps_ms = sequence_timestamps_ms(sample)
    original_length = pose.shape[0]
    if original_length > SEQUENCE_LENGTH:
        pose = resample_track(pose, SEQUENCE_LENGTH, timestamps_ms)
        left = resample_track(left, SEQUENCE_LENGTH, timestamps_ms)
        right = resample_track(right, SEQUENCE_LENGTH, timestamps_ms)
    valid_length = min(original_length, SEQUENCE_LENGTH)

    pose[:, :, 0] *= FRAME_WIDTH
    pose[:, :, 1] *= FRAME_HEIGHT
    left[:, :, 0] *= FRAME_WIDTH
    left[:, :, 1] *= FRAME_HEIGHT
    right[:, :, 0] *= FRAME_WIDTH
    right[:, :, 1] *= FRAME_HEIGHT

    combined = np.concatenate(
        [pose.reshape(pose.shape[0], -1), left.reshape(left.shape[0], -1), right.reshape(right.shape[0], -1)],
        axis=1,
    ).astype(np.float32)
    padded = np.zeros((SEQUENCE_LENGTH, INPUT_SIZE), dtype=np.float32)
    padded[: combined.shape[0]] = combined
    return padded, valid_length


def augment_sequence(values: np.ndarray, valid_length: int) -> np.ndarray:
    if valid_length <= 0:
        return values
    points = values[:valid_length].reshape(valid_length, -1, 2)
    center = np.array([FRAME_WIDTH / 2.0, FRAME_HEIGHT / 2.0], dtype=np.float32)
    angle = np.deg2rad(random.uniform(-5.0, 5.0))
    rotation = np.array(
        [[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]],
        dtype=np.float32,
    )
    nonzero = np.any(points != 0.0, axis=2)
    rotated = (points - center) @ rotation.T + center
    points[nonzero] = rotated[nonzero]
    points[nonzero] += np.random.normal(0.0, 2.2, size=points[nonzero].shape).astype(np.float32)

    if random.random() < 0.25:
        point_mask = np.random.random((valid_length, points.shape[1])) < 0.015
        points[point_mask] = 0.0
    values[:valid_length] = points.reshape(valid_length, INPUT_SIZE)
    return values


def prepare_arrays(
    samples: list[dict[str, Any]],
) -> tuple[dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]], dict[str, str], Counter[str]]:
    label_to_index = {label: index for index, label in enumerate(SELECTED_LABELS)}
    display_names: dict[str, str] = {}
    split_rows: dict[str, list[tuple[np.ndarray, int, int]]] = {"train": [], "val": [], "test": []}
    split_counts: Counter[str] = Counter()

    for sample in samples:
        normalized = normalize_label(str(sample["label"]))
        split = str(sample["split"])
        if split not in split_rows:
            continue
        features, valid_length = sample_to_features(sample)
        split_rows[split].append((features, label_to_index[normalized], valid_length))
        display_names.setdefault(normalized, display_label(str(sample["label"])))
        split_counts[f"{split}:{normalized}"] += 1

    missing_training = [label for label in SELECTED_LABELS if split_counts[f"train:{label}"] == 0]
    if missing_training:
        raise RuntimeError(
            "Selected vocabulary is missing source training samples: " + ", ".join(missing_training)
        )
    for split, rows in split_rows.items():
        if not rows:
            raise RuntimeError(f"The source dataset has no {split} samples for evaluation.")

    arrays: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for split, rows in split_rows.items():
        arrays[split] = (
            np.stack([row[0] for row in rows]),
            np.asarray([row[1] for row in rows], dtype=np.int64),
            np.asarray([row[2] for row in rows], dtype=np.int64),
        )
    return arrays, display_names, split_counts


def audit_signer_splits(samples: list[dict[str, Any]]) -> dict[str, Any]:
    total_count = len(samples)
    available = [sample for sample in samples if str(sample.get("signerHash") or "").strip()]
    available_count = len(available)
    if available_count == 0:
        return {
            "field": "signerHash",
            "status": "not_available",
            "enforced": False,
            "signerDisjoint": None,
            "sequenceCount": total_count,
            "availableSequenceCount": 0,
            "missingSequenceCount": total_count,
            "message": (
                "The source keypoint records contain no signer hashes; signer-disjointness "
                "cannot be verified from this artifact. Source-provided dataset splits are retained."
            ),
        }
    if available_count != total_count:
        raise RuntimeError(
            "signerHash is present for only "
            f"{available_count} of {total_count} selected sequences; refusing a partial signer audit."
        )

    signer_splits: dict[str, set[str]] = {}
    signers_by_split: dict[str, set[str]] = {"train": set(), "val": set(), "test": set()}
    for sample in available:
        signer_hash = str(sample["signerHash"]).strip()
        split = str(sample["split"])
        signer_splits.setdefault(signer_hash, set()).add(split)
        signers_by_split.setdefault(split, set()).add(signer_hash)

    overlap_count = sum(1 for splits in signer_splits.values() if len(splits) > 1)
    if overlap_count:
        raise RuntimeError(
            f"Signer-disjoint split assertion failed: {overlap_count} signer hash(es) occur in multiple splits."
        )
    return {
        "field": "signerHash",
        "status": "verified",
        "enforced": True,
        "signerDisjoint": True,
        "sequenceCount": total_count,
        "availableSequenceCount": available_count,
        "missingSequenceCount": 0,
        "uniqueSignerCount": len(signer_splits),
        "uniqueSignersBySplit": {
            split: len(signers_by_split.get(split, set())) for split in ("train", "val", "test")
        },
        "overlapCount": 0,
        "message": "All selected sequences had signerHash and no signer appeared across splits.",
    }


def select_motion_samples(samples: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    candidates: dict[str, list[dict[str, Any]]] = {label: [] for label in SELECTED_LABELS}
    for sample in samples:
        label = normalize_label(str(sample["label"]))
        if label in candidates and sample["split"] == "train":
            candidates[label].append(sample)

    selected: dict[str, dict[str, Any]] = {}
    for label, rows in candidates.items():
        if not rows:
            raise RuntimeError(f"No training motion is available for exported label {label}.")
        lengths = sorted(len(row["frames"]) for row in rows)
        median_length = lengths[len(lengths) // 2]
        selected[label] = min(
            rows,
            key=lambda row: (
                float(row.get("missingKeypointRatio", 1.0)),
                abs(len(row["frames"]) - median_length),
                str(row["sampleId"]),
            ),
        )
    return selected


def point_list(track: np.ndarray) -> list[dict[str, float]]:
    return [
        {"x": round(float(point[0]), 5), "y": round(float(point[1]), 5)}
        for point in track
    ]


def derive_motion_timing(sample: dict[str, Any], target_frame_count: int) -> dict[str, float | int | str]:
    if target_frame_count < 2:
        raise ValueError("Motion clips require at least two resampled frames.")
    timestamps_ms = sequence_timestamps_ms(sample)
    if timestamps_ms.shape[0] < 2:
        raise RuntimeError(f"Sample {sample.get('sampleId', 'unknown')} has fewer than two timed frames.")

    source_duration_ms = float(timestamps_ms[-1] - timestamps_ms[0])
    effective_fps = round(((target_frame_count - 1) * 1000.0) / source_duration_ms, 6)
    if effective_fps > MAX_RENDERER_FPS:
        raise RuntimeError(
            f"Sample {sample.get('sampleId', 'unknown')} requires {effective_fps:.3f} FPS after "
            f"resampling, above the renderer limit of {MAX_RENDERER_FPS:.0f} FPS."
        )
    playback_duration_ms = ((target_frame_count - 1) / effective_fps) * 1000.0
    duration_error_ms = abs(playback_duration_ms - source_duration_ms)
    if duration_error_ms > MOTION_DURATION_TOLERANCE_MS:
        raise RuntimeError(
            f"Motion playback duration drifted by {duration_error_ms:.6f} ms for "
            f"{sample.get('sampleId', 'unknown')}."
        )
    return {
        "source": "retained-frame-timestamps",
        "sourceFrameCount": int(timestamps_ms.shape[0]),
        "resampledFrameCount": target_frame_count,
        "sourceDurationMs": round(source_duration_ms, 3),
        "playbackDurationMs": round(playback_duration_ms, 3),
        "durationErrorMs": round(duration_error_ms, 6),
        "effectiveFps": effective_fps,
    }


def write_motion_library(
    samples: list[dict[str, Any]], display_names: dict[str, str], output_path: Path, catalog_path: Path
) -> dict[str, float | int | str]:
    selected = select_motion_samples(samples)
    clips: list[dict[str, Any]] = []
    catalog: list[dict[str, Any]] = []
    timing_audits: list[dict[str, float | int | str]] = []

    for label in SELECTED_LABELS:
        sample = selected[label]
        pose, left, right = sequence_to_tracks(sample)
        timestamps_ms = sequence_timestamps_ms(sample)
        pose = resample_track(pose, MOTION_FRAME_COUNT, timestamps_ms)
        left = resample_track(left, MOTION_FRAME_COUNT, timestamps_ms)
        right = resample_track(right, MOTION_FRAME_COUNT, timestamps_ms)
        timing = derive_motion_timing(sample, MOTION_FRAME_COUNT)
        timing_audits.append(timing)
        clip_id = f"include-{label}"
        frames = [
            {
                "pose": point_list(pose[index]),
                "leftHand": point_list(left[index]),
                "rightHand": point_list(right[index]),
            }
            for index in range(MOTION_FRAME_COUNT)
        ]
        clips.append(
            {
                "id": clip_id,
                "label": display_names[label],
                "normalizedLabel": label,
                "fps": timing["effectiveFps"],
                "durationMs": timing["sourceDurationMs"],
                "sourceSampleId": str(sample["sampleId"]),
                "datasetId": "include",
                "expertReviewed": False,
                "timing": timing,
                "frames": frames,
            }
        )
        catalog.append(
            {
                "id": clip_id,
                "label": display_names[label],
                "normalizedLabel": label,
                "datasetId": "include",
                "sourceSampleId": str(sample["sampleId"]),
                "frameCount": MOTION_FRAME_COUNT,
                "fps": timing["effectiveFps"],
                "durationMs": timing["sourceDurationMs"],
                "timingSource": timing["source"],
                "expertReviewed": False,
            }
        )

    max_duration_error_ms = max(float(audit["durationErrorMs"]) for audit in timing_audits)
    timing_summary: dict[str, float | int | str] = {
        "source": "retained-frame-timestamps",
        "resampling": "linear interpolation on the timestamp axis",
        "clipCount": len(timing_audits),
        "durationToleranceMs": MOTION_DURATION_TOLERANCE_MS,
        "maxPlaybackDurationErrorMs": max_duration_error_ms,
        "status": "passed",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    catalog_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            {
                "schemaVersion": 1,
                "source": "INCLUDE real MediaPipe keypoint sequences",
                "license": "Source INCLUDE dataset license applies; research/prototype use",
                "expertReviewStatus": "not_expert_certified",
                "reviewStatement": "Dataset-derived isolated-sign motion; not expert-certified ISL.",
                "timingAudit": timing_summary,
                "clips": clips,
            },
            separators=(",", ":"),
        )
        + "\n",
        encoding="utf-8",
    )
    catalog_path.write_text(
        json.dumps(
            {
                "schemaVersion": 1,
                "clipCount": len(catalog),
                "timingAudit": timing_summary,
                "clips": catalog,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return timing_summary


def set_backbone_trainable(model: IncludeTransformer, trainable: bool) -> None:
    for name, parameter in model.named_parameters():
        parameter.requires_grad = trainable or name.startswith("l2.")


def load_pretrained_backbone(model: IncludeTransformer, path: Path) -> int:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state = checkpoint.get("model", checkpoint)
    backbone_state = {key: value for key, value in state.items() if not key.startswith("l2.")}
    result = model.load_state_dict(backbone_state, strict=False)
    unexpected = [key for key in result.unexpected_keys if not key.startswith("l2.")]
    missing = [key for key in result.missing_keys if not key.startswith("l2.")]
    if unexpected or missing:
        raise RuntimeError(f"Pretrained backbone mismatch; missing={missing}, unexpected={unexpected}")
    return len(backbone_state)


def choose_device() -> torch.device:
    requested = os.environ.get("SIGNSAARTHI_TRAIN_DEVICE")
    if requested:
        return torch.device(requested)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@torch.no_grad()
def predict_torch(
    model: IncludeTransformer, loader: DataLoader[tuple[torch.Tensor, torch.Tensor]], device: torch.device
) -> tuple[np.ndarray, np.ndarray, float]:
    model.eval()
    all_logits: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    losses: list[float] = []
    for values, labels in loader:
        values = values.to(device)
        labels = labels.to(device)
        logits = model(values)
        losses.append(float(F.cross_entropy(logits, labels).item()))
        all_logits.append(logits.detach().cpu().numpy())
        all_labels.append(labels.detach().cpu().numpy())
    return np.concatenate(all_logits), np.concatenate(all_labels), float(np.mean(losses))


def calculate_metrics(logits: np.ndarray, labels: np.ndarray) -> Metrics:
    predictions = logits.argmax(axis=1)
    probabilities = torch.softmax(torch.from_numpy(logits), dim=1).numpy()
    return Metrics(
        accuracy=float(accuracy_score(labels, predictions)),
        macro_f1=float(
            f1_score(
                labels,
                predictions,
                labels=np.arange(len(SELECTED_LABELS)),
                average="macro",
                zero_division=0,
            )
        ),
        top3_accuracy=float(
            top_k_accuracy_score(labels, probabilities, k=3, labels=np.arange(len(SELECTED_LABELS)))
        ),
        sample_count=int(labels.shape[0]),
    )


def train_model(
    arrays: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    pretrained_path: Path,
    work_path: Path,
    warmup_epochs: int,
    fine_tune_epochs: int,
    batch_size: int,
) -> tuple[IncludeTransformer, dict[str, Metrics], dict[str, Any]]:
    device = choose_device()
    model = IncludeTransformer(len(SELECTED_LABELS))
    loaded_tensors = load_pretrained_backbone(model, pretrained_path)
    model.to(device)

    loaders: dict[str, DataLoader[tuple[torch.Tensor, torch.Tensor]]] = {}
    for split in ("train", "val"):
        features, labels, lengths = arrays[split]
        loaders[split] = DataLoader(
            SequenceDataset(features, labels, lengths, augment=split == "train"),
            batch_size=batch_size,
            shuffle=split == "train",
            num_workers=0,
        )

    best_score = (-1.0, -1.0)
    best_state = copy.deepcopy(model.state_dict())
    history: list[dict[str, float | int | str]] = []
    stages = [("head", warmup_epochs), ("fine_tune", fine_tune_epochs)]
    started_at = time.time()

    for stage, epoch_count in stages:
        if epoch_count <= 0:
            continue
        set_backbone_trainable(model, stage == "fine_tune")
        if stage == "head":
            optimizer = torch.optim.AdamW(model.l2.parameters(), lr=8e-4, weight_decay=1e-3)
        else:
            head_parameters = list(model.l2.parameters())
            head_ids = {id(parameter) for parameter in head_parameters}
            backbone_parameters = [
                parameter for parameter in model.parameters() if id(parameter) not in head_ids
            ]
            optimizer = torch.optim.AdamW(
                [
                    {"params": backbone_parameters, "lr": 2e-5},
                    {"params": head_parameters, "lr": 2.5e-4},
                ],
                weight_decay=1e-3,
            )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epoch_count))

        stale_epochs = 0
        for epoch in range(1, epoch_count + 1):
            model.train()
            train_losses: list[float] = []
            for values, labels in loaders["train"]:
                values = values.to(device)
                labels = labels.to(device)
                optimizer.zero_grad(set_to_none=True)
                logits = model(values)
                loss = F.cross_entropy(logits, labels, label_smoothing=0.04)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                train_losses.append(float(loss.item()))
            scheduler.step()

            val_logits, val_labels, val_loss = predict_torch(model, loaders["val"], device)
            val_metrics = calculate_metrics(val_logits, val_labels)
            score = (val_metrics.accuracy, val_metrics.macro_f1)
            history.append(
                {
                    "stage": stage,
                    "epoch": epoch,
                    "trainLoss": float(np.mean(train_losses)),
                    "valLoss": val_loss,
                    "valAccuracy": val_metrics.accuracy,
                    "valMacroF1": val_metrics.macro_f1,
                }
            )
            print(
                f"[{stage} {epoch:02d}/{epoch_count:02d}] "
                f"train_loss={np.mean(train_losses):.4f} val_loss={val_loss:.4f} "
                f"val_acc={val_metrics.accuracy:.4f} val_f1={val_metrics.macro_f1:.4f}",
                flush=True,
            )
            if score > best_score:
                best_score = score
                best_state = copy.deepcopy(model.state_dict())
                work_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(best_state, work_path)
                stale_epochs = 0
            else:
                stale_epochs += 1
            if stage == "fine_tune" and stale_epochs >= 7:
                print("Early stopping after seven non-improving fine-tune epochs.", flush=True)
                break

    model.load_state_dict(best_state)
    model.to(device)
    metrics: dict[str, Metrics] = {}
    for split in ("train", "val"):
        logits, labels, _ = predict_torch(model, loaders[split], device)
        metrics[split] = calculate_metrics(logits, labels)

    training = {
        "device": str(device),
        "seed": SEED,
        "warmupEpochs": warmup_epochs,
        "fineTuneEpochsRequested": fine_tune_epochs,
        "batchSize": batch_size,
        "durationSeconds": round(time.time() - started_at, 2),
        "pretrainedTensorCount": loaded_tensors,
        "history": history,
    }
    return model.cpu().eval(), metrics, training


def export_onnx(
    model: IncludeTransformer, float_path: Path, quantized_path: Path
) -> None:
    float_path.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.zeros((1, SEQUENCE_LENGTH, INPUT_SIZE), dtype=torch.float32)
    torch.onnx.export(
        model,
        dummy,
        float_path,
        input_names=["keypoints"],
        output_names=["logits"],
        dynamic_axes={"keypoints": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )
    quantize_dynamic(float_path, quantized_path, weight_type=QuantType.QInt8)


def predict_onnx(
    model_path: Path, split: tuple[np.ndarray, np.ndarray, np.ndarray]
) -> tuple[np.ndarray, np.ndarray]:
    features, labels, _ = split
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    logits = session.run(["logits"], {"keypoints": features.astype(np.float32)})[0]
    return logits, labels


def softmax_probabilities(logits: np.ndarray) -> np.ndarray:
    exponentials = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    return exponentials / exponentials.sum(axis=1, keepdims=True)


def calculate_selective_metrics(
    logits: np.ndarray, labels: np.ndarray, threshold: float
) -> dict[str, float | int]:
    probabilities = softmax_probabilities(logits)
    predictions = probabilities.argmax(axis=1)
    accepted = probabilities.max(axis=1) >= threshold
    return {
        "threshold": threshold,
        "coverage": float(accepted.mean()),
        "acceptedAccuracy": float(
            np.mean(predictions[accepted] == labels[accepted]) if accepted.any() else 0.0
        ),
        "acceptedCount": int(accepted.sum()),
        "fallbackCount": int((~accepted).sum()),
    }


def calibrate_confidence_threshold(
    logits: np.ndarray,
    labels: np.ndarray,
    calibration_split: str,
    target_accepted_accuracy: float = CALIBRATION_TARGET_ACCEPTED_ACCURACY,
    minimum_coverage: float = CALIBRATION_MIN_COVERAGE,
    thresholds: tuple[float, ...] = CALIBRATION_THRESHOLDS,
) -> dict[str, Any]:
    if calibration_split != "val":
        raise ValueError("Confidence threshold calibration is restricted to the validation split.")
    if labels.shape[0] == 0:
        raise ValueError("Confidence threshold calibration requires validation samples.")

    minimum_accepted_count = max(1, math.ceil(labels.shape[0] * minimum_coverage))
    candidates = [calculate_selective_metrics(logits, labels, threshold) for threshold in thresholds]
    eligible = [
        candidate
        for candidate in candidates
        if int(candidate["acceptedCount"]) >= minimum_accepted_count
        and float(candidate["acceptedAccuracy"]) >= target_accepted_accuracy
    ]
    target_met = bool(eligible)
    if eligible:
        selected = max(
            eligible,
            key=lambda candidate: (
                float(candidate["coverage"]),
                float(candidate["acceptedAccuracy"]),
                -float(candidate["threshold"]),
            ),
        )
    else:
        coverage_eligible = [
            candidate
            for candidate in candidates
            if int(candidate["acceptedCount"]) >= minimum_accepted_count
        ]
        selected = max(
            coverage_eligible,
            key=lambda candidate: (
                float(candidate["acceptedAccuracy"]),
                float(candidate["coverage"]),
                -float(candidate["threshold"]),
            ),
        )

    return {
        "deployedThreshold": float(selected["threshold"]),
        "fallback": "caption",
        "calibration": {
            "split": "val",
            "sampleCount": int(labels.shape[0]),
            "targetAcceptedAccuracy": target_accepted_accuracy,
            "minimumCoverage": minimum_coverage,
            "minimumAcceptedCount": minimum_accepted_count,
            "selectionRule": (
                "Choose maximum validation coverage among thresholds meeting the accepted-accuracy "
                "target and minimum coverage; otherwise maximize validation accepted accuracy at "
                "the minimum coverage."
            ),
            "candidateThresholds": list(thresholds),
            "selectedValidationMetrics": selected,
            "targetMet": target_met,
            "testDataUsed": False,
        },
    }


def write_metadata(
    output_path: Path,
    model_path: Path,
    display_names: dict[str, str],
    split_counts: Counter[str],
    training_metrics: dict[str, Metrics],
    quantized_validation_metrics: Metrics,
    quantized_test_metrics: Metrics,
    confidence_policy: dict[str, Any],
    training: dict[str, Any],
    signer_split_audit: dict[str, Any],
    motion_timing_audit: dict[str, float | int | str],
) -> None:
    labels = [
        {
            "index": index,
            "label": display_names[label],
            "normalizedLabel": label,
            "gloss": display_names[label].upper(),
            "trainSamples": split_counts[f"train:{label}"],
            "valSamples": split_counts[f"val:{label}"],
            "testSamples": split_counts[f"test:{label}"],
            "motionClipId": f"include-{label}",
        }
        for index, label in enumerate(SELECTED_LABELS)
    ]
    record_count = sum(item["trainSamples"] + item["valSamples"] + item["testSamples"] for item in labels)
    labels_with_all_splits = [
        item["normalizedLabel"]
        for item in labels
        if item["trainSamples"] and item["valSamples"] and item["testSamples"]
    ]
    missing_validation_labels = [
        item["normalizedLabel"] for item in labels if item["valSamples"] == 0
    ]
    missing_test_labels = [item["normalizedLabel"] for item in labels if item["testSamples"] == 0]
    model_id = f"signsaarthi-include-temporal-{len(labels)}"
    artifact = {
        "schemaVersion": 1,
        "metadata": {
            "id": model_id,
            "version": "0.4.0",
            "status": "ready",
            "engine": "keypoint_transformer",
            "trainedAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "trainingDataset": {
                "primaryDataset": "include",
                "displayName": f"INCLUDE full local temporal keypoints ({len(labels)} classes)",
                "recordCount": record_count,
                "classCount": len(labels),
                "citationUrls": [
                    "https://github.com/AI4Bharat/INCLUDE",
                    "https://zenodo.org/records/4010759",
                ],
            },
            "notes": [
                "Fine-tuned from the official INCLUDE small-transformer checkpoint.",
                (
                    "Final exported INT8 test evaluation (after model and threshold freeze): "
                    f"{quantized_test_metrics.accuracy * 100:.2f}% top-1 and "
                    f"{quantized_test_metrics.top3_accuracy * 100:.2f}% top-3 on "
                    f"{quantized_test_metrics.sample_count} sequences."
                ),
                (
                    "Confidence threshold selected on validation only. At the frozen test threshold "
                    f"of {float(confidence_policy['deployedThreshold']):.2f}: "
                    f"{float(confidence_policy['acceptedAccuracy']) * 100:.2f}% accepted accuracy "
                    f"at {float(confidence_policy['coverage']) * 100:.2f}% coverage; "
                    "all remaining predictions use captions."
                ),
                "Consumes pose and hand keypoints only; raw video and audio are not stored.",
                "Full locally extracted INCLUDE vocabulary; not a certified interpreter replacement.",
                "Motion clips are dataset-derived isolated signs and are not expert-certified ISL.",
                (
                    f"Source split coverage is complete for {len(labels_with_all_splits)} of "
                    f"{len(labels)} labels; labels without validation or test examples are disclosed "
                    "in sourceSplitCoverage and are not assigned per-class held-out claims."
                ),
                str(signer_split_audit["message"]),
            ],
        },
        "modelFile": model_path.name,
        "modelBytes": model_path.stat().st_size,
        "labels": labels,
        "preprocessing": {
            "sequenceLength": SEQUENCE_LENGTH,
            "inputSize": INPUT_SIZE,
            "poseLandmarks": POSE_COUNT,
            "handLandmarksPerHand": HAND_COUNT,
            "coordinates": ["x", "y"],
            "frameWidth": int(FRAME_WIDTH),
            "frameHeight": int(FRAME_HEIGHT),
            "missingLandmarks": "linear interpolation; nearest endpoint; zero only when entirely absent",
            "longSequences": "timestamp-axis linear resample only when longer than 64 frames",
            "shortSequences": "zero pad after the final observed frame",
        },
        "transferLearning": {
            "source": "AI4Bharat INCLUDE small transformer",
            "checkpointUrl": PRETRAINED_URL,
            "pretrainedValidationScore": 0.7272727272727273,
            "reinitializedHead": True,
        },
        "metrics": {
            "train": asdict(training_metrics["train"]),
            "val": asdict(quantized_validation_metrics),
            "test": asdict(quantized_test_metrics),
            "quantizedTest": asdict(quantized_test_metrics),
        },
        "metricsSemantics": {
            "train": "Selected PyTorch checkpoint on the training split; descriptive only.",
            "val": "Deployed INT8 model on validation; used for confidence calibration.",
            "test": "Single final exported INT8 evaluation after model and confidence policy freeze.",
            "quantizedTest": "Compatibility alias of metrics.test; not a second test evaluation.",
            "pytorchValidation": asdict(training_metrics["val"]),
        },
        "sourceSplitCoverage": {
            "policy": "Preserve source-provided train, validation, and test assignments.",
            "classCount": len(labels),
            "labelsWithAllSplits": len(labels_with_all_splits),
            "missingValidationLabels": missing_validation_labels,
            "missingTestLabels": missing_test_labels,
            "perLabel": [
                {
                    "normalizedLabel": item["normalizedLabel"],
                    "trainSamples": item["trainSamples"],
                    "valSamples": item["valSamples"],
                    "testSamples": item["testSamples"],
                }
                for item in labels
            ],
        },
        "confidencePolicy": confidence_policy,
        "evaluationProtocol": {
            "modelSelectionSplit": "val",
            "thresholdCalibrationSplit": "val",
            "thresholdFrozenBeforeTestEvaluation": True,
            "reportedTestEvaluations": 1,
            "testRole": "final untouched evaluation of the exported INT8 model",
            "testUsedForTrainingOrCalibration": False,
        },
        "signerSplitAudit": signer_split_audit,
        "motionTimingAudit": motion_timing_audit,
        "training": training,
        "rawVideoStored": False,
        "rawAudioStored": False,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_training_results(
    output_path: Path,
    model_path: Path,
    input_path: Path,
    split_summary: dict[str, int],
    training_metrics: dict[str, Metrics],
    quantized_validation_metrics: Metrics,
    quantized_test_metrics: Metrics,
    confidence_policy: dict[str, Any],
    training: dict[str, Any],
    signer_split_audit: dict[str, Any],
    motion_timing_audit: dict[str, float | int | str],
    pretrained_path: Path,
    motion_output_path: Path,
    warmup_epochs: int,
    fine_tune_epochs: int,
    batch_size: int,
) -> None:
    calibration = confidence_policy["calibration"]
    selected_validation = calibration["selectedValidationMetrics"]
    model_mib = model_path.stat().st_size / 1024 / 1024
    signer_statement = str(signer_split_audit["message"])
    lines = [
        f"# SignSaarthi Temporal {len(SELECTED_LABELS)} Training Results",
        "",
        f"Generated: {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}",
        "",
        "## Data And Split Protocol",
        "",
        f"- Input: `{input_path}` (all locally extracted keypoint data; no raw dataset download)",
        f"- Classes: {len(SELECTED_LABELS)}",
        (
            f"- Sequences: {sum(split_summary.values())} total; {split_summary['train']} train, "
            f"{split_summary['val']} validation, {split_summary['test']} test"
        ),
        "- Model selection: validation accuracy, then validation macro F1",
        "- Confidence calibration: validation only",
        "- Test: one final exported INT8 evaluation after model and threshold freeze",
        f"- Signer split audit: {signer_split_audit['status']}. {signer_statement}",
        "",
        "## Calibration",
        "",
        f"- Target accepted accuracy: {float(calibration['targetAcceptedAccuracy']) * 100:.2f}%",
        f"- Minimum validation coverage: {float(calibration['minimumCoverage']) * 100:.2f}%",
        f"- Selected threshold: {float(confidence_policy['deployedThreshold']):.2f}",
        f"- Target met: {str(bool(calibration['targetMet'])).lower()}",
        (
            f"- Validation selective result: {float(selected_validation['acceptedAccuracy']) * 100:.2f}% "
            f"accepted accuracy at {float(selected_validation['coverage']) * 100:.2f}% coverage "
            f"({int(selected_validation['acceptedCount'])}/{int(calibration['sampleCount'])} accepted)"
        ),
        "- Calibration provenance: `split=val`, `testDataUsed=false`",
        "",
        "## Metrics",
        "",
        (
            f"- PyTorch train: {training_metrics['train'].accuracy * 100:.2f}% top-1, "
            f"{training_metrics['train'].macro_f1 * 100:.2f}% macro F1"
        ),
        (
            f"- Exported INT8 validation: {quantized_validation_metrics.accuracy * 100:.2f}% top-1, "
            f"{quantized_validation_metrics.top3_accuracy * 100:.2f}% top-3, "
            f"{quantized_validation_metrics.macro_f1 * 100:.2f}% macro F1 "
            f"({quantized_validation_metrics.sample_count} sequences)"
        ),
        (
            f"- Final untouched exported INT8 test: {quantized_test_metrics.accuracy * 100:.2f}% top-1, "
            f"{quantized_test_metrics.top3_accuracy * 100:.2f}% top-3, "
            f"{quantized_test_metrics.macro_f1 * 100:.2f}% macro F1 "
            f"({quantized_test_metrics.sample_count} sequences)"
        ),
        (
            f"- Frozen-threshold test policy: {float(confidence_policy['acceptedAccuracy']) * 100:.2f}% "
            f"accepted accuracy at {float(confidence_policy['coverage']) * 100:.2f}% coverage; "
            f"{int(confidence_policy['fallbackCount'])} caption fallbacks"
        ),
        "",
        "## Artifacts",
        "",
        f"- INT8 ONNX: `{model_path}` ({model_mib:.2f} MiB)",
        f"- INT8 SHA-256: `{sha256_path(model_path)}`",
        (
            f"- Motion timing: {int(motion_timing_audit['clipCount'])} clips passed "
            f"{float(motion_timing_audit['durationToleranceMs']):.3f} ms tolerance; maximum error "
            f"{float(motion_timing_audit['maxPlaybackDurationErrorMs']):.6f} ms"
        ),
        "",
        "## Reproduction",
        "",
        "```bash",
        (
            f"SIGNSAARTHI_TRAIN_DEVICE={training['device']} .venv-temporal/bin/python "
            "scripts/train_temporal_transformer.py "
            f"--input {input_path} --labels all --pretrained {pretrained_path} "
            f"--output-dir {model_path.parent} "
            f"--motion-output {motion_output_path} --warmup-epochs {warmup_epochs} "
            f"--epochs {fine_tune_epochs} --batch-size {batch_size} "
            f"--target-accepted-accuracy {float(calibration['targetAcceptedAccuracy']):.2f} "
            f"--minimum-calibration-coverage {float(calibration['minimumCoverage']):.2f}"
        ),
        (
            ".venv-temporal/bin/python scripts/evaluate_temporal_transformer.py "
            f"--dataset {input_path} --model {model_path} "
            f"--metadata {model_path.parent / 'isl-temporal-model.json'} "
            f"--output {model_path.parent / 'isl-temporal-evaluation.json'}"
        ),
        ".venv-temporal/bin/python -m unittest discover -s scripts/tests -p 'test_*.py' -v",
        "```",
        "",
        "## Limitations",
        "",
        "- Signer-disjointness is not established when signerHash is unavailable in source keypoint records.",
        (
            "- Source-provided validation and test coverage is incomplete for some labels; aggregate "
            "metrics cover only available held-out samples and have wide uncertainty."
        ),
        "- The exported vocabulary covers all labels in the local corpus, but the inputs are isolated signs, not continuous ISL sentences.",
        "- Motion clips are dataset-derived isolated signs and are not expert-certified ISL.",
        "- Raw video and audio are neither used by this retraining command nor stored in its artifacts.",
        "- The default output is the application's deployed data/models path and is mirrored under models/isl-temporal.",
        "",
        "## Training Runtime",
        "",
        f"- Device: `{training['device']}`",
        f"- Seed: {training['seed']}",
        f"- Fit duration: {float(training['durationSeconds']):.2f} seconds",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("data/isl/keypoints.include-combined.json"))
    parser.add_argument(
        "--labels",
        default=os.environ.get("SIGNSAARTHI_TEMPORAL_LABELS", "all"),
        help="Use 'all' or a comma-separated normalized label list.",
    )
    parser.add_argument(
        "--pretrained",
        type=Path,
        default=Path("data/raw/pretrained/include50-no-cnn-transformer-small.pth"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data/models"))
    parser.add_argument(
        "--mirror-dir",
        type=Path,
        default=Path("models/isl-temporal"),
        help="Mirror deployable artifacts and reports for inspection.",
    )
    parser.add_argument(
        "--motion-output",
        type=Path,
        default=Path("apps/extension/src/assets/motion/isl-motion-library.json"),
    )
    parser.add_argument(
        "--avatar-catalog-output",
        type=Path,
        default=Path("packages/avatar-engine/src/motionCatalog.generated.json"),
    )
    parser.add_argument("--warmup-epochs", type=int, default=int(os.environ.get("SIGNSAARTHI_WARMUP_EPOCHS", "3")))
    parser.add_argument("--epochs", type=int, default=int(os.environ.get("SIGNSAARTHI_TEMPORAL_EPOCHS", "18")))
    parser.add_argument("--batch-size", type=int, default=int(os.environ.get("SIGNSAARTHI_TEMPORAL_BATCH_SIZE", "16")))
    parser.add_argument(
        "--target-accepted-accuracy",
        type=float,
        default=CALIBRATION_TARGET_ACCEPTED_ACCURACY,
    )
    parser.add_argument(
        "--minimum-calibration-coverage",
        type=float,
        default=CALIBRATION_MIN_COVERAGE,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.target_accepted_accuracy <= 1.0:
        raise ValueError("--target-accepted-accuracy must be between 0 and 1.")
    if not 0.0 < args.minimum_calibration_coverage <= 1.0:
        raise ValueError("--minimum-calibration-coverage must be greater than 0 and at most 1.")
    if args.warmup_epochs < 0 or args.epochs < 0 or args.batch_size < 1:
        raise ValueError("Epoch counts must be nonnegative and --batch-size must be positive.")
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))

    if not args.input.exists():
        raise FileNotFoundError(f"Keypoint dataset not found: {args.input}")
    if not args.pretrained.exists():
        raise FileNotFoundError(
            f"Pretrained checkpoint not found: {args.pretrained}. Expected source: {PRETRAINED_URL}"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    labels = resolve_selected_labels(args.input, args.labels)
    print(
        f"Streaming {len(labels)} selected labels from {args.input} ({args.labels})...",
        flush=True,
    )
    samples = stream_selected_samples(args.input)
    signer_split_audit = audit_signer_splits(samples)
    print(
        f"Signer split audit: {signer_split_audit['status']} - {signer_split_audit['message']}",
        flush=True,
    )
    arrays, display_names, split_counts = prepare_arrays(samples)
    split_summary = {split: int(values[1].shape[0]) for split, values in arrays.items()}
    print(f"Selected {len(samples)} keypoint sequences: {split_summary}", flush=True)

    catalog_path = args.output_dir / "isl-motion-catalog.json"
    model_stem = f"signsaarthi-include-temporal-{len(SELECTED_LABELS)}"
    quantized_path = args.output_dir / f"{model_stem}.int8.onnx"
    with tempfile.TemporaryDirectory(prefix="signsaarthi-temporal-training-") as temporary_dir:
        temporary_root = Path(temporary_dir)
        model, training_metrics, training = train_model(
            arrays,
            args.pretrained,
            temporary_root / "best.pt",
            args.warmup_epochs,
            args.epochs,
            args.batch_size,
        )
        export_onnx(
            model,
            temporary_root / f"{model_stem}.onnx",
            quantized_path,
        )

    validation_logits, validation_labels = predict_onnx(quantized_path, arrays["val"])
    quantized_validation_metrics = calculate_metrics(validation_logits, validation_labels)
    calibrated_policy = calibrate_confidence_threshold(
        validation_logits,
        validation_labels,
        calibration_split="val",
        target_accepted_accuracy=args.target_accepted_accuracy,
        minimum_coverage=args.minimum_calibration_coverage,
    )
    frozen_threshold = float(calibrated_policy["deployedThreshold"])
    print(
        f"Froze confidence threshold {frozen_threshold:.2f} from validation calibration; "
        "running final test evaluation.",
        flush=True,
    )

    test_logits, test_labels = predict_onnx(quantized_path, arrays["test"])
    quantized_test_metrics = calculate_metrics(test_logits, test_labels)
    test_selective_metrics = calculate_selective_metrics(test_logits, test_labels, frozen_threshold)
    confidence_policy: dict[str, Any] = {
        "deployedThreshold": frozen_threshold,
        "fallback": "caption",
        "coverage": test_selective_metrics["coverage"],
        "acceptedAccuracy": test_selective_metrics["acceptedAccuracy"],
        "acceptedCount": test_selective_metrics["acceptedCount"],
        "fallbackCount": test_selective_metrics["fallbackCount"],
        "calibration": calibrated_policy["calibration"],
        "thresholdFrozenBeforeTestEvaluation": True,
        "testEvaluation": {
            "split": "test",
            "role": "final untouched evaluation",
            "testDataUsedForPolicySelection": False,
            "selectiveMetrics": test_selective_metrics,
        },
    }

    motion_timing_audit = write_motion_library(
        samples, display_names, args.motion_output, catalog_path
    )
    args.avatar_catalog_output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(catalog_path, args.avatar_catalog_output)
    print(
        f"Wrote {len(SELECTED_LABELS)} timestamp-timed motion clips to {args.motion_output}; "
        f"maximum duration error {float(motion_timing_audit['maxPlaybackDurationErrorMs']):.6f} ms.",
        flush=True,
    )

    training["modelSelectionSplit"] = "val"
    training["testUsedDuringFitting"] = False
    metadata_path = args.output_dir / "isl-temporal-model.json"
    write_metadata(
        metadata_path,
        quantized_path,
        display_names,
        split_counts,
        training_metrics,
        quantized_validation_metrics,
        quantized_test_metrics,
        confidence_policy,
        training,
        signer_split_audit,
        motion_timing_audit,
    )
    results_path = args.output_dir / "TRAINING_RESULTS.md"
    write_training_results(
        results_path,
        quantized_path,
        args.input,
        split_summary,
        training_metrics,
        quantized_validation_metrics,
        quantized_test_metrics,
        confidence_policy,
        training,
        signer_split_audit,
        motion_timing_audit,
        args.pretrained,
        args.motion_output,
        args.warmup_epochs,
        args.epochs,
        args.batch_size,
    )
    args.mirror_dir.mkdir(parents=True, exist_ok=True)
    for artifact_path in (quantized_path, metadata_path, catalog_path, results_path):
        shutil.copy2(artifact_path, args.mirror_dir / artifact_path.name)
    print(
        json.dumps(
            {
                "status": "trained",
                "model": str(quantized_path),
                "modelMiB": round(quantized_path.stat().st_size / 1024 / 1024, 2),
                "trainingMetrics": {
                    key: asdict(value) for key, value in training_metrics.items()
                },
                "quantizedValidation": asdict(quantized_validation_metrics),
                "finalQuantizedTest": asdict(quantized_test_metrics),
                "confidencePolicy": {
                    "deployedThreshold": confidence_policy["deployedThreshold"],
                    "calibration": confidence_policy["calibration"]["selectedValidationMetrics"],
                    "test": confidence_policy["testEvaluation"],
                },
                "metadata": str(metadata_path),
                "trainingResults": str(results_path),
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
