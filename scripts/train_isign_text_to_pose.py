#!/usr/bin/env python3
"""Train a compact text-conditioned iSign pose generator from the private corpus.

This is a research model. Technical pose metrics can qualify an artifact for
experimental preview, but never for certified or expert-reviewed ISL output.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import random
import re
import sqlite3
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


SEED = 186
FRAME_COUNT = 64
POINT_COUNT = 93
COORDINATE_COUNT = 2
MAX_TOKENS = 20
TOKEN_PATTERN = re.compile(r"[\w]+(?:['’-][\w]+)*", re.UNICODE)
PRIVATE_DATA_ROOT = Path("data/private")


class TrainingError(RuntimeError):
    """Raised when a training or evaluation contract is not satisfied."""


@dataclass(frozen=True)
class EvaluationMetrics:
    normalized_mpjpe: float
    velocity_mpjpe: float
    normalized_dtw: float
    dtw_sample_count: int
    predicted_hand_motion_energy: float
    target_hand_motion_energy: float
    hand_motion_energy_ratio: float
    non_static_sample_rate: float
    sample_count: int


def tokenize(value: str) -> tuple[str, ...]:
    return tuple(match.group(0).casefold() for match in TOKEN_PATTERN.finditer(value))


def build_vocabulary(
    train_texts: Iterable[str], max_vocabulary: int = 8_000, minimum_frequency: int = 2
) -> dict[str, int]:
    counts = Counter(token for text in train_texts for token in tokenize(text))
    ranked = sorted(
        (item for item in counts.items() if item[1] >= minimum_frequency),
        key=lambda item: (-item[1], item[0]),
    )[: max(0, max_vocabulary - 2)]
    return {"<pad>": 0, "<unk>": 1, **{token: index + 2 for index, (token, _) in enumerate(ranked)}}


def encode_text(value: str, vocabulary: dict[str, int], max_tokens: int = MAX_TOKENS) -> np.ndarray:
    output = np.zeros(max_tokens, dtype=np.int64)
    tokens = tokenize(value)[:max_tokens]
    output[: len(tokens)] = [vocabulary.get(token, 1) for token in tokens]
    return output


def split_leakage_report(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, set[str]] = {"train": set(), "val": set(), "test": set()}
    for row in rows:
        groups[str(row["split"])].add(str(row["video_hash"]))
    overlaps = {
        "trainVal": len(groups["train"] & groups["val"]),
        "trainTest": len(groups["train"] & groups["test"]),
        "valTest": len(groups["val"] & groups["test"]),
    }
    return {
        "uniqueVideoGroups": {split: len(values) for split, values in groups.items()},
        "overlaps": overlaps,
        "passed": not any(overlaps.values()),
    }


def load_rows(database_path: Path) -> list[dict[str, Any]]:
    connection = sqlite3.connect(f"file:{database_path}?mode=ro", uri=True)
    try:
        rows = [
            {
                "uid_hash": uid_hash,
                "video_hash": video_hash,
                "split": split,
                "text": text,
                "quality_score": quality_score,
                "pose_npz": pose_npz,
            }
            for uid_hash, video_hash, split, text, quality_score, pose_npz in connection.execute(
                "SELECT uid_hash, video_hash, split, text, quality_score, pose_npz FROM samples"
            )
        ]
    finally:
        connection.close()
    if not rows:
        raise TrainingError("The private sentence-pose corpus is empty.")
    report = split_leakage_report(rows)
    if not report["passed"]:
        raise TrainingError("Video-group leakage exists between sentence-pose splits.")
    return rows


def decode_pose_blob(blob: bytes) -> tuple[np.ndarray, np.ndarray]:
    with np.load(io.BytesIO(blob)) as payload:
        xy = np.asarray(payload["xy"], dtype=np.float32)
        confidence = np.asarray(payload["confidence"], dtype=np.float32) / 255.0
    if xy.shape != (FRAME_COUNT, POINT_COUNT, COORDINATE_COUNT):
        raise TrainingError(f"Unexpected pose tensor shape: {xy.shape}")
    if confidence.shape != (FRAME_COUNT, POINT_COUNT):
        raise TrainingError(f"Unexpected confidence tensor shape: {confidence.shape}")
    if not np.isfinite(xy).all() or not np.isfinite(confidence).all():
        raise TrainingError("Pose corpus contains non-finite values.")
    return xy, confidence


def prepare_arrays(
    rows: Sequence[dict[str, Any]], vocabulary: dict[str, int]
) -> dict[str, dict[str, np.ndarray]]:
    by_split: dict[str, list[dict[str, Any]]] = {"train": [], "val": [], "test": []}
    for row in rows:
        by_split[str(row["split"])].append(row)
    arrays: dict[str, dict[str, np.ndarray]] = {}
    for split in ("train", "val", "test"):
        split_rows = sorted(by_split[split], key=lambda row: str(row["uid_hash"]))
        if not split_rows:
            raise TrainingError(f"The {split} split is empty.")
        poses: list[np.ndarray] = []
        confidences: list[np.ndarray] = []
        tokens: list[np.ndarray] = []
        for row in split_rows:
            xy, confidence = decode_pose_blob(row["pose_npz"])
            poses.append(xy)
            confidences.append(confidence)
            tokens.append(encode_text(str(row["text"]), vocabulary))
        arrays[split] = {
            "tokens": np.stack(tokens),
            "pose": np.stack(poses).astype(np.float32),
            "confidence": np.stack(confidences).astype(np.float32),
        }
    return arrays


def weighted_normalization(
    pose: np.ndarray, confidence: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    weights = np.maximum(confidence[..., None], 0.02)
    total = np.maximum(weights.sum(axis=0), 1e-6)
    mean = (pose * weights).sum(axis=0) / total
    variance = (((pose - mean) ** 2) * weights).sum(axis=0) / total
    scale = np.sqrt(np.maximum(variance, 1e-4))
    standardized = (pose - mean) / scale
    return standardized.astype(np.float32), mean.astype(np.float32), scale.astype(np.float32)


def apply_normalization(pose: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return ((pose - mean) / scale).astype(np.float32)


def decode_normalization(values: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return values * scale + mean


def point_weights() -> np.ndarray:
    weights = np.ones((1, 1, POINT_COUNT, 1), dtype=np.float32)
    weights[:, :, 33:75] = 2.0
    weights[:, :, 75:93] = 0.6
    return weights


def weighted_mpjpe(prediction: np.ndarray, target: np.ndarray, confidence: np.ndarray) -> float:
    distance = np.linalg.norm(prediction - target, axis=-1)
    weights = confidence * point_weights()[0, 0, :, 0]
    return float((distance * weights).sum() / max(1e-8, weights.sum()))


def velocity_mpjpe(prediction: np.ndarray, target: np.ndarray, confidence: np.ndarray) -> float:
    prediction_velocity = np.diff(prediction, axis=1)
    target_velocity = np.diff(target, axis=1)
    velocity_confidence = np.minimum(confidence[:, 1:], confidence[:, :-1])
    return weighted_mpjpe(prediction_velocity, target_velocity, velocity_confidence)


def sequence_dtw(first: np.ndarray, second: np.ndarray) -> float:
    first_frames = first.reshape(first.shape[0], -1)
    second_frames = second.reshape(second.shape[0], -1)
    previous = np.full(second_frames.shape[0] + 1, np.inf, dtype=np.float64)
    previous[0] = 0
    for left in first_frames:
        current = np.full(second_frames.shape[0] + 1, np.inf, dtype=np.float64)
        distances = np.sqrt(np.mean((second_frames - left) ** 2, axis=1))
        for index, distance in enumerate(distances, 1):
            current[index] = distance + min(current[index - 1], previous[index], previous[index - 1])
        previous = current
    return float(previous[-1] / max(first_frames.shape[0], second_frames.shape[0]))


def per_sample_hand_motion_energy(
    pose: np.ndarray, confidence: np.ndarray
) -> np.ndarray:
    velocity = np.linalg.norm(np.diff(pose[:, :, 33:75], axis=1), axis=-1)
    visible = np.minimum(confidence[:, 1:, 33:75], confidence[:, :-1, 33:75]) >= 0.08
    numerator = (velocity * visible).sum(axis=(1, 2))
    denominator = np.maximum(1, visible.sum(axis=(1, 2)))
    return numerator / denominator


def evaluate_arrays(
    prediction: np.ndarray,
    target: np.ndarray,
    confidence: np.ndarray,
    dtw_limit: int = 128,
) -> EvaluationMetrics:
    if prediction.shape != target.shape:
        raise TrainingError("Prediction and target pose shapes differ.")
    dtw_count = min(dtw_limit, prediction.shape[0])
    dtw = float(
        np.mean([sequence_dtw(prediction[index], target[index]) for index in range(dtw_count)])
    )
    predicted_energy = per_sample_hand_motion_energy(prediction, confidence)
    target_energy = per_sample_hand_motion_energy(target, confidence)
    mean_predicted_energy = float(predicted_energy.mean())
    mean_target_energy = float(target_energy.mean())
    return EvaluationMetrics(
        normalized_mpjpe=weighted_mpjpe(prediction, target, confidence),
        velocity_mpjpe=velocity_mpjpe(prediction, target, confidence),
        normalized_dtw=dtw,
        dtw_sample_count=dtw_count,
        predicted_hand_motion_energy=mean_predicted_energy,
        target_hand_motion_energy=mean_target_energy,
        hand_motion_energy_ratio=mean_predicted_energy / max(mean_target_energy, 1e-8),
        non_static_sample_rate=float((predicted_energy >= 0.01).mean()),
        sample_count=int(prediction.shape[0]),
    )


def create_model(torch: Any, vocabulary_size: int, hidden_size: int, layers: int, heads: int) -> Any:
    nn = torch.nn

    class TextToPose(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.token_embedding = nn.Embedding(vocabulary_size, hidden_size, padding_idx=0)
            self.token_position = nn.Parameter(torch.zeros(1, MAX_TOKENS, hidden_size))
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=heads,
                dim_feedforward=hidden_size * 4,
                dropout=0.1,
                batch_first=True,
                norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(
                encoder_layer, num_layers=layers, enable_nested_tensor=False
            )
            decoder_layer = nn.TransformerDecoderLayer(
                d_model=hidden_size,
                nhead=heads,
                dim_feedforward=hidden_size * 4,
                dropout=0.1,
                batch_first=True,
                norm_first=True,
            )
            self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=layers)
            self.frame_queries = nn.Parameter(torch.randn(1, FRAME_COUNT, hidden_size) * 0.02)
            self.output = nn.Linear(hidden_size, POINT_COUNT * COORDINATE_COUNT)
            nn.init.normal_(self.token_position, std=0.02)

        def forward(self, token_ids: Any, attention_mask: Any) -> Any:
            safe_attention = attention_mask.bool()
            first_position = torch.zeros_like(safe_attention)
            first_position[:, 0] = True
            safe_attention = safe_attention | (~safe_attention.any(dim=1, keepdim=True) & first_position)
            padding_mask = ~safe_attention
            memory = self.token_embedding(token_ids) + self.token_position
            memory = self.encoder(memory, src_key_padding_mask=padding_mask)
            weights = safe_attention.unsqueeze(-1).to(memory.dtype)
            sentence = (memory * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1)
            queries = self.frame_queries.expand(token_ids.shape[0], -1, -1) + sentence.unsqueeze(1)
            decoded = self.decoder(queries, memory, memory_key_padding_mask=padding_mask)
            return self.output(decoded).reshape(
                token_ids.shape[0], FRAME_COUNT, POINT_COUNT, COORDINATE_COUNT
            )

    return TextToPose()


def torch_loss(torch: Any, prediction: Any, target: Any, confidence: Any) -> Any:
    weights = confidence.unsqueeze(-1).clamp(min=0.02)
    part_weights = torch.ones((1, 1, POINT_COUNT, 1), device=prediction.device)
    part_weights[:, :, 33:75] = 2.0
    part_weights[:, :, 75:93] = 0.6
    weights = weights * part_weights
    position = (((prediction - target) ** 2) * weights).sum() / weights.sum().clamp(min=1)
    velocity_weights = torch.minimum(weights[:, 1:], weights[:, :-1])
    velocity_error = (prediction[:, 1:] - prediction[:, :-1]) - (target[:, 1:] - target[:, :-1])
    velocity = ((velocity_error**2) * velocity_weights).sum() / velocity_weights.sum().clamp(min=1)
    acceleration_error = (prediction[:, 2:] - 2 * prediction[:, 1:-1] + prediction[:, :-2]) - (
        target[:, 2:] - 2 * target[:, 1:-1] + target[:, :-2]
    )
    acceleration_weights = torch.minimum(velocity_weights[:, 1:], velocity_weights[:, :-1])
    acceleration = ((acceleration_error**2) * acceleration_weights).sum() / acceleration_weights.sum().clamp(min=1)
    hand_prediction_velocity = prediction[:, 1:, 33:75] - prediction[:, :-1, 33:75]
    hand_target_velocity = target[:, 1:, 33:75] - target[:, :-1, 33:75]
    hand_visibility = torch.minimum(confidence[:, 1:, 33:75], confidence[:, :-1, 33:75]).clamp(
        min=0.02
    )
    predicted_speed = torch.linalg.vector_norm(hand_prediction_velocity, dim=-1)
    target_speed = torch.linalg.vector_norm(hand_target_velocity, dim=-1)
    speed_alignment = (
        ((predicted_speed - target_speed) ** 2) * hand_visibility
    ).sum() / hand_visibility.sum().clamp(min=1)
    denominator = hand_visibility.sum(dim=(1, 2)).clamp(min=1)
    predicted_energy = (predicted_speed * hand_visibility).sum(dim=(1, 2)) / denominator
    target_energy = (target_speed * hand_visibility).sum(dim=(1, 2)) / denominator
    energy_alignment = ((predicted_energy - target_energy) ** 2).mean()
    return (
        position
        + 0.4 * velocity
        + 0.05 * acceleration
        + 1.5 * speed_alignment
        + 1.5 * energy_alignment
    )


def choose_device(torch: Any, requested: str) -> Any:
    if requested == "auto":
        if torch.backends.mps.is_available():
            requested = "mps"
        elif torch.cuda.is_available():
            requested = "cuda"
        else:
            requested = "cpu"
    return torch.device(requested)


def predict(torch: Any, model: Any, split: dict[str, np.ndarray], device: Any, batch_size: int) -> np.ndarray:
    model.eval()
    outputs: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, split["tokens"].shape[0], batch_size):
            token_ids = torch.from_numpy(split["tokens"][start : start + batch_size]).to(device)
            attention = token_ids.ne(0)
            outputs.append(model(token_ids, attention).cpu().numpy())
    return np.concatenate(outputs)


def train_model(
    torch: Any,
    arrays: dict[str, dict[str, np.ndarray]],
    vocabulary_size: int,
    args: argparse.Namespace,
) -> tuple[Any, list[dict[str, float]], Any]:
    device = choose_device(torch, args.device)
    model = create_model(torch, vocabulary_size, args.hidden_size, args.layers, args.heads).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    generator = torch.Generator().manual_seed(SEED)
    train_tokens = torch.from_numpy(arrays["train"]["tokens"])
    train_pose = torch.from_numpy(arrays["train"]["standardized"])
    train_confidence = torch.from_numpy(arrays["train"]["confidence"])
    dataset = torch.utils.data.TensorDataset(train_tokens, train_pose, train_confidence)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
    )
    history: list[dict[str, float]] = []
    best_state: dict[str, Any] | None = None
    best_selection_score = math.inf
    stale_epochs = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses: list[float] = []
        for token_ids, target, confidence in loader:
            token_ids = token_ids.to(device)
            target = target.to(device)
            confidence = confidence.to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(token_ids, token_ids.ne(0))
            loss = torch_loss(torch, prediction, target, confidence)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        val_prediction = predict(torch, model, arrays["val"], device, args.batch_size)
        val_loss = weighted_mpjpe(
            val_prediction, arrays["val"]["standardized"], arrays["val"]["confidence"]
        )
        predicted_energy = float(
            per_sample_hand_motion_energy(
                val_prediction, arrays["val"]["confidence"]
            ).mean()
        )
        target_energy = float(
            per_sample_hand_motion_energy(
                arrays["val"]["standardized"], arrays["val"]["confidence"]
            ).mean()
        )
        motion_ratio = predicted_energy / max(target_energy, 1e-8)
        selection_score = val_loss + 0.25 * abs(math.log(max(motion_ratio, 1e-6)))
        row = {
            "epoch": float(epoch),
            "trainLoss": float(np.mean(losses)),
            "valStandardizedMpjpe": val_loss,
            "valHandMotionEnergyRatio": motion_ratio,
            "selectionScore": selection_score,
        }
        history.append(row)
        print(
            f"Epoch {epoch}/{args.epochs} train_loss={row['trainLoss']:.5f} "
            f"val_mpjpe={val_loss:.5f} val_motion_ratio={motion_ratio:.4f} "
            f"selection={selection_score:.5f}",
            flush=True,
        )
        if selection_score < best_selection_score - 1e-5:
            best_selection_score = selection_score
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                print(f"Early stopping after {args.patience} non-improving epochs.", flush=True)
                break
    if best_state is None:
        raise TrainingError("Training did not produce a checkpoint.")
    model.load_state_dict(best_state)
    return model.cpu().eval(), history, device


def export_onnx(torch: Any, model: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    token_ids = torch.ones((1, MAX_TOKENS), dtype=torch.long)
    attention = token_ids.ne(0)
    torch.onnx.export(
        model,
        (token_ids, attention),
        path,
        input_names=["token_ids", "attention_mask"],
        output_names=["pose"],
        dynamic_axes={"token_ids": {0: "batch"}, "attention_mask": {0: "batch"}, "pose": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )


def verify_onnx_export(torch: Any, model: Any, path: Path) -> dict[str, Any]:
    try:
        import onnx
        import onnxruntime as ort
    except ImportError as exc:
        raise TrainingError("ONNX and ONNX Runtime are required to validate the exported model.") from exc

    onnx.checker.check_model(onnx.load(path))
    token_ids = np.zeros((3, MAX_TOKENS), dtype=np.int64)
    token_ids[:2, :3] = np.array([[2, 3, 4], [4, 3, 2]], dtype=np.int64)
    attention = token_ids != 0
    with torch.no_grad():
        expected = model(torch.from_numpy(token_ids), torch.from_numpy(attention)).numpy()
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    actual = session.run(None, {"token_ids": token_ids, "attention_mask": attention})[0]
    if actual.shape != expected.shape or not np.isfinite(actual).all():
        raise TrainingError("The exported ONNX model returned an invalid pose tensor.")
    maximum_error = float(np.max(np.abs(expected - actual)))
    if not np.allclose(expected, actual, rtol=2e-4, atol=2e-4):
        raise TrainingError(f"ONNX parity check failed; maximum absolute error was {maximum_error:.6f}.")
    return {
        "passed": True,
        "batchSize": int(actual.shape[0]),
        "outputShape": list(actual.shape),
        "maximumAbsoluteError": maximum_error,
        "provider": "CPUExecutionProvider",
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_private_path(path: Path, label: str) -> None:
    try:
        path.resolve().relative_to(PRIVATE_DATA_ROOT.resolve())
    except ValueError as exc:
        raise TrainingError(
            f"{label} must remain under {PRIVATE_DATA_ROOT} because iSign artifacts are gated and non-redistributable."
        ) from exc


def load_validation_report(
    path: Path,
    database_path: Path,
    plan_path: Path,
    word_library_path: Path,
    expected_rows: int,
) -> dict[str, Any]:
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TrainingError("The required private corpus validation report is unreadable.") from exc
    database_sha256 = sha256_file(database_path)
    if (
        report.get("status") != "valid"
        or report.get("acceptedRows") != expected_rows
        or report.get("databaseSha256") != database_sha256
        or report.get("planSha256") != sha256_file(plan_path)
        or report.get("rawArchiveStored") is not False
        or report.get("fullArchiveDownloaded") is not False
        or report.get("wordMotionLibrary", {}).get("status") != "valid"
        or report.get("wordMotionLibrary", {}).get("fileSha256")
        != sha256_file(word_library_path)
    ):
        raise TrainingError("The private corpus no longer matches its validation report.")
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "databaseSha256": database_sha256,
        "acceptedRows": expected_rows,
    }


def technical_gate(
    validation: EvaluationMetrics,
    test: EvaluationMetrics,
    baseline_validation: EvaluationMetrics,
    baseline_test: EvaluationMetrics,
    sample_count: int,
) -> dict[str, Any]:
    def improvement(value: float, baseline: float) -> float:
        if baseline <= 0:
            return 1.0 if value <= 0 else -math.inf
        return 1 - value / baseline

    validation_improvement = improvement(
        validation.normalized_mpjpe, baseline_validation.normalized_mpjpe
    )
    test_improvement = improvement(test.normalized_mpjpe, baseline_test.normalized_mpjpe)
    validation_velocity_improvement = improvement(
        validation.velocity_mpjpe, baseline_validation.velocity_mpjpe
    )
    validation_dtw_improvement = improvement(
        validation.normalized_dtw, baseline_validation.normalized_dtw
    )
    checks = {
        "minimumCorpusRows": sample_count >= 1_000,
        "validationMpjpeWithinTenPercentOfMean": validation.normalized_mpjpe
        <= baseline_validation.normalized_mpjpe * 1.10,
        "validationVelocityWithinFiftyPercentOfMean": validation.velocity_mpjpe
        <= baseline_validation.velocity_mpjpe * 1.50,
        "validationDtwWithinTenPercentOfMean": validation.normalized_dtw
        <= baseline_validation.normalized_dtw * 1.10,
        "validationMotionEnergyWithinRange": 0.20
        <= validation.hand_motion_energy_ratio
        <= 5.0,
        "validationNonStaticSampleRate": validation.non_static_sample_rate >= 0.90,
        "validationFiniteMetrics": all(
            math.isfinite(value)
            for value in (
                validation.normalized_mpjpe,
                validation.velocity_mpjpe,
                validation.normalized_dtw,
                validation.predicted_hand_motion_energy,
                validation.target_hand_motion_energy,
                validation.hand_motion_energy_ratio,
            )
        ),
    }
    test_values = (
        test.normalized_mpjpe,
        test.velocity_mpjpe,
        test.normalized_dtw,
        test.predicted_hand_motion_energy,
        test.target_hand_motion_energy,
        test.hand_motion_energy_ratio,
    )
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "validationMpjpeImprovement": validation_improvement,
        "validationVelocityImprovement": validation_velocity_improvement,
        "validationDtwImprovement": validation_dtw_improvement,
        "testMpjpeImprovement": test_improvement,
        "testAudit": {
            "reportOnlyNotUsedForModelSelection": True,
            "finite": all(math.isfinite(value) for value in test_values),
        },
        "runtimeEligible": False,
        "runtimeBlocker": (
            "The dynamic geometry gate does not measure linguistic ISL correctness; "
            "Deaf/ISL expert sentence-level validation has not been completed."
        ),
    }


def write_results(path: Path, artifact: dict[str, Any]) -> None:
    metrics = artifact["metrics"]
    gate = artifact["qualityGate"]
    splits = artifact["trainingDataset"]["splits"]
    groups = artifact["trainingDataset"]["leakageAudit"]["uniqueVideoGroups"]
    lines = [
        "# SignSaarthi iSign Text-to-Pose Research Result",
        "",
        f"- Corpus rows: {artifact['trainingDataset']['recordCount']:,}",
        f"- Train/validation/test: {splits['train']:,} / {splits['val']:,} / {splits['test']:,}",
        f"- Unique video groups (train/validation/test): {groups['train']:,} / {groups['val']:,} / {groups['test']:,}",
        f"- Dynamic geometry preview gate: {'passed' if gate['passed'] else 'failed'}",
        "- Runtime eligibility: blocked pending Deaf/ISL expert validation",
        "",
        "## Metrics",
        "",
        f"- Validation normalized MPJPE: {metrics['validation']['normalized_mpjpe']:.6f}",
        f"- Test normalized MPJPE: {metrics['test']['normalized_mpjpe']:.6f}",
        f"- Test velocity MPJPE: {metrics['test']['velocity_mpjpe']:.6f}",
        f"- Test normalized DTW: {metrics['test']['normalized_dtw']:.6f}",
        f"- DTW sample count: {metrics['test']['dtw_sample_count']} of {metrics['test']['sample_count']} held-out sequences",
        f"- Validation hand-motion energy retained: {metrics['validation']['hand_motion_energy_ratio'] * 100:.2f}%",
        f"- Test hand-motion energy retained: {metrics['test']['hand_motion_energy_ratio'] * 100:.2f}%",
        f"- Test non-static generated sequences: {metrics['test']['non_static_sample_rate'] * 100:.2f}%",
        f"- Mean-baseline test MPJPE: {metrics['meanBaselineTest']['normalized_mpjpe']:.6f}",
        f"- ONNX parity maximum absolute error: {artifact['model']['onnxValidation']['maximumAbsoluteError']:.9f}",
        "- Held-out test metrics are report-only and were not used as the model-selection gate.",
        "",
        "## Scope",
        "",
        "This model is trained on gated, noncommercial iSign pose-caption pairs. It is an experimental geometric pose generator, not a certified interpreter. Automated MPJPE/DTW metrics do not establish linguistic accuracy, regional appropriateness, or non-manual grammatical correctness.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, default=Path("data/private/isign-research/sentence-pose-corpus.private.sqlite"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/private/isign-research/text-to-pose"))
    parser.add_argument("--plan", type=Path, default=Path("data/private/isign-research/sentence-plan.private.jsonl"))
    parser.add_argument("--word-library", type=Path, default=Path("data/private/isign-research/word-motion-library.json.gz"))
    parser.add_argument("--epochs", type=int, default=18)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--max-vocabulary", type=int, default=8_000)
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument(
        "--validation-report",
        type=Path,
        default=Path("data/private/isign-research/validation-report.private.json"),
    )
    parser.add_argument("--accept-research-only-terms", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.accept_research_only_terms:
        raise SystemExit("Pass --accept-research-only-terms after accepting the gated iSign terms.")
    if args.epochs < 1 or args.batch_size < 1 or args.hidden_size < 32 or args.layers < 1:
        raise SystemExit("Training dimensions and epoch counts must be positive.")
    for path, label in (
        (args.database, "Training database"),
        (args.output_dir, "Model output"),
        (args.plan, "Sentence plan"),
        (args.word_library, "Word-motion library"),
        (args.validation_report, "Validation report"),
    ):
        require_private_path(path, label)
    try:
        import torch
    except ImportError as exc:
        raise SystemExit("Install scripts/requirements-isign-training.txt before training.") from exc

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    rows = load_rows(args.database)
    validation_report = load_validation_report(
        args.validation_report,
        args.database,
        args.plan,
        args.word_library,
        len(rows),
    )
    leakage = split_leakage_report(rows)
    train_texts = [str(row["text"]) for row in rows if row["split"] == "train"]
    vocabulary = build_vocabulary(train_texts, args.max_vocabulary)
    arrays = prepare_arrays(rows, vocabulary)
    train_standardized, mean, scale = weighted_normalization(
        arrays["train"]["pose"], arrays["train"]["confidence"]
    )
    arrays["train"]["standardized"] = train_standardized
    for split in ("val", "test"):
        arrays[split]["standardized"] = apply_normalization(arrays[split]["pose"], mean, scale)

    model, history, training_device = train_model(torch, arrays, len(vocabulary), args)
    predictions: dict[str, np.ndarray] = {}
    for split in ("val", "test"):
        standardized_prediction = predict(
            torch, model, arrays[split], torch.device("cpu"), args.batch_size
        )
        predictions[split] = decode_normalization(standardized_prediction, mean, scale)

    validation = evaluate_arrays(
        predictions["val"], arrays["val"]["pose"], arrays["val"]["confidence"]
    )
    test = evaluate_arrays(
        predictions["test"], arrays["test"]["pose"], arrays["test"]["confidence"]
    )
    mean_validation_prediction = np.broadcast_to(
        mean, arrays["val"]["pose"].shape
    ).astype(np.float32, copy=False)
    mean_test_prediction = np.broadcast_to(mean, arrays["test"]["pose"].shape).astype(
        np.float32, copy=False
    )
    baseline_validation = evaluate_arrays(
        mean_validation_prediction, arrays["val"]["pose"], arrays["val"]["confidence"]
    )
    baseline_test = evaluate_arrays(
        mean_test_prediction, arrays["test"]["pose"], arrays["test"]["confidence"]
    )
    gate = technical_gate(validation, test, baseline_validation, baseline_test, len(rows))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = args.output_dir / "isign-text-to-pose.onnx"
    export_onnx(torch, model, onnx_path)
    onnx_validation = verify_onnx_export(torch, model, onnx_path)
    normalization_path = args.output_dir / "normalization.npz"
    np.savez_compressed(
        normalization_path,
        mean=mean.astype(np.float32),
        scale=scale.astype(np.float32),
        confidence=np.mean(arrays["train"]["confidence"], axis=0).astype(np.float16),
    )
    vocabulary_path = args.output_dir / "vocabulary.json"
    vocabulary_path.write_text(
        json.dumps(vocabulary, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8"
    )
    splits = {split: int(arrays[split]["tokens"].shape[0]) for split in ("train", "val", "test")}
    artifact = {
        "schemaVersion": 1,
        "model": {
            "id": "signsaarthi-isign-text-to-pose-research-v1",
            "engine": "text_conditioned_temporal_transformer",
            "onnxFile": onnx_path.name,
            "onnxSha256": sha256_file(onnx_path),
            "vocabularyFile": vocabulary_path.name,
            "vocabularySha256": sha256_file(vocabulary_path),
            "normalizationFile": normalization_path.name,
            "normalizationSha256": sha256_file(normalization_path),
            "requiresAtLeastOneToken": True,
            "frameCount": FRAME_COUNT,
            "pointCount": POINT_COUNT,
            "maxTokens": MAX_TOKENS,
            "vocabularySize": len(vocabulary),
            "hiddenSize": args.hidden_size,
            "layers": args.layers,
            "heads": args.heads,
            "onnxValidation": onnx_validation,
        },
        "trainingDataset": {
            "source": "Exploration-Lab/iSign",
            "license": "CC BY-NC-SA 4.0 plus gated noncommercial/no-redistribution terms",
            "recordCount": len(rows),
            "splits": splits,
            "leakageAudit": leakage,
            "rawArchiveStored": False,
            "validationReport": validation_report,
        },
        "training": {
            "seed": SEED,
            "device": str(training_device),
            "epochsRequested": args.epochs,
            "epochsCompleted": len(history),
            "batchSize": args.batch_size,
            "learningRate": args.learning_rate,
            "loss": {
                "position": 1.0,
                "velocity": 0.4,
                "acceleration": 0.05,
                "handSpeedAlignment": 1.5,
                "handEnergyAlignment": 1.5,
            },
            "history": history,
        },
        "metrics": {
            "validation": asdict(validation),
            "test": asdict(test),
            "meanBaselineValidation": asdict(baseline_validation),
            "meanBaselineTest": asdict(baseline_test),
        },
        "qualityGate": gate,
        "expertReview": {
            "status": "pending",
            "requiredForRuntime": True,
            "statement": "No automated metric can certify linguistic ISL correctness.",
        },
    }
    metadata_path = args.output_dir / "metadata.json"
    metadata_path.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    write_results(args.output_dir / "TRAINING_RESULTS.md", artifact)
    print(
        json.dumps(
            {
                "output": str(args.output_dir),
                "rows": len(rows),
                "splits": splits,
                "test": asdict(test),
                "technicalGatePassed": gate["passed"],
                "runtimeEligible": gate["runtimeEligible"],
                "runtimeBlocker": gate["runtimeBlocker"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
