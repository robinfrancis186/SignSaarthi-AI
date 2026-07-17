#!/usr/bin/env python3
"""Run the private iSign Text2Pose ONNX model and render a local research preview."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from train_isign_text_to_pose import MAX_TOKENS, encode_text, tokenize


POSE_CONNECTIONS = ((11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (11, 23), (12, 24), (23, 24))
HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),
)
PRIVATE_DATA_ROOT = Path("data/private")
MIN_PREVIEW_MOTION_ENERGY = 0.001


class PreviewError(RuntimeError):
    """Raised when a private research preview cannot be produced safely."""


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
        raise PreviewError(
            f"{label} must remain under {PRIVATE_DATA_ROOT} because iSign artifacts are gated and non-redistributable."
        ) from exc


def canonicalize_for_avatar(xy: np.ndarray, confidence: np.ndarray) -> np.ndarray:
    output = np.asarray(xy, dtype=np.float32).copy()
    visible = confidence >= 0.08
    visible_points = output[visible]
    if not visible_points.size:
        raise PreviewError("The generated pose has no visible landmarks.")
    minimum = np.percentile(visible_points, 1, axis=0)
    maximum = np.percentile(visible_points, 99, axis=0)
    extent = np.maximum(maximum - minimum, 1e-4)
    scale = min(0.78 / float(extent[0]), 0.82 / float(extent[1]))
    center = (minimum + maximum) / 2
    output = (output - center) * scale + np.asarray([0.5, 0.52], dtype=np.float32)
    return np.clip(output, -0.2, 1.2)


def motion_energy(xy: np.ndarray, confidence: np.ndarray) -> float:
    hand_xy = xy[:, 33:75]
    hand_confidence = np.minimum(confidence[1:, 33:75], confidence[:-1, 33:75])
    velocity = np.linalg.norm(np.diff(hand_xy, axis=0), axis=-1)
    weights = hand_confidence >= 0.08
    return float(velocity[weights].mean()) if weights.any() else 0.0


def frame_payload(xy: np.ndarray, confidence: np.ndarray) -> list[dict[str, Any]]:
    parts = {"pose": (0, 33), "leftHand": (33, 54), "rightHand": (54, 75), "face": (75, 93)}
    return [
        {
            name: [
                {
                    "x": round(float(xy[frame, index, 0]), 5),
                    "y": round(float(xy[frame, index, 1]), 5),
                    "visibility": round(float(confidence[frame, index]), 4),
                }
                for index in range(start, end)
            ]
            for name, (start, end) in parts.items()
        }
        for frame in range(xy.shape[0])
    ]


def render_gif(xy: np.ndarray, confidence: np.ndarray, path: Path, label: str) -> None:
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError as exc:
        raise PreviewError("Install scripts/requirements-isign-training.txt to render GIF previews.") from exc

    size = 640
    scale = 2
    title_font = ImageFont.load_default(size=18 * scale)
    label_font = ImageFont.load_default(size=14 * scale)
    frames = []
    for frame_index in range(xy.shape[0]):
        image = Image.new("RGB", (size * scale, size * scale), "#081116")
        draw = ImageDraw.Draw(image)

        def point(index: int) -> tuple[float, float]:
            return (
                float(xy[frame_index, index, 0] * size * scale),
                float((0.12 + xy[frame_index, index, 1] * 0.78) * size * scale),
            )

        shoulders = [point(11), point(12)]
        hips = [point(23), point(24)]
        draw.polygon([shoulders[0], shoulders[1], hips[1], hips[0]], fill="#162a35", outline="#59d0c7", width=3 * scale)
        for first, second in POSE_CONNECTIONS:
            if min(confidence[frame_index, first], confidence[frame_index, second]) >= 0.08:
                draw.line((point(first), point(second)), fill="#d8eff0", width=7 * scale, joint="curve")

        shoulder_center = ((shoulders[0][0] + shoulders[1][0]) / 2, (shoulders[0][1] + shoulders[1][1]) / 2)
        head_center = point(0)
        radius = max(22 * scale, abs(shoulder_center[1] - head_center[1]) * 0.58)
        draw.ellipse(
            (head_center[0] - radius * 0.72, head_center[1] - radius, head_center[0] + radius * 0.72, head_center[1] + radius),
            fill="#d89d78",
            outline="#f7d6bd",
            width=3 * scale,
        )

        for offset, color in ((33, "#70e3d4"), (54, "#f2b36f")):
            for first, second in HAND_CONNECTIONS:
                left = offset + first
                right = offset + second
                if min(confidence[frame_index, left], confidence[frame_index, right]) >= 0.08:
                    draw.line((point(left), point(right)), fill=color, width=3 * scale, joint="curve")
            for tip in (4, 8, 12, 16, 20):
                index = offset + tip
                if confidence[frame_index, index] >= 0.08:
                    x, y = point(index)
                    radius_tip = 4 * scale
                    draw.ellipse((x - radius_tip, y - radius_tip, x + radius_tip, y + radius_tip), fill="#fff4e6")

        draw.text(
            (24 * scale, 18 * scale),
            "LOCAL RESEARCH PREVIEW",
            fill="#70e3d4",
            font=title_font,
        )
        display_label = label
        if draw.textlength(display_label, font=label_font) > (size - 48) * scale:
            while draw.textlength(f"{display_label}...", font=label_font) > (
                size - 48
            ) * scale:
                display_label = display_label[:-1]
            display_label = f"{display_label.rstrip()}..."
        draw.text(
            (24 * scale, 48 * scale), display_label, fill="#ffffff", font=label_font
        )
        draw.text(
            (24 * scale, (size - 38) * scale),
            f"Frame {frame_index + 1}/64 | not expert-reviewed",
            fill="#9fb2b8",
            font=label_font,
        )
        frames.append(image.resize((size, size), Image.Resampling.LANCZOS))

    path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=120, loop=0, optimize=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=Path("data/private/isign-research/text-to-pose"))
    parser.add_argument("--text", required=True)
    parser.add_argument("--output", type=Path, default=Path("data/private/isign-research/text-to-pose/preview-motion.json"))
    parser.add_argument("--gif", type=Path, default=Path("data/private/isign-research/text-to-pose/preview.gif"))
    parser.add_argument("--accept-research-only-terms", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.accept_research_only_terms:
        raise SystemExit("Pass --accept-research-only-terms after accepting the gated iSign terms.")
    for path, label in (
        (args.model_dir, "Model directory"),
        (args.output, "Preview motion output"),
        (args.gif, "Preview GIF output"),
    ):
        require_private_path(path, label)
    if not tokenize(args.text):
        raise PreviewError("Preview text must contain at least one word token.")
    import onnxruntime as ort

    metadata = json.loads((args.model_dir / "metadata.json").read_text(encoding="utf-8"))
    if metadata.get("qualityGate", {}).get("runtimeEligible") is not False:
        raise PreviewError("Research preview requires an explicitly runtime-blocked artifact.")
    onnx_path = args.model_dir / metadata["model"]["onnxFile"]
    if sha256_file(onnx_path) != metadata["model"]["onnxSha256"]:
        raise PreviewError("ONNX bytes do not match the trained metadata.")
    vocabulary_path = args.model_dir / metadata["model"]["vocabularyFile"]
    normalization_path = args.model_dir / metadata["model"]["normalizationFile"]
    if sha256_file(vocabulary_path) != metadata["model"]["vocabularySha256"]:
        raise PreviewError("Vocabulary bytes do not match the trained metadata.")
    if sha256_file(normalization_path) != metadata["model"]["normalizationSha256"]:
        raise PreviewError("Normalization bytes do not match the trained metadata.")
    vocabulary = json.loads(vocabulary_path.read_text(encoding="utf-8"))
    with np.load(normalization_path) as normalization:
        mean = np.asarray(normalization["mean"], dtype=np.float32)
        scale = np.asarray(normalization["scale"], dtype=np.float32)
        confidence = np.asarray(normalization["confidence"], dtype=np.float32)
    token_ids = encode_text(args.text, vocabulary, MAX_TOKENS)[None, :]
    attention = token_ids != 0
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    standardized = session.run(None, {"token_ids": token_ids, "attention_mask": attention})[0][0]
    xy = canonicalize_for_avatar(standardized * scale + mean, confidence)
    energy = motion_energy(xy, confidence)
    if energy < MIN_PREVIEW_MOTION_ENERGY:
        raise PreviewError(
            f"Generated hand motion energy {energy:.6f} is below the dynamic preview threshold."
        )
    payload = {
        "schemaVersion": 1,
        "kind": "private_text_to_pose_research_preview",
        "text": args.text,
        "modelId": metadata["model"]["id"],
        "technicalGatePassed": bool(metadata["qualityGate"]["passed"]),
        "runtimeEligible": False,
        "expertReviewStatus": "pending",
        "motionEnergy": energy,
        "visibilitySource": "training_mean_prior_not_predicted",
        "clip": {
            "id": "isign-text-to-pose-private-preview",
            "label": args.text,
            "fps": round(1000 / 120, 6),
            "frames": frame_payload(xy, confidence),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, separators=(",", ":")) + "\n", encoding="utf-8")
    render_gif(xy, confidence, args.gif, args.text)
    print(json.dumps({"output": str(args.output), "gif": str(args.gif), "motionEnergy": energy, "runtimeEligible": False}, indent=2))


if __name__ == "__main__":
    main()
