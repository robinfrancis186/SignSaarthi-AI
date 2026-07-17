# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "huggingface-hub>=1.4,<2",
#   "ijson>=3.3,<4",
#   "numpy>=1.26,<3",
#   "onnx>=1.17,<2",
#   "onnxruntime>=1.20,<2",
#   "scikit-learn>=1.5,<2",
#   "torch>=2.4,<3",
#   "transformers>=4.46,<6",
# ]
# ///
"""Hugging Face Job entrypoint for private SignSaarthi temporal training."""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import urllib.request
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download


PRETRAINED_URL = (
    "https://storage.googleapis.com/ai4bharat-public-indic-data/"
    "INCLUDE/pretrained_models/include50/no_cnn/transformer_small.pth"
)


def require_environment(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"{name} is required.")
    return value


def model_card(metadata: dict[str, object]) -> bytes:
    metrics = metadata["metrics"]
    test = metrics["test"]
    policy = metadata["confidencePolicy"]
    return (
        "---\n"
        "library_name: onnxruntime\n"
        "license: cc-by-4.0\n"
        "pipeline_tag: video-classification\n"
        "---\n\n"
        "# SignSaarthi INCLUDE temporal 101\n\n"
        "Private isolated-sign keypoint transformer trained from all locally prepared INCLUDE "
        "sequences. It is not a continuous ISL translator or a certified interpreter.\n\n"
        f"Test top-1: {float(test['accuracy']) * 100:.2f}%. "
        f"Test top-3: {float(test['top3_accuracy']) * 100:.2f}%. "
        f"Frozen threshold: {float(policy['deployedThreshold']):.2f}.\n"
    ).encode("utf-8")


def main() -> None:
    token = require_environment("HF_TOKEN")
    dataset_repo = os.environ.get(
        "HF_DATASET_REPO", "Robin186/signsaarthi-include-keypoints-101"
    )
    model_repo = os.environ.get(
        "HF_MODEL_REPO", "Robin186/signsaarthi-include-temporal-101"
    )
    warmup_epochs = os.environ.get("HF_WARMUP_EPOCHS", "5")
    fine_tune_epochs = os.environ.get("HF_FINE_TUNE_EPOCHS", "30")

    with tempfile.TemporaryDirectory(prefix="signsaarthi-hf-job-") as directory:
        root = Path(directory)
        dataset_path = Path(
            hf_hub_download(
                dataset_repo,
                "data/keypoints.include-combined.json",
                repo_type="dataset",
                token=token,
            )
        )
        trainer_path = Path(
            hf_hub_download(
                dataset_repo,
                "training/train_temporal_transformer.py",
                repo_type="dataset",
                token=token,
            )
        )
        evaluator_path = Path(
            hf_hub_download(
                dataset_repo,
                "training/evaluate_temporal_transformer.py",
                repo_type="dataset",
                token=token,
            )
        )
        checkpoint_path = root / "include50-no-cnn-transformer-small.pth"
        urllib.request.urlretrieve(PRETRAINED_URL, checkpoint_path)
        output_dir = root / "output"
        mirror_dir = root / "mirror"
        subprocess.run(
            [
                "python",
                str(trainer_path),
                "--input",
                str(dataset_path),
                "--labels",
                "all",
                "--pretrained",
                str(checkpoint_path),
                "--output-dir",
                str(output_dir),
                "--mirror-dir",
                str(mirror_dir),
                "--motion-output",
                str(output_dir / "isl-motion-library.json"),
                "--avatar-catalog-output",
                str(output_dir / "motionCatalog.generated.json"),
                "--warmup-epochs",
                warmup_epochs,
                "--epochs",
                fine_tune_epochs,
            ],
            check=True,
        )
        metadata_path = output_dir / "isl-temporal-model.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        model_path = output_dir / str(metadata["modelFile"])
        subprocess.run(
            [
                "python",
                str(evaluator_path),
                "--dataset",
                str(dataset_path),
                "--model",
                str(model_path),
                "--metadata",
                str(metadata_path),
                "--output",
                str(output_dir / "isl-temporal-evaluation.json"),
                "--mirror-output",
                str(mirror_dir / "isl-temporal-evaluation.json"),
            ],
            check=True,
        )
        if metadata["metadata"]["trainingDataset"]["classCount"] != 101:
            raise RuntimeError("Remote training did not export the expected 101 classes.")

        api = HfApi(token=token)
        api.create_repo(repo_id=model_repo, repo_type="model", private=True, exist_ok=True)
        api.upload_folder(folder_path=output_dir, repo_id=model_repo, repo_type="model")
        api.upload_file(
            path_or_fileobj=model_card(metadata),
            path_in_repo="README.md",
            repo_id=model_repo,
            repo_type="model",
        )
        print(
            json.dumps(
                {
                    "status": "trained_and_uploaded",
                    "modelRepo": model_repo,
                    "private": True,
                    "classCount": 101,
                    "modelFile": model_path.name,
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
