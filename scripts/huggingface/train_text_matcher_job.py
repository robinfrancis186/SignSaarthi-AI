# /// script
# requires-python = ">=3.11"
# dependencies = ["huggingface_hub>=0.34,<2"]
# ///
"""Train and publish the SignSaarthi lexical matcher on Hugging Face Jobs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download, snapshot_download


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def model_card(artifact: dict[str, object], dataset_revision: str) -> str:
    metrics = artifact["metrics"]
    assert isinstance(metrics, dict)
    test = metrics["test"]
    assert isinstance(test, dict)
    binary = test["binary"]
    grouped = test["grouped"]
    assert isinstance(binary, dict) and isinstance(grouped, dict)
    return f"""---
library_name: signsaarthi
license: other
tags:
- indian-sign-language
- accessibility
- text-classification
- lexical-matching
datasets:
- Robin186/SignSaarthi-ISL-Training
metrics:
- precision
- f1
---

# SignSaarthi ISL Caption Typo Matcher

Deterministic, text-only lexical pair classifier used to resolve conservative
single-token caption typos against the SignSaarthi vocabulary.

## Evaluation

- Test precision: {binary['precision']}
- Test recall: {binary['recall']}
- Test F1: {binary['f1']}
- Test grouped top-1: {grouped['top1']}
- Test MRR: {grouped['mrr']}
- Dataset revision: `{dataset_revision}`

The threshold and runner-up margin are selected on validation and evaluated
unchanged on test. The artifact is published only when all encoded acceptance
checks pass.

## Limits

This model preserves source text and is restricted to unmatched Latin-script
single tokens. It is not ISL grammar, sentence translation, sign recognition,
pose generation, or avatar motion synthesis. Exact glossary matching and visible
fingerspelling/caption fallbacks remain authoritative.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-repo", required=True)
    parser.add_argument("--dataset-revision", required=True)
    parser.add_argument("--model-repo", required=True)
    parser.add_argument("--trainer-revision", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN must be supplied as a Hugging Face Job secret.")

    dataset_directory = Path(
        snapshot_download(
            repo_id=args.dataset_repo,
            repo_type="dataset",
            revision=args.dataset_revision,
            token=token,
        )
    )
    trainer_path = Path(
        hf_hub_download(
            repo_id=args.model_repo,
            filename="train_isl_text_matcher.py",
            revision=args.trainer_revision,
            token=token,
        )
    )

    with tempfile.TemporaryDirectory(prefix="signsaarthi-matcher-job-") as directory:
        output_directory = Path(directory)
        artifact_path = output_directory / "isl-text-matcher.json"
        subprocess.run(
            [
                sys.executable,
                str(trainer_path),
                "--input-dir",
                str(dataset_directory),
                "--output",
                str(artifact_path),
                "--verify-determinism",
            ],
            check=True,
        )
        artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
        acceptance = artifact.get("metrics", {}).get("acceptance", {})
        if artifact.get("metadata", {}).get("status") != "ready" or not acceptance.get(
            "ready"
        ):
            raise RuntimeError("The matcher did not pass its encoded readiness gates.")

        report = {
            "schemaVersion": 1,
            "dataset": {
                "repoId": args.dataset_repo,
                "revision": args.dataset_revision,
            },
            "trainer": {
                "repoId": args.model_repo,
                "revision": args.trainer_revision,
                "sha256": sha256_file(trainer_path),
            },
            "artifact": {
                "path": "isl-text-matcher.json",
                "sha256": sha256_file(artifact_path),
                "byteCount": artifact_path.stat().st_size,
            },
            "metrics": artifact["metrics"],
            "constraints": artifact["constraints"],
            "status": "passed",
        }
        report_path = output_directory / "training-report.json"
        report_path.write_text(
            json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        card_path = output_directory / "README.md"
        card_path.write_text(
            model_card(artifact, args.dataset_revision), encoding="utf-8"
        )

        commit = HfApi(token=token).create_commit(
            repo_id=args.model_repo,
            repo_type="model",
            commit_message="Train validated SignSaarthi lexical matcher",
            operations=[
                CommitOperationAdd(
                    path_in_repo="isl-text-matcher.json",
                    path_or_fileobj=artifact_path,
                ),
                CommitOperationAdd(
                    path_in_repo="training-report.json",
                    path_or_fileobj=report_path,
                ),
                CommitOperationAdd(path_in_repo="README.md", path_or_fileobj=card_path),
            ],
        )

    print(
        json.dumps(
            {
                "status": "passed",
                "modelCommit": commit.oid,
                "modelUrl": str(commit.repo_url),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
