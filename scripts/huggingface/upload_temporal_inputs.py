#!/usr/bin/env python3
"""Prepare or upload the private Hugging Face temporal-training input repository."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi


DEFAULT_REPO_ID = "Robin186/signsaarthi-include-keypoints-101"
INPUTS = (
    (Path("data/isl/keypoints.include-combined.json"), "data/keypoints.include-combined.json"),
    (Path("scripts/train_temporal_transformer.py"), "training/train_temporal_transformer.py"),
    (Path("scripts/evaluate_temporal_transformer.py"), "training/evaluate_temporal_transformer.py"),
    (Path("scripts/requirements-temporal.txt"), "training/requirements-temporal.txt"),
)


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_manifest(repo_id: str) -> dict[str, Any]:
    files = []
    for local_path, remote_path in INPUTS:
        if not local_path.exists():
            raise FileNotFoundError(f"Required Hugging Face input is missing: {local_path}")
        files.append(
            {
                "localPath": str(local_path),
                "hubPath": remote_path,
                "bytes": local_path.stat().st_size,
                "sha256": sha256_path(local_path),
            }
        )
    return {
        "schemaVersion": 1,
        "repoId": repo_id,
        "visibility": "private",
        "dataset": "AI4Bharat INCLUDE derived MediaPipe keypoints",
        "license": "CC-BY-4.0 source dataset terms apply",
        "rawVideoStored": False,
        "rawAudioStored": False,
        "containsDatasetDerivedPoseData": True,
        "files": files,
    }


def dataset_card(manifest: dict[str, Any]) -> bytes:
    record = next(item for item in manifest["files"] if item["hubPath"].startswith("data/"))
    return (
        "---\n"
        "license: cc-by-4.0\n"
        "pretty_name: SignSaarthi INCLUDE keypoints 101\n"
        "---\n\n"
        "# SignSaarthi INCLUDE keypoints 101\n\n"
        "Private training input for the SignSaarthi isolated-sign temporal classifier. "
        "It contains 1,649 MediaPipe-derived INCLUDE keypoint sequences across 101 labels.\n\n"
        "No raw video or audio is included. The keypoints are still dataset-derived pose data "
        "from real people and must remain private unless a separate privacy and license review "
        "approves wider distribution. Source: https://github.com/AI4Bharat/INCLUDE.\n\n"
        f"Combined corpus SHA256: `{record['sha256']}`.\n"
    ).encode("utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--confirm-private-upload",
        action="store_true",
        help="Required for an upload because the corpus contains dataset-derived pose data.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = build_manifest(args.repo_id)
    if args.dry_run:
        print(json.dumps(manifest, indent=2))
        return
    if not args.confirm_private_upload:
        raise RuntimeError("Refusing upload without --confirm-private-upload.")
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN is required and must be supplied through a secure secret.")

    api = HfApi(token=token)
    api.create_repo(repo_id=args.repo_id, repo_type="dataset", private=True, exist_ok=True)
    for local_path, remote_path in INPUTS:
        api.upload_file(
            path_or_fileobj=local_path,
            path_in_repo=remote_path,
            repo_id=args.repo_id,
            repo_type="dataset",
        )
    api.upload_file(
        path_or_fileobj=dataset_card(manifest),
        path_in_repo="README.md",
        repo_id=args.repo_id,
        repo_type="dataset",
    )
    api.upload_file(
        path_or_fileobj=(json.dumps(manifest, indent=2) + "\n").encode("utf-8"),
        path_in_repo="training-input-manifest.json",
        repo_id=args.repo_id,
        repo_type="dataset",
    )
    print(json.dumps({"status": "uploaded", "repoId": args.repo_id, "private": True}, indent=2))


if __name__ == "__main__":
    main()
