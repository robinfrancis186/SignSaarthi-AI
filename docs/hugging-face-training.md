# Hugging Face Dataset And Model Release

## Verified Hub Revisions

The release uses three revision-pinned Hugging Face repositories:

| Artifact                                                                                                 | Visibility | Verified revision                          | Contents                                                                                                                                           |
| -------------------------------------------------------------------------------------------------------- | ---------- | ------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------- |
| [`Robin186/ISL`](https://huggingface.co/datasets/Robin186/ISL)                                           | Public     | `5bb940424f474d7a476fe25159936844465148e2` | 12,434 compact vocabulary/provenance records, 263 INCLUDE labels, 262 playable motion-metadata records, and one quarantine                         |
| [`Robin186/SignSaarthi-ISL-Training`](https://huggingface.co/datasets/Robin186/SignSaarthi-ISL-Training) | Private    | `6de38cd0370e19ae7c2cbbe954f4351e714ce275` | Canonical rows, all 5,250 INCLUDE metadata rows, leakage-safe lexical match pairs, compact motion metadata, source registry, and validation report |
| [`Robin186/SignSaarthi-ISL-Matcher`](https://huggingface.co/Robin186/SignSaarthi-ISL-Matcher)            | Private    | `5816104c1e3c90d7668434ae6f1e8b58cb697c51` | Ready text-matcher artifact, held-out metrics, model card, revision-pinned training report, trainer, and reusable job entrypoint                   |

The public dataset's `data/isl.jsonl` SHA-256 is
`4a1171f65d60bcc4e75002526ecc7c76d6a3d44c00e26ed47b3704ce45dfcf45`.
The model artifact is 6,015 bytes with SHA-256
`44a5ed18df1cf1389704197556500b50b5412f1ff09801bb399874320b8c72da`.
The exact Hub model revision was downloaded after publication and is
byte-identical to `data/models/isl-text-matcher.json`.

## Training Dataset

Build and validate the private training bundle with:

```bash
pnpm dataset:training:build
pnpm dataset:training:validate
```

The deterministic bundle contains:

- 12,434 canonical records and 12,422 unambiguous matcher representatives.
- 24,837 globally unique normalized query groups and 99,348 candidate rows.
- 79,796 train, 9,708 validation, and 9,844 test candidate rows.
- All 5,250 INCLUDE metadata rows and all 263 vocabulary labels.
- 262 compact playable motion records without frame arrays.
- One explicit quarantine for `Second (Number)` because every upstream path
  conflicts with its label.
- Eleven source-policy records covering admitted, reference-only, and excluded
  corpora.

Splits use a stable SHA-256 bucket of the positive canonical record ID. They do
not reuse the upstream INCLUDE split names for evaluation because exact
`video_path` overlap exists between those upstream partitions.

The bundle validator requires a fixed file allowlist, byte-deterministic output,
complete canonical/input coverage, one positive per match group, split
isolation, globally unique normalized queries, source hashes, no AppleDouble
files, no absolute local paths, and a forbidden-fragment scan.

## Matcher Training And Evaluation

Train the runtime artifact with:

```bash
pnpm model:text-matcher:train
```

The deterministic trainer selects its probability threshold and runner-up
margin on validation and evaluates them unchanged on test. Two independent
in-memory builds must be byte-identical. The published revision reports:

| Metric        | Validation |     Test |
| ------------- | ---------: | -------: |
| Precision     |   0.982095 | 0.985411 |
| Recall        |   0.903997 | 0.905729 |
| F1            |   0.941429 | 0.943892 |
| Grouped top-1 |   0.990935 | 0.993092 |
| MRR           |   0.995468 | 0.996546 |

The exact-match baseline test F1 is `0.666847` and grouped top-1 is `0.581877`.
Every encoded readiness gate passed. The matcher is still deliberately narrow:
it considers only unmatched Latin-script single tokens, preserves the original
source token, requires both the learned threshold and winner margin, and never
marks the selected sign as expert reviewed. It is not sentence translation,
ISL grammar, sign recognition, pose generation, or avatar motion synthesis.

The reusable job entrypoint is
`scripts/huggingface/train_text_matcher_job.py`. It downloads the exact dataset
and trainer revisions, trains with determinism verification, refuses to publish
an unqualified artifact, and writes `training-report.json`. The hosted CPU Job
submission was rejected by Hugging Face with `402 Payment Required` because the
account had no prepaid Jobs credit. The same revision-pinned entrypoint was run
locally and published through the Hugging Face Hub API; the release has no
runtime dependency on Jobs.

## Publication Commands

Publish only the validated allowlisted directories:

```bash
HF_HUB_DISABLE_XET=1 hf upload Robin186/ISL \
  output/huggingface/Robin186-ISL . \
  --repo-type dataset \
  --commit-message "Publish validated SignSaarthi ISL dataset"

HF_HUB_DISABLE_XET=1 hf upload Robin186/SignSaarthi-ISL-Training \
  output/huggingface/Robin186-SignSaarthi-ISL-Training . \
  --repo-type dataset \
  --commit-message "Publish validated SignSaarthi training dataset"
```

Do not upload runtime motion frame arrays, raw media, temporary extraction
files, checkpoints, credentials, absolute paths, or private research artifacts.

## Source And Safety Boundary

INCLUDE is used under CC BY 4.0. ISLRTC contributes compact dictionary metadata
and the official alphabet source under its stated use conditions; dictionary
media is not republished in either dataset.

iSign is gated and declared CC BY-NC-SA 4.0 with terms that prohibit sharing or
re-uploading the source dataset. Authenticated local research may range-read a
bounded set of pose members into `data/private/isign-research`, but no iSign
text, identifiers, media, poses, derived private artifacts, or research model
weights enter either project Hub dataset or a packaged release. The local API
may load the private word-motion artifact only when it already exists on the
same machine. Other discovered sources without a compatible, reviewed grant
remain reference-only or excluded in `source_registry.json`.

The local extractor is pinned to iSign Hub revision
`e4ee6c5f0d9dfcbc74205e3f1388ce94da26c298`. The validated private corpus has
10,005 dynamic sentence clips and the word library has 585 labels. These counts
and aggregate geometric metrics may be documented locally, but the underlying
rows, poses, weights, previews, and model companions must not be uploaded to
Hugging Face under the upstream no-redistribution terms.

The 262 INCLUDE clips are automated isolated-sign transfers and have not been
reviewed by an ISL expert. Isolated signs do not establish sentence-level ISL
grammar. SignSaarthi is AI-assisted, may make mistakes, and is not a certified
interpreter replacement.
