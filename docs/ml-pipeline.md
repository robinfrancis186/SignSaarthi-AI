# SignSaarthi ISL Data And Motion Pipeline

## Release Contract

| Component             | Current state                                                                                                     |
| --------------------- | ----------------------------------------------------------------------------------------------------------------- |
| Lexicon               | 12,434 deterministic entries from the ISLRTC official index plus INCLUDE metadata                                 |
| INCLUDE avatar motion | 262 source-consistent isolated-sign clips with 32 landmark frames; one additional vocabulary label is quarantined |
| Alphabet fallback     | 26 official-source ISLRTC A-Z clips used to build ordered dynamic fingerspelling                                  |
| Private iSign motion  | Optional local-only gated word-motion library; excluded from packages and Hub publication                          |
| Caption planner       | Private local boundary model trained from 119,960 leakage-isolated iSign captions; never treated as ISL grammar   |
| Text-to-pose research | Private bounded iSign sentence corpus and evaluated ONNX research artifact; blocked from release pending expert review |
| Caption typo matcher  | Qualified deterministic pair classifier; 0.985411 test precision and 0.993092 grouped top-1                       |
| Sign recognition      | Not deployed; the API reports recognition unavailable                                                             |

The lexicon and motion catalog support deterministic caption-to-avatar output. They are not a sign-recognition model and do not establish translation accuracy or expert certification.

## Inputs And Provenance

- ISLRTC official index: https://divyangjan.depwd.gov.in/islrtc/
- INCLUDE metadata: `data/isl/include-metadata/train.jsonl`, `val.jsonl`, and `test.jsonl`.
- INCLUDE source record: `data/isl/include-zenodo-record.json`, pinned to Zenodo record `4010759` (CC BY 4.0): https://zenodo.org/records/4010759
- Runtime target catalog: `data/models/isl-motion-catalog.json`.
- Private training bundle: `output/huggingface/Robin186-SignSaarthi-ISL-Training`.
- Runtime matcher: `data/models/isl-text-matcher.json`.

The INCLUDE source archives total approximately 56 GB. The release extractor does not download or retain complete archives.

## Extract The Motion Library

```bash
python3.11 -m venv .venv-mediapipe
.venv-mediapipe/bin/python -m pip install -r scripts/requirements-mediapipe.txt
pnpm dataset:include:all-motions
```

The extractor:

1. Reads the local split metadata and pinned Zenodo record JSON.
2. Indexes relevant remote ZIPs with HTTP range requests.
3. Transfers one selected archive member into a temporary directory.
4. Extracts landmarks for pose, both hands, and a compact face subset with MediaPipe Holistic, validates motion, and resamples exactly 32 frames.
5. Deletes the temporary source video, writes the runtime library/catalog, and updates `data/isl/include-motion-extraction-v2.json`.

Run the same command again after interruption. Completed clips are loaded from the output library and skipped, so extraction resumes without replaying accepted work. Success requires 262 unique source-consistent playable clips, 32 frames per clip, and `expertReviewed: false` for every clip. The 263rd train label, `Second (Number)`, has no source path whose directory matches its label and is therefore retained as vocabulary but quarantined from playback. Release tests require the catalog to equal the train-label set minus that quarantine and independently verify every selected source directory against its label.

The published INCLUDE split names are provenance fields, not a leakage-safe benchmark: exact `video_path` overlap is 120 between train/validation, 277 between train/test, and 27 between validation/test. Any evaluation split must be rebuilt by unique media identity.

## Private iSign Research Lane

The optional iSign workflow requires prior acceptance of the upstream gated terms and local Hugging Face authentication. `scripts/isign_remote_archive.py` pins Hub revision `e4ee6c5f0d9dfcbc74205e3f1388ce94da26c298`, presents the four upstream pose parts as one seekable ZIP, verifies member CRCs, and uses bounded HTTP range requests to fetch selected `.pose` members. It does not reconstruct or retain the full approximately 170 GB pose archive.

```bash
pnpm dataset:isign:plan
pnpm dataset:isign:word-motions
pnpm dataset:isign:sentences
pnpm dataset:isign:validate
pnpm model:isign:train:private
pnpm model:isign:text-to-pose:private
pnpm model:isign:text-to-pose:preview
```

The text-only caption planner uses all 119,960 retained unique captions after video-group splitting and decontamination. Its two deterministic builds are byte-identical; held-out test lexical-token coverage is 0.887212 and exported boundary-feature coverage is 0.331775. The local API uses it only to choose boundaries in long caption chunks. The engine independently verifies that planner output reconstructs the exact cleaned caption and falls back to deterministic chunking if any word is changed, reordered, or removed.

The deterministic pose plan selects at least 10,000 captions with 5-15 words, prioritizes vocabulary balance, and assigns train/validation/test by video-group hash to prevent signer-video leakage. The validated corpus has 10,005 dynamic clips: 8,003 train, 1,002 validation, and 1,000 test, covering 3,274 video groups with zero split overlap. Its accepted compressed source members total 17,871,511,846 bytes; no archive part is stored locally. The compact corpus stores 64 frames with 93 points: body pose, both hands, and a small face subset.

`pnpm dataset:isign:validate` binds every database row to the exact planned text, video hash, split, and word count; verifies the immutable revision, ZIP-directory fingerprint, plan hash, WAL checkpoint, every tensor, actual temporal hand movement, accepted-member byte cap, private word-motion source label, and final database hash. The word library covers all 585 annotated labels; one same-source upstream alias is recorded explicitly. The trainer requires that exact report, builds vocabulary from train only, uses confidence-weighted position, velocity, acceleration, hand-speed, and motion-energy objectives, exports ONNX, hashes its vocabulary and normalization companions, and validates ONNX Runtime parity including an all-padding safety input.

The final model retains 95.31% validation and 94.78% held-out hand-motion energy; every held-out output is non-static. Validation normalized MPJPE is 0.631462 and held-out normalized MPJPE is 0.630077. DTW is evaluated on a declared deterministic sample of 128 sequences. The dynamic geometry preview gate passes, while `runtimeEligible` remains `false`: these geometric metrics do not establish linguistic accuracy, non-manual grammar, regional correctness, or interpreter equivalence.

Private artifacts live under `data/private/isign-research`, which is ignored by source control, package assembly, Graphify, and Hugging Face publication. Technical metric gates never make the model runtime-eligible; sentence-level Deaf/ISL expert evaluation remains mandatory.

## Build The Unified Vocabulary

```bash
pnpm dataset:unified:build
pnpm dataset:unified:validate
pnpm model:train
```

The builder snapshots the ISLRTC A-Z index, merges admissible INCLUDE motion metadata, rejects iSign provenance, and writes `output/huggingface/Robin186-ISL`. The public bundle contains metadata and vocabulary only; it excludes raw media and derived landmark frame arrays. `pnpm model:train` compiles the validated bundle into `data/models/isl-lexicon-model.json`.

## Build And Train The Caption Matcher

```bash
pnpm dataset:training:build
pnpm dataset:training:validate
pnpm model:text-matcher:train
```

The private bundle keeps all 12,434 canonical records, all 5,250 INCLUDE
metadata rows, all 263 vocabulary labels, 262 compact playable motion records,
and the explicit one-label quarantine. It creates 99,348 candidate rows in
24,837 globally unique normalized query groups. Stable canonical-ID hashes
produce independent train, validation, and test partitions; every group has one
positive and deterministic hard negatives.

The matcher is a text-only typo resolver for unmatched Latin-script single
tokens. It preserves the original caption token and requires both a validation-
selected probability threshold and winner margin. Held-out test precision is
0.985411, F1 is 0.943892, grouped top-1 is 0.993092, and MRR is 0.996546. It is
not ISL grammar, sentence translation, recognition, pose generation, or motion
synthesis.

## Verify And Build

```bash
pnpm verify
pnpm build
pnpm --filter @signsaarthi/api start
```

`pnpm verify` runs type checks, lint, TypeScript and Python tests, production builds, unified/training-dataset validation, extension smoke coverage, and explicit 262-playable-plus-one-quarantine assertions. Load the built extension from `apps/extension/dist` in Chrome.

## Recognition API Contract

There is currently no deployed sign-recognition model or ONNX artifact. The release must not substitute fixtures or motion clips for recognition:

- `GET /api/keypoint-model/status` returns `status: "unavailable"`, `model: null`, `metrics: null`, and `recognitionModelReady: false`.
- `POST /api/keypoints/infer` returns HTTP `503` with `error: "recognition_model_unavailable"`, no predictions, and `fallback: "caption"`.

Recognition training utilities are research scaffolding only. A future model requires a separately reviewed dataset, held-out evaluation, a persisted artifact, and an explicit release decision before these API responses may change.

## Output And Review Rules

- Every source word remains represented in its original order.
- Unsupported Latin-script words use official ISLRTC A-Z dynamic fingerspelling.
- Unsupported scripts remain caption/text actions; they are never dropped.
- INCLUDE motion is automated dataset-to-avatar transfer, not expert-certified signing.
- Local iSign motion and text-to-pose output are gated noncommercial research artifacts, not expert-certified signing.
- ISLRTC letter signs use an official source, but MediaPipe extraction, segmentation, interpolation, and avatar rendering are automated and not expert-certified.
- SignSaarthi is an assistive companion, never a certified interpreter replacement.
