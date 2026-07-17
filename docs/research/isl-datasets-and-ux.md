# ISL Dataset And UX Decisions

## Dataset Decision

Primary sources:

- ISLRTC official dictionary index: https://divyangjan.depwd.gov.in/islrtc/
- AI4Bharat INCLUDE repository: https://github.com/AI4Bharat/INCLUDE
- INCLUDE metadata viewer: https://huggingface.co/datasets/ai4bharat/INCLUDE
- INCLUDE media record `4010759`: https://zenodo.org/records/4010759

The deterministic release lexicon has 12,434 entries. ISLRTC is the primary vocabulary authority; INCLUDE contributes 263 isolated-sign vocabulary labels and 262 source-consistent motions. `Second (Number)` remains vocabulary-only because every upstream path conflicts with that label. No iSign row, file, alias, weight, pose, or media asset is included in the distributable release or project Hub datasets. A separate ignored local research lane can use authenticated gated iSign poses.

The private lane follows the iSign paper's video-pose-caption task while avoiding the full archive download. A text-only model uses 119,960 retained unique captions for conservative caption-boundary statistics. A separate video-group-isolated corpus contains 10,005 validated dynamic pose-caption clips and stores compact body/hand/face tensors. The local 585-label iSign word library and 262-label INCLUDE library provide 843 unique native-motion labels after overlap. A separate text-to-pose model is evaluated as research and cannot replace the deterministic all-word fallback without Deaf/ISL expert review.

This is not a complete continuous-ISL dataset. The upstream iSign corpus is much larger, only a small portion is human validated, and no finite word list captures productive signing, regional variants, non-manual grammar, or discourse context. Product completeness is therefore defined as preserving and visibly representing every source word through native motion, official fingerspelling, or explicit text fallback, never as claiming complete linguistic coverage.

The local INCLUDE range extractor uses `data/isl/include-metadata` and `data/isl/include-zenodo-record.json`. `pnpm dataset:include:all-motions` fetches only selected ZIP members, temporarily decodes each video, and produces 262 validated 32-frame avatar motions with landmarks for pose, both hands, and a compact face subset without downloading or storing the complete approximately 56 GB archive set. The upstream split metadata contains exact media-path overlap, so it is retained for provenance and not used as a held-out evaluation boundary.

Public metadata/vocabulary bundle: https://huggingface.co/datasets/Robin186/ISL

The bundle excludes raw media, signer media, derived landmark frames, and recognition artifacts.

## Output Decision

- Known runtime motions use real INCLUDE-derived landmarks for pose, both hands, and a compact face subset.
- When locally available, gated iSign-derived word motions may fill otherwise unresolved words and are visibly labeled as automated research motion.
- Unsupported Latin-script words use ordered dynamic A-Z fingerspelling from the official ISLRTC alphabet source.
- Unsupported scripts remain explicit caption/text actions.
- Every source word remains visible and ordered; matching, phrase handling, and fallback selection may not silently remove a word.
- INCLUDE and alphabet avatar transfer are automated and not expert-certified.
- No sign-recognition model is deployed. Recognition endpoints report unavailable instead of returning fixture labels or experimental metrics.

## W3C UX Findings

Sources:

- W3C WAI Sign Languages: https://www.w3.org/WAI/media/av/sign-languages/
- W3C WAI Media Players: https://www.w3.org/WAI/media/av/player/
- W3C Media Accessibility User Requirements: https://www.w3.org/TR/media-accessibility-reqs/

Applied requirements:

- Keep captions and sign available together. Sign language preference varies, and sign languages differ by region.
- Use a movable picture-in-picture/page overlay or side-panel placement so signing can avoid obscuring important video content.
- Provide pause/resume, replay, speed, avatar size, and position controls, and preserve synchronization when speed changes.
- Keep hands, face, controls, captions, and status text large enough to inspect with readable foreground/background contrast.
- Show explicit regional, provenance, review, and fallback status. Distinguish `INCLUDE motion`, `Official ISLRTC fingerspelling`, `Caption fallback`, and `Text fallback`; never present automated motion as expert-reviewed.
- Keep feedback paths for wrong signs, missing signs, and regional variation.

W3C notes that automatic sign avatars are not robust enough to serve as adequate interpretation. SignSaarthi therefore remains an AI-assisted accessibility companion and never a certified interpreter replacement.
