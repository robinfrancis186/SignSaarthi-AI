# Data Dictionary

The public contracts live in `packages/shared/src/schemas.ts`.

- `TranscriptSegment`: one raw/cleaned transcript unit.
- `SimplifiedChunk`: short meaning unit prepared for signing.
- `ISLGlossItem`: one glossary-backed or fallback sign plan item.
- `ISLInterpretation`: full interpretation response for one segment.
- `ISLInterpretationResponse`: interpretation plus avatar queue, optional model provenance, and `modelBackedGlossCount`.
- `GlossaryEntry`: controlled ISL term record.
- `ModelMetadata`: local model status, engine type, training dataset, class count, and citation URLs.
- `KeypointLandmark`: one pose/hand/face point from an extracted frame.
- `KeypointSequence`: a labeled sign sample made from normalized keypoint frames.
- `VideoModelArtifact`: exported keypoint model with label map, landmark keys, prototypes, metrics, and metadata.
- `VideoInferenceResponse`: top keypoint predictions plus model provenance and `rawVideoStored: false`.
- `TrainingJob`: local metadata contract for future dataset validation, extraction, training, and evaluation jobs.
- `SignAsset`: manifest contract for future keypoint/video/animation assets.
- `FeedbackReport`: wrong-sign or translation feedback.
