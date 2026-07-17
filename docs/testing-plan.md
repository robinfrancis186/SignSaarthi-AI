# Testing Plan

- Unit tests cover shared schemas, text cleanup, chunking, glossary lookup, lexicon training, keypoint model training, keypoint inference, confidence, fallback behavior, and storage helpers.
- API tests use Fastify injection for interpretation, model registry/status, keypoint inference, malformed keypoint payloads, model-backed glossary use, rule fallback, glossary search, feedback, and session lifecycle.
- UI tests cover inactive/active states, API-owned avatar queue dispatch, model provenance display, output-mode gating, signing-plan fallback badges, low-confidence fallback, settings persistence, overlay settings, accessible feedback modal behavior, and feedback success.
- Manual QA loads the unpacked extension on YouTube, verifies the side panel and overlay, tests no-caption fallback, checks output modes, submits feedback, and checks delete-local-data behavior.

Recommended verify loop after every meaningful change:

```bash
pnpm model:train
pnpm model:video:train
pnpm test
pnpm typecheck
pnpm lint
pnpm build
```
