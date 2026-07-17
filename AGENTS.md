# SignSaarthi Engineering Guidance

Use the installed `@ponytail` skill in full mode for coding work: understand the
real flow first, reuse existing code and native platform features, avoid new
dependencies, and make the smallest correct change with a focused check.

When `graphify-out/graph.json` exists, begin architecture and dependency
questions with `graphify query`, `graphify path`, or `graphify explain`. Rebuild
the local code-only graph with `pnpm graph:build`, or refresh it after source
changes with `pnpm graph:update`.

Keep these product contracts intact:

- Preserve every source word in order. Use visible fingerspelling or fallback
  text when a reviewed motion is unavailable; never silently drop a word.
- Do not label text-only vocabulary or unreviewed motion as a verified ISL sign.
- Do not store or publish raw audio, raw video, signer media, private training
  artifacts, credentials, or gated dataset contents.
- When YouTube captions are unavailable, show the explicit fallback state and
  do not present demo text as live captions.
- Preserve accessibility, trust-boundary validation, release integrity, and
  honest AI-assisted interpreter warnings.
- Run `pnpm verify` for release-impacting changes and `pnpm smoke:youtube-live`
  for caption, queue, overlay, or avatar playback changes.
