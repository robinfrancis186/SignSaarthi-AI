# Privacy

- Interpretation starts only after the user clicks Start Interpretation.
- The MVP does not capture live tab audio.
- Raw audio is never stored.
- Raw video is never captured or stored by the extension.
- Optional gated iSign research poses are fetched and stored only by an explicit local developer command under the ignored `data/private` directory; they are never collected from the user's current video.
- After Start Interpretation, the extension reads only captions already visible on the active YouTube page and deduplicates unchanged text.
- Settings and history controls are local-first.
- Session and interpretation records live only in the local API process memory.
- Feedback never includes raw audio or video. It includes transcript/gloss context only after the user submits a report and should avoid personal data.
- API keys are not bundled into the extension.
- The local API accepts browser access only from Chrome extensions and localhost development previews.
- Release packaging and Hugging Face publication exclude the complete `data/private` tree.

SignSaarthi AI is an AI-assisted accessibility tool. It may make mistakes. For legal, medical, emergency, or high-stakes situations, use a qualified human interpreter.
