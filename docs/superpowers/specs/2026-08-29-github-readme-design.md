# SpokenUS GitHub README Design

## Goal

Create a public-facing, English README at the repository root that explains SpokenUS and SpokenTOD, makes the main research contributions easy to scan, and lets visitors listen to representative audio from `spokenwoz_MUL0608`.

## Deliverables

- Add a root-level `README.md`.
- Copy the README figures selected from `docs/static/images/` to `assets/figures/`.
- Copy the complete `spokenwoz_MUL0608` sample directory to `assets/audio_samples/spokenwoz_MUL0608/`.
- Preserve the sample's `data.json`, `synthesis_manifest.json`, and all WAV files so readers can inspect the complete dialogue beyond the clips highlighted in the README.
- Add a corresponding-author marker to Yohan Jo in `docs/index.html` and extend the author-note legend accordingly.
- Preserve all other existing uncommitted changes in `docs/index.html`.

## README Structure

The README will mirror the supplied SimuHome README's academic-project format while using `docs/index.html` as the authoritative source for SpokenUS wording, links, figures, metrics, author list, and citation:

1. A centered lead figure, followed by the venue-prefixed paper title.
2. A compact row of badges for available project resources such as arXiv, the dataset, and the project page. No Python/runtime badge will be added because this repository does not currently expose an installable implementation.
3. A bold project-name lead paragraph and a centered overview/pipeline figure.
4. Horizontal separators matching the SimuHome README's section rhythm.
5. `Spoken User Behaviors`, explaining cross-turn slots, barge-in, disfluency, and emotional prosody.
6. `SpokenTOD`, with its construction figure and dataset statistics.
7. `SpokenUS`, with its model architecture figure and three operating modes.
8. `Audio Samples`, with selected clips from `spokenwoz_MUL0608` and a link to the full dialogue directory.
9. `Results`, summarizing the quantitative claims shown in the project page.
10. `Citation`, using the BibTeX entry from `docs/index.html`.
11. `License`, following SpokenTOD's source-specific data licensing policy. It will state that source-specific licenses control derived portions and that the included `spokenwoz_MUL0608` sample is derived from SpokenWOZ and therefore governed by CC BY-NC 4.0. It will link to the full SpokenTOD dataset licensing information rather than claiming a single repository-wide software license.

The relevant existing figures will be copied from `docs/static/images/` to `assets/figures/` so the root README follows the same asset organization as the supplied SimuHome example. The likely lead figure is `preview.png`, followed by the focused behavior, SpokenTOD, and SpokenUS diagrams in their respective sections.

## Audio Sample Presentation

The README will highlight a small set of representative turns rather than reproduce all 54 audio files inline. The table will cover:

- Disfluency/restart: user turn 10.
- Emotional prosody: a clearly labeled emotional user turn, such as turn 12 or turn 46.
- Barge-in: the assistant/user pair around turns 13–14 or another explicitly tagged interruption.
- Cross-turn slot disclosure: a short sequence showing information spread across adjacent user turns.

Each entry will include the behavior, transcript/context, an HTML audio control, and a normal relative WAV link as a fallback. The fallback keeps samples accessible if a GitHub Markdown renderer suppresses embedded audio controls. A link to the complete copied dialogue directory will appear below the table.

## Data Handling

The user requested that the complete source directory be copied. The source-facing symlink at `../human_eval_app_final/SpokenTOD` is broken in the current environment, so the copy will use its verified backing directory:

`../human_eval_app_spokenwoz/W3-o3/spokenwoz_MUL0608`

File counts, sizes, WAV readability, and JSON validity will be checked after copying.

## Verification

- Confirm the README's relative image, audio, and project links resolve to tracked files or intended external pages.
- Confirm every README audio reference exists.
- Compare source and destination sample file lists and checksums.
- Validate both copied JSON files with `jq`.
- Confirm Yohan Jo alone receives the corresponding-author marker and that the matching legend is present in `docs/index.html`.
- Inspect `git diff` to ensure only the new README, copied figure/audio-sample assets, approved design/plan documents, and the requested corresponding-author addition are introduced; preserve the user's other `docs/index.html` edits.
