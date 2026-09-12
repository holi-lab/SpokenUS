# Speech synthesis

The production path uses a self-hosted Qwen3-TTS server through
`VllmOmniSynthesizer`. The CLI connects augmented dialogue JSONL through English
NeMo normalization to the vLLM-Omni server, writes one WAV per synthesized turn,
and emits a new JSONL manifest.

The server URL defaults to `http://127.0.0.1:8000`. Set
`VLLM_OMNI_BASE_URL` or pass `--base-url` when the server uses another host or
port. This is a local service boundary; the pipeline does not call a hosted TTS
provider.

```bash
uv sync --extra normalization

# Validate JSONL -> NeMo without contacting vLLM-Omni.
uv run --no-sync python src/synthesize.py \
  --input-jsonl datasets/SpokenTOD/sample/train.jsonl \
  --output-jsonl datasets/SpokenTOD/normalized/train.jsonl \
  --normalize-only --limit 1

# Full dialogue synthesis through a running vLLM-Omni server.
uv run --no-sync python src/synthesize.py \
  --input-jsonl datasets/SpokenTOD/sample/train.jsonl \
  --output-jsonl datasets/SpokenTOD/speech/train.jsonl \
  --reference-dir datasets/SpeechAccentArchive/recordings/recordings \
  --base-url http://127.0.0.1:8000 --limit 1

# Equivalent Makefile entry point.
make synthesize VLLM_OMNI_BASE_URL=http://127.0.0.1:8000 \
  SYNTH_ARGS="--limit 1"
```

For each dialogue, the pipeline prepares the selected user and assistant SAA
voices once and reuses them for all turns. `VllmOmniSynthesizer` first attempts
the voice upload/cache endpoints. If those endpoints are unavailable, it falls
back to sending the reference audio and transcript inline to
`/v1/audio/speech`. User emotion categories select a stable per-turn style
keyword; assistant turns use a neutral instruction.

The first NeMo run compiles grammar files into `.cache/nemo`; later runs reuse
the cache. The normalization extra pins NeMo 1.1.0, the version used for the
public release. It uses the same punctuation configuration as the public release.
`goal.normalized` is computed from `goal.text` with the same NeMo normalizer and
punctuation settings used for turns. Existing nonempty `goal.normalized` and
`text_normalized` values are treated as canonical and preserved instead of being
recomputed. Every source field is preserved. Each turn gains `normalized_text`;
synthesized turns also gain `audio_path` and `synthesis` metadata. The augmentation
pipeline appends `<|endoftext|>` to a final user utterance, or adds a standalone
terminal user turn when a dialogue ends with the assistant. Synthesis removes the
marker before TTS; standalone terminal turns remain `control_only` with no audio.
Existing native `audio_path` files are reused.

Output manifests use exclusive creation. On failure, completed dialogues and
generated WAVs remain for diagnosis. Use a new output path when retrying. This
produces turn-level WAVs; mixing barge-in overlap and reproducing the paper's
ASR/human evaluation are separate steps.

Single-utterance smoke test:

```bash
uv run --no-sync python src/synthesize.py \
  --text "Your reservation is confirmed." \
  --emotion neutral \
  --ref-audio datasets/SpeechAccentArchive/recordings/recordings/english105.mp3 \
  --output-path /tmp/qwen3-tts.wav \
  --base-url http://127.0.0.1:8000
```

`local.py` remains as an experimental direct `qwen-tts` adapter. Install its
dependencies with `uv sync --extra local-synthesis`; it is not used by the
dataset CLI.
