# SpokenTOD

<p>
  <a href="https://huggingface.co/datasets/holi-lab/SpokenTOD">
    <img src="https://img.shields.io/badge/huggingface-%23FFD21E.svg?style=for-the-badge&logo=huggingface&logoColor=white" />
  </a>
  <a href="https://arxiv.org/abs/2603.16783">
    <img src="https://img.shields.io/badge/arXiv-2603.16783-b31b1b.svg?style=for-the-badge" />
  </a>
</p>

SpokenTOD is a large-scale spoken task-oriented dialogue (TOD) dataset built by
augmenting existing TOD benchmarks with realistic spoken behaviors and emotion
labels, then optionally synthesizing speech. This repository implements the
SpokenTOD construction pipeline described in our paper. We use SpokenTOD to train [SpokenUS: A Spoken User Simulator for Task-Oriented Dialogue](https://arxiv.org/abs/2603.16783v1).

## We provide

- **Source datasets**: SpokenWOZ, EmoWOZ, SGD, TaskMaster-2, ABCD
- **Augmentations**
  - Cross-turn slots (multi-turn slot construction and self-correction)
  - Barge-in (error recovery, clarification, efficiency)
  - Disfluency (FP, DM, EDIT, REP, RST, COR)
- **Emotion annotation** using the EmoWOZ label set
- **Speaker demographic sampling** using Speech Accent Archive (SAA)
- **Optional speech synthesis** via Qwen3-TTS conditioned on emotion

## Repository layout

- `src/augmentation/` - text augmentation pipeline
- `src/augment.py` - CLI for text augmentation
- `src/synthesize.py` - CLI for single-utterance speech synthesis (Qwen3-TTS)
- `scripts/download_dataset.sh` - dataset downloader

## Setup

```bash
# Install uv once if needed
# https://docs.astral.sh/uv/getting-started/installation/

# Create/update the project virtualenv (.venv) and install the package
uv sync

# Optional NeMo normalization
uv sync --extra normalization

# Optional direct qwen-tts adapter
uv sync --extra local-synthesis

# Activate the environment only if you want an interactive shell inside it
source .venv/bin/activate
```

To see available Makefile targets:

```bash
make help
```

For speech synthesis, run a self-hosted vLLM-Omni server serving Qwen3-TTS and
pass its URL to the dataset synthesis command. The client does not use a hosted
TTS API.

For more detail on speech synthesis, please see [src/synthesis/README.md](src/synthesis/README.md).

## Download datasets

Use the Makefile target (recommended):

```bash
make download DATASETS=emowoz,sgd,abcd,tm2,spokenwoz,saa
```

Direct script usage (same logic):

```bash
bash scripts/download_dataset.sh emowoz,sgd,abcd,tm2,spokenwoz,saa
```

Notes:
- The script will fail if a target directory exists unless `FORCE=1`.
- Datasets are large; plan for storage and download time.

## Text augmentation

```bash
# Full run (Makefile defaults)
make augment

# Quick smoke test
make augment-sample
```

Common overrides:

```bash
make augment DATASETS=emowoz,sgd,abcd SPLITS=train,valid CHUNK_SIZE=100
make augment-sample SAMPLE_SIZE=5 AUGMENT_MODEL=gpt-4.1-mini
make augment-full   # background run with nohup
```

### Model configuration

The pipeline uses LLM calls for emotion tagging, barge-in generation, and some
LLM-based disfluency types through LiteLLM.

- `AUGMENT_MODEL` defaults to `openrouter/qwen/qwen3-32b` and is passed to
  LiteLLM exactly as provided. The project does
  not rewrite, normalize, or alias model names.
- Configure provider credentials with the environment variables LiteLLM expects
  for that provider. The default OpenRouter model uses `OPENROUTER_API_KEY`.
- For OpenAI-compatible local endpoints, set `LITELLM_API_BASE` to the base URL
  to use for augmentation requests.

## Output format

Text augmentation writes JSONL files to `datasets/SpokenTOD/`:

- `train.jsonl`, `valid.jsonl`, `test.jsonl`
- `train_demographics.json`, `valid_demographics.json`, `test_demographics.json`
- `overall_demographics.json`

Each dialogue record contains:

```json
{
  "dialogue_id": "...",
  "source": "emowoz|sgd|abcd|tm2|spokenwoz",
  "goal": {"text": "...", "structured": {...}},
  "turns": [
    {
      "role": "user|assistant",
      "text": "...",
      "slots": [{"slot": "...", "value": "...", "start": 0, "end": 4}],
      "tagged": "...",          // disfluency tags (optional)
      "emotion": {"label": 0, "name": "neutral"},
      "disfluency": [...],       // structured disfluency annotations
      "segment": {...},           // cross-turn segment info
      "state": {...},
      "bargein": {"type": "...", "subtype": "..."},
      "audio_path": "..."        // for datasets with native audio
    }
  ],
  "speaker": {...},
  "assistant_speaker": {...},
  "metadata": {...}
}
```

## Speech synthesis (Qwen3-TTS)

```bash
uv sync --extra normalization
```

```bash
uv run --no-sync python src/synthesize.py \
  --text "Hello, this is a Qwen3-TTS smoke test." \
  --emotion calm \
  --ref-audio datasets/SpeechAccentArchive/recordings/recordings/english105.mp3 \
  --output-path outputs/qwen_smoke_test.wav
```

If you modify dependencies, refresh the lockfile with:

```bash
uv lock
```

For augmented dialogue JSONL → NeMo → turn WAVs and an output manifest:

```bash
make synthesize SYNTH_INPUT=datasets/SpokenTOD/sample/train.jsonl \
  SYNTH_OUTPUT=datasets/SpokenTOD/speech/train.jsonl \
  VLLM_OMNI_BASE_URL=http://127.0.0.1:8000 SYNTH_ARGS="--limit 1"
```

Use a new output path on each run. See [synthesis usage](src/synthesis/README.md)
for normalization-only execution, output fields and native-audio handling.

## Behavior taxonomy (summary)

- **Emotion**: neutral, fearful, dissatisfied, apologetic, abusive, excited, satisfied
- **Cross-turn slots**: phone numbers, emails, reservation IDs, etc.
- **Barge-in**: error recovery, clarification, efficiency
- **Disfluency tags**:
  - FP (filled pause), DM (discourse marker), EDIT (editing term)
  - REP (repetition), RST (restart), COR (correction)

## License

Code: MIT License.

## Dataset License

SpokenTOD contains data derived from multiple source datasets. Each source
dataset remains subject to the license below; the source-specific terms control
the portions derived from that dataset. Please retain the required attribution
and cite the corresponding source publications when using the data.

### SpokenWOZ

[SpokenWOZ](https://spokenwoz.github.io/) is distributed under the
[Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)](https://creativecommons.org/licenses/by-nc/4.0/)
license. Use of SpokenTOD records derived from SpokenWOZ is limited to
non-commercial use and requires attribution to the SpokenWOZ authors.

### EmoWOZ

[EmoWOZ](https://zenodo.org/records/14810836) is released under the
[Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)](https://creativecommons.org/licenses/by-nc/4.0/)
license. Use of SpokenTOD records derived from EmoWOZ is limited to
non-commercial use and requires attribution to the EmoWOZ authors.

### Schema-Guided Dialogue (SGD)

The [Schema-Guided Dialogue dataset](https://github.com/google-research-datasets/dstc8-schema-guided-dialogue)
is released under the
[Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0)](https://creativecommons.org/licenses/by-sa/4.0/)
license. Derived portions must preserve attribution and comply with the
ShareAlike requirement.

### ABCD

The [Action-Based Conversations Dataset (ABCD)](https://github.com/asappresearch/abcd)
is released under the [MIT License](https://github.com/asappresearch/abcd/blob/master/LICENSE).
Please retain the required copyright and license notices for derived portions.

### Taskmaster-2

[Taskmaster-2](https://github.com/google-research-datasets/Taskmaster/tree/master/TM-2-2020)
is made available by Google LLC under the
[Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/)
license. Derived portions require attribution to the Taskmaster-2 authors.

### Speech Accent Archive

Reference audio from the [Speech Accent Archive](https://accent.gmu.edu/)
(Steven H. Weinberger and Matthew C. Kelley, George Mason University) was used
solely as reference audio during speech synthesis. The original Speech Accent
Archive recordings, transcripts, and associated metadata are **not** included
in this release. Copyright © The Speech Accent Archive; use is subject to the
Archive's [official terms and license information](https://accent.gmu.edu/about/).

Please cite the Speech Accent Archive as recommended by its
[citation guidance](https://accent.gmu.edu/howto/), in addition to citing the
SpokenTOD project and any relevant original source datasets.

## Citation

If you use this pipeline or dataset, please cite our paper and the original source datasets (SpokenWOZ, EmoWOZ, SGD, TaskMaster-2, ABCD, SpeechAccentArchive) and Qwen3-TTS.

```
@misc{lee2026spokenusspokenusersimulator,
      title={SpokenUS: A Spoken User Simulator for Task-Oriented Dialogue}, 
      author={Jonggeun Lee and Junseong Pyo and Jeongmin Park and Yohan Jo},
      year={2026},
      eprint={2603.16783},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2603.16783}, 
}
```
