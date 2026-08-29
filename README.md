# [EMNLP 2026] SpokenUS

[![arXiv](https://img.shields.io/badge/arXiv-2603.16783-b31b1b.svg?logo=arxiv&logoColor=red)](https://arxiv.org/abs/2603.16783)
[![Project Page](https://img.shields.io/badge/Project-Page-4C8BF5.svg)](https://holi-lab.github.io/SpokenUS/)
[![SpokenTOD](https://img.shields.io/badge/Code-SpokenTOD-181717.svg?logo=github)](https://github.com/holi-lab/SpokenTOD)
[![Dataset](https://img.shields.io/badge/Hugging_Face-SpokenTOD-FFD21E.svg)](https://huggingface.co/datasets/holi-lab/SpokenTOD)

We introduce **SpokenTOD**, a spoken TOD dataset comprising 52,390 dialogues and 1,034 hours of speech, augmented with four spoken user behaviors—cross-turn slots, barge-in, disfluency, and emotional prosody. Explore the augmentation pipeline on [GitHub](https://github.com/holi-lab/SpokenTOD), and download the full dataset from [Hugging Face](https://huggingface.co/datasets/holi-lab/SpokenTOD).

Building on SpokenTOD, we present **SpokenUS**, a spoken user simulator grounded in TOD with a dedicated architecture for barge-in.

## Main Archiecture

SpokenUS takes the user’s goal and profile, the conversational history as interleaved text and speech tokens, and the current assistant speech, provided either as a complete utterance or as a streaming input. SpokenUS operates in three sequential modes: Listening Mode, which monitors incoming assistant speech to determine when to speak; Pre-scripting Mode, which generates a transcript before speech is produced; and Speaking Mode, which synthesizes the transcript into speech.

![SpokenUS Archiecture](docs/static/images/SpokenUS.png)



## Audio Samples

The sample below presents the assistant greeting and turns 0–5 from the SpokenUS dialogue [`spokenwoz_MUL1830`](assets/audio_samples/spokenwoz_MUL1830/), including the speaker role, emotion, behavior, transcript, and audio for each turn.

| Turn | Role | Emotion | Behavior | Transcript | Audio |
|---:|---|---|---|---|---|
| Greeting | **Assistant** | — | — | Hello! How can I assist you today? | <audio controls src="assets/audio_samples/spokenwoz_MUL1830/audios/assistant_greeting.wav"></audio><br>[WAV](assets/audio_samples/spokenwoz_MUL1830/audios/assistant_greeting.wav) |
| 0 | **User** | Neutral | — | Can you help me find a hotel named home from home? | <audio controls src="assets/audio_samples/spokenwoz_MUL1830/audios/turn_0.wav"></audio><br>[WAV](assets/audio_samples/spokenwoz_MUL1830/audios/turn_0.wav) |
| 1 | **Assistant** | — | — | I found a hotel named Home from Home located at seventy-eight to eighty Milton Road in the north area. It has internet, parking, and a moderate price range with double rooms at seventy-five pounds. Would you like the phone number or more details? | <audio controls src="assets/audio_samples/spokenwoz_MUL1830/audios/turn_1.wav"></audio><br>[WAV](assets/audio_samples/spokenwoz_MUL1830/audios/turn_1.wav) |
| 2 | **User** | Neutral | Disfluency: Correction | I want to— let me just... Does that hotel have free internet? | <audio controls src="assets/audio_samples/spokenwoz_MUL1830/audios/turn_2.wav"></audio><br>[WAV](assets/audio_samples/spokenwoz_MUL1830/audios/turn_2.wav) |
| 3 | **Assistant** | — | — | Yes, the Home from Home hotel does have free internet available. | <audio controls src="assets/audio_samples/spokenwoz_MUL1830/audios/turn_3.wav"></audio><br>[WAV](assets/audio_samples/spokenwoz_MUL1830/audios/turn_3.wav) |
| 4 | **User** | Satisfied | — | Okay, that's good. I want to make a booking there for six people, for three nights starting tuesday. | <audio controls src="assets/audio_samples/spokenwoz_MUL1830/audios/turn_4.wav"></audio><br>[WAV](assets/audio_samples/spokenwoz_MUL1830/audios/turn_4.wav) |
| 5 | **Assistant** | — | — | I can help you book six people at Home from Home for three nights starting Tuesday. Would you like to confirm the booking or need any special requests? | <audio controls src="assets/audio_samples/spokenwoz_MUL1830/audios/turn_5.wav"></audio><br>[WAV](assets/audio_samples/spokenwoz_MUL1830/audios/turn_5.wav) |

## Citation

```bibtex
@article{lee2026spokenus,
  title={{SpokenUS}: A Spoken User Simulator for Task-Oriented Dialogue},
  author={Lee, Jonggeun and Pyo, Junseong and Park, Jeongmin and Jo, Yohan},
  journal={arXiv preprint arXiv:2603.16783},
  year={2026},
  doi={10.48550/arXiv.2603.16783},
  url={https://arxiv.org/abs/2603.16783},
  eprint={2603.16783},
  archivePrefix={arXiv},
  primaryClass={cs.CL}
}
```

------

## License

SpokenTOD contains data derived from multiple source datasets, and each source-specific license governs its derived portion. The included `spokenwoz_MUL1830` example is derived from [SpokenWOZ](https://spokenwoz.github.io/) and is therefore distributed under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/). See the [SpokenTOD dataset card](https://huggingface.co/datasets/holi-lab/SpokenTOD) for the complete licensing and attribution requirements.
