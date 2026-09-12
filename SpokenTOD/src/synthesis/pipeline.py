"""Stream augmented dialogue JSONL through normalization and speech synthesis."""

import hashlib
import json
import random
from pathlib import Path

import numpy as np

from synthesis.base import ReferenceAudio
from synthesis.constants import EMOTION_SUBCATEGORIES, REFERENCE_TRANSCRIPT
from synthesis.normalization import spoken_text, tts_text


def process_jsonl(
    input_path,
    output_path,
    *,
    synthesizer,
    normalizer,
    reference_dir="datasets/SpeechAccentArchive/recordings",
    reference_transcript=REFERENCE_TRANSCRIPT,
    normalize_only=False,
    limit=None,
    seed=0,
    language="English",
    max_new_tokens=None,
):
    """Preserve input fields; append normalized goal/turn text and synthesis metadata.

    Output is exclusive-create. On error it contains only completed dialogues;
    any WAVs already written remain available for diagnosis. Never overwrites input.
    Relative native audio paths are resolved against CWD, then the input directory.
    """
    source, target = Path(input_path).resolve(), Path(output_path).resolve()
    if source == target:
        raise ValueError("Input and output paths must differ")
    if target.exists():
        raise FileExistsError(target)
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")
    if not normalize_only and synthesizer is None:
        raise ValueError("A synthesizer is required")
    reference_dir = Path(reference_dir).resolve()
    audio_root = target.parent / f"{target.stem}_audio"
    stats = {"dialogues": 0, "synthesized_turns": 0, "reused_turns": 0, "control_turns": 0}
    seen = set()
    target.parent.mkdir(parents=True, exist_ok=True)
    with source.open(encoding="utf-8") as reader, target.open("x", encoding="utf-8") as writer:
        for line_number, line in enumerate(reader, 1):
            if not line.strip():
                continue
            if limit is not None and stats["dialogues"] >= limit:
                break
            dialogue_id = f"line {line_number}"
            try:
                record = json.loads(line)
                dialogue_id = record["dialogue_id"]
                if not isinstance(dialogue_id, str) or not dialogue_id:
                    raise ValueError("dialogue_id must be a nonempty string")
                if dialogue_id in seen:
                    raise ValueError("Duplicate dialogue_id")
                seen.add(dialogue_id)
                goal = record.get("goal")
                if not isinstance(goal, dict):
                    raise ValueError("goal.text must be a nonempty string")
                goal_text = goal.get("text")
                if not isinstance(goal_text, str):
                    raise ValueError("goal.text must be a nonempty string")
                goal_speech = spoken_text(goal_text)
                if not goal_speech:
                    raise ValueError("goal.text must be a nonempty string")
                normalized_goal = goal.get("normalized")
                if normalized_goal is None:
                    normalized_goal = normalizer(goal_speech)
                if not isinstance(normalized_goal, str) or not normalized_goal.strip():
                    raise ValueError("goal.normalized must be a nonempty string")
                goal["normalized"] = normalized_goal
                turns = record["turns"]
                if not isinstance(turns, list) or not turns:
                    raise ValueError("turns must be a nonempty list")
                voices = {}
                digest = hashlib.sha256(dialogue_id.encode()).hexdigest()
                for index, turn in enumerate(turns):
                    role = turn["role"]
                    if role not in {"user", "assistant"}:
                        raise ValueError(f"Invalid role at turn {index}: {role}")
                    raw_text = turn["text"]
                    if (
                        raw_text.strip() == "<|endoftext|>"
                        and not turn.get("audio_path")
                    ):
                        normalized = turn.get("text_normalized") or raw_text
                        if not isinstance(normalized, str):
                            raise ValueError(f"Invalid normalized text at turn {index}")
                        turn["normalized_text"] = normalized
                        turn["synthesis"] = {"status": "control_only"}
                        stats["control_turns"] += 1
                        continue
                    text = spoken_text(raw_text)
                    if not text:
                        raise ValueError(f"Empty speech text at turn {index}")
                    normalized = turn.get("text_normalized")
                    if normalized is None:
                        normalized = normalizer(text)
                    if not isinstance(normalized, str) or not normalized.strip():
                        raise ValueError(f"Empty normalized text at turn {index}")
                    turn["normalized_text"] = normalized
                    if turn.get("audio_path"):
                        native = Path(turn["audio_path"])
                        if not native.is_absolute() and not native.is_file():
                            native = source.parent / native
                        if not native.is_file():
                            raise FileNotFoundError(f"Native audio missing: {native}")
                        turn["audio_path"] = str(native.resolve())
                        if not normalize_only:
                            stats["reused_turns"] += 1
                        continue
                    if normalize_only:
                        continue
                    profile = record["speaker" if role == "user" else "assistant_speaker"]
                    filename = profile["filename"]
                    if Path(filename).name != filename:
                        raise ValueError("Speaker filename must be a basename")
                    reference = reference_dir / filename
                    if not reference.is_file():
                        raise FileNotFoundError(f"Reference audio missing: {reference}")
                    if filename not in voices:
                        voices[filename] = synthesizer.prepare_voice(
                            ReferenceAudio(
                                audio_path=reference,
                                transcript=reference_transcript,
                                x_vector_only_mode=False,
                            )
                        )
                    label = (turn.get("emotion") or {}).get("label", 0)
                    if role == "assistant":
                        emotion = "neutral"
                    else:
                        if label not in EMOTION_SUBCATEGORIES:
                            raise ValueError(f"Invalid emotion label: {label}")
                        rng = random.Random(f"{seed}:{dialogue_id}:{index}")
                        emotion = rng.choice(EMOTION_SUBCATEGORIES[label])
                    synthesis_output = synthesizer.synthesize(
                        text=tts_text(normalized),
                        prepared_voice=voices[filename],
                        emotion_name=emotion,
                        language=language,
                        max_new_tokens=max_new_tokens,
                    )
                    if isinstance(synthesis_output.audio, (bytes, bytearray)):
                        if not synthesis_output.audio:
                            raise ValueError(f"Empty audio at turn {index}")
                    else:
                        waveform = np.asarray(synthesis_output.audio)
                        if (
                            waveform.ndim != 1
                            or not waveform.size
                            or not np.isfinite(waveform).all()
                        ):
                            raise ValueError(f"Invalid waveform at turn {index}")
                        if (
                            not isinstance(synthesis_output.sample_rate, (int, np.integer))
                            or synthesis_output.sample_rate <= 0
                        ):
                            raise ValueError(f"Invalid sample rate at turn {index}")
                    audio_path = audio_root / digest / f"{index:05d}.wav"
                    audio_path.parent.mkdir(parents=True, exist_ok=True)
                    if audio_path.exists():
                        raise FileExistsError(audio_path)
                    synthesis_output.write_to_file(audio_path)
                    turn["audio_path"] = str(audio_path)
                    turn["synthesis"] = {
                        "backend": voices[filename].backend,
                        "emotion_keyword": emotion,
                        "reference_audio": str(reference),
                        "voice_name": voices[filename].voice_name,
                        "sample_rate": synthesis_output.sample_rate,
                        "media_type": synthesis_output.media_type,
                        "normalizer": type(normalizer).__name__,
                    }
                    stats["synthesized_turns"] += 1
                writer.write(json.dumps(record, ensure_ascii=False) + "\n")
                writer.flush()
                stats["dialogues"] += 1
            except Exception as exc:
                raise ValueError(
                    f"{source}:{line_number}, dialogue {dialogue_id!r}: {exc}"
                ) from exc
    return stats
