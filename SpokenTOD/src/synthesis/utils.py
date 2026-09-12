from __future__ import annotations

import base64
import hashlib
import mimetypes
import random
import re
from pathlib import Path

from synthesis.constants import EMOTION_CATEGORIES, EMOTION_SUBCATEGORIES


def get_category_name(category_id: int) -> str:
    if category_id not in EMOTION_CATEGORIES:
        raise ValueError(f"Invalid category_id: {category_id}. Must be 0-6.")
    return EMOTION_CATEGORIES[category_id]


def sample_emotion(category_id: int) -> str:
    if category_id not in EMOTION_SUBCATEGORIES:
        raise ValueError(f"Invalid category_id: {category_id}. Must be 0-6.")
    return random.choice(EMOTION_SUBCATEGORIES[category_id])


def sample_emotion_with_category(category_id: int) -> tuple[str, str]:
    category_name = get_category_name(category_id)
    emotion = sample_emotion(category_id)
    return category_name, emotion


def generate_emotion_system_prompt(emotion: str) -> str:
    return f"You are a helpful assistant. Please speak in a {emotion} tone.<|endofprompt|>"


def format_time(seconds: float) -> str:
    """Format seconds into human readable string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    if seconds < 3600:
        mins = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{mins}m {secs}s"
    hours = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    return f"{hours}h {mins}m"


def read_reference_transcript(path: str | Path) -> str:
    transcript_path = Path(path)
    try:
        return transcript_path.read_text(encoding="utf-8").strip()
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Missing {transcript_path}. Run `make download saa`.") from exc


def normalize_media_type(media_type: str | None) -> str:
    if media_type in {"audio/x-wav", "audio/wave", None}:
        return "audio/wav"
    return media_type


def guess_audio_media_type(audio_path: str | Path) -> str:
    media_type, _ = mimetypes.guess_type(str(audio_path))
    return normalize_media_type(media_type)


def encode_audio_as_data_uri(audio_path: str | Path, media_type: str | None = None) -> str:
    path = Path(audio_path)
    resolved_media_type = normalize_media_type(media_type) if media_type else guess_audio_media_type(path)
    audio_b64 = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{resolved_media_type};base64,{audio_b64}"


def build_default_voice_name(
    audio_path: str | Path,
    *,
    transcript: str | None = None,
    x_vector_only_mode: bool | None = None,
) -> str:
    path = Path(audio_path)
    stem = re.sub(r"[^a-z0-9]+", "-", path.stem.lower()).strip("-") or "voice"
    normalized_transcript = transcript.strip() if transcript else ""
    suffix = hashlib.sha1(
        f"{path.resolve()}::{normalized_transcript}::{x_vector_only_mode}".encode()
    ).hexdigest()[:8]
    return f"{stem}-{suffix}"


def _soundfile_write(path: str, wav, samplerate: int, format: str) -> None:
    try:
        import soundfile as sf
    except ImportError as exc:
        raise ImportError(
            "soundfile is required for writing WAV files. "
            "Install the synthesis extra with `uv sync --extra synthesis`."
        ) from exc
    sf.write(path, wav, samplerate=samplerate, format=format)


def write_audio_output(output_path: str | Path, audio, sample_rate: int | None = None) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(audio, (bytes, bytearray)):
        path.write_bytes(bytes(audio))
        return path

    if sample_rate is None:
        raise ValueError("sample_rate is required when writing waveform audio")

    _soundfile_write(str(path), audio, samplerate=sample_rate, format="WAV")
    return path


def save_wav(output_path: str | Path, wav, sample_rate: int) -> None:
    write_audio_output(output_path, wav, sample_rate)
