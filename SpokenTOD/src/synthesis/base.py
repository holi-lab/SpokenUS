from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from synthesis.utils import write_audio_output


@dataclass(slots=True, frozen=True)
class ReferenceAudio:
    audio_path: str | Path
    transcript: str | None = None
    x_vector_only_mode: bool | None = None
    media_type: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "audio_path", Path(self.audio_path))

    @property
    def normalized_transcript(self) -> str | None:
        if self.transcript is None:
            return None
        transcript = self.transcript.strip()
        return transcript or None

    @property
    def filename(self) -> str:
        return self.audio_path.name

    def read_bytes(self) -> bytes:
        return self.audio_path.read_bytes()


@dataclass(slots=True)
class PreparedVoice:
    backend: str
    voice_name: str | None = None
    reference: ReferenceAudio | None = None
    voice_clone_prompt: dict[str, Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class SynthesisOutput:
    audio: bytes | Any
    sample_rate: int | None = None
    media_type: str | None = None
    response_format: str = "wav"
    metadata: dict[str, Any] = field(default_factory=dict)

    def write_to_file(self, output_path: str | Path) -> Path:
        return write_audio_output(output_path, self.audio, self.sample_rate)


class SpeechSynthesizer(ABC):
    @abstractmethod
    def prepare_voice(
        self,
        reference: ReferenceAudio,
        *,
        voice_name: str | None = None,
        force: bool = False,
    ) -> PreparedVoice:
        raise NotImplementedError

    @abstractmethod
    def synthesize(
        self,
        text: str,
        prepared_voice: PreparedVoice,
        emotion_name: str,
        *,
        language: str | None = None,
        response_format: str = "wav",
        **kwargs: Any,
    ) -> SynthesisOutput:
        raise NotImplementedError

    def synthesize_to_file(
        self,
        text: str,
        prepared_voice: PreparedVoice,
        output_path: str | Path,
        emotion_name: str,
        *,
        language: str | None = None,
        response_format: str = "wav",
        **kwargs: Any,
    ) -> Path:
        result = self.synthesize(
            text=text,
            prepared_voice=prepared_voice,
            emotion_name=emotion_name,
            language=language,
            response_format=response_format,
            **kwargs,
        )
        return result.write_to_file(output_path)


__all__ = [
    "PreparedVoice",
    "ReferenceAudio",
    "SpeechSynthesizer",
    "SynthesisOutput",
]
