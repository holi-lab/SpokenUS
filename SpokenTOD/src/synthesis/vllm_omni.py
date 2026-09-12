from __future__ import annotations

from typing import Any
from urllib.parse import quote

import requests

from synthesis.base import PreparedVoice, ReferenceAudio, SpeechSynthesizer, SynthesisOutput
from synthesis.utils import (
    build_default_voice_name,
    encode_audio_as_data_uri,
    generate_emotion_system_prompt,
    guess_audio_media_type,
)

DEFAULT_MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"


class VllmOmniError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        payload: Any = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.payload = payload


class VllmOmniSynthesizer(SpeechSynthesizer):
    def __init__(
        self,
        base_url: str,
        *,
        model: str = DEFAULT_MODEL,
        timeout: float = 120.0,
        session: requests.Session | Any | None = None,
        default_consent: str = "consented",
        enable_cache: bool = True,
        allow_inline_fallback: bool = True,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.session = session or requests.Session()
        self.default_consent = default_consent
        self.enable_cache = enable_cache
        self.allow_inline_fallback = allow_inline_fallback

    def prepare_voice(
        self,
        reference: ReferenceAudio,
        *,
        voice_name: str | None = None,
        force: bool = False,
    ) -> PreparedVoice:
        resolved_voice_name = voice_name or self._default_voice_name(reference)

        try:
            upload_result = self._upload_voice(reference, resolved_voice_name)
        except VllmOmniError as exc:
            if self.allow_inline_fallback and exc.status_code in {404, 405}:
                return PreparedVoice(
                    backend="vllm_omni",
                    reference=reference,
                    metadata={"transport": "inline", "upload_error": str(exc)},
                )
            if "already exists" not in str(exc).lower():
                raise
            upload_result = {"name": resolved_voice_name, "reused": True}

        metadata = {
            "transport": "uploaded",
            "upload": upload_result,
        }

        if self.enable_cache:
            try:
                cache_result = self._create_voice_cache(resolved_voice_name, force=force)
            except VllmOmniError as exc:
                if exc.status_code not in {404, 405}:
                    raise
                metadata["cache_status"] = "unsupported"
                metadata["cache_error"] = str(exc)
            else:
                metadata["cache"] = cache_result
                metadata["cache_status"] = cache_result.get("cache_status")

        return PreparedVoice(
            backend="vllm_omni",
            voice_name=resolved_voice_name,
            reference=reference,
            metadata=metadata,
        )

    def synthesize(
        self,
        text: str,
        prepared_voice: PreparedVoice,
        emotion_name: str,
        *,
        language: str | None = None,
        response_format: str = "wav",
        speed: float = 1.0,
        max_new_tokens: int | None = None,
        **_: Any,
    ) -> SynthesisOutput:
        payload: dict[str, Any] = {
            "input": text,
            "model": self.model,
            "response_format": response_format,
            "instructions": generate_emotion_system_prompt(emotion_name),
            "speed": speed,
        }
        if language is not None:
            payload["language"] = language
        if max_new_tokens is not None:
            payload["max_new_tokens"] = max_new_tokens

        if prepared_voice.voice_name is not None:
            payload["voice"] = prepared_voice.voice_name
        else:
            payload.update(self._build_inline_voice_payload(prepared_voice.reference))

        response = self.session.post(
            self._url("/v1/audio/speech"),
            json=payload,
            timeout=self.timeout,
        )
        self._raise_for_status(response, "Speech synthesis failed")

        media_type = response.headers.get("content-type", f"audio/{response_format}")
        media_type = media_type.split(";", 1)[0]
        return SynthesisOutput(
            audio=response.content,
            media_type=media_type,
            response_format=response_format,
            metadata={"voice_name": prepared_voice.voice_name},
        )

    def _upload_voice(self, reference: ReferenceAudio, voice_name: str) -> dict[str, Any]:
        media_type = reference.media_type or guess_audio_media_type(reference.audio_path)
        data = {
            "name": voice_name,
            "consent": self.default_consent,
        }
        if reference.normalized_transcript is not None:
            data["ref_text"] = reference.normalized_transcript

        response = self.session.post(
            self._url("/v1/audio/voices"),
            data=data,
            files={
                "audio_sample": (
                    reference.filename,
                    reference.read_bytes(),
                    media_type,
                )
            },
            timeout=self.timeout,
        )
        payload = self._decode_json(response)
        self._raise_for_status(response, "Voice upload failed", payload=payload)

        if isinstance(payload, dict) and isinstance(payload.get("voice"), dict):
            return payload["voice"]
        if isinstance(payload, dict):
            return payload
        raise VllmOmniError("Voice upload returned an unexpected response", payload=payload)

    def _create_voice_cache(self, voice_name: str, *, force: bool) -> dict[str, Any]:
        response = self.session.post(
            self._url(f"/v1/audio/voices/{quote(voice_name, safe='')}/cache"),
            params={"force": force},
            timeout=self.timeout,
        )
        payload = self._decode_json(response)
        self._raise_for_status(response, "Voice cache creation failed", payload=payload)

        if isinstance(payload, dict):
            return payload
        raise VllmOmniError("Voice cache endpoint returned an unexpected response", payload=payload)

    def _build_inline_voice_payload(self, reference: ReferenceAudio | None) -> dict[str, Any]:
        if reference is None:
            raise ValueError("reference audio is required when no uploaded voice name is available")

        payload: dict[str, Any] = {
            "task_type": "Base",
            "ref_audio": encode_audio_as_data_uri(reference.audio_path, reference.media_type),
        }
        if reference.x_vector_only_mode is True:
            payload["x_vector_only_mode"] = True
            return payload
        if reference.normalized_transcript is not None:
            payload["ref_text"] = reference.normalized_transcript
            return payload
        payload["x_vector_only_mode"] = True
        return payload

    def _default_voice_name(self, reference: ReferenceAudio) -> str:
        return build_default_voice_name(
            reference.audio_path,
            transcript=reference.normalized_transcript,
            x_vector_only_mode=reference.x_vector_only_mode,
        )

    def _url(self, path: str) -> str:
        return f"{self.base_url}{path}"

    def _decode_json(self, response) -> Any | None:
        try:
            return response.json()
        except ValueError:
            return None

    def _raise_for_status(
        self,
        response,
        default_message: str,
        *,
        payload: Any = None,
    ) -> None:
        if 200 <= response.status_code < 300:
            return

        if payload is None:
            payload = self._decode_json(response)
        message = self._extract_error_message(payload) or getattr(response, "text", "") or default_message
        raise VllmOmniError(message, status_code=response.status_code, payload=payload)

    def _extract_error_message(self, payload: Any) -> str | None:
        if isinstance(payload, dict):
            error = payload.get("error")
            if isinstance(error, str):
                return error
            if isinstance(error, dict):
                nested = error.get("message")
                if isinstance(nested, str):
                    return nested
            message = payload.get("message")
            if isinstance(message, str):
                return message
            detail = payload.get("detail")
            if isinstance(detail, str):
                return detail
        return None


__all__ = [
    "DEFAULT_MODEL",
    "VllmOmniError",
    "VllmOmniSynthesizer",
]
