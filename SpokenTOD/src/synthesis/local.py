from __future__ import annotations

from pathlib import Path

from synthesis.constants import LANGUAGE, MAX_NEW_TOKENS, MODEL_ID, REFERENCE_TRANSCRIPT
from synthesis.utils import (
    generate_emotion_system_prompt,
    save_wav,
)


def parse_torch_dtype(dtype: str):
    import torch

    aliases = {"fp16": "float16", "bf16": "bfloat16", "fp32": "float32"}
    name = aliases.get(dtype, dtype)
    if name not in {"float16", "bfloat16", "float32"}:
        raise ValueError(f"Unsupported dtype: {dtype}")
    return getattr(torch, name)


def load_qwen_model(
    model_id: str = MODEL_ID,
    device_map: str = "cuda:0",
    attn_implementation: str = "sdpa",
    dtype: str = "bfloat16",
):
    try:
        from qwen_tts import Qwen3TTSModel
    except ImportError as exc:
        raise ImportError("Install TTS dependencies with `uv sync --extra synthesis`.") from exc
    return Qwen3TTSModel.from_pretrained(
        model_id,
        device_map=device_map,
        dtype=parse_torch_dtype(dtype),
        attn_implementation=attn_implementation,
    )


def _normalize_token_id(token_id) -> int | None:
    if isinstance(token_id, list):
        token_id = token_id[0] if token_id else None
    return int(token_id) if token_id is not None else None


def _resolve_token_id(obj) -> int | None:
    if obj is None:
        return None

    pad_id = _normalize_token_id(getattr(obj, "pad_token_id", None))
    if pad_id is not None:
        return pad_id

    return _normalize_token_id(getattr(obj, "eos_token_id", None))


# qwen-tts forwards generate() without normalizing pad_token_id, and tokenizer/config special tokens can diverge: https://github.com/QwenLM/Qwen3/issues/927
def _resolve_pad_token_id(model) -> int | None:
    candidates = (
        getattr(getattr(model, "processor", None), "tokenizer", None),
        getattr(getattr(model, "model", None), "config", None),
    )
    for candidate in candidates:
        token_id = _resolve_token_id(candidate)
        if token_id is not None:
            return token_id
    return None


def _build_instruct_id(model, emotion_name: str):
    instruct_text = generate_emotion_system_prompt(emotion_name)
    return model._tokenize_texts([model._build_instruct_text(instruct_text)])[0]


def build_voice_clone_prompt(
    model,
    ref_audio: str | Path,
    ref_text: str = REFERENCE_TRANSCRIPT,
    x_vector_only_mode: bool = False,
):
    prompt = model.create_voice_clone_prompt(
        ref_audio=[str(ref_audio)],
        ref_text=[ref_text],
        x_vector_only_mode=[x_vector_only_mode],
    )
    return prompt[0]


def synthesize_voice_clone(
    model,
    text: str,
    voice_clone_prompt,
    emotion_name: str,
    language: str = LANGUAGE,
    max_new_tokens: int = MAX_NEW_TOKENS,
    pad_token_id: int | None = None,
):
    resolved_pad_token_id = _resolve_pad_token_id(model) if pad_token_id is None else pad_token_id
    instruct_id = _build_instruct_id(model, emotion_name)
    wavs, sample_rate = model.generate_voice_clone(
        text=[text],
        language=[language],
        voice_clone_prompt=[voice_clone_prompt],
        instruct_ids=[instruct_id],
        max_new_tokens=max_new_tokens,
        pad_token_id=resolved_pad_token_id,
    )
    return wavs[0], sample_rate


class QwenTTSSynthesizer:
    def __init__(
        self,
        model,
        reference_transcript: str = REFERENCE_TRANSCRIPT,
        default_language: str = LANGUAGE,
        max_new_tokens: int = MAX_NEW_TOKENS,
    ):
        self.model = model
        self.reference_transcript = reference_transcript
        self.default_language = default_language
        self.max_new_tokens = max_new_tokens

    @classmethod
    def from_pretrained(
        cls,
        model_id: str = MODEL_ID,
        device_map: str = "cuda:0",
        attn_implementation: str = "sdpa",
        dtype: str = "bfloat16",
        reference_transcript: str = REFERENCE_TRANSCRIPT,
        default_language: str = LANGUAGE,
        max_new_tokens: int = MAX_NEW_TOKENS,
    ) -> QwenTTSSynthesizer:
        model = load_qwen_model(
            model_id=model_id,
            device_map=device_map,
            attn_implementation=attn_implementation,
            dtype=dtype,
        )
        return cls(
            model=model,
            reference_transcript=reference_transcript,
            default_language=default_language,
            max_new_tokens=max_new_tokens,
        )

    def build_voice_clone_prompt(
        self,
        ref_audio: str | Path,
        ref_text: str | None = None,
        x_vector_only_mode: bool = False,
    ):
        return build_voice_clone_prompt(
            self.model,
            ref_audio=ref_audio,
            ref_text=ref_text or self.reference_transcript,
            x_vector_only_mode=x_vector_only_mode,
        )

    def synthesize(
        self,
        text: str,
        voice_clone_prompt,
        emotion_name: str,
        language: str | None = None,
        pad_token_id: int | None = None,
    ):
        return synthesize_voice_clone(
            model=self.model,
            text=text,
            voice_clone_prompt=voice_clone_prompt,
            emotion_name=emotion_name,
            language=language or self.default_language,
            max_new_tokens=self.max_new_tokens,
            pad_token_id=pad_token_id,
        )

    def synthesize_to_file(
        self,
        text: str,
        ref_audio: str | Path,
        output_path: str | Path,
        emotion_name: str,
        ref_text: str | None = None,
        language: str | None = None,
    ) -> Path:
        voice_clone_prompt = self.build_voice_clone_prompt(
            ref_audio=ref_audio,
            ref_text=ref_text,
        )
        wav, sample_rate = self.synthesize(
            text=text,
            voice_clone_prompt=voice_clone_prompt,
            emotion_name=emotion_name,
            language=language,
        )
        save_wav(output_path, wav, sample_rate)
        return Path(output_path)
