import argparse
import json
import os
from pathlib import Path

from synthesis.constants import MODEL_ID, REFERENCE_TRANSCRIPT
from synthesis.vllm_omni import VllmOmniSynthesizer

DEFAULT_REFERENCE_DIR = Path("datasets/SpeechAccentArchive/recordings/recordings")


def create_synthesizer(args):
    return VllmOmniSynthesizer(
        base_url=args.base_url,
        model=args.model_id,
        timeout=args.timeout,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Normalize SpokenTOD JSONL and synthesize Qwen3-TTS via vLLM-Omni."
    )
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--text", help="Single utterance to synthesize")
    inputs.add_argument("--input-jsonl", type=Path, help="Augmented dialogue JSONL")
    parser.add_argument("--output-jsonl", type=Path, help="New output manifest (must not exist)")
    parser.add_argument(
        "--reference-dir", type=Path, default=DEFAULT_REFERENCE_DIR
    )
    parser.add_argument(
        "--normalize-only", action="store_true", help="Normalize JSONL without loading TTS"
    )
    parser.add_argument("--normalizer-cache", type=Path, default=Path(".cache/nemo"))
    parser.add_argument("--limit", type=int, help="Maximum number of dialogues")
    parser.add_argument("--seed", type=int, default=0, help="Emotion keyword sampling seed")
    parser.add_argument("--max-new-tokens", type=int, default=4608)
    parser.add_argument(
        "--base-url",
        default=os.getenv("VLLM_OMNI_BASE_URL", "http://127.0.0.1:8000"),
        help="Self-hosted vLLM-Omni URL (or VLLM_OMNI_BASE_URL)",
    )
    parser.add_argument("--timeout", type=float, default=120.0, help="vLLM request timeout")
    parser.add_argument(
        "--emotion",
        default="neutral",
        help="Emotion name used for the system instruction prompt",
    )
    parser.add_argument(
        "--ref-audio",
        help="Reference speaker audio path used for voice cloning",
    )
    parser.add_argument("--output-path", help="Output WAV path")
    parser.add_argument(
        "--model-id",
        default=MODEL_ID,
        help="Hugging Face model id or local model path",
    )
    parser.add_argument(
        "--ref-text",
        default=None,
        help="Optional transcript for the reference audio",
    )
    parser.add_argument(
        "--language",
        default=None,
        help="Optional synthesis language override",
    )

    args = parser.parse_args()

    if args.max_new_tokens < 1 or (args.limit is not None and args.limit < 1):
        parser.error("--max-new-tokens and --limit must be positive")
    if args.input_jsonl:
        if not args.output_jsonl:
            parser.error("--input-jsonl requires --output-jsonl")
        if args.output_jsonl.exists():
            parser.error("--output-jsonl already exists; choose a new output path")
        if not args.input_jsonl.is_file():
            parser.error("--input-jsonl does not exist")
        from synthesis.normalization import NemoNormalizer
        from synthesis.pipeline import process_jsonl

        normalizer = NemoNormalizer(args.normalizer_cache)
        synthesizer = None
        if not args.normalize_only:
            synthesizer = create_synthesizer(args)
        stats = process_jsonl(
            args.input_jsonl,
            args.output_jsonl,
            synthesizer=synthesizer,
            normalizer=normalizer,
            reference_dir=args.reference_dir,
            reference_transcript=args.ref_text or REFERENCE_TRANSCRIPT,
            normalize_only=args.normalize_only,
            limit=args.limit,
            seed=args.seed,
            language=args.language or "English",
            max_new_tokens=args.max_new_tokens,
        )
        print(json.dumps(stats))
        return
    if args.normalize_only or args.output_jsonl:
        parser.error("--normalize-only and --output-jsonl require --input-jsonl")
    if not args.ref_audio or not args.output_path:
        parser.error("--text requires --ref-audio and --output-path")

    from synthesis.base import ReferenceAudio

    synthesizer = create_synthesizer(args)
    voice = synthesizer.prepare_voice(
        ReferenceAudio(
            audio_path=args.ref_audio,
            transcript=args.ref_text or REFERENCE_TRANSCRIPT,
            x_vector_only_mode=False,
        )
    )
    result = synthesizer.synthesize(
        text=args.text,
        prepared_voice=voice,
        emotion_name=args.emotion,
        language=args.language,
        max_new_tokens=args.max_new_tokens,
    )
    result.write_to_file(args.output_path)


if __name__ == "__main__":
    main()
