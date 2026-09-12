"""Prepare spoken text without modifying source transcripts or annotations."""

import re
from pathlib import Path

_ANNOTATION = re.compile(r"\[(?:FP|DM|EDIT|REP|RST|COR)\]")
_TTS_CONTROL = re.compile(r"<bargein>|<\|endoftext\|>", flags=re.IGNORECASE)


def spoken_text(text: str) -> str:
    """Remove inline annotation tags while retaining model control tokens.

    The released SpokenTOD metadata preserves ``<|endoftext|>`` in
    ``text_normalized`` for non-standalone terminal turns.  NeMo leaves that
    token intact, so only augmentation tags are removed here.
    """
    return _ANNOTATION.sub("", text).strip()


def tts_text(text: str) -> str:
    """Remove control markers from the text sent to a speech synthesizer.

    SpokenTOD metadata retains barge-in and terminal markers in its public
    text fields. They describe dialogue control flow and must not be spoken.
    """
    return _TTS_CONTROL.sub("", text).strip()


class NemoNormalizer:
    def __init__(self, cache_dir: str | Path = ".cache/nemo"):
        try:
            from nemo_text_processing.text_normalization.normalize import Normalizer
        except ImportError as exc:
            raise ImportError("Install NeMo with `uv sync --extra normalization`.") from exc
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        self.normalizer = Normalizer(input_case="cased", lang="en", cache_dir=str(cache_dir))

    def __call__(self, text: str) -> str:
        return self.normalizer.normalize(
            spoken_text(text),
            punct_pre_process=False,
            punct_post_process=False,
        )
