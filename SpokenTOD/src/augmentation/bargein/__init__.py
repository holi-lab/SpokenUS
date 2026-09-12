"""Barge-in augmentation module for turn-taking phenomena."""

from augmentation.bargein.types import BargeInResult
from augmentation.bargein.sampler import sample_bargein_turns, select_bargein_type
from augmentation.bargein.injector import (
    inject_bargein_dialogue,
    inject_bargein_dialogues_batch_async,
)

__all__ = [
    "BargeInResult",
    "sample_bargein_turns",
    "select_bargein_type",
    "inject_bargein_dialogue",
    "inject_bargein_dialogues_batch_async",
]
