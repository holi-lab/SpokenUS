"""Shared type definitions for augmentation."""

from typing import NotRequired, TypedDict


class SlotPosition(TypedDict):
    value: str
    start: int | None
    end: int | None
    slot: NotRequired[str]
    original_value: NotRequired[str]
    unrecovered: NotRequired[bool]
    token_start: NotRequired[int | None]
    token_end: NotRequired[int | None]


SlotPositions = dict[str, SlotPosition]
