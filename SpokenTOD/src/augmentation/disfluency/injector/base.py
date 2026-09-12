"""Base classes and dataclasses for disfluency injection."""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class InjectionResult:
    """
    Result of a disfluency injection operation.

    Contains both raw text and TTS-tagged text, plus offset tracking.
    """

    # The modified text (without TTS tags)
    raw_text: str

    # The TTS-tagged version (e.g., "[FP] um, I want...")
    tagged_text: str

    # Number of characters added (for offset tracking)
    char_offset: int

    # Position where insertion occurred (character index)
    insert_position: int

    # Metadata annotation for this injection
    annotation: dict[str, Any] = field(default_factory=dict)


@dataclass
class DisfluencySpec:
    """Specification for a disfluency to be applied."""

    # Type: FP, DM, EDIT, PRO, REP, RST, COR, SLR
    type: str

    # Cognitive phase: planning, articulation, repair
    phase: str

    # Token position (for rule-based)
    token_position: int | None = None

    # Related slot name (if near a slot)
    slot_related: str | None = None

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "DisfluencySpec":
        return cls(
            type=d["type"],
            phase=d["phase"],
            token_position=d.get("token_position"),
            slot_related=d.get("slot_related"),
        )
