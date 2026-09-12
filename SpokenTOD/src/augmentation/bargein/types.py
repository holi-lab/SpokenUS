"""Barge-in data types and structures."""

from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel, Field

# Type aliases for barge-in categories
BargeInType = Literal["ERROR_RECOVERY", "CLARIFICATION", "EFFICIENCY"]
BargeInSubtype = Literal[
    "INCOHERENT_RAW", "INCOHERENT_INTERP",
    "FAIL_RAW", "FAIL_INTERP",
    "REF_IMPL", "REF_RAW", "REF_INTERP"
]


# Pydantic models for structured LLM output
class BargeInTurn(BaseModel):
    """A single dialogue turn in barge-in result."""
    role: Literal["user", "assistant"] = Field(description="Speaker role")
    text: str = Field(description="The utterance text. For first assistant turn, must end with <bargein> tag to indicate truncation.")


class BargeInLLMResponse(BaseModel):
    """Structured response from LLM for barge-in generation."""
    applicable: bool = Field(description="Whether the barge-in pattern can be naturally applied to this dialogue")
    reason: str = Field(default="", description="Brief explanation if not applicable")
    turns: list[BargeInTurn] = Field(default_factory=list, description="Modified dialogue turns with barge-in applied. First turn must be truncated assistant speech ending with <bargein>")
    erroneous_slots: dict[str, str] | None = Field(default=None, description="For ERROR_RECOVERY types (G_INCOHERENT), mapping of slot names to INCORRECT values that the assistant mistakenly said")
    corrected_slots: dict[str, str] | None = Field(default=None, description="For ERROR_RECOVERY types, mapping of slot names to correct values after user correction")


@dataclass
class BargeInResult:
    """Result of barge-in augmentation for a single turn.

    Attributes:
        applied: Whether barge-in was successfully applied
        bargein_type: The main barge-in category (ERROR_RECOVERY, CLARIFICATION, EFFICIENCY)
        bargein_subtype: The specific subtype within the category
        modified_turns: List of modified/new turns to replace the original
        emotion_override: Emotion dict for Error Recovery cases ({label, name})
        erroneous_slots: For G_INCOHERENT, slots with WRONG values {slot_name: wrong_value}
        corrected_slots: For Error Recovery, slots that were corrected {slot_name: correct_value}
    """
    applied: bool
    bargein_type: BargeInType | None = None
    bargein_subtype: BargeInSubtype | None = None
    modified_turns: list[dict] = field(default_factory=list)
    emotion_override: dict | None = None
    erroneous_slots: dict | None = None  # For G_INCOHERENT types
    corrected_slots: dict | None = None  # For state rollback in Error Recovery


@dataclass
class BargeInRequest:
    """Request structure for barge-in LLM call.

    Attributes:
        dialogue_idx: Index of dialogue in batch
        turn_idx: Index of user turn to augment
        bargein_type: Selected barge-in type
        bargein_subtype: Selected subtype
        context: Previous turns for context
        current_turn: The user turn to augment
        next_turn: The assistant turn following (if exists)
        user_goal: The user's goal for this dialogue
    """
    dialogue_idx: int
    turn_idx: int
    bargein_type: BargeInType
    bargein_subtype: BargeInSubtype
    context: list[dict]
    current_turn: dict
    next_turn: dict | None = None
    user_goal: dict | None = None

