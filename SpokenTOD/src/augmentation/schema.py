"""Data structures for augmentation pipeline."""

from dataclasses import dataclass, field
from typing import Any

from demographic_sampler import SpeakerProfile, AssistantSpeaker


@dataclass
class SlotSpan:
    """Slot mention span within an utterance.

    Represents the location of a slot value in the text using character offsets.
    All datasets are normalized to this format.
    """

    slot: str  # Slot name (e.g., "city", "restaurant.name")
    value: str  # Slot value as it appears in text
    start: int  # Start character offset (inclusive)
    end: int  # End character offset (exclusive)

    def to_dict(self) -> dict:
        return {
            "slot": self.slot,
            "value": self.value,
            "start": self.start,
            "end": self.end,
        }


@dataclass
class Turn:
    """Single dialogue turn."""
    role: str  # "user" or "assistant"
    text: str
    slots: list[SlotSpan] = field(default_factory=list)  # Slot spans in this turn
    dialog_act: Any = None  # Original dialogue act info
    tagged: str | None = None
    emotion: dict | None = None  # {"label": int, "token": str}
    disfluency: list[dict] | None = None
    segment: dict | None = None  # cross-turn segment info
    state: dict | None = None  # Cumulative belief state up to this turn
    bargein: dict | None = None  # {"type": str, "subtype": str} for barge-in turns
    audio_path: str | None = None  # Path to existing audio file (e.g. SpokenWOZ)


@dataclass
class StructuredGoal:
    """Structured user goal with intents and slots."""
    domains: list[str]
    intents: list[dict]  # [{"domain", "intent", "slots", "requests"}]
    metadata: dict = field(default_factory=dict)


@dataclass
class Goal:
    """User goal with text and structured form."""
    text: str
    structured: StructuredGoal


@dataclass
class Dialogue:
    """Raw dialogue from source dataset."""
    id: str
    source: str
    turns: list[dict]
    goal: Goal | None = None
    state: dict = field(default_factory=dict)
    emotion_labels: list[int] | None = None  # For EmoWOZ
    metadata: dict = field(default_factory=dict)


@dataclass
class AugmentedDialogue:
    """Augmented dialogue with all features applied."""
    id: str
    source: str
    goal: dict  # {"text": str, "structured": dict}
    turns: list[Turn]
    state: dict
    speaker: SpeakerProfile  # User speaker demographic
    assistant_speaker: AssistantSpeaker  # Assistant speaker from Native pool
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to JSON-serializable dict."""
        return {
            "dialogue_id": self.id,
            "source": self.source,
            "goal": self.goal,
            "turns": [
                {
                    k: v
                    for k, v in {
                        "role": t.role,
                        "text": t.text,
                        "slots": [s.to_dict() for s in t.slots] if t.slots else None,
                        "tagged": t.tagged,
                        "emotion": t.emotion,
                        "disfluency": t.disfluency,
                        "segment": t.segment,
                        "state": t.state,
                        "bargein": t.bargein,
                        "audio_path": t.audio_path,
                    }.items()
                    if v is not None
                }
                for t in self.turns
            ],
            "state": self.state,
            "speaker": self.speaker,
            "assistant_speaker": self.assistant_speaker,
            "metadata": self.metadata,
        }
