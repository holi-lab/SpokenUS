"""Barge-in sampling logic for turn selection and type assignment."""

import random
from typing import Literal

from augmentation.constants import (
    BARGEIN_TYPES,
    BARGEIN_SUBTYPES,
    BARGEIN_SAMPLE_RATE,
)
from augmentation.bargein.types import BargeInType, BargeInSubtype


def sample_bargein_turns(
    turns: list[dict],
    sample_rate: float = BARGEIN_SAMPLE_RATE,
) -> list[int]:
    """Sample user turn indices for barge-in augmentation.
    
    Selects approximately 25% of user turns randomly for barge-in.
    Excludes cross-turn segment turns (they inherit from previous turns).
    
    Args:
        turns: List of dialogue turns
        sample_rate: Fraction of user turns to sample (default: 0.25)
    
    Returns:
        List of turn indices selected for barge-in
    """
    # Get eligible user turn indices (skip cross-turn segments)
    user_indices = [
        i for i, t in enumerate(turns) 
        if t.get("role") == "user" and not t.get("segment")
    ]
    
    if not user_indices:
        return []
    
    # Calculate number of turns to sample
    n_sample = max(1, int(len(user_indices) * sample_rate))
    n_sample = min(n_sample, len(user_indices))
    
    # Random sample without replacement
    return sorted(random.sample(user_indices, n_sample))


def select_bargein_type() -> tuple[BargeInType, BargeInSubtype]:
    """Randomly select a barge-in type and subtype.
    
    Each of the 3 main types has equal probability (1/3).
    Within each type, subtypes have equal probability.
    
    Returns:
        Tuple of (bargein_type, bargein_subtype)
    """
    bargein_type: BargeInType = random.choice(BARGEIN_TYPES)  # type: ignore
    subtypes = BARGEIN_SUBTYPES[bargein_type]
    bargein_subtype: BargeInSubtype = random.choice(subtypes)  # type: ignore
    
    return bargein_type, bargein_subtype


def is_eligible_for_bargein(
    turn: dict,
    turn_idx: int,
    turns: list[dict],
) -> bool:
    """Check if a turn is eligible for barge-in augmentation.
    
    Conditions:
    1. Must be a user turn
    2. Must not be a cross-turn segment
    3. Must have a following assistant turn (for most barge-in types)
    
    Args:
        turn: The turn to check
        turn_idx: Index of the turn
        turns: Full dialogue turns
    
    Returns:
        True if eligible for barge-in
    """
    # Must be user turn
    if turn.get("role") != "user":
        return False
    
    # Skip cross-turn segments
    if turn.get("segment"):
        return False
    
    # Should have following assistant turn (for context)
    if turn_idx + 1 >= len(turns):
        return False
    
    next_turn = turns[turn_idx + 1]
    if next_turn.get("role") != "assistant":
        return False
    
    return True
