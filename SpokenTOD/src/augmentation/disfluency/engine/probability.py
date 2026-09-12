"""Probability engine for sampling disfluencies based on dialogue acts."""

import random
from typing import Any

from augmentation.disfluency.definitions import SlotPositions
from augmentation.disfluency.config import ALL_TYPES

# Disfluency types that go BEFORE slot (Planning/Retrieval - thinking before saying)
BEFORE_SLOT_TYPES = {"FP", "DM", "EDIT"}

# Disfluency types that apply AT slot position (Articulation - during pronunciation)
AT_SLOT_TYPES = {"REP", "SLR", "PRO"}

# Disfluency types that apply BEFORE/AT slot (Repair - correcting mistakes)
REPAIR_TYPES = {"COR", "RST"}


def sample_disfluencies_for_turn(
    words: list[str],
    slot_positions: SlotPositions,
    num_tokens: int,
) -> list[dict[str, Any]]:
    """
    Sample disfluencies based on sentence length and assign positions.

    Probability formula: P(disfluent | L) = 1 - 0.9453^L
    where L is the number of words.

    If disfluent:
    - 50% chance to apply at slot position, 50% at any position
    - Position depends on disfluency type:
      - BEFORE_SLOT_TYPES (FP, DM, EDIT): insert before slot
      - AT_SLOT_TYPES (REP, SLR, PRO): apply at slot value
      - REPAIR_TYPES (COR, RST): apply at/before slot

    Args:
        words: List of words in the utterance
        slot_positions: Slot information with token positions
        num_tokens: Total number of tokens in utterance

    Returns:
        List containing at most one disfluency specification with token_position.
    """
    L = len(words)
    if L == 0:
        return []

    # Probability of disfluency
    prob_disfluent = 1.0 - (0.9453 ** L)

    if random.random() >= prob_disfluent:
        return []

    # Select disfluency type (any type can be selected)
    selected_type = random.choice(ALL_TYPES)

    # Determine if this should be applied at slot position (50%) or any position (50%)
    has_slots = any(
        info.get("token_start") is not None
        for info in slot_positions.values()
    )

    apply_at_slot = has_slots and random.random() < 0.5

    # Determine token position based on type and apply_at_slot
    if apply_at_slot:
        slot_names = list(slot_positions.keys())
        chosen_slot = random.choice(slot_names)
        slot_info = slot_positions[chosen_slot]
        token_start = slot_info.get("token_start", 0)
        token_end = slot_info.get("token_end", token_start)
        if token_start is None:
            token_start = 0
        if token_end is None:
            token_end = token_start
        if token_end < token_start:
            token_end = token_start

        # Position depends on disfluency type
        if selected_type in BEFORE_SLOT_TYPES:
            # Insert BEFORE slot (planning/retrieval hesitation)
            token_position = max(0, token_start - 1)
        elif selected_type in AT_SLOT_TYPES:
            # Apply AT slot value (articulation effects)
            token_position = random.randint(token_start, token_end)
        elif selected_type in REPAIR_TYPES:
            # Repair can be at or before slot
            token_position = token_start
        else:
            token_position = token_start

        slot_related = chosen_slot
    else:
        # Random position across all tokens
        token_position = random.randint(0, max(0, num_tokens - 1))
        slot_related = None

        # Check if position is near a slot
        for slot_name, slot_info in slot_positions.items():
            t_start = slot_info.get("token_start", -10)
            t_end = slot_info.get("token_end", -10)
            if t_end is None:
                t_end = t_start
            if t_start is not None and t_start - 2 <= token_position <= t_end + 2:
                slot_related = slot_name
                break

    result = {
        "type": selected_type,
        "token_position": token_position,
    }
    if slot_related:
        result["slot_related"] = slot_related

    return [result]
