"""Alignment handler for tracking character offsets after insertions."""

from dataclasses import dataclass, field

from augmentation.disfluency.definitions import SlotPositions


@dataclass
class AlignmentHandler:
    """
    Tracks cumulative character offsets from insertions.

    Used for rule-based injections where we can precisely track
    how many characters were inserted at which positions.
    """

    # List of (position, delta) pairs recording each insertion
    offsets: list[tuple[int, int]] = field(default_factory=list)

    def record_insertion(self, position: int, inserted_length: int) -> None:
        """
        Record a new insertion for future offset calculations.

        Args:
            position: Character position where insertion occurred
            inserted_length: Number of characters inserted
        """
        self.offsets.append((position, inserted_length))

    def apply_offset(self, original_pos: int) -> int:
        """
        Calculate new position after all insertions.

        Args:
            original_pos: Original character position

        Returns:
            Adjusted position accounting for all insertions before it
        """
        cumulative_offset = 0

        for insert_pos, delta in self.offsets:
            if insert_pos <= original_pos + cumulative_offset:
                cumulative_offset += delta

        return original_pos + cumulative_offset

    def update_slot_positions(
        self,
        slot_positions: SlotPositions,
    ) -> SlotPositions:
        """
        Recalculate all slot start/end indices after insertions.

        Args:
            slot_positions: Original slot position dictionary

        Returns:
            Updated slot positions with adjusted indices
        """
        updated = {}

        for slot_name, slot_info in slot_positions.items():
            original_start = slot_info.get("start")
            original_end = slot_info.get("end")

            if original_start is not None and original_end is not None:
                updated[slot_name] = {
                    "slot": slot_name,
                    "value": slot_info["value"],
                    "start": self.apply_offset(original_start),
                    "end": self.apply_offset(original_end),
                }
            else:
                # Keep as-is if no position info
                updated[slot_name] = slot_info.copy()

        return updated

    def reset(self) -> None:
        """Clear all recorded offsets."""
        self.offsets.clear()


def compute_insertion_offset(
    text: str,
    position: int,
    insertion: str,
) -> tuple[str, int]:
    """
    Insert text at position and return new text with offset.

    Args:
        text: Original text
        position: Character position to insert at
        insertion: Text to insert

    Returns:
        Tuple of (new_text, inserted_length)
    """
    new_text = text[:position] + insertion + text[position:]
    return new_text, len(insertion)
