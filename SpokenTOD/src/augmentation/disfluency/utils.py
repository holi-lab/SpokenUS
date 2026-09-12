"""Utility functions for tokenization and text manipulation."""

from pathlib import Path
import re


def tokenize_with_positions(text: str) -> tuple[list[str], list[tuple[int, int]]]:
    """
    Whitespace tokenization preserving character positions.

    Args:
        text: Text to tokenize

    Returns:
        tokens: List of token strings
        positions: List of (start_char, end_char) tuples
    """
    tokens: list[str] = []
    positions: list[tuple[int, int]] = []

    for match in re.finditer(r"\S+", text):
        tokens.append(match.group())
        positions.append((match.start(), match.end()))

    return tokens, positions


def char_to_token_positions(
    slot_positions: dict,
    token_char_positions: list[tuple[int, int]],
) -> dict:
    """Map slot character offsets to token offsets."""
    for _, slot_info in slot_positions.items():
        char_start = slot_info.get("start")
        char_end = slot_info.get("end")

        if char_start is None or char_end is None:
            continue

        token_start = None
        token_end = None

        for idx, (t_start, t_end) in enumerate(token_char_positions):
            if t_start <= char_start < t_end:
                token_start = idx
            if t_start < char_end <= t_end:
                token_end = idx

        slot_info["token_start"] = token_start
        slot_info["token_end"] = token_end if token_end is not None else token_start

    return slot_positions


def get_char_position_for_token(
    token_index: int,
    token_char_positions: list[tuple[int, int]],
) -> int:
    """
    Get character position for a token index.

    Args:
        token_index: Index of token
        token_char_positions: List of (start, end) char positions

    Returns:
        Character position (start of token)
    """
    if token_index < 0:
        return 0
    if token_index >= len(token_char_positions):
        if token_char_positions:
            return token_char_positions[-1][0]
        return 0
    return token_char_positions[token_index][0]


def find_word_boundary_before(text: str, position: int) -> int:
    """
    Find the start of a word at or before the given position.

    Args:
        text: The text
        position: Character position

    Returns:
        Character position of word start
    """
    if position <= 0:
        return 0

    # Go backwards to find word boundary
    i = min(position, len(text) - 1)
    while i > 0 and text[i - 1] not in " \t\n":
        i -= 1
    return i


def clean_utterance(text: str) -> str:
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def load_prompt(name: str) -> str:
    """Load a prompt template from src/prompts directory.
    
    Args:
        name: Filename of the prompt (e.g., "correction.txt")
    
    Returns:
        Prompt text content
    """
    prompt_dir = Path(__file__).parent / "prompts"
    prompt_path = prompt_dir / name
    
    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt not found: {prompt_path}")
    
    with open(prompt_path, "r") as f:
        return f.read()
