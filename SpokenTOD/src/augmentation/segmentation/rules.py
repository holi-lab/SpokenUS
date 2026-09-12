"""Segmentation rules for cross-turn slot speaking."""

import random
import re
import string
from dataclasses import dataclass


@dataclass
class Segment:
    """A single segment of a slot value."""
    value: str  # The segment value (e.g., "5 2 5 8")
    idx: int    # Segment index
    total: int  # Total number of segments
    is_correction: bool = False  # Whether this is a correction segment


def segment_phone(value: str, chunk_size: int = 4) -> list[str]:
    """Segment phone number into chunks of digits.
    
    Example: "5258576375249903" -> ["5 2 5 8", "5 7 6 3", "7 5 2 4", "9 9 0 3"]
    """
    # Remove non-digits
    digits = re.sub(r"\D", "", value)
    
    chunks = []
    for i in range(0, len(digits), chunk_size):
        chunk = digits[i:i + chunk_size]
        # Space-separate digits for spoken form
        spaced = " ".join(chunk)
        chunks.append(spaced)
    
    return chunks


def segment_email(value: str) -> list[str]:
    """Segment email into parts.
    
    Example: ₩
    """
    if "@" not in value:
        return [value]
    
    local, domain = value.split("@", 1)
    
    # Convert dots to "dot" for spoken form
    local_spoken = local.replace(".", " dot ").strip()
    domain_spoken = domain.replace(".", " dot ").strip()
    
    return [local_spoken, f"at {domain_spoken}"]


def segment_id_number(value: str, chunk_size: int = 4) -> list[str]:
    """Segment ID/confirmation number into chunks.
    
    Example: "5258576375249903" -> ["5 2 5 8", "5 7 6 3", "7 5 2 4", "9 9 0 3"]
    """
    # Remove non-alphanumeric
    chars = re.sub(r"[^a-zA-Z0-9]", "", value)
    
    chunks = []
    for i in range(0, len(chars), chunk_size):
        chunk = chars[i:i + chunk_size]
        # Space-separate for spoken form
        spaced = " ".join(chunk.upper())
        chunks.append(spaced)
    
    return chunks


def segment_user_name(value: str) -> list[str]:
    """Segment user name into first/last name.
    
    Example: "John Smith" -> ["John", "Smith"]
    """
    parts = value.strip().split()
    if len(parts) >= 2:
        return [parts[0], " ".join(parts[1:])]
    return [value]


def segment_complex_id(value: str) -> list[str]:
    """Segment complex ID (mixed letters/numbers).
    
    Example: "TR1234" -> ["T R", "1 2 3 4"]
    """
    # Separate letters and numbers
    letters = re.findall(r"[a-zA-Z]+", value)
    numbers = re.findall(r"\d+", value)
    
    result = []
    if letters:
        result.append(" ".join(letters[0].upper()))
    if numbers:
        result.append(" ".join(numbers[0]))
    
    return result if result else [value]


# Segmentation function mapping
SEGMENT_FUNCS = {
    "phone": segment_phone,
    "email": segment_email,
    "id_number": segment_id_number,
    "user_name": segment_user_name,
    "complex_id": segment_complex_id,
}


def segment_slot(value: str, slot_type: str) -> list[str]:
    """Segment a slot value based on its type.
    
    Args:
        value: The slot value to segment
        slot_type: One of "phone", "email", "id_number", "user_name", "complex_id"
    
    Returns:
        List of spoken segments
    """
    func = SEGMENT_FUNCS.get(slot_type)
    if func:
        return func(value)
    return [value]


def inject_error_correction(
    segments: list[str],
    error_prob: float = 0.20,
) -> list[tuple[str, bool]]:
    """Inject error correction into segments.
    
    With error_prob probability, insert an incorrect value followed by correction.
    
    Args:
        segments: List of segments
        error_prob: Probability of inserting error correction (default 20%)
    
    Returns:
        List of (segment, is_correction) tuples
    """
    if len(segments) < 2:
        return [(s, False) for s in segments]
    
    result = []
    # Choose one random segment (not first/last) to inject error
    error_idx = random.randint(1, len(segments) - 1) if len(segments) > 2 else 1
    
    for i, seg in enumerate(segments):
        if i == error_idx and random.random() < error_prob:
            # Generate wrong value (swap last two chars or digits)
            wrong = _generate_wrong_value(seg)
            result.append((wrong, False))
            result.append((seg, True))  # Correction
        else:
            result.append((seg, False))
    
    return result


def _generate_wrong_value(segment: str) -> str:
    """Generate a wrong value by modifying the segment."""
    parts = segment.split()
    if len(parts) >= 2:
        # Swap last two parts
        swapped = parts[:]
        swapped[-2], swapped[-1] = swapped[-1], swapped[-2]
        wrong = " ".join(swapped)
        if wrong != segment:
            return wrong
    if not parts:
        return segment + " wrong"
    # Fallback: mutate one part to ensure the wrong value differs
    idx = random.randrange(len(parts))
    parts[idx] = _mutate_token(parts[idx])
    mutated = " ".join(parts)
    if mutated == segment:
        # Last-resort tweak if mutation was ineffective
        parts[idx] = parts[idx] + "x"
        mutated = " ".join(parts)
    return mutated


def _mutate_token(token: str) -> str:
    """Return a slightly altered token to simulate an error."""
    if token.isdigit():
        if len(token) == 1:
            options = [d for d in string.digits if d != token]
            return random.choice(options)
        last = token[-1]
        options = [d for d in string.digits if d != last]
        return token[:-1] + random.choice(options)
    if token.isalpha():
        letters = string.ascii_uppercase if token.isupper() else string.ascii_lowercase
        if len(token) == 1:
            options = [c for c in letters if c != token]
            return random.choice(options)
        last = token[-1]
        options = [c for c in letters if c != last]
        return token[:-1] + random.choice(options)
    return token + "x"
