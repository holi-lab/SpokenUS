"""Rule-based disfluency injectors: FP, DM, EDIT, PRO, REP, SLR.

All injectors are simple rule-based insertions with tag markers.
"""

import random
import re

from augmentation.disfluency.config import DISFLUENCY_TAGS, FILLERS
from augmentation.disfluency.injector.base import InjectionResult


def _choose_filler(category: str) -> str:
    options = FILLERS.get(category, ["um"])
    return random.choice(options)

def _split_trailing_punctuation(text: str) -> tuple[str, str]:
    match = re.match(r"^(.*?)([?!.,;:]+)$", text)
    if match and match.group(1):
        return match.group(1), match.group(2)
    return text, ""


def _inject_filler_with_tag(
    text: str,
    char_position: int,
    filler: str,
    tag: str,
    injection_type: str,
    filler_category: str,
) -> InjectionResult:
    """
    Insert a tagged filler phrase before the specified character position.
    """
    # Include comma for natural pause (no [PAUSE] token needed)
    filler_with_punctuation = f"{filler}, "

    # Build raw text (without tag): "I want to, um, book..."
    raw_text = text[:char_position] + filler_with_punctuation + text[char_position:]

    # Build tagged text: "I want to [FP] um, book..."
    tagged_text = text[:char_position] + f"{tag} {filler_with_punctuation}" + text[char_position:]

    annotation = {
        "type": injection_type,
        "text": filler_with_punctuation,
        "position": char_position,
        "filler_category": filler_category,
    }

    return InjectionResult(
        raw_text=raw_text,
        tagged_text=tagged_text,
        char_offset=len(filler_with_punctuation),
        insert_position=char_position,
        annotation=annotation,
    )


def inject_filled_pause(
    text: str,
    char_position: int,
) -> InjectionResult:
    """
    Insert a filled pause (e.g., um, uh) before the specified character position.
    """
    filler = _choose_filler("filled_pause")
    tag = DISFLUENCY_TAGS["FP"]
    return _inject_filler_with_tag(
        text=text,
        char_position=char_position,
        filler=filler,
        tag=tag,
        injection_type="FP",
        filler_category="filled_pause",
    )


def inject_discourse_marker(
    text: str,
    char_position: int,
) -> InjectionResult:
    """
    Insert a discourse marker (e.g., well, you know) before the specified character position.
    """
    filler = _choose_filler("discourse_marker")
    tag = DISFLUENCY_TAGS["DM"]
    return _inject_filler_with_tag(
        text=text,
        char_position=char_position,
        filler=filler,
        tag=tag,
        injection_type="DM",
        filler_category="discourse_marker",
    )


def inject_editing_term(
    text: str,
    char_position: int,
) -> InjectionResult:
    """
    Insert an explicit editing term (e.g., I mean, sorry) before the specified character position.
    """
    filler = _choose_filler("editing_term")
    tag = DISFLUENCY_TAGS["EDIT"]
    return _inject_filler_with_tag(
        text=text,
        char_position=char_position,
        filler=filler,
        tag=tag,
        injection_type="EDIT",
        filler_category="editing_term",
    )


def inject_prolongation(
    text: str,
    tokens: list[str],
    token_char_positions: list[tuple[int, int]],
    token_index: int,
) -> InjectionResult:
    """
    Mark a word for prolongation without modifying the text itself.
    Tag format: "[PRO] word"
    """
    if token_index >= len(tokens):
        token_index = len(tokens) - 1

    word = tokens[token_index]
    t_start, t_end = token_char_positions[token_index]

    # Raw text stays the same (no modification)
    raw_text = text

    # Tagged text: add [PRO] tag before the word
    tag = DISFLUENCY_TAGS["PRO"]
    tagged_text = text[:t_start] + f"{tag} " + text[t_start:]

    annotation = {
        "type": "PRO",
        "word": word,
        "position": t_start,
    }

    return InjectionResult(
        raw_text=raw_text,
        tagged_text=tagged_text,
        char_offset=0,  # No text modification, only tag insertion
        insert_position=t_start,
        annotation=annotation,
    )


def inject_repetition(
    text: str,
    tokens: list[str],
    token_char_positions: list[tuple[int, int]],
    token_index: int,
    slot_info: dict | None = None,
) -> InjectionResult:
    """
    Repeat word or slot phrase at specified token position.

    If slot_info is provided with token_start/token_end, repeats the entire slot phrase.
    Otherwise, randomly repeats 1-3 words starting from token_index.
    """
    if token_index >= len(tokens):
        token_index = len(tokens) - 1

    # Determine what to repeat: slot phrase or random 1-3 words
    if slot_info and slot_info.get("token_start") is not None:
        slot_start = slot_info["token_start"]
        slot_end = slot_info.get("token_end", slot_start)
        if slot_end is None:
            slot_end = slot_start

        # Get the full slot phrase
        phrase_tokens = tokens[slot_start:slot_end + 1]
        phrase = " ".join(phrase_tokens)

        # Get char positions for the full slot
        t_start = token_char_positions[slot_start][0]
        t_end = token_char_positions[slot_end][1]

        repeated_unit = phrase
    else:
        # Randomly select 1-3 words for repetition
        max_words = min(3, len(tokens) - token_index)
        num_words = random.randint(1, max_words) if max_words > 0 else 1
        
        end_index = token_index + num_words - 1
        phrase_tokens = tokens[token_index:end_index + 1]
        phrase = " ".join(phrase_tokens)
        
        t_start = token_char_positions[token_index][0]
        t_end = token_char_positions[end_index][1]
        repeated_unit = phrase

    phrase_base, trailing_punct = _split_trailing_punctuation(phrase)
    separator = ", "
    if trailing_punct:
        repeated = f"{phrase_base}{separator}{phrase_base}{trailing_punct}"
        repeated_unit = phrase_base
        tag_position = t_start + len(phrase_base) + len(separator)
    else:
        repeated = f"{phrase}{separator}{phrase}"
        tag_position = t_start + len(phrase) + len(separator)
    original_text = text[t_start:t_end]
    char_offset = len(repeated) - len(original_text)

    raw_text = text[:t_start] + repeated + text[t_end:]

    tag = DISFLUENCY_TAGS["REP"]
    if trailing_punct:
        tagged_repeat = f"{phrase_base}, {tag} {phrase_base}{trailing_punct}"
    else:
        tagged_repeat = f"{phrase}, {tag} {phrase}"
    tagged_text = text[:t_start] + tagged_repeat + text[t_end:]

    annotation = {
        "type": "REP",
        "repeated_unit": repeated_unit,
        "position": tag_position,
    }

    return InjectionResult(
        raw_text=raw_text,
        tagged_text=tagged_text,
        char_offset=char_offset,
        insert_position=t_start,
        annotation=annotation,
    )


def inject_slurring(
    text: str,
    tokens: list[str],
    token_char_positions: list[tuple[int, int]],
    token_index: int,
) -> InjectionResult:
    """
    Mark a word for slurring without modifying the text itself.
    Tag format: "[SLR] word"
    """
    if token_index >= len(tokens):
        token_index = len(tokens) - 1

    word = tokens[token_index]
    t_start, t_end = token_char_positions[token_index]

    # Raw text stays the same
    raw_text = text

    # Tagged text: add [SLR] tag before the word
    tag = DISFLUENCY_TAGS["SLR"]
    tagged_text = text[:t_start] + f"{tag} " + text[t_start:]

    annotation = {
        "type": "SLR",
        "word": word,
        "position": t_start,
    }

    return InjectionResult(
        raw_text=raw_text,
        tagged_text=tagged_text,
        char_offset=0,
        insert_position=t_start,
        annotation=annotation,
    )
