from dataclasses import dataclass


@dataclass
class DisfluencyConfig:
    """Configuration for disfluency injection."""

# Vocabulary for rule-based fillers
FILLERS: dict[str, list[str]] = {
    "filled_pause": ["um", "uh", "ah", "er", "hmm"],
    "discourse_marker": ["like", "you know", "well", "so", "i mean",
                         "right", "listen", "basically", "just"],
    "editing_term": ["I mean", "sorry", "or", "wait", "actually",
                     "let me see", "no wait"],
}

# Constants for injection types
FP = "FP"
DM = "DM"
EDIT = "EDIT"
REP = "REP"
COR = "COR"
RST = "RST"

DISFLUENCY_TAGS = {
    FP: "[FP]",
    DM: "[DM]",
    EDIT: "[EDIT]",
    REP: "[REP]",
    COR: "[COR]",
    RST: "[RST]",
}

FP_VOCAB = [
    "um",
    "uh",
    "ah",
    "er",
    "hmm",
    "huh",
    "eh",
    "oops",
    "oh",
]
DM_VOCAB = [
    "like",
    "you know",
    "well",
    "so",
    "right",
    "listen",
    "basically",
    "just",
    "actually",
    "now",
    "see",
    "sure",
]
EDIT_VOCAB = [
    "I mean",
    "sorry",
    "wait",
    "excuse me",
    "let me see",
    "no wait",
]

# Filler groups (mutually exclusive at same position)
FILLER_TYPES: list[str] = [FP, DM, EDIT]

FILLERS: dict[str, list[str]] = {
    "filled_pause": FP_VOCAB,
    "discourse_marker": DM_VOCAB,
    "editing_term": EDIT_VOCAB,
}

# Injection order
ALL_TYPES: list[str] = [RST, COR, FP, DM, EDIT, REP]
