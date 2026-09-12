"""Constants for voice dataset augmentation pipeline."""

# Segmentable slots per dataset (slot_name -> slot_type)
SEGMENTABLE_SLOTS = {
    "sgd": {
        "phone_number": "phone",
        "email": "email",
        "account_number": "id_number",
        "confirmation_number": "id_number",
        "flight_number": "complex_id",
        "event_id": "id_number",
        "passenger_name": "user_name",
        "name": "user_name",
    },
    "abcd": {
        "phone_number": "phone",
        "email": "email",
        "account_id": "id_number",
        "order_id": "id_number",
        "membership_number": "id_number",
        "customer_name": "user_name",
        "full_name": "user_name",
        "phone": "phone",
    },
    "emowoz": {
        "phone": "phone",
        "reference": "id_number",
        "trainID": "complex_id",
    },
    "tm2": {
        "reservation.phone": "phone",
        "delivery.phone": "phone",
        "booking.phone": "phone",
        "reservation.name": "user_name",
        "booking.name": "user_name",
        "passenger.name": "user_name",
        "customer.name": "user_name",
        "booking.confirmation": "id_number",
    },
}

# Emotion token mapping used by augmentation and evaluation helpers
EMOTION_TOKENS = {
    0: "[neutral]",
    1: "[sad]",       # fearful/sad
    2: "[angry]",     # dissatisfied
    3: "[sad]",       # apologetic
    4: "[angry]",     # abusive
    5: "[happy]",     # excited
    6: "[happy]",     # satisfied
}

# EmoWOZ emotion label names
EMOTION_LABELS = {
    0: "neutral",
    1: "fearful",
    2: "dissatisfied",
    3: "apologetic",
    4: "abusive",
    5: "excited",
    6: "satisfied",
}

# Datasets configuration
DATASETS = ["emowoz", "sgd", "abcd", "tm2", "spokenwoz"]

# Cross-turn slot excluded datasets (already has native cross-turn)
CROSSTURN_EXCLUDED = {"spokenwoz"}

# Emotion tagging excluded datasets (already has emotion labels)
EMOTION_EXCLUDED = {"emowoz"}

# Disfluency injection excluded datasets (already has native audio)
DISFLUENCY_EXCLUDED = {"spokenwoz"}

# Barge-in injection excluded datasets (already has native audio)
BARGEIN_EXCLUDED = {"spokenwoz"}

# Error correction probability for cross-turn segmentation
ERROR_CORRECTION_PROB = 0.20

# =============================================================================
# BARGE-IN AUGMENTATION CONSTANTS
# =============================================================================

# Barge-in types
BARGEIN_ERROR_RECOVERY = "ERROR_RECOVERY"
BARGEIN_CLARIFICATION = "CLARIFICATION"
BARGEIN_EFFICIENCY = "EFFICIENCY"

BARGEIN_TYPES = [BARGEIN_ERROR_RECOVERY, BARGEIN_CLARIFICATION, BARGEIN_EFFICIENCY]

BARGEIN_SUBTYPES = {
    "ERROR_RECOVERY": ["INCOHERENT_RAW", "INCOHERENT_INTERP"],
    "CLARIFICATION": ["FAIL_RAW", "FAIL_INTERP"],
    "EFFICIENCY": ["REF_IMPL", "REF_RAW", "REF_INTERP"],
}

# Barge-in sampling rate (percentage of user turns to augment)
BARGEIN_SAMPLE_RATE = 0.25  # 25% of user turns

# Error recovery emotions (randomly select one)
ERROR_RECOVERY_EMOTIONS = [2, 4]  # dissatisfied, abusive


# =============================================================================
# DISFLUENCY INJECTION CONSTANTS
# =============================================================================

FP = "[FP]"
DM = "[DM]"
EDIT = "[EDIT]"
RST = "[RST]"
REP = "[REP]"
COR = "[COR]"

# LLM-based types (structural changes requiring slot recovery)
LLM_BASED_TYPES: list[str] = [RST, COR]

# Rule-based types (surface noise with offset tracking)
RULE_BASED_TYPES: list[str] = [FP, DM, EDIT, REP]

ALL_TYPES: list[str] = ["RST", "COR", "FP", "DM", "EDIT", "REP"]

# =============================================================================
# TTS DISFLUENCY TAGS
# - [FP]: Filled Pause (uh, um)
# - [DM]: Discourse Marker (like, you know, well)
# - [EDIT]: Edit cues (I mean, sorry, wait)
# - [REP]: Repetition
# - [RST]: Restart
# - [COR]: Correction
# =============================================================================

DISFLUENCY_TAGS: dict[str, str] = {
    "FP": FP,
    "DM": DM,
    "EDIT": EDIT,
    "REP": REP,
    "RST": RST,
    "COR": COR,
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

# Rule-based filler vocabulary mapping
FILLERS: dict[str, list[str]] = {
    FP: FP_VOCAB,
    DM: DM_VOCAB,
    EDIT: EDIT_VOCAB,
}
