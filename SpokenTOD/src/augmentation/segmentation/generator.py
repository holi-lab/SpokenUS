"""Multi-turn dialogue generator for cross-turn slot speaking."""

import random
from dataclasses import dataclass

from .rules import segment_slot, inject_error_correction
from augmentation.constants import SEGMENTABLE_SLOTS, ERROR_CORRECTION_PROB


@dataclass
class SegmentedTurn:
    """A turn in a segmented slot dialogue."""
    role: str  # "user" or "assistant"
    text: str
    segment: dict | None = None  # {"slot", "value", "idx", "total", "is_correction"}


# Templates for user utterances
USER_FIRST_TEMPLATES = [
    "My {slot_name} is {value}",
    "The {slot_name} is {value}",
    "It's {value}",
    "Sure, it's {value}",
    "Yes, {value}",
    "Okay, {value}",
    "So {value}",
    "The first part is {value}",
    "Starting with {value}",
    "It starts with {value}",
    "Let me give you my {slot_name}. It's {value}",
    "Here's my {slot_name}: {value}",
    "For the {slot_name}, it's {value}",
    "Yeah, so {value}",
    "Right, so it's {value}",
    "Um, {value}",
    "Let's see, {value}",
    "Alright, {value}",
    "That would be {value}",
    "It begins with {value}",
]

USER_CONTINUE_TEMPLATES = [
    "And then {value}",
    "Then {value}",
    "Next is {value}",
    "Followed by {value}",
    "{value}",
    "And {value}",
    "Next, {value}",
    "Then it's {value}",
    "After that, {value}",
    "Continuing, {value}",
    "The next part is {value}",
    "Then comes {value}",
    "Moving on, {value}",
    "Next up, {value}",
    "And the next is {value}",
    "Following that, {value}",
    "Then we have {value}",
    "Okay, then {value}",
    "Alright, {value}",
    "So then {value}",
]

USER_CONFIRM_CONTINUE_TEMPLATES = [
    "Yes. And then {value}",
    "That's right. Then {value}",
    "Correct. Next is {value}",
    "Yeah. {value}",
    "Right. Then {value}",
    "Yep. And {value}",
    "Correct. {value}",
    "Yes, that's right. {value}",
    "Exactly. Then {value}",
    "That's correct. And {value}",
    "Mm-hmm. {value}",
    "Yes, exactly. {value}",
    "You got it. Then {value}",
    "That's it. Next is {value}",
    "Right, right. {value}",
    "Uh-huh. And then {value}",
]

USER_CORRECTION_TEMPLATES = [
    "I'm sorry, {value}",
    "Wait, no. {value}",
    "Actually, {value}",
    "Sorry, I meant {value}",
    "Oh, sorry. {value}",
    "No wait, {value}",
    "Let me correct that. {value}",
    "My mistake, {value}",
    "Oops, {value}",
    "Hold on, {value}",
    "Sorry, that's wrong. {value}",
    "No, no, {value}",
    "I made a mistake. {value}",
    "That's not right. {value}",
    "Hang on, {value}",
    "Let me fix that. {value}",
    "No, it's {value}",
    "Wait, I said that wrong. {value}",
    "Apologies, {value}",
    "Oh, I misspoke. {value}",
]

KEYWORD_CORRECTION_TEMPLATES = [
    t for t in USER_CORRECTION_TEMPLATES
    if any(w in t.lower() for w in ["sorry", "wait", "actually", "meant"])
]

USER_FINAL_TEMPLATES = [
    "And then {value}",
    "And finally {value}",
    "Last part is {value}",
    "And the last is {value}",
    "Finally, {value}",
    "Ending with {value}",
    "The rest is {value}",
    "And {value}",
    "Last one, {value}",
    "{value}. That's it.",
    "The last part is {value}",
    "And to finish, {value}",
    "That ends with {value}",
    "The final part is {value}",
    "Lastly, {value}",
    "To complete it, {value}",
    "And that's {value}",
    "Wrapping up, {value}",
    "{value}. Done.",
    "{value}. That's all.",
]

USER_DOUBLE_TEMPLATES = [
    "And then double {value}",
    "Then double {value}",
    "Double {value}",
    "And double {value}",
    "{value}, double that",
    "Two {value}s",
    "{value} {value}",
    "That's {value} twice",
    "{value}, repeated",
    "Same {value} again",
]

# Templates for assistant confirmation
ASST_CONFIRM_TEMPLATES = [
    "So it's {value}.",
    "{value}.",
    "Got it, {value}.",
    "Okay, {value}.",
    "{value}, got it.",
    "Alright, {value}.",
    "{value}. Go on.",
    "{value}. And then?",
    "Okay. {value}.",
    "Right, {value}.",
    "{value}. Continue.",
    "I have {value}. What's next?",
    "{value}. Please continue.",
    "That's {value}. Next?",
    "{value}. Go ahead.",
    "Okay, I got {value}.",
    "{value}. What comes next?",
    "So far {value}. Continue.",
    "I've got {value}.",
    "{value}, okay.",
    "Mm-hmm, {value}.",
    "{value}. And?",
    "Alright, {value}. Next?",
    "Sure, {value}.",
    "{value}. Keep going.",
]

ASST_CORRECTION_CONFIRM_TEMPLATES = [
    "Yes. Okay, so it's {value}.",
    "Got it. {value}.",
    "Understood. {value}.",
    "No problem. {value}.",
    "Alright, corrected to {value}.",
    "Okay, {value} then.",
    "Right. So {value}.",
    "Sure, {value}.",
    "I've updated that to {value}.",
    "Noted. {value}.",
    "No worries. {value}.",
    "Okay, changing that to {value}.",
    "I'll fix that. {value}.",
    "Let me update that. {value}.",
    "Of course. {value}.",
    "That's fine. {value}.",
    "Alright, so {value}.",
    "Okay, got it. {value}.",
    "Thanks for correcting. {value}.",
    "I see. {value}.",
]

ASST_FINAL_TEMPLATES = [
    "Got it. The full {slot_name} is {full_value}.",
    "Alright. I have {full_value} as your {slot_name}.",
    "Thank you. Your {slot_name} is {full_value}.",
    "Perfect. So the {slot_name} is {full_value}.",
    "Great. I've recorded {full_value} for the {slot_name}.",
    "Okay. That's {full_value} for your {slot_name}.",
    "Got it. {full_value}. Thank you.",
    "Alright, {full_value}. I have that down.",
    "Thank you. I have {full_value}.",
    "Perfect. {full_value} is confirmed.",
    "Noted. Your {slot_name} is {full_value}.",
    "Great, thank you. The {slot_name} is {full_value}.",
    "Wonderful. I have your {slot_name} as {full_value}.",
    "Excellent. So that's {full_value}.",
    "Thank you. I've got {full_value} for the {slot_name}.",
    "Okay, so the complete {slot_name} is {full_value}.",
    "All set. Your {slot_name} is {full_value}.",
    "I have it. {full_value}.",
    "Thanks. The {slot_name} is confirmed as {full_value}.",
    "Got it all. {full_value}.",
    "Perfect, that's {full_value} for your {slot_name}.",
    "Great. {full_value}. I've noted that.",
    "Alright, I've recorded {full_value}.",
    "Thank you for that. {full_value}.",
]


def _naturalize_slot_name(slot_name: str) -> str:
    """Convert slot name to natural spoken form.
    
    Example: "phone_number" -> "phone number"
    """
    return slot_name.replace("_", " ").replace(".", " ")


def _has_double_chars(segment: str) -> tuple[bool, str]:
    """Check if segment ends with double characters.
    
    Example: "9 9 0 3" -> (True, "9 0 3") for "double 9"
    """
    parts = segment.split()
    if len(parts) >= 2 and parts[-2] == parts[-1]:
        return True, " ".join(parts[:-1])
    return False, segment


def generate_crossturn_dialogue(
    slot_name: str,
    slot_value: str,
    slot_type: str,
    error_prob: float = ERROR_CORRECTION_PROB,
) -> list[SegmentedTurn]:
    """Generate SpokenWOZ-style cross-turn dialogue for a slot value.
    
    Args:
        slot_name: Name of the slot (e.g., "phone_number")
        slot_value: Full value of the slot (e.g., "5258576375249903")
        slot_type: Type for segmentation (e.g., "phone", "email")
        error_prob: Probability of error correction insertion
    
    Returns:
        List of SegmentedTurn objects forming the dialogue
    """
    # Segment the value
    segments = segment_slot(slot_value, slot_type)
    
    if len(segments) == 0:
        return []
    
    # Inject error corrections
    segments_with_errors = inject_error_correction(segments, error_prob)
    
    natural_name = _naturalize_slot_name(slot_name)
    turns = []
    total_segments = len([s for s, is_corr in segments_with_errors if not is_corr])
    seg_idx = 0
    
    for i, (segment, is_correction) in enumerate(segments_with_errors):
        is_first = i == 0
        is_last = i == len(segments_with_errors) - 1
        
        # Check for double pattern in final segment
        has_double, modified_seg = _has_double_chars(segment)
        
        # User turn
        if is_correction:
            template_pool = KEYWORD_CORRECTION_TEMPLATES or USER_CORRECTION_TEMPLATES
            template = random.choice(template_pool)
            user_text = template.format(value=segment)
        elif is_first:
            template = random.choice(USER_FIRST_TEMPLATES)
            user_text = template.format(slot_name=natural_name, value=segment)
        elif is_last and has_double:
            template = random.choice(USER_DOUBLE_TEMPLATES)
            user_text = template.format(value=modified_seg)
        elif is_last:
            template = random.choice(USER_FINAL_TEMPLATES)
            user_text = template.format(value=segment)
        else:
            # Check if previous was correction
            prev_was_correction = i > 0 and segments_with_errors[i - 1][1]
            if prev_was_correction:
                template = random.choice(USER_CONFIRM_CONTINUE_TEMPLATES)
            else:
                template = random.choice(USER_CONTINUE_TEMPLATES)
            user_text = template.format(value=segment)
        
        user_turn = SegmentedTurn(
            role="user",
            text=user_text,
            segment={
                "slot": slot_name,
                "value": segment,
                "idx": seg_idx if not is_correction else seg_idx - 1,
                "total": total_segments,
                "is_correction": is_correction,
            },
        )
        turns.append(user_turn)
        
        if not is_correction:
            seg_idx += 1
        
        # Assistant turn
        if is_correction:
            template = random.choice(ASST_CORRECTION_CONFIRM_TEMPLATES)
            asst_text = template.format(value=segment)
        elif is_last:
            template = random.choice(ASST_FINAL_TEMPLATES)
            asst_text = template.format(
                slot_name=natural_name,
                full_value=slot_value,
            )
        else:
            template = random.choice(ASST_CONFIRM_TEMPLATES)
            asst_text = template.format(value=segment)
        
        asst_turn = SegmentedTurn(role="assistant", text=asst_text)
        turns.append(asst_turn)
    
    return turns


def find_segmentable_slots(
    state: dict,
    dataset: str,
) -> list[tuple[str, str, str]]:
    """Find slots that can be segmented in a dialogue state.
    
    Args:
        state: Dialogue state dict {domain: {slot: value}}
        dataset: Dataset name ("sgd", "abcd", "emowoz", "tm2")
    
    Returns:
        List of (slot_name, slot_value, slot_type) tuples
    """
    segmentable = SEGMENTABLE_SLOTS.get(dataset, {})
    result = []
    
    for domain, slots in state.items():
        if not isinstance(slots, dict):
            continue
        for slot_name, slot_value in slots.items():
            if not slot_value or not isinstance(slot_value, str):
                continue
            
            # Check if slot is segmentable
            slot_type = segmentable.get(slot_name)
            if slot_type:
                result.append((slot_name, slot_value, slot_type))
            
            # Also check domain.slot format
            full_name = f"{domain}.{slot_name}"
            slot_type = segmentable.get(full_name)
            if slot_type:
                result.append((full_name, slot_value, slot_type))
    
    return result
