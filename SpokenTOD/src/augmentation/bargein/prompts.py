"""LLM prompts for barge-in augmentation."""

import json
from typing import Literal

from augmentation.bargein.types import BargeInType, BargeInSubtype


# =============================================================================
# PROMPT TEMPLATES BY BARGE-IN SUBTYPE
# =============================================================================

SUBTYPE_DESCRIPTIONS = {
    # Error Recovery subtypes
    "INCOHERENT_RAW": """The user interrupts the assistant abruptly without explanation.
IMPORTANT: The original assistant's speech must be TRUNCATED mid-sentence, ending with <bargein> tag.
The user says something brief and dismissive like: "No", "That's wrong"
The user does NOT explain what was wrong - just expresses disagreement.
The assistant should apologize and ask what needs to be corrected.""",
    "INCOHERENT_INTERP": """The user interrupts the assistant and explains what was wrong.
IMPORTANT: The original assistant's speech must be TRUNCATED mid-sentence, ending with <bargein> tag.
The user explicitly states the error, like: "No, I said Paris not London", "The destination should be Paris"
The user provides the correction inline.
The assistant should apologize and confirm the corrected information.""",
    # Clarification subtypes
    "FAIL_RAW": """The user didn't understand the assistant and asks for clarification briefly.
IMPORTANT: The original assistant's speech must be TRUNCATED mid-sentence, ending with <bargein> tag.
The user says something like: "Sorry?", "What?", "Come again?", "I didn't catch that"
The user does NOT specify what part was unclear - just signals general confusion.
The assistant should rephrase or repeat their previous message more clearly.""",
    "FAIL_INTERP": """The user asks for clarification about a specific part.
IMPORTANT: The original assistant's speech must be TRUNCATED mid-sentence, ending with <bargein> tag.
The user asks about a specific term or detail, like: "What does PNR mean?", "Which date was that?", "Can you repeat the reference number?"
The user identifies exactly what they didn't understand.
The assistant should explain or clarify just that specific part.""",
    # Efficiency subtypes
    "REF_IMPL": """The user signals understanding with minimal acknowledgment, cutting off the assistant.
IMPORTANT: The original assistant's speech must be TRUNCATED mid-sentence, ending with <bargein> tag.
The user says something brief like: "Uh-huh", "Mm-hmm", "Yeah", "Okay"
This is just a backchannel signal, not a full response.
The assistant should continue briefly or move to the next step.""",
    "REF_RAW": """The user explicitly confirms understanding, cutting off the assistant.
IMPORTANT: The original assistant's speech must be TRUNCATED mid-sentence, ending with <bargein> tag.
The user says something like: "Yes", "I understand", "Got it", "Alright"
This is a clear acknowledgment but no additional information.
The assistant should acknowledge and proceed to the next step.""",
    "REF_INTERP": """The user confirms understanding AND adds relevant information, cutting off the assistant.
IMPORTANT: The original assistant's speech must be TRUNCATED mid-sentence, ending with <bargein> tag.
The user says something like: "Yes, Sunday works for me", "Got it, I prefer the morning flight"
The user shows understanding by adding context or preference.
The assistant should acknowledge the additional information and proceed.""",
}


# Few-shot examples for each subtype
# Note: Examples show TRUNCATED assistant speech before user interruption (ends with <bargein>)
SUBTYPE_EXAMPLES = {
    "INCOHERENT_RAW": [
        {
            "context": [
                {"role": "user", "text": "I want to book a flight to Paris."},
            ],
            "result": [
                {
                    "role": "assistant",
                    "text": "Sure, I'll book a flight to Lon<bargein>",
                },
                {"role": "user", "text": "No, that's wrong."},
                {
                    "role": "assistant",
                    "text": "I apologize. What would you like me to correct?",
                },
                {"role": "user", "text": "I said Paris, not London."},
                {
                    "role": "assistant",
                    "text": "I'm sorry for the confusion. I'll book your flight to Paris instead.",
                },
            ],
            "erroneous_slot": {"destination": "London"},
            "corrected_slot": {"destination": "Paris"},
        }
    ],
    "INCOHERENT_INTERP": [
        {
            "context": [
                {"role": "user", "text": "I need a table for 4 people."},
            ],
            "result": [
                {"role": "assistant", "text": "I've reserved a table for 2<bargein>"},
                {"role": "user", "text": "No, I said 4 people, not 2."},
                {
                    "role": "assistant",
                    "text": "I apologize for the mistake. I'll change the reservation to 4 people.",
                },
            ],
            "erroneous_slot": {"party_size": "2"},
            "corrected_slot": {"party_size": "4"},
        }
    ],
    "FAIL_RAW": [
        {
            "context": [],
            "result": [
                {
                    "role": "assistant",
                    "text": "Your PNR is ABC123 and the flight departs from gate B7 at<bargein>",
                },
                {"role": "user", "text": "Sorry, what was that?"},
                {
                    "role": "assistant",
                    "text": "Let me repeat that. Your booking reference is ABC123. Your flight leaves from gate B7 at 2:35 PM.",
                },
            ],
            "corrected_slot": None,
        }
    ],
    "FAIL_INTERP": [
        {
            "context": [],
            "result": [
                {"role": "assistant", "text": "Your PNR is ABC123 for the<bargein>"},
                {"role": "user", "text": "What's a PNR?"},
                {
                    "role": "assistant",
                    "text": "PNR stands for Passenger Name Record - it's your booking reference number. Yours is ABC123.",
                },
            ],
            "corrected_slot": None,
        }
    ],
    "REF_IMPL": [
        {
            "context": [],
            "result": [
                {
                    "role": "assistant",
                    "text": "So I'll book the 3:00 PM flight on March 15th to<bargein>",
                },
                {"role": "user", "text": "Uh-huh."},
                {"role": "assistant", "text": "Great, I'll proceed with the booking."},
            ],
            "corrected_slot": None,
        }
    ],
    "REF_RAW": [
        {
            "context": [],
            "result": [
                {
                    "role": "assistant",
                    "text": "Your total comes to $250 for the<bargein>",
                },
                {"role": "user", "text": "Got it, that works."},
                {
                    "role": "assistant",
                    "text": "Alright, I'll finalize the booking now.",
                },
            ],
            "corrected_slot": None,
        }
    ],
    "REF_INTERP": [
        {
            "context": [],
            "result": [
                {
                    "role": "assistant",
                    "text": "I found flights available on Saturday and Sun<bargein>",
                },
                {"role": "user", "text": "Yes, Sunday would be better for me."},
                {
                    "role": "assistant",
                    "text": "Understood, I'll book the Sunday flight for you.",
                },
            ],
            "corrected_slot": None,
        }
    ],
}


def build_bargein_prompt(
    bargein_type: BargeInType,
    bargein_subtype: BargeInSubtype,
    context_turns: list[dict],
    current_user_turn: dict,
    next_assistant_turn: dict | None,
    current_state: dict | None = None,
) -> str:
    """Build LLM prompt for barge-in generation.

    Args:
        bargein_type: Main barge-in category
        bargein_subtype: Specific subtype
        context_turns: Previous turns for dialogue context
        current_user_turn: The user turn to potentially modify
        next_assistant_turn: The following assistant turn (if exists)
        current_state: Current dialogue state (for ERROR_RECOVERY types)

    Returns:
        Complete prompt string for LLM
    """
    description = SUBTYPE_DESCRIPTIONS.get(bargein_subtype, "")
    examples = SUBTYPE_EXAMPLES.get(bargein_subtype, [])

    # Format context
    context_str = ""
    for turn in context_turns[-4:]:  # Last 4 turns for context
        role = "User" if turn.get("role") == "user" else "Assistant"
        context_str += f"[{role}]: {turn.get('text', '')}\n"

    # Format current exchange
    current_exchange = f"[User]: {current_user_turn.get('text', '')}\n"
    if next_assistant_turn:
        current_exchange += f"[Assistant]: {next_assistant_turn.get('text', '')}\n"

    # Format following user turn for constraint
    following_constraint = ""

    # Format current dialogue state for ERROR_RECOVERY types
    state_info = ""
    if bargein_subtype in ["INCOHERENT_RAW", "INCOHERENT_INTERP"] and current_state:
        state_info = f"""
## Current Dialogue State (Reference)

The correct dialogue state at this point is:
```json
{json.dumps(current_state, indent=2)}
```

**IMPORTANT for ERROR_RECOVERY:**
- Use this state to identify the CORRECT slot values
- Generate an INCORRECT value for the assistant to mistakenly say (erroneous_slots)
- The corrected_slots MUST match the values in this dialogue state
- This ensures natural flow: the dialogue continues correctly after the barge-in correction
"""

    # Format examples
    examples_str = ""
    for i, ex in enumerate(examples):
        examples_str += f"\nExample {i+1}:\n"
        examples_str += "Context:\n"
        for t in ex.get("context", []):
            role = "User" if t["role"] == "user" else "Assistant"
            examples_str += f"  [{role}]: {t['text']}\n"
        examples_str += "Result (barge-in applied):\n"
        for t in ex.get("result", []):
            role = "User" if t["role"] == "user" else "Assistant"
            examples_str += f"  [{role}]: {t['text']}\n"
        # Only show slot examples for INCOHERENT types
        if bargein_subtype in ["INCOHERENT_RAW", "INCOHERENT_INTERP"]:
            if ex.get("erroneous_slot"):
                examples_str += f"Erroneous slots: {json.dumps(ex['erroneous_slot'])}\n"
            if ex.get("corrected_slot"):
                examples_str += f"Corrected slots: {json.dumps(ex['corrected_slot'])}\n"

    # Determine number of turns based on subtype
    if bargein_subtype == "INCOHERENT_RAW":
        turn_count_instruction = """3. **NUMBER OF TURNS TO GENERATE:**
   - Generate exactly **5 turns** (assistant → user → assistant → user → assistant)
   - User interrupts without explanation, assistant asks for clarification, user explains, assistant resolves
"""
    else:
        turn_count_instruction = """3. **NUMBER OF TURNS TO GENERATE:**
   - Generate exactly **3 turns** (assistant → user → assistant)
   - User interrupts with specific input, assistant responds appropriately
"""

    # Slot extraction instruction only for INCOHERENT types
    if bargein_subtype in ["INCOHERENT_RAW", "INCOHERENT_INTERP"]:
        slot_instruction = """5. For ERROR_RECOVERY types (INCOHERENT_RAW/INCOHERENT_INTERP):
   - Look at the Current Dialogue State to identify the CORRECT slot values
   - Generate an INCORRECT value for the assistant to mistakenly say → put in `erroneous_slots`
   - The corrected_slots MUST use the SAME slot names and values from the Current Dialogue State
   - Slot naming: Use the format from the dialogue state (e.g., "domain.slot" like "flight.destination" or just "destination")
   - Example: State shows {"flight": {"destination": "Paris"}}
     - Assistant incorrectly says "London" → erroneous_slots: {"flight.destination": "London"}
     - User corrects to "Paris" → corrected_slots: {"flight.destination": "Paris"}
6. Keep the dialogue natural and coherent.
7. Maintain all important information from the original exchange."""
    else:
        slot_instruction = """5. Keep the dialogue natural and coherent.
6. Maintain all important information from the original exchange."""

    prompt = f"""You are a dialogue augmentation assistant. Your task is to apply a "barge-in" pattern to a dialogue exchange.
{description}

## Examples
{examples_str}

## Current Dialogue

Previous context:
{context_str}

Current exchange to transform:
{current_exchange}
{state_info}
## Instructions

1. Determine if this barge-in type can be naturally applied to this dialogue exchange.
2. **CRITICAL: Your response should ONLY contain the NEW turns that replace the assistant's response.**
   - **DO NOT include the current User turn** - it will be kept as-is.
   - **START with the truncated Assistant turn** that gets interrupted mid-sentence.
   - End the truncated speech with `<bargein>` tag
   - Example: "I'll book you a flight to Lon<bargein>" (user interrupts before assistant finishes)
{turn_count_instruction}4. If applicable, generate turns following the barge-in pattern as shown in the examples above.
{slot_instruction}
{following_constraint}


## Response Format

Respond in JSON format:
```json
{{
    "applicable": true/false,
    "reason": "Brief explanation if not applicable",
    "turns": [
        {{"role": "assistant", "text": "Truncated assistant speech ending with<bargein>"}},
        {{"role": "user", "text": "User interruption/clarification"}},
        {{"role": "assistant", "text": "Assistant response after interruption"}},
        ...
    ],
    "erroneous_slots": {{"slot_name": "wrong_value"}} or null,
    "corrected_slots": {{"slot_name": "correct_value"}} or null
}}
```

**IMPORTANT**:
- The "turns" array should NOT include the original User turn from the current exchange
- It should start with the truncated Assistant turn
- If not applicable, set "applicable" to false and return empty turns

Generate the barge-in augmented dialogue:"""

    return prompt
