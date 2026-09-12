"""Barge-in injection logic for dialogue augmentation."""

import random
from copy import deepcopy

from loguru import logger

from augmentation.bargein.prompts import build_bargein_prompt
from augmentation.bargein.sampler import (
    is_eligible_for_bargein,
    sample_bargein_turns,
    select_bargein_type,
)
from augmentation.bargein.types import (
    BargeInLLMResponse,
    BargeInRequest,
    BargeInResult,
    BargeInSubtype,
    BargeInType,
)
from augmentation.batch.client import BatchClient, LLMUsageTracker
from augmentation.constants import (
    BARGEIN_SAMPLE_RATE,
    EMOTION_LABELS,
    ERROR_RECOVERY_EMOTIONS,
)


def _get_emotion_for_type(bargein_type: BargeInType) -> dict | None:
    """Get emotion override for barge-in type.

    Error Recovery gets dissatisfied(2) or abusive(4).
    Other types get neutral(0).

    Args:
        bargein_type: The barge-in type

    Returns:
        Emotion dict {label, name} or None
    """
    if bargein_type == "ERROR_RECOVERY":
        label = random.choice(ERROR_RECOVERY_EMOTIONS)
        return {"label": label, "name": EMOTION_LABELS.get(label, "neutral")}
    else:
        return {"label": 0, "name": "neutral"}


def _apply_state_for_bargein(
    bargein_type: BargeInType,
    bargein_subtype: BargeInSubtype,
    original_turns: list[dict],
    modified_turns: list[dict],
    target_turn_idx: int,
    erroneous_slots: dict | None = None,
    corrected_slots: dict | None = None,
) -> list[dict]:
    """Apply appropriate dialogue state to barge-in modified turns.

    State handling by type:
    - ERROR_RECOVERY (G_INCOHERENT): First assistant turn gets erroneous_slots, last gets corrected_slots
    - ERROR_RECOVERY (other): Apply corrected_slots to all turns
    - CLARIFICATION/EFFICIENCY: Preserve previous state

    Args:
        bargein_type: Type of barge-in applied
        bargein_subtype: Subtype of barge-in
        original_turns: Original dialogue turns
        modified_turns: LLM-generated modified turns
        target_turn_idx: Index of the original turn that was augmented
        erroneous_slots: For G_INCOHERENT, slots with WRONG values {name: value}
        corrected_slots: For ERROR_RECOVERY, slots to correct {name: value}

    Returns:
        Modified turns with state applied
    """
    # Get the state from the target turn (User turn)
    # If user turn state is empty, use the previous assistant turn's state
    prev_state = deepcopy(original_turns[target_turn_idx].get("state", {}))
    if not prev_state and target_turn_idx > 0:
        # Try previous turn (should be assistant)
        prev_state = deepcopy(original_turns[target_turn_idx - 1].get("state", {}))

    def apply_slots_to_state(state: dict, slots: dict) -> dict:
        """Helper to apply slot updates to state."""
        updated_state = deepcopy(state)
        for slot_name, slot_value in slots.items():
            # Handle domain.slot format
            if "." in slot_name:
                domain, slot = slot_name.split(".", 1)
                if domain not in updated_state:
                    updated_state[domain] = {}
                updated_state[domain][slot] = slot_value
            else:
                # Try to find domain in existing state
                placed = False
                for domain, domain_slots in updated_state.items():
                    if isinstance(domain_slots, dict) and slot_name in domain_slots:
                        domain_slots[slot_name] = slot_value
                        placed = True
                        break
                if not placed:
                    if "general" not in updated_state:
                        updated_state["general"] = {}
                    updated_state["general"][slot_name] = slot_value
        return updated_state

    if bargein_type == "ERROR_RECOVERY":
        # For G_INCOHERENT types: apply erroneous slots first, then corrected slots
        if bargein_subtype in ["INCOHERENT_RAW", "INCOHERENT_INTERP"] and erroneous_slots:
            # Find first and last assistant turns
            first_assistant_idx = None
            last_assistant_idx = None
            for i, turn in enumerate(modified_turns):
                if turn.get("role") == "assistant":
                    if first_assistant_idx is None:
                        first_assistant_idx = i
                    last_assistant_idx = i

            # Apply states to each turn
            for i, turn in enumerate(modified_turns):
                if turn.get("role") == "assistant":
                    if i == first_assistant_idx:
                        # First assistant turn: apply erroneous slots
                        turn["state"] = apply_slots_to_state(prev_state, erroneous_slots)
                    elif i == last_assistant_idx and corrected_slots:
                        # Last assistant turn: apply corrected slots
                        turn["state"] = apply_slots_to_state(prev_state, corrected_slots)
                    else:
                        # Middle turns: preserve previous state
                        turn["state"] = deepcopy(prev_state)
                else:
                    # User turns: preserve previous state
                    turn["state"] = deepcopy(prev_state)
        elif corrected_slots:
            # Other ERROR_RECOVERY types: apply corrected slots to all turns
            corrected_state = apply_slots_to_state(prev_state, corrected_slots)
            for turn in modified_turns:
                turn["state"] = deepcopy(corrected_state)
        else:
            # No slots to apply
            for turn in modified_turns:
                turn["state"] = deepcopy(prev_state)
    else:
        # Clarification/Efficiency: preserve state
        for turn in modified_turns:
            turn["state"] = deepcopy(prev_state)

    return modified_turns


def inject_bargein_dialogue(
    turns: list[dict],
    model: str = "gpt-4.1-mini",
    sample_rate: float = BARGEIN_SAMPLE_RATE,
) -> list[dict]:
    """Inject barge-in into a single dialogue (synchronous).

    This is a simpler synchronous version for single-dialogue processing.
    For batch processing, use inject_bargein_dialogues_batch_async.

    Args:
        turns: List of dialogue turns
        model: LLM model name
        sample_rate: Fraction of user turns to augment

    Returns:
        List of turns with barge-in applied
    """
    client = BatchClient(model=model)

    # Sample turns for barge-in
    sampled_turns = sample_bargein_turns(turns, sample_rate)

    if not sampled_turns:
        return turns

    result_turns = []
    processed_indices = set()

    for i, turn in enumerate(turns):
        # Skip if we've already processed this turn as part of a barge-in
        if i in processed_indices:
            continue

        if i in sampled_turns and is_eligible_for_bargein(turn, i, turns):
            bargein_type, bargein_subtype = select_bargein_type()

            # Get context (previous turns)
            context = turns[max(0, i-4):i]
            next_turn = turns[i + 1] if i + 1 < len(turns) else None

            # Build prompt and get LLM response
            # Pass current dialogue state for ERROR_RECOVERY types
            current_state = turn.get("state") if bargein_type == "ERROR_RECOVERY" else None
            prompt = build_bargein_prompt(
                bargein_type, bargein_subtype, context, turn, next_turn, current_state
            )

            try:
                # Use structured output for all models (OpenAI and vLLM)
                response = client.chat_completion(
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=1024,
                    temperature=0.7,
                    response_format=BargeInLLMResponse,
                )
                # Convert Pydantic model to dict
                parsed = {
                    "applicable": response.applicable,
                    "reason": response.reason,
                    "turns": [{"role": t.role, "text": t.text} for t in response.turns],
                    "erroneous_slots": response.erroneous_slots,
                    "corrected_slots": response.corrected_slots,
                }

                if parsed["applicable"] and parsed["turns"]:
                    # Validate: first turn must be assistant (truncated)
                    if not parsed["turns"] or parsed["turns"][0].get("role") != "assistant":
                        # Invalid result, skip and keep original turns
                        logger.warning(
                            f"Barge-in result invalid for turn {i}: "
                            f"First turn role is {parsed['turns'][0].get('role') if parsed['turns'] else 'empty'}, "
                            f"expected 'assistant'. Full result: {parsed['turns'][:2]}"
                        )
                        result_turns.append(turn)
                        if next_turn:
                            result_turns.append(next_turn)
                            processed_indices.add(i + 1)
                    else:
                        # Valid barge-in result
                        # Keep the original user turn
                        result_turns.append(deepcopy(turn))

                        # Apply emotion and state to modified turns
                        modified_turns = parsed["turns"]
                        emotion = _get_emotion_for_type(bargein_type)

                        for mt in modified_turns:
                            mt["bargein"] = {
                                "type": bargein_type,
                                "subtype": bargein_subtype,
                            }
                            if mt.get("role") == "user" and emotion:
                                mt["emotion"] = emotion

                        # Apply state management
                        modified_turns = _apply_state_for_bargein(
                            bargein_type, bargein_subtype, turns, modified_turns, i,
                            parsed.get("erroneous_slots"),
                            parsed.get("corrected_slots")
                        )

                        result_turns.extend(modified_turns)

                        # Mark the next assistant turn as processed (we replaced it)
                        if next_turn:
                            processed_indices.add(i + 1)
                else:
                    # Not applicable, keep original
                    result_turns.append(turn)

            except Exception as e:
                logger.error(f"Barge-in injection failed: {e}")
                result_turns.append(turn)
        else:
            result_turns.append(turn)

    return result_turns


def _validate_bargein_turns(turns: list[dict]) -> bool:
    """Validate barge-in turns structure.

    Checks:
    1. Not empty
    2. First turn is assistant
    3. Roles alternate
    """
    if not turns:
        return False

    if turns[0].get("role") != "assistant":
        return False

    prev_role = None
    for turn in turns:
        role = turn.get("role")
        if role == prev_role:
            return False
        prev_role = role

    return True


def _validate_full_dialogue_alternation(turns: list[dict]) -> bool:
    """Validate that the full dialogue has alternating roles.

    Used to check the result after barge-in insertion.

    Args:
        turns: Full list of dialogue turns

    Returns:
        True if turns alternate properly, False otherwise
    """
    if not turns:
        return True

    prev_role = None
    for i, turn in enumerate(turns):
        role = turn.get("role")
        if role == prev_role:
            logger.warning(
                f"Validation failed: consecutive {role} turns at positions {i-1} and {i}. "
                f"Turn {i-1}: '{turns[i-1].get('text', '')[:50]}...', "
                f"Turn {i}: '{turn.get('text', '')[:50]}...'"
            )
            return False
        prev_role = role

    return True


async def inject_bargein_dialogues_batch_async(
    dialogues_turns: list[list[dict]],
    model: str = "gpt-4.1-mini",
    sample_rate: float = BARGEIN_SAMPLE_RATE,
    max_concurrency: int = 50,
    usage_tracker: LLMUsageTracker | None = None,
) -> list[list[dict]]:
    """Inject barge-in into multiple dialogues with batch async LLM calls.

    Pipeline:
    1. Sample 25% of user turns from each dialogue
    2. Assign random barge-in type to each sampled turn
    3. Collect all LLM requests and process concurrently
    4. Apply results, handling state and emotion appropriately

    Args:
        dialogues_turns: List of dialogues, each a list of turn dicts
        model: LLM model name
        sample_rate: Fraction of user turns to augment (default: 0.25)
        max_concurrency: Max concurrent LLM requests

    Returns:
        List of dialogues with barge-in applied
    """
    client = BatchClient(
        model=model,
        usage_tracker=usage_tracker,
        request_tag="barge-in",
    )

    # Phase 1: Collect all barge-in requests
    requests: list[BargeInRequest] = []
    request_prompts: list[dict] = []

    for dlg_idx, turns in enumerate(dialogues_turns):
        sampled_turn_idx = sample_bargein_turns(turns, sample_rate)

        for turn_idx in sampled_turn_idx:
            turn = turns[turn_idx]

            if not is_eligible_for_bargein(turn, turn_idx, turns):
                continue

            bargein_type, bargein_subtype = select_bargein_type()
            context = turns[0:turn_idx] # All previous turns as context
            next_turn = turns[turn_idx + 1] if turn_idx + 1 < len(turns) else None

            request = BargeInRequest(
                dialogue_idx=dlg_idx,
                turn_idx=turn_idx,
                bargein_type=bargein_type,
                bargein_subtype=bargein_subtype,
                context=context,
                current_turn=turn,
                next_turn=next_turn,
            )
            requests.append(request)

            # Build prompt for batch processing
            # Pass current dialogue state for ERROR_RECOVERY types
            current_state = turn.get("state") if bargein_type == "ERROR_RECOVERY" else None
            prompt = build_bargein_prompt(
                bargein_type, bargein_subtype, context, turn, next_turn, current_state
            )
            request_prompts.append({
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 1024,
                "temperature": 0.7,
            })

    # Phase 2: Run all LLM requests concurrently with structured output (with retries)
    parsed_responses_map = {}  # Map request index -> parsed response
    pending_indices = list(range(len(requests)))
    max_retries = 10

    for attempt in range(max_retries + 1):
        if not pending_indices:
            break

        # Select prompts for pending requests
        current_prompts = [request_prompts[i] for i in pending_indices]

        if not current_prompts:
            break

        structured_responses = await client.batch_completion_async(
            current_prompts, max_concurrency=max_concurrency,
            response_format=BargeInLLMResponse
        )

        next_pending_indices = []

        for i, response in zip(pending_indices, structured_responses):
            # Convert Pydantic model to dict
            parsed = {
                "applicable": response.applicable,
                "reason": response.reason,
                "turns": [{"role": t.role, "text": t.text} for t in response.turns],
                "erroneous_slots": response.erroneous_slots,
                "corrected_slots": response.corrected_slots,
            }

            # Helper to check validity
            is_valid = True
            if parsed["applicable"]:
                if not _validate_bargein_turns(parsed["turns"]):
                    is_valid = False
                    logger.warning(
                        f"Barge-in validation failed for dialogue {requests[i].dialogue_idx} "
                        f"(attempt {attempt + 1}/{max_retries + 1}): "
                        f"Invalid turn structure/alternation. "
                        f"Turns: {[t.get('role') for t in parsed['turns']]}"
                    )

            if is_valid:
                parsed_responses_map[i] = parsed
            else:
                if attempt < max_retries:
                    next_pending_indices.append(i)
                else:
                    # Final attempt failed
                    logger.warning(
                        f"Barge-in failed after {max_retries + 1} attempts for "
                        f"dialogue {requests[i].dialogue_idx}. Skipping."
                    )
                    # Mark as not applicable
                    parsed["applicable"] = False
                    parsed["turns"] = []
                    parsed_responses_map[i] = parsed

        pending_indices = next_pending_indices

    # Reconstruct list of responses in original order
    parsed_responses = [parsed_responses_map[i] for i in range(len(requests))]

    # Phase 3: Parse responses and organize by dialogue
    # Map: dialogue_idx -> turn_idx -> BargeInResult
    results_map: dict[int, dict[int, BargeInResult]] = {}

    for req, parsed in zip(requests, parsed_responses):
        if parsed["applicable"] and parsed["turns"]:
            # Additional safety check (should be covered by validation above)
            if not _validate_bargein_turns(parsed["turns"]):
                result = BargeInResult(applied=False)
            else:
                emotion = _get_emotion_for_type(req.bargein_type)

                # Mark turns with barge-in metadata and emotion
                for mt in parsed["turns"]:
                    mt["bargein"] = {
                        "type": req.bargein_type,
                        "subtype": req.bargein_subtype,
                    }
                    if mt.get("role") == "user" and emotion:
                        mt["emotion"] = emotion

                result = BargeInResult(
                    applied=True,
                    bargein_type=req.bargein_type,
                    bargein_subtype=req.bargein_subtype,
                    modified_turns=parsed["turns"],
                    emotion_override=emotion,
                    erroneous_slots=parsed.get("erroneous_slots"),
                    corrected_slots=parsed.get("corrected_slots"),
                )
        else:
            result = BargeInResult(applied=False)

        if req.dialogue_idx not in results_map:
            results_map[req.dialogue_idx] = {}
        results_map[req.dialogue_idx][req.turn_idx] = result

    # Phase 4: Apply results to dialogues
    output_dialogues: list[list[dict]] = []

    for dlg_idx, turns in enumerate(dialogues_turns):
        # Check input for consecutive same-role turns (debugging)
        for i in range(len(turns) - 1):
            if turns[i].get("role") == turns[i+1].get("role"):
                logger.warning(
                    f"Barge-in input dialogue {dlg_idx} has consecutive {turns[i].get('role')} turns "
                    f"at indices {i} and {i+1}. Text: {turns[i].get('text')[:30]}..."
                )

        if dlg_idx not in results_map:
            # No barge-in applied to this dialogue
            output_dialogues.append(deepcopy(turns))
            continue

        dlg_results = results_map[dlg_idx]
        new_turns: list[dict] = []
        skip_next = False
        bargein_for_next: BargeInResult | None = None

        for turn_idx, turn in enumerate(turns):
            if skip_next:
                skip_next = False
                # This is the assistant turn we're replacing with barge-in
                if bargein_for_next and bargein_for_next.applied and bargein_for_next.modified_turns:
                    modified = _apply_state_for_bargein(
                        bargein_for_next.bargein_type,
                        bargein_for_next.bargein_subtype,
                        turns,
                        bargein_for_next.modified_turns,
                        turn_idx - 1,  # The user turn index
                        bargein_for_next.erroneous_slots,
                        bargein_for_next.corrected_slots,
                    )
                    # Trace modified turns roles
                    roles = [t.get("role") for t in modified]
                    if roles and roles[0] != "assistant":
                         logger.error(f"Dlg {dlg_idx}: Modified turns start with {roles[0]}! Expected assistant.")

                    new_turns.extend(modified)
                    bargein_for_next = None
                else:
                    new_turns.append(deepcopy(turn))
                continue

            if turn_idx in dlg_results:
                result = dlg_results[turn_idx]

                if result.applied and result.modified_turns:
                    # Keep the original user turn
                    new_turns.append(deepcopy(turn))

                    # Mark to replace the next assistant turn with barge-in result
                    if turn_idx + 1 < len(turns) and turns[turn_idx + 1].get("role") == "assistant":
                        skip_next = True
                        bargein_for_next = result
                    else:
                        # No following assistant turn, just append barge-in result
                        modified = _apply_state_for_bargein(
                            result.bargein_type,
                            result.bargein_subtype,
                            turns,
                            result.modified_turns,
                            turn_idx,
                            result.erroneous_slots,
                            result.corrected_slots,
                        )
                        new_turns.extend(modified)
                else:
                    new_turns.append(deepcopy(turn))
            else:
                new_turns.append(deepcopy(turn))

        # Validate final dialogue alternation
        if not _validate_full_dialogue_alternation(new_turns):
            # Validation failed - fall back to original dialogue without barge-in
            logger.warning(
                f"Dialogue {dlg_idx}: Final validation failed after barge-in insertion. "
                f"Reverting to original dialogue without barge-in."
            )
            output_dialogues.append(deepcopy(turns))
        else:
            output_dialogues.append(new_turns)

    return output_dialogues
