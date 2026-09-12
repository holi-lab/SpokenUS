import re
from typing import Any

from loguru import logger

from augmentation.batch.client import BatchClient, LLMUsageTracker
from augmentation.disfluency.config import DISFLUENCY_TAGS, DisfluencyConfig
from augmentation.disfluency.data import extract_slot_positions
from augmentation.disfluency.engine import (
    AlignmentHandler,
    sample_disfluencies_for_turn,
)
from augmentation.disfluency.injector import (
    build_correction_prompt,
    build_restart_prompt,
    inject_correction,
    inject_discourse_marker,
    inject_editing_term,
    inject_filled_pause,
    inject_repetition,
    inject_restart,
    process_correction_response,
    process_restart_response,
)
from augmentation.disfluency.utils import (
    char_to_token_positions,
    tokenize_with_positions,
)


def _adjust_annotation_positions(
    annotations: list[dict[str, Any]],
    insert_position: int,
    char_offset: int,
) -> None:
    if not char_offset:
        return
    for annotation in annotations:
        position = annotation.get("position")
        if position is not None and position >= insert_position:
            annotation["position"] = position + char_offset


def _find_correction_position(text: str) -> int | None:
    dash_patterns = [
        r"(\w+)([-–—]\s*)(no[,\s]+)",
        r"(\w+)([-–—]\s*)(wait[,\s]+)",
        r"(\w+)([-–—]\s*)(actually[,\s]+)",
        r"(\w+)([-–—]\s*)(I mean[,\s]+)",
    ]
    for pattern in dash_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.end(2)

    ellipsis_match = re.search(r"(\w+)\.\.\.\s*", text)
    if ellipsis_match:
        return ellipsis_match.end()
    return None


def _find_restart_position(text: str) -> int | None:
    restart_patterns = [
        r"(\w+\s+\w+)\s*[-–—]\s+",
        r"(\w+)\s*[-–—]\s+",
        r"([^.]+)\.\.\.\s*",
    ]
    for pattern in restart_patterns:
        match = re.search(pattern, text)
        if match:
            # Insert tag after the reparandum (before the restart phrase).
            return match.end()
    return None


def _find_slurring_position(
    text: str,
    annotation: dict[str, Any],
    slot_positions: dict[str, Any],
) -> int | None:
    slot_name = annotation.get("slot_name")
    if slot_name and slot_name in slot_positions:
        slot_start = slot_positions[slot_name].get("start")
        if slot_start is not None:
            return slot_start
    for key in ("slurred_value", "original_value"):
        value = annotation.get(key)
        if value:
            match = re.search(re.escape(value), text, re.IGNORECASE)
            if match:
                return match.start()
    return None


def _update_llm_positions(
    annotations: list[dict[str, Any]],
    text: str,
    slot_positions: dict[str, Any],
) -> None:
    def _is_middle_position(pos: int, src: str) -> bool:
        if pos <= 0 or pos >= len(src):
            return False
        before = re.search(r"\S", src[:pos])
        after = re.search(r"\S", src[pos:])
        return before is not None and after is not None

    for annotation in annotations:
        ann_type = annotation.get("type")
        if ann_type == "COR":
            position = _find_correction_position(text)
        elif ann_type == "RST":
            position = _find_restart_position(text)
        elif ann_type == "SLR":
            position = _find_slurring_position(text, annotation, slot_positions)
        else:
            continue
        if position is not None and (ann_type not in {"COR", "RST"} or _is_middle_position(position, text)):
            annotation["position"] = position
        else:
            annotation.pop("position", None)


def _realign_filler_annotations(
    raw_text: str,
    annotations: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Ensure FP/DM/EDIT tags align to their inserted filler text."""
    if not annotations:
        return annotations

    def _find_occurrences(text: str, needle: str) -> list[int]:
        if not needle:
            return []
        escaped = re.escape(needle)
        matches = [m.start() for m in re.finditer(escaped, text)]
        if matches:
            return matches
        return [m.start() for m in re.finditer(escaped, text, flags=re.IGNORECASE)]

    aligned: list[dict[str, Any]] = []
    for annotation in annotations:
        ann_type = annotation.get("type")
        if ann_type not in {"FP", "DM", "EDIT"}:
            aligned.append(annotation)
            continue

        filler_text = annotation.get("text")
        if not filler_text:
            aligned.append(annotation)
            continue

        positions = _find_occurrences(raw_text, filler_text)
        if not positions:
            # Filler text no longer exists after rewrites; drop to avoid invalid tag.
            continue

        existing_pos = annotation.get("position")
        if isinstance(existing_pos, int):
            pos = min(positions, key=lambda p: abs(p - existing_pos))
        else:
            pos = positions[0]

        updated = dict(annotation)
        updated["position"] = pos
        aligned.append(updated)

    return aligned


def build_tagged_utterance(
    raw_text: str,
    annotations: list[dict[str, Any]],
) -> str:
    insertions: list[tuple[int, int, str]] = []
    for idx, annotation in enumerate(annotations):
        ann_type = annotation.get("type")
        tag = annotation.get("tag") or DISFLUENCY_TAGS.get(ann_type)
        position = annotation.get("position")
        if tag is None or position is None:
            continue
        if not isinstance(position, int):
            continue
        if position < 0 or position > len(raw_text):
            continue
        insertions.append((position, idx, tag))

    # Sort by position (ascending), then by idx (ascending) to keep first annotation
    insertions.sort(key=lambda item: (item[0], item[1]))

    # Filter out duplicate positions - keep only the first annotation at each position
    seen_positions: set[int] = set()
    unique_insertions: list[tuple[int, int, str]] = []
    for position, idx, tag in insertions:
        if position not in seen_positions:
            seen_positions.add(position)
            unique_insertions.append((position, idx, tag))

    # Reverse for insertion from end to beginning
    unique_insertions.reverse()

    tagged_text = raw_text
    for position, _, tag in unique_insertions:
        tagged_text = tagged_text[:position] + f"{tag} " + tagged_text[position:]

    return tagged_text


def inject_disfluency(
    turn: dict,
    config: DisfluencyConfig,
    client: BatchClient | None = None,
) -> dict[str, Any]:
    """
    Process a single turn to inject disfluencies.
    Returns:
        dict with keys: text, tagged, disfluency
    """
    utterance: str = turn.get("text", "")

    # Get frames for slot extraction
    # SGD: dialog_act is list of frame dicts with "state", "slots" keys
    # EmoWOZ/MultiWOZ: dialog_act is dict like {"Restaurant-Inform": [...]}
    dialog_act = turn.get("dialog_act")
    if isinstance(dialog_act, list):
        frames = dialog_act  # SGD format
    else:
        frames = []  # Non-SGD format, use slots instead

    # Default result
    result = {
        "text": utterance,
        "tagged": utterance,
        "disfluency": [],
    }

    # Step 1: Extract metadata
    # For non-SGD datasets, try to use the slots field directly
    if frames:
        slot_positions_map = extract_slot_positions(utterance, frames)
    else:
        # Build slot positions from turn's slots field (SlotSpan objects)
        slot_positions_map = {}
        for slot in turn.get("slots", []):
            if hasattr(slot, 'slot'):  # SlotSpan object
                slot_positions_map[slot.slot] = {
                    "value": slot.value,
                    "start": slot.start,
                    "end": slot.end,
                }
            elif isinstance(slot, dict):  # Dict format
                slot_positions_map[slot.get("slot", "")] = {
                    "value": slot.get("value", ""),
                    "start": slot.get("start", 0),
                    "end": slot.get("end", 0),
                }

    # Step 2: Tokenize
    tokens, token_char_positions = tokenize_with_positions(utterance)
    slot_positions = char_to_token_positions(slot_positions_map, token_char_positions)

    # Step 3: Sample disfluencies (use word count for probability, token count for positions)
    words = utterance.split()
    sampled = sample_disfluencies_for_turn(words, slot_positions, len(tokens))

    if not sampled:
        return result

    # Initialize trackers
    current_raw_text = utterance
    current_slots = slot_positions.copy()
    alignment_handler = AlignmentHandler()
    annotations: list[dict[str, Any]] = []

    # Process injections
    for disf in sampled:
        disf_type = disf["type"]
        token_pos = disf.get("token_position", 0)

        # Re-tokenize after each modification
        tokens, token_char_positions = tokenize_with_positions(current_raw_text)

        if token_pos < len(token_char_positions):
            char_pos = token_char_positions[token_pos][0]
        else:
            char_pos = len(current_raw_text)

        # ===== PHASE 1: LLM-based (structural changes) =====
        if disf_type == "RST":
            if client:
                try:
                    rst_result, current_slots = inject_restart(
                        utterance=current_raw_text,
                        tokens=tokens,
                        token_position=token_pos,
                        slot_positions=current_slots,
                        client=client,
                    )
                    current_raw_text = rst_result.raw_text
                    annotations.append(rst_result.annotation)
                    _update_llm_positions(annotations, current_raw_text, current_slots)
                    alignment_handler.reset()
                except Exception as e:
                    logger.error(f"RST injection failed: {e}")
                    continue

        elif disf_type == "COR":
            slot_name = disf.get("slot_related")
            if client and slot_name and slot_name in current_slots:
                try:
                    slot_value = current_slots[slot_name].get("value", "")
                    cor_result, current_slots = inject_correction(
                        utterance=current_raw_text,
                        slot_name=slot_name,
                        slot_value=slot_value,
                        slot_positions=current_slots,
                        client=client,
                    )
                    current_raw_text = cor_result.raw_text
                    annotations.append(cor_result.annotation)
                    _update_llm_positions(annotations, current_raw_text, current_slots)
                    alignment_handler.reset()
                except Exception as e:
                    logger.error(f"COR injection failed: {e}")
                    continue



        # ===== PHASE 2: Rule-based (surface changes) =====
        elif disf_type in ("FP", "DM", "EDIT"):
            if disf_type == "DM":
                rb_result = inject_discourse_marker(
                    text=current_raw_text,
                    char_position=char_pos,
                )
            elif disf_type == "EDIT":
                rb_result = inject_editing_term(
                    text=current_raw_text,
                    char_position=char_pos,
                )
            else:
                rb_result = inject_filled_pause(
                    text=current_raw_text,
                    char_position=char_pos,
                )
            _adjust_annotation_positions(
                annotations,
                rb_result.insert_position,
                rb_result.char_offset,
            )
            current_raw_text = rb_result.raw_text
            alignment_handler.record_insertion(char_pos, rb_result.char_offset)
            annotations.append(rb_result.annotation)



        elif disf_type == "REP":
            # Get slot info if this REP is slot-related
            slot_name = disf.get("slot_related")
            slot_info = current_slots.get(slot_name) if slot_name else None

            rep_result = inject_repetition(
                text=current_raw_text,
                tokens=tokens,
                token_char_positions=token_char_positions,
                token_index=token_pos,
                slot_info=slot_info,
            )
            _adjust_annotation_positions(
                annotations,
                rep_result.insert_position,
                rep_result.char_offset,
            )
            current_raw_text = rep_result.raw_text
            current_slots = alignment_handler.update_slot_positions(
                current_slots
            )
            alignment_handler.record_insertion(char_pos, rep_result.char_offset)
            annotations.append(rep_result.annotation)

    annotations = _realign_filler_annotations(current_raw_text, annotations)
    result["text"] = current_raw_text
    result["tagged"] = build_tagged_utterance(current_raw_text, annotations)
    result["disfluency"] = annotations
    return result


def inject_disfluency_dialogue(
    turns: list[dict],
    config: DisfluencyConfig | None = None,
    model: str = "gpt-4.1-mini",
) -> list[dict]:
    """Inject disfluencies into all user turns.

    If a turn already has disfluency annotations (e.g., from turn merging),
    this function augments them rather than overwriting.
    """
    if config is None:
        config = DisfluencyConfig()

    # Create client once per dialogue if needed
    client = BatchClient(model=model)

    result = []

    for turn in turns:
        new_turn = dict(turn)

        if turn.get("role") == "user":
            # Check for existing disfluency from turn merging
            existing_disfluency = turn.get("disfluency", [])
            existing_tagged = turn.get("tagged")

            # Pass full turn dict as it contains frames/meta
            injection = inject_disfluency(turn, config, client)

            if existing_disfluency and existing_tagged:
                # Merge: keep existing + add new annotations
                # But filter out new annotations that are too close to existing ones
                # to prevent consecutive tags like [FP] [PRO]
                existing_positions = {
                    ann.get("position") for ann in existing_disfluency
                    if ann.get("position") is not None
                }

                # Filter new annotations - skip if within 10 chars of existing position
                min_distance = 10
                filtered_new = []
                for new_ann in injection["disfluency"]:
                    new_pos = new_ann.get("position")
                    if new_pos is None:
                        filtered_new.append(new_ann)
                        continue
                    too_close = any(
                        abs(new_pos - existing_pos) < min_distance
                        for existing_pos in existing_positions
                    )
                    if not too_close:
                        filtered_new.append(new_ann)

                merged_disfluency = list(existing_disfluency) + filtered_new
                new_turn["text"] = injection["text"]
                merged_disfluency = _realign_filler_annotations(
                    injection["text"], merged_disfluency
                )
                new_turn["tagged"] = build_tagged_utterance(
                    injection["text"], merged_disfluency
                )
                new_turn["disfluency"] = merged_disfluency
            else:
                # No existing disfluency, apply injection
                injection = inject_disfluency(turn, config, client)
                new_turn["text"] = injection["text"]
                new_turn["tagged"] = injection["tagged"]
                new_turn["disfluency"] = injection["disfluency"]


        result.append(new_turn)

    return result


async def inject_disfluency_dialogues_batch_async(
    dialogues_turns: list[list[dict]],
    config: DisfluencyConfig | None = None,
    model: str = "gpt-4.1-mini",
    max_concurrency: int = 50,
    usage_tracker: LLMUsageTracker | None = None,
) -> list[list[dict]]:
    """Inject disfluencies into multiple dialogues with batch async LLM calls.

    This function collects all COR/RST LLM requests across dialogues,
    processes them concurrently, then applies results.

    Args:
        dialogues_turns: List of dialogues, each a list of turn dicts
        config: Disfluency configuration
        model: LLM model name
        max_concurrency: Max concurrent LLM requests

    Returns:
        List of dialogues with disfluencies injected
    """
    if config is None:
        config = DisfluencyConfig()

    client = BatchClient(
        model=model,
        usage_tracker=usage_tracker,
        request_tag="disfluency",
    )

    # Phase 1: Collect all LLM requests across all dialogues
    # Track: (dialogue_idx, turn_idx, request_dict, metadata)
    llm_requests = []

    # Pre-process all dialogues to identify LLM requests
    preprocessed = []
    for dlg_idx, turns in enumerate(dialogues_turns):
        dlg_data = []
        for turn_idx, turn in enumerate(turns):
            if turn.get("role") != "user":
                dlg_data.append({"turn": turn, "sampled": None, "llm_indices": []})
                continue

            utterance = turn.get("text", "")

            # Extract slot positions
            dialog_act = turn.get("dialog_act")
            if isinstance(dialog_act, list):
                frames = dialog_act
            else:
                frames = []

            if frames:
                slot_positions_map = extract_slot_positions(utterance, frames)
            else:
                slot_positions_map = {}
                for slot in turn.get("slots", []):
                    if hasattr(slot, 'slot'):
                        slot_positions_map[slot.slot] = {
                            "value": slot.value,
                            "start": slot.start,
                            "end": slot.end,
                        }
                    elif isinstance(slot, dict):
                        slot_positions_map[slot.get("slot", "")] = {
                            "value": slot.get("value", ""),
                            "start": slot.get("start", 0),
                            "end": slot.get("end", 0),
                        }

            # Tokenize and sample
            tokens, token_char_positions = tokenize_with_positions(utterance)
            slot_positions = char_to_token_positions(slot_positions_map, token_char_positions)
            words = utterance.split()
            sampled = sample_disfluencies_for_turn(words, slot_positions, len(tokens))

            # Collect LLM requests for this turn
            llm_indices = []
            for disf in sampled:
                disf_type = disf["type"]
                if disf_type == "RST":
                    token_pos = disf.get("token_position", 0)
                    req = build_restart_prompt(utterance, tokens, token_pos)
                    req["_dlg_idx"] = dlg_idx
                    req["_turn_idx"] = turn_idx
                    req["_slot_positions"] = slot_positions
                    llm_indices.append(len(llm_requests))
                    llm_requests.append(req)
                elif disf_type == "COR":
                    slot_name = disf.get("slot_related")
                    if slot_name and slot_name in slot_positions:
                        slot_value = slot_positions[slot_name].get("value", "")
                        req = build_correction_prompt(utterance, slot_name, slot_value)
                        req["_dlg_idx"] = dlg_idx
                        req["_turn_idx"] = turn_idx
                        req["_slot_positions"] = slot_positions
                        req["_slot_name"] = slot_name
                        req["_slot_value"] = slot_value
                        llm_indices.append(len(llm_requests))
                        llm_requests.append(req)

            dlg_data.append({
                "turn": turn,
                "sampled": sampled,
                "llm_indices": llm_indices,
                "slot_positions": slot_positions,
            })
        preprocessed.append(dlg_data)

    # Phase 2: Run all LLM requests concurrently
    llm_responses = []
    if llm_requests:
        llm_responses = await client.batch_completion_async(
            llm_requests, max_concurrency=max_concurrency
        )

    # Phase 3: Apply results
    results = []
    for dlg_idx, dlg_data in enumerate(preprocessed):
        dlg_result = []
        for turn_data in dlg_data:
            turn = turn_data["turn"]
            new_turn = dict(turn)

            if turn.get("role") != "user" or not turn_data["sampled"]:
                # If merge-based disfluency exists but no tags, render tags now.
                existing_disfluency = new_turn.get("disfluency") or []
                if (
                    new_turn.get("role") == "user"
                    and existing_disfluency
                    and not new_turn.get("tagged")
                ):
                    existing_disfluency = _realign_filler_annotations(
                        new_turn.get("text", ""),
                        existing_disfluency,
                    )
                    new_turn["disfluency"] = existing_disfluency
                    new_turn["tagged"] = build_tagged_utterance(
                        new_turn.get("text", ""),
                        existing_disfluency,
                    )
                dlg_result.append(new_turn)
                continue

            # Process this turn's disfluencies
            utterance = turn.get("text", "")
            current_raw_text = utterance
            current_slots = turn_data.get("slot_positions", {}).copy()
            # Start with any existing (merge-based) disfluencies
            existing_disfluency = turn.get("disfluency") or []
            annotations = [dict(ann) for ann in existing_disfluency]
            alignment_handler = AlignmentHandler()

            llm_response_idx = 0
            llm_indices = turn_data["llm_indices"]

            for disf in turn_data["sampled"]:
                disf_type = disf["type"]

                tokens, token_char_positions = tokenize_with_positions(current_raw_text)
                token_pos = disf.get("token_position", 0)

                if token_pos < len(token_char_positions):
                    char_pos = token_char_positions[token_pos][0]
                else:
                    char_pos = len(current_raw_text)

                if disf_type == "RST" and llm_response_idx < len(llm_indices):
                    req_idx = llm_indices[llm_response_idx]
                    response = llm_responses[req_idx] if req_idx < len(llm_responses) else ""
                    llm_response_idx += 1

                    if response:
                        try:
                            rst_result, current_slots = process_restart_response(
                                response, current_raw_text, token_pos, current_slots
                            )
                            current_raw_text = rst_result.raw_text
                            annotations.append(rst_result.annotation)
                            _update_llm_positions(annotations, current_raw_text, current_slots)
                            alignment_handler.reset()
                        except Exception as e:
                            logger.error(f"RST processing failed: {e}")

                elif disf_type == "COR" and llm_response_idx < len(llm_indices):
                    req_idx = llm_indices[llm_response_idx]
                    response = llm_responses[req_idx] if req_idx < len(llm_responses) else ""
                    req = llm_requests[req_idx] if req_idx < len(llm_requests) else {}
                    llm_response_idx += 1

                    if response:
                        try:
                            slot_name = req.get("_slot_name", "")
                            slot_value = req.get("_slot_value", "")
                            cor_result, current_slots = process_correction_response(
                                response, current_raw_text, slot_name, slot_value, current_slots
                            )
                            current_raw_text = cor_result.raw_text
                            annotations.append(cor_result.annotation)
                            _update_llm_positions(annotations, current_raw_text, current_slots)
                            alignment_handler.reset()
                        except Exception as e:
                            logger.error(f"COR processing failed: {e}")

                elif disf_type in ("FP", "DM", "EDIT"):
                    if disf_type == "DM":
                        rb_result = inject_discourse_marker(text=current_raw_text, char_position=char_pos)
                    elif disf_type == "EDIT":
                        rb_result = inject_editing_term(text=current_raw_text, char_position=char_pos)
                    else:
                        rb_result = inject_filled_pause(text=current_raw_text, char_position=char_pos)
                    _adjust_annotation_positions(annotations, rb_result.insert_position, rb_result.char_offset)
                    current_raw_text = rb_result.raw_text
                    annotations.append(rb_result.annotation)

                elif disf_type == "REP":
                    slot_name = disf.get("slot_related")
                    slot_info = current_slots.get(slot_name) if slot_name else None
                    rep_result = inject_repetition(
                        text=current_raw_text, tokens=tokens, token_char_positions=token_char_positions,
                        token_index=token_pos, slot_info=slot_info
                    )
                    _adjust_annotation_positions(annotations, rep_result.insert_position, rep_result.char_offset)
                    current_raw_text = rep_result.raw_text
                    annotations.append(rep_result.annotation)

            annotations = _realign_filler_annotations(current_raw_text, annotations)
            new_turn["text"] = current_raw_text
            new_turn["tagged"] = build_tagged_utterance(current_raw_text, annotations)
            new_turn["disfluency"] = annotations
            dlg_result.append(new_turn)

        results.append(dlg_result)

    return results
