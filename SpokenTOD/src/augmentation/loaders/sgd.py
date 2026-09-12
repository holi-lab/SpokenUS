"""SGD dataset loader with user goal from slot_values."""

import json
from pathlib import Path
from typing import Iterator

from augmentation.schema import Dialogue, Goal, SlotSpan, StructuredGoal
from .base import BaseLoader


class SGDLoader(BaseLoader):
    """Load SGD dialogues with goals reconstructed from slot_values."""

    # SGD uses "dev" instead of "valid"
    SPLIT_MAP = {"valid": "dev", "validation": "dev"}

    def __init__(self, data_dir: Path, split: str = "train"):
        super().__init__(data_dir, split)
        self._count_cache: int | None = None

    def _get_split_dir(self) -> Path:
        """Get the directory for the current split, mapping standard names to SGD names."""
        split_name = self.SPLIT_MAP.get(self.split, self.split)
        return self.data_dir / split_name

    @property
    def name(self) -> str:
        return "sgd"

    def __len__(self) -> int:
        if self._count_cache is not None:
            return self._count_cache

        split_dir = self._get_split_dir()
        count = 0
        for dlg_path in split_dir.glob("dialogues_*.json"):
            with open(dlg_path, "r") as f:
                dialogues = json.load(f)
                count += len(dialogues)

        self._count_cache = count
        return count

    def load(self) -> Iterator[Dialogue]:
        """Yield SGD dialogues."""
        split_dir = self._get_split_dir()

        # SGD has multiple dialogue files per split
        for dlg_path in sorted(split_dir.glob("dialogues_*.json")):
            with open(dlg_path, "r") as f:
                dialogues = json.load(f)

            for dlg in dialogues:
                dlg_id = dlg["dialogue_id"]
                services = dlg.get("services", [])
                turns_data = dlg.get("turns", [])

                # Extract turns with cumulative state
                turns = []
                cumulative_state = {}  # {service: {slot: value}}
                
                for turn in turns_data:
                    role = "user" if turn["speaker"] == "USER" else "assistant"
                    dialog_act = turn["frames"]
                    text = turn["utterance"]
                    slots = self._extract_slots(turn)
                    
                    # Update cumulative state only on user turns
                    if role == "user":
                        cumulative_state = self._update_cumulative_state(
                            cumulative_state, turn.get("frames", [])
                        )
                    
                    turns.append({
                        "role": role,
                        "text": text,
                        "slots": slots,
                        "dialog_act": dialog_act,
                        "state": dict(cumulative_state),  # Copy current state
                    })

                # Extract goal from last USER turn's slot_values
                goal = self._extract_goal(turns_data, services)

                # Final state is the last cumulative state
                state = cumulative_state

                yield Dialogue(
                    id=dlg_id,
                    source=self.name,
                    turns=turns,
                    goal=goal,
                    state=state,
                    metadata={"services": services},
                )

    def _extract_goal(self, turns: list[dict], services: list[str]) -> Goal:
        """Extract user goal from accumulated slot_values with flow preservation."""
        # Track state to identify new information per turn
        current_slots = {service: {} for service in services}
        current_slot_values = {service: {} for service in services}
        current_intents = {service: None for service in services}
        current_requested = {service: [] for service in services}

        steps = []

        for turn in turns:
            if turn["speaker"] != "USER":
                continue

            frames = turn.get("frames", [])
            for frame in frames:
                service = frame["service"]
                state = frame.get("state", {})

                # Check for intent change
                active_intent = state.get("active_intent", "NONE")
                intent_changed = False
                if active_intent != "NONE" and active_intent != current_intents.get(service):
                    intent_changed = True
                    current_intents[service] = active_intent

                # Check for new slots
                slot_values = state.get("slot_values", {})
                new_slots = {}
                new_slot_values = {}
                for key, values in slot_values.items():
                    if not values:
                        continue
                    raw_values = [str(value) for value in values if value not in (None, "")]
                    if not raw_values:
                        continue
                    val = raw_values[-1]
                    previous_val = current_slots.get(service, {}).get(key)
                    previous_values = current_slot_values.get(service, {}).get(key, [])
                    if previous_val != val or previous_values != raw_values:
                        new_slots[key] = val
                        new_slot_values[key] = raw_values
                        if service not in current_slots:
                            current_slots[service] = {}
                        if service not in current_slot_values:
                            current_slot_values[service] = {}
                        current_slots[service][key] = val
                        current_slot_values[service][key] = raw_values

                # Check for new requested slots
                new_requested = []
                for rs in state.get("requested_slots", []):
                    existing = current_requested.setdefault(service, [])
                    if rs not in existing:
                        existing.append(rs)
                        new_requested.append(rs)

                # Record step if there's new info
                if intent_changed or new_slots or new_slot_values or new_requested:
                    steps.append({
                        "service": service,
                        "intent": active_intent,
                        "new_slots": new_slots,
                        "new_slot_values": new_slot_values,
                        "new_requested": new_requested,
                        "is_intent_change": intent_changed
                    })

        # Build text goal with flow
        text = self._build_goal_text(steps)

        # Build structured goal (accumulated final state)
        structured = self._build_structured_goal(
            current_slots, current_intents, services, current_slot_values, current_requested
        )

        return Goal(text=text, structured=structured)

    def _build_goal_text(
        self,
        steps: list[dict] | dict,
        active_intents: dict | None = None,
        services: list[str] | None = None,
    ) -> str:
        """Build natural language goal from sequence of steps with varied transitions.

        Backward-compatible: if active_intents/services are provided, `steps` is treated as
        flat slot_values and converted into steps.
        """
        if active_intents is not None or services is not None:
            slot_values = steps if isinstance(steps, dict) else {}
            services = services or []
            slot_values = self._normalize_slot_values(slot_values, services)
            steps = []
            for service in services:
                service_slots = slot_values.get(service, {})
                service_slot_values = {}
                for slot, value in service_slots.items():
                    if isinstance(value, list):
                        service_slot_values[slot] = [str(item) for item in value]
                        final_value = service_slot_values[slot][-1]
                        service_slots[slot] = final_value
                    else:
                        service_slot_values[slot] = [str(value)]
                intent = (active_intents or {}).get(service, "NONE")
                if not service_slots and (intent is None or intent == "NONE"):
                    continue
                steps.append({
                    "service": service,
                    "intent": intent,
                    "new_slots": service_slots,
                    "new_slot_values": service_slot_values,
                    "is_intent_change": intent not in (None, "NONE"),
                })

        if not steps:
            return "Your goal is to have a conversation."

        description_parts = []
        
        # Helper to format slot string
        def format_slot_value(slot_name: str, value: str) -> str:
            clean_k = slot_name.replace("_", " ")
            return f"{clean_k} as {value}"

        def format_slots(slots, slot_values=None):
            if not slots:
                return ""
            items = []
            for k, v in slots.items():
                values = []
                if slot_values:
                    values = [str(value) for value in slot_values.get(k, []) if value not in (None, "")]
                if not values:
                    values = [str(v)]
                rendered = [format_slot_value(k, value) for value in values]
                items.extend(rendered)
            return ", ".join(items)

        # Diverse transitions
        first_transitions = [
            "Your objective is to start by",
            "To begin with,",
            "Initially, you'll want to",
            "Your first task is to"
        ]
        mid_transitions = [
            "Next,",
            "After that,",
            "Once you've done that,",
            "Following this,",
            "Moving on,",
            "Additionally, you should"
        ]
        last_transitions = [
            "Conclude by",
            "Lastly,",
            "The final step is to",
            "Rounding things off,"
        ]

        previous_service = None
        total_steps = len(steps)
        
        for i, step in enumerate(steps):
            service = step["service"]
            intent = step["intent"]
            new_slots = step["new_slots"]
            new_slot_values = step.get("new_slot_values", {})
            new_requested = step.get("new_requested", [])
            is_intent_change = step["is_intent_change"]
            
            # Select transition
            if i == 0:
                t = first_transitions[0] # Keeping it simple for the first one usually works best
            elif i == total_steps - 1 and total_steps > 1:
                t = last_transitions[i % len(last_transitions)]
            else:
                t = mid_transitions[i % len(mid_transitions)]

            # Service name cleanup
            service_name = service.split("_")[0].lower()
            if service_name.endswith("s"):
                service_name = service_name[:-1]

            import re
            clean_intent = re.sub(r'(?<!^)(?=[A-Z])', ' ', intent).lower()
            
            parts = []
            
            # Determine if the transition is a full phrase or just a prefix
            use_raw_action = False
            if "objective" in t or "want to" in t or "task is to" in t or "step is to" in t or "by" in t:
                use_raw_action = True

            if is_intent_change and intent != "NONE":
                if use_raw_action:
                    parts.append(f"{t} {clean_intent} for a {service_name}")
                else:
                    parts.append(f"{t} you want to {clean_intent} for a {service_name}")
                    
                if new_slots:
                    parts.append(f"with {format_slots(new_slots, new_slot_values)}")
            elif new_slots:
                # Just providing info
                if use_raw_action:
                    parts.append(f"{t} providing {format_slots(new_slots, new_slot_values)}")
                else:
                    parts.append(f"{t} provide {format_slots(new_slots, new_slot_values)}")
                    
                if previous_service != service:
                    parts.append(f"for the {service_name}")
            
            if new_requested:
                req_str = ", ".join(rs.replace("_", " ") for rs in new_requested)
                parts.append(f"Make sure to ask for the {req_str}.")

            if parts:
                sentence = " ".join(parts)
                if not sentence.endswith("."):
                    sentence += "."
                # Clean up double spaces or awkward phrasing
                sentence = sentence.replace(" .", ".")
                description_parts.append(sentence)

            previous_service = service

        return " ".join(description_parts)

    def _build_structured_goal(
        self,
        slot_values: dict,
        active_intents: dict,
        services: list[str],
        raw_slot_values: dict | None = None,
        requested_slots: dict | None = None,
    ) -> StructuredGoal:
        """Build structured goal representation."""
        slot_values = self._normalize_slot_values(slot_values, services)
        raw_slot_values = self._normalize_slot_values(raw_slot_values or {}, services)
        domains = []
        intents = []

        for service in services:
            domain = service.split("_")[0]
            domains.append(domain)

            # Flatten slots
            flat_slots = {}
            service_slot_values = {}
            if service in slot_values:
                flat_slots = slot_values[service]
            if service in raw_slot_values:
                service_slot_values = raw_slot_values[service]

            service_requests = list(requested_slots.get(service, [])) if requested_slots else []

            intent = active_intents.get(service)
            if flat_slots or (intent and intent != "NONE") or service_requests:
                intent_entry = {
                    "domain": domain,
                    "intent": intent,
                    "slots": flat_slots,
                    "requests": service_requests,
                }
                if service_slot_values:
                    intent_entry["slot_values"] = service_slot_values
                intents.append(intent_entry)

        return StructuredGoal(
            domains=list(set(domains)), 
            intents=intents
        )

    def _normalize_slot_values(self, slot_values: dict, services: list[str]) -> dict:
        """Normalize slot_values to {service: {slot: value}}."""
        if not slot_values:
            return {}
        if any(isinstance(v, dict) for v in slot_values.values()):
            return slot_values

        normalized: dict[str, dict] = {}
        for key, value in slot_values.items():
            if "." in key:
                service, slot = key.split(".", 1)
            else:
                service = services[0] if services else ""
                slot = key
            if not service:
                continue
            normalized.setdefault(service, {})
            normalized[service][slot] = value
        return normalized



    def _update_cumulative_state(self, state: dict, frames: list[dict]) -> dict:
        """Update cumulative state from user turn frames.
        
        Args:
            state: Current cumulative state {service: {slot: value}}
            frames: Frames from the current turn
            
        Returns:
            Updated state dict (new copy)
        """
        new_state = {k: dict(v) for k, v in state.items()}  # Deep copy
        
        for frame in frames:
            service = frame.get("service", "")
            turn_state = frame.get("state", {})
            
            if service not in new_state:
                new_state[service] = {}
            
            for slot, values in turn_state.get("slot_values", {}).items():
                if values:
                    new_state[service][slot] = values[0]
        
        return new_state

    def _extract_slots(self, turn: dict) -> list[SlotSpan]:
        """Extract slot spans from turn frames.

        SGD provides slots with character offsets:
        {'slot': 'city', 'start': 29, 'exclusive_end': 37}
        """
        slots = []
        text = turn.get("utterance", "")

        for frame in turn.get("frames", []):
            for slot_info in frame.get("slots", []):
                slot_name = slot_info.get("slot", "")
                start = slot_info.get("start", 0)
                end = slot_info.get("exclusive_end", 0)
                value = text[start:end]

                slots.append(SlotSpan(
                    slot=slot_name,
                    value=value,
                    start=start,
                    end=end,
                ))

        return slots
