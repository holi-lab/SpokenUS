"""TaskMaster-2 dataset loader with user goal from segments."""

import json
import random
from pathlib import Path
from typing import Iterator

from augmentation.schema import Dialogue, Goal, SlotSpan, StructuredGoal
from augmentation.constants import FILLERS, FP, EDIT, DM
from .base import BaseLoader


# Domain file mapping
TM2_DOMAINS = {
    "restaurants": "restaurant-search.json",
    "food_ordering": "food-ordering.json",
    "hotels": "hotels.json",
    "flights": "flights.json",
    "movies": "movies.json",
    "music": "music.json",
    "sports": "sports.json",
}


class TM2Loader(BaseLoader):
    """Load TaskMaster-2 dialogues with goals from segments."""

    def __init__(self, data_dir: Path, split: str = "train", domains: list[str] | None = None):
        super().__init__(data_dir, split)
        self.domains = domains or list(TM2_DOMAINS.keys())
        self._count_cache: int | None = None

    @property
    def name(self) -> str:
        return "tm2"
        
    def __len__(self) -> int:
        if self._count_cache is not None:
            return self._count_cache
            
        total_count = 0
        for domain in self.domains:
            file_name = TM2_DOMAINS.get(domain)
            if not file_name:
                continue

            data_path = self.data_dir / "data" / file_name
            if not data_path.exists():
                continue

            try:
                with open(data_path, "r") as f:
                    dialogues = json.load(f)
            except json.JSONDecodeError:
                continue

            # Apply partial split logic to get count
            n = len(dialogues)
            if self.split == "train":
                total_count += int(n * 0.8)
            elif self.split in ("valid", "validation", "dev"):
                total_count += int(n * 0.9) - int(n * 0.8)
            else:  # test
                total_count += n - int(n * 0.9)
        
        self._count_cache = total_count
        return total_count

    def load(self) -> Iterator[Dialogue]:
        """Yield TM-2 dialogues."""
        # TM-2 doesn't have split files, we'll create our own split
        for domain in self.domains:
            file_name = TM2_DOMAINS.get(domain)
            if not file_name:
                continue

            data_path = self.data_dir / "data" / file_name
            if not data_path.exists():
                continue

            with open(data_path, "r") as f:
                dialogues = json.load(f)

            # Simple split: 80% train, 10% valid, 10% test
            total = len(dialogues)
            if self.split == "train":
                dialogues = dialogues[:int(total * 0.8)]
            elif self.split in ("valid", "validation", "dev"):
                dialogues = dialogues[int(total * 0.8):int(total * 0.9)]
            else:  # test
                dialogues = dialogues[int(total * 0.9):]

            for dlg in dialogues:
                dlg_id = dlg.get("conversation_id", "")
                instruction_id = dlg.get("instruction_id", "")

                # 1. Early filtering: Check if the dialogue is "perfect" (all info annotated)
                # We skip dialogues that have user turns which likely contain slots but aren't annotated.
                if not self._is_perfect_dialogue(dlg):
                    continue

                # Extract goal from segments
                goal = self._extract_goal(dlg, domain, instruction_id)

                # Extract turns with cumulative state
                turns = []
                cumulative_state = {domain: {}}

                merged_utterances = self._merge_consecutive_turns(
                    dlg.get("utterances", [])
                )

                for utt in merged_utterances:
                    role = "user" if utt.get("speaker") == "USER" else "assistant"
                    text = utt.get("text", "")
                    slots = self._extract_slots(utt)
                    
                    # Update cumulative state only on user turns
                    if role == "user":
                        cumulative_state = self._update_cumulative_state(
                            cumulative_state, slots, domain
                        )
                    
                    turn_dict = {
                        "role": role,
                        "text": text,
                        "slots": slots,
                        "state": {k: dict(v) for k, v in cumulative_state.items()},
                    }
                    
                    # Pass disfluency from merge (for user turns with fillers)
                    merge_disfluency = utt.get("disfluency", [])
                    if role == "user" and merge_disfluency:
                        turn_dict["disfluency"] = merge_disfluency
                    
                    turns.append(turn_dict)

                # Final state is the last cumulative state
                state = cumulative_state

                yield Dialogue(
                    id=dlg_id,
                    source=self.name,
                    turns=turns,
                    goal=goal,
                    state=state,
                    metadata={"domain": domain, "instruction_id": instruction_id},
                )

    def _merge_consecutive_turns(self, utterances: list[dict]) -> list[dict]:
        """Merge consecutive turns from the same speaker into a single turn.
        
        When merging user turns, inserts fillers and tracks them as disfluency
        annotations so inject_disfluency_dialogue can augment rather than duplicate.
        """
        merged = []
        current = None
        fillers = [FP, EDIT, DM]
        # Map tag strings to type codes
        tag_to_type = {FP: "FP", EDIT: "EDIT", DM: "DM"}

        for utt in utterances:
            speaker = utt.get("speaker", "")
            text = utt.get("text", "")
            segments = utt.get("segments", []) or []

            if current and current["speaker"] == speaker:
                if speaker == "USER":
                    filler = random.choice(fillers)
                    filler_word = random.choice(FILLERS[filler])
                    # Keep `text` tag-free; tags are rendered later into `tagged`.
                    join_str = f" {filler_word}, "
                    
                    # Track disfluency annotation for this filler
                    # Position is right after current text + space (where tag starts)
                    filler_position = len(current["text"]) + 1
                    current.setdefault("disfluency", []).append({
                        "type": tag_to_type[filler],
                        "position": filler_position,
                        "tag": filler,
                        "text": f"{filler_word}, ",
                    })
                else:
                    join_str = " "

                offset = len(current["text"]) + len(join_str)
                current["text"] = f"{current['text']}{join_str}{text}"

                for seg in segments:
                    shifted = dict(seg)
                    if shifted.get("start_index") is not None:
                        shifted["start_index"] = shifted["start_index"] + offset
                    if shifted.get("end_index") is not None:
                        shifted["end_index"] = shifted["end_index"] + offset
                    current["segments"].append(shifted)
            else:
                if current:
                    merged.append(current)
                current = {
                    "speaker": speaker,
                    "text": text,
                    "segments": [dict(seg) for seg in segments],
                    "disfluency": [],
                }

        if current:
            merged.append(current)

        return merged

    def _is_perfect_dialogue(self, dlg: dict) -> bool:
        """Strict check to ensure every meaningful user turn has segments.
        
        TaskMaster-2 has many dialogues where information is mentioned but not tagged.
        We filter these out to ensure the User Goal reflects the actual conversation.
        """
        user_utterances = [u for u in dlg.get("utterances", []) if u.get("speaker") == "USER"]
        
        has_any_annotation = False
        
        for utt in user_utterances:
            text = utt.get("text", "").lower().strip()
            segments = utt.get("segments", [])
            
            # Heuristic: if a turn is long and has no segments, it's likely missing info.
            # Example: "I was hoping to get Chinese food tonight." (6 words)
            # Example: "I would like to order takeout, please." (7 words)
            words = text.split()
            
            if segments:
                has_any_annotation = True
            elif len(words) > 5:
                # Long turn with no segments. In TM-2, this is almost always a missing annotation.
                # Common fillers or greetings are usually < 5 words.
                # "I would like to help." (5 words)
                # "Can you find a place?" (5 words)
                return False
                
        # Must have at least one valid annotation to be worthwhile
        return has_any_annotation

    def _extract_goal(self, dlg: dict, domain: str, instruction_id: str) -> Goal:
        """Extract user goal from segment annotations with flow preservation."""
        all_slots = {}
        steps = []
        slot_sequence = []
        
        # Track seen slots to identify new ones per turn
        seen_keys = set()
        
        for utt in dlg.get("utterances", []):
            if utt.get("speaker") != "USER":
                continue
                
            current_turn_new_slots = []
            
            for seg in utt.get("segments", []):
                for ann in seg.get("annotations", []):
                    slot_name = ann.get("name", "")
                    slot_value = seg.get("text", "")
                    
                    if slot_name and slot_value:
                        # Normalize key
                        key = f"{slot_name}:{slot_value.lower().strip()}"
                        
                        # Add to global slots
                        all_slots[slot_name] = slot_value
                        
                        # Check if this is new information
                        if key not in seen_keys:
                            slot_info = {"slot": slot_name, "value": slot_value}
                            current_turn_new_slots.append(slot_info)
                            slot_sequence.append(slot_info)
                            seen_keys.add(key)
            
            if current_turn_new_slots:
                steps.append({
                    "turn_slots": current_turn_new_slots
                })

        # Build text goal with flow
        text = self._build_goal_text(domain, instruction_id, steps)

        # Build structured goal
        structured = self._build_structured_goal(
            domain,
            instruction_id,
            all_slots,
            slot_sequence,
        )

        return Goal(text=text, structured=structured)

    def _build_goal_text(
        self,
        domain: str,
        instruction_id: str,
        steps: list[dict] | None = None,
        slots: dict | None = None,
    ) -> str:
        """Build natural language goal from sequence of steps with varied transitions.

        Backward-compatible: if `slots` is provided, it is converted into a single step.
        """
        parts = []

        # Domain intro
        domain_intros = {
            "restaurants": "You want to find a restaurant.",
            "food_ordering": "You want to order food.",
            "hotels": "You are looking for a hotel.",
            "flights": "You need to book a flight.",
            "movies": "You want to find a movie.",
            "music": "You are looking for music.",
            "sports": "You want sports information.",
        }
        intro = domain_intros.get(domain, f"You need help with {domain.replace('_', ' ')}.")
        parts.append(intro)
        
        if steps is None and slots is not None:
            steps = [{"turn_slots": slots}]

        if not steps:
            return " ".join(parts)

        total_steps = len(steps)
        for i, step in enumerate(steps):
            slots = step["turn_slots"]
            if not slots:
                continue

            if isinstance(slots, dict):
                slot_items = list(slots.items())
            else:
                slot_items = [
                    (slot_info["slot"], slot_info["value"])
                    for slot_info in slots
                ]

            phrases = []
            seen_phrases = set()
            for k, v in slot_items:
                phrase = self._format_goal_phrase(domain, k, v)
                if not phrase:
                    continue
                norm_phrase = phrase.lower().strip()
                if norm_phrase in seen_phrases:
                    continue
                seen_phrases.add(norm_phrase)
                phrases.append(phrase)

            if phrases:
                if i == 0:
                    sentence = self._build_first_step_sentence(domain, phrases)
                else:
                    sentence = self._build_followup_step_sentence(
                        phrases,
                        is_last=i == total_steps - 1,
                    )
                parts.append(sentence)

        return " ".join(parts)

    def _build_first_step_sentence(self, domain: str, phrases: list[str]) -> str:
        """Build the first goal step with a domain-aware subject."""
        subject_map = {
            "restaurants": "Start by looking for a restaurant",
            "food_ordering": "Start by ordering",
            "hotels": "Start by looking for a hotel",
            "flights": "Start by looking for a flight",
            "movies": "Start by looking for a movie",
            "music": "Start by looking for music",
            "sports": "Start by asking for sports information",
        }
        prefix = subject_map.get(domain, "Start by looking for")
        return f"{prefix} {' '.join(phrases)}."

    def _build_followup_step_sentence(
        self,
        phrases: list[str],
        is_last: bool = False,
    ) -> str:
        """Build later goal steps with lighter transitions."""
        prefix = "Finally, mention" if is_last else "Then, mention"
        return f"{prefix} {self._join_phrases(phrases)}."

    def _join_phrases(self, phrases: list[str]) -> str:
        """Join phrases into a readable fragment."""
        if not phrases:
            return ""
        if len(phrases) == 1:
            return phrases[0]
        if len(phrases) == 2:
            return f"{phrases[0]} and {phrases[1]}"
        return ", ".join(phrases[:-1]) + f", and {phrases[-1]}"

    def _format_goal_phrase(self, domain: str, slot_name: str, slot_value: str) -> str:
        """Convert a TM2 slot/value pair into a more natural phrase."""
        value = str(slot_value).strip()
        if not value:
            return ""

        slot = slot_name.lower().replace(" ", "")
        value_lower = value.lower()

        if slot.endswith("num.people") or slot.endswith("num.guests") or slot.endswith("num.pax"):
            return self._format_people_phrase(value)
        if slot.endswith("num.tickets"):
            return self._format_quantity_phrase(value, "ticket")
        if slot.endswith("num.rooms"):
            return self._format_quantity_phrase(value, "room")
        if slot.endswith("type.retrieval"):
            if value_lower in {"takeout", "take out"}:
                return "for takeout"
            if value_lower == "delivery":
                return "for delivery"
            return value
        if "destination" in slot:
            return self._prefix_phrase(value, "to")
        if "origin" in slot:
            return self._prefix_phrase(value, "from")
        if "location" in slot or slot.endswith("sub-location"):
            default = "near" if "sub_location" in slot or "sub-location" in slot else "in"
            return self._prefix_phrase(value, default)
        if "date_range" in slot:
            return self._prefix_phrase(value, "from")
        if slot.endswith("check-in_date"):
            return self._prefix_phrase(value, "starting")
        if slot.endswith("check-out_date"):
            return self._prefix_phrase(value, "until")
        if ".date." in slot or slot.endswith("release_date"):
            return self._prefix_phrase(value, "on")
        if ".time." in slot or slot.endswith(".from.time") or slot.endswith("time.pickup"):
            return self._prefix_phrase(value, "at")
        if slot.endswith("time_of_day"):
            return self._prefix_phrase(value, "in")
        if slot.endswith("seating_class"):
            return self._prefix_phrase(value, "in")
        if slot.endswith("seat_location"):
            return self._prefix_phrase(value, "with")
        if slot.endswith("stops"):
            if value_lower in {"non-stop", "nonstop"}:
                return "non-stop"
            return self._prefix_phrase(value, "with")
        if slot.endswith("airline"):
            return self._prefix_phrase(value, "on")
        if slot.endswith("price_range") or slot.endswith("price.ticket") or slot.endswith("total_fare"):
            return self._prefix_phrase(value, "within")
        if slot.endswith("amenity"):
            return self._prefix_phrase(value, "with")
        if slot.endswith("type.bed") or slot.endswith(".num.beds"):
            return self._prefix_phrase(value, "with")
        if slot.endswith("type.room") or slot.endswith("room"):
            return self._prefix_phrase(value, "with")
        if slot.endswith("star_rating"):
            if "star" in value_lower:
                return self._prefix_phrase(value, "rated")
            return f"rated {value} stars"
        if slot.endswith("customer_rating"):
            return f"with a customer rating of {value}"
        if (
            slot.endswith("critic_rating")
            or slot.endswith("audience_rating")
            or slot.endswith("movie_rating")
            or slot.endswith(".rating")
        ):
            return self._prefix_phrase(value, "rated")
        if slot.endswith("type.food"):
            if domain == "restaurants":
                return self._prefix_phrase(value, "serving")
            return value if "food" in value_lower else f"{value} food"
        if slot.endswith("type.meal"):
            return self._prefix_phrase(value, "for")
        if slot.endswith("name.artist"):
            return self._prefix_phrase(value, "by")
        if slot.endswith("streaming_service"):
            return self._prefix_phrase(value, "on")
        if slot.endswith("genre"):
            if domain == "movies" and "movie" not in value_lower:
                article = "an" if value_lower[:1] in "aeiou" else "a"
                return f"{article} {value} movie"
            return value
        if slot.endswith("other_description.item"):
            if value_lower.startswith("a side"):
                return "as a side"
            return value
        if slot.endswith("other_description"):
            if "music" in value_lower or "bar" in value_lower:
                return self._prefix_phrase(value, "with")
            if value_lower in {"formal", "casual", "family friendly", "family-friendly"}:
                return f"that is {value}"
            return value
        if slot.endswith("other_request") or "describes_" in slot or slot.endswith("offical_description"):
            return value
        if slot.endswith("name.item") and value_lower.startswith("of "):
            return value[3:]
        if ".name." in slot or slot.endswith(".name") or slot.endswith("name.restaurant"):
            return value
        if slot.endswith("real_person"):
            return self._prefix_phrase(value, "with")
        return value

    def _format_people_phrase(self, value: str) -> str:
        """Format group-size phrases without duplicating 'people'."""
        value_lower = value.lower()
        if any(token in value_lower for token in ("people", "person", "guest", "guests", "traveler", "travellers")):
            return self._prefix_phrase(value, "for")
        if value_lower in {"1", "one"}:
            return "for one person"
        return f"for {value} people"

    def _format_quantity_phrase(self, value: str, noun: str) -> str:
        """Format quantity phrases while avoiding duplicate nouns."""
        value_lower = value.lower()
        noun_plural = f"{noun}s"
        if noun in value_lower or noun_plural in value_lower:
            return value
        return f"{value} {noun_plural}"

    def _prefix_phrase(self, value: str, prefix: str) -> str:
        """Prefix a phrase unless it already starts with a suitable preposition."""
        value_lower = value.lower()
        if value_lower.startswith(
            (
                "in ",
                "on ",
                "at ",
                "to ",
                "from ",
                "with ",
                "for ",
                "under ",
                "near ",
                "within ",
                "starting ",
                "until ",
                "rated ",
                "by ",
            )
        ):
            return value
        if prefix == "in" and value_lower.startswith("the "):
            return f"in {value}"
        return f"{prefix} {value}"

    def _build_structured_goal(
        self,
        domain: str,
        instruction_id: str,
        slots: dict,
        slot_sequence: list[dict] | None = None,
    ) -> StructuredGoal:
        """Build structured goal."""
        # Infer intent from instruction_id (e.g., "flight-6" -> "flight")
        intent = instruction_id.split("-")[0] if instruction_id else domain
        slot_sequence = slot_sequence or [
            {"slot": slot_name, "value": slot_value}
            for slot_name, slot_value in slots.items()
        ]
        slot_values = {}
        seen_values = {}
        for slot_info in slot_sequence:
            slot_name = slot_info["slot"]
            slot_value = slot_info["value"]
            norm_value = slot_value.lower().strip()
            if slot_name not in slot_values:
                slot_values[slot_name] = []
                seen_values[slot_name] = set()
            if norm_value in seen_values[slot_name]:
                continue
            slot_values[slot_name].append(slot_value)
            seen_values[slot_name].add(norm_value)

        return StructuredGoal(
            domains=[domain],
            intents=[{
                "domain": domain,
                "intent": intent,
                "slots": slots,
                "slot_values": slot_values,
                "slot_sequence": slot_sequence,
                "requests": [],
            }],
        )

    def _update_cumulative_state(
        self, state: dict, slots: list[SlotSpan], domain: str
    ) -> dict:
        """Update cumulative state from extracted slots.
        
        Args:
            state: Current cumulative state {domain: {slot: value}}
            slots: List of SlotSpan from the current turn
            domain: The domain for this dialogue
            
        Returns:
            Updated state dict
        """
        new_state = {k: dict(v) for k, v in state.items()}  # Deep copy
        
        if domain not in new_state:
            new_state[domain] = {}
        
        for slot in slots:
            new_state[domain][slot.slot] = slot.value
        
        return new_state

    def _extract_slots(self, utt: dict) -> list[SlotSpan]:
        """Extract slot spans from utterance segments.

        TM2 provides segments with character offsets:
        {'start_index': 26, 'end_index': 34, 'text': 'sandwich',
         'annotations': [{'name': 'restaurant.type.food'}]}
        """
        slots = []

        for seg in utt.get("segments", []):
            start = seg.get("start_index", 0)
            end = seg.get("end_index", 0)
            value = seg.get("text", "")

            for ann in seg.get("annotations", []):
                slot_name = ann.get("name", "")
                if slot_name:
                    slots.append(SlotSpan(
                        slot=slot_name,
                        value=value,
                        start=start,
                        end=end,
                    ))

        return slots
