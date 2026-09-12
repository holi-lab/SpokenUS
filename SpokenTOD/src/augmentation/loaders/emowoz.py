"""EmoWOZ dataset loader with user goal from MultiWOZ."""

import json
import re
from pathlib import Path
from typing import Iterator

from augmentation.schema import Dialogue, Goal, SlotSpan, StructuredGoal
from .base import BaseLoader


class EmoWOZLoader(BaseLoader):
    """Load EmoWOZ dialogues with emotion labels and goals from MultiWOZ."""

    TASK_PREFIX_RE = re.compile(r"^Task\s+\d+:\s*")

    def __init__(
        self,
        data_dir: Path,
        split: str = "train",
    ):
        super().__init__(data_dir, split)
        self.multiwoz_dir = Path("datasets") / "MultiWOZ_2.1"
        self._multiwoz_cache: dict | None = None
        self._split_ids: set | None = None

    @property
    def name(self) -> str:
        return "emowoz"

    def _load_multiwoz(self) -> dict:
        """Load and cache MultiWOZ data."""
        if self._multiwoz_cache is None:
            mwoz_path = self.multiwoz_dir / "data.json"
            try:
                with open(mwoz_path, "r") as f:
                    self._multiwoz_cache = json.load(f)
            except json.JSONDecodeError:
                with open(mwoz_path, "r") as f:
                    raw = f.read()
                repaired = re.sub(r'("goal"\s*:\s*)\{SNG\d+', r"\1{", raw)
                if repaired == raw:
                    raise
                self._multiwoz_cache = json.loads(repaired)
        return self._multiwoz_cache

    def _load_data(self) -> dict:
        """Load and cache EmoWOZ data."""
        emowoz_path = self.data_dir / "emowoz-multiwoz.json"
        if not emowoz_path.exists():
            return {}
            
        with open(emowoz_path, "r") as f:
            return json.load(f)

    def _get_split_ids(self) -> list[str]:
        """Get dialogue IDs for current split."""
        if self._split_ids is None:
            data = self._load_data()
            all_ids = sorted(data.keys())
            total = len(all_ids)
            
            if self.split == "train":
                self._split_ids = all_ids[:int(total * 0.8)]
            elif self.split in ("valid", "validation", "dev"):
                self._split_ids = all_ids[int(total * 0.8):int(total * 0.9)]
            else:  # test
                self._split_ids = all_ids[int(total * 0.9):]
                
        return self._split_ids

    def __len__(self) -> int:
        return len(self._get_split_ids())

    def load(self) -> Iterator[Dialogue]:
        """Yield EmoWOZ dialogues."""
        data = self._load_data()
        split_ids = self._get_split_ids()

        if not data:
            return

        multiwoz = self._load_multiwoz()

        for dlg_id in split_ids:
            dlg_data = data[dlg_id]
            mwoz_data = multiwoz.get(dlg_id, {})
            mwoz_log = mwoz_data.get("log", [])
            normalized_id = self._normalize_dialogue_id(dlg_id)

            # Get goal from MultiWOZ
            goal = self._extract_goal(dlg_id, multiwoz)

            # Extract turns with emotion labels and state from MultiWOZ metadata
            turns = []
            emotion_labels = []
            cumulative_state = {}  # Fallback if metadata missing
            emowoz_log = dlg_data.get("log", [])
            
            # Ensure alignment
            limit = min(len(emowoz_log), len(mwoz_log))
            
            for i in range(limit):
                turn_data = emowoz_log[i]
                mwoz_turn = mwoz_log[i]
                
                role = "user" if i % 2 == 0 else "assistant"
                text = turn_data.get("text", "")
                dialogue_act = turn_data.get("dialog_act", {})
                slots = self._extract_slots(turn_data)
                
                # Get state from MultiWOZ metadata
                metadata = mwoz_turn.get("metadata", {})
                if metadata:
                    state = self._extract_belief_state(metadata)
                    # Update fallback cumulative state for consistency
                    cumulative_state = state
                else:
                    # Fallback to cumulative state from slots
                    if role == "user" and slots:
                        cumulative_state = self._update_cumulative_state(
                            cumulative_state, slots
                        )
                    state = cumulative_state
                
                # Emotion extraction
                emotion_raw = turn_data.get("emotion", [])
                if role == "assistant" or not emotion_raw:
                    emotion = -1
                else:
                    emotion = self._extract_emotion_value(emotion_raw)

                turns.append({
                    "role": role,
                    "text": text,
                    "slots": slots,
                    "dialog_act": dialogue_act,
                    "state": {k: dict(v) for k, v in state.items()},  # Deep copy
                })
                emotion_labels.append(emotion)

            # Final state from cumulative

            yield Dialogue(
                id=normalized_id,
                source=self.name,
                turns=turns,
                goal=goal,
                state=cumulative_state,
                emotion_labels=emotion_labels,
            )

    def _extract_emotion_value(self, emotion_raw: list) -> int:
        """Extract emotion value from EmoWOZ annotation list.
        
        Format: [{'annotator': '...', 'annotation': 0}, ..., {'emotion': 0, 'sentiment': 0}]
        The last item typically has the final 'emotion' value.
        """
        if not emotion_raw:
            return 0
        
        # Try to find 'emotion' key in last item
        last_item = emotion_raw[-1]
        if isinstance(last_item, dict):
            if 'emotion' in last_item:
                return int(last_item['emotion'])
            if 'annotation' in last_item:
                return int(last_item['annotation'])
        
        # If first item has annotation
        first_item = emotion_raw[0]
        if isinstance(first_item, dict) and 'annotation' in first_item:
            return int(first_item['annotation'])
        
        return 0

    def _extract_goal(self, dlg_id: str, multiwoz: dict) -> Goal:
        """Extract user goal from MultiWOZ by dialogue ID."""
        mwoz_data = multiwoz.get(dlg_id, {})
        goal_data = mwoz_data.get("goal", {})
        messages = goal_data.get("message", [])

        # Build natural language goal text
        text = self._messages_to_text(messages)

        # Build structured goal
        structured = self._goal_to_structured(goal_data)

        if not text:
            text = self._structured_goal_to_text(structured)

        return Goal(text=text, structured=structured)

    def _messages_to_text(self, messages: list[str]) -> str:
        """Convert MultiWOZ messages to natural language goal."""
        messages = self._normalize_goal_messages(messages)
        if isinstance(messages, str):
            messages = [messages]

        # Clean HTML spans and ensure proper sentence endings
        clean_msgs = []
        for msg in messages:
            clean = self._normalize_goal_message(msg)
            if clean:
                clean_msgs.append(clean)
        return " ".join(clean_msgs)

    def _normalize_goal_messages(self, messages: str | list[str]) -> str | list[str]:
        """Normalize raw MultiWOZ goal messages while preserving container type."""
        if isinstance(messages, str):
            return self._normalize_goal_message(messages)
        return [self._normalize_goal_message(msg) for msg in messages]

    def _normalize_goal_message(self, msg: str) -> str:
        """Strip markup and synthetic task IDs from MultiWOZ goal text."""
        clean = re.sub(r"<[^>]+>", "", msg).strip()
        clean = self.TASK_PREFIX_RE.sub("", clean).strip()
        if clean and clean[-1] not in ".!?":
            clean += "."
        return clean

    def _normalize_dialogue_id(self, dlg_id: str) -> str:
        """Remove file suffixes from dialogue IDs for stable downstream keys."""
        if dlg_id.endswith(".json"):
            return dlg_id[:-5]
        return dlg_id

    def _goal_to_structured(self, goal_data: dict) -> StructuredGoal:
        """Convert MultiWOZ goal to structured format."""
        domains = []
        intents = []
        metadata = {}

        if goal_data.get("message"):
            metadata["message"] = self._normalize_goal_messages(goal_data["message"])
        if goal_data.get("topic"):
            metadata["topic"] = dict(goal_data["topic"])

        for domain in ["restaurant", "hotel", "attraction", "taxi", "train", "hospital", "police"]:
            if domain not in goal_data or not goal_data[domain]:
                continue

            domains.append(domain)
            d_goal = goal_data[domain]

            intent = {
                "domain": domain,
                "intent": self._infer_intent(d_goal),
                "slots": {},
                "requests": [],
            }

            # Info slots (constraints)
            if "info" in d_goal:
                intent["slots"].update(d_goal["info"])

            # Book slots
            if "book" in d_goal:
                intent["slots"].update({f"book_{k}": v for k, v in d_goal["book"].items()})

            # Requested slots
            if "reqt" in d_goal:
                intent["requests"] = d_goal["reqt"]

            extra_goal = {
                key: value
                for key, value in d_goal.items()
                if key not in {"info", "book", "reqt"} and value
            }
            if extra_goal:
                intent["metadata"] = extra_goal

            intents.append(intent)

        return StructuredGoal(domains=domains, intents=intents, metadata=metadata)

    def _structured_goal_to_text(self, structured: StructuredGoal) -> str:
        """Convert structured goals into a simple natural language summary."""
        if not structured.intents:
            return ""

        sentences = []
        for intent in structured.intents:
            domain = intent.get("domain", "general")
            action = intent.get("intent", "inform").replace("_", " ")
            slots = intent.get("slots", {})

            slot_parts = []
            for key, value in slots.items():
                slot_parts.append(f"{key}={value}")
            slot_text = ", ".join(slot_parts) if slot_parts else "no constraints"

            sentences.append(f"For {domain}, user wants to {action} with {slot_text}.")

        return " ".join(sentences)

    def _infer_intent(self, domain_goal: dict) -> str:
        """Infer intent from domain goal."""
        has_book = "book" in domain_goal and domain_goal["book"]
        has_info = "info" in domain_goal and domain_goal["info"]

        if has_book and has_info:
            return "find_and_book"
        elif has_book:
            return "book"
        elif has_info:
            return "find"
        else:
            return "inform"

    def _extract_slots(self, turn_data: dict) -> list[SlotSpan]:
        """Extract slot spans from turn span_info.

        EmoWOZ span_info format: [act, slot, value, start_word_idx, end_word_idx]
        e.g., ['Restaurant-Inform', 'Food', 'european', 8, 8]

        We find the value in text to get character offsets.
        """
        slots = []
        text = turn_data.get("text", "")
        text_lower = text.lower()
        span_info = turn_data.get("span_info", [])

        for span in span_info:
            if len(span) < 5:
                continue

            act, slot_name, value, _, _ = span
            if not value:
                continue

            # Extract domain.slot format from act (e.g., "Restaurant-Inform" -> "restaurant")
            domain = act.split("-")[0].lower() if "-" in act else ""
            full_slot = f"{domain}.{slot_name}".lower() if domain else slot_name.lower()

            # Find value in text (case-insensitive)
            value_lower = value.lower()
            start = text_lower.find(value_lower)

            if start != -1:
                # Use actual text case for value
                end = start + len(value)
                actual_value = text[start:end]
                slots.append(SlotSpan(
                    slot=full_slot,
                    value=actual_value,
                    start=start,
                    end=end,
                ))

        return slots

    def _extract_belief_state(self, metadata: dict) -> dict:
        """Extract belief state from MultiWOZ turn metadata.
        
        MultiWOZ metadata structure:
        {
            "restaurant": {
                "semi": {"food": "indian", "pricerange": "expensive", "name": "...", "area": "..."},
                "book": {"booked": [], "people": "6", "day": "saturday", "time": "19:30"}
            },
            "hotel": {...},
            ...
        }
        
        Returns:
            Dict of {domain: {slot: value}} with non-empty values only.
        """
        state = {}
        
        domains = ["restaurant", "hotel", "attraction", "taxi", "train", "hospital", "police"]
        
        for domain in domains:
            if domain not in metadata or not metadata[domain]:
                continue
            
            domain_meta = metadata[domain]
            domain_state = {}
            
            # Extract from "semi" (constraints/informable slots)
            semi = domain_meta.get("semi", {})
            for slot, value in semi.items():
                if value and value != "not mentioned" and value != "none":
                    domain_state[slot] = value
            
            # Extract from "book" (booking slots)
            book = domain_meta.get("book", {})
            for slot, value in book.items():
                if slot == "booked":
                    # "booked" is a list of booking confirmations
                    if value and isinstance(value, list) and len(value) > 0:
                        # Extract reference number from first booking
                        first_booking = value[0]
                        if "reference" in first_booking:
                            domain_state["book_reference"] = first_booking["reference"]
                        if "name" in first_booking:
                            domain_state["book_name"] = first_booking["name"]
                elif value and value != "not mentioned":
                    domain_state[f"book_{slot}"] = value
            
            if domain_state:
                state[domain] = domain_state
        
        return state

    def _update_cumulative_state(self, state: dict, slots: list) -> dict:
        """Update cumulative state from extracted slots.
        
        Args:
            state: Current cumulative state {domain: {slot: value}}
            slots: List of SlotSpan from the current turn
            
        Returns:
            Updated state dict
        """
        new_state = {k: dict(v) for k, v in state.items()}  # Deep copy
        
        for slot in slots:
            # Parse domain from slot name (e.g., "restaurant.food" -> "restaurant")
            slot_name = slot.slot if hasattr(slot, 'slot') else slot.get('slot', '')
            slot_value = slot.value if hasattr(slot, 'value') else slot.get('value', '')
            
            if "." in slot_name:
                domain, name = slot_name.split(".", 1)
            else:
                domain = "general"
                name = slot_name
            
            if domain not in new_state:
                new_state[domain] = {}
            new_state[domain][name] = slot_value
        
        return new_state
