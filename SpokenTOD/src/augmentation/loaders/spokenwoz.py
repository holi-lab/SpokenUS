"""SpokenWOZ dataset loader."""

import json
import re
from collections.abc import Iterator
from copy import deepcopy
from pathlib import Path

from augmentation.schema import Dialogue, Goal, StructuredGoal

from .base import BaseLoader


class SpokenWOZLoader(BaseLoader):
    """Load SpokenWOZ dialogues.

    SpokenWOZ already has native audio, so skip_augmentation=True is set
    to signal the pipeline to skip cross-turn, disfluency, and barge-in.
    """

    def __init__(
        self,
        data_dir: Path,
        split: str = "train",
    ):
        super().__init__(data_dir, split)

        # Turn-level audio directory
        self.audio_dir = data_dir / "audio"

        self._data_cache = None

    @property
    def name(self) -> str:
        return "spokenwoz"

    def __len__(self) -> int:
        """Return number of dialogues."""
        if self._data_cache is None:
            self._load_data()
        return len(self._data_cache)

    def _load_data(self) -> None:
        """Load SpokenWOZ data."""
        target_file = "test.json" if self.split == "test" else "train.json"
        data_path = self.data_dir / target_file

        if not data_path.exists():
            self._data_cache = {}
            return

        full_data = self._parse_concatenated_json(data_path)

        # If test, just return
        if self.split == "test":
            self._data_cache = full_data
            return

        # For train/valid, we need to split based on valListFile.json
        val_list_path = self.data_dir / "valListFile.json"
        if not val_list_path.exists():
            # If no validation list, treat all as train
            if self.split == "valid":
                self._data_cache = {}
            else:
                self._data_cache = full_data
            return

        val_ids = set()
        with open(val_list_path) as f:
            for line in f:
                val_ids.add(line.strip())

        filtered_data = {}
        for dlg_id, dlg_data in full_data.items():
            is_val = dlg_id in val_ids

            if self.split == "valid" and is_val:
                filtered_data[dlg_id] = dlg_data
            elif self.split == "train" and not is_val:
                filtered_data[dlg_id] = dlg_data

        self._data_cache = filtered_data

    def _get_turn_audio_path(self, dlg_id: str, turn_idx: int) -> str | None:
        """Get audio file path for a specific turn if it exists.

        Audio files are named: {dialogue_id}_{turn_idx:03d}.wav
        """
        audio_file = self.audio_dir / f"{dlg_id}_{turn_idx:03d}.wav"
        if audio_file.exists():
            return str(audio_file)
        return None

    def load(self) -> Iterator[Dialogue]:
        """Yield SpokenWOZ dialogues."""
        # SpokenWOZ uses train.json / test.json
        # These files have concatenated JSON objects like {id1: {...}}{id2: {...}}
        if self._data_cache is None:
            self._load_data()

        for dlg_id, dlg_data in self._data_cache.items():
            goal = self._extract_goal(dlg_data)

            # Extract turns
            turns = []
            log = dlg_data.get("log", [])
            cumulative_state = {}
            has_audio = False

            for i, turn_data in enumerate(log):
                role = "user" if i % 2 == 0 else "assistant"
                text = turn_data.get("text", "")

                # Update cumulative state with this turn's metadata
                meta = turn_data.get("metadata", {})
                if meta:
                    for domain, domain_state in meta.items():
                        if domain not in cumulative_state:
                            cumulative_state[domain] = {}

                        if "semi" in domain_state:
                            cumulative_state[domain].update(domain_state["semi"])
                        if "book" in domain_state:
                            cumulative_state[domain].update({f"book_{k}": v for k, v in domain_state["book"].items()})

                turn = {
                    "role": role,
                    "text": text,
                    "state": deepcopy(cumulative_state) if cumulative_state else None
                }

                # Add turn-level audio_path
                audio_path = self._get_turn_audio_path(dlg_id, i)
                if audio_path:
                    turn["audio_path"] = audio_path
                    has_audio = True
                turns.append(turn)

            yield Dialogue(
                id=dlg_id,
                source=self.name,
                turns=turns,
                goal=goal,
                state=cumulative_state,
                metadata={
                    "has_native_crossturn": True,
                    "skip_augmentation": True,  # Signal to skip all augmentation steps
                    "has_audio": has_audio,
                },
            )

    def _parse_concatenated_json(self, path: Path) -> dict:
        """Parse SpokenWOZ's concatenated JSON format.

        The format is: {"id1": {...}}{"id2": {...}} with no newlines between.
        """
        with open(path) as f:
            content = f.read()

        # SpokenWOZ format starts with {"dialogue_id": {data}}
        # Split by pattern: }{"MUL or }{SNG
        result = {}

        # Use regex to find all dialogue IDs and extract their content
        # Pattern: "DIALOGUE_ID": { ... entire object ... }
        pattern = r'"(MUL\d+|SNG\d+)"\s*:\s*'

        parts = re.split(pattern, content)
        # parts will be: ['', 'MUL0001', '{...}', 'MUL0002', '{...}', ...]

        i = 1
        while i < len(parts) - 1:
            dlg_id = parts[i]
            json_str = parts[i + 1]

            # Find the matching closing brace for this dialogue
            # Simple approach: take until we see the pattern again or EOF
            try:
                # Parse just the value part (the {...})
                obj = json.loads(json_str.rstrip('}') + '}')
                result[dlg_id] = obj
            except json.JSONDecodeError:
                # Try to extract valid JSON
                brace_count = 0
                end_idx = 0
                for j, c in enumerate(json_str):
                    if c == '{':
                        brace_count += 1
                    elif c == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            end_idx = j + 1
                            break

                if end_idx > 0:
                    try:
                        obj = json.loads(json_str[:end_idx])
                        result[dlg_id] = obj
                    except json.JSONDecodeError:
                        pass

            i += 2

        return result

    def _extract_goal(self, dlg_data: dict) -> Goal:
        """Extract user goal from SpokenWOZ dialogue data."""
        goal_data = dlg_data.get("goal", {})

        text = self._goal_to_natural_language(goal_data)
        structured = self._goal_to_structured(goal_data)

        return Goal(text=text, structured=structured)

    def _goal_to_natural_language(self, goal_data: dict) -> str:
        """Convert SpokenWOZ structured goal to natural language.

        Example input:
        {
            "hotel": {"book": {"day": "saturday", "people": "3", "stay": "5"},
                      "info": {"internet": "yes", "parking": "yes", "stars": "0", "type": "guesthouse"}},
            "restaurant": {"book": {"day": "saturday", "people": "3", "time": "11:30"},
                           "info": {"area": "west", "food": "indian"}},
            "taxi": {"info": {"arriveBy": "11:30"}, "reqt": ["car type"]},
            "profile": {"info": {"name": "kathryn romaine"}}
        }

        Example output:
        "Your goal is to find a guesthouse with free internet and parking.
         First, book it for 3 people for 5 nights starting saturday.
         Next, find an indian restaurant in the west and book a table for 3 people at 11:30 on saturday.
         Finally, arrange a taxi arriving by 11:30. Make sure to ask for the car type."
        """
        sentences = []

        profile = goal_data.get("profile", {})
        profile_info = profile.get("info", {})
        user_name = profile_info.get("name", "")

        profile_parts = self._format_profile_info(profile_info)
        if profile_parts:
            profile_text = ", and ".join(profile_parts) if len(profile_parts) == 2 else profile_parts[0]
            if len(profile_parts) > 2:
                profile_text = ", ".join(profile_parts[:-1]) + ", and " + profile_parts[-1]
            sentences.append(f"For this request, {profile_text}.")

        # Collect active domains
        domain_order = ["hotel", "restaurant", "attraction", "train", "taxi", "hospital", "police"]
        active_domains = [d for d in domain_order if d in goal_data and goal_data[d]]

        for idx, domain in enumerate(active_domains):
            d_goal = goal_data[domain]
            info = d_goal.get("info", {})
            book = d_goal.get("book", {})
            reqt = d_goal.get("reqt", [])

            is_first = idx == 0
            is_last = idx == len(active_domains) - 1

            domain_sentences = self._domain_to_sentences(
                domain, info, book, reqt, user_name, is_first, is_last
            )
            sentences.extend(domain_sentences)

        return " ".join(sentences) if sentences else ""

    def _format_profile_info(self, profile_info: dict) -> list[str]:
        """Render every profile field into the goal text."""
        if not profile_info:
            return []

        field_templates = {
            "name": "your name is {value}",
            "phonenumber": "your phone number is {value}",
            "idnumber": "your ID number is {value}",
            "email": "your email is {value}",
            "platenumber": "your plate number is {value}",
        }

        parts = []
        handled = set()
        for field, template in field_templates.items():
            value = profile_info.get(field)
            if value in (None, "", [], {}):
                continue
            parts.append(template.format(value=value))
            handled.add(field)

        for field, value in profile_info.items():
            if field in handled or value in (None, "", [], {}):
                continue
            label = field.replace("_", " ")
            parts.append(f"your {label} is {value}")

        return parts

    def _domain_to_sentences(
        self,
        domain: str,
        info: dict,
        book: dict,
        reqt: list,
        user_name: str = "",
        is_first: bool = False,
        is_last: bool = False,
    ) -> list[str]:
        """Convert a single domain goal to natural language sentences."""
        sentences = []

        if domain == "hotel":
            sentences.extend(self._hotel_to_sentences(info, book, reqt, user_name, is_first, is_last))
        elif domain == "restaurant":
            sentences.extend(self._restaurant_to_sentences(info, book, reqt, user_name, is_first, is_last))
        elif domain == "attraction":
            sentences.extend(self._attraction_to_sentences(info, reqt, is_first, is_last))
        elif domain == "train":
            sentences.extend(self._train_to_sentences(info, book, reqt, is_first, is_last))
        elif domain == "taxi":
            sentences.extend(self._taxi_to_sentences(info, reqt, is_first, is_last))
        elif domain == "hospital":
            sentences.extend(self._hospital_to_sentences(info, reqt, is_first, is_last))
        elif domain == "police":
            sentences.extend(self._police_to_sentences(reqt, is_first, is_last))

        return sentences

    def _get_transition(self, is_first: bool, is_last: bool) -> str:
        """Get appropriate transition word based on position."""
        if is_first:
            return "Your goal is to"
        elif is_last:
            return "Finally,"
        else:
            return "Next,"

    def _hotel_to_sentences(
        self,
        info: dict,
        book: dict,
        reqt: list,
        user_name: str = "",
        is_first: bool = False,
        is_last: bool = False,
    ) -> list[str]:
        """Generate hotel-related sentences."""
        sentences = []
        transition = self._get_transition(is_first, is_last)

        # Info constraints
        constraints = []
        if info.get("type"):
            constraints.append(f"a {info['type']}")
        else:
            constraints.append("a hotel")

        if info.get("area"):
            constraints.append(f"in the {info['area']}")
        if info.get("stars") and info["stars"] != "0":
            constraints.append(f"with {info['stars']} stars")
        if info.get("pricerange"):
            constraints.append(f"in the {info['pricerange']} price range")
        if info.get("internet") == "yes":
            constraints.append("with free internet")
        if info.get("parking") == "yes":
            constraints.append("with free parking")

        if info.get("name"):
            if is_first:
                sentences.append(f"{transition} find {info['name']}.")
            else:
                sentences.append(f"{transition} find {info['name']}.")
        elif constraints:
            if is_first:
                sentences.append(f"{transition} find {' '.join(constraints)}.")
            else:
                sentences.append(f"{transition} find {' '.join(constraints)}.")

        # Booking
        if book:
            book_parts = []
            if book.get("people"):
                book_parts.append(f"for {book['people']} people")
            if book.get("stay"):
                book_parts.append(f"for {book['stay']} nights")
            if book.get("day"):
                book_parts.append(f"starting {book['day']}")
            if book_parts:
                name_part = f" under the name {user_name}" if user_name else ""
                sentences.append(f"Book it {' '.join(book_parts)}{name_part}.")

        # Requests
        if reqt:
            sentences.append(f"Make sure to ask for the {', '.join(reqt)}.")

        return sentences

    def _restaurant_to_sentences(
        self,
        info: dict,
        book: dict,
        reqt: list,
        user_name: str = "",
        is_first: bool = False,
        is_last: bool = False,
    ) -> list[str]:
        """Generate restaurant-related sentences."""
        sentences = []
        transition = self._get_transition(is_first, is_last)

        # Info constraints
        constraints = []
        if info.get("food"):
            constraints.append(f"a {info['food']} restaurant")
        else:
            constraints.append("a restaurant")

        if info.get("area"):
            constraints.append(f"in the {info['area']}")
        if info.get("pricerange"):
            constraints.append(f"in the {info['pricerange']} price range")

        if info.get("name"):
            if is_first:
                sentences.append(f"{transition} find {info['name']}.")
            else:
                sentences.append(f"{transition} find {info['name']}.")
        elif constraints:
            if is_first:
                sentences.append(f"{transition} find {' '.join(constraints)}.")
            else:
                sentences.append(f"{transition} find {' '.join(constraints)}.")

        # Booking
        if book:
            book_parts = []
            if book.get("people"):
                book_parts.append(f"for {book['people']} people")
            if book.get("time"):
                book_parts.append(f"at {book['time']}")
            if book.get("day"):
                book_parts.append(f"on {book['day']}")
            if book_parts:
                name_part = f" under the name {user_name}" if user_name else ""
                sentences.append(f"Book a table {' '.join(book_parts)}{name_part}.")

        # Requests
        if reqt:
            sentences.append(f"Make sure to ask for the {', '.join(reqt)}.")

        return sentences

    def _attraction_to_sentences(
        self,
        info: dict,
        reqt: list,
        is_first: bool = False,
        is_last: bool = False,
    ) -> list[str]:
        """Generate attraction-related sentences."""
        sentences = []
        transition = self._get_transition(is_first, is_last)

        constraints = []
        if info.get("type"):
            constraints.append(f"a {info['type']}")
        else:
            constraints.append("an attraction")

        if info.get("area"):
            constraints.append(f"in the {info['area']}")

        if info.get("name"):
            if is_first:
                sentences.append(f"{transition} find {info['name']}.")
            else:
                sentences.append(f"{transition} find {info['name']}.")
        elif constraints:
            if is_first:
                sentences.append(f"{transition} find {' '.join(constraints)}.")
            else:
                sentences.append(f"{transition} find {' '.join(constraints)}.")

        if reqt:
            sentences.append(f"Make sure to ask for the {', '.join(reqt)}.")

        return sentences

    def _train_to_sentences(
        self,
        info: dict,
        book: dict,
        reqt: list,
        is_first: bool = False,
        is_last: bool = False,
    ) -> list[str]:
        """Generate train-related sentences."""
        sentences = []
        transition = self._get_transition(is_first, is_last)

        parts = []
        if info.get("departure"):
            parts.append(f"from {info['departure']}")
        if info.get("destination"):
            parts.append(f"to {info['destination']}")
        if info.get("day"):
            parts.append(f"on {info['day']}")
        if info.get("leaveAt"):
            parts.append(f"leaving at {info['leaveAt']}")
        if info.get("arriveBy"):
            parts.append(f"arriving by {info['arriveBy']}")

        if parts:
            if is_first:
                sentences.append(f"{transition} find a train {' '.join(parts)}.")
            else:
                sentences.append(f"{transition} find a train {' '.join(parts)}.")

        if book and book.get("people"):
            sentences.append(f"Book tickets for {book['people']} people.")

        if reqt:
            sentences.append(f"Make sure to ask for the {', '.join(reqt)}.")

        return sentences

    def _taxi_to_sentences(
        self,
        info: dict,
        reqt: list,
        is_first: bool = False,
        is_last: bool = False,
    ) -> list[str]:
        """Generate taxi-related sentences."""
        sentences = []
        transition = self._get_transition(is_first, is_last)

        parts = []
        if info.get("departure"):
            parts.append(f"from {info['departure']}")
        if info.get("destination"):
            parts.append(f"to {info['destination']}")
        if info.get("leaveAt"):
            parts.append(f"leaving at {info['leaveAt']}")
        if info.get("arriveBy"):
            parts.append(f"arriving by {info['arriveBy']}")

        if parts:
            if is_first:
                sentences.append(f"{transition} book a taxi {' '.join(parts)}.")
            else:
                sentences.append(f"{transition} book a taxi {' '.join(parts)}.")
        else:
            if is_first:
                sentences.append(f"{transition} book a taxi.")
            else:
                sentences.append(f"{transition} book a taxi.")

        if reqt:
            sentences.append(f"Make sure to ask for the {', '.join(reqt)}.")

        return sentences

    def _hospital_to_sentences(
        self,
        info: dict,
        reqt: list,
        is_first: bool = False,
        is_last: bool = False,
    ) -> list[str]:
        """Generate hospital-related sentences."""
        sentences = []
        transition = self._get_transition(is_first, is_last)

        if info.get("department"):
            if is_first:
                sentences.append(f"{transition} find the {info['department']} department at the hospital.")
            else:
                sentences.append(f"{transition} find the {info['department']} department at the hospital.")
        else:
            if is_first:
                sentences.append(f"{transition} find a hospital.")
            else:
                sentences.append(f"{transition} find a hospital.")

        if reqt:
            sentences.append(f"Make sure to ask for the {', '.join(reqt)}.")

        return sentences

    def _police_to_sentences(
        self,
        reqt: list,
        is_first: bool = False,
        is_last: bool = False,
    ) -> list[str]:
        """Generate police-related sentences."""
        sentences = []
        transition = self._get_transition(is_first, is_last)

        if is_first:
            sentences.append(f"{transition} find the police station.")
        else:
            sentences.append(f"{transition} find the police station.")

        if reqt:
            sentences.append(f"Make sure to ask for the {', '.join(reqt)}.")

        return sentences

    def _goal_to_structured(self, goal_data: dict) -> StructuredGoal:
        """Convert SpokenWOZ goal to structured format."""
        domains = []
        intents = []
        metadata = {}

        if goal_data.get("profile"):
            metadata["profile"] = goal_data["profile"]

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

            if "info" in d_goal:
                intent["slots"].update(d_goal["info"])
            if "book" in d_goal:
                intent["slots"].update({f"book_{k}": v for k, v in d_goal["book"].items()})
            if "reqt" in d_goal:
                intent["requests"] = d_goal["reqt"]

            intents.append(intent)

        return StructuredGoal(domains=domains, intents=intents, metadata=metadata)

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
