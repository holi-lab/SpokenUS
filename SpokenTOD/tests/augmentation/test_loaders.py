"""Unit tests for dataset loaders."""

import json
import pytest
from pathlib import Path
from unittest.mock import patch

from augmentation.loaders.emowoz import EmoWOZLoader
from augmentation.loaders.spokenwoz import SpokenWOZLoader
from augmentation.loaders.sgd import SGDLoader
from augmentation.loaders.abcd import ABCDLoader
from augmentation.loaders.tm2 import TM2Loader
from augmentation.constants import FP, FILLERS


SAMPLE_EMOWOZ = {
    "MUL0001.json": {
        "log": {
            "text": ["I'm looking for a restaurant", "Sure, what type of food?"],
            "emotion": [0, -1],
        }
    }
}

SAMPLE_SGD = [
    {
        "dialogue_id": "1_00000",
        "services": ["Restaurants_1"],
        "turns": [
            {
                "frames": [
                    {
                        "actions": [
                            {
                                "act": "INFORM",
                                "canonical_values": ["San Jose"],
                                "slot": "city",
                                "values": ["San Jose"],
                            }
                        ],
                        "service": "Restaurants_1",
                        "slots": [{"exclusive_end": 37, "slot": "city", "start": 29}],
                        "state": {
                            "active_intent": "FindRestaurants",
                            "requested_slots": [],
                            "slot_values": {"city": ["San Jose"]},
                        },
                    }
                ],
                "speaker": "USER",
                "utterance": "I would like for it to be in San Jose.",
            },
            {
                "frames": [
                    {
                        "actions": [
                            {
                                "act": "REQUEST",
                                "canonical_values": ["Mexican", "Italian"],
                                "slot": "cuisine",
                                "values": ["Mexican", "Italian"],
                            }
                        ],
                        "service": "Restaurants_1",
                        "slots": [
                            {"exclusive_end": 59, "slot": "cuisine", "start": 52},
                            {"exclusive_end": 68, "slot": "cuisine", "start": 61},
                        ],
                    }
                ],
                "speaker": "SYSTEM",
                "utterance": "Is there a specific cuisine type you enjoy, such as Mexican, Italian or something else?",
            },
        ],
    }
]

SAMPLE_ABCD = {
    "train": [
        {
            "convo_id": "3592",
            "scenario": {
                "personal": {
                    "customer_name": "John Smith",
                    "email": "john@example.com",
                    "phone_number": "555-1234",
                },
                "flow": "order_issue",
                "subflow": "missing_item",
                "order": {"order_id": "ORD123", "product_names": ["Blue Jacket"]},
                "product": {},
            },
            "delexed": [
                {"speaker": "agent", "text": "hi!", "turn_count": 1},
                {"speaker": "agent", "text": "how can i help you?", "turn_count": 2},
                {
                    "speaker": "customer",
                    "text": "My order is missing an item",
                    "turn_count": 3,
                },
                {"speaker": "agent", "text": "I can help with that", "turn_count": 4},
            ],
        }
    ]
}

SAMPLE_TM2 = [
    {
        "conversation_id": "dlg-00100680-00e0-40fe-8321-6d81b21bfc4f",
        "instruction_id": "flight-6",
        "utterances": [
            {
                "index": 0,
                "speaker": "USER",
                "text": "Hello. I'd like to find a round trip commercial airline flight from San Francisco to Denver.",
                "segments": [
                    {
                        "start_index": 26,
                        "end_index": 36,
                        "text": "round trip",
                        "annotations": [{"name": "flight_search.type"}],
                    },
                    {
                        "start_index": 68,
                        "end_index": 81,
                        "text": "San Francisco",
                        "annotations": [{"name": "flight_search.origin"}],
                    },
                    {
                        "start_index": 85,
                        "end_index": 91,
                        "text": "Denver",
                        "annotations": [{"name": "flight_search.destination1"}],
                    },
                ],
            },
            {"index": 1, "speaker": "ASSISTANT", "text": "Hello, how can I help you?"},
        ],
    }
]


class TestEmoWOZLoader:
    """Tests for EmoWOZ loader."""

    def test_messages_to_text(self):
        """Verify HTML span removal from messages."""
        loader = EmoWOZLoader(Path("."), Path("."))
        messages = [
            "Find a <span class='emphasis'>restaurant</span>",
            "in the <span>centre</span>",
        ]
        result = loader._messages_to_text(messages)
        assert "restaurant" in result
        assert "centre" in result
        assert "<span" not in result

    def test_remove_task_prefix(self):
        """Some MultiWOZ test items store message as a single string."""
        loader = EmoWOZLoader(Path("."), Path("."))
        message = "Task 11193: You are looking for an expensive restaurant."
        result = loader._messages_to_text(message)
        assert result == "You are looking for an expensive restaurant."

    def test_remove_json_suffix(self):
        """EmoWOZ dialogue IDs should not retain file extensions."""
        loader = EmoWOZLoader(Path("."), Path("."))
        assert loader._normalize_dialogue_id("WOZ20224.json") == "WOZ20224"
        assert loader._normalize_dialogue_id("WOZ20224") == "WOZ20224"

    def test_goal_to_structured_normalizes_task_prefix_in_message_metadata(self):
        """Structured metadata should not preserve synthetic task IDs."""
        loader = EmoWOZLoader(Path("."), Path("."))
        goal_data = {
            "message": "Task 11193: You are looking for an expensive restaurant.",
            "restaurant": {"info": {"food": "chinese"}},
        }
        result = loader._goal_to_structured(goal_data)
        assert (
            result.metadata["message"] == "You are looking for an expensive restaurant."
        )

    def test_goal_to_structured(self):
        """Verify structured goal extraction."""
        loader = EmoWOZLoader(Path("."), Path("."))
        goal_data = {
            "message": "Task 11193: You are looking for an expensive restaurant.",
            "restaurant": {
                "info": {"food": "chinese"},
                "book": {"people": "4"},
                "reqt": ["phone"],
                "fail_info": {"food": "indian"},
            },
        }
        result = loader._goal_to_structured(goal_data)
        assert "restaurant" in result.domains
        assert len(result.intents) == 1
        assert result.intents[0]["slots"]["food"] == "chinese"
        assert result.intents[0]["requests"] == ["phone"]
        assert result.intents[0]["metadata"]["fail_info"]["food"] == "indian"
        assert (
            result.metadata["message"] == "You are looking for an expensive restaurant."
        )

    def test_infer_intent(self):
        """Verify intent inference logic."""
        loader = EmoWOZLoader(Path("."), Path("."))

        # Book + info -> find_and_book
        assert (
            loader._infer_intent({"info": {"a": 1}, "book": {"b": 2}})
            == "find_and_book"
        )
        # Only book -> book
        assert loader._infer_intent({"book": {"b": 2}}) == "book"
        # Only info -> find
        assert loader._infer_intent({"info": {"a": 1}}) == "find"


class TestSpokenWOZLoader:
    """Tests for SpokenWOZ loader."""

    def test_goal_construction_uses_spokenwoz_goal(self):
        """SpokenWOZ goal construction uses its own embedded goal."""
        loader = SpokenWOZLoader(Path("."), Path("."))
        loader._data_cache = {
            "MUL0001": {
                "goal": {
                    "restaurant": {
                        "info": {"food": "chinese", "area": "centre"},
                        "book": {"people": "4", "day": "friday"},
                        "reqt": ["phone"],
                    },
                    "profile": {
                        "info": {
                            "name": "alice",
                            "phonenumber": "12345",
                            "idnumber": "9999",
                            "email": "alice@example.com",
                            "platenumber": "ab12cde",
                        }
                    },
                },
                "log": [],
            }
        }

        with patch.object(loader, "_get_turn_audio_path", return_value=None):
            dialogue = next(loader.load())

        assert "chinese restaurant" in dialogue.goal.text.lower()
        assert "centre" in dialogue.goal.text.lower()
        assert "book a table" in dialogue.goal.text.lower()
        assert "alice" in dialogue.goal.text.lower()
        assert "12345" in dialogue.goal.text
        assert "9999" in dialogue.goal.text
        assert "alice@example.com" in dialogue.goal.text
        assert "ab12cde" in dialogue.goal.text

    def test_goal_to_structured_domains_match_emowoz(self):
        """Structured goal domains align with EmoWOZ domain set."""
        spoken_loader = SpokenWOZLoader(Path("."), Path("."))
        emowoz_loader = EmoWOZLoader(Path("."), Path("."))

        goal_data = {
            "restaurant": {"info": {"food": "chinese"}},
            "profile": {"info": {"name": "alice"}},
        }

        spoken_structured = spoken_loader._goal_to_structured(goal_data)
        emowoz_structured = emowoz_loader._goal_to_structured(goal_data)

        assert spoken_structured.domains == emowoz_structured.domains
        assert spoken_structured.metadata["profile"]["info"]["name"] == "alice"


class TestSGDLoader:
    """Tests for SGD loader."""

    def test_build_goal_text(self):
        """Verify goal text generation."""
        loader = SGDLoader(Path("."))
        slot_values = {
            "Restaurants_1.city": "San Jose",
            "Restaurants_1.cuisine": "Mexican",
        }
        active_intents = {"Restaurants_1": "FindRestaurants"}
        services = ["Restaurants_1"]

        result = loader._build_goal_text(slot_values, active_intents, services)
        # Check for natural language output
        assert "looking for" in result.lower() or "restaurant" in result.lower()
        assert "San Jose" in result

    def test_build_structured_goal(self):
        """Verify structured goal generation."""
        loader = SGDLoader(Path("."))
        slot_values = {"Restaurants_1.city": "San Jose"}
        active_intents = {"Restaurants_1": "FindRestaurants"}
        services = ["Restaurants_1"]

        result = loader._build_structured_goal(slot_values, active_intents, services)
        assert "Restaurants" in result.domains
        assert len(result.intents) == 1
        assert result.intents[0]["slots"]["city"] == "San Jose"

    def test_build_structured_goal_preserves_raw_slot_values(self):
        """Structured SGD goals should keep all raw slot values."""
        loader = SGDLoader(Path("."))
        result = loader._build_structured_goal(
            slot_values={
                "Restaurants_1.restaurant_name": "Bj's restaurant & Brewhouse"
            },
            active_intents={"Restaurants_1": "ReserveRestaurant"},
            services=["Restaurants_1"],
            raw_slot_values={
                "Restaurants_1": {
                    "restaurant_name": [
                        "Bj's",
                        "Bj's Restaurant & Brewhouse",
                        "Bj's restaurant & Brewhouse",
                    ]
                }
            },
        )
        assert result.intents[0]["slot_values"]["restaurant_name"] == [
            "Bj's",
            "Bj's Restaurant & Brewhouse",
            "Bj's restaurant & Brewhouse",
        ]

    def test_build_goal_text_preserves_all_raw_slot_values(self):
        """Text goal should mention every raw SGD slot value variant."""
        loader = SGDLoader(Path("."))
        result = loader._build_goal_text(
            steps=[
                {
                    "service": "Restaurants_1",
                    "intent": "ReserveRestaurant",
                    "new_slots": {
                        "restaurant_name": "Bj's restaurant & Brewhouse",
                        "time": "11:30 am",
                    },
                    "new_slot_values": {
                        "restaurant_name": [
                            "Bj's",
                            "Bj's Restaurant & Brewhouse",
                            "Bj's restaurant & Brewhouse",
                        ],
                        "time": ["11:15 am", "11:30 am"],
                    },
                    "is_intent_change": True,
                }
            ]
        )
        assert "Bj's" in result
        assert "Bj's Restaurant & Brewhouse" in result
        assert "Bj's restaurant & Brewhouse" in result
        assert "11:15 am" in result
        assert "11:30 am" in result


class TestABCDLoader:
    """Tests for ABCD loader."""

    def test_merge_consecutive_turns_inserts_filler_word_not_tag(self):
        loader = ABCDLoader(Path("."))
        delexed_turns = [
            {"speaker": "customer", "text": "hi"},
            {"speaker": "customer", "text": "need help"},
        ]
        original_turns = [
            ["customer", "hi"],
            ["customer", "need help"],
        ]

        filler_word = FILLERS[FP][0]
        with patch(
            "augmentation.loaders.abcd.random.choice", side_effect=[FP, filler_word]
        ):
            merged = loader._merge_consecutive_turns(delexed_turns, original_turns)

        assert len(merged) == 1
        text = merged[0]["original_text"]
        assert FP not in text
        assert filler_word in text
        assert merged[0]["disfluency"] and merged[0]["disfluency"][0]["tag"] == FP

    def test_build_goal_text(self):
        """Verify goal text generation."""
        loader = ABCDLoader(Path("."))
        result = loader._build_goal_text(
            flow="order_issue",
            subflow="missing_item",
            personal={
                "customer_name": "John",
                "security_answer": "alexander",
                "password": "secret123",
            },
            order={"product_names": ["Jacket"]},
            product={},
        )
        # Check for natural language - "order issue" or "issue with your order"
        assert "order" in result.lower() and "issue" in result.lower()
        # Check for subflow mention
        assert "missing" in result.lower()
        assert "John" in result
        assert "alexander" in result
        assert "secret123" in result

    def test_build_structured_goal(self):
        """Verify structured goal generation."""
        loader = ABCDLoader(Path("."))
        result = loader._build_structured_goal(
            flow="order_issue",
            subflow="missing_item",
            personal={"customer_name": "John"},
            order={},
            product={"names": ["Blue Jacket"], "amounts": [42]},
        )
        assert result.domains == ["customer_service"]
        assert result.intents[0]["intent"] == "order_issue.missing_item"
        assert result.intents[0]["slots"]["customer_name"] == "John"
        assert result.metadata["product"]["names"] == ["Blue Jacket"]


class TestTM2Loader:
    """Tests for TM-2 loader."""

    def test_merge_consecutive_turns_inserts_filler_word_not_tag(self):
        loader = TM2Loader(Path("."))
        utterances = [
            {"speaker": "USER", "text": "hi", "segments": []},
            {"speaker": "USER", "text": "need help", "segments": []},
        ]

        filler_word = FILLERS[FP][0]
        with patch(
            "augmentation.loaders.tm2.random.choice", side_effect=[FP, filler_word]
        ):
            merged = loader._merge_consecutive_turns(utterances)

        assert len(merged) == 1
        text = merged[0]["text"]
        assert FP not in text
        assert filler_word in text
        assert merged[0]["disfluency"] and merged[0]["disfluency"][0]["tag"] == FP

    def test_build_goal_text(self):
        """Verify goal text generation."""
        loader = TM2Loader(Path("."))
        result = loader._build_goal_text(
            domain="flights",
            instruction_id="flight-6",
            slots={"flight.destination": "New York"},
        )
        # Check for natural language - "flight" in some form
        assert "flight" in result.lower()
        assert "New York" in result

    def test_build_goal_text_uses_natural_tm2_phrasing(self):
        """Common TM2 slot patterns should avoid mechanical phrasing."""
        loader = TM2Loader(Path("."))
        result = loader._build_goal_text(
            domain="food_ordering",
            instruction_id="food-ordering-2",
            steps=[
                {
                    "turn_slots": [
                        {"slot": "food_order.name.item", "value": "sandwich"},
                        {"slot": "food_order.type.retrieval", "value": "takeout"},
                        {"slot": "food_order.num.people", "value": "one"},
                    ]
                }
            ],
        )
        assert "with item" not in result.lower()
        assert "people people" not in result.lower()
        assert "takeout" in result.lower()
        assert "one person" in result.lower()

    def test_build_goal_text_uses_natural_location_phrasing(self):
        """Location slots should read naturally."""
        loader = TM2Loader(Path("."))
        result = loader._build_goal_text(
            domain="movies",
            instruction_id="movie-26",
            steps=[
                {
                    "turn_slots": [
                        {
                            "slot": "movie_search.location.theater",
                            "value": "Davis, California",
                        },
                        {"slot": "movie_search.genre", "value": "Action"},
                    ]
                }
            ],
        )
        assert "with theater" not in result.lower()
        assert "in Davis, California" in result
        assert "an Action movie" in result

    def test_build_structured_goal(self):
        """Verify structured goal generation."""
        loader = TM2Loader(Path("."))
        result = loader._build_structured_goal(
            domain="flights",
            instruction_id="flight-6",
            slots={"flight.destination": "New York"},
            slot_sequence=[{"slot": "flight.destination", "value": "New York"}],
        )
        assert result.domains == ["flights"]
        assert result.intents[0]["intent"] == "flight"
        assert result.intents[0]["slots"]["flight.destination"] == "New York"
        assert result.intents[0]["slot_values"]["flight.destination"] == ["New York"]

    def test_build_goal_text_preserves_multiple_values_for_same_slot(self):
        """Repeated slot names should keep all distinct values in text."""
        loader = TM2Loader(Path("."))
        result = loader._build_goal_text(
            domain="food_ordering",
            instruction_id="food-1",
            steps=[
                {
                    "turn_slots": [
                        {
                            "slot": "food_order.name.item",
                            "value": "one BLT on rye bread",
                        },
                        {"slot": "food_order.name.item", "value": "a lemonade"},
                    ]
                }
            ],
        )
        assert "one BLT on rye bread" in result
        assert "a lemonade" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
