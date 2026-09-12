"""Unit tests for emotion tagging module."""

import asyncio

import pytest

from augmentation.batch.client import MockBatchClient
from augmentation.emotion.prompts import (
    FEWSHOT_EXAMPLES_SIMPLE,
    build_emotion_prompt,
    parse_emotion_response,
)
from augmentation.emotion.tagger import (
    EmotionTagger,
    tag_dialogue_emotions,
)


class TestEmotionPrompts:
    """Tests for emotion prompts."""

    def test_few_shot_examples_coverage(self):
        """Verify all emotion labels have examples."""
        labels_with_examples = set(label for _, label in FEWSHOT_EXAMPLES_SIMPLE)

        # Should have examples for all 7 labels (0-6)
        for label in range(7):
            assert label in labels_with_examples, f"Missing example for label {label}"

    def test_build_emotion_prompt(self):
        """Test prompt construction."""
        prompt = build_emotion_prompt("I need a restaurant")

        # Should contain instructions
        assert "emotion classifier" in prompt.lower()
        assert "0-6" in prompt or "0:" in prompt

        # Should contain the utterance
        assert "I need a restaurant" in prompt

        # Should contain examples
        assert "Examples" in prompt

class TestParseEmotionResponse:
    """Tests for response parsing."""

    def test_parse_single_digit(self):
        """Parse simple digit."""
        assert parse_emotion_response("0") == 0
        assert parse_emotion_response("6") == 6
        assert parse_emotion_response("3") == 3

    def test_parse_with_text(self):
        """Parse digit with surrounding text."""
        assert parse_emotion_response("The emotion is 2 (dissatisfied)") == 2
        assert parse_emotion_response("Label: 5") == 5

    def test_parse_invalid(self):
        """Invalid responses should return 0."""
        assert parse_emotion_response("") == 0
        assert parse_emotion_response("invalid") == 0
        assert parse_emotion_response("99") == 0  # First digit 9 is out of range


class TestMockBatchClient:
    """Tests for mock batch client."""

    def test_chat_completion(self):
        """Test basic chat completion output."""
        client = MockBatchClient(default_emotion=2)
        response = client.chat_completion([{"role": "user", "content": "hello"}])
        assert response == "2"

    def test_batch_completion_async(self):
        """Test async batch completions."""
        client = MockBatchClient(default_emotion=4)
        responses = asyncio.run(
            client.batch_completion_async(
                [
                    {"messages": [{"role": "user", "content": "one"}]},
                    {"messages": [{"role": "user", "content": "two"}]},
                ]
            )
        )
        assert responses == ["4", "4"]


def test_tag_utterances_use_batch_routes_through_async_path(monkeypatch):
    tagger = EmotionTagger(model="gpt-4.1-mini")

    async def fake_tag_async(utterances, max_concurrency=50):
        assert utterances == ["hello", "thanks"]
        assert max_concurrency == 3
        return [
            {"label": 1, "name": "fearful"},
            {"label": 6, "name": "satisfied"},
        ]

    monkeypatch.setattr(tagger, "tag_utterances_async", fake_tag_async)

    result = tagger.tag_utterances(
        ["hello", "thanks"],
        use_batch=True,
        max_workers=3,
    )

    assert result == [
        {"label": 1, "name": "fearful"},
        {"label": 6, "name": "satisfied"},
    ]

class TestTagDialogueEmotions:
    """Tests for dialogue emotion tagging."""

    def test_with_existing_labels(self):
        """Test with pre-existing labels (EmoWOZ)."""
        turns = [
            {"role": "user", "text": "Hello"},
            {"role": "assistant", "text": "Hi there"},
            {"role": "user", "text": "Thanks"},
        ]
        existing_labels = [0, -1, 6]  # -1 for assistant

        result = tag_dialogue_emotions(turns, "emowoz", existing_labels=existing_labels)

        assert result[0]["emotion"]["label"] == 0
        assert result[0]["emotion"]["name"] == "neutral"
        assert "emotion" not in result[1]
        assert result[2]["emotion"]["label"] == 6
        assert result[2]["emotion"]["name"] == "satisfied"

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
