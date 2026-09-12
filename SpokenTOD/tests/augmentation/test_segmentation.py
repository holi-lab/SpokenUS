"""Unit tests for cross-turn slot segmentation."""

import pytest

from augmentation.segmentation.rules import (
    segment_phone,
    segment_email,
    segment_id_number,
    segment_user_name,
    segment_complex_id,
    segment_slot,
    inject_error_correction,
)
from augmentation.segmentation.generator import (
    generate_crossturn_dialogue,
    find_segmentable_slots,
    SegmentedTurn,
)


class TestSegmentationRules:
    """Tests for segmentation rules."""

    def test_segment_phone(self):
        """Test phone number segmentation."""
        result = segment_phone("5258576375249903")
        assert len(result) == 4
        assert result[0] == "5 2 5 8"
        assert result[1] == "5 7 6 3"
        assert result[2] == "7 5 2 4"
        assert result[3] == "9 9 0 3"

    def test_segment_phone_with_separators(self):
        """Test phone with dashes/spaces."""
        result = segment_phone("555-123-4567")
        assert len(result) == 3
        assert result[0] == "5 5 5 1"
        assert result[1] == "2 3 4 5"
        assert result[2] == "6 7"

    def test_segment_email(self):
        """Test email segmentation."""
        result = segment_email("john.smith@gmail.com")
        assert len(result) == 2
        assert result[0] == "john dot smith"
        assert result[1] == "at gmail dot com"

    def test_segment_email_simple(self):
        """Test simple email."""
        result = segment_email("test@example.org")
        assert len(result) == 2
        assert result[0] == "test"
        assert "example dot org" in result[1]

    def test_segment_id_number(self):
        """Test ID number segmentation."""
        result = segment_id_number("5258576375249903")
        assert len(result) == 4
        assert result[0] == "5 2 5 8"

    def test_segment_user_name(self):
        """Test user name segmentation."""
        result = segment_user_name("John Smith")
        assert len(result) == 2
        assert result[0] == "John"
        assert result[1] == "Smith"

    def test_segment_user_name_single(self):
        """Test single word name."""
        result = segment_user_name("Madonna")
        assert len(result) == 1
        assert result[0] == "Madonna"

    def test_segment_complex_id(self):
        """Test complex ID segmentation."""
        result = segment_complex_id("TR1234")
        assert len(result) == 2
        assert result[0] == "T R"
        assert result[1] == "1 2 3 4"

    def test_segment_slot_dispatch(self):
        """Test segment_slot dispatches correctly."""
        phone = segment_slot("5551234567", "phone")
        assert len(phone) >= 2

        email = segment_slot("test@test.com", "email")
        assert len(email) == 2

        name = segment_slot("John Doe", "user_name")
        assert len(name) == 2


class TestErrorInjection:
    """Tests for error correction injection."""

    def test_no_error_on_single_segment(self):
        """Single segment should not have error."""
        result = inject_error_correction(["test"], error_prob=1.0)
        assert len(result) == 1
        assert result[0] == ("test", False)

    def test_error_injection_structure(self):
        """Error injection should add correction tuple."""
        # Force error by setting prob to 1.0
        segments = ["5 2 5 8", "5 7 6 3", "7 5 2 4", "9 9 0 3"]
        result = inject_error_correction(segments, error_prob=1.0)
        
        # Should have one more item due to correction
        assert len(result) == 5
        
        # One item should be marked as correction
        corrections = [r for r in result if r[1]]
        assert len(corrections) == 1

    def test_no_error_with_zero_prob(self):
        """Zero probability should not inject errors."""
        segments = ["5 2 5 8", "5 7 6 3"]
        result = inject_error_correction(segments, error_prob=0.0)
        
        assert len(result) == 2
        assert all(not is_corr for _, is_corr in result)


class TestDialogueGenerator:
    """Tests for multi-turn dialogue generator."""

    def test_generate_phone_dialogue(self):
        """Test phone number dialogue generation."""
        turns = generate_crossturn_dialogue(
            slot_name="phone_number",
            slot_value="5258576375249903",
            slot_type="phone",
            error_prob=0.0,  # Disable error for predictable test
        )
        
        # Should have alternating user/assistant turns
        assert len(turns) >= 8  # 4 segments * 2 turns each
        
        user_turns = [t for t in turns if t.role == "user"]
        asst_turns = [t for t in turns if t.role == "assistant"]
        
        assert len(user_turns) == 4
        assert len(asst_turns) == 4
        
        # First user turn should mention phone number
        assert "phone number" in turns[0].text.lower() or "5 2 5 8" in turns[0].text

    def test_generate_email_dialogue(self):
        """Test email dialogue generation."""
        turns = generate_crossturn_dialogue(
            slot_name="email",
            slot_value="john@gmail.com",
            slot_type="email",
            error_prob=0.0,
        )
        
        user_turns = [t for t in turns if t.role == "user"]
        assert len(user_turns) == 2  # local + domain

    def test_generate_with_error_correction(self):
        """Test dialogue with error correction."""
        turns = generate_crossturn_dialogue(
            slot_name="id_number",
            slot_value="5258576375249903",
            slot_type="id_number",
            error_prob=1.0,  # Force error
        )
        
        # Should have correction turn
        corrections = [t for t in turns if t.segment and t.segment.get("is_correction")]
        assert len(corrections) >= 1
        
        # Correction text should include sorry/wait/actually
        correction_texts = [t.text.lower() for t in corrections]
        assert any(
            any(word in text for word in ["sorry", "wait", "actually", "meant"])
            for text in correction_texts
        )

    def test_segment_metadata(self):
        """Test segment metadata in turns."""
        turns = generate_crossturn_dialogue(
            slot_name="phone",
            slot_value="55512345",
            slot_type="phone",
            error_prob=0.0,
        )
        
        user_turns = [t for t in turns if t.role == "user"]
        for i, turn in enumerate(user_turns):
            assert turn.segment is not None
            assert turn.segment["slot"] == "phone"
            assert turn.segment["idx"] == i
            assert turn.segment["total"] == len(user_turns)


class TestFindSegmentableSlots:
    """Tests for finding segmentable slots."""

    def test_find_sgd_slots(self):
        """Test finding SGD segmentable slots."""
        state = {
            "Restaurants_1": {
                "phone_number": "555-123-4567",
                "city": "San Jose",
            }
        }
        result = find_segmentable_slots(state, "sgd")
        
        assert len(result) >= 1
        slot_names = [r[0] for r in result]
        assert "phone_number" in slot_names

    def test_find_abcd_slots(self):
        """Test finding ABCD segmentable slots."""
        state = {
            "customer_service": {
                "email": "test@test.com",
                "account_id": "ABC123",
            }
        }
        result = find_segmentable_slots(state, "abcd")
        
        slot_names = [r[0] for r in result]
        assert "email" in slot_names
        assert "account_id" in slot_names

    def test_empty_state(self):
        """Test with empty state."""
        result = find_segmentable_slots({}, "sgd")
        assert result == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
