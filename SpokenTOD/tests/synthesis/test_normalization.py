import pytest

from synthesis.normalization import NemoNormalizer, spoken_text, tts_text


@pytest.fixture(scope="module")
def normalizer(tmp_path_factory):
    return NemoNormalizer(tmp_path_factory.mktemp("nemo"))


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("u2<|endoftext|>", "u two <|endoftext|>"),
        ("Yes =(", "Yes equal sign ("),
        ("thank you :)", "thank you:)"),
        (
            "0213 Woodshore St  La Fayette, MI 91969",
            "zero two one three Woodshore St La Fayette, Michigan nine one nine six nine",
        ),
        (
            "You said 29.99, but the listen, actual price is 39.99.",
            "You said two nine dot nine nine comma but the listen, actual price is thirty nine point nine nine.",
        ),
    ],
)
def test_nemo_normalization_matches_public_metadata(normalizer, source, expected):
    assert normalizer(source) == expected


def test_spoken_text_removes_annotation_tags_but_preserves_endoftext():
    assert spoken_text("[FP] I need 2 tickets.<|endoftext|>") == "I need 2 tickets.<|endoftext|>"


def test_tts_text_removes_nonverbal_and_terminal_control_markers():
    assert tts_text("Please wait<bargein>") == "Please wait"
    assert tts_text("Thank you.<|endoftext|>") == "Thank you."
