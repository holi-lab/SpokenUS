import pytest

from src.demographic_sampler import ASSISTANT_NATIVE_POOL, DemographicSampler


def test_selected_speakers_mode_uses_only_manifest_entries(tmp_path):
    archive = tmp_path / "SpeechAccentArchive"
    recordings = archive / "recordings" / "recordings"
    recordings.mkdir(parents=True)
    (recordings / "selected.mp3").touch()
    (archive / "selected_speakers.csv").write_text(
        "filename,category,cohort,sex,age,country,native_language\n"
        "selected.mp3,African,20-30,female,27,ethiopia,amharic\n",
        encoding="utf-8",
    )

    sampler = DemographicSampler(
        base_dir=str(archive),
        with_selected_speakers=True,
    )

    assert sampler.speakers == [
        {
            "filename": "selected.mp3",
            "category": "African",
            "cohort": "20-30",
            "sex": "female",
            "age": 27,
            "country": "ethiopia",
            "native_language": "amharic",
        }
    ]


def test_selected_speakers_mode_requires_manifest(tmp_path):
    with pytest.raises(FileNotFoundError, match="selected speaker manifest"):
        DemographicSampler(
            base_dir=str(tmp_path / "SpeechAccentArchive"),
            with_selected_speakers=True,
        )


def test_demographic_sampler_loads_speakers():
    sampler = DemographicSampler()
    assert sampler.speakers


def test_origin_country_category_mapping():
    sampler = DemographicSampler(category_strategy="origin_country")

    african = [s for s in sampler.speakers if s.get("country") == "ethiopia"]
    assert african, "Expected at least one Ethiopia speaker in SAA metadata"
    assert all(s.get("category") == "African" for s in african)

    native = [s for s in sampler.speakers if s.get("country") == "canada"]
    assert native, "Expected at least one Canada speaker in SAA metadata"
    assert all(s.get("category") == "Native" for s in native)

    indian = [s for s in sampler.speakers if s.get("country") == "india"]
    assert indian, "Expected at least one India speaker in SAA metadata"
    assert all(s.get("category") == "Indian" for s in indian)


def test_sample_assistant_speaker_returns_valid_speaker():
    """Test that sample_assistant_speaker returns a speaker from the Native pool."""
    sampler = DemographicSampler()
    
    speaker = sampler.sample_assistant_speaker()
    assert speaker is not None
    assert speaker["filename"].endswith(".mp3")
    assert speaker["sex"] in ("male", "female")
    assert speaker["country"] in ("usa", "uk", "canada", "australia")


def test_find_speaker_excludes_assistant_pool():
    """Test that find_speaker never returns a speaker from ASSISTANT_NATIVE_POOL."""
    sampler = DemographicSampler()
    
    # Get all assistant pool filenames
    assistant_filenames = set(
        s["filename"] for s in ASSISTANT_NATIVE_POOL["male"] + ASSISTANT_NATIVE_POOL["female"]
    )
    
    # Sample many times to ensure no collision
    for _ in range(50):
        demo = sampler.sample_demographic()
        speaker = sampler.find_speaker(demo)
        if speaker:
            assert speaker["filename"] not in assistant_filenames, \
                f"User speaker {speaker['filename']} should not be in assistant pool"


def test_sample_assistant_speaker_returns_from_pool():
    """Test that sample_assistant_speaker returns a speaker from the Native pool."""
    sampler = DemographicSampler()
    
    assistant_filenames = set(
        s["filename"] for s in ASSISTANT_NATIVE_POOL["male"] + ASSISTANT_NATIVE_POOL["female"]
    )
    
    for _ in range(20):
        speaker = sampler.sample_assistant_speaker()
        assert speaker is not None
        assert speaker["filename"] in assistant_filenames


def test_assistant_native_pool_has_correct_structure():
    """Verify the predefined pool has 5 male and 5 female speakers."""
    assert len(ASSISTANT_NATIVE_POOL["male"]) == 5
    assert len(ASSISTANT_NATIVE_POOL["female"]) == 5
    
    # All should have .mp3 extension
    for sex in ("male", "female"):
        for speaker in ASSISTANT_NATIVE_POOL[sex]:
            assert speaker["filename"].endswith(".mp3")
            assert "country" in speaker
