import pytest

from synthesis import utils


def test_read_reference_transcript_strips_contents(tmp_path):
    transcript_path = tmp_path / "reading-passage.txt"
    transcript_path.write_text("  hello world  \n", encoding="utf-8")

    assert utils.read_reference_transcript(transcript_path) == "hello world"


def test_guess_audio_media_type_normalizes_wav_alias(tmp_path):
    audio_path = tmp_path / "voice.wav"
    audio_path.write_bytes(b"RIFFfake")

    assert utils.guess_audio_media_type(audio_path) == "audio/wav"


def test_encode_audio_as_data_uri_uses_normalized_media_type(tmp_path):
    audio_path = tmp_path / "voice.wav"
    audio_path.write_bytes(b"RIFFfake")

    data_uri = utils.encode_audio_as_data_uri(audio_path)

    assert data_uri.startswith("data:audio/wav;base64,")


def test_build_default_voice_name_is_stable(tmp_path):
    audio_path = tmp_path / "My Voice.wav"
    audio_path.write_bytes(b"RIFFfake")

    voice_name = utils.build_default_voice_name(
        audio_path,
        transcript="reference transcript",
        x_vector_only_mode=False,
    )

    assert voice_name.startswith("my-voice-")
    assert len(voice_name.split("-")[-1]) == 8


def test_write_audio_output_writes_encoded_bytes(tmp_path):
    output_path = tmp_path / "nested" / "sample.wav"

    written_path = utils.write_audio_output(output_path, b"wav-bytes")

    assert written_path == output_path
    assert output_path.read_bytes() == b"wav-bytes"


def test_read_reference_transcript_raises_with_download_hint(tmp_path):
    missing_path = tmp_path / "missing.txt"

    with pytest.raises(FileNotFoundError, match="make download saa"):
        utils.read_reference_transcript(missing_path)
