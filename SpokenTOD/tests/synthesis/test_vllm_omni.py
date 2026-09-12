import importlib
import sys
import types

import pytest


class FakeResponse:
    def __init__(self, status_code=200, json_data=None, content=b"", headers=None):
        self.status_code = status_code
        self._json_data = json_data
        self.content = content
        self.headers = headers or {}

    @property
    def text(self):
        if self._json_data is not None:
            return str(self._json_data)
        return self.content.decode("utf-8", errors="replace")

    def json(self):
        if self._json_data is None:
            raise ValueError("no json body")
        return self._json_data


class FakeSession:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def post(self, url, **kwargs):
        self.calls.append({"url": url, **kwargs})
        if not self.responses:
            raise AssertionError(f"Unexpected POST {url}")
        return self.responses.pop(0)


@pytest.fixture
def synthesis_modules(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(bfloat16="bfloat16"))
    monkeypatch.setitem(sys.modules, "qwen_tts", types.SimpleNamespace(Qwen3TTSModel=object))

    for module_name in list(sys.modules):
        if module_name == "synthesis" or module_name.startswith("synthesis."):
            del sys.modules[module_name]

    base_module = importlib.import_module("synthesis.base")
    vllm_module = importlib.import_module("synthesis.vllm_omni")
    return base_module, vllm_module


def test_prepare_voice_uploads_and_caches(synthesis_modules, tmp_path):
    base_module, vllm_module = synthesis_modules
    ref_audio = tmp_path / "voice.wav"
    ref_audio.write_bytes(b"RIFFfake")
    session = FakeSession(
        [
            FakeResponse(json_data={"success": True, "voice": {"name": "speaker-a"}}),
            FakeResponse(json_data={"voice": "speaker-a", "cache_status": "ready"}),
        ]
    )

    synthesizer = vllm_module.VllmOmniSynthesizer(
        base_url="http://localhost:8000",
        session=session,
    )

    prepared = synthesizer.prepare_voice(
        base_module.ReferenceAudio(audio_path=ref_audio, transcript="reference transcript"),
        voice_name="speaker-a",
        force=True,
    )

    assert prepared.voice_name == "speaker-a"
    assert prepared.metadata["cache_status"] == "ready"
    assert session.calls[0]["url"] == "http://localhost:8000/v1/audio/voices"
    assert session.calls[0]["data"]["ref_text"] == "reference transcript"
    assert session.calls[0]["files"]["audio_sample"][0] == "voice.wav"
    assert session.calls[1]["url"] == "http://localhost:8000/v1/audio/voices/speaker-a/cache"
    assert session.calls[1]["params"] == {"force": True}


def test_prepare_voice_falls_back_to_inline_reference_when_upload_unavailable(
    synthesis_modules,
    tmp_path,
):
    base_module, vllm_module = synthesis_modules
    ref_audio = tmp_path / "voice.wav"
    ref_audio.write_bytes(b"RIFFfake")
    session = FakeSession(
        [
            FakeResponse(status_code=404, json_data={"error": {"message": "not found"}}),
        ]
    )

    synthesizer = vllm_module.VllmOmniSynthesizer(
        base_url="http://localhost:8000",
        session=session,
    )

    prepared = synthesizer.prepare_voice(
        base_module.ReferenceAudio(audio_path=ref_audio, transcript="reference transcript"),
        voice_name="speaker-a",
    )

    assert prepared.voice_name is None
    assert prepared.reference.audio_path == ref_audio
    assert prepared.metadata["transport"] == "inline"


def test_synthesize_uses_cached_voice_without_reference_overrides(synthesis_modules, tmp_path):
    base_module, vllm_module = synthesis_modules
    ref_audio = tmp_path / "voice.wav"
    ref_audio.write_bytes(b"RIFFfake")
    session = FakeSession(
        [
            FakeResponse(content=b"wav-bytes", headers={"content-type": "audio/wav"}),
        ]
    )
    synthesizer = vllm_module.VllmOmniSynthesizer(
        base_url="http://localhost:8000",
        session=session,
    )
    prepared = base_module.PreparedVoice(
        backend="vllm_omni",
        voice_name="speaker-a",
        reference=base_module.ReferenceAudio(audio_path=ref_audio, transcript="reference transcript"),
        metadata={"cache_status": "ready"},
    )

    result = synthesizer.synthesize(
        text="Hello there",
        prepared_voice=prepared,
        emotion_name="calm",
    )

    assert result.audio == b"wav-bytes"
    assert result.media_type == "audio/wav"
    payload = session.calls[0]["json"]
    assert payload["voice"] == "speaker-a"
    assert payload["instructions"].startswith("You are a helpful assistant.")
    assert "ref_audio" not in payload
    assert "ref_text" not in payload
    assert "x_vector_only_mode" not in payload


def test_synthesize_inlines_reference_audio_when_voice_is_not_registered(
    synthesis_modules,
    tmp_path,
):
    base_module, vllm_module = synthesis_modules
    ref_audio = tmp_path / "voice.wav"
    ref_audio.write_bytes(b"RIFFfake")
    session = FakeSession(
        [
            FakeResponse(content=b"wav-bytes", headers={"content-type": "audio/wav"}),
        ]
    )
    synthesizer = vllm_module.VllmOmniSynthesizer(
        base_url="http://localhost:8000",
        session=session,
    )
    prepared = base_module.PreparedVoice(
        backend="vllm_omni",
        reference=base_module.ReferenceAudio(audio_path=ref_audio),
        metadata={"transport": "inline"},
    )

    synthesizer.synthesize(
        text="Hello there",
        prepared_voice=prepared,
        emotion_name="calm",
    )

    payload = session.calls[0]["json"]
    assert payload["task_type"] == "Base"
    assert payload["ref_audio"].startswith("data:audio/wav;base64,")
    assert payload["x_vector_only_mode"] is True
    assert "ref_text" not in payload


def test_synthesize_to_file_writes_encoded_audio_bytes(synthesis_modules, tmp_path):
    base_module, vllm_module = synthesis_modules
    ref_audio = tmp_path / "voice.wav"
    ref_audio.write_bytes(b"RIFFfake")
    session = FakeSession(
        [
            FakeResponse(content=b"wav-bytes", headers={"content-type": "audio/wav"}),
        ]
    )
    synthesizer = vllm_module.VllmOmniSynthesizer(
        base_url="http://localhost:8000",
        session=session,
    )
    prepared = base_module.PreparedVoice(
        backend="vllm_omni",
        voice_name="speaker-a",
        reference=base_module.ReferenceAudio(audio_path=ref_audio, transcript="reference transcript"),
        metadata={"cache_status": "ready"},
    )
    output_path = tmp_path / "nested" / "sample.wav"

    written_path = synthesizer.synthesize_to_file(
        text="Hello there",
        prepared_voice=prepared,
        output_path=output_path,
        emotion_name="calm",
    )

    assert written_path == output_path
    assert output_path.read_bytes() == b"wav-bytes"
