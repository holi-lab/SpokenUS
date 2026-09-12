import copy
import json
import wave
from pathlib import Path

import pytest

from synthesis.base import PreparedVoice, SynthesisOutput


def fixture_dialogue():
    return {
        "dialogue_id": "../unsafe/id",
        "speaker": {"filename": "user.mp3"},
        "assistant_speaker": {"filename": "agent.mp3"},
        "goal": {"text": "book 2 tickets"},
        "turns": [
            {
                "role": "user",
                "text": "I need 2 tickets.<|endoftext|>",
                "tagged": "[FP] I need 2 tickets.",
                "emotion": {"label": 5, "name": "excited"},
                "state": {"count": "2"},
            },
            {"role": "assistant", "text": "For 2 people?"},
            {"role": "user", "text": "Yes."},
        ],
    }


class TinySynthesizer:
    def __init__(self):
        self.voices = []
        self.calls = []

    def prepare_voice(self, reference):
        self.voices.append(str(reference.audio_path))
        return PreparedVoice(backend="tiny", reference=reference, voice_name=reference.filename)

    def synthesize(
        self, text, prepared_voice, emotion_name, language=None, max_new_tokens=None
    ):
        self.calls.append((text, prepared_voice, emotion_name, max_new_tokens))
        return SynthesisOutput(audio=[0.0, 0.2, -0.2] * 160, sample_rate=16000)


class BytesSynthesizer(TinySynthesizer):
    def prepare_voice(self, reference):
        return PreparedVoice(
            backend="vllm_omni", reference=reference, voice_name=reference.filename
        )

    def synthesize(
        self, text, prepared_voice, emotion_name, language=None, max_new_tokens=None
    ):
        return SynthesisOutput(audio=b"RIFFserver-wav", media_type="audio/wav")


def write_pcm(path, wav, sample_rate):
    import struct

    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(sample_rate)
        stream.writeframes(b"".join(struct.pack("<h", int(x * 32767)) for x in wav))


def run(tmp_path, monkeypatch, record=None, **kwargs):
    from synthesis.pipeline import process_jsonl

    record = record or fixture_dialogue()
    source = tmp_path / "input.jsonl"
    source.write_text(json.dumps(record) + "\n")
    refs = tmp_path / "refs"
    refs.mkdir(exist_ok=True)
    for name in ["user.mp3", "agent.mp3"]:
        (refs / name).write_bytes(b"reference")
    synth = TinySynthesizer()
    target = tmp_path / "output.jsonl"
    stats = process_jsonl(
        source,
        target,
        synthesizer=synth,
        normalizer=lambda t: t.replace("2", "two"),
        reference_dir=refs,
        **kwargs,
    )
    return record, json.loads(target.read_text()), synth, stats


def test_jsonl_to_audio_preserves_annotations_and_reuses_speakers(tmp_path, monkeypatch):
    original, result, synth, stats = run(tmp_path, monkeypatch)
    assert result["goal"]["text"] == original["goal"]["text"]
    assert result["goal"]["normalized"] == "book two tickets"
    assert result["turns"][0]["state"] == original["turns"][0]["state"]
    assert result["turns"][0]["text"] == original["turns"][0]["text"]
    assert result["turns"][0]["normalized_text"] == "I need two tickets.<|endoftext|>"
    assert synth.calls[0][0] == "I need two tickets."
    assert len(synth.voices) == 2
    assert synth.calls[0][1].voice_name == synth.calls[2][1].voice_name
    assert synth.calls[1][2] == "neutral"
    assert stats == {"dialogues": 1, "synthesized_turns": 3, "reused_turns": 0, "control_turns": 0}
    for t in result["turns"]:
        p = Path(t["audio_path"])
        assert p.is_relative_to(tmp_path)
        with wave.open(str(p)) as audio:
            assert audio.getnframes() == 480
            assert audio.getframerate() == 16000


def test_pipeline_does_not_speak_bargein_marker(tmp_path, monkeypatch):
    record = fixture_dialogue()
    record["turns"][1]["text"] = "For 2 people?<bargein>"
    _, result, synth, _ = run(tmp_path, monkeypatch, record)

    assert result["turns"][1]["normalized_text"] == "For two people?<bargein>"
    assert synth.calls[1][0] == "For two people?"


def test_normalize_only_does_not_load_speakers_or_synthesize(tmp_path, monkeypatch):
    record = fixture_dialogue()
    del record["speaker"]
    del record["assistant_speaker"]
    _, result, synth, stats = run(tmp_path, monkeypatch, record, normalize_only=True)
    assert not synth.calls
    assert result["turns"][1]["normalized_text"] == "For two people?"
    assert stats["synthesized_turns"] == 0


def test_existing_normalized_text_is_reused_verbatim(tmp_path, monkeypatch):
    record = fixture_dialogue()
    record["turns"][0]["text_normalized"] = "canonical public normalization"
    _, result, synth, _ = run(tmp_path, monkeypatch, record)

    assert result["turns"][0]["normalized_text"] == "canonical public normalization"
    assert synth.calls[0][0] == "canonical public normalization"


def test_goal_and_turns_use_the_same_normalizer_instance(tmp_path):
    from synthesis.pipeline import process_jsonl

    record = fixture_dialogue()
    record["goal"]["text"] = "[FP] book 2 tickets"
    source = tmp_path / "input.jsonl"
    source.write_text(json.dumps(record) + "\n")
    target = tmp_path / "output.jsonl"
    calls = []

    def normalizer(text):
        calls.append(text)
        return f"normalized::{text}"

    process_jsonl(
        source,
        target,
        synthesizer=None,
        normalizer=normalizer,
        normalize_only=True,
    )

    result = json.loads(target.read_text())
    assert calls == [
        "book 2 tickets",
        "I need 2 tickets.<|endoftext|>",
        "For 2 people?",
        "Yes.",
    ]
    assert result["goal"]["text"] == "[FP] book 2 tickets"
    assert result["goal"]["normalized"] == "normalized::book 2 tickets"
    assert result["turns"][0]["normalized_text"] == ("normalized::I need 2 tickets.<|endoftext|>")


def test_existing_normalized_goal_is_reused_verbatim(tmp_path, monkeypatch):
    record = fixture_dialogue()
    record["goal"]["normalized"] = "canonical normalized goal"

    _, result, _, _ = run(tmp_path, monkeypatch, record, normalize_only=True)

    assert result["goal"]["normalized"] == "canonical normalized goal"


@pytest.mark.parametrize("goal", [None, {}, {"text": None}, {"text": ""}])
def test_invalid_goal_has_dialogue_context(tmp_path, monkeypatch, goal):
    record = fixture_dialogue()
    record["goal"] = goal

    with pytest.raises(ValueError, match=r"unsafe/id.*goal.text"):
        run(tmp_path, monkeypatch, record, normalize_only=True)


def test_existing_audio_is_reused(tmp_path, monkeypatch):
    native = tmp_path / "native.wav"
    write_pcm(native, [0.1] * 100, 16000)
    record = fixture_dialogue()
    record["turns"][0]["audio_path"] = str(native)
    _, result, synth, stats = run(tmp_path, monkeypatch, record)
    assert result["turns"][0]["audio_path"] == str(native)
    assert len(synth.calls) == 2
    assert stats["reused_turns"] == 1


def test_existing_manifest_is_not_overwritten(tmp_path, monkeypatch):
    target = tmp_path / "output.jsonl"
    target.write_text("keep")
    with pytest.raises(FileExistsError):
        run(tmp_path, monkeypatch)
    assert target.read_text() == "keep"


def test_bad_reference_has_dialogue_context(tmp_path, monkeypatch):
    record = copy.deepcopy(fixture_dialogue())
    record["speaker"]["filename"] = "missing.mp3"
    with pytest.raises(ValueError, match="unsafe/id"):
        run(tmp_path, monkeypatch, record)


def test_normalization_keeps_relative_native_audio_resolvable(tmp_path):
    from synthesis.pipeline import process_jsonl

    source_dir = tmp_path / "source"
    source_dir.mkdir()
    native = source_dir / "native.wav"
    write_pcm(native, [0.1] * 100, 16000)
    source = source_dir / "input.jsonl"
    source.write_text(
        json.dumps(
            {
                "dialogue_id": "native",
                "goal": {"text": "reuse native audio"},
                "turns": [{"role": "user", "text": "Hello.", "audio_path": "native.wav"}],
            }
        )
        + "\n"
    )
    normalized = tmp_path / "normalized" / "result.jsonl"
    process_jsonl(source, normalized, synthesizer=None, normalizer=lambda t: t, normalize_only=True)
    output = tmp_path / "audio.jsonl"
    stats = process_jsonl(normalized, output, synthesizer=TinySynthesizer(), normalizer=lambda t: t)
    assert stats["reused_turns"] == 1
    assert json.loads(output.read_text())["turns"][0]["audio_path"] == str(native)


def test_endoftext_turn_is_preserved_without_synthesizing_empty_speech(tmp_path, monkeypatch):
    record = fixture_dialogue()
    record["turns"].append({"role": "user", "text": "<|endoftext|>"})
    _, result, synth, stats = run(tmp_path, monkeypatch, record)
    terminal = result["turns"][-1]
    assert terminal["text"] == "<|endoftext|>"
    assert terminal["normalized_text"] == "<|endoftext|>"
    assert terminal["synthesis"]["status"] == "control_only"
    assert "audio_path" not in terminal
    assert len(synth.calls) == 3
    assert stats["control_turns"] == 1


def test_invalid_waveform_does_not_commit_dialogue(tmp_path, monkeypatch):
    monkeypatch.setattr(
        TinySynthesizer,
        "synthesize",
        lambda *a, **k: SynthesisOutput(audio=[float("nan")], sample_rate=16000),
    )
    with pytest.raises(ValueError, match="Invalid waveform"):
        run(tmp_path, monkeypatch)
    assert (tmp_path / "output.jsonl").read_text() == ""
    assert not list(tmp_path.rglob("*.wav"))


def test_server_wav_bytes_are_written_without_local_audio_dependencies(tmp_path):
    from synthesis.pipeline import process_jsonl

    source = tmp_path / "input.jsonl"
    source.write_text(json.dumps(fixture_dialogue()) + "\n")
    refs = tmp_path / "refs"
    refs.mkdir()
    for name in ["user.mp3", "agent.mp3"]:
        (refs / name).write_bytes(b"reference")
    target = tmp_path / "output.jsonl"

    process_jsonl(
        source,
        target,
        synthesizer=BytesSynthesizer(),
        normalizer=lambda text: text,
        reference_dir=refs,
    )

    record = json.loads(target.read_text())
    assert Path(record["turns"][0]["audio_path"]).read_bytes() == b"RIFFserver-wav"
    assert record["turns"][0]["synthesis"]["backend"] == "vllm_omni"


def test_generation_limit_is_forwarded_to_vllm(tmp_path, monkeypatch):
    _, _, synth, _ = run(tmp_path, monkeypatch, max_new_tokens=321)

    assert all(call[3] == 321 for call in synth.calls)


def test_jsonl_pipeline_uses_vllm_voice_and_speech_endpoints(tmp_path):
    from synthesis.pipeline import process_jsonl
    from synthesis.vllm_omni import VllmOmniSynthesizer

    class Response:
        def __init__(self, *, payload=None, content=b"", content_type="application/json"):
            self.status_code = 200
            self._payload = payload
            self.content = content
            self.headers = {"content-type": content_type}
            self.text = ""

        def json(self):
            if self._payload is None:
                raise ValueError("not JSON")
            return self._payload

    class Session:
        def __init__(self):
            self.calls = []
            self.responses = [
                Response(payload={"voice": {"name": "user"}}),
                Response(payload={"cache_status": "ready"}),
                Response(content=b"RIFF-user-0", content_type="audio/wav"),
                Response(payload={"voice": {"name": "agent"}}),
                Response(payload={"cache_status": "ready"}),
                Response(content=b"RIFF-agent-1", content_type="audio/wav"),
                Response(content=b"RIFF-user-2", content_type="audio/wav"),
            ]

        def post(self, url, **kwargs):
            self.calls.append({"url": url, **kwargs})
            return self.responses.pop(0)

    source = tmp_path / "input.jsonl"
    source.write_text(json.dumps(fixture_dialogue()) + "\n")
    refs = tmp_path / "refs"
    refs.mkdir()
    for name in ["user.mp3", "agent.mp3"]:
        (refs / name).write_bytes(b"reference")
    session = Session()
    synthesizer = VllmOmniSynthesizer(
        "http://127.0.0.1:8000", session=session, enable_cache=True
    )
    target = tmp_path / "output.jsonl"

    process_jsonl(
        source,
        target,
        synthesizer=synthesizer,
        normalizer=lambda text: text,
        reference_dir=refs,
        max_new_tokens=123,
    )

    urls = [call["url"] for call in session.calls]
    assert urls.count("http://127.0.0.1:8000/v1/audio/voices") == 2
    assert urls.count("http://127.0.0.1:8000/v1/audio/speech") == 3
    speech_payloads = [call["json"] for call in session.calls if call["url"].endswith("/speech")]
    assert all(payload["max_new_tokens"] == 123 for payload in speech_payloads)
    record = json.loads(target.read_text())
    assert Path(record["turns"][0]["audio_path"]).read_bytes() == b"RIFF-user-0"
    assert record["turns"][1]["synthesis"]["backend"] == "vllm_omni"
