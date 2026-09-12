import importlib
import sys
import types

import pytest


class FakeTorch(types.SimpleNamespace):
    float16 = "float16"
    bfloat16 = "bfloat16"
    float32 = "float32"
    Tensor = object
    dtype = object


class FakeProcessor:
    def __init__(self, pad_token_id=None, eos_token_id=None):
        self.tokenizer = types.SimpleNamespace(
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
        )


class FakeModelConfig:
    def __init__(self, pad_token_id=None, eos_token_id=None):
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id


class FakeInnerModel:
    def __init__(self, pad_token_id=None, eos_token_id=None):
        self.config = FakeModelConfig(pad_token_id=pad_token_id, eos_token_id=eos_token_id)


class FakeQwenModel:
    def __init__(self):
        self.processor = FakeProcessor(pad_token_id=None, eos_token_id=[17])
        self.model = FakeInnerModel(pad_token_id=None, eos_token_id=None)
        self.tokenized_inputs = []
        self.voice_clone_calls = []
        self.generate_calls = []

    def _build_instruct_text(self, text):
        return f"instruct::{text}"

    def _tokenize_texts(self, texts):
        self.tokenized_inputs.append(list(texts))
        return [[101, 202, 303]]

    def create_voice_clone_prompt(self, **kwargs):
        self.voice_clone_calls.append(kwargs)
        return [{"speaker": "prompt"}]

    def generate_voice_clone(self, **kwargs):
        self.generate_calls.append(kwargs)
        return [["wav-bytes"]], 24000


@pytest.fixture
def qwen_module(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", FakeTorch())
    if "synthesis.local" in sys.modules:
        del sys.modules["synthesis.local"]
    return importlib.import_module("synthesis.local")


def test_parse_torch_dtype_supports_aliases(qwen_module):
    assert qwen_module.parse_torch_dtype("fp16") == "float16"
    assert qwen_module.parse_torch_dtype("bf16") == "bfloat16"
    assert qwen_module.parse_torch_dtype("fp32") == "float32"


def test_resolve_pad_token_id_falls_back_to_tokenizer_eos(qwen_module):
    model = FakeQwenModel()
    assert qwen_module._resolve_pad_token_id(model) == 17


def test_build_voice_clone_prompt_uses_reference_transcript(qwen_module):
    model = FakeQwenModel()

    prompt = qwen_module.build_voice_clone_prompt(model, "/tmp/ref.wav")

    assert prompt == {"speaker": "prompt"}
    assert model.voice_clone_calls == [
        {
            "ref_audio": ["/tmp/ref.wav"],
            "ref_text": [qwen_module.REFERENCE_TRANSCRIPT],
            "x_vector_only_mode": [False],
        }
    ]


def test_synthesize_voice_clone_builds_instruct_ids_and_unwraps_output(qwen_module):
    model = FakeQwenModel()

    wav, sample_rate = qwen_module.synthesize_voice_clone(
        model=model,
        text="Hello there",
        voice_clone_prompt={"speaker": "prompt"},
        emotion_name="joyful",
    )

    assert wav == ["wav-bytes"]
    assert sample_rate == 24000
    assert model.tokenized_inputs == [
        ["instruct::You are a helpful assistant. Please speak in a joyful tone.<|endofprompt|>"]
    ]
    assert model.generate_calls == [
        {
            "text": ["Hello there"],
            "language": [qwen_module.LANGUAGE],
            "voice_clone_prompt": [{"speaker": "prompt"}],
            "instruct_ids": [[101, 202, 303]],
            "max_new_tokens": qwen_module.MAX_NEW_TOKENS,
            "pad_token_id": 17,
        }
    ]


def test_save_wav_creates_parent_directory_and_writes_file(qwen_module, monkeypatch, tmp_path):
    calls = []

    def fake_write(path, wav, samplerate, format):
        calls.append((path, wav, samplerate, format))

    from synthesis import utils

    monkeypatch.setattr(utils, "_soundfile_write", fake_write)
    output_path = tmp_path / "nested" / "sample.wav"

    qwen_module.save_wav(output_path, [0.1, 0.2], 16000)

    assert output_path.parent.exists()
    assert calls == [(str(output_path), [0.1, 0.2], 16000, "WAV")]


def test_load_qwen_model_passes_normalized_dtype(monkeypatch, qwen_module):
    captured = {}

    class FakeQwen3TTSModel:
        @classmethod
        def from_pretrained(cls, model_id, device_map, dtype, attn_implementation):
            captured.update(
                {
                    "model_id": model_id,
                    "device_map": device_map,
                    "dtype": dtype,
                    "attn_implementation": attn_implementation,
                }
            )
            return "loaded-model"

    monkeypatch.setitem(
        sys.modules,
        "qwen_tts",
        types.SimpleNamespace(Qwen3TTSModel=FakeQwen3TTSModel),
    )

    model = qwen_module.load_qwen_model(
        model_id="Qwen/test",
        device_map="cuda:1",
        dtype="bf16",
        attn_implementation="flash_attention_2",
    )

    assert model == "loaded-model"
    assert captured == {
        "model_id": "Qwen/test",
        "device_map": "cuda:1",
        "dtype": "bfloat16",
        "attn_implementation": "flash_attention_2",
    }
