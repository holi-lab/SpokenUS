import asyncio

import pytest
from click.testing import CliRunner
from httpx import Request
from litellm.exceptions import APIConnectionError
from pydantic import BaseModel

import augment
import augmentation.batch.client as client_module
from augmentation.batch.client import BatchClient, LLMUsageTracker


class DummyMessage:
    def __init__(self, content=None, parsed=None):
        self.content = content
        self.parsed = parsed


class DummyChoice:
    def __init__(self, content=None, parsed=None):
        self.message = DummyMessage(content=content, parsed=parsed)


class DummyResponse:
    def __init__(self, content=None, parsed=None, usage=None):
        self.choices = [DummyChoice(content=content, parsed=parsed)]
        self.usage = usage


class StructuredPayload(BaseModel):
    applicable: bool
    reason: str = ""


def test_format_request_error_includes_configured_api_base():
    client = BatchClient(model="gpt-4.1-mini", api_base="http://localhost:4000/v1")

    message = client._format_request_error(
        APIConnectionError(
            message="Connection error.",
            llm_provider="openai",
            model="gpt-4.1-mini",
            request=Request("POST", "http://localhost:4000/v1/chat/completions"),
        )
    )

    assert "Failed to connect to API for model 'gpt-4.1-mini'" in message
    assert "http://localhost:4000/v1" in message


def test_chat_completion_uses_litellm_completion(monkeypatch):
    captured = {}

    def fake_completion(**kwargs):
        captured.update(kwargs)
        return DummyResponse("2")

    def fail_openai_client(*args, **kwargs):
        raise AssertionError("OpenAI client should not be constructed")

    monkeypatch.setattr(client_module, "completion", fake_completion, raising=False)
    monkeypatch.setattr(client_module, "OpenAI", fail_openai_client, raising=False)

    client = BatchClient(model="gpt-4.1-mini")
    response = client.chat_completion(
        messages=[{"role": "user", "content": "hello"}],
        max_tokens=7,
        temperature=0.2,
    )

    assert response == "2"
    assert captured["model"] == "gpt-4.1-mini"
    assert captured["messages"] == [{"role": "user", "content": "hello"}]
    assert captured["max_tokens"] == 7
    assert captured["temperature"] == 0.2


def test_chat_completion_passes_response_format_to_litellm(monkeypatch):
    captured = {}

    def fake_completion(**kwargs):
        captured.update(kwargs)
        return DummyResponse(parsed=StructuredPayload(applicable=True, reason="ok"))

    def fail_openai_client(*args, **kwargs):
        raise AssertionError("OpenAI client should not be constructed")

    monkeypatch.setattr(client_module, "completion", fake_completion, raising=False)
    monkeypatch.setattr(client_module, "OpenAI", fail_openai_client, raising=False)

    client = BatchClient(model="gpt-4.1-mini")
    response = client.chat_completion(
        messages=[{"role": "user", "content": "hello"}],
        response_format=StructuredPayload,
    )

    assert captured["response_format"] is StructuredPayload
    assert response == StructuredPayload(applicable=True, reason="ok")


def test_chat_completion_tracks_usage_and_cost(monkeypatch):
    def fake_completion(**kwargs):
        return DummyResponse(
            content="2",
            usage={
                "prompt_tokens": 11,
                "completion_tokens": 7,
                "total_tokens": 18,
            },
        )

    monkeypatch.setattr(client_module, "completion", fake_completion, raising=False)
    monkeypatch.setattr(client_module, "completion_cost", lambda **kwargs: 0.123456, raising=False)

    tracker = LLMUsageTracker()
    client = BatchClient(
        model="openai/gpt-4.1-mini",
        usage_tracker=tracker,
        request_tag="barge-in",
    )

    response = client.chat_completion(messages=[{"role": "user", "content": "hello"}])
    snapshot = tracker.snapshot()

    assert response == "2"
    assert snapshot.total.requests == 1
    assert snapshot.total.prompt_tokens == 11
    assert snapshot.total.completion_tokens == 7
    assert snapshot.total.total_tokens == 18
    assert snapshot.total.estimated_cost_usd == pytest.approx(0.123456)
    assert snapshot.by_stage["barge-in"].requests == 1
    assert snapshot.by_stage["barge-in"].estimated_cost_usd == pytest.approx(0.123456)


def test_augment_cli_prints_connection_help(monkeypatch, tmp_path):
    data_dir = tmp_path / "datasets"
    saa_root = data_dir / "SpeechAccentArchive"
    saa_root.mkdir(parents=True)
    (saa_root / "speakers_all.csv").write_text("speaker_id\n", encoding="utf-8")
    (saa_root / "recordings").mkdir()
    monkeypatch.setenv("LITELLM_API_BASE", "http://localhost:4000/v1")

    class FakePipeline:
        def __init__(self, config):
            self.config = config

        def run(self, splits):
            raise APIConnectionError(
                message="Connection error.",
                llm_provider="openai",
                model="gpt-4.1-mini",
                request=Request("POST", "http://localhost:4000/v1/chat/completions"),
            )

    monkeypatch.setattr(augment, "AugmentationPipeline", FakePipeline)

    result = CliRunner().invoke(
        augment.main,
        [
            "--datasets",
            "saa",
            "--data-dir",
            str(data_dir),
            "--output-dir",
            str(tmp_path / "out"),
            "--sample-size",
            "1",
            "--chunk-size",
            "1",
            "--model",
            "Qwen/Qwen3-32B",
        ],
    )

    assert result.exit_code != 0
    assert "Failed to connect to API for model 'gpt-4.1-mini'" in result.output
    assert "http://localhost:4000/v1" in result.output


def test_augment_cli_prints_cost_summary(monkeypatch, tmp_path):
    data_dir = tmp_path / "datasets"
    saa_root = data_dir / "SpeechAccentArchive"
    saa_root.mkdir(parents=True)
    (saa_root / "speakers_all.csv").write_text("speaker_id\n", encoding="utf-8")
    (saa_root / "recordings").mkdir()

    monkeypatch.setattr(client_module, "completion_cost", lambda **kwargs: 0.5, raising=False)

    tracker = LLMUsageTracker()
    tracker.record_response(
        DummyResponse(
            content="2",
            usage={
                "prompt_tokens": 20,
                "completion_tokens": 5,
                "total_tokens": 25,
            },
        ),
        model="openai/gpt-4.1-mini",
        stage="barge-in",
    )

    class FakePipeline:
        def __init__(self, config):
            self.llm_usage_tracker = tracker

        def run(self, splits):
            return {"saa": 1}

    monkeypatch.setattr(augment, "AugmentationPipeline", FakePipeline)

    result = CliRunner().invoke(
        augment.main,
        [
            "--datasets",
            "saa",
            "--data-dir",
            str(data_dir),
            "--output-dir",
            str(tmp_path / "out"),
            "--sample-size",
            "1",
            "--chunk-size",
            "1",
            "--model",
            "openai/gpt-4.1-mini",
        ],
    )

    assert result.exit_code == 0
    assert "saa" in result.output
    assert "Dataset Counts" in result.output
    assert "LLM Requests" in result.output
    assert "1 (barge-in: 1, disfluency: 0, emotion: 0)" in result.output
    assert "25 (barge-in: 25, disfluency: 0, emotion: 0)" in result.output
    assert "Estimated Cost" in result.output
    assert "$0.500000 (barge-in: $0.500000, disfluency: $0.000000," in result.output
    assert "emotion: $0.000000)" in result.output


def test_batch_completion_async_raises_when_all_requests_fail(monkeypatch):
    def fake_completion(**kwargs):
        raise APIConnectionError(
            message="Connection error.",
            llm_provider="openai",
            model="gpt-4.1-mini",
            request=Request("POST", "http://localhost:4000/v1/chat/completions"),
        )

    monkeypatch.setattr(client_module, "completion", fake_completion, raising=False)

    client = BatchClient(model="gpt-4.1-mini", api_base="http://localhost:4000/v1")

    with pytest.raises(APIConnectionError):
        asyncio.run(
            client.batch_completion_async(
                [{"messages": [{"role": "user", "content": "hello"}]}],
                max_concurrency=1,
            )
        )


def test_batch_completion_async_does_not_use_litellm_acompletion(monkeypatch):
    async def fail_acompletion(**kwargs):
        raise AssertionError("acompletion should not be used for augmentation batch calls")

    def fake_completion(**kwargs):
        return DummyResponse(content="2")

    monkeypatch.setattr(client_module, "acompletion", fail_acompletion, raising=False)
    monkeypatch.setattr(client_module, "completion", fake_completion, raising=False)

    client = BatchClient(model="gpt-4.1-mini")
    result = asyncio.run(
        client.batch_completion_async(
            [{"messages": [{"role": "user", "content": "hello"}]}],
            max_concurrency=1,
        )
    )

    assert result == ["2"]
