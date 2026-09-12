"""LiteLLM client helpers for augmentation tasks."""

import asyncio
import os
import re
import threading
from dataclasses import dataclass
from typing import Any, TypeVar

from litellm import completion, completion_cost
from litellm.exceptions import APIConnectionError
from loguru import logger
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)

_JSON_FENCE_RE = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.DOTALL)


@dataclass
class LLMUsageStats:
    requests: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    estimated_cost_usd: float = 0.0


@dataclass
class LLMUsageSnapshot:
    total: LLMUsageStats
    by_stage: dict[str, LLMUsageStats]


def _usage_field(usage: Any, field_name: str) -> int:
    if usage is None:
        return 0
    if isinstance(usage, dict):
        value = usage.get(field_name, 0)
    else:
        value = getattr(usage, field_name, 0)
    return int(value or 0)


class LLMUsageTracker:
    """Thread-safe usage and cost tracker for LiteLLM responses."""

    def __init__(self):
        self._total = LLMUsageStats()
        self._by_stage: dict[str, LLMUsageStats] = {}
        self._lock = threading.Lock()

    def record_response(self, response: Any, model: str, stage: str) -> None:
        usage = getattr(response, "usage", None)
        prompt_tokens = _usage_field(usage, "prompt_tokens")
        completion_tokens = _usage_field(usage, "completion_tokens")
        total_tokens = _usage_field(usage, "total_tokens")

        try:
            estimated_cost = float(
                completion_cost(
                    completion_response=response,
                    model=model,
                )
            )
        except Exception:
            estimated_cost = 0.0

        with self._lock:
            stage_stats = self._by_stage.setdefault(stage, LLMUsageStats())
            self._accumulate(
                self._total,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                estimated_cost=estimated_cost,
            )
            self._accumulate(
                stage_stats,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                estimated_cost=estimated_cost,
            )

    def snapshot(self) -> LLMUsageSnapshot:
        with self._lock:
            total = LLMUsageStats(**vars(self._total))
            by_stage = {
                stage: LLMUsageStats(**vars(stats))
                for stage, stats in self._by_stage.items()
            }
        return LLMUsageSnapshot(total=total, by_stage=by_stage)

    @staticmethod
    def _accumulate(
        stats: LLMUsageStats,
        prompt_tokens: int,
        completion_tokens: int,
        total_tokens: int,
        estimated_cost: float,
    ) -> None:
        stats.requests += 1
        stats.prompt_tokens += prompt_tokens
        stats.completion_tokens += completion_tokens
        stats.total_tokens += total_tokens
        stats.estimated_cost_usd += estimated_cost


class BatchClient:
    """Thin LiteLLM wrapper used across augmentation flows."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gpt-4.1-mini",
        max_retries: int = 3,
        api_base: str | None = None,
        usage_tracker: LLMUsageTracker | None = None,
        request_tag: str = "default",
    ):
        self._api_key = api_key
        self.api_base = api_base or os.getenv("LITELLM_API_BASE")
        self.model = model
        self.max_retries = max_retries
        self.usage_tracker = usage_tracker
        self.request_tag = request_tag

    def _build_request_kwargs(
        self,
        messages: list[dict[str, Any]],
        max_tokens: int,
        temperature: float,
        response_format: type[T] | None = None,
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if response_format is not None:
            kwargs["response_format"] = response_format
        if self._api_key:
            kwargs["api_key"] = self._api_key
        if self.api_base:
            kwargs["base_url"] = self.api_base
        return kwargs

    def _extract_content(self, response: Any) -> str:
        message = response.choices[0].message
        content = getattr(message, "content", "")

        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                    continue
                if isinstance(item, dict):
                    text = item.get("text")
                    if isinstance(text, str):
                        parts.append(text)
                    continue
                text = getattr(item, "text", None)
                if isinstance(text, str):
                    parts.append(text)
            return "".join(parts)
        return str(content or "")

    def _parse_structured_content(
        self,
        content: str,
        response_format: type[T],
    ) -> T:
        raw_content = content.strip()
        fenced = _JSON_FENCE_RE.match(raw_content)
        if fenced:
            raw_content = fenced.group(1).strip()
        return response_format.model_validate_json(raw_content)

    def _coerce_response(
        self,
        response: Any,
        response_format: type[T] | None,
    ) -> str | T:
        if response_format is None:
            return self._extract_content(response)
        parsed = getattr(response.choices[0].message, "parsed", None)
        if parsed is not None:
            return parsed
        content = self._extract_content(response)
        return self._parse_structured_content(content, response_format)

    def _build_error_response(
        self,
        response_format: type[T],
        error: Exception,
    ) -> T:
        payload: dict[str, Any] = {}
        if "applicable" in response_format.model_fields:
            payload["applicable"] = False
        if "reason" in response_format.model_fields:
            payload["reason"] = f"API Error: {self._format_request_error(error)}"
        return response_format.model_validate(payload)

    def _format_request_error(self, error: Exception) -> str:
        if isinstance(error, APIConnectionError):
            request_url = getattr(getattr(error, "request", None), "url", None)
            model_name = getattr(error, "model", self.model)
            endpoint = self.api_base or request_url or "configured provider endpoint"
            return (
                f"Failed to connect to API for model '{model_name}' at {endpoint}. "
                "Check network connectivity and endpoint configuration."
            )
        return str(error)

    def chat_completion(
        self,
        messages: list[dict],
        max_tokens: int = 5,
        temperature: float = 0.0,
        response_format: type[T] | None = None,
    ) -> str | T:
        response = completion(
            **self._build_request_kwargs(
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                response_format=response_format,
            )
        )
        self._record_response_usage(response)
        return self._coerce_response(response, response_format)

    async def chat_completion_async(
        self,
        messages: list[dict],
        max_tokens: int = 5,
        temperature: float = 0.0,
        response_format: type[T] | None = None,
    ) -> str | T:
        response = await asyncio.to_thread(
            completion,
            **self._build_request_kwargs(
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                response_format=response_format,
            )
        )
        self._record_response_usage(response)
        return self._coerce_response(response, response_format)

    def _record_response_usage(self, response: Any) -> None:
        if self.usage_tracker is None:
            return
        self.usage_tracker.record_response(
            response=response,
            model=self.model,
            stage=self.request_tag,
        )

    async def batch_completion_async(
        self,
        request_list: list[dict],
        max_concurrency: int = 50,
        response_format: type[T] | None = None,
    ) -> list[str | T]:
        sem = asyncio.Semaphore(max_concurrency)
        errors: list[Exception] = []

        async def _process_single(req: dict) -> str | T:
            async with sem:
                try:
                    return await self.chat_completion_async(
                        messages=req["messages"],
                        max_tokens=req.get("max_tokens", 5),
                        temperature=req.get("temperature", 0.0),
                        response_format=response_format,
                    )
                except Exception as error:
                    errors.append(error)
                    if response_format is not None:
                        return self._build_error_response(response_format, error)
                    return ""

        tasks = [_process_single(req) for req in request_list]
        results = await asyncio.gather(*tasks)

        if errors:
            if len(errors) == len(request_list):
                raise errors[0]
            logger.error(
                f"{len(errors)}/{len(request_list)} async requests failed for model "
                f"{self.model}. First error: {self._format_request_error(errors[0])}"
            )

        return results


class MockBatchClient:
    """Mock client for testing without external API calls."""

    def __init__(self, default_emotion: int = 0):
        self.default_emotion = default_emotion

    def chat_completion(
        self,
        messages: list[dict],
        **kwargs,
    ) -> str:
        return str(self.default_emotion)

    async def batch_completion_async(
        self,
        request_list: list[dict],
        **kwargs,
    ) -> list[str]:
        return [str(self.default_emotion) for _ in request_list]
