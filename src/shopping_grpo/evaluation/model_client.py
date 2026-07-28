"""Minimal OpenAI-compatible JSON client for frozen Rubric/Judge prompts."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from copy import deepcopy
import json
import os
from http.client import RemoteDisconnected
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


DEFAULT_FLASH_MODEL = "deepseek-v4-flash"
DEFAULT_PRO_MODEL = "deepseek-v4-pro"


class ModelResponseError(ValueError):
    """Raised when a provider response cannot satisfy the JSON contract."""


class OpenAIJSONClient:
    """Call one chat-completions model without serializing credentials."""

    def __init__(
        self,
        *,
        model: str,
        base_url: str,
        api_key: str,
        max_tokens: int = 4096,
        timeout: float = 120,
        retries: int = 2,
        retry_delay_seconds: float = 2,
        response_format_json: bool = False,
        transport: Callable | None = None,
    ):
        if not str(model).strip():
            raise ValueError("model is required")
        if not str(base_url).strip():
            raise ValueError("base_url is required")
        if not str(api_key):
            raise ValueError("api_key is required")
        if int(max_tokens) < 1:
            raise ValueError("max_tokens must be positive")
        if int(retries) < 0:
            raise ValueError("retries cannot be negative")
        self.model = str(model)
        self.base_url = str(base_url).rstrip("/")
        self.api_key = str(api_key)
        self.max_tokens = int(max_tokens)
        self.timeout = float(timeout)
        self.retries = int(retries)
        self.retry_delay_seconds = float(retry_delay_seconds)
        self.response_format_json = bool(response_format_json)
        self.transport = transport

    def _request_payload(self, messages: list[Mapping]) -> dict:
        payload = {
            "model": self.model,
            "messages": deepcopy(messages),
            "temperature": 0.0,
            "top_p": 1.0,
            "max_tokens": self.max_tokens,
        }
        if self.model.casefold().startswith("deepseek-v4"):
            payload["thinking"] = {"type": "disabled"}
        if self.response_format_json:
            payload["response_format"] = {"type": "json_object"}
        return payload

    def complete_json(self, messages: list[Mapping]) -> dict:
        """Return parsed JSON plus non-secret request metadata."""

        if not isinstance(messages, list) or not messages:
            raise ValueError("messages must be a non-empty list")
        payload = self._request_payload(messages)
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
            "User-Agent": "shopping-grpo-longhorizon-evaluator/0.1",
        }
        url = f"{self.base_url}/chat/completions"
        started = time.monotonic()
        response = None
        attempts = 0
        for attempt in range(self.retries + 1):
            attempts = attempt + 1
            try:
                if self.transport is not None:
                    response = self.transport(
                        url,
                        payload,
                        headers,
                        self.timeout,
                    )
                else:
                    request = Request(
                        url,
                        data=json.dumps(payload).encode("utf-8"),
                        headers=headers,
                        method="POST",
                    )
                    with urlopen(request, timeout=self.timeout) as raw:
                        response = json.loads(raw.read().decode("utf-8"))
                break
            except HTTPError:
                raise
            except (RemoteDisconnected, TimeoutError, URLError):
                if attempt >= self.retries:
                    raise
                if self.retry_delay_seconds > 0:
                    time.sleep(self.retry_delay_seconds * (attempt + 1))
        latency = time.monotonic() - started
        if not isinstance(response, Mapping):
            raise ModelResponseError("provider response must be an object")
        choices = response.get("choices")
        if not isinstance(choices, list) or not choices:
            raise ModelResponseError("provider response is missing choices")
        first = choices[0]
        message = first.get("message") if isinstance(first, Mapping) else None
        content = message.get("content") if isinstance(message, Mapping) else None
        if not isinstance(content, str) or not content.strip():
            raise ModelResponseError(
                "provider response message.content must be non-empty JSON text"
            )
        try:
            result = json.loads(content)
        except json.JSONDecodeError as exc:
            raise ModelResponseError(
                "provider response content is not strict JSON"
            ) from exc
        if not isinstance(result, dict):
            raise ModelResponseError(
                "provider response JSON root must be an object"
            )
        usage = response.get("usage")
        usage = deepcopy(dict(usage)) if isinstance(usage, Mapping) else {}
        return {
            "result": result,
            "metadata": {
                "provider_request_id": response.get("id"),
                "provider_model": response.get("model") or self.model,
                "requested_model": self.model,
                "attempts": attempts,
                "latency_seconds": latency,
                "usage": usage,
            },
        }


def client_from_environment(
    *,
    model: str,
    max_tokens: int,
    timeout: float = 120,
    retries: int = 2,
    response_format_json: bool = False,
) -> OpenAIJSONClient:
    """Use the same environment-variable convention as Teacher collection."""

    base_url = os.environ.get("OPENAI_BASE_URL")
    api_key = os.environ.get("OPENAI_API_KEY")
    if not base_url:
        raise ValueError("OPENAI_BASE_URL is required")
    if not api_key:
        raise ValueError("OPENAI_API_KEY is required")
    return OpenAIJSONClient(
        model=model,
        base_url=base_url,
        api_key=api_key,
        max_tokens=max_tokens,
        timeout=timeout,
        retries=retries,
        response_format_json=response_format_json,
    )
