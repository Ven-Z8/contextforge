from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

from contextforge.core.config import settings


class OpenRouterError(RuntimeError):
    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


@dataclass(frozen=True)
class LLMResponse:
    text: str
    input_tokens: int
    output_tokens: int
    response_id: str | None = None


class OpenRouterClient:
    """Small OpenRouter REST client for benchmark calls.

    The project intentionally avoids the OpenAI SDK. OpenRouter exposes an
    OpenAI-compatible HTTP schema, so direct HTTP is enough for this benchmark.
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        app_title: str = "ContextForge Benchmark",
        referer: str = "https://github.com/venkatesh/contextforge",
    ) -> None:
        self._api_key = settings.openrouter_api_key if api_key is None else api_key
        self._base_url = (base_url or settings.openrouter_base_url).rstrip("/")
        self._app_title = app_title
        self._referer = referer

    def chat(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        max_tokens: int = 256,
        temperature: float = 0.0,
        response_format: dict[str, Any] | None = None,
    ) -> LLMResponse:
        if not self._api_key:
            raise OpenRouterError("CONTEXTFORGE_OPENROUTER_API_KEY is required")

        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if response_format:
            payload["response_format"] = response_format

        req = urllib.request.Request(
            f"{self._base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": self._referer,
                "X-OpenRouter-Title": self._app_title,
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise OpenRouterError(body or str(exc), status_code=exc.code) from exc
        except urllib.error.URLError as exc:
            raise OpenRouterError(str(exc)) from exc

        try:
            text = data["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError) as exc:
            raise OpenRouterError(f"Malformed OpenRouter response: {data}") from exc

        usage = data.get("usage") or {}
        return LLMResponse(
            text=text,
            input_tokens=int(usage.get("prompt_tokens") or 0),
            output_tokens=int(usage.get("completion_tokens") or 0),
            response_id=data.get("id"),
        )


def retry_delay(attempt: int) -> float:
    return min(60.0, (2**attempt) + (time.monotonic() % 1.0))
