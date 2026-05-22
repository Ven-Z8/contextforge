import json
import urllib.error
from io import BytesIO

import pytest

from contextforge.llm import OpenRouterClient, OpenRouterError


class _FakeResponse:
    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def read(self):
        return json.dumps(
            {
                "id": "gen-123",
                "choices": [{"message": {"content": "answer"}}],
                "usage": {"prompt_tokens": 11, "completion_tokens": 3},
            }
        ).encode("utf-8")


def test_openrouter_client_parses_chat_response(monkeypatch):
    captured = {}

    def fake_urlopen(req, timeout):
        captured["url"] = req.full_url
        captured["timeout"] = timeout
        captured["body"] = json.loads(req.data.decode("utf-8"))
        return _FakeResponse()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    client = OpenRouterClient(api_key="test-key")

    result = client.chat(
        model="deepseek/deepseek-v4-flash",
        messages=[{"role": "user", "content": "hi"}],
    )

    assert captured["url"].endswith("/chat/completions")
    assert captured["timeout"] == 120
    assert captured["body"]["model"] == "deepseek/deepseek-v4-flash"
    assert result.text == "answer"
    assert result.input_tokens == 11
    assert result.output_tokens == 3
    assert result.response_id == "gen-123"


def test_openrouter_client_requires_api_key():
    client = OpenRouterClient(api_key="")

    with pytest.raises(OpenRouterError, match="OPENROUTER_API_KEY"):
        client.chat(model="deepseek/deepseek-v4-flash", messages=[])


def test_openrouter_client_wraps_http_errors(monkeypatch):
    def fake_urlopen(_req, **_kwargs):
        raise urllib.error.HTTPError(
            url="https://openrouter.ai/api/v1/chat/completions",
            code=429,
            msg="rate limited",
            hdrs={},
            fp=BytesIO(b"rate limited"),
        )

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    client = OpenRouterClient(api_key="test-key")

    with pytest.raises(OpenRouterError) as exc:
        client.chat(model="deepseek/deepseek-v4-flash", messages=[])

    assert exc.value.status_code == 429
