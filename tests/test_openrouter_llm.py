import json

import pytest

from thirawat_mapper_beta.models.rag_llm import OpenRouterConfig, OpenRouterLLMClient


def test_openrouter_config_uses_env(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    cfg = OpenRouterConfig(api_key=None)
    assert cfg.resolve_api_key() == "test-key"


def test_openrouter_client_builds_request(monkeypatch):
    captured = {}

    class DummyResponse:
        def __init__(self, body: str) -> None:
            self._body = body.encode("utf-8")
            self.headers = {"content-type": "application/json"}

        def read(self) -> bytes:
            return self._body

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            return None

    def fake_urlopen(req, *args, **kwargs):  # type: ignore[override]
        captured["url"] = req.full_url
        captured["headers"] = dict(req.headers)
        captured["payload"] = json.loads(req.data.decode("utf-8"))
        body = json.dumps({"choices": [{"message": {"content": "A, B, C"}}]})
        return DummyResponse(body)

    monkeypatch.setenv("OPENROUTER_API_KEY", "secret")
    monkeypatch.setattr("thirawat_mapper_beta.models.rag_llm.request.urlopen", fake_urlopen)

    cfg = OpenRouterConfig()
    client = OpenRouterLLMClient(cfg)
    output = client.generate("Rank these")

    assert output == "A, B, C"
    assert captured["url"].endswith("/chat/completions")
    assert captured["headers"]["Authorization"].startswith("Bearer ")
    assert all(key.lower() not in {"http-referer", "referer", "x-title"} for key in captured["headers"].keys())
    assert captured["payload"]["model"] == cfg.model_name
    assert captured["payload"]["messages"][0]["role"] == "system"
    assert "temperature" not in captured["payload"]
    assert "top_p" not in captured["payload"]
