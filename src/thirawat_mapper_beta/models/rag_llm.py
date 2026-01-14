"""LLM client helpers for the beta RAG pipeline."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol, Sequence
from urllib import error, request

LOGGER = logging.getLogger(__name__)

_STRUCTURED_RANK_SCHEMA = {
    "type": "object",
    "properties": {"concept_ids": {"type": "array", "items": {"type": "integer"}}},
    "required": ["concept_ids"],
    "additionalProperties": False,
}


def _build_plain_messages(system_prompt: str, user_prompt: str) -> List[Dict[str, str]]:
    """Return chat messages compatible with the Workers AI run endpoint."""

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def _build_responses_messages(system_prompt: str, user_prompt: str) -> List[Dict[str, str]]:
    """Return chat messages shaped for the Responses API."""

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def _extract_text_from_cf_payload(payload: Any) -> Optional[str]:
    """Best-effort extraction of assistant text from Workers AI payloads."""

    stack: List[Any] = [payload]
    visited: set[int] = set()
    priority_keys = ("response", "text", "output_text", "delta", "content", "value")
    nested_keys = ("messages", "output", "result", "choices")

    while stack:
        current = stack.pop()
        if current is None:
            continue
        if isinstance(current, str):
            stripped = current.strip()
            if stripped:
                return stripped
            continue
        current_id = id(current)
        if isinstance(current, (list, dict)) and current_id in visited:
            continue
        if isinstance(current, (list, dict)):
            visited.add(current_id)
        if isinstance(current, list):
            stack.extend(reversed(current))
            continue
        if isinstance(current, dict):
            for key in priority_keys:
                if key in current:
                    stack.append(current[key])
            for key in nested_keys:
                if key in current:
                    stack.append(current[key])
            for key, value in current.items():
                if key in priority_keys or key in nested_keys or key == "type":
                    continue
                stack.append(value)
            continue
    return None


class BaseLLMClient(Protocol):
    """Protocol describing the minimal interface required by the RAG pipeline."""

    def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        stop: Optional[Sequence[str]] = None,
    ) -> str:  # pragma: no cover - structural type
        """Return the generated text for the supplied prompt."""


# --- Cloudflare Workers AI -------------------------------------------------


@dataclass
class CloudflareConfig:
    account_id: Optional[str] = None
    api_token: Optional[str] = None
    model_name: str = "@cf/openai/gpt-oss-20b"
    base_url: str = "https://api.cloudflare.com/client/v4"
    max_tokens: int = 512
    temperature: float = 0.0
    top_p: float = 0.95
    use_responses_api: bool = True
    reasoning_effort: Optional[str] = None  # low|medium|high
    reasoning_summary: Optional[str] = None  # auto|concise|detailed

    def resolve_account_id(self) -> str:
        account_id = self.account_id or os.getenv("CLOUDFLARE_ACCOUNT_ID")
        if not account_id:
            raise ValueError("Cloudflare account ID is required; set CLOUDFLARE_ACCOUNT_ID in your environment.")
        return account_id

    def resolve_api_token(self) -> str:
        token = self.api_token or os.getenv("CLOUDFLARE_API_TOKEN")
        if not token:
            raise ValueError("Cloudflare API token is required; set CLOUDFLARE_API_TOKEN in your environment.")
        return token

    def endpoint(self) -> str:
        account_id = self.resolve_account_id()
        base = self.base_url.rstrip("/")
        if self.use_responses_api:
            return f"{base}/accounts/{account_id}/ai/v1/responses"
        model = self.model_name.strip()
        return f"{base}/accounts/{account_id}/ai/run/{model}"


class CloudflareLLMClient(BaseLLMClient):
    """Thin wrapper around Cloudflare Workers AI text generation endpoints."""

    def __init__(self, cfg: CloudflareConfig) -> None:
        self.cfg = cfg

    def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        stop: Optional[Sequence[str]] = None,
    ) -> str:
        if not prompt:
            raise ValueError("Prompt must be non-empty.")
        endpoint = self.cfg.endpoint()
        system_prompt = (
            "You help rank medical terminology candidates. "
            "Reply with a comma-separated list or JSON array; do not add commentary."
        )
        plain_messages = _build_plain_messages(system_prompt, prompt)
        payload: Dict[str, object]
        if endpoint.endswith("/responses"):
            payload = {
                "model": self.cfg.model_name,
                "input": _build_responses_messages(system_prompt, prompt),
            }
            effort = self.cfg.reasoning_effort
            summary = self.cfg.reasoning_summary
            if effort or summary:
                reasoning: Dict[str, str] = {}
                if effort:
                    reasoning["effort"] = effort
                if summary:
                    reasoning["summary"] = summary
                payload["reasoning"] = reasoning
        else:
            payload = {"messages": plain_messages}
        data = json.dumps(payload).encode("utf-8")
        headers = {
            "Authorization": f"Bearer {self.cfg.resolve_api_token()}",
            "Content-Type": "application/json",
        }
        req = request.Request(endpoint, data=data, headers=headers, method="POST")
        try:
            with request.urlopen(req) as resp:
                raw_bytes = resp.read()
                content_type = resp.headers.get("content-type", "")
        except error.HTTPError as exc:  # pragma: no cover - network
            detail = exc.read().decode("utf-8", errors="ignore") if hasattr(exc, "read") else ""
            message = detail or exc.reason
            raise RuntimeError(f"Cloudflare API error {exc.code}: {message}") from exc
        except error.URLError as exc:  # pragma: no cover - network
            raise RuntimeError(f"Failed to reach Cloudflare API: {exc.reason}") from exc

        raw = raw_bytes.decode("utf-8")
        if content_type.startswith("text/event-stream"):
            chunks: List[str] = []
            for line in raw.splitlines():
                if not line.startswith("data:"):
                    continue
                payload_str = line[5:].strip()
                if not payload_str or payload_str == "[DONE]":
                    continue
                try:
                    event = json.loads(payload_str)
                except json.JSONDecodeError:
                    continue
                chunk = _extract_text_from_cf_payload(event)
                if chunk:
                    chunks.append(chunk)
            text = "".join(chunks).strip()
        else:
            try:
                response = json.loads(raw) if raw else {}
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Cloudflare API returned invalid JSON: {raw}") from exc
            text = _extract_text_from_cf_payload(response)
            if text is None:
                raise RuntimeError(f"Cloudflare API response missing text payload: {raw}")
        output = text.strip()
        if stop:
            for token in stop:
                if token and token in output:
                    output = output.split(token, 1)[0].rstrip()
                    break
        return output


# --- OpenRouter (OpenAI-compatible REST API) --------------------------------


@dataclass
class OpenRouterConfig:
    api_key: Optional[str] = None
    model_name: str = "openrouter/polaris-alpha"
    base_url: str = "https://openrouter.ai/api/v1"
    structured: bool = True

    def resolve_api_key(self) -> str:
        token = self.api_key or os.getenv("OPENROUTER_API_KEY")
        if not token:
            raise ValueError("OpenRouter API key is required; set OPENROUTER_API_KEY in your environment.")
        return token

    def endpoint(self) -> str:
        return self.base_url.rstrip("/") + "/chat/completions"


class OpenRouterLLMClient(BaseLLMClient):
    """Client for the OpenRouter API (OpenAI-compatible chat completions)."""

    def __init__(self, cfg: OpenRouterConfig) -> None:
        self.cfg = cfg

    def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        stop: Optional[Sequence[str]] = None,
    ) -> str:
        if not prompt:
            raise ValueError("Prompt must be non-empty.")
        system_prompt = (
            "You help rank medical terminology candidates. "
            "Reply with JSON only: {\"concept_ids\":[...]} and no extra text."
        )
        payload: Dict[str, object] = {
            "model": self.cfg.model_name,
            "messages": _build_plain_messages(system_prompt, prompt),
            "temperature": 0.0,
        }
        if self.cfg.structured:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "ranking", "strict": True, "schema": _STRUCTURED_RANK_SCHEMA},
            }
        if max_new_tokens is not None:
            payload["max_tokens"] = int(max_new_tokens)
        if temperature is not None:
            payload["temperature"] = max(0.0, float(temperature))
        if top_p is not None:
            payload["top_p"] = float(top_p)
        if stop:
            payload["stop"] = list(stop)
        data = json.dumps(payload).encode("utf-8")
        headers = {
            "Authorization": f"Bearer {self.cfg.resolve_api_key()}",
            "Content-Type": "application/json",
        }
        req = request.Request(self.cfg.endpoint(), data=data, headers=headers, method="POST")
        try:
            with request.urlopen(req) as resp:
                raw_bytes = resp.read()
        except error.HTTPError as exc:  # pragma: no cover - network
            detail = exc.read().decode("utf-8", errors="ignore") if hasattr(exc, "read") else ""
            message = detail or exc.reason
            raise RuntimeError(f"OpenRouter API error {exc.code}: {message}") from exc
        except error.URLError as exc:  # pragma: no cover - network
            raise RuntimeError(f"Failed to reach OpenRouter API: {exc.reason}") from exc

        raw = raw_bytes.decode("utf-8") if raw_bytes else ""
        try:
            response = json.loads(raw) if raw else {}
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"OpenRouter API returned invalid JSON: {raw}") from exc
        choices = response.get("choices") if isinstance(response, dict) else None
        if not choices:
            raise RuntimeError(f"OpenRouter response missing choices: {raw}")
        message = choices[0].get("message") if isinstance(choices[0], dict) else None
        if not isinstance(message, dict):
            raise RuntimeError(f"OpenRouter response missing message: {raw}")
        content = message.get("content")
        if isinstance(content, list):
            # Some models can emit structured content blocks; join text entries.
            text_parts = [str(item.get("text", "")) for item in content if isinstance(item, dict)]
            output = "".join(text_parts).strip()
        else:
            output = str(content or "").strip()
        if stop and output:
            for token in stop:
                if token and token in output:
                    output = output.split(token, 1)[0].rstrip()
                    break
        return output


# --- Ollama (local REST API) -----------------------------------------------


@dataclass
class OllamaConfig:
    base_url: str = "http://localhost:11434"
    model: str = "qwen2:7b"
    timeout: int = 120
    keep_alive: Optional[str] = None
    temperature: float = 0.0
    top_p: float = 0.95

    def endpoint(self) -> str:
        return self.base_url.rstrip("/") + "/api/generate"


class OllamaLLMClient(BaseLLMClient):
    """Client for an Ollama-compatible local inference server."""

    def __init__(self, cfg: OllamaConfig) -> None:
        self.cfg = cfg

    def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        stop: Optional[Sequence[str]] = None,
    ) -> str:
        if not prompt:
            raise ValueError("Prompt must be non-empty.")
        payload: Dict[str, object] = {
            "model": self.cfg.model,
            "prompt": prompt,
            "stream": False,
        }
        if self.cfg.keep_alive:
            payload["keep_alive"] = self.cfg.keep_alive
        options: Dict[str, object] = {}
        if stop:
            options["stop"] = list(stop)
        if options:
            payload["options"] = options
        data = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        req = request.Request(self.cfg.endpoint(), data=data, headers=headers, method="POST")
        try:
            with request.urlopen(req, timeout=self.cfg.timeout) as resp:
                raw_bytes = resp.read()
        except error.HTTPError as exc:  # pragma: no cover - network
            detail = exc.read().decode("utf-8", errors="ignore") if hasattr(exc, "read") else ""
            raise RuntimeError(f"Ollama server returned {exc.code}: {detail or exc.reason}") from exc
        except error.URLError as exc:  # pragma: no cover - network
            raise RuntimeError(f"Failed to contact Ollama server: {exc.reason}") from exc

        raw = raw_bytes.decode("utf-8") if raw_bytes else ""
        try:
            response = json.loads(raw) if raw else {}
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Ollama server returned invalid JSON: {raw}") from exc
        text = (
            response.get("response")
            or response.get("output")
            or response.get("text")
            or ""
        )
        output = str(text).strip()
        return output


# --- llama.cpp -------------------------------------------------------------


@dataclass
class LlamaCppConfig:
    model_path: str
    n_ctx: int = 8192
    n_gpu_layers: int = -1
    n_threads: Optional[int] = None
    chat_format: Optional[str] = None
    system_prompt: Optional[str] = None
    max_tokens: int = 512
    temperature: Optional[float] = None
    top_p: Optional[float] = None


class LlamaCppLLMClient(BaseLLMClient):
    """Client using llama.cpp via the llama-cpp-python bindings."""

    def __init__(self, cfg: LlamaCppConfig) -> None:
        try:
            from llama_cpp import Llama  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError("llama-cpp-python is required for the llamacpp provider.") from exc
        self.cfg = cfg
        kwargs: Dict[str, object] = {
            "model_path": cfg.model_path,
            "n_ctx": int(cfg.n_ctx),
            "n_gpu_layers": int(cfg.n_gpu_layers),
        }
        if cfg.n_threads is not None:
            kwargs["n_threads"] = int(cfg.n_threads)
        if cfg.chat_format:
            kwargs["chat_format"] = cfg.chat_format
        self._llama = Llama(**kwargs)

    def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        stop: Optional[Sequence[str]] = None,
    ) -> str:
        if not prompt:
            raise ValueError("Prompt must be non-empty.")
        max_tokens = int(max_new_tokens or self.cfg.max_tokens)
        temp = temperature if temperature is not None else self.cfg.temperature
        nucleus = top_p if top_p is not None else self.cfg.top_p

        if self.cfg.chat_format:
            messages = []
            if self.cfg.system_prompt:
                messages.append({"role": "system", "content": self.cfg.system_prompt})
            messages.append({"role": "user", "content": prompt})
            chat_kwargs: Dict[str, object] = {
                "messages": messages,
                "max_tokens": max_tokens,
            }
            if stop:
                chat_kwargs["stop"] = list(stop)
            if temp is not None:
                chat_kwargs["temperature"] = max(0.0, float(temp))
            if nucleus is not None:
                chat_kwargs["top_p"] = float(nucleus)
            result = self._llama.create_chat_completion(**chat_kwargs)
            choices = result.get("choices", [])
            if not choices:
                return ""
            message = choices[0].get("message", {})
            content = message.get("content", "")
            return str(content).strip()

        final_prompt = prompt
        if self.cfg.system_prompt:
            final_prompt = f"{self.cfg.system_prompt.strip()}\n\n{prompt}"
        completion_kwargs: Dict[str, object] = {
            "prompt": final_prompt,
            "max_tokens": max_tokens,
        }
        if stop:
            completion_kwargs["stop"] = list(stop)
        if temp is not None:
            completion_kwargs["temperature"] = max(0.0, float(temp))
        if nucleus is not None:
            completion_kwargs["top_p"] = float(nucleus)
        result = self._llama.create_completion(**completion_kwargs)
        choices = result.get("choices", [])
        if not choices:
            return ""
        text = choices[0].get("text", "")
        return str(text).strip()


@dataclass
class LlamaCppServerConfig:
    base_url: str = "http://127.0.0.1:8080"
    timeout: int = 120
    model_name: Optional[str] = None
    cache_prompt: bool = True
    slot_id: Optional[int] = None

    def endpoint(self) -> str:
        return self.base_url.rstrip("/") + "/completion"


class LlamaCppServerLLMClient(BaseLLMClient):
    """HTTP client for llama.cpp's llama-server (completion endpoint)."""

    def __init__(self, cfg: LlamaCppServerConfig) -> None:
        self.cfg = cfg

    def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        stop: Optional[Sequence[str]] = None,
    ) -> str:
        if not prompt:
            raise ValueError("Prompt must be non-empty.")

        payload: Dict[str, object] = {
            "prompt": prompt,
            "stream": False,
        }
        if self.cfg.model_name:
            payload["model"] = self.cfg.model_name
        if self.cfg.cache_prompt:
            payload["cache_prompt"] = True
        if self.cfg.slot_id is not None:
            payload["id_slot"] = int(self.cfg.slot_id)
        if max_new_tokens is not None:
            payload["n_predict"] = max_new_tokens
        if temperature is not None:
            payload["temperature"] = max(0.0, float(temperature))
        if top_p is not None:
            payload["top_p"] = float(top_p)
        if stop:
            payload["stop"] = list(stop)

        data = json.dumps(payload).encode("utf-8")
        req = request.Request(
            self.cfg.endpoint(),
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=self.cfg.timeout) as resp:
                raw = resp.read()
        except error.URLError as exc:  # pragma: no cover - network
            raise RuntimeError(f"Failed to reach llama.cpp server: {exc.reason}") from exc
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="ignore") if hasattr(exc, "read") else exc.reason
            raise RuntimeError(f"llama.cpp server returned {exc.code}: {detail}") from exc

        try:
            response = json.loads(raw) if raw else {}
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"llama.cpp server returned invalid JSON: {raw}") from exc

        text = (
            response.get("content")
            or response.get("completion")
            or response.get("response")
        )
        if not text and isinstance(response.get("results"), list):
            parts = []
            for result in response["results"]:
                if isinstance(result, dict):
                    parts.append(str(result.get("content") or result.get("text") or ""))
            text = "".join(parts)
        if not text:
            return ""
        return str(text).strip()


__all__ = [
    "BaseLLMClient",
    "CloudflareConfig",
    "CloudflareLLMClient",
    "OllamaConfig",
    "OllamaLLMClient",
    "LlamaCppConfig",
    "LlamaCppLLMClient",
    "LlamaCppServerConfig",
    "LlamaCppServerLLMClient",
]
