"""LLM client helpers for the beta RAG pipeline."""

from __future__ import annotations

import json
import os
import logging
from dataclasses import dataclass
from typing import Dict, Optional, Protocol, Sequence
from urllib import error, request

LOGGER = logging.getLogger(__name__)


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
            raise ValueError("Cloudflare account ID is required (set CLOUDFLARE_ACCOUNT_ID or pass --cloudflare-account-id).")
        return account_id

    def resolve_api_token(self) -> str:
        token = self.api_token or os.getenv("CLOUDFLARE_API_TOKEN")
        if not token:
            raise ValueError("Cloudflare API token is required (set CLOUDFLARE_API_TOKEN or pass --cloudflare-api-token).")
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
        messages = [
            {
                "role": "system",
                "content": (
                    "You help rank medical terminology candidates. "
                    "Reply with a comma-separated list or JSON array; do not add commentary."
                ),
            },
            {"role": "user", "content": prompt},
        ]
        payload: Dict[str, object]
        if endpoint.endswith("/responses"):
            payload = {
                "model": self.cfg.model_name,
                "input": messages,
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
            payload = {"input": messages}
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
            chunks = []
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
                message = event.get("response") or event.get("text") or event.get("output_text")
                if message:
                    chunks.append(str(message))
            text = "".join(chunks).strip()
        else:
            try:
                response = json.loads(raw) if raw else {}
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Cloudflare API returned invalid JSON: {raw}") from exc
            result = response.get("result") if isinstance(response, dict) else None
            text: Optional[str] = None
            if isinstance(result, dict):
                text = (
                    result.get("response")
                    or result.get("output_text")
                    or result.get("text")
                    or result.get("content")
                )
                if text is None and isinstance(result.get("messages"), list):
                    for message in reversed(result["messages"]):
                        if isinstance(message, dict) and message.get("role") == "assistant":
                            content = message.get("content")
                            if isinstance(content, str):
                                text = content
                                break
            if text is None and isinstance(response, dict):
                text = response.get("response") or response.get("output_text") or response.get("text")
            if text is None:
                raise RuntimeError(f"Cloudflare API response missing text payload: {raw}")
        output = text.strip()
        if stop:
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
    temperature: float = 0.0
    top_p: float = 0.95


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
        temp = float(self.cfg.temperature if temperature is None else temperature)
        nucleus = float(self.cfg.top_p if top_p is None else top_p)

        if self.cfg.chat_format:
            messages = []
            if self.cfg.system_prompt:
                messages.append({"role": "system", "content": self.cfg.system_prompt})
            messages.append({"role": "user", "content": prompt})
            result = self._llama.create_chat_completion(
                messages=messages,
                max_tokens=max_tokens,
                temperature=max(temp, 0.0),
                top_p=nucleus,
                stop=list(stop) if stop else None,
            )
            choices = result.get("choices", [])
            if not choices:
                return ""
            message = choices[0].get("message", {})
            content = message.get("content", "")
            return str(content).strip()

        final_prompt = prompt
        if self.cfg.system_prompt:
            final_prompt = f"{self.cfg.system_prompt.strip()}\n\n{prompt}"
        result = self._llama.create_completion(
            prompt=final_prompt,
            max_tokens=max_tokens,
            temperature=max(temp, 0.0),
            top_p=nucleus,
            stop=list(stop) if stop else None,
        )
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
