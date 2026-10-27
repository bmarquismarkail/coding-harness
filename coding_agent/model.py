from __future__ import annotations

import os
from typing import Any, Mapping, Protocol, Sequence

import httpx

from .config import AgentConfig, ModelSettings


class ModelError(RuntimeError):
    """Raised when an OpenAI-compatible model cannot produce a response."""


class ModelProvider(Protocol):
    def invoke(self, messages: Sequence[Mapping[str, str]], *, role: str) -> str:
        ...


class OpenAICompatibleModel:
    """Minimal provider boundary for llama.cpp, vLLM, Ollama-compatible proxies, etc."""

    def __init__(self, settings: ModelSettings, *, json_mode: bool = True, client: httpx.Client | None = None):
        self.settings = settings
        self.json_mode = json_mode
        self._client = client

    def invoke(self, messages: Sequence[Mapping[str, str]]) -> str:
        headers = {"Content-Type": "application/json"}
        if self.settings.api_key_env:
            api_key = os.environ.get(self.settings.api_key_env)
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"
        payload: dict[str, Any] = {
            "model": self.settings.model,
            "messages": list(messages),
            "temperature": self.settings.temperature,
            "max_tokens": self.settings.max_tokens,
        }
        if self.json_mode:
            payload["response_format"] = {"type": "json_object"}
        if self.settings.reasoning_effort is not None:
            payload["reasoning_effort"] = self.settings.reasoning_effort
        if self.settings.enable_thinking is not None:
            payload["chat_template_kwargs"] = {"enable_thinking": self.settings.enable_thinking}
        url = self.settings.base_url.rstrip("/") + "/chat/completions"
        client = self._client or httpx.Client(timeout=self.settings.timeout_seconds)
        should_close = self._client is None
        try:
            try:
                response = client.post(url, headers=headers, json=payload)
            except httpx.HTTPError as exc:
                raise ModelError(f"request to {url} failed: {exc}") from exc
            if response.status_code >= 400:
                detail = response.text[:1000]
                raise ModelError(f"model endpoint returned HTTP {response.status_code}: {detail}")
            try:
                data = response.json()
                choice = data["choices"][0]
                content = choice["message"]["content"]
            except (ValueError, KeyError, IndexError, TypeError) as exc:
                raise ModelError("model response did not contain choices[0].message.content") from exc
            if choice.get("finish_reason") == "length":
                raise ModelError("model exhausted max_tokens before completing its answer (finish_reason=length)")
            if isinstance(content, list):
                content = "".join(
                    str(part.get("text", "")) if isinstance(part, dict) else str(part)
                    for part in content
                )
            if not isinstance(content, str) or not content.strip():
                raise ModelError("model response contained empty content")
            return content
        finally:
            if should_close:
                client.close()


class DualModelProvider:
    def __init__(self, config: AgentConfig):
        self.main = OpenAICompatibleModel(config.main, json_mode=config.json_mode)
        self.subagents = OpenAICompatibleModel(config.subagents, json_mode=config.json_mode)

    def invoke(self, messages: Sequence[Mapping[str, str]], *, role: str) -> str:
        return (self.main if role == "main" else self.subagents).invoke(messages)
