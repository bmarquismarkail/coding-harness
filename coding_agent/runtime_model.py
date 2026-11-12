"""Recorded, cancellable llama-server requests for durable jobs."""
from __future__ import annotations

import asyncio
import os
import time

import httpx

from .model import ModelError
from .types import action_schema


class RecoverableResponseError(ModelError):
    def __init__(self, category: str, message: str):
        super().__init__(message)
        self.category = category


class RecordingProvider:
    def __init__(self, config, store, run_id, cancelled):
        self.config, self.store, self.run_id, self.cancelled = config, store, run_id, cancelled

    def invoke(self, messages, *, role, schema=None):
        self.store.consume(self.run_id, kind="model")
        settings = {"main": self.config.main, "subagent": self.config.subagents,
                    "reviewer": self.config.reviewer}.get(role)
        if settings is None:
            raise ModelError(f"unknown model role: {role}")
        response_format = self.response_format(role) if schema is None else {"type": "json_object", "schema": schema}
        request = {"role": role, "model": settings.model, "base_url": settings.base_url,
                   "max_tokens": settings.max_tokens, "enable_thinking": settings.enable_thinking,
                   "messages": messages, "response_format": response_format}
        artifact = self.store.artifact(self.run_id, request)
        self.store.event(self.run_id, "model_started", {"role": role, "request_artifact": artifact})
        started = time.monotonic()
        result_id = None
        try:
            body = asyncio.run(self._request(settings, messages, role, response_format))
            record = {"request_artifact": artifact, "response": body, "http_status": 200,
                      "elapsed_seconds": round(time.monotonic()-started, 3)}
            result_id = self.store.artifact(self.run_id, record)
            choice = body["choices"][0]
            finish_reason = choice.get("finish_reason")
            if finish_reason == "length":
                raise RecoverableResponseError("truncated", "Response reached the token limit. Return a shorter action.")
            if finish_reason != "stop":
                raise ModelError(f"model did not complete its response: {finish_reason}")
            content = choice.get("message", {}).get("content")
            if content is None or isinstance(content, str) and not content.strip():
                raise RecoverableResponseError("empty_content", "Response had no final content. Return one complete action.")
            if not isinstance(content, str):
                raise ModelError("model returned non-text final content")
            self.store.event(self.run_id, "model_completed", {"role": role, "response_artifact": result_id,
                                                            "usage": body.get("usage", {})})
            return content
        except Exception as exc:
            message = str(exc)
            for key in ("LLAMA_API_KEY", "OPENAI_API_KEY", "HARNESS_MCP_API_KEY"):
                if os.environ.get(key):
                    message = message.replace(os.environ[key], "[redacted]")
            failure = {"role": role, "error": message, "request_artifact": artifact}
            if result_id is not None:
                failure["response_artifact"] = result_id
            if isinstance(exc, RecoverableResponseError):
                failure["category"] = exc.category
            self.store.event(self.run_id, "model_failed", failure)
            if isinstance(exc, InterruptedError):
                raise InterruptedError(message) from None
            if isinstance(exc, RecoverableResponseError):
                raise RecoverableResponseError(exc.category, message) from None
            raise ModelError(message) from None

    async def _request(self, settings, messages, role, response_format):
        headers = {}
        if settings.api_key_env:
            key = os.environ.get(settings.api_key_env)
            if not key:
                raise ModelError(f"required credential environment variable {settings.api_key_env} is missing")
            headers["Authorization"] = f"Bearer {key}"
        payload = {"model": settings.model, "messages": messages, "temperature": settings.temperature,
                   "max_tokens": settings.max_tokens, "response_format": response_format}
        if role == "reviewer" and settings.model == "gpt-6-luna":
            payload["max_completion_tokens"] = payload.pop("max_tokens")
            del payload["temperature"]
        if role != "reviewer" and settings.reasoning_effort is not None:
            payload["reasoning_effort"] = settings.reasoning_effort
        if role != "reviewer" and settings.enable_thinking is not None:
            payload["chat_template_kwargs"] = {"enable_thinking": settings.enable_thinking}
        async with httpx.AsyncClient(timeout=settings.timeout_seconds) as client:
            started = time.monotonic()
            pending = asyncio.create_task(client.post(settings.base_url + "/chat/completions", headers=headers, json=payload))
            try:
                while not pending.done():
                    if self.cancelled():
                        raise InterruptedError("model request cancelled")
                    if time.monotonic()-started >= settings.timeout_seconds:
                        raise ModelError("model request wall-clock timeout exceeded")
                    status = self.store.status(self.run_id)
                    if status.active_seconds >= self.store.get(self.run_id)["profile"]["max_active_seconds"]:
                        raise ModelError("run active-time budget exhausted during model request")
                    await asyncio.wait({pending}, timeout=0.1)
                response = await pending
                if response.status_code >= 400:
                    raise ModelError(f"model endpoint returned HTTP {response.status_code}: {response.text[:1000]}")
                return response.json()
            finally:
                if not pending.done():
                    pending.cancel()
                    await asyncio.gather(pending, return_exceptions=True)

    def response_format(self, role="main"):
        value = {"type": "json_object"}
        # llama-server accepts the extended schema form; OpenAI's compatible
        # endpoint currently accepts JSON-object mode but rejects schema here.
        if role != "reviewer":
            run = self.store.get(self.run_id)
            if run["profile"].get("constrain_actions", True):
                value["schema"] = action_schema(
                    typed_findings=role == "main" and run["request"].get("mode") == "review"
                )
        return value
