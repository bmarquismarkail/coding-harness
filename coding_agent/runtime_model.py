"""Recorded, cancellable llama-server requests for durable jobs."""
from __future__ import annotations

import asyncio
import os
import time

import httpx

from .model import ModelError
from .types import action_schema


class RecordingProvider:
    def __init__(self, config, store, run_id, cancelled):
        self.config, self.store, self.run_id, self.cancelled = config, store, run_id, cancelled

    def invoke(self, messages, *, role):
        self.store.consume(self.run_id, kind="model")
        settings = self.config.main if role == "main" else self.config.subagents
        request = {"role": role, "model": settings.model, "base_url": settings.base_url,
                   "max_tokens": settings.max_tokens, "enable_thinking": settings.enable_thinking,
                   "messages": messages, "response_format": self.response_format()}
        artifact = self.store.artifact(self.run_id, request)
        self.store.event(self.run_id, "model_started", {"role": role, "request_artifact": artifact})
        started = time.monotonic()
        try:
            body = asyncio.run(self._request(settings, messages))
            record = {"request_artifact": artifact, "response": body, "http_status": 200,
                      "elapsed_seconds": round(time.monotonic()-started, 3)}
            result_id = self.store.artifact(self.run_id, record)
            choice = body["choices"][0]
            if choice.get("finish_reason") != "stop":
                raise ModelError(f"model did not complete its response: {choice.get('finish_reason')}")
            content = choice["message"]["content"]
            if not isinstance(content, str) or not content.strip():
                raise ModelError("model returned no final content")
            self.store.event(self.run_id, "model_completed", {"role": role, "response_artifact": result_id,
                                                            "usage": body.get("usage", {})})
            return content
        except Exception as exc:
            message = str(exc)
            for key in ("LLAMA_API_KEY", "HARNESS_MCP_API_KEY"):
                if os.environ.get(key):
                    message = message.replace(os.environ[key], "[redacted]")
            self.store.event(self.run_id, "model_failed", {"role": role, "error": message, "request_artifact": artifact})
            if isinstance(exc, InterruptedError):
                raise InterruptedError(message) from None
            raise ModelError(message) from None

    async def _request(self, settings, messages):
        headers = {}
        if settings.api_key_env:
            key = os.environ.get(settings.api_key_env)
            if not key:
                raise ModelError(f"required credential environment variable {settings.api_key_env} is missing")
            headers["Authorization"] = f"Bearer {key}"
        payload = {"model": settings.model, "messages": messages, "temperature": settings.temperature,
                   "max_tokens": settings.max_tokens, "response_format": self.response_format()}
        if settings.reasoning_effort is not None:
            payload["reasoning_effort"] = settings.reasoning_effort
        if settings.enable_thinking is not None:
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

    def response_format(self):
        value = {"type": "json_object"}
        if self.store.get(self.run_id)["profile"].get("constrain_actions", True):
            value["schema"] = action_schema()
        return value
