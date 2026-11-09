from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urlparse


class ConfigError(ValueError):
    """Raised when the agent configuration is unsafe or incomplete."""


DEFAULT_ROLES = [
    "repository architect: map modules, boundaries, and likely change points",
    "Python specialist: inspect Python APIs, packaging, typing, tests, and runtime behavior",
    "C/C++ specialist: inspect headers, ownership, ABI, build systems, and compiler concerns",
    "cross-language specialist: reason about interfaces, generated code, FFI, and toolchains",
    "test engineer: propose focused regression tests and negative cases",
    "debugger: identify likely failure modes and the evidence needed to distinguish them",
    "security reviewer: identify unsafe execution, path, dependency, and secret-handling risks",
    "performance reviewer: identify concurrency, latency, memory, and scalability risks",
    "implementation planner: turn the evidence into the smallest verifiable change plan",
]


def _url(value: str, field_name: str) -> str:
    value = value.strip().rstrip("/")
    parsed = urlparse(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ConfigError(
            f"{field_name} must be an absolute http(s) URL, including the scheme"
        )
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ConfigError(f"{field_name} must not contain credentials, query parameters, or fragments")
    if not parsed.path or parsed.path == "/":
        value += "/v1"
    return value


@dataclass(frozen=True)
class ModelSettings:
    base_url: str
    model: str
    api_key_env: str | None
    timeout_seconds: float
    temperature: float
    max_tokens: int
    reasoning_effort: str | None = None
    enable_thinking: bool | None = None

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any], name: str) -> "ModelSettings":
        model = str(data.get("model", "")).strip()
        if not model:
            raise ConfigError(f"{name}.model is required")
        api_key_env = data.get("api_key_env")
        if api_key_env is not None:
            api_key_env = str(api_key_env).strip() or None
        timeout = float(data.get("timeout_seconds", 120))
        temperature = float(data.get("temperature", 0.1))
        max_tokens = int(data.get("max_tokens", 2048))
        reasoning_effort = data.get("reasoning_effort")
        enable_thinking = data.get("enable_thinking")
        if enable_thinking is not None and not isinstance(enable_thinking, bool):
            raise ConfigError(f"{name}.enable_thinking must be a boolean or null")
        if reasoning_effort is not None and reasoning_effort not in {"none", "minimal", "low", "medium", "high"}:
            raise ConfigError(f"{name}.reasoning_effort must be none, minimal, low, medium, high, or null")
        if timeout <= 0 or max_tokens <= 0 or not 0 <= temperature <= 2:
            raise ConfigError(f"{name} has invalid timeout, temperature, or max_tokens")
        return cls(
            base_url=_url(str(data.get("base_url", "")), f"{name}.base_url"),
            model=model,
            api_key_env=api_key_env,
            timeout_seconds=timeout,
            temperature=temperature,
            max_tokens=max_tokens,
            reasoning_effort=reasoning_effort,
            enable_thinking=enable_thinking,
        )


@dataclass(frozen=True)
class ContextSettings:
    max_files: int = 120
    max_bytes_per_file: int = 24_000
    max_total_bytes: int = 120_000

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "ContextSettings":
        result = cls(
            max_files=int(data.get("max_files", 120)),
            max_bytes_per_file=int(data.get("max_bytes_per_file", 24_000)),
            max_total_bytes=int(data.get("max_total_bytes", 120_000)),
        )
        if min(result.max_files, result.max_bytes_per_file, result.max_total_bytes) <= 0:
            raise ConfigError("context limits must be positive")
        return result


@dataclass(frozen=True)
class ZvecSettings:
    executable: str = "zg"
    mode: str = "direct"
    query_limit: int = 7
    timeout_seconds: float = 120.0
    embedding: str | None = None

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "ZvecSettings":
        result = cls(
            executable=str(data.get("executable", "zg")).strip() or "zg",
            mode=str(data.get("mode", "direct")),
            query_limit=int(data.get("query_limit", 7)),
            timeout_seconds=float(data.get("timeout_seconds", 120.0)),
            embedding=(str(data["embedding"]).strip() if data.get("embedding") else None),
        )
        if result.mode not in {"direct", "server", "auto"}:
            raise ConfigError("zvec.mode must be direct, server, or auto")
        if result.query_limit <= 0 or result.timeout_seconds <= 0:
            raise ConfigError("zvec query_limit and timeout_seconds must be positive")
        return result


@dataclass(frozen=True)
class AgentConfig:
    main: ModelSettings
    subagents: ModelSettings
    reviewer: ModelSettings
    parallelism: int = 9
    json_mode: bool = True
    context: ContextSettings = field(default_factory=ContextSettings)
    zvec: ZvecSettings = field(default_factory=ZvecSettings)
    specialist_roles: tuple[str, ...] = tuple(DEFAULT_ROLES)

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "AgentConfig":
        sub_data = dict(data.get("subagents", {}))
        parallelism = int(sub_data.pop("parallelism", data.get("parallelism", 9)))
        roles = tuple(str(role).strip() for role in data.get("specialist_roles", DEFAULT_ROLES))
        if not 1 <= parallelism <= 64:
            raise ConfigError("subagents.parallelism must be between 1 and 64")
        if len(roles) < parallelism:
            raise ConfigError(
                f"specialist_roles has {len(roles)} entries but parallelism is {parallelism}"
            )
        if any(not role for role in roles):
            raise ConfigError("specialist_roles cannot contain empty values")
        reviewer_data = dict(data.get("reviewer", {}))
        reviewer_data.setdefault("base_url", "https://api.openai.com/v1")
        reviewer_data.setdefault("model", "gpt-4.1-mini")
        reviewer_data.setdefault("api_key_env", "OPENAI_API_KEY")
        return cls(
            main=ModelSettings.from_mapping(data.get("main", {}), "main"),
            subagents=ModelSettings.from_mapping(sub_data, "subagents"),
            reviewer=ModelSettings.from_mapping(reviewer_data, "reviewer"),
            parallelism=parallelism,
            json_mode=bool(data.get("json_mode", True)),
            context=ContextSettings.from_mapping(data.get("context", {})),
            zvec=ZvecSettings.from_mapping(data.get("zvec", {})),
            specialist_roles=roles,
        )

    @staticmethod
    def _endpoint(value: str) -> str:
        return value.rstrip("/").lower()

    def __post_init__(self):
        endpoints = {self._endpoint(self.main.base_url), self._endpoint(self.subagents.base_url),
                     self._endpoint(self.reviewer.base_url)}
        if len(endpoints) != 3:
            raise ConfigError("main, subagents, and reviewer must use different endpoints")

    def with_environment(self, environ: Mapping[str, str] | None = None) -> "AgentConfig":
        """Apply explicit, non-secret overrides without changing credential selection."""
        env = os.environ if environ is None else environ
        raw = {
            "main": {
                "base_url": env.get("MAIN_BASE_URL", self.main.base_url),
                "model": env.get("MAIN_MODEL", self.main.model),
                "api_key_env": self.main.api_key_env,
                "timeout_seconds": self.main.timeout_seconds,
                "temperature": self.main.temperature,
                "max_tokens": self.main.max_tokens,
                "reasoning_effort": self.main.reasoning_effort,
                "enable_thinking": self.main.enable_thinking,
            },
            "subagents": {
                "base_url": env.get("SUBAGENT_BASE_URL", self.subagents.base_url),
                "model": env.get("SUBAGENT_MODEL", self.subagents.model),
                "api_key_env": self.subagents.api_key_env,
                "parallelism": env.get("SUBAGENT_PARALLELISM", self.parallelism),
                "timeout_seconds": self.subagents.timeout_seconds,
                "temperature": self.subagents.temperature,
                "max_tokens": self.subagents.max_tokens,
                "reasoning_effort": self.subagents.reasoning_effort,
                "enable_thinking": self.subagents.enable_thinking,
            },
            "reviewer": {
                "base_url": env.get("REVIEWER_BASE_URL", self.reviewer.base_url),
                "model": env.get("REVIEWER_MODEL", self.reviewer.model),
                "api_key_env": env.get("REVIEWER_API_KEY_ENV", self.reviewer.api_key_env),
                "timeout_seconds": self.reviewer.timeout_seconds,
                "temperature": self.reviewer.temperature,
                "max_tokens": self.reviewer.max_tokens,
                "reasoning_effort": self.reviewer.reasoning_effort,
                "enable_thinking": self.reviewer.enable_thinking,
            },
            "json_mode": self.json_mode,
            "context": {
                "max_files": self.context.max_files,
                "max_bytes_per_file": self.context.max_bytes_per_file,
                "max_total_bytes": self.context.max_total_bytes,
            },
            "zvec": {
                "executable": self.zvec.executable,
                "mode": self.zvec.mode,
                "query_limit": self.zvec.query_limit,
                "timeout_seconds": self.zvec.timeout_seconds,
                "embedding": self.zvec.embedding,
            },
            "specialist_roles": list(self.specialist_roles),
        }
        return AgentConfig.from_mapping(raw)


def _defaults() -> dict[str, Any]:
    return {
        "main": {
            "base_url": "https://lead.example.invalid/v1",
            "model": "lead-model",
            "api_key_env": "LLAMA_API_KEY",
        },
        "subagents": {
            "base_url": "https://specialist.example.invalid/v1",
            "model": "specialist-model",
            "api_key_env": "LLAMA_API_KEY",
            "parallelism": 9,
            "reasoning_effort": "none",
            "enable_thinking": False,
        },
        "reviewer": {
            "base_url": "https://api.openai.com/v1",
            "model": "gpt-4.1-mini",
            "api_key_env": "OPENAI_API_KEY",
            "reasoning_effort": "none",
            "enable_thinking": False,
        },
        "specialist_roles": DEFAULT_ROLES,
    }


def load_config(path: str | Path | None = None, *, environ: Mapping[str, str] | None = None) -> AgentConfig:
    """Load JSON configuration and apply explicit environment overrides."""
    config_path = Path(path) if path else Path("agent-config.json")
    if config_path.exists():
        try:
            data = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ConfigError(f"cannot read configuration {config_path}: {exc}") from exc
    else:
        data = _defaults()
    if not isinstance(data, dict):
        raise ConfigError("configuration root must be a JSON object")
    return AgentConfig.from_mapping(data).with_environment(environ)
