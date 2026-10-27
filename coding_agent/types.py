"""Public job contracts and strictly validated agent actions."""
from __future__ import annotations

from typing import Any, Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid")


class CheckSpec(Contract):
    id: str = Field(pattern=r"^[A-Za-z0-9_-]{1,64}$")
    argv: list[str] = Field(min_length=1, max_length=100)
    cwd: str = "."
    kind: Literal["acceptance", "regression", "build"] = "acceptance"
    timeout_seconds: float = Field(default=600, gt=0)
    requires: list[str] = Field(default_factory=list)


class Profile(Contract):
    name: str
    enable_thinking: bool = False
    main_max_tokens: int = Field(default=4096, ge=128, le=65536)
    worker_max_tokens: int = Field(default=2048, ge=128, le=32768)
    model_timeout_seconds: float = Field(default=300, gt=0)
    max_actions: int = Field(default=200, ge=1, le=10000)
    max_model_requests: int = Field(default=200, ge=1, le=10000)
    max_active_seconds: float = Field(default=3600, gt=0)
    command_timeout_seconds: float = Field(default=600, gt=0)
    max_specialists: int = Field(default=9, ge=0, le=9)
    worker_max_actions: int = Field(default=12, ge=1, le=100)
    main_context_bytes: int = Field(default=24000, ge=4000, le=120000)
    worker_context_bytes: int = Field(default=12000, ge=2000, le=60000)
    allow_network: bool = False
    allow_install: bool = False
    constrain_actions: bool = True


class TaskRequest(Contract):
    root: str = Field(min_length=1)
    task: str = Field(min_length=1, max_length=40000)
    mode: Literal["implement", "review"] = "implement"
    task_type: Literal["bugfix", "feature", "refactor"] = "bugfix"
    profile: str | None = None
    checks: list[CheckSpec] = Field(default_factory=list, max_length=50)
    readonly_dependencies: list[str] = Field(default_factory=list, max_length=20)

    @model_validator(mode="after")
    def check_graph(self):
        known: set[str] = set()
        for check in self.checks:
            if check.id in known or not set(check.requires) <= known:
                raise ValueError("check IDs must be unique and prerequisites must occur earlier")
            known.add(check.id)
        if not self.task.strip():
            raise ValueError("task must not be blank")
        return self


RunState = Literal["queued", "running", "interrupted", "needs_input", "completed", "failed", "cancelled"]


class RunStatus(Contract):
    run_id: str
    state: RunState
    attempt: int = 0
    actions: int = 0
    model_requests: int = 0
    active_seconds: float = 0
    cancel_requested: bool = False
    error: str | None = None
    created_at: float
    updated_at: float


class RunResult(Contract):
    run_id: str
    outcome: Literal["verified", "unverified", "failed", "cancelled"]
    summary: str
    patch: str = ""
    patch_artifact: str | None = None
    checks: list[dict[str, Any]] = Field(default_factory=list)
    evidence: list[str] = Field(default_factory=list)
    findings: list[dict[str, Any]] = Field(default_factory=list)
    gaps: list[str] = Field(default_factory=list)


class ToolAction(Contract):
    type: Literal["tool"]
    tool: str = Field(min_length=1, max_length=80)
    args: dict[str, Any] = Field(default_factory=dict)


class FinishAction(Contract):
    type: Literal["finish"]
    summary: str = Field(min_length=1, max_length=12000)
    findings: list[dict[str, Any]] = Field(default_factory=list, max_length=30)
    gaps: list[str] = Field(default_factory=list, max_length=30)


class InputAction(Contract):
    type: Literal["input"]
    question: str = Field(min_length=1, max_length=2000)


def parse_action(raw: str) -> ToolAction | FinishAction | InputAction:
    import json
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError("action must be a JSON object")
    cls = {"tool": ToolAction, "finish": FinishAction, "input": InputAction}.get(value.get("type"))
    if cls is None:
        raise ValueError("action type must be tool, finish, or input")
    return cls.model_validate(value)


def action_schema():
    from pydantic import TypeAdapter
    return TypeAdapter(ToolAction | FinishAction | InputAction).json_schema()
