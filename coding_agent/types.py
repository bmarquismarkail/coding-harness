"""Public job contracts and strictly validated agent actions."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Literal
from pydantic import Field, model_validator
from .contracts import Contract
from .review_types import FindingProposal, DefectProbeSpec, ReviewDetails, ReviewerResponse



class CheckSpec(Contract):
    id: str = Field(pattern=r"^[A-Za-z0-9_-]{1,64}$")
    argv: list[str] = Field(min_length=1, max_length=100)
    cwd: str = "."
    kind: Literal["acceptance", "regression", "build"] = "acceptance"
    timeout_seconds: float = Field(default=600, gt=0)
    requires: list[str] = Field(default_factory=list)
    failure_categories: dict[int, Literal["missing_required_change", "unexpected_change", "content_mismatch", "compiler_input_failure"]] = Field(default_factory=dict)


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
    protocol_failure_limit: int = Field(default=3, ge=1, le=20)
    discovery_edges: int = Field(default=2000, ge=1)
    discovery_paths: int = Field(default=200, ge=1)
    discovery_bytes: int = Field(default=4194304, ge=1)
    discovery_seconds: float = Field(default=2, gt=0)
    review_no_progress_actions: int = Field(default=24, ge=6, le=200)
    main_context_bytes: int = Field(default=24000, ge=4000, le=120000)
    worker_context_bytes: int = Field(default=12000, ge=2000, le=60000)
    allow_network: bool = False
    allow_install: bool = False
    constrain_actions: bool = True
    preserve_model_settings: bool = False
    check_after_edit: bool = False


class ReplacementSpec(Contract):
    target_id: str = Field(pattern=r"^[A-Za-z0-9_-]{1,64}$")
    source_path: str
    selector_path: str
    original_selector_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    selected_selector_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    selector_argv: list[str] = Field(min_length=2, max_length=100)
    verifier_path: str
    verifier_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    compile_check_id: str


class TaskRequest(Contract):
    root: str = Field(min_length=1)
    task: str = Field(min_length=1, max_length=40000)
    mode: Literal["implement", "review"] = "implement"
    task_type: Literal["bugfix", "feature", "refactor"] = "bugfix"
    profile: str | None = None
    project_profile: str | None = Field(default=None, pattern=r"^[A-Za-z0-9_-]{1,64}$")
    skill_ids: list[str] | None = Field(default=None, max_length=20)
    checks: list[CheckSpec] | None = Field(default=None, max_length=50)
    readonly_dependencies: list[str] = Field(default_factory=list, max_length=20)
    defect_probes: list[DefectProbeSpec] = Field(default_factory=list, max_length=50)
    focus_paths: list[str] = Field(default_factory=list, max_length=40)
    required_patch_paths: list[str] = Field(default_factory=list, max_length=40)

    replacement: ReplacementSpec | None = None

    @model_validator(mode="after")
    def check_graph(self):
        if self.required_patch_paths and self.mode != "implement":
            raise ValueError("required patch paths apply only to implementation jobs")
        known: set[str] = set()
        for check in self.checks or []:
            if check.id in known or not set(check.requires) <= known:
                raise ValueError("check IDs must be unique and prerequisites must occur earlier")
            known.add(check.id)
        replacement = self.replacement
        if replacement:
            paths = {replacement.source_path, replacement.selector_path}
            if (self.mode != "implement" or len(paths) != 2 or not paths <= set(self.required_patch_paths)
                    or any(not path or Path(path).is_absolute() or ".." in Path(path).parts
                           or Path(path).as_posix() != path for path in paths)):
                raise ValueError("replacement needs two distinct required workspace paths")
            verifier = Path(replacement.verifier_path)
            if (not verifier.is_absolute() or replacement.verifier_path not in replacement.selector_argv
                    or replacement.original_selector_sha256 == replacement.selected_selector_sha256
                    or not any(verifier.is_relative_to(Path(dep)) for dep in self.readonly_dependencies)):
                raise ValueError("replacement needs a distinct selector splice and a mounted verifier")
            checks = self.checks or []
            if (not checks or checks[0].id != replacement.compile_check_id or checks[0].kind != "build"
                    or checks[0].requires or any(replacement.verifier_path not in c.argv for c in checks)
                    or any(replacement.compile_check_id not in c.requires for c in checks[1:])):
                raise ValueError("replacement needs a first build check and dependent verification checks")
        if len({p.check_id for p in self.defect_probes}) != len(self.defect_probes):
            raise ValueError("one trusted probe is permitted per check")
        if len({p.defect_id for p in self.defect_probes}) != len(self.defect_probes):
            raise ValueError("duplicate defect ID")
        if self.checks is not None and any(p.check_id not in known for p in self.defect_probes):
            raise ValueError("probe must reference a caller-configured check")
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


class EvidencePackage(Contract):
    baseline_checks: list[dict[str, Any]] = Field(default_factory=list)
    reproducer_artifacts: list[str] = Field(default_factory=list)
    diagnosis: str = ""
    diagnosis_sources: list[str] = Field(default_factory=list)
    patch_artifact: str | None = None
    patch_source_hash: str | None = None
    verification_checks: list[dict[str, Any]] = Field(default_factory=list)
    remaining_gaps: list[str] = Field(default_factory=list)
    skill_snapshots: list[dict[str, Any]] = Field(default_factory=list)


class RunResult(Contract):
    review: ReviewDetails | None = None
    run_id: str
    outcome: Literal["verified", "unverified", "failed", "cancelled"]
    summary: str
    patch: str = ""
    patch_artifact: str | None = None
    checks: list[dict[str, Any]] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)
    evidence: EvidencePackage | None = None
    findings: list[dict[str, Any]] = Field(default_factory=list)
    gaps: list[str] = Field(default_factory=list)


class ToolAction(Contract):
    type: Literal["tool"]
    tool: str = Field(min_length=1, max_length=80)
    args: dict[str, Any] = Field(default_factory=dict)


class FinishAction(Contract):
    review: ReviewerResponse | None = None
    type: Literal["finish"]
    summary: str = Field(min_length=1, max_length=12000)
    findings: list[dict[str, Any]] = Field(default_factory=list, max_length=30)
    gaps: list[str] = Field(default_factory=list, max_length=30)
    diagnosis: str = Field(default="", max_length=12000)
    diagnosis_sources: list[str] = Field(default_factory=list, max_length=50)


class ReviewFinishAction(FinishAction):
    findings: list[FindingProposal] = Field(default_factory=list, max_length=30)


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


def action_schema(*, typed_findings=False):
    from pydantic import TypeAdapter
    finish = ReviewFinishAction if typed_findings else FinishAction
    return TypeAdapter(ToolAction | finish | InputAction).json_schema()
