"""Versioned review contracts. Models propose; the harness adjudicates."""
from typing import Literal
from pydantic import Field, model_validator
from .contracts import Contract


class FindingProposal(Contract):
    id: str
    title: str
    claim: str
    path: str
    start_line: int = Field(default=1, ge=1)
    end_line: int = Field(default=1, ge=1)
    phase: Literal['baseline', 'candidate'] = 'candidate'
    severity: Literal['critical', 'high', 'medium', 'low'] = 'medium'
    category: str = 'correctness'
    evidence_ids: list[str] = Field(default_factory=list)
    defect_id: str | None = None

    @model_validator(mode='after')
    def range_valid(self):
        if self.end_line < self.start_line:
            raise ValueError('invalid source range')
        return self


class DefectProbeSpec(Contract):
    defect_id: str
    observation: str
    check_id: str
    path: str
    start_line: int = Field(default=1, ge=1)
    end_line: int = Field(default=1, ge=1)
    phase: Literal['baseline', 'candidate'] = 'baseline'
    source_hash: str = Field(pattern=r'^[0-9a-f]{64}$')
    # Caller-supplied wrapper bytes, frozen at submission and mounted read-only.
    wrapper: str = Field(min_length=1, max_length=100000)
    observed_exit_codes: list[int] = Field(min_length=1)
    not_observed_exit_codes: list[int] = Field(min_length=1)

    @model_validator(mode='after')
    def signals(self):
        codes = self.observed_exit_codes + self.not_observed_exit_codes
        if set(self.observed_exit_codes) & set(self.not_observed_exit_codes):
            raise ValueError('probe signals must be disjoint')
        if any(code < 0 or code > 125 for code in codes) or self.end_line < self.start_line:
            raise ValueError('invalid probe signal or range')
        return self


class Coverage(Contract):
    intended_paths: list[str] = Field(default_factory=list)
    examined_paths: list[str] = Field(default_factory=list)
    omitted_paths: list[str] = Field(default_factory=list)
    limit_reached: str | None = None
    unenumerated_remainder: bool = False


class ImmutableEvidenceBundle(Contract):
    version: Literal[1] = 1
    run_id: str
    revision: int
    baseline_hash: str
    candidate_hash: str
    patch_artifact_id: str
    definitions: list[dict] = Field(default_factory=list)
    checks: list[dict] = Field(default_factory=list)
    source_evidence: list[dict] = Field(default_factory=list)
    defect_probes: list[dict] = Field(default_factory=list)
    coverage: Coverage = Field(default_factory=Coverage)
    integrity_gaps: list[str] = Field(default_factory=list)


class ReviewerEvidenceView(Contract):
    version: Literal[1] = 1
    bundle_id: str
    run_id: str
    checks: list[dict] = Field(default_factory=list)
    sources: list[dict] = Field(default_factory=list)
    defect_probes: list[dict] = Field(default_factory=list)
    omitted_references: list[str] = Field(default_factory=list)
    omitted_manifest_id: str | None = None
    omitted_count: int = 0


class EvidenceChallenge(Contract):
    kind: Literal['check_missing', 'check_passed', 'source_exists', 'source_examined']
    check_id: str | None = None
    phase: Literal['baseline', 'candidate'] = 'candidate'
    evidence_id: str | None = None
    view_id: str | None = None


class ReviewerResponse(Contract):
    findings: list[FindingProposal] = Field(default_factory=list)
    observations: list[str] = Field(default_factory=list)
    challenges: list[EvidenceChallenge] = Field(default_factory=list)
    semantic_concerns: list[str] = Field(default_factory=list)


class AdjudicatedFinding(Contract):
    proposal: FindingProposal
    status: Literal['hypothesis', 'reproduced', 'needs_human_review'] = 'hypothesis'
    observation: str | None = None
    reason: str
    resolved: bool = False
    proposal_artifact: str


class ReviewDetails(Contract):
    version: Literal[1] = 1
    completion: Literal['completed', 'partial']
    partial_reasons: list[str] = Field(default_factory=list)
    findings: list[AdjudicatedFinding] = Field(default_factory=list)
    bundle_id: str | None = None
    adjudication_id: str | None = None
    coverage: Coverage = Field(default_factory=Coverage)
