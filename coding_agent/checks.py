"""Source-bound checks and deterministic completion gates."""
from __future__ import annotations

import json
import shutil
import sys
import hashlib
import time
from pathlib import Path

from .sandbox import SandboxRunner, CommandCancelled
from .store import Store
from .types import CheckSpec, EvidencePackage, RunResult, TaskRequest
from .workspace import Workspace, checked_path, digest, manifest, tree_hash


class Checks:
    def __init__(self, store: Store, run_id: str, workspace: Workspace, runner: SandboxRunner):
        self.store, self.run_id, self.workspace, self.runner = store, run_id, workspace, runner
        self.path = workspace.folder / "checks.json"
        self.required_patch_paths: list[str] = []

    def initialize(self, request: TaskRequest):
        self._request_specs = list(request.checks or [])
        self.required_patch_paths = request.required_patch_paths
        if self.path.exists():
            return
        checks = [{"spec": c.model_dump(), "origin": "request", "oracle": {}} for c in (request.checks or [])]
        root = self.workspace.baseline
        if not any(c["spec"]["kind"] == "regression" for c in checks):
            discovered = []
            if (root / "tests").is_dir() and any((root/"tests").rglob("test*.py")):
                discovered = [CheckSpec(id="original-python-tests", argv=[sys.executable, "-m", "unittest", "discover", "-s", "tests"], kind="regression")]
            if (root / "CMakeLists.txt").exists():
                discovered += [
                    CheckSpec(id="original-configure", argv=["cmake", "-S", ".", "-B", "build-harness"], kind="build"),
                    CheckSpec(id="original-build", argv=["cmake", "--build", "build-harness", "-j2"], kind="build", requires=["original-configure"]),
                    CheckSpec(id="original-ctest", argv=["ctest", "--test-dir", "build-harness", "--output-on-failure", "--no-tests=error"], kind="regression", requires=["original-build"]),
                ]
            known = {c["spec"]["id"] for c in checks}
            checks.extend({"spec": c.model_dump(), "origin": "discovered", "oracle": {}}
                          for c in discovered if c.id not in known)
        self._save(checks)

    def _save(self, checks: list):
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(checks, indent=2))
        tmp.replace(self.path)

    def definitions(self) -> list[dict]:
        return json.loads(self.path.read_text())

    def register(self, spec: CheckSpec, test_paths: list[str]) -> dict:
        checks = self.definitions()
        if spec.id in {c["spec"]["id"] for c in checks}:
            raise ValueError("check ID already exists; checks cannot be replaced")
        known = {c["spec"]["id"] for c in checks}
        if not set(spec.requires) <= known:
            raise ValueError("check prerequisite is not registered")
        oracle = {}
        for relative in test_paths:
            p = checked_path(self.workspace.root, relative)
            name = Path(relative).name
            if not (name.startswith(("test_", "smoke_")) or "tests" in Path(relative).parts
                    or spec.kind == "build" and name in {"CMakeLists.txt", "Makefile", "pyproject.toml", "meson.build"}):
                raise ValueError("oracle additions must be test files")
            original = self.workspace.baseline / relative
            if spec.kind == "acceptance" and original.exists():
                raise ValueError("new acceptance checks must use added tests; original oracles cannot be replaced")
            if spec.kind != "acceptance" and (not original.is_file() or digest(original) != digest(p)):
                raise ValueError("existing regression/build checks must bind unchanged original test/build files")
            raw = p.read_bytes()
            oracle[relative] = {"artifact": self.store.artifact(self.run_id, raw), "sha256": digest(p),
                                "mode": p.stat().st_mode & 0o777}
        if not oracle:
            raise ValueError("register a new acceptance test or bind an unchanged original regression/build file")
        checks.append({"spec": spec.model_dump(), "origin": "model", "oracle": oracle})
        self._save(checks)
        return {"registered": spec.id, "oracle": oracle}

    def missing_patch_paths(self) -> list[str]:
        if not self.required_patch_paths:
            return []
        changed = set(self.workspace.changed_paths())
        return [path for path in self.required_patch_paths
                if Path(path).as_posix() not in changed or not checked_path(self.workspace.root, path).is_file()]

    def execute(self, check_id: str, phase: str, *, key: str) -> dict:
        if phase not in {"baseline", "candidate"}:
            raise ValueError("phase must be baseline or candidate")
        definition = next((c for c in self.definitions() if c["spec"]["id"] == check_id), None)
        if definition is None:
            raise ValueError("unknown check ID")
        missing = self.missing_patch_paths() if phase == "candidate" else []
        if missing:
            raise ValueError("Candidate checks require a patch containing: " + ", ".join(missing))
        spec = CheckSpec.model_validate(definition["spec"])
        root = self.workspace.baseline_check_root() if phase == "baseline" else self.workspace.root
        expected_baseline = manifest(self.workspace.baseline)
        for registered in self.definitions():
            for relative, info in registered["oracle"].items():
                candidate = self.workspace.root / relative
                if digest(candidate) != info["sha256"]:
                    raise RuntimeError("registered oracle changed; cannot count its results")
                if phase == "baseline":
                    destination = root / relative
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(candidate, destination)
                    expected_baseline[relative] = {"sha256": info["sha256"], "mode": info["mode"]}
        if phase == "baseline" and manifest(root) != expected_baseline:
            raise RuntimeError("baseline execution sources changed; retained checks cannot establish a reproduction")
        before = manifest(root)
        fingerprint = tree_hash(root)
        previous = self.store.receipt(self.run_id, key)
        if previous and previous["state"] == "completed" and previous["data"].get("source_hash") == fingerprint:
            return {**previous["data"], "output": previous["data"].get("output", "")[:6000]}
        records = [r["data"] for r in self.store.receipts(self.run_id, "check") if r["state"] == "completed"]
        for required in spec.requires:
            if not any(r["check_id"] == required and r["phase"] == phase and r.get("passed")
                       and r["source_hash"] == fingerprint for r in records):
                result = {"check_id": check_id, "phase": phase, "source_hash": fingerprint,
                          "passed": False, "skipped": True, "reason": f"prerequisite {required} has not passed on these sources"}
                self.store.save_receipt(self.run_id, key, "check", "completed", result)
                return result
        self.store.save_receipt(self.run_id, key, "check", "started", {"check_id": check_id, "phase": phase, "source_hash": fingerprint})
        probe_definitions = []
        for receipt in self.store.receipts(self.run_id, "probe_definition"):
            from .evidence import read_json
            probe = read_json(self.store, self.run_id, receipt["data"]["artifact_id"])
            if probe["check_id"] == check_id:
                probe_definitions.append((receipt["data"]["artifact_id"], probe))
        if len(probe_definitions) > 1:
            raise ValueError("one trusted probe is permitted per check")
        runner, argv = self.runner, spec.argv
        probe_id = None
        if probe_definitions:
            probe_id, probe = probe_definitions[0]
            if probe.get("check_spec") != spec.model_dump():
                raise RuntimeError("trusted check definition changed")
            wrapper = self.store.run_dir(self.run_id) / "artifacts" / probe["wrapper_artifact"]
            if hashlib.sha256(wrapper.read_bytes()).hexdigest() != probe["wrapper_artifact"]:
                raise RuntimeError("trusted wrapper integrity mismatch")
            runner = SandboxRunner(self.runner.profile, self.runner.dependencies, self.runner.cancelled)
            runner.protected_files = {"/trusted-probe.py": wrapper}
            argv = [sys.executable, "-I", "/trusted-probe.py", *spec.argv]
        try:
            result = runner.run(root, argv, cwd=spec.cwd, timeout=spec.timeout_seconds)
        except CommandCancelled as exc:
            data = {**exc.result, "check_id": check_id, "phase": phase, "source_hash": fingerprint, "passed": False}
            data["artifact_id"] = self.store.artifact(self.run_id, data)
            self.store.save_receipt(self.run_id, key, "check", "interrupted", data)
            self.store.event(self.run_id, "check_interrupted", {"artifact_id": data["artifact_id"]})
            raise
        altered = manifest(root) != before
        result.update(check_id=check_id, phase=phase, source_hash=fingerprint, origin=definition["origin"],
                      passed=(result["exit_code"] in probe["not_observed_exit_codes"] if probe_id else result["exit_code"] == 0) and not result["timed_out"] and not altered and not result["output_truncated"],
                      source_modified=altered, skipped=False, executed_at=time.time())
        if probe_id:
            result["probe_definition_id"] = probe_id
            try:
                target = checked_path(root, probe['path'])
            except ValueError:
                result['probe_range_valid'] = False
                result['probe_target_valid'] = False
            else:
                result['probe_range_valid'] = (target.is_file()
                    and len(target.read_text(errors='replace').splitlines()) >= probe['end_line'])
                result['probe_target_valid'] = (phase == probe['phase'] and fingerprint == probe['source_hash']
                    and result['probe_range_valid'])
            result['probe_observed'] = result['exit_code'] in probe['observed_exit_codes']
        artifact = self.store.artifact(self.run_id, result)
        result["artifact_id"] = artifact
        self.store.save_receipt(self.run_id, key, "check", "completed", result)
        self.store.event(self.run_id, "check", {k: result[k] for k in ("check_id", "phase", "passed", "artifact_id")})
        return {**result, "output": result["output"][:6000]}

    def run_all(self, phase: str, prefix: str):
        results = []
        for item in self.definitions():
            self.store.consume(self.run_id)
            results.append(self.execute(item["spec"]["id"], phase, key=f"{prefix}:{item['spec']['id']}"))
        return results

    def run_requested(self, phase: str, prefix: str):
        """Run only checks explicitly requested for a review job."""
        requested = {c.id for c in self._request_specs}
        return [self.execute(check_id, phase, key=f"{prefix}:{check_id}") for check_id in requested]

    def finish(self, request: TaskRequest, summary: str, findings: list, gaps: list[str], *,
               diagnosis: str = "", diagnosis_sources: list[str] | None = None,
               skill_snapshots: list[dict] | None = None) -> RunResult:
        patch = self.workspace.patch()
        fingerprint = tree_hash(self.workspace.root)
        records = [r["data"] for r in self.store.receipts(self.run_id, "check") if r["state"] == "completed"]
        gaps = list(gaps)
        for path in self.missing_patch_paths():
            gaps.append(f"Required patch path is missing or unchanged: {path}")
        definitions = self.definitions()
        candidates = sorted([r for r in records if r.get("phase") == "candidate" and r.get("source_hash") == fingerprint],
                            key=lambda r:r.get("executed_at", 0))
        final_by_id = {r["check_id"]: r for r in candidates}
        for check in definitions if request.mode == "implement" else []:
            last = final_by_id.get(check["spec"]["id"])
            if not last or not last.get("passed"):
                gaps.append(f"Required check {check['spec']['id']} did not pass on the final source.")
        acceptance = [c for c in definitions if c["spec"]["kind"] == "acceptance"]
        regressions = [c for c in definitions if c["spec"]["kind"] == "regression"]
        if request.mode == "implement" and not acceptance:
            gaps.append("No executable acceptance check was established.")
        if request.mode == "implement" and not regressions:
            gaps.append("No independent existing regression suite was established.")
        baseline_root = self.workspace.baseline_check_root()
        expected_baseline = manifest(self.workspace.baseline)
        for definition in definitions:
            expected_baseline.update({p:{"sha256":info["sha256"], "mode":info["mode"]} for p,info in definition["oracle"].items()})
        if manifest(baseline_root) != expected_baseline:
            gaps.append("Baseline execution sources no longer match the original snapshot and frozen oracles.")
        baseline_hash = tree_hash(baseline_root)
        for check in acceptance if request.mode == "implement" else []:
            base = [r for r in records if r.get("phase") == "baseline" and r["check_id"] == check["spec"]["id"]
                    and r.get("source_hash") == baseline_hash]
            if request.task_type in {"bugfix", "feature"} and not any(
                    not r.get("passed") and not r.get("skipped") and not r.get("timed_out")
                    and not r.get("source_modified") and not r.get("output_truncated")
                    and (r.get("probe_observed") and r.get("probe_target_valid") if r.get("probe_definition_id")
                         else r.get("exit_code", 0) not in {0, 77, 126, 127, -9}) for r in base):
                gaps.append(f"Acceptance check {check['spec']['id']} did not reproduce a baseline failure.")
            if request.task_type == "refactor" and not any(r.get("passed") for r in base):
                gaps.append(f"Refactor baseline check {check['spec']['id']} did not pass.")
        for relative in self.workspace.assert_original_tests():
            gaps.append(f"Original test/build oracle changed: {relative}")
        if request.mode == "implement" and not patch:
            gaps.append("Implementation produced no patch.")
        if request.mode == "implement" and not diagnosis.strip():
            gaps.append("Evidence package has no diagnosis.")
        if request.mode == "review":
            gaps.append("Review findings are hypotheses; model agreement is not independent verification.")
        # A model cannot promote its own classification to a proof.
        findings = [{**finding, "classification": "hypothesis"} for finding in findings]
        artifact = self.store.artifact(self.run_id, patch)
        evidence = {r["artifact_id"] for r in records if r.get("artifact_id")}
        history = {r["artifact_id"]:r for r in records if r.get("artifact_id")}
        with self.store.db() as db:
            for row in db.execute("SELECT kind,data FROM events WHERE run_id=?", (self.run_id,)):
                value = json.loads(row["data"])
                evidence.update(value[k] for k in ("artifact_id", "request_artifact", "response_artifact") if value.get(k))
                if row["kind"] == "check" and value.get("artifact_id") not in history:
                    raw = (self.store.run_dir(self.run_id)/"artifacts"/value["artifact_id"]).read_bytes()
                    if hashlib.sha256(raw).hexdigest() != value["artifact_id"]:
                        raise RuntimeError("retained check artifact failed its integrity check")
                    history[value["artifact_id"]] = {**json.loads(raw), "artifact_id":value["artifact_id"]}
        sources = list(dict.fromkeys(diagnosis_sources or []))
        if request.mode == "implement" and not sources:
            gaps.append("Evidence package diagnosis has no source citations.")
        evidence_package = EvidencePackage(
            baseline_checks=[{k:r.get(k) for k in ("check_id", "source_hash", "passed", "exit_code", "artifact_id", "skipped", "timed_out")}
                             for r in records if r.get("phase") == "baseline"],
            reproducer_artifacts=list(dict.fromkeys(r["artifact_id"] for r in records
                                                    if r.get("phase") == "baseline" and r.get("artifact_id")
                                                    and r.get("check_id") in {c["spec"]["id"] for c in acceptance})),
            diagnosis=diagnosis,
            diagnosis_sources=sources,
            patch_artifact=artifact,
            patch_source_hash=fingerprint,
            verification_checks=[{k:r.get(k) for k in ("check_id", "source_hash", "passed", "exit_code", "artifact_id", "skipped", "timed_out")}
                                 for r in candidates],
            remaining_gaps=sorted(set(gaps)),
            skill_snapshots=skill_snapshots or [],
        )
        result = RunResult(run_id=self.run_id, outcome="verified" if not gaps else "unverified", summary=summary,
                           patch=patch, patch_artifact=artifact,
                           checks=[{k: v for k, v in r.items() if k != "output"} for r in [*history.values(), *[r for r in records if not r.get("artifact_id")]]],
                           evidence_ids=sorted(evidence), evidence=evidence_package,
                           findings=findings, gaps=sorted(set(gaps)))
        return result
