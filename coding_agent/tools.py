"""A capability-scoped tool broker shared by the writer and specialists."""
from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import tempfile
from pathlib import Path
from typing import Literal

from pydantic import Field

from .checks import Checks
from .sandbox import CommandCancelled
from .store import Store
from .types import CheckSpec, Contract
from .review_types import FindingProposal
from .workspace import Workspace, checked_path, digest, paths, tree_hash
from .zvec import ZvecGrepTool, ZvecGrepError


class ListArgs(Contract):
    pattern: str = "*"
    offset: int = Field(default=0, ge=0)
    limit: int = Field(default=100, ge=1, le=200)


class ReadArgs(Contract):
    path: str
    start: int = Field(default=1, ge=1)
    lines: int = Field(default=100, ge=1, le=200)
    baseline: bool = False


class SearchArgs(Contract):
    query: str = Field(min_length=1, max_length=1000)
    limit: int = Field(default=20, ge=1, le=30)


class EvidenceArgs(Contract):
    artifact_id: str


class ArtifactArgs(EvidenceArgs):
    offset: int = Field(default=0, ge=0)
    limit: int = Field(default=4000, ge=1, le=6000)


class PageArgs(Contract):
    offset: int = Field(default=0, ge=0)
    limit: int = Field(default=10, ge=1, le=20)


class Change(Contract):
    path: str
    expected_hash: str | None
    content: str | None = Field(max_length=200000)
    executable: bool | None = None


class PatchArgs(Contract):
    changes: list[Change] = Field(min_length=1, max_length=20)


class CommandArgs(Contract):
    argv: list[str] = Field(min_length=1, max_length=100)
    cwd: str = "."
    timeout_seconds: float | None = Field(default=None, gt=0)


class CheckArgs(Contract):
    check_id: str
    phase: Literal["baseline", "candidate"] = "candidate"


class RegisterArgs(Contract):
    check: CheckSpec
    test_paths: list[str] = Field(min_length=1, max_length=10)


class WorkerTask(Contract):
    id: str = Field(pattern=r"^[A-Za-z0-9_-]{1,64}$")
    task: str = Field(min_length=1, max_length=4000)
    paths: list[str] = Field(min_length=1, max_length=20)


class DelegateArgs(Contract):
    tasks: list[WorkerTask] = Field(min_length=1, max_length=9)


class CollectArgs(Contract):
    ids: list[str] = Field(min_length=1, max_length=9)


class ReportFindingsArgs(Contract):
    findings: list[FindingProposal] = Field(default_factory=list, max_length=30)


class EmptyArgs(Contract):
    pass


READ_TOOLS = {"list_files": ListArgs, "read_file": ReadArgs, "search_exact": SearchArgs,
              "search_semantic": SearchArgs, "read_evidence": EvidenceArgs, "read_artifact": ArtifactArgs,
              "list_checks": PageArgs, "inspect_diff": EmptyArgs}
WRITE_TOOLS = {"report_findings": ReportFindingsArgs, "apply_patch": PatchArgs, "run_command": CommandArgs, "run_check": CheckArgs,
               "register_check": RegisterArgs, "delegate": DelegateArgs, "collect": CollectArgs}


class ToolBroker:
    def __init__(self, store: Store, run_id: str, workspace: Workspace, checks: Checks, *,
                 mode: str = "implement", scope: list[str] | None = None, zvec=None):
        self.store, self.run_id, self.workspace, self.checks = store, run_id, workspace, checks
        self.mode, self.scope = mode, scope
        self.zvec = zvec or ZvecGrepTool()
        self.pool = None

    def schemas(self):
        allowed = dict(READ_TOOLS)
        if self.scope is None:
            allowed.update(WRITE_TOOLS)
            if self.mode == "review":
                allowed.pop("apply_patch")
        allowed["report_findings"] = ReportFindingsArgs
        descriptions = {
            "apply_patch": "Create, replace, or delete files. Read existing files first and supply their sha256 as expected_hash; use null only for new files. Preserve the original tests.",
            "run_check": "Execute a registered check on baseline or candidate sources; use this rather than run_command for acceptance evidence.",
            "register_check": "Freeze new acceptance test files, or add a regression/build check bound to unchanged original test/build files. Check IDs and original oracles cannot be replaced.",
            "delegate": "Start focused read-only specialists with explicit file/directory scopes. Collect them before editing or executing commands.",
            "collect": "Wait for named specialists and collect their reports. A required worker failure can be resumed without repeating completed workers.",
            "read_evidence": "Recover previously issued evidence only if its source hash is still valid.",
        }
        if self.mode == "review" or self.checks.runner.profile.check_after_edit:
            descriptions["run_command"] = "Run a command with a read-only workspace and writable /tmp. Use run_check for registered checks."
        return {name: {"description": descriptions.get(name, name.replace("_", " ")),
                       "parameters": cls.model_json_schema()} for name, cls in allowed.items()}

    def invoke(self, name: str, args: dict, *, key: str):
        if name not in self.schemas():
            raise PermissionError(f"tool {name!r} is not available for this role")
        schema = {**READ_TOOLS, **WRITE_TOOLS}[name]
        parsed = schema.model_validate(args)
        self.store.consume(self.run_id)
        if name in {"apply_patch", "run_command", "run_check", "register_check"} and self.pool and self.pool.pending():
            raise ValueError("collect running specialists before changing or executing the workspace")
        result = getattr(self, name)(parsed, key=key)
        artifact = self.store.artifact(self.run_id, result)
        self.store.event(self.run_id, "tool", {"tool": name, "artifact_id": artifact})
        # Return a bounded observation; complete data stays recoverable by ID.
        raw = json.dumps(result, ensure_ascii=False)
        if len(raw) > 8000:
            return {"artifact_id": artifact, "preview": raw[:6000], "truncated": True}
        return {"artifact_id": artifact, "result": result}

    def report_findings(self, args: ReportFindingsArgs, *, key):
        artifact = self.store.artifact(self.run_id, args.model_dump())
        self.store.save_receipt(self.run_id, key, "draft_findings", "completed", {"artifact_id": artifact})
        return {"saved": len(args.findings), "artifact_id": artifact}

    def _path(self, relative: str, baseline=False):
        scope = self.scope
        if scope is not None and Path(relative).name == "AGENTS.md" and any(
                Path(relative).parent == Path(".") or Path(relative).parent in Path(s).parents
                or Path(relative).parent == Path(s) for s in scope):
            scope = None
        return checked_path(self.workspace.baseline if baseline else self.workspace.root, relative, scope=scope)

    def list_files(self, args: ListArgs, **_):
        import fnmatch
        items = []
        for relative, path in paths(self.workspace.root):
            try:
                self._path(relative)
            except PermissionError:
                continue
            if fnmatch.fnmatch(relative, args.pattern):
                items.append(relative)
        return {"files": items[args.offset:args.offset+args.limit], "total": len(items),
                "next_offset": args.offset+args.limit, "has_more": args.offset+args.limit<len(items)}

    def read_file(self, args: ReadArgs, *, record_read=True, **_):
        path = self._path(args.path, args.baseline)
        if path.is_dir():
            raise ValueError(f"{args.path} is a directory; use list_files to find a file to read")
        if path.stat().st_size > 1024*1024:
            raise ValueError("file exceeds the text-read limit")
        raw = path.read_bytes()
        if b"\0" in raw:
            raise ValueError("binary file cannot be read as source text")
        lines = raw.decode("utf-8", errors="replace").splitlines()
        selected = []
        used = 0
        truncated_line = False
        for line in lines[args.start-1:args.start-1+args.lines]:
            numbered = f"{args.start+len(selected)}: {line}"
            if used + len(numbered) > 6500:
                if not selected:
                    selected.append(numbered[:6500])
                    truncated_line = True
                break
            selected.append(numbered)
            used += len(numbered)+1
        evidence = {"kind": "source", "path": args.path, "baseline": args.baseline,
                    "sha256": digest(path), "start": args.start, "end": args.start+len(selected)-1,
                    "text": "\n".join(selected), "truncated_line": truncated_line,
                    "total_lines": len(lines), "next_start": args.start+len(selected)}
        evidence["evidence_id"] = self.store.artifact(self.run_id, evidence)
        if not record_read:
            return evidence
        receipt_key = f"read:{args.baseline}:{args.path}"
        if self.scope is not None:
            receipt_key = "worker:"+hashlib.sha256(json.dumps(sorted(self.scope)).encode()).hexdigest()+":"+receipt_key
        prior = self.store.receipt(self.run_id, receipt_key)
        ranges = prior["data"]["ranges"] if prior and prior["data"].get("sha256") == evidence["sha256"] else []
        if not truncated_line:
            ranges.append([evidence["start"], evidence["end"]])
        self.store.save_receipt(self.run_id, receipt_key, "source", "completed",
                                {"sha256": evidence["sha256"], "ranges": ranges, "evidence_id": evidence["evidence_id"]})
        self.store.event(self.run_id, "source", {"path": args.path, "sha256": evidence["sha256"], "artifact_id": evidence["evidence_id"]})
        return evidence

    def require_instructions(self, relative):
        for parent in [Path("."), *reversed(Path(relative).parents[:-1])]:
            instruction = self.workspace.root / parent / "AGENTS.md"
            if not instruction.is_file():
                continue
            key = f"read:False:{(parent/'AGENTS.md').as_posix()}"
            receipt = self.store.receipt(self.run_id, key)
            total = len(instruction.read_text(errors="replace").splitlines())
            covered = set()
            if receipt and receipt["data"]["sha256"] == digest(instruction):
                for start, end in receipt["data"]["ranges"]:
                    covered.update(range(start, end+1))
            if not all(n in covered for n in range(1, total+1)):
                raise ValueError(f"read all applicable instructions before editing: {(parent/'AGENTS.md').as_posix()}")

    def read_evidence(self, args: EvidenceArgs, **_):
        data = self.store.read_artifact(self.run_id, args.artifact_id, limit=32000)
        if data["has_more"]:
            raise ValueError("artifact is not a bounded source evidence record")
        record = json.loads(data["content"])
        if record.get("kind") != "source":
            raise ValueError("artifact is not source evidence")
        source = self._path(record["path"], record.get("baseline", False))
        if digest(source) != record["sha256"]:
            baseline = self._path(record["path"], True)
            if record.get("baseline", False) or digest(baseline) != record["sha256"]:
                raise ValueError("source evidence is stale; read the current file again")
            record["validated_against"] = "baseline"
        record.setdefault("evidence_id", args.artifact_id)
        return record

    def read_artifact(self, args: ArtifactArgs, **_):
        if self.scope is not None:
            # A specialist must not recover an unscoped writer/worker transcript.
            self.read_evidence(EvidenceArgs(artifact_id=args.artifact_id))
        return self.store.read_artifact(self.run_id, args.artifact_id, args.offset, args.limit)

    def list_checks(self, args: PageArgs, **_):
        definitions = self.checks.definitions()
        from .evidence import read_json
        probes = [read_json(self.store, self.run_id, receipt['data']['artifact_id'])
                  for receipt in self.store.receipts(self.run_id, 'probe_definition')]
        for definition in definitions:
            definition['trusted_observations'] = [
                {k: p[k] for k in ('defect_id', 'observation', 'path', 'start_line', 'end_line', 'phase', 'source_hash')}
                for p in probes if p['check_id'] == definition['spec']['id']]
        return {"checks": definitions[args.offset:args.offset+args.limit], "total": len(definitions),
                "has_more": args.offset+args.limit<len(definitions), "next_offset": args.offset+args.limit}

    def search_exact(self, args: SearchArgs, **_):
        hits = []
        for relative, path in paths(self.workspace.root):
            try:
                self._path(relative)
            except PermissionError:
                continue
            if path.stat().st_size > 1024*1024:
                continue
            for number, line in enumerate(path.read_text(errors="replace").splitlines(), 1):
                if args.query in line:
                    hits.append({"path": relative, "line": number, "text": line[:1000], "sha256": digest(path)})
                    if len(hits) >= args.limit:
                        return {"hits": hits, "limited": True}
        return {"hits": hits, "limited": False}

    def search_semantic(self, args: SearchArgs, **_):
        metadata = json.loads(self.workspace.meta.read_text())
        source = Path(metadata["source"])
        try:
            raw = self.zvec.query(source, args.query, limit=min(args.limit, 7))
        except ZvecGrepError as exc:
            return {"status": "semantic_unavailable", "reason": str(exc)[:1000],
                    "fallback": self.search_exact(args), "index_created": False}
        hits = []
        stale = []
        for match in re.finditer(r"([\w./-]+\.(?:py|pyi|cpp|hpp|cc|cxx|h|c|md|rs|go|ts|js|toml)):(\d+)", raw):
            path = Path(match[1])
            if path.is_absolute():
                if not path.is_relative_to(source):
                    continue
                path = path.relative_to(source)
            relative = path.as_posix()
            try:
                current = self._path(relative)
                baseline = metadata["baseline"].get(relative)
                if not baseline or digest(current) != baseline["sha256"] or digest(source/relative) != baseline["sha256"]:
                    stale.append(relative)
                    continue
                # Never forward raw indexed snippets: issue current source evidence.
                hit = self.read_file(ReadArgs(path=relative, start=max(1, int(match[2])-3), lines=16))
            except (ValueError, PermissionError, OSError):
                continue
            if hit["evidence_id"] not in {h["evidence_id"] for h in hits}:
                hits.append(hit)
            if len(hits) >= min(args.limit, 5):
                break
        return {"status": "source_verified_navigation", "hits": hits, "stale_paths": sorted(set(stale)),
                "fallback": self.search_exact(args) if not hits else None}

    def inspect_diff(self, args: EmptyArgs, **_):
        if self.scope is not None:
            patch = self.workspace.patch(self.scope)
            return {"patch": patch[:6500], "truncated": len(patch)>6500,
                    "note": "Read scoped baseline and candidate files for full source evidence."}
        patch = self.workspace.patch()
        return {"patch": patch[:6500], "patch_artifact": self.store.artifact(self.run_id, patch), "truncated": len(patch)>6500}

    def apply_patch(self, args: PatchArgs, *, key: str):
        receipt = self.store.receipt(self.run_id, key)
        action_hash = hashlib.sha256(args.model_dump_json().encode()).hexdigest()
        if receipt and receipt["data"].get("action_hash") != action_hash:
            raise RuntimeError("edit receipt does not match the pending action; refusing ambiguous replay")
        seen = set()
        changes = []
        for change in args.changes:
            self.require_instructions(change.path)
            if change.path in seen:
                raise ValueError("duplicate patch path")
            seen.add(change.path)
            path = self._path(change.path)
            if any(p.is_symlink() for p in [path, *path.parents] if p.is_relative_to(self.workspace.root)):
                raise ValueError("editing through symlinks is forbidden")
            if path.is_dir():
                raise ValueError(f"{change.path} is a directory; apply_patch requires a file path")
            current = digest(path)
            after = hashlib.sha256(change.content.encode()).hexdigest() if change.content is not None else None
            if receipt and current == after:
                complete = change.executable is None or not path.exists() or bool(path.stat().st_mode & 0o111) == change.executable
                changes.append((change, path, after, complete))
                continue
            if current != change.expected_hash:
                raise ValueError(f"source changed for {change.path}; read it again before editing")
            changes.append((change, path, after, False))
        desired = {c.path: {"before": c.expected_hash, "after": after} for c, p, after, done in changes}
        self.store.save_receipt(self.run_id, key, "edit", "started", {"files": desired, "action_hash": action_hash})
        for change, path, after, done in changes:
            if done:
                continue
            if change.content is None:
                if path.exists():
                    path.unlink()
                continue
            mode = stat.S_IMODE(path.stat().st_mode) if path.exists() else 0o644
            if change.executable is not None:
                mode = (mode | 0o111) if change.executable else (mode & ~0o111)
            path.parent.mkdir(parents=True, exist_ok=True)
            fd, temporary = tempfile.mkstemp(prefix=".harness-write-", dir=path.parent)
            try:
                with os.fdopen(fd, "wb") as handle:
                    handle.write(change.content.encode())
                    handle.flush()
                    os.fsync(handle.fileno())
                os.chmod(temporary, mode)
                os.replace(temporary, path)
            finally:
                if Path(temporary).exists():
                    Path(temporary).unlink()
        result = {"files": desired, "source_hash": tree_hash(self.workspace.root), "action_hash": action_hash}
        self.store.save_receipt(self.run_id, key, "edit", "completed", result)
        return result

    def run_command(self, args: CommandArgs, *, key: str):
        before = tree_hash(self.workspace.root)
        self.store.save_receipt(self.run_id, key, "command", "started", {"argv": args.argv, "cwd":args.cwd, "source_hash":before})
        try:
            result = self.checks.runner.run(
                self.workspace.root, args.argv, cwd=args.cwd, timeout=args.timeout_seconds,
                read_only=self.mode == "review" or self.checks.runner.profile.check_after_edit,
            )
        except CommandCancelled as exc:
            data = {**exc.result, "source_hash": before, "verified": False}
            data["artifact_id"] = self.store.artifact(self.run_id, data)
            self.store.save_receipt(self.run_id, key, "command", "interrupted", data)
            self.store.event(self.run_id, "command_interrupted", {"artifact_id": data["artifact_id"]})
            raise
        result["source_hash"] = before
        result["after_source_hash"] = tree_hash(self.workspace.root)
        artifact = self.store.artifact(self.run_id, result)
        result["artifact_id"] = artifact
        self.store.save_receipt(self.run_id, key, "command", "completed", result)
        return {**result, "output": result["output"][:6000], "note": "Only registered checks count as verification."}

    def register_check(self, args: RegisterArgs, **_):
        return self.checks.register(args.check, args.test_paths)

    def run_check(self, args: CheckArgs, *, key: str):
        return self.checks.execute(args.check_id, args.phase, key=key)

    def delegate(self, args: DelegateArgs, **_):
        if self.pool is None:
            raise RuntimeError("specialist pool is not available")
        return self.pool.start(args.tasks)

    def collect(self, args: CollectArgs, **_):
        if self.pool is None:
            raise RuntimeError("specialist pool is not available")
        return self.pool.collect(args.ids)
