"""Shared public job service used by Python, batch CLI, and MCP."""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
import threading
from dataclasses import asdict
from pathlib import Path

from .config import AgentConfig, load_config
from .profiles import resolve_profile
from .project_profiles import load_definitions, resolve_request, skill_metadata
from .store import Store
from .types import RunResult, TaskRequest
from .workspace import checked_path

TERMINAL = {"completed", "cancelled", "failed", "interrupted", "needs_input"}


class Harness:
    def __init__(self, state_dir: str | Path | None = None, *, config_path: str | Path | None = None,
                 config: AgentConfig | None = None, profiles: dict | None = None,
                 project_profiles: dict | None = None,
                 allowed_roots: list[str] | None = None, allowed_dependencies: list[str] | None = None,
                 autostart: bool = True):
        self.store = Store(state_dir)
        self.config = config or load_config(config_path)
        self.profiles = dict(profiles or {})
        selected_config = Path(config_path or "agent-config.json")
        self.project_profiles = load_definitions()
        self.project_profiles.update(project_profiles or {})
        if selected_config.exists():
            raw_config = json.loads(selected_config.read_text())
            self.profiles = {**raw_config.get("profiles", {}), **self.profiles}
            self.project_profiles.update(raw_config.get("project_profiles", {}))
        self.allowed_roots = [Path(p).expanduser().resolve() for p in allowed_roots] if allowed_roots is not None else None
        self.allowed_dependencies = [Path(p).expanduser().resolve() for p in (allowed_dependencies or [])]
        self.autostart = autostart

    def _authorize(self, root: str, *, require_exists=True) -> Path:
        path = Path(root).expanduser().resolve()
        if require_exists and not path.is_dir() or path == Path("/") or path == Path.home():
            raise ValueError("select a specific existing project directory")
        if self.allowed_roots is not None and not any(path.is_relative_to(p) for p in self.allowed_roots):
            raise PermissionError("repository is outside the configured allowed roots")
        if path == self.store.root or self.store.root.is_relative_to(path):
            raise ValueError("job state directory must be outside the source snapshot")
        return path

    def _authorize_run(self, run_id: str):
        row = self.store.get(run_id)
        self._authorize(row["request"]["root"], require_exists=False)
        return row

    def submit(self, request: TaskRequest | dict) -> str:
        """Authorize a request and queue it with frozen configuration and probes."""
        request = TaskRequest.model_validate(request)
        source = self._authorize(request.root)
        for focus in request.focus_paths:
            checked_path(source, focus)
        request, skills = resolve_request(request, source, self.project_profiles)
        for dependency in request.readonly_dependencies:
            p = Path(dependency).expanduser().resolve()
            if self.allowed_roots is not None and not any(p.is_relative_to(r) for r in self.allowed_dependencies):
                raise PermissionError("dependency mount is not configured for this MCP server")
            if not p.exists() or p == Path("/") or p == Path.home():
                raise ValueError("dependency must be a specific existing path")
        request = request.model_copy(update={"root": str(source), "readonly_dependencies": [str(Path(p).resolve()) for p in request.readonly_dependencies]})
        profile = resolve_profile(request.profile or ("review" if request.mode == "review" else "coding"), self.config, self.profiles)
        frozen_config = asdict(self.config)
        frozen_config["_resolved_skills"] = skills
        frozen_config["_skill_metadata"] = skill_metadata(skills)
        with self.store.queue_lock():
            run_id = self.store.create(request.model_dump(), frozen_config, profile.model_dump())
            for probe in request.defect_probes:
                definition = probe.model_dump()
                definition["check_spec"] = next(c.model_dump() for c in request.checks if c.id == probe.check_id)
                definition["wrapper_artifact"] = self.store.artifact(run_id, probe.wrapper)
                definition_id = self.store.artifact(run_id, definition)
                self.store.save_receipt(run_id, "probe:" + probe.defect_id, "probe_definition", "completed",
                                        {"artifact_id": definition_id})
            if self.autostart:
                self._start_supervisor()
        return run_id

    def _start_supervisor(self):
        # Supervisor uses an OS flock, so simultaneous CLI/MCP clients cannot
        # admit competing writers. It runs independently of the calling process.
        env = os.environ.copy()
        package_root = str(Path(__file__).resolve().parent.parent)
        env["PYTHONPATH"] = package_root + (os.pathsep+env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        with (self.store.root / "supervisor.log").open("ab") as log:
            process = subprocess.Popen([sys.executable, "-m", "coding_agent.supervisor", "--state-dir", str(self.store.root)],
                             stdin=subprocess.DEVNULL, stdout=log, stderr=log, start_new_session=True,
                             cwd=package_root, env=env, close_fds=True)
        threading.Thread(target=process.wait, daemon=True).start()

    def status(self, run_id: str):
        self._authorize_run(run_id)
        return self.store.status(run_id)

    def events(self, run_id: str, cursor: int = 0, limit: int = 50):
        self._authorize_run(run_id)
        return self.store.events(run_id, cursor, limit)

    def result(self, run_id: str) -> RunResult | None:
        row = self._authorize_run(run_id)
        return RunResult.model_validate(row["result"]) if row["result"] else None

    def artifact(self, run_id: str, artifact_id: str, offset: int = 0, limit: int = 8000):
        self._authorize_run(run_id)
        return self.store.read_artifact(run_id, artifact_id, offset, limit)

    def list_runs(self, offset: int = 0, limit: int = 20):
        if offset < 0 or not 1 <= limit <= 100:
            raise ValueError("invalid run page")
        with self.store.db() as db:
            clauses, parameters = [], []
            for root in self.allowed_roots or []:
                prefix = str(root).rstrip(os.sep)+os.sep
                clauses.append("(json_extract(request,'$.root')=? OR substr(json_extract(request,'$.root'),1,?)=?)")
                parameters.extend([str(root), len(prefix), prefix])
            where = "1" if self.allowed_roots is None else " OR ".join(clauses) or "0"
            total = db.execute(f"SELECT COUNT(*) FROM runs WHERE {where}", parameters).fetchone()[0]
            rows = db.execute(f"SELECT id FROM runs WHERE {where} ORDER BY created_at DESC LIMIT ? OFFSET ?",
                              [*parameters, limit, offset]).fetchall()
        return {"items": [self.store.status(row["id"]).model_dump() for row in rows],
                "total": total, "has_more": offset+limit<total, "next_offset": offset+limit}

    def cancel(self, run_id: str):
        with self.store.queue_lock():
            return self._cancel(run_id)

    def _cancel(self, run_id: str):
        row = self._authorize_run(run_id)
        if row["state"] in {"completed", "cancelled", "failed"}:
            return self.status(run_id)
        with self.store.db() as db:
            db.execute("UPDATE runs SET cancel_requested=1,updated_at=? WHERE id=?", (time.time(), run_id))
        if row["state"] != "running":
            retained = row["result"] or RunResult(run_id=run_id, outcome="cancelled", summary="Cancelled before execution").model_dump()
            retained.update(outcome="cancelled", summary="Cancelled by caller; completed work remains retained")
            self.store.transition(run_id, "cancelled", error="cancelled by caller", result=retained)
        return self.status(run_id)

    def resume(self, run_id: str, *, answer: str | None = None, profile: str | None = None):
        with self.store.queue_lock():
            return self._resume(run_id, answer=answer, profile=profile)

    def _resume(self, run_id: str, *, answer: str | None = None, profile: str | None = None):
        row = self._authorize_run(run_id)
        if row["state"] not in {"interrupted", "needs_input"}:
            # A crashed worker can leave a running row; do not steal a live lease.
            if row["state"] != "running" or self._alive(row["pid"]):
                raise ValueError("only an interrupted or needs_input run can be resumed")
            self.store.transition(run_id, "interrupted", error="worker exited without a terminal receipt")
            row = self.store.get(run_id)
        config = dict(row["config"])
        selected = row["profile"]
        if profile and profile != selected["name"]:
            selected = resolve_profile(profile, self.config, self.profiles).model_dump()
            config["_revision"] = config.get("_revision", 0)+1
        if row["state"] == "needs_input":
            if profile and profile != row["profile"]["name"]:
                raise ValueError("answer the pending question before changing profiles")
            if not answer or not answer.strip():
                raise ValueError("this run needs an answer to its pending question")
            with self.store.db() as db:
                question = db.execute("SELECT data FROM events WHERE run_id=? AND kind='question' ORDER BY seq DESC LIMIT 1", (run_id,)).fetchone()
            config.update(_answer=answer, _answer_for=json.loads(question["data"])["id"])
        with self.store.db() as db:
            db.execute("UPDATE runs SET config=?,profile=?,cancel_requested=0,result=NULL,pid=NULL WHERE id=?",
                       (json.dumps(config), json.dumps(selected), run_id))
        self.store.transition(run_id, "queued")
        if self.autostart:
            self._start_supervisor()
        return self.status(run_id)

    def wait(self, run_id: str, timeout: float = 3600):
        end = time.monotonic()+timeout
        while time.monotonic()<end:
            status = self.status(run_id)
            if status.state in TERMINAL:
                return status
            time.sleep(0.1)
        raise TimeoutError("job is still running; inspect it using its run ID")

    @staticmethod
    def _alive(pid):
        if not pid:
            return False
        try:
            os.kill(pid, 0)
            proc = Path(f"/proc/{pid}/stat")
            if proc.exists() and proc.read_text().rsplit(")", 1)[1].split()[0] == "Z":
                return False
            return True
        except (ProcessLookupError, FileNotFoundError):
            return False
