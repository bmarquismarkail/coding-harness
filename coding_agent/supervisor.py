"""One durable local queue, independent of the CLI/MCP process lifetime."""
from __future__ import annotations

import argparse
import fcntl
import os
import signal
import subprocess
import sys
import time

from .store import Store
from .types import EvidencePackage, RunResult
from .workspace import Workspace


def record_exit(store, run_id, reason):
    current = store.get(run_id)
    state = "cancelled" if current["cancel_requested"] else "interrupted"
    workspace = Workspace(store.run_dir(run_id))
    gaps = ["Worker exited before completion; interrupted actions remain unverified."]
    patch = ""
    if (workspace.folder/"baseline.index").exists():
        try:
            patch = workspace.patch()
        except Exception as exc:
            gaps.append(f"Partial patch could not be exported: {type(exc).__name__}")
    records = store.receipts(run_id,"check")
    baseline = [{k:r["data"].get(k) for k in ("check_id", "source_hash", "passed", "exit_code", "artifact_id", "skipped", "timed_out")}
                for r in records if r["data"].get("phase") == "baseline"]
    verification = [{k:r["data"].get(k) for k in ("check_id", "source_hash", "passed", "exit_code", "artifact_id", "skipped", "timed_out")}
                    for r in records if r["data"].get("phase") == "candidate"]
    config = store.get(run_id)["config"]
    package = EvidencePackage(baseline_checks=baseline,
                              reproducer_artifacts=[r["data"]["artifact_id"] for r in records
                                                    if r["data"].get("phase") == "baseline" and r["data"].get("artifact_id")],
                              patch_artifact=store.artifact(run_id, patch),
                              verification_checks=verification, remaining_gaps=gaps,
                              skill_snapshots=config.get("_skill_metadata", []))
    result = RunResult(run_id=run_id, outcome="cancelled" if state == "cancelled" else "unverified",
                       summary=reason, patch=patch, patch_artifact=store.artifact(run_id,patch),
                       checks=[{**{k:v for k,v in r['data'].items() if k != 'output'},"receipt_state":r['state']} for r in records],
                       evidence_ids=sorted({r["data"]["artifact_id"] for r in records if r["data"].get("artifact_id")}),
                       evidence=package,
                       gaps=gaps)
    store.transition(run_id,state,error=reason,result=result.model_dump())


def serve(state_dir):
    store = Store(state_dir)
    with (store.root / "supervisor.lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return
        (store.root / "supervisor.pid").write_text(str(os.getpid()))
        # Only the lock owner reconciles an orphan; another live writer is never stolen.
        with store.db() as db:
            running = db.execute("SELECT id,pid FROM runs WHERE state='running'").fetchall()
        for row in running:
            from .service import Harness
            if Harness._alive(row["pid"]):
                # An orphaned job is still working; wait rather than over-admit.
                stopping_since = None
                while Harness._alive(row["pid"]):
                    status = store.status(row["id"])
                    if status.cancel_requested or status.active_seconds > store.get(row["id"])["profile"]["max_active_seconds"]+5:
                        if stopping_since is None:
                            try:
                                os.kill(row["pid"], signal.SIGTERM)
                            except ProcessLookupError:
                                break
                            stopping_since = time.monotonic()
                        elif time.monotonic()-stopping_since > 5:
                            try:
                                os.killpg(row["pid"], signal.SIGKILL)
                            except ProcessLookupError:
                                break
                    time.sleep(0.2)
            if store.get(row["id"])["state"] == "running":
                record_exit(store, row["id"], "worker exited without a final checkpoint")
        idle_since = time.monotonic()
        while True:
            with store.db() as db:
                row = db.execute("SELECT id FROM runs WHERE state='queued' AND cancel_requested=0 ORDER BY created_at LIMIT 1").fetchone()
            if row is None:
                if time.monotonic()-idle_since >= 60:
                    with store.queue_lock():
                        with store.db() as db:
                            pending = db.execute("SELECT 1 FROM runs WHERE state='queued' AND cancel_requested=0 LIMIT 1").fetchone()
                        if not pending:
                            fcntl.flock(lock, fcntl.LOCK_UN)
                            return
                time.sleep(0.2)
                continue
            idle_since = time.monotonic()
            run_id = row["id"]
            with (store.run_dir(run_id) / "worker.log").open("ab") as log:
                with store.queue_lock():
                    if store.get(run_id)["state"] != "queued":
                        continue
                    store.transition(run_id, "running")
                    process = subprocess.Popen([sys.executable, "-m", "coding_agent.worker", "--state-dir", str(store.root), "--run-id", run_id],
                                               stdin=subprocess.DEVNULL, stdout=log, stderr=log, start_new_session=True)
                    with store.db() as db:
                        db.execute("UPDATE runs SET pid=?,attempt=attempt+1 WHERE id=?", (process.pid, run_id))
                while process.poll() is None:
                    status = store.status(run_id)
                    if status.cancel_requested or status.active_seconds > store.get(run_id)["profile"]["max_active_seconds"]+5:
                        process.terminate()
                        try:
                            process.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            os.killpg(process.pid, signal.SIGKILL)
                    time.sleep(0.1)
                current = store.get(run_id)
                if current["pid"] == process.pid and current["state"] in {"queued", "running"}:
                    record_exit(store, run_id, f"worker exited with code {process.returncode}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--state-dir", required=True)
    serve(parser.parse_args().state_dir)
