"""SQLite registry, immutable content-addressed artifacts, and action receipts."""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import time
import uuid
import fcntl
import tempfile
from contextlib import contextmanager
from pathlib import Path

from .types import RunStatus


def default_state_dir() -> Path:
    return Path(os.environ.get("XDG_STATE_HOME", str(Path.home() / ".local/state"))) / "coding-agent"


class Store:
    def __init__(self, root: str | Path | None = None):
        self.root = Path(root or default_state_dir()).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.db_path = self.root / "registry.sqlite"
        with self.db() as db:
            db.executescript("""
            CREATE TABLE IF NOT EXISTS runs (
                id TEXT PRIMARY KEY, request TEXT NOT NULL, config TEXT NOT NULL, profile TEXT NOT NULL,
                state TEXT NOT NULL, attempt INTEGER DEFAULT 0, actions INTEGER DEFAULT 0,
                active_seconds REAL DEFAULT 0, active_from REAL, pid INTEGER,
                cancel_requested INTEGER DEFAULT 0, error TEXT, result TEXT,
                created_at REAL NOT NULL, updated_at REAL NOT NULL);
            CREATE TABLE IF NOT EXISTS events (
                seq INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT NOT NULL, kind TEXT NOT NULL,
                data TEXT NOT NULL, created_at REAL NOT NULL);
            CREATE INDEX IF NOT EXISTS events_run ON events(run_id,seq);
            CREATE TABLE IF NOT EXISTS receipts (
                run_id TEXT NOT NULL, key TEXT NOT NULL, kind TEXT NOT NULL, state TEXT NOT NULL,
                data TEXT NOT NULL, PRIMARY KEY(run_id,key));
            CREATE TABLE IF NOT EXISTS specialists (
                run_id TEXT NOT NULL, key TEXT NOT NULL, state TEXT NOT NULL, data TEXT NOT NULL,
                PRIMARY KEY(run_id,key));
            """)
            db.execute("BEGIN IMMEDIATE")
            if "model_requests" not in {r[1] for r in db.execute("PRAGMA table_info(runs)")}:
                db.execute("ALTER TABLE runs ADD COLUMN model_requests INTEGER NOT NULL DEFAULT 0")

    @contextmanager
    def db(self):
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def run_dir(self, run_id: str) -> Path:
        if len(run_id) != 32 or any(c not in "0123456789abcdef" for c in run_id):
            raise ValueError("invalid run ID")
        return self.root / "runs" / run_id

    @contextmanager
    def queue_lock(self):
        with (self.root / "queue.lock").open("a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            yield

    def create(self, request: dict, config: dict, profile: dict) -> str:
        run_id = uuid.uuid4().hex
        now = time.time()
        folder = self.run_dir(run_id)
        folder.mkdir(parents=True, mode=0o700)
        (folder / "artifacts").mkdir(mode=0o700)
        with self.db() as db:
            db.execute("INSERT INTO runs(id,request,config,profile,state,created_at,updated_at) VALUES(?,?,?,?,?,?,?)",
                       (run_id, json.dumps(request), json.dumps(config), json.dumps(profile), "queued", now, now))
        self.event(run_id, "submitted", {"profile": profile["name"], "root": request["root"]})
        return run_id

    def get(self, run_id: str) -> dict:
        self.run_dir(run_id)
        with self.db() as db:
            row = db.execute("SELECT * FROM runs WHERE id=?", (run_id,)).fetchone()
        if row is None:
            raise KeyError(f"unknown run {run_id}")
        result = dict(row)
        for key in ("request", "config", "profile", "result"):
            if result[key] is not None:
                result[key] = json.loads(result[key])
        return result

    def status(self, run_id: str) -> RunStatus:
        run = self.get(run_id)
        active = run["active_seconds"]
        if run["active_from"] is not None:
            active += max(0, time.time() - run["active_from"])
        return RunStatus(run_id=run_id, state=run["state"], attempt=run["attempt"], actions=run["actions"],
                         model_requests=run["model_requests"],
                         active_seconds=active, cancel_requested=bool(run["cancel_requested"]), error=run["error"],
                         created_at=run["created_at"], updated_at=run["updated_at"])

    def transition(self, run_id: str, state: str, *, error: str | None = None, result: dict | None = None):
        now = time.time()
        with self.db() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT * FROM runs WHERE id=?", (run_id,)).fetchone()
            if row is None:
                raise KeyError(run_id)
            active = row["active_seconds"] + (max(0, now-row["active_from"]) if row["active_from"] else 0)
            db.execute("UPDATE runs SET state=?,error=?,result=COALESCE(?,result),active_seconds=?,active_from=?,updated_at=? WHERE id=?",
                       (state, error, json.dumps(result) if result is not None else None, active,
                        now if state == "running" else None, now, run_id))
        self.event(run_id, "state", {"state": state, "error": error})

    def consume(self, run_id: str, *, kind="tool"):
        with self.db() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT * FROM runs WHERE id=?", (run_id,)).fetchone()
            p = json.loads(row["profile"])
            active = row["active_seconds"] + (time.time()-row["active_from"] if row["active_from"] else 0)
            column = "model_requests" if kind == "model" else "actions"
            limit = p.get("max_model_requests", 200) if kind == "model" else p["max_actions"]
            if row[column] >= limit or active >= p["max_active_seconds"]:
                raise RuntimeError("run budget exhausted")
            if row["cancel_requested"]:
                raise InterruptedError("run cancelled")
            db.execute(f"UPDATE runs SET {column}={column}+1 WHERE id=?", (run_id,))

    def event(self, run_id: str, kind: str, data: dict):
        with self.db() as db:
            db.execute("INSERT INTO events(run_id,kind,data,created_at) VALUES(?,?,?,?)",
                       (run_id, kind, json.dumps(data), time.time()))

    def events(self, run_id: str, cursor: int = 0, limit: int = 50) -> dict:
        self.get(run_id)
        if cursor < 0 or not 1 <= limit <= 100:
            raise ValueError("cursor must be nonnegative and limit must be 1..100")
        with self.db() as db:
            rows = db.execute("SELECT * FROM events WHERE run_id=? AND seq>? ORDER BY seq LIMIT ?",
                              (run_id, cursor, limit+1)).fetchall()
        items = [{**dict(r), "data": json.loads(r["data"])} for r in rows[:limit]]
        return {"items": items, "has_more": len(rows)>limit,
                "next_cursor": items[-1]["seq"] if items else cursor}

    def artifact(self, run_id: str, data: dict | str | bytes) -> str:
        raw = data if isinstance(data, bytes) else (data.encode() if isinstance(data, str) else json.dumps(data, sort_keys=True).encode())
        key = hashlib.sha256(raw).hexdigest()
        path = self.run_dir(run_id) / "artifacts" / key
        fd, temporary = tempfile.mkstemp(prefix=".pending-", dir=path.parent)
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(raw)
                handle.flush()
                os.fsync(handle.fileno())
            try:
                os.link(temporary, path)
            except FileExistsError:
                if path.read_bytes() != raw:
                    raise RuntimeError("artifact integrity mismatch")
        finally:
            os.unlink(temporary)
        return key

    def read_artifact(self, run_id: str, key: str, offset: int = 0, limit: int = 8000) -> dict:
        self.get(run_id)
        if len(key) != 64 or any(c not in "0123456789abcdef" for c in key):
            raise ValueError("invalid artifact ID")
        if offset < 0 or not 1 <= limit <= 32000:
            raise ValueError("invalid artifact page")
        path = self.run_dir(run_id) / "artifacts" / key
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != key:
            raise RuntimeError("artifact hash mismatch")
        content = raw.decode("utf-8", errors="replace")
        return {"artifact_id": key, "content": content[offset:offset+limit], "offset": offset,
                "next_offset": min(len(content), offset+limit), "has_more": offset+limit<len(content)}

    def receipt(self, run_id: str, key: str) -> dict | None:
        with self.db() as db:
            row = db.execute("SELECT * FROM receipts WHERE run_id=? AND key=?", (run_id, key)).fetchone()
        return {**dict(row), "data": json.loads(row["data"])} if row else None

    def save_receipt(self, run_id: str, key: str, kind: str, state: str, data: dict):
        with self.db() as db:
            db.execute("INSERT INTO receipts VALUES(?,?,?,?,?) ON CONFLICT(run_id,key) DO UPDATE SET state=excluded.state,data=excluded.data",
                       (run_id, key, kind, state, json.dumps(data)))

    def receipts(self, run_id: str, kind: str | None = None) -> list[dict]:
        with self.db() as db:
            rows = db.execute("SELECT * FROM receipts WHERE run_id=? ORDER BY rowid", (run_id,)).fetchall()
        return [{**dict(r), "data": json.loads(r["data"])} for r in rows if kind is None or r["kind"] == kind]
