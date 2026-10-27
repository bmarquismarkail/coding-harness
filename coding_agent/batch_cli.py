from __future__ import annotations

import argparse
import json
import sys

from .service import Harness
from .types import TaskRequest


def parser():
    cli = argparse.ArgumentParser(description="Durable isolated coding jobs (legacy positional tasks remain read-only)")
    commands = cli.add_subparsers(dest="command", required=True)
    for name in ("run", "submit", "status", "events", "result", "resume", "cancel", "list-runs", "artifact", "serve-mcp"):
        sub = commands.add_parser(name)
        sub.add_argument("--state-dir")
        sub.add_argument("--config")
        if name in {"run", "submit"}:
            sub.add_argument("task")
            sub.add_argument("--root", default=".")
            sub.add_argument("--mode", choices=["implement", "review"], default="implement")
            sub.add_argument("--task-type", choices=["bugfix", "feature", "refactor"], default="bugfix")
            sub.add_argument("--profile")
            sub.add_argument("--checks", help="JSON file containing a list of CheckSpec objects")
            sub.add_argument("--dependency", action="append", default=[])
            sub.add_argument("--wait", action="store_true")
            sub.add_argument("--timeout", type=float, default=3600)
        if name in {"status", "events", "result", "resume", "cancel", "artifact"}:
            sub.add_argument("run_id")
        if name in {"events", "list-runs", "artifact"}:
            sub.add_argument("--limit", type=int, default=8000 if name == "artifact" else 20)
        if name == "events":
            sub.add_argument("--cursor", type=int, default=0)
        if name in {"list-runs", "artifact"}:
            sub.add_argument("--offset", type=int, default=0)
        if name == "artifact":
            sub.add_argument("artifact_id")
        if name == "resume":
            sub.add_argument("--answer")
            sub.add_argument("--profile")
        if name == "serve-mcp":
            sub.add_argument("--transport", choices=["stdio", "streamable-http"], default="stdio")
            sub.add_argument("--host", default="127.0.0.1")
            sub.add_argument("--port", type=int, default=8765)
            sub.add_argument("--allow-root", action="append", required=True)
            sub.add_argument("--allow-dependency", action="append", default=[])
            sub.add_argument("--allowed-host", action="append", default=[])
            sub.add_argument("--allowed-origin", action="append", default=[])
    return cli


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        if args.command == "serve-mcp":
            from .mcp_server import serve
            serve(args)
            return 0
        harness = Harness(args.state_dir, config_path=args.config)
        name = args.command
        if name in {"run", "submit"}:
            from pathlib import Path
            checks = json.loads(Path(args.checks).read_text()) if args.checks else []
            run_id = harness.submit(TaskRequest(root=args.root, task=args.task, mode=args.mode, task_type=args.task_type,
                                                profile=args.profile, checks=checks, readonly_dependencies=args.dependency))
            if not args.wait:
                result = {"run_id": run_id, "state": "queued"}
            else:
                status = harness.wait(run_id, args.timeout)
                run_result = harness.result(run_id)
                result = {"status": status.model_dump(), "result": run_result.model_dump() if run_result else None}
                print(json.dumps(result, indent=2))
                return 0 if run_result and run_result.outcome == "verified" else 1
        elif name == "status":
            result = harness.status(args.run_id).model_dump()
        elif name == "events":
            result = harness.events(args.run_id, args.cursor, args.limit)
        elif name == "result":
            value = harness.result(args.run_id)
            result = value.model_dump() if value else {"status": harness.status(args.run_id).model_dump(), "result": None}
        elif name == "resume":
            result = harness.resume(args.run_id, answer=args.answer, profile=args.profile).model_dump()
        elif name == "cancel":
            result = harness.cancel(args.run_id).model_dump()
        elif name == "list-runs":
            result = harness.list_runs(args.offset, args.limit)
        else:
            result = harness.artifact(args.run_id, args.artifact_id, args.offset, args.limit)
        print(json.dumps(result, indent=2))
        return 0
    except (ValueError, KeyError, PermissionError, OSError, TimeoutError) as exc:
        print(f"coding-agent: {exc}", file=sys.stderr)
        return 1
