from __future__ import annotations

import argparse
import json
import sys

from .config import ConfigError, load_config
from .graph import run_agent
from .zvec import ZvecGrepError, ZvecGrepSettings, ZvecGrepTool


def legacy_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the configurable LangGraph coding agent")
    parser.add_argument("task", nargs="?", help="coding task for the agent")
    parser.add_argument("--root", default=".", help="repository root to inspect")
    parser.add_argument("--config", default="agent-config.json", help="JSON configuration file")
    parser.add_argument("--print-config", action="store_true", help="validate and print non-secret configuration")
    parser.add_argument("--zvec-index", action="store_true", help="explicitly build the zvec-grep index")
    parser.add_argument("--zvec-rebuild", action="store_true", help="rebuild the zvec-grep index")
    parser.add_argument("--zvec-status", action="store_true", help="show zvec-grep index status")
    parser.add_argument("--zvec-query", help="query the existing zvec-grep index before planning")
    parser.add_argument("--zvec-route", choices=["hybrid", "fts", "vector", "rg"], default="hybrid")
    parser.add_argument("--zvec-limit", type=int, default=None)
    parser.add_argument("--zvec-fuse", action="store_true")
    parser.add_argument("--zvec-embedding", help="embedding model for explicit zvec-grep indexing")
    parser.add_argument("--zvec-allow-remote", action="store_true", help="allow remote embeddings for indexing")
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config)
    except ConfigError as exc:
        parser.error(str(exc))
    if args.print_config:
        print(json.dumps({
            "main_base_url": config.main.base_url,
            "main_model": config.main.model,
            "subagent_base_url": config.subagents.base_url,
            "subagent_model": config.subagents.model,
            "parallelism": config.parallelism,
            "zvec_executable": config.zvec.executable,
            "zvec_mode": config.zvec.mode,
            "zvec_query_limit": config.zvec.query_limit,
            "specialist_roles": list(config.specialist_roles),
        }, indent=2))
        if not args.task:
            return 0
    zvec = ZvecGrepTool(ZvecGrepSettings(
        executable=config.zvec.executable,
        mode=config.zvec.mode,  # type: ignore[arg-type]
        query_limit=config.zvec.query_limit,
        timeout_seconds=config.zvec.timeout_seconds,
        embedding=config.zvec.embedding,
    ))
    try:
        if args.zvec_status:
            print(zvec.status(args.root))
            if not args.task and not args.zvec_query:
                return 0
        if args.zvec_index:
            print(zvec.index(
                args.root,
                embedding=args.zvec_embedding,
                rebuild=args.zvec_rebuild,
                allow_remote=args.zvec_allow_remote,
            ))
            if not args.task and not args.zvec_query:
                return 0
        if args.zvec_query and not args.task:
            print(zvec.query(
                args.root,
                args.zvec_query,
                route=args.zvec_route,
                limit=args.zvec_limit,
                fuse=args.zvec_fuse,
            ))
            return 0
    except ZvecGrepError as exc:
        print(f"coding-agent: {exc}", file=sys.stderr)
        return 1
    if not args.task:
        parser.error("task is required unless --print-config is used")
    try:
        result = run_agent(args.task, args.root, config, zvec_query=args.zvec_query, zvec_tool=zvec)
    except Exception as exc:  # CLI boundary: preserve a useful nonzero failure
        print(f"coding-agent: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({
        "plan": result.get("plan", {}),
        "specialist_results": result.get("subagent_results", []),
        "synthesis": result.get("synthesis", {}),
    }, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    from .batch_cli import main as batch_main
    args = sys.argv[1:] if argv is None else argv
    commands = {"run", "submit", "status", "events", "result", "resume", "cancel", "list-runs", "artifact", "serve-mcp"}
    if args and (args[0] in commands or args[0] in {"-h", "--help"}):
        return batch_main(args)
    return legacy_main(args)


if __name__ == "__main__":
    raise SystemExit(main())
