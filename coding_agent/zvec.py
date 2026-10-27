from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal

from pydantic import BaseModel, Field

from langchain_core.tools import StructuredTool


class ZvecGrepError(RuntimeError):
    """Raised when the zvec-grep CLI cannot complete an operation."""


QueryRoute = Literal["hybrid", "fts", "vector", "rg"]


@dataclass(frozen=True)
class ZvecGrepSettings:
    executable: str = "zg"
    mode: Literal["direct", "server", "auto"] = "direct"
    query_limit: int = 7
    timeout_seconds: float = 120.0
    embedding: str | None = None

    def __post_init__(self) -> None:
        if self.query_limit <= 0 or self.timeout_seconds <= 0:
            raise ValueError("zvec-grep query_limit and timeout_seconds must be positive")


Runner = Callable[..., subprocess.CompletedProcess[str]]


class ZvecGrepTool:
    """Fixed-root adapter for zvec-grep indexing, status, and query operations.

    The adapter never invokes a shell. Query uses ``--refresh off`` so an agent
    lookup cannot silently create or rebuild a persistent index. Indexing is an
    explicit method/operation and can use the embedding configured by zg or the
    supplied model name.
    """

    def __init__(self, settings: ZvecGrepSettings | None = None, *, runner: Runner | None = None):
        self.settings = settings or ZvecGrepSettings()
        self._runner = runner or subprocess.run

    @staticmethod
    def _root(root: str | Path) -> Path:
        path = Path(root).expanduser().resolve()
        if not path.is_dir():
            raise ZvecGrepError(f"zvec-grep root is not a directory: {path}")
        return path

    def _run(self, args: list[str], root: Path) -> str:
        command = [self.settings.executable, *args]
        try:
            completed = self._runner(
                command,
                cwd=str(root),
                capture_output=True,
                text=True,
                timeout=self.settings.timeout_seconds,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise ZvecGrepError(f"zvec-grep command failed to start or timed out: {exc}") from exc
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout or "no diagnostic").strip()
            raise ZvecGrepError(f"zvec-grep exited with {completed.returncode}: {detail[:2000]}")
        return completed.stdout.strip()

    def index(
        self,
        root: str | Path,
        *,
        embedding: str | None = None,
        rebuild: bool = False,
        allow_remote: bool = False,
    ) -> str:
        """Build or explicitly rebuild the workspace index."""
        root_path = self._root(root)
        args = ["index", str(root_path), "--mode", self.settings.mode]
        model = embedding or self.settings.embedding
        if model:
            args.extend(["--embedding", model])
        if rebuild:
            args.append("--rebuild")
        if allow_remote:
            args.append("--allow-remote")
        return self._run(args, root_path)

    def status(self, root: str | Path) -> str:
        root_path = self._root(root)
        return self._run(["status", str(root_path), "--mode", self.settings.mode], root_path)

    def query(
        self,
        root: str | Path,
        query: str,
        *,
        route: QueryRoute = "hybrid",
        limit: int | None = None,
        fuse: bool = False,
    ) -> str:
        """Query an existing index, or use managed ripgrep for the ``rg`` route."""
        if not query.strip():
            raise ZvecGrepError("zvec-grep query must not be empty")
        root_path = self._root(root)
        result_limit = self.settings.query_limit if limit is None else int(limit)
        if result_limit <= 0:
            raise ZvecGrepError("zvec-grep query limit must be positive")
        args = ["query", "--mode", self.settings.mode]
        if route == "hybrid":
            args.extend(["--refresh", "off", "--limit", str(result_limit), query])
        elif route == "fts":
            args.extend(["--refresh", "off", "--limit", str(result_limit), "--fts", query])
        elif route == "vector":
            args.extend(["--refresh", "off", "--limit", str(result_limit), "--vector", query])
        elif route == "rg":
            args.extend(["--rg", query, str(root_path)])
        else:
            raise ZvecGrepError(f"unsupported zvec-grep query route: {route}")
        if fuse and route != "rg":
            args.append("--fuse")
        return self._run(args, root_path)

    def run(
        self,
        action: Literal["index", "query", "status"],
        root: str | Path,
        *,
        query: str | None = None,
        route: QueryRoute = "hybrid",
        limit: int | None = None,
        fuse: bool = False,
        embedding: str | None = None,
        rebuild: bool = False,
        allow_remote: bool = False,
    ) -> str:
        if action == "index":
            return self.index(root, embedding=embedding, rebuild=rebuild, allow_remote=allow_remote)
        if action == "status":
            return self.status(root)
        if action == "query":
            if query is None:
                raise ZvecGrepError("query is required for the zvec-grep query action")
            return self.query(root, query, route=route, limit=limit, fuse=fuse)
        raise ZvecGrepError(f"unsupported zvec-grep action: {action}")

    def as_langchain_tool(self) -> StructuredTool:
        """Expose the adapter as a typed LangChain tool for tool-capable models."""

        def invoke(
            action: Literal["index", "query", "status"],
            root: str,
            query: str | None = None,
            route: QueryRoute = "hybrid",
            limit: int | None = None,
            fuse: bool = False,
            embedding: str | None = None,
            rebuild: bool = False,
            allow_remote: bool = False,
        ) -> str:
            return self.run(
                action,
                root,
                query=query,
                route=route,
                limit=limit,
                fuse=fuse,
                embedding=embedding,
                rebuild=rebuild,
                allow_remote=allow_remote,
            )

        return StructuredTool.from_function(
            func=invoke,
            name="zvec_grep",
            description=(
                "Index or query a fixed repository root with zvec-grep. "
                "Indexing is explicit; query uses refresh=off and never silently rebuilds an index."
            ),
            args_schema=ZvecGrepInput,
        )


class ZvecGrepInput(BaseModel):
    action: Literal["index", "query", "status"] = Field(description="Operation to perform")
    root: str = Field(description="Absolute or workspace-relative repository root")
    query: str | None = Field(default=None, description="Search text for the query action")
    route: QueryRoute = Field(default="hybrid", description="hybrid, fts, vector, or managed rg")
    limit: int | None = Field(default=None, ge=1, description="Maximum result count")
    fuse: bool = Field(default=False, description="Fuse hybrid query groups")
    embedding: str | None = Field(default=None, description="Embedding model for explicit indexing")
    rebuild: bool = Field(default=False, description="Explicitly rebuild an existing index")
    allow_remote: bool = Field(default=False, description="Allow remote embedding for this index operation")


def zvec_grep_tool(settings: ZvecGrepSettings | None = None) -> StructuredTool:
    return ZvecGrepTool(settings).as_langchain_tool()
