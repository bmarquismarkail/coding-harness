"""Thin stdio/HTTP MCP adapters over the same Harness job service."""
from __future__ import annotations

import asyncio
import hmac
import os
from typing import Any
from functools import wraps

from mcp.server import MCPServer
from mcp.server.transport_security import TransportSecuritySettings
from mcp.types import ToolAnnotations
from mcp.server.mcpserver.exceptions import ToolError

from .service import Harness
from .types import TaskRequest


def create_server(harness: Harness) -> MCPServer:
    if not harness.allowed_roots:
        raise ValueError("MCP requires explicitly configured allowed repository roots")
    server = MCPServer("coding_agent_mcp", version="0.2.0", instructions=(
        "Submit an isolated coding/review job, poll status/events, and retrieve its result and patch artifact. "
        "Only outcome=verified establishes completion. Model findings may remain hypotheses. "
        "Jobs survive this MCP connection; cancel explicitly to stop them."))

    def tool(name, read=True):
        def register(fn):
            @wraps(fn)
            async def safe(**kwargs):
                try:
                    return await fn(**kwargs)
                except (ValueError, KeyError, PermissionError, FileNotFoundError) as exc:
                    raise ToolError(str(exc)[:1500]) from None
            return server.tool(name=name, annotations=ToolAnnotations(readOnlyHint=read, destructiveHint=False,
                               idempotentHint=read, openWorldHint=not read))(safe)
        return register

    @tool("harness_submit", False)
    async def submit(request: TaskRequest) -> dict[str, Any]:
        """Submit a task for an allowed repository; returns a run ID immediately."""
        return {"run_id": await asyncio.to_thread(harness.submit, request)}

    @tool("harness_status")
    async def status(run_id: str) -> dict[str, Any]:
        """Read durable execution status and budget use for one run."""
        return (await asyncio.to_thread(harness.status, run_id)).model_dump()

    @tool("harness_events")
    async def events(run_id: str, cursor: int = 0, limit: int = 20) -> dict[str, Any]:
        """Read a bounded event page; pass next_cursor for subsequent events."""
        return await asyncio.to_thread(harness.events, run_id, cursor, limit)

    @tool("harness_result")
    async def result(run_id: str) -> dict[str, Any]:
        """Get the final verdict, checks, gaps and patch artifact ID; patch content is paginated separately."""
        value = await asyncio.to_thread(harness.result, run_id)
        if value is None:
            return {"result": None, "status": (await asyncio.to_thread(harness.status, run_id)).model_dump()}
        data = value.model_dump(exclude={"patch"})
        return data

    @tool("harness_artifact")
    async def artifact(run_id: str, artifact_id: str, offset: int = 0, limit: int = 8000) -> dict[str, Any]:
        """Read a bounded immutable artifact page. Paths on the host are not accepted."""
        return await asyncio.to_thread(harness.artifact, run_id, artifact_id, offset, limit)

    @tool("harness_resume", False)
    async def resume(run_id: str, answer: str | None = None, profile: str | None = None) -> dict[str, Any]:
        """Resume an interrupted job or answer its pending question, optionally selecting a configured profile."""
        return (await asyncio.to_thread(harness.resume, run_id, answer=answer, profile=profile)).model_dump()

    @tool("harness_cancel", False)
    async def cancel(run_id: str) -> dict[str, Any]:
        """Stop a queued/running job and its model/command work while retaining evidence."""
        return (await asyncio.to_thread(harness.cancel, run_id)).model_dump()

    @tool("harness_list_runs")
    async def list_runs(offset: int = 0, limit: int = 20) -> dict[str, Any]:
        """List jobs visible under this server's configured repository roots."""
        return await asyncio.to_thread(harness.list_runs, offset, limit)

    return server


class BearerMiddleware:
    def __init__(self, app, token: str):
        self.app, self.token = app, token.encode()

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            headers = dict(scope.get("headers", []))
            value = headers.get(b"authorization", b"")
            scheme, _, key = value.partition(b" ")
            if scheme.lower() != b"bearer" or not hmac.compare_digest(key, self.token):
                await send({"type": "http.response.start", "status": 401,
                            "headers": [(b"www-authenticate", b"Bearer"), (b"content-type", b"application/json")]})
                await send({"type": "http.response.body", "body": b'{"error":"valid harness Bearer token required"}'})
                return
        await self.app(scope, receive, send)


def http_app(harness: Harness, *, token: str | None = None, allowed_hosts=None, allowed_origins=None):
    token = token if token is not None else os.environ.get("HARNESS_MCP_API_KEY")
    if not token:
        raise ValueError("HTTP MCP requires HARNESS_MCP_API_KEY (separate from LLAMA_API_KEY)")
    model_key = os.environ.get("LLAMA_API_KEY")
    if model_key and hmac.compare_digest(token.encode(), model_key.encode()):
        raise ValueError("HARNESS_MCP_API_KEY must differ from LLAMA_API_KEY")
    server = create_server(harness)
    app = server.streamable_http_app(json_response=True, stateless_http=True, transport_security=TransportSecuritySettings(
        enable_dns_rebinding_protection=True,
        allowed_hosts=allowed_hosts or ["127.0.0.1:*", "localhost:*", "[::1]:*"],
        allowed_origins=allowed_origins or []))
    return BearerMiddleware(app, token)


def serve(args):
    harness = Harness(args.state_dir, config_path=args.config, allowed_roots=args.allow_root,
                      allowed_dependencies=args.allow_dependency)
    if args.transport == "stdio":
        create_server(harness).run(transport="stdio")
    else:
        import uvicorn
        app = http_app(harness, allowed_hosts=args.allowed_host or None, allowed_origins=args.allowed_origin)
        uvicorn.run(app, host=args.host, port=args.port, log_level="info")
