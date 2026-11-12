"""Real SDK clients, real transports, detached workers and a local model server."""
from __future__ import annotations

import asyncio
import json
import os
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from collections import deque
from contextlib import AsyncExitStack
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

import httpx
import httpx2
from mcp import Client, StdioServerParameters
from mcp.client.streamable_http import streamable_http_client

from coding_agent import Harness, CheckSpec
from coding_agent.engine import Engine
from coding_agent.evidence import read_json
from coding_agent.review_types import DefectProbeSpec
from coding_agent.workspace import Workspace, tree_hash
from coding_agent.mcp_server import http_app
from test_v2 import EMPTY_REVIEW_JSON, cited_source, fixture


class ModelServer:
    def __init__(self):
        self.requests = []
        self.scripts = {}
        self.blocked = threading.Event()
        self.writer_finished = threading.Event()
        self.release = threading.Event()
        owner = self
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_):
                pass
            def do_POST(self):
                data = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                owner.requests.append((self.path, self.headers.get("Authorization"), data))
                contents = [m["content"] for m in data["messages"]]
                if "TASK: slow" in "\n".join(contents):
                    owner.release.wait(15)
                answered = any('"user_answer"' in s for s in contents)
                action = {"type":"finish","summary":"Scoped review completed","findings":[],"gaps":[]}
                if self.path == "/reviewer/v1/chat/completions":
                    if data.get("response_format", {}).get("type") == "json_object" and not any(
                        "json" in content.lower() for content in contents
                    ):
                        self.send_error(400, "JSON-object mode requires JSON in the messages")
                        return
                    action = json.loads(EMPTY_REVIEW_JSON)
                elif "TASK: question" in "\n".join(contents) and not answered:
                    action = {"type":"input","question":"Which compatibility target should this review use?"}
                raw = json.dumps({"choices":[{"finish_reason":"stop","message":{"content":json.dumps(action)}}]}).encode()
                role = self.path.split("/")[1]
                script = owner.scripts.get(role)
                if script:
                    choice = script.popleft()
                    if role == "main" and not script:
                        owner.writer_finished.set()
                    if role == "subagents":
                        owner.writer_finished.wait(5)
                    if callable(choice):
                        choice = choice(data["messages"])
                    if choice is None:
                        owner.blocked.set()
                        owner.release.wait(15)
                    else:
                        raw = json.dumps({"choices": [choice]}).encode()
                try:
                    self.send_response(200); self.send_header("Content-Type","application/json")
                    self.send_header("Content-Length",str(len(raw))); self.end_headers(); self.wfile.write(raw)
                except (BrokenPipeError, ConnectionResetError):
                    pass
        self.server = ThreadingHTTPServer(("127.0.0.1",0),Handler)
        self.thread = threading.Thread(target=self.server.serve_forever,daemon=True); self.thread.start()

    def close(self):
        self.release.set(); self.server.shutdown(); self.server.server_close(); self.thread.join()


class HttpTransport:
    def __init__(self,url,token):
        self.url,self.token=url,token
    async def __aenter__(self):
        self.stack=AsyncExitStack()
        http=await self.stack.enter_async_context(httpx2.AsyncClient(headers={"Authorization":f"Bearer {self.token}"}))
        streams=await self.stack.enter_async_context(streamable_http_client(self.url,http_client=http))
        return streams[:2]
    async def __aexit__(self,*args):
        return await self.stack.__aexit__(*args)


class MCPAcceptance(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp=tempfile.TemporaryDirectory(); self.root=Path(self.temp.name)
        self.source=self.root/"source"; fixture(self.source)
        self.model=ModelServer(); self.env={**os.environ,"LLAMA_API_KEY":"fixture-model-key","HARNESS_MCP_API_KEY":"fixture-mcp-key","PYTHONDONTWRITEBYTECODE":"1"}
        config=json.loads((Path(__file__).resolve().parents[1]/"agent-config.json").read_text())
        for role in ("main","subagents","reviewer"):
            config[role]["base_url"]=f"http://127.0.0.1:{self.model.server.server_port}/{role}/v1"
            config[role]["api_key_env"]="LLAMA_API_KEY"
        self.config=self.root/"config.json"; self.config.write_text(json.dumps(config))
        self.state=self.root/"state"; self.harness=Harness(self.state,config_path=self.config,autostart=False)
        self.processes=[]

    async def asyncTearDown(self):
        self.model.release.set()
        for process in self.processes:
            if process.poll() is None:
                process.terminate()
            await asyncio.to_thread(process.wait,timeout=10)
        # Only stop this fixture's detached supervisor after every job is terminal.
        pidfile=self.state/"supervisor.pid"
        if pidfile.exists():
            try:
                os.kill(int(pidfile.read_text()),signal.SIGTERM)
            except ProcessLookupError:
                pass
        await asyncio.to_thread(self.model.close)
        self.temp.cleanup()

    async def call(self,client,name,**arguments):
        result=await client.call_tool("harness_"+name,arguments)
        self.assertFalse(result.is_error, result.content)
        return result.structured_content

    async def wait_state(self,run_id,states):
        for _ in range(150):
            status=self.harness.status(run_id)
            if status.state in states:
                return status
            await asyncio.sleep(.05)
        folder=self.harness.store.run_dir(run_id)
        self.fail(f"Timed out: {self.harness.status(run_id)}; log={(folder/'worker.log').read_text() if (folder/'worker.log').exists() else ''}")

    async def exercise(self,connect):
        # The MCP server may disconnect after submission; the durable worker continues.
        async with Client(connect()) as client:
            tools=await client.list_tools()
            self.assertEqual(len(tools.tools),8)
            submission=await self.call(client,"submit",request={"root":str(self.source),"task":"question","mode":"review"})
            run_id=submission["run_id"]
        status=await self.wait_state(run_id,{"needs_input","interrupted","failed"})
        self.assertEqual(status.state,"needs_input",status.error)
        async with Client(connect()) as client:
            self.assertEqual((await self.call(client,"status",run_id=run_id))["state"],"needs_input")
            first=await self.call(client,"events",run_id=run_id,limit=1)
            self.assertTrue(first["has_more"])
            second=await self.call(client,"events",run_id=run_id,cursor=first["next_cursor"],limit=1)
            self.assertGreater(second["next_cursor"],first["next_cursor"])
            await self.call(client,"resume",run_id=run_id,answer="Python 3.14 and C++17")
            status=await self.wait_state(run_id,{"completed","interrupted","failed"})
            self.assertEqual(status.state,"completed",status.error)
            result=await self.call(client,"result",run_id=run_id)
            self.assertEqual(result["outcome"],"unverified")
            self.assertEqual(result["review"], self.harness.result(run_id).review.model_dump())
            self.assertTrue(any(event["kind"] == "independent_model_review"
                                for event in self.harness.events(run_id, limit=100)["items"]))
            self.assertNotIn("patch",result)
            artifact=await self.call(client,"artifact",run_id=run_id,artifact_id=result["patch_artifact"],limit=1)
            self.assertEqual(artifact["content"],"")
            self.assertTrue((await self.call(client,"list_runs",limit=1))["items"])
            denied=await client.call_tool("harness_submit",{"request":{"root":str(self.root),"task":"outside root","mode":"review"}})
            self.assertTrue(denied.is_error)
            slow=(await self.call(client,"submit",request={"root":str(self.source),"task":"slow","mode":"review"}))["run_id"]
            await self.wait_state(slow,{"running"})
            for _ in range(100):
                if any(e["kind"]=="model_started" for e in self.harness.events(slow)["items"]):
                    break
                await asyncio.sleep(.02)
            await self.call(client,"cancel",run_id=slow)
            self.assertEqual((await self.wait_state(slow,{"cancelled","failed"})).state,"cancelled")
            self.assertEqual((await self.call(client,"result",run_id=slow))["outcome"],"cancelled")
        self.assertTrue(self.model.requests)
        self.assertTrue(any(path == "/reviewer/v1/chat/completions" for path, _, _ in self.model.requests))
        self.assertTrue(all(auth=="Bearer fixture-model-key" for _,auth,_ in self.model.requests))

    async def test_stdio_real_client_survives_disconnect_resume_and_cancel(self):
        def connect():
            return StdioServerParameters(command=sys.executable,args=["-m","coding_agent.cli","serve-mcp","--transport","stdio",
                "--state-dir",str(self.state),"--config",str(self.config),"--allow-root",str(self.source)],env=self.env)
        await self.exercise(connect)

    async def test_http_real_client_auth_host_origin_resume_and_cancel(self):
        with socket.socket() as sock:
            sock.bind(("127.0.0.1",0)); port=sock.getsockname()[1]
        url=f"http://127.0.0.1:{port}/mcp"
        log=(self.root/"http.log").open("wb")
        process=subprocess.Popen([sys.executable,"-m","coding_agent.cli","serve-mcp","--transport","streamable-http",
            "--port",str(port),"--state-dir",str(self.state),"--config",str(self.config),"--allow-root",str(self.source)],
            env=self.env,stdin=subprocess.DEVNULL,stdout=log,stderr=log)
        self.processes.append(process); log.close()
        async with httpx.AsyncClient() as http:
            for _ in range(100):
                try:
                    response=await http.get(url)
                    break
                except httpx.ConnectError:
                    await asyncio.sleep(.05)
            else:
                self.fail((self.root/"http.log").read_text())
            self.assertEqual(response.status_code,401)
            self.assertEqual((await http.get(url,headers={"Authorization":"Bearer wrong"})).status_code,401)
            self.assertEqual((await http.get(url,headers={"Authorization":"Bearer fixture-model-key"})).status_code,401)
            for headers in ({"Host":"attacker.invalid"},{"Origin":"https://attacker.invalid"}):
                response=await http.get(url,headers={"Authorization":"Bearer fixture-mcp-key",**headers})
                self.assertIn(response.status_code,{400,403,421},response.text)
        await self.exercise(lambda:HttpTransport(url,"fixture-mcp-key"))

    async def test_http_requires_separate_token(self):
        with unittest.mock.patch.dict(os.environ,{"HARNESS_MCP_API_KEY":"","LLAMA_API_KEY":"model-only"}):
            with self.assertRaisesRegex(ValueError,"HARNESS_MCP_API_KEY"):
                http_app(self.harness)

    async def test_batch_cli_submission_disconnect_and_python_parity(self):
        async def cli(*args):
            result=await asyncio.to_thread(subprocess.run,[sys.executable,"-m","coding_agent.cli",*args,
                "--state-dir",str(self.state),"--config",str(self.config)],env=self.env,capture_output=True,text=True,timeout=10)
            self.assertEqual(result.returncode,0,result.stderr)
            return json.loads(result.stdout)
        submitted=await cli("submit","question","--root",str(self.source),"--mode","review")
        run_id=submitted["run_id"]
        self.assertEqual((await self.wait_state(run_id,{"needs_input","interrupted"})).state,"needs_input")
        status=await cli("status",run_id)
        self.assertEqual(status["state"],self.harness.status(run_id).state)
        await cli("resume",run_id,"--answer","Preserve the public API")
        self.assertEqual((await self.wait_state(run_id,{"completed","interrupted"})).state,"completed")
        self.assertEqual(await cli("result",run_id),self.harness.result(run_id).model_dump())

    async def run_response_script(self, role, responses, *, max_requests=20, cancel=False):
        self.model.requests.clear()
        self.model.scripts.clear()
        self.model.writer_finished.clear()
        finish = {"type": "finish", "summary": "Done"}
        draft = {"type": "tool", "tool": "report_findings", "args": {"findings": [
            {"id": "latency", "title": "Inspect latency", "claim": "Possible issue", "path": "latency.py"}]}}

        def choice(action):
            return {"finish_reason": "stop", "message": {"content": json.dumps(action)}}

        if role == "subagents":
            delegate = {"type": "tool", "tool": "delegate", "args": {"tasks": [
                {"id": "latency", "task": "Review latency", "paths": ["latency.py"]}]}}
            self.model.scripts["main"] = deque(map(choice, [delegate, finish]))
        self.model.scripts[role] = deque([choice(draft), *responses, choice(finish)])
        self.harness.profiles["recovery"] = {"protocol_failure_limit": 2, "max_model_requests": max_requests}
        rid = self.harness.submit({"root": str(self.source), "task": "Test response recovery", "mode": "review",
                                   "profile": "recovery", "focus_paths": ["latency.py"]})
        with patch.dict(os.environ, self.env):
            execution = asyncio.create_task(asyncio.to_thread(Engine(self.harness.store, rid).execute))
            if cancel:
                try:
                    self.assertTrue(await asyncio.to_thread(self.model.blocked.wait, 5))
                    self.harness.cancel(rid)
                finally:
                    await asyncio.wait_for(execution, 10)
            else:
                await asyncio.wait_for(execution, 10)
        return rid

    async def test_loopback_recovers_response_failures_in_both_roles(self):
        from coding_agent.qualification import report

        read = {"finish_reason": "stop", "message": {"content": json.dumps({
            "type": "tool", "tool": "read_file", "args": {"path": "latency.py"}})}}
        cases = [
            ("truncated", {"finish_reason": "length", "message": read["message"]}),
            ("empty_content", {"finish_reason": "stop", "message": {}}),
            ("empty_content", {"finish_reason": "stop", "message": {"content": None}}),
            ("empty_content", {"finish_reason": "stop", "message": {"content": " \n "}}),
        ]
        attempts = []
        for role in ("main", "subagents"):
            for category, failure in cases:
                with self.subTest(role=role, failure=failure):
                    rid = await self.run_response_script(role, [failure, read])
                    result = self.harness.result(rid)
                    self.assertEqual(self.harness.status(rid).state, "completed")
                    self.assertNotIn("protocol_failure", result.review.partial_reasons)
                    events = self.harness.events(rid, limit=100)["items"]
                    tools = [e["data"]["tool"] for e in events if e["kind"] == "tool"]
                    self.assertEqual(tools.count("read_file"), 1)
                    requests = [payload for path, _, payload in self.model.requests if path.startswith(f"/{role}/")]
                    self.assertEqual(len(requests), 4)
                    failures = [e["data"] for e in events if e["kind"] == "model_failed"]
                    self.assertEqual([e["category"] for e in failures], [category])
                    self.assertIn(failures[0]["error"], requests[2]["messages"][-1]["content"])
                    artifact = self.harness.artifact(rid, failures[0]["response_artifact"])
                    self.assertEqual(json.loads(artifact["content"])["response"]["choices"], [failure])
                    self.assertEqual(self.harness.status(rid).model_requests, len(self.model.requests))
                    attempts.append([False, True])
        metrics = report({"protocol": attempts})["metrics"]
        self.assertEqual(metrics["first_response"]["observed"], 0)
        self.assertEqual(metrics["after_recovery"]["observed"], 1)

    async def test_supervisor_recovers_drafts_after_worker_is_killed(self):
        (self.source / "value.py").write_text("value = 99\n")
        for case in ("valid", "missing", "stale", "cancelled"):
            with self.subTest(case=case):
                self.model.blocked.clear()
                self.model.release.clear()
                read = {"type": "tool", "tool": "read_file", "args": {"path": "value.py"}}

                def choice(action):
                    return {"finish_reason": "stop", "message": {"content": json.dumps(action)}}

                def draft(messages):
                    return choice({"type": "tool", "tool": "report_findings", "args": {"findings": [
                        {"id": "upper", "title": "Upper bound", "claim": "Check value",
                         "path": "value.py", "defect_id": "upper", "evidence_ids": cited_source(messages)}]}})

                self.model.scripts = {"main": deque([choice(read), draft, None])}
                probe = DefectProbeSpec(defect_id="upper", observation="The fixture returns 99.", check_id="probe",
                    path="value.py", phase="candidate", source_hash=tree_hash(self.source),
                    wrapper='from pathlib import Path\nimport sys\nsys.exit(42 if "99" in Path("value.py").read_text() else 0)\n',
                    observed_exit_codes=[42], not_observed_exit_codes=[0])
                with patch.dict(os.environ, self.env):
                    service = Harness(self.state, config_path=self.config)
                    rid = service.submit({"root": str(self.source), "task": "Review value", "mode": "review",
                        "focus_paths": ["value.py"], "checks": [CheckSpec(id="probe", argv=["ignored"]).model_dump()],
                        "defect_probes": [probe.model_dump()]})
                try:
                    self.assertTrue(await asyncio.to_thread(self.model.blocked.wait, 10))
                    receipt, = service.store.receipts(rid, "draft_findings")
                    if case == "missing":
                        with service.store.db() as db:
                            db.execute("DELETE FROM receipts WHERE run_id=? AND kind='check' "
                                       "AND json_extract(data,'$.phase')='candidate'", (rid,))
                    if case == "stale":
                        workspace = Workspace(service.store.run_dir(rid))
                        (workspace.root / "unrelated.txt").write_text("Changed candidate snapshot\n")
                    # A corrupt draft must not hide the subsequent valid draft.
                    with service.store.db() as db:
                        db.execute("DELETE FROM receipts WHERE run_id=? AND key=?", (rid, receipt["key"]))
                    service.store.save_receipt(rid, "broken-draft", "draft_findings", "completed",
                                               {"artifact_id": "0"*64})
                    service.store.save_receipt(rid, receipt["key"], receipt["kind"], receipt["state"], receipt["data"])
                    checks = service.store.receipts(rid, "check")
                    requests = len(self.model.requests)
                    expected_engine = Engine(service.store, rid, object())
                    try:
                        expected = expected_engine.partial_result("unverified", "worker interrupted")
                    finally:
                        expected_engine.pool.close()
                    if case == "cancelled":
                        service.cancel(rid)
                    pid = service.store.get(rid)["pid"]
                    os.kill(pid, signal.SIGKILL)
                    await self.wait_state(rid, {"cancelled" if case == "cancelled" else "interrupted"})
                    result = service.result(rid)
                    self.assertEqual(result.outcome, "cancelled" if case == "cancelled" else "unverified")
                    self.assertEqual(result.review.completion, "partial")
                    self.assertEqual([f.model_dump() for f in result.review.findings], expected["review"]["findings"])
                    finding, = result.review.findings
                    self.assertEqual(finding.status, "reproduced" if case in {"valid", "cancelled"} else "needs_human_review")
                    self.assertTrue(finding.proposal.evidence_ids)
                    for aid in finding.proposal.evidence_ids:
                        source = read_json(service.store, rid, aid)
                        self.assertEqual(source["path"], "value.py")
                    read_json(service.store, rid, result.review.bundle_id)
                    read_json(service.store, rid, result.review.adjudication_id)
                    self.assertTrue(any("broken-draft" in gap for gap in result.gaps))
                    self.assertEqual(service.store.receipts(rid, "check"), checks)
                    self.assertEqual(len(self.model.requests), requests)
                finally:
                    self.model.release.set()

    async def test_loopback_response_recovery_respects_terminal_boundaries(self):
        length = {"finish_reason": "length", "message": {"content": ""}}
        for role in ("main", "subagents"):
            for boundary in ("protocol_failure", "budget_exhausted", "cancelled"):
                with self.subTest(role=role, boundary=boundary):
                    # The specialist shares the writer's delegation and finish requests.
                    extra = 2 if role == "subagents" else 0
                    responses = [length, length] if boundary == "protocol_failure" else [length]
                    if boundary == "cancelled":
                        self.model.blocked.clear()
                        responses.append(None)
                    limit = 2 + extra if boundary == "budget_exhausted" else 20
                    rid = await self.run_response_script(role, responses, max_requests=limit,
                                                         cancel=boundary == "cancelled")
                    result = self.harness.result(rid)
                    self.assertEqual(result.outcome, "cancelled" if boundary == "cancelled" else "unverified")
                    self.assertEqual(result.review.completion, "partial")
                    if boundary != "cancelled":
                        self.assertIn(boundary, result.review.partial_reasons)
                    self.assertEqual([f.proposal.id for f in result.review.findings], ["latency"])
                    expected = 2 if boundary == "budget_exhausted" else 3
                    self.assertEqual(sum(path.startswith(f"/{role}/") for path, _, _ in self.model.requests), expected)
                    self.assertEqual(self.harness.status(rid).model_requests, len(self.model.requests))

    async def test_supervisor_admits_only_one_active_job(self):
        with patch.dict(os.environ,self.env):
            service=Harness(self.state,config_path=self.config)
            first=service.submit({"root":str(self.source),"task":"slow","mode":"review"})
            await self.wait_state(first,{"running"})
            second=service.submit({"root":str(self.source),"task":"quick","mode":"review"})
            await asyncio.sleep(.3)
            self.assertEqual(service.status(second).state,"queued")
            self.assertFalse(any(e["kind"]=="model_started" for e in service.events(second)["items"]))
            # Restart only this fixture's supervisor; its orphaned worker must
            # retain its lease while the queue waits, then cancellation works.
            pid=int((self.state/"supervisor.pid").read_text())
            os.kill(pid,signal.SIGKILL)
            for _ in range(100):
                if not Harness._alive(pid): break
                await asyncio.sleep(.02)
            service._start_supervisor()
            service.cancel(first)
            self.assertEqual((await self.wait_state(first,{"cancelled"})).state,"cancelled")
            self.assertEqual((await self.wait_state(second,{"completed","interrupted"})).state,"completed")


if __name__=="__main__":
    unittest.main()
