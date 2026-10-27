from __future__ import annotations

import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx

from coding_agent.config import AgentConfig, ConfigError, load_config
from coding_agent.graph import run_agent
from coding_agent.model import ModelError, OpenAICompatibleModel
from coding_agent.zvec import ZvecGrepSettings, ZvecGrepTool


class FakeProvider:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def invoke(self, messages, *, role: str) -> str:
        self.calls.append(role)
        user = messages[-1]["content"]
        if role == "main" and "SPECIALIST REPORTS" not in user:
            return json.dumps({"summary": "plan", "work_items": ["inspect"], "verification": []})
        if role == "subagent":
            marker = user.split("role: ", 1)[-1].split(".", 1)[0]
            return json.dumps({"role": marker, "findings": ["evidence"], "evidence": ["README.md"]})
        return json.dumps({
            "status": "ready",
            "summary": "synthesized",
            "implementation_steps": ["implement", "verify"],
            "verification_commands": [],
        })


class FakeZvec:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def query(self, root: str, query: str) -> str:
        self.calls.append((root, query))
        return "hit: coding_agent/graph.py:1"


class AgentTests(unittest.TestCase):
    def test_config_defaults_keep_two_endpoints_and_nine_specialists(self) -> None:
        config = load_config("/does/not/exist.json", environ={})
        self.assertEqual(config.main.base_url, "https://lead.example.invalid/v1")
        self.assertEqual(config.subagents.base_url, "https://specialist.example.invalid/v1")
        self.assertEqual(config.main.api_key_env, "LLAMA_API_KEY")
        self.assertEqual(config.subagents.api_key_env, "LLAMA_API_KEY")
        self.assertEqual(config.parallelism, 9)
        self.assertEqual(len(config.specialist_roles), 9)

    def test_config_requires_absolute_url(self) -> None:
        with self.assertRaises(ConfigError):
            AgentConfig.from_mapping({
                "main": {"base_url": "lead.example.invalid/v1", "model": "main"},
                "subagents": {"base_url": "https://host/minicoder/v1", "model": "sub"},
            })

    def test_graph_runs_all_nine_specialists_and_synthesizes(self) -> None:
        config = load_config("/does/not/exist.json", environ={})
        fake = FakeProvider()
        fake_zvec = FakeZvec()
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, "README.md").write_text("test repository\n", encoding="utf-8")
            result = run_agent(
                "add a feature",
                directory,
                config,
                provider=fake,
                zvec_query="where is the graph built?",
                zvec_tool=fake_zvec,
            )
        self.assertEqual(len(result["subagent_results"]), 9)
        self.assertEqual(result["synthesis"]["status"], "ready")
        self.assertEqual(fake.calls.count("subagent"), 9)
        self.assertEqual(fake.calls.count("main"), 2)
        self.assertEqual(fake_zvec.calls[0][1], "where is the graph built?")


class ModelTests(unittest.TestCase):
    def test_specialist_reasoning_setting_reaches_the_server(self) -> None:
        config = load_config("/does/not/exist.json", environ={})
        def handler(request):
            self.assertEqual(json.loads(request.content)["reasoning_effort"], "none")
            self.assertEqual(json.loads(request.content)["chat_template_kwargs"], {"enable_thinking": False})
            return httpx.Response(200, json={"choices": [{"message": {"content": "{}"}}]})
        with httpx.Client(transport=httpx.MockTransport(handler)) as client:
            OpenAICompatibleModel(config.subagents, client=client).invoke([{"role": "user", "content": "test"}])

    def test_openai_compatible_adapter_posts_expected_request(self) -> None:
        seen: dict[str, object] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["url"] = str(request.url)
            seen["body"] = json.loads(request.content)
            return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

        config = load_config("/does/not/exist.json", environ={})
        client = httpx.Client(transport=httpx.MockTransport(handler))
        model = OpenAICompatibleModel(config.main, client=client)
        try:
            self.assertEqual(model.invoke([{"role": "user", "content": "hi"}]), "ok")
        finally:
            client.close()
        self.assertEqual(seen["url"], "https://lead.example.invalid/v1/chat/completions")
        self.assertEqual(seen["body"]["model"], "JetBrains/Mellum2-12B-A2.5B-Thinking")
        self.assertEqual(seen["body"]["response_format"], {"type": "json_object"})

    def test_openai_compatible_adapter_reports_http_failure(self) -> None:
        client = httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(503, text="offline")))
        config = load_config("/does/not/exist.json", environ={})
        model = OpenAICompatibleModel(config.main, client=client)
        try:
            with self.assertRaises(ModelError):
                model.invoke([{"role": "user", "content": "hi"}])
        finally:
            client.close()

    def test_output_limit_is_not_accepted_as_a_completed_answer(self) -> None:
        response = {"choices": [{"finish_reason": "length", "message": {"content": '{"summary": "partial"}'}}]}
        config = load_config("/does/not/exist.json", environ={})
        with httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(200, json=response))) as client:
            with self.assertRaisesRegex(ModelError, "exhausted max_tokens"):
                OpenAICompatibleModel(config.main, client=client).invoke([{"role": "user", "content": "test"}])

    def test_openai_compatible_adapter_uses_llama_api_key_environment_variable(self) -> None:
        seen: dict[str, str | None] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["authorization"] = request.headers.get("authorization")
            return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

        config = load_config("/does/not/exist.json", environ={})
        client = httpx.Client(transport=httpx.MockTransport(handler))
        try:
            with patch.dict(os.environ, {"LLAMA_API_KEY": "test-secret"}, clear=False):
                self.assertEqual(model := OpenAICompatibleModel(config.main, client=client).invoke([
                    {"role": "user", "content": "hi"}
                ]), "ok")
        finally:
            client.close()
        self.assertEqual(seen["authorization"], "Bearer test-secret")


class ZvecGrepTests(unittest.TestCase):
    def test_index_and_query_use_argument_vectors_and_refresh_off(self) -> None:
        commands: list[list[str]] = []

        def runner(command, **kwargs):
            commands.append(command)
            return subprocess.CompletedProcess(command, 0, stdout="ok", stderr="")

        tool = ZvecGrepTool(ZvecGrepSettings(mode="direct"), runner=runner)
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(tool.index(directory, embedding="local/potion-code-16m-v2"), "ok")
            self.assertEqual(tool.query(directory, "graph construction", route="hybrid"), "ok")
        self.assertEqual(commands[0][0:3], ["zg", "index", commands[0][2]])
        self.assertIn("--embedding", commands[0])
        self.assertNotIn("--rebuild", commands[0])
        self.assertIn("--refresh", commands[1])
        self.assertIn("off", commands[1])
        self.assertIn("graph construction", commands[1])

    def test_langchain_tool_exposes_explicit_index_query_and_status_actions(self) -> None:
        def runner(command, **kwargs):
            return subprocess.CompletedProcess(command, 0, stdout="result", stderr="")

        tool = ZvecGrepTool(runner=runner).as_langchain_tool()
        self.assertEqual(tool.invoke({"action": "status", "root": tempfile.gettempdir()}), "result")
        self.assertEqual(tool.name, "zvec_grep")


if __name__ == "__main__":
    unittest.main()
