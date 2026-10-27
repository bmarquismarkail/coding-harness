"""Opt-in integration test using real llama-server models and independent checks.

Run from the repository root:
  RUN_LIVE_MODEL_TESTS=1 /home/user/python-venv/langchain-suite/bin/python \
      -m unittest discover -s tests -p test_live_models.py -v

Each invocation retains its fixture, model replies, timings, and verdict under
development-evidence/. Ordinary test discovery skips live inference.
"""

from __future__ import annotations

import ast
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx

from coding_agent.config import load_config
from coding_agent.graph import run_agent
from coding_agent.model import OpenAICompatibleModel


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = Path(__file__).parent / "fixtures" / "latency"
TASK = """Repair mean_latency in latency.py for the request-monitoring application.
Follow README.md. It must support lists and single-use generators of numbers,
use constant auxiliary memory and consume the input once, return the arithmetic
mean (including floats, negatives and singleton inputs), and raise ValueError
for empty lists AND exhausted generators. Keep mean_latency(samples).

Deliver a concrete fix in the FINAL synthesis: files_to_change MUST be a JSON
array with exactly one object, {"path": "latency.py", "content": "<complete replacement Python source>"}.
The content must contain only def mean_latency(samples) and its body, without
imports, annotations, decorators or top-level executable statements. Use basic
Python arithmetic and iteration, with ValueError as needed. Include a brief
diagnosis in summary. This is a proposed replacement; do not claim tests ran.
"""

# Host-authored validation. The generated function has access only to the
# named builtins below, after an AST check, in a time-limited child process.
VALIDATOR = r'''
import json
import math
import resource
import sys
resource.setrlimit(resource.RLIMIT_CPU, (3, 3))
resource.setrlimit(resource.RLIMIT_AS, (256 * 1024 * 1024, 256 * 1024 * 1024))
resource.setrlimit(resource.RLIMIT_FSIZE, (1024 * 1024, 1024 * 1024))
source = sys.stdin.read()
allowed = {"ValueError": ValueError, "sum": sum, "len": len, "float": float,
           "list": list, "tuple": tuple, "iter": iter, "next": next,
           "enumerate": enumerate}
namespace = {"__builtins__": allowed}
exec(compile(source, "latency.py", "exec"), namespace)
function = namespace["mean_latency"]
cases = [
    ("list", lambda: [10, 20, 30], 20.0),
    ("generator", lambda: (x for x in [5, 15, 40]), 20.0),
    ("float_generator", lambda: (x for x in [0.25, 0.75]), 0.5),
    ("negative_generator", lambda: (x for x in [-10, 5, 20]), 5.0),
    ("singleton", lambda: iter([17]), 17.0),
    ("zeros", lambda: (x for x in [0, 0, 0]), 0.0),
    ("large_generator", lambda: (x for x in range(100000)), 49999.5),
    ("empty_list", lambda: [], None),
    ("empty_generator", lambda: (x for x in []), None),
]
results = []
for name, factory, expected in cases:
    try:
        actual = function(factory())
        passed = expected is not None and math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-10)
        results.append({"case": name, "passed": passed, "actual": actual})
    except Exception as exc:
        results.append({"case": name, "passed": expected is None and isinstance(exc, ValueError),
                        "error": type(exc).__name__})
print(json.dumps(results))
'''


def validate_source(source: str) -> list[dict]:
    """Check a single pure function without granting model code host capabilities."""
    if len(source) > 16000:
        raise ValueError("replacement is unexpectedly large")
    tree = ast.parse(source)
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
        raise ValueError("replacement must contain exactly one function")
    function = tree.body[0]
    if (function.name != "mean_latency" or function.decorator_list or function.returns
            or function.args.defaults or function.args.kw_defaults
            or function.args.vararg or function.args.kwarg or function.args.kwonlyargs
            or function.args.posonlyargs or len(function.args.args) != 1
            or function.args.args[0].arg != "samples" or function.args.args[0].annotation
            or getattr(function, "type_params", [])):
        raise ValueError("replacement must preserve def mean_latency(samples)")
    allowed_calls = {"ValueError", "sum", "len", "float", "list", "tuple", "iter", "next", "enumerate"}
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom, ast.Attribute, ast.Global, ast.Nonlocal, ast.ClassDef)):
            raise ValueError(f"unsupported construct: {type(node).__name__}")
        if isinstance(node, ast.Name) and node.id.startswith("__"):
            raise ValueError("dunder names are not supported")
        if isinstance(node, ast.Call) and (not isinstance(node.func, ast.Name) or node.func.id not in allowed_calls):
            raise ValueError("only arithmetic helper builtins may be called")
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in {"list", "tuple"}:
            raise ValueError("materializing the iterable violates the constant-memory requirement")
        if isinstance(node, (ast.List, ast.ListComp, ast.Set, ast.SetComp, ast.Dict, ast.DictComp)):
            raise ValueError("container allocation is outside this fixture's constant-memory contract")
    completed = subprocess.run(
        [sys.executable, "-I", "-c", VALIDATOR], input=source,
        capture_output=True, text=True, timeout=10, check=True,
        env={"PATH": "/usr/bin:/bin"},
    )
    return json.loads(completed.stdout)


class RecordedProvider:
    """Record responses and counters, excluding credentials and HTTP headers."""

    def __init__(self, config, evidence: Path):
        self.config = config
        self.evidence = evidence
        self.calls: list[dict] = []
        self.lock = threading.Lock()
        self.active = 0
        self.peak = 0

    def invoke(self, messages, *, role: str) -> str:
        settings = self.config.main if role == "main" else self.config.subagents
        started = time.monotonic()
        with self.lock:
            number = len(self.calls) + 1
            record = {"number": number, "role": role, "model": settings.model,
                      "base_url": settings.base_url, "state": "started",
                      "max_tokens": settings.max_tokens, "reasoning_effort": settings.reasoning_effort,
                      "enable_thinking": settings.enable_thinking}
            self.calls.append(record)
            if role == "subagent":
                self.active += 1
                self.peak = max(self.peak, self.active)
        print(f"live call {number}: {role} started", flush=True)

        def response_hook(response):
            response.read()
            record["http_status"] = response.status_code
            body = response.json()
            record["response"] = body
            record["served_model"] = body.get("model")
            choices = body.get("choices", [])
            if choices:
                record["finish_reason"] = choices[0].get("finish_reason")

        try:
            with httpx.Client(timeout=settings.timeout_seconds, event_hooks={"response": [response_hook]}) as client:
                content = OpenAICompatibleModel(settings, json_mode=self.config.json_mode, client=client).invoke(messages)
            parsed = json.loads(content)
            if not isinstance(parsed, dict) or not parsed:
                raise ValueError("model did not return a nonempty JSON object")
            if record.get("finish_reason") != "stop":
                raise ValueError(f"model did not finish normally: {record.get('finish_reason')}")
            record["state"] = "completed"
            return content
        except Exception as exc:
            record["state"] = "failed"
            record["error"] = type(exc).__name__
            raise
        finally:
            record["elapsed_seconds"] = round(time.monotonic() - started, 3)
            with self.lock:
                if role == "subagent":
                    self.active -= 1
                (self.evidence / f"call-{number:02d}.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
            print(f"live call {number}: {record['state']} ({record['elapsed_seconds']}s)", flush=True)


@unittest.skipUnless(os.environ.get("RUN_LIVE_MODEL_TESTS") == "1", "set RUN_LIVE_MODEL_TESTS=1 for live inference")
class LiveModelTest(unittest.TestCase):
    def test_models_and_generator_bug_repair(self) -> None:
        config = load_config(ROOT / "agent-config.json")
        for settings in (config.main, config.subagents):
            self.assertTrue(settings.api_key_env and os.environ.get(settings.api_key_env), "configured API key is missing")
        evidence_root = ROOT / "development-evidence"
        evidence_root.mkdir(exist_ok=True)
        evidence = Path(tempfile.mkdtemp(prefix="live-model-smoke-", dir=evidence_root))
        print(f"Evidence: {evidence}", flush=True)
        workspace = evidence / "workspace"
        shutil.copytree(FIXTURE, workspace)
        report = {"status": "running", "main_calls": 0, "subagent_calls": 0,
                  "task": TASK, "peak_subagent_requests": 0}
        provider = RecordedProvider(config, evidence)
        graph_call_start = None
        try:
            source = (workspace / "latency.py").read_text(encoding="utf-8")
            source = "def mean_latency" + source.split("def mean_latency", 1)[1]
            report["baseline"] = validate_source(source)
            self.assertTrue(any(not case["passed"] for case in report["baseline"]), "fixture must reproduce the bug")

            handshake = [
                {"role": "system", "content": "Reply with a JSON object only."},
                {"role": "user", "content": 'Compute 17 + 25. Return the result in a JSON object with the key "answer".'},
            ]
            with ThreadPoolExecutor(max_workers=2) as pool:
                replies = list(pool.map(lambda role: provider.invoke(handshake, role=role), ["main", "subagent"]))
            for reply in replies:
                self.assertEqual(json.loads(reply).get("answer"), 42)
            report["communication"] = "passed"
            graph_call_start = len(provider.calls)
            result = run_agent(TASK, str(workspace), config, provider=provider)
            (evidence / "graph-result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
            graph_calls = provider.calls[graph_call_start:]
            report["main_calls"] = sum(call["role"] == "main" for call in graph_calls)
            report["subagent_calls"] = sum(call["role"] == "subagent" for call in graph_calls)
            self.assertEqual(report["main_calls"], 2)
            self.assertEqual(report["subagent_calls"], config.parallelism)
            self.assertEqual(len(result["subagent_results"]), config.parallelism)
            if config.parallelism > 1:
                self.assertGreater(provider.peak, 1, "specialist requests must overlap")
            changes = result["synthesis"].get("files_to_change")
            self.assertIsInstance(changes, list, "synthesis must contain replacement source")
            self.assertEqual(len(changes), 1)
            self.assertEqual(changes[0].get("path"), "latency.py")
            replacement = changes[0].get("content")
            self.assertIsInstance(replacement, str)
            report["candidate"] = validate_source(replacement)
            (workspace / "latency.py").write_text(replacement, encoding="utf-8")
            self.assertTrue(all(case["passed"] for case in report["candidate"]), report["candidate"])
            report["status"] = "passed"
        except Exception as exc:
            report["status"] = "failed"
            report["error_type"] = type(exc).__name__
            raise
        finally:
            if graph_call_start is not None:
                graph_calls = provider.calls[graph_call_start:]
                report["main_calls"] = sum(call["role"] == "main" for call in graph_calls)
                report["subagent_calls"] = sum(call["role"] == "subagent" for call in graph_calls)
            report["peak_subagent_requests"] = provider.peak
            report["completed_calls"] = sum(call["state"] == "completed" for call in provider.calls)
            (evidence / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
            print(f"Live test {report['status']}; evidence: {evidence}", flush=True)


if __name__ == "__main__":
    unittest.main()
