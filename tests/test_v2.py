from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
import socket
import unittest
from pathlib import Path
from unittest.mock import patch

from coding_agent import CheckSpec, Harness, TaskRequest
from coding_agent.config import AgentConfig, ConfigError
from coding_agent.checks import Checks
from coding_agent.engine import Engine
from coding_agent.model import ModelError
from coding_agent.replacement import select_replacement
from coding_agent.sandbox import SandboxRunner, SandboxError
from coding_agent.store import Store
from coding_agent.tools import Change, PatchArgs, ReadArgs, EvidenceArgs, ToolBroker, WorkerTask
from coding_agent.types import Profile, ReplacementSpec, parse_action
from coding_agent.workspace import Workspace, digest

EMPTY_REVIEW_JSON = '{"findings":[],"observations":[],"challenges":[],"semantic_concerns":[]}'

BAD = 'def mean_latency(samples):\n    if not samples:\n        raise ValueError("empty")\n    return sum(samples) / len(samples)\n'
GOOD = 'def mean_latency(samples):\n    total = 0\n    count = 0\n    for value in samples:\n        total += value\n        count += 1\n    if count == 0:\n        raise ValueError("empty")\n    return total / count\n'
EXISTING = 'import unittest\nfrom latency import mean_latency\nclass Existing(unittest.TestCase):\n    def test_list(self):\n        self.assertEqual(mean_latency([10,20,30]),20)\n'
ACCEPTANCE = '''import unittest
from latency import mean_latency
class Acceptance(unittest.TestCase):
    def test_generator(self):
        self.assertEqual(mean_latency(x for x in [10,20,30]),20)
    def test_empty(self):
        with self.assertRaises(ValueError):
            mean_latency(x for x in [])
'''


def fixture(root: Path):
    root.mkdir()
    (root / "latency.py").write_text(BAD)
    (root / "tests").mkdir()
    (root / "tests/test_existing.py").write_text(EXISTING)
    (root / "tests/test_acceptance.py").write_text(ACCEPTANCE)


def request(root: Path, **kwargs):
    return TaskRequest(root=str(root), task="Repair mean_latency for generators and empty iterables.",
                       checks=[CheckSpec(id="acceptance", argv=[sys.executable, "-m", "unittest", "discover", "-s", "tests", "-p", "test_acceptance.py"])], **kwargs)


def cited_source(messages):
    for message in reversed(messages):
        if message.get("role") != "user":
            continue
        try:
            evidence_id = json.loads(message["content"]).get("result", {}).get("evidence_id")
            if evidence_id:
                return [evidence_id]
        except (ValueError, TypeError, AttributeError):
            pass
    return []


class RepairProvider:
    def __init__(self, content=GOOD):
        self.main_calls = 0
        self.worker_calls = 0
        self.content = content

    def invoke(self, messages, *, role):
        if role == "reviewer":
            return EMPTY_REVIEW_JSON
        if role == "subagent":
            self.worker_calls += 1
            return json.dumps({"type": "finish", "summary": "Reviewed scoped candidate", "findings": [], "gaps": []})
        self.main_calls += 1
        if self.main_calls == 1:
            return json.dumps({"type": "tool", "tool": "read_file", "args": {"path": "latency.py"}})
        if self.main_calls == 2:
            return json.dumps({"type": "tool", "tool": "apply_patch", "args": {"changes": [
                {"path": "latency.py", "expected_hash": hashlib.sha256(BAD.encode()).hexdigest(), "content": self.content}]}})
        return json.dumps({"type": "finish", "summary": "Fixed generator averaging", "diagnosis": "The implementation used len() and repeated iteration, which fails for one-shot iterators.", "diagnosis_sources": cited_source(messages), "findings": [], "gaps": []})


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.source = self.root / "source"
        fixture(self.source)
        self.harness = Harness(self.root / "state", autostart=False)

    def tearDown(self):
        self.temp.cleanup()

    def test_real_sandbox_python_repair_is_verified_and_original_unchanged(self):
        run_id = self.harness.submit(request(self.source))
        provider = RepairProvider()
        Engine(self.harness.store, run_id, provider).execute()
        status = self.harness.status(run_id)
        self.assertEqual(status.state, "completed", status.error)
        result = self.harness.result(run_id)
        self.assertEqual(result.outcome, "verified", result.gaps)
        self.assertIn("+    count = 0", result.patch)
        self.assertEqual((self.source / "latency.py").read_text(), BAD)
        self.assertEqual(provider.worker_calls, 1)
        self.assertTrue(any(c["phase"] == "baseline" and not c["passed"] for c in result.checks))
        self.assertTrue(all(c["passed"] for c in result.checks if c["phase"] == "candidate"))
        self.assertEqual(result.evidence.patch_artifact, result.patch_artifact)
        self.assertTrue(result.evidence.reproducer_artifacts)
        self.assertTrue(result.evidence.diagnosis_sources)
        self.assertEqual(result.evidence.patch_source_hash,
                         next(c["source_hash"] for c in result.checks if c.get("phase") == "candidate" and c.get("check_id") == "acceptance"))

    def test_repair_recovers_from_empty_finish_and_rejected_edits(self):
        self.harness.profiles["post-edit"] = {"check_after_edit": True}
        run_id = self.harness.submit(request(self.source, profile="post-edit"))
        test = self

        class RecoveringProvider(RepairProvider):
            def invoke(self, messages, *, role):
                if role != "main":
                    return super().invoke(messages, role=role)
                self.main_calls += 1
                step = self.main_calls
                if step == 1:
                    return json.dumps({"type": "finish", "summary": "Premature repair claim"})
                if step == 2:
                    test.assertIn("No edit landed", messages[-1]["content"])
                    return json.dumps({"type": "tool", "tool": "read_file", "args": {"path": "latency.py"}})
                if step == 3:
                    self.citation = cited_source(messages)
                    self.edit = {"path": "latency.py", "expected_hash": "0" * 64, "content": GOOD}
                    return json.dumps({"type": "tool", "tool": "apply_patch", "args": self.edit})
                if step == 4:
                    test.assertIn("Use args.changes", messages[-1]["content"])
                    test.assertEqual(json.loads(messages[-1]["content"])["error_code"], "invalid_arguments")
                    test.assertEqual((engine.workspace.root / "latency.py").read_text(), BAD)
                    return json.dumps({"type": "tool", "tool": "apply_patch", "args": {"changes": [self.edit]}})
                if step == 5:
                    test.assertIn("Reread the target", messages[-1]["content"])
                    test.assertEqual(json.loads(messages[-1]["content"])["error_code"], "stale_hash")
                    test.assertEqual((engine.workspace.root / "latency.py").read_text(), BAD)
                    return json.dumps({"type": "tool", "tool": "read_file", "args": {"path": "latency.py"}})
                if step == 6:
                    self.edit["expected_hash"] = json.loads(messages[-1]["content"])["edit_reference"]["expected_hash"]
                    return json.dumps({"type": "tool", "tool": "apply_patch", "args": {"changes": [self.edit]}})
                observation = json.loads(messages[-1]["content"])
                checks = observation["candidate_checks"]
                test.assertTrue(checks)
                test.assertTrue(all(check["passed"] for check in checks))
                return json.dumps({"type": "finish", "summary": "Fixed generator averaging",
                                   "diagnosis": "Generators have no length.",
                                   "diagnosis_sources": observation["finish_template"]["diagnosis_sources"]})

        provider = RecoveringProvider()
        engine = Engine(self.harness.store, run_id, provider)
        engine.execute()
        result = self.harness.result(run_id)
        self.assertEqual(result.outcome, "verified", result.gaps)
        self.assertTrue(result.patch)
        self.assertGreater(provider.worker_calls, 0)

    def test_recovery_constrains_creation_of_unread_required_file(self):
        self.assert_required_creation_recovery({"type": "finish", "summary": "Premature finish"})

    def test_blocked_candidate_check_immediately_constrains_creation(self):
        self.assert_required_creation_recovery({
            "type": "tool", "tool": "run_check", "args": {"check_id": "acceptance", "phase": "candidate"},
        })

    def test_wrong_path_edit_immediately_constrains_creation(self):
        self.assert_required_creation_recovery({
            "type": "tool", "tool": "apply_patch", "args": {"changes": [
                {"path": "other.py", "expected_hash": None, "content": "pass"},
            ]},
        })

    def assert_required_creation_recovery(self, first_action):
        source = self.root / "empty-source"
        source.mkdir()
        (source / "README.md").write_text("Fixture")
        target = "required.py"
        request = TaskRequest(root=str(source), task="Create the required source.",
                              required_patch_paths=[target], checks=[
            CheckSpec(id="acceptance", argv=[sys.executable, "-c",
                     "from pathlib import Path; assert Path('required.py').is_file()"]),
            CheckSpec(id="regression", kind="regression", argv=[sys.executable, "-c", "pass"]),
        ])
        run_id = self.harness.submit(request)
        engine = Engine(self.harness.store, run_id)
        calls = 0

        def invoke(messages, *, role, schema=None):
            nonlocal calls
            if role == "reviewer":
                return EMPTY_REVIEW_JSON
            if role == "subagent":
                return '{"type":"finish","summary":"Reviewed candidate"}'
            calls += 1
            if calls == 1:
                return json.dumps(first_action)
            if 2 <= calls <= 5:
                if calls <= 3:
                    patch_schema, inspection_schema = schema["anyOf"]
                    self.assertEqual(inspection_schema["properties"]["tool"]["enum"], ["read_file", "inspect_diff"])
                else:
                    patch_schema = schema
                props = patch_schema["properties"]
                self.assertEqual(props["tool"]["const"], "apply_patch")
                change = props["args"]["properties"]["changes"]["items"]["anyOf"][0]["properties"]
                self.assertEqual(change["path"]["const"], target)
                self.assertIsNone(change["expected_hash"]["const"])
                self.assertEqual(change["content"]["type"], "string")
                if calls == 2:
                    self.assertIn(target, messages[-1]["content"])
                    return '{"type":"tool","tool":"read_file","args":{"path":"README.md"}}'
                if calls == 3:
                    self.assertEqual(json.loads(messages[-1]["content"])["remaining_inspections"], 1)
                    return '{"type":"tool","tool":"inspect_diff","args":{}}'
                if calls == 4:
                    self.assertEqual(json.loads(messages[-1]["content"])["remaining_inspections"], 0)
                    return '{"type":"tool","tool":"inspect_diff","args":{}}'
                self.assertIn("Inspection limit reached", messages[-1]["content"])
                return json.dumps({"type": "tool", "tool": "apply_patch", "args": {"changes": [
                    {"path": target, "expected_hash": None, "content": "pass\n"}
                ]}})
            if calls == 6:
                return json.dumps({"type": "tool", "tool": "read_file", "args": {"path": target}})
            return json.dumps({"type": "finish", "summary": "Created required source",
                               "diagnosis": "The required module was absent.", "diagnosis_sources": cited_source(messages)})

        with patch.object(engine.provider, "invoke", side_effect=invoke):
            engine.execute()
        result = self.harness.result(run_id)
        self.assertEqual(result.outcome, "verified", result.gaps)
        self.assertIn("new file mode", result.patch)
        self.assertEqual(calls, 7)

    def replacement_engine(self):
        verifier = self.root / "verify.py"
        verifier.write_text("# Frozen test verifier\n")
        selector = self.source / "selector.s"
        selector.write_text("original\n")
        target = "replacement.c"
        expected = hashlib.sha256(b"selected\n").hexdigest()
        argv = [sys.executable, str(verifier)]
        spec = ReplacementSpec(target_id="test-target", source_path=target, selector_path=selector.name,
                               original_selector_sha256=digest(selector), selected_selector_sha256=expected,
                               selector_argv=[*argv, "select"], verifier_path=str(verifier),
                               verifier_sha256=digest(verifier), compile_check_id="compile")
        req = TaskRequest(root=str(self.source), task="Replace the target", replacement=spec,
                          required_patch_paths=[target, selector.name], readonly_dependencies=[str(verifier)],
                          checks=[CheckSpec(id="compile", argv=[*argv, "compile"], kind="build"),
                                  CheckSpec(id="matching", argv=[*argv, "match"], requires=["compile"])])
        run_id = self.harness.submit(req)
        engine = Engine(self.harness.store, run_id, RepairProvider())
        engine.workspace.prepare(self.source)
        engine.checks.initialize(engine.request)
        (engine.workspace.root / target).write_text("PLACEHOLDER")
        return engine

    def test_automatic_selection_fails_closed(self):
        for failure in ("stale", "source", "third_file", "verifier"):
            with self.subTest(failure=failure):
                engine = self.replacement_engine()
                root = engine.workspace.root
                if failure == "stale":
                    (root / "selector.s").write_text("stale\n")
                if failure == "verifier":
                    Path(engine.request.replacement.verifier_path).write_text("# Drifted verifier\n")
                before = digest(root / "selector.s")

                def transform(_runner, stage, _argv):
                    (stage / "selector.s").write_text("selected\n")
                    if failure == "source":
                        (stage / "replacement.c").write_text("tampered")
                    if failure == "third_file":
                        (stage / "unrelated.txt").write_text("tampered")
                    return {"exit_code": 0, "timed_out": False, "output_truncated": False, "output": ""}

                with patch.object(SandboxRunner, "run", autospec=True, side_effect=transform) as runner:
                    result = select_replacement(engine.checks)
                self.assertIn("error", result)
                self.assertEqual(runner.call_count, int(failure in {"source", "third_file"}))
                self.assertEqual(digest(root / "selector.s"), before)
                self.assertEqual((root / "replacement.c").read_text(), "PLACEHOLDER")
                self.assertFalse((root / "unrelated.txt").exists())
                self.assertIn("selector.s", engine.checks.missing_patch_paths())

    def test_automatic_selection_cancelled_before_publication(self):
        engine = self.replacement_engine()

        def transform(_runner, stage, _argv):
            (stage / "selector.s").write_text("selected\n")
            return {"exit_code": 0, "timed_out": False, "output_truncated": False, "output": ""}

        engine.checks.runner.cancelled = lambda: True
        with patch.object(SandboxRunner, "run", autospec=True, side_effect=transform):
            with self.assertRaises(InterruptedError):
                select_replacement(engine.checks)
        self.assertEqual((engine.workspace.root / "selector.s").read_text(), "original\n")
        receipt = engine.store.receipt(engine.run_id, "replacement:test-target")
        self.assertEqual(receipt["state"], "interrupted")
        self.assertIn("selector.s", engine.checks.missing_patch_paths())

    def test_automatic_selection_resumes_after_publication(self):
        engine = self.replacement_engine()
        store = engine.store
        save = store.save_receipt

        def transform(_runner, stage, _argv):
            (stage / "selector.s").write_text("selected\n")
            return {"exit_code": 0, "timed_out": False, "output_truncated": False, "output": ""}

        def interrupt(run_id, key, kind, state, data):
            if state == "completed":
                raise InterruptedError("after publication")
            return save(run_id, key, kind, state, data)

        with patch.object(SandboxRunner, "run", autospec=True, side_effect=transform) as runner:
            with patch.object(store, "save_receipt", side_effect=interrupt):
                with self.assertRaises(InterruptedError):
                    select_replacement(engine.checks)
            self.assertEqual(store.receipt(engine.run_id, "replacement:test-target")["state"], "prepared")
            first = select_replacement(engine.checks)
            (engine.workspace.root / "replacement.c").write_text("corrected")
            self.assertEqual(select_replacement(engine.checks), first)
            self.assertEqual(runner.call_count, 1)
        self.assertEqual(first["c_before"], first["c_after"])
        self.assertEqual(engine.checks.missing_patch_paths(), [])

    def test_required_patch_path_blocks_checks_after_wrong_file_edit(self):
        req = request(self.source, required_patch_paths=["latency.py"])
        run_id = self.harness.submit(req)
        engine = Engine(self.harness.store, run_id, RepairProvider())
        engine.workspace.prepare(self.source)
        engine.checks.initialize(engine.request)
        broker = engine.broker

        broker.invoke("apply_patch", {"changes": [
            {"path": "other.py", "expected_hash": None, "content": "pass\n"},
        ]}, key="wrong-edit")
        self.assertEqual(engine.checks.missing_patch_paths(), ["latency.py"])
        with self.assertRaisesRegex(ValueError, "Candidate checks require a patch containing: latency.py"):
            broker.invoke("run_check", {"check_id": "acceptance", "phase": "candidate"}, key="blocked")
        self.assertIsNone(self.harness.store.receipt(run_id, "blocked"))
        baseline = broker.invoke("run_check", {"check_id": "acceptance", "phase": "baseline"}, key="baseline")["result"]
        self.assertIn("TypeError", baseline["output"])

        read = broker.invoke("read_file", {"path": "latency.py"}, key="read-required")["result"]
        broker.invoke("apply_patch", {"changes": [
            {"path": "latency.py", "expected_hash": read["sha256"], "content": GOOD},
        ]}, key="required-edit")
        self.assertEqual(engine.checks.missing_patch_paths(), [])
        candidate = broker.invoke("run_check", {"check_id": "acceptance", "phase": "candidate"}, key="candidate")["result"]
        self.assertTrue(candidate["passed"], candidate["output"])

    def test_request_check_failure_category_is_retained_with_output(self):
        check = CheckSpec(id="comparison", argv=[sys.executable, "-c", "print('byte mismatch'); raise SystemExit(80)"],
                          failure_categories={80: "content_mismatch"})
        req = request(self.source).model_copy(update={"checks": [check]})
        run_id = self.harness.submit(req)
        engine = Engine(self.harness.store, run_id, RepairProvider())
        engine.workspace.prepare(self.source)
        engine.checks.initialize(engine.request)

        result = engine.checks.execute("comparison", "baseline", key="comparison")
        self.assertFalse(result["passed"])
        self.assertEqual(result["failure_category"], "content_mismatch")
        artifact = json.loads(self.harness.artifact(run_id, result["artifact_id"])["content"])
        self.assertEqual(artifact["failure_category"], "content_mismatch")
        self.assertIn("byte mismatch", artifact["output"])

    def test_review_command_cannot_invalidate_source_evidence(self):
        run_id = self.harness.submit(request(self.source, mode="review"))
        engine = Engine(self.harness.store, run_id, RepairProvider())
        engine.workspace.prepare(self.source)
        engine.checks.initialize(engine.request)
        broker = engine.broker
        read = broker.invoke("read_file", {"path": "latency.py"}, key="read")["result"]
        attempt = broker.invoke("run_command", {
            "argv": [sys.executable, "-c", "from pathlib import Path; Path('latency.py').write_text('changed')"],
        }, key="write")["result"]
        self.assertNotEqual(attempt["exit_code"], 0)
        self.assertEqual(attempt["source_hash"], attempt["after_source_hash"])
        self.assertEqual((engine.workspace.root / "latency.py").read_text(), BAD)
        evidence = broker.invoke("read_evidence", {"artifact_id": read["evidence_id"]}, key="cite")["result"]
        self.assertEqual(evidence["sha256"], read["sha256"])
        self.assertEqual(evidence["evidence_id"], read["evidence_id"])
        check = broker.invoke("run_check", {"check_id": "acceptance", "phase": "baseline"}, key="check")["result"]
        self.assertFalse(check["passed"])
        self.assertIn("TypeError", check["output"])
        self.assertFalse(check["source_modified"])

    def test_invalid_independent_review_blocks_verified_repair(self):
        class SchemaEchoProvider(RepairProvider):
            def invoke(self, messages, *, role):
                if role == "reviewer":
                    return '{"$defs":{"EvidenceChallenge":{"type":"object"}}}'
                return super().invoke(messages, role=role)

        run_id = self.harness.submit(request(self.source))
        Engine(self.harness.store, run_id, SchemaEchoProvider()).execute()
        self.assertEqual(self.harness.status(run_id).state, "completed")
        result = self.harness.result(run_id)
        self.assertTrue(result.evidence.verification_checks)
        self.assertTrue(all(check["passed"] for check in result.evidence.verification_checks))
        self.assertEqual(result.outcome, "unverified")
        self.assertEqual(result.review.completion, "partial")
        self.assertIn("protocol_failure", result.review.partial_reasons)
        self.assertTrue(any(gap.startswith("Invalid reviewer response:") for gap in result.gaps))

    def test_unscanned_inventory_does_not_block_verified_repair(self):
        unrelated = [f"filler_{number:03}.cpp" for number in range(500)]
        for path in unrelated:
            (self.source / path).write_text("int filler = 0;\n")
        run_id = self.harness.submit(request(self.source))
        Engine(self.harness.store, run_id, RepairProvider()).execute()
        result = self.harness.result(run_id)
        self.assertIsNotNone(result, self.harness.status(run_id).error)
        self.assertEqual(result.outcome, "verified", result.gaps)
        self.assertNotIn("scope_limit", result.review.partial_reasons)
        coverage = result.review.coverage
        self.assertEqual(coverage.intended_paths, ["latency.py"])
        self.assertEqual(coverage.omitted_paths, [])
        self.assertEqual(coverage.unscanned_paths, [
            *unrelated[199:], "tests/test_acceptance.py", "tests/test_existing.py"])
        self.assertIsNone(coverage.limit_reached)
        self.assertEqual(coverage.scan_limit_reached, "paths")
        self.assertTrue(coverage.unenumerated_remainder)

    def test_investigative_draft_does_not_block_verified_repair(self):
        """Keep a working draft available without treating it as a final finding."""
        draft = {"id": "iterator", "title": "Iterator bug", "claim": "Generators have no len()",
                 "path": "latency.py"}

        class DraftProvider(RepairProvider):
            def invoke(self, messages, *, role):
                """File one draft before the writer proceeds with the repair."""
                if role == "main" and self.main_calls == 1 and not hasattr(self, "drafted"):
                    self.drafted = True
                    return json.dumps({"type": "tool", "tool": "report_findings", "args": {"findings": [draft]}})
                return super().invoke(messages, role=role)

        run_id = self.harness.submit(request(self.source))
        Engine(self.harness.store, run_id, DraftProvider()).execute()
        result = self.harness.result(run_id)
        self.assertEqual(result.outcome, "verified", result.gaps)
        self.assertEqual(result.review.completion, "partial")
        self.assertIn("unexamined_scope", result.review.partial_reasons)
        receipts = self.harness.store.receipts(run_id, "draft_findings")
        self.assertEqual(len(receipts), 1)
        artifact = receipts[0]["data"]["artifact_id"]
        raw = (self.harness.store.run_dir(run_id) / "artifacts" / artifact).read_text()
        self.assertEqual(json.loads(raw)["findings"][0]["id"], draft["id"])

    def test_final_specialist_finding_blocks_verification(self):
        """Require resolution of concrete findings raised by the final specialist."""
        finding = {"id": "regression", "title": "Candidate defect", "claim": "Needs resolution",
                   "path": "latency.py"}

        class FindingProvider(RepairProvider):
            def invoke(self, messages, *, role):
                """Return a concrete finding from the specialist review."""
                if role == "subagent":
                    return json.dumps({"type": "finish", "summary": "Reviewed candidate",
                                       "findings": [finding], "review": {"observations": ["Checks ran"]}})
                return super().invoke(messages, role=role)

        run_id = self.harness.submit(request(self.source))
        Engine(self.harness.store, run_id, FindingProvider()).execute()
        result = self.harness.result(run_id)
        self.assertEqual(result.outcome, "unverified")
        self.assertIn("Concrete review findings require resolution.", result.gaps)
        self.assertEqual([item.proposal.id for item in result.review.findings], [finding["id"]])

    def test_wrong_patch_cannot_claim_verification(self):
        run_id = self.harness.submit(request(self.source))
        Engine(self.harness.store, run_id, RepairProvider('def mean_latency(samples):\n    return 20\n')).execute()
        result = self.harness.result(run_id)
        self.assertIsNotNone(result, self.harness.status(run_id).error)
        self.assertEqual(result.outcome, "unverified")

    def test_action_validation_and_worker_permissions(self):
        with self.assertRaises(ValueError):
            parse_action('{"type":"tool","tool":"read_file","args":{},"execute":"anything"}')
        run_id = self.harness.submit(request(self.source))
        engine = Engine(self.harness.store, run_id, RepairProvider())
        engine.workspace.prepare(self.source)
        engine.checks.initialize(engine.request)
        worker = ToolBroker(engine.store, run_id, engine.workspace, engine.checks, scope=["latency.py"])
        for tool in ("apply_patch", "run_command", "delegate"):
            with self.assertRaises(PermissionError):
                worker.invoke(tool, {}, key="bad")
        with self.assertRaises(PermissionError):
            worker.read_file(ReadArgs(path="tests/test_existing.py"))
        with self.assertRaises(ValueError):
            worker.read_file(ReadArgs(path="../source/latency.py"))
        engine.pool.close()

    def test_stale_evidence_and_interrupted_patch_reconciliation(self):
        run_id = self.harness.submit(request(self.source))
        engine = Engine(self.harness.store, run_id, RepairProvider())
        engine.workspace.prepare(self.source)
        engine.checks.initialize(engine.request)
        read = engine.broker.read_file(ReadArgs(path="latency.py"))
        args = PatchArgs(changes=[Change(path="latency.py", expected_hash=read["sha256"], content=GOOD),
                                 Change(path="new.py", expected_hash=None, content="answer = 42\n")])
        engine.store.save_receipt(run_id, "edit-test", "edit", "started",
                                  {"action_hash": hashlib.sha256(args.model_dump_json().encode()).hexdigest()})
        (engine.workspace.root / "latency.py").write_text(GOOD)
        result = engine.broker.apply_patch(args, key="edit-test")
        self.assertEqual((engine.workspace.root / "new.py").read_text(), "answer = 42\n")
        self.assertEqual(engine.broker.read_evidence(EvidenceArgs(artifact_id=read["evidence_id"]))["validated_against"], "baseline")
        self.assertEqual(engine.broker.apply_patch(args, key="edit-test")["files"], result["files"])
        candidate_read = engine.broker.read_file(ReadArgs(path="latency.py"))
        (engine.workspace.root / "latency.py").write_text("def mean_latency(samples):\n    return 0\n")
        with self.assertRaisesRegex(ValueError, "stale"):
            engine.broker.read_evidence(EvidenceArgs(artifact_id=candidate_read["evidence_id"]))
        engine.pool.close()

    def test_bubblewrap_cannot_see_host_credentials_or_write_source(self):
        runner = SandboxRunner(Profile(name="offline"))
        with socket.socket() as listener:
            listener.bind(("127.0.0.1",0)); listener.listen()
            script = ('import os,pathlib,socket; '
                      'assert "LLAMA_API_KEY" not in os.environ; '
                      f'assert not pathlib.Path({str(self.source)!r}).exists(); '
                      f's=socket.socket(); assert s.connect_ex(("127.0.0.1",{listener.getsockname()[1]})) != 0; print("isolated")')
            with patch.dict(os.environ, {"LLAMA_API_KEY": "test-host-secret"}):
                result = runner.run(self.source, [sys.executable, "-c", script])
        self.assertEqual(result["exit_code"], 0, result["output"])
        self.assertIn("isolated", result["output"])

    def test_declared_dependencies_are_read_only_and_network_requires_profile(self):
        dependency=self.root/"dependency";dependency.mkdir();(dependency/"data").write_text("fixture")
        runner=SandboxRunner(Profile(name="offline"),[str(dependency)])
        script=f'''from pathlib import Path
p=Path({str(dependency/'data')!r})
assert p.read_text()=="fixture"
try: p.write_text("overwrite")
except OSError: pass
else: raise AssertionError("dependency was writable")
'''
        result=runner.run(self.source,[sys.executable,"-c",script])
        self.assertEqual(result["exit_code"],0,result["output"])
        self.assertEqual((dependency/"data").read_text(),"fixture")
        with socket.socket() as listener:
            listener.bind(("127.0.0.1",0));listener.listen()
            result=SandboxRunner(Profile(name="connected",allow_network=True)).run(self.source,[sys.executable,"-c",
                f'import socket;s=socket.socket();assert s.connect_ex(("127.0.0.1",{listener.getsockname()[1]}))==0'])
            self.assertEqual(result["exit_code"],0,result["output"])

    def test_sandbox_start_failure_has_no_host_fallback(self):
        runner = SandboxRunner(Profile(name="offline"))
        with patch("coding_agent.sandbox.subprocess.Popen", side_effect=FileNotFoundError("missing bwrap")):
            with self.assertRaises(SandboxError):
                runner.run(self.source, ["true"])

    def test_git_dirty_baseline_patch_only_contains_agent_changes(self):
        def git(*args):
            return subprocess.run(["git", "-C", str(self.source), *args], capture_output=True, check=True)
        git("init")
        git("add", ".")
        git("-c", "user.name=Test", "-c", "user.email=test@example.invalid", "commit", "-m", "fixture")
        (self.source / "user.txt").write_text("user's untracked work\n")
        (self.source / "latency.py").write_text(BAD+"\n# user change\n")
        status_before = git("status", "--porcelain").stdout
        folder = self.root / "git-job"; folder.mkdir()
        work = Workspace(folder)
        work.prepare(self.source)
        (work.root / "latency.py").write_text(GOOD+"\n# user change\n")
        diff = work.patch()
        self.assertNotIn("user.txt", diff)
        self.assertNotIn("+# user change", diff)
        self.assertEqual(git("status", "--porcelain").stdout, status_before)
        self.assertEqual((self.source/"latency.py").read_text(), BAD+"\n# user change\n")

    def test_git_snapshot_uses_gitignore_for_generated_files(self):
        def git(*args):
            return subprocess.run(["git", "-C", str(self.source), *args], capture_output=True, check=True)
        git("init")
        (self.source / ".gitignore").write_text("graphify-out/\n")
        (self.source / "graphify-out").mkdir()
        (self.source / "graphify-out/graph.json").write_text("x" * (11 * 1024 * 1024))
        git("add", ".gitignore", "latency.py", "tests")
        git("-c", "user.name=Test", "-c", "user.email=test@example.invalid", "commit", "-m", "fixture")
        folder = self.root / "git-ignore-job"; folder.mkdir()
        work = Workspace(folder)
        work.prepare(self.source)
        self.assertFalse((work.root / "graphify-out/graph.json").exists())
        self.assertTrue((work.root / "latency.py").exists())

    def test_review_repeated_missing_reads_get_a_bounded_finish_prompt(self):
        run_id = self.harness.submit(TaskRequest(root=str(self.source), task="Review the latency implementation.", mode="review"))
        class Provider:
            def __init__(self): self.calls = 0
            def invoke(self, messages, *, role):
                if role == "reviewer":
                    return EMPTY_REVIEW_JSON
                self.calls += 1
                if self.calls >= 8:
                    self.assert_boundary(messages)
                    return json.dumps({"type":"finish", "summary":"Review completed with a gap.", "findings":[], "gaps":["No additional evidence established."]})
                return json.dumps({"type":"tool", "tool":"read_file", "args":{"path":"missing-file.cpp"}})
            def assert_boundary(self, messages):
                self_seen = any("Complete the review now" in m.get("content", "") for m in messages)
                assert self_seen
        provider = Provider()
        Engine(self.harness.store, run_id, provider).execute()
        self.assertEqual(self.harness.status(run_id).state, "completed", self.harness.status(run_id).error)
        self.assertEqual(self.harness.result(run_id).outcome, "unverified")

    def test_reviewer_is_configurable_and_must_use_a_distinct_endpoint(self):
        data = {
            "main": {"base_url": "https://main.example/v1", "model": "writer", "api_key_env": "LLAMA_API_KEY"},
            "subagents": {"base_url": "https://subs.example/v1", "model": "worker", "api_key_env": "LLAMA_API_KEY"},
            "reviewer": {"base_url": "https://api.openai.com/v1", "model": "gpt-4.1-mini"},
        }
        config = AgentConfig.from_mapping(data)
        self.assertEqual(config.reviewer.api_key_env, "OPENAI_API_KEY")
        self.assertEqual(config.reviewer.base_url, "https://api.openai.com/v1")
        with self.assertRaises(ConfigError):
            AgentConfig.from_mapping({**data, "reviewer": {"base_url": "https://main.example/v1", "model": "reviewer"}})

    def test_cancel_queued_job_and_pagination(self):
        one = self.harness.submit(request(self.source))
        two = self.harness.submit(request(self.source))
        self.assertTrue(self.harness.list_runs(limit=1)["has_more"])
        self.assertEqual(self.harness.cancel(one).state, "cancelled")
        self.assertEqual(self.harness.status(two).state, "queued")
        with self.assertRaises(ValueError):
            self.harness.events(two, limit=10000)


class CheckpointTests(unittest.TestCase):
    def test_one_failed_worker_of_nine_resumes_only_missing_work(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); source = root/"source"; fixture(source)
            harness = Harness(root/"state", autostart=False)
            run_id = harness.submit(TaskRequest(root=str(source), task="Review scoped files", mode="review"))
            class Provider:
                def __init__(self):
                    self.main = 0; self.calls = {}; self.lock = threading.Lock()
                def invoke(self, messages, *, role):
                    if role == "reviewer":
                        return EMPTY_REVIEW_JSON
                    if role == "main":
                        self.main += 1
                        if self.main == 1:
                            return json.dumps({"type":"tool", "tool":"delegate", "args":{"tasks":[
                                {"id":f"w{i}","task":f"worker-{i}","paths":["latency.py"]} for i in range(9)]}})
                        if self.main == 2:
                            return json.dumps({"type":"tool","tool":"collect","args":{"ids":[f"w{i}" for i in range(9)]}})
                        return json.dumps({"type":"finish","summary":"Review complete","findings":[],"gaps":[]})
                    marker = messages[1]["content"].split("\n",1)[0]
                    with self.lock:
                        self.calls[marker] = self.calls.get(marker,0)+1
                        count = self.calls[marker]
                    if marker == "worker-8" and count == 1:
                        raise ModelError("fixture transient transport failure")
                    return json.dumps({"type":"finish","summary":"Scoped report","findings":[],"gaps":[]})
            provider=Provider()
            Engine(harness.store,run_id,provider).execute()
            self.assertEqual(harness.status(run_id).state,"interrupted")
            harness.resume(run_id)
            Engine(harness.store,run_id,provider).execute()
            self.assertEqual(harness.status(run_id).state,"completed",harness.status(run_id).error)
            self.assertEqual(provider.main,3)
            self.assertEqual(provider.calls["worker-8"],2)
            self.assertTrue(all(provider.calls[f"worker-{i}"]==1 for i in range(8)),provider.calls)


if __name__ == "__main__":
    unittest.main()
