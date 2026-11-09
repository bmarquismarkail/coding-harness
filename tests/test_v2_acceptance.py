"""Executable acceptance cases; models here are deterministic transport fixtures."""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

from coding_agent import CheckSpec, Harness, TaskRequest
from coding_agent.engine import Engine
from coding_agent.tools import Change, CommandArgs, PatchArgs, ReadArgs
from coding_agent.workspace import digest, tree_hash
from test_v2 import BAD, GOOD, RepairProvider, cited_source, fixture, request

CPP_BAD = "inline int clamp_level(int v) { if (v < 0) return 0; if (v > 100) return 99; return v; }\n"
CPP_GOOD = CPP_BAD.replace("return 99", "return 100")


def cpp_fixture(source):
    source.mkdir(); (source/"tests").mkdir()
    (source/"level.hpp").write_text(CPP_BAD)
    (source/"tests/acceptance.cpp").write_text('#include "level.hpp"\nint main(){return clamp_level(101)!=100 || clamp_level(2147483647)!=100;}\n')
    (source/"tests/regression.cpp").write_text('#include "level.hpp"\nint main(){return clamp_level(-1)!=0 || clamp_level(50)!=50 || clamp_level(100)!=100;}\n')
    checks = []
    for name, kind in [("acceptance", "acceptance"), ("regression", "regression")]:
        checks.extend([
            CheckSpec(id=f"build-{name}", kind="build", argv=["g++", "-std=c++17", "-Wall", "-Wextra", "-Werror", "-I.", f"tests/{name}.cpp", "-o", f"build-{name}"]),
            CheckSpec(id=name, kind=kind, argv=[f"./build-{name}"], requires=[f"build-{name}"]),
        ])
    return TaskRequest(root=str(source), task="Repair clamp_level for levels above 100; retain existing behavior otherwise.", checks=checks)


class AcceptanceTests(unittest.TestCase):
    def test_compiled_c_repair(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); source=root/"source"
            req=cpp_fixture(source)
            before=CPP_BAD.replace("inline int", "static inline int")
            after=before.replace("return 99", "return 100")
            (source/"level.hpp").write_text(before)
            req=req.model_copy(update={"checks":[c.model_copy(update={"argv":["gcc","-x","c","-std=c11",*c.argv[2:]]})
                if c.kind=="build" else c for c in req.checks]})
            class Provider:
                def __init__(self): self.calls=0
                def invoke(self,messages,*,role):
                    if role=="subagent":
                        return '{"type":"finish","summary":"Reviewed C repair","findings":[],"gaps":[]}'
                    self.calls+=1
                    if self.calls==1:
                        return json.dumps({"type":"tool","tool":"read_file","args":{"path":"level.hpp"}})
                    if self.calls==2:
                        return json.dumps({"type":"tool","tool":"apply_patch","args":{"changes":[
                            {"path":"level.hpp","expected_hash":hashlib.sha256(before.encode()).hexdigest(),"content":after}]}})
                    return json.dumps({"type":"finish","summary":"Repaired C clamp","diagnosis":"The clamp returned 99 for values above its maximum.","diagnosis_sources":cited_source(messages),"findings":[],"gaps":[]})
            harness=Harness(root/"state",autostart=False); run_id=harness.submit(req)
            Engine(harness.store,run_id,Provider()).execute()
            result=harness.result(run_id)
            self.assertEqual(result.outcome,"verified",result.gaps)

    def test_compiled_cpp_repair_and_independent_regression(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); source = root/"source"
            req = cpp_fixture(source)
            class Provider(RepairProvider):
                def invoke(self, messages, *, role):
                    if role == "subagent":
                        return super().invoke(messages, role=role)
                    self.main_calls += 1
                    if self.main_calls == 1:
                        return json.dumps({"type":"tool","tool":"read_file","args":{"path":"level.hpp"}})
                    if self.main_calls == 2:
                        return json.dumps({"type":"tool","tool":"apply_patch","args":{"changes":[
                            {"path":"level.hpp","expected_hash":hashlib.sha256(CPP_BAD.encode()).hexdigest(),"content":CPP_GOOD}]}})
                    return json.dumps({"type":"finish","summary":"Fixed upper-bound clamping","diagnosis":"The function returned 99 where its contract requires a maximum of 100.","diagnosis_sources":cited_source(messages),"findings":[],"gaps":[]})
            harness = Harness(root/"state", autostart=False)
            run_id = harness.submit(req)
            Engine(harness.store, run_id, Provider()).execute()
            result = harness.result(run_id)
            self.assertIsNotNone(result, harness.status(run_id).error)
            self.assertEqual(result.outcome, "verified", result.gaps)
            self.assertNotIn("diff --git a/build-", result.patch)
            self.assertEqual((source/"level.hpp").read_text(), CPP_BAD)
            self.assertTrue(any(c["check_id"] == "acceptance" and c["phase"] == "baseline" and c["exit_code"] == 1 for c in result.checks))

    def test_original_tests_cannot_be_bypassed_by_a_passing_replacement(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); source = root/"source"; fixture(source)
            harness = Harness(root/"state", autostart=False); run_id = harness.submit(request(source))
            class Bypass(RepairProvider):
                def invoke(self, messages, *, role):
                    if role == "main" and self.main_calls == 2:
                        self.main_calls += 1
                        return json.dumps({"type":"tool","tool":"apply_patch","args":{"changes":[
                            {"path":"tests/test_acceptance.py","expected_hash":digest(source/"tests/test_acceptance.py"),
                             "content":"# emptied oracle\n"}]}})
                    return super().invoke(messages, role=role)
            Engine(harness.store, run_id, Bypass()).execute()
            result = harness.result(run_id)
            self.assertEqual(result.outcome, "unverified", result.gaps)
            self.assertTrue(any("oracle changed" in gap for gap in result.gaps))

    def test_interrupted_command_is_rerun_and_cancellation_stops_execution(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); source = root/"source"; fixture(source)
            harness = Harness(root/"state", autostart=False); run_id = harness.submit(request(source))
            engine = Engine(harness.store, run_id, RepairProvider())
            engine.workspace.prepare(source); engine.checks.initialize(engine.request)
            harness.store.save_receipt(run_id, "command", "command", "started", {"argv":["false"]})
            result = engine.broker.run_command(CommandArgs(argv=[sys.executable,"-c","print('rerun proof')"]), key="command")
            self.assertEqual(result["exit_code"], 0); self.assertIn("rerun proof", result["output"])
            self.assertEqual(harness.store.receipt(run_id, "command")["state"], "completed")
            engine.store.transition(run_id,"running")
            # Leave enough time for the child process to start and flush its
            # first line before cancellation; otherwise slow CI hosts can kill
            # it during interpreter startup and make this assertion flaky.
            timer = threading.Timer(0.75, engine.stop.set); timer.start()
            started = time.monotonic()
            with self.assertRaises(InterruptedError):
                engine.broker.run_command(CommandArgs(argv=[sys.executable,"-u","-c","import time;print('before cancel');time.sleep(60)"]), key="cancelled-command")
            timer.join(); self.assertLess(time.monotonic()-started, 3)
            receipt=harness.store.receipt(run_id,"cancelled-command")
            self.assertEqual(receipt["state"],"interrupted")
            self.assertIn("before cancel",receipt["data"]["output"])
            engine.pool.close()

    def test_nested_project_instructions_are_required_before_editing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); source = root/"source"; fixture(source)
            (source/"tests/AGENTS.md").write_text("Keep every original test.\n")
            harness=Harness(root/"state",autostart=False); run_id=harness.submit(request(source))
            engine=Engine(harness.store,run_id,RepairProvider()); engine.workspace.prepare(source)
            args=PatchArgs(changes=[Change(path="tests/new.py", expected_hash=None,content="pass\n")])
            with self.assertRaisesRegex(ValueError,"AGENTS.md"):
                engine.broker.apply_patch(args,key="nested")
            engine.broker.read_file(ReadArgs(path="tests/AGENTS.md"))
            engine.broker.apply_patch(args,key="nested")
            engine.pool.close()

    def test_patch_contains_deletions_modes_and_new_ignored_source_only(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); source = root/"source"; fixture(source)
            harness=Harness(root/"state",autostart=False); run_id=harness.submit(request(source))
            engine=Engine(harness.store,run_id,RepairProvider()); engine.workspace.prepare(source)
            work=engine.workspace.root
            (work/"latency.py").unlink()
            (work/"tool.py").write_text("pass\n"); (work/"tool.py").chmod(0o755)
            (work/"build-output").mkdir(); (work/"build-output/binary").write_bytes(b"\x00data")
            diff=engine.workspace.patch()
            self.assertIn("deleted file mode",diff)
            self.assertIn("new file mode 100755",diff)
            self.assertNotIn("build-output",diff)
            engine.pool.close()

    def test_retained_proto_time_false_positives_cannot_become_proof(self):
        evidence = Path(__file__).resolve().parents[1]/"development-evidence/proto-plugin-review-jQXQ6q"
        retained = (evidence/"direct-review/source-bundle-focused/01.md").read_text()
        self.assertLess(retained.index("if (registrationBlocks_ != 0)"), retained.index("Entry entry;"))
        self.assertIn("cacheInterfaces", retained)
        rejected = ["PluginManager::add lacks a registration guard", "plugin_at is used without a null check",
                    "captureLastError lacks fallback handling", "onAttach exceptions bypass initialize's catch"]
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); source=root/"source"; source.mkdir()
            (source/"PluginManager-excerpt.md").write_text(retained)
            harness=Harness(root/"state",autostart=False)
            run_id=harness.submit(TaskRequest(root=str(source),task="Review the retained plugin source",mode="review"))
            class FalsePositive:
                def invoke(self,messages,*,role):
                    return json.dumps({"type":"finish","summary":"Model claims confirmed defects", "findings":[
                        {"title":title,"classification":"confirmed_gap"} for title in rejected],"gaps":[]})
            Engine(harness.store,run_id,FalsePositive()).execute()
            result=harness.result(run_id)
            self.assertEqual(result.outcome,"unverified")
            self.assertTrue(all(f["classification"]=="hypothesis" for f in result.findings))


if __name__ == "__main__":
    unittest.main()
