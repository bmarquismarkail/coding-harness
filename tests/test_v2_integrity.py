from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from coding_agent import Harness, TaskRequest
from coding_agent.engine import Engine
from coding_agent.profiles import configured_models, resolve_profile
from coding_agent.tools import Change, PatchArgs, ReadArgs, WorkerTask
from coding_agent.types import CheckSpec, Profile
from coding_agent.workspace import Workspace, digest
from test_v2 import GOOD, RepairProvider, fixture, request


class IntegrityTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(); self.root=Path(self.temp.name)
        self.source=self.root/"source"; fixture(self.source)
        self.harness=Harness(self.root/"state",autostart=False)
        self.run_id=self.harness.submit(request(self.source))
        self.engine=Engine(self.harness.store,self.run_id,RepairProvider())
        self.engine.workspace.prepare(self.source); self.engine.checks.initialize(self.engine.request)

    def tearDown(self):
        self.engine.pool.close(); self.temp.cleanup()

    def test_frozen_new_acceptance_test_cannot_be_replaced(self):
        work=self.engine.workspace.root
        oracle=work/"tests/test_added.py"; oracle.write_text("assert True\n")
        self.engine.checks.register(CheckSpec(id="added",argv=[sys.executable,"tests/test_added.py"]),["tests/test_added.py"])
        self.engine.checks.execute("added","baseline",key="baseline-added")
        oracle.write_text("# bypass\n")
        with self.assertRaisesRegex(RuntimeError,"oracle changed"):
            self.engine.checks.execute("added","candidate",key="candidate-added")

    def test_baseline_execution_sources_are_immutable(self):
        self.engine.checks.execute("acceptance","baseline",key="baseline-1")
        root=self.engine.workspace.baseline_check_root()
        (root/"latency.py").write_text(GOOD)
        with self.assertRaisesRegex(RuntimeError,"baseline execution sources changed"):
            self.engine.checks.execute("acceptance","baseline",key="baseline-2")

    def test_check_that_modifies_sources_does_not_pass(self):
        check=CheckSpec(id="mutating",argv=[sys.executable,"-c","from pathlib import Path;Path('latency.py').write_text('pass')"])
        checks=self.engine.checks.definitions(); checks.append({"spec":check.model_dump(),"origin":"request","oracle":{}})
        self.engine.checks._save(checks)
        result=self.engine.checks.execute("mutating","candidate",key="mutation")
        self.assertFalse(result["passed"]); self.assertTrue(result["source_modified"])

    def test_edit_replay_requires_the_identical_action_and_reconciles_modes(self):
        args=PatchArgs(changes=[Change(path="script.py",expected_hash=None,content="pass\n",executable=True)])
        action_hash=hashlib.sha256(args.model_dump_json().encode()).hexdigest()
        target=self.engine.workspace.root/"script.py"; target.write_text("pass\n"); target.chmod(0o644)
        self.harness.store.save_receipt(self.run_id,"partial","edit","started",{"action_hash":action_hash})
        self.engine.broker.apply_patch(args,key="partial")
        self.assertTrue(target.stat().st_mode&0o111)
        with self.assertRaisesRegex(RuntimeError,"does not match"):
            self.engine.broker.apply_patch(PatchArgs(changes=[Change(path="script.py",expected_hash=digest(target),content="changed\n")]),key="partial")

    def test_symlinks_fail_instead_of_silently_omitting_source(self):
        (self.source/"linked.py").symlink_to("latency.py")
        folder=self.root/"new-job";folder.mkdir()
        with self.assertRaisesRegex(ValueError,"symlink"):
            Workspace(folder).prepare(self.source)

    def test_profiles_keep_models_legacy_settings_and_deep_defaults(self):
        config=self.harness.config
        self.assertIs(configured_models(config,resolve_profile("legacy",config)),config)
        deep=resolve_profile("deep",config,{"deep":{"max_actions":10}})
        self.assertTrue(deep.enable_thinking); self.assertEqual(deep.main_max_tokens,8192)
        coding=configured_models(config,resolve_profile("coding",config))
        self.assertEqual(coding.main.model,config.main.model)
        self.assertEqual(coding.main.base_url,config.main.base_url)
        self.assertFalse(coding.main.enable_thinking); self.assertFalse(coding.subagents.enable_thinking)
        self.assertEqual(Profile(name="custom",command_timeout_seconds=1200).command_timeout_seconds,1200)

    def test_model_and_tool_budgets_are_independent_and_never_reset(self):
        profile=self.harness.store.get(self.run_id)["profile"]
        profile.update(max_actions=1,max_model_requests=2)
        with self.harness.store.db() as db:
            db.execute("UPDATE runs SET profile=? WHERE id=?",(json.dumps(profile),self.run_id))
        for _ in range(2): self.harness.store.consume(self.run_id,kind="model")
        self.assertEqual(self.harness.status(self.run_id).actions,0)
        self.harness.store.consume(self.run_id)
        self.assertEqual(self.harness.status(self.run_id).model_requests,2)
        for kind in ("model","tool"):
            with self.assertRaisesRegex(RuntimeError,"budget exhausted"):
                self.harness.store.consume(self.run_id,kind=kind)

    def test_profile_change_invalidates_active_specialist_cache(self):
        task=WorkerTask(id="scope",task="Review this source",paths=["latency.py"])
        self.engine.pool.start([task])
        for future in self.engine.pool.futures.values(): future.result()
        self.assertTrue(self.engine.pool.pending())
        self.harness.store.transition(self.run_id,"interrupted")
        self.harness.resume(self.run_id,profile="deep")
        restarted=Engine(self.harness.store,self.run_id,RepairProvider())
        restarted.pool.reconcile()
        self.assertFalse(restarted.pool.pending())
        self.assertEqual(restarted.pool.rows()[0]["state"],"completed")
        self.assertIn("invalidated",restarted.pool.rows()[0]["data"])
        with self.assertRaisesRegex(ValueError,"profile changed"):
            restarted.pool.collect(["scope"])
        restarted.pool.close()

    def test_cancelled_orphan_keeps_cancelled_verdict_and_partial_patch(self):
        from coding_agent.supervisor import record_exit
        (self.engine.workspace.root/"latency.py").write_text(GOOD)
        self.harness.store.transition(self.run_id,"running")
        with self.harness.store.db() as db:
            db.execute("UPDATE runs SET cancel_requested=1 WHERE id=?",(self.run_id,))
        record_exit(self.harness.store,self.run_id,"orphaned worker exited")
        self.assertEqual(self.harness.status(self.run_id).state,"cancelled")
        result=self.harness.result(self.run_id)
        self.assertEqual(result.outcome,"cancelled")
        self.assertIn("+    count = 0",result.patch)

    def test_orphan_with_unreadable_patch_still_gets_terminal_result(self):
        from coding_agent.supervisor import record_exit

        (self.engine.workspace.folder / "baseline.index").write_bytes(b"invalid index")
        self.harness.store.transition(self.run_id, "running")
        record_exit(self.harness.store, self.run_id, "orphaned worker exited")
        self.assertEqual(self.harness.status(self.run_id).state, "interrupted")
        result = self.harness.result(self.run_id)
        self.assertEqual(result.outcome, "unverified")
        self.assertEqual(result.review.completion, "partial")
        self.assertTrue(any("Partial patch could not be exported" in gap for gap in result.gaps))

    def test_status_and_artifacts_remain_readable_after_source_moves(self):
        key=self.harness.store.artifact(self.run_id,"retained")
        self.source.rename(self.root/"moved")
        self.assertEqual(self.harness.status(self.run_id).state,"queued")
        self.assertEqual(self.harness.artifact(self.run_id,key)["content"],"retained")

    def test_git_filters_cannot_execute_project_code_on_host(self):
        def git(*args):
            return subprocess.run(["git","-C",str(self.source),*args],capture_output=True,check=True)
        git("init");git("add",".")
        git("-c","user.name=Fixture","-c","user.email=fixture@example.invalid","commit","-m","fixture")
        marker=self.root/"host-filter-ran"
        command=f'{sys.executable} -c "from pathlib import Path;Path(\'{marker}\').touch()"'
        git("config","filter.dangerous.clean",command);git("config","filter.dangerous.smudge",command)
        (self.source/".gitattributes").write_text("*.py filter=dangerous\n")
        folder=self.root/"filtered-job";folder.mkdir()
        workspace=Workspace(folder); workspace.prepare(self.source)
        (workspace.root/"latency.py").write_text(GOOD); workspace.patch()
        self.assertFalse(marker.exists())

    def test_malformed_actions_are_feedback_and_not_executed(self):
        class BadThenRepair(RepairProvider):
            def __init__(self):
                super().__init__(); self.bad=True
            def invoke(self,messages,*,role):
                if role=="main" and self.bad:
                    self.bad=False
                    return '{"type":"run_command","args":{"argv":["touch","not-allowed"]}}'
                return super().invoke(messages,role=role)
        self.engine.pool.close()
        Engine(self.harness.store,self.run_id,BadThenRepair()).execute()
        result=self.harness.result(self.run_id)
        self.assertEqual(result.outcome,"verified",result.gaps)
        self.assertFalse((self.engine.workspace.root/"not-allowed").exists())


if __name__=="__main__":
    unittest.main()
