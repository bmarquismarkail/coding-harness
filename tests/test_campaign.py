import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from coding_agent.campaign import MODEL_OVERRIDES, protocol_attempt, run_protocol, run_suite, save
from coding_agent.config import load_config
from coding_agent.model import ModelError
from coding_agent.runtime_model import RecoverableResponseError
from coding_agent.service import Harness

ACTION = {"type": "tool", "tool": "list_files", "args": {}}


class CampaignTests(unittest.TestCase):
    def test_only_live_suite_inherits_model_overrides(self):
        overrides = {name: "override" for name in MODEL_OVERRIDES}
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, overrides), patch(
            "coding_agent.campaign.subprocess.run"
        ) as run:
            run.return_value.returncode = 0
            directory = Path(tmp)
            self.assertTrue(run_suite(directory))
            self.assertTrue(MODEL_OVERRIDES.isdisjoint(run.call_args.kwargs["env"]))
            self.assertTrue(run_suite(directory, directory / "config.json"))
            self.assertEqual({key: run.call_args.kwargs["env"][key] for key in overrides}, overrides)

    def test_protocol_active_budget_stops_requests(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            fixtures = directory / "fixtures"
            fixtures.mkdir()
            save(fixtures / "protocol.json", [
                {"root": str(fixtures), "id": i, "requested_action": ACTION} for i in range(100)
            ])
            harness = Harness(directory / "runtime", config=load_config("agent-config.json"),
                              profiles={"protocol-smoke": {"max_active_seconds": 2}}, autostart=False)
            now = 1000

            def invoke(*args, **kwargs):
                nonlocal now
                now += 2
                return json.dumps(ACTION)

            with patch("coding_agent.store.time.time", side_effect=lambda: now), patch(
                "coding_agent.campaign.RecordingProvider.invoke", side_effect=invoke
            ) as model:
                metrics = run_protocol(harness, directory)
                self.assertEqual(model.call_count, 2)
                self.assertTrue(all(item["budget_exhausted"] and not item["passed"] for item in metrics.values()))
                now += 1000
                for role in metrics:
                    journal = json.loads((directory / f"protocol-{role}.json").read_text())
                    self.assertEqual(harness.store.status(journal["run_id"]).active_seconds, 2)

    def test_recovery_classifies_provider_and_action_failures(self):
        class Provider:
            def invoke(self, messages, *, role):
                value = next(responses)
                if isinstance(value, Exception):
                    raise value
                return value

        responses = iter([ModelError("timed out"), "not JSON",
                          RecoverableResponseError("truncated", "token limit"),
                          json.dumps(ACTION)])
        for expected in ("timeout", "malformed_action", "truncated", "valid"):
            self.assertEqual(protocol_attempt(Provider(), "main", ACTION, []), expected)

    def test_cancelled_attempt_retained_and_resume_does_not_repeat_completed_samples(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            fixtures = directory / "fixtures"
            fixtures.mkdir()
            save(fixtures / "protocol.json", [
                {"root": str(fixtures), "id": i, "requested_action": ACTION} for i in range(100)
            ])
            harness = Harness(directory / "runtime", config=load_config("agent-config.json"),
                              profiles={"protocol-smoke": {"max_model_requests": 300}}, autostart=False)
            with patch("coding_agent.campaign.RecordingProvider.invoke",
                       side_effect=[json.dumps(ACTION), InterruptedError]):
                with self.assertRaises(InterruptedError):
                    run_protocol(harness, directory)
            with patch("coding_agent.campaign.RecordingProvider.invoke", return_value=json.dumps(ACTION)) as model:
                metrics = run_protocol(harness, directory)
            self.assertEqual(model.call_count, 199)
            self.assertTrue(all(m["passed"] for m in metrics.values()))
            journal = json.loads((directory / "protocol-main.json").read_text())
            self.assertEqual(journal["samples"]["1"], ["cancelled", "valid"])
