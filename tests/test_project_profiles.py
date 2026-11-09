from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from coding_agent import CheckSpec, Harness, TaskRequest
from coding_agent.project_profiles import resolve_request
from coding_agent.types import EvidencePackage, RunResult


class ProjectProfileTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name) / "repo"
        self.root.mkdir()
        skill = self.root / ".agents/skills/test-domain"
        skill.mkdir(parents=True)
        (skill / "SKILL.md").write_text("---\nname: test-domain\ndescription: test skill\n---\nUse the test tool.\n")
        self.definitions = {"sample": {"roots": [str(self.root)], "skills": ["test-domain"], "checks": [
            {"id": "default-check", "argv": ["true"], "kind": "regression"}
        ]}}

    def tearDown(self):
        self.temp.cleanup()

    def test_project_root_selects_profile_checks_and_read_only_skill_snapshot(self):
        harness = Harness(Path(self.temp.name) / "state", project_profiles=self.definitions, autostart=False)
        run_id = harness.submit(TaskRequest(root=str(self.root), task="Inspect project"))
        row = harness.store.get(run_id)
        self.assertEqual(row["request"]["project_profile"], "sample")
        self.assertEqual(row["request"]["skill_ids"], ["test-domain"])
        self.assertEqual(row["request"]["checks"][0]["id"], "default-check")
        snapshot = row["config"]["_resolved_skills"][0]
        self.assertEqual(snapshot["id"], "test-domain")
        self.assertEqual(snapshot["version"], snapshot["sha256"][:16])
        self.assertIn("Use the test tool", snapshot["content"])

    def test_explicit_checks_and_empty_skill_list_override_profile_defaults(self):
        request = TaskRequest(root=str(self.root), task="Inspect project", project_profile="sample",
                              checks=[CheckSpec(id="explicit", argv=["true"], kind="regression")], skill_ids=[])
        resolved, skills = resolve_request(request, self.root, self.definitions)
        self.assertEqual([c.id for c in resolved.checks], ["explicit"])
        self.assertEqual(resolved.skill_ids, [])
        self.assertEqual(skills, [])

    def test_unknown_skills_and_mismatched_profile_roots_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "skill not found"):
            resolve_request(TaskRequest(root=str(self.root), task="Inspect", skill_ids=["missing"]), self.root, self.definitions)
        other = Path(self.temp.name) / "other"
        other.mkdir()
        with self.assertRaisesRegex(ValueError, "outside profile"):
            resolve_request(TaskRequest(root=str(other), task="Inspect", project_profile="sample"), other, self.definitions)

    def test_evidence_package_is_structured_in_public_result(self):
        package = EvidencePackage(baseline_checks=[{"check_id": "acceptance", "passed": False}],
                                  reproducer_artifacts=["a" * 64], diagnosis="Incorrect boundary.",
                                  diagnosis_sources=["b" * 64], patch_artifact="c" * 64,
                                  patch_source_hash="d" * 64,
                                  verification_checks=[{"check_id": "acceptance", "passed": True}],
                                  remaining_gaps=[])
        result = RunResult(run_id="e" * 32, outcome="verified", summary="fixed", evidence=package)
        self.assertEqual(result.evidence.baseline_checks[0]["passed"], False)
        self.assertEqual(result.evidence.patch_source_hash, "d" * 64)
        self.assertEqual(result.model_dump()["evidence"]["diagnosis_sources"], ["b" * 64])


if __name__ == "__main__":
    unittest.main()
