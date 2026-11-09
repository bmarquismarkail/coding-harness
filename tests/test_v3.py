"""Provider-free v0.3 acceptance and adversarial evidence regressions."""
import hashlib
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from coding_agent import Harness, TaskRequest, CheckSpec
from coding_agent.engine import Engine
from coding_agent.evidence import freeze_bundle, reviewer_view, adjudicate_response, read_json
from coding_agent.investigation import discover_scope, evidence_progress
from coding_agent.review_types import DefectProbeSpec, FindingProposal, ReviewDetails, Coverage
from coding_agent.types import Profile, RunResult
from coding_agent.tools import ReadArgs
from coding_agent.workspace import tree_hash

FIXTURE = Path(__file__).parent / 'fixtures/cpp-8uqnat7r'


class EvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.record = json.loads((FIXTURE/'record.json').read_text())
        self.source = self.root/'source'
        shutil.copytree(FIXTURE/'baseline', self.source)
        self.harness = Harness(self.root/'state', autostart=False)
        self.request = TaskRequest.model_validate({**self.record['request'], 'root': str(self.source), 'profile': 'coding'})
        self.rid = self.harness.submit(self.request)
        self.engine = Engine(self.harness.store, self.rid, provider=object())
        self.engine.workspace.prepare(self.source)
        shutil.copy2(FIXTURE/'workspace/level.hpp', self.engine.workspace.root/'level.hpp')
        self.engine.checks.initialize(self.request)
        for receipt in self.record['receipts']:
            data = receipt['data']
            if data.get('artifact_id'):
                aid = data['artifact_id']
                raw = (FIXTURE/'artifacts'/aid).read_bytes()
                self.assertEqual(hashlib.sha256(raw).hexdigest(), aid)
                self.assertEqual(self.harness.store.artifact(self.rid, raw), aid)
            self.harness.store.save_receipt(self.rid, receipt['key'], receipt['kind'], receipt['state'], data)
        self.raw = json.loads((FIXTURE/'artifacts'/self.record['review_artifact']).read_text())['results'][0]
        # Explicit fixture projection of the archived report into the new reviewer protocol.
        # Original bytes remain in the proposal artifact, not rewritten in the archive.
        self.typed = {**self.raw, 'review': {
            'observations': [self.raw['findings'][0]['note']],
            'challenges': [{'kind': 'check_missing', 'check_id': 'acceptance', 'phase': p}
                           for p in ('baseline', 'candidate')]}}

    def tearDown(self):
        self.engine.pool.close()
        self.temp.cleanup()

    def bundle(self):
        return freeze_bundle(self.engine)

    def finish(self, gaps):
        return self.engine.checks.finish(self.request, 'Recorded repair', [], gaps,
            diagnosis='Clamp observation', diagnosis_sources=['retained-fixture'])

    def test_original_bytes_reproduce_baseline_and_revised_acceptance_pair(self):
        baseline = json.loads((FIXTURE/'baseline-result.json').read_text())
        original = self.finish(self.raw['gaps'])
        self.assertEqual(original.outcome, baseline['outcome'])
        self.assertEqual(original.gaps, baseline['gaps'])
        bundle, aid = self.bundle()
        findings, gaps, decision = adjudicate_response(self.harness.store, bundle, aid, self.typed)
        self.assertFalse(findings)
        self.assertFalse(gaps)
        self.assertEqual(self.finish(gaps).outcome, 'verified')
        self.assertTrue(read_json(self.harness.store, self.rid, decision)['decisions'])
        with self.harness.store.db() as db:
            db.execute("DELETE FROM receipts WHERE run_id=? AND kind='check' AND json_extract(data,'$.phase')='candidate'", (self.rid,))
        bundle, aid = self.bundle()
        _, gaps, _ = adjudicate_response(self.harness.store, bundle, aid, self.typed)
        self.assertTrue(gaps)
        self.assertEqual(self.finish(gaps).outcome, 'unverified')

    def test_independent_reviewer_challenge_uses_same_canonical_receipts(self):
        self.engine.bundle, self.engine.bundle_id = self.bundle()
        class Reviewer:
            def invoke(self, messages, *, role):
                assert role == 'reviewer'
                assert 'evidence_view' in messages[1]['content']
                return json.dumps({'challenges': [{'kind': 'check_missing', 'check_id': 'acceptance', 'phase': 'candidate'}]})
        self.engine.provider = Reviewer()
        findings, gaps = self.engine.independent_review('Repair', [], [])
        self.assertFalse(findings or gaps)

    def test_projection_omissions_do_not_change_bundle(self):
        self.engine.broker.read_file(ReadArgs(path='level.hpp'))
        bundle, aid = self.bundle()
        original = bundle.model_dump()
        view = reviewer_view(self.harness.store, bundle, aid, budget=200)
        self.assertTrue(view.omitted_count)
        self.assertEqual(bundle.model_dump(), original)
        self.assertTrue(read_json(self.harness.store, self.rid, aid)['source_evidence'])

    def test_tampered_receipt_never_satisfies_challenge(self):
        with self.harness.store.db() as db:
            db.execute("UPDATE receipts SET data=json_set(data,'$.passed',0) WHERE run_id=? AND json_extract(data,'$.phase')='candidate'", (self.rid,))
        bundle, aid = self.bundle()
        self.assertTrue(bundle.integrity_gaps)
        _, gaps, _ = adjudicate_response(self.harness.store, bundle, aid, {'challenges': [
            {'kind': 'check_missing', 'check_id': 'acceptance', 'phase': 'candidate'}]})
        self.assertTrue(gaps)

    def test_cross_run_artifact_is_not_resolved(self):
        bundle, aid = self.bundle()
        other = self.harness.submit(self.request)
        with self.assertRaises(FileNotFoundError):
            read_json(self.harness.store, other, aid)

    def test_wrong_snapshot_and_omitted_range_cannot_support_claim(self):
        evidence = self.engine.broker.read_file(ReadArgs(path='level.hpp', baseline=True))
        bundle, aid = self.bundle()
        proposal = dict(id='clamp', title='Clamp', claim='Wrong bound', path='level.hpp', evidence_ids=[evidence['evidence_id']])
        found, _, _ = adjudicate_response(self.harness.store, bundle, aid, {'findings': [proposal]})
        self.assertEqual(found[0].status, 'needs_human_review')
        _, gaps, _ = adjudicate_response(self.harness.store, bundle, aid, {'challenges': [
            {'kind': 'source_examined', 'evidence_id': '0'*64}]})
        self.assertTrue(gaps)

    def test_model_cannot_assign_its_own_status(self):
        with self.assertRaises(ValueError):
            FindingProposal(id='x', title='x', claim='x', path='level.hpp', status='reproduced')

    def test_legacy_results_and_review_roundtrip(self):
        old = RunResult(run_id=self.rid, outcome='unverified', summary='old', findings=[{'note': 'legacy'}])
        self.assertIsNone(old.review)
        old.review = ReviewDetails(completion='partial', partial_reasons=['budget_exhausted'])
        self.assertEqual(RunResult.model_validate_json(old.model_dump_json()), old)

    def test_source_move_makes_probe_target_inconclusive(self):
        bundle, aid = self.bundle()
        probe = dict(defect_id='bound', observation='Upper bound violated', check_id='acceptance', path='old.hpp',
            start_line=1, end_line=1, phase='candidate', source_hash=bundle.candidate_hash,
            wrapper='pass', observed_exit_codes=[0], not_observed_exit_codes=[1], artifact_id='f'*64)
        bundle.defect_probes = [probe]
        for check in bundle.checks:
            check['probe_definition_id'] = 'f'*64
        aid = self.harness.store.artifact(self.rid, bundle.model_dump())
        self.harness.store.event(self.rid, 'evidence_bundle', {'artifact_id': aid})
        found, _, _ = adjudicate_response(self.harness.store, bundle, aid, {'findings': [dict(
            id='bound', title='Bound', claim='Root cause speculation', path='level.hpp', defect_id='bound')]})
        self.assertEqual(found[0].status, 'needs_human_review')


class InvestigationTests(unittest.TestCase):
    def test_new_ranges_overlap_and_revision_refresh(self):
        result = dict(kind='source', path='x.py', sha256='a', start=1, end=10)
        ledger, new = evidence_progress({}, {'result': result}, ['x.py'])
        self.assertTrue(new)
        ledger, new = evidence_progress(ledger, {'result': {**result, 'start': 2, 'end': 5}}, ['x.py'])
        self.assertFalse(new)
        _, new = evidence_progress(ledger, {'result': {**result, 'sha256': 'b'}}, ['x.py'])
        self.assertTrue(new)
        _, new = evidence_progress({}, {'result': result}, ['other.py'])
        self.assertFalse(new)

    def test_scope_caps_and_deleted_paths(self):
        from coding_agent.workspace import Workspace
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); source = root/'source'; source.mkdir()
            (source/'a.py').write_text('import b\ndef exported(): return 1\n')
            (source/'caller.py').write_text('value = exported()\n')
            (source/'b.py').write_text('value = 1\n')
            (root/'run').mkdir()
            workspace = Workspace(root/'run'); workspace.prepare(source)
            (workspace.root/'a.py').unlink()
            scope = discover_scope(workspace, Profile(name='test'), implementation=True)
            self.assertIn('a.py', scope.intended_paths)
            self.assertIn('caller.py', scope.intended_paths)
            scope = discover_scope(workspace, Profile(name='test', discovery_paths=1))
            self.assertEqual(scope.limit_reached, 'paths')
            self.assertTrue(scope.omitted_paths)
            scope = discover_scope(workspace, Profile(name='test', discovery_bytes=1))
            self.assertEqual(scope.limit_reached, 'bytes')
            ticks = iter([0, 3])
            scope = discover_scope(workspace, Profile(name='test'), clock=lambda: next(ticks))
            self.assertEqual(scope.limit_reached, 'seconds')

    def test_drafts_survive_protocol_exhaustion_and_cancellation(self):
        for cancel in (False, True):
            with self.subTest(cancel=cancel), tempfile.TemporaryDirectory() as d:
                root = Path(d); source = root/'source'; source.mkdir(); (source/'x.py').write_text('x=1\n')
                harness = Harness(root/'state', autostart=False)
                rid = harness.submit(TaskRequest(root=str(source), task='Review x', mode='review'))
                class Provider:
                    calls = 0
                    def invoke(self, messages, *, role):
                        if role == 'reviewer':
                            return '{}'
                        self.calls += 1
                        if self.calls == 1:
                            return json.dumps({'type': 'tool', 'tool': 'report_findings', 'args': {'findings': [
                                dict(id='x', title='Possible x issue', claim='Needs investigation', path='x.py')]}})
                        if cancel:
                            harness.cancel(rid)
                        return 'invalid JSON'
                engine = Engine(harness.store, rid, Provider()); engine.execute()
                result = harness.result(rid)
                self.assertEqual(result.review.completion, 'partial')
                self.assertEqual(len(result.review.findings), 1)
                self.assertEqual(result.outcome, 'cancelled' if cancel else 'unverified')


class QualificationTests(unittest.TestCase):
    def test_denominators_recovery_and_confidence(self):
        from coding_agent.qualification import report
        result = report({'protocol': [[True]]*99 + [[False, True]], 'cases': [
            {'id': 'bug', 'language': 'cpp', 'expected': ['upper'], 'detected': ['upper']},
            {'id': 'clean', 'language': 'cpp'}], 'promotions': [True]})
        self.assertEqual(result['metrics']['first_response']['observed'], .99)
        self.assertEqual(result['metrics']['after_recovery']['observed'], 1)
        self.assertFalse(result['statistical_probability_99_established'])
        self.assertTrue(result['observed_gates_pass'])
        self.assertFalse(report({'purpose': 'release', 'protocol': [[True]]*100})['observed_gates']['protocol_sample'])
        self.assertIsNone(report({})['metrics']['valid_promotions']['observed'])
        self.assertFalse(report({})['observed_gates_pass'])


class TrustedProbeTests(unittest.TestCase):
    def test_trusted_observation_and_inconclusive_execution(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); source = root/'source'; source.mkdir()
            (source/'value.py').write_text('value = 99\n')
            fingerprint = tree_hash(source)
            probe = DefectProbeSpec(defect_id='upper', observation='The fixture returns 99.', check_id='probe',
                path='value.py', source_hash=fingerprint, phase='candidate',
                wrapper='from pathlib import Path\nimport sys\nsys.exit(42 if "99" in Path("value.py").read_text() else 0)\n',
                observed_exit_codes=[42], not_observed_exit_codes=[0])
            harness = Harness(root/'state', autostart=False)
            rid = harness.submit(TaskRequest(root=str(source), task='Review value', mode='review',
                checks=[CheckSpec(id='probe', argv=['ignored-by-wrapper'])], defect_probes=[probe]))
            engine = Engine(harness.store, rid, object()); engine.workspace.prepare(source); engine.checks.initialize(engine.request)
            try:
                receipt = engine.checks.execute('probe', 'candidate', key='probe')
                self.assertEqual(receipt['exit_code'], 42)
                bundle, aid = freeze_bundle(engine)
                view = reviewer_view(harness.store, bundle, aid, budget=12000)
                self.assertEqual(view.defect_probes[0]['defect_id'], 'upper')
                from coding_agent.tools import PageArgs
                self.assertEqual(engine.broker.list_checks(PageArgs())['checks'][0]['trusted_observations'][0]['defect_id'], 'upper')
                raw = {'findings': [dict(id='upper', title='Upper bound', claim='Possibly incorrect branch',
                                        path='value.py', defect_id='upper')]}
                findings, _, _ = adjudicate_response(harness.store, bundle, aid, raw)
                self.assertEqual(findings[0].status, 'reproduced')
                self.assertEqual(findings[0].observation, probe.observation)
                self.assertEqual(findings[0].proposal.claim, 'Possibly incorrect branch')
                # The immutable definition survives caller/workspace edits.
                (engine.workspace.root/'value.py').rename(engine.workspace.root/'moved.py')
                bundle, aid = freeze_bundle(engine)
                findings, _, _ = adjudicate_response(harness.store, bundle, aid, raw)
                self.assertEqual(findings[0].status, 'needs_human_review')
            finally:
                engine.pool.close()


class BenchmarkFixtureTests(unittest.TestCase):
    def test_python_c_cpp_clean_and_defective_oracles(self):
        from coding_agent.qualification import prepare_fixtures
        from coding_agent.sandbox import SandboxRunner
        with tempfile.TemporaryDirectory() as d:
            requests, gold = prepare_fixtures(Path(d)/'fixtures')
            self.assertEqual(len(requests), 6)
            from coding_agent.types import parse_action
            opportunities = json.loads((Path(d)/'fixtures/protocol.json').read_text())
            self.assertEqual(len(opportunities), 100)
            for opportunity in opportunities:
                parse_action(json.dumps(opportunity['requested_action']))
            runner = SandboxRunner(Profile(name='fixture'))
            for task, expected in zip(requests, gold):
                request = TaskRequest.model_validate(task['request'])
                for spec in request.checks:
                    result = runner.run(Path(request.root), spec.argv)
                    if spec.kind == 'build':
                        self.assertEqual(result['exit_code'], 0, result['output'])
                    else:
                        self.assertEqual(result['exit_code'] != 0, bool(expected['expected']), result['output'])


class ProgressAcceptanceTests(unittest.TestCase):
    def test_scoped_reads_continue_but_unrelated_new_files_stagnate(self):
        for relevant in (True, False):
            with self.subTest(relevant=relevant), tempfile.TemporaryDirectory() as d:
                root = Path(d); source = root/'source'; source.mkdir()
                (source/'target.py').write_text('a = 1\n'*40)
                for i in range(8):
                    (source/f'other{i}.py').write_text('z = 1\n')
                harness = Harness(root/'state', autostart=False,
                    profiles={'review': {'review_no_progress_actions': 6}})
                rid = harness.submit(TaskRequest(root=str(source), task='Review target', mode='review', focus_paths=['target.py']))
                class Provider:
                    calls = 0
                    def invoke(self, messages, *, role):
                        if role == 'reviewer': return '{}'
                        self.calls += 1
                        if self.calls <= 8:
                            args = {'path': 'target.py', 'start': self.calls, 'lines': 1} if relevant else {'path': f'other{self.calls-1}.py'}
                            return json.dumps({'type': 'tool', 'tool': 'read_file', 'args': args})
                        return json.dumps({'type': 'finish', 'summary': 'Review finished'})
                provider = Provider(); Engine(harness.store, rid, provider).execute()
                review = harness.result(rid).review
                self.assertEqual(review.completion, 'completed' if relevant else 'partial')
                self.assertEqual(provider.calls, 9 if relevant else 6)
