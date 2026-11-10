"""Provider-free v0.3 acceptance and adversarial evidence regressions."""
import hashlib
import json
import os
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
from test_v2 import EMPTY_REVIEW_JSON

FIXTURE = Path(__file__).parent / 'fixtures/cpp-8uqnat7r'


class EvidenceTests(unittest.TestCase):
    def setUp(self):
        """Restore the archived workspace, check receipts, and reviewer response."""
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
        target = self.engine.workspace.root/'level.hpp'
        original_mtime = target.stat().st_mtime_ns
        shutil.copyfile(FIXTURE/'workspace/level.hpp', target)
        # The archived variants have the same size. Give Git a distinct stat entry.
        os.utime(target, ns=(original_mtime, original_mtime + 2_000_000_000))
        self.engine.checks.initialize(self.request)
        for receipt in self.record['receipts']:
            data = receipt['data']
            if data.get('artifact_id'):
                aid = data['artifact_id']
                raw = (FIXTURE/'artifacts'/aid).read_bytes()
                self.assertEqual(hashlib.sha256(raw).hexdigest(), aid)
                self.assertEqual(self.harness.store.artifact(self.rid, raw), aid)
            self.harness.store.save_receipt(self.rid, receipt['key'], receipt['kind'], receipt['state'], data)
        archive_id = self.record['review_artifact']
        archive = (FIXTURE/'artifacts'/archive_id).read_bytes()
        self.assertEqual(self.harness.store.artifact(self.rid, archive), archive_id)
        self.raw = json.loads(archive)['results'][0]
        # Migrate only the archived assertion represented by the two challenges.
        # Any other legacy gaps still go through normal semantic-concern handling.
        migrated_gap = (
            'No independent reproduction of the baseline acceptance failure (exit 1) was performed; '
            'candidate acceptance pass is not yet independently confirmed against a fresh run.'
        )
        self.assertIn(migrated_gap, self.raw['gaps'])
        observation, *other_findings = self.raw['findings']
        self.typed = {**self.raw, 'findings': other_findings,
                      'gaps': [gap for gap in self.raw['gaps'] if gap != migrated_gap], 'review': {
            'observations': [observation['note']],
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
        """Replay the archived check against the original and revised snapshots."""
        baseline = json.loads((FIXTURE/'baseline-result.json').read_text())
        original = self.finish(self.raw['gaps'])
        self.assertEqual(original.outcome, baseline['outcome'])
        self.assertEqual(original.gaps, baseline['gaps'])
        bundle, aid = self.bundle()
        findings, gaps, decision = adjudicate_response(self.harness.store, bundle, aid, self.typed)
        self.assertFalse(findings)
        self.assertFalse(gaps)
        self.assertEqual(self.finish(gaps).outcome, 'verified')
        decisions = read_json(self.harness.store, self.rid, decision)['decisions']
        self.assertEqual([item['supported'] for item in decisions], [False, False])
        with self.harness.store.db() as db:
            db.execute("DELETE FROM receipts WHERE run_id=? AND kind='check' AND json_extract(data,'$.phase')='candidate'", (self.rid,))
        bundle, aid = self.bundle()
        _, gaps, _ = adjudicate_response(self.harness.store, bundle, aid, self.typed)
        self.assertEqual(gaps, ['Required evidence missing: candidate/acceptance'])
        self.assertEqual(self.finish(gaps).outcome, 'unverified')

    def test_independent_reviewer_challenge_uses_same_canonical_receipts(self):
        self.engine.bundle, self.engine.bundle_id = self.bundle()
        class Reviewer:
            def invoke(self, messages, *, role):
                assert role == 'reviewer'
                assert 'evidence_view' in messages[1]['content']
                return json.dumps({**json.loads(EMPTY_REVIEW_JSON), 'challenges': [
                    {'kind': 'check_missing', 'check_id': 'acceptance', 'phase': 'candidate'}]})
        self.engine.provider = Reviewer()
        findings, gaps = self.engine.independent_review('Repair', [], [])
        self.assertFalse(findings or gaps)

    def test_live_review_gates_archived_cpp_repair(self):
        self.engine.bundle, self.engine.bundle_id = self.bundle()
        replies = ((EMPTY_REVIEW_JSON, 'verified'), ('{}', 'unverified'),
                   ('{"error":"review could not be performed"}', 'unverified'),
                   ('{"findings":[]}', 'unverified'))
        for event_count, (reply, outcome) in enumerate(replies, start=1):
            with self.subTest(reply=reply):
                with patch.object(self.engine, 'provider') as provider:
                    provider.invoke.return_value = reply
                    findings, gaps = self.engine.independent_review('Repair', [], [])
                    provider.invoke.assert_called_once()
                    prompt = provider.invoke.call_args.args[0][0]['content']
                    schema = json.loads(prompt.split('must match this schema: ', 1)[1])
                self.assertFalse(findings)
                self.assertEqual(self.finish(gaps).outcome, outcome)
                self.assertEqual(set(schema.get('required', [])), set(json.loads(EMPTY_REVIEW_JSON)))
                events = self.harness.events(self.rid)['items']
                review_events = [e for e in events if e['kind'] in {
                    'independent_model_review', 'review_protocol_failure'}]
                self.assertEqual(len(review_events), event_count)
                event = review_events[-1]
                if outcome == 'unverified':
                    self.assertEqual(event['kind'], 'review_protocol_failure')
                    artifact = self.harness.artifact(self.rid, event['data']['artifact_id'])
                    self.assertEqual(artifact['content'], reply)
        findings, gaps, _ = adjudicate_response(self.harness.store, self.engine.bundle,
                                                self.engine.bundle_id, {})
        self.assertFalse(findings or gaps)

    def test_projection_omissions_do_not_change_bundle(self):
        self.engine.broker.read_file(ReadArgs(path='level.hpp'))
        bundle, aid = self.bundle()
        original = bundle.model_dump()
        view = reviewer_view(self.harness.store, bundle, aid, budget=400)
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
        legacy = RunResult.model_validate({**old.model_dump(), 'review': {
            'completion': 'partial', 'partial_reasons': ['scope_limit'],
            'coverage': {'limit_reached': 'paths', 'unenumerated_remainder': True}}})
        self.assertEqual(legacy.review.coverage.limit_reached, 'paths')
        self.assertIsNone(legacy.review.coverage.scan_limit_reached)
        self.assertEqual(legacy.review.coverage.unscanned_paths, [])
        old.review = ReviewDetails(completion='completed', coverage=Coverage(
            intended_paths=['a.py'], examined_paths=['a.py'], unscanned_paths=['b.py'],
            scan_limit_reached='paths', unenumerated_remainder=True))
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

    def test_genuine_concerns_survive_rejected_factual_challenges(self):
        """Retain semantic concerns when a factual challenge is rejected."""
        bundle, aid = self.bundle()
        legacy_gap = 'The acceptance check exercises only one difficulty.'
        semantic_concern = 'The original report does not assess negative inputs.'
        report = {**self.typed, 'gaps': [legacy_gap],
                  'review': {**self.typed['review'], 'semantic_concerns': [semantic_concern]}}
        _, gaps, _ = adjudicate_response(self.harness.store, bundle, aid, report)
        self.assertEqual(gaps, [semantic_concern, legacy_gap])
        self.assertEqual(self.finish(gaps).outcome, 'unverified')

    def test_complete_view_budget_and_manifest(self):
        bundle, aid = self.bundle()
        bundle.checks *= 30
        view = reviewer_view(self.harness.store, bundle, aid, budget=2400)
        self.assertLessEqual(len(view.model_dump_json().encode()), 2400)
        omitted = read_json(self.harness.store, self.rid, view.omitted_manifest_id)
        self.assertEqual(view.omitted_count, len(omitted))
        self.assertEqual(len(view.checks) + len(omitted), len(bundle.checks))

    def test_latest_execution_wins_over_receipt_order(self):
        bundle, aid = self.bundle()
        check = next(c for c in bundle.checks if c['phase'] == 'candidate')
        bundle.checks = [{**check, 'passed': False, 'executed_at': 20},
                         {**check, 'passed': True, 'executed_at': 10}]
        aid = self.harness.store.artifact(self.rid, bundle.model_dump())
        self.harness.store.event(self.rid, 'evidence_bundle', {'artifact_id': aid})
        _, gaps, _ = adjudicate_response(self.harness.store, bundle, aid, {'challenges': [
            {'kind': 'check_passed', 'check_id': check['check_id'], 'phase': 'candidate'}]})
        self.assertTrue(gaps)

    def test_joint_citation_coverage_and_duplicate_proposals(self):
        (self.engine.workspace.root/'level.hpp').write_text('one\ntwo\nthree\nfour\n')
        left = self.engine.broker.read_file(ReadArgs(path='level.hpp', start=1, lines=1))
        right = self.engine.broker.read_file(ReadArgs(path='level.hpp', start=2, lines=2))
        self.engine.bundle, self.engine.bundle_id = self.bundle()
        proposal = dict(id='range', title='Range', claim='Possible issue', path='level.hpp',
                        start_line=1, end_line=3, evidence_ids=[left['evidence_id'], right['evidence_id']])
        self.engine.review_findings = []; self.engine.adjudication_ids = []
        found, gaps = self.engine.evaluate_review({'findings': [proposal, proposal]})
        self.assertFalse(gaps)
        self.assertEqual(found[0].status, 'hypothesis')
        self.assertEqual(len(self.engine.review_findings), 1)
        proposal['end_line'] = 4
        found, _ = self.engine.evaluate_review({'findings': [proposal]})
        self.assertEqual(found[0].status, 'needs_human_review')

    def test_specialist_can_only_report_drafts_and_read(self):
        from coding_agent.tools import ToolBroker
        broker = ToolBroker(self.harness.store, self.rid, self.engine.workspace, self.engine.checks, scope=['level.hpp'])
        self.assertIn('report_findings', broker.schemas())
        for name in ('apply_patch', 'run_command', 'run_check', 'delegate', 'register_check'):
            with self.assertRaises(PermissionError):
                broker.invoke(name, {}, key=name)
        broker.invoke('report_findings', {'findings': [dict(id='draft', title='Draft', claim='Investigate', path='level.hpp')]}, key='draft')
        self.assertTrue(self.harness.store.receipts(self.rid, 'draft_findings'))

    def test_probe_signals_paths_and_unrelated_errors(self):
        from coding_agent.sandbox import SandboxRunner
        from coding_agent.evidence import read_json
        check = self.engine.checks.definitions()[0]['spec']
        wrapper = self.harness.store.artifact(self.rid, 'pass')
        for code, expected in ((0, False), (42, True)):
            probe = dict(check_id=check['id'], check_spec=check, wrapper_artifact=wrapper,
                         path='level.hpp', phase='candidate', source_hash=tree_hash(self.engine.workspace.root),
                         end_line=1, observed_exit_codes=[0], not_observed_exit_codes=[42])
            aid = self.harness.store.artifact(self.rid, probe)
            self.harness.store.save_receipt(self.rid, 'probe', 'probe_definition', 'completed', {'artifact_id': aid})
            execution = dict(exit_code=code, timed_out=False, output_truncated=False, output='')
            with patch.object(SandboxRunner, 'run', return_value=execution):
                result = self.engine.checks.execute(check['id'], 'candidate', key=f'signal-{code}')
            self.assertEqual(result['passed'], expected)
            self.assertTrue(result['probe_target_valid'])
            with patch.object(SandboxRunner, 'run', return_value=dict(execution)), patch('coding_agent.checks.checked_path', side_effect=ValueError('excluded')):
                result = self.engine.checks.execute(check['id'], 'candidate', key=f'invalid-{code}')
            self.assertFalse(result['probe_target_valid'])
            self.assertFalse(read_json(self.harness.store, self.rid, result['artifact_id'])['probe_target_valid'])
            with patch.object(SandboxRunner, 'run', return_value=dict(execution)), patch('coding_agent.checks.checked_path', side_effect=RuntimeError('unexpected')):
                with self.assertRaises(RuntimeError):
                    self.engine.checks.execute(check['id'], 'candidate', key=f'error-{code}')

    def test_specialist_top_level_findings_are_retained(self):
        from coding_agent.tools import WorkerTask
        proposal = dict(id='draft', title='Draft', claim='Investigate', path='level.hpp')
        class Provider:
            def invoke(self, messages, *, role):
                return json.dumps({'type': 'finish', 'summary': 'Done', 'findings': [proposal]})
        self.engine.provider = Provider()
        self.engine.pool.start([WorkerTask(id='draft', task='Review source', paths=['level.hpp'])])
        self.engine.pool.collect(['draft'])
        receipts = self.harness.store.receipts(self.rid, 'draft_findings')
        self.assertEqual(len(receipts), 1)
        self.assertEqual(read_json(self.harness.store, self.rid, receipts[0]['data']['artifact_id'])['findings'][0]['id'], 'draft')

    def test_baseline_resolution_requires_latest_clean_candidate(self):
        source = self.engine.broker.read_file(ReadArgs(path='level.hpp', baseline=True))
        bundle, _ = self.bundle()
        probe = dict(artifact_id='f'*64, defect_id='bound', observation='Wrong bound',
                     check_id='acceptance', path='level.hpp', phase='baseline',
                     start_line=1, end_line=1, source_hash=bundle.baseline_hash,
                     observed_exit_codes=[0], not_observed_exit_codes=[42], check_spec={})
        bundle.defect_probes = [probe]
        common = dict(check_id='acceptance', receipt_state='completed', artifact_id='e'*64,
                      probe_definition_id='f'*64, probe_target_valid=True, probe_range_valid=True)
        bundle.checks = [{**common, 'phase': 'baseline', 'source_hash': bundle.baseline_hash,
                          'passed': False, 'exit_code': 0, 'executed_at': 1},
                         {**common, 'phase': 'candidate', 'source_hash': bundle.candidate_hash,
                          'passed': True, 'exit_code': 42, 'executed_at': 2}]
        proposal = dict(id='bound', title='Bound', claim='Wrong', path='level.hpp', phase='baseline',
                        defect_id='bound', evidence_ids=[source['evidence_id']])
        for resolved in (True, False):
            if not resolved:
                bundle.checks.insert(0, {**bundle.checks[-1], 'passed': False, 'exit_code': 0, 'executed_at': 3})
            aid = self.harness.store.artifact(self.rid, bundle.model_dump())
            self.harness.store.event(self.rid, 'evidence_bundle', {'artifact_id': aid})
            found, _, _ = adjudicate_response(self.harness.store, bundle, aid, {'findings': [proposal]})
            self.assertEqual(found[0].status, 'reproduced')
            self.assertEqual(found[0].resolved, resolved)


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

    def test_focus_paths_use_workspace_path_safety(self):
        """Reject unsafe requests and exclude unsafe paths from older jobs' scopes."""
        from coding_agent.workspace import Workspace
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'source'
            source.mkdir()
            (source / 'deleted.py').write_text('x = 1\n')
            harness = Harness(root / 'state', autostart=False)
            request = dict(root=str(source), task='Review paths', mode='review')
            invalid = ['../outside.py', 'src/../deleted.py', '.git/config', '.env', str(root / 'outside.py')]
            for focus in invalid:
                with self.subTest(focus=focus), self.assertRaises(ValueError):
                    harness.submit({**request, 'focus_paths': [focus]})

            valid = ['./deleted.py', 'new/module.py', '.env.example']
            run_id = harness.submit({**request, 'focus_paths': valid})
            workspace = Workspace(harness.store.run_dir(run_id))
            workspace.prepare(source)
            (workspace.root / 'deleted.py').unlink()
            scope = discover_scope(workspace, Profile(name='test'), [*invalid, *valid])
            self.assertEqual(scope.intended_paths, ['deleted.py', 'new/module.py', '.env.example'])

    def test_inventory_path_limit_keeps_small_patch_and_focused_review_complete(self):
        """Report optional inventory omissions separately from an unfinished review."""
        from coding_agent.workspace import Workspace
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); source = root/'source'; source.mkdir()
            unrelated = [f'f{i:03d}.cpp' for i in range(500)]
            for path in unrelated:
                (source/path).write_text('int filler = 1;\n')
            target = 'z_changed.cpp'
            (source/target).write_text('int value = 1;\n')
            (root/'run').mkdir()
            workspace = Workspace(root/'run'); workspace.prepare(source)
            (workspace.root/target).write_text('int value = 2;\n')
            scope = discover_scope(workspace, Profile(name='test'), implementation=True)
            self.assertEqual(scope.intended_paths, [target])
            self.assertEqual(scope.omitted_paths, [])
            self.assertEqual(scope.unscanned_paths, unrelated[199:])
            self.assertIsNone(scope.limit_reached)
            self.assertEqual(scope.scan_limit_reached, 'paths')
            self.assertTrue(scope.unenumerated_remainder)

            harness = Harness(root/'state', autostart=False, profiles={'one': {'discovery_paths': 1}})
            for focus, read, profile in (([target], target, None), ([], unrelated[0], 'one')):
                with self.subTest(focus=focus):
                    rid = harness.submit(TaskRequest(root=str(source), task='Review source', mode='review',
                                                     focus_paths=focus, profile=profile))
                    actions = iter([
                        {'type': 'tool', 'tool': 'read_file', 'args': {'path': read}},
                        {'type': 'finish', 'summary': 'Finished review'},
                    ])

                    class Provider:
                        def invoke(self, messages, *, role):
                            return EMPTY_REVIEW_JSON if role == 'reviewer' else json.dumps(next(actions))

                    Engine(harness.store, rid, Provider()).execute()
                    review = harness.result(rid).review
                    self.assertEqual(review.coverage.examined_paths, [read])
                    if focus:
                        self.assertEqual(review.completion, 'completed')
                        self.assertEqual(review.partial_reasons, [])
                        self.assertEqual(review.coverage, scope.model_copy(update={'examined_paths': [read]}))
                    else:
                        self.assertEqual(review.completion, 'partial')
                        self.assertEqual(review.partial_reasons, ['scope_limit'])
                        self.assertEqual(review.coverage.omitted_paths, [*unrelated[1:], target])
                        self.assertEqual(review.coverage.unscanned_paths, [*unrelated, target])

    def test_dependencies_take_priority_over_optional_inventory(self):
        from coding_agent.workspace import Workspace
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); source = root/'source'; source.mkdir()
            (source/'a.py').write_text('import z\n')
            (source/'b.py').write_text('# oversized\n' * 20)
            (source/'z.py').write_text('value = 1\n')
            (root/'run').mkdir()
            workspace = Workspace(root/'run'); workspace.prepare(source)
            for profile, reason in ((Profile(name='test', discovery_paths=2), 'paths'),
                                    (Profile(name='test', discovery_bytes=30), 'bytes')):
                with self.subTest(reason=reason):
                    scope = discover_scope(workspace, profile, ['a.py'])
                    self.assertEqual(scope.intended_paths, ['a.py', 'z.py'])
                    self.assertEqual(scope.unscanned_paths, ['b.py'])
                    self.assertEqual(scope.omitted_paths, [])
                    self.assertIsNone(scope.limit_reached)
                    self.assertEqual(scope.scan_limit_reached, reason)
                    self.assertTrue(scope.unenumerated_remainder)

    def test_required_scan_interruption_preserves_unread_inventory(self):
        from coding_agent.workspace import Workspace
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); source = root/'source'; source.mkdir()
            (source/'a.py').write_text('import b\n' + '# ' + 'x' * 65536)
            (source/'b.py').write_text('value = 1\n')
            (root/'run').mkdir()
            workspace = Workspace(root/'run'); workspace.prepare(source)
            ticks = iter([0, 0, 0, 3])  # Expire before reading the second chunk of a.py.
            scope = discover_scope(workspace, Profile(name='test'), ['a.py'], clock=lambda: next(ticks))
            self.assertEqual(scope.intended_paths, ['a.py'])
            self.assertEqual(scope.unscanned_paths, ['a.py', 'b.py'])
            self.assertEqual(scope.omitted_paths, [])
            self.assertEqual(scope.limit_reached, 'seconds')
            self.assertIsNone(scope.scan_limit_reached)
            self.assertTrue(scope.unenumerated_remainder)

    def test_optional_scan_timeout_leaves_known_scope_complete(self):
        from coding_agent.workspace import Workspace
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); source = root/'source'; source.mkdir()
            for path in ('a.py', 'b.py', 'c.py'):
                (source/path).write_text('value = 1\n')
            (root/'run').mkdir()
            workspace = Workspace(root/'run'); workspace.prepare(source)
            elapsed = 0
            open_source = Path.open

            def expire_during_optional_read(path, *args, **kwargs):
                nonlocal elapsed
                if path == workspace.root/'b.py':
                    elapsed = 3
                return open_source(path, *args, **kwargs)

            with patch.object(Path, 'open', expire_during_optional_read):
                scope = discover_scope(workspace, Profile(name='test'), ['a.py'], clock=lambda: elapsed)
            self.assertEqual(scope.intended_paths, ['a.py'])
            self.assertEqual(scope.unscanned_paths, ['b.py', 'c.py'])
            self.assertEqual(scope.omitted_paths, [])
            self.assertIsNone(scope.limit_reached)
            self.assertEqual(scope.scan_limit_reached, 'seconds')
            self.assertTrue(scope.unenumerated_remainder)

    def test_revisits_cached_dependents_after_discovering_new_names(self):
        from coding_agent.workspace import Workspace
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); source = root/'source'; source.mkdir()
            (source/'entry.py').write_text('value = 1\n')
            (source/'a_caller.py').write_text('value = exported()\n')
            (source/'b_bridge.py').write_text('import entry\nimport z\n')
            (source/'z.py').write_text('def exported(): return 1\n')
            (root/'run').mkdir()
            workspace = Workspace(root/'run'); workspace.prepare(source)
            scope = discover_scope(workspace, Profile(name='test', discovery_paths=4), ['entry.py'])
            self.assertEqual(scope.intended_paths, ['entry.py', 'b_bridge.py', 'z.py', 'a_caller.py'])
            self.assertEqual(scope.unscanned_paths, [])
            self.assertIsNone(scope.limit_reached)
            self.assertIsNone(scope.scan_limit_reached)
            self.assertFalse(scope.unenumerated_remainder)

    def test_scope_caps_retain_every_known_omitted_target(self):
        from coding_agent.workspace import Workspace
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); source = root/'source'; source.mkdir()
            (source/'a.py').write_text('import b\nimport c\nimport d\n')
            for path in ('b.py', 'c.py', 'd.py'):
                (source/path).write_text('value = 1\n')
            (root/'run').mkdir()
            workspace = Workspace(root/'run'); workspace.prepare(source)
            for profile, reason in ((Profile(name='test', discovery_paths=2), 'paths'),
                                    (Profile(name='test', discovery_edges=1), 'edges')):
                with self.subTest(reason=reason):
                    scope = discover_scope(workspace, profile, ['a.py'])
                    self.assertEqual(scope.intended_paths, ['a.py', 'b.py'])
                    self.assertEqual(scope.omitted_paths, ['c.py', 'd.py'])
                    self.assertEqual(scope.unscanned_paths, ['b.py', 'c.py', 'd.py'])
                    self.assertEqual(scope.limit_reached, reason)
                    self.assertIsNone(scope.scan_limit_reached)
                    self.assertTrue(scope.unenumerated_remainder)

    def test_review_completion_requires_evidence_for_each_intended_path(self):
        """Require valid candidate reads of both files for complete review coverage."""
        for second_read in (None, {'start': 2}, {'baseline': True}, {}):
            with self.subTest(second_read=second_read), tempfile.TemporaryDirectory() as directory:
                root = Path(directory); source = root/'source'; source.mkdir()
                (source/'src').mkdir()
                (source/'src/read.py').write_text('first = 1\n')
                (source/'src/unread.py').write_text('second = 2\n')
                harness = Harness(root/'state', autostart=False)
                rid = harness.submit(TaskRequest(root=str(source), task='Review both files', mode='review',
                                                 focus_paths=['src']))
                actions = [{'type': 'tool', 'tool': 'read_file', 'args': {'path': 'src/read.py'}}]
                if second_read is not None:
                    actions.append({'type': 'tool', 'tool': 'read_file',
                                    'args': {'path': 'src/unread.py', **second_read}})
                actions = iter([*actions, {'type': 'finish', 'summary': 'Finished review'}])

                class Provider:
                    def invoke(self, messages, *, role):
                        """Replay the selected reads and return an empty independent review."""
                        return EMPTY_REVIEW_JSON if role == 'reviewer' else json.dumps(next(actions))

                Engine(harness.store, rid, Provider()).execute()
                review = harness.result(rid).review
                covered = second_read == {}
                self.assertEqual(review.coverage.intended_paths, ['src/read.py', 'src/unread.py'])
                self.assertEqual(review.coverage.examined_paths,
                                 ['src/read.py', 'src/unread.py'] if covered else ['src/read.py'])
                self.assertEqual(review.completion, 'completed' if covered else 'partial')
                self.assertEqual(review.partial_reasons, [] if covered else ['unexamined_scope'])

    def test_independent_reviewer_requires_an_explicit_report(self):
        replies = {
            'schema echo': '{"$defs":{"EvidenceChallenge":{"type":"object"}}}',
            'empty object': '{}',
            'unknown field': json.dumps({**json.loads(EMPTY_REVIEW_JSON), 'unexpected': []}),
            'invalid field value': json.dumps({**json.loads(EMPTY_REVIEW_JSON), 'findings': {}}),
            'invalid nested type': json.dumps({**json.loads(EMPTY_REVIEW_JSON), 'findings': [dict(
                id='value', title='Value', claim='Check value', path='value.py', start_line='1')]}),
            'malformed JSON': '{"findings":',
            'non-object': '[]',
            'explicit empty report': EMPTY_REVIEW_JSON,
        }
        for case, reply in replies.items():
            with self.subTest(case=case), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                source = root / 'source'
                source.mkdir()
                (source / 'value.py').write_text('value = 1\n')
                harness = Harness(root / 'state', autostart=False)
                run_id = harness.submit(TaskRequest(root=str(source), task='Review value', mode='review',
                                                    focus_paths=['value.py']))
                actions = iter([
                    {'type': 'tool', 'tool': 'read_file', 'args': {'path': 'value.py'}},
                    {'type': 'finish', 'summary': 'Finished review'},
                ])

                class Provider:
                    reviewer_calls = 0

                    def invoke(self, messages, *, role):
                        if role == 'reviewer':
                            self.reviewer_calls += 1
                            return reply
                        return json.dumps(next(actions))

                provider = Provider()
                Engine(harness.store, run_id, provider).execute()
                self.assertEqual(harness.status(run_id).state, 'completed')
                self.assertEqual(provider.reviewer_calls, 1)
                result = harness.result(run_id)
                self.assertEqual(result.review.coverage.examined_paths, ['value.py'])
                events = harness.events(run_id)['items']
                failures = [event for event in events if event['kind'] == 'review_protocol_failure']
                accepted = [event for event in events if event['kind'] == 'independent_model_review']
                if case == 'explicit empty report':
                    self.assertEqual(result.review.completion, 'completed')
                    self.assertFalse(failures)
                    self.assertEqual(len(accepted), 1)
                else:
                    self.assertEqual(result.review.completion, 'partial')
                    self.assertEqual(result.review.partial_reasons, ['protocol_failure'])
                    self.assertTrue(any(gap.startswith('Invalid reviewer response:') for gap in result.gaps))
                    self.assertFalse(accepted)
                    self.assertEqual(len(failures), 1)
                    raw = harness.artifact(run_id, failures[0]['data']['artifact_id'])
                    self.assertEqual(raw['content'], reply)

    def test_transitive_discovery_mode_changes_and_cancellation(self):
        from coding_agent.workspace import Workspace
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); source = root/'source'; source.mkdir()
            (source/'z.py').write_text('import a\n')
            (source/'a.py').write_text('import b\n')
            (source/'b.py').write_text('import c\n')
            (source/'c.py').write_text('value = 1\n')
            (root/'run').mkdir()
            workspace = Workspace(root/'run'); workspace.prepare(source)
            (workspace.root/'z.py').chmod(0o755)
            scope = discover_scope(workspace, Profile(name='test'), implementation=True)
            self.assertEqual(set(scope.intended_paths), {'z.py', 'a.py', 'b.py', 'c.py'})
            scope = discover_scope(workspace, Profile(name='test'), cancelled=lambda: True)
            self.assertEqual(scope.limit_reached, 'cancelled')
            self.assertTrue(scope.unenumerated_remainder)

    def test_drafts_survive_protocol_exhaustion_and_cancellation(self):
        """Recover draft findings when review or implementation stops early."""
        for mode, cancel in (('review', False), ('review', True), ('implement', False)):
            with self.subTest(mode=mode, cancel=cancel), tempfile.TemporaryDirectory() as d:
                root = Path(d); source = root/'source'; source.mkdir(); (source/'x.py').write_text('x=1\n')
                harness = Harness(root/'state', autostart=False)
                rid = harness.submit(TaskRequest(root=str(source), task='Investigate x', mode=mode))
                class Provider:
                    calls = 0
                    def invoke(self, messages, *, role):
                        if role == 'reviewer':
                            return EMPTY_REVIEW_JSON
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
                if not cancel:
                    self.assertIn('protocol_failure', result.review.partial_reasons)
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
        no_clean = report({'cases': [{'id': 'bug', 'language': 'python', 'expected': ['x']}]})
        self.assertTrue(no_clean['observed_gates']['zero_unsupported_critical'])
        self.assertFalse(no_clean['observed_gates']['clean_cases_present'])


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
                        if role == 'reviewer': return EMPTY_REVIEW_JSON
                        self.calls += 1
                        if self.calls <= 8:
                            args = {'path': 'target.py', 'start': self.calls, 'lines': 1} if relevant else {'path': f'other{self.calls-1}.py'}
                            return json.dumps({'type': 'tool', 'tool': 'read_file', 'args': args})
                        return json.dumps({'type': 'finish', 'summary': 'Review finished'})
                provider = Provider(); Engine(harness.store, rid, provider).execute()
                review = harness.result(rid).review
                self.assertEqual(review.completion, 'completed' if relevant else 'partial')
                self.assertEqual(provider.calls, 9 if relevant else 6)
