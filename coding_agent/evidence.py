"""Immutable evidence, bounded presentation, and mechanical adjudication."""
import hashlib
import json
from .review_types import (ImmutableEvidenceBundle, ReviewerEvidenceView, ReviewerResponse,
                           FindingProposal, AdjudicatedFinding, ReviewDetails, Coverage)
from .workspace import tree_hash, checked_path, digest


def read_json(store, run_id, artifact_id):
    # Never resolve an artifact outside this run, even if another run owns identical bytes.
    if len(artifact_id) != 64 or any(c not in '0123456789abcdef' for c in artifact_id):
        raise ValueError('invalid artifact reference')
    raw = (store.run_dir(run_id) / 'artifacts' / artifact_id).read_bytes()
    if hashlib.sha256(raw).hexdigest() != artifact_id:
        raise ValueError('artifact hash mismatch')
    return json.loads(raw)


def freeze_bundle(engine, coverage=None):
    store, rid, workspace = engine.store, engine.run_id, engine.workspace
    checks, sources, probes, gaps = [], [], [], []
    for receipt in store.receipts(rid, 'check'):
        data = receipt['data']
        item = {**data, 'receipt_state': receipt['state'], 'receipt_key': receipt['key']}
        if data.get('artifact_id'):
            try:
                actual = read_json(store, rid, data['artifact_id'])
                if actual != {k: v for k, v in data.items() if k != 'artifact_id'}:
                    raise ValueError('receipt differs from retained bytes')
            except (ValueError, OSError) as exc:
                item['integrity_error'] = str(exc)
                gaps.append(f"Invalid check evidence: {receipt['key']}")
        elif receipt['state'] == 'completed' and not data.get('skipped'):
            item['integrity_error'] = 'missing execution artifact'
            gaps.append(f"Missing check artifact: {receipt['key']}")
        checks.append(item)
    with store.db() as db:
        events = db.execute("SELECT data FROM events WHERE run_id=? AND kind='source' ORDER BY seq", (rid,)).fetchall()
    for artifact_id in dict.fromkeys(json.loads(e['data'])['artifact_id'] for e in events):
        try:
            source = read_json(store, rid, artifact_id)
            phase = 'baseline' if source.get('baseline') else 'candidate'
            root = workspace.baseline if phase == 'baseline' else workspace.root
            valid = digest(checked_path(root, source['path'])) == source['sha256']
            sources.append({'artifact_id': artifact_id, 'phase': phase, 'valid': valid, **source})
        except (ValueError, OSError, KeyError) as exc:
            gaps.append(f'Invalid source evidence {artifact_id}: {exc}')
    for receipt in store.receipts(rid, 'probe_definition'):
        artifact_id = receipt['data']['artifact_id']
        probes.append({'artifact_id': artifact_id, **read_json(store, rid, artifact_id)})
    bundle = ImmutableEvidenceBundle(run_id=rid, revision=engine.revision,
        baseline_hash=tree_hash(workspace.baseline_check_root()), candidate_hash=tree_hash(workspace.root),
        patch_artifact_id=store.artifact(rid, workspace.patch()), definitions=engine.checks.definitions(),
        checks=checks, source_evidence=sources, defect_probes=probes,
        coverage=coverage or Coverage(), integrity_gaps=gaps)
    bundle.coverage.examined_paths = sorted({s["path"] for s in sources if s["valid"] and s["phase"] == "candidate"})
    artifact_id = store.artifact(rid, bundle.model_dump())
    store.event(rid, 'evidence_bundle', {'artifact_id': artifact_id})
    return bundle, artifact_id


def reviewer_view(store, bundle, bundle_id, budget=6000):
    view = ReviewerEvidenceView(bundle_id=bundle_id, run_id=bundle.run_id)
    # Keep complete references in the bundle; each projection explicitly records omissions.
    for category, items in [('checks', bundle.checks), ('defect_probes', bundle.defect_probes),
                            ('sources', bundle.source_evidence)]:
        for item in items:
            projected = {k: v for k, v in item.items() if k not in {'output', 'wrapper'}}
            target = getattr(view, category)
            target.append(projected)
            if len(view.model_dump_json().encode()) > budget:
                target.pop()
                view.omitted_references.append(item.get('artifact_id', item.get('receipt_key', 'unknown')))
    omitted = list(view.omitted_references)
    view.omitted_references = []
    # The manifest is authoritative; inline references are optional excerpts.
    while True:
        view.omitted_count = len(omitted)
        view.omitted_manifest_id = store.artifact(bundle.run_id, omitted) if omitted else None
        if len(view.model_dump_json().encode()) <= budget:
            break
        category = next((c for c in ('sources', 'defect_probes', 'checks') if getattr(view, c)), None)
        if category is None:
            raise ValueError('budget cannot hold evidence view metadata')
        item = getattr(view, category).pop()
        omitted.append(item.get('artifact_id', item.get('receipt_key', 'unknown')))
    artifact = store.artifact(bundle.run_id, view.model_dump())
    store.event(bundle.run_id, 'reviewer_view', {'artifact_id': artifact, 'bundle_id': bundle_id})
    return view


def current_checks(bundle, check_id, phase):
    expected = bundle.baseline_hash if phase == 'baseline' else bundle.candidate_hash
    return sorted([c for c in bundle.checks if c.get('check_id') == check_id and c.get('phase') == phase
            and c.get('source_hash') == expected and c.get('receipt_state') == 'completed'
            and c.get('artifact_id') and not c.get('integrity_error')], key=lambda c: c.get('executed_at', 0))


def normalize_response(raw):
    """Accept typed responses and legacy reports without silently dropping concerns."""
    if raw.get('review') is not None:
        response = ReviewerResponse.model_validate(raw['review'])
        response.semantic_concerns = list(dict.fromkeys([*response.semantic_concerns, *raw.get('gaps', [])]))
        return response
    if any(k in raw for k in ('challenges', 'observations', 'semantic_concerns')):
        response = ReviewerResponse.model_validate({k: value for k, value in raw.items() if k != 'gaps'})
        response.semantic_concerns = list(dict.fromkeys([*response.semantic_concerns, *raw.get('gaps', [])]))
        return response
    # Legacy empty reports remain valid. Unstructured concerns are retained conservatively.
    return ReviewerResponse(findings=[FindingProposal.model_validate(f) for f in raw.get('findings', [])],
                            semantic_concerns=raw.get('gaps', []))


def adjudicate_response(store, bundle, bundle_id, raw):
    stored = read_json(store, bundle.run_id, bundle_id)
    if stored != bundle.model_dump():
        raise ValueError("adjudication bundle differs from immutable artifact")
    with store.db() as db:
        registered = db.execute("SELECT data FROM events WHERE run_id=? AND kind='evidence_bundle'", (bundle.run_id,)).fetchall()
    if not any(json.loads(row['data']).get('artifact_id') == bundle_id for row in registered):
        raise ValueError("bundle is not registered to this run")
    proposal_id = store.artifact(bundle.run_id, raw)
    response = normalize_response(raw)
    gaps = list(response.semantic_concerns)
    decisions = []
    for challenge in response.challenges:
        records = current_checks(bundle, challenge.check_id, challenge.phase)
        if challenge.kind == 'check_missing':
            supported = not records
        elif challenge.kind == 'check_passed':
            supported = bool(records and records[-1].get('passed'))
        else:
            supported = any(s['artifact_id'] == challenge.evidence_id and s['valid']
                            and s['phase'] == challenge.phase for s in bundle.source_evidence)
        if challenge.kind == 'source_examined':
            try:
                view = ReviewerEvidenceView.model_validate(read_json(store, bundle.run_id, challenge.view_id or ''))
                with store.db() as db:
                    views = db.execute("SELECT data FROM events WHERE run_id=? AND kind='reviewer_view'", (bundle.run_id,)).fetchall()
                registered_view = any(json.loads(row['data']).get('artifact_id') == challenge.view_id for row in views)
                supported = supported and registered_view and view.run_id == bundle.run_id and any(
                    source['artifact_id'] == challenge.evidence_id for source in view.sources)
            except (ValueError, OSError):
                supported = False
        decisions.append({'challenge': challenge.model_dump(), 'supported': supported})
        if challenge.kind == 'check_missing' and supported:
            gaps.append(f'Required evidence missing: {challenge.phase}/{challenge.check_id}')
        if challenge.kind != 'check_missing' and not supported:
            gaps.append(f'Unsupported evidence assertion: {challenge.kind}')
    findings = []
    for proposal in response.findings:
        status, reason, observation = 'hypothesis', 'Source reasoning remains a model proposal.', None
        citations = [s for s in bundle.source_evidence if s['artifact_id'] in proposal.evidence_ids]
        cursor = proposal.start_line
        for source in sorted(citations, key=lambda s: s['start']):
            if source['start'] > cursor:
                break
            cursor = max(cursor, source['end'] + 1)
        if (citations and cursor <= proposal.end_line) or len(citations) != len(set(proposal.evidence_ids)) or any(
                not s['valid'] or s['phase'] != proposal.phase or s['path'] != proposal.path
                or s.get('truncated_line')
                for s in citations):
            status, reason = 'needs_human_review', 'Invalid or wrong-snapshot source citation.'
        probe = next((p for p in bundle.defect_probes if p['defect_id'] == proposal.defect_id), None)
        if proposal.defect_id and probe is None:
            reason = 'No trusted probe definition supports this observation.'
        if probe:
            records = current_checks(bundle, probe['check_id'], probe['phase'])
            record = records[-1] if records else {}
            expected = bundle.baseline_hash if probe['phase'] == 'baseline' else bundle.candidate_hash
            valid = (probe['source_hash'] == expected and proposal.phase == probe['phase']
                     and proposal.path == probe['path'] and proposal.start_line == probe['start_line']
                     and proposal.end_line == probe['end_line']
                     and record.get('probe_definition_id') == probe['artifact_id']
                     and record.get('probe_target_valid') is True
                     and all(bool(current_checks(bundle, prerequisite, probe['phase']))
                             and current_checks(bundle, prerequisite, probe['phase'])[-1].get('passed')
                             for prerequisite in probe.get('check_spec', {}).get('requires', []))
                     and not any(record.get(k) for k in ('timed_out', 'cancelled', 'skipped', 'source_modified', 'output_truncated')))
            if not valid or record.get('exit_code') not in probe['observed_exit_codes'] + probe['not_observed_exit_codes']:
                status, reason = 'needs_human_review', 'Trusted probe execution or target is inconclusive.'
            elif record['exit_code'] in probe['observed_exit_codes'] and status != 'needs_human_review':
                status, reason, observation = 'reproduced', 'Only the registered observation was reproduced.', probe['observation']
        resolved = False
        if probe and proposal.phase == 'baseline' and status == 'reproduced':
            candidates = current_checks(bundle, probe['check_id'], 'candidate')
            latest = candidates[-1] if candidates else {}
            # Only a clean execution of the same trusted oracle on the final
            # candidate resolves a reproduced baseline observation.
            resolved = (latest.get('probe_definition_id') == probe['artifact_id']
                        and latest.get('passed') is True
                        and latest.get('probe_range_valid') is True
                        and latest.get('exit_code') in probe['not_observed_exit_codes']
                        and not any(latest.get(k) for k in ('timed_out', 'cancelled', 'skipped',
                                                          'source_modified', 'output_truncated')))
        findings.append(AdjudicatedFinding(proposal=proposal, status=status, reason=reason,
                                           observation=observation, resolved=resolved, proposal_artifact=proposal_id))
    adjudication_id = store.artifact(bundle.run_id, {'bundle_id': bundle_id, 'proposal_id': proposal_id,
        'decisions': decisions, 'findings': [f.model_dump() for f in findings], 'gaps': gaps})
    store.event(bundle.run_id, 'adjudication', {'artifact_id': adjudication_id})
    return findings, gaps, adjudication_id
