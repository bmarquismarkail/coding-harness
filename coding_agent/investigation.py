"""Bounded dependency discovery and snapshot-aware progress accounting."""
import re
import time
from pathlib import Path
from .review_types import Coverage
from .workspace import paths


def discover_scope(workspace, profile, focus_paths=(), *, implementation=False,
                   clock=time.monotonic, cancelled=lambda: False):
    import hashlib
    started = clock()
    used = 0
    limit = None
    omitted = []
    snapshots = []
    known = set()

    def stop():
        nonlocal limit
        if limit:
            return True
        if cancelled():
            limit = 'cancelled'
        elif clock() - started >= profile.discovery_seconds:
            limit = 'seconds'
        return limit is not None

    # Bound preparation as well as navigation. Hash in chunks so cancellation
    # and the clock are checked even within large files.
    for root in (workspace.baseline, workspace.root):
        snapshot = {}
        iterator = iter(paths(root))
        while not stop():
            item = next(iterator, None)
            if item is None:
                break
            path, source = item
            if path not in known and len(known) >= profile.discovery_paths:
                omitted.append(path)
                limit = 'paths'
                break
            known.add(path)
            size = source.stat().st_size
            if used + size > profile.discovery_bytes:
                omitted.append(path)
                limit = 'bytes'
                break
            raw = bytearray()
            hasher = hashlib.sha256()
            with source.open('rb') as stream:
                while not stop():
                    chunk = stream.read(min(65536, profile.discovery_bytes - used + 1))
                    if not chunk:
                        break
                    used += len(chunk)
                    if used > profile.discovery_bytes:
                        limit = 'bytes'
                        break
                    hasher.update(chunk)
                    raw.extend(chunk)
            if limit:
                omitted.append(path)
                break
            snapshot[path] = (hasher.hexdigest(), source.stat().st_mode & 0o777,
                              raw.decode(errors='replace'))
        snapshots.append(snapshot)
        if limit:
            break
    baseline = snapshots[0]
    candidate = snapshots[1] if len(snapshots) > 1 else {}
    available = {**baseline, **candidate}
    if implementation:
        seeds = sorted(p for p in available if baseline.get(p, (None, None))[:2]
                       != candidate.get(p, (None, None))[:2])
    elif focus_paths:
        seeds = sorted(p for p in available if any(p == f or f == '.' or p.startswith(f.rstrip('/') + '/') for f in focus_paths))
    else:
        seeds = sorted(available)
    seeds += [p for p in focus_paths if p not in seeds and p not in available]
    included = list(dict.fromkeys(seeds[:profile.discovery_paths]))
    omitted.extend(seeds[profile.discovery_paths:])
    if len(seeds) > profile.discovery_paths and not limit:
        limit = 'paths'
    names = {Path(p).stem for p in included}
    edges = set()
    # Revisit cached source when new declarations or modules expand the scope.
    changed = True
    while changed and not stop():
        changed = False
        ordered = sorted(available.items(), key=lambda item: (item[0] not in included, item[0]))
        for path, (_, _, text) in ordered:
            if stop():
                break
            tokens = set(re.findall(r'[A-Za-z_][A-Za-z_0-9]*', text))
            links = re.findall(r'(?:#\s*include\s*["<]([^">]+)|(?:from|import)\s+([\w.]+))', text)
            related = []
            if path in included:
                declarations = re.findall(r'\bdef\s+([A-Za-z_]\w*)|\b[A-Za-z_]\w*[\s*&]+([A-Za-z_]\w*)\s*\(', text)
                new_names = {name for declaration in declarations for name in declaration if name}
                if not new_names <= names:
                    names.update(new_names)
                    changed = True
                for include, module in links:
                    target = include or module.replace('.', '/') + '.py'
                    for possible in (str(Path(path).parent / target), target):
                        if possible in available:
                            related.append(possible)
            if tokens & names or Path(path).name in {'CMakeLists.txt', 'Makefile', 'pyproject.toml'}:
                related.append(path)
            for target in related:
                edge = (path, target)
                if edge in edges:
                    continue
                if len(edges) >= profile.discovery_edges:
                    omitted.extend(t for t in related if t not in included)
                    limit = 'edges'
                    break
                edges.add(edge)
                if target not in included:
                    if len(included) >= profile.discovery_paths:
                        omitted.append(target)
                        limit = 'paths'
                        break
                    included.append(target)
                    names.add(Path(target).stem)
                    changed = True
    return Coverage(intended_paths=included, omitted_paths=sorted(set(omitted)),
                    limit_reached=limit, unenumerated_remainder=bool(limit))


def evidence_progress(previous, observation, scope):
    """Count new covered intervals, never response text or repeated check executions."""
    ledger = dict(previous)
    result = observation.get('result', {})
    progress = False
    if result.get('kind') == 'source' and result.get('path') in scope and not result.get('truncated_line'):
        key = f"source:{result.get('baseline', False)}:{result['path']}:{result['sha256']}"
        intervals = ledger.get(key, [])
        start, end = result['start'], result['end']
        cursor = start
        for left, right in sorted(intervals):
            if left > cursor:
                break
            cursor = max(cursor, right + 1)
        progress = end >= start and cursor <= end
        merged = []
        for left, right in sorted([*intervals, [start, end]]):
            if merged and left <= merged[-1][1] + 1:
                merged[-1][1] = max(merged[-1][1], right)
            else:
                merged.append([left, right])
        ledger[key] = merged
    if result.get('check_id') and result.get('artifact_id') and not result.get('skipped'):
        key = 'check:' + ':'.join(str(result.get(k)) for k in ('check_id', 'phase', 'source_hash', 'exit_code'))
        progress = key not in ledger
        ledger[key] = True
    return ledger, progress
