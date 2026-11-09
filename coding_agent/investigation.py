"""Bounded dependency discovery and snapshot-aware progress accounting."""
import re
import time
from pathlib import Path
from .review_types import Coverage
from .workspace import paths, digest


def discover_scope(workspace, profile, focus_paths=(), *, implementation=False, clock=time.monotonic):
    baseline = dict(paths(workspace.baseline))
    candidate = dict(paths(workspace.root))
    available = {**baseline, **candidate}
    if implementation:
        seeds = sorted(p for p in available if digest(baseline.get(p, Path('/nonexistent')))
                       != digest(candidate.get(p, Path('/nonexistent'))))
    elif focus_paths:
        seeds = sorted(p for p in available if any(p == f or f == '.' or p.startswith(f.rstrip('/') + '/') for f in focus_paths))
    else:
        seeds = sorted(available)
    seeds += [p for p in focus_paths if p not in seeds and p not in available]
    included = list(dict.fromkeys(seeds[:profile.discovery_paths]))
    omitted = list(dict.fromkeys(seeds[profile.discovery_paths:]))
    limit = 'paths' if omitted else None
    started = clock()
    used, edges = 0, 0
    unenumerated = bool(limit)
    # Discover imports/includes and references to scoped file stems, including reverse callers.
    names = {Path(p).stem for p in included}
    ordered = sorted(available.items(), key=lambda item: (item[0] not in included, item[0]))
    for path, source in ordered:
        if limit:
            break
        if clock() - started >= profile.discovery_seconds:
            limit = 'seconds'
            unenumerated = True
            break
        size = source.stat().st_size
        if used + size > profile.discovery_bytes:
            limit = 'bytes'
            unenumerated = True
            break
        used += size
        text = source.read_text(errors='replace')
        tokens = set(re.findall(r'[A-Za-z_][A-Za-z_0-9]*', text))
        links = re.findall(r'(?:#\s*include\s*["<]([^">]+)|(?:from|import)\s+([\w.]+))', text)
        related = []
        if path in included:
            declarations = re.findall(r'\bdef\s+([A-Za-z_]\w*)|\b[A-Za-z_]\w*[\s*&]+([A-Za-z_]\w*)\s*\(', text)
            names.update(name for declaration in declarations for name in declaration if name)
            for include, module in links:
                target = include or module.replace('.', '/') + '.py'
                for possible in (str(Path(path).parent / target), target):
                    if possible in available:
                        related.append(possible)
        if tokens & names or (Path(path).name in {'CMakeLists.txt', 'Makefile', 'pyproject.toml'}):
            related.append(path)
        for target in related:
            if edges >= profile.discovery_edges:
                limit = 'edges'
                unenumerated = True
                break
            edges += 1
            if target not in included:
                if len(included) >= profile.discovery_paths:
                    omitted.append(target)
                    limit = 'paths'
                    unenumerated = True
                    break
                included.append(target)
                names.add(Path(target).stem)
    return Coverage(intended_paths=included, omitted_paths=sorted(set(omitted)),
                    limit_reached=limit, unenumerated_remainder=unenumerated)


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
