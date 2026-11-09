"""Bounded dependency discovery and snapshot-aware progress accounting."""
import re
import time
from pathlib import Path
from .review_types import Coverage
from .workspace import checked_path, paths


def discover_scope(workspace, profile, focus_paths=(), *, implementation=False,
                   clock=time.monotonic, cancelled=lambda: False):
    """Expand patch and focus paths into a scope bounded by the discovery profile."""
    if cancelled():
        return Coverage(limit_reached='cancelled', unenumerated_remainder=True)

    # Establish scope from the complete path inventory and the exact patch
    # before applying budgets to lexical dependency discovery.
    available = dict(paths(workspace.root))
    patch_paths = workspace.changed_paths() if implementation else []
    for path in patch_paths:
        available.setdefault(path, workspace.baseline / path)
    focused = []
    for focus in focus_paths:
        try:
            focus = checked_path(workspace.root, focus).relative_to(workspace.root).as_posix()
        except ValueError:
            # Older queued requests may contain paths that now fail intake validation.
            continue
        matches = sorted(path for path in available
                         if path == focus or focus == '.' or path.startswith(focus + '/'))
        focused.extend(matches or [focus])
    seeds = list(dict.fromkeys([*patch_paths, *focused]))
    if not implementation and not focus_paths:
        seeds = sorted(available)

    included = seeds[:profile.discovery_paths]
    omitted = seeds[profile.discovery_paths:]
    limit = 'paths' if omitted else None
    started = clock()

    def stop():
        nonlocal limit
        if limit:
            return True
        if cancelled():
            limit = 'cancelled'
        elif clock() - started >= profile.discovery_seconds:
            limit = 'seconds'
        return limit is not None

    used = 0
    sources = {}
    ordered = sorted(available, key=lambda path: (path not in included, path))
    for path in ordered[:profile.discovery_paths]:
        if stop():
            break
        source = available[path]
        if used + source.stat().st_size > profile.discovery_bytes:
            limit = 'bytes'
            break
        raw = bytearray()
        with source.open('rb') as stream:
            while not stop():
                chunk = stream.read(min(65536, profile.discovery_bytes - used + 1))
                if not chunk:
                    break
                used += len(chunk)
                if used > profile.discovery_bytes:
                    limit = 'bytes'
                    break
                raw.extend(chunk)
        if limit:
            break
        sources[path] = raw.decode(errors='replace')

    names = {Path(p).stem for p in included}
    edges = set()
    # Revisit cached source when new declarations or modules expand the scope.
    changed = True
    while changed and not stop():
        changed = False
        ordered_sources = sorted(sources, key=lambda path: (path not in included, path))
        for path in ordered_sources:
            if stop():
                break
            text = sources[path]
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
    if not limit and len(available) > profile.discovery_paths:
        limit = 'paths'
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
