"""Bounded dependency discovery and snapshot-aware progress accounting."""
import re
import time
from collections import deque
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

    included = dict.fromkeys(seeds[:profile.discovery_paths])
    omitted = set(seeds[profile.discovery_paths:])
    limit = 'paths' if omitted else None
    scan_limit = None
    started = clock()

    def interrupted():
        if cancelled():
            return 'cancelled'
        if clock() - started >= profile.discovery_seconds:
            return 'seconds'
        return None

    used = 0
    sources = {}

    def scan(path):
        nonlocal used
        if reason := interrupted():
            return reason
        if len(sources) >= profile.discovery_paths:
            return 'paths'
        source = available[path]
        if used + source.stat().st_size > profile.discovery_bytes:
            return 'bytes'
        raw = bytearray()
        with source.open('rb') as stream:
            while True:
                if reason := interrupted():
                    return reason
                chunk = stream.read(min(65536, profile.discovery_bytes - used + 1))
                if not chunk:
                    break
                used += len(chunk)
                if used > profile.discovery_bytes:
                    return 'bytes'
                raw.extend(chunk)
        text = raw.decode(errors='replace')
        tokens = set(re.findall(r'[A-Za-z_][A-Za-z_0-9]*', text))
        if reason := interrupted():
            return reason
        sources[path] = text, tokens
        return None

    names = {Path(path).stem for path in included}
    edges = set()
    pending = deque(path for path in included if path in available)
    background = iter(sorted(available))
    unmatched = {}
    revisit = False

    def add_names(discovered):
        nonlocal revisit
        if not discovered <= names:
            names.update(discovered)
            revisit = True

    def expand(path, related):
        reason = interrupted()
        for target in related:
            if reason:
                break
            edge = (path, target)
            if edge in edges:
                continue
            if len(edges) >= profile.discovery_edges:
                reason = 'edges'
                break
            if target not in included:
                if len(included) >= profile.discovery_paths:
                    reason = 'paths'
                    break
                included[target] = None
                pending.append(target)
                add_names({Path(target).stem})
            edges.add(edge)
        if reason:
            omitted.update(target for target in related if target not in included)
        return reason

    def matches(path, tokens):
        return bool(tokens & names) or Path(path).name in {'CMakeLists.txt', 'Makefile', 'pyproject.toml'}

    while not (limit or scan_limit):
        if pending:
            path = pending.popleft()
            if path not in sources:
                limit = scan(path)
                if limit:
                    break
            text, tokens = sources[path]
            declarations = re.findall(r'\bdef\s+([A-Za-z_]\w*)|\b[A-Za-z_]\w*[\s*&]+([A-Za-z_]\w*)\s*\(', text)
            add_names({name for declaration in declarations for name in declaration if name})
            related = []
            for include, module in re.findall(r'(?:#\s*include\s*["<]([^">]+)|(?:from|import)\s+([\w.]+))', text):
                target = include or module.replace('.', '/') + '.py'
                for possible in (str(Path(path).parent / target), target):
                    if possible in available:
                        related.append(possible)
            if matches(path, tokens):
                related.append(path)
                unmatched.pop(path, None)
            else:
                unmatched[path] = tokens
            limit = expand(path, related)
        elif revisit:
            # Newly discovered names can turn previously scanned files into dependents.
            revisit = False
            for path, tokens in list(unmatched.items()):
                if limit := interrupted():
                    break
                if matches(path, tokens):
                    del unmatched[path]
                    if limit := expand(path, [path]):
                        break
        else:
            path = next(background, None)
            if path is None:
                break
            if path in sources:
                continue
            reason = scan(path)
            if reason == 'cancelled':
                limit = reason
            elif reason:
                scan_limit = reason
            elif matches(path, sources[path][1]):
                limit = expand(path, [path])
            else:
                unmatched[path] = sources[path][1]

    return Coverage(intended_paths=list(included), omitted_paths=sorted(omitted),
                    unscanned_paths=sorted(available.keys() - sources.keys()), limit_reached=limit,
                    scan_limit_reached=scan_limit, unenumerated_remainder=bool(limit or scan_limit))


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
