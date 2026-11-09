"""Effective checkout snapshots, baseline-relative diffs, and bounded paths."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import subprocess
import re
from pathlib import Path

EXCLUDED_DIRS = {".git", ".venv", "venv", "node_modules", "__pycache__", ".pytest_cache", ".mypy_cache", "bin", "obj",
                 ".zvec-grep", ".harness", ".cache", "build", "dist", "target", "development-evidence"}
SECRET_NAMES = {".env", ".netrc", ".npmrc", ".pypirc", "credentials.json", "id_rsa", "id_ed25519"}
OUTPUT_SUFFIXES = {".pyc", ".o", ".a", ".so", ".dll", ".exe", ".class"}


def excluded(relative: str) -> bool:
    p = Path(relative)
    return (any(part in EXCLUDED_DIRS or part.startswith(("build-", ".venv-", "venv-")) for part in p.parts)
            or p.name in SECRET_NAMES or p.name.startswith(".env.") and p.name != ".env.example"
            or p.suffix in {".pem", ".key"} or p.suffix in OUTPUT_SUFFIXES)


def checked_path(root: Path, relative: str, *, scope: list[str] | None = None) -> Path:
    p = Path(relative)
    if p.is_absolute() or ".." in p.parts or excluded(relative):
        raise ValueError("path must be a non-secret workspace-relative path")
    lexical = root / p
    resolved = lexical.resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError("path escapes workspace through a symlink")
    if scope is not None and not any(p == Path(s) or Path(s) in p.parents or s == "." for s in scope):
        raise PermissionError("path is outside specialist scope")
    # Never replace/traverse symlinks through editing tools, even in-root ones.
    return lexical


def paths(root: Path):
    for current, dirs, files in os.walk(root, followlinks=False):
        dirs[:] = sorted(d for d in dirs if not excluded(str((Path(current)/d).relative_to(root)))
                         and not (Path(current)/d).is_symlink())
        for name in sorted(files):
            p = Path(current) / name
            relative = p.relative_to(root).as_posix()
            if not excluded(relative) and p.is_file() and not p.is_symlink():
                yield relative, p


def git_paths(root: Path):
    """Yield the effective Git checkout, including nonignored dirty files.

    Git's pathspec machinery is the authority for ignored generated output;
    tracked files remain included even if a repository later ignores them.
    """
    result = safe_git(root, "ls-files", "--cached", "--others", "--exclude-standard", "-z")
    for raw in result.stdout.split(b"\0"):
        if not raw:
            continue
        relative = os.fsdecode(raw)
        if excluded(relative):
            continue
        path = root / relative
        if path.is_file() and not path.is_symlink():
            yield relative, path


def digest(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() and not path.is_symlink() else None


def manifest(root: Path) -> dict:
    reject_source_symlinks(root)
    return {relative: {"sha256": digest(path), "mode": stat.S_IMODE(path.stat().st_mode)}
            for relative, path in paths(root)}


def tree_hash(root: Path) -> str:
    return hashlib.sha256(json.dumps(manifest(root), sort_keys=True).encode()).hexdigest()


def copy_source(source: Path, target: Path, *, source_paths=None):
    target.mkdir(parents=True, exist_ok=True)
    for relative, path in (source_paths or paths(source)):
        if path.stat().st_size > 10 * 1024 * 1024:
            raise ValueError(f"source file exceeds 10 MiB snapshot limit: {relative}")
        destination = target / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)


def safe_git(root: Path, *args, index: Path | None = None, check=True):
    """Git metadata operations may not run project hooks, monitors or filters."""
    env = {k:v for k,v in os.environ.items() if not k.startswith("GIT_")}
    env.update(GIT_TERMINAL_PROMPT="0", GIT_CONFIG_NOSYSTEM="1", GIT_CONFIG_GLOBAL="/dev/null")
    if index:
        env["GIT_INDEX_FILE"] = str(index)
    prefix = ["git", "--literal-pathspecs", "-c", "core.hooksPath=/dev/null", "-c", "core.fsmonitor=false", "-c", "submodule.recurse=false", "-C", str(root)]
    configured = subprocess.run([*prefix, "config", "--name-only", "--get-regexp", r"^filter\..*\.(clean|smudge|process|required)$"],
                                env=env, capture_output=True, text=True)
    filters = {match[1] for key in configured.stdout.splitlines() if (match := re.fullmatch(r"filter\.(.+)\.(?:clean|smudge|process|required)", key))}
    for name in sorted(filters):
        for option, value in (("clean", ""), ("smudge", ""), ("process", ""), ("required", "false")):
            prefix.extend(["-c", f"filter.{name}.{option}={value}"])
    return subprocess.run([*prefix, *args], env=env, capture_output=True, check=check)


def reject_source_symlinks(source: Path):
    for current, dirs, files in os.walk(source, followlinks=False):
        dirs[:] = [d for d in dirs if not excluded(str((Path(current)/d).relative_to(source)))]
        for name in [*dirs, *files]:
            path = Path(current)/name
            if path.is_symlink() and not excluded(str(path.relative_to(source))):
                raise ValueError(f"source symlink requires an explicit materialized copy: {path.relative_to(source)}")


class Workspace:
    def __init__(self, folder: Path):
        self.folder = folder
        self.root = folder / "workspace"
        self.baseline = folder / "baseline"
        self.meta = folder / "workspace.json"

    def git(self, *args, index: Path | None = None, check: bool = True):
        return safe_git(self.root, *args, index=index, check=check)

    def prepare(self, source: Path):
        if self.meta.exists():
            metadata = json.loads(self.meta.read_text())
            if manifest(self.baseline) != metadata["baseline"]:
                raise RuntimeError("baseline integrity check failed")
            return metadata
        if self.root.exists():
            raise RuntimeError("incomplete workspace preparation; retained for inspection")
        reject_source_symlinks(source)
        probe = safe_git(source, "rev-parse", "--show-toplevel", check=False)
        head_probe = safe_git(source, "rev-parse", "--verify", "HEAD", check=False)
        is_git = probe.returncode == 0 and Path(os.fsdecode(probe.stdout).strip()).resolve() == source and head_probe.returncode == 0
        source_paths = list(git_paths(source)) if is_git else list(paths(source))
        initial = {relative: {"sha256": digest(path), "mode": stat.S_IMODE(path.stat().st_mode)}
                   for relative, path in source_paths}
        head = None
        if is_git:
            head = os.fsdecode(head_probe.stdout).strip()
            safe_git(source, "worktree", "add", "--detach", str(self.root), head)
            # Use the effective checkout, not just HEAD. Remove files deleted in
            # the user's working tree and omitted secrets/generated material.
            for relative, path in list(paths(self.root)):
                if relative not in initial:
                    path.unlink()
            for current, dirs, files in os.walk(self.root, topdown=False):
                for name in files:
                    p = Path(current)/name
                    relative = p.relative_to(self.root).as_posix()
                    if name != ".git" and (p.is_symlink() or excluded(relative)):
                        p.unlink()
        else:
            self.root.mkdir()
        copy_source(source, self.root, source_paths=source_paths)
        if {relative: {"sha256": digest(path), "mode": stat.S_IMODE(path.stat().st_mode)}
            for relative, path in (list(git_paths(source)) if is_git else list(paths(source)))} != initial:
            raise RuntimeError("source changed while preparing snapshot")
        if not is_git:
            self.git("init")
        index = self.folder / "baseline.index"
        self.git("-c", "core.attributesFile=/dev/null", "add", "--force", "--all", index=index)
        copy_source(self.root, self.baseline)
        metadata = {"source": str(source), "head": head, "git_worktree": is_git, "baseline": manifest(self.baseline)}
        self.meta.write_text(json.dumps(metadata, indent=2))
        return metadata

    def changed_paths(self) -> list[str]:
        baseline = json.loads(self.meta.read_text())["baseline"]
        candidate = manifest(self.root)
        return sorted(path for path in baseline.keys() | candidate.keys()
                      if baseline.get(path) != candidate.get(path))

    def patch(self, scope: list[str] | None = None) -> str:
        index = self.folder / "diff.index"
        shutil.copy2(self.folder / "baseline.index", index)
        # Stage only intent for new selected source files. Do not stage deletions
        # (that would hide them from the index/worktree diff) or build outputs.
        selected = [relative for relative, _ in paths(self.root)]
        for offset in range(0, len(selected), 200):
            self.git("add", "--force", "--intent-to-add", "--", *selected[offset:offset+200], index=index)
        extra = ["--", *scope] if scope else []
        return self.git("diff", "--binary", "--full-index", "--no-ext-diff", "--no-textconv", *extra, index=index).stdout.decode("utf-8", errors="replace")

    def baseline_check_root(self) -> Path:
        root = self.folder / "baseline-check"
        if not root.exists():
            copy_source(self.baseline, root)
        return root

    def assert_original_tests(self) -> list[str]:
        baseline = json.loads(self.meta.read_text())["baseline"]
        changes = []
        for relative, info in baseline.items():
            p = Path(relative)
            if "tests" in p.parts or "test" in p.parts or p.name.startswith(("test_", "smoke_")) or p.name in {"CMakeLists.txt", "pyproject.toml", "pytest.ini", "setup.cfg"}:
                candidate = self.root / relative
                if digest(candidate) != info["sha256"] or candidate.is_file() and stat.S_IMODE(candidate.stat().st_mode) != info["mode"]:
                    changes.append(relative)
        return changes
