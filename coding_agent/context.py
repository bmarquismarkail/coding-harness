from __future__ import annotations

import os
from pathlib import Path

from .config import ContextSettings


EXCLUDED_DIRS = {
    ".git", ".hg", ".svn", ".venv", "venv", "node_modules", "__pycache__",
    ".mypy_cache", ".pytest_cache", "build", "dist", "target", ".tox",
}
IMPORTANT_NAMES = {
    "AGENTS.md", "README.md", "README.rst", "pyproject.toml", "setup.cfg",
    "CMakeLists.txt", "Makefile", "Cargo.toml", "compile_commands.json",
    "Dockerfile", "BUILD", "WORKSPACE",
}
TEXT_SUFFIXES = {
    ".py", ".pyi", ".c", ".h", ".cc", ".cpp", ".cxx", ".hh", ".hpp", ".hxx",
    ".rs", ".go", ".java", ".js", ".ts", ".tsx", ".cs", ".swift", ".zig",
    ".rb", ".php", ".lua", ".pl", ".pm", ".ex", ".exs", ".kt", ".kts", ".scala",
    ".dart", ".r", ".m", ".mm", ".asm", ".s", ".sh", ".fish", ".sql", ".proto",
    ".graphql", ".vue", ".svelte", ".html", ".css", ".scss", ".xml", ".gradle",
    ".nix", ".jl", ".hs", ".fs", ".fsx", ".erl", ".hrl", ".clj", ".cljs",
    ".md", ".rst", ".toml", ".yaml", ".yml", ".json", ".cmake", ".txt",
}


def _language(path: Path) -> str:
    suffix = path.suffix.lower()
    return {
        ".py": "Python", ".pyi": "Python", ".c": "C", ".h": "C/C++",
        ".cc": "C++", ".cpp": "C++", ".cxx": "C++", ".hh": "C++", ".hpp": "C++",
        ".rs": "Rust", ".go": "Go", ".java": "Java", ".js": "JavaScript",
        ".ts": "TypeScript", ".tsx": "TypeScript", ".cs": "C#", ".swift": "Swift",
        ".zig": "Zig",
    }.get(suffix, "configuration/documentation")


def build_context(root: str | Path, settings: ContextSettings) -> str:
    """Build bounded, deterministic repository context without executing repository code."""
    root_path = Path(root).expanduser().resolve()
    if not root_path.is_dir():
        raise ValueError(f"repository root is not a directory: {root_path}")
    candidates: list[Path] = []
    for current, dirs, files in os.walk(root_path, followlinks=False):
        dirs[:] = sorted(d for d in dirs if d not in EXCLUDED_DIRS and not d.startswith("."))
        for filename in sorted(files):
            path = Path(current) / filename
            if path.is_symlink():
                continue
            if filename in IMPORTANT_NAMES or path.suffix.lower() in TEXT_SUFFIXES:
                candidates.append(path)
    candidates.sort(key=lambda p: (p.name not in IMPORTANT_NAMES, str(p.relative_to(root_path))))
    chunks: list[str] = [f"REPOSITORY ROOT: {root_path}"]
    total = 0
    included = 0
    for path in candidates:
        if included >= settings.max_files or total >= settings.max_total_bytes:
            break
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        text = text[: settings.max_bytes_per_file]
        relative = path.relative_to(root_path)
        block = f"\n--- {relative} [{_language(path)}] ---\n{text}"
        if total + len(block.encode("utf-8")) > settings.max_total_bytes:
            break
        chunks.append(block)
        total += len(block.encode("utf-8"))
        included += 1
    if not candidates:
        chunks.append("\nNo supported text files were found; treat the repository as empty until proven otherwise.")
    else:
        chunks.insert(1, f"FILE COUNT CONSIDERED: {len(candidates)}; FILES INCLUDED: {included}")
    return "".join(chunks)
