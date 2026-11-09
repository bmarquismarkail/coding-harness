"""Project-aware check defaults and immutable, read-only skill snapshots."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from .types import CheckSpec, TaskRequest

_ID = re.compile(r"^[a-z0-9][a-z0-9-]{0,62}$")
_MAX_SKILL_BYTES = 40_000
_MAX_TOTAL_SKILL_BYTES = 120_000


def load_definitions(path: str | Path | None = None) -> dict[str, dict]:
    selected = Path(path) if path else Path(__file__).resolve().parent.parent / "project-profiles.json"
    if not selected.is_file():
        return {}
    raw = json.loads(selected.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or raw.get("version") != 1 or not isinstance(raw.get("projects"), dict):
        raise ValueError("project profile registry must have version=1 and a projects object")
    return raw["projects"]


def _skill_dirs(project_root: Path):
    yield project_root / ".agents" / "skills"
    yield project_root / ".codex" / "skills"
    yield Path.home() / ".agents" / "skills"
    yield Path.home() / ".codex" / "skills"


def load_skill(skill_id: str, project_root: Path) -> dict:
    if not _ID.fullmatch(skill_id):
        raise ValueError(f"invalid skill ID: {skill_id!r}")
    for parent in _skill_dirs(project_root):
        directory = parent / skill_id
        entry = directory / "SKILL.md"
        if not entry.is_file():
            continue
        resolved = entry.resolve()
        if not resolved.is_relative_to(directory.resolve()):
            raise ValueError(f"skill entry escapes its directory: {skill_id}")
        raw = resolved.read_bytes()
        if len(raw) > _MAX_SKILL_BYTES:
            raise ValueError(f"skill entry exceeds {_MAX_SKILL_BYTES} bytes: {skill_id}")
        return {"id": skill_id, "source": str(resolved), "sha256": hashlib.sha256(raw).hexdigest(),
                "version": hashlib.sha256(raw).hexdigest()[:16], "content": raw.decode("utf-8")}
    raise ValueError(f"skill not found: {skill_id}")


def resolve_request(request: TaskRequest, root: Path, definitions: dict[str, dict]):
    selected = request.project_profile
    if selected is not None and selected not in definitions:
        raise ValueError(f"unknown project profile: {selected}")
    if selected is None:
        candidates = []
        for name, definition in definitions.items():
            for candidate in definition.get("roots", []):
                path = Path(candidate).expanduser().resolve()
                if root == path or root.is_relative_to(path):
                    candidates.append((len(path.parts), name))
        if candidates:
            selected = max(candidates)[1]
    definition = definitions.get(selected, {}) if selected else {}
    if selected and not any(root == Path(p).expanduser().resolve() or root.is_relative_to(Path(p).expanduser().resolve())
                             for p in definition.get("roots", [])):
        raise ValueError(f"project root is outside profile {selected!r}")

    checks = list(request.checks or [])
    if request.checks is None:
        checks = [CheckSpec.model_validate(value) for value in definition.get("checks", [])]
    skill_ids = request.skill_ids if request.skill_ids is not None else definition.get("skills", [])
    if not isinstance(skill_ids, list) or len(skill_ids) > 20 or len(set(skill_ids)) != len(skill_ids):
        raise ValueError("skill_ids must contain up to 20 unique skill names")
    skills = [load_skill(skill_id, root) for skill_id in skill_ids]
    if sum(len(item["content"].encode("utf-8")) for item in skills) > _MAX_TOTAL_SKILL_BYTES:
        raise ValueError("selected skills exceed the combined context limit")
    resolved = request.model_copy(update={"root": str(root), "project_profile": selected,
                                         "skill_ids": list(skill_ids), "checks": checks})
    return resolved, skills


def skill_context(skills: list[dict]) -> str:
    if not skills:
        return "No project skills were selected."
    blocks = []
    for skill in skills:
        blocks.append(f"## {skill['id']} (snapshot {skill['version']}; sha256 {skill['sha256']})\n"
                      f"Source: {skill['source']}\n\n{skill['content']}")
    return "\n\n---\n\n".join(blocks)


def skill_metadata(skills: list[dict]) -> list[dict]:
    return [{key: value for key, value in skill.items() if key != "content"} for skill in skills]
