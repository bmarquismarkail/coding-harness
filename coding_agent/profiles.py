from __future__ import annotations

from dataclasses import replace
from .config import AgentConfig
from .types import Profile


def resolve_profile(name: str, config: AgentConfig, overrides: dict | None = None) -> Profile:
    defaults = {
        "coding": Profile(name="coding"),
        "review": Profile(name="review"),
        "deep": Profile(name="deep", enable_thinking=True, main_max_tokens=8192,
                        review_no_progress_actions=40),
        "legacy": Profile(name="legacy", enable_thinking=config.main.enable_thinking is not False,
                          main_max_tokens=config.main.max_tokens, worker_max_tokens=config.subagents.max_tokens,
                          model_timeout_seconds=config.main.timeout_seconds, constrain_actions=False),
    }
    if name in (overrides or {}):
        base = defaults.get(name, Profile(name=name)).model_dump()
        return Profile.model_validate({**base, **overrides[name], "name": name})
    if name not in defaults:
        raise ValueError(f"unknown profile {name!r}; configure it explicitly")
    profile = defaults[name]
    return profile.model_copy(update={"max_specialists": min(9, config.parallelism)})


def configured_models(config: AgentConfig, profile: Profile) -> AgentConfig:
    if profile.name == "legacy" or profile.preserve_model_settings:
        return config
    return replace(config,
        main=replace(config.main, max_tokens=profile.main_max_tokens,
                     timeout_seconds=profile.model_timeout_seconds,
                     enable_thinking=profile.enable_thinking,
                     reasoning_effort=None if profile.enable_thinking else "none"),
        subagents=replace(config.subagents, max_tokens=profile.worker_max_tokens,
                          timeout_seconds=profile.model_timeout_seconds,
                          enable_thinking=False, reasoning_effort="none"),
        reviewer=replace(config.reviewer, max_tokens=profile.worker_max_tokens,
                         timeout_seconds=profile.model_timeout_seconds,
                         enable_thinking=False, reasoning_effort="none"))
