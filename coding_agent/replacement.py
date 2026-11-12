"""Caller-configured selector publication and verifier pinning."""
import os
import stat
import tempfile
from pathlib import Path

from .sandbox import CommandCancelled, SandboxError, SandboxRunner
from .workspace import checked_path, copy_source, digest, manifest


def verifier_runner(store, run_id, runner, spec):
    original = Path(spec.verifier_path)
    frozen = store.run_dir(run_id) / "artifacts" / spec.verifier_sha256
    if digest(original) != spec.verifier_sha256 or digest(frozen) != spec.verifier_sha256:
        raise ValueError("frozen replacement verifier digest mismatch")
    pinned = SandboxRunner(runner.profile, runner.dependencies, runner.cancelled)
    pinned.protected_files = {**runner.protected_files, spec.verifier_path: frozen}
    return pinned


def select_replacement(checks):
    spec = checks.replacement
    store, run_id, workspace = checks.store, checks.run_id, checks.workspace
    key = f"replacement:{spec.target_id}"
    receipt = store.receipt(run_id, key)
    selector = checked_path(workspace.root, spec.selector_path)
    source = checked_path(workspace.root, spec.source_path)
    current = digest(selector)
    c_hash = digest(source)
    if c_hash is None:
        return {"error": "Create the required C source before automatic selection."}

    data = {"target_id": spec.target_id, "argv": spec.selector_argv,
            "selector_before": current, "c_before": c_hash}

    def record(state, result):
        artifact = store.artifact(run_id, result)
        result = {**result, "artifact_id": artifact}
        store.save_receipt(run_id, key, "replacement", state, result)
        store.event(run_id, "replacement_selection", {"state": state, "artifact_id": artifact})
        return result

    try:
        runner = verifier_runner(store, run_id, checks.runner, spec)
        if current == spec.selected_selector_sha256 and receipt:
            if receipt["state"] == "completed":
                return receipt["data"]
            if receipt["state"] == "prepared" and receipt["data"]["c_before"] == c_hash:
                return record("completed", {k: v for k, v in receipt["data"].items() if k != "artifact_id"})
        if (receipt and receipt["state"] == "completed") or current != spec.original_selector_sha256:
            raise ValueError("selector SHA-256 does not match the frozen original")
        store.consume(run_id)
        with tempfile.TemporaryDirectory(prefix="replacement-", dir=workspace.folder) as directory:
            staging = Path(directory) / "source"
            original = manifest(workspace.root)
            copy_source(workspace.root, staging)
            staged_paths = sorted(staging.rglob("*"))
            try:
                result = runner.run(staging, spec.selector_argv)
            except CommandCancelled as exc:
                record("interrupted", {**data, "command": exc.result})
                raise
            data["command"] = result
            if result["exit_code"] or result["timed_out"] or result["output_truncated"]:
                raise ValueError("frozen selector command failed")
            changed = manifest(staging)
            expected = dict(original)
            expected[spec.selector_path] = {**original[spec.selector_path], "sha256": spec.selected_selector_sha256}
            if (changed != expected or sorted(staging.rglob("*")) != staged_paths
                    or manifest(workspace.root) != original):
                raise ValueError("selector transformation changed unexpected source or selector bytes")
            if runner.cancelled():
                record("interrupted", data)
                raise InterruptedError("run cancelled before selector publication")
            data.update(selector_after=spec.selected_selector_sha256, c_after=c_hash)
            record("prepared", data)
            with tempfile.NamedTemporaryFile(dir=directory, delete=False) as pending:
                pending.write((staging / spec.selector_path).read_bytes())
                pending.flush()
                os.fchmod(pending.fileno(), stat.S_IMODE(selector.stat().st_mode))
                os.fsync(pending.fileno())
            os.replace(pending.name, selector)
        return record("completed", data)
    except InterruptedError:
        raise
    except (ValueError, OSError, SandboxError) as exc:
        if store.receipt(run_id, key) and digest(selector) == spec.selected_selector_sha256 and current != spec.selected_selector_sha256:
            raise
        # Preserve a prepared/completed receipt if a later attempt encounters drift.
        if receipt and receipt["state"] in {"prepared", "completed"}:
            return {**data, "error": str(exc), "artifact_id": store.artifact(run_id, {**data, "error": str(exc)})}
        return record("failed", {**data, "error": str(exc)})
