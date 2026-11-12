"""Prepare or execute a bounded live qualification campaign from a source checkout."""
import argparse
import fcntl
import hashlib
import json
import os
import re
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

from .config import load_config
from .qualification import prepare_fixtures, rate
from .runtime_model import RecordingProvider, RecoverableResponseError
from .model import ModelError
from .service import Harness
from .types import parse_action

ROOT = Path(__file__).resolve().parents[1]
PROFILE = {"max_actions": 200, "max_active_seconds": 600, "max_model_requests": 200,
           "preserve_model_settings": True}
REPAIR_PROFILE = {**PROFILE, "check_after_edit": True}
ROLES = ("main", "subagent")
MODEL_OVERRIDES = {
    "MAIN_BASE_URL", "MAIN_MODEL", "SUBAGENT_BASE_URL", "SUBAGENT_MODEL",
    "SUBAGENT_PARALLELISM", "REVIEWER_BASE_URL", "REVIEWER_MODEL", "REVIEWER_API_KEY_ENV",
}


def save(path, data):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def load(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def manifest(directory, config, config_path):
    tracked = subprocess.check_output(["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard", "--", "coding_agent", "tests", "pyproject.toml"], cwd=ROOT).decode().split("\0")
    return {
        "revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "source": {name: digest(ROOT / name) for name in sorted(set(tracked) |
                   {"coding_agent/campaign.py"}) if name and (ROOT / name).is_file()},
        "fixtures": {str(p.relative_to(directory)): digest(p)
                     for p in sorted((directory / "fixtures").rglob("*")) if p.is_file()},
        "configuration": json.loads(json.dumps(asdict(config))),
        "config_sha256": digest(config_path),
        "project_registry_sha256": digest(ROOT / "project-profiles.json") if
        (ROOT / "project-profiles.json").is_file() else None,
        "protocol": {"opportunities_per_role": 100, "attempts": 3},
        "job_profile": PROFILE,
        "repair_profile": REPAIR_PROFILE,
        "protocol_model_request_budget_per_role": 300,
    }


def protocol_attempt(provider, role, action, previous):
    prompt = [{"role": "system", "content": "Return one JSON action in the harness action envelope."},
              {"role": "user", "content": "Return this action: " + json.dumps(action)}]
    if previous:
        prompt.append({"role": "user", "content": "The previous attempt failed validation. Return the requested action."})
    try:
        response = parse_action(provider.invoke(prompt, role=role))
    except RecoverableResponseError as exc:
        return exc.category
    except ModelError as exc:
        return "timeout" if "timed out" in str(exc) or "timeout" in str(exc) else "provider_error"
    except ValueError:
        return "malformed_action"
    return "valid" if response == parse_action(json.dumps(action)) else "wrong_action"


def run_protocol(harness, directory):
    opportunities = load(directory / "fixtures/protocol.json")
    metrics = {}
    for role in ROLES:
        path = directory / f"protocol-{role}.json"
        journal = load(path) if path.exists() else {"run_id": harness.submit({
            "root": opportunities[0]["root"], "task": "Protocol qualification",
            "profile": "protocol-smoke",
        }), "samples": {}}
        save(path, journal)
        run_id = journal["run_id"]
        provider = RecordingProvider(harness.config, harness.store, run_id,
                                     lambda: harness.store.get(run_id)["cancel_requested"])
        budget = harness.store.get(run_id)["profile"]["max_active_seconds"]
        exhausted = False
        finished = False
        harness.store.transition(run_id, "running")
        try:
            for opportunity in opportunities:
                attempts = journal["samples"].setdefault(str(opportunity["id"]), [])
                if attempts and attempts[-1] == "started":
                    attempts[-1] = "interrupted"
                    save(path, journal)
                while len(attempts) < 3 and "valid" not in attempts:
                    if harness.store.status(run_id).active_seconds >= budget:
                        exhausted = True
                        break
                    attempts.append("started")
                    save(path, journal)
                    try:
                        attempts[-1] = protocol_attempt(provider, role, opportunity["requested_action"], attempts[:-1])
                    except (KeyboardInterrupt, InterruptedError):
                        attempts[-1] = "cancelled"
                        save(path, journal)
                        raise
                    save(path, journal)
                if exhausted:
                    break
            finished = True
        finally:
            harness.store.transition(run_id, "failed" if exhausted else "completed" if finished else "interrupted",
                                     error="protocol active-time budget exhausted" if exhausted else None)
        samples = list(journal["samples"].values())
        first = sum(bool(a and a[0] == "valid") for a in samples)
        recovered = sum("valid" in a for a in samples)
        metrics[role] = {"first_response": rate(first, len(opportunities)),
                         "after_recovery": rate(recovered, len(opportunities)),
                         "budget_exhausted": exhausted,
                         "passed": first >= 99 and recovered == 100 and not exhausted}
    return metrics


def grade_review(result, gold):
    review = result.review if result else None
    findings = review.findings if review else []
    decisions = []
    for finding in findings:
        proposal = finding.proposal
        description = f"{proposal.title} {proposal.claim}"
        matched = bool(gold["expected"] and proposal.path == gold["path"] and
                       re.search(r"\b99\b", description) and re.search(r"\b100\b", description)
                       and finding.status != "needs_human_review")
        decisions.append({"finding": finding.model_dump(), "matches_seed": matched})
    detected = gold["expected"] if any(d["matches_seed"] for d in decisions) else []
    # These review fixtures have no registered trusted probes.
    promotions_valid = all(f.status != "reproduced" for f in findings)
    unsupported = sum(d["finding"]["proposal"]["severity"] == "critical" and
                      not d["matches_seed"] for d in decisions)
    complete = bool(review and review.completion == "completed")
    return {"completion": review.completion if review else None, "expected": gold["expected"],
            "detected": detected, "unsupported_critical": unsupported, "promotions_valid": promotions_valid,
            "decisions": decisions, "passed": complete and detected == gold["expected"] and
            (bool(gold["expected"]) or not findings) and not unsupported and promotions_valid}


def run_reviews(harness, directory):
    fixtures = directory / "fixtures"
    gold = {row["id"]: row for row in load(fixtures / "gold.json")}
    grades = {}
    for case in load(fixtures / "requests.json"):
        identity = case["id"]
        path = directory / f"review-{identity}.json"
        record = load(path) if path.exists() else {
            "run_id": harness.submit({**case["request"], "profile": "campaign-review"})}
        save(path, record)
        if "grade" not in record:
            try:
                status = harness.wait(record["run_id"], timeout=620)
            except (TimeoutError, KeyboardInterrupt):
                harness.cancel(record["run_id"])
                raise
            result = harness.result(record["run_id"])
            record.update(status=status.model_dump(), result=result.model_dump() if result else None,
                          grade=grade_review(result, gold[identity]))
            save(path, record)
        grades[identity] = record["grade"]
    return grades


def run_suite(directory, config_path=None):
    repair = config_path is not None
    path = directory / ("repairs" if repair else "baseline")
    path.mkdir(exist_ok=True)
    receipt = path / "exit.json"
    if receipt.exists():
        return load(receipt)["exit_code"] == 0
    env = {k: v for k, v in os.environ.items() if k not in ("RUN_V2_LIVE", "RUN_LIVE_MODEL_TESTS")}
    if not repair:
        for name in MODEL_OVERRIDES:
            env.pop(name, None)
    args = [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"]
    if repair:
        env.update(RUN_V2_LIVE="1", HARNESS_LIVE_CONFIG=str(config_path),
                   HARNESS_LIVE_EVIDENCE_DIR=str(path))
        args.extend(["-p", "test_v2_live.py"])
    started = path / "started.json"
    if repair and started.exists():
        # A killed driver may have left a supervised job alive. Preserve its
        # run ID and require explicit investigation instead of starting it twice.
        save(receipt, {"exit_code": 1, "reason": "interrupted repair driver"})
        return False
    if repair:
        save(started, {"state": "started"})
    with (path / "suite.log").open("ab") as log:
        code = subprocess.run(args, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
    save(receipt, {"exit_code": code})
    return code == 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "baseline", "run"))
    parser.add_argument("--review-only", action="store_true", help="Stop after the review gate")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--evidence-dir", required=True, type=Path)
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = load_config(config_path)
    directory = args.evidence_dir.resolve()
    if any(".invalid" in item.base_url for item in
                                  (config.main, config.subagents, config.reviewer)):
        parser.error("explicit lead, specialist and reviewer endpoints are required")
    frozen = directory / "manifest.json"
    if args.phase == "prepare":
        directory.mkdir(mode=0o700, parents=True)
        prepare_fixtures(directory / "fixtures")
        save(frozen, manifest(directory, config, config_path))
        print(frozen)
        return 0
    if load(frozen) != manifest(directory, config, config_path):
        parser.error("campaign inputs changed. Prepare a new evidence directory.")
    with (directory / "campaign.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.phase == "baseline":
            return 0 if run_suite(directory) else 1
        return run_campaign(directory, config, config_path, review_only=args.review_only)


def run_campaign(directory, config, config_path, *, review_only=False):
    summary = directory / "summary.json"
    if not run_suite(directory):
        save(summary, {"failed_stage": "baseline"})
        return 1
    # Keep protocol requests out of the supervised queue.
    protocol = Harness(directory / "protocol-state", config=config, config_path=config_path, autostart=False,
                       profiles={"protocol-smoke": {**PROFILE, "max_model_requests": 300}})
    metrics = run_protocol(protocol, directory)
    report = {"baseline": "passed", "protocol": metrics}
    if not all(item["passed"] for item in metrics.values()):
        save(summary, {**report, "failed_stage": "protocol"})
        return 1
    save(summary, report)
    reviews = Harness(directory / "review-state", config=config, config_path=config_path,
                      profiles={"campaign-review": PROFILE})
    report["reviews"] = run_reviews(reviews, directory)
    if not all(item["passed"] for item in report["reviews"].values()):
        save(summary, {**report, "failed_stage": "review"})
        return 1
    save(summary, report)
    if review_only:
        save(summary, {**report, "review_gate_passed": True})
        return 0
    passed = run_suite(directory, config_path)
    report["repairs"] = {"passed": passed, "reports": [
        str(path.relative_to(directory)) for path in sorted((directory / "repairs").glob("*/acceptance.json"))
    ]}
    report["eligible_for_pilot_retry"] = passed
    if not passed:
        report["failed_stage"] = "repair"
    save(summary, report)
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
