#!/usr/bin/env python3
"""Compare bounded model proposals with the existing VDA acceptance runner.

No model transport is added. Callers provide an existing, timeout-aware adapter,
or the CLI replays explicit response captures. All generations finish before any
candidate is tested. Passing code is an artifact for this task, never a global
model promotion. A fresh-copy rerun uses the same acceptance tests; those tests
are readable inside the existing Docker container and are not protected holdouts.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, replace
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import tempfile
import threading
import time

import verified_dev_agent as vda

SCHEMA = "mj.candidate-comparison/1.0"
CAPTURE_SCHEMA = "mj.candidate-captures/1.0"
BASE_COMMIT = "381a80cc89df4a0f6e1b89f288194c50e58008b7"
BASE_SHA256 = "b93c583e79c410de85ea2e723747536ac28e0d9a1feaf0bed5f450b860c73f5b"
MAX_PROFILES = 3
MAX_ATTEMPTS = 2
MAX_CAPTURE_BYTES = 2_000_000


def digest(data: str | bytes) -> str:
    return hashlib.sha256(data.encode("utf-8") if isinstance(data, str) else data).hexdigest()


def _error_code(value, default="GENERATION_FAILED"):
    return value if isinstance(value, str) and re.fullmatch(r"[A-Z0-9_]{3,80}", value) else default


def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("DUPLICATE_JSON_KEY")
        result[key] = value
    return result


@dataclass(frozen=True)
class CandidateSpec:
    profile: str
    expected_model: str
    expected_upstream_model: str | None = None
    family: str | None = None


def _validate(specs, attempts, concurrency, budget_seconds):
    if (not isinstance(specs, (list, tuple)) or not 1 <= len(specs) <= MAX_PROFILES
            or type(attempts) is not int or not 1 <= attempts <= MAX_ATTEMPTS
            or type(concurrency) is not int or not 1 <= concurrency <= MAX_PROFILES
            or type(budget_seconds) not in (int, float)
            or not math.isfinite(budget_seconds) or not 5 <= budget_seconds <= 3600):
        raise ValueError("COMPARISON_BOUNDS_INVALID")
    names = set()
    for spec in specs:
        if (not isinstance(spec, CandidateSpec)
                or not isinstance(spec.profile, str)
                or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", spec.profile)
                or spec.profile in names):
            raise ValueError("CANDIDATE_PROFILE_INVALID")
        names.add(spec.profile)
        if (not isinstance(spec.expected_model, str) or not 1 <= len(spec.expected_model) <= 1024
                or (spec.expected_upstream_model is not None
                    and (not isinstance(spec.expected_upstream_model, str)
                         or not 1 <= len(spec.expected_upstream_model) <= 1024))
                or (spec.family is not None and
                    (not isinstance(spec.family, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", spec.family)))):
            raise ValueError("CANDIDATE_IDENTITY_INVALID")


def _safe_task(task):
    vda.audit_workspace(task.workspace)
    if not task.targets or len(task.targets) > vda.MAX_TARGET_FILES:
        raise ValueError("TARGET_COUNT_INVALID")
    for name in task.targets:
        vda.valid_path(name)
        path = Path(name)
        if (any(part.casefold() in ("test", "tests") for part in path.parts)
                or path.name.casefold().startswith("test_") or path.name.casefold().endswith("_test.py")):
            raise ValueError("ACCEPTANCE_TEST_CANNOT_BE_TARGET")
        source = task.workspace / name
        if not source.is_file() or source.is_symlink() or source.stat().st_size > vda.MAX_FILE_BYTES:
            raise ValueError("TARGET_SOURCE_INVALID")


def _summary(check):
    return {
        "passed": check.passed, "exit_code": check.exit_code,
        "tests_run": check.tests_run, "seconds": check.elapsed_seconds,
        "diagnostic_sha256": digest(check.diagnostic),
    }


def _check(runner, workspace, deadline):
    left = deadline - time.monotonic()
    if left < 5:
        return vda.Check(False, None, 0.0, "COMPARISON_BUDGET_EXCEEDED")
    actual = runner
    if type(runner) is vda.DockerRunner:
        actual = vda.DockerRunner(runner.image, min(runner.timeout, int(left)))
    try:
        result = actual.check(workspace)
        if (not isinstance(result, vda.Check) or type(result.passed) is not bool
                or type(result.elapsed_seconds) not in (int, float)
                or not math.isfinite(result.elapsed_seconds) or result.elapsed_seconds < 0
                or not isinstance(result.diagnostic, str)):
            return vda.Check(False, None, 0.0, "RUNNER_RESULT_INVALID")
        if time.monotonic() > deadline:
            return vda.Check(False, None, result.elapsed_seconds, "COMPARISON_BUDGET_EXCEEDED")
        return result
    except Exception:
        return vda.Check(False, None, 0.0, "RUNNER_FAILED")


def _passed(check, expected_count):
    return (check.passed is True and type(check.exit_code) is int and check.exit_code == 0
            and type(check.tests_run) is int and check.tests_run == expected_count)


def _response(response, spec):
    if not isinstance(response, dict):
        raise ValueError("RESPONSE_NOT_OBJECT")
    if "error" in response:
        error = response["error"]
        raise ValueError(_error_code(error.get("code") if isinstance(error, dict) else None))
    if response.get("model") != spec.expected_model:
        raise ValueError("MODEL_ID_MISMATCH")
    extra = response.get("mj", {})
    if not isinstance(extra, dict) or extra.get("paid_fallback") is True:
        raise ValueError("UNEXPECTED_PROVIDER_METADATA")
    upstream = extra.get("upstream_model")
    if spec.expected_upstream_model is not None and upstream != spec.expected_upstream_model:
        raise ValueError("UPSTREAM_MODEL_ID_MISMATCH")
    choices = response.get("choices")
    if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
        raise ValueError("CHOICES_INVALID")
    choice = choices[0]
    if (choice.get("finish_reason") != "stop"
            or type(choice.get("index", 0)) is not int or choice.get("index", 0) != 0):
        raise ValueError("INCOMPLETE_GENERATION")
    message = choice.get("message")
    if (not isinstance(message, dict) or message.get("role") != "assistant"
            or message.get("tool_calls") or message.get("refusal")):
        raise ValueError("FINAL_MESSAGE_INVALID")
    content = message.get("content")
    if not isinstance(content, str) or not content.strip() or len(content.encode("utf-8")) > vda.MAX_RESPONSE_BYTES:
        raise ValueError("FINAL_CONTENT_INVALID")
    if upstream is not None and (not isinstance(upstream, str) or len(upstream) > 1024):
        raise ValueError("UPSTREAM_MODEL_ID_INVALID")
    return content, upstream


def _observations(response):
    """Keep bounded failure evidence without echoing arbitrary provider text."""
    result = {}
    if not isinstance(response, dict):
        return result
    extra = response.get("mj", {})
    for key, value in (("actual_model", response.get("model")),
                       ("actual_upstream_model", extra.get("upstream_model") if isinstance(extra, dict) else None)):
        if isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9._:/+@%~=-]{1,1024}", value):
            result[key] = value
    choices = response.get("choices")
    if isinstance(choices, list) and len(choices) == 1 and isinstance(choices[0], dict):
        finish = choices[0].get("finish_reason")
        if finish in ("stop", "length", "tool_calls", "content_filter"):
            result["finish_reason"] = finish
        message = choices[0].get("message")
        content = message.get("content") if isinstance(message, dict) else None
        if isinstance(content, str) and len(content.encode("utf-8")) <= vda.MAX_RESPONSE_BYTES:
            result["raw_content_sha256"] = digest(content)
    return result


def _normalize(content):
    match = re.fullmatch(r"\x60\x60\x60(?:json)?[ \t]*\r?\n([\s\S]*?)\r?\n\x60\x60\x60", content.strip())
    return (match.group(1), "outer_json_fence_only") if match else (content, "none")


def compare(task, model, runner, specs, *, attempts=1, concurrency=2, budget_seconds=900):
    """Return integrity receipt and one independently rechecked task artifact.

    Adapter contract: generate_evidence(profile, prompt, timeout_seconds) returns
    an OpenAI-compatible final response. The existing adapter must honor its
    timeout; Python threads cannot forcibly terminate an arbitrary callback.
    Late responses are discarded, never promoted. No test feedback is supplied.
    """
    _validate(specs, attempts, concurrency, budget_seconds)
    if digest(Path(vda.__file__).read_bytes()) != BASE_SHA256:
        raise ValueError("VDA_SOURCE_PIN_MISMATCH")
    _safe_task(task)
    manifest = vda.workspace_manifest(task.workspace)
    start = time.monotonic()
    deadline = start + budget_seconds
    rows = []
    selected = None
    selected_id = None
    prompt_sha = None
    initial = None
    status = "NO_VERIFIED_CANDIDATE"
    frozen = False
    with tempfile.TemporaryDirectory(prefix="mj-candidate-compare-") as scratch:
        root = Path(scratch)
        snapshot = root / "frozen-source"
        shutil.copytree(task.workspace, snapshot)
        frozen_manifest = vda.workspace_manifest(snapshot)
        if frozen_manifest != manifest:
            raise ValueError("SOURCE_CHANGED_DURING_CAPTURE")
        manifest_sha = vda.sha(frozen_manifest)
        if getattr(model, "expected_workspace_sha256", manifest_sha) != manifest_sha:
            raise ValueError("CAPTURE_TASK_MANIFEST_MISMATCH")
        fixed_task = replace(task, workspace=snapshot)
        prompt = vda.prompt_for(fixed_task)
        prompt_sha = digest(prompt)
        if getattr(model, "expected_prompt_sha256", prompt_sha) != prompt_sha:
            raise ValueError("CAPTURE_PROMPT_MISMATCH")
        baseline = root / "baseline"
        shutil.copytree(snapshot, baseline)
        initial = _check(runner, baseline, deadline)
        if initial.passed:
            status = "NO_FAILING_BASELINE"
        elif (type(initial.tests_run) is not int or initial.tests_run <= 0
              or type(initial.exit_code) is not int or initial.exit_code != 1):
            status = "BASELINE_RUNNER_UNAVAILABLE"
        else:
            plan = [(number, attempt, spec) for number, spec in enumerate(specs)
                    for attempt in range(1, attempts + 1)]
            locks = {spec.family or spec.profile: threading.Lock() for spec in specs}

            def generate(item):
                number, attempt, spec = item
                row = {
                    "candidate_id": f"{spec.profile}-{attempt}", "profile": spec.profile,
                    "attempt": attempt, "profile_order": number, "status": "GENERATION_FAILED",
                    "expected_model": spec.expected_model,
                    "expected_upstream_model": spec.expected_upstream_model,
                    "prompt_sha256": prompt_sha,
                }
                content = None
                try:
                    with locks[spec.family or spec.profile]:
                        left = deadline - time.monotonic()
                        if left < 1:
                            raise ValueError("COMPARISON_BUDGET_EXCEEDED")
                        begun = time.monotonic()
                        response = model.generate_evidence(spec.profile, prompt, timeout_seconds=left)
                        row["generation_seconds"] = round(time.monotonic() - begun, 6)
                    row.update(_observations(response))
                    if time.monotonic() > deadline:
                        raise ValueError("GENERATION_BUDGET_EXCEEDED")
                    content, upstream = _response(response, spec)
                    row.update(actual_model=response["model"], actual_upstream_model=upstream,
                               finish_reason="stop", raw_content_sha256=digest(content),
                               status="GENERATION_CAPTURED")
                except Exception as exc:
                    row["error_code"] = _error_code(str(exc)) if isinstance(exc, ValueError) else "GENERATION_FAILED"
                    content = None
                return row, content

            generated = {}
            with ThreadPoolExecutor(max_workers=min(concurrency, len(plan))) as pool:
                futures = {pool.submit(generate, item): (item[0], item[1]) for item in plan}
                for future in as_completed(futures):
                    generated[futures[future]] = future.result()
            frozen = True
            # Never add diagnostics or candidate source to any generation prompt.
            for number, attempt, spec in plan:
                row, content = generated[number, attempt]
                rows.append(row)
                if content is None:
                    continue
                if time.monotonic() >= deadline:
                    row.update(status="BUDGET_EXCEEDED", error_code="COMPARISON_BUDGET_EXCEEDED")
                    continue
                try:
                    normalized, normalization = _normalize(content)
                    row.update(normalization=normalization,
                               strict_json_contract=normalization == "none",
                               normalized_content_sha256=digest(normalized))
                    json.loads(normalized, object_pairs_hook=_unique)
                    edits = vda.parse_edits(normalized, fixed_task.targets)
                    row["edits_sha256"] = vda.sha(edits)
                    candidate = root / f"candidate-{number}-{attempt}"
                    shutil.copytree(snapshot, candidate)
                    vda.apply_edits(candidate, edits)
                    before = vda.workspace_manifest(candidate)
                    row["candidate_workspace_sha256"] = vda.sha(before)
                    check = _check(runner, candidate, deadline)
                    row["acceptance"] = _summary(check)
                    if vda.workspace_manifest(candidate) != before:
                        raise ValueError("CANDIDATE_WORKSPACE_CHANGED")
                    if not _passed(check, initial.tests_run):
                        row["status"] = "ACCEPTANCE_FAILED"
                        continue
                    fresh = root / f"recheck-{number}-{attempt}"
                    shutil.copytree(snapshot, fresh)
                    vda.apply_edits(fresh, edits)
                    recheck = _check(runner, fresh, deadline)
                    row["independent_recheck"] = _summary(recheck)
                    if vda.workspace_manifest(fresh) != before:
                        raise ValueError("RECHECK_WORKSPACE_CHANGED")
                    if not _passed(recheck, initial.tests_run):
                        row["status"] = "INDEPENDENT_RECHECK_FAILED"
                        continue
                    row["status"] = "PASS_ACCEPTANCE_RECHECK"
                    # Declared profile order, then attempt; never race completion,
                    # parameter counts or self-reported model confidence.
                    if selected is None:
                        selected, selected_id = edits, row["candidate_id"]
                except (ValueError, OSError, UnicodeError) as exc:
                    row.update(status="INVALID_CANDIDATE",
                               error_code=_error_code(str(exc), "CANDIDATE_INVALID"))
            if selected is not None:
                status = "PASS_ACCEPTANCE_RECHECK"
    record = vda.signed({
        "schema": SCHEMA, "task_id": task.task_id,
        "task_sha256": vda.sha({"id": task.task_id, "issue": task.issue,
                               "targets": task.targets, "workspace": manifest}),
        "input_workspace_sha256": vda.sha(manifest),
        "task_manifest_sha256": vda.sha(manifest),
        "prompt_sha256": prompt_sha, "status": status,
        "baseline": _summary(initial), "candidates": rows,
        "selected_candidate_id": selected_id,
        "selected_edits_sha256": vda.sha(selected) if selected is not None else None,
        "selection_rule": "first_passing_declared_profile_then_attempt",
        "all_generations_frozen_before_candidate_tests": frozen,
        "max_parallel_generations": concurrency, "attempts_per_profile": attempts,
        "budget_seconds": budget_seconds,
        "deadline_contract": "existing_adapter_timeout_required_late_results_rejected",
        "generation_mode": getattr(model, "evidence_mode", "caller_adapter_reported"),
        "model_identity_basis": "response_reported_not_weight_attestation",
        "test_runner": "existing_vda_docker" if type(runner) is vda.DockerRunner else "injected_runner_unattested",
        "test_scope": "same_acceptance_tests_on_fresh_workspace_not_protected_holdout",
        "quality_scope": "one_task_artifact_not_global_model_promotion",
        "foundation_model_training": False,
        "new_model_transport_added": False,
        "source_provenance": {
            "repository": "MusicJapanLLC/music-japan-cloud-inference",
            "base_commit": BASE_COMMIT, "verified_dev_agent_sha256": BASE_SHA256,
            "comparator_sha256": digest(Path(__file__).read_bytes()),
        },
        "elapsed_seconds": round(time.monotonic() - start, 6),
        "observed_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    })
    return record, selected


class CapturedModel:
    evidence_mode = "provided_response_captures_not_live_generation"

    def __init__(self, path):
        raw = Path(path).read_bytes()
        if not 1 <= len(raw) <= MAX_CAPTURE_BYTES:
            raise ValueError("CAPTURE_SIZE_INVALID")
        data = json.loads(raw, object_pairs_hook=_unique)
        if (not isinstance(data, dict) or set(data) != {
                    "schema", "prompt_sha256", "task_manifest_sha256", "candidates"}
                or data["schema"] != CAPTURE_SCHEMA
                or not isinstance(data["prompt_sha256"], str)
                or not re.fullmatch("[a-f0-9]{64}", data["prompt_sha256"])
                or not isinstance(data["task_manifest_sha256"], str)
                or not re.fullmatch("[a-f0-9]{64}", data["task_manifest_sha256"])
                or not isinstance(data["candidates"], list)):
            raise ValueError("CAPTURE_SCHEMA_INVALID")
        self.expected_prompt_sha256 = data["prompt_sha256"]
        self.expected_workspace_sha256 = data["task_manifest_sha256"]
        self.specs, self.responses = [], {}
        for row in data["candidates"]:
            if not isinstance(row, dict) or set(row) - {
                    "profile", "expected_model", "expected_upstream_model", "family", "responses"}:
                raise ValueError("CAPTURE_CANDIDATE_INVALID")
            spec = CandidateSpec(row.get("profile"), row.get("expected_model"),
                                 row.get("expected_upstream_model"), row.get("family"))
            replies = row.get("responses")
            if not isinstance(replies, list) or not 1 <= len(replies) <= MAX_ATTEMPTS:
                raise ValueError("CAPTURE_RESPONSE_COUNT_INVALID")
            self.specs.append(spec)
            self.responses[spec.profile] = iter(replies)
        _validate(self.specs, 1, 1, 900)
        self.capture_sha256 = digest(raw)

    def generate_evidence(self, profile, prompt, timeout_seconds):
        if digest(prompt) != self.expected_prompt_sha256:
            raise ValueError("CAPTURE_PROMPT_MISMATCH")
        try:
            return next(self.responses[profile])
        except StopIteration:
            raise ValueError("CAPTURE_RESPONSE_MISSING") from None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", type=Path, required=True)
    parser.add_argument("--captures", type=Path)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--attempts", type=int, default=1)
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--budget-seconds", type=float, default=900)
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--selected", type=Path)
    args = parser.parse_args(argv)
    try:
        task = vda.load_task(args.task)
        _safe_task(task)
        prompt_sha = digest(vda.prompt_for(task))
        if not args.execute:
            print(json.dumps({"mode": "dry_run", "task_id": task.task_id,
                              "prompt_sha256": prompt_sha, "generation_performed": False,
                              "task_manifest_sha256": vda.sha(vda.workspace_manifest(task.workspace)),
                              "containers_started": False, "capture_schema": CAPTURE_SCHEMA}))
            return 0
        if not args.captures or not args.receipt:
            raise ValueError("CAPTURES_AND_RECEIPT_REQUIRED")
        for output in (args.receipt, args.selected):
            if output is not None and (output.exists() or output.is_symlink()):
                raise ValueError("FRESH_OUTPUT_PATH_REQUIRED")
        model = CapturedModel(args.captures)
        result, edits = compare(task, model, vda.DockerRunner(), model.specs,
                                attempts=args.attempts, concurrency=args.concurrency,
                                budget_seconds=args.budget_seconds)
        result = vda.signed({**result, "capture_file_sha256": model.capture_sha256})
        args.receipt.parent.mkdir(parents=True, exist_ok=True)
        with args.receipt.open("xb") as destination:
            destination.write(vda.canonical(result) + b"\n")
        if edits is not None and args.selected:
            args.selected.parent.mkdir(parents=True, exist_ok=True)
            with args.selected.open("xb") as destination:
                destination.write(vda.canonical({
                    "task_sha256": result["task_sha256"],
                    "comparison_receipt_sha256": result["sha256"], "edits": edits,
                }) + b"\n")
        print(json.dumps(result, ensure_ascii=False))
        return 0 if edits is not None else 2
    except (ValueError, OSError, TypeError) as exc:
        print(json.dumps({"status": "FAILED", "error_code": _error_code(str(exc), "COMPARISON_INPUT_INVALID")}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
