"""Offline orchestration tests; no model calls or generated-code execution."""
import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

import compare_candidates as c
import verified_dev_agent as vda


def edits(content="def answer():\n    return 7\n", path="answer.py"):
    return json.dumps({"edits": [{"path": path, "content": content}]})


def response(model="model-a", content=None, *, finish="stop", upstream=None):
    value = {"model": model, "choices": [{"index": 0, "finish_reason": finish,
              "message": {"role": "assistant", "content": edits() if content is None else content}}]}
    if upstream is not None:
        value["mj"] = {"upstream_model": upstream, "paid_fallback": False}
    return value


class FakeModel:
    def __init__(self, outputs):
        self.outputs = outputs
        self.calls = []
        self.finished = 0
        self.lock = threading.Lock()

    def generate_evidence(self, profile, prompt, timeout_seconds):
        with self.lock:
            self.calls.append((profile, prompt, timeout_seconds))
        answer = self.outputs[profile]
        if callable(answer):
            answer = answer()
        with self.lock:
            self.finished += 1
        if isinstance(answer, Exception):
            raise answer
        return answer


class AcceptanceRunner:
    def __init__(self, model=None, expected_finished=None, fail_recheck=False, wrong_count=False):
        self.calls = []
        self.model, self.expected_finished = model, expected_finished
        self.fail_recheck, self.wrong_count = fail_recheck, wrong_count

    def check(self, workspace):
        self.calls.append(workspace)
        if workspace.name != "baseline" and self.model is not None:
            if self.model.finished != self.expected_finished:
                raise AssertionError("candidate tests started before all generations froze")
        good = "return 7" in (workspace / "answer.py").read_text()
        if self.fail_recheck and workspace.name.startswith("recheck"):
            good = False
        count = 1 if self.wrong_count and workspace.name != "baseline" else 2
        return vda.Check(good, 0 if good else 1, 0.01,
                         "OK" if good else "PRIVATE_TEST_DIAGNOSTIC_SENTINEL", count)


class ComparisonTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.workspace = self.root / "workspace"
        self.workspace.mkdir()
        (self.workspace / "answer.py").write_text("def answer():\n    return 0\n")
        (self.workspace / "tests").mkdir()
        (self.workspace / "tests/test_answer.py").write_text("# PRIVATE_TEST_SOURCE_SENTINEL\n")
        self.task = vda.Task("bounded-answer", "Repair answer while preserving its public interface.",
                             self.workspace, ("answer.py",))
        self.specs = [c.CandidateSpec("a", "model-a"), c.CandidateSpec("b", "model-b")]

    def tearDown(self):
        self.temp.cleanup()

    def compare(self, outputs, *, specs=None, runner=None, **kwargs):
        model = FakeModel(outputs)
        check = runner or AcceptanceRunner()
        result, selected = c.compare(self.task, model, check, specs or self.specs, **kwargs)
        return result, selected, model, check

    def test_wrong_candidate_rejected_and_right_candidate_rechecked(self):
        result, selected, model, runner = self.compare({
            "a": response(content=edits("def answer():\n    return 0\n")),
            "b": response("model-b"),
        })
        self.assertEqual(result["selected_candidate_id"], "b-1")
        self.assertEqual([r["status"] for r in result["candidates"]],
                         ["ACCEPTANCE_FAILED", "PASS_ACCEPTANCE_RECHECK"])
        self.assertEqual(len(runner.calls), 4)
        self.assertEqual(result["selected_edits_sha256"], vda.sha(selected))
        self.assertTrue(vda.verified(result))
        self.assertEqual((self.workspace / "answer.py").read_text(), "def answer():\n    return 0\n")

    def test_all_fail_never_promotes_a_model_or_artifact(self):
        bad = edits("def answer():\n    return -1\n")
        result, selected, _, _ = self.compare({"a": response(content=bad), "b": response("model-b", bad)})
        self.assertEqual(result["status"], "NO_VERIFIED_CANDIDATE")
        self.assertIsNone(selected)
        self.assertIsNone(result["selected_edits_sha256"])
        self.assertEqual(len(result["candidates"]), 2)

    def test_identity_terminal_and_empty_final_fail_before_tests(self):
        variants = [
            (response("different-model"), "MODEL_ID_MISMATCH"),
            (response(finish="length"), "INCOMPLETE_GENERATION"),
            (response(content=" "), "FINAL_CONTENT_INVALID"),
            ({"choices": response()["choices"]}, "MODEL_ID_MISMATCH"),
        ]
        for bad, code in variants:
            with self.subTest(code=code):
                result, selected, _, runner = self.compare({"a": bad}, specs=self.specs[:1])
                self.assertIsNone(selected)
                self.assertEqual(result["candidates"][0]["error_code"], code)
                self.assertEqual(len(runner.calls), 1)

    def test_upstream_identity_is_strict_even_when_gateway_alias_matches(self):
        specs = [c.CandidateSpec("a", "model-a", "expected-upstream")]
        result, selected, _, _ = self.compare(
            {"a": response(upstream="wrong-upstream")}, specs=specs)
        self.assertIsNone(selected)
        self.assertEqual(result["candidates"][0]["error_code"], "UPSTREAM_MODEL_ID_MISMATCH")

    def test_rejected_response_keeps_actual_identity_and_output_hash(self):
        raw = edits()
        result, selected, _, _ = self.compare(
            {"a": response("wrong-model", raw, finish="length")}, specs=self.specs[:1])
        row = result["candidates"][0]
        self.assertIsNone(selected)
        self.assertEqual(row["actual_model"], "wrong-model")
        self.assertEqual(row["finish_reason"], "length")
        self.assertEqual(row["raw_content_sha256"], c.digest(raw))

    def test_boolean_choice_index_is_not_an_integer_index(self):
        value = response()
        value["choices"][0]["index"] = False
        result, selected, _, _ = self.compare({"a": value}, specs=self.specs[:1])
        self.assertIsNone(selected)
        self.assertEqual(result["candidates"][0]["error_code"], "INCOMPLETE_GENERATION")

    def test_path_escape_or_test_edit_cannot_be_applied(self):
        for path in ("../outside.py", "tests/test_answer.py"):
            with self.subTest(path=path):
                result, selected, _, runner = self.compare(
                    {"a": response(content=edits(path=path))}, specs=self.specs[:1])
                self.assertIsNone(selected)
                self.assertEqual(result["candidates"][0]["status"], "INVALID_CANDIDATE")
                self.assertEqual(len(runner.calls), 1)
                self.assertFalse((self.root / "outside.py").exists())

    def test_tests_and_diagnostics_never_enter_prompts(self):
        model = FakeModel({"a": response(), "b": response("model-b")})
        runner = AcceptanceRunner(model, 2)
        result, _ = c.compare(self.task, model, runner, self.specs)
        self.assertTrue(result["all_generations_frozen_before_candidate_tests"])
        for _, prompt, _ in model.calls:
            self.assertNotIn("PRIVATE_TEST_SOURCE_SENTINEL", prompt)
            self.assertNotIn("PRIVATE_TEST_DIAGNOSTIC_SENTINEL", prompt)
        self.assertEqual(model.calls[0][1], model.calls[1][1])
        unsafe_task = vda.Task("bad", self.task.issue, self.workspace, ("tests/test_answer.py",))
        with self.assertRaisesRegex(ValueError, "ACCEPTANCE_TEST_CANNOT_BE_TARGET"):
            c.compare(unsafe_task, model, runner, self.specs)
        (self.workspace / "Tests").mkdir(exist_ok=True)
        (self.workspace / "Tests/private_case.py").write_text("# PRIVATE_CASE_TEST")
        case_task = vda.Task("case", self.task.issue, self.workspace, ("Tests/private_case.py",))
        with self.assertRaisesRegex(ValueError, "ACCEPTANCE_TEST_CANNOT_BE_TARGET"):
            c.compare(case_task, model, runner, self.specs)

    def test_selection_uses_declared_order_not_fastest_generation(self):
        def delayed():
            time.sleep(0.02)
            return response(content=edits("# first profile\n" + "def answer():\n    return 7\n"))
        result, _, _, _ = self.compare({"a": delayed, "b": response("model-b")})
        self.assertEqual(result["selected_candidate_id"], "a-1")
        self.assertTrue(all(row["status"] == "PASS_ACCEPTANCE_RECHECK" for row in result["candidates"]))

    def test_same_provider_family_is_serialized_across_profiles_and_attempts(self):
        lock, running, maximum = threading.Lock(), [0], [0]
        def make(model):
            def call():
                with lock:
                    running[0] += 1
                    maximum[0] = max(maximum[0], running[0])
                time.sleep(0.01)
                with lock:
                    running[0] -= 1
                return response(model)
            return call
        specs = [c.CandidateSpec("a", "model-a", family="shared"),
                 c.CandidateSpec("b", "model-b", family="shared")]
        result, _, model, _ = self.compare(
            {"a": make("model-a"), "b": make("model-b")}, specs=specs, attempts=2, concurrency=3)
        self.assertEqual(maximum[0], 1)
        self.assertEqual(len(model.calls), 4)
        self.assertEqual(len(result["candidates"]), 4)

    def test_network_errors_preserved_without_false_success_or_secret_leak(self):
        result, selected, _, runner = self.compare({
            "a": {"error": {"code": "GATEWAY_HTTP_503"}},
            "b": RuntimeError("Authorization: Bearer secret-value"),
        })
        self.assertIsNone(selected)
        self.assertEqual(len(runner.calls), 1)
        self.assertEqual(result["candidates"][0]["error_code"], "GATEWAY_HTTP_503")
        self.assertNotIn("secret-value", json.dumps(result))
        self.assertEqual(result["candidates"][1]["error_code"], "GENERATION_FAILED")

    def test_budget_and_candidate_count_rejected_before_model_calls(self):
        model = FakeModel({})
        for kwargs in ({"attempts": 0}, {"attempts": 3}, {"concurrency": 4},
                       {"budget_seconds": float("nan")}, {"budget_seconds": 3601}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                c.compare(self.task, model, AcceptanceRunner(), self.specs, **kwargs)
        with self.assertRaises(ValueError):
            c.compare(self.task, model, AcceptanceRunner(),
                      [c.CandidateSpec(str(i), f"m{i}") for i in range(4)])
        self.assertEqual(model.calls, [])

    def test_late_generation_is_not_tested_or_promoted(self):
        clock = [100.0]
        def late():
            clock[0] += 20
            return response()
        with patch.object(c.time, "monotonic", side_effect=lambda: clock[0]):
            result, selected, _, runner = self.compare(
                {"a": late}, specs=self.specs[:1], budget_seconds=10)
        self.assertIsNone(selected)
        self.assertEqual(result["candidates"][0]["error_code"], "GENERATION_BUDGET_EXCEEDED")
        self.assertEqual(len(runner.calls), 1)

    def test_failed_independent_recheck_blocks_selection(self):
        result, selected, _, _ = self.compare(
            {"a": response()}, specs=self.specs[:1], runner=AcceptanceRunner(fail_recheck=True))
        self.assertIsNone(selected)
        self.assertEqual(result["candidates"][0]["status"], "INDEPENDENT_RECHECK_FAILED")

    def test_changed_test_count_blocks_selection(self):
        result, selected, _, _ = self.compare(
            {"a": response()}, specs=self.specs[:1], runner=AcceptanceRunner(wrong_count=True))
        self.assertIsNone(selected)
        self.assertEqual(result["candidates"][0]["status"], "ACCEPTANCE_FAILED")

    def test_runner_mutation_of_candidate_is_detected(self):
        class MutatingRunner(AcceptanceRunner):
            def check(self, workspace):
                result = super().check(workspace)
                if workspace.name != "baseline":
                    (workspace / "tests/test_answer.py").write_text("# changed")
                return result
        result, selected, _, _ = self.compare(
            {"a": response()}, specs=self.specs[:1], runner=MutatingRunner())
        self.assertIsNone(selected)
        self.assertEqual(result["candidates"][0]["error_code"], "CANDIDATE_WORKSPACE_CHANGED")

    def test_outer_fence_only_is_recorded_without_editing_code(self):
        raw = "\x60\x60\x60json\n" + edits() + "\n\x60\x60\x60"
        result, selected, _, _ = self.compare({"a": response(content=raw)}, specs=self.specs[:1])
        row = result["candidates"][0]
        self.assertEqual(row["normalization"], "outer_json_fence_only")
        self.assertFalse(row["strict_json_contract"])
        self.assertEqual(row["raw_content_sha256"], c.digest(raw))
        self.assertEqual(selected[0]["content"], "def answer():\n    return 7\n")
        result, selected, _, _ = self.compare(
            {"a": response(content="Here is the answer:\n" + raw)}, specs=self.specs[:1])
        self.assertIsNone(selected)

    def test_duplicate_json_fields_fail_closed(self):
        bad = '{"edits":[],"edits":[{"path":"answer.py","content":"pass"}]}'
        result, selected, _, _ = self.compare({"a": response(content=bad)}, specs=self.specs[:1])
        self.assertIsNone(selected)
        self.assertEqual(result["candidates"][0]["error_code"], "DUPLICATE_JSON_KEY")

    def test_baseline_already_passing_does_not_claim_repair(self):
        (self.workspace / "answer.py").write_text("def answer():\n    return 7\n")
        result, selected, model, _ = self.compare({})
        self.assertIsNone(selected)
        self.assertEqual(result["status"], "NO_FAILING_BASELINE")
        self.assertEqual(model.calls, [])

    def test_capture_prompt_binding_and_replay_label(self):
        path = self.root / "captures.json"
        value = {"schema": c.CAPTURE_SCHEMA,
                 "prompt_sha256": c.digest(vda.prompt_for(self.task)),
                 "task_manifest_sha256": vda.sha(vda.workspace_manifest(self.task.workspace)),
                 "candidates": [{"profile": "a", "expected_model": "model-a",
                                 "responses": [response()]}]}
        path.write_text(json.dumps(value))
        model = c.CapturedModel(path)
        result, selected = c.compare(self.task, model, AcceptanceRunner(), model.specs)
        self.assertIsNotNone(selected)
        self.assertEqual(result["generation_mode"], "provided_response_captures_not_live_generation")
        value["prompt_sha256"] = "0" * 64
        path.write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError, "CAPTURE_PROMPT_MISMATCH"):
            model = c.CapturedModel(path)
            c.compare(self.task, model, AcceptanceRunner(), model.specs)

    def test_capture_missing_or_invalid_task_manifest_is_rejected(self):
        path = self.root / "captures.json"
        value = {"schema": c.CAPTURE_SCHEMA,
                 "prompt_sha256": c.digest(vda.prompt_for(self.task)),
                 "candidates": [{"profile": "a", "expected_model": "model-a",
                                 "responses": [response()]}]}
        path.write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError, "CAPTURE_SCHEMA_INVALID"):
            c.CapturedModel(path)
        for invalid in ("", "g" * 64, None, 123):
            with self.subTest(invalid=invalid):
                value["task_manifest_sha256"] = invalid
                path.write_text(json.dumps(value))
                with self.assertRaisesRegex(ValueError, "CAPTURE_SCHEMA_INVALID"):
                    c.CapturedModel(path)

    def test_capture_binds_tests_even_when_generation_prompt_is_unchanged(self):
        path = self.root / "captures.json"
        expected_manifest = vda.sha(vda.workspace_manifest(self.task.workspace))
        value = {"schema": c.CAPTURE_SCHEMA,
                 "prompt_sha256": c.digest(vda.prompt_for(self.task)),
                 "task_manifest_sha256": expected_manifest,
                 "candidates": [{"profile": "a", "expected_model": "model-a",
                                 "responses": [response()]}]}
        path.write_text(json.dumps(value))
        model = c.CapturedModel(path)
        self.assertEqual(model.expected_workspace_sha256, expected_manifest)
        (self.workspace / "tests/test_answer.py").write_text("# changed acceptance tests\n")
        self.assertEqual(c.digest(vda.prompt_for(self.task)), value["prompt_sha256"])
        runner = AcceptanceRunner()
        with self.assertRaisesRegex(ValueError, "CAPTURE_TASK_MANIFEST_MISMATCH"):
            c.compare(self.task, model, runner, model.specs)
        self.assertEqual(runner.calls, [])

    def test_receipt_does_not_disclose_generated_source_or_test_diagnostics(self):
        result, _, _, _ = self.compare({"a": response()}, specs=self.specs[:1])
        serialized = json.dumps(result)
        self.assertNotIn("return 7", serialized)
        self.assertNotIn("PRIVATE_TEST", serialized)
        self.assertEqual(result["quality_scope"], "one_task_artifact_not_global_model_promotion")
        self.assertEqual(result["test_scope"], "same_acceptance_tests_on_fresh_workspace_not_protected_holdout")
        result["selected_candidate_id"] = "forged"
        self.assertFalse(vda.verified(result))

    def test_cli_default_dry_run_starts_no_container_or_generation(self):
        sample = Path(vda.__file__).parent / "samples/sliding-window/task.json"
        run = subprocess.run([sys.executable, str(Path(c.__file__)), "--task", str(sample)],
                             capture_output=True, text=True, timeout=10)
        self.assertEqual(run.returncode, 0, run.stderr)
        value = json.loads(run.stdout)
        self.assertFalse(value["generation_performed"])
        self.assertFalse(value["containers_started"])

    def test_cli_refuses_existing_selected_artifact_instead_of_leaving_stale_success(self):
        sample = Path(vda.__file__).parent / "samples/sliding-window/task.json"
        selected = self.root / "selected.json"
        selected.write_text('{"old":"artifact"}')
        receipt = self.root / "receipt.json"
        stdout = io.StringIO()
        with patch.object(c.vda, "DockerRunner") as runner, contextlib.redirect_stdout(stdout):
            result = c.main(["--task", str(sample), "--captures", str(self.root / "missing.json"),
                             "--execute", "--receipt", str(receipt), "--selected", str(selected)])
            runner.assert_not_called()
        self.assertEqual(result, 2)
        self.assertEqual(json.loads(stdout.getvalue())["error_code"], "FRESH_OUTPUT_PATH_REQUIRED")
        self.assertEqual(selected.read_text(), '{"old":"artifact"}')
        self.assertFalse(receipt.exists())


if __name__ == "__main__":
    unittest.main()
