"""Offline contracts: no provider calls or credentials are used by these tests."""
import contextlib
import io
import json
import os
from pathlib import Path
import stat
import subprocess
import tempfile
import unittest
from unittest.mock import Mock, patch

import free_cloud_inference as f


MODEL = "qwen-flash-next"
ID = f.MODELS[MODEL]["id"]


def event(content="", finish=None, **fields):
    value = {"id": "request-example", "model": ID,
             "choices": [{"index": 0, "delta": {"content": content}, "finish_reason": finish}]}
    value.update(fields)
    return ("data: " + json.dumps(value) + "\n").encode()


class StreamTests(unittest.TestCase):
    def test_complete_answer_keeps_exact_text_usage_and_identity(self):
        value = f.parse_sse([event(" answer "), event(finish="stop", usage={"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5}), b"data: [DONE]\n"], ID)
        self.assertEqual(value["raw_answer"], " answer ")
        self.assertEqual(value["usage"]["total_tokens"], 5)
        self.assertEqual(value["completion_id"], "request-example")

    def test_rejects_model_switch_id_conflict_and_bad_usage(self):
        cases = [(event("x", model="different"), "REPORTED_MODEL_MISMATCH"),
                 (event("x", id="another-id"), "SSE_ID_CONFLICT"),
                 (event(usage={"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 6}), "USAGE_TOTAL_MISMATCH"),
                 (event(usage={"prompt_tokens": True, "completion_tokens": 3, "total_tokens": 4}), "USAGE_INVALID")]
        for bad, code in cases:
            with self.subTest(code=code), self.assertRaisesRegex(f.InferenceError, code):
                f.parse_sse([event("a"), bad, event(finish="stop"), b"data: [DONE]\n"], ID)

    def test_stop_done_and_final_content_all_required(self):
        cases = [[event("x", finish="length"), b"data: [DONE]\n"],
                 [event("x", finish="stop")], [event("x"), b"data: [DONE]\n"],
                 [event(finish="stop"), b"data: [DONE]\n"],
                 [event("x", finish="stop"), event("late"), b"data: [DONE]\n"]]
        for lines in cases:
            with self.subTest(lines=lines), self.assertRaises(f.InferenceError):
                f.parse_sse(lines, ID)

    def test_duplicate_json_keys_and_stream_cap_rejected(self):
        with self.assertRaisesRegex(f.InferenceError, "SSE_INVALID"):
            f.parse_sse([b'data: {"model":"x","model":"y"}\n'], ID)
        with self.assertRaisesRegex(f.InferenceError, "SSE_LIMIT"):
            f.parse_sse([b"x" * (f.MAX_LINE_BYTES + 1)], ID)

    def test_truncated_or_unfinished_output_is_retained_as_failed_evidence(self):
        for lines, code in [([event("partial", finish="length")], "FINISH_REASON_NOT_STOP"),
                            ([event("partial", finish="stop")], "SSE_UNFINISHED")]:
            with self.subTest(code=code), self.assertRaisesRegex(f.InferenceError, code) as caught:
                f.parse_sse(lines, ID)
            self.assertEqual(caught.exception.evidence["raw_answer"], "partial")
            self.assertEqual(caught.exception.evidence["completion_id"], "request-example")


class ExecutionTests(unittest.TestCase):
    def test_fresh_challenge_and_strict_grading(self):
        prompt, expected = f.fresh_challenge()
        self.assertRegex(expected["nonce"], r"^[0-9a-f]{16}$")
        self.assertIn(expected["nonce"], prompt)
        self.assertNotEqual(expected["nonce"], f.fresh_challenge()[1]["nonce"])
        self.assertTrue(f.grade(json.dumps(expected), expected)["passed"])
        wrong = dict(expected, nonce="0" * 16)
        self.assertFalse(f.grade(json.dumps(wrong), expected)["passed"])
        self.assertFalse(f.grade("```json\n" + json.dumps(expected) + "\n```", expected)["passed"])
        self.assertFalse(f.grade(json.dumps(dict(expected, product=True)), expected)["passed"])

    def test_catalog_gate_prevents_inference_and_no_retry_on_429(self):
        transport = Mock()
        transport.request.return_value = {"data": [{"id": "wrong"}]}
        with self.assertRaisesRegex(f.InferenceError, "CATALOG_MODEL_MISMATCH") as caught:
            f.perform(MODEL, "public example", 128, False, transport=transport)
        self.assertEqual(transport.request.call_count, 1)
        self.assertFalse(caught.exception.evidence["inference_attempted"])
        self.assertEqual(caught.exception.evidence["inference_requests_attempted"], 0)
        transport.reset_mock()
        transport.request.side_effect = [{"data": [{"id": ID}]}, f.InferenceError("UPSTREAM_HTTP_429")]
        with self.assertRaisesRegex(f.InferenceError, "UPSTREAM_HTTP_429"):
            f.perform(MODEL, "public example", 128, False, transport=transport)
        self.assertEqual(transport.request.call_count, 2)

    def test_expiry_wins_over_late_success(self):
        transport = Mock()
        transport.request.side_effect = [{"data": [{"id": ID}]}, {"raw_answer": "late"}]
        with self.assertRaisesRegex(f.InferenceError, "DEADLINE"):
            f.perform(MODEL, "public example", 128, False, transport=transport,
                      clock=Mock(side_effect=[0, 0, 1, f.TOTAL_SECONDS]))

    def test_reported_completion_limit_boundary_and_missing_usage(self):
        for tokens in (None, 2, 3):
            with self.subTest(completion_tokens=tokens):
                usage = None if tokens is None else {"prompt_tokens": 5, "completion_tokens": tokens, "total_tokens": 5 + tokens}
                response = f.parse_sse([event("answer", finish="stop", usage=usage), b"data: [DONE]\n"], ID)
                transport = Mock()
                transport.request.side_effect = [{"data": [{"id": ID}]}, response]
                if tokens is not None and tokens > 2:
                    with self.assertRaisesRegex(f.InferenceError, "REPORTED_OUTPUT_LIMIT_EXCEEDED") as caught:
                        f.perform(MODEL, "public example", 2, False, transport=transport)
                    result = caught.exception.evidence
                    self.assertEqual(result["usage"]["completion_tokens"], 3)
                    self.assertEqual(result["raw_answer"], "answer")
                else:
                    result = f.perform(MODEL, "public example", 2, False, transport=transport)
                self.assertEqual(result["completion_token_count_observed"], tokens is not None)
                self.assertEqual(result["completion_token_count_basis"], "not_reported" if tokens is None else "provider_usage")
                self.assertEqual(result["inference_requests_attempted"], 1)
                self.assertEqual(transport.request.call_count, 2)

    def test_two_explicit_models_but_no_arbitrary_host_or_budget(self):
        for model in f.MODELS:
            body = f.build_request(model, "public example", 384)
            self.assertEqual(body["model"], f.MODELS[model]["id"])
            self.assertEqual(body["max_tokens"], 384)
        for model, prompt, tokens in [("other", "example", 1), (MODEL, "a" * 8193, 1), (MODEL, "example", 385), (MODEL, "example", True)]:
            with self.assertRaises(f.InferenceError):
                f.build_request(model, prompt, tokens)
        with self.assertRaisesRegex(f.InferenceError, "TRANSPORT_SCOPE"):
            f.Transport().request(MODEL, "https://other.example/", timeout=1)

    def test_network_child_deadline_kills_and_waits_without_ambient_secrets(self):
        child = Mock()
        child.communicate.side_effect = [subprocess.TimeoutExpired("fixed", 1), (b"", b"")]
        child.poll.return_value = None
        with patch.object(f.subprocess, "Popen", return_value=child) as launch:
            with self.assertRaisesRegex(f.InferenceError, "DEADLINE"):
                f.isolated_call(MODEL, "public example", 128, False, deadline=f.time.monotonic() + 2)
        child.kill.assert_called_once()
        self.assertEqual(child.communicate.call_count, 2)
        self.assertEqual(launch.call_args.kwargs["env"], {})
        self.assertNotIn("shell", launch.call_args.kwargs)


class CliTests(unittest.TestCase):
    def run_main(self, args):
        with contextlib.redirect_stdout(io.StringIO()):
            return f.main(args)

    def test_default_plan_no_network_or_prompt_reads_or_receipt(self):
        with patch.object(f, "isolated_call") as call, patch.object(f, "read_prompt") as read:
            self.assertEqual(self.run_main(["--prompt-file", "/not/read", "--receipt", "/not/written"]), 0)
            call.assert_not_called()
            read.assert_not_called()

    def test_explicit_public_required_and_error_receipt_is_0600_exclusive(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "result.json"
            args = ["--execute", "--verify", "--receipt", str(path)]
            with patch.object(f, "isolated_call") as call:
                self.assertEqual(self.run_main(args), 2)
                call.assert_not_called()
            original = path.read_bytes()
            self.assertEqual(json.loads(original)["error_code"], "PUBLIC_INPUT_CONFIRMATION_REQUIRED")
            if os.name != "nt":
                self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)
            self.assertEqual(self.run_main(args), 2)
            self.assertEqual(path.read_bytes(), original)

    def test_verification_records_actual_answer_expected_grade_and_identity(self):
        def fake(model, prompt, tokens, verify, *, deadline):
            self.assertTrue(verify)
            expected = {"nonce": "0123456789abcdef", "product": 123 * 456, "reverse": "tibro"}
            return {"raw_answer": json.dumps(expected), "reported_model": ID,
                    "finish_reason": "stop", "completion_id": "fresh-id", "usage": None}
        expected = {"nonce": "0123456789abcdef", "product": 123 * 456, "reverse": "tibro"}
        with tempfile.TemporaryDirectory() as folder, patch.object(f, "fresh_challenge", return_value=("public test", expected)), patch.object(f, "isolated_call", side_effect=fake) as call:
            path = Path(folder) / "result.json"
            self.assertEqual(self.run_main(["--execute", "--public", "--verify", "--receipt", str(path)]), 0)
            result = json.loads(path.read_text())
            self.assertEqual(result["status"], "GENERATED_AND_GRADED")
            self.assertEqual(json.loads(result["raw_answer"]), expected)
            self.assertEqual(result["completion_id"], "fresh-id")
            self.assertFalse(result["weight_bytes_attested"])
            self.assertIsNone(result["usage"])
            self.assertFalse(result["completion_token_count_observed"])
            self.assertEqual(result["completion_token_count_basis"], "not_reported")
            call.assert_called_once()

    def test_worker_output_limit_failure_preserves_usage_in_failed_receipt(self):
        response = f.parse_sse([event("received answer", finish="stop", usage={"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8}), b"data: [DONE]\n"], ID)
        transport = Mock()
        transport.request.side_effect = [{"data": [{"id": ID}]}, response]
        with self.assertRaisesRegex(f.InferenceError, "REPORTED_OUTPUT_LIMIT_EXCEEDED") as caught:
            f.perform(MODEL, "public example", 2, False, transport=transport)
        child = Mock()
        child.returncode = child.poll.return_value = 0
        child.communicate.return_value = (json.dumps({"error_code": str(caught.exception), "evidence": caught.exception.evidence}).encode(), b"")
        with tempfile.TemporaryDirectory() as folder, patch.object(f.subprocess, "Popen", return_value=child) as launch:
            path = Path(folder) / "result.json"
            self.assertEqual(self.run_main(["--execute", "--public", "--verify", "--max-tokens", "2", "--receipt", str(path)]), 2)
            result = json.loads(path.read_text())
        self.assertEqual(result["status"], "FAILED")
        self.assertEqual(result["error_code"], "REPORTED_OUTPUT_LIMIT_EXCEEDED")
        self.assertEqual(result["requested_output_tokens"], 2)
        self.assertEqual(result["usage"]["completion_tokens"], 3)
        self.assertEqual(result["raw_answer"], "received answer")
        self.assertTrue(result["completion_token_count_observed"])
        self.assertEqual(result["completion_token_count_basis"], "provider_usage")
        self.assertEqual(result["inference_requests_attempted"], 1)
        launch.assert_called_once()

    def test_failure_receipt_and_no_second_call(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(f, "isolated_call", side_effect=f.InferenceError("UPSTREAM_HTTP_503")) as call:
            path = Path(folder) / "result.json"
            self.assertEqual(self.run_main(["--execute", "--public", "--verify", "--receipt", str(path)]), 2)
            self.assertEqual(json.loads(path.read_text())["error_code"], "UPSTREAM_HTTP_503")
            call.assert_called_once()


if __name__ == "__main__":
    unittest.main()
