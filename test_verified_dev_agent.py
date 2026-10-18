"""All mocked inference tests are offline and never call paid/proprietary models."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import verified_dev_agent as vda

SAMPLE = Path(__file__).parent / "samples" / "sliding-window"


class FakeModel:
    def __init__(self, replies):
        self.replies = iter(replies)
        self.calls = []

    def generate(self, model_name, prompt):
        self.calls.append((model_name, prompt))
        return next(self.replies)


class ContentRunner:
    """Fake test-runner for verifying CEGIS control flow, not a model/benchmark."""
    def __init__(self, successful_fragment="threading.Lock"):
        self.calls = 0
        self.successful_fragment = successful_fragment

    def check(self, workspace):
        self.calls += 1
        good = self.successful_fragment in (workspace / "retry_budget.py").read_text("utf-8")
        return vda.Check(good, 0 if good else 1, 0.1, "FAILED: test_capacity" if not good else "OK", 4)


class RunnerError:
    def check(self, workspace):
        return vda.Check(False, 125, .01, "DOCKER_UNAVAILABLE", None)


class AlreadyGood:
    def check(self, workspace):
        return vda.Check(True, 0, .01, "OK", 4)


class VerifiedAgentTests(unittest.TestCase):
    def setUp(self):
        self.task = vda.load_task(SAMPLE / "task.json")

    def test_fixture_has_real_behavioral_regression(self):
        proc = subprocess.run([sys.executable, "-B", "-m", "unittest", "discover", "-s", "tests", "-p", "test_*.py"],
                              cwd=self.task.workspace, capture_output=True, text=True, timeout=15)
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("FAILED", proc.stderr)

    def test_task_does_not_include_test_sources_in_model_prompt(self):
        question = vda.prompt_for(self.task)
        self.assertIn("try_acquire", question)
        self.assertNotIn("def test_capacity", question)
        self.assertNotIn("class RollingBudgetAcceptance", question)

    def test_task_validation_blocks_absolute_and_traversal(self):
        for bad in ("/etc/passwd", "../db.env", "a/../../a.py", "C:/x.py", "a\\b.py", "a//b.py", "./src.py", "readme.sh", "a\x00b.py"):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                vda.valid_path(bad)

    def test_unapproved_modifications_are_denied(self):
        for payload in [
            {"edits": [{"path": "tests/test_retry_budget.py", "content": "pass"}]},
            {"edits": [{"path": "../secrets.txt", "content": "pass"}]},
            {"edits": [{"path": "retry_budget.py", "content": "x"}, {"path": "retry_budget.py", "content": "y"}]},
            {"edits": [{"path": "retry_budget.py", "content": ""}]},
            {"edits": [{"path": "retry_budget.py", "content": "x"}], "override": True},
        ]:
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                vda.parse_edits(json.dumps(payload), self.task.targets)

    def test_no_markdown_or_code_injection_in_agent_response(self):
        with self.assertRaises(ValueError):
            vda.parse_edits('```json\n{"edits":[]}\n```', self.task.targets)

    def test_repair_loop_can_select_evidence_verified_candidate(self):
        client = FakeModel([
            json.dumps({"edits": [{"path": "retry_budget.py", "content": "# attempt 1\n"}]}),
            json.dumps({"edits": [{"path": "retry_budget.py", "content": "import threading\nlock = threading.Lock()\n"}]}),
        ])
        runner = ContentRunner()
        receipt, edits = vda.execute(self.task, client, runner, ["local-oss-a", "local-oss-b"], attempts=2)
        self.assertEqual(receipt["status"], "PASS_VERIFIED")
        self.assertEqual(receipt["selected_profile"], "local-oss-b")
        self.assertEqual(runner.calls, 4)  # baseline + 2 candidates + independent recheck
        self.assertTrue(vda.verified(receipt))
        self.assertEqual(receipt["selected_edits_sha256"], vda.sha(edits))
        self.assertIn("UNTRUSTED TEST FAILURE TEXT", client.calls[1][1])
        self.assertEqual(receipt["direct_vendor_api_calls_by_agent"], 0)
        self.assertEqual(receipt["upstream_billing_provenance"], "unknown_localhost_endpoint")
        self.assertFalse(receipt["own_model_weights_claimed"])
        # No private source or repair details stored in the publication-safe receipt.
        self.assertNotIn("threading.Lock", json.dumps(receipt))

    def test_failure_is_never_described_as_completed_patch(self):
        candidate = json.dumps({"edits": [{"path": "retry_budget.py", "content": "# still broken\n"}]})
        receipt, edits = vda.execute(self.task, FakeModel([candidate, candidate]), ContentRunner(), ["local"], attempts=2)
        self.assertEqual(receipt["status"], "NO_VERIFIED_FIX")
        self.assertIsNone(edits)
        self.assertIsNone(receipt["selected_edits_sha256"])

    def test_baseline_already_passes_is_not_a_fabricated_improvement(self):
        receipt, edits = vda.execute(self.task, FakeModel([]), AlreadyGood(), ["local"])
        self.assertEqual(receipt["status"], "NO_FAILING_BASELINE")
        self.assertIsNone(edits)
        self.assertEqual(receipt["candidate_attempts"], [])

    def test_container_failure_aborts_without_asking_model(self):
        receipt, edits = vda.execute(self.task, FakeModel([]), RunnerError(), ["local"])
        self.assertEqual(receipt["status"], "RUNNER_UNAVAILABLE_OR_UNTRUSTWORTHY")
        self.assertIsNone(edits)

    def test_bad_model_output_recorded_as_failure_not_pass(self):
        bad = FakeModel(['not json'])
        receipt, edits = vda.execute(self.task, bad, ContentRunner(), ["local"], attempts=1)
        self.assertEqual(receipt["status"], "NO_VERIFIED_FIX")
        self.assertFalse(receipt["candidate_attempts"][0]["tests_passed"])
        self.assertIsNone(edits)

    def test_receipt_tamper_is_detected(self):
        receipt, _ = vda.execute(self.task, FakeModel([]), AlreadyGood(), ["local"])
        self.assertTrue(vda.verified(receipt))
        receipt["status"] = "PASS_VERIFIED"
        self.assertFalse(vda.verified(receipt))

    def test_docker_test_contract_enforces_isolation(self):
        cmd = vda.DockerRunner().command(self.task.workspace)
        for required in ("--network", "none", "--read-only", "--cap-drop", "ALL", "no-new-privileges", "--pull", "never", "--memory", "512m", "--user", "65534:65534"):
            self.assertIn(required, cmd)
        self.assertNotIn("sh", cmd)
        self.assertNotIn("--privileged", cmd)
        self.assertNotIn("/var/run/docker.sock", " ".join(cmd))

    def test_local_http_only_no_upstream_vendor(self):
        import inspect
        source = inspect.getsource(vda.LocalModel)
        self.assertIn('HTTPConnection("127.0.0.1"', source)
        self.assertNotIn("api.openai.com", source)
        self.assertNotIn("api.anthropic.com", source)
        with self.assertRaises(ValueError):
            vda.LocalModel(443)

    def test_workspace_symlink_denied(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            try:
                (p / "link.py").symlink_to(self.task.workspace / "retry_budget.py")
            except OSError as exc:
                if getattr(exc, "winerror", None) == 1314:
                    self.skipTest("Windows account cannot create symlinks; covered in Linux CI")
                raise
            with self.assertRaises(ValueError):
                vda.audit_workspace(p)

    def test_dry_run_has_no_model_or_docker_cost(self):
        cmd = [sys.executable, str(Path(vda.__file__)), "--task", str(SAMPLE / "task.json"),
               "--models", "local-qwen", "--dry-run"]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        payload = json.loads(proc.stdout)
        self.assertFalse(payload["models_called"])
        self.assertFalse(payload["containers_started"])


if __name__ == "__main__":
    unittest.main()
