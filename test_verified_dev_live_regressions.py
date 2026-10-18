"""Regressions for issues found before the first real code-repair run."""
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch
import verified_dev_agent as v
from test_verified_dev_agent import FakeModel, ContentRunner, RunnerError, SAMPLE


class LiveRegressions(unittest.TestCase):
    def setUp(self):
        self.task = v.load_task(SAMPLE / 'task.json')

    def test_missing_baseline_test_count_prevents_inference(self):
        class NoCount:
            def check(self, workspace):
                return v.Check(False, 1, .1, 'Import or infrastructure error', None)
        model = FakeModel([])
        record, edits = v.execute(self.task, model, NoCount(), ['local'], 1)
        self.assertEqual(record['status'], 'RUNNER_UNAVAILABLE_OR_UNTRUSTWORTHY')
        self.assertEqual(model.calls, [])
        self.assertIsNone(edits)

    def test_retry_contains_previous_candidate_source(self):
        first = json.dumps({'edits': [{'path': 'retry_budget.py', 'content': '# PREVIOUS_CANDIDATE_42\n'}]})
        second = json.dumps({'edits': [{'path': 'retry_budget.py', 'content': 'import threading\nlock = threading.Lock()\n'}]})
        model = FakeModel([first, second])
        result, _ = v.execute(self.task, model, ContentRunner(), ['local'], 2)
        self.assertEqual(result['status'], 'PASS_VERIFIED')
        self.assertIn('PREVIOUS_CANDIDATE_42', model.calls[1][1])
        self.assertIn('FAILED', model.calls[1][1])

    def test_task_digest_includes_tests(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / 'sample'
            shutil.copytree(SAMPLE, root)
            task = v.load_task(root / 'task.json')
            a, _ = v.execute(task, FakeModel([]), RunnerError(), ['local'], 1)
            test = next((task.workspace / 'tests').glob('test_*.py'))
            test.write_text(test.read_text('utf-8') + '\n# New independent acceptance condition\n', 'utf-8')
            b, _ = v.execute(task, FakeModel([]), RunnerError(), ['local'], 1)
            self.assertNotEqual(a['task_sha256'], b['task_sha256'])

    def test_selected_candidate_is_rechecked_independently(self):
        class Flaky:
            def __init__(self): self.calls = 0
            def check(self, workspace):
                self.calls += 1
                ok = self.calls == 2
                return v.Check(ok, 0 if ok else 1, .1, 'OK' if ok else 'FAILED', 4)
        model = FakeModel([json.dumps({'edits': [{'path': 'retry_budget.py', 'content': '# repair\n'}]})])
        runner = Flaky()
        result, edits = v.execute(self.task, model, runner, ['local'], 1)
        self.assertEqual(runner.calls, 3)
        self.assertEqual(result['status'], 'FINAL_RECHECK_FAILED')
        self.assertIsNone(edits)

    def test_docker_timeout_removes_named_container(self):
        runner = v.DockerRunner(timeout=5)
        def invoke(command, **kwargs):
            if command[:2] == ['docker', 'run']:
                self.assertIn('--name', command)
                raise subprocess.TimeoutExpired(command, 5)
            self.assertEqual(command[:3], ['docker', 'rm', '-f'])
            return subprocess.CompletedProcess(command, 0, '', '')
        with patch.object(v.subprocess, 'run', side_effect=invoke) as mocked:
            checked = runner.check(self.task.workspace)
        self.assertFalse(checked.passed)
        self.assertEqual(checked.diagnostic, 'TEST_TIMEOUT')
        self.assertEqual(mocked.call_count, 2)
        run_cmd = mocked.call_args_list[0].args[0]
        rm_cmd = mocked.call_args_list[1].args[0]
        self.assertEqual(run_cmd[run_cmd.index('--name') + 1], rm_cmd[-1])

    def test_workspace_rejects_fifo_and_other_special_files(self):
        if not hasattr(__import__('os'), 'mkfifo'):
            self.skipTest('POSIX special-file test is covered in Linux CI')
        import os
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            os.mkfifo(root / 'input.pipe')
            with self.assertRaisesRegex(ValueError, 'SPECIAL'):
                v.audit_workspace(root)


if __name__ == '__main__':
    unittest.main()
