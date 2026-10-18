"""Pure protocol tests: no model, credentials, git dispatch or candidate execution."""
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import live_dev_bridge as bridge


class OwnerUpperModelTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.client = self.root / 'MusicJapan/FrontierRuntime/upper_client.py'
        self.client.parent.mkdir(parents=True)
        self.client.write_text('# synthetic trusted client; never executed\n', encoding='utf-8')
        self.digest = hashlib.sha256(self.client.read_bytes()).hexdigest()
        self.env = mock.patch.dict(os.environ, {'LOCALAPPDATA': str(self.root)})
        self.env.start(); self.addCleanup(self.env.stop)

    def model(self, **kwargs):
        self.assertTrue(hasattr(bridge, 'OwnerUpperModel'), 'owner upper-model adapter missing')
        with mock.patch.object(bridge, 'UPPER_CLIENT_SHA256', self.digest):
            return bridge.OwnerUpperModel(self.client, self.root / 'evidence', **kwargs)

    def receipt(self, argv):
        prompt = Path(argv[argv.index('--prompt-file') + 1]).read_bytes().decode('utf-8-sig')
        model = argv[argv.index('--model') + 1]
        profile = bridge.UPPER_PROFILES[model]
        return {'schema': 'mj.upper-client.receipt/1.0',
                'started_at': '2026-10-18T10:00:00+00:00', 'completed_at': '2026-10-18T10:00:01+00:00',
                'elapsed_seconds': 1.0, 'requested_model': model, 'actual_model': profile['upstream'],
                'source_model': profile['source'], 'upstream_origin': profile['origin'],
                'identity_basis': 'upstream_reported_not_byte_verified', 'weight_bytes_verified': False,
                'independent_quality_verification': False, 'cost_mode': 'public_demo',
                'company_credentials_sent_upstream': False, 'paid_fallback': False,
                'requested_reasoning_effort': argv[argv.index('--effort') + 1],
                'reasoning_effort_attested': False, 'requested_max_tokens': int(argv[argv.index('--max-tokens') + 1]),
                'response_format_requested': 'json_object', 'finish_reason': 'stop',
                'usage': {'prompt_tokens': 40, 'completion_tokens': 80, 'total_tokens': 120},
                'usage_basis': 'upstream_reported', 'prompt_sha256': hashlib.sha256(prompt.encode()).hexdigest(),
                'public_input_confirmed': True,
                'final_content': '{"edits":[{"path":"retry_budget.py","content":"# synthetic proposal only\\n"}]}'}

    def fake_client(self, mutation=None, seen=None):
        def run(argv, **kwargs):
            if seen is not None:
                seen.append((argv, kwargs))
            data = self.receipt(argv)
            if mutation:
                mutation(data)
            Path(argv[argv.index('--out') + 1]).write_text(json.dumps(data), encoding='utf-8')
            return SimpleNamespace(returncode=0, stdout='ignored untrusted child stdout', stderr='not exported')
        return run

    def test_explicit_pinned_client_xhigh_json_receipt_and_cleanup(self):
        model = self.model()
        seen = []
        with mock.patch.object(bridge.subprocess, 'run', side_effect=self.fake_client(seen=seen)):
            result = model.generate('mj-flash-next', 'public synthetic source and issue')
        self.assertEqual(json.loads(result)['edits'][0]['path'], 'retry_budget.py')
        argv, options = seen[0]
        self.assertIn('--public', argv)
        self.assertEqual(argv[argv.index('--effort') + 1], 'xhigh')
        self.assertEqual(argv[argv.index('--response-format') + 1], 'json_object')
        self.assertFalse(options['shell'])
        self.assertFalse(Path(argv[argv.index('--prompt-file') + 1]).exists())
        capture = json.loads((self.root / 'evidence/generation-1.json').read_text())
        self.assertEqual(capture['client_receipt']['source_model'], 'Qwen/Qwen3.8-Flash-Next')
        self.assertFalse(capture['independent_quality_verification'])
        self.assertNotIn('ignored untrusted', json.dumps(capture))
        self.assertEqual(model.calls[0]['status'], 'complete')

    def test_multiline_unicode_prompt_keeps_exact_wire_bytes(self):
        model = self.model()
        prompt = 'Public example\nUnicode: 日本語\r\nExisting CRLF stays intact\n'
        fake = self.fake_client()
        def run(argv, **kwargs):
            raw = Path(argv[argv.index('--prompt-file') + 1]).read_bytes()
            self.assertEqual(raw, prompt.encode('utf-8'))
            return fake(argv, **kwargs)
        with mock.patch.object(bridge.subprocess, 'run', side_effect=run):
            model.generate('mj-flash-next', prompt)
        capture = json.loads((self.root / 'evidence/generation-1.json').read_text())
        self.assertEqual(capture['client_receipt']['prompt_sha256'],
                         hashlib.sha256(prompt.encode('utf-8')).hexdigest())

    def test_substitution_paid_unknown_fields_hash_and_truncation_rejected(self):
        changes = [('actual_model', 'fake-astra'), ('paid_fallback', True),
                   ('prompt_sha256', '0' * 64), ('finish_reason', 'length'),
                   ('requested_reasoning_effort', 'low'), ('company_credentials_sent_upstream', True),
                   ('usage', {'prompt_tokens': 40, 'completion_tokens': 80, 'total_tokens': 999}),
                   ('authorization', 'synthetic-secret-should-not-be-captured')]
        for key, value in changes:
            with self.subTest(key=key):
                model = self.model(effort='xhigh')
                with mock.patch.object(bridge.subprocess, 'run',
                        side_effect=self.fake_client(lambda data: data.update({key: value}))), \
                        self.assertRaises(ValueError):
                    model.generate('mj-flash-next', 'public')
                self.assertEqual(model.calls[-1]['status'], 'failed')
                self.assertNotIn('synthetic-secret', json.dumps(model.calls))

    def test_disallowed_file_replacements_and_duplicate_json_are_rejected(self):
        for text in ('{"edits":[{"path":"tests/test_retry_budget.py","content":"pass"}]}',
                     '{"edits":[],"edits":[]}'):
            model = self.model()
            with mock.patch.object(bridge.subprocess, 'run',
                    side_effect=self.fake_client(lambda data: data.update(final_content=text))), \
                    self.assertRaises(ValueError):
                model.generate('mj-flash-next', 'public')

    def test_budget_at_most_initial_and_one_repair(self):
        model = self.model()
        with mock.patch.object(bridge.subprocess, 'run', side_effect=self.fake_client()) as run:
            model.generate('mj-flash-next', 'public')
            model.generate('mj-flash-next', 'public repair feedback')
            with self.assertRaisesRegex(ValueError, 'BUDGET'):
                model.generate('mj-flash-next', 'third call forbidden')
        self.assertEqual(run.call_count, 2)

    def test_explicit_deepseek_profile_keeps_its_actual_provenance(self):
        model = self.model(model='mj-deepseek-v4-flash', effort='high')
        with mock.patch.object(bridge.subprocess, 'run', side_effect=self.fake_client()):
            model.generate('mj-deepseek-v4-flash', 'public task')
        self.assertEqual(model.calls[0]['source_model'], 'deepseek-ai/DeepSeek-V4-Flash-0731')
        with self.assertRaises(ValueError):
            self.model(model='mj-deepseek-v4-flash', effort='xhigh')

    def test_failure_reason_is_bounded_without_exporting_child_text(self):
        model = self.model()
        done = SimpleNamespace(returncode=2, stdout='', stderr='MusicJapan upper API: Response is incomplete, mismatched, or invalid for the requested format\n')
        with mock.patch.object(bridge.subprocess, 'run', return_value=done), self.assertRaises(ValueError):
            model.generate('mj-flash-next', 'public')
        self.assertEqual(model.calls[-1]['reason'], 'UPPER_RESPONSE_INCOMPLETE_OR_INVALID')
        model = self.model()
        with mock.patch.object(bridge.subprocess, 'run', side_effect=RuntimeError('private text must never be exported')), self.assertRaises(ValueError):
            model.generate('mj-flash-next', 'public')
        self.assertEqual(model.calls[-1]['reason'], 'UPPER_RESPONSE_PROCESSING_FAILED')
        self.assertNotIn('private text', json.dumps(model.calls))

    def test_registered_renderer_only_accepts_its_own_target(self):
        model = self.model(task_id='redline-analysis-html-report')
        renderer = json.dumps({'edits': [{'path': 'render_analysis.py', 'content': '# proposal only\n'}]})
        with mock.patch.object(bridge.subprocess, 'run',
                side_effect=self.fake_client(lambda data: data.update(final_content=renderer))):
            self.assertEqual(model.generate('mj-flash-next', 'public renderer task'), renderer)
        with mock.patch.object(bridge.subprocess, 'run', side_effect=self.fake_client()), self.assertRaises(ValueError):
            model.generate('mj-flash-next', 'wrong task target')
        with mock.patch.object(bridge.subprocess, 'run') as run, self.assertRaises(ValueError):
            self.model(task_id='../unregistered')
        run.assert_not_called()

    def test_client_pin_profile_effort_and_input_bounds_fail_before_subprocess(self):
        with self.assertRaises(ValueError):
            bridge.OwnerUpperModel(self.client, self.root / 'evidence')
        with self.assertRaises(ValueError):
            self.model(effort='high')
        model = self.model()
        with mock.patch.object(bridge.subprocess, 'run') as run:
            for profile, prompt in [('gpt-6-astra', 'public'), ('mj-flash-next', '日' * 12000)]:
                with self.assertRaises(ValueError):
                    model.generate(profile, prompt)
            run.assert_not_called()

    def test_cli_dry_run_never_starts_model_or_cloud(self):
        with mock.patch.object(bridge, 'OwnerUpperModel') as model, \
                mock.patch.object(bridge, 'GitHubRunner') as runner, \
                contextlib.redirect_stdout(io.StringIO()) as stdout:
            self.assertEqual(bridge.main(['--model', 'mj-flash-next', '--effort', 'xhigh']), 0)
        self.assertEqual(json.loads(stdout.getvalue())['model'], 'mj-flash-next')
        model.assert_not_called(); runner.assert_not_called()

    def test_real_orchestrator_one_repair_and_fresh_recheck_with_fake_cloud(self):
        model = self.model()
        task = bridge.v.load_task(Path(bridge.__file__).parent / bridge.TASKS['sliding-window-retry-budget'])
        checked = []
        diagnostics = ['BASELINE_BROKEN', 'CANDIDATE_BOUNDARY_FAILURE', 'CANDIDATE_OK', 'RECHECK_OK']
        class FakeCloud:
            evidence_name = 'test_fake_cloud_no_execution'
            def check(inner, workspace):
                checked.append(workspace)
                index = len(checked) - 1
                passed = index >= 2
                return bridge.v.Check(passed, 0 if passed else 1, .1, diagnostics[index], 4)
        seen = []
        with mock.patch.object(bridge.subprocess, 'run', side_effect=self.fake_client(seen=seen)):
            result, edits = bridge.v.execute(task, model, FakeCloud(), ['mj-flash-next'], 2)
        self.assertEqual(result['status'], 'PASS_VERIFIED')
        self.assertEqual(len(checked), 4)
        self.assertEqual(len(model.calls), 2)
        self.assertEqual([item['tests_passed'] for item in result['candidate_attempts']], [False, True])
        self.assertTrue(result['independent_recheck']['passed'])
        self.assertEqual(edits[0]['path'], 'retry_budget.py')
        self.assertEqual(len(seen), 2)
        self.assertEqual(result['evidence_type'], 'reproducible_executable_tests_not_frontier_benchmark')

    def test_explicit_execution_reuses_registered_task_runner_and_two_attempts(self):
        model = SimpleNamespace(calls=[])
        runner = SimpleNamespace(receipts=[])
        out = self.root / 'result'
        with mock.patch.object(bridge, 'OwnerUpperModel', return_value=model) as factory, \
                mock.patch.object(bridge, 'GitHubRunner', return_value=runner), \
                mock.patch.object(bridge.v, 'execute', return_value=({'status': 'NO_VERIFIED_FIX'}, None)) as execute, \
                contextlib.redirect_stdout(io.StringIO()):
            result = bridge.main(['--execute', '--public', '--client', str(self.client),
                         '--model', 'mj-flash-next', '--effort', 'xhigh', '--attempts', '2',
                         '--branch', 'runs/vda/public-upper-review', '--output', str(out)])
        self.assertEqual(result, 2)
        self.assertEqual(execute.call_args.args[0].task_id, 'sliding-window-retry-budget')
        self.assertEqual(execute.call_args.args[1:], (model, runner, ['mj-flash-next'], 2))
        self.assertEqual(factory.call_args.kwargs['effort'], 'xhigh')
        self.assertEqual(json.loads((out / 'receipt.json').read_text())['status'], 'NO_VERIFIED_FIX')


if __name__ == '__main__':
    unittest.main()
