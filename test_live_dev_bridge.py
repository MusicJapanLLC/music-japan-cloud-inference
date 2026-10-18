"""Protocol tests use synthetic responses; live generation is separately recorded."""
import dataclasses
import json
from pathlib import Path
import shutil
import tempfile
import unittest
import verified_dev_agent as v
import live_dev_bridge as b

ROOT = Path(__file__).parent


def response(content='{"edits":[{"path":"retry_budget.py","content":"# candidate\\n"}]}'):
    return {'model': b.MODEL, 'choices': [{'finish_reason': 'stop', 'message': {'content': content}}],
            'usage': {'completion_tokens': 20}}


class ResponseTests(unittest.TestCase):
    def test_preserve_valid_json(self):
        text, info = b.decode_response(response(), b.MODEL)
        self.assertTrue(info['strict_json'])
        self.assertEqual(json.loads(text)['edits'][0]['path'], 'retry_budget.py')

    def test_outer_json_fence_is_explicitly_recorded(self):
        text, info = b.decode_response(response('```json\n{"edits":[]}\n```'), b.MODEL)
        self.assertEqual(text, '{"edits":[]}')
        self.assertFalse(info['strict_json'])
        self.assertEqual(info['normalization'], 'outer_json_fence_only')

    def test_truncation_and_model_substitution_fail(self):
        for key, value in [('model', 'substitute'), ('finish_reason', 'length')]:
            data = response()
            if key == 'model': data[key] = value
            else: data['choices'][0][key] = value
            with self.assertRaises(ValueError): b.decode_response(data, b.MODEL)

    def test_ambiguous_or_non_json_response_fails(self):
        for raw in ['Here you go: {"edits":[]}', '```python\npass\n```', 'null', '[]', '']:
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                b.decode_response(response(raw), b.MODEL)

    def test_missing_choice_and_oversize_fail(self):
        for data in [{}, {'model': b.MODEL, 'choices': []}, response('x' * (v.MAX_RESPONSE_BYTES + 1))]:
            with self.assertRaises(ValueError): b.decode_response(data, b.MODEL)

    def test_duplicate_json_keys_are_not_silently_accepted(self):
        with self.assertRaises(ValueError):
            b.decode_response(response('{"edits":[],"edits":[]}'), b.MODEL)


class RequestTests(unittest.TestCase):
    def setUp(self):
        self.task = v.load_task(ROOT / b.TASKS['sliding-window-retry-budget'])
        self.tmp = tempfile.TemporaryDirectory()
        self.work = Path(self.tmp.name) / 'candidate'
        shutil.copytree(self.task.workspace, self.work)

    def tearDown(self): self.tmp.cleanup()

    def test_candidate_request_round_trip(self):
        (self.work / 'retry_budget.py').write_bytes(b'# NEW_CANDIDATE\n')
        request = b.build_request(self.task, self.work, 'a' * 32)
        task, edits = b.validate_request(request, ROOT)
        self.assertEqual(task.task_id, self.task.task_id)
        self.assertEqual(edits, [{'path': 'retry_budget.py', 'content': '# NEW_CANDIDATE\n'}])
        self.assertTrue(v.verified(request))

    def test_baseline_request_has_no_edits(self):
        request = b.build_request(self.task, self.work, 'b' * 32)
        self.assertEqual(b.validate_request(request, ROOT)[1], [])

    def test_modified_tests_never_leave_controller(self):
        f = next((self.work / 'tests').glob('test_*.py'))
        f.write_text('# bypass', 'utf-8')
        with self.assertRaisesRegex(ValueError, 'NON_TARGET'):
            b.build_request(self.task, self.work, 'c' * 32)

    def test_unregistered_task_cannot_be_sent_to_public_runner(self):
        other = dataclasses.replace(self.task, task_id='company-private-project')
        with self.assertRaises(ValueError): b.build_request(other, self.work, 'd' * 32)

    def test_tampered_request_and_baseline_manifest_rejected(self):
        data = b.build_request(self.task, self.work, 'e' * 32)
        data['task_id'] = 'other'
        with self.assertRaises(ValueError): b.validate_request(data, ROOT)
        data = b.build_request(self.task, self.work, 'e' * 32)
        data['baseline_manifest']['retry_budget.py'] = '0' * 64
        data = v.signed(data)
        with self.assertRaises(ValueError): b.validate_request(data, ROOT)

    def test_missing_public_authorization_rejected(self):
        data = b.build_request(self.task, self.work, 'f' * 32)
        data['data_class'] = 'private'
        with self.assertRaises(ValueError): b.validate_request(v.signed(data), ROOT)

    def test_path_traversal_in_response_or_request_is_rejected(self):
        for bad in ['../xx', '', 'g' * 32, 'A' * 32]:
            with self.assertRaises(ValueError): b.build_request(self.task, self.work, bad)

    def test_actual_worker_checks_copy_not_original(self):
        request = b.build_request(self.task, self.work, 'a' * 32)
        class FakeRunner:
            def check(inner, work):
                self.assertNotEqual(work, self.task.workspace)
                return v.Check(False, 1, .1, 'FAILED', 4)
        result = b.verify_request(request, ROOT, FakeRunner())
        self.assertEqual(result['request_sha256'], request['sha256'])
        self.assertEqual(b.decode_check(result, request).tests_run, 4)
        self.assertTrue(v.verified(result))

    def test_forged_pass_missing_count_or_wrong_parent_rejected(self):
        request = b.build_request(self.task, self.work, 'a' * 32)
        class FakeRunner:
            def check(inner, work): return v.Check(True, 0, .1, 'OK', 4)
        result = b.verify_request(request, ROOT, FakeRunner())
        for modified in [{'request_sha256': '0' * 64}, {'request_id': 'b' * 32},
                         {'check': {'passed': True, 'exit_code': 0, 'tests_run': None,
                                    'elapsed_seconds': .1, 'diagnostic': 'OK'}}]:
            with self.assertRaises(ValueError):
                b.decode_check(v.signed({**result, **modified}), request)

    def test_git_runner_rejects_main_and_arbitrary_repo(self):
        for branch in ['main', 'production', '../main']:
            with self.assertRaises(ValueError):
                b.GitHubRunner(ROOT, self.task, branch, public=True)
        with self.assertRaises(ValueError):
            b.GitHubRunner(ROOT, self.task, 'runs/vda/test', public=False)


if __name__ == '__main__': unittest.main()
