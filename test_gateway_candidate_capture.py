import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import gateway_candidate_capture as g
import verified_dev_agent as v


class MockGatewayError(Exception):
    def __init__(self, code):
        self.code = code


def fake_engine():
    routes = {key: SimpleNamespace(model=key + '-model', family=key) for key in g.ALLOWED_ROUTES}
    def validate(data, route):
        if data.get('model') != route.model:
            raise MockGatewayError('MODEL_ID_MISMATCH')
        if data['choices'][0]['finish_reason'] != 'stop':
            raise MockGatewayError('INCOMPLETE_GENERATION')
    return SimpleNamespace(ROUTES=routes, Task=lambda **kw: SimpleNamespace(**kw),
                           GatewayError=MockGatewayError, validate_answer=validate)


def response(model):
    return {'model': model, 'choices': [{'finish_reason': 'stop', 'message': {
        'role': 'assistant', 'content': '{"edits":[]}', 'reasoning_content': 'do not export'}}],
        'mj': {'upstream_model': 'approved-model', 'paid_fallback': False,
               'unexpected_private_metadata': 'do not export'}, 'private_config': 'do not export'}


class CaptureTests(unittest.TestCase):
    def setUp(self):
        self.task = v.load_task(Path(__file__).parent / 'samples/lease-finalization/task.json')
        self.engine = fake_engine()

    def test_keeps_final_answer_and_strips_unapproved_fields(self):
        actual = g.public_response(response('qwen27-model'), 'qwen27-model', self.engine)
        self.assertEqual(actual['choices'][0]['message']['content'], '{"edits":[]}')
        self.assertNotIn('do not export', json.dumps(actual))
        self.assertEqual(actual['mj'], {'upstream_model': 'approved-model', 'paid_fallback': False})

    def test_rejects_wrong_or_incomplete_response(self):
        with self.assertRaises(MockGatewayError):
            g.public_response(response('wrong'), 'qwen27-model', self.engine)
        data = response('qwen27-model')
        data['choices'][0]['finish_reason'] = 'length'
        with self.assertRaises(MockGatewayError):
            g.public_response(data, 'qwen27-model', self.engine)

    def test_does_not_accept_tool_call_as_final_code(self):
        data = response('qwen27-model')
        data['choices'][0]['message']['tool_calls'] = [{'function': {'name': 'anything'}}]
        with self.assertRaisesRegex(ValueError, 'NON_FINAL_RESPONSE'):
            g.public_response(data, 'qwen27-model', self.engine)

    def test_same_prompt_for_each_model_and_failure_is_preserved(self):
        calls = []
        def generate(task, route):
            calls.append(task)
            if task.route == 'deepseek':
                raise MockGatewayError('GATEWAY_HTTP_503')
            return response(route.model)
        record = g.capture(self.task, self.engine, SimpleNamespace(generate=generate),
                           list(g.ALLOWED_ROUTES))
        self.assertEqual(len(calls), 3)
        self.assertEqual(len({x.prompt for x in calls}), 1)
        self.assertTrue(all(x.data_class == 'public' for x in calls))
        self.assertEqual(record['candidates'][2]['responses'], [{'error': {'code': 'GATEWAY_HTTP_503'}}])
        self.assertFalse(record['generated_source_executed_on_controller'])
        self.assertNotIn('test_job_receipt', calls[0].prompt)

    def test_sensitive_exception_text_is_not_exported(self):
        def fail(*args):
            raise RuntimeError('Authorization: Bearer sensitive-example')
        record = g.capture(self.task, self.engine, SimpleNamespace(generate=fail), ['qwen27'])
        self.assertNotIn('sensitive-example', json.dumps(record))
        self.assertEqual(record['candidates'][0]['capture_status'], 'error')

    def test_invalid_budget_or_duplicate_route_makes_no_call(self):
        def forbidden(*args):
            self.fail('transport must not run')
        caller = SimpleNamespace(generate=forbidden)
        for routes, tokens in [(['qwen27', 'qwen27'], 4096), (['unknown'], 4096), (['qwen27'], 5000)]:
            with self.assertRaises(ValueError):
                g.capture(self.task, self.engine, caller, routes, max_tokens=tokens)

    def test_same_family_is_not_parallelized_under_different_names(self):
        self.engine.ROUTES['flashnext'].family = 'qwen27'
        with self.assertRaisesRegex(ValueError, 'ONE_CANDIDATE_PER_PROVIDER_FAMILY'):
            g.capture(self.task, self.engine, None, ['qwen27', 'flashnext'])

    def test_changed_engine_is_rejected_before_import(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / 'parallel_engine.py'
            path.write_text('raise RuntimeError("untrusted engine executed")')
            with self.assertRaisesRegex(ValueError, 'EXISTING_ENGINE_SHA256_MISMATCH'):
                g.load_engine(path)

    def test_split_capture_replays_through_real_comparator_contract(self):
        from compare_candidates import CapturedModel
        caller = SimpleNamespace(generate=lambda task, route: response(route.model))
        rich = g.capture(self.task, self.engine, caller, ['qwen27'])
        replay, metadata = g.split_capture(rich)
        self.assertNotIn('responses', metadata['candidates'][0])
        self.assertEqual(metadata['fresh_model_calls'], 1)
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / 'captures.json'
            path.write_text(json.dumps(replay))
            model = CapturedModel(path)
            actual = model.generate_evidence('qwen27', v.prompt_for(self.task), 30)
            self.assertEqual(actual['model'], 'qwen27-model')


if __name__ == '__main__':
    unittest.main()
