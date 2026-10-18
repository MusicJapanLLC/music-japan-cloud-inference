import copy
import hashlib
import http.client
import json
import tempfile
import unittest
from pathlib import Path

import redline397b as r
import clean_runner as cleanup


class EvaluationContractTests(unittest.TestCase):
    def setUp(self):
        self.cases = json.loads((Path(__file__).parent / 'cases.json').read_text('utf8'))['cases']

    def test_vulnerable_and_fixed_cases_score_independently(self):
        good = ['{"label":"IDOR","evidence":["E1"]}', '{"label":"CLEAR","evidence":[]}', '{"label":"CLEAR","evidence":[]}']
        self.assertEqual([r.score(c, a, 'stop')['status'] for c, a in zip(self.cases, good)], ['CORRECT'] * 3)
        self.assertEqual(r.score(self.cases[0], good[1], 'stop')['status'], 'WRONG_ANSWER')
        self.assertEqual(r.score(self.cases[2], '{"label":"SQLI","evidence":["E3"]}', 'stop')['status'], 'WRONG_ANSWER')

    def test_parser_requires_exact_json_schema_and_supported_evidence(self):
        for bad in ['CLEAR', '{}', '{"label":"CLEAR","evidence":[],"extra":1}', '{"label":"IDOR","evidence":["E9"]}', '{"label":"CLEAR","evidence":["E1"]}']:
            self.assertEqual(r.score(self.cases[0], bad, 'stop')['status'], 'INVALID_OUTPUT')

    def test_truncation_and_transport_failure_are_not_wrong_answers(self):
        self.assertEqual(r.score(self.cases[0], '{', 'length')['status'], 'OUTPUT_LIMIT')
        result = r.evaluate_case(self.cases[0], lambda *args: (_ for _ in ()).throw(TimeoutError()), 20)
        self.assertEqual(result['status'], 'TIME_LIMIT')
        result = r.evaluate_case(self.cases[0], lambda *args: (_ for _ in ()).throw(OSError()), 20)
        self.assertEqual(result['status'], 'MODEL_FAILURE')

    def test_case_input_hash_changes_when_evidence_changes_and_hides_gold(self):
        before = r.case_input(self.cases[0])
        after = copy.deepcopy(self.cases[0]); after['source'] += '\n# changed'
        self.assertNotEqual(r.digest(before), r.digest(r.case_input(after)))
        self.assertNotIn('expected', json.dumps(before))

    def test_exact_context_gate_refuses_over_budget(self):
        r.context_gate(608)
        with self.assertRaisesRegex(ValueError, 'CONTEXT_BUDGET'):
            r.context_gate(609)

    def test_resource_gate_fails_below_all_shards_plus_reserve(self):
        r.capacity_gate(16, 84.2)
        with self.assertRaisesRegex(ValueError, 'DISK_CAPACITY'):
            r.capacity_gate(16, 84.0)
        with self.assertRaisesRegex(ValueError, 'RAM_CAPACITY'):
            r.capacity_gate(8, 100)

    def test_receipt_binds_model_runtime_input_and_raw_output(self):
        def response(messages, timeout):
            return {'choices': [{'finish_reason':'stop','message':{'content':'{"label":"IDOR","evidence":["E1"]}'}}], 'usage':{'prompt_tokens':200,'completion_tokens':20}}
        result = r.evaluate_case(self.cases[0], response, 20)
        self.assertEqual(result['status'], 'CORRECT')
        self.assertEqual(result['output_sha256'], hashlib.sha256(result['raw_output'].encode()).hexdigest())
        self.assertEqual(len(result['input_sha256']), 64)
        record = r.make_receipt([result], {'source_commit':r.RUNTIME_SHA, 'binary_sha256':'1'*64}, False)
        self.assertFalse(record['real_inference'])
        self.assertEqual(record['model']['revision'], '69ed2f05fdcea0e9a880dd4c518be64c01663157')
        self.assertEqual(len(record['model_sha256']), 64)
        signed = record.pop('receipt_sha256')
        self.assertEqual(signed, r.digest(record))

    def test_runtime_sha_mismatch_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'RUNTIME_IDENTITY'):
            r.validate_runtime({'source_commit':'0'*40,'binary_sha256':'1'*64})

    def test_suite_tampering_rejected_before_network(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/'cases.json'; p.write_text('{"cases":[]}', 'utf8')
            with self.assertRaisesRegex(ValueError, 'SUITE_IDENTITY'):
                r.load_cases(p)

    def test_higher_precision_requires_measured_extra_space(self):
        model = r.select_model('IQ1_M')
        self.assertEqual(sum(p['size'] for p in model['parts']), 98280612832)
        with self.assertRaisesRegex(ValueError, 'DISK_CAPACITY'):
            r.capacity_gate(16, 85.46, model)
        r.capacity_gate(16, 94, model)

    def test_cleanup_refuses_local_host_arbitrary_paths_and_nested_mounts(self):
        with self.assertRaisesRegex(ValueError, 'HOSTED_RUNNER_ONLY'):
            cleanup.host_gate({})
        for path in ['/home/runner/work', '/', '/usr', '/usr/share/dotnet-other']:
            with self.assertRaisesRegex(ValueError, 'TARGET'):
                cleanup.target_gate(path, '/home/runner/work/repo', [])
        with self.assertRaisesRegex(ValueError, 'MOUNT'):
            cleanup.target_gate('/usr/share/dotnet', '/home/runner/work/repo', ['/usr/share/dotnet/shared'])

    def test_no_more_requests_after_uncertain_model_timeout(self):
        result = r.evaluate_suite(self.cases, lambda *args: (_ for _ in ()).throw(TimeoutError()), float('inf'))
        self.assertEqual([row['status'] for row in result], ['TIME_LIMIT', 'NOT_RUN_TIME_LIMIT', 'NOT_RUN_TIME_LIMIT'])
        result = r.evaluate_suite(self.cases, lambda *args: (_ for _ in ()).throw(OSError()), float('inf'))
        self.assertEqual([row['status'] for row in result], ['MODEL_FAILURE', 'NOT_RUN_MODEL_FAILURE', 'NOT_RUN_MODEL_FAILURE'])

    def test_expired_suite_never_calls_the_model(self):
        def forbidden(*args):
            self.fail('expired suite dispatched inference')
        result = r.evaluate_suite(self.cases, forbidden, 0)
        self.assertEqual([row['status'] for row in result], ['NOT_RUN_TIME_LIMIT'] * 3)

    def test_malformed_http_response_is_model_failure(self):
        result = r.evaluate_case(self.cases[0], lambda *args: (_ for _ in ()).throw(http.client.HTTPException()), 20)
        self.assertEqual(result['status'], 'MODEL_FAILURE')

    def test_incomplete_evaluation_is_not_successful_but_wrong_answers_are_complete(self):
        self.assertFalse(r.evaluation_complete([{'status':'TIME_LIMIT'}] * 3))
        self.assertFalse(r.evaluation_complete([{'status':'CORRECT'}]))
        self.assertFalse(r.evaluation_complete([{'status':'INVALID_OUTPUT'}] * 3))
        self.assertTrue(r.evaluation_complete([{'status':'WRONG_ANSWER'}, {'status':'CORRECT'}, {'status':'CORRECT'}]))

    def test_best_fit_uses_actual_capacity_and_never_selects_a_too_large_quant(self):
        for free, quant in [(85.46,'IQ1_S'), (94,'IQ1_M'), (108,'IQ2_XXS'), (120,'IQ2_XS')]:
            self.assertEqual(r.best_fit_model(free, 15.61)['quantization'], quant)
        with self.assertRaisesRegex(ValueError, 'DISK_CAPACITY'):
            r.best_fit_model(80, 15.61)

    def test_each_case_receives_at_most_ten_minutes_even_with_unlimited_suite_time(self):
        budgets = []
        def response(data, timeout):
            budgets.append(timeout)
            answer = '{"label":"CLEAR","evidence":[]}'
            return {'choices':[{'message':{'content':answer},'finish_reason':'stop'}],
                    'usage':{'prompt_tokens':200,'completion_tokens':20}}
        results = r.evaluate_suite(self.cases, response, float('inf'))
        self.assertEqual(budgets, [600, 600, 600])
        self.assertEqual(len(results), 3)


if __name__ == '__main__':
    unittest.main()
